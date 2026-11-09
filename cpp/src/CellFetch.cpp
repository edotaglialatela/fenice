// CellParallel: the global facts of a mesh (counts, splitters) and the fetch of the leaves around a region from their owners.
#include "fv/CellParallel.hpp"
#include "fv/Profile.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include "CellParallelDetail.hpp"
#include "fv/CellFlux.hpp"
#include "fv/Profile.hpp"
#include "fv/CellRemesh.hpp"

namespace fv {

using detail::kNoKey;
using detail::leaf_key;

GlobalFacts CellParallel::global_facts(const std::vector<std::vector<uint64_t>>& keys) const {
    const int R = comm_.size, M = mesh_.max_level;
    const size_t L = static_cast<size_t>(M) + 1;
    GlobalFacts gf;
    gf.count.assign(L, 0);
    gf.split.assign(L, std::vector<uint64_t>(static_cast<size_t>(R) + 1, kNoKey));
    // all ranks' values of a vector of doubles, in rank order
    const auto gather = [&](const std::vector<double>& mine, std::vector<double>& all) {
        const int n = static_cast<int>(mine.size());
        all.resize(static_cast<size_t>(n) * R);
        std::vector<int> counts(static_cast<size_t>(R), n), displs(static_cast<size_t>(R));
        for (int r = 0; r < R; ++r) displs[static_cast<size_t>(r)] = r * n;
        if (R > 1) comm_.allgatherv(mine.data(), n, all.data(), counts.data(), displs.data());
        else all = mine;
    };
    std::vector<double> mine(L), all;
    for (size_t l = 0; l < L; ++l) mine[l] = static_cast<double>(keys[l].size());
    gather(mine, all);
    for (int r = 0; r < R; ++r)
        for (size_t l = 0; l < L; ++l) gf.count[l] += static_cast<long long>(all[static_cast<size_t>(r) * L + l]);
    // the target position of every (level, rank): the key found is the one of that position in the level's order
    struct Target {
        size_t l;
        int r;
        long long pos;
        uint64_t lo, hi;
    };
    std::vector<Target> tg;
    int rounds = 0;
    for (size_t l = 0; l < L; ++l) {
        int bits = 1;
        const int dim = mesh_.dim, ext = std::max({mesh_.nx_at(static_cast<int>(l)), mesh_.ny_at(static_cast<int>(l)), mesh_.nz_at(static_cast<int>(l))});
        while ((1 << bits) < ext) ++bits;
        rounds = std::max(rounds, dim * bits);   // (the key interleaves dim coordinates of `bits` bits)
        for (int r = 0; r < R; ++r) {
            const long long pos = gf.count[l] * r / R;
            if (pos < gf.count[l]) tg.push_back({l, r, pos, 0, (static_cast<uint64_t>(1) << (dim * bits)) - 1});
        }
    }
    std::vector<double> cnt(tg.size()), tot;
    for (int round = 0; round <= rounds; ++round) {
        for (size_t k = 0; k < tg.size(); ++k) {   // the keys <= mid of this rank
            const uint64_t mid = tg[k].lo + (tg[k].hi - tg[k].lo) / 2;
            const std::vector<uint64_t>& kl = keys[tg[k].l];
            cnt[k] = static_cast<double>(std::upper_bound(kl.begin(), kl.end(), mid) - kl.begin());
        }
        gather(cnt, tot);
        for (size_t k = 0; k < tg.size(); ++k) {
            long long below = 0;
            for (int r = 0; r < R; ++r) below += static_cast<long long>(tot[static_cast<size_t>(r) * tg.size() + k]);
            const uint64_t mid = tg[k].lo + (tg[k].hi - tg[k].lo) / 2;
            if (below >= tg[k].pos + 1) tg[k].hi = mid;
            else tg[k].lo = mid + 1;
        }
    }
    for (const Target& t : tg) gf.split[t.l][static_cast<size_t>(t.r)] = t.lo;
    return gf;
}

void CellParallel::exchange_sized(const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv) const {
    const int R = comm_.size, me = comm_.rank;
    recv.assign(static_cast<size_t>(R), {});
    if (R == 1) return;
    std::vector<std::vector<double>> cs(static_cast<size_t>(R)), cr(static_cast<size_t>(R));
    for (int r = 0; r < R; ++r)
        if (r != me) {
            cs[static_cast<size_t>(r)].assign(1, static_cast<double>(send[static_cast<size_t>(r)].size()));
            cr[static_cast<size_t>(r)].assign(1, 0.0);
        }
    comm_.exchange(cs, cr);
    for (int r = 0; r < R; ++r)
        if (r != me) recv[static_cast<size_t>(r)].resize(static_cast<size_t>(cr[static_cast<size_t>(r)][0]));
    comm_.exchange(send, recv);
}

int CellParallel::owner_in(const std::vector<std::vector<uint64_t>>& split, int level, uint64_t key) {
    const std::vector<uint64_t>& sp = split[static_cast<size_t>(level)];
    // (a cell below the first key of the level has no leaf: rank 0 answers that there is none)
    return std::max(0, static_cast<int>(std::upper_bound(sp.begin(), sp.end() - 1, key) - sp.begin()) - 1);
}

void CellParallel::fetch_leaves(const std::vector<std::vector<uint64_t>>& split, const CellMesh& responder, const Field* resp_state,
                                const std::vector<RegionItem>& items, CellMesh& out, FetchPlan* plan) const {
    const int R = comm_.size, me = comm_.rank, M = responder.max_level, dim = responder.dim;
    const int nch = dim == 3 ? 8 : 4;
    const size_t crec = dim == 3 ? 3 : 2;   // the coordinates of a cell in a message
    const bool states = resp_state != nullptr;
    if (plan) {
        plan->from.assign(static_cast<size_t>(R), {});
        plan->served.assign(static_cast<size_t>(R), {});
    }
    std::vector<int> lv, ii, jj, kk;
    Field U;
    // sorted cell numbers ((k * ny + j) * nx + i, the 2-D j * nx + i) of the leaves found, per level
    std::vector<std::vector<long long>> found(static_cast<size_t>(M) + 1);
    const auto cell_no = [&](int t, int i, int j, int k) {
        return (static_cast<long long>(k) * responder.ny_at(t) + j) * responder.nx_at(t) + i;
    };
    const auto cell_of = [&](int t, long long no, int& i, int& j, int& k) {
        const long long nx = responder.nx_at(t), ny = responder.ny_at(t);
        i = static_cast<int>(no % nx);
        j = static_cast<int>((no / nx) % ny);
        k = static_cast<int>(no / (nx * ny));
    };
    // Per item the cells of its box (at its own level), and the cells of the level reached that hold no leaf (they are split: the
    // next level asks about their children)
    struct Work {
        const RegionItem* it;
        std::vector<int> bi, bj, bk;
        std::vector<long long> split_cells;
    };
    std::vector<Work> work;
    for (const RegionItem& it : items) {
        Work w;
        w.it = &it;
        detail::for_box(it.box, [&](int i, int j, int k) {
            w.bi.push_back(i);
            w.bj.push_back(j);
            w.bk.push_back(k);
        });
        if (!w.bi.empty()) work.push_back(std::move(w));
    }
    // v sorted and without duplicates. The cell numbers of a region are dense in their range (a block of a few rows and columns of
    // a mesh), so a scan over a byte per number in the range replaces the sort when the range is not much wider than the list
    // (the result is the same either way; a sparse list takes the sort).
    std::vector<char> seen;
    const auto sort_unique = [&seen](std::vector<long long>& v) {
        if (v.size() > 64) {
            long long lo = v[0], hi = v[0];
            for (const long long x : v) {
                lo = std::min(lo, x);
                hi = std::max(hi, x);
            }
            const long long span = hi - lo + 1;
            if (span <= 16 * static_cast<long long>(v.size()) + 4096) {
                seen.assign(static_cast<size_t>(span), 0);
                for (const long long x : v) seen[static_cast<size_t>(x - lo)] = 1;
                size_t n = 0;
                for (long long e = 0; e < span; ++e)
                    if (seen[static_cast<size_t>(e)]) v[n++] = lo + e;
                v.resize(n);
                return;
            }
        }
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    for (int t = 0; t <= M; ++t) {
        // 1) the cells of level t each item asks about: its box's ancestors above its level, the box, the children of the split cells below
        ProfileScope ps_front(Section::FetchFrontier);
        std::vector<std::vector<long long>> frontier(work.size());
        std::vector<long long> ask_cells, want_cells;
        for (size_t q = 0; q < work.size(); ++q) {
            Work& w = work[q];
            const int L = w.it->level;
            std::vector<long long>& fr = frontier[q];
            if (t > w.it->tmax) continue;
            if (t < L) {
                if (t < w.it->tmin) continue;
                for (size_t e = 0; e < w.bi.size(); ++e) fr.push_back(cell_no(t, w.bi[e] >> (L - t), w.bj[e] >> (L - t), w.bk[e] >> (L - t)));
                sort_unique(fr);
            } else if (t == L) {
                for (size_t e = 0; e < w.bi.size(); ++e) fr.push_back(cell_no(t, w.bi[e], w.bj[e], w.bk[e]));
            } else {
                for (const long long no : w.split_cells) {
                    int i, j, k;
                    cell_of(t - 1, no, i, j, k);
                    for (int c = 0; c < nch; ++c) fr.push_back(cell_no(t, 2 * i + (c & 1), 2 * j + ((c >> 1) & 1), dim == 3 ? 2 * k + ((c >> 2) & 1) : 0));
                }
                sort_unique(fr);
            }
            ask_cells.insert(ask_cells.end(), fr.begin(), fr.end());
            if (t >= w.it->tmin) want_cells.insert(want_cells.end(), fr.begin(), fr.end());
        }
        sort_unique(ask_cells);
        sort_unique(want_cells);
        ps_front.stop();
        // 2) those not inside a leaf already found go to the owners: (i, j[, k], wanted)
        ProfileScope ps_class(Section::FetchClassify);
        std::vector<std::vector<double>> ask(static_cast<size_t>(R));
        std::vector<long long> mine, covered_cells;
        // want_cells is a sorted subset of ask_cells: one pointer walks it; the ancestors of consecutive cells are mostly the same,
        // so the last answer of each level above is kept
        size_t wp = 0;
        std::vector<long long> last_anc(static_cast<size_t>(t), -1);
        std::vector<char> last_hit(static_cast<size_t>(t), 0);
        for (const long long no : ask_cells) {
            int i, j, k;
            cell_of(t, no, i, j, k);
            bool covered = false;
            for (int s = 0; s < t && !covered; ++s) {
                const long long a = cell_no(s, i >> (t - s), j >> (t - s), k >> (t - s));
                if (a != last_anc[static_cast<size_t>(s)]) {
                    last_anc[static_cast<size_t>(s)] = a;
                    last_hit[static_cast<size_t>(s)] =
                        std::binary_search(found[static_cast<size_t>(s)].begin(), found[static_cast<size_t>(s)].end(), a) ? 1 : 0;
                }
                covered = last_hit[static_cast<size_t>(s)] != 0;
            }
            while (wp < want_cells.size() && want_cells[wp] < no) ++wp;
            const bool wanted_cell = wp < want_cells.size() && want_cells[wp] == no;
            if (covered) {
                covered_cells.push_back(no);
                continue;
            }
            const int o = owner_in(split, t, detail::cell_morton(dim, i, j, k));
            if (o == me) {
                mine.push_back(no);
            } else {
                std::vector<double>& a = ask[static_cast<size_t>(o)];
                a.push_back(i);
                a.push_back(j);
                if (dim == 3) a.push_back(k);
                a.push_back(wanted_cell ? 1.0 : 0.0);
            }
        }
        ps_class.stop();
        std::vector<std::vector<double>> got_ask, reply(static_cast<size_t>(R)), got_reply;
        {
            ProfileScope ps(Section::FetchExchange);
            exchange_sized(ask, got_ask);
        }
        ProfileScope ps_serve(Section::FetchServe);
        for (int r = 0; r < R; ++r) {   // the leaves asked of this rank that it has: (i, j[, k], wanted) and, if wanted, the state
            if (r == me) continue;
            const std::vector<double>& q = got_ask[static_cast<size_t>(r)];
            for (size_t e = 0; e + crec < q.size(); e += crec + 1) {
                const int c = responder.find(t, static_cast<int>(q[e]), static_cast<int>(q[e + 1]), dim == 3 ? static_cast<int>(q[e + 2]) : 0);
                if (c < 0) continue;
                std::vector<double>& rp = reply[static_cast<size_t>(r)];
                rp.insert(rp.end(), q.begin() + static_cast<std::ptrdiff_t>(e), q.begin() + static_cast<std::ptrdiff_t>(e + crec + 1));
                if (q[e + crec] != 0.0) {
                    if (states)
                        for (int v = 0; v < kNVar; ++v) rp.push_back((*resp_state)[static_cast<size_t>(c)][static_cast<size_t>(v)]);
                    if (plan) plan->served[static_cast<size_t>(r)].push_back(c);
                }
            }
        }
        ps_serve.stop();
        {
            ProfileScope ps(Section::FetchExchange);
            exchange_sized(reply, got_reply);
        }
        ProfileScope ps_collect(Section::FetchCollect);
        std::vector<long long>& fl = found[static_cast<size_t>(t)];
        size_t wq = 0;   // (mine is a subsequence of ask_cells, so of want_cells' order)
        for (const long long no : mine) {
            int i, j, k;
            cell_of(t, no, i, j, k);
            const int c = responder.find(t, i, j, k);
            if (c < 0) continue;
            fl.push_back(no);
            while (wq < want_cells.size() && want_cells[wq] < no) ++wq;
            if (!(wq < want_cells.size() && want_cells[wq] == no)) continue;
            lv.push_back(t);
            ii.push_back(i);
            jj.push_back(j);
            kk.push_back(k);
            if (states) U.push_back((*resp_state)[static_cast<size_t>(c)]);
        }
        for (int r = 0; r < R; ++r) {
            if (r == me) continue;
            const std::vector<double>& a = got_reply[static_cast<size_t>(r)];
            for (size_t e = 0; e + crec + 1 <= a.size();) {
                const int i = static_cast<int>(a[e]), j = static_cast<int>(a[e + 1]), k = dim == 3 ? static_cast<int>(a[e + 2]) : 0;
                const bool wanted = a[e + crec] != 0.0;
                fl.push_back(cell_no(t, i, j, k));
                e += crec + 1;
                if (!wanted) continue;
                lv.push_back(t);
                ii.push_back(i);
                jj.push_back(j);
                kk.push_back(k);
                if (states) {
                    State u;
                    for (int v = 0; v < kNVar; ++v) u[static_cast<size_t>(v)] = a[e + static_cast<size_t>(v)];
                    U.push_back(u);
                    e += static_cast<size_t>(kNVar);
                }
                if (plan) plan->from[static_cast<size_t>(r)].push_back({t, i, j, k});
            }
        }
        std::sort(fl.begin(), fl.end());
        // 3) the cells reached that hold no leaf are split: their children are asked next, down to the item's deepest level
        for (size_t q = 0; q < work.size(); ++q) {
            Work& w = work[q];
            w.split_cells.clear();
            if (t < w.it->level || t >= w.it->tmax || t >= M) continue;
            for (const long long no : frontier[q])
                if (!std::binary_search(covered_cells.begin(), covered_cells.end(), no) && !std::binary_search(fl.begin(), fl.end(), no))
                    w.split_cells.push_back(no);
        }
    }
    ProfileScope ps_assemble(Section::FetchCollect);
    if (dim == 3)
        out.assign_partial_leaves(lv, ii, jj, kk, states ? &U : nullptr);
    else
        out.assign_partial_leaves(lv, ii, jj, states ? &U : nullptr);
}

}   // namespace fv
