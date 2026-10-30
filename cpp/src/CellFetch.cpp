// CellParallel: the global facts of a mesh (counts, splitters) and the fetch of the leaves around a region from their owners.
#include "fv/CellParallel.hpp"

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
        while ((1 << bits) < std::max(mesh_.nx_at(static_cast<int>(l)), mesh_.ny_at(static_cast<int>(l)))) ++bits;
        rounds = std::max(rounds, 2 * bits);
        for (int r = 0; r < R; ++r) {
            const long long pos = gf.count[l] * r / R;
            if (pos < gf.count[l]) tg.push_back({l, r, pos, 0, (static_cast<uint64_t>(1) << (2 * bits)) - 1});
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
    const int R = comm_.size, me = comm_.rank, M = responder.max_level;
    const bool states = resp_state != nullptr;
    if (plan) {
        plan->from.assign(static_cast<size_t>(R), {});
        plan->served.assign(static_cast<size_t>(R), {});
    }
    std::vector<int> lv, ii, jj;
    Field U;
    std::vector<std::vector<long long>> found(static_cast<size_t>(M) + 1);   // sorted cell numbers (j * nx + i) of the leaves found, per level
    const auto cell_no = [&](int t, int i, int j) { return static_cast<long long>(j) * responder.nx_at(t) + i; };
    // Per item the cells of its box (at its own level), and the cells of the level reached that hold no leaf (they are split: the
    // next level asks about their children)
    struct Work {
        const RegionItem* it;
        std::vector<int> bi, bj;
        std::vector<long long> split_cells;
    };
    std::vector<Work> work;
    for (const RegionItem& it : items) {
        Work w;
        w.it = &it;
        for (int j = 0; j < it.box.h; ++j)
            for (int i = 0; i < it.box.w; ++i)
                if (it.box.g[static_cast<size_t>(j) * it.box.w + i]) {
                    w.bi.push_back(it.box.i0 + i);
                    w.bj.push_back(it.box.j0 + j);
                }
        if (!w.bi.empty()) work.push_back(std::move(w));
    }
    const auto sort_unique = [](std::vector<long long>& v) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    };
    for (int t = 0; t <= M; ++t) {
        // 1) the cells of level t each item asks about: its box's ancestors above its level, the box, the children of the split cells below
        std::vector<std::vector<long long>> frontier(work.size());
        std::vector<long long> ask_cells, want_cells;
        for (size_t k = 0; k < work.size(); ++k) {
            Work& w = work[k];
            const int L = w.it->level;
            std::vector<long long>& fr = frontier[k];
            if (t > w.it->tmax) continue;
            if (t < L) {
                if (t < w.it->tmin) continue;
                for (size_t q = 0; q < w.bi.size(); ++q) fr.push_back(cell_no(t, w.bi[q] >> (L - t), w.bj[q] >> (L - t)));
                sort_unique(fr);
            } else if (t == L) {
                for (size_t q = 0; q < w.bi.size(); ++q) fr.push_back(cell_no(t, w.bi[q], w.bj[q]));
            } else {
                const int nxp = responder.nx_at(t - 1);
                for (const long long no : w.split_cells) {
                    const int i = static_cast<int>(no % nxp), j = static_cast<int>(no / nxp);
                    for (int q = 0; q < 4; ++q) fr.push_back(cell_no(t, 2 * i + (q & 1), 2 * j + (q >> 1)));
                }
                sort_unique(fr);
            }
            ask_cells.insert(ask_cells.end(), fr.begin(), fr.end());
            if (t >= w.it->tmin) want_cells.insert(want_cells.end(), fr.begin(), fr.end());
        }
        sort_unique(ask_cells);
        sort_unique(want_cells);
        // 2) those not inside a leaf already found go to the owners: (i, j, wanted)
        std::vector<std::vector<double>> ask(static_cast<size_t>(R));
        std::vector<long long> mine, covered_cells;
        const int nx = responder.nx_at(t);
        for (const long long no : ask_cells) {
            const int i = static_cast<int>(no % nx), j = static_cast<int>(no / nx);
            bool covered = false;
            for (int s = 0; s < t && !covered; ++s)
                covered = std::binary_search(found[static_cast<size_t>(s)].begin(), found[static_cast<size_t>(s)].end(),
                                             cell_no(s, i >> (t - s), j >> (t - s)));
            if (covered) {
                covered_cells.push_back(no);
                continue;
            }
            const int o = owner_in(split, t, morton_key(static_cast<uint32_t>(i), static_cast<uint32_t>(j)));
            if (o == me) {
                mine.push_back(no);
            } else {
                ask[static_cast<size_t>(o)].push_back(i);
                ask[static_cast<size_t>(o)].push_back(j);
                ask[static_cast<size_t>(o)].push_back(std::binary_search(want_cells.begin(), want_cells.end(), no) ? 1.0 : 0.0);
            }
        }
        std::vector<std::vector<double>> got_ask, reply(static_cast<size_t>(R)), got_reply;
        exchange_sized(ask, got_ask);
        for (int r = 0; r < R; ++r) {   // the leaves asked of this rank that it has: (i, j, wanted) and, if wanted, the state
            if (r == me) continue;
            const std::vector<double>& q = got_ask[static_cast<size_t>(r)];
            for (size_t k = 0; k + 2 < q.size(); k += 3) {
                const int c = responder.find(t, static_cast<int>(q[k]), static_cast<int>(q[k + 1]));
                if (c < 0) continue;
                std::vector<double>& rp = reply[static_cast<size_t>(r)];
                rp.push_back(q[k]);
                rp.push_back(q[k + 1]);
                rp.push_back(q[k + 2]);
                if (q[k + 2] != 0.0) {
                    if (states)
                        for (int v = 0; v < kNVar; ++v) rp.push_back((*resp_state)[static_cast<size_t>(c)][static_cast<size_t>(v)]);
                    if (plan) plan->served[static_cast<size_t>(r)].push_back(c);
                }
            }
        }
        exchange_sized(reply, got_reply);
        std::vector<long long>& fl = found[static_cast<size_t>(t)];
        for (const long long no : mine) {
            const int i = static_cast<int>(no % nx), j = static_cast<int>(no / nx);
            const int c = responder.find(t, i, j);
            if (c < 0) continue;
            fl.push_back(no);
            if (!std::binary_search(want_cells.begin(), want_cells.end(), no)) continue;
            lv.push_back(t);
            ii.push_back(i);
            jj.push_back(j);
            if (states) U.push_back((*resp_state)[static_cast<size_t>(c)]);
        }
        for (int r = 0; r < R; ++r) {
            if (r == me) continue;
            const std::vector<double>& a = got_reply[static_cast<size_t>(r)];
            for (size_t k = 0; k + 3 <= a.size();) {
                const int i = static_cast<int>(a[k]), j = static_cast<int>(a[k + 1]);
                const bool wanted = a[k + 2] != 0.0;
                fl.push_back(cell_no(t, i, j));
                k += 3;
                if (!wanted) continue;
                lv.push_back(t);
                ii.push_back(i);
                jj.push_back(j);
                if (states) {
                    State u;
                    for (int v = 0; v < kNVar; ++v) u[static_cast<size_t>(v)] = a[k + static_cast<size_t>(v)];
                    U.push_back(u);
                    k += static_cast<size_t>(kNVar);
                }
                if (plan) plan->from[static_cast<size_t>(r)].push_back({t, i, j});
            }
        }
        std::sort(fl.begin(), fl.end());
        // 3) the cells reached that hold no leaf are split: their children are asked next, down to the item's deepest level
        for (size_t k = 0; k < work.size(); ++k) {
            Work& w = work[k];
            w.split_cells.clear();
            if (t < w.it->level || t >= w.it->tmax || t >= M) continue;
            for (const long long no : frontier[k])
                if (!std::binary_search(covered_cells.begin(), covered_cells.end(), no) && !std::binary_search(fl.begin(), fl.end(), no))
                    w.split_cells.push_back(no);
        }
    }
    out.assign_partial_leaves(lv, ii, jj, states ? &U : nullptr);
}

}   // namespace fv
