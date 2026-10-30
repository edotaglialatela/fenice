// CellParallel: the remesh of a localized run, with no rank holding the topology of the whole mesh.
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
#include "fv/CellRemesh.hpp"

namespace fv {

using detail::kNoKey;
using detail::leaf_key;

bool CellParallel::remesh(CellAdaptor& ad) {
    if (!localized_) throw std::logic_error("CellParallel::remesh: the mesh is not localized");
    using Clock = std::chrono::steady_clock;
    const auto sec = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    const int R = comm_.size, me = comm_.rank, M = mesh_.max_level;
    const auto t0 = Clock::now();
    // 1) the owned leaves, and the region the indicator and the decisions read, with its state, from the owners
    std::vector<int> owned;
    for (int l = 0; l <= M; ++l)
        for (int c = engine_.owned_begin(l); c < engine_.owned_end(l); ++c) owned.push_back(c);
    std::vector<RegionItem> items;
    {
        std::vector<std::vector<std::array<int, 2>>> seeds(static_cast<size_t>(M) + 1);
        for (const int c : owned) {
            const int l = mesh_.level[static_cast<size_t>(c)], i = mesh_.ci[static_cast<size_t>(c)], j = mesh_.cj[static_cast<size_t>(c)];
            seeds[static_cast<size_t>(l)].push_back({i, j});
            if (l > 0) seeds[static_cast<size_t>(l) - 1].push_back({i >> 1, j >> 1});
        }
        for (int t = 0; t <= M; ++t) {   // (the cells CellIndicator::region marks: the ring of the smoothing, which holds the one of the ripple)
            if (seeds[static_cast<size_t>(t)].empty()) continue;
            RegionItem it;
            it.level = t;
            it.box = dilate_cells(seeds[static_cast<size_t>(t)], mesh_.nx_at(t), mesh_.ny_at(t), ad.indicator().diffusion + 1 + region_shrink,
                                  mesh_.periodic_x, mesh_.periodic_y);
            it.tmin = 0;
            it.tmax = M;
            items.push_back(std::move(it));
        }
    }
    CellMesh W(mesh_.Lx, mesh_.Ly, mesh_.nx, mesh_.ny, M, mesh_.periodic_x, mesh_.periodic_y, mesh_.x0, mesh_.y0);
    FetchPlan plan;
    {
        ProfileScope ps(Section::Fetch);
        fetch_leaves(split_, mesh_, &mesh_.U, items, W, &plan);
    }
    const int NW = W.num_leaves();
    std::vector<int> owned_w(owned.size());
    for (size_t k = 0; k < owned.size(); ++k) {
        const int c = owned[k];
        owned_w[k] = W.find(mesh_.level[static_cast<size_t>(c)], mesh_.ci[static_cast<size_t>(c)], mesh_.cj[static_cast<size_t>(c)]);
        if (owned_w[k] < 0) throw std::logic_error("CellParallel::remesh: an owned leaf is not in the region fetched");
    }
    const auto t1 = Clock::now();
    // 2) the decisions of the owned leaves (code: refine + 2 coarsen), and the mirror lists to exchange them with the halo
    std::vector<char> rf, cf;
    if (flag_hook) flag_hook(W, owned_w, rf, cf);
    else ad.flags(W, W.U, owned_w, rf, cf);
    std::vector<char> code(static_cast<size_t>(NW), 0);
    long long nr_local = 0;
    for (size_t k = 0; k < owned.size(); ++k) {
        code[static_cast<size_t>(owned_w[k])] = static_cast<char>(rf[k] + 2 * cf[k]);
        nr_local += rf[k];
    }
    std::vector<std::vector<int>> send_w(static_cast<size_t>(R)), recv_w(static_cast<size_t>(R));
    for (int r = 0; r < R; ++r) {
        if (r == me) continue;
        for (const int c : plan.served[static_cast<size_t>(r)])
            send_w[static_cast<size_t>(r)].push_back(W.find(mesh_.level[static_cast<size_t>(c)], mesh_.ci[static_cast<size_t>(c)], mesh_.cj[static_cast<size_t>(c)]));
        for (const std::array<int, 3>& t : plan.from[static_cast<size_t>(r)]) recv_w[static_cast<size_t>(r)].push_back(W.find(t[0], t[1], t[2]));
    }
    const auto exchange_codes = [&]() {
        std::vector<std::vector<double>> snd(static_cast<size_t>(R)), rcv;
        for (int r = 0; r < R; ++r)
            for (const int w : send_w[static_cast<size_t>(r)]) snd[static_cast<size_t>(r)].push_back(static_cast<double>(code[static_cast<size_t>(w)]));
        exchange_sized(snd, rcv);
        for (int r = 0; r < R; ++r)
            for (size_t k = 0; k < recv_w[static_cast<size_t>(r)].size(); ++k)
                code[static_cast<size_t>(recv_w[static_cast<size_t>(r)][k])] = static_cast<char>(rcv[static_cast<size_t>(r)][k]);
    };
    std::vector<char> Rf(static_cast<size_t>(NW), 0), Cf(static_cast<size_t>(NW), 0);
    const auto split_codes = [&]() {
        for (int w = 0; w < NW; ++w) {
            Rf[static_cast<size_t>(w)] = code[static_cast<size_t>(w)] & 1;
            Cf[static_cast<size_t>(w)] = code[static_cast<size_t>(w)] >> 1;
        }
    };
    exchange_codes();
    split_codes();
    const auto t2 = Clock::now();
    // 3) the ripple: a leaf with a refined finer leaf around it is refined too; rounds of (own pass, exchange) until no rank changes.
    // A rank that fails (a region too small) says so in the same reduction, so that all stop together.
    std::string failure;
    for (int round = 0;; ++round) {
        double flag = 0.0;
        try {
            for (size_t k = 0; k < owned.size(); ++k) {
                const int w = owned_w[k];
                if (!Rf[static_cast<size_t>(w)] && W.forced_by_finer(w, Rf, true)) {
                    Rf[static_cast<size_t>(w)] = 1;
                    code[static_cast<size_t>(w)] |= 1;
                    flag = 1.0;
                }
            }
        } catch (const std::logic_error& e) {
            failure = e.what();
            flag = 2.0;
        }
        const double any = reduce_max(flag);
        if (any == 2.0) throw std::logic_error(failure.empty() ? "CellParallel::remesh: another rank's region was too small" : failure);
        if (any == 0.0) break;
        if (round > 4 * (M + 2)) throw std::logic_error("CellParallel::remesh: the ripple does not converge");
        exchange_codes();
        split_codes();
    }
    // 4) the merges: every owner of a member of a group evaluates it on the same flags; the owner of the anchor makes the parent
    enum Role { Keep = 0, Refine = 1, Absorbed = 2, Anchor = 3 };
    std::vector<char> role(owned.size(), Keep);
    double any_change = 0.0;
    try {
        std::map<int, bool> group;   // by anchor (index in W)
        for (size_t k = 0; k < owned.size(); ++k) {
            const int w = owned_w[k], l = W.level[static_cast<size_t>(w)];
            if (Rf[static_cast<size_t>(w)]) {
                role[k] = Refine;
                any_change = 1.0;
                continue;
            }
            if (l == 0) continue;
            const int ai = W.ci[static_cast<size_t>(w)] & ~1, aj = W.cj[static_cast<size_t>(w)] & ~1;
            const int a = W.find(l, ai, aj);
            if (a < 0) {
                if (!W.has_leaf_under(l, ai, aj)) throw std::logic_error("CellParallel::remesh: the anchor cell of a group is not in the region fetched");
                continue;
            }
            auto it = group.find(a);
            if (it == group.end()) it = group.emplace(a, W.merge_ok(a, Cf, Rf, true)).first;
            if (!it->second) continue;
            role[k] = a == w ? Anchor : Absorbed;
            if (a == w) any_change = 1.0;
        }
    } catch (const std::logic_error& e) {
        failure = e.what();
        any_change = 2.0;
    }
    {
        const double any = reduce_max(any_change);
        if (any == 2.0) throw std::logic_error(failure.empty() ? "CellParallel::remesh: another rank's region was too small" : failure);
        if (any == 0.0) {
            t_topo += sec(t2, Clock::now());
            profile_add(Section::RemeshTopology, sec(t2, Clock::now()));
            t_region += sec(t0, t1);
            profile_add(Section::RemeshRegion, sec(t0, t1));
            t_flags += sec(t1, t2);
            profile_add(Section::RemeshFlags, sec(t1, t2));
            ad.t_indicator += sec(t0, t2);
            return false;
        }
    }
    // 5) the new leaves this rank makes, with their state: the children of its refined leaves (injection), the parents of the groups
    // it anchors (mean of the four), the leaves that stay
    std::vector<int> lv, ii, jj;
    Field nu;
    std::vector<std::vector<uint64_t>> keys(static_cast<size_t>(M) + 1);
    const auto push = [&](int l, int i, int j, const State& u) {
        lv.push_back(l);
        ii.push_back(i);
        jj.push_back(j);
        nu.push_back(u);
        keys[static_cast<size_t>(l)].push_back(morton_key(static_cast<uint32_t>(i), static_cast<uint32_t>(j)));
    };
    for (size_t k = 0; k < owned.size(); ++k) {
        const int w = owned_w[k], l = W.level[static_cast<size_t>(w)], i = W.ci[static_cast<size_t>(w)], j = W.cj[static_cast<size_t>(w)];
        switch (role[k]) {
        case Keep: push(l, i, j, W.U[static_cast<size_t>(w)]); break;
        case Refine:
            for (int q = 0; q < 4; ++q) push(l + 1, 2 * i + (q & 1), 2 * j + (q >> 1), W.U[static_cast<size_t>(w)]);
            break;
        case Anchor: {
            const State &a = W.U[static_cast<size_t>(w)], &b = W.U[static_cast<size_t>(W.find(l, i + 1, j))],
                        &c = W.U[static_cast<size_t>(W.find(l, i, j + 1))], &d = W.U[static_cast<size_t>(W.find(l, i + 1, j + 1))];
            State m;
            for (size_t q = 0; q < static_cast<size_t>(kNVar); ++q) m[q] = ((a[q] + b[q]) + (c[q] + d[q])) * 0.25;
            push(l - 1, i >> 1, j >> 1, m);
            break;
        }
        default: break;   // absorbed into the parent of its anchor
        }
    }
    for (std::vector<uint64_t>& kl : keys) std::sort(kl.begin(), kl.end());
    // 6) the facts of the new mesh (counts and splitters), then every new leaf goes to its owner
    const GlobalFacts gf = global_facts(keys);
    const long long nr = static_cast<long long>(reduce_sum(static_cast<double>(nr_local)));
    const long long before = static_cast<long long>(reduce_sum(static_cast<double>(owned.size())));
    long long after = 0;
    for (const long long n : gf.count) after += n;
    ad.n_refined += nr;
    ad.n_joined += std::max(0LL, before + 3 * nr - after) / 3;
    const size_t rec = 3 + static_cast<size_t>(kNVar);
    std::vector<std::vector<double>> out(static_cast<size_t>(R)), in;
    std::vector<int> ml, mi, mj;
    Field mu;
    for (size_t k = 0; k < lv.size(); ++k) {
        const int o = owner_in(gf.split, lv[k], morton_key(static_cast<uint32_t>(ii[k]), static_cast<uint32_t>(jj[k])));
        if (o == me) {
            ml.push_back(lv[k]);
            mi.push_back(ii[k]);
            mj.push_back(jj[k]);
            mu.push_back(nu[k]);
        } else {
            std::vector<double>& s = out[static_cast<size_t>(o)];
            s.push_back(lv[k]);
            s.push_back(ii[k]);
            s.push_back(jj[k]);
            s.insert(s.end(), nu[k].begin(), nu[k].end());
        }
    }
    exchange_sized(out, in);
    for (int r = 0; r < R; ++r) {
        const std::vector<double>& a = in[static_cast<size_t>(r)];
        for (size_t k = 0; k + rec <= a.size(); k += rec) {
            ml.push_back(static_cast<int>(a[k]));
            mi.push_back(static_cast<int>(a[k + 1]));
            mj.push_back(static_cast<int>(a[k + 2]));
            State u;
            for (size_t q = 0; q < static_cast<size_t>(kNVar); ++q) u[q] = a[k + 3 + q];
            mu.push_back(u);
        }
    }
    {   // the share must be the equal-count block the facts give
        double bad = 0.0;
        std::vector<long long> got(static_cast<size_t>(M) + 1, 0);
        for (const int l : ml) ++got[static_cast<size_t>(l)];
        for (int l = 0; l <= M; ++l) {
            const long long n = gf.count[static_cast<size_t>(l)];
            if (got[static_cast<size_t>(l)] != n * (me + 1) / R - n * me / R) bad = 1.0;
        }
        if (reduce_max(bad) != 0.0) throw std::logic_error("CellParallel::remesh: the new leaves are not spread as the splitters say");
    }
    const auto t3 = Clock::now();
    // 7) the mesh and the engine of the new share, the partition, the halo
    localized_ = false;
    auto t4 = Clock::now();
    build_share_fetched(ml, mi, mj, mu, gf, &t4);
    for (int l = 0; l <= M; ++l) exchange_final(l);
    const auto t5 = Clock::now();
    t_topo += sec(t2, t3);
    profile_add(Section::RemeshTopology, sec(t2, t3));
    t_region += sec(t0, t1);
    profile_add(Section::RemeshRegion, sec(t0, t1));
    t_flags += sec(t1, t2);
    profile_add(Section::RemeshFlags, sec(t1, t2));
    t_migrate += sec(t2, t3);
    profile_add(Section::RemeshMigrate, sec(t2, t3));
    t_rebuild += sec(t3, t4);
    profile_add(Section::RemeshRebuild, sec(t3, t4));
    t_repartition += sec(t4, t5);
    profile_add(Section::RemeshPartition, sec(t4, t5));
    ad.t_indicator += sec(t0, t2);
    ad.t_adapt += sec(t2, t3);
    ad.t_build += sec(t3, t4);
    if (shadow) check_against_whole_build();
    return true;
}

void CellParallel::check_against_whole_build() {
    // the whole new mesh and its engine, the way a run with the topology on every rank makes them, and the share cut from them
    std::vector<int> lv, ci, cj;
    Field U;
    collect(lv, ci, cj, U, false);
    CellMesh whole(mesh_.Lx, mesh_.Ly, mesh_.nx, mesh_.ny, mesh_.max_level, mesh_.periodic_x, mesh_.periodic_y, mesh_.x0, mesh_.y0);
    whole.assign_leaves(lv, ci, cj, &U);
    CellFluxEngine eng(engine_);
    eng.build(whole);
    CellParallel ref(whole, eng, comm_);
    ref.partition();
    ref.localize();
    for (int l = 0; l <= mesh_.max_level; ++l) ref.exchange_final(l);
    const bool sm = mesh_.same_arrays(whole), se = engine_.same_tables(eng, &mesh_), sl = same_lists(ref);
    if (!sm || !se || !sl) std::fprintf(stderr, "shadow rank %d: mesh %d engine %d lists %d\n", comm_.rank, sm, se, sl);
    // every rank throws together (a rank that threw alone would leave the others waiting in a collective)
    if (reduce_max(sm && se && sl ? 0.0 : 1.0) != 0.0)
        throw std::logic_error("CellParallel::remesh: the directly built share differs from the one cut from the whole mesh");
}

}   // namespace fv
