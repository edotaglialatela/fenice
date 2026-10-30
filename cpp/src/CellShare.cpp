// CellParallel: the share of the mesh a rank keeps, built from the topology it holds or from fetched leaves; the first adaptation.
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
#include "fv/Profile.hpp"
#include "fv/CellRemesh.hpp"

namespace fv {

using detail::kNoKey;
using detail::leaf_key;

void CellParallel::region_mask(const CellMesh& topo, const std::vector<int>& owned, int extra, std::vector<char>& keep) const {
    const int N = topo.num_leaves(), M = topo.max_level;
    keep.assign(static_cast<size_t>(N), 0);
    std::vector<std::vector<std::array<int, 2>>> seeds(static_cast<size_t>(M) + 1);
    for (const int c : owned) {
        keep[static_cast<size_t>(c)] = 1;
        seeds[static_cast<size_t>(topo.level[static_cast<size_t>(c)])].push_back({topo.ci[static_cast<size_t>(c)], topo.cj[static_cast<size_t>(c)]});
    }
    // rings of the owned leaf's level within which a leaf of another level is read (docs/design/amr-distributed.md section 8.2, a
    // ring more than the airfoil needs; `extra` more rings on a retry), by level difference d = owner level - level read
    const auto rings = [](int d) { return d < 0 ? 4 : (d == 0 ? 6 : (d == 1 ? 8 : (d == 2 ? 9 : 11))); };
    const bool body = engine_.has_body();
    for (int o = 0; o <= M; ++o) {
        if (seeds[static_cast<size_t>(o)].empty()) continue;
        for (int t = 0; t <= M; ++t) {
            if (topo.level_begin[t + 1] == topo.level_begin[t]) continue;
            int r = std::max(0, rings(o - t) + extra);
            if (body && t == o && o == M) r = std::max(r, 8 + extra);   // the images of the ghost cells (7 rings)
            const CellBox b = dilate_cells(seeds[static_cast<size_t>(o)], topo.nx_at(o), topo.ny_at(o), r, topo.periodic_x, topo.periodic_y);
            for (int j = 0; j < b.h; ++j)
                for (int i = 0; i < b.w; ++i) {
                    if (!b.g[static_cast<size_t>(j) * b.w + i]) continue;
                    const int ci = b.i0 + i, cj = b.j0 + j;
                    if (t <= o) {
                        const int c = topo.find(t, ci >> (o - t), cj >> (o - t));
                        if (c >= 0) keep[static_cast<size_t>(c)] = 1;
                    } else {   // the leaves of level t under this cell
                        const int n = 1 << (t - o);
                        for (int qj = 0; qj < n; ++qj)
                            for (int qi = 0; qi < n; ++qi) {
                                const int c = topo.find(t, ci * n + qi, cj * n + qj);
                                if (c >= 0) keep[static_cast<size_t>(c)] = 1;
                            }
                    }
                }
        }
    }
    // the leaves along a mass-flow inlet (its average reads them on every rank)
    for (int s = 0; s < 4; ++s) {
        if (!engine_.mass_flow_side(s)) continue;
        for (int c = 0; c < N; ++c) {
            const int l = topo.level[static_cast<size_t>(c)];
            const bool on = s == 0 ? topo.ci[static_cast<size_t>(c)] == 0
                                   : (s == 1 ? topo.ci[static_cast<size_t>(c)] == topo.nx_at(l) - 1
                                             : (s == 2 ? topo.cj[static_cast<size_t>(c)] == 0 : topo.cj[static_cast<size_t>(c)] == topo.ny_at(l) - 1));
            if (on) keep[static_cast<size_t>(c)] = 1;
        }
    }
}

void CellParallel::build_share_fetched(const std::vector<int>& lv, const std::vector<int>& ii, const std::vector<int>& jj, const Field& U,
                                         const GlobalFacts& gf, std::chrono::steady_clock::time_point* built) {
    const int M = mesh_.max_level, me = comm_.rank;
    Field own_u = U;
    CellMesh own(mesh_.Lx, mesh_.Ly, mesh_.nx, mesh_.ny, M, mesh_.periodic_x, mesh_.periodic_y, mesh_.x0, mesh_.y0);
    own.assign_partial_leaves(lv, ii, jj, &own_u);   // what this rank answers from
    std::vector<std::vector<std::array<int, 2>>> seeds(static_cast<size_t>(M) + 1);
    for (int c = 0; c < own.num_leaves(); ++c)
        seeds[static_cast<size_t>(own.level[static_cast<size_t>(c)])].push_back({own.ci[static_cast<size_t>(c)], own.cj[static_cast<size_t>(c)]});
    // the ghost-cell wall is active if the cells it reads are all finest-level leaves of the whole mesh (a collective count)
    int ghost_hint = -1;
    if (engine_.ghost_enabled()) {
        const std::vector<int> cells = engine_.ghost_referenced_cells(mesh_);
        const int NX = mesh_.nx_at(M);
        double found = 0.0;
        for (const int fc : cells) {
            const int i = fc % NX, j = fc / NX;
            if (owner_in(gf.split, M, morton_key(static_cast<uint32_t>(i), static_cast<uint32_t>(j))) == me && own.find(M, i, j) >= 0) found += 1.0;
        }
        ghost_hint = reduce_sum(found) == static_cast<double>(cells.size()) ? 1 : 0;
    }
    // rings of the owned leaf's level within which a leaf of another level is read (see region_mask), by level difference
    const auto rings = [](int d) { return d < 0 ? 4 : (d == 0 ? 6 : (d == 1 ? 8 : (d == 2 ? 9 : 11))); };
    for (int extra = first_extra;; extra += 2) {
        std::vector<RegionItem> items;
        for (int o = 0; o <= M; ++o) {   // one item per level of the owned leaves, with the widest of the rings of the levels read (a superset of what region_mask keeps)
            if (seeds[static_cast<size_t>(o)].empty()) continue;
            int r = 0;
            for (int t = 0; t <= M; ++t) {
                if (gf.count[static_cast<size_t>(t)] == 0) continue;
                int rt = std::max(0, rings(o - t) + extra);
                if (engine_.has_body() && t == o && o == M) rt = std::max(rt, 8 + extra);   // the images of the ghost cells
                // one cell more than region_mask reads, in the units of the seeds' level, so that the leaves at the border of the
                // share have the neighbours that make their faces: one cell of the leaf's level, or of the next coarser
                r = std::max(r, rt + (t >= o ? 1 : 2 * (1 << (o - t))));
            }
            RegionItem it;
            it.level = o;
            it.box = dilate_cells(seeds[static_cast<size_t>(o)], mesh_.nx_at(o), mesh_.ny_at(o), r, mesh_.periodic_x, mesh_.periodic_y);
            it.tmin = 0;
            it.tmax = M;
            items.push_back(std::move(it));
        }
        for (int s = 0; s < 4; ++s) {   // the leaves along a mass-flow inlet (its average reads them on every rank)
            if (!engine_.mass_flow_side(s)) continue;
            for (int t = 0; t <= M; ++t) {
                if (gf.count[static_cast<size_t>(t)] == 0) continue;
                const int nx = mesh_.nx_at(t), ny = mesh_.ny_at(t);
                RegionItem it;
                it.level = t;
                it.tmin = it.tmax = t;
                it.box.i0 = s == 1 ? std::max(0, nx - 2) : 0;
                it.box.j0 = s == 3 ? std::max(0, ny - 2) : 0;
                it.box.w = s < 2 ? std::min(2, nx) : nx;
                it.box.h = s < 2 ? ny : std::min(2, ny);
                it.box.g.assign(static_cast<size_t>(it.box.w) * it.box.h, 1);
                items.push_back(std::move(it));
            }
        }
        // (one cell more around the region than region_mask reads, in the units of the seeds' level, so that the leaves at the border of
        // the share have the neighbours that make their faces: one cell of the leaf's level, or of the next coarser)
        CellMesh T(mesh_.Lx, mesh_.Ly, mesh_.nx, mesh_.ny, M, mesh_.periodic_x, mesh_.periodic_y, mesh_.x0, mesh_.y0);
        {
            ProfileScope ps(Section::Fetch);
            fetch_leaves(gf.split, own, nullptr, items, T);   // (no state: the halo is filled by exchange_final)
        }
        std::vector<int> owned_t;
        for (int c = 0; c < own.num_leaves(); ++c)
            owned_t.push_back(T.find(own.level[static_cast<size_t>(c)], own.ci[static_cast<size_t>(c)], own.cj[static_cast<size_t>(c)]));
        std::vector<char> keep;
        {
            ProfileScope ps(Section::ShareRegion);
            region_mask(T, owned_t, extra, keep);
        }
        bool ok = true;
        try {
            {
                ProfileScope ps(Section::ShareMesh);
                mesh_.build_from(T, keep, nullptr);
            }
            for (int c = 0; c < own.num_leaves(); ++c)   // the state of the owned leaves
                mesh_.U[static_cast<size_t>(mesh_.find(own.level[static_cast<size_t>(c)], own.ci[static_cast<size_t>(c)], own.cj[static_cast<size_t>(c)]))] = own.U[static_cast<size_t>(c)];
            engine_.set_ghost_whole_ready(ghost_hint);
            {
                ProfileScope ps(Section::ShareEngine);
                engine_.build(mesh_, &T);
            }
            engine_.set_ghost_whole_ready(-1);
            {
                ProfileScope ps(Section::SharePartition);
                ok = partition_local(&T, &gf);
            }
        } catch (const std::logic_error&) {
            ok = false;
        }
        if (reduce_max(ok ? 0.0 : 1.0) == 0.0) break;
        ++region_retries;
        if (extra > 15) throw std::logic_error("CellParallel: no reach of the leaves covers what the faces read");
    }
    if (built) *built = std::chrono::steady_clock::now();
    {
        ProfileScope ps(Section::SharePartition);
        partition_exchange();
    }
    {
        ProfileScope ps(Section::ShareLocalize);
        localize();
    }
}

void CellParallel::owned_leaves(const CellMesh& topo, std::vector<int>& out) const {
    const int R = comm_.size, me = comm_.rank;
    out.clear();
    for (int l = 0; l <= topo.max_level; ++l) {
        const long long lb = topo.level_begin[static_cast<size_t>(l)], n = topo.level_begin[static_cast<size_t>(l) + 1] - lb;
        for (long long c = lb + n * me / R; c < lb + n * (me + 1) / R; ++c) out.push_back(static_cast<int>(c));
    }
}

void CellParallel::build_share(const CellMesh& topo, const std::vector<int>& mine, const Field* nU,
                               std::chrono::steady_clock::time_point* built) {
    // the leaves within the reach of the owned ones: a partial mesh, its engine tables, the partition; a rank whose share does
    // not cover what its faces read (before anything is sent) makes every rank try again with a wider reach
    for (int extra = first_extra;; extra += 2) {
        bool ok = true;
        std::vector<char> keep;
        {
            ProfileScope ps(Section::ShareRegion);
            region_mask(topo, mine, extra, keep);
        }
        try {
            {
                ProfileScope ps(Section::ShareMesh);
                mesh_.build_from(topo, keep, nU);
            }
            {
                ProfileScope ps(Section::ShareEngine);
                engine_.build(mesh_, &topo);
            }
            {
                ProfileScope ps(Section::SharePartition);
                ok = partition_local(&topo);
            }
        } catch (const std::logic_error&) {
            ok = false;
        }
        if (reduce_max(ok ? 0.0 : 1.0) == 0.0) break;
        ++region_retries;
        if (extra > 15) throw std::logic_error("CellParallel: no reach of the leaves covers what the faces read");
    }
    if (built) *built = std::chrono::steady_clock::now();
    {
        ProfileScope ps(Section::SharePartition);
        partition_exchange();
    }
    {
        ProfileScope ps(Section::ShareLocalize);
        localize();
    }
}

int CellParallel::init_adapt(CellMesh& topo, CellAdaptor& ad, const std::function<void(CellMesh&)>& fill) {
    localized_ = false;
    std::vector<int> mine;
    owned_leaves(topo, mine);
    build_share(topo, mine, nullptr, nullptr);
    int passes = 0;
    for (; passes < topo.max_level + 1; ++passes) {
        fill(mesh_);
        if (!(replicated_topology ? remesh(topo, ad) : remesh(ad))) {
            ++passes;
            break;
        }
    }
    fill(mesh_);
    return passes;
}

}   // namespace fv
