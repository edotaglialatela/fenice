// AMR in three dimensions, phase Q5 (docs/design/amr-octree.md): the indicator, the remesh and the initial adaptation on octree
// leaves, the body band, the hot-start helpers, and dynamic adaptive runs against uniform ones.
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_amr_common.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <tuple>

namespace {

using BCs = std::array<const char*, 6>;
using KV = std::vector<std::pair<const char*, const char*>>;
using Cell3 = std::tuple<int, int, int, int>;   // (level, i, j, k)

// Independent oracle of CellIndicator::evaluate on a 3-D tree: the cells of every level found by walking up from every leaf, the
// value of a split cell the mean of its eight children (recursive, in key order), the neighbour across a side the cell of the same
// level there or else the coarser leaf covering it (found by CellMesh::covering), the flags of every interface, the smoothing through
// the interface rule CellIndicator::diffuse. Returns xi and xi_parent per leaf.
void indicator_oracle(const CellMesh& m, const EOS& eos, const CellIndicator& ind, std::vector<double>& xi_leaf,
                      std::vector<double>& xi_parent) {
    const int M = m.max_level, N = m.num_leaves();
    std::vector<std::map<std::array<int, 3>, int>> index(static_cast<size_t>(M) + 1);   // per level: cell -> its number
    std::vector<std::vector<std::array<int, 3>>> cells(static_cast<size_t>(M) + 1);
    for (int c = 0; c < N; ++c)
        for (int l = m.level[c]; l >= 0; --l) {
            const int s = m.level[c] - l;
            const std::array<int, 3> x = {m.ci[c] >> s, m.cj[c] >> s, m.ck[c] >> s};
            auto& ix = index[static_cast<size_t>(l)];
            if (!ix.count(x)) {
                ix[x] = static_cast<int>(cells[static_cast<size_t>(l)].size());
                cells[static_cast<size_t>(l)].push_back(x);
            }
        }
    std::function<State(int, int, int, int)> value = [&](int l, int i, int j, int k) {
        const int leaf = m.find(l, i, j, k);
        if (leaf >= 0) return m.U[static_cast<size_t>(leaf)];
        State v[8];
        for (int q = 0; q < 8; ++q) v[q] = value(l + 1, 2 * i + (q & 1), 2 * j + ((q >> 1) & 1), 2 * k + ((q >> 2) & 1));
        State s;
        for (int q = 0; q < kNVar; ++q) s[q] = (((v[0][q] + v[1][q]) + (v[2][q] + v[3][q])) + ((v[4][q] + v[5][q]) + (v[6][q] + v[7][q]))) * 0.125;
        return s;
    };
    const auto quantities = [&](const State& u, double& rho, double& p, double& sp) {
        rho = u[0];
        const State w = eos.cons_to_prim(u);
        p = w[3];
        sp = std::sqrt(w[1] * w[1] + w[2] * w[2] + w[4] * w[4]);
    };
    const auto test = [&](double a, double b, double floor_value) {
        double vmin = std::min(std::abs(a), std::abs(b));
        if (vmin < ind.value_floor) vmin = floor_value;
        return std::abs(b - a) / vmin >= ind.criterion;
    };
    const auto flagged = [&](const State& a, const State& b) {
        double ra, pa, sa, rb, pb, sb;
        quantities(a, ra, pa, sa);
        quantities(b, rb, pb, sb);
        return (ind.use_density && test(ra, rb, ind.value_floor)) || (ind.use_pressure && test(pa, pb, ind.value_floor)) ||
               (ind.use_velocity && test(sa, sb, ind.velocity_floor));
    };
    const bool per[3] = {m.periodic_x, m.periodic_y, m.periodic_z};
    std::vector<std::vector<double>> xi(static_cast<size_t>(M) + 1);
    for (int l = 0; l <= M; ++l) {
        const auto& cl = cells[static_cast<size_t>(l)];
        const int n = static_cast<int>(cl.size()), n3[3] = {m.nx_at(l), m.ny_at(l), m.nz_at(l)};
        std::vector<double>& x = xi[static_cast<size_t>(l)];
        x.assign(static_cast<size_t>(n), 0.0);
        std::vector<State> val(static_cast<size_t>(n));
        for (int c = 0; c < n; ++c) val[static_cast<size_t>(c)] = value(l, cl[c][0], cl[c][1], cl[c][2]);
        std::vector<std::array<int, 2>> ifaces;
        for (int c = 0; c < n; ++c)
            for (int s = 0; s < 6; ++s) {
                std::array<int, 3> y = cl[static_cast<size_t>(c)];
                const int a = s / 2;
                y[a] += (s & 1) ? 1 : -1;
                if (y[a] < 0 || y[a] >= n3[a]) {
                    if (!per[a]) continue;
                    y[a] = (y[a] + n3[a]) % n3[a];
                }
                const auto it = index[static_cast<size_t>(l)].find(y);
                if (it != index[static_cast<size_t>(l)].end()) {
                    if (flagged(val[static_cast<size_t>(c)], val[static_cast<size_t>(it->second)])) x[c] = x[it->second] = 1.0;
                    if (s & 1) ifaces.push_back({c, it->second});   // each same-level pair once (from its lower cell)
                } else {
                    const int cov = m.covering(l, y[0], y[1], y[2]);
                    CHECK(cov >= 0 && m.level[cov] < l);
                    if (flagged(val[static_cast<size_t>(c)], m.U[static_cast<size_t>(cov)])) x[c] = 1.0;
                    ifaces.push_back({c, -1});
                }
            }
        ind.diffuse(x, ifaces);
    }
    xi_leaf.assign(static_cast<size_t>(N), 0.0);
    xi_parent.assign(static_cast<size_t>(N), 1e300);
    for (int c = 0; c < N; ++c) {
        const int l = m.level[c];
        xi_leaf[c] = xi[static_cast<size_t>(l)][static_cast<size_t>(index[static_cast<size_t>(l)].at({m.ci[c], m.cj[c], m.ck[c]}))];
        if (l > 0)
            xi_parent[c] = xi[static_cast<size_t>(l) - 1][static_cast<size_t>(
                index[static_cast<size_t>(l) - 1].at({m.ci[c] >> 1, m.cj[c] >> 1, m.ck[c] >> 1}))];
    }
}

// A 3-D blob and a step of density and pressure (the indicator has something to see on every axis)
State blob_state(const EOS& eos, double x, double y, double z) {
    const double r2 = (x - 0.9) * (x - 0.9) + (y - 0.5) * (y - 0.5) + (z - 0.45) * (z - 0.45);
    return eos.prim_to_cons({1.0 + 0.8 * std::exp(-30.0 * r2) + (x > 1.3 ? 0.3 : 0.0), 0.3, -0.2, 1.0 + 0.5 * std::exp(-20.0 * r2), 0.1});
}

}   // namespace

TEST(cellindicator3d_matches_a_brute_force_oracle) {
    // The 3-D indicator (6 sides, split cells holding the mean of 8 children, periodic z) against the oracle above on random
    // octrees and on a level-0 one, density + pressure + velocity, with and without periodic axes: xi and xi_parent of every leaf.
    const EOS eos = EOS::ideal(1.4);
    int flagged_total = 0;
    for (int cfg = 0; cfg < 4; ++cfg) {
        const bool periodic = cfg & 1;
        CellMesh m = CellMesh::make_3d(1.7, 1.1, 0.9, 8, 6, 5, cfg >= 2 ? 0 : 2, periodic, false, periodic);
        if (cfg < 2) random_adapt(m, 101u + static_cast<unsigned>(cfg), 3, 12, 20);
        m.U.resize(static_cast<size_t>(m.num_leaves()));
        for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = blob_state(eos, m.xc(c), m.yc(c), m.zc(c));
        CellIndicator ind;
        ind.use_pressure = true;
        ind.use_velocity = cfg == 1;
        const CellIndicator::Result r = ind.evaluate(m, m.U, eos);
        std::vector<double> xo, po;
        indicator_oracle(m, eos, ind, xo, po);
        double worst = 0.0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            worst = std::max(worst, std::abs(r.xi[c] - xo[c]));
            if (m.level[c] > 0) worst = std::max(worst, std::abs(r.xi_parent[c] - po[c]));
            else CHECK(r.xi_parent[c] == 1e300);
            flagged_total += r.xi[c] >= ind.split;
        }
        CHECK(worst < 1e-12);
        if (!(worst < 1e-12)) std::printf("    cfg %d: worst difference %.3e\n", cfg, worst);
    }
    CHECK(flagged_total > 50);
}

namespace {

struct Rig3 {
    Simulation sim;
    std::unique_ptr<CellMesh> m;
    std::unique_ptr<CellFluxEngine> eng;
    Rig3(const BCs& bc, KV kv, int nx, int ny, int nz, int max_level) {
        const std::string sx = std::to_string(nx), sy = std::to_string(ny), sz = std::to_string(nz);
        kv.push_back({"nx", sx.c_str()});
        kv.push_back({"ny", sy.c_str()});
        kv.push_back({"nz", sz.c_str()});
        setup_3d(sim, bc, kv);
        m = std::make_unique<CellMesh>(octree_of(sim, max_level));
        eng = std::make_unique<CellFluxEngine>(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        eng->set_physics(physics_of(sim));
    }
};

}   // namespace

TEST(cellremesh3d_step_is_stable_conservative_and_deterministic) {
    // init_adapt of a blob and a step, twice: the same leaves and states (deterministic); remeshing the static field again changes
    // nothing; the field moves, the remesh follows it and the five totals are conserved (joins average, refines inject).
    const BCs per = {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"};
    const auto make = [&](std::unique_ptr<Rig3>& r, std::unique_ptr<CellAdaptor>& ad, double shift) {
        // (8 x 6 x 5: on a smaller base the 26-neighbour balance refines every base cell and level 0 is empty)
        r = std::make_unique<Rig3>(per, KV{{"reconstruction", "muscl"}, {"Lx", "1.7"}, {"Ly", "1.1"}, {"Lz", "0.9"}}, 8, 6, 5, 2);
        CellIndicator ind;
        ind.use_pressure = true;
        ad = std::make_unique<CellAdaptor>(*r->m, *r->eng, r->sim.fluid, ind);
        const auto init = [&, shift](CellMesh& mm) {
            mm.U.resize(static_cast<size_t>(mm.num_leaves()));
            for (int c = 0; c < mm.num_leaves(); ++c) mm.U[c] = blob_state(r->sim.fluid, mm.xc(c) - shift, mm.yc(c), mm.zc(c));
        };
        init(*r->m);
        r->eng->build(*r->m);
        return ad->init_adapt(init);
    };
    std::unique_ptr<Rig3> r1, r2;
    std::unique_ptr<CellAdaptor> a1, a2;
    const int p1 = make(r1, a1, 0.0), p2 = make(r2, a2, 0.0);
    CellMesh& m1 = *r1->m;
    CHECK(p1 >= 2 && p1 == p2);
    CHECK(m1.ci == r2->m->ci && m1.cj == r2->m->cj && m1.ck == r2->m->ck && m1.level == r2->m->level && m1.U == r2->m->U);
    CHECK(m1.level_begin[3] > m1.level_begin[2] && m1.level_begin[1] > 0);   // three levels
    const std::vector<int> ci = m1.ci, cj = m1.cj, ck = m1.ck, lv = m1.level;
    const Field U = m1.U;
    CHECK(!a1->remesh());   // a static field: nothing to change
    CHECK(m1.ci == ci && m1.cj == cj && m1.ck == ck && m1.level == lv && m1.U == U);
    // the field moves: the remesh follows it and conserves the totals
    for (int c = 0; c < m1.num_leaves(); ++c) m1.U[c] = blob_state(r1->sim.fluid, m1.xc(c) - 0.25, m1.yc(c), m1.zc(c));
    double before[kNVar];
    for (int q = 0; q < kNVar; ++q) before[q] = cell_total_nd(m1, q);
    bool changed = false;
    for (int k = 0; k < 4; ++k) changed = a1->remesh() || changed;
    CHECK(changed);
    for (int q = 0; q < kNVar; ++q) CHECK_NEAR(cell_total_nd(m1, q), before[q], 1e-12 * std::max(1.0, std::abs(before[q])));
    // (the topology after adapt is checked by the brute-force oracle of Q1, which is quadratic in the leaves: 25 s at this size)
    // the engine was rebuilt with the leaves: its tables equal a fresh build
    CellFluxEngine fresh(r1->sim.fluid, r1->sim.riemann, r1->sim.muscl.get(), r1->sim.boundaries);
    fresh.build(m1);
    CHECK(r1->eng->same_tables(fresh));
    std::printf("    %d leaves after init_adapt (%d passes), %d after the field moved\n", static_cast<int>(lv.size()), p1, m1.num_leaves());
}

TEST(cellremesh3d_forced_box_refines_the_leaves_it_overlaps_and_only_those) {
    const BCs walls = {"wall", "wall", "wall", "wall", "wall", "wall"};
    Rig3 r(walls, {{"Lx", "2"}, {"Ly", "1"}, {"Lz", "1"}}, 8, 4, 4, 2);
    CellMesh& m = *r.m;
    CellIndicator ind;
    CellAdaptor ad(m, *r.eng, r.sim.fluid, ind);
    const double b[6] = {0.5, 0.8, 0.25, 0.5, 0.3, 0.6};
    ad.set_force_box3(b[0], b[1], b[2], b[3], b[4], b[5]);
    const auto fill = [&](CellMesh& mm) { mm.U.assign(static_cast<size_t>(mm.num_leaves()), r.sim.fluid.prim_to_cons({1.0, 0.0, 0.0, 1.0, 0.0})); };
    fill(m);
    r.eng->build(m);
    for (int k = 0; k <= 3; ++k) {
        ad.remesh();
        fill(m);
    }
    int overlapping = 0, far_coarse = 0;
    for (int c = 0; c < m.num_leaves(); ++c) {
        const int l = m.level[c];
        const double lo[3] = {m.xc(c) - 0.5 * m.dx(l), m.yc(c) - 0.5 * m.dy(l), m.zc(c) - 0.5 * m.dz(l)};
        const double hi[3] = {m.xc(c) + 0.5 * m.dx(l), m.yc(c) + 0.5 * m.dy(l), m.zc(c) + 0.5 * m.dz(l)};
        bool in = true;
        for (int a = 0; a < 3; ++a) in = in && hi[a] > b[2 * a] && lo[a] < b[2 * a + 1];
        if (in) {
            ++overlapping;
            CHECK(l == 2);
        }
        far_coarse += lo[0] > 1.2 && l == 0;
    }
    CHECK(overlapping > 0);
    CHECK(far_coarse > 0);
    check_cell_mesh_nd(m);
}

TEST(cellremesh3d_keeps_the_body_band_on_the_finest_level) {
    // A sphere and an extruded NACA section in a uniform flow: after init_adapt every leaf within the band (the distance of its centre
    // to the wall within its half diagonal plus `margin` finest widths) is at the finest level and nothing else is refined; with
    // margin 6 the ghost-cell wall of the engine is ready (the band and its readiness rule agree), and a remesh keeps all of it.
    for (const char* shape : {"Sphere", "NACA0012"})
        for (const int margin : {2, 6}) {
            Rig3 r({"vel_inlet", "outlet", "wall", "wall", "periodic", "periodic"},
                   {{"use_ibm", "true"}, {"ibm_method", margin == 6 ? "ghost" : "mask"}, {"ibm_shape", shape}, {"ibm_radius", "0.27"},
                    {"ibm_cx", "0.83"}, {"ibm_cy", "0.52"}, {"ibm_cz", "0.47"}, {"ibm_chord", "0.9"}, {"ibm_aoa", "12"},
                    {"reconstruction", "muscl"}},
                   10, 6, 5, 2);
            CellPhysics ph = physics_of(r.sim);
            ph.ibm_ghost = margin == 6;
            ph.ibm_slip = true;
            r.eng->set_physics(ph);
            CellMesh& m = *r.m;
            CellIndicator ind;
            CellAdaptor ad(m, *r.eng, r.sim.fluid, ind);
            ad.set_body(r.sim.ibm_factory, margin);
            const auto init = [&](CellMesh& mm) { mm.U.assign(static_cast<size_t>(mm.num_leaves()), r.sim.fluid.prim_to_cons({1.0, 0.5, 0.0, 1.0, 0.0})); };
            init(m);
            r.eng->build(m);
            const int passes = ad.init_adapt(init);
            const ImmersedBoundary geo = r.sim.ibm_factory(r.sim.mesh);
            {   // init_adapt reached the fixed point: no band leaf below the finest level (a band rule that is not consistent between a
                // parent and its children joins and refines the same leaves for ever and fails here)
                int missing = 0;
                for (int c = 0; c < m.num_leaves(); ++c) missing += ad.in_body_band(c) && m.level[c] < m.max_level;
                CHECK(missing == 0);
                CHECK(passes <= m.max_level + 1);
            }
            const int M = m.max_level;
            const double hm = std::max({m.dx(M), m.dy(M), m.dz(M)});
            int band = 0, outside_fine = 0;
            const auto check = [&]() {
                band = outside_fine = 0;
                // brute force: every finest cell under the leaf, its centre's distance to the wall
                const auto dist = [&](double x, double y, double z) {
                    double wx, wy, wz, nx, ny, nz;
                    geo.closest_wall_point3(x, y, z, wx, wy, wz, nx, ny, nz);
                    return std::sqrt((x - wx) * (x - wx) + (y - wy) * (y - wy) + (z - wz) * (z - wz));
                };
                for (int c = 0; c < m.num_leaves(); ++c) {
                    const int l = m.level[c], w = 1 << (M - l);
                    bool near = false;
                    for (int kk = 0; kk < w && !near; ++kk)
                        for (int jj = 0; jj < w && !near; ++jj)
                            for (int ii = 0; ii < w && !near; ++ii)
                                near = dist((m.ci[c] * w + ii + 0.5) * m.dx(M) + m.x0, (m.cj[c] * w + jj + 0.5) * m.dy(M) + m.y0,
                                            (m.ck[c] * w + kk + 0.5) * m.dz(M) + m.z0) <= margin * hm;
                    if (near) {
                        ++band;
                        CHECK(l == M);
                    } else if (l == M && dist(m.xc(c), m.yc(c), m.zc(c)) > (margin + 4) * hm) {
                        ++outside_fine;   // finest leaves far from the band: only the 2:1 balance could put them there
                    }
                }
            };
            check();
            CHECK(band > 100);
            CHECK(outside_fine == 0);
            if (margin == 6) CHECK(r.eng->ghost_ready());
            CHECK(!ad.remesh());
            check();
            std::printf("    %s, margin %d: %d leaves, %d in the band%s\n", shape, margin, m.num_leaves(), band,
                        margin == 6 ? (r.eng->ghost_ready() ? ", ghost wall ready" : ", ghost wall NOT ready") : "");
        }
}

TEST(hot_start_helpers_3d) {
    // GridInterp3D has the arithmetic of map_solution_3d (the same bits at the fine centres); leaves_to_uniform_3d writes every leaf
    // onto the block of finest cells it covers (totals kept) and rejects a set of leaves that does not cover the domain.
    const EOS eos = EOS::ideal(1.4);
    const int nc[3] = {5, 4, 3}, nf[3] = {11, 7, 6};
    const double L[3] = {1.7, 1.1, 0.9};
    Field Uc(static_cast<size_t>(nc[0]) * nc[1] * nc[2]);
    for (size_t c = 0; c < Uc.size(); ++c) Uc[c] = blob_state(eos, 0.1 * static_cast<double>(c % 7), 0.2 * static_cast<double>(c % 5), 0.3);
    const Field Uf = map_solution_3d(Uc, nc[0], nc[1], nc[2], nf[0], nf[1], nf[2], L[0], L[1], L[2]);
    const auto centres = [](double l, int n) { return linspace(l / n / 2.0, l - l / n / 2.0, n); };
    const GridInterp3D gi(centres(L[0], nc[0]), centres(L[1], nc[1]), centres(L[2], nc[2]), Uc);
    const auto xf = centres(L[0], nf[0]), yf = centres(L[1], nf[1]), zf = centres(L[2], nf[2]);
    bool same = true;
    for (int k = 0; k < nf[2]; ++k)
        for (int j = 0; j < nf[1]; ++j)
            for (int i = 0; i < nf[0]; ++i) same = same && gi(xf[i], yf[j], zf[k]) == Uf[(static_cast<size_t>(k) * nf[1] + j) * nf[0] + i];
    CHECK(same);
    // leaves -> uniform
    CellMesh m = CellMesh::make_3d(L[0], L[1], L[2], 4, 3, 2, 2);
    random_adapt(m, 7u, 3);
    m.U.resize(static_cast<size_t>(m.num_leaves()));
    for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = blob_state(eos, m.xc(c), m.yc(c), m.zc(c));
    int nx, ny, nz;
    const Field u = leaves_to_uniform_3d(m.U, m.level, m.ci, m.cj, m.ck, m.nx, m.ny, m.nz, nx, ny, nz);
    CHECK(nx == 16 && ny == 12 && nz == 8);
    const double v = (L[0] / nx) * (L[1] / ny) * (L[2] / nz);
    double tot = 0.0;
    for (const State& s : u) tot += s[0] * v;
    CHECK_NEAR(tot, cell_total_nd(m, 0), 1e-12 * tot);
    for (int c = 0; c < m.num_leaves(); ++c) {   // the cell at the leaf's corner holds the leaf's value
        const int w = 1 << (2 - m.level[c]);
        CHECK(u[(static_cast<size_t>(m.ck[c] * w) * ny + m.cj[c] * w) * nx + m.ci[c] * w] == m.U[c]);
    }
    std::vector<int> lv = m.level;
    lv.pop_back();
    Field U2(m.U.begin(), m.U.end() - 1);
    std::vector<int> i2(m.ci.begin(), m.ci.end() - 1), j2(m.cj.begin(), m.cj.end() - 1), k2(m.ck.begin(), m.ck.end() - 1);
    bool threw = false;
    try {
        leaves_to_uniform_3d(U2, lv, i2, j2, k2, m.nx, m.ny, m.nz, nx, ny, nz);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(cellremesh3d_dynamic_sod_follows_the_waves) {
    // Bands fixed before the first run (those of the 2-D test): a Sod tube along x of base 50 x 2 x 2 (periodic transverse axes) with
    // two levels and the indicator, remeshed every 4 coarse steps, against the exact density at t = 0.2: error within 1.6 x the
    // uniform-200 one and below the uniform-50 one, a mean leaf count below 0.6 x the 200 x 8 x 8 of the finest level, the shock region on
    // the finest level at the end, mass conserved to 1e-12.
    const BCs bc = {"transmissive", "transmissive", "periodic", "periodic", "periodic", "periodic"};
    const auto run = [&](int nx, int max_level, double& mean_leaves, int& shock_level) {
        const int nt = nx / 25;   // square cells: 0.04 across at 50
        Rig3 r(bc, {{"Lx", "1"}, {"Ly", "0.04"}, {"Lz", "0.04"}, {"reconstruction", "muscl"}}, nx, nt, nt, max_level);
        CellMesh& m = *r.m;
        CellFluxEngine& eng = *r.eng;
        CellIndicator ind;
        CellAdaptor ad(m, eng, r.sim.fluid, ind);
        const auto init = [&](CellMesh& mm) {
            mm.U.resize(static_cast<size_t>(mm.num_leaves()));
            for (int c = 0; c < mm.num_leaves(); ++c)
                mm.U[c] = mm.xc(c) < 0.5 ? r.sim.fluid.prim_to_cons({1.0, 0.0, 0.0, 1.0, 0.0}) : r.sim.fluid.prim_to_cons({0.125, 0.0, 0.0, 0.1, 0.0});
        };
        init(m);
        eng.build(m);
        if (max_level > 0) ad.init_adapt(init);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4);
        CellAdvance adv(m, eng, *ts);
        const double mass0 = cell_total_nd(m, 0);
        double S = eng.wave_speed(m, m.U), t = 0.0, leaf_sum = 0.0;
        int steps = 0;
        while (t < 0.2) {
            const double dt = std::min(0.4 * m.dx(0) / S, 0.2 - t);
            S = adv.step(dt, t);
            t += dt;
            ++steps;
            leaf_sum += m.num_leaves();
            if (max_level > 0 && steps % 4 == 0 && ad.remesh()) S = eng.wave_speed(m, m.U);
        }
        CHECK_NEAR(cell_total_nd(m, 0), mass0, 1e-12 * mass0);
        mean_leaves = leaf_sum / steps;
        shock_level = 0;
        for (int c = 0; c < m.num_leaves(); ++c)
            if (std::abs(m.xc(c) - 0.8528) < 0.02) shock_level = std::max(shock_level, m.level[c]);
        double err = 0.0, vol = 0.0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            err += std::abs(m.U[c][0] - sod_exact_density(m.xc(c), 0.2)) * m.volume(c);
            vol += m.volume(c);
        }
        return err / vol;
    };
    double lc = 0, lf = 0, la = 0;
    int sc = 0, sf = 0, sa = 0;
    const double ec = run(50, 0, lc, sc), ef = run(200, 0, lf, sf), ea = run(50, 2, la, sa);
    std::printf("    3-D dynamic Sod L1(rho): uniform 50 %.4e, uniform 200 %.4e, adaptive %.4e (mean %.0f leaves of %d, shock level %d)\n", ec, ef, ea,
                la, 200 * 8 * 8, sa);
    CHECK(ea < 1.6 * ef);
    CHECK(ea < ec);
    CHECK(la < 0.6 * 200 * 8 * 8);
    CHECK(sa == 2);
}

TEST(cellremesh3d_taylor_green_against_uniform) {
    // The viscous 3-D Taylor-Green vortex at t = 1 on base 12^3 with one level (density + velocity indicator, remeshed every 4 steps)
    // against the uniform 24^3 run; the velocity difference to the uniform 24^3 (volume-weighted L2, the 24^3 cells averaged onto each
    // leaf). The Taylor-Green vortex is active everywhere: the Q0 estimate (section 11) put its octree at 0.99 of the uniform cells, a
    // correctness case, not an economy one. History, for the record: the first version expected the indicator to refine part of the
    // domain (and held a band on the uniform runs' kinetic energy that was wrong as a measure: 12^3 is 3.5 % off); the default criterion
    // refines every leaf, and so do 0.2, 0.4 and 0.8 (the velocity floor near the many stagnation points, then the smoothing), as
    // section 11 predicted. The checks of this version: (a) with the indicator every leaf is refined and the run equals the uniform
    // 24^3 one up to its other time steps (difference below 1e-3 x the uniform 12^3 one); (b) a static box over half the domain
    // (amr_force_box, no indicator) beats the uniform 12^3 with at most 0.6 x the 24^3 leaves: level jumps through a vortical flow.
    const char* name = "Taylor-Green Vortex 3D";
    double criterion = 0.1;
    bool half_box = false;
    const auto run = [&](int n, int max_level, Field& out, CellMesh*& keep, double& mean_leaves) {
        const std::string ns = std::to_string(n);
        Simulation* sim = new Simulation;
        sim->setup(params_for(name, {{"nx", ns.c_str()}, {"ny", ns.c_str()}, {"nz", ns.c_str()}, {"use_mpi", "false"}}), false);
        CellMesh* m = new CellMesh(octree_of(*sim, max_level));
        CellFluxEngine eng(sim->fluid, sim->riemann, sim->muscl.get(), sim->boundaries);
        eng.set_physics(physics_of(*sim));
        CellIndicator ind;
        ind.use_velocity = true;
        ind.criterion = half_box ? 1e300 : criterion;
        CellAdaptor ad(*m, eng, sim->fluid, ind);
        if (half_box) ad.set_force_box3(-1.0, 0.5 * m->Lx + m->x0, -1e300, 1e300, -1e300, 1e300);
        const auto init = [&](CellMesh& mm) {
            mm.U.assign(static_cast<size_t>(mm.num_leaves()), State{});
            for (int l = 0; l <= mm.max_level; ++l) {
                if (mm.level_begin[l + 1] == mm.level_begin[l]) continue;
                const Mesh ml = generate_cartesian_mesh_3d(mm.Lx, mm.Ly, mm.Lz, mm.nx_at(l), mm.ny_at(l), mm.nz_at(l), mm.x0, mm.y0, mm.z0);
                const Field F = get_test_case(name).initialize(ml, sim->fluid, sim->p.source);
                for (int c = mm.level_begin[l]; c < mm.level_begin[l + 1]; ++c) mm.U[c] = F[ml.cell(mm.ci[c], mm.cj[c], mm.ck[c])];
            }
        };
        init(*m);
        eng.build(*m);
        if (max_level > 0) ad.init_adapt(init);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(sim->p.time_scheme), sim->p.cfl);
        CellAdvance adv(*m, eng, *ts);
        const double L = 2.0 / (1.0 / m->dx(0) + 1.0 / m->dy(0) + 1.0 / m->dz(0));
        double S = eng.wave_speed(*m, m->U), t = 0.0, sum = 0.0;
        int steps = 0;
        while (t < 1.0) {
            const double dt = std::min(sim->p.cfl * L / S, 1.0 - t);
            S = adv.step(dt, t);
            t += dt;
            ++steps;
            sum += m->num_leaves();
            if (max_level > 0 && steps % 4 == 0 && ad.remesh()) S = eng.wave_speed(*m, m->U);
        }
        mean_leaves = sum / steps;
        out = m->U;
        keep = m;
        delete sim;
    };
    Field Uc, Uf;
    CellMesh *mc = nullptr, *mf = nullptr;
    double lc, lf;
    run(12, 0, Uc, mc, lc);
    run(24, 0, Uf, mf, lf);
    const EOS eos = EOS::ideal(1.4);
    // the uniform 24^3 velocity averaged onto a leaf (level 0: 8 cells, level 1: its own cell)
    const auto fine_vel = [&](const CellMesh& m, int c, int a) {
        const int w = m.level[c] == 0 ? 2 : 1;
        double s = 0.0;
        for (int dk = 0; dk < w; ++dk)
            for (int dj = 0; dj < w; ++dj)
                for (int di = 0; di < w; ++di) {
                    const int f = mf->find(0, m.ci[c] * w + di, m.cj[c] * w + dj, m.ck[c] * w + dk);
                    s += eos.cons_to_prim(Uf[static_cast<size_t>(f)])[kMom[a]];
                }
        return s / (w * w * w);
    };
    const auto diff = [&](const CellMesh& m, const Field& U) {
        double e = 0.0, v = 0.0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            const State w = eos.cons_to_prim(U[c]);
            for (int a = 0; a < 3; ++a) {
                const double d = w[kMom[a]] - fine_vel(m, c, a);
                e += d * d * m.volume(c);
            }
            v += m.volume(c);
        }
        return std::sqrt(e / v);
    };
    const auto ke = [&](const CellMesh& m, const Field& U) {
        double k = 0.0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            const State w = eos.cons_to_prim(U[c]);
            k += 0.5 * w[0] * (w[1] * w[1] + w[2] * w[2] + w[4] * w[4]) * m.volume(c);
        }
        return k;
    };
    const double dc = diff(*mc, Uc), kc = ke(*mc, Uc), kf = ke(*mf, Uf);
    std::printf("    3-D Taylor-Green at t = 1, velocity L2 difference to uniform 24^3: uniform 12^3 %.4e (KE 12^3 %.5f, 24^3 %.5f)\n", dc, kc, kf);
    for (int pass = 0; pass < 2; ++pass) {
        half_box = pass == 1;
        Field Ua;
        CellMesh* ma = nullptr;
        double la;
        run(12, 1, Ua, ma, la);
        const double da = diff(*ma, Ua), ka = ke(*ma, Ua);
        std::printf("      %s: adaptive %.4e (%.3e of the 12^3 one), mean %.0f leaves (%.2f of 24^3), KE %.5f\n",
                    half_box ? "half box forced" : "indicator (criterion 0.1)", da, da / dc, la, la / (24.0 * 24 * 24), ka);
        if (half_box) {
            CHECK(da < dc);
            CHECK(la <= 0.6 * 24 * 24 * 24);
        } else {
            CHECK(la == 24.0 * 24 * 24);
            CHECK(da < 1e-3 * dc);
        }
        delete ma;
    }
    delete mc;
    delete mf;
}
