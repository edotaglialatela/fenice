// AMR in three dimensions, phase Q3 (docs/design/amr-octree.md): CellAdvance (subcycling with flux replacement) on octree leaves.
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_amr_common.hpp"

#include <algorithm>
#include <string>

namespace {

using BCs = std::array<const char*, 6>;

// The time-step length of the 3-D uniform solver (Simulation::dt_length) at level 0 of a mesh: 2 / (1/dx + 1/dy + 1/dz), min(dx, dy) in 2-D
double dt_length0(const CellMesh& m) {
    return m.dim == 3 ? 2.0 / (1.0 / m.dx(0) + 1.0 / m.dy(0) + 1.0 / m.dz(0)) : std::min(m.dx(0), m.dy(0));
}

// The state of the 3-D uniform mesh of `sim`, filled with test_state3, and the level-0 octree holding the same values
void level0_pair(const Simulation& sim, int max_level, Field& Uu, CellMesh& cm) {
    const Mesh& u = sim.mesh;
    Uu.assign(static_cast<size_t>(u.num_cells()), State{});
    for (int k = 0; k < u.nz; ++k)
        for (int j = 0; j < u.ny; ++j)
            for (int i = 0; i < u.nx; ++i) Uu[u.cell(i, j, k)] = test_state3(sim.fluid, u.xc(i), u.yc(j), u.zc(k));
    cm = octree_of(sim, max_level);
    cm.U.resize(static_cast<size_t>(cm.num_leaves()));
    for (int c = 0; c < cm.num_leaves(); ++c) cm.U[c] = Uu[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
}

// Advance the 3-D uniform solver and the level-0 octree by the same steps; true if every leaf equals its cell bit for bit
bool level0_steps_equal(Simulation& sim, int max_level, int nsteps) {
    Field Uu;
    CellMesh cm = octree_of(sim, 0);
    level0_pair(sim, max_level, Uu, cm);
    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    eng.build(cm);
    std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(sim.p.time_scheme), 0.4);
    CellAdvance adv(cm, eng, *ts);
    const double dt = 2e-3;
    const ResidualFn res = [&](const Field& Uin, Field& R) { return sim.residual(Uin, R); };
    double Su = 0.0, Sc = 0.0;
    for (int k = 0; k < nsteps; ++k) {
        Su = sim.scheme->advance_fixed_dt(Uu, res, dt);
        Sc = adv.step(dt, k * dt);
    }
    const Mesh& u = sim.mesh;
    bool same = Su == Sc;
    for (int c = 0; c < cm.num_leaves(); ++c)
        for (int q = 0; q < kNVar; ++q) same = same && cm.U[c][q] == Uu[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])][q];
    return same;
}

// A random octree on a fully periodic 8 x 6 x 6 base, max_level 2, with the given reconstruction. (On the 4 x 3 x 3 base of the Q2
// rig the 26-neighbour balance refines every base cell: level 0 is empty and only one level jump is nested.)
CellMesh periodic_octree(Simulation& sim, const char* recon, const char* scheme, unsigned seed) {
    const BCs per = {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"};
    setup_3d(sim, per, {{"nx", "8"}, {"ny", "6"}, {"nz", "6"}, {"reconstruction", recon}, {"muscl_limiter", "Van Leer"},
                        {"time_scheme", scheme}});
    CellMesh m = octree_of(sim, 2);
    random_adapt(m, seed, 3, 6, 30);
    return m;
}

int coarse_fine_faces(const CellMesh& m, int axis) {
    int n = 0;
    for (const CellFace& f : m.faces) n += f.kind == CellFaceKind::CoarseFine && f.axis == axis;
    return n;
}

}   // namespace

TEST(celladvance3d_level0_steps_match_uniform_steps_bitwise) {
    // A level-0 octree advanced by CellAdvance (max_level 0, and 2: the recursion with nothing to recurse into) against the 3-D
    // uniform solver's own time scheme, bit for bit (state and returned speed), for the three schemes, first order, MUSCL,
    // WENO3-Z and boundary_order 1 / 2.
    const BCs sets[] = {{"wall", "outlet", "no_slip", "moving_wall", "transmissive", "wall"},
                        {"periodic", "periodic", "wall", "wall", "transmissive", "no_slip"},
                        {"vel_inlet", "transmissive", "periodic", "periodic", "periodic", "periodic"}};
    struct Scheme {
        const char *recon, *bo;
        int sc_lo, sc_hi;   // time schemes kSchemeNames[sc_lo .. sc_hi]
    };
    const Scheme schemes[] = {{"none", "1", 0, 2}, {"muscl", "1", 0, 2}, {"muscl", "2", 0, 2}, {"weno3", "1", 2, 2}, {"weno3", "2", 2, 2}};
    int n_cases = 0;
    for (const BCs& bc : sets)
        for (const Scheme& s : schemes)
            for (int sc = s.sc_lo; sc <= s.sc_hi; ++sc)
                for (int max_level : {0, 2}) {
                    Simulation sim;
                    setup_3d(sim, bc, {{"reconstruction", s.recon}, {"boundary_order", s.bo}, {"time_scheme", kSchemeNames[sc]}});
                    const bool same = level0_steps_equal(sim, max_level, 4);
                    CHECK(same);
                    if (!same)
                        std::printf("    bc %s/%s/%s/%s/%s/%s %s bo %s %s max_level %d\n", bc[0], bc[1], bc[2], bc[3], bc[4], bc[5], s.recon,
                                    s.bo, kSchemeNames[sc], max_level);
                    ++n_cases;
                }
    CHECK(n_cases == 66);
    // above kOmpMinWork the loops of the engine, the registers and the time scheme run in parallel: same bits
    for (const char* recon : {"muscl", "weno3"}) {
        Simulation sim;
        setup_3d(sim, BCs{"wall", "outlet", "periodic", "periodic", "no_slip", "transmissive"},
                 {{"nx", "24"}, {"ny", "20"}, {"nz", "10"}, {"reconstruction", recon}, {"muscl_limiter", "MC"}, {"boundary_order", "2"},
                  {"time_scheme", "SSP Runge-Kutta 3"}});
        CHECK(sim.mesh.num_cells() >= kOmpMinWork);
        CHECK(level0_steps_equal(sim, 2, 3));
    }
}

TEST(celladvance3d_fully_refined_mesh_equals_the_finer_uniform_mesh) {
    // Every leaf at level 1 (level 0 empty): the same arithmetic as the level-0 octree of 2n^3 stepped with two substeps of dt / 2.
    struct Cfg {
        BCs bc;
        const char *recon, *bo, *scheme;
    };
    const Cfg cfgs[] = {{{"wall", "outlet", "no_slip", "wall", "transmissive", "wall"}, "muscl", "2", "SSP Runge-Kutta 3"},
                        {{"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"}, "weno3", "1", "SSP Runge-Kutta 3"},
                        {{"transmissive", "transmissive", "wall", "wall", "periodic", "periodic"}, "none", "1", "Explicit Euler"}};
    for (const Cfg& cf : cfgs) {
        Simulation coarse, fine;
        setup_3d(coarse, cf.bc, {{"nx", "4"}, {"ny", "3"}, {"nz", "3"}, {"reconstruction", cf.recon}, {"boundary_order", cf.bo}, {"time_scheme", cf.scheme}});
        setup_3d(fine, cf.bc, {{"nx", "8"}, {"ny", "6"}, {"nz", "6"}, {"reconstruction", cf.recon}, {"boundary_order", cf.bo}, {"time_scheme", cf.scheme}});
        CellMesh amr = octree_of(coarse, 1), uni = octree_of(fine, 0);
        amr.adapt(std::vector<char>(static_cast<size_t>(amr.num_leaves()), 1));
        CHECK(amr.num_leaves() == uni.num_leaves() && amr.level_begin[1] == 0);
        CHECK(amr.dx(1) == uni.dx(0) && amr.dy(1) == uni.dy(0) && amr.dz(1) == uni.dz(0));
        uni.U.resize(static_cast<size_t>(uni.num_leaves()));
        for (int c = 0; c < uni.num_leaves(); ++c) uni.U[c] = test_state3(fine.fluid, uni.xc(c), uni.yc(c), uni.zc(c));
        amr.U.resize(static_cast<size_t>(amr.num_leaves()));
        std::vector<int> to_uni(static_cast<size_t>(amr.num_leaves()));
        for (int c = 0; c < amr.num_leaves(); ++c) {
            to_uni[c] = uni.find(0, amr.ci[c], amr.cj[c], amr.ck[c]);
            amr.U[c] = uni.U[static_cast<size_t>(to_uni[c])];
        }
        CellFluxEngine ea(coarse.fluid, coarse.riemann, coarse.muscl.get(), coarse.boundaries),
            eu(fine.fluid, fine.riemann, fine.muscl.get(), fine.boundaries);
        ea.build(amr);
        eu.build(uni);
        std::unique_ptr<TimeScheme> ta = make_time_scheme(parse_time_scheme(cf.scheme), 0.4), tu = make_time_scheme(parse_time_scheme(cf.scheme), 0.4);
        CellAdvance aa(amr, ea, *ta), au(uni, eu, *tu);
        const double dt = 2e-3;
        for (int k = 0; k < 3; ++k) {
            aa.step(dt, k * dt);
            au.step(dt / 2, k * dt);
            au.step(dt / 2, k * dt + dt / 2);
        }
        bool same = true;
        for (int c = 0; c < amr.num_leaves(); ++c) same = same && amr.U[c] == uni.U[static_cast<size_t>(to_uni[c])];
        CHECK(same);
        if (!same) std::printf("    %s bo %s %s: the fully refined octree differs from the finer uniform one\n", cf.recon, cf.bo, cf.scheme);
    }
}

TEST(celladvance3d_conserves_across_level_jumps_and_the_control_fails) {
    // Fully periodic random octree, three levels, coarse-fine faces normal to all three axes: the totals of the five conserved
    // quantities (rho w included) stay constant to roundoff with flux replacement, and drift without it (negative control).
    struct Cfg {
        const char* recon;
        int sc;
    };
    const Cfg cfgs[] = {{"none", 0}, {"muscl", 0}, {"muscl", 1}, {"muscl", 2}, {"weno3", 2}};
    for (const Cfg& cf : cfgs) {
        Simulation sim;
        const CellMesh m = periodic_octree(sim, cf.recon, kSchemeNames[cf.sc], 41u + static_cast<unsigned>(cf.sc));
        CHECK(m.level_begin[1] > 0 && m.level_begin[2] > m.level_begin[1] && m.level_begin[3] > m.level_begin[2]);   // three levels
        std::printf("    levels: %d %d %d of %d leaves; coarse-fine faces x/y/z %d %d %d\n", m.level_begin[1] - m.level_begin[0],
                    m.level_begin[2] - m.level_begin[1], m.level_begin[3] - m.level_begin[2], m.num_leaves(), coarse_fine_faces(m, 0),
                    coarse_fine_faces(m, 1), coarse_fine_faces(m, 2));
        for (int a = 0; a < 3; ++a) CHECK(coarse_fine_faces(m, a) > 50);
        const auto run = [&](bool replace) {
            CellMesh mm = m;
            mm.U.resize(static_cast<size_t>(mm.num_leaves()));
            for (int c = 0; c < mm.num_leaves(); ++c) mm.U[c] = test_state3(sim.fluid, mm.xc(c), mm.yc(c), mm.zc(c));
            CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
            eng.build(mm);
            std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(kSchemeNames[cf.sc]), 0.4);
            CellAdvance adv(mm, eng, *ts);
            adv.flux_replacement = replace;
            double S = eng.wave_speed(mm, mm.U), before[kNVar], mag[kNVar], worst = 0.0;
            for (int q = 0; q < kNVar; ++q) {
                before[q] = cell_total_nd(mm, q);
                mag[q] = cell_total_nd(mm, q, true);
            }
            for (int k = 0; k < 6; ++k) S = adv.step(0.4 * dt_length0(mm) / S, k * 0.01);
            for (int q = 0; q < kNVar; ++q) worst = std::max(worst, std::abs(cell_total_nd(mm, q) - before[q]) / mag[q]);
            return worst;
        };
        const double with = run(true), without = run(false);
        CHECK(with < 1e-13);
        CHECK(without > 1e-8);   // the register is what conserves
        std::printf("    %s %s: conservation error with replacement %.3e, without %.3e\n", cf.recon, kSchemeNames[cf.sc], with, without);
    }
}

TEST(celladvance3d_second_order_wall_faces_conserve_mass_and_energy_in_a_closed_box) {
    // Six slip walls with ghost layers (boundary_order = 2), three levels with jumps that meet the walls: no mass and no energy
    // cross a wall, so the totals stay constant to roundoff (momentum is not conserved: wall pressure).
    for (const char* recon : {"muscl", "weno3"}) {
        Simulation sim;
        setup_3d(sim, BCs{"wall", "wall", "wall", "wall", "wall", "wall"},
                 {{"nx", "8"}, {"ny", "6"}, {"nz", "6"}, {"reconstruction", recon}, {"boundary_order", "2"}, {"time_scheme", "SSP Runge-Kutta 3"}});
        CellMesh m = octree_of(sim, 2);
        random_adapt(m, 57u, 3, 6, 30);
        CHECK(m.level_begin[1] > 0 && m.level_begin[2] > m.level_begin[1] && m.level_begin[3] > m.level_begin[2]);   // three levels
        int wall_jumps = 0;   // leaves of the finest level on a wall
        for (int c = m.level_begin[2]; c < m.level_begin[3]; ++c)
            for (int s = 0; s < 6; ++s) wall_jumps += m.neighbour(c, static_cast<Face>(s)).kind == CellNeighbour::Kind::Boundary;
        CHECK(wall_jumps > 20);
        m.U.resize(static_cast<size_t>(m.num_leaves()));
        for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state3(sim.fluid, m.xc(c), m.yc(c), m.zc(c));
        CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        eng.build(m);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK3, 0.4);
        CellAdvance adv(m, eng, *ts);
        double S = eng.wave_speed(m, m.U);
        const double m0 = cell_total_nd(m, 0), e0 = cell_total_nd(m, 3);
        for (int k = 0; k < 8; ++k) S = adv.step(0.4 * dt_length0(m) / S, k * 0.01);
        const double dm = std::abs(cell_total_nd(m, 0) - m0) / m0, de = std::abs(cell_total_nd(m, 3) - e0) / e0;
        CHECK(dm < 1e-13);
        CHECK(de < 1e-13);
        std::printf("    %s: mass error %.3e, energy error %.3e\n", recon, dm, de);
    }
}

TEST(celladvance3d_free_stream_and_stage_times) {
    for (int sc = 0; sc < 3; ++sc) {
        Simulation sim;
        CellMesh m = periodic_octree(sim, "muscl", kSchemeNames[sc], 77u);
        CHECK(m.level_begin[3] > m.level_begin[2] && m.level_begin[2] > m.level_begin[1]);
        m.U.assign(static_cast<size_t>(m.num_leaves()), sim.fluid.prim_to_cons({1.3, 0.8, -0.45, 2.1, 0.35}));
        const Field U0 = m.U;
        CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        eng.build(m);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(kSchemeNames[sc]), 0.4);
        CellAdvance adv(m, eng, *ts);
        std::vector<std::pair<int, double>> seen;
        adv.stage_hook = [&](int l, double st) { seen.push_back({l, st}); };
        const double dt = 0.003, t0 = 0.125;
        for (int k = 0; k < 3; ++k) adv.step(dt, t0 + k * dt);
        double drift = 0.0;
        for (int c = 0; c < m.num_leaves(); ++c)
            for (int q = 0; q < kNVar; ++q) drift = std::max(drift, std::abs(m.U[c][q] - U0[c][q]));
        if (sc < 2)
            CHECK(drift == 0.0);   // zero residual in every stage: bitwise unchanged
        else
            CHECK(drift < 1e-14);   // the stage weights of SSP-RK3 sum to 1 only up to rounding
        const std::vector<double> off = ts->stage_offsets();
        std::vector<std::pair<int, double>> want;
        std::function<void(int, double, double)> emit = [&](int l, double d, double t) {
            if (l < m.max_level && m.level_begin[l + 2] > m.level_begin[l + 1])
                for (int sub = 0; sub < 2; ++sub) emit(l + 1, d / 2, t + sub * (d / 2));
            if (m.level_begin[l + 1] > m.level_begin[l])
                for (double o : off) want.push_back({l, o == 0.0 ? t : (o == 1.0 ? t + d : t + o * d)});
        };
        for (int k = 0; k < 3; ++k) emit(0, dt, t0 + k * dt);
        CHECK(seen == want);
        CHECK(seen.size() > 3 * off.size() * 3);
    }
}

TEST(celladvance3d_invariant_axis_run_equals_the_2d_engine) {
    // The run version of cellflux3d_invariant_axis_equals_the_2d_engine: a field independent of axis a on the 2-D adapted tree
    // extruded along a (periodic), advanced with the same fixed dt by the 3-D and the 2-D CellAdvance; after 8 steps the states
    // agree to round-off (relative 1e-12 of each component's scale; 4 registers per coarse side against 2, other face areas) and
    // the a-momentum is exactly 0. Three axes, the three schemes.
    struct Scheme {
        const char *recon, *bo;
        int sc;
    };
    const Scheme schemes[] = {{"none", "1", 0}, {"muscl", "1", 1}, {"muscl", "2", 2}, {"weno3", "2", 2}};
    int checked_jumps = 0;
    for (int a = 0; a < 3; ++a) {
        const int p = a == 0 ? 1 : 0, q = a == 2 ? 1 : 2;   // the in-plane axes (2-D x <- p, 2-D y <- q)
        const int n2[2] = {9, 7};
        const double L2[2] = {1.7, 1.1}, La = 0.6;
        const char* bc2[4] = {"transmissive", "wall", "wall", "outlet"};
        for (const Scheme& sc : schemes) {
            Simulation s2;
            s2.setup(cell_flux_params({{"bc_left", bc2[0]}, {"bc_right", bc2[1]}, {"bc_bottom", bc2[2]}, {"bc_top", bc2[3]},
                                       {"reconstruction", sc.recon}, {"boundary_order", sc.bo}}),
                     false);
            BCs bc3;
            bc3[2 * p] = bc2[0];
            bc3[2 * p + 1] = bc2[1];
            bc3[2 * q] = bc2[2];
            bc3[2 * q + 1] = bc2[3];
            bc3[2 * a] = bc3[2 * a + 1] = "periodic";
            std::string n3[3], L3[3];
            n3[p] = std::to_string(n2[0]);
            n3[q] = std::to_string(n2[1]);
            n3[a] = "2";
            for (int ax = 0; ax < 3; ++ax) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.17g", ax == p ? L2[0] : (ax == q ? L2[1] : La));
                L3[ax] = buf;
            }
            Simulation s3;
            setup_3d(s3, bc3,
                     {{"nx", n3[0].c_str()}, {"ny", n3[1].c_str()}, {"nz", n3[2].c_str()}, {"Lx", L3[0].c_str()}, {"Ly", L3[1].c_str()},
                      {"Lz", L3[2].c_str()}, {"reconstruction", sc.recon}, {"boundary_order", sc.bo}});
            CellMesh m2(L2[0], L2[1], n2[0], n2[1], 2, false, false, s2.mesh.x0, s2.mesh.y0);
            random_adapt(m2, 41u + static_cast<unsigned>(a), 4, 18, 20);
            CellMesh m3 = octree_of(s3, 2);
            std::vector<int> lv, ii, jj, kk;
            for (int c = 0; c < m2.num_leaves(); ++c) {
                const int l = m2.level[c], layers = 2 << l;
                for (int t = 0; t < layers; ++t) {
                    int x[3];
                    x[p] = m2.ci[c];
                    x[q] = m2.cj[c];
                    x[a] = t;
                    lv.push_back(l);
                    ii.push_back(x[0]);
                    jj.push_back(x[1]);
                    kk.push_back(x[2]);
                }
            }
            m3.assign_leaves(lv, ii, jj, kk);
            m2.U.resize(static_cast<size_t>(m2.num_leaves()));
            for (int c = 0; c < m2.num_leaves(); ++c) m2.U[c] = test_state(s2.fluid, m2.xc(c), m2.yc(c));
            m3.U.resize(static_cast<size_t>(m3.num_leaves()));
            std::vector<int> to2(static_cast<size_t>(m3.num_leaves()));
            for (int c = 0; c < m3.num_leaves(); ++c) {
                const int x[3] = {m3.ci[c], m3.cj[c], m3.ck[c]};
                const int c2 = m2.find(m3.level[c], x[p], x[q]);
                to2[static_cast<size_t>(c)] = c2;
                const State W2 = s2.fluid.cons_to_prim(m2.U[c2]);
                State W3{};
                W3[0] = W2[0];
                W3[3] = W2[3];
                W3[kMom[p]] = W2[1];
                W3[kMom[q]] = W2[2];
                m3.U[c] = s3.fluid.prim_to_cons(W3);
            }
            CellFluxEngine e2(s2.fluid, s2.riemann, s2.muscl.get(), s2.boundaries), e3(s3.fluid, s3.riemann, s3.muscl.get(), s3.boundaries);
            e2.build(m2);
            e3.build(m3);
            std::unique_ptr<TimeScheme> t2 = make_time_scheme(parse_time_scheme(kSchemeNames[sc.sc]), 0.3),
                                        t3 = make_time_scheme(parse_time_scheme(kSchemeNames[sc.sc]), 0.3);
            CellAdvance a2(m2, e2, *t2), a3(m3, e3, *t3);
            const double dt = 0.3 * std::min(m2.dx(0), m2.dy(0)) / e2.wave_speed(m2, m2.U);   // one fixed dt for both
            for (int k = 0; k < 8; ++k) {
                a2.step(dt, k * dt);
                a3.step(dt, k * dt);
            }
            const int q3[4] = {0, kMom[p], kMom[q], 3}, q2[4] = {0, 1, 2, 3};
            double worst = 0.0;
            bool a_zero = true;
            for (int k = 0; k < 4; ++k) {
                double scale = 0.0;
                for (const State& u : m2.U) scale = std::max(scale, std::abs(u[q2[k]]));
                for (int c = 0; c < m3.num_leaves(); ++c)
                    worst = std::max(worst, std::abs(m3.U[c][q3[k]] - m2.U[static_cast<size_t>(to2[c])][q2[k]]) / scale);
            }
            for (int c = 0; c < m3.num_leaves(); ++c) a_zero = a_zero && m3.U[c][kMom[a]] == 0.0;
            CHECK(worst <= 1e-12);
            CHECK(a_zero);
            std::printf("    axis %d %s bo %s %s: worst relative difference to the 2-D run %.3e\n", a, sc.recon, sc.bo, kSchemeNames[sc.sc], worst);
            for (const CellFace& f : m3.faces) checked_jumps += f.kind == CellFaceKind::CoarseFine && f.axis != a;
        }
    }
    CHECK(checked_jumps > 1000);
}

namespace {

// One Sod run on a long axis of 50 base cells ([0, 1]) with two transverse axes of 2 periodic base cells (0.04), the band
// [0.2, 0.92] refined to level 2, SSP-RK2 + MUSCL Van Leer, to t = 0.2. dim 2: the long axis is x (the 2-D reference);
// dim 3: the long axis is `a`. `dts` empty: the run takes its own dt (0.4 dx / S) and records it; else it replays them.
struct SodRun {
    CellMesh m;
    double mass0 = 0.0;
};
SodRun sod_run(int dim, int a, int max_level, bool band, std::vector<double>& dts, int n_long = 50) {
    const std::string nl = std::to_string(n_long);
    Simulation sim;
    int p = 1, q = 2;
    if (dim == 2) {
        sim.setup(cell_flux_params({{"nx", nl.c_str()}, {"ny", "2"}, {"Lx", "1"}, {"Ly", "0.04"}, {"bc_left", "transmissive"}, {"bc_right", "transmissive"},
                                    {"bc_bottom", "periodic"}, {"bc_top", "periodic"}, {"reconstruction", "muscl"}, {"muscl_limiter", "Van Leer"}}),
                  false);
    } else {
        p = a == 0 ? 1 : 0;
        q = a == 2 ? 1 : 2;
        BCs bc;
        bc[2 * a] = bc[2 * a + 1] = "transmissive";
        bc[2 * p] = bc[2 * p + 1] = bc[2 * q] = bc[2 * q + 1] = "periodic";
        std::string n[3], L[3];
        for (int ax = 0; ax < 3; ++ax) {
            n[ax] = ax == a ? nl : "2";
            L[ax] = ax == a ? "1" : "0.04";
        }
        setup_3d(sim, bc, {{"nx", n[0].c_str()}, {"ny", n[1].c_str()}, {"nz", n[2].c_str()}, {"Lx", L[0].c_str()}, {"Ly", L[1].c_str()},
                           {"Lz", L[2].c_str()}, {"reconstruction", "muscl"}, {"muscl_limiter", "Van Leer"}});
    }
    const auto make_mesh = [&]() {
        if (dim == 2) return CellMesh(1.0, 0.04, n_long, 2, max_level, false, true, 0.0, 0.0);
        int n[3] = {2, 2, 2};
        double L[3] = {0.04, 0.04, 0.04};
        n[a] = n_long;
        L[a] = 1.0;
        return CellMesh::make_3d(L[0], L[1], L[2], n[0], n[1], n[2], max_level, a != 0, a != 1, a != 2);
    };
    SodRun r{make_mesh()};
    CellMesh& m = r.m;
    const auto along = [&](int c) { return dim == 2 || a == 0 ? m.xc(c) : (a == 1 ? m.yc(c) : m.zc(c)); };
    if (band)
        for (int pass = 0; pass < max_level; ++pass) {
            std::vector<char> f(static_cast<size_t>(m.num_leaves()), 0);
            for (int c = 0; c < m.num_leaves(); ++c) f[c] = along(c) > 0.2 && along(c) < 0.92;
            m.adapt(f);
        }
    m.U.resize(static_cast<size_t>(m.num_leaves()));
    for (int c = 0; c < m.num_leaves(); ++c) {
        State W{};
        W[0] = along(c) < 0.5 ? 1.0 : 0.125;
        W[3] = along(c) < 0.5 ? 1.0 : 0.1;
        m.U[c] = sim.fluid.prim_to_cons(W);
    }
    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    eng.build(m);
    std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4);
    CellAdvance adv(m, eng, *ts);
    r.mass0 = cell_total_nd(m, 0);
    const bool record = dts.empty();
    double S = eng.wave_speed(m, m.U), t = 0.0;
    for (size_t k = 0; record ? t < 0.2 : k < dts.size(); ++k) {
        const double dt = record ? std::min(0.4 * m.dx(0) / S, 0.2 - t) : dts[k];
        if (record) dts.push_back(dt);
        S = adv.step(dt, t);
        t += dt;
    }
    return r;
}

double sod_l1(const CellMesh& m, int a) {
    double err = 0.0, vol = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c) {
        const double x = m.dim == 2 || a == 0 ? m.xc(c) : (a == 1 ? m.yc(c) : m.zc(c));
        err += std::abs(m.U[c][0] - sod_exact_density(x, 0.2)) * m.volume(c);
        vol += m.volume(c);
    }
    return err / vol;
}

}   // namespace

TEST(celladvance3d_sod_along_each_axis_on_a_static_refined_band) {
    // The same Sod tube along x, along y and along z of an octree (band [0.2, 0.92] at level 2, 1-D in the transverse periodic
    // axes) with the dt sequence of the 2-D run. Bands fixed before the first run: (1) each 3-D run equals the 2-D one up to the
    // permutation to round-off (density, axial momentum and energy within 1e-12 of their scale; transverse momenta exactly 0);
    // (2) the three axes agree with each other to the same tolerance; (3) mass to 1e-12; (4) the L1 density error against the
    // exact solution is below half the uniform-50 one (the 2-D test's band), and no overshoot is born at a jump.
    std::vector<double> dts, dts_coarse;
    const SodRun r2 = sod_run(2, 0, 2, true, dts);
    const SodRun c2 = sod_run(2, 0, 0, false, dts_coarse);
    const double e2 = sod_l1(r2.m, 0), e_coarse = sod_l1(c2.m, 0);
    std::printf("    Sod 2-D: L1(rho) uniform 50 %.4e, AMR %.4e (%d leaves, %zu steps)\n", e_coarse, e2, r2.m.num_leaves(), dts.size());
    CHECK(e2 < 0.5 * e_coarse);
    std::vector<SodRun> r3;
    for (int a = 0; a < 3; ++a) r3.push_back(sod_run(3, a, 2, true, dts));
    double between = 0.0;
    bool bitwise = true;
    for (int a = 0; a < 3; ++a) {
        const CellMesh& m = r3[static_cast<size_t>(a)].m;
        const int p = a == 0 ? 1 : 0, q = a == 2 ? 1 : 2;
        const int q3[3] = {0, kMom[a], 3}, q2[3] = {0, 1, 3};
        double scale[3] = {0.0, 0.0, 0.0}, worst = 0.0, rho_min = 1e9, rho_max = 0.0;
        for (const State& u : r2.m.U)
            for (int k = 0; k < 3; ++k) scale[k] = std::max(scale[k], std::abs(u[q2[k]]));
        bool transverse_zero = true;
        for (int c = 0; c < m.num_leaves(); ++c) {
            const int x[3] = {m.ci[c], m.cj[c], m.ck[c]};
            const int c2 = r2.m.find(m.level[c], x[a], x[p]);
            CHECK(c2 >= 0);
            for (int k = 0; k < 3; ++k) worst = std::max(worst, std::abs(m.U[c][q3[k]] - r2.m.U[static_cast<size_t>(c2)][q2[k]]) / scale[k]);
            transverse_zero = transverse_zero && m.U[c][kMom[p]] == 0.0 && m.U[c][kMom[q]] == 0.0;
            rho_min = std::min(rho_min, m.U[c][0]);
            rho_max = std::max(rho_max, m.U[c][0]);
            // the same leaf of the other axes' runs (the axial coordinate on their axis, the transverse ones in order)
            for (int b = 0; b < 3; ++b) {
                if (b == a) continue;
                const CellMesh& mb = r3[static_cast<size_t>(b)].m;
                const int pb = b == 0 ? 1 : 0, qb = b == 2 ? 1 : 2;
                int y[3];
                y[b] = x[a];
                y[pb] = x[p];
                y[qb] = x[q];
                const int cb = mb.find(m.level[c], y[0], y[1], y[2]);
                CHECK(cb >= 0);
                const State& ub = mb.U[static_cast<size_t>(cb)];
                const int qa3[3] = {0, kMom[b], 3};
                for (int k = 0; k < 3; ++k) {
                    between = std::max(between, std::abs(m.U[c][q3[k]] - ub[qa3[k]]) / scale[k]);
                    bitwise = bitwise && m.U[c][q3[k]] == ub[qa3[k]];
                }
            }
        }
        const double e3 = sod_l1(m, a), dm = std::abs(cell_total_nd(m, 0) - r3[static_cast<size_t>(a)].mass0) / r3[static_cast<size_t>(a)].mass0;
        std::printf("    Sod along axis %d: %d leaves, L1(rho) %.4e, worst relative difference to the 2-D run %.3e, mass error %.3e\n", a,
                    m.num_leaves(), e3, worst, dm);
        CHECK(worst <= 1e-12);
        CHECK(transverse_zero);
        CHECK(dm < 1e-12);
        CHECK(e3 < 0.5 * e_coarse);
        CHECK(rho_min > 0.125 - 0.01 && rho_max < 1.0 + 0.01);
    }
    std::printf("    the three axes against each other: worst relative difference %.3e (%s)\n", between, bitwise ? "bit for bit" : "not bitwise");
    CHECK(between <= 1e-12);
}

namespace {

// Advected density wave: rho = 1 + 0.1 sum_a sin(2 pi (x_a - u_a t)), u = (1, 0.5, 0.25), p = 1, periodic [0, 1]^d; the exact
// cell average of sin(2 pi (x - s)) over [x0, x1] is (cos 2 pi (x0 - s) - cos 2 pi (x1 - s)) / (2 pi (x1 - x0)).
const double kWaveU[3] = {1.0, 0.5, 0.25};

double wave_rho_average(int dim, const double lo[3], const double hi[3], double t) {
    const double tp = 2.0 * 3.14159265358979323846;
    double r = 1.0;
    for (int a = 0; a < dim; ++a) {
        const double s = kWaveU[a] * t;
        r += 0.1 * (std::cos(tp * (lo[a] - s)) - std::cos(tp * (hi[a] - s))) / (tp * (hi[a] - lo[a]));
    }
    return r;
}

void leaf_box(const CellMesh& m, int c, double lo[3], double hi[3]) {
    const int l = m.level[c];
    const double h[3] = {m.dx(l), m.dy(l), m.dim == 3 ? m.dz(l) : 0.0};
    const int x[3] = {m.ci[c], m.cj[c], m.dim == 3 ? m.ck[c] : 0};
    for (int a = 0; a < 3; ++a) {
        lo[a] = x[a] * h[a];
        hi[a] = (x[a] + 1) * h[a];
    }
}

// L1 density error (whole domain; within 2 base widths of the refined box's surface; more than 3 base widths outside it) after
// the wave ran to t_end, base n^d with the box [0.25, 0.75]^d one level finer (refine = false: the uniform base mesh).
struct WaveErrors {
    double all, near, far;
    int leaves;
};
WaveErrors wave_errors(int dim, int n, bool refine, double t_end) {
    const std::string ns = std::to_string(n);
    Simulation sim;
    if (dim == 2)
        sim.setup(cell_flux_params({{"nx", ns.c_str()}, {"ny", ns.c_str()}, {"Lx", "1"}, {"Ly", "1"}, {"bc_left", "periodic"}, {"bc_right", "periodic"},
                                    {"bc_bottom", "periodic"}, {"bc_top", "periodic"}, {"reconstruction", "muscl"}, {"muscl_limiter", "Van Leer"}}),
                  false);
    else
        setup_3d(sim, {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"},
                 {{"nx", ns.c_str()}, {"ny", ns.c_str()}, {"nz", ns.c_str()}, {"Lx", "1"}, {"Ly", "1"}, {"Lz", "1"},
                  {"reconstruction", "muscl"}, {"muscl_limiter", "Van Leer"}});
    CellMesh m = dim == 2 ? CellMesh(1.0, 1.0, n, n, 1, true, true, 0.0, 0.0) : CellMesh::make_3d(1.0, 1.0, 1.0, n, n, n, 1, true, true, true);
    const auto signed_dist = [&](int c) {   // max-norm distance of the leaf centre to the box surface (< 0 inside)
        double lo[3], hi[3], s = -1e300;
        leaf_box(m, c, lo, hi);
        for (int a = 0; a < dim; ++a) {
            const double x = 0.5 * (lo[a] + hi[a]);
            s = std::max(s, std::max(0.25 - x, x - 0.75));
        }
        return s;
    };
    if (refine) {
        std::vector<char> f(static_cast<size_t>(m.num_leaves()));
        for (int c = 0; c < m.num_leaves(); ++c) f[c] = signed_dist(c) < 0.0;
        m.adapt(f);
    }
    m.U.resize(static_cast<size_t>(m.num_leaves()));
    for (int c = 0; c < m.num_leaves(); ++c) {
        double lo[3], hi[3];
        leaf_box(m, c, lo, hi);
        State W{};
        W[0] = wave_rho_average(dim, lo, hi, 0.0);
        W[1] = kWaveU[0];
        W[2] = kWaveU[1];
        W[3] = 1.0;
        if (dim == 3) W[4] = kWaveU[2];
        m.U[c] = sim.fluid.prim_to_cons(W);
    }
    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    eng.build(m);
    std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4);
    CellAdvance adv(m, eng, *ts);
    double S = eng.wave_speed(m, m.U), t = 0.0;
    while (t < t_end) {
        const double dt = std::min(0.4 * dt_length0(m) / S, t_end - t);
        S = adv.step(dt, t);
        t += dt;
    }
    const double h0 = 1.0 / n;
    double ea = 0.0, va = 0.0, en = 0.0, vn = 0.0, ef = 0.0, vf = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c) {
        double lo[3], hi[3];
        leaf_box(m, c, lo, hi);
        const double e = std::abs(m.U[c][0] - wave_rho_average(dim, lo, hi, t_end)) * m.volume(c), v = m.volume(c), s = signed_dist(c);
        ea += e;
        va += v;
        if (std::abs(s) < 2.0 * h0) {
            en += e;
            vn += v;
        } else if (s > 3.0 * h0) {
            ef += e;
            vf += v;
        }
    }
    return {ea / va, en / vn, ef / std::max(vf, 1e-300), m.num_leaves()};
}

}   // namespace

TEST(celladvance3d_near_jump_time_study) {
    // The time-run question left by the Q2 truncation table: the residual is O(1) at a jump, what does the solution converge at?
    // A density wave advected through a box refined one level (base n = 8, 16, 32, t = 0.25: the wave moves 1/4 of the domain in x),
    // the same measure in 2-D and 3-D, L1 errors against the exact cell averages, SSP-RK2 + MUSCL Van Leer, CFL 0.4 of dt_length.
    // Bands fixed before the first run: (0) the uniform base runs converge with order >= 1.5 (16 -> 32) in both dimensions (the
    // measure); (1) the 3-D adaptive order (whole domain, 16 -> 32) >= the 2-D one - 0.3; (2) the 3-D near-jump order >= the
    // 2-D one - 0.3; (3) the adaptive / uniform-base error ratio at n = 32 in 3-D <= 1.2 x the 2-D one.
    const int ns[3] = {8, 16, 32};
    const double t_end = 0.25;
    WaveErrors amr[2][3], uni[2][3];
    for (int d = 0; d < 2; ++d)
        for (int k = 0; k < 3; ++k) {
            amr[d][k] = wave_errors(d + 2, ns[k], true, t_end);
            uni[d][k] = wave_errors(d + 2, ns[k], false, t_end);
        }
    const auto order = [](double a, double b) { return std::log2(a / b); };
    std::printf("    advected density wave, box [0.25, 0.75]^d one level finer, L1(rho) at t = 0.25, n = 8 / 16 / 32 (orders 8 -> 16, 16 -> 32):\n");
    for (int d = 0; d < 2; ++d) {
        std::printf("    %d-D uniform   all %.3e %.3e %.3e  orders %.2f %.2f\n", d + 2, uni[d][0].all, uni[d][1].all, uni[d][2].all,
                    order(uni[d][0].all, uni[d][1].all), order(uni[d][1].all, uni[d][2].all));
        std::printf("    %d-D adaptive  all %.3e %.3e %.3e  orders %.2f %.2f  (leaves at 32: %d, uniform %d)\n", d + 2, amr[d][0].all, amr[d][1].all,
                    amr[d][2].all, order(amr[d][0].all, amr[d][1].all), order(amr[d][1].all, amr[d][2].all), amr[d][2].leaves, uni[d][2].leaves);
        // (no leaf is more than 3 base widths outside the box at n = 8: the far measure starts at 16)
        std::printf("    %d-D adaptive  near %.3e %.3e %.3e  orders %.2f %.2f   far - %.3e %.3e  order %.2f\n", d + 2, amr[d][0].near,
                    amr[d][1].near, amr[d][2].near, order(amr[d][0].near, amr[d][1].near), order(amr[d][1].near, amr[d][2].near),
                    amr[d][1].far, amr[d][2].far, order(amr[d][1].far, amr[d][2].far));
        std::printf("    %d-D adaptive / uniform at 32: all %.3f, near %.3f, far %.3f\n", d + 2, amr[d][2].all / uni[d][2].all,
                    amr[d][2].near / uni[d][2].near, amr[d][2].far / uni[d][2].far);
    }
    for (int d = 0; d < 2; ++d) CHECK(order(uni[d][1].all, uni[d][2].all) >= 1.5);
    CHECK(order(amr[1][1].all, amr[1][2].all) >= order(amr[0][1].all, amr[0][2].all) - 0.3);
    CHECK(order(amr[1][1].near, amr[1][2].near) >= order(amr[0][1].near, amr[0][2].near) - 0.3);
    CHECK(amr[1][2].all / uni[1][2].all <= 1.2 * amr[0][2].all / uni[0][2].all);
}
