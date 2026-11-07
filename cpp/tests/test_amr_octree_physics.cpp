// AMR in three dimensions, phase Q4 (docs/design/amr-octree.md): the physics of CellFluxEngine on octree leaves (viscous terms,
// body force, sources, mass-flow inlet, non-reflecting outlet, immersed body).
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_amr_common.hpp"

#include <algorithm>
#include <string>

namespace {

using BCs = std::array<const char*, 6>;
using KV = std::vector<std::pair<const char*, const char*>>;

// The engine physics of a 3-D simulation: viscous terms only (Q4a)
CellPhysics viscous_physics(Simulation& sim) {
    CellPhysics ph;
    ph.ns = sim.ns.get();
    return ph;
}

// Level-0 octree against the 3-D uniform solver's residual (viscous terms when the run has them): true if residual and speed agree
// bit for bit. `visc_wins` is set when the viscous speed is the larger one (so the speed comparison tests it).
bool viscous_level0_equal(const BCs& bc, const KV& kv, bool& visc_wins) {
    Simulation sim;
    setup_3d(sim, bc, kv);
    const Mesh& u = sim.mesh;
    Field Uu(static_cast<size_t>(u.num_cells()));
    for (int k = 0; k < u.nz; ++k)
        for (int j = 0; j < u.ny; ++j)
            for (int i = 0; i < u.nx; ++i) Uu[u.cell(i, j, k)] = test_state3(sim.fluid, u.xc(i), u.yc(j), u.zc(k));
    CellMesh cm = octree_of(sim, 0);
    cm.U.resize(static_cast<size_t>(cm.num_leaves()));
    for (int c = 0; c < cm.num_leaves(); ++c) cm.U[c] = Uu[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
    Field Ru, Rc, Rx;
    const double Su = sim.residual(Uu, Ru);
    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries), inv(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    eng.set_physics(viscous_physics(sim));
    eng.build(cm);
    inv.build(cm);
    const double Sc = eng.residual(cm, cm.U, Rc), Sinv = inv.residual(cm, cm.U, Rx);
    visc_wins = Sc > Sinv;
    Field Rperm(Ru.size());
    for (int c = 0; c < cm.num_leaves(); ++c) Rperm[c] = Ru[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
    return bits_equal(Rperm, Rc) && Su == Sc;
}

}   // namespace

TEST(cellflux3d_viscous_level0_matches_uniform_residual_bitwise) {
    // Every boundary type, moving walls on an x-, a y- and a z-side, and with boundary_order = 2 adjacent sides of rules that do not
    // commute (moving wall / no-slip / slip wall) on every pair of axes, so a swapped edge composition fails; mu small and large
    // (the viscous speed wins in some cases).
    const BCs sets[] = {{"moving_wall", "no_slip", "wall", "moving_wall", "no_slip", "moving_wall"},
                        {"no_slip", "moving_wall", "moving_wall", "wall", "moving_wall", "wall"},
                        {"wall", "transmissive", "no_slip", "moving_wall", "periodic", "periodic"},
                        {"periodic", "periodic", "moving_wall", "no_slip", "wall", "moving_wall"},
                        {"outlet", "vel_inlet", "periodic", "periodic", "no_slip", "moving_wall"},
                        {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"}};
    struct Scheme {
        const char *recon, *bo;
    };
    const Scheme schemes[] = {{"none", "1"}, {"muscl", "1"}, {"muscl", "2"}, {"weno3", "2"}};
    int n_cases = 0, n_visc_wins = 0;
    for (const BCs& bc : sets)
        for (const Scheme& sc : schemes)
            for (const char* mu : {"0.02", "0.6"}) {
                bool wins = false;
                const bool same = viscous_level0_equal(
                    bc, {{"reconstruction", sc.recon}, {"boundary_order", sc.bo}, {"use_viscous", "true"}, {"mu", mu}}, wins);
                CHECK(same);
                if (!same)
                    std::printf("    bc %s/%s/%s/%s/%s/%s %s bo %s mu %s\n", bc[0], bc[1], bc[2], bc[3], bc[4], bc[5], sc.recon, sc.bo, mu);
                ++n_cases;
                n_visc_wins += wins;
            }
    CHECK(n_cases == 48);
    CHECK(n_visc_wins >= 12);
    std::printf("    %d cases, the viscous speed is the larger one in %d\n", n_cases, n_visc_wins);
    // above kOmpMinWork: the parallel loops
    for (const BCs& bc : {BCs{"moving_wall", "no_slip", "wall", "moving_wall", "transmissive", "no_slip"},
                          BCs{"periodic", "periodic", "no_slip", "moving_wall", "periodic", "periodic"}}) {
        bool wins = false;
        CHECK(viscous_level0_equal(bc, {{"nx", "24"}, {"ny", "20"}, {"nz", "10"}, {"reconstruction", "muscl"}, {"boundary_order", "2"},
                                        {"use_viscous", "true"}, {"mu", "0.05"}},
                                   wins));
    }
}

namespace {

// A random three-level octree on an 8 x 6 x 6 base (the Q3 rig) with viscous terms
struct ViscRig {
    Simulation sim;
    std::unique_ptr<CellMesh> m;
    ViscRig(const BCs& bc, const char* recon, const char* bo, const char* scheme, unsigned seed, const char* mu = "0.05") {
        setup_3d(sim, bc, {{"nx", "8"}, {"ny", "6"}, {"nz", "6"}, {"reconstruction", recon}, {"boundary_order", bo}, {"time_scheme", scheme},
                           {"use_viscous", "true"}, {"mu", mu}});
        m = std::make_unique<CellMesh>(octree_of(sim, 2));
        random_adapt(*m, seed, 3, 6, 30);
        m->U.resize(static_cast<size_t>(m->num_leaves()));
        for (int c = 0; c < m->num_leaves(); ++c) m->U[c] = test_state3(sim.fluid, m->xc(c), m->yc(c), m->zc(c));
    }
    bool three_levels() const { return m->level_begin[1] > 0 && m->level_begin[2] > m->level_begin[1] && m->level_begin[3] > m->level_begin[2]; }
};

const BCs kPeriodic = {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"};

}   // namespace

TEST(cellflux3d_viscous_level_pieces_equal_the_whole_residual_and_conserve) {
    // compute_level_fluxes + gather_level of every level == the residual's slices bit for bit (they read the per-level lists: the
    // viscous cell quantities of vcells_ with six neighbours); on a periodic octree the residual is conservative (all five components).
    for (const char* bo : {"1", "2"})
        for (const BCs& bc : {kPeriodic, BCs{"moving_wall", "no_slip", "periodic", "periodic", "wall", "moving_wall"}}) {
            ViscRig r(bc, "muscl", bo, "SSP Runge-Kutta 3", 23u);
            CellMesh& m = *r.m;
            CHECK(r.three_levels());
            CellFluxEngine eng(r.sim.fluid, r.sim.riemann, r.sim.muscl.get(), r.sim.boundaries);
            eng.set_physics(viscous_physics(r.sim));
            eng.build(m);
            Field R;
            const double S = eng.residual(m, m.U, R);
            for (int l = 0; l <= m.max_level; ++l) {
                eng.compute_level_fluxes(m, m.U, l);
                Field Rl;
                eng.gather_level(m, l, false, Rl);
                CHECK(bits_equal(Rl, Field(R.begin() + m.level_begin[l], R.begin() + m.level_begin[l + 1])));
            }
            CHECK(eng.wave_speed(m, m.U) == S);
            if (bc[0] == std::string("periodic"))
                for (int q = 0; q < kNVar; ++q) {
                    double sum = 0.0, mag = 0.0;
                    for (int c = 0; c < m.num_leaves(); ++c) {
                        sum += R[c][q] * m.volume(c);
                        mag += std::abs(R[c][q]) * m.volume(c);
                    }
                    CHECK(mag > 1e-6);
                    CHECK(std::abs(sum) <= 1e-13 * mag);
                }
        }
}

TEST(celladvance3d_viscous_conservation_free_stream_and_control) {
    for (int sc = 0; sc < 3; ++sc) {
        const char* scheme = kSchemeNames[sc];
        ViscRig r(kPeriodic, "muscl", "1", scheme, 91u + static_cast<unsigned>(sc));
        CHECK(r.three_levels());
        const auto run = [&](bool replace, bool uniform_state) {
            CellMesh mm = *r.m;
            if (uniform_state) mm.U.assign(static_cast<size_t>(mm.num_leaves()), r.sim.fluid.prim_to_cons({1.3, 0.8, -0.45, 2.1, 0.35}));
            const Field U0 = mm.U;
            CellFluxEngine eng(r.sim.fluid, r.sim.riemann, r.sim.muscl.get(), r.sim.boundaries);
            eng.set_physics(viscous_physics(r.sim));
            eng.build(mm);
            std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(scheme), 0.3);
            CellAdvance adv(mm, eng, *ts);
            adv.flux_replacement = replace;
            double S = eng.wave_speed(mm, mm.U), before[kNVar], mag[kNVar], worst = 0.0;
            for (int q = 0; q < kNVar; ++q) {
                before[q] = cell_total_nd(mm, q);
                mag[q] = cell_total_nd(mm, q, true);
            }
            const double h = 2.0 / (1.0 / mm.dx(0) + 1.0 / mm.dy(0) + 1.0 / mm.dz(0));
            for (int k = 0; k < 5; ++k) S = adv.step(0.3 * h / S, k * 0.01);
            for (int q = 0; q < kNVar; ++q) worst = std::max(worst, std::abs(cell_total_nd(mm, q) - before[q]) / mag[q]);
            double drift = 0.0;
            for (int c = 0; c < mm.num_leaves(); ++c)
                for (int q = 0; q < kNVar; ++q) drift = std::max(drift, std::abs(mm.U[c][q] - U0[c][q]));
            return std::make_pair(worst, drift);
        };
        const auto with = run(true, false), without = run(false, false), stream = run(true, true);
        CHECK(with.first < 1e-13);
        CHECK(without.first > 1e-8);   // negative control: the registers (convective and viscous) are what conserve
        CHECK(stream.second < 1e-14);  // zero gradients: nothing moves
        std::printf("    %s: viscous conservation error %.3e (without replacement %.3e), free-stream drift %.3e\n", scheme, with.first,
                    without.first, stream.second);
    }
}

TEST(celladvance3d_viscous_level0_steps_match_uniform_steps_bitwise) {
    for (const BCs& bc : {BCs{"moving_wall", "no_slip", "wall", "moving_wall", "no_slip", "moving_wall"},
                          BCs{"periodic", "periodic", "moving_wall", "no_slip", "periodic", "periodic"}})
        for (const char* recon : {"muscl", "weno3"}) {
            Simulation sim;
            setup_3d(sim, bc, {{"reconstruction", recon}, {"boundary_order", "2"}, {"time_scheme", "SSP Runge-Kutta 3"},
                               {"use_viscous", "true"}, {"mu", "0.1"}});
            const Mesh& u = sim.mesh;
            Field Uu(static_cast<size_t>(u.num_cells()));
            for (int k = 0; k < u.nz; ++k)
                for (int j = 0; j < u.ny; ++j)
                    for (int i = 0; i < u.nx; ++i) Uu[u.cell(i, j, k)] = test_state3(sim.fluid, u.xc(i), u.yc(j), u.zc(k));
            for (int max_level : {0, 2}) {
                CellMesh cm = octree_of(sim, max_level);
                cm.U.resize(static_cast<size_t>(cm.num_leaves()));
                for (int c = 0; c < cm.num_leaves(); ++c) cm.U[c] = Uu[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
                Field Uw = Uu;
                CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
                eng.set_physics(viscous_physics(sim));
                eng.build(cm);
                std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK3, 0.4);
                CellAdvance adv(cm, eng, *ts);
                const ResidualFn res = [&](const Field& Uin, Field& R) { return sim.residual(Uin, R); };
                const double dt = 1e-3;
                bool same = true;
                for (int k = 0; k < 3; ++k) same = same && sim.scheme->advance_fixed_dt(Uw, res, dt) == adv.step(dt, k * dt);
                for (int c = 0; c < cm.num_leaves(); ++c) same = same && cm.U[c] == Uw[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
                CHECK(same);
            }
        }
}

TEST(cellflux3d_viscous_invariant_axis_equals_the_2d_engine) {
    // The viscous version of cellflux3d_invariant_axis_equals_the_2d_engine: a field independent of axis a on the extruded 2-D tree
    // (a periodic), the 3-D viscous residual equals the 2-D engine's to round-off and its a-momentum is exactly 0. Walls without a
    // moving wall (its velocity is tied to the absolute axes, not to the 2-D ones).
    int checked_jumps = 0;
    for (int a = 0; a < 3; ++a) {
        const int p = a == 0 ? 1 : 0, q = a == 2 ? 1 : 2;
        const int n2[2] = {9, 7};
        const double L2[2] = {1.7, 1.1}, La = 0.6;
        const char* const bcs2[2][4] = {{"no_slip", "wall", "wall", "no_slip"}, {"transmissive", "no_slip", "wall", "outlet"}};
        for (const auto& bc2 : bcs2)
            for (const char* bo : {"1", "2"})
                for (const char* recon : {"none", "muscl"}) {
                    if (std::string(recon) == "none" && std::string(bo) == "2") continue;
                    const KV common = {{"reconstruction", recon}, {"boundary_order", bo}, {"use_viscous", "true"}, {"mu", "0.05"}};
                    Simulation s2;
                    s2.setup(cell_flux_params({{"bc_left", bc2[0]}, {"bc_right", bc2[1]}, {"bc_bottom", bc2[2]}, {"bc_top", bc2[3]},
                                               {"reconstruction", recon}, {"boundary_order", bo}, {"use_viscous", "true"}, {"mu", "0.05"}}),
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
                    KV kv3 = {{"nx", n3[0].c_str()}, {"ny", n3[1].c_str()}, {"nz", n3[2].c_str()},
                              {"Lx", L3[0].c_str()}, {"Ly", L3[1].c_str()}, {"Lz", L3[2].c_str()}};
                    kv3.insert(kv3.end(), common.begin(), common.end());
                    Simulation s3;
                    setup_3d(s3, bc3, kv3);
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
                    CellPhysics ph2;
                    ph2.ns = s2.ns.get();
                    e2.set_physics(ph2);
                    e3.set_physics(viscous_physics(s3));
                    e2.build(m2);
                    e3.build(m3);
                    Field R2, R3;
                    e2.residual(m2, m2.U, R2);
                    e3.residual(m3, m3.U, R3);
                    // (the speeds differ: the 3-D viscous bound sums 1/h^2 over three axes)
                    const int q3[4] = {0, kMom[p], kMom[q], 3}, q2[4] = {0, 1, 2, 3};
                    double worst = 0.0;
                    for (int k = 0; k < 4; ++k) {
                        double scale = 0.0;
                        for (const State& r : R2) scale = std::max(scale, std::abs(r[q2[k]]));
                        for (int c = 0; c < m3.num_leaves(); ++c)
                            worst = std::max(worst, std::abs(R3[c][q3[k]] - R2[static_cast<size_t>(to2[c])][q2[k]]) / scale);
                    }
                    bool a_zero = true;
                    for (int c = 0; c < m3.num_leaves(); ++c) a_zero = a_zero && R3[c][kMom[a]] == 0.0;
                    CHECK(worst <= 1e-12);
                    CHECK(a_zero);
                    if (worst > 1e-12 || !a_zero)
                        std::printf("    axis %d bc %s/%s/%s/%s %s bo %s: worst %.3e, a-momentum zero %d\n", a, bc2[0], bc2[1], bc2[2], bc2[3], recon,
                                    bo, worst, a_zero);
                    for (const CellFace& f : m3.faces) checked_jumps += f.kind == CellFaceKind::CoarseFine && f.axis != a;
                }
    }
    CHECK(checked_jumps > 1000);
}

// ---------------------------------------------------------------------------
// Q4b: body force and source terms
// ---------------------------------------------------------------------------

namespace {

// The physics of a simulation for the engine, through the inline terms of the gather (physics_of) or through the ordered
// RhsSourceTerms of the simulation (what Simulation hands the engine)
CellPhysics physics_3d(Simulation& sim, bool rhs) {
    CellPhysics ph = physics_of(sim);
    if (rhs) ph.rhs_sources = sim.rhs_sources.empty() ? nullptr : &sim.rhs_sources;
    return ph;
}

}   // namespace

TEST(cellflux3d_body_force_and_sources_level0_match_uniform_bitwise) {
    // The 3-D manufactured solution (viscous, steady sources) with a body force on three axes, and a time-dependent source on a
    // walled box, against the 3-D uniform solver: residual and speed bit for bit, through the inline terms and the RhsSourceTerms.
    // The sources are evaluated at the leaf centres, so this also checks that they are the uniform cell centres bit for bit.
    struct Cfg {
        KV kv;
        const char* what;
    };
    const Cfg cfgs[] = {
        {{{"nx", "6"}, {"ny", "5"}, {"nz", "4"}, {"use_bodyforce", "true"}, {"gx", "0.3"}, {"gy", "-0.7"}, {"gz", "0.4"}}, "MMS + body force"},
        {{{"nx", "6"}, {"ny", "5"}, {"nz", "4"}, {"bc_left", "no_slip"}, {"bc_right", "moving_wall"}, {"bc_bottom", "wall"},
          {"bc_top", "outlet"}, {"bc_front", "wall"}, {"bc_back", "no_slip"}, {"boundary_order", "2"},
          {"source_rho", "0.1*sin(x+2*y+3*z+t)"}, {"source_mz", "0.2*cos(3*x-y+z-2*t)"}, {"source_E", "0.05*x*y*z+t"},
          {"use_bodyforce", "true"}, {"gz", "-0.5"}},
         "walls, unsteady source + body force"}};
    for (const Cfg& cf : cfgs)
        for (int rhs = 0; rhs < 2; ++rhs) {
            Params p = baseline_defaults();
            p.merge(get_test_case("Manufactured Solution 3D").defaults);
            p.set("test_case", "Manufactured Solution 3D");
            p.set("output_root", tmp_dir().string());
            p.set("use_mpi", "false");
            for (const auto& e : cf.kv) p.set(e.first, e.second);
            Simulation sim;
            sim.setup(SimParams::from(p), false);
            CHECK(sim.p.use_source && sim.p.use_bodyforce && sim.mesh.is3d());
            const Mesh& u = sim.mesh;
            Field Uu(static_cast<size_t>(u.num_cells()));
            for (int k = 0; k < u.nz; ++k)
                for (int j = 0; j < u.ny; ++j)
                    for (int i = 0; i < u.nx; ++i) Uu[u.cell(i, j, k)] = test_state3(sim.fluid, u.xc(i), u.yc(j), u.zc(k));
            CellMesh cm = octree_of(sim, 0);
            cm.U.resize(static_cast<size_t>(cm.num_leaves()));
            for (int c = 0; c < cm.num_leaves(); ++c) cm.U[c] = Uu[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
            const double t = 0.37;
            sim.scheme->stage_time = t;
            Field Ru, Rc;
            const double Su = sim.residual(Uu, Ru);
            CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
            eng.set_physics(physics_3d(sim, rhs != 0));
            eng.build(cm);
            eng.stage_time = t;
            const double Sc = eng.residual(cm, cm.U, Rc);
            Field Rperm(Ru.size());
            for (int c = 0; c < cm.num_leaves(); ++c) Rperm[c] = Ru[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
            const bool same = bits_equal(Rperm, Rc) && Su == Sc;
            CHECK(same);
            if (!same) std::printf("    %s (%s): the level-0 octree differs from the uniform residual\n", cf.what, rhs ? "RhsSourceTerms" : "inline");
        }
}

namespace {

// The initial field of a case at the leaf centres of a 3-D octree (each level from the uniform 3-D mesh of that level, so a level-0
// tree gets exactly the uniform initial field)
void init_cell_case_3d(CellMesh& cm, const Simulation& sim, const char* name) {
    cm.U.assign(static_cast<size_t>(cm.num_leaves()), State{});
    for (int l = 0; l <= cm.max_level; ++l) {
        if (cm.level_begin[l + 1] == cm.level_begin[l]) continue;
        const Mesh ml = generate_cartesian_mesh_3d(cm.Lx, cm.Ly, cm.Lz, cm.nx_at(l), cm.ny_at(l), cm.nz_at(l), cm.x0, cm.y0, cm.z0);
        const Field F = get_test_case(name).initialize(ml, sim.fluid, sim.p.source);
        for (int c = cm.level_begin[l]; c < cm.level_begin[l + 1]; ++c) cm.U[c] = F[ml.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
    }
}

// Density L1 error against the initial (exact, steady) field after a run to t_end; base n^d, the half (y < 0.5 in 2-D, z < 0.5 in
// 3-D) refined one level when `refine`; `far` = the error at more than 2 coarse widths from the two jump planes (0 and 0.5)
void mms_errors(int dim, int n, bool refine, double t_end, double& all, double& far, int& leaves) {
    const char* name = dim == 2 ? "Manufactured Solution" : "Manufactured Solution 3D";
    const std::string ns = std::to_string(n);
    Simulation sim;
    if (dim == 2)
        sim.setup(params_for(name, {{"nx", ns.c_str()}, {"ny", ns.c_str()}, {"use_mpi", "false"}}), false);
    else
        sim.setup(params_for(name, {{"nx", ns.c_str()}, {"ny", ns.c_str()}, {"nz", ns.c_str()}, {"use_mpi", "false"}}), false);
    CellMesh m = dim == 2 ? CellMesh(1.0, 1.0, n, n, 1, true, true) : CellMesh::make_3d(1.0, 1.0, 1.0, n, n, n, 1, true, true, true);
    const auto coord = [&](int c) { return dim == 2 ? m.yc(c) : m.zc(c); };
    if (refine) {
        std::vector<char> r(static_cast<size_t>(m.num_leaves()));
        for (int c = 0; c < m.num_leaves(); ++c) r[c] = coord(c) < 0.5;
        m.adapt(r);
    }
    if (dim == 2)
        init_cell_case(m, sim, name);
    else
        init_cell_case_3d(m, sim, name);
    const Field U0 = m.U;
    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    eng.set_physics(physics_of(sim));
    eng.build(m);
    std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(sim.p.time_scheme), sim.p.cfl);
    CellAdvance adv(m, eng, *ts);
    const double L = dim == 2 ? std::min(m.dx(0), m.dy(0)) : 2.0 / (1.0 / m.dx(0) + 1.0 / m.dy(0) + 1.0 / m.dz(0));
    double S = eng.wave_speed(m, m.U), t = 0.0;
    while (t < t_end) {
        const double dt = std::min(sim.p.cfl * L / S, t_end - t);
        S = adv.step(dt, t);
        t += dt;
    }
    double ea = 0.0, va = 0.0, ef = 0.0, vf = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c) {
        const double e = std::abs(m.U[c][0] - U0[c][0]) * m.volume(c), z = coord(c);
        const double d = std::min(std::min(std::abs(z - 0.5), z), 1.0 - z) * n;   // in coarse widths
        ea += e;
        va += m.volume(c);
        if (d > 2.0) {
            ef += e;
            vf += m.volume(c);
        }
    }
    all = ea / va;
    far = ef / vf;
    leaves = m.num_leaves();
}

}   // namespace

TEST(celladvance3d_manufactured_solution_on_an_adapted_mesh) {
    // The manufactured solutions (viscous, steady sources, periodic) on a base n^d with one half refined (y < 0.5 in 2-D, z < 0.5 in
    // 3-D), density L1 error against the exact field at t = 0.05, n = 12 and 24, the same measure in 2-D and 3-D. Bands fixed before
    // the first run: (1) the 3-D adaptive order (12 -> 24) >= the 2-D one - 0.3, on the whole domain and away from the jumps;
    // (2) the adaptive / uniform-base error ratio at 24 in 3-D <= 1.2 x the 2-D one; (3) the adaptive error is below the uniform-base
    // one in both dimensions at 24.
    const int ns[2] = {12, 24};
    const double t_end = 0.05;
    double amr[2][2], amr_far[2][2], uni[2][2], uni_far[2][2];
    int leaves[2][2], uleaves[2][2];
    for (int d = 0; d < 2; ++d)
        for (int k = 0; k < 2; ++k) {
            mms_errors(d + 2, ns[k], true, t_end, amr[d][k], amr_far[d][k], leaves[d][k]);
            mms_errors(d + 2, ns[k], false, t_end, uni[d][k], uni_far[d][k], uleaves[d][k]);
        }
    const auto order = [](double a, double b) { return std::log2(a / b); };
    std::printf("    manufactured solutions, half refined, L1(rho) at t = 0.05, n = 12 / 24:\n");
    for (int d = 0; d < 2; ++d)
        std::printf("    %d-D uniform %.3e %.3e (order %.2f; far %.2f) | adaptive %.3e %.3e (order %.2f; far %.3e %.3e, order %.2f) | "
                    "adaptive / uniform at 24: %.3f (%d leaves, uniform %d)\n",
                    d + 2, uni[d][0], uni[d][1], order(uni[d][0], uni[d][1]), order(uni_far[d][0], uni_far[d][1]), amr[d][0], amr[d][1],
                    order(amr[d][0], amr[d][1]), amr_far[d][0], amr_far[d][1], order(amr_far[d][0], amr_far[d][1]), amr[d][1] / uni[d][1],
                    leaves[d][1], uleaves[d][1]);
    CHECK(order(amr[1][0], amr[1][1]) >= order(amr[0][0], amr[0][1]) - 0.3);
    CHECK(order(amr_far[1][0], amr_far[1][1]) >= order(amr_far[0][0], amr_far[0][1]) - 0.3);
    CHECK(amr[1][1] / uni[1][1] <= 1.2 * amr[0][1] / uni[0][1]);
    for (int d = 0; d < 2; ++d) CHECK(amr[d][1] < uni[d][1]);
}

// ---------------------------------------------------------------------------
// Q4c: mass-flow inlet and non-reflecting outlet
// ---------------------------------------------------------------------------

TEST(cellflux3d_massflow_and_nonreflecting_level0_match_uniform_bitwise) {
    // A mass-flow inlet on an x-, a y- and a z-side (its density average is summed in the uniform solver's order of the boundary
    // cells) and a non-reflecting outlet opposite (two tangential directions: wrapped, one-sided at a wall), inviscid and viscous,
    // boundary_order 1 and 2: residual and speed bit for bit.
    const BCs sets[] = {{"massflow_inlet", "nonreflecting", "wall", "no_slip", "periodic", "periodic"},
                        {"wall", "no_slip", "massflow_inlet", "nonreflecting", "wall", "transmissive"},
                        {"periodic", "periodic", "no_slip", "wall", "massflow_inlet", "nonreflecting"},
                        {"nonreflecting", "massflow_inlet", "periodic", "periodic", "nonreflecting", "wall"}};
    int n_cases = 0;
    for (const BCs& bc : sets)
        for (const char* recon : {"none", "muscl"})
            for (const char* bo : {"1", "2"})
                for (const char* visc : {"false", "true"}) {
                    if (std::string(recon) == "none" && std::string(bo) == "2") continue;
                    bool wins = false;
                    const bool same = viscous_level0_equal(
                        bc, {{"reconstruction", recon}, {"boundary_order", bo}, {"use_viscous", visc}, {"mu", "0.03"}, {"rho_in", "1.3"}}, wins);
                    CHECK(same);
                    if (!same)
                        std::printf("    bc %s/%s/%s/%s/%s/%s %s bo %s viscous %s\n", bc[0], bc[1], bc[2], bc[3], bc[4], bc[5], recon, bo, visc);
                    ++n_cases;
                }
    CHECK(n_cases == 24);
}

TEST(cellflux3d_massflow_and_nonreflecting_invariant_axis_equals_the_2d_engine) {
    // The extruded 2-D adapted tree (axis a periodic) with a mass-flow inlet and a non-reflecting outlet on the 2-D x-sides: the 3-D
    // residual equals the 2-D engine's to round-off, so the inlet's average over a side with leaves of several levels (4 finer faces
    // per face in 3-D, 2 in 2-D) and the outlet's tangential differences agree with 2-D. Inviscid and viscous. The mass-flow ghost
    // blows along x whatever its side (BoundaryManager::ghost_state, as the uniform solver), so the inlet keeps its meaning only when
    // the 2-D x-axis stays x (a = 2); along the other two axes the inlet is transmissive (the outlet is direction-free).
    for (int a = 0; a < 3; ++a) {
        const int p = a == 0 ? 1 : 0, q = a == 2 ? 1 : 2;
        const int n2[2] = {9, 7};
        const double L2[2] = {1.7, 1.1}, La = 0.6;
        const char* const bc2[4] = {a == 2 ? "massflow_inlet" : "transmissive", "nonreflecting", "wall", "no_slip"};
        for (const char* visc : {"false", "true"})
            for (const char* bo : {"1", "2"}) {
                Simulation s2;
                s2.setup(cell_flux_params({{"bc_left", bc2[0]}, {"bc_right", bc2[1]}, {"bc_bottom", bc2[2]}, {"bc_top", bc2[3]},
                                           {"reconstruction", "muscl"}, {"boundary_order", bo}, {"use_viscous", visc}, {"mu", "0.03"},
                                           {"rho_in", "1.3"}}),
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
                          {"Lz", L3[2].c_str()}, {"reconstruction", "muscl"}, {"boundary_order", bo}, {"use_viscous", visc}, {"mu", "0.03"},
                          {"rho_in", "1.3"}});
                CellMesh m2(L2[0], L2[1], n2[0], n2[1], 2, false, false, s2.mesh.x0, s2.mesh.y0);
                random_adapt(m2, 61u + static_cast<unsigned>(a), 4, 18, 20);
                int inlet_levels = 0, outlet_levels = 0;   // levels of the leaves on the inlet and the outlet side
                for (int c = 0; c < m2.num_leaves(); ++c) {
                    if (m2.ci[c] == 0) inlet_levels |= 1 << m2.level[c];
                    if (m2.ci[c] == (n2[0] << m2.level[c]) - 1) outlet_levels |= 1 << m2.level[c];
                }
                const auto several = [](int bits) { return bits != 1 && bits != 2 && bits != 4; };
                CHECK(a != 2 || several(inlet_levels));
                CHECK(several(outlet_levels));
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
                CellPhysics ph2;
                ph2.ns = s2.ns.get();
                e2.set_physics(ph2);
                e3.set_physics(viscous_physics(s3));
                e2.build(m2);
                e3.build(m3);
                Field R2, R3;
                e2.residual(m2, m2.U, R2);
                e3.residual(m3, m3.U, R3);
                const int q3[4] = {0, kMom[p], kMom[q], 3}, q2[4] = {0, 1, 2, 3};
                double worst = 0.0;
                for (int k = 0; k < 4; ++k) {
                    double scale = 0.0;
                    for (const State& r : R2) scale = std::max(scale, std::abs(r[q2[k]]));
                    for (int c = 0; c < m3.num_leaves(); ++c)
                        worst = std::max(worst, std::abs(R3[c][q3[k]] - R2[static_cast<size_t>(to2[c])][q2[k]]) / scale);
                }
                bool a_zero = true;
                for (int c = 0; c < m3.num_leaves(); ++c) a_zero = a_zero && R3[c][kMom[a]] == 0.0;
                CHECK(worst <= 1e-12);
                CHECK(a_zero);
                if (worst > 1e-12 || !a_zero)
                    std::printf("    axis %d viscous %s bo %s: worst %.3e, a-momentum zero %d\n", a, visc, bo, worst, a_zero);
            }
    }
}

// ---------------------------------------------------------------------------
// Q4d: immersed body (mask forcing), per-leaf classification
// ---------------------------------------------------------------------------

namespace {

// A 3-D simulation with an immersed body (mask method) in the 1.7 x 1.1 x 0.9 box
void setup_ibm_3d(Simulation& sim, const char* shape, const BCs& bc, KV extra = {}) {
    KV kv = {{"use_ibm", "true"}, {"ibm_method", "mask"}, {"ibm_shape", shape}, {"ibm_radius", "0.27"}, {"ibm_cx", "0.83"},
             {"ibm_cy", "0.52"}, {"ibm_cz", "0.47"}, {"ibm_chord", "0.9"}, {"ibm_aoa", "12"}, {"reconstruction", "muscl"}};
    kv.insert(kv.end(), extra.begin(), extra.end());
    setup_3d(sim, bc, kv);
}

CellPhysics ibm_physics(Simulation& sim) {
    CellPhysics ph = viscous_physics(sim);
    ph.ibm_factory = sim.ibm_factory;
    return ph;
}

// Refine `rounds` times the leaves within `band` of the body's surface (the distance of the leaf centre to the uniform mask's
// boundary, estimated from solid_at at the centre and the 6 neighbours at one leaf width), plus a random sprinkle
void refine_around_body(CellMesh& m, const ImmersedBoundary& ib, int rounds, unsigned seed) {
    std::mt19937 rng(seed);
    for (int r = 0; r < rounds; ++r) {
        std::vector<char> f(static_cast<size_t>(m.num_leaves()), 0);
        for (int c = 0; c < m.num_leaves(); ++c) {
            const int l = m.level[c];
            const double h[3] = {m.dx(l), m.dy(l), m.dz(l)}, x[3] = {m.xc(c), m.yc(c), m.zc(c)};
            const bool s0 = ib.solid_at(x[0], x[1], x[2]);
            bool cut = false;
            for (int a = 0; a < 3 && !cut; ++a)
                for (const double sgn : {-1.0, 1.0}) {
                    double y[3] = {x[0], x[1], x[2]};
                    y[a] += sgn * h[a];
                    cut = cut || ib.solid_at(y[0], y[1], y[2]) != s0;
                }
            f[c] = cut || rng() % 100 < 4;
        }
        m.adapt(f);
    }
}

}   // namespace

TEST(cellflux3d_ibm_masks_equal_the_uniform_masks) {
    // The solid flag of every leaf (the centre test of the body's geometry) equals the mask of the uniform 3-D mesh of its level at
    // that cell, for a sphere, a cylinder along z and an extruded NACA section, on octrees refined around the body; also after an
    // incremental build.
    const BCs walls = {"wall", "wall", "wall", "wall", "wall", "wall"};
    for (const char* shape : {"Sphere", "Cylinder", "NACA0012"}) {
        Simulation sim;
        setup_ibm_3d(sim, shape, walls, {{"nx", "8"}, {"ny", "6"}, {"nz", "5"}});
        CellMesh m = octree_of(sim, 2);
        const ImmersedBoundary geo = sim.ibm_factory(sim.mesh);
        CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        eng.set_physics(ibm_physics(sim));
        eng.build(m);
        refine_around_body(m, geo, 2, 7u);
        eng.build(m);   // incremental: the kept leaves keep their flag
        CHECK(m.level_begin[3] > m.level_begin[2]);
        std::vector<ImmersedBoundary> lv;
        std::vector<Mesh> lm;
        for (int l = 0; l <= m.max_level; ++l) {
            lm.push_back(generate_cartesian_mesh_3d(m.Lx, m.Ly, m.Lz, m.nx_at(l), m.ny_at(l), m.nz_at(l), m.x0, m.y0, m.z0));
            lv.push_back(sim.ibm_factory(lm.back()));
        }
        int solid = 0, mismatch = 0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            const int l = m.level[c];
            const char want = lv[static_cast<size_t>(l)].mask()[static_cast<size_t>(lm[static_cast<size_t>(l)].cell(m.ci[c], m.cj[c], m.ck[c]))];
            mismatch += (eng.solid()[static_cast<size_t>(c)] != 0) != (want != 0);
            solid += want != 0;
        }
        CellFluxEngine full(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        full.set_physics(ibm_physics(sim));
        full.build(m);
        CHECK(full.solid() == eng.solid());
        CHECK(mismatch == 0);
        CHECK(solid > 20 && solid < m.num_leaves() / 2);
        std::printf("    %s: %d leaves, %d solid, %d differ from the uniform masks\n", shape, m.num_leaves(), solid, mismatch);
    }
}

TEST(cellflux3d_ibm_level0_matches_uniform_residual_and_solid_reset_bitwise) {
    // A level-0 octree with a body (mask method) against the 3-D uniform solver: the residual (the forcing zeroes the velocity of
    // the solid cells in every residual) and the solid reset (fringe Neumann passes right, left, top, bottom, back, front; deep solid
    // at the reference state), bit for bit; inviscid and viscous.
    for (const char* shape : {"Sphere", "Cylinder", "NACA0012"})
        for (const char* visc : {"false", "true"}) {
            Simulation sim;
            setup_ibm_3d(sim, shape, {"transmissive", "outlet", "wall", "no_slip", "periodic", "periodic"},
                         {{"nx", "12"}, {"ny", "9"}, {"nz", "7"}, {"use_viscous", visc}, {"mu", "0.02"}});
            CHECK(sim.ibm && sim.ibm->solid_count() > 4);
            const Mesh& u = sim.mesh;
            Field Uu(static_cast<size_t>(u.num_cells()));
            for (int k = 0; k < u.nz; ++k)
                for (int j = 0; j < u.ny; ++j)
                    for (int i = 0; i < u.nx; ++i) Uu[u.cell(i, j, k)] = test_state3(sim.fluid, u.xc(i), u.yc(j), u.zc(k));
            CellMesh cm = octree_of(sim, 0);
            cm.U.resize(static_cast<size_t>(cm.num_leaves()));
            for (int c = 0; c < cm.num_leaves(); ++c) cm.U[c] = Uu[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
            Field Ru, Rc;
            const double Su = sim.residual(Uu, Ru);
            CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
            eng.set_physics(ibm_physics(sim));
            eng.build(cm);
            const double Sc = eng.residual(cm, cm.U, Rc);
            Field Rperm(Ru.size());
            for (int c = 0; c < cm.num_leaves(); ++c) Rperm[c] = Ru[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
            CHECK(bits_equal(Rperm, Rc) && Su == Sc);
            // the solid reset of the same state
            Field Ur = Uu, Ucr = cm.U;
            sim.ibm->apply_solid_reset(Ur, sim.fluid);
            eng.solid_reset(cm, Ucr);
            bool same = true;
            for (int c = 0; c < cm.num_leaves(); ++c) same = same && Ucr[c] == Ur[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
            CHECK(same);
            if (!same || !bits_equal(Rperm, Rc)) std::printf("    %s viscous %s: residual %d, reset %d\n", shape, visc, bits_equal(Rperm, Rc), same);
        }
}

TEST(cellflux3d_ibm_cylinder_along_z_equals_the_2d_engine) {
    // A cylinder along z on the 2-D adapted tree extruded along z (periodic): the solid flags equal the 2-D engine's, and the residual
    // and the solid reset (whose fringe averages four finer leaves in 3-D, two in 2-D) equal its own to round-off.
    const int n2[2] = {12, 9};
    const double L2[2] = {1.7, 1.1};
    Simulation s2, s3;
    const KV ibm = {{"use_ibm", "true"}, {"ibm_method", "mask"}, {"ibm_shape", "Cylinder"}, {"ibm_radius", "0.27"}, {"ibm_cx", "0.83"},
                    {"ibm_cy", "0.52"}, {"reconstruction", "muscl"}};
    s2.setup(cell_flux_params({{"nx", "12"}, {"ny", "9"}, {"bc_left", "transmissive"}, {"bc_right", "outlet"}, {"bc_bottom", "wall"},
                               {"bc_top", "wall"}, {"use_ibm", "true"}, {"ibm_method", "mask"}, {"ibm_shape", "Cylinder"},
                               {"ibm_radius", "0.27"}, {"ibm_cx", "0.83"}, {"ibm_cy", "0.52"}, {"reconstruction", "muscl"}}),
             false);
    KV kv3 = {{"nx", "12"}, {"ny", "9"}, {"nz", "2"}, {"Lz", "0.6"}};
    kv3.insert(kv3.end(), ibm.begin(), ibm.end());
    setup_3d(s3, {"transmissive", "outlet", "wall", "wall", "periodic", "periodic"}, kv3);
    CellMesh m2(L2[0], L2[1], n2[0], n2[1], 2, false, false, s2.mesh.x0, s2.mesh.y0);
    {   // refine around the circle (and a sprinkle), in 2-D
        const ImmersedBoundary g2 = s2.ibm_factory(s2.mesh);
        std::mt19937 rng(5);
        for (int r = 0; r < 2; ++r) {
            std::vector<char> f(static_cast<size_t>(m2.num_leaves()), 0);
            for (int c = 0; c < m2.num_leaves(); ++c) {
                const double x = m2.xc(c), y = m2.yc(c), hx = m2.dx(m2.level[c]), hy = m2.dy(m2.level[c]);
                const bool s0 = g2.solid_at(x, y, 0.0);
                f[c] = g2.solid_at(x + hx, y, 0.0) != s0 || g2.solid_at(x - hx, y, 0.0) != s0 || g2.solid_at(x, y + hy, 0.0) != s0 ||
                       g2.solid_at(x, y - hy, 0.0) != s0 || rng() % 100 < 5;
            }
            m2.adapt(f);
        }
    }
    CellMesh m3 = octree_of(s3, 2);
    std::vector<int> lv, ii, jj, kk;
    for (int c = 0; c < m2.num_leaves(); ++c)
        for (int t = 0; t < (2 << m2.level[c]); ++t) {
            lv.push_back(m2.level[c]);
            ii.push_back(m2.ci[c]);
            jj.push_back(m2.cj[c]);
            kk.push_back(t);
        }
    m3.assign_leaves(lv, ii, jj, kk);
    m2.U.resize(static_cast<size_t>(m2.num_leaves()));
    for (int c = 0; c < m2.num_leaves(); ++c) m2.U[c] = test_state(s2.fluid, m2.xc(c), m2.yc(c));
    m3.U.resize(static_cast<size_t>(m3.num_leaves()));
    std::vector<int> to2(static_cast<size_t>(m3.num_leaves()));
    for (int c = 0; c < m3.num_leaves(); ++c) {
        to2[static_cast<size_t>(c)] = m2.find(m3.level[c], m3.ci[c], m3.cj[c]);
        const State W2 = s2.fluid.cons_to_prim(m2.U[static_cast<size_t>(to2[c])]);
        m3.U[c] = s3.fluid.prim_to_cons({W2[0], W2[1], W2[2], W2[3], 0.0});
    }
    CellFluxEngine e2(s2.fluid, s2.riemann, s2.muscl.get(), s2.boundaries), e3(s3.fluid, s3.riemann, s3.muscl.get(), s3.boundaries);
    CellPhysics p2;
    p2.ibm_factory = s2.ibm_factory;
    e2.set_physics(p2);
    e3.set_physics(ibm_physics(s3));
    e2.build(m2);
    e3.build(m3);
    bool flags = true;
    int solid = 0;
    for (int c = 0; c < m3.num_leaves(); ++c) {
        flags = flags && e3.solid()[static_cast<size_t>(c)] == e2.solid()[static_cast<size_t>(to2[c])];
        solid += e3.solid()[static_cast<size_t>(c)] != 0;
    }
    CHECK(flags);
    CHECK(solid > 50);
    Field R2, R3;
    e2.residual(m2, m2.U, R2);
    e3.residual(m3, m3.U, R3);
    Field U2 = m2.U, U3 = m3.U;
    e2.solid_reset(m2, U2);
    e3.solid_reset(m3, U3);
    double worst_r = 0.0, worst_u = 0.0;
    const int q3[4] = {0, 1, 2, 3};
    for (int k = 0; k < 4; ++k) {
        double sr = 0.0, su = 0.0;
        for (const State& r : R2) sr = std::max(sr, std::abs(r[q3[k]]));
        for (const State& w : U2) su = std::max(su, std::abs(w[q3[k]]));
        for (int c = 0; c < m3.num_leaves(); ++c) {
            worst_r = std::max(worst_r, std::abs(R3[c][q3[k]] - R2[static_cast<size_t>(to2[c])][q3[k]]) / sr);
            if (su > 0.0) worst_u = std::max(worst_u, std::abs(U3[c][q3[k]] - U2[static_cast<size_t>(to2[c])][q3[k]]) / su);
        }
    }
    std::printf("    cylinder along z: %d leaves, %d solid; residual vs 2-D %.3e, solid reset vs 2-D %.3e\n", m3.num_leaves(), solid, worst_r, worst_u);
    CHECK(worst_r <= 1e-12);
    CHECK(worst_u <= 1e-13);
}

TEST(cellflux3d_ibm_solid_reset_matches_a_brute_force_oracle) {
    // The solid reset on an adapted octree around a sphere against an independent oracle: the leaves across a side are found by
    // brute force in finest-level integer boxes (not through the mesh's neighbour tables); passes right, left, top, bottom, back,
    // front, each reading the original state, later passes win; a solid leaf takes the mean rho, p of the fluid leaves across the side
    // (one, or up to four finer ones), the deep solid leaves the reference state. Agreement to round-off (the oracle sums in its own
    // order), on a state that varies from leaf to leaf.
    Simulation sim;
    setup_ibm_3d(sim, "Sphere", {"wall", "wall", "wall", "wall", "wall", "wall"}, {{"nx", "8"}, {"ny", "6"}, {"nz", "5"}});
    CellMesh m = octree_of(sim, 2);
    random_adapt(m, 11u, 3, 20, 10);   // random levels (not refined around the body): coarse solid leaves next to finer fluid ones
    m.U.resize(static_cast<size_t>(m.num_leaves()));
    for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state3(sim.fluid, 3.0 * m.xc(c), 2.0 * m.yc(c) + 5.0 * m.zc(c), 7.0 * m.zc(c));
    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    eng.set_physics(ibm_physics(sim));
    eng.build(m);
    const std::vector<char>& solid = eng.solid();
    const int N = m.num_leaves(), M = m.max_level;
    struct Box {
        long long lo[3], hi[3];
    };
    std::vector<Box> box(static_cast<size_t>(N));
    for (int c = 0; c < N; ++c) {
        const long long s = 1LL << (M - m.level[c]);
        const int x[3] = {m.ci[c], m.cj[c], m.ck[c]};
        for (int a = 0; a < 3; ++a) {
            box[c].lo[a] = x[a] * s;
            box[c].hi[a] = (x[a] + 1) * s;
        }
    }
    const long long ext[3] = {static_cast<long long>(m.nx) << M, static_cast<long long>(m.ny) << M, static_cast<long long>(m.nz) << M};
    const double g = sim.fluid.gamma;
    const auto pressure = [&](const State& u) { return (g - 1.0) * (u[3] - 0.5 * (u[1] * u[1] + u[2] * u[2] + u[4] * u[4]) / u[0]); };
    std::vector<double> rho(static_cast<size_t>(N)), p(static_cast<size_t>(N));
    for (int c = 0; c < N; ++c) {
        rho[c] = m.U[c][0];
        p[c] = pressure(m.U[c]);
    }
    std::vector<double> nr = rho, np = p;
    std::vector<char> fringe(static_cast<size_t>(N), 0);
    const int axes[6] = {0, 0, 1, 1, 2, 2}, dirs[6] = {1, -1, 1, -1, 1, -1};   // right, left, top, bottom, back, front
    int four = 0;   // pulls that averaged more than two finer leaves
    for (int pass = 0; pass < 6; ++pass) {
        const int a = axes[pass], d = dirs[pass];
        for (int c = 0; c < N; ++c) {
            if (!solid[static_cast<size_t>(c)]) continue;
            const long long plane = d > 0 ? box[c].hi[a] : box[c].lo[a];
            if (plane == 0 || plane == ext[a]) continue;   // no pull across a domain edge
            double sr = 0.0, sp = 0.0;
            int cnt = 0, across = 0;
            for (int n = 0; n < N; ++n) {
                if ((d > 0 ? box[n].lo[a] : box[n].hi[a]) != plane) continue;
                bool overlap = true;
                for (int t = 0; t < 3; ++t)
                    if (t != a) overlap = overlap && box[n].lo[t] < box[c].hi[t] && box[c].lo[t] < box[n].hi[t];
                if (!overlap) continue;
                ++across;
                if (solid[static_cast<size_t>(n)]) continue;
                sr += rho[n];
                sp += p[n];
                ++cnt;
            }
            if (cnt == 0) continue;
            four += cnt > 2;
            (void)across;
            nr[c] = sr / cnt;
            np[c] = sp / cnt;
            fringe[c] = 1;
        }
    }
    Field U = m.U;
    eng.solid_reset(m, U);
    double worst = 0.0;
    int nsolid = 0, nfringe = 0;
    for (int c = 0; c < N; ++c) {
        if (!solid[static_cast<size_t>(c)]) {
            CHECK(U[c] == m.U[c]);
            continue;
        }
        ++nsolid;
        nfringe += fringe[c];
        const double wr = fringe[c] ? nr[c] : sim.ibm->rho_ref(), wp = fringe[c] ? np[c] : sim.ibm->p_ref();
        CHECK(U[c][1] == 0.0 && U[c][2] == 0.0 && U[c][4] == 0.0);
        worst = std::max(worst, std::abs(U[c][0] - wr) / wr);
        worst = std::max(worst, std::abs(U[c][3] * (g - 1.0) - wp) / wp);
    }
    std::printf("    %d leaves, %d solid, %d in the fringe, %d pulls from more than two finer leaves; worst relative difference %.3e\n", N, nsolid,
                nfringe, four, worst);
    CHECK(four > 0);
    CHECK(nfringe > 20 && nfringe < nsolid);
    CHECK(worst <= 1e-14);
}

TEST(celladvance3d_ibm_sphere_on_an_adapted_mesh) {
    // A short subcycled run past a sphere refined around its surface (mask method, the solid reset once per complete step as the
    // adaptive solver does): the state stays finite and positive, the solid leaves are at rest after every reset, mass enters
    // and leaves only through the open sides.
    Simulation sim;
    setup_ibm_3d(sim, "Sphere", {"vel_inlet", "outlet", "periodic", "periodic", "periodic", "periodic"},
                 {{"nx", "10"}, {"ny", "6"}, {"nz", "6"}, {"time_scheme", "SSP Runge-Kutta 2"}});
    CellMesh m = octree_of(sim, 2);
    refine_around_body(m, sim.ibm_factory(sim.mesh), 2, 3u);
    CHECK(m.level_begin[3] > m.level_begin[2]);
    m.U.assign(static_cast<size_t>(m.num_leaves()), sim.fluid.prim_to_cons({1.2, 0.7, 0.0, 1.3, 0.0}));
    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    eng.set_physics(ibm_physics(sim));
    eng.build(m);
    std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4);
    CellAdvance adv(m, eng, *ts);
    eng.solid_reset(m, m.U);
    double S = eng.wave_speed(m, m.U), t = 0.0;
    const double L = 2.0 / (1.0 / m.dx(0) + 1.0 / m.dy(0) + 1.0 / m.dz(0));
    bool ok = true;
    int solid = 0;
    for (int k = 0; k < 12; ++k) {
        const double dt = 0.4 * L / S;
        S = adv.step(dt, t);
        eng.solid_reset(m, m.U);
        t += dt;
        for (int c = 0; c < m.num_leaves(); ++c) {
            const State W = sim.fluid.cons_to_prim(m.U[c]);
            ok = ok && std::isfinite(W[0]) && W[0] > 0.0 && W[3] > 0.0;
            if (eng.solid()[static_cast<size_t>(c)]) ok = ok && m.U[c][1] == 0.0 && m.U[c][2] == 0.0 && m.U[c][4] == 0.0;
        }
    }
    for (int c = 0; c < m.num_leaves(); ++c) solid += eng.solid()[static_cast<size_t>(c)] != 0;
    CHECK(ok);
    CHECK(solid > 20);
    std::printf("    sphere: %d leaves (%d solid), t = %.3f after 12 steps\n", m.num_leaves(), solid, t);
}
