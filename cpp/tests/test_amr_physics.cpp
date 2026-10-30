// AMR: viscous, forces, sources, immersed body, non-reflecting outlet, physics on adapted meshes
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_amr_common.hpp"

// ---------------------------------------------------------------------------
// CellFluxEngine physics (viscous, forces, sources, immersed body, non-reflecting outlet)
// ---------------------------------------------------------------------------


TEST(cellphys_viscous_matches_uniform_bitwise) {
    struct BCs {
        const char *l, *r, *b, *t;
    };
    const BCs sets[] = {{"wall", "wall", "wall", "wall"},
                        {"transmissive", "outlet", "no_slip", "moving_wall"},
                        {"vel_inlet", "outlet", "wall", "transmissive"},
                        {"massflow_inlet", "transmissive", "no_slip", "no_slip"},
                        {"periodic", "periodic", "no_slip", "moving_wall"},
                        {"moving_wall", "no_slip", "periodic", "periodic"},
                        {"periodic", "periodic", "periodic", "periodic"}};
    int n = 0;
    for (const BCs& bc : sets)
        for (const char* muscl : {"false", "true"}) {
            ParityRig rig({{"bc_left", bc.l}, {"bc_right", bc.r}, {"bc_bottom", bc.b}, {"bc_top", bc.t},
                           {"use_viscous", "true"}, {"mu", "0.05"}, {"use_muscl", muscl}});
            const bool same = rig.residual_equal();
            CHECK(same);
            if (!same) std::printf("    viscous %s/%s/%s/%s muscl %s\n", bc.l, bc.r, bc.b, bc.t, muscl);
            ++n;
        }
    CHECK(n == 14);
    // parallel size
    ParityRig big({{"nx", "96"}, {"ny", "64"}, {"bc_left", "periodic"}, {"bc_right", "periodic"}, {"bc_bottom", "no_slip"},
                   {"bc_top", "moving_wall"}, {"use_viscous", "true"}, {"mu", "0.05"}, {"use_muscl", "true"}});
    CHECK(big.residual_equal());
}

TEST(cellphys_forces_and_sources_match_uniform_bitwise) {
    ParityRig bf({{"use_viscous", "true"}, {"mu", "0.03"}, {"use_bodyforce", "true"}, {"gx", "0.3"}, {"gy", "-0.7"},
                  {"bc_bottom", "no_slip"}, {"use_muscl", "true"}});
    CHECK(bf.sim.body_force != nullptr && bf.residual_equal());
    ParityRig steady({{"use_viscous", "true"}, {"mu", "0.03"}, {"source_rho", "0.1*sin(3*x)*cos(2*y)"},
                      {"source_mx", "0.2*x*y"}, {"source_my", "0.3"}, {"source_E", "0.5*cos(x+y)"}, {"use_muscl", "true"}});
    CHECK(steady.sim.p.use_source && !steady.sim.p.source_unsteady && steady.residual_equal());
    ParityRig unsteady({{"source_rho", "0.1*sin(3*x+t)"}, {"source_E", "0.5*cos(x+y)*t"}, {"use_muscl", "true"}});
    CHECK(unsteady.sim.p.source_unsteady);
    CHECK(unsteady.residual_equal(0.0) && unsteady.residual_equal(0.37) && unsteady.residual_equal(1.25));
    // the time actually enters
    unsteady.sim.scheme->stage_time = 0.0;
    unsteady.eng->stage_time = 0.0;
    Field a, b;
    unsteady.eng->residual(*unsteady.cm, unsteady.cm->U, a);
    unsteady.eng->stage_time = 0.9;
    unsteady.eng->residual(*unsteady.cm, unsteady.cm->U, b);
    CHECK(!bits_equal(a, b));
}

TEST(cellphys_immersed_body_matches_uniform_bitwise) {
    for (const char* viscous : {"false", "true"}) {
        ParityRig rig({{"nx", "36"}, {"ny", "24"}, {"use_ibm", "true"}, {"ibm_shape", "Cylinder"}, {"ibm_radius", "0.25"},
                       {"ibm_cx", "0.8"}, {"ibm_cy", "0.55"}, {"use_viscous", viscous}, {"mu", "0.02"},
                       {"use_muscl", "true"}, {"bc_left", "vel_inlet"}, {"bc_right", "outlet"}});
        CHECK(rig.sim.ibm->solid_count() > 20);
        CHECK(rig.eng->has_body());
        CHECK(rig.residual_equal());
        // the solid reset: fringe Neumann + deep solid
        rig.sim.ibm->apply_solid_reset(rig.Uu, rig.sim.fluid);
        rig.eng->solid_reset(*rig.cm, rig.cm->U);
        const Mesh& u = rig.sim.mesh;
        Field perm(rig.cm->U.size());
        for (int c = 0; c < rig.cm->num_leaves(); ++c) perm[c] = rig.Uu[rig.cm->cj[c] * u.nx + rig.cm->ci[c]];
        CHECK(bits_equal(perm, rig.cm->U));
        CHECK(rig.residual_equal());
    }
}

TEST(cellphys_nonreflecting_matches_uniform_bitwise) {
    for (const char* beta : {"-1", "0.3", "1"}) {
        ParityRig rig({{"bc_right", "nonreflecting"}, {"bc_top", "nonreflecting"}, {"bc_left", "outlet"}, {"bc_bottom", "wall"},
                       {"p_out", "17.5"}, {"nrbc_beta", beta}, {"use_muscl", "true"}});
        CHECK(rig.residual_equal());
        ParityRig per({{"bc_right", "nonreflecting"}, {"bc_left", "wall"}, {"bc_bottom", "periodic"}, {"bc_top", "periodic"},
                       {"p_out", "17.5"}, {"nrbc_beta", beta}, {"use_viscous", "true"}, {"mu", "0.02"}});
        CHECK(per.residual_equal());
    }
}

TEST(cellphys_viscous_ghost_matches_the_padded_state) {
    // BoundaryManager::viscous_ghost is the 2-D rule of get_padded_U for every boundary type
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 5, 4);
    const EOS eos = EOS::ideal(1.4);
    for (BCType t : {BCType::Wall, BCType::NoSlip, BCType::MovingWall, BCType::VelInlet, BCType::MassflowInlet, BCType::Outlet,
                     BCType::Transmissive, BCType::NonReflecting})
        for (int side = 0; side < 4; ++side) {
            BoundaryManager bm(m, eos);
            bm.set_boundary_types(BCType::Transmissive, BCType::Transmissive, BCType::Transmissive, BCType::Transmissive);
            bm.set_type(static_cast<Face>(side), t);
            BCParams bp;
            bp.u_in = 0.8;
            bp.v_in = -0.3;
            bm.set_params(bp);
            Field U(20);
            for (int c = 0; c < 20; ++c) U[c] = test_state(eos, 0.2 * c, 0.1 * c + 0.3);
            Field P;
            bm.get_padded_U(U, P);
            const int W = 7;
            for (int k = 0; k < (side < 2 ? 4 : 5); ++k) {
                int cell, pad;
                switch (side) {
                case 0: cell = k * 5; pad = (k + 1) * W; break;
                case 1: cell = k * 5 + 4; pad = (k + 1) * W + 6; break;
                case 2: cell = k; pad = k + 1; break;
                default: cell = 15 + k; pad = 5 * W + k + 1; break;
                }
                const State g = bm.viscous_ghost(static_cast<Face>(side), U[cell]);
                CHECK(g == P[pad]);
            }
        }
}

// ---------------------------------------------------------------------------
// Physics on adapted meshes
// ---------------------------------------------------------------------------


TEST(celladvance_viscous_conservation_free_stream_and_control) {
    const EOS eos = EOS::ideal(1.4);
    NavierStokes ns(eos, 0.08, 0.72);
    for (int sc = 0; sc < 3; ++sc) {
        CellSetup su(true);
        CellMesh m = random_adapted_mesh(true, true, 91u + static_cast<unsigned>(sc), 6);
        CHECK(m.level_begin[3] < m.num_leaves());
        CellPhysics ph;
        ph.ns = &ns;
        const auto run = [&](bool replace, bool uniform_state) {
            CellMesh mm = m;
            mm.U.resize(mm.num_leaves());
            for (int c = 0; c < mm.num_leaves(); ++c)
                mm.U[c] = uniform_state ? eos.prim_to_cons({1.3, 0.8, -0.45, 2.1}) : test_state(eos, mm.xc(c), mm.yc(c));
            const Field U0 = mm.U;
            CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
            eng.set_physics(ph);
            eng.build(mm);
            std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(kSchemeNames[sc]), 0.3, 1.0, 5, 1e-3);
            CellAdvance adv(mm, eng, *ts);
            adv.flux_replacement = replace;
            Field R;
            double S = eng.residual(mm, mm.U, R);
            double before[4], mag[4], worst = 0.0;
            for (int q = 0; q < 4; ++q) {
                before[q] = cell_total(mm, q);
                mag[q] = 0.0;
                for (int c = 0; c < mm.num_leaves(); ++c) mag[q] += std::abs(mm.U[c][q]) * mm.area(c);
            }
            for (int k = 0; k < 5; ++k) S = adv.step(0.3 * mm.dx(0) / S, k * 0.01);
            for (int q = 0; q < 4; ++q) worst = std::max(worst, std::abs(cell_total(mm, q) - before[q]) / mag[q]);
            double drift = 0.0;
            for (int c = 0; c < mm.num_leaves(); ++c)
                for (int q = 0; q < kNVar; ++q) drift = std::max(drift, std::abs(mm.U[c][q] - U0[c][q]));
            return std::make_pair(worst, drift);
        };
        const auto with = run(true, false), without = run(false, false), stream = run(true, true);
        CHECK(with.first < 1e-13);
        CHECK(without.first > 1e-8);   // negative control
        CHECK(stream.second < 1e-14);   // a uniform state has zero gradients: nothing moves
        if (!(with.first < 1e-13)) std::printf("    %s: viscous conservation error %.3e\n", kSchemeNames[sc], with.first);
    }
}

TEST(celladvance_fully_refined_mesh_equals_the_finer_uniform_mesh) {
    // every leaf at level 1 (level 0 empty): the same arithmetic as the uniform 64x64 mesh with two substeps of dt/2
    Simulation sim;
    sim.setup(params_for("Manufactured Solution", {{"nx", "32"}, {"ny", "32"}}), false);
    CellMesh amr(sim.mesh.Lx, sim.mesh.Ly, 32, 32, 1, true, true);
    amr.adapt(std::vector<char>(static_cast<size_t>(amr.num_leaves()), 1));
    CellMesh uni(sim.mesh.Lx, sim.mesh.Ly, 64, 64, 0, true, true);
    CHECK(amr.num_leaves() == uni.num_leaves() && amr.level_begin[1] == 0);
    init_cell_case(amr, sim, "Manufactured Solution");
    init_cell_case(uni, sim, "Manufactured Solution");
    CHECK(amr.U == uni.U);
    CellFluxEngine ea(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries), eu(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    ea.set_physics(physics_of(sim));
    eu.set_physics(physics_of(sim));
    ea.build(amr);
    eu.build(uni);
    std::unique_ptr<TimeScheme> ta = make_time_scheme(parse_time_scheme(sim.p.time_scheme), 0.4, 1.0, 5, 1e-3);
    std::unique_ptr<TimeScheme> tu = make_time_scheme(parse_time_scheme(sim.p.time_scheme), 0.4, 1.0, 5, 1e-3);
    CellAdvance aa(amr, ea, *ta), au(uni, eu, *tu);
    const double dt = 1e-3;
    for (int k = 0; k < 3; ++k) {
        aa.step(dt, k * dt);
        au.step(dt / 2, k * dt);
        au.step(dt / 2, k * dt + dt / 2);
    }
    CHECK(amr.U == uni.U);
}

TEST(celladvance_adapted_manufactured_solutions_regression_guards) {
    // Base 32x32 with the lower half refined once, against the uniform 32 (coarse) and 64 (fine, the reference). Acceptance,
    // fixed before the first run: the error to the fine solution is below 0.75 x the coarse one. With the injected jump closure
    // it was not met (1.1-1.3 x: the first-order coarse side of a jump puts an error of the size of the whole coarse-grid error
    // in a strip along each jump, see the closure study); with the default closure (interpolated fine side, second-order coarse
    // side) it is (about 0.7 x). The split below is a regression guard taken from measurement (default closure): away from the
    // jumps the adaptive error must be below 0.8 x the coarse one (measured 0.6), near them below 1.2 x (measured 0.85).
    struct Case {
        const char* name;
        double t_end;
    };
    for (const Case& cs : {Case{"Manufactured Solution", 0.05}, Case{"Manufactured Solution Unsteady Euler", 0.05}}) {
        Simulation s32, s64;
        s32.setup(params_for(cs.name, {{"nx", "32"}, {"ny", "32"}}), false);
        s64.setup(params_for(cs.name, {{"nx", "64"}, {"ny", "64"}}), false);
        CellMesh fine(1.0, 1.0, 64, 64, 0, true, true), coarse(1.0, 1.0, 32, 32, 0, true, true), amr(1.0, 1.0, 32, 32, 1, true, true);
        std::vector<char> r(static_cast<size_t>(amr.num_leaves()), 0);
        for (int c = 0; c < amr.num_leaves(); ++c) r[c] = amr.yc(c) < 0.5;
        amr.adapt(r);
        CHECK(amr.num_leaves() > 32 * 32 && amr.num_leaves() < 64 * 64);
        init_cell_case(fine, s64, cs.name);
        init_cell_case(coarse, s32, cs.name);
        init_cell_case(amr, s32, cs.name);
        run_cell_mesh(fine, s64, cs.t_end);
        run_cell_mesh(coarse, s32, cs.t_end);
        run_cell_mesh(amr, s32, cs.t_end);
        const double e_coarse = density_error(coarse, fine), e_amr = density_error(amr, fine);
        std::printf("    %s: L1 density difference to the 64x64 solution: uniform 32 %.4e, adaptive %.4e (%d leaves)\n", cs.name,
                    e_coarse, e_amr, amr.num_leaves());
        {   // where is the error? away from the two jump lines (y = 0 ~ 1 and y = 0.5) or within 0.12 of them
            std::vector<double> grid(64 * 64);
            for (int c = 0; c < fine.num_leaves(); ++c) grid[static_cast<size_t>(fine.cj[c]) * 64 + fine.ci[c]] = fine.U[c][0];
            double far_a = 0, near_a = 0, far_c = 0, near_c = 0, far_area = 0, near_area = 0;
            const auto err_of = [&](const CellMesh& m, double& far_e, double& near_e, bool count) {
                for (int c = 0; c < m.num_leaves(); ++c) {
                    const int rr = 64 / m.nx_at(m.level[c]);
                    double mean = 0.0;
                    for (int j = 0; j < rr; ++j)
                        for (int i = 0; i < rr; ++i) mean += grid[static_cast<size_t>(m.cj[c] * rr + j) * 64 + m.ci[c] * rr + i];
                    mean /= rr * rr;
                    const double y = m.yc(c), d = std::min(std::min(std::abs(y - 0.5), y), 1.0 - y);
                    const double e = std::abs(m.U[c][0] - mean) * m.area(c);
                    if (d > 0.12) { far_e += e; if (count) far_area += m.area(c); }
                    else { near_e += e; if (count) near_area += m.area(c); }
                }
            };
            err_of(amr, far_a, near_a, true);
            err_of(coarse, far_c, near_c, false);
            std::printf("    error split (sum |d rho| area): away from the jumps adaptive %.3e coarse %.3e | near the jumps adaptive %.3e coarse %.3e\n",
                        far_a, far_c, near_a, near_c);
            CHECK(far_a < 0.8 * far_c);
            CHECK(near_a < 1.2 * near_c);
        }
        CHECK(e_coarse > 1e-7);
        CHECK(e_amr < 0.75 * e_coarse);
    }
}

TEST(celladvance_jump_closure_study) {
    // How big is the error of the jump closure, and does it shrink with resolution?  Steady manufactured solution, base
    // n x n with the half y < 0.5 refined once, against the uniform n x n run and the uniform 2n x 2n reference at t = 0.05.
    // Mean absolute density difference to the reference per unit area in bins of the distance to the nearest jump line (in
    // coarse cell widths), on the fine and on the coarse side. Closures: 0 injection of the coarse value (the original),
    // 1 the coarse value interpolated to the virtual fine cell (option b), 2 = 1 plus a second-order coarse side (option c,
    // an experiment).
    const char* name = "Manufactured Solution";
    const char* const closure_name[4] = {"injection", "interpolated (b)", "interpolated + 2nd-order coarse side (b+c)", "injection + 2nd-order coarse side (c)"};
    double ratio_prev[4] = {0.0, 0.0, 0.0, 0.0};
    for (int n : {16, 32, 64}) {
        Simulation sn, s2;
        sn.setup(params_for(name, {{"nx", std::to_string(n).c_str()}, {"ny", std::to_string(n).c_str()}}), false);
        s2.setup(params_for(name, {{"nx", std::to_string(2 * n).c_str()}, {"ny", std::to_string(2 * n).c_str()}}), false);
        CellMesh ref(1.0, 1.0, 2 * n, 2 * n, 0, true, true), coarse(1.0, 1.0, n, n, 0, true, true);
        init_cell_case(ref, s2, name);
        init_cell_case(coarse, sn, name);
        run_cell_mesh(ref, s2, 0.05);
        run_cell_mesh(coarse, sn, 0.05);
        std::vector<double> grid(static_cast<size_t>(4 * n) * n);
        for (int c = 0; c < ref.num_leaves(); ++c) grid[static_cast<size_t>(ref.cj[c]) * 2 * n + ref.ci[c]] = ref.U[c][0];
        const double hc = 1.0 / n;
        // bins: [0,1), [1,2), [2,4), [4,inf) coarse widths; index = side * 4 + bin, side 0 = fine region (y < 0.5)
        const auto accumulate = [&](const CellMesh& m, double* e, double* a) {
            for (int c = 0; c < m.num_leaves(); ++c) {
                const int rr = 2 * n / m.nx_at(m.level[c]);
                double mean = 0.0;
                for (int j = 0; j < rr; ++j)
                    for (int i = 0; i < rr; ++i) mean += grid[static_cast<size_t>(m.cj[c] * rr + j) * 2 * n + m.ci[c] * rr + i];
                mean /= rr * rr;
                const double y = m.yc(c), d = std::min(std::min(std::abs(y - 0.5), y), 1.0 - y) / hc;
                const int bin = d < 1 ? 0 : (d < 2 ? 1 : (d < 4 ? 2 : 3)), idx = (y < 0.5 ? 0 : 4) + bin;
                e[idx] += std::abs(m.U[c][0] - mean) * m.area(c);
                if (a) a[idx] += m.area(c);
            }
        };
        double ec[8] = {}, ar[8] = {};
        accumulate(coarse, ec, nullptr);
        for (int closure = 0; closure < 4; ++closure) {
            CellMesh amr(1.0, 1.0, n, n, 1, true, true);
            std::vector<char> r(static_cast<size_t>(amr.num_leaves()), 0);
            for (int c = 0; c < amr.num_leaves(); ++c) r[c] = amr.yc(c) < 0.5;
            amr.adapt(r);
            init_cell_case(amr, sn, name);
            run_cell_mesh(amr, sn, 0.05, 100000, closure);
            double ea[8] = {};
            std::fill(ar, ar + 8, 0.0);
            accumulate(amr, ea, ar);
            std::printf("    n = %d, closure %s: error per unit area, adaptive (ratio to uniform n)\n", n, closure_name[closure]);
            for (int side = 0; side < 2; ++side) {
                std::printf("      %s side:", side == 0 ? "fine  " : "coarse");
                for (int b = 0; b < 4; ++b) {
                    const int i = side * 4 + b;
                    if (ar[i] > 0.0) std::printf("  bin%d %.2e (%.2f)", b, ea[i] / ar[i], ea[i] / ec[i]);
                }
                std::printf("\n");
            }
            double near_a = 0.0, near_c = 0.0;
            for (int i : {0, 1, 4, 5}) {
                near_a += ea[i];
                near_c += ec[i];
            }
            const double ratio = near_a / near_c;
            std::printf("      within 2 coarse widths of a jump: adaptive / uniform = %.2f\n", ratio);
            if (n > 16) CHECK(ratio < 4.0 * ratio_prev[closure] + 1.0);   // loose guard against a runaway
            ratio_prev[closure] = ratio;
        }
    }
}

TEST(cellflux_incremental_rebuild_equals_full_rebuild) {
    // After every adapt of a run of random refinements and joins, the faces and gather lists that CellMesh::adapt updates
    // (copied for the clean leaves) and the engine tables that CellFluxEngine::build updates must equal a full rebuild of the
    // same leaves bit for bit: walled and periodic axes, interpolated jumps on and off, an immersed body, steady sources.
    const EOS eos = EOS::ideal(1.4);
    unsigned seed = 2024u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<double>((seed >> 8) & 0xFFFFu) / 65536.0;
    };
    for (int cfg = 0; cfg < 4; ++cfg) {
        const bool periodic = cfg & 1, ip = (cfg & 2) == 0;
        const int nx = 24, ny = 16, maxl = 3;
        const double Lx = 3.0, Ly = 2.0;
        const Mesh base = generate_cartesian_mesh(Lx, Ly, nx, ny);
        BoundaryManager bm(base, eos);
        if (periodic)
            bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Wall, BCType::Wall);
        else
            bm.set_boundary_types(BCType::Transmissive, BCType::Transmissive, BCType::Wall, BCType::Wall);
        RiemannSolver rs(eos, RiemannType::HLLC);
        MusclReconstruction mu(eos, Limiter::VanLeer);
        Params src;
        src.set("source_rho", "0.1*sin(x)*cos(y)");
        src.set("source_E", "x*y");
        CellPhysics ph;
        ph.source = &src;
        ph.source_mesh = &base;
        ph.ibm_factory = [](const Mesh& m) { return ImmersedBoundary(m, "Cylinder", 0.3, 1.2, 1.0, 1.0, 1.0); };
        const auto make_engine = [&]() {
            auto e = std::make_unique<CellFluxEngine>(eos, rs, &mu, bm);
            e->interpolate_jumps = ip;
            e->set_physics(ph);
            return e;
        };
        CellMesh m(Lx, Ly, nx, ny, maxl, periodic, false);
        auto eng = make_engine();
        eng->build(m);
        int incremental = 0, clean_total = 0, bad_mesh = 0, bad_eng = 0;
        for (int it = 0; it < 14; ++it) {
            // refine near a moving spot and at random, join at random
            const double sx = 0.4 + 0.15 * it, sy = 1.0 + 0.5 * std::sin(0.7 * it);
            std::vector<char> r(static_cast<size_t>(m.num_leaves()), 0), co(r.size(), 0);
            for (int c = 0; c < m.num_leaves(); ++c) {
                const double d = std::hypot(m.xc(c) - sx, m.yc(c) - sy);
                r[c] = (d < 0.35 && rnd() < 0.8) || rnd() < 0.01;
                co[c] = !r[c] && rnd() < 0.7;
            }
            if (!m.adapt(r, co)) continue;
            CHECK(static_cast<int>(m.prev_leaf.size()) == m.num_leaves());
            for (const char d : m.topo_dirty) clean_total += d ? 0 : 1;
            eng->build(m);
            ++incremental;
            CellMesh full(Lx, Ly, nx, ny, maxl, periodic, false);
            full.assign_leaves(m.level, m.ci, m.cj);
            bool same = full.faces.size() == m.faces.size() && full.side_ptr == m.side_ptr && full.side_face == m.side_face &&
                        full.face_level_begin == m.face_level_begin;
            for (size_t f = 0; same && f < m.faces.size(); ++f) {
                const CellFace &a = m.faces[f], &b = full.faces[f];
                same = a.left == b.left && a.right == b.right && a.level == b.level && a.tpos == b.tpos && a.axis == b.axis &&
                       a.kind == b.kind && a.side == b.side && std::memcmp(&a.length, &b.length, sizeof(double)) == 0;
            }
            bad_mesh += same ? 0 : 1;
            auto ref = make_engine();
            ref->build(m);   // a new engine: a full build
            bad_eng += eng->same_tables(*ref) ? 0 : 1;
        }
        CHECK(incremental >= 8);
        CHECK(clean_total > 0);
        CHECK(bad_mesh == 0);
        CHECK(bad_eng == 0);
        if (bad_mesh || bad_eng)
            std::printf("    periodic %d interpolate %d: %d of %d meshes, %d engines differ\n", periodic, ip, bad_mesh, incremental,
                        bad_eng);
    }
}

TEST(cellflux_interpolated_jump_values_are_exact_for_a_linear_field) {
    // A linear primitive field: the value interpolated to the virtual fine cell across a jump is the exact one there (all
    // limiters return the exact slope on linear data). Control: the injected coarse value is not.
    const EOS eos = EOS::ideal(1.4);
    const auto prim = [&](double x, double y) {
        return State{1.0 + 0.05 * x + 0.02 * y, 0.3 + 0.04 * x - 0.01 * y, -0.2 + 0.03 * y, 2.0 - 0.06 * x + 0.05 * y, 0.0};
    };
    for (const Limiter lim : {Limiter::Minmod, Limiter::VanLeer, Limiter::MC, Limiter::Koren}) {
        CellMesh m(8.0, 8.0, 8, 8, 2);
        std::vector<char> r(static_cast<size_t>(m.num_leaves()), 0);
        r[m.find(0, 3, 3)] = 1;   // one interior block refined: its coarse neighbours have same-level neighbours only
        m.adapt(r);
        BoundaryManager bm(generate_cartesian_mesh(8.0, 8.0, 8, 8), eos);
        bm.set_boundary_types(BCType::Transmissive, BCType::Transmissive, BCType::Transmissive, BCType::Transmissive);
        RiemannSolver rs(eos, RiemannType::HLLC);
        MusclReconstruction mu(eos, lim);
        CellFluxEngine eng(eos, rs, &mu, bm);
        eng.build(m);
        m.U.resize(m.num_leaves());
        for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = eos.prim_to_cons(prim(m.xc(c), m.yc(c)));
        Field R;
        eng.residual(m, m.U, R);
        CHECK(eng.interpolated().size() == 8);   // each of the 4 fine leaves faces a coarse leaf on two sides
        double worst = 0.0, injected = 0.0;
        for (size_t k = 0; k < eng.interpolated().size(); ++k) {
            const CellFluxEngine::InterpVirtual& iv = eng.interpolated()[k];
            const double h = m.dx(m.level[iv.fine]);
            const int s = static_cast<int>(iv.side);
            const double x = m.xc(iv.fine) + (s == 1 ? h : (s == 0 ? -h : 0.0)), y = m.yc(iv.fine) + (s == 3 ? h : (s == 2 ? -h : 0.0));
            const State want = prim(x, y), got = eng.interpolated_state(static_cast<int>(k));
            const State coarse_value = eos.cons_to_prim(m.U[iv.coarse]);
            for (int q = 0; q < 4; ++q) {
                worst = std::max(worst, std::abs(got[q] - want[q]));
                injected = std::max(injected, std::abs(coarse_value[q] - want[q]));
            }
        }
        std::printf("    interpolation vs injection error on a linear field: %.3e vs %.3e\n", worst, injected);
        // not exact to roundoff: the coarse leaf's neighbour over the refined block is the mean of CONSERVATIVE fine values,
        // which is not the primitive-linear value (difference of order 1e-5 here); the control is a thousand times worse
        CHECK(worst < 1e-4);
        CHECK(injected > 1e-2 && worst < 1e-2 * injected);
        // with the interpolation disabled there is no such cell
        CellFluxEngine off(eos, rs, &mu, bm);
        off.interpolate_jumps = false;
        off.build(m);
        CHECK(off.interpolated().empty());
    }
}

