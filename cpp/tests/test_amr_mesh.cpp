// AMR: CellMesh topology, the face-list flux engine, subcycling
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_amr_common.hpp"

// ---------------------------------------------------------------------------
// CellMesh (cell-based AMR topology)
// ---------------------------------------------------------------------------


TEST(cellmesh_level0_matches_uniform_mesh) {
    for (int per = 0; per < 4; ++per) {
        const bool px = per & 1, py = per & 2;
        CellMesh m(3.7, 1.9, 7, 5, 2, px, py, -0.3, 0.45);
        const Mesh u = generate_cartesian_mesh(3.7, 1.9, 7, 5, -0.3, 0.45);
        CHECK(m.num_leaves() == 35);
        for (int c = 0; c < m.num_leaves(); ++c) {
            CHECK(m.dx(0) == u.dx && m.dy(0) == u.dy);
            CHECK(m.xc(c) == u.xc(m.ci[c]) && m.yc(c) == u.yc(m.cj[c]));
            CHECK(m.area(c) == u.cell_area());
        }
        CHECK(m.dx(2) == u.dx / 4.0 && m.dy(1) == u.dy / 2.0);
        const int nxf = px ? 7 : 8, nyf = py ? 5 : 6;
        CHECK(static_cast<int>(m.faces.size()) == nxf * 5 + 7 * nyf);
        check_cell_mesh(m);
    }
}

TEST(cellmesh_small_periodic_domains) {
    // nx = 1 (a leaf is its own neighbour) and nx = 2 (two faces between the same pair of leaves)
    for (const auto& n : std::vector<std::pair<int, int>>{{1, 1}, {1, 3}, {2, 2}, {2, 1}}) {
        CellMesh m(1.0, 1.0, n.first, n.second, 3, true, true);
        check_cell_mesh(m);
        std::mt19937 rng(7u + static_cast<unsigned>(n.first * 10 + n.second));
        for (int round = 0; round < 6; ++round) {
            std::vector<char> r(m.num_leaves()), c(m.num_leaves());
            for (int k = 0; k < m.num_leaves(); ++k) {
                r[k] = rng() % 100 < 25;
                c[k] = rng() % 100 < 50;
            }
            m.adapt(r, c);
            check_cell_mesh(m);
        }
    }
    CellMesh two(1.0, 1.0, 2, 2, 0, true, true);
    CHECK(two.faces.size() == 8);
}

TEST(cellmesh_random_adapt_invariants) {
    const bool flags[4][2] = {{false, false}, {true, false}, {false, true}, {true, true}};
    for (int cfg = 0; cfg < 4; ++cfg) {
        CellMesh m(2.0, 1.5, 6, 5, 3, flags[cfg][0], flags[cfg][1]);
        std::mt19937 rng(1234u + static_cast<unsigned>(cfg));
        random_state(m, rng);
        for (int round = 0; round < 10; ++round) {
            double before[kNVar];
            for (int q = 0; q < kNVar; ++q) before[q] = cell_mesh_total(m, q);
            std::vector<char> r(m.num_leaves()), c(m.num_leaves());
            for (int k = 0; k < m.num_leaves(); ++k) {
                r[k] = rng() % 100 < 12;
                c[k] = rng() % 100 < 60;
            }
            m.adapt(r, c);
            check_cell_mesh(m);
            CHECK(static_cast<int>(m.U.size()) == m.num_leaves());
            for (int q = 0; q < kNVar; ++q) CHECK_NEAR(cell_mesh_total(m, q), before[q], 1e-12 * before[q]);
        }
    }
}

TEST(cellmesh_refine_then_coarsen_restores_state) {
    CellMesh m(1.0, 1.0, 5, 4, 2);
    std::mt19937 rng(99);
    random_state(m, rng);
    const Field U0 = m.U;
    const std::vector<int> ci0 = m.ci, cj0 = m.cj;
    std::vector<char> r(m.num_leaves());
    for (char& v : r) v = rng() % 100 < 40;
    m.adapt(r);
    CHECK(m.num_leaves() > 20);
    std::vector<char> c(m.num_leaves());
    for (int k = 0; k < m.num_leaves(); ++k) c[k] = m.level[k] == 1;
    m.adapt({}, c);
    CHECK(m.num_leaves() == 20 && m.ci == ci0 && m.cj == cj0);
    CHECK(m.U == U0);   // bitwise: injection, then the mean of four equal values
    check_cell_mesh(m);
}

TEST(cellmesh_ripple_and_merge_refusal) {
    CellMesh m(4.0, 4.0, 4, 4, 3);
    const auto flag_cell = [](const CellMesh& mesh, int l, int i, int j) {
        std::vector<char> r(mesh.num_leaves(), 0);
        const int c = mesh.find(l, i, j);
        CHECK(c >= 0);
        if (c >= 0) r[c] = 1;
        return r;
    };
    m.adapt(flag_cell(m, 0, 1, 1));
    CHECK(m.num_leaves() == 19 && m.find(1, 2, 2) >= 0);
    // refine the child touching the corner of (0;2,2): its level-0 neighbours must follow (ripple)
    m.adapt(flag_cell(m, 1, 3, 3));
    CHECK(m.find(0, 2, 2) < 0 && m.find(0, 2, 1) < 0 && m.find(0, 1, 2) < 0);
    CHECK(m.find(2, 6, 6) >= 0 && m.find(1, 4, 4) >= 0 && m.find(0, 0, 0) >= 0 && m.find(0, 3, 3) >= 0);
    CHECK(m.num_leaves() == 31);
    check_cell_mesh(m);
    {
        // the group of (0;2,2) cannot merge back: its corner neighbour block (1,1) is two levels deeper there
        std::vector<char> c(m.num_leaves(), 0);
        for (int k = 0; k < m.num_leaves(); ++k) c[k] = m.level[k] == 1 && m.ci[k] / 2 == 2 && m.cj[k] / 2 == 2;
        m.adapt({}, c);
        CHECK(m.num_leaves() == 31);
    }
    {
        const CellNeighbour nb = m.neighbour(m.find(0, 0, 1), Face::Right);
        CHECK(nb.kind == CellNeighbour::Kind::Finer && m.cj[nb.a] < m.cj[nb.b]);
        const CellNeighbour back = m.neighbour(nb.a, Face::Left);
        CHECK(back.kind == CellNeighbour::Kind::Coarser && back.a == m.find(0, 0, 1));
    }

    // a merge refused because a neighbour is refined in the same call (and the ripple it causes)
    CellMesh q(4.0, 4.0, 4, 4, 3);
    std::vector<char> r(q.num_leaves(), 0);
    r[q.find(0, 1, 1)] = 1;
    r[q.find(0, 2, 1)] = 1;
    q.adapt(r);
    CHECK(q.num_leaves() == 22);
    std::mt19937 rng(5);
    random_state(q, rng);
    std::vector<char> co(q.num_leaves(), 0);
    for (int k = 0; k < q.num_leaves(); ++k) co[k] = q.level[k] == 1 && q.ci[k] / 2 == 1 && q.cj[k] / 2 == 1;
    q.adapt(flag_cell(q, 1, 4, 3), co);   // right of the block (1,1)
    CHECK(q.find(1, 3, 3) >= 0 && q.find(2, 8, 6) >= 0);   // block (1,1) kept, (1;4,3) split
    CHECK(q.find(0, 1, 2) < 0 && q.find(0, 2, 2) < 0);      // ripple to the level-0 corner neighbours
    CHECK(q.num_leaves() == 31);
    check_cell_mesh(q);
    // asked again: still refused, the level-2 children are next to the block
    co.assign(q.num_leaves(), 0);
    for (int k = 0; k < q.num_leaves(); ++k) co[k] = q.level[k] == 1 && q.ci[k] / 2 == 1 && q.cj[k] / 2 == 1;
    q.adapt({}, co);
    CHECK(q.find(1, 3, 3) >= 0);
    // coarsen the level-2 children first, then the block merges
    co.assign(q.num_leaves(), 0);
    for (int k = 0; k < q.num_leaves(); ++k) co[k] = q.level[k] == 2;
    q.adapt({}, co);
    CHECK(q.find(1, 4, 3) >= 0);
    co.assign(q.num_leaves(), 0);
    for (int k = 0; k < q.num_leaves(); ++k) co[k] = q.level[k] == 1 && q.ci[k] / 2 == 1 && q.cj[k] / 2 == 1;
    q.adapt({}, co);
    CHECK(q.find(0, 1, 1) >= 0);
    check_cell_mesh(q);
}

// ---------------------------------------------------------------------------
// CellFluxEngine (convective residual on the face list)
// ---------------------------------------------------------------------------


TEST(cellflux_level0_matches_uniform_residual_bitwise) {
    struct BCs {
        const char *l, *r, *b, *t;
    };
    const BCs sets[] = {{"wall", "wall", "wall", "wall"},
                        {"transmissive", "outlet", "no_slip", "moving_wall"},
                        {"vel_inlet", "outlet", "wall", "transmissive"},
                        {"massflow_inlet", "transmissive", "wall", "wall"},
                        {"periodic", "periodic", "wall", "wall"},
                        {"wall", "transmissive", "periodic", "periodic"},
                        {"periodic", "periodic", "periodic", "periodic"}};
    struct Scheme {
        const char *muscl, *limiter, *riemann;
    };
    const Scheme schemes[] = {{"false", "Van Leer", "HLLC"},    {"true", "Van Leer", "HLLC"},
                              {"true", "Minmod", "HLLC"},       {"true", "MC", "lowmach"},
                              {"true", "Koren", "preconditioned"}};
    int n_cases = 0;
    for (const BCs& bc : sets)
        for (const Scheme& sc : schemes) {
            Simulation sim;
            sim.setup(cell_flux_params({{"bc_left", bc.l},
                                        {"bc_right", bc.r},
                                        {"bc_bottom", bc.b},
                                        {"bc_top", bc.t},
                                        {"use_muscl", sc.muscl},
                                        {"muscl_limiter", sc.limiter},
                                        {"riemann_solver", sc.riemann}}),
                      false);
            const Mesh& u = sim.mesh;
            const bool px = sim.boundaries.type(Face::Left) == BCType::Periodic;
            const bool py = sim.boundaries.type(Face::Bottom) == BCType::Periodic;
            CellMesh cm(u.Lx, u.Ly, u.nx, u.ny, 0, px, py, u.x0, u.y0);
            Field Uu(static_cast<size_t>(u.num_cells()));
            for (int j = 0; j < u.ny; ++j)
                for (int i = 0; i < u.nx; ++i) Uu[j * u.nx + i] = test_state(sim.fluid, u.xc(i), u.yc(j));
            cm.U.resize(cm.num_leaves());
            for (int c = 0; c < cm.num_leaves(); ++c) cm.U[c] = Uu[cm.cj[c] * u.nx + cm.ci[c]];

            Field Ru;
            const double Su = sim.residual(Uu, Ru);
            CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
            eng.build(cm);
            Field Rc;
            const double Sc = eng.residual(cm, cm.U, Rc);
            Field Rperm(Ru.size());
            for (int c = 0; c < cm.num_leaves(); ++c) Rperm[c] = Ru[cm.cj[c] * u.nx + cm.ci[c]];
            const bool same = bits_equal(Rperm, Rc) && Su == Sc;
            CHECK(same);
            if (!same) std::printf("    case bc %s/%s/%s/%s  muscl %s %s %s\n", bc.l, bc.r, bc.b, bc.t, sc.muscl, sc.limiter, sc.riemann);
            ++n_cases;
        }
    CHECK(n_cases == 35);

    // above kOmpMinWork the loops run in parallel: same bits
    for (const BCs& bc : {BCs{"wall", "outlet", "no_slip", "moving_wall"}, BCs{"periodic", "periodic", "wall", "wall"}}) {
        Simulation sim;
        sim.setup(cell_flux_params({{"nx", "96"}, {"ny", "64"}, {"bc_left", bc.l}, {"bc_right", bc.r}, {"bc_bottom", bc.b},
                                    {"bc_top", bc.t}, {"use_muscl", "true"}, {"muscl_limiter", "MC"}}),
                  false);
        const Mesh& u = sim.mesh;
        CellMesh cm(u.Lx, u.Ly, u.nx, u.ny, 0, sim.boundaries.type(Face::Left) == BCType::Periodic, false, u.x0, u.y0);
        CHECK(cm.num_leaves() >= kOmpMinWork);
        Field Uu(static_cast<size_t>(u.num_cells()));
        for (int j = 0; j < u.ny; ++j)
            for (int i = 0; i < u.nx; ++i) Uu[j * u.nx + i] = test_state(sim.fluid, u.xc(i), u.yc(j));
        cm.U.resize(cm.num_leaves());
        for (int c = 0; c < cm.num_leaves(); ++c) cm.U[c] = Uu[cm.cj[c] * u.nx + cm.ci[c]];
        Field Ru, Rc, Rperm(Uu.size());
        const double Su = sim.residual(Uu, Ru);
        CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        eng.build(cm);
        const double Sc = eng.residual(cm, cm.U, Rc);
        for (int c = 0; c < cm.num_leaves(); ++c) Rperm[c] = Ru[cm.cj[c] * u.nx + cm.ci[c]];
        CHECK(bits_equal(Rperm, Rc) && Su == Sc);
    }
}


TEST(cellflux_free_stream_is_exact_across_level_jumps) {
    const EOS eos = EOS::ideal(1.4);
    for (int per = 0; per < 4; ++per) {
        const bool px = per & 1, py = per & 2;
        for (const char* limiter : {"Van Leer", "Koren"}) {
            CellMesh m = random_adapted_mesh(px, py, 17u + static_cast<unsigned>(per));
            int jumps = 0;
            for (const CellFace& f : m.faces) jumps += f.kind == CellFaceKind::CoarseFine;
            CHECK(jumps > 10);
            BoundaryManager bm(generate_cartesian_mesh(1.5, 1.0, 6, 5), eos);
            bm.set_boundary_types(px ? BCType::Periodic : BCType::Transmissive, px ? BCType::Periodic : BCType::Transmissive,
                                  py ? BCType::Periodic : BCType::Transmissive, py ? BCType::Periodic : BCType::Transmissive);
            RiemannSolver rs(eos, RiemannType::HLLC);
            MusclReconstruction mu(eos, parse_limiter(limiter));
            CellFluxEngine eng(eos, rs, &mu, bm);
            eng.build(m);
            CHECK(eng.num_virtual() > 0);
            m.U.assign(static_cast<size_t>(m.num_leaves()), eos.prim_to_cons({1.3, 0.8, -0.45, 2.1}));
            Field R;
            eng.residual(m, m.U, R);
            double worst = 0.0;
            for (const State& r : R)
                for (int q = 0; q < kNVar; ++q) worst = std::max(worst, std::abs(r[q]));
            CHECK(worst == 0.0);
        }
    }
}

TEST(cellflux_residual_is_conservative_on_adapted_meshes) {
    const EOS eos = EOS::ideal(1.4);
    for (int per = 0; per < 4; ++per) {
        CellMesh m = random_adapted_mesh(true, true, 5u + static_cast<unsigned>(per));
        BoundaryManager bm(generate_cartesian_mesh(1.5, 1.0, 6, 5), eos);
        bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Periodic, BCType::Periodic);
        RiemannSolver rs(eos, per % 2 ? RiemannType::LowMachHLLC : RiemannType::HLLC);
        MusclReconstruction mu(eos, Limiter::MC);
        CellFluxEngine eng(eos, rs, per < 2 ? &mu : nullptr, bm);
        eng.build(m);
        m.U.resize(m.num_leaves());
        for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state(eos, m.xc(c), m.yc(c));
        Field R;
        const double S = eng.residual(m, m.U, R);
        CHECK(S > 0.5);
        for (int q = 0; q < 4; ++q) {
            double sum = 0.0, mag = 0.0;
            for (int c = 0; c < m.num_leaves(); ++c) {
                sum += R[c][q] * m.area(c);
                mag += std::abs(R[c][q]) * m.area(c);
            }
            CHECK(mag > 1e-6);
            CHECK(std::abs(sum) <= 1e-13 * mag);   // each face flux is used by both sides with the same length
        }
    }
}

// ---------------------------------------------------------------------------
// CellAdvance (subcycling with flux replacement)
// ---------------------------------------------------------------------------


TEST(celladvance_level0_steps_match_uniform_steps_bitwise) {
    struct BCs {
        const char *l, *r, *b, *t;
    };
    const BCs sets[] = {{"wall", "outlet", "no_slip", "moving_wall"}, {"periodic", "periodic", "wall", "wall"},
                        {"transmissive", "vel_inlet", "periodic", "periodic"}};
    for (int big = 0; big < 2; ++big)
        for (const BCs& bc : sets)
            for (int sc = 0; sc < 3; ++sc) {
                Simulation sim;
                sim.setup(cell_flux_params({{"nx", big ? "96" : "9"}, {"ny", big ? "64" : "7"}, {"bc_left", bc.l},
                                            {"bc_right", bc.r}, {"bc_bottom", bc.b}, {"bc_top", bc.t},
                                            {"use_muscl", "true"}, {"time_scheme", kSchemeNames[sc]}}),
                          false);
                const Mesh& u = sim.mesh;
                const bool px = sim.boundaries.type(Face::Left) == BCType::Periodic;
                const bool py = sim.boundaries.type(Face::Bottom) == BCType::Periodic;
                for (int max_level : {0, 2}) {   // 2: the recursion with nothing to recurse into
                    CellMesh cm(u.Lx, u.Ly, u.nx, u.ny, max_level, px, py, u.x0, u.y0);
                    Field Uu(static_cast<size_t>(u.num_cells()));
                    for (int j = 0; j < u.ny; ++j)
                        for (int i = 0; i < u.nx; ++i) Uu[j * u.nx + i] = test_state(sim.fluid, u.xc(i), u.yc(j));
                    cm.U.resize(cm.num_leaves());
                    for (int c = 0; c < cm.num_leaves(); ++c) cm.U[c] = Uu[cm.cj[c] * u.nx + cm.ci[c]];
                    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
                    eng.build(cm);
                    std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(kSchemeNames[sc]), 0.4, 1.0, 5, 1e-3);
                    CellAdvance adv(cm, eng, *ts);
                    const double dt = 2e-3;
                    const ResidualFn res = [&](const Field& Uin, Field& R) { return sim.residual(Uin, R); };
                    bool same = true;
                    for (int k = 0; k < 4; ++k) {
                        sim.scheme->advance_fixed_dt(Uu, res, dt);
                        adv.step(dt, k * dt);
                    }
                    for (int c = 0; c < cm.num_leaves(); ++c)
                        for (int q = 0; q < kNVar; ++q) same = same && cm.U[c][q] == Uu[cm.cj[c] * u.nx + cm.ci[c]][q];
                    CHECK(same);
                    if (!same) std::printf("    %s %s/%s/%s/%s big %d max_level %d\n", kSchemeNames[sc], bc.l, bc.r, bc.b, bc.t, big, max_level);
                }
            }
}

TEST(celladvance_conserves_across_level_jumps_and_the_control_fails) {
    for (int sc = 0; sc < 3; ++sc) {
        CellSetup su(true);
        CellMesh m = random_adapted_mesh(true, true, 41u + static_cast<unsigned>(sc), 6);
        CHECK(m.level_begin[3] < m.num_leaves());   // three levels in play
        const auto run = [&](bool replace) {
            CellMesh mm = m;
            mm.U.resize(mm.num_leaves());
            for (int c = 0; c < mm.num_leaves(); ++c) mm.U[c] = test_state(su.eos, mm.xc(c), mm.yc(c));
            CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
            eng.build(mm);
            std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(kSchemeNames[sc]), 0.4, 1.0, 5, 1e-3);
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
            for (int k = 0; k < 6; ++k) S = adv.step(0.4 * mm.dx(0) / S, k * 0.01);
            for (int q = 0; q < 4; ++q) worst = std::max(worst, std::abs(cell_total(mm, q) - before[q]) / mag[q]);
            return worst;
        };
        const double with = run(true), without = run(false);
        CHECK(with < 1e-13);
        CHECK(without > 1e-8);   // negative control: the register is what conserves
        if (!(with < 1e-13) || !(without > 1e-8))
            std::printf("    %s: conservation error with replacement %.3e, without %.3e\n", kSchemeNames[sc], with, without);
    }
}

TEST(celladvance_free_stream_and_stage_times) {
    for (int sc = 0; sc < 3; ++sc) {
        CellSetup su(true);
        CellMesh m = random_adapted_mesh(true, true, 77u, 6);
        CHECK(m.level_begin[3] < m.num_leaves());
        m.U.assign(static_cast<size_t>(m.num_leaves()), su.eos.prim_to_cons({1.3, 0.8, -0.45, 2.1}));
        const Field U0 = m.U;
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        eng.build(m);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(kSchemeNames[sc]), 0.4, 1.0, 5, 1e-3);
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
            CHECK(drift < 1e-14);   // RK4's stage weights sum to 1 only up to rounding
        // expected order of the stage evaluations: finer level first, two substeps, stage times as TimeScheme::step()
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

TEST(celladvance_sod_on_a_static_refined_band) {
    // Bands fixed before the first run: the adaptive run (base 50, two levels, equivalent to 200) is within
    // 1.6 x of the uniform 200 error against the exact density, and below half the uniform 50 error.
    CellSetup su(false);
    const auto run = [&](int nx, int max_level, bool band, double xlo = 0.2, double xhi = 0.92,
                         double* win_err = nullptr, double* rho_max_out = nullptr) {
        CellMesh m(1.0, 0.08, nx, nx / 12 > 0 ? std::max(4, nx / 12) : 4, max_level);   // square cells
        if (band)
            for (int pass = 0; pass < max_level; ++pass) {
                std::vector<char> r(m.num_leaves(), 0);
                for (int c = 0; c < m.num_leaves(); ++c) r[c] = m.xc(c) > xlo && m.xc(c) < xhi;
                m.adapt(r);
            }
        m.U.resize(m.num_leaves());
        for (int c = 0; c < m.num_leaves(); ++c)
            m.U[c] = m.xc(c) < 0.5 ? su.eos.prim_to_cons({1.0, 0.0, 0.0, 1.0}) : su.eos.prim_to_cons({0.125, 0.0, 0.0, 0.1});
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        eng.build(m);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4, 1.0, 5, 1e-3);
        CellAdvance adv(m, eng, *ts);
        const double mass0 = cell_total(m, 0);
        Field R;
        double S = eng.residual(m, m.U, R), t = 0.0;
        while (t < 0.2) {
            const double dt = std::min(0.4 * m.dx(0) / S, 0.2 - t);
            S = adv.step(dt, t);
            t += dt;
        }
        CHECK_NEAR(cell_total(m, 0), mass0, 1e-12 * mass0);
        double err = 0.0, area = 0.0, rho_min = 1e9, rho_max = 0.0, werr = 0.0, warea = 0.0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            rho_min = std::min(rho_min, m.U[c][0]);
            rho_max = std::max(rho_max, m.U[c][0]);
            if (m.xc(c) > 0.55 && m.xc(c) < 0.75) {
                werr += std::abs(m.U[c][0] - sod_exact_density(m.xc(c), 0.2)) * m.area(c);
                warea += m.area(c);
            }
            err += std::abs(m.U[c][0] - sod_exact_density(m.xc(c), 0.2)) * m.area(c);
            area += m.area(c);
        }
        CHECK(rho_min > 0.1 && std::isfinite(err));
        if (win_err) *win_err = werr / warea;
        if (rho_max_out) {
            *rho_max_out = rho_max;
            *win_err = werr / warea;
            CHECK(rho_min > 0.125 - 0.01 && rho_max < 1.0 + 0.01);   // TVD: no overshoot born at a jump
        }
        return std::make_pair(err / area, m.num_leaves());
    };
    const auto coarse = run(50, 0, false), fine = run(200, 0, false), amr = run(50, 2, true);
    std::printf("    Sod L1(rho): uniform 50 %.4e (%d cells), uniform 200 %.4e (%d), AMR %.4e (%d)\n", coarse.first,
                coarse.second, fine.first, fine.second, amr.first, amr.second);
    CHECK(amr.first < 1.6 * fine.first);
    CHECK(amr.first < 0.5 * coarse.first);
    CHECK(amr.second < fine.second);

    // Narrow bands, so that waves cross the coarse-fine jumps (mass to 1e-12 is checked in run; density in
    // [0.115, 1.01], i.e. no overshoot born at a jump; global error below the uniform-50 error).
    //  (a) band [0.55, 0.75], fixed first: the shock enters it at t ~ 0.03 and leaves at ~ 0.14, the contact
    //      enters at ~ 0.05 after being smeared on the coarse grid, the rarefaction stays coarse. Window error
    //      in [0.55, 0.75] required below the uniform-50 one (the contact smear made on the coarse grid stays).
    //  (b) band [0.4, 0.8], the contact is fine from t = 0 and the rarefaction head (t ~ 0.085) and the shock
    //      (t ~ 0.17) cross the jumps moving out: window error within 1.6 x of the uniform-200 one there.
    double w_a = 0.0, w_b = 0.0, w_fine = 0.0, w_coarse = 0.0, rmax = 0.0, unused = 0.0;
    const auto a = run(50, 2, true, 0.55, 0.75, &w_a, &rmax);
    const auto b = run(50, 2, true, 0.4, 0.8, &w_b, &unused);
    run(200, 0, false, 0.2, 0.92, &w_fine, &unused);
    run(50, 0, false, 0.2, 0.92, &w_coarse, &unused);
    std::printf("    Sod, waves crossing the jumps: L1 (a) %.4e (%d cells), (b) %.4e (%d cells); error in [0.55, 0.75]: (a) %.4e, (b) %.4e, "
                "uniform 200 %.4e, uniform 50 %.4e\n",
                a.first, a.second, b.first, b.second, w_a, w_b, w_fine, w_coarse);
    CHECK(a.first < coarse.first && b.first < coarse.first);
    CHECK(w_a < w_coarse);
    CHECK(w_b < 1.6 * w_fine);
}

