// EOS, Riemann solvers, boundary conditions, residual, time schemes, IBM, mapping and I/O, behavioural checks
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_common.hpp"

// ---------------------------------------------------------------------------
// EOS / mesh / limiters
// ---------------------------------------------------------------------------

TEST(fluid_roundtrip) {
    for (const EOS& eos : {EOS::ideal(1.4), EOS::stiffened(4.4, 6e8)}) {
        const State W = {998.0, 1.5, -0.7, 1.0e5};
        const State W2 = eos.cons_to_prim(eos.prim_to_cons(W));
        for (int q = 0; q < kNVar; ++q) CHECK_NEAR(W2[q], W[q], 1e-9 * std::max(1.0, std::abs(W[q])));
    }
    CHECK_NEAR(EOS::ideal(1.4).sound_speed(1.0, 1.0), std::sqrt(1.4), 1e-14);
    // Pressure floor
    CHECK(EOS::ideal().cons_to_prim({1.0, 0.0, 0.0, -5.0})[3] == kEosEps);
}

TEST(mesh_topology) {
    const Mesh m = generate_cartesian_mesh(2.0, 1.0, 4, 3);
    CHECK(m.num_cells() == 12);
    CHECK(m.num_faces() == 5 * 3 + 4 * 4);
    CHECK(m.boundary_face(Face::Right, 2) == 2 * 5 + 4);
    CHECK(m.boundary_face(Face::Top, 1) == 15 + 3 * 4 + 1);
    CHECK_NEAR(m.xc(3), 1.75, 1e-15);
}

TEST(limiters) {
    for (Limiter l : {Limiter::Minmod, Limiter::VanLeer, Limiter::MC, Limiter::Koren}) {
        CHECK(apply_limiter(l, 1.0, -2.0) == 0.0);
        CHECK(apply_limiter(l, 0.0, 3.0) == 0.0);
        CHECK_NEAR(apply_limiter(l, 0.7, 0.7), 0.7, 1e-15);
        CHECK_NEAR(apply_limiter(l, -0.7, -0.7), -0.7, 1e-15);
    }
    CHECK_NEAR(lim_minmod(1.0, 2.0), 1.0, 1e-15);
    CHECK_NEAR(lim_vanleer(1.0, 3.0), 1.5, 1e-15);
    CHECK_NEAR(lim_mc(1.0, 3.0), 2.0, 1e-15);
    CHECK_NEAR(lim_koren(1.0, 3.0), 5.0 / 3.0, 1e-15);
}

// ---------------------------------------------------------------------------
// Riemann solvers
// ---------------------------------------------------------------------------

TEST(riemann_consistency) {
    const EOS eos = EOS::ideal(1.4);
    const State U = eos.prim_to_cons({1.2, 0.3, -0.4, 2.0});
    const State W = eos.cons_to_prim(U);
    for (RiemannType t : {RiemannType::HLLC, RiemannType::LowMachHLLC, RiemannType::PreconditionedHLLC}) {
        const RiemannSolver rs(eos, t);
        double s;
        const State Fx = rs.face_flux(U, U, 0, s);
        CHECK_NEAR(Fx[0], W[0] * W[1], 1e-12);
        CHECK_NEAR(Fx[1], W[0] * W[1] * W[1] + W[3], 1e-12);
        CHECK_NEAR(Fx[3], W[1] * (U[3] + W[3]), 1e-12);
        const State Fy = rs.face_flux(U, U, 1, s);
        CHECK_NEAR(Fy[2], W[0] * W[2] * W[2] + W[3], 1e-12);
    }
}

TEST(riemann_supersonic_upwind) {
    const EOS eos = EOS::ideal(1.4);
    const State UL = eos.prim_to_cons({1.0, 5.0, 0.0, 1.0});   // Mach ~4.2 to the right
    const State UR = eos.prim_to_cons({0.5, 4.0, 0.0, 0.8});
    const RiemannSolver rs(eos, RiemannType::HLLC);
    double s;
    const State F = rs.face_flux(UL, UR, 0, s);
    CHECK_NEAR(F[0], 5.0, 1e-12);
    CHECK_NEAR(F[1], 1.0 * 25.0 + 1.0, 1e-12);
    CHECK(s > 5.0);
}

TEST(riemann_sod_flux_symmetry) {
    // Mirrored states must give mirrored fluxes
    const EOS eos = EOS::ideal(1.4);
    const State A = eos.prim_to_cons({1.0, 0.2, 0.0, 1.0}), B = eos.prim_to_cons({0.125, -0.1, 0.0, 0.1});
    const State Am = eos.prim_to_cons({1.0, -0.2, 0.0, 1.0}), Bm = eos.prim_to_cons({0.125, 0.1, 0.0, 0.1});
    const RiemannSolver rs(eos, RiemannType::HLLC);
    double s1, s2;
    const State F1 = rs.face_flux(A, B, 0, s1);
    const State F2 = rs.face_flux(Bm, Am, 0, s2);
    CHECK_NEAR(F1[0], -F2[0], 1e-12);
    CHECK_NEAR(F1[1], F2[1], 1e-12);
    CHECK_NEAR(F1[3], -F2[3], 1e-12);
    CHECK_NEAR(s1, s2, 1e-12);
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------

TEST(bc_ghost_states) {
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 4, 3);
    Field U = uniform_field(eos, m.num_cells(), 1.0, 0.3, 0.2, 1.0);
    BoundaryManager bm(m, eos);
    bm.set_boundary_types(BCType::Wall, BCType::NoSlip, BCType::Periodic, BCType::Periodic);
    Field UL, UR;
    bm.apply_boundary_conditions(U, UL, UR);
    // Slip wall on the left: u reversed, v kept
    const State Wg = eos.cons_to_prim(UL[m.boundary_face(Face::Left, 1)]);
    CHECK_NEAR(Wg[1], -0.3, 1e-12);
    CHECK_NEAR(Wg[2], 0.2, 1e-12);
    // No-slip on the right: both reversed
    const State Wr = eos.cons_to_prim(UR[m.boundary_face(Face::Right, 0)]);
    CHECK_NEAR(Wr[1], -0.3, 1e-12);
    CHECK_NEAR(Wr[2], -0.2, 1e-12);
    // Periodic bottom: ghost = top-row cell
    U[(m.ny - 1) * m.nx + 2][0] = 7.0;
    bm.apply_boundary_conditions(U, UL, UR);
    CHECK(UL[m.boundary_face(Face::Bottom, 2)][0] == 7.0);

    // Padded array: no-slip right negates both momenta, slip-wall left only rho*u
    Field P;
    bm.get_padded_U(U, P);
    const int PW = m.nx + 2;
    CHECK_NEAR(P[2 * PW + m.nx + 1][1], -U[1 * m.nx + m.nx - 1][1], 1e-15);
    CHECK_NEAR(P[2 * PW + m.nx + 1][2], -U[1 * m.nx + m.nx - 1][2], 1e-15);
    CHECK_NEAR(P[2 * PW][1], -U[1 * m.nx][1], 1e-15);
    CHECK_NEAR(P[2 * PW][2], U[1 * m.nx][2], 1e-15);
}

TEST(nonreflecting_outlet) {
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(2.0, 1.0, 8, 4);
    BoundaryManager bm(m, eos);
    bm.set_boundary_types(BCType::NonReflecting, BCType::NonReflecting, BCType::Wall, BCType::Wall);
    BCParams bp;
    bp.p_out = 1.0;
    bp.nrbc_sigma = 0.28;
    bp.length[0] = 2.0;
    bm.set_params(bp);
    const double rho = 1.2, u = 0.3, p = 1.25;
    Field U = uniform_field(eos, m.num_cells(), rho, u, 0.1, p);
    Field UL, UR;
    bm.apply_boundary_conditions(U, UL, UR);
    const double c = eos.sound_speed(p, rho);
    // Each side: only the incoming acoustic characteristic jumps, by -alpha (p - p_out),
    // alpha = sigma (1 + M) h / L with M the outward Mach number
    for (int side = 0; side < 2; ++side) {
        const double s = side == 0 ? -1.0 : 1.0;   // outward normal: -x on the left, +x on the right
        const State Wg =
            eos.cons_to_prim(side == 0 ? UL[m.boundary_face(Face::Left, 1)] : UR[m.boundary_face(Face::Right, 1)]);
        const double dp = Wg[3] - p, du = s * (Wg[1] - u), drho = Wg[0] - rho;
        const double alpha = 0.28 * (1.0 + s * u / c) * m.dx / 2.0;
        CHECK_NEAR(dp - rho * c * du, -alpha * (p - 1.0), 1e-12);   // incoming w1 = p - rho c u_out
        CHECK_NEAR(dp + rho * c * du, 0.0, 1e-12);                  // outgoing acoustic wave unchanged
        CHECK_NEAR(drho - dp / (c * c), 0.0, 1e-12);                // entropy unchanged
        CHECK_NEAR(Wg[2], 0.1, 1e-12);                              // vorticity (tangential velocity) unchanged
    }
    // sigma = 0, and supersonic outflow: the ghost is the cell (transmissive)
    auto same = [](const State& x, const State& y) {
        double d = 0.0;
        for (int q = 0; q < kNVar; ++q) d = std::max(d, std::abs(x[q] - y[q]) / (1.0 + std::abs(y[q])));
        return d < 1e-14;
    };
    bp.nrbc_sigma = 0.0;
    bm.set_params(bp);
    bm.apply_boundary_conditions(U, UL, UR);
    CHECK(same(UR[m.boundary_face(Face::Right, 0)], U[m.nx - 1]));
    bp.nrbc_sigma = 0.28;
    bm.set_params(bp);
    U = uniform_field(eos, m.num_cells(), rho, 2.5 * c, 0.0, p);
    bm.apply_boundary_conditions(U, UL, UR);
    CHECK(same(UR[m.boundary_face(Face::Right, 0)], U[m.nx - 1]));
    CHECK(parse_bc("Non-Reflecting Outlet") == BCType::NonReflecting &&
          bc_name(BCType::NonReflecting) == "nonreflecting");

    // A tube open at both ends with p above p_out: the NRBC relaxes the pressure to p_out,
    // transmissive keeps it (perfectly non-reflecting, so nothing anchors the level)
    for (const char* bc : {"transmissive", "nonreflecting"}) {
        Simulation sim;
        sim.setup(params_for("Sod Shock Tube", {{"nx", "100"},
                                                {"ny", "2"},
                                                {"bc_left", bc},
                                                {"bc_right", bc},
                                                {"p_out", "1.0"},
                                                {"nrbc_sigma", "1.0"},
                                                {"t_end", "30"}}),
                  false);
        sim.U = uniform_field(sim.fluid, sim.mesh.num_cells(), 1.0, 0.0, 0.0, 1.1);
        const ResidualFn res = [&](const Field& Uin, Field& R) { return sim.residual(Uin, R); };
        double t = 0.0;
        while (t < 30.0) t += sim.scheme->step(sim.U, res, sim.dt_length, t, 30.0).dt;
        const double p_mid = sim.fluid.cons_to_prim(sim.U[50])[3];
        std::printf("    %s: p after 30 acoustic times %.6f (p_out 1, start 1.1)\n", bc, p_mid);
        if (std::string(bc) == "transmissive")
            CHECK(std::abs(p_mid - 1.1) < 1e-9);
        else
            CHECK(std::abs(p_mid - 1.0) < 0.01);
    }
}

TEST(airfoil_geometry) {
    // NACA 0012: symmetric, 12 % thick at 30 % chord, closed trailing edge
    const Polygon2D naca = naca4("0012");
    CHECK(naca_digits("NACA 0012") == "0012" && naca_digits("naca2412") == "2412" && naca_digits("Cylinder").empty());
    CHECK(naca.contains(0.3, 0.0) && naca.contains(0.3, 0.055) && !naca.contains(0.3, 0.065) &&
          !naca.contains(-0.01, 0.0));
    double area = 0.0;   // shoelace; NACA 4-digit: 0.6851 t c^2
    for (size_t i = 0, j = naca.pts.size() - 1; i < naca.pts.size(); j = i++)
        area += naca.pts[j][0] * naca.pts[i][1] - naca.pts[i][0] * naca.pts[j][1];
    CHECK_NEAR(std::abs(area) / 2.0, 0.6851 * 0.12, 2e-3);
    // Cambered 2412: more of the section above the chord line
    CHECK(naca4("2412").contains(0.4, 0.07) && !naca.contains(0.4, 0.07));
    // Placement: quarter chord at (2, 1), chord 2, 10 deg nose up -> leading edge above the trailing edge
    const Polygon2D placed = place_airfoil(naca, 2.0, 10.0, 2.0, 1.0);
    CHECK(placed.contains(2.0, 1.0));
    double y_le = 0.0, y_te = 0.0, x_min = 1e9, x_max = -1e9;
    for (const auto& q : placed.pts) {
        if (q[0] < x_min) {
            x_min = q[0];
            y_le = q[1];
        }
        if (q[0] > x_max) {
            x_max = q[0];
            y_te = q[1];
        }
    }
    CHECK(y_le > 1.0 && y_te < 1.0);
    CHECK_NEAR(y_le - 1.0, 0.5 * std::sin(10.0 * 3.14159265358979 / 180.0), 0.02);

    // Coordinate files: Selig (name line, closed loop) and Lednicer (counts, two blocks)
    const fs::path dir = tmp_dir() / "airfoils";
    fs::create_directories(dir);
    {
        std::ofstream sel((dir / "selig.dat").string());
        sel << "NACA 0012 (Selig)\n";
        for (const auto& q : naca.pts) sel << 2.0 * q[0] + 5.0 << " " << 2.0 * q[1] << "\n";   // chord 2, shifted
        sel << 2.0 * naca.pts[0][0] + 5.0 << " " << 2.0 * naca.pts[0][1] << "\n";
        std::ofstream led((dir / "lednicer.dat").string());
        const int n = 200;   // naca4's upper surface: pts[0..n] from TE to LE
        led << "NACA 0012 (Lednicer)\n" << n + 1 << ". " << n << ".\n\n";
        for (int i = n; i >= 0; --i) led << naca.pts[i][0] << " " << naca.pts[i][1] << "\n";
        led << "\n" << naca.pts[n][0] << " " << naca.pts[n][1] << "\n";   // lower: LE ...
        for (size_t i = n + 1; i < naca.pts.size(); ++i) led << naca.pts[i][0] << " " << naca.pts[i][1] << "\n";
    }
    for (const char* f : {"selig.dat", "lednicer.dat"}) {
        const Polygon2D a = load_airfoil((dir / f).string());
        bool same = a.pts.size() == naca.pts.size() || a.pts.size() + 1 == naca.pts.size();
        for (const double x : {0.05, 0.3, 0.7, 0.95})
            for (const double y : {-0.05, -0.02, 0.0, 0.02, 0.05})
                same = same && a.contains(x, y) == naca.contains(x, y);
        CHECK(same);
    }
    bool threw = false;
    {
        std::ofstream bad((dir / "bad.dat").string());
        bad << "name\n0 0\n1 x\n";
    }
    try {
        load_airfoil((dir / "bad.dat").string());
    } catch (const std::runtime_error& e) { threw = std::string(e.what()).find("bad.dat:3") != std::string::npos; }
    CHECK(threw);

    // IBM mask of a placed section: area within 2 % of the polygon's on 100 cells per chord
    const Mesh m = generate_cartesian_mesh(2.0, 1.0, 200, 100);
    const ImmersedBoundary ib(m, place_airfoil(naca, 1.0, 0.0, 0.75, 0.5), 1.0, 1.0);
    CHECK_NEAR(ib.solid_count() * m.dx * m.dy, 0.6851 * 0.12, 0.02 * 0.6851 * 0.12);
}

TEST(reynolds_and_ibm_loads) {
    // reynolds sets mu = rho_in u_in L / Re (cylinder: L = diameter)
    const SimParams sp = params_for("IBM Flow over Cylinder", {{"reynolds", "250"}, {"use_mpi", "false"}});
    CHECK_NEAR(sp.mu, 1.0 * 1.0 * 0.2 / 250.0, 1e-15);
    // Loads: the cylinder in its free stream feels a drag along +x and, symmetric, no lift
    Simulation sim;
    sim.setup(params_for("IBM Flow over Cylinder", {{"nx", "160"}, {"ny", "80"}, {"use_mpi", "false"}}), false);
    sim.initialize_field();
    const ResidualFn res = [&](const Field& Uin, Field& R) { return sim.residual(Uin, R); };
    double t = 0.0;
    for (int k = 0; k < 200; ++k) {
        t += sim.scheme->step(sim.U, res, sim.dt_length, t, 1e9).dt;
        sim.ibm->apply_solid_reset(sim.U, sim.fluid);
    }
    Field R;
    sim.residual(sim.U, R);
    const auto loads = sim.ibm_loads(sim.mesh, *sim.ibm, sim.U, R);
    std::printf("    cylinder after 200 steps: Fx %.4f, Fy %.2e, Mz %.2e\n", loads[0], loads[1], loads[2]);
    CHECK(loads[0] > 0.0);
    CHECK(std::abs(loads[1]) < 1e-3 * loads[0]);
    CHECK(std::abs(loads[2]) < 1e-3 * loads[0]);
}

TEST(bc_parse_labels) {
    CHECK(parse_bc("Inviscid Wall (Slip)") == BCType::Wall);
    CHECK(parse_bc("Mass Flow Inlet") == BCType::MassflowInlet);
    CHECK(parse_bc("no_slip") == BCType::NoSlip);
    CHECK(parse_riemann("HLLC (Low-Mach Thornber)") == RiemannType::LowMachHLLC);
    CHECK(parse_time_scheme("BDF2 Dual Time") == TimeSchemeType::BDF2DualTime);
}

// ---------------------------------------------------------------------------
// Full residual
// ---------------------------------------------------------------------------

TEST(uniform_flow_is_preserved) {
    // A uniform free stream with zero-gradient BCs must give a zero residual
    // for every Riemann solver, with/without MUSCL and viscous fluxes.
    for (const char* rs : {"HLLC", "HLLC (Low-Mach Thornber)", "HLLC (Preconditioned)"})
        for (const char* muscl : {"false", "true"})
            for (const char* visc : {"false", "true"}) {
                Simulation sim;
                sim.setup(params_for("Sod Shock Tube", {{"nx", "12"},
                                                        {"ny", "8"},
                                                        {"riemann_solver", rs},
                                                        {"use_muscl", muscl},
                                                        {"use_viscous", visc},
                                                        {"bc_left", "transmissive"},
                                                        {"bc_right", "transmissive"},
                                                        {"bc_bottom", "transmissive"},
                                                        {"bc_top", "transmissive"}}),
                          false);
                const Field U = uniform_field(sim.fluid, sim.mesh.num_cells(), 1.3, 0.4, -0.2, 2.0);
                Field R;
                const double s = sim.residual(U, R);
                CHECK(max_abs(R) < 1e-12);
                CHECK(s > 0.0);
            }
}

TEST(sod_50_steps_physical) {
    Simulation sim;
    sim.setup(params_for("Sod Shock Tube", {{"nx", "100"}, {"ny", "5"}, {"cfl", "0.4"}}), false);
    sim.U = get_test_case("Sod Shock Tube").initialize(sim.mesh, sim.fluid, sim.p.source);
    const ResidualFn res = [&](const Field& U, Field& R) { return sim.residual(U, R); };
    const double min_delta = std::min(sim.mesh.Lx / sim.mesh.nx, sim.mesh.Ly / sim.mesh.ny);
    double t = 0.0;
    for (int k = 0; k < 50; ++k) t += sim.scheme->step(sim.U, res, min_delta, t, 0.2).dt;
    CHECK(t > 0.0);
    double mass = 0.0, rho_max = 0.0;
    bool finite = true, positive = true;
    for (const State& c : sim.U) {
        const State W = sim.fluid.cons_to_prim(c);
        for (double x : c) finite = finite && std::isfinite(x);
        positive = positive && c[0] > 0 && W[3] > 0;
        rho_max = std::max(rho_max, W[0]);
        mass += c[0] * sim.mesh.cell_area();
    }
    CHECK(finite);
    CHECK(positive);
    CHECK(rho_max < 5.0);
    const double mass0 = (1.0 * 0.5 + 0.125 * 0.5) * 0.1;
    CHECK(std::abs(mass - mass0) / mass0 < 0.01);
}

TEST(muscl_linear_field_exact) {
    // On a linear field every TVD limiter returns the exact face value.
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 8, 6);
    Field U(m.num_cells());
    for (int j = 0; j < m.ny; ++j)
        for (int i = 0; i < m.nx; ++i) U[j * m.nx + i] = eos.prim_to_cons({1.0 + 0.1 * i + 0.05 * j, 0.0, 0.0, 1.0});
    BoundaryManager bm(m, eos);
    bm.set_boundary_types(BCType::Transmissive, BCType::Transmissive, BCType::Transmissive, BCType::Transmissive);
    for (Limiter l : {Limiter::Minmod, Limiter::VanLeer, Limiter::MC, Limiter::Koren}) {
        Field UL, UR;
        bm.apply_boundary_conditions(U, UL, UR);
        MusclReconstruction(eos, l).reconstruct(UL, UR, U, m.nx, m.ny);
        const int f = 2 * (m.nx + 1) + 3;   // x-face between cells i=2 and i=3 on row j=2
        CHECK_NEAR(UL[f][0], 1.0 + 0.1 * 2.5 + 0.05 * 2, 1e-12);
        CHECK_NEAR(UR[f][0], 1.0 + 0.1 * 2.5 + 0.05 * 2, 1e-12);
    }
}

TEST(viscous_uniform_zero_and_couette) {
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 6, 10);
    BoundaryManager bm(m, eos);
    bm.set_boundary_types(BCType::Transmissive, BCType::Transmissive, BCType::NoSlip, BCType::MovingWall);
    bm.set_params({1.0, 0.0, 1.0, 1.0});
    const NavierStokes ns(eos, 0.01, 0.72);
    // Linear Couette profile u = y between a fixed and a u=1 lid is a steady solution
    Field U(m.num_cells());
    for (int j = 0; j < m.ny; ++j)
        for (int i = 0; i < m.nx; ++i) U[j * m.nx + i] = eos.prim_to_cons({1.0, m.yc(j), 0.0, 1.0});
    Field P, gx, gy, R;
    bm.get_padded_U(U, P);
    compute_gradients(P, m.nx, m.ny, m.dx, m.dy, gx, gy);
    const double dt = ns.viscous_residual(U, gx, gy, m, P, R);
    double xmom = 0.0;
    for (const State& r : R) xmom = std::max(xmom, std::abs(r[1]));
    CHECK(xmom < 1e-12);
    CHECK(dt > 0.0);
}

// ---------------------------------------------------------------------------
// Time schemes
// ---------------------------------------------------------------------------

TEST(time_schemes_linear_decay) {
    // dU/dt = -U, single "cell": check orders of accuracy roughly via one step
    const ResidualFn res = [](const Field& U, Field& R) {
        R.resize(U.size());
        for (size_t c = 0; c < U.size(); ++c)
            for (int q = 0; q < kNVar; ++q) R[c][q] = -U[c][q];
        return 1.0;
    };
    const double exact = std::exp(-0.1);
    for (TimeSchemeType tt : {TimeSchemeType::ExplicitEuler, TimeSchemeType::SSPRK2, TimeSchemeType::RungeKutta4}) {
        auto ts = make_time_scheme(tt, 0.1, 0.1, 50, 1e-3);
        Field U(1, State{1, 1, 1, 1});
        const StepResult r = ts->step(U, res, 1.0, 0.0, 10.0);   // dt = cfl*min_delta/S = 0.1
        CHECK_NEAR(r.dt, 0.1, 1e-9);
        const double tol = tt == TimeSchemeType::ExplicitEuler ? 1e-2 : (tt == TimeSchemeType::SSPRK2 ? 1e-3 : 1e-7);
        CHECK_NEAR(U[0][0], exact, tol);
    }
    // Step clamped to t_max
    auto e = make_time_scheme(TimeSchemeType::ExplicitEuler, 0.5, 0.1, 50, 1e-3);
    Field U(1, State{1, 1, 1, 1});
    CHECK_NEAR(e->step(U, res, 1.0, 0.99, 1.0).dt, 0.01, 1e-12);
}

TEST(bdf2_steady_state_converges) {
    // Relaxation dU/dt = 1 - U: BDF2 dual time must approach U = 1
    const ResidualFn res = [](const Field& U, Field& R) {
        R.resize(U.size());
        for (size_t c = 0; c < U.size(); ++c)
            for (int q = 0; q < kNVar; ++q) R[c][q] = 1.0 - U[c][q];
        return 0.0;
    };
    BDF2DualTime bdf(0.5, 0.5, 200, 1e-8);
    Field U(3, State{0, 0, 0, 0});
    double t = 0.0;
    for (int k = 0; k < 40; ++k) t += bdf.step(U, res, 1.0, t, 1e9).dt;
    CHECK_NEAR(U[1][2], 1.0, 1e-6);
    CHECK(bdf.inner_iter_last() >= 1);
}

// ---------------------------------------------------------------------------
// IBM
// ---------------------------------------------------------------------------

TEST(ibm_mask_and_reset) {
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 100, 100);
    const ImmersedBoundary ib(m, "Cylinder", 0.2, 0.5, 0.5, 1.0, 1.0);
    CHECK(std::abs(ib.solid_count() - 3.14159265 * 0.04 * 1e4) < 60);
    Field U = uniform_field(eos, m.num_cells(), 1.1, 0.5, 0.2, 1.3);
    ib.apply_forcing(U, eos);
    CHECK(U[50 * 100 + 50][1] == 0.0);
    CHECK_NEAR(U[5][1], 1.1 * 0.5, 1e-12);
    ib.apply_solid_reset(U, eos);
    CHECK(U[50 * 100 + 50][0] == 1.0);                                                 // deep solid -> rho_ref
    CHECK_NEAR(U[50 * 100 + 50][3], 1.0 / 0.4, 1e-12);                                 // p_ref / (gamma-1)
    CHECK(ImmersedBoundary(m, "cylinder", 0.2, 0.5, 0.5, 1, 1).mask() == ib.mask());   // case-insensitive
    bool threw = false;
    try {
        ImmersedBoundary(m, "Circle", 0.2, 0.5, 0.5, 1, 1);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);   // unknown shape
}

TEST(ibm_ghost_cells_reproduce_a_linear_wall_profile) {
    // A rectangle tilted by 30 degrees (not aligned with the grid): with a velocity that grows linearly with the distance d from
    // its lower wall (u = k d t along the wall, zero at the wall) and a uniform pressure, the no-slip ghost state is the exact
    // linear extension (bilinear interpolation is exact on linear fields): u_ghost = -k d_ghost t. Slip: a normal velocity
    // c d n is reflected (exact as well) and a uniform tangential one kept. Closest points and normals are checked on the way.
    const EOS eos = EOS::ideal(1.4);
    const double pi = std::acos(-1.0), a = pi / 6.0, tx = std::cos(a), ty = std::sin(a), nxw = -ty, nyw = tx;
    // wall through (0.2, 0.3) along t; the body lies below it (outward normal n = (-sin a, cos a))
    const double ox = 0.2, oy = 0.3;
    Polygon2D poly;
    for (const auto& q : std::vector<std::array<double, 2>>{{-1.0, 0.0}, {2.0, 0.0}, {2.0, -1.0}, {-1.0, -1.0}})
        poly.pts.push_back({ox + q[0] * tx + q[1] * nxw, oy + q[0] * ty + q[1] * nyw});
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 64, 64);
    const auto dist = [&](double x, double y) { return (x - ox) * nxw + (y - oy) * nyw; };   // signed, > 0 in the fluid
    for (const bool slip : {false, true}) {
        ImmersedBoundary ib(m, poly, 1.0, 1.0);
        ib.enable_ghost_cells(m, slip);
        CHECK(ib.ghost_cells());
        CHECK(ib.ghost_count() > 0);
        const double k = 0.7, cn = 0.3, ut = 0.4;
        Field U(static_cast<size_t>(m.num_cells()));
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                const double d = dist(m.xc(i), m.yc(j));
                const double u = slip ? ut * tx + cn * d * nxw : k * d * tx, v = slip ? ut * ty + cn * d * nyw : k * d * ty;
                U[m.cell(i, j)] = eos.prim_to_cons({1.2, u, v, 2.0, 0.0});
            }
        ib.apply_forcing(U, eos);
        double worst = 0.0, worst_wall = 0.0;
        int ghosts = 0;
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                const int c = m.cell(i, j);
                if (!ib.mask()[c]) continue;
                const double d = dist(m.xc(i), m.yc(j));   // <= 0 inside
                if (d < -2.9 / 64.0) continue;             // deep solid cells: at rest, not ghosts
                double wx, wy, gnx, gny;
                ib.closest_wall_point(m.xc(i), m.yc(j), wx, wy, gnx, gny);
                worst_wall = std::max({worst_wall, std::abs(dist(wx, wy)), std::abs(gnx - nxw), std::abs(gny - nyw)});
                const State W = eos.cons_to_prim(U[c]);
                // a ghost? (solid with a fluid cell within two cells) -- here all cells with d > -2 h * sqrt(2)
                bool near = false;
                for (int dj = -2; dj <= 2; ++dj)
                    for (int di = -2; di <= 2; ++di) {
                        const int ii = i + di, jj = j + dj;
                        if (ii >= 0 && ii < m.nx && jj >= 0 && jj < m.ny && !ib.mask()[m.cell(ii, jj)]) near = true;
                    }
                if (!near) continue;
                // only cells whose image stencil stays inside the mesh (away from the domain edges)
                if (m.xc(i) < 0.1 || m.xc(i) > 0.9 || m.yc(j) < 0.1 || m.yc(j) > 0.9) continue;
                ++ghosts;
                const double eu = slip ? ut * tx + cn * d * nxw : k * d * tx, ev = slip ? ut * ty + cn * d * nyw : k * d * ty;
                worst = std::max({worst, std::abs(W[1] - eu), std::abs(W[2] - ev), std::abs(W[0] - 1.2), std::abs(W[3] - 2.0)});
            }
        CHECK(ghosts > 20);
        CHECK(worst < 1e-10);        // roundoff: normals from short ghost-to-wall vectors
        CHECK(worst_wall < 1e-10);
        if (worst >= 1e-10 || worst_wall >= 1e-10) std::printf("    slip %d: ghost error %.3e over %d ghosts, wall %.3e\n", slip, worst, ghosts, worst_wall);
    }
    // loads: a uniform pressure gives no net force on a closed body; a body at rest in uniform flow at rest feels none either
    const Mesh mc = generate_cartesian_mesh(2.0, 2.0, 80, 80);
    ImmersedBoundary cyl(mc, "Cylinder", 0.3, 1.0, 1.0, 1.0, 1.0);
    cyl.enable_ghost_cells(mc, false);
    const Field Uc = uniform_field(eos, mc.num_cells(), 1.0, 0.0, 0.0, 1.0);
    const std::array<double, 3> F = cyl.surface_loads(Uc, eos, 0.01, 1.0, 1.0);
    CHECK(std::abs(F[0]) < 1e-12 && std::abs(F[1]) < 1e-12 && std::abs(F[2]) < 1e-12);
    // only geometry bodies in 2-D
    bool threw = false;
    try {
        ImmersedBoundary mk(mc.nx, mc.ny, std::vector<char>(static_cast<size_t>(mc.num_cells()), 0), 1.0, 1.0);
        mk.enable_ghost_cells(mc, false);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// ---------------------------------------------------------------------------
// Mapping / IO
// ---------------------------------------------------------------------------

TEST(solution_mapping) {
    const int nxc = 10, nyc = 5;
    Field Uc(nxc * nyc);
    const auto xc = linspace(0.05, 0.95, nxc), yc = linspace(0.1, 0.9, nyc);
    for (int j = 0; j < nyc; ++j)
        for (int i = 0; i < nxc; ++i) Uc[j * nxc + i] = {1.0 + 2.0 * xc[i] - yc[j], 3.0, 0.0, 1.0};
    const Field Uf = map_solution(Uc, nxc, nyc, 40, 20, 1.0, 1.0);
    const auto xf = linspace(0.0125, 0.9875, 40), yf = linspace(0.025, 0.975, 20);
    double err = 0.0;
    for (int j = 0; j < 20; ++j)
        for (int i = 0; i < 40; ++i) {
            err = std::max(
                err, std::abs(Uf[j * 40 + i][0] - (1.0 + 2.0 * xf[i] - yf[j])));   // linear: exact (extrapolated too)
            err = std::max(err, std::abs(Uf[j * 40 + i][1] - 3.0));
        }
    CHECK(err < 1e-12);
}

TEST(npz_roundtrip) {
    const std::string path = (tmp_dir() / "rt.npz").string();
    Field U = {{1, 2, 3, 4}, {5, 6, 7, 8}, {-1.5, 0, 1e-300, 1e300}};
    NpzArchive a;
    a["U"] = field_to_npy(U);
    a["t"] = NpyArray::from_double(0.125);
    a["step"] = NpyArray::from_int(42);
    a["files"] = NpyArray::from_strings({"sod_00010.vtr", "x"});
    write_npz(path, a);
    const NpzArchive b = read_npz(path);
    const Field V = npy_to_field(b.at("U"));
    CHECK(V == U);
    CHECK(b.at("t").scalar() == 0.125);
    CHECK(b.at("step").scalar() == 42.0);
    CHECK(b.at("files").as_strings() == std::vector<std::string>({"sod_00010.vtr", "x"}));
}

TEST(checkpoint_roundtrip) {
    Simulation a;
    a.setup(params_for("Sod Shock Tube", {{"nx", "20"}, {"ny", "4"}}), false);
    a.U = get_test_case("Sod Shock Tube").initialize(a.mesh, a.fluid, a.p.source);
    a.t = 0.0375;
    a.exporter.collection.push_back({0.0, "sod_shock_tube_00000.vtr"});
    a.save_checkpoint(17);
    Simulation b;
    b.setup(params_for("Sod Shock Tube", {{"nx", "20"}, {"ny", "4"}}), false);
    CHECK(b.load_checkpoint() == 17);
    CHECK(b.t == 0.0375);
    CHECK(b.U == a.U);
    CHECK(b.exporter.collection.size() == 1 && b.exporter.collection[0].filename == "sod_shock_tube_00000.vtr");
    // Hot start 20x4 -> 40x8
    Simulation c;
    c.setup(params_for("Sod Shock Tube", {{"nx", "40"}, {"ny", "8"}}), false);
    const Field Uh = c.load_hot_start(a.checkpoint_path());
    CHECK(static_cast<int>(Uh.size()) == 40 * 8);
}

// ---------------------------------------------------------------------------
// Additional behavioural checks (BCs, solvers, AMR structure, I/O, checkpoints)
// ---------------------------------------------------------------------------

TEST(fluid_stiffened_reduces_to_ideal) {
    const EOS ideal = EOS::ideal(1.4), stiff = EOS::stiffened(1.4, 0.0);
    const State W = {1.3, 0.2, -0.5, 2.5};
    const State a = ideal.prim_to_cons(W), b = stiff.prim_to_cons(W);
    for (int q = 0; q < kNVar; ++q) CHECK_NEAR(a[q], b[q], 1e-14);
    CHECK_NEAR(ideal.sound_speed(2.5, 1.3), stiff.sound_speed(2.5, 1.3), 1e-14);
    CHECK(ideal.cons_to_prim({1e-20, 0, 0, 1.0})[0] == kEosEps);   // density floor
}

TEST(limiter_tvd_bounds) {
    const double vals[] = {-3.0, -1.0, -0.25, 0.1, 0.5, 2.0, 4.0};
    for (double a : vals)
        for (double b : vals)
            for (Limiter l : {Limiter::Minmod, Limiter::VanLeer, Limiter::MC, Limiter::Koren}) {
                const double phi = apply_limiter(l, a, b);
                CHECK(std::abs(phi) <= 2.0 * std::abs(a) + 1e-14);
                CHECK(std::abs(phi) <= 2.0 * std::abs(b) + 1e-14);
                CHECK(phi == 0.0 || sgn(phi) == sgn(a));
                CHECK(std::abs(lim_minmod(a, b)) <= std::abs(phi) + 1e-14);   // minmod is the most diffusive
            }
    CHECK_NEAR(lim_vanleer(0.3, 1.7), lim_vanleer(1.7, 0.3), 1e-15);   // symmetric
}

TEST(riemann_variants) {
    const EOS eos = EOS::ideal(1.4);
    // Supersonic states: Thornber scaling is inactive (z = 1), so LowMach == HLLC
    const State UL = eos.prim_to_cons({1.0, 2.5, 0.3, 1.0}), UR = eos.prim_to_cons({0.8, 2.2, 0.1, 0.7});
    double s1, s2, s3, s4;
    const State F1 = RiemannSolver(eos, RiemannType::HLLC).face_flux(UL, UR, 0, s1);
    const State F2 = RiemannSolver(eos, RiemannType::LowMachHLLC).face_flux(UL, UR, 0, s2);
    for (int q = 0; q < kNVar; ++q) CHECK_NEAR(F1[q], F2[q], 1e-12 * (1 + std::abs(F1[q])));
    // Preconditioning with Ma_cut = 1 recovers the Einfeldt bounds exactly
    const State A = eos.prim_to_cons({1.0, 0.05, 0.0, 1.0}), B = eos.prim_to_cons({0.9, 0.02, 0.0, 0.95});
    const State Fh = RiemannSolver(eos, RiemannType::HLLC).face_flux(A, B, 0, s3);
    const State Fp = RiemannSolver(eos, RiemannType::PreconditionedHLLC, 1.0).face_flux(A, B, 0, s4);
    for (int q = 0; q < kNVar; ++q) CHECK_NEAR(Fh[q], Fp[q], 1e-12 * (1 + std::abs(Fh[q])));
    CHECK_NEAR(s3, s4, 1e-12);
    // At low Mach the preconditioned wave speeds are smaller than the acoustic ones
    double s5;
    RiemannSolver(eos, RiemannType::PreconditionedHLLC, 0.1).face_flux(A, B, 0, s5);
    CHECK(s5 < s3);
    // Fluid at rest: no mass flux; Sod data: finite, positive max speed
    const State rest = eos.prim_to_cons({1.0, 0, 0, 1.0});
    CHECK(std::abs(RiemannSolver(eos, RiemannType::HLLC).face_flux(rest, rest, 1, s1)[0]) < 1e-15);
    const EOS water = EOS::stiffened(4.4, 6e8);
    const State W1 = water.prim_to_cons({1000, 1, 0, 1e5}), W2 = water.prim_to_cons({998, 0, 0, 2e5});
    const State Fw = RiemannSolver(water, RiemannType::HLLC).face_flux(W1, W2, 0, s1);
    for (double x : Fw) CHECK(std::isfinite(x));
    CHECK(s1 > 1000.0);   // liquid sound speed ~ 1.6 km/s
}

TEST(bc_inlets_outlets_moving_wall) {
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 5, 4);
    const Field U = uniform_field(eos, m.num_cells(), 1.2, 0.3, 0.1, 2.0);
    BoundaryManager bm(m, eos);
    bm.set_params({0.8, 0.05, 1.5, 0.7});
    Field UL, UR;
    auto faceavg = [&](Face f, int k, int comp) {
        const int fi = m.boundary_face(f, k);
        return 0.5 * (eos.cons_to_prim(UL[fi])[comp] + eos.cons_to_prim(UR[fi])[comp]);
    };
    // Velocity inlet (left): face-average velocity = prescribed, rho / p from the interior
    bm.set_boundary_types(BCType::VelInlet, BCType::Outlet, BCType::MovingWall, BCType::MassflowInlet);
    bm.apply_boundary_conditions(U, UL, UR);
    CHECK_NEAR(faceavg(Face::Left, 1, 1), 0.8, 1e-12);
    CHECK_NEAR(faceavg(Face::Left, 1, 2), 0.05, 1e-12);
    CHECK_NEAR(eos.cons_to_prim(UL[m.boundary_face(Face::Left, 1)])[3], 2.0, 1e-12);
    // Pressure outlet (right): p fixed, outflow velocity kept
    const State Wo = eos.cons_to_prim(UR[m.boundary_face(Face::Right, 2)]);
    CHECK_NEAR(Wo[3], 0.7, 1e-12);
    CHECK_NEAR(Wo[1], 0.3, 1e-12);
    // Moving wall (bottom): face average = wall velocity (u_in, v_in)
    CHECK_NEAR(faceavg(Face::Bottom, 3, 1), 0.8, 1e-12);
    // Mass-flow inlet (top): face-average velocity = rho_in*u_in / <rho>
    CHECK_NEAR(faceavg(Face::Top, 0, 1), 1.5 * 0.8 / 1.2, 1e-12);
    // Outlet backflow is clamped to zero
    const Field Ub = uniform_field(eos, m.num_cells(), 1.0, -0.4, 0.0, 1.0);
    bm.apply_boundary_conditions(Ub, UL, UR);
    CHECK_NEAR(eos.cons_to_prim(UR[m.boundary_face(Face::Right, 0)])[1], 0.0, 1e-15);
    // Periodic left/right ghosts come from the opposite column
    Field Up = U;
    for (int j = 0; j < m.ny; ++j) Up[j * m.nx + m.nx - 1][0] = 5.0 + j;
    bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Wall, BCType::Wall);
    bm.apply_boundary_conditions(Up, UL, UR);
    CHECK(UL[m.boundary_face(Face::Left, 2)][0] == 7.0);
    Field P;
    bm.get_padded_U(Up, P);
    CHECK(P[3 * (m.nx + 2)][0] == 7.0);
}

TEST(gradients_and_viscous_properties) {
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(2.0, 1.0, 10, 6);
    Field U(m.num_cells());
    for (int j = 0; j < m.ny; ++j)
        for (int i = 0; i < m.nx; ++i) U[j * m.nx + i] = {1.0 + 0.3 * m.xc(i) - 0.2 * m.yc(j), 0.1 * m.yc(j), 0.0, 2.5};
    BoundaryManager bm(m, eos);
    bm.set_boundary_types(BCType::Transmissive, BCType::Transmissive, BCType::Transmissive, BCType::Transmissive);
    Field P, gx, gy;
    bm.get_padded_U(U, P);
    compute_gradients(P, m.nx, m.ny, m.dx, m.dy, gx, gy);
    const int c = 3 * m.nx + 4;   // interior: central differences are exact on linear fields
    CHECK_NEAR(gx[c][0], 0.3, 1e-12);
    CHECK_NEAR(gy[c][0], -0.2, 1e-12);
    CHECK_NEAR(gy[c][1], 0.1, 1e-12);
    // Shear flow: no mass residual, residual grows with viscosity
    Field Rlo, Rhi;
    NavierStokes(eos, 1e-3, 0.72).viscous_residual(U, gx, gy, m, P, Rlo);
    NavierStokes(eos, 1e-1, 0.72).viscous_residual(U, gx, gy, m, P, Rhi);
    double mass = 0.0;
    for (const State& r : Rhi) mass = std::max(mass, std::abs(r[0]));
    CHECK(mass == 0.0);
    CHECK(max_abs(Rhi) > max_abs(Rlo));
    // Stiffened-gas fluid works too
    Field Rs;
    CHECK(NavierStokes(EOS::stiffened(4.4, 1.0), 1e-3, 7.0).viscous_residual(U, gx, gy, m, P, Rs) > 0.0);
}

TEST(body_force_terms) {
    Field U = {{2.0, 0.6, -0.4, 5.0}};
    Field R = {{0, 0, 0, 0}};
    ConstantBodyForce{0.5, -9.81}.add_to(U, R);
    CHECK(R[0][0] == 0.0);
    CHECK_NEAR(R[0][1], 2.0 * 0.5, 1e-15);
    CHECK_NEAR(R[0][2], 2.0 * -9.81, 1e-15);
    CHECK_NEAR(R[0][3], 0.6 * 0.5 + -0.4 * -9.81, 1e-15);   // power of the force: (rho g) . u
}

TEST(ibm_mask_shapes) {
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 20, 20);
    CHECK(ImmersedBoundary(m, "Cylinder", 0.0, 0.5, 0.5, 1, 1).solid_count() == 0);
    CHECK(ImmersedBoundary(m, "Cylinder", 5.0, 0.5, 0.5, 1, 1).solid_count() == 400);
    CHECK(ImmersedBoundary(m, "Sphere", 0.3, 0.4, 0.6, 1, 1).mask() ==
          ImmersedBoundary(m, "Cylinder", 0.3, 0.4, 0.6, 1, 1).mask());
    // Fringe cells take rho / p from their fluid neighbour
    const EOS eos = EOS::ideal(1.4);
    const ImmersedBoundary ib(m, "Cylinder", 0.2, 0.5, 0.5, 9.0, 9.0);
    Field U = uniform_field(eos, m.num_cells(), 1.7, 0.0, 0.0, 1.1);
    ib.apply_solid_reset(U, eos);
    int fringe = 0, deep = 0;
    for (int c = 0; c < m.num_cells(); ++c)
        if (ib.mask()[c]) (std::abs(U[c][0] - 1.7) < 1e-14 ? fringe : deep) += 1;
    CHECK(fringe > 0);
    CHECK(deep > 0);
}

TEST(time_scheme_stages_and_bdf2) {
    int calls = 0;
    double smax_seq[4] = {1.0, 3.0, 2.0, 0.5};
    const ResidualFn res = [&](const Field& U, Field& R) {
        R.assign(U.size(), State{0, 0, 0, 0});
        return smax_seq[calls++ % 4];
    };
    Field U(2, State{1, 1, 1, 1});
    calls = 0;
    const StepResult r4 = RungeKutta4(0.5).step(U, res, 1.0, 0.0, 10.0);
    CHECK(calls == 4);
    CHECK(r4.s_max == 3.0);                          // max over the stages
    CHECK_NEAR(r4.dt, 0.5 / (1.0 + 1e-10), 1e-12);   // dt from the first stage
    calls = 0;
    SSPRK2(0.5).step(U, res, 1.0, 0.0, 10.0);
    CHECK(calls == 2);
    // BDF2: invalid dt rejected, zero RHS converges in one iteration, dt == dt_phys
    bool threw = false;
    try {
        BDF2DualTime bad(0.5, 0.0);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    BDF2DualTime bdf(0.5, 0.25, 7, 1e-3);
    calls = 0;
    const StepResult rb = bdf.step(U, res, 1.0, 0.0, 10.0);
    CHECK(bdf.inner_iter_last() == 1);
    CHECK_NEAR(rb.dt, 0.25, 1e-15);
    CHECK_NEAR(bdf.step(U, res, 1.0, 9.9, 10.0).dt, 0.1, 1e-12);   // clamped near the end
    // Iteration cap honoured on a residual that never converges
    const ResidualFn grow = [](const Field& V, Field& R) {
        R.resize(V.size());
        for (size_t c = 0; c < V.size(); ++c)
            for (int q = 0; q < kNVar; ++q) R[c][q] = 1e3 * (c + 1);
        return 1.0;
    };
    BDF2DualTime capped(0.5, 1e9, 7, 1e-12);
    capped.step(U, grow, 1.0, 0.0, 1e12);
    CHECK(capped.inner_iter_last() == 7);
}

TEST(vtk_export_layout) {
    const fs::path dir = tmp_dir() / "vtk";
    fs::remove_all(dir);
    VTKExporter ex(dir.string(), false);
    CHECK(ex.collection.empty());
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(1.0, 0.5, 4, 2);
    Field U = uniform_field(eos, m.num_cells(), 1.0, 0.1, 0.0, 1.0);
    U[3][0] = std::nan("");   // must be sanitised, not crash
    ex.export_vtr("case", eos, U, m, 0.0, 0);
    ex.export_vtr("case", eos, U, m, 0.5, 7);
    ex.write_pvd("case");
    CHECK(fs::exists(dir / "case_00007.vtr"));
    CHECK(ex.collection.size() == 2 && ex.collection[1].time == 0.5);
    std::ifstream in((dir / "case_00007.vtr").string());
    const std::string vtr((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(vtr.find("<RectilinearGrid WholeExtent=\"0 4 0 2 0 0\">") != std::string::npos);
    CHECK(vtr.find("Name=\"Density\"") != std::string::npos && vtr.find("nan") == std::string::npos);
    // Default binary: inline base64 of a UInt32 byte count + Float32 values (8 cells -> 32 bytes)
    CHECK(vtr.find("format=\"binary\"") != std::string::npos && vtr.find("format=\"ascii\"") == std::string::npos);
    CHECK(vtr.find("NumberOfComponents=\"1\" format=\"binary\">\n          IAAAA") != std::string::npos);
    VTKExporter ascii_ex((dir / "ascii").string(), false, false);
    ascii_ex.export_vtr("case", eos, U, m, 0.0, 0);
    std::ifstream ain((dir / "ascii" / "case_00000.vtr").string());
    const std::string avtr((std::istreambuf_iterator<char>(ain)), std::istreambuf_iterator<char>());
    CHECK(avtr.find("Name=\"Density\" NumberOfComponents=\"1\" format=\"ascii\">\n1.00000e+00\n") != std::string::npos);
    CHECK(avtr.find("format=\"binary\"") == std::string::npos && avtr.find("nan") == std::string::npos);
    std::ifstream pin((dir / "case.pvd").string());
    const std::string pvd((std::istreambuf_iterator<char>(pin)), std::istreambuf_iterator<char>());
    CHECK(pvd.find("timestep=\"5.000000e-01\"") != std::string::npos);
    // adaptive mesh: one .vtu of VTK_QUAD cells (several ranks: a piece per rank and a .pvtu)
    CellMesh cm(1.0, 1.0, 4, 4, 1);
    std::vector<char> refine(static_cast<size_t>(cm.num_leaves()), 0);
    refine[static_cast<size_t>(cm.find(0, 1, 1))] = 1;
    cm.adapt(refine);
    cm.U.assign(static_cast<size_t>(cm.num_leaves()), eos.prim_to_cons({1, 0, 0, 1}));
    std::vector<int> all_leaves(static_cast<size_t>(cm.num_leaves()));
    for (int c = 0; c < cm.num_leaves(); ++c) all_leaves[static_cast<size_t>(c)] = c;
    const fs::path adir = tmp_dir() / "vtk_amr";
    fs::remove_all(adir);
    VTKExporter ea(adir.string(), false, false);
    ea.export_cells("amr", eos, cm, all_leaves, 0.0, 3);
    ea.write_pvd("amr");
    CHECK(fs::exists(adir / "amr_00003.vtu") && ea.collection.size() == 1);
    std::ifstream lin((adir / "amr_00003.vtu").string());
    const std::string leaf((std::istreambuf_iterator<char>(lin)), std::istreambuf_iterator<char>());
    CHECK(leaf.find("NumberOfPoints=\"76\" NumberOfCells=\"19\"") != std::string::npos);
    CHECK(leaf.find("Name=\"AMR_Level\"") != std::string::npos && leaf.find("Name=\"types\"") != std::string::npos);
    std::vector<int> half_a, half_b;
    for (int c = 0; c < cm.num_leaves(); ++c) (c % 2 ? half_b : half_a).push_back(c);
    VTKExporter ep(adir.string(), false, false);
    ep.export_cells("par", eos, cm, half_b, 0.0, 5, 1, 2);   // rank 1 first: only its piece
    CHECK(ep.collection.empty() && fs::exists(adir / "par_00005" / "r0001.vtu"));
    ep.export_cells("par", eos, cm, half_a, 0.0, 5, 0, 2);
    CHECK(ep.collection.size() == 1 && fs::exists(adir / "par_00005.pvtu") && fs::exists(adir / "par_00005" / "r0000.vtu"));
    int tmp_files = 0;
    for (const auto& e : fs::recursive_directory_iterator(adir))
        if (e.path().extension() == ".tmp") ++tmp_files;
    CHECK(tmp_files == 0);
}

TEST(checkpoint_edge_cases) {
    const std::string ck = (tmp_dir() / "ck").string(), ck2 = (tmp_dir() / "ck2").string();
    Simulation a;
    a.setup(params_for("Sod Shock Tube", {{"nx", "8"}, {"ny", "2"}, {"output_root", ck.c_str()}}), false);
    fs::remove(a.checkpoint_path());
    CHECK(a.load_checkpoint() == 0);   // missing checkpoint: start from zero
    a.U = uniform_field(a.fluid, 16, 1, 0, 0, 1);
    a.save_checkpoint(1);
    a.U[0][0] = 3.0;
    a.save_checkpoint(2);   // overwrites atomically
    CHECK(!fs::exists(a.checkpoint_path().substr(0, a.checkpoint_path().size() - 4) + "_tmp.npz"));
    Simulation b;
    b.setup(params_for("Sod Shock Tube", {{"nx", "8"}, {"ny", "2"}, {"output_root", ck.c_str()}}), false);
    CHECK(b.load_checkpoint() == 2 && b.U[0][0] == 3.0);
    // Legacy hot-start file without metadata: needs manual dims
    const std::string legacy = (tmp_dir() / "legacy.npz").string();
    write_npz(legacy, {{"U", field_to_npy(a.U)}});
    bool threw = false;
    try {
        b.load_hot_start(legacy);
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    Simulation c;
    c.setup(params_for("Sod Shock Tube", {{"nx", "16"},
                                          {"ny", "4"},
                                          {"hot_start_src_nx", "8"},
                                          {"hot_start_src_ny", "2"},
                                          {"output_root", ck2.c_str()}}),
            false);
    CHECK(c.load_hot_start(legacy).size() == 64);
}

TEST(solution_mapping_properties) {
    const EOS eos = EOS::ideal(1.4);
    Field U(6 * 4);
    for (int c = 0; c < 24; ++c) U[c] = eos.prim_to_cons({1.0 + 0.1 * (c % 6), 0.2, 0.0, 1.0 + 0.05 * (c / 6)});
    CHECK(map_solution(U, 6, 4, 6, 4, 1.0, 1.0) == U);            // same grid: identity
    const Field coarse = map_solution(U, 6, 4, 3, 2, 1.0, 1.0);   // coarsening also works
    CHECK(coarse.size() == 6);
    const Field fine = map_solution(U, 6, 4, 24, 16, 1.0, 1.0);
    double m0 = 0, m1 = 0;
    bool positive = true;
    for (const State& c : U) m0 += c[0] / 24.0;
    for (const State& c : fine) {
        m1 += c[0] / 384.0;
        positive = positive && c[0] > 0 && eos.cons_to_prim(c)[3] > 0;
    }
    CHECK(positive);
    CHECK(std::abs(m1 - m0) / m0 < 1e-2);
}

// ---------------------------------------------------------------------------
// Environment switches
// ---------------------------------------------------------------------------

namespace {
void set_env(const char* key, const char* value) {   // value == nullptr: unset
#ifdef _WIN32
    _putenv_s(key, value ? value : "");   // an empty value removes the variable on Windows
#else
    if (value) setenv(key, value, 1);
    else unsetenv(key);
#endif
}
}   // namespace

TEST(env_switches_are_a_closed_table) {
    std::set<std::string> seen;
    for (const EnvSwitch& s : env_switches()) {
        CHECK(std::string(s.name).rfind("FENICE_", 0) == 0);
        CHECK(seen.insert(s.name).second);   // no name twice
        CHECK(std::string(s.kind) == "flag" || std::string(s.kind) == "path");
        CHECK(std::string(s.reader) == "solver" || std::string(s.reader) == "python");
        CHECK(std::strlen(s.effect) > 20);
    }
    bool threw = false;
    try {
        switch_on("FENICE_NO_SUCH_SWITCH");
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);   // a name that is not in the table cannot be read
    threw = false;
    try {
        switch_on("FENICE_CASES");   // a path, not a flag
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
    const char* f = "FENICE_DEBUG_TABLES";
    const char* old_f = std::getenv(f);   // (restored at the end: the other tests must see the environment they were started in)
    const std::string saved_f = old_f ? old_f : "";
    const bool had_f = old_f != nullptr;
    const char* old_c = std::getenv("FENICE_CASES");
    const std::string saved_c = old_c ? old_c : "";
    const bool had_c = old_c != nullptr;
    set_env(f, nullptr);
    CHECK(!switch_on(f));
    set_env(f, "0");
    CHECK(!switch_on(f));
    set_env(f, "1");
    CHECK(switch_on(f));
    set_env(f, "yes");
    CHECK(switch_on(f));
    set_env(f, nullptr);
    CHECK(!switch_on(f));
    set_env("FENICE_CASES", "some_folder");
    CHECK(switch_value("FENICE_CASES") == "some_folder");
    set_env("FENICE_CASES", nullptr);
    CHECK(switch_value("FENICE_CASES").empty());
    set_env(f, had_f ? saved_f.c_str() : nullptr);
    set_env("FENICE_CASES", had_c ? saved_c.c_str() : nullptr);
}
