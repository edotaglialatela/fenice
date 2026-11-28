// EOS, Riemann solvers, boundary conditions, residual, time schemes, IBM, mapping and I/O, behavioural checks
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstring>
#include <map>
#include "fv/ThreadPlacement.hpp"
#include "test_common.hpp"
#include <chrono>

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
    for (RiemannType t : {RiemannType::HLLC, RiemannType::LowMachHLLC}) {
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
                                                {"bc_xmin", bc},
                                                {"bc_xmax", bc},
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

TEST(ordered_loads_equal_the_serial_loads_whatever_the_cut) {
    // The loads that drive a body are the sum of per-cell terms in ascending global cell order: the bits of ibm_loads (which adds in that order), and
    // the same bits from the terms of strips placed anywhere in the mesh, from global coordinates, handed over in any order.
    Simulation sim;
    sim.setup(params_for("IBM Flow over Cylinder", {{"nx", "160"}, {"ny", "80"}, {"use_mpi", "false"}}), false);
    sim.initialize_field();
    const ResidualFn res = [&](const Field& Uin, Field& R) { return sim.residual(Uin, R); };
    double t = 0.0;
    for (int k = 0; k < 40; ++k) {
        t += sim.scheme->step(sim.U, res, sim.dt_length, t, 1e9).dt;
        sim.ibm->apply_solid_reset(sim.U, sim.fluid);
    }
    Field R;
    sim.residual(sim.U, R);
    const std::array<double, 3> serial = sim.ibm_loads(sim.mesh, *sim.ibm, sim.U, R);
    std::vector<Simulation::LoadTerm> whole;
    sim.ibm_load_terms(sim.mesh, sim.mesh, 0, 0, sim.ibm->mask(), sim.U, R, sim.p.ibm_cx, sim.p.ibm_cy, whole);
    CHECK(whole.size() > 30);
    std::vector<Simulation::LoadTerm> all = whole;
    const std::array<double, 6> sum = Simulation::sum_load_terms(all);
    CHECK(sum[0] == serial[0] && sum[1] == serial[1] && sum[2] == serial[2]);
    // strips of 17, 40 and 23 rows, the way the MPI loop cuts and places them (the local origin rounds differently from the global one)
    std::vector<Simulation::LoadTerm> strips;
    const int cuts[4] = {0, 17, 57, 80};
    for (int s = 2; s >= 0; --s) {   // the last strip first: any order
        const int j0 = cuts[s], nyl = cuts[s + 1] - cuts[s];
        const double dy = sim.mesh.Ly / sim.mesh.ny;
        Mesh lm = generate_cartesian_mesh_3d(sim.mesh.Lx, dy * nyl, sim.mesh.Lz, sim.mesh.nx, nyl, 1, 0.0, j0 * dy);
        lm.dx = sim.mesh.dx;
        lm.dy = sim.mesh.dy;
        lm.dz = sim.mesh.dz;
        const size_t c0 = static_cast<size_t>(j0) * sim.mesh.nx, n = static_cast<size_t>(nyl) * sim.mesh.nx;
        const Field Ul(sim.U.begin() + c0, sim.U.begin() + c0 + n), Rl(R.begin() + c0, R.begin() + c0 + n);
        const std::vector<char> ml(sim.ibm->mask().begin() + c0, sim.ibm->mask().begin() + c0 + n);
        sim.ibm_load_terms(lm, sim.mesh, 0, j0, ml, Ul, Rl, sim.p.ibm_cx, sim.p.ibm_cy, strips);
    }
    CHECK(strips.size() == whole.size());
    const std::array<double, 6> cut = Simulation::sum_load_terms(strips);
    CHECK(cut[0] == serial[0] && cut[1] == serial[1] && cut[2] == serial[2]);
    for (size_t q = 0; q < strips.size(); ++q)   // term by term
        CHECK(strips[q].cell == whole[q].cell && strips[q].fx == whole[q].fx && strips[q].fy == whole[q].fy && strips[q].mz == whole[q].mz);
    // the order matters: the same terms added in descending order are not the same bits (otherwise the test above would prove nothing)
    double rev = 0.0;
    for (size_t q = whole.size(); q-- > 0;) rev += whole[q].fx;
    CHECK(rev != serial[0]);
}

TEST(body_update_is_second_order_on_a_spring) {
    // A body on a spring with no fluid load (fx = fy = 0): x'' = -k x / m, x(0) = 0, x'(0) = 1, m = 1, k = 4: x = sin(2 t) / 2. The update is
    // Adams-Bashforth 2 for the velocity and the trapezoid for the position, started with an Euler step: the error at t = 3 falls by a factor
    // of about four when the step is halved (band: at least 3.2), and is below 2e-3 at 300 steps.
    const auto run = [](int steps) {
        Simulation sim;
        sim.setup(params_for("IBM Flow over Cylinder", {{"nx", "40"}, {"ny", "20"}, {"ibm_mass", "1.0"}, {"ibm_stiff_x", "4.0"}, {"ibm_vx", "1.0"},
                                                        {"use_mpi", "false"}}),
                  false);
        CHECK(sim.body_.u == 1.0 && sim.body_.x == 0.0);
        const double dt = 3.0 / steps;
        for (int k = 0; k < steps; ++k) {
            sim.body_.captured = true;
            sim.advance_body(dt);
        }
        return std::abs(sim.body_.x - 0.5 * std::sin(6.0));
    };
    const double e1 = run(300), e2 = run(600);
    std::printf("    spring, no fluid: error %.3e at 300 steps, %.3e at 600 (ratio %.2f)\n", e1, e2, e1 / e2);
    CHECK(e1 < 2e-3);
    CHECK(e1 / e2 > 3.2);
}

TEST(body_rotation_update_is_second_order_on_a_torsional_spring) {
    // The same law for the angle: a body pinned at its placement on a torsional spring with no fluid torque, I = 1, k = 4, theta(0) = 0, omega(0) = 1:
    // theta = sin(2 t) / 2. The error at t = 3 falls by about four when the step is halved (band: at least 3.2) and is below 2e-3 at 300 steps.
    const auto run = [](int steps) {
        Simulation sim;
        sim.setup(params_for("IBM Flow over Cylinder", {{"nx", "40"}, {"ny", "20"}, {"ibm_inertia", "1.0"}, {"ibm_stiff_theta", "4.0"}, {"ibm_omega", "1.0"},
                                                        {"use_mpi", "false"}}),
                  false);
        CHECK(sim.body_.om == 1.0 && sim.body_.th == 0.0 && sim.body_.u == 0.0);
        const double dt = 3.0 / steps;
        for (int k = 0; k < steps; ++k) {
            sim.body_.captured = true;
            sim.advance_body(dt);
        }
        CHECK(sim.body_.x == 0.0 && sim.body_.y == 0.0);   // a pinned body does not translate
        return std::abs(sim.body_.th - 0.5 * std::sin(6.0));
    };
    const double e1 = run(300), e2 = run(600);
    std::printf("    torsional spring, no fluid: error %.3e at 300 steps, %.3e at 600 (ratio %.2f)\n", e1, e2, e1 / e2);
    CHECK(e1 < 2e-3);
    CHECK(e1 / e2 > 3.2);
}

TEST(body_with_an_off_centre_mass_follows_rigid_body_motion) {
    // No fluid loads. (a) Free fall with the centre of mass off the pivot: it falls with g (x_cm = x0 + u t, y_cm = y0 + v t - g t^2 / 2) and the body
    // turns at the constant rate, so the pivot is the centre of mass minus the turned offset. (b) The two linear degrees of freedom of a sprung
    // section (plunge y and angle, the centre of mass behind the pivot at S = m r_x): second order against the normal-mode solution.
    const double m = 2.0, rx = 0.3, ip = 0.5;
    {
        const auto run = [&](int steps) {
            Simulation sim;
            sim.setup(params_for("IBM Flow over Cylinder", {{"nx", "40"}, {"ny", "20"}, {"ibm_mass", "2.0"}, {"ibm_inertia", "0.5"}, {"ibm_cm_x", "0.3"}, {"ibm_cm_y", "0.0"},
                                                            {"use_bodyforce", "true"}, {"gx", "0"}, {"gy", "-1.5"}, {"ibm_omega", "2.0"}, {"ibm_vx", "0.4"}, {"ibm_vy", "0.2"},
                                                            {"use_mpi", "false"}}),
                      false);
            const double dt = 1.0 / steps;
            for (int k = 0; k < steps; ++k) {
                sim.body_.captured = true;
                sim.advance_body(dt);
            }
            CHECK_NEAR(sim.body_.om, 2.0, 1e-12);
            CHECK_NEAR(sim.body_.th, 2.0, 1e-12);
            // initial velocity of the centre of mass = pivot velocity + omega z x r
            const double ucm = 0.4, vcm = 0.2 + 2.0 * rx;
            const double xp = rx + ucm * 1.0 - rx * std::cos(2.0), yp = vcm * 1.0 - 0.5 * 1.5 - rx * std::sin(2.0);
            return std::hypot(sim.body_.x - xp, sim.body_.y - yp);
        };
        const double e1 = run(400), e2 = run(800);
        std::printf("    free fall, offset centre of mass: pivot error %.3e at 400 steps, %.3e at 800 (ratio %.2f)\n", e1, e2, e1 / e2);
        CHECK(e1 < 2e-3);
        CHECK(e1 / e2 > 3.2);
    }
    {
        // M q'' + K q = 0, q = (y, theta), M = [[m, S], [S, I_P]], S = m r_x, K = diag(ky, kt): released from y = a, theta = 0 at rest (small a).
        const double ky = 3.0, kt = 1.0, S = m * rx, a = 1e-9, T = 3.0;
        const double A = m * ip - S * S, B = -(m * kt + ip * ky), C = ky * kt;
        const double w1 = std::sqrt((-B - std::sqrt(B * B - 4 * A * C)) / (2 * A)), w2 = std::sqrt((-B + std::sqrt(B * B - 4 * A * C)) / (2 * A));
        const auto mode = [&](double w) { return -(ky - w * w * m) / (-w * w * S); };   // theta per unit y in the mode (first row)
        const double r1 = mode(w1), r2 = mode(w2);
        const double c1 = a * r2 / (r2 - r1), c2 = -a * r1 / (r2 - r1);   // y = c1 cos w1 t + c2 cos w2 t, theta = c1 r1 cos + c2 r2 cos
        const auto run = [&](int steps) {
            Simulation sim;
            sim.setup(params_for("IBM Flow over Cylinder", {{"nx", "40"}, {"ny", "20"}, {"ibm_mass", "2.0"}, {"ibm_inertia", "0.5"}, {"ibm_cm_x", "0.3"}, {"ibm_cm_y", "0.0"},
                                                            {"ibm_stiff_y", "3.0"}, {"ibm_stiff_theta", "1.0"}, {"use_mpi", "false"}}),
                      false);
            sim.body_.y = a;
            const double dt = T / steps;
            double worst = 0.0;   // over the whole record: at one instant the errors of the two modes can cancel
            for (int k = 1; k <= steps; ++k) {
                sim.body_.captured = true;
                sim.advance_body(dt);
                const double y = c1 * std::cos(w1 * k * dt) + c2 * std::cos(w2 * k * dt), th = c1 * r1 * std::cos(w1 * k * dt) + c2 * r2 * std::cos(w2 * k * dt);
                worst = std::max(worst, (std::abs(sim.body_.y - y) + std::abs(sim.body_.th - th)) / a);
            }
            return worst;
        };
        const double e1 = run(600), e2 = run(1200);
        std::printf("    two degrees of freedom (w = %.3f, %.3f): relative error %.3e at 600 steps, %.3e at 1200 (ratio %.2f)\n", w1, w2, e1, e2, e1 / e2);
        CHECK(e1 < 5e-3);
        CHECK(e1 / e2 > 3.2);
    }
}

TEST(centre_of_mass_offset_is_in_the_body_frame) {
    // An airfoil is placed turned by ibm_aoa, nose up: an offset along x of the body frame must point from the pivot (the quarter chord) to the trailing
    // edge of the placed polygon.
    Simulation sim;
    sim.setup(params_for("IBM Airfoil Pitch", {{"nx", "120"}, {"ny", "60"}, {"use_amr", "false"}, {"ibm_aoa", "20"}, {"ibm_omega", "0"}, {"ibm_omega_from", "0"},
                                               {"ibm_omega_until", "0"}, {"ibm_mass", "2"}, {"ibm_inertia", "1"}, {"ibm_cm_x", "0.2"}, {"ibm_cm_y", "0"}}),
              false);
    const Polygon2D placed = place_airfoil(naca4("0012"), 1.0, 20.0, 0.0, 0.0);
    double tx = -1e9, ty = 0.0;
    for (const auto& q : placed.pts)
        if (q[0] > tx) {
            tx = q[0];
            ty = q[1];
        }
    CHECK(ty < 0.0);   // nose up: the trailing edge is below the pivot
    CHECK_NEAR(sim.body_cm0_[0], 0.2 * std::cos(20.0 * std::acos(-1.0) / 180.0), 1e-12);
    CHECK_NEAR(sim.body_cm0_[1] / sim.body_cm0_[0], ty / tx, 0.01);
}

TEST(moving_wall_piston_matches_the_exact_shock) {
    // A flat wall (a cylinder of radius 50, its sagitta over the 0.1 high tube is 1e-4) moves at u_p into a gas at rest: a shock runs ahead
    // of it and the gas between them is the exact piston state. Band fixed beforehand: plateau mean within 2 % of the Rankine-Hugoniot
    // values of pressure, density and velocity.
    const double gamma = 1.4, up = 0.5, p1 = 1.0, rho1 = 1.0, x0 = 0.4, t_end = 0.5;
    const double c0 = std::sqrt(gamma * p1 / rho1);
    const double k = 0.5 * up * (gamma + 1.0) / c0;   // M - 1/M = k
    const double M = 0.5 * (k + std::sqrt(k * k + 4.0));
    const double p2 = p1 * (1.0 + 2.0 * gamma / (gamma + 1.0) * (M * M - 1.0));
    const double rho2 = rho1 * (gamma + 1.0) * M * M / ((gamma - 1.0) * M * M + 2.0);
    Simulation sim;
    sim.setup(params_for("Sod Shock Tube", {{"Lx", "2.0"}, {"nx", "400"}, {"use_ibm", "true"}, {"ibm_shape", "Cylinder"},
                                            {"ibm_radius", "50"}, {"ibm_cx", "-49.6"}, {"ibm_cy", "0.05"}, {"ibm_vx", "0.5"},
                                            {"rho_in", "1.0"}, {"p_out", "1.0"}, {"use_viscous", "false"}, {"use_mpi", "false"}}),
              false);
    CHECK(sim.ibm->moving());
    sim.U = uniform_field(sim.fluid, sim.mesh.num_cells(), rho1, 0.0, 0.0, p1);
    const ResidualFn res = [&](const Field& Uin, Field& R) { return sim.residual(Uin, R); };
    double t = 0.0;
    while (t < t_end) {
        ImmersedBoundary::BodyPose pose;
        pose.x = up * t;
        pose.u = up;
        if (sim.ibm->move_to(sim.mesh, pose)) sim.ibm->fill_fresh(sim.U, sim.fluid);
        t += sim.scheme->step(sim.U, res, sim.dt_length, t, t_end).dt;
        sim.ibm->apply_solid_reset(sim.U, sim.fluid);
    }
    const double x_piston = x0 + up * t, x_shock = x0 + M * c0 * t;
    double sp = 0.0, sr = 0.0, su = 0.0;
    int n = 0;
    const int j = sim.mesh.ny / 2;
    for (int i = 0; i < sim.mesh.nx; ++i) {
        const double x = sim.mesh.xc(i);
        if (x < x_piston + 0.1 || x > x_shock - 0.1) continue;
        const State W = sim.fluid.cons_to_prim(sim.U[sim.mesh.cell(i, j, 0)]);
        sr += W[0];
        su += W[1];
        sp += W[3];
        ++n;
    }
    CHECK(n > 20);
    std::printf("    piston plateau (%d cells): p %.4f / %.4f, rho %.4f / %.4f, u %.4f / %.4f\n", n, sp / n, p2, sr / n, rho2, su / n, up);
    CHECK_NEAR(sp / n, p2, 0.02 * p2);
    CHECK_NEAR(sr / n, rho2, 0.02 * rho2);
    CHECK_NEAR(su / n, up, 0.02 * up);
}

TEST(oscillating_cylinder_feels_the_added_mass) {
    // A cylinder of radius R moves as x = A (1 - cos(w t)) in a gas at rest, inviscid, low Mach number (A = 0.2 R: the body crosses two cells at
    // 10 cells per diameter, so the mask rebuilds and fresh cells occur). The exact linear compressible answer (outgoing 2-D waves, kR = 0.169)
    // is F = -K (1.0535 cos(w t) + 0.0467 sin(w t)), K = rho pi R^2 A w^2: the added mass, 5 % above the incompressible one, and a small
    // radiation damping. The fit window t in [0.6, 1.2] ends before the reflections of the open boundaries disturb it (a longer one moved the
    // fine mesh by 10 %). MUSCL reconstruction: with the first-order scheme the dissipation of the scheme added a spurious in-phase force of
    // 0.3 K at 20 cells per diameter (sin component -0.30 against -0.05), which is not the wall's. Bands (set after the runs: 1.9 % / 0.8 % in cos,
    // 0.023 / 0.012 in sin at 20 / 40 cells per diameter): cos within 4 % at 20 cells per diameter; the error of both components falls by at least
    // 0.6 from 20 to 40; the rebuilds cost under 10 % of the run.
    const double Rc = 0.1, A = 0.02, w = 2.0, rho = 1.0;
    const double K = rho * std::acos(-1.0) * Rc * Rc * A * w * w;
    const double a_exact = -1.0535, b_exact = -0.0467;
    const auto run = [&](const char* n, double& alpha, double& beta) {
        Simulation sim;
        sim.setup(params_for("Sod Shock Tube", {{"Lx", "2.4"}, {"Ly", "2.4"}, {"nx", n}, {"ny", n}, {"use_ibm", "true"},
                                                {"ibm_shape", "Cylinder"}, {"ibm_radius", "0.1"}, {"ibm_cx", "1.2"}, {"ibm_cy", "1.2"},
                                                {"ibm_omega", "1e-300"}, {"reconstruction", "muscl"},   // any nonzero motion key switches the wall to the moving one
                                                {"rho_in", "1.0"}, {"p_out", "1.0"}, {"use_viscous", "false"}, {"bc_ymin", "transmissive"},
                                                {"bc_ymax", "transmissive"}, {"use_mpi", "false"}}),
                  false);
        sim.U = uniform_field(sim.fluid, sim.mesh.num_cells(), rho, 0.0, 0.0, 1.0);
        const ResidualFn res = [&](const Field& Uin, Field& R) { return sim.residual(Uin, R); };
        double move_time = 0.0;
        const auto place = [&](double tt) {
            const auto t0 = std::chrono::steady_clock::now();
            ImmersedBoundary::BodyPose pose;
            pose.x = A * (1.0 - std::cos(w * tt));
            pose.u = A * w * std::sin(w * tt);
            if (sim.ibm->move_to(sim.mesh, pose)) sim.ibm->fill_fresh(sim.U, sim.fluid);
            move_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        };
        const auto wall0 = std::chrono::steady_clock::now();
        double t = 0.0, scc = 0.0, sss = 0.0, scs = 0.0, sfc = 0.0, sfs = 0.0;
        Field R;
        while (t < 1.2) {
            place(t);
            t += sim.scheme->step(sim.U, res, sim.dt_length, t, 1.2).dt;
            sim.ibm->apply_solid_reset(sim.U, sim.fluid);
            place(t);
            if (t < 0.6) continue;
            sim.residual(sim.U, R);
            const double fx = sim.ibm_loads(sim.mesh, *sim.ibm, sim.U, R)[0];
            const double c = std::cos(w * t), s = std::sin(w * t);
            scc += c * c;
            sss += s * s;
            scs += c * s;
            sfc += fx * c;
            sfs += fx * s;
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
        const double det = scc * sss - scs * scs;   // least squares F = alpha cos + beta sin
        alpha = (sfc * sss - sfs * scs) / det / K;
        beta = (sfs * scc - sfc * scs) / det / K;
        std::printf("    %s cells: force/K = %.4f cos + %.4f sin (exact %.4f, %.4f); body placement %.1f %% of the run\n", n, alpha, beta,
                    a_exact, b_exact, 100.0 * move_time / wall);
        return move_time / wall;
    };
    double a1, b1, a2, b2;
    const double share = run("240", a1, b1);
    run("480", a2, b2);
    CHECK_NEAR(a1, a_exact, 0.04 * std::abs(a_exact));
    const double e1 = std::abs(a1 - a_exact) + std::abs(b1 - b_exact), e2 = std::abs(a2 - a_exact) + std::abs(b2 - b_exact);
    CHECK(e2 < 0.6 * e1);
    CHECK(share < 0.1);
}

TEST(bc_parse_labels) {
    CHECK(parse_bc("Inviscid Wall (Slip)") == BCType::Wall);
    CHECK(parse_bc("Mass Flow Inlet") == BCType::MassflowInlet);
    CHECK(parse_bc("no_slip") == BCType::NoSlip);
    CHECK(parse_riemann("HLLC (Low-Mach Thornber)") == RiemannType::LowMachHLLC);
}

// ---------------------------------------------------------------------------
// Full residual
// ---------------------------------------------------------------------------

TEST(uniform_flow_is_preserved) {
    // A uniform free stream with zero-gradient BCs must give a zero residual
    // for every Riemann solver, with/without MUSCL and viscous fluxes.
    for (const char* rs : {"HLLC", "HLLC (Low-Mach Thornber)"})
        for (const char* muscl : {"none", "muscl"})
            for (const char* visc : {"false", "true"}) {
                Simulation sim;
                sim.setup(params_for("Sod Shock Tube", {{"nx", "12"},
                                                        {"ny", "8"},
                                                        {"riemann_solver", rs},
                                                        {"reconstruction", muscl},
                                                        {"use_viscous", visc},
                                                        {"bc_xmin", "transmissive"},
                                                        {"bc_xmax", "transmissive"},
                                                        {"bc_ymin", "transmissive"},
                                                        {"bc_ymax", "transmissive"}}),
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

TEST(weno5_faces_near_a_body_take_the_four_cell_stencil) {
    // Uniform WENO5-Z with a solid mask: a face with a solid cell among its six cells has the WENO3-Z states of its four cells, bit
    // for bit; every other interior face keeps the WENO5-Z states of the run without a mask.
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 24, 20);
    Field U(m.num_cells());
    for (int j = 0; j < m.ny; ++j)
        for (int i = 0; i < m.nx; ++i)
            U[m.cell(i, j)] = eos.prim_to_cons({1.0 + 0.3 * std::sin(0.7 * i) * std::cos(0.4 * j), 0.2 * std::cos(0.5 * i), 0.1 * std::sin(0.3 * j),
                                                1.0 + 0.2 * std::cos(0.6 * i + 0.2 * j)});
    std::vector<char> solid(static_cast<size_t>(m.num_cells()), 0);
    for (int j = 8; j <= 11; ++j)
        for (int i = 10; i <= 13; ++i) solid[static_cast<size_t>(m.cell(i, j))] = 1;
    BoundaryManager bm(m, eos);
    bm.set_boundary_types(BCType::Transmissive, BCType::Transmissive, BCType::Transmissive, BCType::Transmissive);
    Field L5, R5, L3, R3, Ln, Rn;
    bm.apply_boundary_conditions(U, L5, R5);
    L3 = Ln = L5;
    R3 = Rn = R5;
    const MusclReconstruction w5(eos, Limiter::VanLeer, Reconstruction::Weno5), w3(eos, Limiter::VanLeer, Reconstruction::Weno3);
    w5.reconstruct(L5, R5, U, m.nx, m.ny);
    w3.reconstruct(L3, R3, U, m.nx, m.ny);
    Halos h;
    h.depth = 3;
    h.narrow = &solid;
    w5.reconstruct(Ln, Rn, U, m.nx, m.ny, h);
    const auto same = [](const State& a, const State& b) { return std::memcmp(a.data(), b.data(), sizeof(State)) == 0; };
    int four = 0, six = 0, bad = 0;
    for (int axis = 0; axis < 2; ++axis)
        for (int j = (axis == 1 ? 3 : 0); j < (axis == 1 ? m.ny - 2 : m.ny); ++j)
            for (int i = (axis == 0 ? 3 : 0); i < (axis == 0 ? m.nx - 2 : m.nx); ++i) {   // faces whose six cells are inside the mesh
                const int f = axis == 0 ? m.xface(i, j) : m.yface(i, j);
                bool near = false;
                for (int s = -3; s <= 2; ++s) near = near || solid[static_cast<size_t>(axis == 0 ? m.cell(i + s, j) : m.cell(i, j + s))] != 0;
                const bool ok = near ? same(Ln[f], L3[f]) && same(Rn[f], R3[f]) : same(Ln[f], L5[f]) && same(Rn[f], R5[f]);
                (near ? four : six) += 1;
                if (!ok) ++bad;
            }
    CHECK(four > 50 && six > 500);
    CHECK(bad == 0);
    if (bad) std::printf("    %d of %d faces differ (%d near the body)\n", bad, four + six, four);
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

// The fused passes of the uniform solver (reconstruction + Riemann solver, with the flux balance in 2-D, and the first-order
// scheme) give, bit for bit, what the arrays of left / right states followed by RiemannSolver::compute_flux and the divergence
// give. Walls and transmissive sides, periodic wrap-around halos, every limiter and Riemann solver, 2-D and 3-D.
TEST(fused_residual_passes_equal_the_two_pass_residual_bit_for_bit) {
    const EOS eos = EOS::ideal(1.4);
    unsigned seed = 12345u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<double>(seed >> 8) / 16777216.0;
    };
    long long compared = 0;
    for (const int nz : {1, 5}) {
        const Mesh m = nz == 1 ? generate_cartesian_mesh(1.0, 0.8, 19, 11) : generate_cartesian_mesh_3d(1.0, 0.8, 0.6, 9, 7, nz);
        Field U(m.num_cells());
        for (auto& u : U) u = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, 0.4 + rnd(), nz > 1 ? rnd() - 0.5 : 0.0});
        BoundaryManager bm(m, eos);
        bm.set_boundary_types(BCType::Wall, BCType::Transmissive, BCType::Transmissive, BCType::Periodic);
        for (const RiemannType rt : {RiemannType::HLLC, RiemannType::LowMachHLLC}) {
            const RiemannSolver rs(eos, rt, 0.2);
            for (const Limiter lim : {Limiter::Minmod, Limiter::VanLeer, Limiter::MC, Limiter::Koren}) {
                const MusclReconstruction mu(eos, lim);
                // periodic wrap-around halos on y (and on z in 3-D), as Simulation::assemble builds them
                Field perB, perT, perF, perK;
                Halos h;
                const int D = kHaloDepth;
                if (nz == 1) {
                    perB.assign(U.end() - static_cast<std::ptrdiff_t>(D) * m.nx, U.end());
                    perT.assign(U.begin(), U.begin() + static_cast<std::ptrdiff_t>(D) * m.nx);
                    h.b = &perB;
                    h.t = &perT;
                }
                Field UL, UR, F2, R2;
                bm.apply_boundary_conditions(U, UL, UR);
                // two passes
                if (nz == 1)
                    mu.reconstruct(UL, UR, U, m.nx, m.ny, h);
                else
                    mu.reconstruct_3d(UL, UR, U, m, h);
                const double s2 = rs.compute_flux(UL, UR, m, F2);
                // fused flux pass, from the first-order states of the sides only
                Field ULs, URs, F1;
                bm.apply_boundary_conditions(U, ULs, URs, nullptr, false);
                const double s1 = mu.muscl_fluxes(rs, ULs, URs, U, m, h, F1);
                CHECK(s1 == s2);
                CHECK(F1.size() == F2.size());
                bool same = F1.size() == F2.size();
                for (size_t f = 0; same && f < F1.size(); ++f)
                    for (int q = 0; q < kNVar; ++q) same = same && F1[f][q] == F2[f][q];
                CHECK(same);
                compared += static_cast<long long>(F1.size());
                if (nz > 1) {
                    // 3-D flux balance in the same pass
                    Field R1;
                    const double s3 = mu.muscl_residual_3d(rs, ULs, URs, U, m, h, R1);
                    CHECK(s3 == s2);
                    const double Ax = m.face_area(0), Ay = m.face_area(1), Az = m.face_area(2), inv_vol = 1.0 / m.cell_volume();
                    bool rsame = R1.size() == U.size();
                    for (int k = 0; rsame && k < m.nz; ++k)
                        for (int j = 0; j < m.ny; ++j)
                            for (int i = 0; i < m.nx; ++i)
                                for (int q = 0; q < kNVar; ++q) {
                                    const State &fl = F2[m.xface(i, j, k)], &fr = F2[m.xface(i + 1, j, k)], &fb = F2[m.yface(i, j, k)],
                                                &ft = F2[m.yface(i, j + 1, k)], &ff = F2[m.zface(i, j, k)], &fk = F2[m.zface(i, j, k + 1)];
                                    const double r = (((0.0 - (fr[q] * Ax - fl[q] * Ax)) - (ft[q] * Ay - fb[q] * Ay)) - (fk[q] * Az - ff[q] * Az)) * inv_vol;
                                    rsame = rsame && R1[m.cell(i, j, k)][q] == r;
                                }
                    CHECK(rsame);
                }
                if (nz == 1) {
                    // flux balance in the same pass
                    Field R1;
                    const double s3 = mu.muscl_residual_2d(rs, ULs, URs, U, m, h, R1);
                    CHECK(s3 == s2);
                    const double inv_area = 1.0 / m.cell_area();
                    const int nxf = m.num_xfaces();
                    bool rsame = R1.size() == U.size();
                    for (int j = 0; rsame && j < m.ny; ++j)
                        for (int i = 0; i < m.nx; ++i)
                            for (int q = 0; q < kNVar; ++q) {
                                const double fl = F2[j * (m.nx + 1) + i][q], fr = F2[j * (m.nx + 1) + i + 1][q];
                                const double fb = F2[nxf + j * m.nx + i][q], ft = F2[nxf + (j + 1) * m.nx + i][q];
                                const double r = ((0.0 - (fr * m.dy - fl * m.dy)) - (ft * m.dx - fb * m.dx)) * inv_area;
                                rsame = rsame && R1[j * m.nx + i][q] == r;
                            }
                    CHECK(rsame);
                }
            }
            if (nz == 1) {
                // first order: states between cells are the cells themselves
                Field UL, UR, F2, ULs, URs, R1;
                bm.apply_boundary_conditions(U, UL, UR);
                const double s2 = rs.compute_flux(UL, UR, m, F2);
                bm.apply_boundary_conditions(U, ULs, URs, nullptr, false);
                const double s1 = rs.first_order_residual_2d(ULs, URs, U, m, R1);
                CHECK(s1 == s2);
                const double inv_area = 1.0 / m.cell_area();
                const int nxf = m.num_xfaces();
                bool rsame = R1.size() == U.size();
                for (int j = 0; rsame && j < m.ny; ++j)
                    for (int i = 0; i < m.nx; ++i)
                        for (int q = 0; q < kNVar; ++q) {
                            const double fl = F2[j * (m.nx + 1) + i][q], fr = F2[j * (m.nx + 1) + i + 1][q];
                            const double fb = F2[nxf + j * m.nx + i][q], ft = F2[nxf + (j + 1) * m.nx + i][q];
                            rsame = rsame && R1[j * m.nx + i][q] == ((0.0 - (fr * m.dy - fl * m.dy)) - (ft * m.dx - fb * m.dx)) * inv_area;
                        }
                CHECK(rsame);
            }
        }
    }
    std::printf("    %lld face fluxes compared\n", compared);
}

// The MUSCL-Hancock residual in one pass over rows gives, bit for bit, what the arrays of face states, the predictor, the Riemann solver
// and the divergence give (the path of FENICE_NO_FUSED = 1): physical sides with and without ghost layers, wrap-around halos, every
// limiter and Riemann solver, ideal and stiffened gas, a step small enough to keep every state and one that does not, and meshes
// whose rows are not a multiple of four, one row high, and large enough for the threads.
TEST(fused_hancock_residual_equals_the_two_pass_residual_bit_for_bit) {
    unsigned seed = 777u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<double>(seed >> 8) / 16777216.0;
    };
    long long compared = 0, stayed_total = 0;
    struct Dim { int nx, ny; };
    for (const EOS eos : {EOS::ideal(1.4), EOS::stiffened(4.4, 100.0)})
        for (const Dim d : {Dim{19, 11}, Dim{6, 1}, Dim{5, 7}, Dim{72, 66}}) {
            const Mesh m = generate_cartesian_mesh(1.0, 0.8, d.nx, d.ny);
            Field U(m.num_cells());
            const double ps = eos.type == EOS::Type::IdealGas ? 1.0 : 200.0;
            for (auto& u : U) u = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, ps * (0.4 + rnd()), 0.0});
            // 0: walls and transmissive sides; 1: periodic in y; 2: periodic in both; 3: ghost layers on every side
            for (int cfg = 0; cfg < 4; ++cfg) {
                BoundaryManager bm(m, eos);
                if (cfg == 0) bm.set_boundary_types(BCType::Wall, BCType::Transmissive, BCType::Transmissive, BCType::Wall);
                if (cfg == 1) bm.set_boundary_types(BCType::Wall, BCType::Transmissive, BCType::Periodic, BCType::Periodic);
                if (cfg == 2) bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Periodic, BCType::Periodic);
                if (cfg == 3) bm.set_boundary_types(BCType::Wall, BCType::Transmissive, BCType::Wall, BCType::Transmissive);
                const int D = kHaloDepth;
                Field perL, perR, perB, perT, ghL, ghR, ghB, ghT;
                Halos h;
                const bool wrap_x = cfg == 2, wrap_y = cfg == 1 || cfg == 2;
                if (wrap_x && m.nx >= D) {
                    perL.resize(static_cast<size_t>(m.ny) * D);
                    perR.resize(static_cast<size_t>(m.ny) * D);
                    for (int j = 0; j < m.ny; ++j)
                        for (int k = 0; k < D; ++k) {
                            perL[j * D + k] = U[j * m.nx + m.nx - D + k];
                            perR[j * D + k] = U[j * m.nx + k];
                        }
                    h.l = &perL;
                    h.r = &perR;
                }
                if (wrap_y && m.ny >= D) {
                    perB.assign(U.end() - static_cast<std::ptrdiff_t>(D) * m.nx, U.end());
                    perT.assign(U.begin(), U.begin() + static_cast<std::ptrdiff_t>(D) * m.nx);
                    h.b = &perB;
                    h.t = &perT;
                }
                if (cfg == 3) {
                    if (bm.fill_ghost_slab(U, Face::Left, D, ghL)) h.l = &ghL;
                    if (bm.fill_ghost_slab(U, Face::Right, D, ghR)) h.r = &ghR;
                    if (bm.fill_ghost_slab(U, Face::Bottom, D, ghB)) h.b = &ghB;
                    if (bm.fill_ghost_slab(U, Face::Top, D, ghT)) h.t = &ghT;
                }
                const bool px = h.l == &perL && h.r == &perR, py = h.b == &perB && h.t == &perT;
                // 0: plain; 1: some cells of a body are not predicted; 2: a cell source moves the states; 3: both
                std::vector<char> body(m.num_cells(), 0);
                Field src(m.num_cells());
                for (int c = 0; c < m.num_cells(); ++c) {
                    body[c] = rnd() < 0.1 ? 1 : 0;
                    for (int q = 0; q < kNVar; ++q) src[c][q] = (rnd() - 0.5) * (q == 3 ? 4.0 : 1.0);
                }
                // bit 4: an MPI neighbour on both sides supplies the predicted states across the interfaces (ghost-layer halos only)
                Field from_lo(m.nx), from_hi(m.nx), sent_lo, sent_hi;
                for (int i = 0; i < m.nx; ++i) {
                    from_lo[i] = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, ps * (0.4 + rnd()), 0.0});
                    from_hi[i] = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, ps * (0.4 + rnd()), 0.0});
                }
                // bit 8: the same along x (strips cut along x: one state per row)
                Field xfrom_lo(m.ny), xfrom_hi(m.ny);
                for (int j = 0; j < m.ny; ++j) {
                    xfrom_lo[j] = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, ps * (0.4 + rnd()), 0.0});
                    xfrom_hi[j] = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, ps * (0.4 + rnd()), 0.0});
                }
                for (int extra = 0; extra < 16; ++extra) {
                if ((extra & 12) == 12) continue;
                if ((extra & 4) && (cfg != 3 || m.ny < 2)) continue;
                if ((extra & 8) && (cfg != 3 || m.nx < 2)) continue;
                const bool xs = (extra & 8) != 0;
                const char axis = xs ? 'x' : 'y';
                h.solid = (extra & 1) ? &body : nullptr;
                const Field* source = (extra & 2) ? &src : nullptr;
                h.exchange_axis = axis;
                h.exchange_faces = (extra & 12) ? FaceExchange([&](const Field& to_lo, const Field& to_hi, Field& lo, Field& hi) {
                                                      sent_lo = to_lo;
                                                      sent_hi = to_hi;
                                                      lo = xs ? xfrom_lo : from_lo;
                                                      hi = xs ? xfrom_hi : from_hi;
                                                  })
                                                : FaceExchange();
                for (const RiemannType rt : {RiemannType::HLLC, RiemannType::LowMachHLLC}) {
                    const RiemannSolver rs(eos, rt, 0.2);
                    for (const Limiter lim : {Limiter::Minmod, Limiter::VanLeer, Limiter::MC, Limiter::Koren})
                        for (const double half : {0.002, 0.06}) {
                            const MusclReconstruction mu(eos, lim);
                            Field ULs, URs;
                            bm.apply_boundary_conditions(U, ULs, URs, nullptr, false);
                            // two passes
                            Field UL = ULs, UR = URs, F2;
                            mu.reconstruct(UL, UR, U, m.nx, m.ny, h);
                            const int stayed2 = mu.hancock_predict(UL, UR, m, half, px, py, h.solid, source, (extra & 12) ? &h.exchange_faces : nullptr,
                                                                   nullptr, axis);
                            const Field two_lo = sent_lo, two_hi = sent_hi;
                            const double s2 = rs.compute_flux(UL, UR, m, F2);
                            // one pass
                            Field R1;
                            long long stayed1 = -1;
                            const double s1 = mu.hancock_residual_2d(rs, ULs, URs, U, m, h, half, px, py, R1, &stayed1, source);
                            CHECK(s1 == s2);
                            if (extra & 12) CHECK(sent_lo == two_lo && sent_hi == two_hi);
                            CHECK(stayed1 == stayed2);
                            stayed_total += stayed2;
                            const double inv_area = 1.0 / m.cell_area();
                            const int nxf = m.num_xfaces();
                            bool same = R1.size() == U.size();
                            for (int j = 0; same && j < m.ny; ++j)
                                for (int i = 0; i < m.nx; ++i)
                                    for (int q = 0; q < kNVar; ++q) {
                                        const double fl = F2[j * (m.nx + 1) + i][q], fr = F2[j * (m.nx + 1) + i + 1][q];
                                        const double fb = F2[nxf + j * m.nx + i][q], ft = F2[nxf + (j + 1) * m.nx + i][q];
                                        same = same && R1[j * m.nx + i][q] == ((0.0 - (fr * m.dy - fl * m.dy)) - (ft * m.dx - fb * m.dx)) * inv_area;
                                    }
                            CHECK(same);
                            compared += static_cast<long long>(U.size());
                        }
                }
                }
            }
        }
    std::printf("    %lld cells compared, %lld states left unpredicted in the large steps\n", compared, stayed_total);
    CHECK(stayed_total > 0);   // the fallback is exercised
}

// The 3-D MUSCL-Hancock residual in one pass over planes gives, bit for bit, what the arrays of face states, the predictor, the wall ghosts,
// the Riemann solver and the divergence give (FENICE_NO_FUSED = 1): reflecting walls, transmissive sides, wrap-around halos on every axis, ghost
// layers on every side, a body with its wall sides, a cell source, the states of a neighbour across a y interface, ideal and stiffened gas, a step
// that keeps every state and one that does not, and meshes whose rows are not a multiple of four, two planes thick, and large enough for threads.
TEST(fused_hancock_3d_residual_equals_the_two_pass_residual_bit_for_bit) {
    unsigned seed = 4711u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<double>(seed >> 8) / 16777216.0;
    };
    long long compared = 0, stayed_total = 0;
    struct Dim { int nx, ny, nz; };
    for (const EOS eos : {EOS::ideal(1.4), EOS::stiffened(4.4, 100.0)})
        for (const Dim d : {Dim{9, 6, 3}, Dim{5, 4, 2}, Dim{4, 3, 5}, Dim{22, 18, 17}}) {
            const Mesh m = generate_cartesian_mesh_3d(1.0, 0.8, 0.6, d.nx, d.ny, d.nz);
            Field U(m.num_cells());
            const double ps = eos.type == EOS::Type::IdealGas ? 1.0 : 200.0;
            for (auto& u : U) u = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, ps * (0.4 + rnd()), rnd() - 0.5});
            // 0: walls and transmissive sides; 1: periodic on every axis; 2: ghost layers on every side (walls on x and z)
            for (int cfg = 0; cfg < 3; ++cfg) {
                BoundaryManager bm(m, eos);
                if (cfg == 0) {
                    bm.set_boundary_types(BCType::Wall, BCType::Transmissive, BCType::Transmissive, BCType::Wall);
                    bm.set_type(Face::Front, BCType::Wall);
                    bm.set_type(Face::Back, BCType::Transmissive);
                }
                if (cfg == 1) {
                    bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Periodic, BCType::Periodic);
                    bm.set_type(Face::Front, BCType::Periodic);
                    bm.set_type(Face::Back, BCType::Periodic);
                }
                if (cfg == 2) {
                    bm.set_boundary_types(BCType::Wall, BCType::Transmissive, BCType::Transmissive, BCType::Transmissive);
                    bm.set_type(Face::Front, BCType::Wall);
                    bm.set_type(Face::Back, BCType::Transmissive);
                }
                const int D = kHaloDepth;
                Field perL, perR, perB, perT, perF, perK, ghs[6];
                Halos h;
                const auto cell = [&](int i, int j, int k) -> const State& { return U[m.cell(i, j, k)]; };
                if (cfg == 1) {
                    if (m.nx >= D) {
                        perL.resize(static_cast<size_t>(m.ny) * m.nz * D);
                        perR.resize(perL.size());
                        for (int k = 0; k < m.nz; ++k)
                            for (int j = 0; j < m.ny; ++j)
                                for (int e = 0; e < D; ++e) {
                                    perL[(static_cast<size_t>(k) * m.ny + j) * D + e] = cell(m.nx - D + e, j, k);
                                    perR[(static_cast<size_t>(k) * m.ny + j) * D + e] = cell(e, j, k);
                                }
                        h.l = &perL;
                        h.r = &perR;
                    }
                    if (m.ny >= D) {
                        perB.resize(static_cast<size_t>(m.nx) * m.nz * D);
                        perT.resize(perB.size());
                        for (int e = 0; e < D; ++e)
                            for (int k = 0; k < m.nz; ++k)
                                for (int i = 0; i < m.nx; ++i) {
                                    perB[(static_cast<size_t>(e) * m.nz + k) * m.nx + i] = cell(i, m.ny - D + e, k);
                                    perT[(static_cast<size_t>(e) * m.nz + k) * m.nx + i] = cell(i, e, k);
                                }
                        h.b = &perB;
                        h.t = &perT;
                    }
                    if (m.nz >= D) {
                        perF.resize(static_cast<size_t>(m.nx) * m.ny * D);
                        perK.resize(perF.size());
                        for (int e = 0; e < D; ++e)
                            for (int j = 0; j < m.ny; ++j)
                                for (int i = 0; i < m.nx; ++i) {
                                    perF[(static_cast<size_t>(e) * m.ny + j) * m.nx + i] = cell(i, j, m.nz - D + e);
                                    perK[(static_cast<size_t>(e) * m.ny + j) * m.nx + i] = cell(i, j, e);
                                }
                        h.f = &perF;
                        h.bk = &perK;
                    }
                }
                if (cfg == 2) {
                    const Field* Halos::*slot[6] = {&Halos::l, &Halos::r, &Halos::b, &Halos::t, &Halos::f, &Halos::bk};
                    for (int s = 0; s < 6; ++s)
                        if (bm.fill_ghost_slab(U, static_cast<Face>(s), D, ghs[s])) h.*slot[s] = &ghs[s];
                }
                const bool per[3] = {cfg == 1 && h.l == &perL && h.r == &perR, cfg == 1 && h.b == &perB && h.t == &perT,
                                     cfg == 1 && h.f == &perF && h.bk == &perK};
                // bit 1: some cells of a body (and the sides that face them) are not predicted; 2: a cell source; 4: a neighbour across y
                std::vector<char> body(m.num_cells(), 0);
                Field src(m.num_cells());
                for (int c = 0; c < m.num_cells(); ++c) {
                    body[c] = rnd() < 0.1 ? 1 : 0;
                    for (int q = 0; q < kNVar; ++q) src[c][q] = (rnd() - 0.5) * (q == 3 ? 4.0 : 1.0);
                }
                const std::vector<unsigned char> sides = solid_side_mask(m, body);
                Field from_lo(static_cast<size_t>(m.nx) * m.nz), from_hi(from_lo.size()), sent_lo, sent_hi;
                for (auto* f : {&from_lo, &from_hi})
                    for (auto& u : *f) u = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, ps * (0.4 + rnd()), rnd() - 0.5});
                h.wall_ghost = [&bm](int side, const State& inner) { return bm.ghost_state(static_cast<Face>(side), inner, 0.0); };
                // bit 8: the same across an x interface (strips cut along x: one state per row of every plane, index k * ny + j)
                Field xfrom_lo(static_cast<size_t>(m.ny) * m.nz), xfrom_hi(xfrom_lo.size());
                for (auto* f : {&xfrom_lo, &xfrom_hi})
                    for (auto& u : *f) u = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, ps * (0.4 + rnd()), rnd() - 0.5});
                for (int extra = 0; extra < 16; ++extra) {
                    if ((extra & 12) == 12) continue;
                    if ((extra & 4) && (cfg != 2 || m.ny < 2)) continue;
                    if ((extra & 8) && (cfg != 2 || m.nx < 2)) continue;
                    const bool xs = (extra & 8) != 0;
                    const char axis = xs ? 'x' : 'y';
                    h.exchange_axis = axis;
                    h.solid = (extra & 1) ? &body : nullptr;
                    h.solid_sides = (extra & 1) ? &sides : nullptr;
                    const Field* source = (extra & 2) ? &src : nullptr;
                    // the reflecting sides of the box; a y interface is not a wall
                    for (int s = 0; s < 6; ++s) h.wall_side[s] = bm.type(static_cast<Face>(s)) == BCType::Wall;
                    if (extra & 4) h.wall_side[2] = h.wall_side[3] = false;
                    if (extra & 8) h.wall_side[0] = h.wall_side[1] = false;
                    h.exchange_faces = (extra & 12) ? FaceExchange([&](const Field& to_lo, const Field& to_hi, Field& lo, Field& hi) {
                                                          sent_lo = to_lo;
                                                          sent_hi = to_hi;
                                                          lo = xs ? xfrom_lo : from_lo;
                                                          hi = xs ? xfrom_hi : from_hi;
                                                      })
                                                    : FaceExchange();
                    for (const RiemannType rt : {RiemannType::HLLC, RiemannType::LowMachHLLC}) {
                        const RiemannSolver rs(eos, rt, 0.2);
                        for (const Limiter lim : {Limiter::Minmod, Limiter::VanLeer, Limiter::MC, Limiter::Koren})
                            for (const double half : {0.002, 0.06}) {
                                const MusclReconstruction mu(eos, lim);
                                Field ULs, URs;
                                bm.apply_boundary_conditions(U, ULs, URs, nullptr, false);
                                // two passes
                                Field UL = ULs, UR = URs, F2;
                                mu.reconstruct_3d(UL, UR, U, m, h);
                                const int stayed2 = mu.hancock_predict_3d(UL, UR, m, half, per, h.solid, source,
                                                                          (extra & 12) ? &h.exchange_faces : nullptr, h.solid_sides, axis);
                                const Field two_lo = sent_lo, two_hi = sent_hi;
                                mu.hancock_wall_ghosts(UL, UR, m, h);
                                const double s2 = rs.compute_flux(UL, UR, m, F2);
                                // one pass
                                Field R1;
                                long long stayed1 = -1;
                                const double s1 = mu.hancock_residual_3d(rs, ULs, URs, U, m, h, half, per, R1, &stayed1, source);
                                CHECK(s1 == s2);
                                if (extra & 12) CHECK(sent_lo == two_lo && sent_hi == two_hi);
                                CHECK(stayed1 == stayed2);
                                stayed_total += stayed2;
                                const double Ax = m.face_area(0), Ay = m.face_area(1), Az = m.face_area(2), inv_vol = 1.0 / m.cell_volume();
                                bool same = R1.size() == U.size();
                                for (int k = 0; same && k < m.nz; ++k)
                                    for (int j = 0; j < m.ny; ++j)
                                        for (int i = 0; i < m.nx; ++i)
                                            for (int q = 0; q < kNVar; ++q) {
                                                const double fl = F2[m.xface(i, j, k)][q], fr = F2[m.xface(i + 1, j, k)][q];
                                                const double fb = F2[m.yface(i, j, k)][q], ft = F2[m.yface(i, j + 1, k)][q];
                                                const double ff = F2[m.zface(i, j, k)][q], fk = F2[m.zface(i, j, k + 1)][q];
                                                same = same && R1[m.cell(i, j, k)][q] ==
                                                                   (((0.0 - (fr * Ax - fl * Ax)) - (ft * Ay - fb * Ay)) - (fk * Az - ff * Az)) * inv_vol;
                                            }
                                CHECK(same);
                                compared += static_cast<long long>(U.size());
                            }
                    }
                }
            }
        }
    std::printf("    %lld cells compared, %lld states left unpredicted in the large steps\n", compared, stayed_total);
    CHECK(stayed_total > 0);
}

TEST(fused_weno_fluxes_equal_the_two_pass_fluxes_bit_for_bit) {
    const EOS eos = EOS::ideal(1.4);
    unsigned seed = 4242u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<double>(seed >> 8) / 16777216.0;
    };
    long long compared = 0;
    for (const int nz : {1, 5}) {
        const Mesh m = nz == 1 ? generate_cartesian_mesh(1.0, 0.8, 21, 13) : generate_cartesian_mesh_3d(1.0, 0.8, 0.6, 11, 9, nz);
        Field U(m.num_cells());
        for (auto& u : U) u = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, 0.4 + rnd(), nz > 1 ? rnd() - 0.5 : 0.0});
        std::vector<char> solid(m.num_cells(), 0);
        for (int c = 0; c < m.num_cells(); ++c) solid[c] = rnd() < 0.04 ? 1 : 0;
        BoundaryManager bm(m, eos);
        bm.set_boundary_types(BCType::Wall, BCType::Transmissive, BCType::Transmissive, BCType::Periodic);
        const RiemannSolver rs(eos, RiemannType::HLLC, 0.2);
        for (const Reconstruction scheme : {Reconstruction::Weno3, Reconstruction::Weno5}) {
            const int pad = scheme == Reconstruction::Weno5 ? 3 : kHaloDepth;
            for (const bool with_solid : {false, true}) {
                if (with_solid && scheme != Reconstruction::Weno5) continue;
                for (const bool with_halo : {false, true}) {
                    const MusclReconstruction mu(eos, Limiter::VanLeer, scheme, 1e-6);
                    // periodic wrap-around halos on y (2-D) or z (3-D), `pad` deep, as Simulation::assemble builds them
                    Field perB, perT;
                    Halos h;
                    h.depth = pad;
                    if (with_solid) h.narrow = &solid;
                    if (with_halo) {
                        const size_t layer = static_cast<size_t>(nz == 1 ? m.nx : m.nx * m.ny);
                        perB.assign(U.end() - static_cast<std::ptrdiff_t>(pad * layer), U.end());
                        perT.assign(U.begin(), U.begin() + static_cast<std::ptrdiff_t>(pad * layer));
                        if (nz == 1) {
                            h.b = &perB;
                            h.t = &perT;
                        } else {
                            h.f = &perB;
                            h.bk = &perT;
                        }
                    }
                    Field UL, UR, F2, ULs, URs, F1;
                    bm.apply_boundary_conditions(U, UL, UR);
                    if (nz == 1)
                        mu.reconstruct(UL, UR, U, m.nx, m.ny, h);
                    else
                        mu.reconstruct_3d(UL, UR, U, m, h);
                    const double s2 = rs.compute_flux(UL, UR, m, F2);
                    bm.apply_boundary_conditions(U, ULs, URs, nullptr, false);
                    const double s1 = mu.muscl_fluxes(rs, ULs, URs, U, m, h, F1);
                    CHECK(s1 == s2);
                    CHECK(F1.size() == F2.size());
                    bool same = F1.size() == F2.size();
                    for (size_t f = 0; same && f < F1.size(); ++f)
                        for (int q = 0; q < kNVar; ++q) same = same && F1[f][q] == F2[f][q];
                    CHECK(same);
                    compared += static_cast<long long>(F1.size());
                    // the flux balance in the same pass
                    Field R1;
                    const double s3 = nz == 1 ? mu.muscl_residual_2d(rs, ULs, URs, U, m, h, R1) : mu.muscl_residual_3d(rs, ULs, URs, U, m, h, R1);
                    CHECK(s3 == s2);
                    const double Ax = m.face_area(0), Ay = m.face_area(1), Az = m.face_area(2), inv_vol = 1.0 / m.cell_volume();
                    bool rsame = R1.size() == U.size();
                    for (int k = 0; rsame && k < m.nz; ++k)
                        for (int j = 0; j < m.ny; ++j)
                            for (int i = 0; i < m.nx; ++i)
                                for (int q = 0; q < kNVar; ++q) {
                                    const State &fl = F2[m.xface(i, j, k)], &fr = F2[m.xface(i + 1, j, k)], &fb = F2[m.yface(i, j, k)],
                                                &ft = F2[m.yface(i, j + 1, k)];
                                    double r;
                                    if (nz == 1) {
                                        r = ((0.0 - (fr[q] * m.dy - fl[q] * m.dy)) - (ft[q] * m.dx - fb[q] * m.dx)) * (1.0 / m.cell_area());
                                    } else {
                                        const State &ff = F2[m.zface(i, j, k)], &fk = F2[m.zface(i, j, k + 1)];
                                        r = (((0.0 - (fr[q] * Ax - fl[q] * Ax)) - (ft[q] * Ay - fb[q] * Ay)) - (fk[q] * Az - ff[q] * Az)) * inv_vol;
                                    }
                                    rsame = rsame && R1[m.cell(i, j, k)][q] == r;
                                }
                    CHECK(rsame);
                }
            }
        }
    }
    std::printf("    %lld WENO face fluxes compared\n", compared);
}

TEST(fourth_order_flux_properties) {
    // flux_quadrature = fourth: the transverse correction vanishes on a field that does not vary across the faces (the flux is then the
    // midpoint flux bit for bit), changes the flux of a field that does, keeps the balance conservative on a periodic box, treats an
    // extruded 3-D field as its 2-D layer, and leaves the faces at a shock on the flux of their averaged states.
    const EOS eos = EOS::ideal(1.4);
    const RiemannSolver rs(eos, RiemannType::HLLC, 0.2);
    const MusclReconstruction mu(eos, Limiter::VanLeer, Reconstruction::Weno5, 1e-6);
    const double pi = 3.14159265358979323846;
    // periodic halos, three deep, of a mesh and its field (the layouts of Halos)
    struct PerHalos {
        Field l, r, b, t, f, bk;
        Halos h;
    };
    const auto make_halos = [](const Mesh& m, const Field& U, PerHalos& ph) {
        const int D = 3, nx = m.nx, ny = m.ny, nz = m.nz;
        ph.l.resize(static_cast<size_t>(ny) * nz * D);
        ph.r.resize(ph.l.size());
        ph.b.resize(static_cast<size_t>(D) * nz * nx);
        ph.t.resize(ph.b.size());
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int d = 0; d < D; ++d) {
                    ph.l[(k * ny + j) * D + d] = U[m.cell(nx - D + d, j, k)];
                    ph.r[(k * ny + j) * D + d] = U[m.cell(d, j, k)];
                }
        for (int d = 0; d < D; ++d)
            for (int k = 0; k < nz; ++k)
                for (int i = 0; i < nx; ++i) {
                    ph.b[(d * nz + k) * nx + i] = U[m.cell(i, m.ny - D + d, k)];
                    ph.t[(d * nz + k) * nx + i] = U[m.cell(i, d, k)];
                }
        ph.h.depth = D;
        ph.h.l = &ph.l;
        ph.h.r = &ph.r;
        ph.h.b = &ph.b;
        ph.h.t = &ph.t;
        if (nz > 1) {
            ph.f.resize(static_cast<size_t>(D) * ny * nx);
            ph.bk.resize(ph.f.size());
            for (int d = 0; d < D; ++d)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        ph.f[(d * ny + j) * nx + i] = U[m.cell(i, j, nz - D + d)];
                        ph.bk[(d * ny + j) * nx + i] = U[m.cell(i, j, d)];
                    }
            ph.h.f = &ph.f;
            ph.h.bk = &ph.bk;
        }
    };
    const bool per[3] = {true, true, true};
    const auto smooth = [&](const Mesh& m, bool vary_y, bool vary_z) {
        Field U(m.num_cells());
        for (int k = 0; k < m.nz; ++k)
            for (int j = 0; j < m.ny; ++j)
                for (int i = 0; i < m.nx; ++i) {
                    const double x = (i + 0.5) / m.nx, y = vary_y ? (j + 0.5) / m.ny : 0.0, z = vary_z ? (k + 0.5) / m.nz : 0.0;
                    const double ph = 2.0 * pi * (x + y + z);
                    U[m.cell(i, j, k)] = eos.prim_to_cons({1.0 + 0.2 * std::sin(ph), 0.5 + 0.1 * std::cos(ph), vary_y ? 0.3 * std::sin(ph) : 0.0,
                                                           1.0 + 0.1 * std::sin(ph + 1.0), 0.0});
                }
        return U;
    };
    // the same arrays as Simulation::assemble: the sides filled by the boundary condition (unused here: everything periodic)
    const auto fluxes = [&](const Mesh& m, const Field& U, Field& F2, Field& F4) {
        PerHalos ph;
        make_halos(m, U, ph);
        Field UL, UR;
        UL.resize(m.num_faces());
        UR.resize(m.num_faces());
        mu.reconstruct_conserved(UL, UR, U, m, ph.h);
        rs.compute_flux(UL, UR, m, F2);
        fourth_order_fluxes(rs, eos, UL, UR, m, ph.h, per, F4);
        return std::make_pair(UL, UR);
    };
    {
        // x-only field: no transverse variation, the correction is zero
        const Mesh m = generate_cartesian_mesh(1.0, 1.0, 24, 12);
        Field F2, F4;
        fluxes(m, smooth(m, false, false), F2, F4);
        bool same = F2.size() == F4.size();
        for (int f = 0; same && f < m.num_xfaces(); ++f)   // the x faces: their transverse axis is y
            for (int q = 0; q < kNVar; ++q) same = same && F2[f][q] == F4[f][q];
        CHECK(same);
    }
    {
        // 2-D field varying in both directions: the flux changes (by a truncation-sized amount) and the balance is conservative
        const Mesh m = generate_cartesian_mesh(1.0, 1.0, 24, 24);
        Field F2, F4;
        fluxes(m, smooth(m, true, false), F2, F4);
        double dmax = 0.0;
        for (size_t f = 0; f < F2.size(); ++f)
            for (int q = 0; q < kNVar; ++q) dmax = std::max(dmax, std::abs(F4[f][q] - F2[f][q]));
        CHECK(dmax > 1e-7);
        CHECK(dmax < 1e-2);
        double sum[kNVar] = {};
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i)
                for (int q = 0; q < kNVar; ++q)
                    sum[q] += (F4[m.xface(i + 1, j)][q] - F4[m.xface(i, j)][q]) * m.dy + (F4[m.yface(i, j + 1)][q] - F4[m.yface(i, j)][q]) * m.dx;
        for (int q = 0; q < kNVar; ++q) CHECK(std::abs(sum[q]) < 1e-12);
    }
    {
        // a 3-D field that does not vary along z is its 2-D layer: the same x and y fluxes
        const Mesh m2 = generate_cartesian_mesh(1.0, 1.0, 16, 16);
        const Mesh m3 = generate_cartesian_mesh_3d(1.0, 1.0, 0.5, 16, 16, 6);
        Field F2a, F4a, F2b, F4b;
        fluxes(m2, smooth(m2, true, false), F2a, F4a);
        fluxes(m3, smooth(m3, true, false), F2b, F4b);
        bool same = true;
        for (int k = 0; k < m3.nz; ++k)
            for (int j = 0; j < m3.ny; ++j)
                for (int i = 0; i <= m3.nx; ++i)
                    for (int q = 0; q < kNVar; ++q) same = same && F4b[m3.xface(i, j, k)][q] == F4a[m2.xface(i, j)][q];
        for (int k = 0; k < m3.nz; ++k)
            for (int j = 0; j <= m3.ny; ++j)
                for (int i = 0; i < m3.nx; ++i)
                    for (int q = 0; q < kNVar; ++q) same = same && F4b[m3.yface(i, j, k)][q] == F4a[m2.yface(i, j)][q];
        CHECK(same);
        // and a field that varies along z has z fluxes that are corrected too
        Field F2c, F4c;
        fluxes(m3, smooth(m3, true, true), F2c, F4c);
        double dz = 0.0;
        for (int k = 1; k < m3.nz; ++k)
            for (int j = 0; j < m3.ny; ++j)
                for (int i = 0; i < m3.nx; ++i)
                    for (int q = 0; q < kNVar; ++q) dz = std::max(dz, std::abs(F4c[m3.zface(i, j, k)][q] - F2c[m3.zface(i, j, k)][q]));
        CHECK(dz > 1e-8);
    }
    {
        // a strong pressure jump: the shock sensor fires on the faces next to it, which keep the flux of their averaged states
        const Mesh m = generate_cartesian_mesh(1.0, 1.0, 24, 24);
        Field U = smooth(m, true, false);
        for (int j = 8; j < 16; ++j)
            for (int i = 12; i < 24; ++i) U[m.cell(i, j)] = eos.prim_to_cons({1.0, 0.0, 0.0, 30.0, 0.0});
        Field F2, F4;
        const auto st = fluxes(m, U, F2, F4);
        const int f = m.xface(12, 10);   // between the jump cell (12) and its neighbour
        double sm = 0.0;
        const State mid = rs.face_flux(st.first[f], st.second[f], 0, sm);
        bool same = true;
        for (int q = 0; q < kNVar; ++q) same = same && F4[f][q] == mid[q];
        CHECK(same);
        for (const State& s : F4)
            for (int q = 0; q < kNVar; ++q) CHECK(std::isfinite(s[q]));
    }
}

TEST(fused_fourth_order_residual_equals_the_two_pass_residual_bit_for_bit) {
    // flux_quadrature = fourth on a 2-D mesh: the pass over blocks of rows (reconstruction, point values, Riemann solves, correction and balance)
    // gives what reconstruct_conserved, fourth_order_fluxes and the divergence of the face fluxes give, for any thread count (the blocks and
    // their repeated seam rows change, the values do not), with walls, outflow and periodic seams, a smooth field with noise and a pressure jump
    // (the shock sensor and the positivity fallbacks fire on part of the faces)
    const EOS eos = EOS::ideal(1.4);
    const double pi = 3.14159265358979323846;
    unsigned seed = 777u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<double>(seed >> 8) / 16777216.0;
    };
    const Mesh m = generate_cartesian_mesh(1.0, 0.8, 23, 53);
    const int nx = m.nx, ny = m.ny;
    Field U(m.num_cells());
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double ph = 2.0 * pi * ((i + 0.5) / nx + (j + 0.5) / ny);
            double p = 1.0 + 0.1 * std::sin(ph + 1.0) + 0.01 * (rnd() - 0.5);
            if (j >= 20 && j < 26 && i > 8) p *= 4.0;   // a pressure jump
            U[j * nx + i] = eos.prim_to_cons({1.0 + 0.2 * std::sin(ph) + 0.01 * (rnd() - 0.5), 0.5 + 0.1 * std::cos(ph), 0.3 * std::sin(ph), p, 0.0});
        }
    constexpr int D = 3;
    long long corrected = 0;
    for (const Reconstruction rec : {Reconstruction::Weno3, Reconstruction::Weno5}) {
        const MusclReconstruction mu(eos, Limiter::VanLeer, rec, 1e-6);
        for (const RiemannType rt : {RiemannType::HLLC, RiemannType::LowMachHLLC}) {
            const RiemannSolver rs(eos, rt, 0.2);
            for (const bool per_x : {false, true}) {
                BoundaryManager bm(m, eos);
                if (per_x)
                    bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Wall, BCType::Transmissive);
                else
                    bm.set_boundary_types(BCType::Wall, BCType::Transmissive, BCType::Periodic, BCType::Periodic);
                // wrap-around halos on the periodic axis, as Simulation::assemble builds them
                Field perL, perR, perB, perT;
                Halos h;
                h.depth = D;
                if (per_x) {
                    perL.resize(static_cast<size_t>(ny) * D);
                    perR.resize(perL.size());
                    for (int j = 0; j < ny; ++j)
                        for (int d = 0; d < D; ++d) {
                            perL[j * D + d] = U[j * nx + nx - D + d];
                            perR[j * D + d] = U[j * nx + d];
                        }
                    h.l = &perL;
                    h.r = &perR;
                } else {
                    perB.assign(U.end() - static_cast<std::ptrdiff_t>(D) * nx, U.end());
                    perT.assign(U.begin(), U.begin() + static_cast<std::ptrdiff_t>(D) * nx);
                    h.b = &perB;
                    h.t = &perT;
                }
                const bool per3[3] = {per_x, !per_x, false}, per2[2] = {per_x, !per_x};
                // two passes
                Field UL, UR, F, F0, R2;
                bm.apply_boundary_conditions(U, UL, UR);
                mu.reconstruct_conserved(UL, UR, U, m, h);
                rs.compute_flux(UL, UR, m, F0);
                const double s2 = fourth_order_fluxes(rs, eos, UL, UR, m, h, per3, F);
                R2.resize(U.size());
                const double inv_area = 1.0 / m.cell_area(), dxl = m.dx, dyl = m.dy;
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i)
                        for (int q = 0; q < kNVar; ++q) {
                            const State &fl = F[m.xface(i, j)], &fr = F[m.xface(i + 1, j)], &fb = F[m.yface(i, j)], &ft = F[m.yface(i, j + 1)];
                            R2[j * nx + i][q] = ((0.0 - (fr[q] * dyl - fl[q] * dyl)) - (ft[q] * dxl - fb[q] * dxl)) * inv_area;
                        }
                for (size_t f = 0; f < F.size(); ++f)
                    for (int q = 0; q < kNVar; ++q) corrected += F[f][q] != F0[f][q];
                // the fused pass, from the first-order states of the sides only
                Field ULs, URs;
                bm.apply_boundary_conditions(U, ULs, URs, nullptr, false);
                for (const int threads : {1, 2, 3, 7}) {
#ifdef _OPENMP
                    omp_set_num_threads(threads);
#endif
                    Field R1;
                    const double s1 = mu.fourth_order_residual_2d(rs, ULs, URs, U, m, h, per2, R1);
                    CHECK(s1 == s2);
                    bool same = R1.size() == R2.size();
                    for (size_t c = 0; same && c < R1.size(); ++c)
                        for (int q = 0; q < kNVar; ++q) same = same && R1[c][q] == R2[c][q];
                    CHECK(same);
                }
            }
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs());
#endif
    CHECK(corrected > 1000);   // the correction was exercised (the fourth-order fluxes differ from those of the averaged states on many faces)
}

TEST(fused_fourth_order_residual_3d_equals_the_two_pass_residual_bit_for_bit) {
    // the same for a 3-D mesh: tiles (strips of rows times blocks of planes, with the repeated rows and planes at their seams) give what
    // reconstruct_conserved, fourth_order_fluxes and the divergence of the face fluxes give, for any thread count, with walls, outflow and
    // periodic axes (seams that wrap in the row, in the plane and across the tiles), noise and a pressure jump
    const EOS eos = EOS::ideal(1.4);
    const double pi = 3.14159265358979323846;
    unsigned seed = 4242u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<double>(seed >> 8) / 16777216.0;
    };
    const Mesh m = generate_cartesian_mesh_3d(1.0, 0.8, 0.6, 13, 70, 50);
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    Field U(m.num_cells());
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double ph = 2.0 * pi * ((i + 0.5) / nx + (j + 0.5) / ny + (k + 0.5) / nz);
                double p = 1.0 + 0.1 * std::sin(ph + 1.0) + 0.01 * (rnd() - 0.5);
                if (j >= 30 && j < 36 && k >= 20 && k < 27) p *= 4.0;   // a pressure jump
                U[m.cell(i, j, k)] = eos.prim_to_cons({1.0 + 0.2 * std::sin(ph) + 0.01 * (rnd() - 0.5), 0.5 + 0.1 * std::cos(ph),
                                                       0.3 * std::sin(ph), p, 0.2 * std::cos(ph + 0.5)});
            }
    constexpr int D = 3;
    long long corrected = 0;
    for (const Reconstruction rec : {Reconstruction::Weno3, Reconstruction::Weno5}) {
        const MusclReconstruction mu(eos, Limiter::VanLeer, rec, 1e-6);
        const RiemannSolver rs(eos, RiemannType::HLLC, 0.2);
        for (int variant = 0; variant < 2; ++variant) {
            const bool pa[3] = {variant == 1, variant == 0, variant == 0};
            BoundaryManager bm(m, eos);
            const Face faces[6] = {Face::Left, Face::Right, Face::Bottom, Face::Top, Face::Front, Face::Back};
            for (int f = 0; f < 6; ++f) bm.set_type(faces[f], pa[f / 2] ? BCType::Periodic : (f % 2 == 0 ? BCType::Wall : BCType::Transmissive));
            Field hl, hr, hb, ht, hf, hk;
            Halos h;
            h.depth = D;
            if (pa[0]) {
                hl.resize(static_cast<size_t>(ny) * nz * D);
                hr.resize(hl.size());
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j)
                        for (int d = 0; d < D; ++d) {
                            hl[(k * ny + j) * D + d] = U[m.cell(nx - D + d, j, k)];
                            hr[(k * ny + j) * D + d] = U[m.cell(d, j, k)];
                        }
                h.l = &hl;
                h.r = &hr;
            }
            if (pa[1]) {
                hb.resize(static_cast<size_t>(D) * nz * nx);
                ht.resize(hb.size());
                for (int d = 0; d < D; ++d)
                    for (int k = 0; k < nz; ++k)
                        for (int i = 0; i < nx; ++i) {
                            hb[(d * nz + k) * nx + i] = U[m.cell(i, ny - D + d, k)];
                            ht[(d * nz + k) * nx + i] = U[m.cell(i, d, k)];
                        }
                h.b = &hb;
                h.t = &ht;
            }
            if (pa[2]) {
                hf.resize(static_cast<size_t>(D) * ny * nx);
                hk.resize(hf.size());
                for (int d = 0; d < D; ++d)
                    for (int j = 0; j < ny; ++j)
                        for (int i = 0; i < nx; ++i) {
                            hf[(d * ny + j) * nx + i] = U[m.cell(i, j, nz - D + d)];
                            hk[(d * ny + j) * nx + i] = U[m.cell(i, j, d)];
                        }
                h.f = &hf;
                h.bk = &hk;
            }
            // two passes
            Field UL, UR, F, F0, R2;
            bm.apply_boundary_conditions(U, UL, UR);
            mu.reconstruct_conserved(UL, UR, U, m, h);
            rs.compute_flux(UL, UR, m, F0);
            const double s2 = fourth_order_fluxes(rs, eos, UL, UR, m, h, pa, F);
            R2.resize(U.size());
            const double Ax = m.face_area(0), Ay = m.face_area(1), Az = m.face_area(2), inv_vol = 1.0 / m.cell_volume();
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i)
                        for (int q = 0; q < kNVar; ++q) {
                            const State &fl = F[m.xface(i, j, k)], &fr = F[m.xface(i + 1, j, k)], &fb = F[m.yface(i, j, k)],
                                        &ft = F[m.yface(i, j + 1, k)], &ff = F[m.zface(i, j, k)], &fk = F[m.zface(i, j, k + 1)];
                            R2[m.cell(i, j, k)][q] =
                                (((0.0 - (fr[q] * Ax - fl[q] * Ax)) - (ft[q] * Ay - fb[q] * Ay)) - (fk[q] * Az - ff[q] * Az)) * inv_vol;
                        }
            for (size_t f = 0; f < F.size(); ++f)
                for (int q = 0; q < kNVar; ++q) corrected += F[f][q] != F0[f][q];
            // the fused pass, from the first-order states of the sides only
            Field ULs, URs;
            bm.apply_boundary_conditions(U, ULs, URs, nullptr, false);
            for (const int threads : {1, 3, 7}) {
#ifdef _OPENMP
                omp_set_num_threads(threads);
#endif
                Field R1;
                const double s1 = mu.fourth_order_residual_3d(rs, ULs, URs, U, m, h, pa, R1);
                CHECK(s1 == s2);
                bool same = R1.size() == R2.size();
                for (size_t c = 0; same && c < R1.size(); ++c)
                    for (int q = 0; q < kNVar; ++q) same = same && R1[c][q] == R2[c][q];
                CHECK(same);
            }
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs());
#endif
    CHECK(corrected > 1000);
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

// The one-sweep viscous residual (rows in cache, no gradient field, no padded or primitive array: the cell field and the ghost
// ring only) must give the bits of the gradient version on the padded array: random state, a mesh above kOmpMinWork so that
// the thread blocks have seams, 1 / 3 / 7 threads, walls (also with second-order boundary faces and fixed corners),
// neighbour-rank layers in the ring and periodic seams on both axes, replacing and accumulating.
TEST(viscous_rows_equal_the_gradient_version_bit_for_bit) {
    const EOS eos = EOS::ideal(1.4);
    // 70 columns: the vector kernel of the row cells covers the interior exactly; 77: three leftover columns; 6: the smallest row it runs on
    long long compared = 0;
    for (const int nxc : {70, 77, 6}) {
    const Mesh m = generate_cartesian_mesh(1.0, 0.8, nxc, nxc == 6 ? 700 : (nxc == 77 ? 57 : 61));
    unsigned seed = 4242u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<double>(seed >> 8) / 16777216.0;
    };
    Field U(m.num_cells());
    for (auto& u : U) u = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, 0.4 + rnd(), 0.0});
    // 0: periodic x and y, 1: walls, 2: walls with second-order boundary faces (boundary_order = 2, corners fixed),
    // 3: as 2 with neighbour-rank layers in the ghost ring (an x strip: columns, then a y strip: rows)
    for (const int variant : {0, 1, 2, 3, 4}) {
        BoundaryManager bm(m, eos);
        if (variant == 0)
            bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Periodic, BCType::Periodic);
        else
            bm.set_boundary_types(BCType::NoSlip, BCType::MovingWall, BCType::NoSlip, BCType::Transmissive);
        BCParams bp;
        bp.u_in = 1.0;
        bp.v_in = 0.3;
        bp.boundary_order = variant >= 2 ? 2 : 1;
        bm.set_params(bp);
        NavierStokes ns(eos, 0.02, 0.72);
        ns.set_second_order_boundary(variant >= 2);
        // the ghost ring as the solver builds it; the padded array of the reference is this ring around U
        PaddedFrame F;
        bm.get_padded_frame(U, F);
        const int mpi_axis = variant == 3 ? 0 : (variant == 4 ? 1 : -1);
        if (variant == 3) {
            for (auto& s : F.left) s = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, 0.4 + rnd(), 0.0});
            for (auto& s : F.right) s = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, 0.4 + rnd(), 0.0});
        } else if (variant == 4) {
            for (int i = 0; i < m.nx; ++i) {
                F.bottom[i + 1] = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, 0.4 + rnd(), 0.0});
                F.top[i + 1] = eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, 0.4 + rnd(), 0.0});
            }
        }
        bm.fix_frame_corners(F, mpi_axis, mpi_axis >= 0, mpi_axis >= 0);
        Field P((m.ny + 2) * (m.nx + 2)), gx, gy;
        for (int jj = 0; jj < m.ny + 2; ++jj)
            for (int ii = 0; ii < m.nx + 2; ++ii) {
                State& p = P[jj * (m.nx + 2) + ii];
                if (jj == 0) p = F.bottom[ii];
                else if (jj == m.ny + 1) p = F.top[ii];
                else if (ii == 0) p = F.left[jj - 1];
                else if (ii == m.nx + 1) p = F.right[jj - 1];
                else p = U[(jj - 1) * m.nx + ii - 1];
            }
        compute_gradients(P, m.nx, m.ny, m.dx, m.dy, gx, gy);
        CrossGhosts cg;
        cg.periodic_x = cg.periodic_y = (variant == 0);
        for (const bool accumulate : {false, true}) {
            Field Ra(m.num_cells()), Rb;
            // the zero slots start as -0 in every other cell: the balance of the rows must give the sign the reference gives
            size_t cc = 0;
            for (auto& r : Ra) {
                r = {0.1, 0.2, 0.3, 0.4};
                if (cc++ % 2) r[0] = r[4] = -0.0;
            }
            Rb = Ra;
            const double sa = ns.viscous_residual(U, gx, gy, m, P, Ra, &cg, accumulate);
            for (const int threads : {1, 3, 7}) {
#ifdef _OPENMP
                omp_set_num_threads(threads);
#endif
                Field Rc = Rb;
                const double sb = ns.viscous_residual_rows(U, F, m, Rc, &cg, accumulate);
                CHECK(std::memcmp(&sa, &sb, sizeof sa) == 0);
                bool same = Ra.size() == Rc.size();
                for (size_t c = 0; same && c < Ra.size(); ++c)
                    for (int q = 0; q < kNVar; ++q) same = same && std::memcmp(&Ra[c][q], &Rc[c][q], sizeof(double)) == 0;
                CHECK(same);
                compared += static_cast<long long>(Ra.size());
            }
        }
    }
    }
#ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs());
#endif
    std::printf("    %lld cell residuals compared\n", compared);
}

// The 3-D viscous sweep by planes (face fluxes of a plane at a time, balance and speed bound in the same loop) must give the bits of the
// sweep by passes (one face array per axis): random state, meshes with a width that is not a multiple of four, the thread blocks with
// seams (1 / 3 / 7 threads), walls on all six sides also with second-order boundary faces and fixed edges, periodic seams on every axis,
// neighbour-rank layers of an x strip and of a y strip with their cross-derivative slabs, replacing and accumulating, the speed bound.
TEST(viscous_3d_plane_sweep_equals_the_passes_bit_for_bit) {
    const EOS eos = EOS::ideal(1.4);
    long long compared = 0;
    struct Dims {
        int nx, ny, nz;
    };
    // above kOmpMinWork cells so that the thread blocks run; 90 x 12 x 5: fewer planes than 7 threads (the passes take over), 13 / 37: leftover columns
    for (const Dims dm : {Dims{17, 14, 20}, Dims{13, 16, 24}, Dims{37, 5, 25}, Dims{90, 12, 5}}) {
        const Mesh m = generate_cartesian_mesh_3d(1.0, 0.8, 0.6, dm.nx, dm.ny, dm.nz);
        unsigned seed = 777u + static_cast<unsigned>(dm.nx);
        const auto rnd = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return static_cast<double>(seed >> 8) / 16777216.0;
        };
        const auto random_state = [&]() { return eos.prim_to_cons({0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, 0.4 + rnd(), rnd() - 0.5}); };
        Field U(m.num_cells());
        for (auto& u : U) u = random_state();
        // 0: periodic on all axes, 1: walls on all six sides, 2: as 1 with boundary_order = 2, 3: periodic z with walls in x and y (order 2),
        // 4: x strip (neighbour layers on both x sides), 5: y strip, both with second-order boundary faces and the other sides walls
        for (const int variant : {0, 1, 2, 3, 4, 5}) {
            BoundaryManager bm(m, eos);
            if (variant == 0) {
                bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Periodic, BCType::Periodic);
                bm.set_type(Face::Front, BCType::Periodic);
                bm.set_type(Face::Back, BCType::Periodic);
            } else {
                bm.set_boundary_types(BCType::NoSlip, BCType::MovingWall, BCType::NoSlip, BCType::Transmissive);
                bm.set_type(Face::Front, variant == 3 ? BCType::Periodic : BCType::NoSlip);
                bm.set_type(Face::Back, variant == 3 ? BCType::Periodic : BCType::Wall);
            }
            BCParams bp;
            bp.u_in = 1.0;
            bp.v_in = 0.3;
            bp.boundary_order = variant >= 2 ? 2 : 1;
            bm.set_params(bp);
            NavierStokes ns(eos, 0.02, 0.72);
            ns.set_second_order_boundary(variant >= 2);
            Field P;
            bm.get_padded_U(U, P);
            const int PX = m.nx + 2, PY = m.ny + 2;
            const int mpi_axis = variant == 4 ? 0 : (variant == 5 ? 1 : -1);
            if (mpi_axis == 0) {   // the layers of the neighbour strips: columns 0 and nx + 1 of the interior rows and planes
                for (int k = 0; k < m.nz; ++k)
                    for (int j = 0; j < m.ny; ++j) {
                        P[(static_cast<size_t>(k + 1) * PY + j + 1) * PX + 0] = random_state();
                        P[(static_cast<size_t>(k + 1) * PY + j + 1) * PX + m.nx + 1] = random_state();
                    }
            } else if (mpi_axis == 1) {
                for (int k = 0; k < m.nz; ++k)
                    for (int i = 0; i < m.nx; ++i) {
                        P[(static_cast<size_t>(k + 1) * PY + 0) * PX + i + 1] = random_state();
                        P[(static_cast<size_t>(k + 1) * PY + m.ny + 1) * PX + i + 1] = random_state();
                    }
            }
            bm.fix_padded_corners(P, mpi_axis, mpi_axis >= 0, mpi_axis >= 0);
            // the cross-derivative slabs of the neighbour strips (x strip: slab index k * ny + j, y strip: k * nx + i)
            std::vector<double> slab[8];
            CrossGhosts cg;
            cg.periodic_x = cg.periodic_y = cg.periodic_z = (variant == 0);
            if (variant == 3) cg.periodic_z = true;
            if (mpi_axis >= 0) {
                cg.dir = mpi_axis == 0 ? 'x' : 'y';
                const size_t len = static_cast<size_t>(m.nz) * (mpi_axis == 0 ? m.ny : m.nx);
                const double** slots[8] = {&cg.lo_a, &cg.lo_b, &cg.lo_c, &cg.lo_d, &cg.hi_a, &cg.hi_b, &cg.hi_c, &cg.hi_d};
                for (int s = 0; s < 8; ++s) {
                    slab[s].resize(len);
                    for (auto& v : slab[s]) v = rnd() - 0.5;
                    *slots[s] = slab[s].data();
                }
            }
            for (const bool accumulate : {false, true}) {
                Field Rb(m.num_cells());
                size_t cc = 0;
                for (auto& r : Rb) {
                    r = {0.1, 0.2, 0.3, 0.4, 0.5};
                    if (cc++ % 2) r[0] = -0.0;   // the zero slot starts as -0 in every other cell
                }
                Field Ra = Rb;
#ifdef _OPENMP
                omp_set_num_threads(1);
#endif
                ns.set_fused_sweep(false);
                const double sa = ns.viscous_residual_3d(U, m, P, Ra, &cg, 0.0, accumulate);
                ns.set_fused_sweep(true);
                for (const int threads : {1, 3, 7}) {
#ifdef _OPENMP
                    omp_set_num_threads(threads);
#endif
                    Field Rc = Rb;
                    const double sb = ns.viscous_residual_3d(U, m, P, Rc, &cg, 0.0, accumulate);
                    CHECK(std::memcmp(&sa, &sb, sizeof sa) == 0);
                    bool same = Ra.size() == Rc.size();
                    for (size_t c = 0; same && c < Ra.size(); ++c)
                        for (int q = 0; q < kNVar; ++q) same = same && std::memcmp(&Ra[c][q], &Rc[c][q], sizeof(double)) == 0;
                    CHECK(same);
                    compared += static_cast<long long>(Ra.size());
                }
            }
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(omp_get_num_procs());
#endif
    std::printf("    %lld cell residuals compared\n", compared);
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
    for (TimeSchemeType tt : {TimeSchemeType::ExplicitEuler, TimeSchemeType::SSPRK2, TimeSchemeType::SSPRK3}) {
        auto ts = make_time_scheme(tt, 0.1);
        Field U(1, State{1, 1, 1, 1});
        const StepResult r = ts->step(U, res, 1.0, 0.0, 10.0);   // dt = cfl*min_delta/S = 0.1
        CHECK_NEAR(r.dt, 0.1, 1e-9);
        const double tol = tt == TimeSchemeType::ExplicitEuler ? 1e-2 : (tt == TimeSchemeType::SSPRK2 ? 1e-3 : 1e-5);
        CHECK_NEAR(U[0][0], exact, tol);
    }
    // Step clamped to t_max
    auto e = make_time_scheme(TimeSchemeType::ExplicitEuler, 0.5);
    Field U(1, State{1, 1, 1, 1});
    CHECK_NEAR(e->step(U, res, 1.0, 0.99, 1.0).dt, 0.01, 1e-12);
}

// A field above the size at which the first-stage update of SSP-RK2 / RK3 writes past the caches (odd cell count: the ragged ends and the
// unaligned start of every thread's range) gives the same bits as the plain loops of the scheme.
TEST(time_update_past_the_caches_equals_the_plain_loops_bit_for_bit) {
    const size_t n = 1100003;   // 44 MB of State
    Field U(n);
    for (size_t c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) U[c][q] = std::sin(0.001 * double(c) + q) + 2.0;
    const ResidualFn res = [](const Field& V, Field& R) {
        R.resize(V.size());
        for (size_t c = 0; c < V.size(); ++c)
            for (int q = 0; q < kNVar; ++q) R[c][q] = -V[c][q] * V[c][q] + 0.1 * q;
        return 1.0;
    };
    const double cfl = 0.4;
    const double dt = cfl * 1.0 / (1.0 + 1e-10);
    // plain SSP-RK2 reference
    Field U1(n), R(n), expect(n);
    res(U, R);
    for (size_t c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) U1[c][q] = U[c][q] + dt * R[c][q];
    res(U1, R);
    for (size_t c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) expect[c][q] = 0.5 * U[c][q] + 0.5 * (U1[c][q] + dt * R[c][q]);
    auto ts = make_time_scheme(TimeSchemeType::SSPRK2, cfl);
    ts->step(U, res, 1.0, 0.0, 10.0);
    CHECK(std::memcmp(U.data(), expect.data(), n * sizeof(State)) == 0);
}

// ---------------------------------------------------------------------------
// IBM
// ---------------------------------------------------------------------------

TEST(ibm_mask_and_reset) {
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 100, 100);
    ImmersedBoundary ib(m, "Cylinder", 0.2, 0.5, 0.5, 1.0, 1.0);
    CHECK(std::abs(ib.solid_count() - 3.14159265 * 0.04 * 1e4) < 60);
    Field U = uniform_field(eos, m.num_cells(), 1.1, 0.5, 0.2, 1.3);
    bool refused = false;   // the forcing is the wall's: not before enable_ghost_cells
    try {
        ib.apply_forcing(U, eos);
    } catch (const std::logic_error&) { refused = true; }
    CHECK(refused);
    ib.enable_ghost_cells(m, false);
    ib.apply_forcing(U, eos);
    CHECK(U[50 * 100 + 50][1] == 0.0);                                                 // deep solid: stopped
    CHECK(U[50 * 100 + 50][0] == 1.1);                                                 // ... density and pressure kept in a stage
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

TEST(ibm_ghost_pressure_follows_the_hydrostatic_gradient_with_gravity) {
    // A gas of uniform density rho at rest under gravity g has the linear pressure p = p0 + rho g.x. The image state is bilinear, exact on it, so with
    // set_gravity the ghost pressure is the exact extension of the field to the ghost cell; without it (the plain copy, zero normal gradient) it is not.
    const EOS eos = EOS::ideal(1.4);
    const double pi = std::acos(-1.0), a = pi / 6.0, tx = std::cos(a), ty = std::sin(a), nxw = -ty, nyw = tx;
    const double ox = 0.2, oy = 0.3, rho = 1.2, p0 = 2.0, gx = 0.3, gy = -0.8;
    Polygon2D poly;
    for (const auto& q : std::vector<std::array<double, 2>>{{-1.0, 0.0}, {2.0, 0.0}, {2.0, -1.0}, {-1.0, -1.0}})
        poly.pts.push_back({ox + q[0] * tx + q[1] * nxw, oy + q[0] * ty + q[1] * nyw});
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 64, 64);
    const auto pressure = [&](double x, double y) { return p0 + rho * (gx * x + gy * y); };
    double worst_with = 0.0, worst_without = 0.0;
    for (const bool gravity : {true, false}) {
        ImmersedBoundary ib(m, poly, rho, p0);
        if (gravity) ib.set_gravity(gx, gy, 0.0);
        ib.enable_ghost_cells(m, false);
        Field U(static_cast<size_t>(m.num_cells()));
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) U[m.cell(i, j)] = eos.prim_to_cons({rho, 0.0, 0.0, pressure(m.xc(i), m.yc(j)), 0.0});
        ib.apply_forcing(U, eos);
        double& worst = gravity ? worst_with : worst_without;
        for (const int c : ib.ghost_cell_list()) {
            if (c % m.nx < 4) continue;   // the image stencils of the cells at the domain edge are clipped by it: not exact there
            worst = std::max(worst, std::abs(eos.cons_to_prim(U[c])[3] - pressure(m.xc(c % m.nx), m.yc(c / m.nx))));
        }
    }
    CHECK(worst_with < 1e-12);
    CHECK(worst_without > 1e-3);   // the plain copy misses the gradient: the test can fail
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

TEST(ibm_ghost_cells_without_an_image_point_take_the_nearest_fluid_cell) {
    // A block with a slot one cell wide: the image points of the ghost cells on the two walls of the slot lie in the opposite wall, at
    // any distance up to four cells. Those cells take the density and pressure of the nearest fluid cell (the slot cell of their row)
    // and stay at rest (no-slip) or keep its velocity along the wall (slip); the build does not stop, and counts them.
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 40, 40);
    Polygon2D poly;
    for (const auto& q : std::vector<std::array<double, 2>>{{0.2, 0.2}, {0.8, 0.2}, {0.8, 0.8}, {0.525, 0.8}, {0.525, 0.4}, {0.5, 0.4},
                                                           {0.5, 0.8}, {0.2, 0.8}})
        poly.pts.push_back({q[0], q[1]});
    const int islot = 20;   // the column of the slot (centre 0.5125)
    for (const bool slip : {false, true}) {
        ImmersedBoundary ib(m, poly, 1.0, 1.0);
        CHECK(!ib.mask()[m.cell(islot, 24)] && ib.mask()[m.cell(islot - 1, 24)] && ib.mask()[m.cell(islot + 1, 24)]);
        ib.enable_ghost_cells(m, slip);
        CHECK(ib.ghost_fallback_count() > 0);
        Field U(static_cast<size_t>(m.num_cells()));
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) U[m.cell(i, j)] = eos.prim_to_cons({1.0 + m.yc(j), 0.3, 0.5, 2.0 + m.yc(j), 0.0});
        ib.apply_forcing(U, eos);
        double worst = 0.0;
        int seen = 0;
        for (int j = 21; j <= 27; ++j)   // rows well inside the slot (0.525 < y < 0.7)
            for (const int i : {islot - 2, islot - 1, islot + 1, islot + 2}) {
                const State W = eos.cons_to_prim(U[m.cell(i, j)]);
                ++seen;
                worst = std::max({worst, std::abs(W[0] - (1.0 + m.yc(j))), std::abs(W[3] - (2.0 + m.yc(j))), std::abs(W[1]),
                                  std::abs(W[2] - (slip ? 0.5 : 0.0))});
            }
        CHECK(seen == 28);
        CHECK(worst < 1e-12);
        if (worst >= 1e-12) std::printf("    slip %d: fallback ghost error %.3e\n", slip, worst);
    }
    // a convex body needs none
    ImmersedBoundary cyl(m, "Cylinder", 0.2, 0.5, 0.5, 1.0, 1.0);
    cyl.enable_ghost_cells(m, false);
    CHECK(cyl.ghost_fallback_count() == 0);
}

TEST(ibm_ghost_cells_of_a_sphere_reproduce_a_linear_field) {
    // 3-D sphere on a mesh with different spacings: with density and pressure uniform and a velocity linear in space, the trilinear
    // image state is exact, so every ghost state is the reflection of the analytic velocity at its image point (no-slip: -ratio u;
    // slip: the normal part reflected). The image point is on the normal through the ghost cell, di = dg / ratio beyond the wall.
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh_3d(1.0, 1.2, 0.9, 40, 48, 30);
    const double R = 0.25, cx = 0.5, cy = 0.6, cz = 0.45;
    const double a[3] = {0.3, -0.2, 0.15}, B[3][3] = {{0.4, 0.1, -0.2}, {-0.3, 0.2, 0.25}, {0.05, -0.15, 0.3}};
    const auto vel = [&](double x, double y, double z, double* u) {
        for (int q = 0; q < 3; ++q) u[q] = a[q] + B[q][0] * x + B[q][1] * y + B[q][2] * z;
    };
    for (const bool slip : {false, true}) {
        ImmersedBoundary ib(m, "Sphere", R, cx, cy, 1.0, 1.0, cz);
        ib.enable_ghost_cells(m, slip);
        CHECK(ib.ghost_cells());
        CHECK(ib.ghost_count() > 200);
        Field U(static_cast<size_t>(m.num_cells()));
        for (int k = 0; k < m.nz; ++k)
            for (int j = 0; j < m.ny; ++j)
                for (int i = 0; i < m.nx; ++i) {
                    double u[3];
                    vel(m.xc(i), m.yc(j), m.zc(k), u);
                    U[m.cell(i, j, k)] = eos.prim_to_cons({1.2, u[0], u[1], 2.0, u[2]});
                }
        const Field U0 = U;
        ib.apply_forcing(U, eos);
        const std::vector<int> cells = ib.ghost_cell_list();
        const std::vector<double> ratios = ib.ghost_ratios();
        std::vector<int> reads(static_cast<size_t>(m.num_cells()), 0);
        for (const auto& pr : ib.ghost_reads()) ++reads[static_cast<size_t>(pr.first)];
        double worst = 0.0;
        int checked = 0;
        for (size_t q = 0; q < cells.size(); ++q) {
            const int c = cells[q];
            if (reads[static_cast<size_t>(c)] != 8) continue;   // a solid cell dropped from the stencil: not exact
            const int i = c % m.nx, j = (c / m.nx) % m.ny, k = c / (m.nx * m.ny);
            const double dx = m.xc(i) - cx, dy = m.yc(j) - cy, dz = m.zc(k) - cz, r = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double n[3] = {dx / r, dy / r, dz / r}, dg = R - r, di = dg / ratios[q];
            CHECK(dg > 0.0 && ratios[q] > 0.0 && ratios[q] <= 1.0);
            double ui[3];
            vel(cx + (R + di) * n[0], cy + (R + di) * n[1], cz + (R + di) * n[2], ui);
            const double un = ui[0] * n[0] + ui[1] * n[1] + ui[2] * n[2];
            const State W = eos.cons_to_prim(U[c]);
            const double got[3] = {W[1], W[2], W[4]};
            for (int d = 0; d < 3; ++d) {
                const double want = slip ? ui[d] - (1.0 + ratios[q]) * un * n[d] : -ratios[q] * ui[d];
                worst = std::max(worst, std::abs(got[d] - want));
            }
            worst = std::max({worst, std::abs(W[0] - 1.2), std::abs(W[3] - 2.0)});
            ++checked;
        }
        CHECK(checked > 100);
        CHECK(worst < 1e-10);
        if (worst >= 1e-10) std::printf("    slip %d: worst ghost error %.3e over %d ghosts\n", slip, worst, checked);
        // the cells that are not ghosts keep their state outside the body, the deep solid cells are at rest
        for (int c = 0; c < m.num_cells(); ++c) {
            if (!ib.mask()[c]) { CHECK(U[c][0] == U0[c][0]); continue; }
        }
    }
    // a sphere in a 2-D mesh is a circle: the ghost wall of a 2-D "Sphere" works and has no z-component
    const Mesh m2 = generate_cartesian_mesh(1.0, 1.0, 40, 40);
    ImmersedBoundary circ(m2, "Sphere", 0.2, 0.5, 0.5, 1.0, 1.0);
    circ.enable_ghost_cells(m2, true);
    CHECK(circ.ghost_count() > 20);
}

TEST(ibm_curvature_corrects_the_ghost_pressure_on_a_curved_wall) {
    // Supersonic vortex around a cylinder (u_theta = A/r, isentropic, Mach 2.25 at the wall): the exact pressure grows outwards with
    // dp/dr = rho u^2 / r, which the zero-gradient ghost state ignores. With the correction the ghost pressure follows the exact
    // field (error down by an order of magnitude), a polygon approximating the circle gives the same ghost states as the circle
    // (its curvature estimate is 1/R), and a flat wall is left untouched.
    const EOS eos = EOS::ideal(1.4);
    const double g = 1.4, Mi = 2.25, R = 1.0, c0 = 2.0;
    const Mesh m = generate_cartesian_mesh(4.0, 4.0, 640, 640);   // 160 cells per radius
    const auto exact = [&](double x, double y) {
        const double dx = x - c0, dy = y - c0, r2 = std::max(dx * dx + dy * dy, 0.5 * R * R);   // the analytic continuation inside the body (ghost cells)
        const double c2 = 1.0 + 0.5 * (g - 1.0) * Mi * Mi * (1.0 - R * R / r2), rho = std::pow(c2, 1.0 / (g - 1.0));
        return std::array<double, 4>{rho, -Mi * R * dy / r2, Mi * R * dx / r2, rho * c2 / g};
    };
    Field U0(static_cast<size_t>(m.num_cells()));
    for (int j = 0; j < m.ny; ++j)
        for (int i = 0; i < m.nx; ++i) {
            const auto w = exact(m.xc(i), m.yc(j));
            U0[m.cell(i, j)] = eos.prim_to_cons({w[0], w[1], w[2], w[3], 0.0});
        }
    // worst relative pressure error over the ghost cells within one cell of the wall (the deeper ones extrapolate further)
    const auto ghost_pressure_error = [&](ImmersedBoundary& ib, std::map<int, double>* out) {
        Field U = U0;
        ib.apply_forcing(U, eos);
        double worst = 0.0;
        for (const int c : ib.ghost_cell_list()) {
            const int i = c % m.nx, j = c / m.nx;
            if (std::hypot(m.xc(i) - c0, m.yc(j) - c0) < R - m.dx) continue;
            const double pe = exact(m.xc(i), m.yc(j))[3], pg = eos.cons_to_prim(U[c])[3];
            worst = std::max(worst, std::abs(pg / pe - 1.0));
            if (out) (*out)[c] = pg;
        }
        return worst;
    };
    ImmersedBoundary off(m, "Cylinder", R, c0, c0, 1.0, 1.0), on(m, "Cylinder", R, c0, c0, 1.0, 1.0);
    off.enable_ghost_cells(m, true, false);
    on.enable_ghost_cells(m, true, true);
    const double e_off = ghost_pressure_error(off, nullptr);
    std::map<int, double> p_circle;
    const double e_on = ghost_pressure_error(on, &p_circle);
    CHECK(e_off > 0.05);
    CHECK(e_on < 0.1 * e_off);
    if (e_on >= 0.1 * e_off) std::printf("    ghost pressure error: %.3e without, %.3e with the correction\n", e_off, e_on);
    // an 8000-gon of radius R (the estimate counts vertex turns inside a window of 3 cells: it needs many edges per window): same ghost pressures as the circle to a fraction of the correction
    const double pi = std::acos(-1.0);
    Polygon2D poly;
    for (int k = 0; k < 8000; ++k) poly.pts.push_back({c0 + R * std::cos(2.0 * pi * k / 8000.0), c0 + R * std::sin(2.0 * pi * k / 8000.0)});
    ImmersedBoundary pg(m, poly, 1.0, 1.0);
    pg.enable_ghost_cells(m, true, true);
    std::map<int, double> p_poly;
    ghost_pressure_error(pg, &p_poly);
    double worst_diff = 0.0;
    int common = 0;
    for (const auto& kv : p_circle) {
        const auto it = p_poly.find(kv.first);
        if (it == p_poly.end()) continue;
        ++common;
        worst_diff = std::max(worst_diff, std::abs(it->second / kv.second - 1.0));
    }
    CHECK(common > 100);
    CHECK(worst_diff < 0.01);
    if (worst_diff >= 0.01) std::printf("    polygon against circle: %.3e\n", worst_diff);
    // a flat wall (uniform field, tilted wall): the correction changes nothing
    const double a = pi / 6.0, tx = std::cos(a), ty = std::sin(a), nxw = -ty, nyw = tx;
    Polygon2D flat;
    for (const auto& q : std::vector<std::array<double, 2>>{{-8.0, 0.0}, {8.0, 0.0}, {8.0, -2.0}, {-8.0, -2.0}})
        flat.pts.push_back({1.0 + q[0] * tx + q[1] * nxw, 1.5 + q[0] * ty + q[1] * nyw});
    ImmersedBoundary f0(m, flat, 1.0, 1.0), f1(m, flat, 1.0, 1.0);
    f0.enable_ghost_cells(m, true, false);
    f1.enable_ghost_cells(m, true, true);
    Field Ua(static_cast<size_t>(m.num_cells())), Ub;
    for (int j = 0; j < m.ny; ++j)
        for (int i = 0; i < m.nx; ++i) Ua[m.cell(i, j)] = eos.prim_to_cons({1.2, 0.5 * tx, 0.5 * ty, 2.0, 0.0});
    Ub = Ua;
    f0.apply_forcing(Ua, eos);
    f1.apply_forcing(Ub, eos);
    double dmax = 0.0;
    for (size_t c = 0; c < Ua.size(); ++c)
        for (int q = 0; q < 4; ++q) dmax = std::max(dmax, std::abs(Ua[c][q] - Ub[c][q]));
    CHECK(dmax < 1e-9);
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
    double s1, s2;
    const State F1 = RiemannSolver(eos, RiemannType::HLLC).face_flux(UL, UR, 0, s1);
    const State F2 = RiemannSolver(eos, RiemannType::LowMachHLLC).face_flux(UL, UR, 0, s2);
    for (int q = 0; q < kNVar; ++q) CHECK_NEAR(F1[q], F2[q], 1e-12 * (1 + std::abs(F1[q])));
    // The removed preconditioned solver is refused with the name of its replacement
    bool refused = false;
    try {
        parse_riemann("HLLC (Preconditioned)");
    } catch (const std::invalid_argument& e) {
        refused = std::string(e.what()).find("Low-Mach Thornber") != std::string::npos;
    }
    CHECK(refused);
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
    // Ghost cells take rho / p from the fluid at their image point, the deeper solid cells the reference state
    const EOS eos = EOS::ideal(1.4);
    ImmersedBoundary ib(m, "Cylinder", 0.2, 0.5, 0.5, 9.0, 9.0);
    ib.enable_ghost_cells(m, false);
    Field U = uniform_field(eos, m.num_cells(), 1.7, 0.0, 0.0, 1.1);
    ib.apply_solid_reset(U, eos);
    int ghost = 0, deep = 0, other = 0;
    for (int c = 0; c < m.num_cells(); ++c)
        if (ib.mask()[c]) (std::abs(U[c][0] - 1.7) < 1e-12 ? ghost : (U[c][0] == 9.0 ? deep : other)) += 1;
    CHECK(ghost == ib.ghost_count());
    CHECK(deep > 0);
    CHECK(other == 0);
}

TEST(time_scheme_auto_takes_ssp_rk2_for_viscous_muscl) {
    const auto resolved = [](const char* scheme, const char* reconstruction, const char* viscous) {
        Params p = baseline_defaults();
        p.merge(get_test_case("Sod Shock Tube").defaults);
        p.set("test_case", "Sod Shock Tube");
        p.set("time_scheme", scheme);
        p.set("reconstruction", reconstruction);
        p.set("use_viscous", viscous);
        return SimParams::from(p).time_scheme;
    };
    CHECK(resolved("auto", "muscl", "false") == "MUSCL-Hancock");
    CHECK(resolved("auto", "muscl", "true") == "SSP Runge-Kutta 2");
    CHECK(resolved("auto", "none", "false") == "Explicit Euler");
    CHECK(resolved("auto", "none", "true") == "Explicit Euler");
    CHECK(resolved("MUSCL-Hancock", "muscl", "true") == "MUSCL-Hancock");   // by name it stays available
}

TEST(time_scheme_stages) {
    int calls = 0;
    double smax_seq[4] = {1.0, 3.0, 2.0, 0.5};
    const ResidualFn res = [&](const Field& U, Field& R) {
        R.assign(U.size(), State{0, 0, 0, 0});
        return smax_seq[calls++ % 4];
    };
    Field U(2, State{1, 1, 1, 1});
    calls = 0;
    const StepResult r3 = SSPRK3(0.5).step(U, res, 1.0, 0.0, 10.0);
    CHECK(calls == 3);
    CHECK(r3.s_max == 3.0);                          // max over the stages
    CHECK_NEAR(r3.dt, 0.5 / (1.0 + 1e-10), 1e-12);   // dt from the first stage
    calls = 0;
    SSPRK2(0.5).step(U, res, 1.0, 0.0, 10.0);
    CHECK(calls == 2);
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

    // format_version: written as 1; a file without it (older) still loads; a newer one is refused with a clear message
    NpzArchive cur = read_npz(a.checkpoint_path());
    CHECK(cur.count("format_version") && cur.at("format_version").scalar() == 1.0);
    NpzArchive old = cur;
    old.erase("format_version");
    write_npz(a.checkpoint_path(), old);
    CHECK(b.load_checkpoint() == 2 && b.U[0][0] == 3.0);
    NpzArchive future = cur;
    future["format_version"] = NpyArray::from_int(99);
    write_npz(a.checkpoint_path(), future);
    std::string what;
    try {
        b.load_checkpoint();
    } catch (const std::runtime_error& e) { what = e.what(); }
    CHECK(what.find("format_version 99") != std::string::npos);
    what.clear();
    try {
        b.load_hot_start(a.checkpoint_path());
    } catch (const std::runtime_error& e) { what = e.what(); }
    CHECK(what.find("format_version 99") != std::string::npos);
}

TEST(npz_file_reads_chosen_rows_and_entries) {
    const fs::path dir = tmp_dir() / "npzfile";
    fs::create_directories(dir);
    const std::string path = (dir / "a.npz").string();
    std::vector<double> d(200 * 4);
    for (size_t k = 0; k < d.size(); ++k) d[k] = 0.5 * static_cast<double>(k) + 1.0;
    NpzArchive arch;
    arch["U"] = NpyArray::from_doubles(d, {200, 4});
    arch["t"] = NpyArray::from_double(2.5);
    arch["step"] = NpyArray::from_int(17);
    write_npz(path, arch);
    const NpzFile f(path);
    CHECK(f.has("U") && f.has("t") && !f.has("nothing"));
    CHECK(f.read("t").scalar() == 2.5 && f.read("step").scalar() == 17.0);
    CHECK(f.length("U") == 200);
    const std::vector<int64_t> rows = {7, 8, 9, 3, 150, 151, 199, 0};   // runs and jumps, not sorted
    int ncol = 0;
    const std::vector<double> got = f.read_rows("U", rows, ncol);
    CHECK(ncol == 4 && got.size() == rows.size() * 4);
    bool same = true;
    for (size_t k = 0; k < rows.size(); ++k)
        for (int q = 0; q < 4; ++q) same = same && got[k * 4 + q] == d[static_cast<size_t>(rows[k]) * 4 + q];
    CHECK(same);
    CHECK(f.read("U").as_doubles() == d);
    bool threw = false;
    try {
        f.read_rows("U", {200}, ncol);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
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

TEST(leaves_to_uniform_fills_the_blocks_of_the_finest_level) {
    // 2 x 2 base mesh; the cell (0, 0) is split in four level-1 leaves, one of which is split again (level 2)
    std::vector<int> lev, ci, cj;
    Field U;
    const auto add = [&](int l, int i, int j, double rho) {
        lev.push_back(l);
        ci.push_back(i);
        cj.push_back(j);
        U.push_back(State{rho, 0, 0, 1, 0});
    };
    add(0, 1, 0, 10);
    add(0, 0, 1, 20);
    add(0, 1, 1, 30);
    add(1, 1, 0, 2);
    add(1, 0, 1, 3);
    add(1, 1, 1, 4);
    add(2, 0, 0, 5);
    add(2, 1, 0, 6);
    add(2, 0, 1, 7);
    add(2, 1, 1, 8);
    int nx = 0, ny = 0;
    const Field out = leaves_to_uniform(U, lev, ci, cj, 2, 2, nx, ny);
    CHECK(nx == 8 && ny == 8 && out.size() == 64);
    const auto at = [&](int i, int j) { return out[static_cast<size_t>(j) * nx + i][0]; };
    CHECK(at(0, 0) == 5 && at(1, 0) == 6 && at(0, 1) == 7 && at(1, 1) == 8);   // level 2 leaves: one finest cell each
    CHECK(at(2, 0) == 2 && at(3, 1) == 2);                                       // level 1 leaf (1,0): 2 x 2 finest cells
    CHECK(at(0, 2) == 3 && at(1, 3) == 3 && at(2, 2) == 4 && at(3, 3) == 4);
    CHECK(at(4, 0) == 10 && at(7, 3) == 10 && at(0, 4) == 20 && at(7, 7) == 30);   // level 0 leaves: 4 x 4 finest cells
    // a missing leaf is refused
    lev.pop_back();
    ci.pop_back();
    cj.pop_back();
    U.pop_back();
    bool threw = false;
    try {
        leaves_to_uniform(U, lev, ci, cj, 2, 2, nx, ny);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
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
        CHECK(std::string(s.audience) == "user" || std::string(s.audience) == "developer");
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

TEST(moving_wall_ghost_has_the_pressure_of_the_interior_at_second_order) {
    // The ghost of a moving wall flips and shifts the momentum; with boundary_order = 2 its total energy follows, so that the ghost has
    // the pressure of the interior cell (copying E would put a jump of the order of the kinetic-energy difference across the wall).
    // With boundary_order = 1 the energy is copied. Every side, an interior state with all three velocity components (the w term of the energy is zero in 2-D, so only a 3-D state tests it).
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 4, 4);
    const State in = eos.prim_to_cons({1.3, 0.7, -0.4, 0.9, 0.5});
    const double p_in = eos.cons_to_prim(in)[3];
    for (const int order : {1, 2})
        for (const Face side : {Face::Left, Face::Right, Face::Bottom, Face::Top}) {
            BoundaryManager bm(m, eos);
            bm.set_boundary_types(BCType::Wall, BCType::Wall, BCType::Wall, BCType::Wall);
            bm.set_type(side, BCType::MovingWall);
            BCParams bp;
            bp.u_in = 1.5;
            bp.v_in = -0.8;
            bp.boundary_order = order;
            bm.set_params(bp);
            const State gv = bm.viscous_ghost(side, in), gs = bm.ghost_state(side, in);
            CHECK(gv[0] == in[0] && gs[0] == in[0]);
            CHECK_NEAR(eos.cons_to_prim(gs)[3], p_in, 1e-12);   // the first-order ghost state is built from the primitives
            if (order == 2) {
                CHECK_NEAR(eos.cons_to_prim(gv)[3], p_in, 1e-12);
                CHECK(gv[3] != in[3]);   // the energy did change: the ghost moves
            } else {
                CHECK(gv[3] == in[3]);   // the viscous ghost of the first-order boundary copies E
            }
        }
}

TEST(npz_writer_pieces_give_the_file_of_write_npz_and_zip64_reads_back) {
    const auto bytes = [](const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    };
    const size_t n = 1000;
    std::vector<double> u(4 * n), lev(n);
    for (size_t k = 0; k < u.size(); ++k) u[k] = std::sin(0.37 * static_cast<double>(k)) * 1e3;
    for (size_t k = 0; k < n; ++k) lev[k] = static_cast<double>(k % 3);
    NpzArchive a;
    a["U"] = NpyArray::from_doubles(u, {static_cast<int64_t>(n), 4});
    a["leaf_level"] = NpyArray::from_doubles(lev, {static_cast<int64_t>(n)});
    a["step"] = NpyArray::from_int(7);
    a["t"] = NpyArray::from_double(0.25);
    a["files"] = NpyArray::from_strings({"a_00001.vtu", "b"});
    const std::string one = (tmp_dir() / "writer_one.npz").string(), parts = (tmp_dir() / "writer_parts.npz").string();
    write_npz(one, a);
    // the same entries in key order, the two arrays reserved and filled in pieces of uneven size, out of order
    const auto in_parts = [&](const std::string& path, uint64_t zip64_from) {
        NpzWriter w(path, zip64_from);
        int hu = -1, hl = -1;
        for (const auto& kv : a) {
            if (kv.first == "U") hu = w.reserve("U", {static_cast<int64_t>(n), 4});
            else if (kv.first == "leaf_level") hl = w.reserve("leaf_level", {static_cast<int64_t>(n)});
            else w.add(kv.first, kv.second);
        }
        const size_t cut[5] = {0, 4 * 13, 4 * 500, 4 * 501, 4 * n};
        for (const int q : {2, 0, 3, 1}) w.put(hu, cut[q], u.data() + cut[q], cut[q + 1] - cut[q]);
        w.put(hl, 700, lev.data() + 700, n - 700);
        w.put(hl, 0, lev.data(), 700);
        w.put(hl, 0, lev.data(), 0);   // an empty piece (a rank without leaves of a level) is nothing
        w.close();
    };
    in_parts(parts, 0xFFFFFFFFull);
    CHECK(bytes(one) == bytes(parts) && !bytes(one).empty());
    // zip64 form (forced from 100 bytes on): the two readers give the arrays back
    const std::string big = (tmp_dir() / "writer_zip64.npz").string();
    in_parts(big, 100);
    CHECK(bytes(big) != bytes(one));
    const NpzArchive b = read_npz(big);
    CHECK(b.size() == a.size());
    for (const auto& kv : a) CHECK(b.count(kv.first) == 1 && b.at(kv.first).data == kv.second.data && b.at(kv.first).shape == kv.second.shape);
    const NpzFile f(big);
    CHECK(f.read("U").data == a.at("U").data && f.length("leaf_level") == static_cast<int64_t>(n));
    int ncol = 0;
    const std::vector<double> rows = f.read_rows("U", {999, 0, 501}, ncol);
    CHECK(ncol == 4 && rows.size() == 12 && rows[0] == u[4 * 999] && rows[4] == u[0] && rows[11] == u[4 * 501 + 3]);
    // an entry with a missing piece, or with one given twice, is refused
    for (const bool twice : {false, true}) {
        bool refused = false;
        try {
            NpzWriter w((tmp_dir() / "writer_bad.npz").string());
            const int h = w.reserve("x", {10});
            w.put(h, 0, lev.data(), 6);
            if (twice) w.put(h, 4, lev.data(), 6);
            w.close();
        } catch (const std::logic_error&) { refused = true; }
        CHECK(refused);
    }
}

TEST(mesh_too_large_for_32_bit_indices_is_refused) {
    // 1000^3 cells have 3.003e9 faces: refused with a message instead of a wrapped index
    const auto refused = [](const char* nx, const char* ny, const char* nz) {
        try {
            params_for("Sod Shock Tube", {{"nx", nx}, {"ny", ny}, {"nz", nz}});
        } catch (const std::invalid_argument& e) { return std::string(e.what()).find("32-bit") != std::string::npos; }
        return false;
    };
    CHECK(refused("1000", "1000", "1000"));
    CHECK(refused("40000", "40000", "1"));                      // 2-D: 3.2e9 faces
    CHECK(refused("2000000000", "2000000000", "2000000000"));   // the product itself exceeds 64 bits
    CHECK(!refused("800", "800", "800"));                       // 1.5e9 faces fit
    CHECK(!refused("30000", "30000", "1"));
}

TEST(weno5_line_batch_equals_the_scalar_function_bit_for_bit) {
    // The vector kernel of the uniform WENO5-Z reconstruction (weno5_line_simd, AVX2 or NEON) gives the bits of weno5_value and
    // prim_to_cons, for the left and the right state of every face of a line, along a stride other than one, with smooth data, jumps, zeros, signed zeros,
    // tiny and huge values.
#ifdef FV_SIMD
    if (!fv::detail::simd_available()) return;
    unsigned long long seed = 0x2545F4914F6CDD1Dull;
    const auto rnd = [&]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(seed >> 11) * (1.0 / 9007199254740992.0);
    };
    const auto same = [](double a, double b) { return std::memcmp(&a, &b, 8) == 0 || (std::isnan(a) && std::isnan(b)); };
    const EOS eos = EOS::ideal(1.4);
    for (double eps : {1e-6, 1e-2}) {
        const MusclReconstruction w5(eos, Limiter::VanLeer, Reconstruction::Weno5, eps);
        for (size_t stride : {size_t(1), size_t(7)}) {
            const int n = 64;
            const size_t len = static_cast<size_t>(n) + 5 * stride + 8;
            std::vector<double> plane[5];
            for (int q = 0; q < 5; ++q) {
                plane[q].resize(len);
                for (size_t k = 0; k < len; ++k) {
                    const double r = rnd();
                    double x = (q == 3 ? 1.0 : 0.0) + (r - 0.5) * (r < 0.2 ? 5.0 : 0.05);
                    if (r > 0.9) x = 0.0;   // exact zeros: the regularisation alone keeps the weights finite
                    if (r > 0.95) x = -0.0;
                    if (q == 1 && r > 0.85 && r <= 0.9) x = 1.0e150 * (r - 0.8);
                    if (q == 2 && r > 0.85 && r <= 0.9) x = 1.0e-160 * (r - 0.8);
                    plane[q][k] = x;
                }
            }
            const double* pl[5] = {plane[0].data(), plane[1].data(), plane[2].data(), plane[3].data(), plane[4].data()};
            std::vector<State> UL(n), UR(n);
            fv::detail::weno5_line_simd(eos, eps, pl, 3, stride, n, UL.data(), UR.data());
            int different = 0;
            for (int k = 0; k < n; ++k) {
                State wl, wr;
                for (int q = 0; q < 5; ++q) {
                    const double* c = pl[q] + 3 + k;
                    wl[q] = w5.weno5_value(c[0], c[stride], c[2 * stride], c[3 * stride], c[4 * stride]);
                    wr[q] = w5.weno5_value(c[5 * stride], c[4 * stride], c[3 * stride], c[2 * stride], c[stride]);
                }
                const State el = eos.prim_to_cons(wl), er = eos.prim_to_cons(wr);
                for (int q = 0; q < 5; ++q)
                    if (!same(el[q], UL[k][q]) || !same(er[q], UR[k][q])) ++different;
            }
            CHECK(different == 0);
        }
    }
#endif
}

TEST(weno_face_states_equal_the_scalar_functions_bit_for_bit) {
    // The face states of the adaptive engine (MusclReconstruction::weno_face_states, vector kernel for runs of four faces) are the bits of
    // weno3_value / weno5_value and prim_to_cons per face, for cells read from anywhere in the primitive array (repeated cells included).
    unsigned long long seed = 0x9E3779B97F4A7C15ull;
    const auto rnd = [&]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(seed >> 11) * (1.0 / 9007199254740992.0);
    };
    const auto same = [](double a, double b) { return std::memcmp(&a, &b, 8) == 0 || (std::isnan(a) && std::isnan(b)); };
    const EOS eos = EOS::ideal(1.4);
    const int ncell = 200, n = 67;
    std::vector<State> W(ncell);
    for (auto& w : W) {
        const double r = rnd();
        w = {0.5 + rnd(), (rnd() - 0.5) * (r < 0.2 ? 5.0 : 0.05), r > 0.9 ? 0.0 : rnd() - 0.5, 0.4 + rnd(), 0.0};
    }
    for (const double eps : {1e-6, 1e-2}) {
        const MusclReconstruction mu(eos, Limiter::VanLeer, Reconstruction::Weno5, eps);
        for (const bool five : {false, true}) {
            const int ns = five ? 6 : 4;
            std::vector<int> st(static_cast<size_t>(ns) * n);
            for (int k = 0; k < n; ++k) {
                const int c0 = static_cast<int>(rnd() * (ncell - 8));
                for (int p = 0; p < ns; ++p) st[static_cast<size_t>(ns) * k + p] = rnd() < 0.1 ? c0 + (p > 0 ? p - 1 : 0) : c0 + p;
            }
            std::vector<State> UL(n), UR(n);
            mu.weno_face_states(five, W.data(), st.data(), n, UL.data(), UR.data());
            int different = 0;
            for (int k = 0; k < n; ++k) {
                const int* c = &st[static_cast<size_t>(ns) * k];
                State wl, wr;
                for (int q = 0; q < kNVar; ++q) {
                    if (five) {
                        wl[q] = mu.weno5_value(W[c[0]][q], W[c[1]][q], W[c[2]][q], W[c[3]][q], W[c[4]][q]);
                        wr[q] = mu.weno5_value(W[c[5]][q], W[c[4]][q], W[c[3]][q], W[c[2]][q], W[c[1]][q]);
                    } else {
                        wl[q] = mu.weno3_value(W[c[0]][q], W[c[1]][q], W[c[2]][q]);
                        wr[q] = mu.weno3_value(W[c[3]][q], W[c[2]][q], W[c[1]][q]);
                    }
                }
                const State el = eos.prim_to_cons(wl), er = eos.prim_to_cons(wr);
                for (int q = 0; q < kNVar; ++q)
                    if (!same(el[q], UL[k][q]) || !same(er[q], UR[k][q])) ++different;
            }
            CHECK(different == 0);
        }
    }
}

TEST(weno3_line_batch_equals_the_scalar_function_bit_for_bit) {
    // The vector kernel of the uniform WENO3-Z reconstruction (weno3_line_simd, AVX2 or NEON) gives the bits of weno3_value and
    // prim_to_cons, for the left and the right state of every face of a line, along a stride other than one, with smooth data, jumps, zeros, signed zeros,
    // tiny and huge values.
#ifdef FV_SIMD
    if (!fv::detail::simd_available()) return;
    unsigned long long seed = 0x2545F4914F6CDD1Dull;
    const auto rnd = [&]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(seed >> 11) * (1.0 / 9007199254740992.0);
    };
    const auto same = [](double a, double b) { return std::memcmp(&a, &b, 8) == 0 || (std::isnan(a) && std::isnan(b)); };
    const EOS eos = EOS::ideal(1.4);
    for (double eps : {1e-6, 1e-2}) {
        const MusclReconstruction w3(eos, Limiter::VanLeer, Reconstruction::Weno3, eps);
        for (size_t stride : {size_t(1), size_t(7)}) {
            const int n = 64;
            const size_t len = static_cast<size_t>(n) + 3 * stride + 8;
            std::vector<double> plane[5];
            for (int q = 0; q < 5; ++q) {
                plane[q].resize(len);
                for (size_t k = 0; k < len; ++k) {
                    const double r = rnd();
                    double x = (q == 3 ? 1.0 : 0.0) + (r - 0.5) * (r < 0.2 ? 5.0 : 0.05);
                    if (r > 0.9) x = 0.0;   // exact zeros: the regularisation alone keeps the weights finite
                    if (r > 0.95) x = -0.0;
                    if (q == 1 && r > 0.85 && r <= 0.9) x = 1.0e150 * (r - 0.8);
                    if (q == 2 && r > 0.85 && r <= 0.9) x = 1.0e-160 * (r - 0.8);
                    plane[q][k] = x;
                }
            }
            const double* pl[5] = {plane[0].data(), plane[1].data(), plane[2].data(), plane[3].data(), plane[4].data()};
            std::vector<State> UL(n), UR(n);
            fv::detail::weno3_line_simd(eos, eps, pl, 3, stride, n, UL.data(), UR.data());
            int different = 0;
            for (int k = 0; k < n; ++k) {
                State wl, wr;
                for (int q = 0; q < 5; ++q) {
                    const double* c = pl[q] + 3 + k;
                    wl[q] = w3.weno3_value(c[0], c[stride], c[2 * stride]);
                    wr[q] = w3.weno3_value(c[3 * stride], c[2 * stride], c[stride]);
                }
                const State el = eos.prim_to_cons(wl), er = eos.prim_to_cons(wr);
                for (int q = 0; q < 5; ++q)
                    if (!same(el[q], UL[k][q]) || !same(er[q], UR[k][q])) ++different;
            }
            CHECK(different == 0);
        }
    }
#endif
}

// ---- thread placement: one thread per physical core ----------------------------------------------------------------

TEST(thread_placement_takes_one_cpu_per_core) {
    const std::vector<fv::CoreInfo> smt = {{{0, 4}, 0}, {{1, 5}, 0}, {{2, 6}, 0}, {{3, 7}, 0}};
    const std::vector<int> all = {0, 1, 2, 3, 4, 5, 6, 7};
    CHECK(fv::choose_thread_cpus(smt, all, 4) == (std::vector<int>{0, 1, 2, 3}));
    CHECK(fv::choose_thread_cpus(smt, all, 2) == (std::vector<int>{0, 1}));
    CHECK(fv::choose_thread_cpus(smt, all, 5).empty());   // more threads than cores: left to the scheduler
    CHECK(fv::choose_thread_cpus(smt, all, 1).empty());
    CHECK(fv::choose_thread_cpus({}, all, 4).empty());
}

TEST(thread_placement_leaves_alone_what_it_cannot_improve) {
    const std::vector<fv::CoreInfo> no_smt = {{{0}, 0}, {{1}, 0}, {{2}, 0}, {{3}, 0}};
    CHECK(fv::choose_thread_cpus(no_smt, {0, 1, 2, 3}, 4).empty());
    // a process already restricted to one logical CPU per core (by the user or by mpiexec)
    const std::vector<fv::CoreInfo> smt = {{{0, 4}, 0}, {{1, 5}, 0}, {{2, 6}, 0}, {{3, 7}, 0}};
    CHECK(fv::choose_thread_cpus(smt, {0, 1, 2, 3}, 4).empty());
}

TEST(thread_placement_uses_only_the_allowed_cpus) {
    const std::vector<fv::CoreInfo> smt = {{{0, 4}, 0}, {{1, 5}, 0}, {{2, 6}, 0}, {{3, 7}, 0}};
    CHECK(fv::choose_thread_cpus(smt, {2, 3, 6, 7}, 2) == (std::vector<int>{2, 3}));
    CHECK(fv::choose_thread_cpus(smt, {2, 3, 6, 7}, 3).empty());   // only two cores allowed
}

TEST(thread_placement_fills_the_fast_cores_first) {
    // two fast cores with SMT (class 1), then two slow cores without (class 0), listed slow first
    const std::vector<fv::CoreInfo> hybrid = {{{4}, 0}, {{5}, 0}, {{0, 1}, 1}, {{2, 3}, 1}};
    const std::vector<int> all = {0, 1, 2, 3, 4, 5};
    CHECK(fv::choose_thread_cpus(hybrid, all, 2) == (std::vector<int>{0, 2}));
    CHECK(fv::choose_thread_cpus(hybrid, all, 3) == (std::vector<int>{0, 2, 4}));
    CHECK(fv::choose_thread_cpus(hybrid, all, 5).empty());
}

TEST(thread_placement_runs_and_keeps_the_threads_working) {
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    const int n = std::min(4, saved);
    omp_set_num_threads(n);
    const fv::Placement p = fv::place_threads(n);   // may pin or not, depending on the machine: either way a region still runs on n threads
    long sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < 1000; ++i) sum += i;
    CHECK(sum == 499500);
    (void)p;
    omp_set_num_threads(saved);
#endif
}

namespace {
// The Mach 10 cylinder of stress_shocks.py (100 x 150 cells, slip wall, inviscid) as a case file: its wake opens to near vacuum.
SimParams mach10_cylinder_params(const char* fallback) {
    std::ifstream in(fs::path(cases_directory()) / "ibm_supersonic_flow_over_cylinder" / "case.toml", std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto replace = [&](const std::string& a, const std::string& b) {
        const size_t at = text.find(a);
        if (at == std::string::npos) throw std::runtime_error("case text changed: " + a);
        text.replace(at, a.size(), b);
    };
    replace("u_in = 2.0", "u_in = 10.0");
    replace("u = 2.0,", "u = 10.0,");
    replace("nx = 800", "nx = 100");
    replace("ny = 1200", "ny = 150");
    replace("use_mpi = true", "use_mpi = false");
    const fs::path dir = tmp_dir() / "positivity_cylinder";
    fs::create_directories(dir);
    {
        std::ofstream out(dir / "case.toml", std::ios::binary);
        out << text;
    }
    return params_for((dir / "case.toml").string(), {{"reconstruction", "weno5"}, {"time_scheme", "SSP Runge-Kutta 3"}, {"cfl", "0.3"},
                                                      {"positivity_fallback", fallback}});
}

bool admissible(const Field& U, const EOS& eos) {
    for (const State& u : U) {
        if (!std::isfinite(u[0]) || !std::isfinite(u[3]) || !(u[0] > 0.0)) return false;
        const double ke = 0.5 * (u[1] * u[1] + u[2] * u[2]) / u[0];
        if (!(eos.pressure_law().from_internal_energy_density(u[3] - ke) > 0.0)) return false;
    }
    return true;
}
}   // namespace

TEST(positivity_fallback_keeps_the_wake_of_a_mach_10_cylinder_admissible) {
    // WENO5-Z with the ghost wall loses the positivity of the wake cells around step 28 (from positive face states, through the update)
    // and blows up by step 45. With positivity_fallback the stages that leave the admissible set are repeated with first-order faces
    // around the offending cells: the run goes on, and the repair is on the record. Without the key nothing changes.
    constexpr int kSteps = 70;
    const auto run = [&](const char* fallback, long long* repairs) {
        Simulation sim;
        sim.setup(mach10_cylinder_params(fallback), false);
        sim.initialize_field();
        const ResidualFn res = [&](const Field& U, Field& R) { return sim.residual(U, R); };
        double t = 0.0;
        const double md = std::min(sim.mesh.dx, sim.mesh.dy);
        for (int k = 0; k < kSteps; ++k) {
            t += sim.scheme->step(sim.U, res, md, t, 1e9).dt;
            if (sim.ibm) sim.ibm->apply_solid_reset(sim.U, sim.fluid);
        }
        if (repairs) *repairs = sim.positivity_repairs();
        return admissible(sim.U, sim.fluid);
    };
    long long repairs = 0;
    CHECK(!run("false", nullptr));   // the failure this guards against
    CHECK(run("true", &repairs));
    std::printf("    %lld repair passes in %d steps\n", repairs, kSteps);
    CHECK(repairs > 0);
}

TEST(a_spinning_sphere_carries_the_rigid_rotation_velocity_in_its_deep_cells) {
    // a sphere that translates (0.1, -0.2, 0.05) and spins about all three axes: the solid cells away from the wall hold rho (u_b + omega x r)
    Simulation sim;
    sim.setup(params_for("Sod Shock Tube", {{"Lx", "1.0"}, {"Ly", "1.0"}, {"Lz", "1.0"}, {"nx", "32"}, {"ny", "32"}, {"nz", "32"}, {"use_ibm", "true"},
                                            {"ibm_shape", "Sphere"}, {"ibm_radius", "0.35"}, {"ibm_cx", "0.5"}, {"ibm_cy", "0.5"}, {"ibm_cz", "0.5"},
                                            {"ibm_vx", "0.1"}, {"ibm_vy", "-0.2"}, {"ibm_vz", "0.05"}, {"ibm_omega", "0.9"}, {"ibm_omega_x", "0.4"},
                                            {"ibm_omega_y", "-0.7"}, {"use_mpi", "false"}}),
              false);
    const ImmersedBoundary::BodyPose pose = sim.prescribed_pose(0.0);
    CHECK(pose.omega == 0.9 && pose.omega_x == 0.4 && pose.omega_y == -0.7);
    sim.ibm->move_to(sim.mesh, pose);
    const double rho = 1.3;
    Field U(static_cast<size_t>(sim.mesh.num_cells()), sim.fluid.prim_to_cons({rho, 0.0, 0.0, 1.0, 0.0}));
    const Field before = U;
    sim.ibm->apply_solid_reset(U, sim.fluid);
    int deep = 0;
    double worst = 0.0;
    for (int c = 0; c < sim.mesh.num_cells(); ++c) {
        if (U[static_cast<size_t>(c)] == before[static_cast<size_t>(c)]) continue;
        const int i = c % sim.mesh.nx, j = (c / sim.mesh.nx) % sim.mesh.ny, k = c / (sim.mesh.nx * sim.mesh.ny);
        const double rx = sim.mesh.xc(i) - 0.5, ry = sim.mesh.yc(j) - 0.5, rz = sim.mesh.zc(k) - 0.5;
        if (rx * rx + ry * ry + rz * rz > 0.22 * 0.22) continue;   // the cells next to the wall are ghost cells
        const double u = 0.1 - 0.9 * ry + (-0.7) * rz, v = -0.2 + 0.9 * rx - 0.4 * rz, w = 0.05 + 0.4 * ry - (-0.7) * rx;
        const State W = sim.fluid.cons_to_prim(U[static_cast<size_t>(c)]);
        worst = std::max({worst, std::abs(W[1] - u), std::abs(W[2] - v), std::abs(W[4] - w)});
        ++deep;
    }
    CHECK(deep > 300);
    CHECK(worst < 1e-12);
    std::printf("    %d deep cells, worst velocity error %.2e\n", deep, worst);
}

TEST(prescribed_rotation_stops_at_ibm_omega_until_and_holds_the_angle) {
    const auto pose_at = [](const char* until, double t, const char* from = "0.0") {
        Simulation sim;
        sim.setup(params_for("Sod Shock Tube", {{"Lx", "2.4"}, {"Ly", "2.4"}, {"nx", "16"}, {"ny", "16"}, {"use_ibm", "true"}, {"ibm_shape", "Cylinder"},
                                                {"ibm_radius", "0.1"}, {"ibm_cx", "1.2"}, {"ibm_cy", "1.2"}, {"ibm_omega", "-0.5"},
                                                {"ibm_omega_until", until}, {"ibm_omega_from", from}, {"use_mpi", "false"}}),
                  false);
        return sim.prescribed_pose(t);
    };
    const ImmersedBoundary::BodyPose a = pose_at("2.0", 1.0), b = pose_at("2.0", 5.0), c = pose_at("0.0", 5.0);
    CHECK(a.angle == -0.5 && a.omega == -0.5);   // before the stop: as without it
    CHECK(b.angle == -1.0 && b.omega == 0.0);    // after: the angle held, no angular velocity
    CHECK(c.angle == -2.5 && c.omega == -0.5);   // ibm_omega_until = 0: the rotation never stops (the old expression, bit for bit)
    // ibm_omega_from: at rest first, then the rotation, then held
    const ImmersedBoundary::BodyPose d0 = pose_at("4.0", 0.5, "1.0"), d1 = pose_at("4.0", 3.0, "1.0"), d2 = pose_at("4.0", 9.0, "1.0"), e = pose_at("0.0", 3.0, "1.0");
    CHECK(d0.angle == 0.0 && d0.omega == 0.0);
    CHECK(d1.angle == -1.0 && d1.omega == -0.5);
    CHECK(d2.angle == -1.5 && d2.omega == 0.0);
    CHECK(e.angle == -1.0 && e.omega == -0.5);   // no stop: from alone delays the start
}
