// Periodic flows: translation invariance, Taylor-Green decay, isentropic vortex, 3-D, sources
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_common.hpp"

// ---------------------------------------------------------------------------
// Periodic flows: translation invariance, Taylor-Green decay, isentropic vortex
// ---------------------------------------------------------------------------

namespace {
constexpr double kPi = 3.14159265358979323846;

// Advance `U0` with the run's scheme until `t_end` (or `max_steps`); case BCs applied.
Field run_to(const SimParams& sp, Field U0, double t_end, int max_steps = 1 << 30) {
    Simulation sim;
    sim.setup(sp, false);
    sim.U = std::move(U0);
    const ResidualFn res = [&](const Field& U, Field& R) { return sim.residual(U, R); };
    const double md = std::min(sim.mesh.dx, sim.mesh.dy);
    double t = 0.0;
    for (int k = 0; k < max_steps && t < t_end; ++k) t += sim.scheme->step(sim.U, res, md, t, t_end).dt;
    return sim.U;
}

Field roll(const Field& U, int nx, int ny, int sx, int sy) {
    Field out(U.size());
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) out[((j + sy) % ny) * nx + (i + sx) % nx] = U[j * nx + i];
    return out;
}

double kinetic_energy(const Field& U) {
    double ke = 0.0;
    for (const State& c : U) ke += 0.5 * (c[1] * c[1] + c[2] * c[2]) / c[0];
    return ke;
}

// Relative L2 error of the velocity against the exact Taylor-Green field at time t.
double tgv_velocity_error(const Field& U, const Mesh& m, const EOS& eos, double t, double mu) {
    double num = 0.0, den = 0.0;
    for (int j = 0; j < m.ny; ++j)
        for (int i = 0; i < m.nx; ++i) {
            const State W = eos.cons_to_prim(U[j * m.nx + i]);
            const State E = taylor_green_exact(m.xc(i), m.yc(j), t, m.Lx, mu);
            num += (W[1] - E[1]) * (W[1] - E[1]) + (W[2] - E[2]) * (W[2] - E[2]);
            den += E[1] * E[1] + E[2] * E[2];
        }
    return std::sqrt(num / den);
}

// L1 density error of the isentropic vortex at time t on an n x n mesh.
double vortex_density_error(int n, double t_end) {
    const SimParams sp =
        params_for("Isentropic Vortex", {{"nx", std::to_string(n).c_str()}, {"ny", std::to_string(n).c_str()}});
    Simulation init;
    init.setup(sp, false);
    const Field U =
        run_to(sp, get_test_case("Isentropic Vortex").initialize(init.mesh, init.fluid, init.p.source), t_end);
    double err = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            err += std::abs(U[j * n + i][0] -
                            isentropic_vortex_exact(init.mesh.xc(i), init.mesh.yc(j), t_end, 10.0, 10.0, 1.4)[0]);
    return err / (n * n);
}
}   // namespace

TEST(periodic_translation_invariance) {
    // A periodic domain has no special place: shifting the initial field by whole cells
    // must shift the solution exactly, with MUSCL and viscous terms on or off.
    const int nx = 16, ny = 12, sx = 5, sy = 3;
    for (const char* muscl : {"false", "true"})
        for (const char* visc : {"false", "true"}) {
            const SimParams sp =
                params_for("Taylor-Green Vortex",
                           {{"nx", "16"}, {"ny", "12"}, {"use_muscl", muscl}, {"use_viscous", visc}, {"mu", "0.05"}});
            Simulation s;
            s.setup(sp, false);
            Field U(nx * ny);
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const double x = 2 * kPi * (i + 0.5) / nx, y = 2 * kPi * (j + 0.5) / ny;
                    U[j * nx + i] = s.fluid.prim_to_cons(
                        {1.0 + 0.1 * std::sin(x + 0.3) * std::cos(y), 0.3 * std::sin(y + 1.0),
                         0.2 * std::cos(x - 0.5) + 0.1 * std::sin(2 * y), 100.0 + 2.0 * std::cos(x + y)});
                }
            const Field A = run_to(sp, U, 1e9, 6);
            const Field B = run_to(sp, roll(U, nx, ny, sx, sy), 1e9, 6);
            const Field Ar = roll(A, nx, ny, sx, sy);
            double diff = 0.0;
            for (size_t c = 0; c < B.size(); ++c)
                for (int q = 0; q < kNVar; ++q)
                    diff = std::max(diff, std::abs(Ar[c][q] - B[c][q]) / (1.0 + std::abs(B[c][q])));
            std::printf("    muscl=%s viscous=%s: max shifted difference %.3e\n", muscl, visc, diff);
            CHECK(diff < 1e-13);
        }
}

TEST(taylor_green_viscous_decay) {
    // Kinetic energy decays as exp(-4 nu k^2 t); subtracting the decay of the same run
    // without viscous fluxes (numerical dissipation) must recover the physical nu.
    const double mu = 0.05, t_end = 1.0;
    const SimParams visc = params_for("Taylor-Green Vortex", {{"nx", "32"}, {"ny", "32"}, {"mu", "0.05"}});
    const SimParams invisc =
        params_for("Taylor-Green Vortex", {{"nx", "32"}, {"ny", "32"}, {"mu", "0.05"}, {"use_viscous", "false"}});
    Simulation s;
    s.setup(visc, false);
    const Field U0 = get_test_case("Taylor-Green Vortex").initialize(s.mesh, s.fluid, s.p.source);
    const Field Uv = run_to(visc, U0, t_end), Ui = run_to(invisc, U0, t_end);
    const double ke0 = kinetic_energy(U0);
    const double nu_total = -std::log(kinetic_energy(Uv) / ke0) / (4.0 * t_end);
    const double nu_num = -std::log(kinetic_energy(Ui) / ke0) / (4.0 * t_end);
    const double err = tgv_velocity_error(Uv, s.mesh, s.fluid, t_end, mu);
    std::printf("    nu = %.4f: measured total %.5f, numerical %.5f, physical %.5f; velocity L2 error %.3e\n", mu,
                nu_total, nu_num, nu_total - nu_num, err);
    CHECK(std::abs((nu_total - nu_num) - mu) < 0.02 * mu);   // measured 0.4 % at 32x32
    CHECK(err < 0.015);
}

TEST(taylor_green_convergence) {
    // Velocity error against the exact solution must fall at (close to) 2nd order
    const double mu = 0.05, t_end = 0.5;
    double e[3];
    const int ns[3] = {16, 32, 64};
    for (int k = 0; k < 3; ++k) {
        const std::string n = std::to_string(ns[k]);
        const SimParams sp = params_for("Taylor-Green Vortex", {{"nx", n.c_str()}, {"ny", n.c_str()}, {"mu", "0.05"}});
        Simulation s;
        s.setup(sp, false);
        e[k] = tgv_velocity_error(
            run_to(sp, get_test_case("Taylor-Green Vortex").initialize(s.mesh, s.fluid, s.p.source), t_end), s.mesh,
            s.fluid, t_end, mu);
    }
    const double o1 = std::log2(e[0] / e[1]), o2 = std::log2(e[1] / e[2]);
    std::printf("    errors %.3e %.3e %.3e, observed orders %.2f %.2f\n", e[0], e[1], e[2], o1, o2);
    CHECK(o2 > 1.7);
}

TEST(taylor_green_viscous_time_step_budget) {
    // With a large viscosity the convective and the viscous eigenvalues must share the stability budget of the
    // time step. Taking only the smaller of the two limits let CFL 0.5 go unstable here (velocity error 0.37
    // at t = 1 for nu = 0.2, the energy decaying too fast for nu = 0.1), the combined limit keeps it exact.
    for (const double mu : {0.1, 0.2}) {
        const std::string mus = std::to_string(mu);
        const SimParams sp = params_for("Taylor-Green Vortex", {{"nx", "64"}, {"ny", "64"}, {"mu", mus.c_str()}});
        Simulation s;
        s.setup(sp, false);
        const double err = tgv_velocity_error(
            run_to(sp, get_test_case("Taylor-Green Vortex").initialize(s.mesh, s.fluid, s.p.source), 2.0), s.mesh,
            s.fluid, 2.0, mu);
        std::printf("    nu = %.1f: velocity error %.2e at t = 2\n", mu, err);
        CHECK(err < 5e-3);
    }
}

TEST(source_terms_enter_the_residual) {
    // A uniform state has no flux divergence, so one Explicit Euler step moves every cell by dt S(x, y): this
    // isolates the parsing of the source expressions, their slot in the state and the cell-centre positions
    const SimParams sp = params_for("Taylor-Green Vortex",
                                    {{"nx", "16"}, {"ny", "8"}, {"time_scheme", "Explicit Euler"},
                                     {"source_rho", "0.25"}, {"source_mx", "sin(2*pi*x/Lx)"},
                                     {"source_my", "0.5*cos(2*pi*y/Ly)"}, {"source_E", "x*y + mu"}});
    CHECK(sp.use_source);
    Simulation sim;
    sim.setup(sp, false);
    const Field U0 = uniform_field(sim.fluid, sim.mesh.num_cells(), 1.0, 0.0, 0.0, 1.0);
    sim.U = U0;
    const ResidualFn res = [&](const Field& U, Field& R) { return sim.residual(U, R); };
    const double dt = sim.scheme->step(sim.U, res, std::min(sim.mesh.dx, sim.mesh.dy), 0.0, 1e9).dt;
    const Mesh& m = sim.mesh;
    double err = 0.0;
    for (int j = 0; j < m.ny; ++j)
        for (int i = 0; i < m.nx; ++i) {
            const double x = m.xc(i), y = m.yc(j);
            const State expect = {0.25, std::sin(2 * kPi * x / m.Lx), 0.5 * std::cos(2 * kPi * y / m.Ly), x * y + sp.mu,
                                  0.0};
            const State& a = sim.U[m.cell(i, j)];
            const State& b = U0[m.cell(i, j)];
            for (int q = 0; q < kNVar; ++q) err = std::max(err, std::abs((a[q] - b[q]) - dt * expect[q]));
        }
    std::printf("    max |dU - dt S| = %.2e (dt = %.3e)\n", err, dt);
    CHECK(err < 1e-14);
}

TEST(source_terms_3d_slots) {
    // Only source_mz: only the w-momentum slot of the 3-D state moves, with the z dependence of the expression
    const SimParams sp = params_for("Taylor-Green Vortex 3D",
                                    {{"nx", "8"}, {"ny", "8"}, {"nz", "8"}, {"time_scheme", "Explicit Euler"},
                                     {"source_mz", "cos(2*pi*z/Lz)*(1+0.5*sin(x))"}});
    Simulation sim;
    sim.setup(sp, false);
    const Mesh& m = sim.mesh;
    const Field U0(static_cast<size_t>(m.num_cells()), sim.fluid.prim_to_cons({1.0, 0.0, 0.0, 1.0, 0.0}));
    sim.U = U0;
    const ResidualFn res = [&](const Field& U, Field& R) { return sim.residual(U, R); };
    const double dt = sim.scheme->step(sim.U, res, sim.dt_length, 0.0, 1e9).dt;
    double err = 0.0, other = 0.0;
    for (int k = 0; k < m.nz; ++k)
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                const State& a = sim.U[m.cell(i, j, k)];
                const State& b = U0[m.cell(i, j, k)];
                const double s = std::cos(2 * kPi * m.zc(k) / m.Lz) * (1 + 0.5 * std::sin(m.xc(i)));
                err = std::max(err, std::abs((a[4] - b[4]) - dt * s));
                for (int q = 0; q < 4; ++q) other = std::max(other, std::abs(a[q] - b[q]));
            }
    std::printf("    w-momentum error %.2e, other slots %.2e\n", err, other);
    CHECK(err < 1e-14);
    CHECK(other == 0.0);
}

namespace {
// Density after one step of the scheme on a uniform periodic state with only source_rho = f(t), started at t0.
// A uniform state has no flux divergence, so the step integrates drho/dt = f(t) with the scheme's own stage times.
double density_after_step(const char* scheme, const char* f, double t0, double* dt_out, int steps = 1) {
    const SimParams sp = params_for("Taylor-Green Vortex",
                                    {{"nx", "8"}, {"ny", "8"}, {"time_scheme", scheme}, {"source_rho", f},
                                     {"dt_phys", "0.05"}, {"max_inner_iter", "100000"}, {"inner_tol_rel", "1e-14"}});
    Simulation sim;
    sim.setup(sp, false);
    sim.U = uniform_field(sim.fluid, sim.mesh.num_cells(), 1.0, 0.0, 0.0, 1.0);
    const ResidualFn res = [&](const Field& U, Field& R) { return sim.residual(U, R); };
    double t = t0;
    for (int n = 0; n < steps; ++n) {
        const StepResult r = sim.scheme->step(sim.U, res, std::min(sim.mesh.dx, sim.mesh.dy), t, 1e9);
        t += r.dt;
        *dt_out = r.dt;
    }
    return sim.U[0][0];
}
double f_poly(double t) { return 1.0 + 2.0 * t + 3.0 * t * t + 4.0 * t * t * t; }
}   // namespace

TEST(time_dependent_source_stage_times) {
    // Each scheme must evaluate the source at ITS stage times, started from an arbitrary absolute time t0 (resume):
    // Euler at t0; SSP-RK2 (Heun) at t0 and t0 + dt, the trapezoid rule; RK4 at t0, twice t0 + dt/2, t0 + dt, which
    // is Simpson's rule. This is independent of the spatial scheme and of the limiter.
    const char* f = "1 + 2*t + 3*t*t + 4*t*t*t";
    const double t0 = 0.37;
    double dt = 0.0;
    double rho = density_after_step("Explicit Euler", f, t0, &dt);
    CHECK_NEAR(rho - 1.0, dt * f_poly(t0), 1e-14);
    rho = density_after_step("SSP Runge-Kutta 2", f, t0, &dt);
    CHECK_NEAR(rho - 1.0, 0.5 * dt * (f_poly(t0) + f_poly(t0 + dt)), 1e-14);
    rho = density_after_step("Runge-Kutta 4", f, t0, &dt);
    CHECK_NEAR(rho - 1.0, dt / 6.0 * (f_poly(t0) + 4.0 * f_poly(t0 + 0.5 * dt) + f_poly(t0 + dt)), 1e-14);
    // Simpson integrates a cubic exactly: the step is the integral of f over [t0, t0 + dt]
    const auto F = [](double t) { return t + t * t + t * t * t + t * t * t * t; };
    CHECK_NEAR(rho - 1.0, F(t0 + dt) - F(t0), 1e-14);
}

TEST(time_dependent_source_bdf2) {
    // BDF2 evaluates the source at the new time in every inner iteration: with a tight inner tolerance the first step
    // is BDF1, rho1 = rho0 + dt f(t0 + dt), and the second is (3 rho2 - 4 rho1 + rho0) / (2 dt) = f(t0 + 2 dt)
    const char* f = "1 + 2*t + 3*t*t + 4*t*t*t";
    const double t0 = 0.2;
    double dt = 0.0;
    const double rho1 = density_after_step("BDF2 Dual Time", f, t0, &dt, 1);
    CHECK_NEAR(rho1 - 1.0, dt * f_poly(t0 + dt), 1e-9);
    const double rho2 = density_after_step("BDF2 Dual Time", f, t0, &dt, 2);
    CHECK_NEAR(rho2, (4.0 * rho1 - 1.0 + 2.0 * dt * f_poly(t0 + 2.0 * dt)) / 3.0, 1e-9);
}

TEST(source_time_dependence_is_detected_by_identifier) {
    // `t` as a whole identifier only: sqrt, tan, t_end and the exponent of 1e-3 do not count
    const auto unsteady = [](const char* e) {
        return params_for("Taylor-Green Vortex", {{"source_rho", e}}).source_unsteady;
    };
    CHECK(!unsteady("0"));
    CHECK(!unsteady("sqrt(2)*x + tan(y) + 1e-3 + 2.5e+2*x"));
    CHECK(!unsteady("sin(x)*cos(y) + 5e-1"));
    CHECK(unsteady("sin(t)"));
    CHECK(unsteady("x*t"));
    CHECK(unsteady("1e-3*t"));
    CHECK(unsteady("pow(x,2)+pow(t,2)"));
    CHECK(unsteady("(t)"));
}

TEST(source_terms_are_rejected_where_unsupported) {
    bool ibm_throws = false;
    CHECK(params_for("Sod Shock Tube", {{"use_amr", "true"}, {"source_rho", "1"}}).use_source);   // sources work on adaptive meshes
    try {
        params_for("IBM Flow over Cylinder", {{"source_mx", "x"}});
    } catch (const std::invalid_argument&) {
        ibm_throws = true;
    }
    CHECK(ibm_throws);
    CHECK(!params_for("Sod Shock Tube", {}).use_source);
    CHECK(!params_for("Sod Shock Tube", {{"source_rho", "0.0"}}).use_source);   // a zero number is no source
}

TEST(isentropic_vortex_convergence) {
    // Smooth inviscid advection: MUSCL + SSP-RK2 must converge at (close to) 2nd order
    const double e20 = vortex_density_error(20, 1.0), e40 = vortex_density_error(40, 1.0),
                 e80 = vortex_density_error(80, 1.0);
    const double o1 = std::log2(e20 / e40), o2 = std::log2(e40 / e80);
    std::printf("    L1 density errors %.3e %.3e %.3e, observed orders %.2f %.2f\n", e20, e40, e80, o1, o2);
    CHECK(o2 > 1.7);
}

// ---------------------------------------------------------------------------
// 3-D: consistency with 2-D, symmetry, Taylor-Green dissipation
// ---------------------------------------------------------------------------

namespace {
// Full run setup (initial field, IBM forcing, case BCs) advanced by `steps` steps.
// `dt_length` > 0 overrides the CFL length scale (to step a 3-D run exactly like a 2-D one).
Simulation run_case(const SimParams& sp, int steps, double dt_length = 0.0) {
    Simulation sim;
    sim.setup(sp, false);
    if (dt_length > 0.0) sim.dt_length = dt_length;
    sim.initialize_field();
    const ResidualFn res = [&](const Field& U, Field& R) { return sim.residual(U, R); };
    const double md = sim.dt_length;
    double t = 0.0;
    for (int k = 0; k < steps && t < sp.t_end; ++k) {
        t += sim.scheme->step(sim.U, res, md, t, sp.t_end).dt;
        if (sp.use_ibm) sim.ibm->apply_solid_reset(sim.U, sim.fluid);
    }
    return sim;
}

double rel_diff(const State& a, const State& b) {
    double d = 0.0;
    for (int q = 0; q < kNVar; ++q) d = std::max(d, std::abs(a[q] - b[q]) / (1.0 + std::abs(b[q])));
    return d;
}

// Advance an explicit 3-D field (periodic box, case defaults + overrides) by `steps` steps.
// `dt_length` > 0 overrides the CFL length scale.
Field run_field_3d(const SimParams& sp, Field U0, int steps, double dt_length = 0.0) {
    Simulation sim;
    sim.setup(sp, false);
    if (dt_length > 0.0) sim.dt_length = dt_length;
    sim.U = std::move(U0);
    const ResidualFn res = [&](const Field& U, Field& R) { return sim.residual(U, R); };
    double t = 0.0;
    for (int k = 0; k < steps; ++k) t += sim.scheme->step(sim.U, res, sim.dt_length, t, 1e9).dt;
    return sim.U;
}

// Smooth, fully 3-D periodic test field on [0, 2 pi]^3 (all velocity components non-zero).
State smooth_3d(const EOS& eos, double x, double y, double z) {
    return eos.prim_to_cons({1.0 + 0.1 * std::sin(x + 0.3) * std::cos(y - 0.2) * std::cos(z + 0.5),
                             0.4 * std::sin(y + 1.0) * std::cos(z), 0.3 * std::cos(x - 0.5) * std::sin(z + 0.7),
                             100.0 + 2.0 * std::cos(x + y - z), 0.2 * std::sin(x) * std::cos(y + 0.4)});
}
}   // namespace

TEST(extrusion_matches_2d) {
    // A 2-D case extruded along a periodic z must reproduce the 2-D solution in every layer
    struct Case {
        const char* name;
        std::vector<std::pair<const char*, const char*>> ov;
        int steps;
    };
    const std::vector<Case> cases = {
        {"Sod Shock Tube", {{"nx", "40"}, {"ny", "4"}, {"use_muscl", "true"}, {"time_scheme", "ssprk2"}}, 15},
        {"Lid Driven Cavity", {{"nx", "12"}, {"ny", "12"}, {"use_muscl", "true"}}, 20},
        {"Poiseuille Flow", {{"nx", "16"}, {"ny", "8"}}, 30},
        {"IBM Flow over Cylinder", {{"nx", "40"}, {"ny", "20"}, {"use_mpi", "false"}}, 10},
        {"Isentropic Vortex", {{"nx", "20"}, {"ny", "20"}}, 10},
    };
    for (const Case& c : cases) {
        Params p2 = baseline_defaults();
        p2.merge(get_test_case(c.name).defaults);
        p2.set("test_case", c.name);
        p2.set("output_root", tmp_dir().string());
        for (const auto& kv : c.ov) p2.set(kv.first, kv.second);
        Params p3 = p2;
        p3.set("nz", "3");
        p3.set("Lz", "3.0e5");   // deep cells: dz must neither set the CFL step nor add to the viscous one
        const Simulation a = run_case(SimParams::from(p2), c.steps);
        const Simulation b = run_case(SimParams::from(p3), c.steps, a.dt_length);
        double diff = 0.0, wmax = 0.0;
        for (int k = 0; k < 3; ++k)
            for (int c2 = 0; c2 < a.mesh.num_cells(); ++c2) {
                const State& s3 = b.U[k * a.mesh.num_cells() + c2];
                State s3_2d = s3;
                s3_2d[4] = 0.0;
                diff = std::max(diff, rel_diff(s3_2d, a.U[c2]));
                wmax = std::max(wmax, std::abs(s3[4]));
            }
        std::printf("    %-24s extruded vs 2-D: max rel diff %.2e, max |rho w| %.2e\n", c.name, diff, wmax);
        CHECK(diff < 1e-10);
        CHECK(wmax < 1e-12);
    }
}

TEST(taylor_green_rotated_planes) {
    // The 2-D Taylor-Green vortex placed in the x-z and y-z planes of a 3-D box must
    // reproduce the 2-D (x-y) run with the velocity components rotated accordingly.
    const int n = 16;
    const SimParams p2 = params_for("Taylor-Green Vortex", {{"nx", "16"}, {"ny", "16"}, {"mu", "0.05"}});
    Simulation s2;
    s2.setup(p2, false);
    const Field A =
        run_to(p2, get_test_case("Taylor-Green Vortex").initialize(s2.mesh, s2.fluid, s2.p.source), 1e9, 12);
    for (int plane = 0; plane < 2; ++plane) {   // 0: x-z plane, 1: y-z plane
        const char* nx = plane == 0 ? "16" : "3";
        const char* ny = plane == 0 ? "3" : "16";
        // The 3-cell direction is made very long: its 1/h^2 would otherwise add to the viscous time-step limit
        const char* lx = plane == 0 ? "6.283185307179586" : "6.283185307179586e5";
        const char* ly = plane == 0 ? "6.283185307179586e5" : "6.283185307179586";
        const SimParams p3 = params_for("Taylor-Green Vortex 3D",
                                        {{"nx", nx},
                                         {"ny", ny},
                                         {"nz", "16"},
                                         {"Lx", lx},
                                         {"Ly", ly},
                                         {"mu", "0.05"},
                                         {"riemann_solver", "HLLC (Low-Mach Thornber)"}});
        Simulation s3;
        s3.setup(p3, false);
        const Mesh& m = s3.mesh;
        Field U0(m.num_cells());
        for (int k = 0; k < m.nz; ++k)
            for (int j = 0; j < m.ny; ++j)
                for (int i = 0; i < m.nx; ++i) {
                    const int a = plane == 0 ? i : j;   // in-plane first coordinate
                    const State W =
                        taylor_green_exact(m.xc(0) * 0 + (a + 0.5) * 2 * kPi / n, m.zc(k), 0.0, 2 * kPi, 0.05);
                    // 2-D (u, v) -> 3-D (u, w) for x-z, (v, w) for y-z
                    const State W3 =
                        plane == 0 ? State{W[0], W[1], 0.0, W[3], W[2]} : State{W[0], 0.0, W[1], W[3], W[2]};
                    U0[m.cell(i, j, k)] = s3.fluid.prim_to_cons(W3);
                }
        const Field B = run_field_3d(p3, U0, 12, std::min(s2.mesh.dx, s2.mesh.dy));   // same steps as the 2-D run
        double diff = 0.0;
        for (int k = 0; k < n; ++k)
            for (int a = 0; a < n; ++a) {
                const State& u2 = A[k * n + a];
                const State expect =
                    plane == 0 ? State{u2[0], u2[1], 0.0, u2[3], u2[2]} : State{u2[0], 0.0, u2[1], u2[3], u2[2]};
                for (int b = 0; b < 3; ++b)
                    diff = std::max(diff, rel_diff(B[plane == 0 ? m.cell(a, b, k) : m.cell(b, a, k)], expect));
            }
        std::printf("    %s plane vs 2-D x-y run: max rel diff %.2e\n", plane == 0 ? "x-z" : "y-z", diff);
        CHECK(diff < 1e-10);
    }
}

TEST(axis_permutation_3d) {
    // Cyclic relabelling x -> y -> z of a fully 3-D field must relabel the solution
    const int n = 10;
    const std::string ns = std::to_string(n);
    const SimParams sp = params_for("Taylor-Green Vortex 3D",
                                    {{"nx", ns.c_str()}, {"ny", ns.c_str()}, {"nz", ns.c_str()}, {"mu", "0.05"}});
    Simulation s;
    s.setup(sp, false);
    const Mesh& m = s.mesh;
    Field A0(m.num_cells()), B0(m.num_cells());
    // B(x', y', z') = A(x = z', y = x', z = y') with velocities (u', v', w') = (v, w, u)
    auto rot = [](const State& a) { return State{a[0], a[2], a[4], a[3], a[1]}; };
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) A0[m.cell(i, j, k)] = smooth_3d(s.fluid, m.xc(i), m.yc(j), m.zc(k));
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) B0[m.cell(i, j, k)] = rot(A0[m.cell(k, i, j)]);
    const Field A = run_field_3d(sp, A0, 8), B = run_field_3d(sp, B0, 8);
    double diff = 0.0;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) diff = std::max(diff, rel_diff(B[m.cell(i, j, k)], rot(A[m.cell(k, i, j)])));
    std::printf("    permuted vs original: max rel diff %.2e\n", diff);
    CHECK(diff < 1e-11);
}

TEST(periodic_translation_invariance_3d) {
    const int nx = 12, ny = 10, nz = 8, sx = 5, sy = 3, sz = 2;
    const SimParams sp =
        params_for("Taylor-Green Vortex 3D", {{"nx", "12"}, {"ny", "10"}, {"nz", "8"}, {"mu", "0.05"}});
    Simulation s;
    s.setup(sp, false);
    const Mesh& m = s.mesh;
    Field U(m.num_cells()), Us(m.num_cells());
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) U[m.cell(i, j, k)] = smooth_3d(s.fluid, m.xc(i), m.yc(j), m.zc(k));
    auto shift = [&](const Field& F) {
        Field out(F.size());
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    out[m.cell((i + sx) % nx, (j + sy) % ny, (k + sz) % nz)] = F[m.cell(i, j, k)];
        return out;
    };
    const Field A = shift(run_field_3d(sp, U, 6)), B = run_field_3d(sp, shift(U), 6);
    double diff = 0.0;
    for (size_t c = 0; c < A.size(); ++c) diff = std::max(diff, rel_diff(A[c], B[c]));
    std::printf("    max shifted difference %.3e\n", diff);
    CHECK(diff < 1e-13);
}

TEST(uniform_flow_is_preserved_3d) {
    for (const char* rs : {"HLLC", "HLLC (Low-Mach Thornber)", "HLLC (Preconditioned)"})
        for (const char* visc : {"false", "true"}) {
            Simulation sim;
            sim.setup(params_for("Taylor-Green Vortex 3D", {{"nx", "6"},
                                                            {"ny", "5"},
                                                            {"nz", "4"},
                                                            {"riemann_solver", rs},
                                                            {"use_viscous", visc},
                                                            {"bc_left", "transmissive"},
                                                            {"bc_right", "transmissive"},
                                                            {"bc_bottom", "transmissive"},
                                                            {"bc_top", "transmissive"},
                                                            {"bc_front", "transmissive"},
                                                            {"bc_back", "transmissive"}}),
                      false);
            const Field U(sim.mesh.num_cells(), sim.fluid.prim_to_cons({1.3, 0.4, -0.2, 2.0, 0.3}));
            Field R;
            sim.residual(U, R);
            CHECK(max_abs(R) < 1e-11);
        }
}

TEST(taylor_green_3d_default_is_stable) {
    // Regression for the 3-D CFL length: with min(dx, dy, dz) the default 32^3 case (CFL 0.5, three
    // summed Courant numbers) blew up after ~30 steps; the kinetic energy must only decay.
    const SimParams sp = params_for("Taylor-Green Vortex 3D", {{"t_end", "1e9"}});
    Simulation sim;
    sim.setup(sp, false);
    sim.initialize_field();
    const ResidualFn res = [&](const Field& U, Field& R) { return sim.residual(U, R); };
    auto ke = [&] {
        double e = 0.0;
        for (const State& c : sim.U) e += 0.5 * (c[1] * c[1] + c[2] * c[2] + c[4] * c[4]) / c[0];
        return e / sim.U.size();
    };
    double prev = ke(), t = 0.0;
    bool monotone = true;
    for (int k = 0; k < 80; ++k) {
        t += sim.scheme->step(sim.U, res, sim.dt_length, t, 1e9).dt;
        const double e = ke();
        monotone = monotone && e <= prev * (1.0 + 1e-12);
        prev = e;
    }
    std::printf("    80 steps to t = %.3f: KE %.6f (monotone decay: %s)\n", t, prev, monotone ? "yes" : "no");
    CHECK(monotone);
    CHECK(prev < 0.125 && prev > 0.11);
}

TEST(taylor_green_3d_initial_dissipation) {
    // At t = 0 the 3-D Taylor-Green kinetic energy decays at eps = nu <|omega|^2> = 0.75 nu
    const double mu = 0.01, t_end = 0.1;
    const SimParams visc = params_for("Taylor-Green Vortex 3D",
                                      {{"nx", "24"}, {"ny", "24"}, {"nz", "24"}, {"mu", "0.01"}, {"t_end", "0.1"}});
    const SimParams inv = params_for(
        "Taylor-Green Vortex 3D",
        {{"nx", "24"}, {"ny", "24"}, {"nz", "24"}, {"mu", "0.01"}, {"t_end", "0.1"}, {"use_viscous", "false"}});
    auto ke_mean = [](const Field& U) {
        double ke = 0.0;
        for (const State& c : U) ke += 0.5 * (c[1] * c[1] + c[2] * c[2] + c[4] * c[4]) / c[0];
        return ke / U.size();
    };
    const Simulation a = run_case(visc, 1 << 30), b = run_case(inv, 1 << 30);
    const double eps = (ke_mean(b.U) - ke_mean(a.U)) / t_end;
    std::printf("    physical dissipation %.5e vs exact 0.75 nu = %.5e (ratio %.4f); KE0 %.4f\n", eps, 0.75 * mu,
                eps / (0.75 * mu), ke_mean(a.initial_field()));
    CHECK(std::abs(eps / (0.75 * mu) - 1.0) < 0.05);
    CHECK(std::abs(ke_mean(a.initial_field()) - 0.125) < 2e-3);
}

