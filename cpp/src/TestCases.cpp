#include "fv/TestCases.hpp"

#include <cstdio>
#include <stdexcept>

namespace fv {

namespace {

Params make(std::initializer_list<std::pair<const char*, const char*>> kv) {
    Params p;
    for (const auto& e : kv) p.set(e.first, e.second);
    return p;
}

Field uniform(const Mesh& m, double rho, double u, double v, double p, double gamma) {
    // rho*E = p/(gamma-1) + 0.5*rho*(u^2+v^2)  (set directly, not via the EOS)
    const State U = {rho, rho * u, rho * v, p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)};
    return Field(static_cast<size_t>(m.num_cells()), U);
}

// --- Sod shock tube (Sod 1978) ---
TestCase sod() {
    TestCase tc;
    tc.key = "Sod Shock Tube";
    tc.defaults = make({{"Lx", "1.0"}, {"Ly", "0.1"}, {"nx", "200"}, {"ny", "20"},
                        {"use_viscous", "false"}, {"t_end", "0.2"}, {"cfl", "0.5"},
                        {"save_interval", "20"}, {"riemann_solver", "HLLC"},
                        {"bc_left", "transmissive"}, {"bc_right", "transmissive"},
                        {"bc_bottom", "wall"}, {"bc_top", "wall"},
                        {"use_bodyforce", "false"}, {"gx", "0.0"}, {"gy", "0.0"}});
    tc.initialize = [](const Mesh& m, const EOS& eos, double) {
        std::printf("[TEST CASE] Initialising Sod Shock Tube...\n");
        Field U(static_cast<size_t>(m.num_cells()));
        const double mid = m.Lx / 2.0;
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                const bool left = m.xc(i) <= mid;
                U[j * m.nx + i] = eos.prim_to_cons(left ? State{1.0, 0, 0, 1.0} : State{0.125, 0, 0, 0.1});
            }
        return U;
    };
    tc.configure_boundaries = [](BoundaryManager& bm, bool) {
        bm.set_boundary_types(BCType::Transmissive, BCType::Transmissive, BCType::Wall, BCType::Wall);
    };
    return tc;
}

// --- Hagen-Poiseuille channel ---
TestCase poiseuille() {
    constexpr double rho_0 = 1.0, u_in = 0.01, p_out = 1.0;
    TestCase tc;
    tc.key = "Poiseuille Flow";
    tc.requires_viscosity = true;
    tc.defaults = make({{"Lx", "3.0"}, {"Ly", "1.0"}, {"nx", "60"}, {"ny", "20"},
                        {"use_viscous", "true"}, {"mu", "0.02"}, {"u_in", "0.01"}, {"rho_0", "1.0"},
                        {"p_out", "1.0"}, {"t_end", "80.0"}, {"save_interval", "1000"}, {"cfl", "0.1"},
                        {"riemann_solver", "HLLC (Low-Mach Thornber)"},
                        {"bc_left", "massflow_inlet"}, {"bc_right", "outlet"},
                        {"bc_bottom", "no_slip"}, {"bc_top", "no_slip"},
                        {"use_bodyforce", "false"}, {"gx", "0.0"}, {"gy", "0.0"},
                        {"dt_phys", "0.5"}, {"max_inner_iter", "50"}, {"inner_tol_rel", "1e-3"}});
    tc.initialize = [](const Mesh& m, const EOS& eos, double mu) {
        std::printf("[TEST CASE] Initialising Poiseuille flow...\n");
        const double dp_dx = -12.0 * mu * u_in / (m.Ly * m.Ly);   // analytical pressure gradient
        Field U(static_cast<size_t>(m.num_cells()));
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                const double p = p_out - dp_dx * (m.Lx - m.xc(i));
                U[j * m.nx + i] = eos.prim_to_cons({rho_0, u_in, 0.0, p});
            }
        return U;
    };
    tc.configure_boundaries = [](BoundaryManager& bm, bool verbose) {
        bm.set_boundary_types(BCType::MassflowInlet, BCType::Outlet, BCType::NoSlip, BCType::NoSlip);
        bm.set_params({u_in, 0.0, rho_0, p_out});
        if (verbose)
            std::printf("[POISEUILLE] Mass-flow inlet rho=%g u=%g  |  Pressure outlet p=%g  |  No-slip walls\n",
                        rho_0, u_in, p_out);
    };
    return tc;
}

// --- Lid-driven cavity (Re = 400) ---
TestCase cavity() {
    constexpr double rho_0 = 1.0, p_out = 100.0;
    TestCase tc;
    tc.key = "Lid Driven Cavity";
    tc.requires_viscosity = true;
    tc.defaults = make({{"Lx", "1.0"}, {"Ly", "1.0"}, {"nx", "30"}, {"ny", "30"},
                        {"use_viscous", "true"}, {"mu", "2.5e-3"}, {"u_in", "1.0"}, {"rho_in", "1.0"},
                        {"p_out", "100.0"}, {"t_end", "15.0"}, {"save_interval", "2000"}, {"cfl", "0.4"},
                        {"riemann_solver", "HLLC (Low-Mach Thornber)"}, {"time_scheme", "Runge-Kutta 4"},
                        {"bc_left", "no_slip"}, {"bc_right", "no_slip"},
                        {"bc_bottom", "no_slip"}, {"bc_top", "moving_wall"},
                        {"use_bodyforce", "false"},
                        {"dt_phys", "0.05"}, {"max_inner_iter", "50"}, {"inner_tol_rel", "1e-3"}});
    tc.initialize = [](const Mesh& m, const EOS& eos, double) {
        return Field(static_cast<size_t>(m.num_cells()), State{rho_0, 0.0, 0.0, p_out / (eos.gamma - 1.0)});
    };
    return tc;
}

// --- IBM cylinder, von Karman street (Re = 100) ---
TestCase cylinder() {
    TestCase tc;
    tc.key = "IBM Flow over Cylinder";
    tc.requires_viscosity = true;
    tc.defaults = make({{"Lx", "4.0"}, {"Ly", "2.0"}, {"nx", "400"}, {"ny", "200"},
                        {"use_viscous", "true"}, {"mu", "2e-3"}, {"u_in", "1.0"}, {"rho_in", "1.0"},
                        {"p_out", "100.0"}, {"t_end", "12.0"}, {"save_interval", "1000"}, {"cfl", "0.5"},
                        {"riemann_solver", "HLLC (Low-Mach Thornber)"}, {"time_scheme", "SSP Runge-Kutta 2"},
                        {"bc_left", "vel_inlet"}, {"bc_right", "transmissive"},
                        {"bc_bottom", "wall"}, {"bc_top", "wall"},
                        {"use_bodyforce", "false"}, {"use_muscl", "true"}, {"muscl_limiter", "Van Leer"},
                        {"use_mpi", "true"}, {"n_cpus", "16"},
                        {"use_ibm", "true"}, {"ibm_shape", "Cylinder"}, {"ibm_radius", "0.1"},
                        {"ibm_cx", "1.0"}, {"ibm_cy", "1.0"}});
    tc.initialize = [](const Mesh& m, const EOS& eos, double) { return uniform(m, 1.0, 1.0, 0.0, 100.0, eos.gamma); };
    return tc;
}

// --- Shu-Osher shock / entropy-wave interaction ---
TestCase shu_osher() {
    TestCase tc;
    tc.key = "Shu-Osher Test";
    tc.defaults = make({{"Lx", "10.0"}, {"Ly", "0.1"}, {"nx", "400"}, {"ny", "2"},
                        {"use_viscous", "false"}, {"gamma", "1.4"}, {"t_end", "1.8"},
                        {"save_interval", "100"}, {"cfl", "0.4"}, {"riemann_solver", "HLLC"},
                        {"time_scheme", "SSP Runge-Kutta 2"},
                        {"bc_left", "transmissive"}, {"bc_right", "transmissive"},
                        {"bc_bottom", "wall"}, {"bc_top", "wall"},
                        {"use_muscl", "true"}, {"muscl_limiter", "MC"}, {"use_mpi", "false"}});
    tc.initialize = [](const Mesh& m, const EOS& eos, double) {
        std::printf("[TEST CASE] Initialising 1D: Shu-Osher Problem...\n");
        constexpr double rho_L = 3.857143, u_L = 2.629369, p_L = 10.33333;
        constexpr double eps = 0.2, k = 5.0, x_shift = 5.0, x_shock = 1.0;
        Field U(static_cast<size_t>(m.num_cells()));
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                const double x = m.xc(i);
                const State W = (x < x_shock) ? State{rho_L, u_L, 0.0, p_L}
                                             : State{1.0 + eps * std::sin(k * (x - x_shift)), 0.0, 0.0, 1.0};
                U[j * m.nx + i] = eos.prim_to_cons(W);
            }
        return U;
    };
    tc.configure_boundaries = [](BoundaryManager& bm, bool verbose) {
        if (verbose) std::printf("[TEST CASE] Forcing 1D boundary conditions for 1D: Shu-Osher Problem\n");
        bm.set_boundary_types(BCType::Transmissive, BCType::Transmissive, BCType::Wall, BCType::Wall);
    };
    return tc;
}

// --- IBM supersonic cylinder (Mach 2 bow shock) ---
TestCase supersonic_cylinder() {
    constexpr double u_in = 2.0, rho_in = 1.0, p_out = 1.0 / 1.4;
    TestCase tc;
    tc.key = "IBM Supersonic Flow over Cylinder";
    tc.defaults = make({{"Lx", "4.0"}, {"Ly", "4.0"}, {"nx", "400"}, {"ny", "400"},
                        {"use_viscous", "false"}, {"u_in", "2.0"}, {"rho_in", "1.0"},
                        {"p_out", "0.7142857142857143"}, {"t_end", "5.0"}, {"save_interval", "250"},
                        {"cfl", "0.4"}, {"riemann_solver", "HLLC"}, {"time_scheme", "SSP Runge-Kutta 2"},
                        {"bc_left", "vel_inlet"}, {"bc_right", "transmissive"},
                        {"bc_bottom", "transmissive"}, {"bc_top", "transmissive"},
                        {"use_bodyforce", "false"}, {"use_muscl", "true"}, {"muscl_limiter", "Minmod"},
                        {"use_mpi", "true"}, {"n_cpus", "16"},
                        {"use_ibm", "true"}, {"ibm_shape", "Cylinder"}, {"ibm_radius", "0.25"},
                        {"ibm_cx", "1.0"}, {"ibm_cy", "2.0"}});
    tc.initialize = [](const Mesh& m, const EOS& eos, double) {
        const double c = std::sqrt(eos.gamma * p_out / rho_in);
        std::printf("[TEST CASE] Initialising IBM: Supersonic Cylinder (Mach 2) at Mach %.2f...\n", u_in / c);
        return uniform(m, rho_in, u_in, 0.0, p_out, eos.gamma);
    };
    return tc;
}

void all_periodic(BoundaryManager& bm, bool) {
    bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Periodic, BCType::Periodic);
}

// --- Taylor-Green vortex: decaying viscous solution on a periodic square ---
TestCase taylor_green() {
    TestCase tc;
    tc.key = "Taylor-Green Vortex";
    tc.requires_viscosity = true;
    tc.defaults = make({{"Lx", "6.283185307179586"}, {"Ly", "6.283185307179586"}, {"nx", "64"}, {"ny", "64"},
                        {"use_viscous", "true"}, {"mu", "0.02"}, {"t_end", "5.0"}, {"cfl", "0.5"},
                        {"save_interval", "200"}, {"riemann_solver", "HLLC (Low-Mach Thornber)"},
                        {"time_scheme", "SSP Runge-Kutta 2"}, {"use_muscl", "true"}, {"muscl_limiter", "Van Leer"},
                        {"bc_left", "periodic"}, {"bc_right", "periodic"}, {"bc_bottom", "periodic"},
                        {"bc_top", "periodic"}, {"use_bodyforce", "false"}});
    tc.initialize = [](const Mesh& m, const EOS& eos, double mu) {
        std::printf("[TEST CASE] Initialising Taylor-Green vortex...\n");
        Field U(static_cast<size_t>(m.num_cells()));
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) U[j * m.nx + i] = eos.prim_to_cons(taylor_green_exact(m.xc(i), m.yc(j), 0.0, m.Lx, mu));
        return U;
    };
    tc.configure_boundaries = all_periodic;
    return tc;
}

// --- Isentropic vortex: inviscid advection of an exact vortex (2nd-order accuracy check) ---
TestCase isentropic_vortex() {
    TestCase tc;
    tc.key = "Isentropic Vortex";
    tc.defaults = make({{"Lx", "10.0"}, {"Ly", "10.0"}, {"nx", "80"}, {"ny", "80"}, {"use_viscous", "false"},
                        {"gamma", "1.4"}, {"t_end", "10.0"}, {"cfl", "0.5"}, {"save_interval", "50"},
                        {"riemann_solver", "HLLC"}, {"time_scheme", "SSP Runge-Kutta 2"}, {"use_muscl", "true"},
                        {"muscl_limiter", "MC"}, {"bc_left", "periodic"}, {"bc_right", "periodic"},
                        {"bc_bottom", "periodic"}, {"bc_top", "periodic"}, {"use_bodyforce", "false"}});
    tc.initialize = [](const Mesh& m, const EOS& eos, double) {
        std::printf("[TEST CASE] Initialising isentropic vortex...\n");
        Field U(static_cast<size_t>(m.num_cells()));
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i)
                U[j * m.nx + i] = eos.prim_to_cons(isentropic_vortex_exact(m.xc(i), m.yc(j), 0.0, m.Lx, m.Ly, eos.gamma));
        return U;
    };
    tc.configure_boundaries = all_periodic;
    return tc;
}

// --- 3-D Taylor-Green vortex (Re = 1600, Ma = 0.1): transition to turbulence ---
TestCase taylor_green_3d() {
    TestCase tc;
    tc.key = "Taylor-Green Vortex 3D";
    tc.requires_viscosity = true;
    tc.three_d = true;
    tc.defaults = make({{"Lx", "6.283185307179586"}, {"Ly", "6.283185307179586"}, {"Lz", "6.283185307179586"},
                        {"nx", "32"}, {"ny", "32"}, {"nz", "32"}, {"use_viscous", "true"}, {"mu", "6.25e-4"},
                        {"t_end", "20.0"}, {"cfl", "0.5"}, {"save_interval", "100"},
                        {"riemann_solver", "HLLC (Low-Mach Thornber)"}, {"time_scheme", "SSP Runge-Kutta 2"},
                        {"use_muscl", "true"}, {"muscl_limiter", "Van Leer"},
                        {"bc_left", "periodic"}, {"bc_right", "periodic"}, {"bc_bottom", "periodic"},
                        {"bc_top", "periodic"}, {"bc_front", "periodic"}, {"bc_back", "periodic"},
                        {"use_bodyforce", "false"}, {"use_mpi", "false"}});
    tc.initialize = [](const Mesh& m, const EOS& eos, double) {
        std::printf("[TEST CASE] Initialising 3-D Taylor-Green vortex...\n");
        Field U(static_cast<size_t>(m.num_cells()));
        for (int k = 0; k < m.nz; ++k)
            for (int j = 0; j < m.ny; ++j)
                for (int i = 0; i < m.nx; ++i)
                    U[m.cell(i, j, k)] = eos.prim_to_cons(taylor_green_3d_initial(m.xc(i), m.yc(j), m.zc(k), eos.gamma));
        return U;
    };
    tc.configure_boundaries = [](BoundaryManager& bm, bool) {
        bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Periodic, BCType::Periodic);
        bm.set_type(Face::Front, BCType::Periodic);
        bm.set_type(Face::Back, BCType::Periodic);
    };
    return tc;
}

}  // namespace

State taylor_green_3d_initial(double x, double y, double z, double gamma) {
    constexpr double U0 = 1.0, rho0 = 1.0, Ma = 0.1;
    const double p0 = rho0 * U0 * U0 / (gamma * Ma * Ma);
    const double u = U0 * std::sin(x) * std::cos(y) * std::cos(z);
    const double v = -U0 * std::cos(x) * std::sin(y) * std::cos(z);
    const double p = p0 + rho0 * U0 * U0 / 16.0 * (std::cos(2.0 * x) + std::cos(2.0 * y)) * (std::cos(2.0 * z) + 2.0);
    return {rho0, u, v, p, 0.0};
}

State taylor_green_exact(double x, double y, double t, double L, double mu) {
    constexpr double PI = 3.14159265358979323846, U0 = 1.0, rho0 = 1.0, p0 = 100.0;
    const double k = 2.0 * PI / L;
    const double F = std::exp(-2.0 * (mu / rho0) * k * k * t);
    const double u = -U0 * std::cos(k * x) * std::sin(k * y) * F;
    const double v = U0 * std::sin(k * x) * std::cos(k * y) * F;
    const double p = p0 - 0.25 * rho0 * U0 * U0 * (std::cos(2.0 * k * x) + std::cos(2.0 * k * y)) * F * F;
    return {rho0, u, v, p};
}

State isentropic_vortex_exact(double x, double y, double t, double Lx, double Ly, double gamma) {
    constexpr double PI = 3.14159265358979323846, beta = 5.0, u_inf = 1.0, v_inf = 1.0;
    // Vortex centre advected with the free stream; minimum-image distance on the torus
    auto wrap = [](double d, double L) { return d - L * std::round(d / L); };
    const double dx = wrap(x - (0.5 * Lx + u_inf * t), Lx);
    const double dy = wrap(y - (0.5 * Ly + v_inf * t), Ly);
    const double r2 = dx * dx + dy * dy;
    const double g = beta / (2.0 * PI) * std::exp(0.5 * (1.0 - r2));
    const double T = 1.0 - (gamma - 1.0) * beta * beta / (8.0 * gamma * PI * PI) * std::exp(1.0 - r2);
    const double rho = std::pow(T, 1.0 / (gamma - 1.0));
    return {rho, u_inf - g * dy, v_inf + g * dx, rho * T};
}

const std::vector<TestCase>& test_registry() {
    static const std::vector<TestCase> reg = {sod(), poiseuille(), cavity(), cylinder(), shu_osher(),
                                              supersonic_cylinder(), taylor_green(), isentropic_vortex(),
                                              taylor_green_3d()};
    return reg;
}

const TestCase& get_test_case(const std::string& key) {
    for (const auto& tc : test_registry())
        if (tc.key == key) return tc;
    throw std::invalid_argument("Unknown test case: '" + key + "'");
}

}  // namespace fv
