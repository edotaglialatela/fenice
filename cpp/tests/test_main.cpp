// Unit tests for the C++ solver.
// Self-contained harness: run `fenice_tests` (exit code = number of failed checks).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <condition_variable>
#include <iterator>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "fv/CellAdvance.hpp"
#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"
#include "fv/CellParallel.hpp"
#include "fv/CellRemesh.hpp"
#include "fv/Simulation.hpp"
#include "fv/SolutionMapping.hpp"

using namespace fv;
namespace fs = std::filesystem;

namespace {

int g_fail = 0, g_checks = 0;

struct TestCaseReg {
    const char* name;
    void (*fn)();
};
std::vector<TestCaseReg>& registry() {
    static std::vector<TestCaseReg> r;
    return r;
}
struct Registrar {
    Registrar(const char* n, void (*f)()) { registry().push_back({n, f}); }
};

#define TEST(name)                                                                                                     \
    static void name();                                                                                                \
    static Registrar reg_##name(#name, &name);                                                                         \
    static void name()

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        ++g_checks;                                                                                                    \
        if (!(cond)) {                                                                                                 \
            ++g_fail;                                                                                                  \
            std::printf("    FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                                            \
        }                                                                                                              \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                          \
    do {                                                                                                               \
        ++g_checks;                                                                                                    \
        const double va_ = (a), vb_ = (b);                                                                             \
        if (!(std::abs(va_ - vb_) <= (tol))) {                                                                         \
            ++g_fail;                                                                                                  \
            std::printf("    FAIL %s:%d  %s = %.15g, expected %.15g (tol %g)\n", __FILE__, __LINE__, #a, va_, vb_,     \
                        (double)(tol));                                                                                \
        }                                                                                                              \
    } while (0)

fs::path tmp_dir() {
    fs::path d = fs::temp_directory_path() / "fv_cpp_tests";
    fs::create_directories(d);
    return d;
}

SimParams params_for(const std::string& tc, std::initializer_list<std::pair<const char*, const char*>> kv) {
    Params p = baseline_defaults();
    p.merge(get_test_case(tc).defaults);
    p.set("test_case", tc);
    p.set("output_root", tmp_dir().string());
    for (const auto& e : kv) p.set(e.first, e.second);
    return SimParams::from(p);
}

Field uniform_field(const EOS& eos, int n, double rho, double u, double v, double p) {
    return Field(static_cast<size_t>(n), eos.prim_to_cons({rho, u, v, p}));
}

// Smooth, non-trivial state with nonzero velocity everywhere (also inside an immersed body)
State test_state(const EOS& eos, double x, double y) {
    return eos.prim_to_cons({1.0 + 0.1 * std::sin(1.3 * x) * std::cos(0.9 * y), 1.0 + 0.2 * std::sin(0.7 * y),
                             0.15 * std::cos(1.1 * x), 17.0 + 0.5 * std::cos(0.6 * x + 0.4 * y)});
}

SimParams airfoil_params(std::initializer_list<std::pair<const char*, const char*>> kv) {
    Params pr = baseline_defaults();
    pr.merge(get_test_case("IBM Airfoil").defaults);
    pr.set("test_case", "IBM Airfoil");
    pr.set("output_root", tmp_dir().string());
    pr.set("use_mpi", "false");
    for (const auto& e : kv) pr.set(e.first, e.second);
    return SimParams::from(pr);
}

double max_abs(const Field& R) {
    double m = 0.0;
    for (const State& r : R)
        for (double x : r) m = std::max(m, std::abs(x));
    return m;
}

}   // namespace

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

// ---------------------------------------------------------------------------
// Case files
// ---------------------------------------------------------------------------

namespace {
// Write `text` as <tmp>/<name>/case.toml and return the folder.
std::string write_case(const std::string& name, const std::string& text) {
    const fs::path d = tmp_dir() / "casefiles" / name;
    fs::create_directories(d);
    std::ofstream(d / "case.toml", std::ios::binary) << text;
    return d.string();
}

// The error message load_case_file throws for `text`, or "" if it loads.
std::string case_error(const std::string& name, const std::string& text) {
    try {
        load_case_file(write_case(name, text));
    } catch (const std::exception& e) { return e.what(); }
    return "";
}

const char* kMinimalCase = R"toml(
[case]
name = "Minimal"
[initial]
state = { rho = 1.0, p = 1.0 }
)toml";
}   // namespace

TEST(every_parameter_has_a_case_section) {
    const Params base = baseline_defaults();
    for (const auto& [k, v] : base.values()) {
        if (k == "test_case") continue;
        const bool ok = !param_section(k).empty();
        if (!ok) std::printf("    parameter '%s' has no case-file section\n", k.c_str());
        CHECK(ok);
    }
}

// Bad --set / --config values fail up front instead of being silently ignored or truncated.
TEST(parameter_validation) {
    auto error_of = [](std::initializer_list<std::pair<const char*, const char*>> sets) -> std::string {
        Params p = baseline_defaults();
        for (const auto& kv : sets) p.set(kv.first, kv.second);
        try {
            check_parameter_keys(p, "--set");
            SimParams::from(p);
        } catch (const std::invalid_argument& e) { return e.what(); }
        return "";
    };
    CHECK(error_of({}).empty());
    CHECK(error_of({{"nx", "200.0"}, {"cfl", " 0.4 "}, {"eos_type", "stiffened gas"}}).empty());
    CHECK(error_of({{"nX", "200"}}).find("did you mean 'nx'") != std::string::npos);
    CHECK(error_of({{"nx", "200.5"}}).find("not an integer") != std::string::npos);
    CHECK(error_of({{"cfl", "0.5x"}}).find("not a number") != std::string::npos);
    CHECK(error_of({{"cfl", "0"}}).find("cfl") != std::string::npos);
    CHECK(error_of({{"eos_type", "ideal"}}).find("eos_type") != std::string::npos);
    CHECK(error_of({{"bc_left", "periodic"}}).find("in pairs") != std::string::npos);
    CHECK(error_of({{"bc_left", "Periodic"}, {"bc_right", "periodic"}}).empty());
    // adaptive mesh: any base mesh, resume, periodic and mass-flow boundaries and sources are fine now
    CHECK(error_of({{"use_amr", "true"}, {"resume", "true"}}).empty());
    CHECK(error_of({{"use_amr", "true"}, {"nx", "100"}}).empty());
    CHECK(error_of({{"use_amr", "true"}, {"nx", "96"}, {"ny", "96"}, {"bc_left", "periodic"}, {"bc_right", "periodic"}}).empty());
    CHECK(error_of({{"use_amr", "true"}, {"bc_left", "massflow_inlet"}}).empty());
    CHECK(error_of({{"amr_patch_nx", "16"}}).find("removed") != std::string::npos);
    CHECK(error_of({{"amr_patch_ny", "16"}}).find("level-0 grid") != std::string::npos);
    CHECK(error_of({{"amr_buffer_layers", "1"}}).find("amr_diffusion") != std::string::npos);
    CHECK(error_of({{"amr_indicator", "density+vorticity"}}).find("amr_indicator") != std::string::npos);
    CHECK(error_of({{"amr_indicator", "pressure+density+velocity"}}).empty());
    CHECK(error_of({{"amr_derefine_threshold", "0.3"}, {"amr_refine_threshold", "0.2"}}).find("derefine") != std::string::npos);
    CHECK(error_of({{"use_amr", "true"}, {"amr_criterion", "0"}}).find("amr_criterion") != std::string::npos);
    CHECK(error_of({{"use_amr", "true"}, {"nx", "96"}, {"ny", "96"}, {"time_scheme", "BDF2 Dual Time"}}).find("BDF2") !=
          std::string::npos);
    CHECK(error_of({{"use_amr", "true"}, {"nx", "96"}, {"ny", "96"}}).empty());
    bool threw = false;
    try {
        parse_limiter("superbee");
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

TEST(case_file_parsing) {
    const CaseFile cf = load_case_file(write_case("parse", R"toml(
[case]
name = "Parse test"
description = "all value types"
order = 7
requires_viscous = true

[mesh]
Lx = 2.5
nx = 40

[physics]
use_viscous = true
mu = 2.5e-3

[numerics]
riemann_solver = "HLLC (Low-Mach Thornber)"

[initial]
state = { rho = 1.0, u = "0.5*x", p = 1.0 }

[[initial.region]]
shape = "sphere"
center = ["Lx/2", 0.5]
radius = 0.25
state = { rho = 0.1 }
)toml"));
    CHECK(cf.name == "Parse test");
    CHECK(cf.description == "all value types");
    CHECK(cf.order == 7);
    CHECK(cf.requires_viscous);
    CHECK(!cf.requires_3d);
    CHECK(cf.params.str("Lx") == "2.5");
    CHECK(cf.params.str("nx") == "40");
    CHECK(cf.params.str("use_viscous") == "true");
    CHECK(cf.params.num("mu") == 2.5e-3);
    CHECK(cf.params.str("riemann_solver") == "HLLC (Low-Mach Thornber)");
    CHECK(cf.initial.state.at("u") == "0.5*x");
    CHECK(cf.initial.regions.size() == 1);
    CHECK(cf.initial.regions[0].center[0] == "Lx/2");
    CHECK(cf.initial.regions[0].state.at("rho") == "0.1");
}

TEST(case_file_errors_are_helpful) {
    const std::string base = kMinimalCase;
    auto has = [](const std::string& msg, const char* needle) {
        const bool ok = msg.find(needle) != std::string::npos;
        if (!ok) std::printf("    expected '%s' in: %s\n", needle, msg.c_str());
        return ok;
    };
    CHECK(has(case_error("e1", base + "[thermo]\ngama = 1.4\n"), "did you mean 'gamma'"));
    CHECK(has(case_error("e2", base + "[run]\nnx = 10\n"), "'nx' belongs in [mesh], not [run]"));
    CHECK(has(case_error("e3", base + "[numeric]\ncfl = 0.5\n"), "did you mean 'numerics'"));
    CHECK(has(
        case_error("e4",
                   base + "[[initial.region]]\nshape = \"sphear\"\ncenter = [0, 0]\nradius = 1\nstate = { rho = 2 }\n"),
        "did you mean 'sphere'"));
    CHECK(has(case_error("e5", base + "[[initial.region]]\nshape = \"box\"\nmin = [0, 0]\nstate = { rho = 2 }\n"),
              "a box needs 'min' and 'max'"));
    CHECK(has(case_error("e6", "[case]\nname = \"x\"\n[initial]\nstate = { rho = 1.0 }\n"), "at least rho and p"));
    CHECK(has(case_error("e7", "[case]\nname = \"x\"\n[initial]\nstate = { rho = \"1 + sin(xx)\", p = 1.0 }\n"),
              "did you mean 'x'"));
    CHECK(has(case_error("e8", "[case]\nname = \"x\"\n[initial]\nstate = { rh = 1.0, p = 1.0 }\n"),
              "did you mean 'rho'"));
    CHECK(has(case_error("e9", base + "[mesh]\nnx = = 3\n"), "case.toml:"));   // TOML syntax, with a line
    CHECK(has(case_error("e10", base + "[mesh]\ntest_case = \"Sod\"\n"), "cannot be set in a case file"));
    CHECK(has(case_error("e11", "[initial]\nstate = { rho = 1.0, p = 1.0 }\n"), "missing [case] name"));
    CHECK(case_error("ok", base).empty());
}

TEST(initial_regions_and_expressions) {
    // 10 x 10 cells on [0, 1]^2: centres at 0.05, 0.15, ..., 0.95
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 10, 10);
    const EOS eos = EOS::ideal(1.4);
    Params run = baseline_defaults();
    run.set("u_in", "0.75");
    InitialSpec spec;
    spec.state = {{"rho", "1 + 0.5*x"}, {"u", "u_in"}, {"v", "0.0"}, {"p", "2.0"}};
    InitialRegion box;   // later regions override earlier ones
    box.shape = "box";
    box.min = {"0.0", "0.0"};
    box.max = {"0.5", "0.5"};
    box.state = {{"rho", "3.0"}};
    InitialRegion half;
    half.shape = "halfspace";
    half.point = {"0.0", "0.8"};
    half.normal = {"0.0", "1.0"};   // y >= 0.8
    half.state = {{"p", "5.0"}, {"v", "-1.0"}};
    InitialRegion circle;
    circle.shape = "sphere";
    circle.center = {"0.3", "0.3"};   // a grid vertex: the 4 cells around it are inside
    circle.radius = "0.08";
    circle.state = {{"rho", "9.0"}};
    InitialRegion cyl;
    cyl.shape = "cylinder";
    cyl.center = {"0.7", "0.2"};
    cyl.radius = "Lx/12";
    cyl.state = {{"rho", "7.0"}};
    spec.regions = {box, half, circle, cyl};
    const Field U = evaluate_initial(spec, m, eos, run);

    bool ok = true;
    for (int j = 0; j < 10; ++j)
        for (int i = 0; i < 10; ++i) {
            const double x = m.xc(i), y = m.yc(j);
            State W{1.0 + 0.5 * x, 0.75, 0.0, 2.0, 0.0};
            if (x <= 0.5 && y <= 0.5) W[0] = 3.0;
            if (y >= 0.8) {
                W[3] = 5.0;
                W[2] = -1.0;
            }
            if ((x - 0.3) * (x - 0.3) + (y - 0.3) * (y - 0.3) <= 0.08 * 0.08) W[0] = 9.0;
            if ((x - 0.7) * (x - 0.7) + (y - 0.2) * (y - 0.2) <= (1.0 / 12) * (1.0 / 12)) W[0] = 7.0;
            const State E = eos.prim_to_cons(W);
            for (int q = 0; q < kNVar; ++q) ok = ok && U[j * 10 + i][q] == E[q];
        }
    CHECK(ok);
    // The circle and the cylinder each cover the 4 cells around their centre
    int n9 = 0, n7 = 0;
    for (const State& s : U) {
        n9 += s[0] == 9.0;
        n7 += s[0] == 7.0;
    }
    CHECK(n9 == 4);
    CHECK(n7 == 4);

    // A profile with a region on top: untouched cells keep the profile's state bit for bit
    InitialSpec ps;
    InitialRegion bubble = circle;
    bubble.state = {{"rho", "0.5"}};
    ps.regions = {bubble};
    Field base(static_cast<size_t>(m.num_cells()));
    for (int c = 0; c < m.num_cells(); ++c) base[c] = eos.prim_to_cons({1.0 + 0.01 * c, 0.1, 0.2, 3.0, 0.0});
    const Field P = evaluate_initial(ps, m, eos, run, &base);
    int kept = 0, changed = 0;
    for (int c = 0; c < m.num_cells(); ++c) {
        if (P[c] == base[c])
            ++kept;
        else if (std::abs(eos.cons_to_prim(P[c])[0] - 0.5) < 1e-14)
            ++changed;
    }
    CHECK(kept == 96);
    CHECK(changed == 4);

    // Usual precedence: -a^2 = -(a^2) (a Gaussian must not become exp(+x^2)), a^b^c = a^(b^c)
    InitialSpec pr;
    pr.state = {{"rho", "2^3^2"}, {"u", "-2^2"}, {"v", "exp(-x^2)"}, {"p", "1.0"}};
    const Field Q = evaluate_initial(pr, m, eos, run);
    const State Wq = eos.cons_to_prim(Q[3]);   // cell (3, 0): x = 0.35
    CHECK(Wq[0] == 512.0);
    CHECK_NEAR(Wq[1], -4.0, 1e-12);
    CHECK_NEAR(Wq[2], std::exp(-0.35 * 0.35), 1e-12);
}

TEST(case_file_save_load_round_trip) {
    CaseFile cf = load_case_file(write_case("rt_src", R"toml(
[case]
name = "Round trip"
description = "quote \" and backslash \\ survive"
order = 3
requires_3d = true

[mesh]
Lx = 6.283185307179586
nz = 8

[boundaries]
bc_front = "wall"
p_out = 0.7142857142857143

[numerics]
inner_tol_rel = 1e-3

[initial]
state = { rho = 1.0, u = "sin(x)*cos(y)", w = 0.25, p = "1/(gamma*0.1*0.1)" }

[[initial.region]]
shape = "cylinder"
center = ["Lx/2", 1.0, 0.0]
radius = 0.5
axis = "x"
state = { rho = 2.0 }

[[initial.region]]
shape = "halfspace"
point = [0.0, 0.0, "Lz/2"]
normal = [0.0, 0.0, 1.0]
state = { p = 9.0 }
)toml"));
    const std::string out = (tmp_dir() / "casefiles" / "rt_saved").string();
    save_case_file(cf, out);
    const CaseFile back = load_case_file(out);
    CHECK(back.name == cf.name);
    CHECK(back.description == cf.description);
    CHECK(back.order == 3 && back.requires_3d && !back.requires_viscous);
    CHECK(back.params.values().size() == cf.params.values().size());
    for (const auto& [k, v] : cf.params.values()) {
        const std::string w = back.params.str(k);
        char* e1 = nullptr;
        char* e2 = nullptr;
        const double a = std::strtod(v.c_str(), &e1), b = std::strtod(w.c_str(), &e2);
        CHECK(w == v || (*e1 == '\0' && *e2 == '\0' && a == b));
    }
    CHECK(back.initial.state == cf.initial.state);
    CHECK(back.initial.regions.size() == 2);
    CHECK(back.initial.regions[0].axis == "x");
    CHECK(back.initial.regions[1].point == cf.initial.regions[1].point);
    // Same initial field, bit for bit
    const Mesh m = generate_cartesian_mesh_3d(6.283185307179586, 2.0, 1.0, 12, 6, 8);
    Params run = baseline_defaults();
    run.merge(cf.params);
    const Field a = evaluate_initial(cf.initial, m, EOS::ideal(1.4), run);
    const Field b = evaluate_initial(back.initial, m, EOS::ideal(1.4), run);
    CHECK(a == b);
}

TEST(builtin_cases_and_save_case) {
    // The shipped case files: names, order and flags
    const std::vector<std::string> names = {
        "Sod Shock Tube",      "Poiseuille Flow",   "Lid Driven Cavity",     "IBM Flow over Cylinder",
        "IBM Airfoil",         "IBM Airfoil AMR",   "Shu-Osher Test",        "IBM Supersonic Flow over Cylinder",
        "Taylor-Green Vortex", "Isentropic Vortex", "Taylor-Green Vortex 3D"};
    std::vector<std::string> got;
    for (const TestCase& tc : test_registry()) got.push_back(tc.key);
    CHECK(std::search(got.begin(), got.end(), names.begin(), names.end()) == got.begin());
    CHECK(get_test_case("Taylor-Green Vortex 3D").three_d);
    CHECK(get_test_case("Lid Driven Cavity").requires_viscosity);
    CHECK(!get_test_case("Sod Shock Tube").requires_viscosity);
    CHECK(&get_test_case("sod_shock_tube") == &get_test_case("Sod Shock Tube"));   // folder name
    bool threw = false;
    try {
        get_test_case("Sod Shok Tube");
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).find("did you mean 'Sod Shock Tube'") != std::string::npos;
    }
    CHECK(threw);
    CHECK(case_folder_name("IBM Flow over Cylinder") == "ibm_flow_over_cylinder");
    CHECK(case_folder_name("Re = 1600 (3-D)") == "re_1600_3_d");

    // save_case keeps the source's parameters and anything changed; it never overwrites
    Params run = baseline_defaults();
    run.merge(get_test_case("Lid Driven Cavity").defaults);
    run.set("nx", "48");
    run.set("output_root", "somewhere");
    const std::string dir = (tmp_dir() / "casefiles" / "saved_cavity").string();
    fs::remove_all(dir);
    const std::string file = save_case(get_test_case("Lid Driven Cavity"), run, dir, "Saved cavity");
    const TestCase& saved = get_test_case(dir);
    CHECK(saved.key == "Saved cavity");
    CHECK(saved.defaults.str("nx") == "48");
    CHECK(saved.defaults.str("bc_top") == "moving_wall");
    CHECK(!saved.defaults.has("output_root"));
    CHECK(saved.requires_viscosity);
    bool refused = false;
    try {
        save_case(get_test_case("Lid Driven Cavity"), run, dir, "Saved cavity 2");
    } catch (const std::runtime_error&) { refused = true; }
    CHECK(refused);
    CHECK(fs::exists(file));
}

// ---------------------------------------------------------------------------
// CellMesh (cell-based AMR topology)
// ---------------------------------------------------------------------------

namespace {

// Independent oracle: leaf boxes in integer coordinates of the finest level, every touching pair found by brute
// force (periodic axes by shifted copies); the mesh's own lookup tables are never used.
struct LeafBox {
    long long x0, x1, y0, y1;
    int level;
};

// axis, left, right, start of the segment (finer cell's own transverse start), kind (0 interior, 1 coarse-fine,
// 2 boundary), level, boundary side
using FaceTuple = std::tuple<int, int, int, long long, int, int, int>;

void check_cell_mesh(const CellMesh& m) {
    const int N = m.num_leaves(), M = m.max_level;
    const long long W = static_cast<long long>(m.nx) << M, H = static_cast<long long>(m.ny) << M;
    std::vector<LeafBox> box(N);
    long long area = 0;
    for (int c = 0; c < N; ++c) {
        const long long s = 1ll << (M - m.level[c]);
        box[c] = {m.ci[c] * s, (m.ci[c] + 1) * s, m.cj[c] * s, (m.cj[c] + 1) * s, m.level[c]};
        area += s * s;
    }
    CHECK(area == W * H);   // with the pairwise test below: the leaves tile the domain

    // order: level runs, Morton inside a level
    for (int c = 1; c < N; ++c) {
        CHECK(m.level[c] >= m.level[c - 1]);
        if (m.level[c] == m.level[c - 1])
            CHECK(morton_key(m.ci[c], m.cj[c]) > morton_key(m.ci[c - 1], m.cj[c - 1]));
    }
    CHECK(static_cast<int>(m.level_begin.size()) == M + 2 && m.level_begin[M + 1] == N);
    for (int l = 0; l <= M; ++l)
        for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c) CHECK(m.level[c] == l);

    const std::vector<long long> shx = m.periodic_x ? std::vector<long long>{-W, 0, W} : std::vector<long long>{0};
    const std::vector<long long> shy = m.periodic_y ? std::vector<long long>{-H, 0, H} : std::vector<long long>{0};
    std::vector<FaceTuple> want;
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b)
            for (long long sx : shx)
                for (long long sy : shy) {
                    if (a == b && sx == 0 && sy == 0) continue;
                    const LeafBox& A = box[a];
                    const LeafBox B = {box[b].x0 + sx, box[b].x1 + sx, box[b].y0 + sy, box[b].y1 + sy, box[b].level};
                    const long long ox = std::min(A.x1, B.x1) - std::max(A.x0, B.x0);   // overlap; 0 = touching
                    const long long oy = std::min(A.y1, B.y1) - std::max(A.y0, B.y0);
                    if (ox < 0 || oy < 0) continue;
                    CHECK(!(ox > 0 && oy > 0));   // no overlap
                    CHECK(std::abs(A.level - B.level) <= 1);   // corner 2:1 balance for every touching pair
                    const int kind = A.level == B.level ? 0 : 1, lev = std::max(A.level, B.level);
                    const LeafBox& fine = A.level >= B.level ? box[a] : box[b];
                    if (A.x1 == B.x0 && oy > 0) want.emplace_back(0, a, b, fine.y0, kind, lev, 0);
                    if (A.y1 == B.y0 && ox > 0) want.emplace_back(1, a, b, fine.x0, kind, lev, 0);
                }
    for (int a = 0; a < N; ++a) {
        const LeafBox& A = box[a];
        if (!m.periodic_x && A.x0 == 0) want.emplace_back(0, -1, a, A.y0, 2, A.level, static_cast<int>(Face::Left));
        if (!m.periodic_x && A.x1 == W) want.emplace_back(0, a, -1, A.y0, 2, A.level, static_cast<int>(Face::Right));
        if (!m.periodic_y && A.y0 == 0) want.emplace_back(1, -1, a, A.x0, 2, A.level, static_cast<int>(Face::Bottom));
        if (!m.periodic_y && A.y1 == H) want.emplace_back(1, a, -1, A.x0, 2, A.level, static_cast<int>(Face::Top));
    }
    std::vector<FaceTuple> got;
    for (const CellFace& f : m.faces) {
        const long long s = 1ll << (M - f.level);
        const int kind = f.kind == CellFaceKind::Interior ? 0 : (f.kind == CellFaceKind::CoarseFine ? 1 : 2);
        got.emplace_back(f.axis, f.left, f.right, f.tpos * s, kind, f.level, kind == 2 ? static_cast<int>(f.side) : 0);
        CHECK(f.length == (f.axis == 0 ? m.dy(f.level) : m.dx(f.level)));
    }
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    CHECK(want == got);   // exactly one face per shared segment, with the right orientation, kind and level

    for (size_t f = 1; f < m.faces.size(); ++f) CHECK(m.faces[f].level >= m.faces[f - 1].level);
    CHECK(static_cast<int>(m.face_level_begin.size()) == M + 2 &&
          m.face_level_begin[M + 1] == static_cast<int>(m.faces.size()));
    for (int l = 0; l <= M; ++l)
        for (int f = m.face_level_begin[l]; f < m.face_level_begin[l + 1]; ++f) CHECK(m.faces[f].level == l);

    // gather lists: every side covered exactly once by one or two faces, lower transverse position first
    for (int c = 0; c < N; ++c)
        for (int s = 0; s < 4; ++s) {
            const int b = m.side_ptr[4 * c + s], e = m.side_ptr[4 * c + s + 1];
            CHECK(e - b == 1 || e - b == 2);
            long long covered = 0;
            for (int k = b; k < e; ++k) {
                const CellFace& f = m.faces[m.side_face[k]];
                covered += 1ll << (M - f.level);
                CHECK(f.axis == s / 2);
                CHECK((s & 1 ? f.left : f.right) == c);
                if (k > b) CHECK(f.tpos > m.faces[m.side_face[k - 1]].tpos);
            }
            CHECK(covered == 1ll << (M - m.level[c]));
            const CellNeighbour nb = m.neighbour(c, static_cast<Face>(s));
            if (nb.kind == CellNeighbour::Kind::Boundary) CHECK(nb.a < 0);
            if (nb.kind == CellNeighbour::Kind::Same) CHECK(m.level[nb.a] == m.level[c]);
            if (nb.kind == CellNeighbour::Kind::Coarser) CHECK(m.level[nb.a] == m.level[c] - 1);
            if (nb.kind == CellNeighbour::Kind::Finer)
                CHECK(m.level[nb.a] == m.level[c] + 1 && m.level[nb.b] == m.level[c] + 1 && nb.a != nb.b);
        }
}

double cell_mesh_total(const CellMesh& m, int q) {
    double s = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c) s += m.U[c][q] * m.area(c);
    return s;
}

void random_state(CellMesh& m, std::mt19937& rng) {
    m.U.resize(m.num_leaves());
    for (State& u : m.U)
        for (double& v : u) v = 0.001 * static_cast<double>(rng() % 100000) + 1.0;
}

}   // namespace

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

namespace {

SimParams cell_flux_params(std::initializer_list<std::pair<const char*, const char*>> kv) {
    std::vector<std::pair<const char*, const char*>> all = {{"nx", "9"},          {"ny", "7"},          {"Lx", "1.7"},
                                                           {"Ly", "1.1"},        {"u_in", "0.7"},      {"v_in", "0.2"},
                                                           {"rho_in", "1.2"},    {"p_out", "1.3"}};
    all.insert(all.end(), kv.begin(), kv.end());
    Params p = baseline_defaults();
    p.merge(get_test_case("Sod Shock Tube").defaults);
    p.set("test_case", "Sod Shock Tube");
    p.set("output_root", tmp_dir().string());
    p.set("use_mpi", "false");
    for (const auto& e : all) p.set(e.first, e.second);
    return SimParams::from(p);
}

bool bits_equal(const Field& a, const Field& b) {
    if (a.size() != b.size()) return false;
    for (size_t c = 0; c < a.size(); ++c)
        for (int q = 0; q < kNVar; ++q)
            if (!(a[c][q] == b[c][q])) return false;
    return true;
}

}   // namespace

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

namespace {

CellMesh random_adapted_mesh(bool px, bool py, unsigned seed, int rounds = 5) {
    CellMesh m(1.5, 1.0, 6, 5, 3, px, py);
    std::mt19937 rng(seed);
    for (int round = 0; round < rounds; ++round) {
        std::vector<char> r(m.num_leaves()), c(m.num_leaves());
        for (int k = 0; k < m.num_leaves(); ++k) {
            r[k] = rng() % 100 < 15;
            c[k] = rng() % 100 < 30;
        }
        m.adapt(r, c);
    }
    return m;
}

}   // namespace

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

namespace {

// Exact density of the Sod problem (rho, u, p = 1, 0, 1 | 0.125, 0, 0.1, gamma 1.4, diaphragm at x = 0.5) at time t.
double sod_exact_density(double x, double t) {
    const double g = 1.4, rL = 1.0, pL = 1.0, rR = 0.125, pR = 0.1;
    const double cL = std::sqrt(g * pL / rL), cR = std::sqrt(g * pR / rR);
    const auto fK = [&](double p, double rK, double pK, double cK) {
        if (p > pK) return (p - pK) * std::sqrt((2.0 / ((g + 1.0) * rK)) / (p + (g - 1.0) / (g + 1.0) * pK));
        return 2.0 * cK / (g - 1.0) * (std::pow(p / pK, (g - 1.0) / (2.0 * g)) - 1.0);
    };
    double p = 0.3;
    for (int it = 0; it < 100; ++it) {
        const double f = fK(p, rL, pL, cL) + fK(p, rR, pR, cR);
        const double h = 1e-7 * p;
        const double df = (fK(p + h, rL, pL, cL) + fK(p + h, rR, pR, cR) - f) / h;
        const double pn = std::max(p - f / df, 1e-6);
        if (std::abs(pn - p) < 1e-14) {
            p = pn;
            break;
        }
        p = pn;
    }
    const double us = 0.5 * (fK(p, rR, pR, cR) - fK(p, rL, pL, cL));
    const double S = (x - 0.5) / t;
    const double rsL = rL * std::pow(p / pL, 1.0 / g);
    const double rsR = rR * ((p / pR + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * p / pR + 1.0));
    const double SR = cR * std::sqrt((g + 1.0) / (2.0 * g) * p / pR + (g - 1.0) / (2.0 * g));
    const double cs = cL * std::pow(p / pL, (g - 1.0) / (2.0 * g));
    if (S < -cL) return rL;
    if (S < us - cs) {
        const double c = 2.0 / (g + 1.0) * (cL - (g - 1.0) / 2.0 * S);
        return rL * std::pow(c / cL, 2.0 / (g - 1.0));
    }
    if (S < us) return rsL;
    if (S < SR) return rsR;
    return rR;
}

struct CellSetup {
    EOS eos = EOS::ideal(1.4);
    RiemannSolver rs;
    MusclReconstruction mu;
    BoundaryManager bm;
    CellSetup(bool periodic, Limiter lim = Limiter::VanLeer)
        : rs(eos, RiemannType::HLLC), mu(eos, lim), bm(generate_cartesian_mesh(1.0, 1.0, 4, 4), eos) {
        if (periodic)
            bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Periodic, BCType::Periodic);
        else
            bm.set_boundary_types(BCType::Transmissive, BCType::Transmissive, BCType::Wall, BCType::Wall);
    }
};

const char* const kSchemeNames[3] = {"Explicit Euler", "SSP Runge-Kutta 2", "Runge-Kutta 4"};

double cell_total(const CellMesh& m, int q) {
    double s = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c) s += m.U[c][q] * m.area(c);
    return s;
}

}   // namespace

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

// ---------------------------------------------------------------------------
// CellFluxEngine physics (viscous, forces, sources, immersed body, non-reflecting outlet)
// ---------------------------------------------------------------------------

namespace {

// A uniform-solver simulation and a level-0 cell mesh holding the same state, with the physics of the run.
struct ParityRig {
    Simulation sim;
    std::unique_ptr<CellMesh> cm;
    CellPhysics ph;
    std::unique_ptr<CellFluxEngine> eng;
    Field Uu;

    explicit ParityRig(std::initializer_list<std::pair<const char*, const char*>> kv, int max_level = 0) {
        sim.setup(cell_flux_params(kv), false);
        const Mesh& u = sim.mesh;
        ph.ns = sim.ns.get();
        ph.body = sim.body_force.get();
        if (sim.p.use_source) {
            ph.source = &sim.p.source;
            ph.source_unsteady = sim.p.source_unsteady;
            ph.source_mesh = &sim.mesh;
        }
        if (sim.p.use_ibm) ph.ibm_factory = sim.ibm_factory;
        const bool px = sim.boundaries.type(Face::Left) == BCType::Periodic;
        const bool py = sim.boundaries.type(Face::Bottom) == BCType::Periodic;
        cm = std::make_unique<CellMesh>(u.Lx, u.Ly, u.nx, u.ny, max_level, px, py, u.x0, u.y0);
        Uu.resize(static_cast<size_t>(u.num_cells()));
        for (int j = 0; j < u.ny; ++j)
            for (int i = 0; i < u.nx; ++i) Uu[j * u.nx + i] = test_state(sim.fluid, u.xc(i), u.yc(j));
        cm->U.resize(cm->num_leaves());
        for (int c = 0; c < cm->num_leaves(); ++c) cm->U[c] = Uu[cm->cj[c] * u.nx + cm->ci[c]];
        eng = std::make_unique<CellFluxEngine>(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        eng->set_physics(ph);
        eng->build(*cm);
    }

    // residual and S_max of both solvers at stage time t, equal bit for bit?
    bool residual_equal(double t = 0.0) {
        sim.scheme->stage_time = t;
        eng->stage_time = t;
        Field Ru, Rc;
        const double Su = sim.residual(Uu, Ru);
        const double Sc = eng->residual(*cm, cm->U, Rc);
        Field perm(Ru.size());
        const Mesh& u = sim.mesh;
        for (int c = 0; c < cm->num_leaves(); ++c) perm[c] = Ru[cm->cj[c] * u.nx + cm->ci[c]];
        return bits_equal(perm, Rc) && Su == Sc;
    }
};

}   // namespace

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

namespace {

// Initial state of a test case at the leaf centres (evaluated on the uniform mesh of each level, so a level-0 mesh
// gets exactly the uniform initial field).
void init_cell_case(CellMesh& cm, const Simulation& sim, const char* name) {
    cm.U.assign(static_cast<size_t>(cm.num_leaves()), State{});
    for (int l = 0; l <= cm.max_level; ++l) {
        if (cm.level_begin[l + 1] == cm.level_begin[l]) continue;
        const Mesh ml = generate_cartesian_mesh(cm.Lx, cm.Ly, cm.nx_at(l), cm.ny_at(l), cm.x0, cm.y0);
        const Field F = get_test_case(name).initialize(ml, sim.fluid, sim.p.source);
        for (int c = cm.level_begin[l]; c < cm.level_begin[l + 1]; ++c) cm.U[c] = F[cm.cj[c] * cm.nx_at(l) + cm.ci[c]];
    }
}

CellPhysics physics_of(Simulation& sim) {
    CellPhysics ph;
    ph.ns = sim.ns.get();
    ph.body = sim.body_force.get();
    if (sim.p.use_source) {
        ph.source = &sim.p.source;
        ph.source_unsteady = sim.p.source_unsteady;
        ph.source_mesh = &sim.mesh;
    }
    if (sim.p.use_ibm) ph.ibm_factory = sim.ibm_factory;
    return ph;
}

// Run the cell mesh to t_end with the scheme and CFL of the simulation.
void run_cell_mesh(CellMesh& cm, Simulation& sim, double t_end, int nsteps_max = 100000, int closure = 2) {
    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    eng.interpolate_jumps = closure == 1 || closure == 2;   // 0 injection, 1 interpolated coarse value, 2 = 1 + a second-order coarse side, 3 = injection + second-order coarse side
    eng.coarse_side_second_order = closure >= 2;
    eng.set_physics(physics_of(sim));
    eng.build(cm);
    std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(sim.p.time_scheme), sim.p.cfl, 1.0, 5, 1e-3);
    CellAdvance adv(cm, eng, *ts);
    Field R;
    double S = eng.residual(cm, cm.U, R), t = 0.0;
    for (int k = 0; k < nsteps_max && t < t_end; ++k) {
        const double dt = std::min(sim.p.cfl * std::min(cm.dx(0), cm.dy(0)) / S, t_end - t);
        S = adv.step(dt, t);
        t += dt;
    }
}

// L1 difference of the density of `cm` against a level-0 reference mesh `ref` (mean of the reference cells under a leaf)
double density_error(const CellMesh& cm, const CellMesh& ref) {
    std::vector<double> grid(static_cast<size_t>(ref.nx) * ref.ny);
    for (int c = 0; c < ref.num_leaves(); ++c) grid[static_cast<size_t>(ref.cj[c]) * ref.nx + ref.ci[c]] = ref.U[c][0];
    double err = 0.0, area = 0.0;
    for (int c = 0; c < cm.num_leaves(); ++c) {
        const int r = ref.nx / cm.nx_at(cm.level[c]);   // reference cells per leaf side
        double mean = 0.0;
        for (int j = 0; j < r; ++j)
            for (int i = 0; i < r; ++i) mean += grid[static_cast<size_t>(cm.cj[c] * r + j) * ref.nx + cm.ci[c] * r + i];
        mean /= r * r;
        err += std::abs(cm.U[c][0] - mean) * cm.area(c);
        area += cm.area(c);
    }
    return err / area;
}

}   // namespace

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

// ---------------------------------------------------------------------------
// Indicator and remesh (P5)
// ---------------------------------------------------------------------------

TEST(cellindicator_matches_a_structured_reference) {
    // Level-0 mesh: the indicator against a plain 4-neighbour implementation on the structured grid.
    const EOS eos = EOS::ideal(1.4);
    const int nx = 40, ny = 12;
    CellMesh m(1.0, 0.3, nx, ny, 0);
    std::mt19937 rng(3);
    std::vector<double> rho(static_cast<size_t>(nx) * ny);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            rho[static_cast<size_t>(j) * nx + i] = (i < 17 ? 1.0 : 0.4) + (j == 6 && i > 30 ? 0.35 : 0.0) + 0.001 * (rng() % 10);
    m.U.resize(m.num_leaves());
    for (int c = 0; c < m.num_leaves(); ++c)
        m.U[c] = eos.prim_to_cons({rho[static_cast<size_t>(m.cj[c]) * nx + m.ci[c]], 0.3, -0.1, 1.0});
    CellIndicator ind;
    ind.criterion = 0.1;
    const std::vector<double> got = ind.xi(m, m.U, eos);
    // reference
    std::vector<double> x(rho.size(), 0.0);
    const auto test = [&](int a, int b) {
        const double vmin = std::max(std::min(std::abs(rho[a]), std::abs(rho[b])), 1e-2);
        if (std::abs(rho[b] - rho[a]) / vmin >= 0.1) x[a] = x[b] = 1.0;
    };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            if (i + 1 < nx) test(j * nx + i, j * nx + i + 1);
            if (j + 1 < ny) test(j * nx + i, (j + 1) * nx + i);
        }
    for (int it = 0; it < 2; ++it) {
        std::vector<double> add(x.size(), 0.0);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const int a = j * nx + i;
                for (const int b : {i + 1 < nx ? a + 1 : -1, j + 1 < ny ? a + nx : -1})
                    if (b >= 0 && (x[a] > 0.05 || x[b] > 0.05)) add[a] += 0.1, add[b] += 0.1;
            }
        for (size_t k = 0; k < x.size(); ++k) x[k] += add[k];
    }
    double worst = 0.0;
    int flagged = 0;
    for (int c = 0; c < m.num_leaves(); ++c) {
        worst = std::max(worst, std::abs(got[c] - x[static_cast<size_t>(m.cj[c]) * nx + m.ci[c]]));
        flagged += got[c] >= ind.split;
    }
    CHECK(worst < 1e-12);
    CHECK(flagged > 12 && flagged < 150);   // the step columns with their ring, and the bump
}

TEST(cellindicator_diffusion_rule_at_level_jumps) {
    // Interfaces of a level: {a, b} two cells of the level (either one above the trigger spreads to both), {a, -1} a cell of
    // the level against a coarser leaf (only the cell decides and only it receives).
    CellIndicator ind;
    ind.diffusion = 1;
    const std::vector<std::array<int, 2>> ifaces = {{0, 1}, {1, 2}, {2, -1}, {3, -1}};
    std::vector<double> xi = {1.0, 0.0, 0.0, 0.0};
    ind.diffuse(xi, ifaces);
    CHECK_NEAR(xi[0], 1.1, 1e-12);
    CHECK_NEAR(xi[1], 0.1, 1e-12);   // cell 0 spreads to cell 1; cell 1 starts at 0, so nothing reaches cell 2 in this iteration
    CHECK(xi[2] == 0.0 && xi[3] == 0.0);
    xi = {0.0, 0.0, 1.0, 0.0};
    ind.diffuse(xi, ifaces);
    CHECK_NEAR(xi[2], 1.2, 1e-12);   // cell 2 spreads to cell 1 (pair) and to itself once more through {2, -1}
    CHECK_NEAR(xi[1], 0.1, 1e-12);
    CHECK(xi[0] == 0.0 && xi[3] == 0.0);
    xi = {0.0, 0.0, 0.0, 0.0};
    ind.diffuse(xi, ifaces);
    CHECK(xi == std::vector<double>({0.0, 0.0, 0.0, 0.0}));
}

TEST(riemann_batch_equals_scalar_bit_for_bit) {
    // face_flux_batch (AVX2 where the CPU has it) must give the bits of face_flux for every solver, both equations of state,
    // every axis, a batch length that is not a multiple of four, and the awkward states: densities and pressures under the
    // floor, negative internal energy, identical and mirrored states, speeds equal to zero to the last bits, signed zeros, NaN
    // (all region tests false: zero flux, a NaN wave speed). Two NaN count as equal.
    unsigned long long seed = 0x9E3779B97F4A7C15ull;
    const auto rnd = [&]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(seed >> 11) * (1.0 / 9007199254740992.0);
    };
    const auto same = [](double a, double b) {
        unsigned long long x, y;
        std::memcpy(&x, &a, 8);
        std::memcpy(&y, &b, 8);
        return x == y || (std::isnan(a) && std::isnan(b));
    };
    for (int kind = 0; kind < 2; ++kind) {
        const EOS eos = kind == 0 ? EOS::ideal(1.4) : EOS::stiffened(4.4, 6.0e8);
        const double p0 = kind == 0 ? 1.0 : 1.0e8;
        const int n = 4099;   // not a multiple of four: the tail runs the scalar code
        std::vector<State> UL(n), UR(n);
        std::vector<int> axis(n);
        for (int k = 0; k < n; ++k) {
            const bool strong = rnd() < 0.15;
            const double jump = strong ? 4.0 : 0.02;
            State a = {0.5 + rnd(), 2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0, p0 * (0.5 + rnd()), 2.0 * rnd() - 1.0};
            State b = a;
            b[0] *= 1.0 + jump * rnd();
            b[1] += jump * (rnd() - 0.5);
            b[2] += jump * (rnd() - 0.5);
            b[3] *= 1.0 + jump * rnd();
            b[4] += 0.1 * jump * (rnd() - 0.5);
            UL[k] = eos.prim_to_cons(a);
            UR[k] = eos.prim_to_cons(b);
            axis[k] = static_cast<int>(rnd() * 3.0) % 3;
        }
        // awkward states in the first entries
        int k = 0;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const auto put = [&](State l, State r, int ax) {
            UL[k] = l;
            UR[k] = r;
            axis[k] = ax;
            ++k;
        };
        for (int ax = 0; ax < 3; ++ax) {
            const State a = eos.prim_to_cons({1.0, 0.3, -0.2, p0, 0.1});
            put(a, a, ax);
            put(a, eos.prim_to_cons({1.0, -0.3, -0.2, p0, 0.1}), ax);
            put(eos.prim_to_cons({1.0, 0.0, 0.0, p0, 0.0}), eos.prim_to_cons({0.125, -0.0, 0.0, 0.1 * p0, -0.0}), ax);
            put({1e-20, 0.0, 0.0, 1e-20, 0.0}, {1.0, 0.0, 0.0, p0, 0.0}, ax);
            put({1.0, 5.0, 0.0, 0.1, 0.0}, {1.0, 5.0, 0.0, 0.1, 0.0}, ax);
            put({-1.0, 1.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, p0, 0.0}, ax);
            put({0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, ax);
            put({nan, 0.0, 0.0, p0, 0.0}, {1.0, 0.0, 0.0, p0, 0.0}, ax);
            put(a, {1.0, nan, 0.0, p0, 0.0}, ax);
            put(a, {1.0, 0.0, 0.0, nan, 0.0}, ax);
            for (int ulp = -3; ulp <= 3; ++ulp) {   // normal velocity equal to the sound speed to the ulp
                const double c = std::sqrt(eos.gamma * (p0 + (kind == 0 ? 0.0 : eos.p_inf)) / 1.0);
                double u = c;
                for (int q = 0; q < std::abs(ulp); ++q) u = std::nextafter(u, ulp < 0 ? 0.0 : 1e300);
                State W = {1.0, 0.0, 0.0, p0, 0.0};
                W[ax == 0 ? 1 : (ax == 1 ? 2 : 4)] = u;
                State Wr = W;
                Wr[3] *= 1.0 + 1e-3;
                put(eos.prim_to_cons(W), eos.prim_to_cons(Wr), ax);
                put(eos.prim_to_cons(Wr), eos.prim_to_cons(W), ax);
            }
        }
        for (const RiemannType type : {RiemannType::HLLC, RiemannType::LowMachHLLC, RiemannType::PreconditionedHLLC}) {
            const RiemannSolver rs(eos, type, 0.1);
            std::vector<State> Fb(n);
            std::vector<double> sb(n);
            rs.face_flux_batch(UL.data(), UR.data(), axis.data(), n, Fb.data(), sb.data());
            int bad = 0;
            for (int q = 0; q < n; ++q) {
                double ss;
                const State Fs = rs.face_flux(UL[q], UR[q], axis[q], ss);
                bool ok = same(ss, sb[q]);
                for (int c = 0; c < kNVar; ++c) ok = ok && same(Fs[c], Fb[q][c]);
                bad += ok ? 0 : 1;
            }
            CHECK(bad == 0);
            if (bad) std::printf("    eos %d solver %d: %d of %d faces differ\n", kind, static_cast<int>(type), bad, n);
        }
    }
}

TEST(cellindicator_cell_gather_diffusion_equals_the_interface_rule_bit_for_bit) {
    // diffuse_cells (gather per cell, parallel) must give the bits of diffuse (scatter over the interface list) on any table:
    // random same-level neighbours (also a cell twice, as on a 1-cell periodic axis, and itself), coarser sides and edges.
    unsigned seed = 12345u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) & 0xFFFFu;
    };
    for (int rep = 0; rep < 6; ++rep) {
        const int n = rep < 3 ? 40 : 6000;   // above kOmpMinWork the loops run in parallel
        std::vector<int> nbr(static_cast<size_t>(4) * n, -2);
        std::vector<std::array<int, 2>> ifaces;
        for (int c = 0; c < n; ++c)
            for (int s = 0; s < 4; ++s) {
                const unsigned r = rnd() % 8;
                if (r < 5) {
                    const int b = static_cast<int>(rnd() % static_cast<unsigned>(n));
                    nbr[static_cast<size_t>(4) * c + s] = b;
                } else if (r == 5) {
                    nbr[static_cast<size_t>(4) * c + s] = -1;
                }
            }
        // the interface list the table stands for: a same-level entry c -> b is an interface {c, b} that gives to both cells once
        // per listing, which is what each cell counts in its own row only for the pairs listed from its side.  The gather rule counts
        // a pair for a cell from the cell's own sides, so build the list from the rows of the lower index and mirror the rest.
        std::vector<int> sym(nbr.size(), -2);
        for (int c = 0; c < n; ++c)
            for (int s = 0; s < 4; ++s) {
                const int b = nbr[static_cast<size_t>(4) * c + s];
                if (b == -1) {
                    sym[static_cast<size_t>(4) * c + s] = -1;
                    ifaces.push_back({c, -1});
                }
            }
        // same-level pairs: make the table symmetric by construction (each listed pair occupies one free side in each cell)
        std::vector<int> used(static_cast<size_t>(n), 0);
        for (int c = 0; c < n; ++c)
            for (int s = 0; s < 4; ++s) {
                const int b = nbr[static_cast<size_t>(4) * c + s];
                if (b < 0 || used[static_cast<size_t>(c)] >= 4 || used[static_cast<size_t>(b)] >= 4) continue;
                int sc = -1, sb = -1;
                for (int q = 0; q < 4 && sc < 0; ++q)
                    if (sym[static_cast<size_t>(4) * c + q] == -2) sc = q;
                if (b == c) {
                    if (sc < 0) continue;
                    int sc2 = -1;
                    for (int q = sc + 1; q < 4 && sc2 < 0; ++q)
                        if (sym[static_cast<size_t>(4) * c + q] == -2) sc2 = q;
                    if (sc2 < 0) continue;
                    sym[static_cast<size_t>(4) * c + sc] = c;
                    sym[static_cast<size_t>(4) * c + sc2] = c;
                    ifaces.push_back({c, c});
                    used[static_cast<size_t>(c)] += 2;
                    continue;
                }
                for (int q = 0; q < 4 && sb < 0; ++q)
                    if (sym[static_cast<size_t>(4) * b + q] == -2) sb = q;
                if (sc < 0 || sb < 0) continue;
                sym[static_cast<size_t>(4) * c + sc] = b;
                sym[static_cast<size_t>(4) * b + sb] = c;
                ifaces.push_back({c, b});
                ++used[static_cast<size_t>(c)];
                ++used[static_cast<size_t>(b)];
            }
        for (int it = 1; it <= 3; ++it) {
            CellIndicator ind;
            ind.diffusion = it;
            ind.diffusion_coeff = 0.1;
            ind.diffusion_trigger = 0.05;
            std::vector<double> xa(static_cast<size_t>(n)), xb;
            for (int c = 0; c < n; ++c) xa[c] = (rnd() % 5 == 0) ? 1.0 : (rnd() % 7 == 0 ? 0.07 : 0.0);
            xb = xa;
            ind.diffuse(xa, ifaces);
            ind.diffuse_cells(xb, sym);
            CHECK(xa == xb);
        }
    }
}

TEST(cellflux_wave_speed_equals_the_residual_bound) {
    // the level-by-level pass returns the value residual() does (convective and viscous), on random adapted meshes
    for (int sc = 0; sc < 3; ++sc) {
        CellSetup su(true);
        CellMesh m = random_adapted_mesh(true, true, 90u + static_cast<unsigned>(sc), 6);
        CHECK(m.level_begin[3] < m.num_leaves());
        m.U.resize(m.num_leaves());
        for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state(su.eos, m.xc(c), m.yc(c));
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        eng.build(m);
        Field R;
        const double a = eng.residual(m, m.U, R);
        const double b = eng.wave_speed(m, m.U);
        CHECK(a == b);
        CHECK(a > 0.0);
    }
}

TEST(cellindicator_levels_see_only_their_own_interfaces) {
    // A density jump far inside the coarse region must not flag the fine leaves of a refined block (they only see their own
    // level's interfaces and those to a coarser leaf), and a jump next to the block flags the fine cells through the
    // coarse-fine interface.
    const EOS eos = EOS::ideal(1.4);
    CellMesh m(8.0, 8.0, 8, 8, 2);
    std::vector<char> r(static_cast<size_t>(m.num_leaves()), 0);
    r[m.find(0, 1, 1)] = 1;
    m.adapt(r);
    CellIndicator ind;
    m.U.resize(m.num_leaves());
    for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = eos.prim_to_cons({m.xc(c) > 5.0 ? 3.0 : 1.0, 0.0, 0.0, 1.0});
    CellIndicator::Result far = ind.evaluate(m, m.U, eos);
    double fine_xi = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c)
        if (m.level[c] == 1) fine_xi = std::max(fine_xi, far.xi[c]);
    CHECK(fine_xi == 0.0);   // the jump at x = 5 is three coarse cells away
    for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = eos.prim_to_cons({m.xc(c) > 2.0 ? 3.0 : 1.0, 0.0, 0.0, 1.0});
    CellIndicator::Result near = ind.evaluate(m, m.U, eos);
    double fine_near = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c)
        if (m.level[c] == 1 && m.xc(c) > 1.6) fine_near = std::max(fine_near, near.xi[c]);
    CHECK(fine_near >= 1.0);   // the jump sits on the interface of the block
    // the parent cell of the block at level 0 has its own xi: it is next to the jump too
    int parent_flagged = 0;
    for (int c = 0; c < m.num_leaves(); ++c)
        if (m.level[c] == 1 && near.xi_parent[c] >= 1.0) ++parent_flagged;
    CHECK(parent_flagged == 4);
}

TEST(cellremesh_a_marginal_ramp_reaches_a_fixed_point_with_the_parent_rule_and_flickers_without) {
    // A smooth density ramp whose variation per coarse face (0.14 at x = 0) is above the criterion 0.1 and per fine face
    // (0.07) below it. Expectation fixed beforehand: with the parent's xi the mesh stops changing within max_level + 1
    // remeshes of the static field and stays; with the children's xi it keeps changing (refine, join, refine ...).
    CellSetup su(false);
    const auto changes = [&](bool parent_rule, int remeshes, std::vector<int>& counts) {
        CellMesh m(1.0, 0.25, 32, 8, 2);
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        CellIndicator ind;
        ind.parent_join = parent_rule;
        CellAdaptor ad(m, eng, su.eos, ind);
        const auto fill = [&](CellMesh& mm) {
            mm.U.resize(mm.num_leaves());
            for (int c = 0; c < mm.num_leaves(); ++c) mm.U[c] = su.eos.prim_to_cons({1.0 + 4.5 * mm.xc(c), 0.0, 0.0, 1.0});
        };
        fill(m);
        eng.build(m);
        counts.clear();
        int n_changed = 0;
        for (int k = 0; k < remeshes; ++k) {
            const bool ch = ad.remesh();
            fill(m);   // the static field again, exactly, at the new leaves
            counts.push_back(m.num_leaves());
            if (ch) ++n_changed;
        }
        return n_changed;
    };
    std::vector<int> with_parent, with_children;
    const int ch_parent = changes(true, 10, with_parent);
    const int ch_children = changes(false, 10, with_children);
    std::printf("    marginal ramp, leaves after each of 10 remeshes: parent rule");
    for (int n : with_parent) std::printf(" %d", n);
    std::printf(" | children rule");
    for (int n : with_children) std::printf(" %d", n);
    std::printf("\n");
    CHECK(ch_parent <= 3);                                                   // max_level + 1 remeshes, then nothing
    CHECK(with_parent[4] == with_parent[9] && with_parent[3] == with_parent[9]);
    CHECK(ch_children >= 8);                                                 // the old rule keeps changing
    CHECK(with_children[8] != with_children[9] || with_children[7] != with_children[8]);
}

TEST(cellremesh_step_is_stable_conservative_and_deterministic) {
    CellSetup su(false);
    const auto build = [&](CellMesh& m, CellFluxEngine& eng, CellAdaptor& ad) {
        const auto initial = [&](CellMesh& mm) {
            mm.U.resize(mm.num_leaves());
            for (int c = 0; c < mm.num_leaves(); ++c)
                mm.U[c] = su.eos.prim_to_cons({mm.xc(c) < 0.5 ? 1.0 : 0.3, 0.2, 0.0, 1.0});
        };
        initial(m);
        eng.build(m);
        return ad.init_adapt(initial);
    };
    CellMesh m1(1.0, 0.25, 64, 16, 2), m2(1.0, 0.25, 64, 16, 2);
    CellFluxEngine e1(su.eos, su.rs, &su.mu, su.bm), e2(su.eos, su.rs, &su.mu, su.bm);
    CellIndicator ind;
    CellAdaptor a1(m1, e1, su.eos, ind), a2(m2, e2, su.eos, ind);
    const int p1 = build(m1, e1, a1), p2 = build(m2, e2, a2);
    CHECK(p1 >= 2 && p1 == p2);
    CHECK(m1.ci == m2.ci && m1.cj == m2.cj && m1.level == m2.level && m1.U == m2.U);   // deterministic
    CHECK(m1.num_leaves() > 64 * 16 && m1.num_leaves() < 256 * 64);
    // the finest level sits at the step (x = 0.5): its leaves are within the diffusion band of it
    int fine_far = 0;
    for (int c = 0; c < m1.num_leaves(); ++c)
        if (m1.level[c] == 2 && std::abs(m1.xc(c) - 0.5) > 0.1) ++fine_far;
    CHECK(fine_far == 0);
    // remeshing a static field again changes nothing, and the leaves stay bitwise the same
    const std::vector<int> ci = m1.ci, cj = m1.cj, lv = m1.level;
    const Field U = m1.U;
    CHECK(!a1.remesh());
    CHECK(m1.ci == ci && m1.cj == cj && m1.level == lv && m1.U == U);
    // the step moves: remeshing follows it and conserves the totals
    double before[4];
    for (int q = 0; q < 4; ++q) before[q] = cell_total(m1, q);
    for (int c = 0; c < m1.num_leaves(); ++c)
        m1.U[c] = su.eos.prim_to_cons({m1.xc(c) < 0.62 ? 1.0 : 0.3, 0.2, 0.0, 1.0});
    for (int q = 0; q < 4; ++q) before[q] = cell_total(m1, q);
    bool changed = false;
    for (int k = 0; k < 4; ++k) changed = a1.remesh() || changed;
    CHECK(changed);
    for (int q = 0; q < 4; ++q) CHECK_NEAR(cell_total(m1, q), before[q], 1e-12 * std::abs(before[q]));
    check_cell_mesh(m1);
}

TEST(cellremesh_keeps_the_immersed_body_on_the_finest_level) {
    ParityRig rig({{"nx", "36"}, {"ny", "24"}, {"use_ibm", "true"}, {"ibm_shape", "Cylinder"}, {"ibm_radius", "0.25"},
                   {"ibm_cx", "0.8"}, {"ibm_cy", "0.55"}, {"use_muscl", "true"}},
                  2);
    CellMesh& m = *rig.cm;
    CellIndicator ind;
    CellAdaptor ad(m, *rig.eng, rig.sim.fluid, ind);
    ad.set_body(rig.sim.ibm_factory, 2);
    const auto initial = [&](CellMesh& mm) {
        mm.U.assign(static_cast<size_t>(mm.num_leaves()), rig.sim.fluid.prim_to_cons({1.0, 0.5, 0.0, 1.0}));
    };
    ad.init_adapt(initial);
    const auto check_body = [&]() {
        const ImmersedBoundary fine = rig.sim.ibm_factory(generate_cartesian_mesh(m.Lx, m.Ly, m.nx_at(2), m.ny_at(2), m.x0, m.y0));
        int body_leaves = 0, coarse_leaves = 0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            const int r = 1 << (2 - m.level[c]);
            bool has_solid = false;
            for (int j = m.cj[c] * r; j < (m.cj[c] + 1) * r; ++j)
                for (int i = m.ci[c] * r; i < (m.ci[c] + 1) * r; ++i) has_solid = has_solid || fine.mask()[static_cast<size_t>(j) * m.nx_at(2) + i];
            if (has_solid) {
                ++body_leaves;
                CHECK(m.level[c] == 2);
            }
            coarse_leaves += m.level[c] == 0;
            if (rig.eng->solid()[c]) CHECK(m.level[c] == 2);
        }
        CHECK(body_leaves > 20);
        CHECK(coarse_leaves > 100);   // a uniform flow refines nothing else
    };
    check_body();
    CHECK(!ad.remesh());
    check_body();
    check_cell_mesh(m);
}

TEST(cellremesh_dynamic_sod_follows_the_waves) {
    // Bands fixed before the first run: base 50 + two levels with the indicator, remeshed every 4 coarse steps,
    // against the exact density: error within 1.6 x the uniform-200 one, a mean leaf count below 0.6 x 3200, the
    // shock region on the finest level at the end, mass conserved to 1e-12.
    CellSetup su(false);
    const auto init = [&](CellMesh& mm) {
        mm.U.resize(mm.num_leaves());
        for (int c = 0; c < mm.num_leaves(); ++c)
            mm.U[c] = mm.xc(c) < 0.5 ? su.eos.prim_to_cons({1.0, 0.0, 0.0, 1.0}) : su.eos.prim_to_cons({0.125, 0.0, 0.0, 0.1});
    };
    const auto run = [&](int nx, int max_level, double& mean_leaves, int& shock_level) {
        CellMesh m(1.0, 0.08, nx, std::max(4, nx / 12), max_level);
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        CellIndicator ind;
        CellAdaptor ad(m, eng, su.eos, ind);
        init(m);
        eng.build(m);
        if (max_level > 0) ad.init_adapt(init);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4, 1.0, 5, 1e-3);
        CellAdvance adv(m, eng, *ts);
        const double mass0 = cell_total(m, 0);
        Field R;
        double S = eng.residual(m, m.U, R), t = 0.0, leaf_sum = 0.0;
        int steps = 0;
        while (t < 0.2) {
            const double dt = std::min(0.4 * m.dx(0) / S, 0.2 - t);
            S = adv.step(dt, t);
            t += dt;
            ++steps;
            leaf_sum += m.num_leaves();
            if (max_level > 0 && steps % 4 == 0 && ad.remesh()) S = eng.residual(m, m.U, R);   // the speed bound follows the new leaves
        }
        CHECK_NEAR(cell_total(m, 0), mass0, 1e-12 * mass0);
        mean_leaves = leaf_sum / steps;
        shock_level = 0;
        for (int c = 0; c < m.num_leaves(); ++c)
            if (std::abs(m.xc(c) - 0.8528) < 0.02) shock_level = std::max(shock_level, m.level[c]);
        double err = 0.0, area = 0.0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            err += std::abs(m.U[c][0] - sod_exact_density(m.xc(c), 0.2)) * m.area(c);
            area += m.area(c);
        }
        return err / area;
    };
    double leaves_c = 0, leaves_f = 0, leaves_a = 0;
    int sl_c = 0, sl_f = 0, sl_a = 0;
    const double e_coarse = run(50, 0, leaves_c, sl_c), e_fine = run(200, 0, leaves_f, sl_f), e_amr = run(50, 2, leaves_a, sl_a);
    std::printf("    dynamic Sod L1(rho): uniform 50 %.4e, uniform 200 %.4e, adaptive %.4e (mean %.0f leaves, shock level %d)\n",
                e_coarse, e_fine, e_amr, leaves_a, sl_a);
    CHECK(e_amr < 1.6 * e_fine);
    CHECK(e_amr < e_coarse);
    CHECK(leaves_a < 0.6 * 3200);
    CHECK(sl_a == 2);
}

TEST(cellremesh_sod_indicator_study) {
    // How the indicator settings change the adaptive Sod run (base 50 + 2 levels, remesh every 4 steps): L1(rho) against the
    // exact solution and the mean number of leaves; uniform 50 and 200 for scale. Prints a table; the guards only catch breakage.
    CellSetup su(false);
    const auto init = [&](CellMesh& mm) {
        mm.U.resize(mm.num_leaves());
        for (int c = 0; c < mm.num_leaves(); ++c)
            mm.U[c] = mm.xc(c) < 0.5 ? su.eos.prim_to_cons({1.0, 0.0, 0.0, 1.0}) : su.eos.prim_to_cons({0.125, 0.0, 0.0, 0.1});
    };
    const auto run = [&](const CellIndicator& ind, int nx, int max_level, double& mean_leaves) {
        CellMesh m(1.0, 0.08, nx, std::max(4, nx / 12), max_level);
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        CellAdaptor ad(m, eng, su.eos, ind);
        init(m);
        eng.build(m);
        if (max_level > 0) ad.init_adapt(init);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4, 1.0, 5, 1e-3);
        CellAdvance adv(m, eng, *ts);
        Field R;
        double S = eng.residual(m, m.U, R), t = 0.0, sum = 0.0;
        int steps = 0;
        while (t < 0.2) {
            const double dt = std::min(0.4 * m.dx(0) / S, 0.2 - t);
            S = adv.step(dt, t);
            t += dt;
            ++steps;
            sum += m.num_leaves();
            if (max_level > 0 && steps % 4 == 0 && ad.remesh()) S = eng.residual(m, m.U, R);   // the speed bound follows the new leaves
        }
        mean_leaves = sum / steps;
        double err = 0.0, area = 0.0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            err += std::abs(m.U[c][0] - sod_exact_density(m.xc(c), 0.2)) * m.area(c);
            area += m.area(c);
        }
        return err / area;
    };
    double leaves = 0.0;
    CellIndicator base;
    std::printf("    Sod indicator study (L1 rho, mean leaves): uniform 50 %.3e, uniform 200 %.3e\n", run(base, 50, 0, leaves), run(base, 200, 0, leaves));
    for (const double crit : {0.02, 0.05, 0.1})
        for (const int vars : {0, 1, 2}) {   // density; density + pressure; density + velocity
            CellIndicator ind;
            ind.criterion = crit;
            ind.use_pressure = vars == 1;
            ind.use_velocity = vars == 2;
            const double e = run(ind, 50, 2, leaves);
            std::printf("      criterion %.2f, %-16s: %.3e, %6.0f leaves\n", crit, vars == 0 ? "density" : (vars == 1 ? "density+pressure" : "density+velocity"), e, leaves);
            CHECK(e > 0.0 && e < 0.02);
        }
}

// ---------------------------------------------------------------------------
// Distributed run (P6): ranks emulated by threads
// ---------------------------------------------------------------------------

namespace {

// In-process stand-in for MPI: `size` threads call the same collectives in the same order.
class FakeHub {
public:
    explicit FakeHub(int n) : n_(n), mail_(static_cast<size_t>(n), std::vector<std::vector<double>>(static_cast<size_t>(n))),
                              slot_(static_cast<size_t>(n)), dslot_(static_cast<size_t>(n), 0.0) {}

    CellComm comm(int rank) {
        CellComm c;
        c.rank = rank;
        c.size = n_;
        c.allgatherv = [this, rank](const double* send, int n, double* recv, const int* counts, const int* displs) {
            slot_[static_cast<size_t>(rank)].assign(send, send + n);
            barrier();
            for (int r = 0; r < n_; ++r) {
                if (static_cast<int>(slot_[static_cast<size_t>(r)].size()) != counts[r]) throw std::logic_error("allgatherv: counts differ");
                std::copy(slot_[static_cast<size_t>(r)].begin(), slot_[static_cast<size_t>(r)].end(), recv + displs[r]);
            }
            barrier();
        };
        c.allreduce_max = [this, rank](double v) {
            dslot_[static_cast<size_t>(rank)] = v;
            barrier();
            double g = dslot_[0];
            for (int r = 1; r < n_; ++r) g = std::max(g, dslot_[static_cast<size_t>(r)]);
            barrier();
            return g;
        };
        c.exchange = [this, rank](const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv) {
            for (int r = 0; r < n_; ++r) mail_[static_cast<size_t>(rank)][static_cast<size_t>(r)] = send[static_cast<size_t>(r)];
            barrier();
            bool bad = false;
            for (int r = 0; r < n_; ++r) {
                const std::vector<double>& in = mail_[static_cast<size_t>(r)][static_cast<size_t>(rank)];
                if (in.size() != recv[static_cast<size_t>(r)].size()) bad = true;   // the lists of the two ranks disagree
                else recv[static_cast<size_t>(r)] = in;
            }
            barrier();
            if (bad) throw std::logic_error("exchange: a message does not match what the receiver expects");
        };
        return c;
    }

private:
    void barrier() {
        std::unique_lock<std::mutex> lock(m_);
        const int gen = gen_;
        if (++count_ == n_) {
            count_ = 0;
            ++gen_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [&] { return gen != gen_; });
        }
    }
    int n_;
    std::mutex m_;
    std::condition_variable cv_;
    int count_ = 0, gen_ = 0;
    std::vector<std::vector<std::vector<double>>> mail_;
    std::vector<std::vector<double>> slot_;
    std::vector<double> dslot_;
};

struct DistSpec {
    const char* name;
    bool px, py;
    BCType bc[4];   // left, right, bottom, top (periodic axes: both sides periodic)
    bool viscous, force, source, unsteady_source, body;
    unsigned seed;
    bool remesh;
};

// Everything one rank (or the serial reference) needs, built from the spec.
struct DistEnv {
    EOS eos = EOS::ideal(1.4);
    RiemannSolver rs;
    MusclReconstruction mu;
    BoundaryManager bm;
    NavierStokes ns;
    ConstantBodyForce bf;
    Params src;
    Mesh base;
    CellPhysics ph;
    explicit DistEnv(const DistSpec& sp)
        : rs(eos, RiemannType::HLLC), mu(eos, Limiter::VanLeer), bm(generate_cartesian_mesh(1.5, 1.0, 6, 5), eos),
          ns(eos, 0.03, 0.72), bf{0.3, -0.5, 0.0}, base(generate_cartesian_mesh(1.5, 1.0, 6, 5)) {
        bm.set_boundary_types(sp.bc[0], sp.bc[1], sp.bc[2], sp.bc[3]);
        BCParams bp;
        bp.u_in = 0.6;
        bp.v_in = 0.1;
        bp.rho_in = 1.1;
        bp.p_out = 17.0;
        bm.set_params(bp);
        if (sp.viscous) ph.ns = &ns;
        if (sp.force) ph.body = &bf;
        if (sp.source) {
            src = baseline_defaults();
            src.set("source_rho", "0.1*sin(3*x)*cos(2*y)");
            src.set("source_mx", sp.unsteady_source ? "0.2*x*y*cos(t)" : "0.2*x*y");
            src.set("source_E", "0.5*cos(x+y)");
            ph.source = &src;
            ph.source_unsteady = sp.unsteady_source;
            ph.source_mesh = &base;
        }
        if (sp.body)
            ph.ibm_factory = [](const Mesh& m) { return ImmersedBoundary(m, "Cylinder", 0.18, 0.7, 0.5, 1.0, 17.0); };
    }
};

// Run the case for `steps` steps on `ranks` ranks (0: the serial path without CellParallel); returns the final state as every
// rank sees it after gather_state (all identical, checked), and the held / halo counts of the last rank.
Field run_distributed(const DistSpec& sp, int ranks, int steps, std::vector<int>* counts = nullptr) {
    CellMesh base = random_adapted_mesh(sp.px, sp.py, sp.seed, 6);
    const int Nranks = std::max(ranks, 1);
    FakeHub hub(Nranks);
    std::vector<Field> result(static_cast<size_t>(Nranks));
    std::vector<std::string> error(static_cast<size_t>(Nranks));
    std::vector<int> held(static_cast<size_t>(Nranks), 0), halo(static_cast<size_t>(Nranks), 0);
    const auto body = [&](int rank) {
        try {
            DistEnv env(sp);
            CellMesh m = base;
            m.U.resize(m.num_leaves());
            for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state(env.eos, m.xc(c), m.yc(c));
            CellFluxEngine eng(env.eos, env.rs, &env.mu, env.bm);
            eng.set_physics(env.ph);
            eng.build(m);
            CellIndicator ind;
            ind.criterion = 0.02;
            CellAdaptor ad(m, eng, env.eos, ind);
            if (sp.body) ad.set_body(env.ph.ibm_factory, 1);
            std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4, 1.0, 5, 1e-3);
            std::unique_ptr<CellParallel> par;
            if (ranks > 0) {
                par = std::make_unique<CellParallel>(m, eng, hub.comm(rank));
                par->poison = true;
            }
            CellAdvance adv(m, eng, *ts);
            adv.par = par.get();
            Field R;
            double S = eng.residual(m, m.U, R), t = 0.0;
            if (par) par->partition();
            for (int k = 0; k < steps; ++k) {
                const double dt = 0.4 * m.dx(0) / S;
                S = adv.step(dt, t);
                t += dt;
                if (sp.body) {
                    eng.solid_reset(m, m.U);
                    if (par) par->exchange_solid();
                }
                if (sp.remesh && (k + 1) % 2 == 0) {
                    if (par) par->gather_state();
                    const bool changed = ad.remesh();
                    if (changed) S = eng.residual(m, m.U, R);   // every leaf is valid here
                    if (par) par->partition();
                }
            }
            if (par) par->gather_state();
            result[static_cast<size_t>(rank)] = m.U;
            if (par) {
                held[static_cast<size_t>(rank)] = par->held_count();
                halo[static_cast<size_t>(rank)] = par->halo_count();
            }
        } catch (const std::exception& e) {
            error[static_cast<size_t>(rank)] = e.what();
        }
    };
    std::vector<std::thread> threads;
    for (int r = 0; r < Nranks; ++r) threads.emplace_back(body, r);
    for (std::thread& th : threads) th.join();
    for (int r = 0; r < Nranks; ++r)
        if (!error[static_cast<size_t>(r)].empty()) std::printf("    rank %d: %s\n", r, error[static_cast<size_t>(r)].c_str());
    for (int r = 1; r < Nranks; ++r) CHECK(bits_equal(result[0], result[static_cast<size_t>(r)]));   // all ranks agree
    if (counts) *counts = {held[static_cast<size_t>(Nranks - 1)], halo[static_cast<size_t>(Nranks - 1)]};
    return result[0];
}

}   // namespace

TEST(cellparallel_equals_serial_bitwise_for_any_rank_count) {
    const BCType per = BCType::Periodic;
    const DistSpec specs[] = {
        // name, px, py, bcs, viscous, force, source, unsteady, body, seed, remesh
        {"euler periodic", true, true, {per, per, per, per}, false, false, false, false, false, 11u, true},
        {"viscous + force + source periodic", true, true, {per, per, per, per}, true, true, true, false, false, 12u, true},
        {"unsteady source, mixed", true, false, {per, per, BCType::NoSlip, BCType::MovingWall}, true, false, true, true, false, 13u, false},
        {"walls, outlet, nrbc, viscous", false, false, {BCType::Wall, BCType::NonReflecting, BCType::NoSlip, BCType::Outlet}, true, false, false, false, false, 14u, true},
        {"inlet + mass flow", false, false, {BCType::MassflowInlet, BCType::Transmissive, BCType::Wall, BCType::VelInlet}, true, false, false, false, false, 15u, false},
        {"immersed body, viscous", false, false, {BCType::VelInlet, BCType::Outlet, BCType::Wall, BCType::Wall}, true, false, false, false, true, 16u, true},
    };
    for (const DistSpec& sp : specs) {
        const Field serial = run_distributed(sp, 0, 6);
        for (const int ranks : {1, 2, 3, 5, 7}) {
            std::vector<int> counts;
            const Field dist = run_distributed(sp, ranks, 6, &counts);
            const bool same = bits_equal(serial, dist);
            CHECK(same);
            if (!same) std::printf("    %s on %d ranks differs from the serial run\n", sp.name, ranks);
            if (ranks >= 5) CHECK(counts[0] < static_cast<int>(serial.size()));   // a rank does not hold everything
            if (ranks >= 2) CHECK(counts[1] > 0);                                  // and it does have a halo
        }
    }
}

TEST(cell_engine_result_does_not_depend_on_the_thread_count) {
    // Adapted mesh above kOmpMinWork (every parallel loop of the engine runs in parallel), physics on: the result of three
    // steps with 1 thread and with several threads is bit-identical (all reductions are maxima, no sums across threads).
#ifdef _OPENMP
    const EOS eos = EOS::ideal(1.4);
    NavierStokes ns(eos, 0.02, 0.72);
    ConstantBodyForce bf{0.2, -0.3, 0.0};
    Params src = baseline_defaults();
    src.set("source_rho", "0.05*sin(5*x)*cos(3*y)");
    src.set("source_E", "0.3*cos(x+2*y)");
    const Mesh base = generate_cartesian_mesh(1.0, 0.75, 64, 48);
    CellSetup su(true);
    CellPhysics ph;
    ph.ns = &ns;
    ph.body = &bf;
    ph.source = &src;
    ph.source_mesh = &base;
    const auto run = [&](int threads) {
        omp_set_num_threads(threads);
        CellMesh m(1.0, 0.75, 64, 48, 2, true, true);
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<char> r(static_cast<size_t>(m.num_leaves()), 0);
            for (int c = 0; c < m.num_leaves(); ++c) r[c] = std::abs(m.xc(c) - 0.4) < 0.15 && m.yc(c) < 0.5;
            m.adapt(r);
        }
        m.U.resize(m.num_leaves());
        for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state(su.eos, m.xc(c), m.yc(c));
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        eng.set_physics(ph);
        eng.build(m);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4, 1.0, 5, 1e-3);
        CellAdvance adv(m, eng, *ts);
        Field R;
        double S = eng.residual(m, m.U, R), t = 0.0;
        for (int k = 0; k < 3; ++k) {
            const double dt = 0.4 * m.dx(0) / S;
            S = adv.step(dt, t);
            t += dt;
        }
        return std::make_pair(m.U, m.num_leaves());
    };
    const auto one = run(1), many = run(std::max(2, std::min(8, omp_get_num_procs())));
    omp_set_num_threads(omp_get_num_procs());
    CHECK(one.second > 4096 && one.second == many.second);
    CHECK(bits_equal(one.first, many.first));
#endif
}

int main() {
    int failed_tests = 0;
    for (const auto& t : registry()) {
        const int before = g_fail;
        std::printf("[ RUN  ] %s\n", t.name);
        std::fflush(stdout);
        try {
            t.fn();
        } catch (const std::exception& e) {
            ++g_fail;
            std::printf("    EXCEPTION: %s\n", e.what());
        }
        const bool ok = g_fail == before;
        failed_tests += ok ? 0 : 1;
        std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", t.name);
    }
    std::printf("\n%zu tests, %d checks, %d failed checks, %d failed tests\n", registry().size(), g_checks, g_fail,
                failed_tests);
    return failed_tests == 0 ? 0 : 1;
}
