// Unit tests for the C++ solver.
// Self-contained harness: run `fenice_tests` (exit code = number of failed checks).
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

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

#define TEST(name)                                   \
    static void name();                              \
    static Registrar reg_##name(#name, &name);       \
    static void name()

#define CHECK(cond)                                                                    \
    do {                                                                               \
        ++g_checks;                                                                    \
        if (!(cond)) { ++g_fail; std::printf("    FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); } \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                          \
    do {                                                                               \
        ++g_checks;                                                                    \
        const double va_ = (a), vb_ = (b);                                             \
        if (!(std::abs(va_ - vb_) <= (tol))) {                                         \
            ++g_fail;                                                                  \
            std::printf("    FAIL %s:%d  %s = %.15g, expected %.15g (tol %g)\n", __FILE__, __LINE__, #a, va_, vb_, (double)(tol)); \
        }                                                                              \
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

double total(const AMRTree& tree, int q) {
    double s = 0.0;
    for (const AMRPatch* lf : tree.leaves())
        for (const State& c : lf->U) s += c[q] * lf->mesh.cell_area();
    return s;
}

double max_abs(const Field& R) {
    double m = 0.0;
    for (const State& r : R)
        for (double x : r) m = std::max(m, std::abs(x));
    return m;
}

}  // namespace

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
        const State Wg = eos.cons_to_prim(side == 0 ? UL[m.boundary_face(Face::Left, 1)]
                                                    : UR[m.boundary_face(Face::Right, 1)]);
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
    CHECK(parse_bc("Non-Reflecting Outlet") == BCType::NonReflecting && bc_name(BCType::NonReflecting) == "nonreflecting");

    // A tube open at both ends with p above p_out: the NRBC relaxes the pressure to p_out,
    // transmissive keeps it (perfectly non-reflecting, so nothing anchors the level)
    for (const char* bc : {"transmissive", "nonreflecting"}) {
        Simulation sim;
        sim.setup(params_for("Sod Shock Tube", {{"nx", "100"}, {"ny", "2"}, {"bc_left", bc}, {"bc_right", bc},
                                                {"p_out", "1.0"}, {"nrbc_sigma", "1.0"}, {"t_end", "30"}}),
                  false);
        sim.U = uniform_field(sim.fluid, sim.mesh.num_cells(), 1.0, 0.0, 0.0, 1.1);
        const ResidualFn res = [&](const Field& Uin, Field& R) { return sim.residual(Uin, R); };
        double t = 0.0;
        while (t < 30.0) t += sim.scheme->step(sim.U, res, sim.dt_length, t, 30.0).dt;
        const double p_mid = sim.fluid.cons_to_prim(sim.U[50])[3];
        std::printf("    %s: p after 30 acoustic times %.6f (p_out 1, start 1.1)\n", bc, p_mid);
        if (std::string(bc) == "transmissive") CHECK(std::abs(p_mid - 1.1) < 1e-9);
        else                                   CHECK(std::abs(p_mid - 1.0) < 0.01);
    }
}

TEST(airfoil_geometry) {
    // NACA 0012: symmetric, 12 % thick at 30 % chord, closed trailing edge
    const Polygon2D naca = naca4("0012");
    CHECK(naca_digits("NACA 0012") == "0012" && naca_digits("naca2412") == "2412" && naca_digits("Cylinder").empty());
    CHECK(naca.contains(0.3, 0.0) && naca.contains(0.3, 0.055) && !naca.contains(0.3, 0.065) && !naca.contains(-0.01, 0.0));
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
        if (q[0] < x_min) { x_min = q[0]; y_le = q[1]; }
        if (q[0] > x_max) { x_max = q[0]; y_te = q[1]; }
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
    { std::ofstream bad((dir / "bad.dat").string()); bad << "name\n0 0\n1 x\n"; }
    try { load_airfoil((dir / "bad.dat").string()); } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find("bad.dat:3") != std::string::npos;
    }
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
                sim.setup(params_for("Sod Shock Tube", {{"nx", "12"}, {"ny", "8"}, {"riemann_solver", rs},
                                                        {"use_muscl", muscl}, {"use_viscous", visc},
                                                        {"bc_left", "transmissive"}, {"bc_right", "transmissive"},
                                                        {"bc_bottom", "transmissive"}, {"bc_top", "transmissive"}}),
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
    CHECK(U[50 * 100 + 50][0] == 1.0);                                    // deep solid -> rho_ref
    CHECK_NEAR(U[50 * 100 + 50][3], 1.0 / 0.4, 1e-12);                    // p_ref / (gamma-1)
    CHECK(ImmersedBoundary(m, "cylinder", 0.2, 0.5, 0.5, 1, 1).mask() == ib.mask());   // case-insensitive
    bool threw = false;
    try { ImmersedBoundary(m, "Circle", 0.2, 0.5, 0.5, 1, 1); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);   // unknown shape
}

// ---------------------------------------------------------------------------
// AMR
// ---------------------------------------------------------------------------

TEST(amr_refine_derefine_conservative) {
    AMRTree tree(1.0, 1.0, 8, 8, 2);
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i) tree.root->U[j * 8 + i] = {1.0 + i + 3.0 * j, 0.1 * i, 0.2, 2.0 + j};
    const double m0 = total(tree, 0), e0 = total(tree, 3);
    tree.refine_patch(tree.root);
    CHECK(tree.leaves().size() == 4);
    CHECK_NEAR(total(tree, 0), m0, 1e-12);
    tree.refine_patch(tree.leaves()[0]);
    tree.enforce_2to1_balance();
    CHECK_NEAR(total(tree, 0), m0, 1e-12);
    tree.derefine_patch(tree.root->children[0]);
    tree.derefine_patch(tree.root);
    CHECK(tree.leaves().size() == 1);
    CHECK_NEAR(total(tree, 0), m0, 1e-12);
    CHECK_NEAR(total(tree, 3), e0, 1e-12);
    bool odd_rejected = false;
    try { AMRTree bad(1.0, 1.0, 5, 8, 1); } catch (const std::invalid_argument&) { odd_rejected = true; }
    CHECK(odd_rejected);
}

TEST(amr_2to1_balance) {
    AMRTree tree(1.0, 1.0, 4, 4, 3);
    tree.refine_patch(tree.root);
    AMRPatch* bl = tree.leaves()[0];
    tree.refine_patch(bl);
    tree.refine_patch(bl->children[3]);   // level 3 in the middle of the domain
    tree.enforce_2to1_balance();
    for (AMRPatch* p : tree.leaves())
        for (Face f : kAllFaces)
            for (AMRPatch* nb : tree.neighbour_leaves(p, f)) CHECK(std::abs(p->level - nb->level) <= 1);
}

TEST(amr_error_indicator) {
    AMRTree tree(1.0, 1.0, 8, 8, 1);
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i) tree.root->U[j * 8 + i] = {1.0 + 0.3 * i, 0, 0, 1};
    const AMRErrorIndicator ind;
    CHECK(ind.indicator(*tree.root) < 1e-12);   // linear ramp
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i) tree.root->U[j * 8 + i][0] = i < 4 ? 1.0 : 0.125;
    CHECK(ind.indicator(*tree.root) > 0.3);     // discontinuity
}

namespace {
SimParams amr_params(const char* scheme = "SSP Runge-Kutta 2", const char* bcs = "transmissive") {
    return params_for("Sod Shock Tube", {{"Lx", "1.0"}, {"Ly", "1.0"}, {"nx", "8"}, {"ny", "8"},
                                         {"bc_left", bcs}, {"bc_right", bcs}, {"bc_bottom", "wall"}, {"bc_top", "wall"},
                                         {"time_scheme", scheme}, {"use_amr", "true"}, {"amr_max_level", "2"},
                                         {"amr_patch_nx", "8"}, {"amr_patch_ny", "8"}});
}
AMRPatch* sorted_leaf(AMRTree& tree, int k) {
    auto lv = tree.leaves();
    std::sort(lv.begin(), lv.end(), [](AMRPatch* a, AMRPatch* b) {
        return a->ymin != b->ymin ? a->ymin < b->ymin : a->xmin < b->xmin;
    });
    return lv[k];
}
}  // namespace

TEST(amr_sample_leaf_state) {
    AMRTree tree(1.0, 1.0, 4, 4, 1);
    tree.root->U.assign(16, State{0, 0, 0, 0});
    for (int c = 0; c < 16; ++c) tree.root->U[c][0] = c;
    const State* s = sample_leaf_state(tree.leaves(), 2.5 / 4, 1.5 / 4);
    CHECK(s && (*s)[0] == 6.0);
    CHECK(sample_leaf_state(tree.leaves(), -0.1, 0.5) == nullptr);
}

TEST(amr_patch_coupling) {
    Simulation sim;
    sim.setup(amr_params(), false);
    AMRTree& tree = *sim.amr_tree;
    tree.refine_patch(tree.root);
    AMRPatch *bl = sorted_leaf(tree, 0), *br = sorted_leaf(tree, 1), *tl = sorted_leaf(tree, 2), *tr = sorted_leaf(tree, 3);
    const int n = bl->mesh.num_cells();
    // Uniform state everywhere -> zero residual across internal faces
    for (AMRPatch* p : tree.leaves()) p->U = uniform_field(sim.fluid, n, 1.0, 0, 0, 1.0);
    Field R, fx, fy;
    sim.patch_residual(*bl, tree, bl->U, R, fx, fy);
    CHECK(max_abs(R) < 1e-8);
    // High | low across the shared vertical face: only that column is driven
    br->U = uniform_field(sim.fluid, n, 0.125, 0, 0, 0.1);
    tr->U = br->U;
    (void)tl;
    sim.patch_residual(*bl, tree, bl->U, R, fx, fy);
    double right_col = 0.0, left_col = 0.0;
    for (int j = 0; j < bl->ny(); ++j) {
        right_col = std::max(right_col, std::abs(R[j * bl->nx() + bl->nx() - 1][1]));
        left_col  = std::max(left_col, max_abs(Field{R[j * bl->nx()]}));
    }
    CHECK(right_col > 1e-8);
    CHECK(left_col < 1e-8);
    // Coarse-fine: refine br; bl must still couple to the finer side
    tree.refine_patch(br);
    tree.enforce_2to1_balance();
    for (AMRPatch* p : tree.leaves())
        if (p != bl) p->U = uniform_field(sim.fluid, n, 0.125, 0, 0, 0.1);
    sim.patch_residual(*bl, tree, bl->U, R, fx, fy);
    right_col = 0.0;
    for (int j = 0; j < bl->ny(); ++j) right_col = std::max(right_col, std::abs(R[j * bl->nx() + bl->nx() - 1][1]));
    CHECK(right_col > 1e-8);
}

TEST(amr_ghost_prolongation_linear) {
    AMRTree tree(1.0, 1.0, 8, 8, 2);
    tree.refine_patch(tree.root);
    AMRPatch* bl = sorted_leaf(tree, 0);
    AMRPatch* br = sorted_leaf(tree, 1);
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i) bl->U[j * 8 + i] = {1.0 + 2.0 * bl->mesh.yc(j), 0, 0, 2.5};
    tree.refine_patch(br);
    AMRPatch* child = br->children[0];   // bottom-left child abuts bl across its LEFT face
    const Field g = prolong_ghost_face(*child, *bl, Face::Left);
    for (int j = 0; j < child->ny(); ++j) {
        CHECK_NEAR(g[j][0], 1.0 + 2.0 * child->mesh.yc(j), 1e-9);
        if (j > 0) CHECK(std::abs(g[j][0] - g[j - 1][0]) > 1e-4);
    }
}

namespace {
void seed_blast(AMRTree& tree, const EOS& eos) {
    for (AMRPatch* p : tree.leaves())
        for (int j = 0; j < p->ny(); ++j)
            for (int i = 0; i < p->nx(); ++i) {
                const double dx = p->mesh.xc(i) - 0.45, dy = p->mesh.yc(j) - 0.25;
                p->U[j * p->nx() + i] = eos.prim_to_cons({1.0, 0, 0, 1.0 + 2.0 * std::exp(-(dx * dx + dy * dy) / 0.01)});
            }
}
void run_closed_box(double& m0, double& m1, double& e0, double& e1) {
    Simulation sim;
    sim.setup(amr_params("Explicit Euler", "wall"), false);
    AMRTree& tree = *sim.amr_tree;
    tree.refine_patch(tree.root);
    tree.refine_patch(sorted_leaf(tree, 0));
    tree.enforce_2to1_balance();
    seed_blast(tree, sim.fluid);
    m0 = total(tree, 0); e0 = total(tree, 3);
    const PatchResidualFn pr = [&](AMRPatch& p, AMRTree& t, const Field& U, Field& R, Field& fx, Field& fy) {
        return sim.patch_residual(p, t, U, R, fx, fy);
    };
    double finest = 1e9, S0 = 0.0;
    Field R, fx, fy;
    for (AMRPatch* lf : tree.leaves()) {
        finest = std::min(finest, lf->mesh.Lx / lf->nx());
        S0 = std::max(S0, sim.patch_residual(*lf, tree, lf->U, R, fx, fy));
    }
    const double dt = 0.3 * finest / (S0 + 1e-12);
    double t = 0.0;
    for (int k = 0; k < 15; ++k) { advance_amr_level(tree, 0, dt, t, 1e9, pr, *sim.scheme); t += dt; }
    m1 = total(tree, 0); e1 = total(tree, 3);
}
}  // namespace

TEST(amr_reflux_conservation) {
    double m0, m1, e0, e1;
    run_closed_box(m0, m1, e0, e1);
    CHECK(std::abs(m1 - m0) / m0 < 1e-10);
    CHECK(std::abs(e1 - e0) / e0 < 1e-10);
    // Negative control: without refluxing the coarse-fine seam leaks mass
    g_amr_disable_reflux = true;
    run_closed_box(m0, m1, e0, e1);
    g_amr_disable_reflux = false;
    CHECK(std::abs(m1 - m0) / m0 > 1e-9);
}

// ---------------------------------------------------------------------------
// AMR with IBM / viscous terms / body force
// ---------------------------------------------------------------------------

namespace {
// Smooth, non-trivial state with nonzero velocity everywhere (also inside the body)
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
}  // namespace

TEST(amr_root_patch_matches_uniform_residual) {
    // One root patch (no interfaces) must reproduce the uniform-mesh residual: IBM forcing,
    // viscous flux, MUSCL and the physical BCs all go through the AMR kernel.
    for (const char* gy : {"0.0", "-0.3"}) {
        const bool gravity = std::string(gy) != "0.0";
        Simulation uni, amr;
        uni.setup(airfoil_params({{"nx", "192"}, {"ny", "96"}, {"use_bodyforce", gravity ? "true" : "false"}, {"gy", gy}}), false);
        amr.setup(airfoil_params({{"nx", "192"}, {"ny", "96"}, {"use_bodyforce", gravity ? "true" : "false"}, {"gy", gy},
                                  {"use_amr", "true"}, {"amr_patch_nx", "192"}, {"amr_patch_ny", "96"},
                                  {"amr_max_level", "0"}}), false);
        uni.initialize_field();
        amr.initialize_field();
        CHECK(uni.ibm->solid_count() > 20);
        AMRPatch& root = *amr.amr_tree->root;
        for (int j = 0; j < uni.mesh.ny; ++j)
            for (int i = 0; i < uni.mesh.nx; ++i)
                uni.U[j * uni.mesh.nx + i] = root.U[j * uni.mesh.nx + i] = test_state(uni.fluid, uni.mesh.xc(i), uni.mesh.yc(j));
        Field Ru, Ra, fx, fy;
        const double Su = uni.residual(uni.U, Ru);
        const double Sa = amr.patch_residual(root, *amr.amr_tree, root.U, Ra, fx, fy);
        double d = 0.0, m = 0.0;
        for (size_t c = 0; c < Ru.size(); ++c)
            for (int q = 0; q < 4; ++q) { d = std::max(d, std::abs(Ru[c][q] - Ra[c][q])); m = std::max(m, std::abs(Ru[c][q])); }
        CHECK(m > 1e-3);
        CHECK(d <= 1e-12 * m);
        CHECK_NEAR(Sa, Su, 1e-12 * Su);
        // Solid reset: identical to the uniform one
        uni.ibm->apply_solid_reset(uni.U, uni.fluid);
        amr.amr_solid_reset(*amr.amr_tree);
        double dr = 0.0;
        for (size_t c = 0; c < uni.U.size(); ++c)
            for (int q = 0; q < 4; ++q) dr = std::max(dr, std::abs(uni.U[c][q] - root.U[c][q]));
        CHECK(dr < 1e-14);
    }
}

TEST(amr_ibm_masks_and_reset_across_patches) {
    // The body straddles the patch interfaces: the patch masks must tile the uniform mask, and the
    // reset (fringe cells read across interfaces) must not depend on how the domain is split.
    Simulation uni, sim;
    uni.setup(airfoil_params({{"nx", "96"}, {"ny", "48"}, {"ibm_cx", "6.0"}, {"ibm_cy", "3.0"}}), false);
    sim.setup(airfoil_params({{"nx", "96"}, {"ny", "48"}, {"ibm_cx", "6.0"}, {"ibm_cy", "3.0"}, {"use_amr", "true"},
                              {"amr_patch_nx", "48"}, {"amr_patch_ny", "24"}, {"amr_max_level", "1"}, {"amr_refine_ibm", "false"}}), false);
    uni.initialize_field();
    sim.initialize_field();
    AMRTree& tree = *sim.amr_tree;
    CHECK(tree.roots.size() == 4);   // four 48 x 24 root patches: the 96 x 48 uniform grid
    int solid = 0;
    for (AMRPatch* lf : tree.leaves()) {
        sim.ensure_patch_masks(*lf);
        const int i0 = static_cast<int>(std::lround(lf->xmin / uni.mesh.dx));
        const int j0 = static_cast<int>(std::lround(lf->ymin / uni.mesh.dy));
        for (int j = 0; j < lf->ny(); ++j)
            for (int i = 0; i < lf->nx(); ++i) {
                CHECK(lf->solid[j * lf->nx() + i] == uni.ibm->mask()[(j0 + j) * uni.mesh.nx + i0 + i]);
                solid += lf->solid[j * lf->nx() + i];
                lf->U[j * lf->nx() + i] = test_state(uni.fluid, lf->mesh.xc(i), lf->mesh.yc(j));
                uni.U[(j0 + j) * uni.mesh.nx + i0 + i] = lf->U[j * lf->nx() + i];
            }
    }
    CHECK(solid == uni.ibm->solid_count());
    uni.ibm->apply_solid_reset(uni.U, uni.fluid);
    sim.amr_solid_reset(tree);
    double d = 0.0;
    for (AMRPatch* lf : tree.leaves()) {
        const int i0 = static_cast<int>(std::lround(lf->xmin / uni.mesh.dx));
        const int j0 = static_cast<int>(std::lround(lf->ymin / uni.mesh.dy));
        for (int j = 0; j < lf->ny(); ++j)
            for (int i = 0; i < lf->nx(); ++i)
                for (int q = 0; q < 4; ++q)
                    d = std::max(d, std::abs(lf->U[j * lf->nx() + i][q] - uni.U[(j0 + j) * uni.mesh.nx + i0 + i][q]));
    }
    CHECK(d < 1e-14);
}

TEST(amr_same_level_patches_match_uniform_residual) {
    // Four same-level patches tiling the uniform grid, the body on their common corner, with the
    // airfoil BCs (velocity inlet, non-reflecting outlet, walls): every patch residual equals the
    // uniform one, so interface ghosts, corner BC ghosts and solid ghosts are exact (1st order:
    // MUSCL has a 1st-order seam at patch interfaces).
    Simulation uni, sim;
    uni.setup(airfoil_params({{"nx", "96"}, {"ny", "48"}, {"ibm_cx", "6.0"}, {"ibm_cy", "3.0"}, {"use_muscl", "false"}}), false);
    sim.setup(airfoil_params({{"nx", "96"}, {"ny", "48"}, {"ibm_cx", "6.0"}, {"ibm_cy", "3.0"}, {"use_muscl", "false"},
                              {"use_amr", "true"}, {"amr_patch_nx", "48"}, {"amr_patch_ny", "24"}, {"amr_max_level", "1"}, {"amr_refine_ibm", "false"}}), false);
    uni.initialize_field();
    sim.initialize_field();
    AMRTree& tree = *sim.amr_tree;
    for (int j = 0; j < uni.mesh.ny; ++j)
        for (int i = 0; i < uni.mesh.nx; ++i)
            uni.U[j * uni.mesh.nx + i] = test_state(uni.fluid, uni.mesh.xc(i), uni.mesh.yc(j));
    for (AMRPatch* lf : tree.leaves()) {
        const int i0 = static_cast<int>(std::lround(lf->xmin / uni.mesh.dx));
        const int j0 = static_cast<int>(std::lround(lf->ymin / uni.mesh.dy));
        for (int j = 0; j < lf->ny(); ++j)
            for (int i = 0; i < lf->nx(); ++i) lf->U[j * lf->nx() + i] = uni.U[(j0 + j) * uni.mesh.nx + i0 + i];
    }
    Field Ru;
    const double Su = uni.residual(uni.U, Ru);
    double m = 0.0;
    for (const State& r : Ru) for (int q = 0; q < 4; ++q) m = std::max(m, std::abs(r[q]));
    double d = 0.0, Smax = 0.0;
    for (AMRPatch* lf : tree.leaves()) {
        Field R, fx, fy;
        Smax = std::max(Smax, sim.patch_residual(*lf, tree, lf->U, R, fx, fy));
        const int i0 = static_cast<int>(std::lround(lf->xmin / uni.mesh.dx));
        const int j0 = static_cast<int>(std::lround(lf->ymin / uni.mesh.dy));
        for (int j = 0; j < lf->ny(); ++j)
            for (int i = 0; i < lf->nx(); ++i)
                for (int q = 0; q < 4; ++q)
                    d = std::max(d, std::abs(R[j * lf->nx() + i][q] - Ru[(j0 + j) * uni.mesh.nx + i0 + i][q]));
    }
    CHECK(m > 1e-3);
    CHECK(d <= 1e-12 * m);
    CHECK_NEAR(Smax, Su, 1e-12 * Su);
}

TEST(amr_body_kept_at_max_level) {
    // Tiled root, airfoil: after seeding every leaf that touches the body is at max_level, the tree is
    // 2:1 balanced, and a remesh (which sees a uniform flow, eta = 0) never coarsens the body
    Simulation sim;
    sim.setup(airfoil_params({{"nx", "192"}, {"ny", "96"}, {"use_amr", "true"}, {"amr_patch_nx", "16"},
                              {"amr_patch_ny", "16"}, {"amr_max_level", "2"}}), false);
    sim.initialize_field();
    AMRTree& tree = *sim.amr_tree;
    auto check = [&] {
        int body = 0;
        for (AMRPatch* lf : tree.leaves()) {
            if (sim.patch_touches_body(*lf)) { CHECK(lf->level == tree.max_level); ++body; }
            for (Face f : kAllFaces)
                for (AMRPatch* nb : tree.neighbour_leaves(lf, f)) CHECK(std::abs(nb->level - lf->level) <= 1);
        }
        CHECK(body > 4);
    };
    check();
    std::vector<AMRPatch*> to_ref, to_der;
    for (int pass = 0; pass < 3; ++pass) {
        sim.amr_indicator->mark(tree, to_ref, to_der);
        for (AMRPatch* p : to_ref) if (p->is_leaf() && p->level < tree.max_level) tree.refine_patch(p);
        for (AMRPatch* par : to_der) {
            bool kids = !par->is_leaf();
            for (AMRPatch* c : par->children) kids = kids && c->is_leaf();
            if (kids) tree.derefine_patch(par);
        }
        tree.enforce_2to1_balance();
        check();
    }
}

namespace {
// Largest difference between a tiled (single-level) AMR run and the uniform solver after `nsteps` fixed-dt steps
double tiled_vs_uniform(const char* scheme, const char* muscl, int nsteps, bool ibm_visc) {
    std::vector<std::pair<const char*, const char*>> base = {
        {"nx", "96"}, {"ny", "48"}, {"time_scheme", scheme}, {"use_muscl", muscl}};
    auto make = [&](bool amr) {
        Params pr = baseline_defaults();
        if (ibm_visc) {
            pr.merge(get_test_case("IBM Airfoil").defaults);
            pr.set("test_case", "IBM Airfoil");
            pr.set("ibm_cx", "6.0");  pr.set("ibm_cy", "3.0");
        } else {
            pr.merge(get_test_case("Sod Shock Tube").defaults);
            pr.set("test_case", "Sod Shock Tube");
            pr.set("Lx", "12.0");  pr.set("Ly", "6.0");
        }
        pr.set("output_root", tmp_dir().string());
        pr.set("use_mpi", "false");
        for (const auto& e : base) pr.set(e.first, e.second);
        if (amr) {
            pr.set("use_amr", "true");  pr.set("amr_patch_nx", "48");  pr.set("amr_patch_ny", "24");
            pr.set("amr_max_level", "0");
        }
        return SimParams::from(pr);
    };
    Simulation uni, sim;
    uni.setup(make(false), false);  uni.initialize_field();
    sim.setup(make(true), false);   sim.initialize_field();
    AMRTree& tree = *sim.amr_tree;
    for (int j = 0; j < uni.mesh.ny; ++j)
        for (int i = 0; i < uni.mesh.nx; ++i)
            uni.U[j * uni.mesh.nx + i] = test_state(uni.fluid, uni.mesh.xc(i), uni.mesh.yc(j));
    auto scatter = [&] {
        for (AMRPatch* lf : tree.leaves()) {
            const int i0 = static_cast<int>(std::lround(lf->xmin / uni.mesh.dx));
            const int j0 = static_cast<int>(std::lround(lf->ymin / uni.mesh.dy));
            for (int j = 0; j < lf->ny(); ++j)
                for (int i = 0; i < lf->nx(); ++i) lf->U[j * lf->nx() + i] = uni.U[(j0 + j) * uni.mesh.nx + i0 + i];
        }
    };
    scatter();
    const PatchResidualFn pr = [&](AMRPatch& p, AMRTree& t, const Field& U, Field& R, Field& fx, Field& fy) {
        return sim.patch_residual(p, t, U, R, fx, fy);
    };
    const ResidualFn res = [&](const Field& Uin, Field& R) { return uni.residual(Uin, R); };
    const double dt = 2e-3;
    double t = 0.0;
    for (int k = 0; k < nsteps; ++k) {
        uni.scheme->advance_fixed_dt(uni.U, res, dt);
        if (ibm_visc) uni.ibm->apply_solid_reset(uni.U, uni.fluid);
        advance_amr_level(tree, 0, dt, t, 1e9, pr, *sim.scheme);
        if (ibm_visc) sim.amr_solid_reset(tree);
        t += dt;
    }
    double d = 0.0;
    for (AMRPatch* lf : tree.leaves()) {
        const int i0 = static_cast<int>(std::lround(lf->xmin / uni.mesh.dx));
        const int j0 = static_cast<int>(std::lround(lf->ymin / uni.mesh.dy));
        for (int j = 0; j < lf->ny(); ++j)
            for (int i = 0; i < lf->nx(); ++i)
                for (int q = 0; q < 4; ++q)
                    d = std::max(d, std::abs(lf->U[j * lf->nx() + i][q] - uni.U[(j0 + j) * uni.mesh.nx + i0 + i][q]));
    }
    return d;
}
}  // namespace

TEST(amr_tiled_matches_uniform_steps) {
    // A level-0 tiling must be the uniform solver: the same fixed-dt steps give the same state, for every
    // time scheme (stage-synchronised ghosts) with and without MUSCL (reconstruction across patch seams)
    for (const char* sch : {"Explicit Euler", "SSP Runge-Kutta 2", "Runge-Kutta 4"}) {
        CHECK(tiled_vs_uniform(sch, "false", 3, false) < 1e-15);
        CHECK(tiled_vs_uniform(sch, "true", 3, false) < 1e-15);
        CHECK(tiled_vs_uniform(sch, "true", 3, true) < 1e-12);   // airfoil: IBM, viscous, NRBC, solid reset
    }
}

namespace {
void run_viscous_box(double& e0, double& e1, double& m0, double& m1) {
    Simulation sim;
    sim.setup(params_for("Lid Driven Cavity",
                         {{"Lx", "1.0"}, {"Ly", "1.0"}, {"nx", "8"}, {"ny", "8"}, {"use_mpi", "false"},
                          {"bc_top", "no_slip"}, {"time_scheme", "Explicit Euler"}, {"use_muscl", "false"},
                          {"mu", "0.02"}, {"use_amr", "true"}, {"amr_max_level", "2"},
                          {"amr_patch_nx", "8"}, {"amr_patch_ny", "8"}}), false);
    AMRTree& tree = *sim.amr_tree;
    tree.refine_patch(tree.root);
    tree.refine_patch(sorted_leaf(tree, 0));
    tree.enforce_2to1_balance();
    for (AMRPatch* lf : tree.leaves())
        for (int j = 0; j < lf->ny(); ++j)
            for (int i = 0; i < lf->nx(); ++i) {
                const double x = lf->mesh.xc(i), y = lf->mesh.yc(j);
                lf->U[j * lf->nx() + i] = sim.fluid.prim_to_cons(
                    {1.0 + 0.2 * std::exp(-30 * ((x - 0.3) * (x - 0.3) + (y - 0.3) * (y - 0.3))),
                     0.5 * std::sin(3.14159 * x) * std::sin(2 * 3.14159 * y),
                     0.3 * std::sin(2 * 3.14159 * x) * std::sin(3.14159 * y), 1.0});
            }
    e0 = total(tree, 3);  m0 = total(tree, 0);
    const PatchResidualFn pr = [&](AMRPatch& p, AMRTree& t, const Field& U, Field& R, Field& fx, Field& fy) {
        return sim.patch_residual(p, t, U, R, fx, fy);
    };
    double finest = 1e9, S0 = 0.0;
    Field R, fx, fy;
    for (AMRPatch* lf : tree.leaves()) {
        finest = std::min(finest, lf->mesh.Lx / lf->nx());
        S0 = std::max(S0, sim.patch_residual(*lf, tree, lf->U, R, fx, fy));
    }
    const double dt = 0.3 * finest / (S0 + 1e-12);
    double t = 0.0;
    for (int k = 0; k < 20; ++k) { advance_amr_level(tree, 0, dt, t, 1e9, pr, *sim.scheme); t += dt; }
    e1 = total(tree, 3);  m1 = total(tree, 0);
}
}  // namespace

TEST(amr_viscous_reflux_conservation) {
    // Closed no-slip box: viscous work only heats the gas, so total energy and mass are conserved.
    // Only the viscous flux across the coarse-fine seams can break it, which refluxing must repair.
    double e0, e1, m0, m1;
    run_viscous_box(e0, e1, m0, m1);
    CHECK(std::abs(e1 - e0) / e0 < 1e-12);
    CHECK(std::abs(m1 - m0) / m0 < 1e-12);
    g_amr_disable_reflux = true;
    run_viscous_box(e0, e1, m0, m1);
    g_amr_disable_reflux = false;
    CHECK(std::abs(e1 - e0) / e0 > 1e-9);
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
            err = std::max(err, std::abs(Uf[j * 40 + i][0] - (1.0 + 2.0 * xf[i] - yf[j])));   // linear: exact (extrapolated too)
            err = std::max(err, std::abs(Uf[j * 40 + i][1] - 3.0));
        }
    CHECK(err < 1e-12);
    const Field Ur = restrict_patch(Uf, 40, 20, 20, 10);
    CHECK_NEAR(Ur[0][0], 0.25 * (Uf[0][0] + Uf[1][0] + Uf[40][0] + Uf[41][0]), 1e-14);
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
    CHECK(r4.s_max == 3.0);              // max over the stages
    CHECK_NEAR(r4.dt, 0.5 / (1.0 + 1e-10), 1e-12);   // dt from the first stage
    calls = 0;
    SSPRK2(0.5).step(U, res, 1.0, 0.0, 10.0);
    CHECK(calls == 2);
    // BDF2: invalid dt rejected, zero RHS converges in one iteration, dt == dt_phys
    bool threw = false;
    try { BDF2DualTime bad(0.5, 0.0); } catch (const std::invalid_argument&) { threw = true; }
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

TEST(amr_tree_structure) {
    AMRTree tree(2.0, 1.0, 4, 4, 2);
    CHECK(tree.leaves().size() == 1 && tree.root->level == 0);
    CHECK(tree.root->xmax == 2.0 && tree.root->ymax == 1.0);
    auto kids = tree.refine_patch(tree.root);
    CHECK(kids.size() == 4);
    double area = 0.0;
    for (AMRPatch* k : kids) { CHECK(k->level == 1); area += (k->xmax - k->xmin) * (k->ymax - k->ymin); }
    CHECK_NEAR(area, 2.0, 1e-14);
    bool threw = false;
    try { tree.refine_patch(tree.root); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);   // non-leaf
    tree.refine_patch(kids[1]);
    threw = false;
    try { tree.refine_patch(kids[1]->children[0]); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);   // at max_level
    // A coarse face sees the two finer leaves across it; domain edges have none
    CHECK(tree.neighbour_leaves(kids[0], Face::Right).size() == 2);
    CHECK(tree.neighbour_leaves(kids[0], Face::Left).empty());
    CHECK(tree.neighbour_leaves(kids[0], Face::Top).size() == 1);
    AMRTree no_refine(1.0, 1.0, 5, 5, 0);   // odd dims are fine without refinement
    CHECK(no_refine.leaves().size() == 1);
}

TEST(amr_marking_and_buffer) {
    AMRTree tree(1.0, 1.0, 8, 8, 2);
    tree.refine_patch(tree.root);
    for (AMRPatch* p : tree.leaves()) p->U.assign(64, State{1, 0, 0, 2.5});
    const AMRErrorIndicator ind(0.3, 0.1, 1);
    std::vector<AMRPatch*> ref, der;
    ind.mark(tree, ref, der);
    CHECK(ref.empty());
    CHECK(der.size() == 1 && der[0] == tree.root);   // uniform siblings collapse
    AMRPatch* bl = sorted_leaf(tree, 0);
    for (int j = 0; j < 8; ++j)
        for (int i = 0; i < 8; ++i) bl->U[j * 8 + i][0] = i < 4 ? 1.0 : 0.125;   // discontinuity
    ind.mark(tree, ref, der);
    CHECK(std::find(ref.begin(), ref.end(), bl) != ref.end());
    CHECK(ref.size() == 3);                           // + the two face neighbours (buffer ring)
    CHECK(der.empty());                               // guard: buffered child protects the parent
    ind.mark(tree, ref, der);
    const AMRErrorIndicator tight(0.3, 0.1, 0);
    tight.mark(tree, ref, der);
    CHECK(ref.size() == 1);
}

TEST(vtk_export_layout) {
    const fs::path dir = tmp_dir() / "vtk";
    fs::remove_all(dir);
    VTKExporter ex(dir.string(), false);
    CHECK(ex.collection.empty());
    const EOS eos = EOS::ideal(1.4);
    const Mesh m = generate_cartesian_mesh(1.0, 0.5, 4, 2);
    Field U = uniform_field(eos, m.num_cells(), 1.0, 0.1, 0.0, 1.0);
    U[3][0] = std::nan("");                            // must be sanitised, not crash
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
    // AMR: one .vtr per leaf in the step folder, only .vtm + .pvd at top level
    AMRTree tree(1.0, 1.0, 4, 4, 1);
    tree.root->U.assign(16, eos.prim_to_cons({1, 0, 0, 1}));
    tree.refine_patch(tree.root);
    const fs::path adir = tmp_dir() / "vtk_amr";
    fs::remove_all(adir);
    VTKExporter ea(adir.string(), false);
    ea.export_amr("amr", eos, tree, 0.0, 3);
    ea.write_pvd("amr");
    int top_vtr = 0, leaf_vtr = 0, tmp_files = 0;
    for (const auto& e : fs::recursive_directory_iterator(adir)) {
        if (e.path().extension() == ".tmp") ++tmp_files;
        if (e.path().extension() == ".vtr") (e.path().parent_path() == adir ? top_vtr : leaf_vtr) += 1;
    }
    CHECK(top_vtr == 0 && leaf_vtr == 4 && tmp_files == 0);
    CHECK(fs::exists(adir / "amr_00003.vtm") && ea.collection.size() == 1);
    std::ifstream lin((adir / "amr_00003" / "p0000.vtr").string());
    const std::string leaf((std::istreambuf_iterator<char>(lin)), std::istreambuf_iterator<char>());
    CHECK(leaf.find("Name=\"AMR_Level\"") != std::string::npos);
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
    a.save_checkpoint(2);              // overwrites atomically
    CHECK(!fs::exists(a.checkpoint_path().substr(0, a.checkpoint_path().size() - 4) + "_tmp.npz"));
    Simulation b;
    b.setup(params_for("Sod Shock Tube", {{"nx", "8"}, {"ny", "2"}, {"output_root", ck.c_str()}}), false);
    CHECK(b.load_checkpoint() == 2 && b.U[0][0] == 3.0);
    // Legacy hot-start file without metadata: needs manual dims
    const std::string legacy = (tmp_dir() / "legacy.npz").string();
    write_npz(legacy, {{"U", field_to_npy(a.U)}});
    bool threw = false;
    try { b.load_hot_start(legacy); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
    Simulation c;
    c.setup(params_for("Sod Shock Tube", {{"nx", "16"}, {"ny", "4"}, {"hot_start_src_nx", "8"}, {"hot_start_src_ny", "2"},
                                          {"output_root", ck2.c_str()}}), false);
    CHECK(c.load_hot_start(legacy).size() == 64);
}

TEST(solution_mapping_properties) {
    const EOS eos = EOS::ideal(1.4);
    Field U(6 * 4);
    for (int c = 0; c < 24; ++c) U[c] = eos.prim_to_cons({1.0 + 0.1 * (c % 6), 0.2, 0.0, 1.0 + 0.05 * (c / 6)});
    CHECK(map_solution(U, 6, 4, 6, 4, 1.0, 1.0) == U);                     // same grid: identity
    const Field coarse = map_solution(U, 6, 4, 3, 2, 1.0, 1.0);            // coarsening also works
    CHECK(coarse.size() == 6);
    const Field fine = map_solution(U, 6, 4, 24, 16, 1.0, 1.0);
    double m0 = 0, m1 = 0;
    bool positive = true;
    for (const State& c : U) m0 += c[0] / 24.0;
    for (const State& c : fine) { m1 += c[0] / 384.0; positive = positive && c[0] > 0 && eos.cons_to_prim(c)[3] > 0; }
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
    const SimParams sp = params_for("Isentropic Vortex", {{"nx", std::to_string(n).c_str()}, {"ny", std::to_string(n).c_str()}});
    Simulation init;
    init.setup(sp, false);
    const Field U = run_to(sp, get_test_case("Isentropic Vortex").initialize(init.mesh, init.fluid, init.p.source), t_end);
    double err = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            err += std::abs(U[j * n + i][0] -
                            isentropic_vortex_exact(init.mesh.xc(i), init.mesh.yc(j), t_end, 10.0, 10.0, 1.4)[0]);
    return err / (n * n);
}
}  // namespace

TEST(periodic_translation_invariance) {
    // A periodic domain has no special place: shifting the initial field by whole cells
    // must shift the solution exactly, with MUSCL and viscous terms on or off.
    const int nx = 16, ny = 12, sx = 5, sy = 3;
    for (const char* muscl : {"false", "true"})
        for (const char* visc : {"false", "true"}) {
            const SimParams sp = params_for("Taylor-Green Vortex", {{"nx", "16"}, {"ny", "12"}, {"use_muscl", muscl},
                                                                    {"use_viscous", visc}, {"mu", "0.05"}});
            Simulation s;
            s.setup(sp, false);
            Field U(nx * ny);
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const double x = 2 * kPi * (i + 0.5) / nx, y = 2 * kPi * (j + 0.5) / ny;
                    U[j * nx + i] = s.fluid.prim_to_cons({1.0 + 0.1 * std::sin(x + 0.3) * std::cos(y), 0.3 * std::sin(y + 1.0),
                                                          0.2 * std::cos(x - 0.5) + 0.1 * std::sin(2 * y), 100.0 + 2.0 * std::cos(x + y)});
                }
            const Field A = run_to(sp, U, 1e9, 6);
            const Field B = run_to(sp, roll(U, nx, ny, sx, sy), 1e9, 6);
            const Field Ar = roll(A, nx, ny, sx, sy);
            double diff = 0.0;
            for (size_t c = 0; c < B.size(); ++c)
                for (int q = 0; q < kNVar; ++q) diff = std::max(diff, std::abs(Ar[c][q] - B[c][q]) / (1.0 + std::abs(B[c][q])));
            std::printf("    muscl=%s viscous=%s: max shifted difference %.3e\n", muscl, visc, diff);
            CHECK(diff < 1e-13);
        }
}

TEST(taylor_green_viscous_decay) {
    // Kinetic energy decays as exp(-4 nu k^2 t); subtracting the decay of the same run
    // without viscous fluxes (numerical dissipation) must recover the physical nu.
    const double mu = 0.05, t_end = 1.0;
    const SimParams visc = params_for("Taylor-Green Vortex", {{"nx", "32"}, {"ny", "32"}, {"mu", "0.05"}});
    const SimParams invisc = params_for("Taylor-Green Vortex", {{"nx", "32"}, {"ny", "32"}, {"mu", "0.05"}, {"use_viscous", "false"}});
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
        e[k] = tgv_velocity_error(run_to(sp, get_test_case("Taylor-Green Vortex").initialize(s.mesh, s.fluid, s.p.source), t_end),
                                  s.mesh, s.fluid, t_end, mu);
    }
    const double o1 = std::log2(e[0] / e[1]), o2 = std::log2(e[1] / e[2]);
    std::printf("    errors %.3e %.3e %.3e, observed orders %.2f %.2f\n", e[0], e[1], e[2], o1, o2);
    CHECK(o2 > 1.7);
}

TEST(isentropic_vortex_convergence) {
    // Smooth inviscid advection: MUSCL + SSP-RK2 must converge at (close to) 2nd order
    const double e20 = vortex_density_error(20, 1.0), e40 = vortex_density_error(40, 1.0), e80 = vortex_density_error(80, 1.0);
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
}  // namespace

TEST(extrusion_matches_2d) {
    // A 2-D case extruded along a periodic z must reproduce the 2-D solution in every layer
    struct Case { const char* name; std::vector<std::pair<const char*, const char*>> ov; int steps; };
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
        p3.set("Lz", "30.0");   // deep cells: dz must not set the CFL step
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
    const Field A = run_to(p2, get_test_case("Taylor-Green Vortex").initialize(s2.mesh, s2.fluid, s2.p.source), 1e9, 12);
    for (int plane = 0; plane < 2; ++plane) {   // 0: x-z plane, 1: y-z plane
        const char* nx = plane == 0 ? "16" : "3";
        const char* ny = plane == 0 ? "3" : "16";
        const SimParams p3 = params_for("Taylor-Green Vortex 3D", {{"nx", nx}, {"ny", ny}, {"nz", "16"}, {"mu", "0.05"},
                                                                  {"riemann_solver", "HLLC (Low-Mach Thornber)"}});
        Simulation s3;
        s3.setup(p3, false);
        const Mesh& m = s3.mesh;
        Field U0(m.num_cells());
        for (int k = 0; k < m.nz; ++k)
            for (int j = 0; j < m.ny; ++j)
                for (int i = 0; i < m.nx; ++i) {
                    const int a = plane == 0 ? i : j;   // in-plane first coordinate
                    const State W = taylor_green_exact(m.xc(0) * 0 + (a + 0.5) * 2 * kPi / n, m.zc(k), 0.0, 2 * kPi, 0.05);
                    // 2-D (u, v) -> 3-D (u, w) for x-z, (v, w) for y-z
                    const State W3 = plane == 0 ? State{W[0], W[1], 0.0, W[3], W[2]} : State{W[0], 0.0, W[1], W[3], W[2]};
                    U0[m.cell(i, j, k)] = s3.fluid.prim_to_cons(W3);
                }
        const Field B = run_field_3d(p3, U0, 12, std::min(s2.mesh.dx, s2.mesh.dy));   // same steps as the 2-D run
        double diff = 0.0;
        for (int k = 0; k < n; ++k)
            for (int a = 0; a < n; ++a) {
                const State& u2 = A[k * n + a];
                const State expect = plane == 0 ? State{u2[0], u2[1], 0.0, u2[3], u2[2]} : State{u2[0], 0.0, u2[1], u2[3], u2[2]};
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
    const SimParams sp = params_for("Taylor-Green Vortex 3D", {{"nx", ns.c_str()}, {"ny", ns.c_str()}, {"nz", ns.c_str()},
                                                              {"mu", "0.05"}});
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
    const SimParams sp = params_for("Taylor-Green Vortex 3D", {{"nx", "12"}, {"ny", "10"}, {"nz", "8"}, {"mu", "0.05"}});
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
                for (int i = 0; i < nx; ++i) out[m.cell((i + sx) % nx, (j + sy) % ny, (k + sz) % nz)] = F[m.cell(i, j, k)];
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
            sim.setup(params_for("Taylor-Green Vortex 3D", {{"nx", "6"}, {"ny", "5"}, {"nz", "4"}, {"riemann_solver", rs},
                                                            {"use_viscous", visc}, {"bc_left", "transmissive"},
                                                            {"bc_right", "transmissive"}, {"bc_bottom", "transmissive"},
                                                            {"bc_top", "transmissive"}, {"bc_front", "transmissive"},
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
    const SimParams visc = params_for("Taylor-Green Vortex 3D", {{"nx", "24"}, {"ny", "24"}, {"nz", "24"}, {"mu", "0.01"}, {"t_end", "0.1"}});
    const SimParams inv = params_for("Taylor-Green Vortex 3D", {{"nx", "24"}, {"ny", "24"}, {"nz", "24"}, {"mu", "0.01"},
                                                               {"t_end", "0.1"}, {"use_viscous", "false"}});
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
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

const char* kMinimalCase = R"toml(
[case]
name = "Minimal"
[initial]
state = { rho = 1.0, p = 1.0 }
)toml";
}  // namespace

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
        } catch (const std::invalid_argument& e) {
            return e.what();
        }
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
    CHECK(error_of({{"use_amr", "true"}, {"resume", "true"}}).find("AMR") != std::string::npos);
    CHECK(error_of({{"use_amr", "true"}, {"nx", "100"}}).find("multiples") != std::string::npos);   // patch 8
    CHECK(error_of({{"use_amr", "true"}, {"nx", "96"}, {"ny", "96"}, {"bc_left", "periodic"}, {"bc_right", "periodic"}})
              .find("periodic") != std::string::npos);
    CHECK(error_of({{"use_amr", "true"}, {"nx", "96"}, {"ny", "96"}, {"time_scheme", "BDF2 Dual Time"}}).find("BDF2") != std::string::npos);
    CHECK(error_of({{"use_amr", "true"}, {"nx", "96"}, {"ny", "96"}}).empty());
    bool threw = false;
    try { parse_limiter("superbee"); } catch (const std::invalid_argument&) { threw = true; }
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
    CHECK(has(case_error("e4", base + "[[initial.region]]\nshape = \"sphear\"\ncenter = [0, 0]\nradius = 1\nstate = { rho = 2 }\n"),
              "did you mean 'sphere'"));
    CHECK(has(case_error("e5", base + "[[initial.region]]\nshape = \"box\"\nmin = [0, 0]\nstate = { rho = 2 }\n"),
              "a box needs 'min' and 'max'"));
    CHECK(has(case_error("e6", "[case]\nname = \"x\"\n[initial]\nstate = { rho = 1.0 }\n"), "at least rho and p"));
    CHECK(has(case_error("e7", "[case]\nname = \"x\"\n[initial]\nstate = { rho = \"1 + sin(xx)\", p = 1.0 }\n"),
              "did you mean 'x'"));
    CHECK(has(case_error("e8", "[case]\nname = \"x\"\n[initial]\nstate = { rh = 1.0, p = 1.0 }\n"), "did you mean 'rho'"));
    CHECK(has(case_error("e9", base + "[mesh]\nnx = = 3\n"), "case.toml:"));                // TOML syntax, with a line
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
    half.normal = {"0.0", "1.0"};        // y >= 0.8
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
            if (y >= 0.8) { W[3] = 5.0; W[2] = -1.0; }
            if ((x - 0.3) * (x - 0.3) + (y - 0.3) * (y - 0.3) <= 0.08 * 0.08) W[0] = 9.0;
            if ((x - 0.7) * (x - 0.7) + (y - 0.2) * (y - 0.2) <= (1.0 / 12) * (1.0 / 12)) W[0] = 7.0;
            const State E = eos.prim_to_cons(W);
            for (int q = 0; q < kNVar; ++q) ok = ok && U[j * 10 + i][q] == E[q];
        }
    CHECK(ok);
    // The circle and the cylinder each cover the 4 cells around their centre
    int n9 = 0, n7 = 0;
    for (const State& s : U) { n9 += s[0] == 9.0; n7 += s[0] == 7.0; }
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
        if (P[c] == base[c]) ++kept;
        else if (std::abs(eos.cons_to_prim(P[c])[0] - 0.5) < 1e-14) ++changed;
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
    const std::vector<std::string> names = {"Sod Shock Tube", "Poiseuille Flow", "Lid Driven Cavity",
                                            "IBM Flow over Cylinder", "IBM Airfoil", "IBM Airfoil AMR", "Shu-Osher Test",
                                            "IBM Supersonic Flow over Cylinder", "Taylor-Green Vortex",
                                            "Isentropic Vortex", "Taylor-Green Vortex 3D"};
    std::vector<std::string> got;
    for (const TestCase& tc : test_registry()) got.push_back(tc.key);
    CHECK(std::search(got.begin(), got.end(), names.begin(), names.end()) == got.begin());
    CHECK(get_test_case("Taylor-Green Vortex 3D").three_d);
    CHECK(get_test_case("Lid Driven Cavity").requires_viscosity);
    CHECK(!get_test_case("Sod Shock Tube").requires_viscosity);
    CHECK(&get_test_case("sod_shock_tube") == &get_test_case("Sod Shock Tube"));   // folder name
    bool threw = false;
    try { get_test_case("Sod Shok Tube"); } catch (const std::invalid_argument& e) {
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
    try { save_case(get_test_case("Lid Driven Cavity"), run, dir, "Saved cavity 2"); } catch (const std::runtime_error&) { refused = true; }
    CHECK(refused);
    CHECK(fs::exists(file));
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
