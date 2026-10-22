// Wall-time gate of the cell-based AMR: residual cost per cell of the face engine against the uniform
// Simulation::residual on the same physics, plus the cost of adapting and rebuilding the stencils.
//   bench_cellflux [nx ny reps]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"
#include "fv/Simulation.hpp"

using namespace fv;
using Clock = std::chrono::steady_clock;

namespace {

double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

State smooth_state(const EOS& eos, double x, double y) {
    return eos.prim_to_cons({1.0 + 0.1 * std::sin(1.3 * x) * std::cos(0.9 * y), 1.0 + 0.2 * std::sin(0.7 * y),
                             0.15 * std::cos(1.1 * x), 17.0 + 0.5 * std::cos(0.6 * x + 0.4 * y)});
}

}   // namespace

int main(int argc, char** argv) {
    const int nx = argc > 1 ? std::atoi(argv[1]) : 768, ny = argc > 2 ? std::atoi(argv[2]) : 384;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    Params p = baseline_defaults();
    p.merge(get_test_case("Sod Shock Tube").defaults);
    p.set("test_case", "Sod Shock Tube");
    p.set("use_mpi", "false");
    p.set("nx", std::to_string(nx));
    p.set("ny", std::to_string(ny));
    p.set("Lx", "12.0");
    p.set("Ly", "6.0");
    p.set("bc_left", "transmissive");
    p.set("bc_right", "transmissive");
    p.set("bc_bottom", "wall");
    p.set("bc_top", "wall");
    p.set("use_muscl", "true");
    p.set("muscl_limiter", "Van Leer");
    Simulation sim;
    sim.setup(SimParams::from(p), false);
    const Mesh& u = sim.mesh;

    // uniform
    Field Uu(static_cast<size_t>(u.num_cells())), R;
    for (int j = 0; j < u.ny; ++j)
        for (int i = 0; i < u.nx; ++i) Uu[j * u.nx + i] = smooth_state(sim.fluid, u.xc(i), u.yc(j));
    sim.residual(Uu, R);
    auto t0 = Clock::now();
    for (int r = 0; r < reps; ++r) sim.residual(Uu, R);
    const double t_uni = seconds(t0, Clock::now()) / reps;

    // face engine on the same level-0 mesh
    CellMesh m0(u.Lx, u.Ly, u.nx, u.ny, 0);
    m0.U.resize(m0.num_leaves());
    for (int c = 0; c < m0.num_leaves(); ++c) m0.U[c] = smooth_state(sim.fluid, m0.xc(c), m0.yc(c));
    CellFluxEngine e0(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    e0.build(m0);
    e0.residual(m0, m0.U, R);
    t0 = Clock::now();
    for (int r = 0; r < reps; ++r) e0.residual(m0, m0.U, R);
    const double t_face0 = seconds(t0, Clock::now()) / reps;
    {
        Field Ru, Rf;
        sim.residual(Uu, Ru);
        e0.residual(m0, m0.U, Rf);
        for (int c = 0; c < m0.num_leaves(); ++c)
            for (int q = 0; q < kNVar; ++q)
                if (!(Ru[m0.cj[c] * u.nx + m0.ci[c]][q] == Rf[c][q])) {
                    std::printf("MISMATCH between the uniform and the face residual\n");
                    return 1;
                }
    }

    // uniform at the size of the adapted mesh (the working set changes ns/cell)
    p.set("nx", std::to_string(nx / 2));
    p.set("ny", std::to_string(ny / 2));
    Simulation small;
    small.setup(SimParams::from(p), false);
    Field Us(static_cast<size_t>(small.mesh.num_cells())), Rs;
    for (int j = 0; j < small.mesh.ny; ++j)
        for (int i = 0; i < small.mesh.nx; ++i) Us[j * small.mesh.nx + i] = smooth_state(small.fluid, small.mesh.xc(i), small.mesh.yc(j));
    small.residual(Us, Rs);
    t0 = Clock::now();
    for (int r = 0; r < reps; ++r) small.residual(Us, Rs);
    const double cus = seconds(t0, Clock::now()) / reps / small.mesh.num_cells();

    // face engine on an adapted mesh: two levels, refined around a moving-front-like band
    CellMesh ma(u.Lx, u.Ly, u.nx / 4, u.ny / 4, 2);
    for (int lev = 0; lev < 2; ++lev) {
        std::vector<char> ref(ma.num_leaves(), 0);
        for (int c = 0; c < ma.num_leaves(); ++c) ref[c] = std::abs(ma.xc(c) - 4.0 - 0.8 * std::sin(1.5 * ma.yc(c))) < 0.9 * (lev + 1);
        ma.adapt(ref);
    }
    t0 = Clock::now();
    CellFluxEngine ea(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    ea.build(ma);
    const double t_build = seconds(t0, Clock::now());
    ma.U.resize(ma.num_leaves());
    for (int c = 0; c < ma.num_leaves(); ++c) ma.U[c] = smooth_state(sim.fluid, ma.xc(c), ma.yc(c));
    ea.residual(ma, ma.U, R);
    t0 = Clock::now();
    for (int r = 0; r < reps; ++r) ea.residual(ma, ma.U, R);
    const double t_adapt = seconds(t0, Clock::now()) / reps;
    std::vector<char> none(ma.num_leaves(), 0);
    t0 = Clock::now();
    ma.adapt(none, none);
    ea.build(ma);
    const double t_remesh = seconds(t0, Clock::now());

    const double cu = t_uni / u.num_cells(), cf0 = t_face0 / m0.num_leaves(), cfa = t_adapt / ma.num_leaves();
    std::printf("uniform %dx%d (%d cells): %.4f s/residual, %.1f ns/cell\n", nx, ny, u.num_cells(), t_uni, 1e9 * cu);
    std::printf("face engine, level-0 same mesh: %.4f s, %.1f ns/cell, ratio %.2f\n", t_face0, 1e9 * cf0, cf0 / cu);
    std::printf("face engine, adapted (%d leaves, %d virtual): %.4f s, %.1f ns/cell, ratio %.2f\n", ma.num_leaves(),
                ea.num_virtual(), t_adapt, 1e9 * cfa, cfa / cu);
    std::printf("uniform %dx%d (%d cells), matched size: %.1f ns/cell; adapted / matched = %.2f\n", small.mesh.nx, small.mesh.ny,
                small.mesh.num_cells(), 1e9 * cus, cfa / cus);
    // cost of one coarse step (SSP-RK2: 2 stages, level l advanced 2^l times) and the share of a remesh every 10 steps
    double leaf_evals = 0.0;
    for (int l = 0; l <= ma.max_level; ++l) {
        const int n = ma.level_begin[l + 1] - ma.level_begin[l];
        std::printf("  level %d: %d leaves\n", l, n);
        leaf_evals += 2.0 * (1 << l) * n;
    }
    const double t_step = leaf_evals * cfa;
    std::printf("coarse step ~ %.4f s (%.1f full-mesh residuals); remesh every 10 steps = %.1f %% of the run\n", t_step,
                leaf_evals / ma.num_leaves(), 100.0 * t_remesh / (10.0 * t_step + t_remesh));
    std::printf("stencil build (%d leaves): %.4f s = %.2f residuals; adapt-without-change + build: %.4f s = %.2f residuals\n",
                ma.num_leaves(), t_build, t_build / t_adapt, t_remesh, t_remesh / t_adapt);
    return 0;
}
