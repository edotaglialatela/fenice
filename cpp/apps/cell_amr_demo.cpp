// Development driver of the cell-based AMR: runs a case file on an adaptive CellMesh and writes the leaves.
//   cell_amr_demo "<case>" [key=value ...] [--out dir] [--levels n] [--every n] [--tend t]
//                 [--criterion x] [--split x] [--join x] [--vars rho,p,u] [--diffusion n]
// Parameters come from the case as usual; amr_max_level / amr_refine_every are used when --levels / --every are absent.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "fv/CellAdvance.hpp"
#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"
#include "fv/CellRemesh.hpp"
#include "fv/Simulation.hpp"

using namespace fv;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: cell_amr_demo \"<case>\" [key=value ...] [--out dir] [--levels n] [--every n] [--tend t] "
                    "[--criterion x] [--split x] [--join x] [--vars rho,p,u] [--diffusion n]\n");
        return 1;
    }
    const std::string name = argv[1];
    Params p = baseline_defaults();
    p.merge(get_test_case(name).defaults);
    p.set("test_case", name);
    p.set("use_mpi", "false");
    p.set("use_amr", "false");
    std::string out = "results/cell_amr/" + name;
    int levels = -1, every = -1;
    double tend = -1.0;
    CellIndicator ind;
    for (int a = 2; a < argc; ++a) {
        const std::string s = argv[a];
        const auto next = [&]() { return std::string(argv[++a]); };
        if (s == "--out") out = next();
        else if (s == "--levels") levels = std::atoi(next().c_str());
        else if (s == "--every") every = std::atoi(next().c_str());
        else if (s == "--tend") tend = std::atof(next().c_str());
        else if (s == "--criterion") ind.criterion = std::atof(next().c_str());
        else if (s == "--split") ind.split = std::atof(next().c_str());
        else if (s == "--join") ind.join = std::atof(next().c_str());
        else if (s == "--diffusion") ind.diffusion = std::atoi(next().c_str());
        else if (s == "--vars") {
            const std::string v = next();
            ind.use_density = v.find("rho") != std::string::npos;
            ind.use_pressure = v.find('p') != std::string::npos;
            ind.use_velocity = v.find('u') != std::string::npos;
        } else if (s.find('=') != std::string::npos)
            p.set(s.substr(0, s.find('=')), s.substr(s.find('=') + 1));
    }
    Simulation sim;
    sim.setup(SimParams::from(p), false);
    const Mesh& u = sim.mesh;
    if (levels < 0) levels = sim.p.amr_max_level;
    if (every < 0) every = sim.p.amr_refine_every;
    if (tend < 0) tend = sim.p.t_end;
    const bool px = sim.boundaries.type(Face::Left) == BCType::Periodic;
    const bool py = sim.boundaries.type(Face::Bottom) == BCType::Periodic;
    CellMesh mesh(u.Lx, u.Ly, u.nx, u.ny, levels, px, py, u.x0, u.y0);

    CellPhysics ph;
    ph.ns = sim.ns.get();
    ph.body = sim.body_force.get();
    if (sim.p.use_source) {
        ph.source = &sim.p.source;
        ph.source_unsteady = sim.p.source_unsteady;
        ph.source_mesh = &sim.mesh;
    }
    if (sim.p.use_ibm) ph.ibm_factory = sim.ibm_factory;
    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    eng.set_physics(ph);
    CellAdaptor adaptor(mesh, eng, sim.fluid, ind);
    if (sim.p.use_ibm) adaptor.set_body(sim.ibm_factory, 2);

    // initial condition at the leaf centres, each leaf on the uniform mesh of its own level
    const auto initial = [&](CellMesh& m) {
        m.U.assign(static_cast<size_t>(m.num_leaves()), State{});
        for (int l = 0; l <= m.max_level; ++l) {
            if (m.level_begin[l + 1] == m.level_begin[l]) continue;
            const Mesh ml = generate_cartesian_mesh(m.Lx, m.Ly, m.nx_at(l), m.ny_at(l), m.x0, m.y0);
            const Field F = get_test_case(name).initialize(ml, sim.fluid, sim.p.source);
            for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c) m.U[c] = F[m.cj[c] * m.nx_at(l) + m.ci[c]];
        }
    };
    initial(mesh);
    eng.build(mesh);
    const int passes = levels > 0 ? adaptor.init_adapt(initial) : 0;
    if (eng.has_body()) eng.solid_reset(mesh, mesh.U);
    std::printf("[cell-amr] %s: base %d x %d, %d levels, %d initial passes -> %d leaves\n", name.c_str(), u.nx, u.ny, levels,
                passes, mesh.num_leaves());

    std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(sim.p.time_scheme), sim.p.cfl, 1.0, 5, 1e-3);
    CellAdvance adv(mesh, eng, *ts);
    Field R;
    double S = eng.residual(mesh, mesh.U, R), t = 0.0, remesh_wall = 0.0, leaf_sum = 0.0;
    int step = 0;
    const auto t0 = Clock::now();
    while (t < tend) {
        const double dt = std::min(sim.p.cfl * std::min(mesh.dx(0), mesh.dy(0)) / S, tend - t);
        S = adv.step(dt, t);
        if (eng.has_body()) eng.solid_reset(mesh, mesh.U);
        t += dt;
        ++step;
        leaf_sum += mesh.num_leaves();
        if (levels > 0 && step % every == 0) {
            const auto r0 = Clock::now();
            adaptor.remesh();
            remesh_wall += std::chrono::duration<double>(Clock::now() - r0).count();
        }
        if (step % 100 == 0) std::printf("step %6d  t = %.5f  leaves %d\n", step, t, mesh.num_leaves());
    }
    const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
    std::printf("[cell-amr] remesh parts: indicator %.2f s, adapt %.2f s, engine build %.2f s\n", adaptor.t_indicator, adaptor.t_adapt, adaptor.t_build);
    std::vector<int> per_level(static_cast<size_t>(levels) + 1, 0);
    for (int c = 0; c < mesh.num_leaves(); ++c) ++per_level[mesh.level[c]];
    std::printf("[cell-amr] %d steps, wall %.2f s (remesh %.2f s = %.1f %%), mean leaves %.0f, final leaves %d (", step, wall,
                remesh_wall, 100.0 * remesh_wall / wall, leaf_sum / step, mesh.num_leaves());
    for (int l = 0; l <= levels; ++l) std::printf("%sL%d %d", l ? ", " : "", l, per_level[l]);
    std::printf(")\n");

    std::filesystem::create_directories(out);
    std::ofstream f(out + "/leaves.csv");
    f << "x,y,dx,level,rho,u,v,p,solid\n";
    for (int c = 0; c < mesh.num_leaves(); ++c) {
        const State W = sim.fluid.cons_to_prim(mesh.U[c]);
        f << mesh.xc(c) << ',' << mesh.yc(c) << ',' << mesh.dx(mesh.level[c]) << ',' << mesh.level[c] << ',' << W[0] << ','
          << W[1] << ',' << W[2] << ',' << W[3] << ',' << (eng.has_body() && eng.solid()[c] ? 1 : 0) << '\n';
    }
    std::printf("[cell-amr] leaves written to %s/leaves.csv\n", out.c_str());
    return 0;
}
