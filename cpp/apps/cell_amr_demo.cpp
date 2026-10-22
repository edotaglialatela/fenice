// Development driver of the cell-based AMR: runs a case file on an adaptive CellMesh and writes the leaves.
//   cell_amr_demo "<case>" [key=value ...] [--out dir] [--levels n] [--every n] [--tend t]
//                 [--criterion x] [--split x] [--join x] [--vars rho,p,u] [--diffusion n] [--closure 0..2]
//                 [--monitor n] [--steps n] [--poison]
// Parameters come from the case as usual; amr_max_level / amr_refine_every are used when --levels / --every are absent.
// Under mpiexec the leaves are distributed over the ranks (CellParallel); the final state in <out>/final.bin is bit-identical
// for any number of ranks. --monitor n: loads every n steps (needs the state on every rank, so it costs a gather); --poison:
// the state of the leaves a rank does not hold is NaN (finds a missing halo cell).
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef FV_USE_MPI
#include <mpi.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "fv/CellAdvance.hpp"
#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"
#include "fv/CellParallel.hpp"
#include "fv/CellRemesh.hpp"
#include "fv/Simulation.hpp"

using namespace fv;
using Clock = std::chrono::steady_clock;

namespace {
double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }
}   // namespace

int main(int argc, char** argv) {
    int rank = 0, nranks = 1;
#ifdef FV_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
#endif
#ifdef _OPENMP
    if (nranks > 1 && !std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(1);   // avoid oversubscription
#endif
    const bool root = rank == 0;
    if (argc < 2) {
        if (root)
            std::printf("usage: cell_amr_demo \"<case>\" [key=value ...] [--out dir] [--levels n] [--every n] [--tend t] "
                        "[--criterion x] [--split x] [--join x] [--vars rho,p,u] [--diffusion n] [--closure 0..2] "
                        "[--monitor n] [--poison]\n");
        return 1;
    }
    const std::string name = argv[1];
    Params p = baseline_defaults();
    p.merge(get_test_case(name).defaults);
    p.set("test_case", name);
    p.set("use_mpi", "false");
    p.set("use_amr", "false");
    std::string out = "results/cell_amr/" + name;
    int levels = -1, every = -1, closure = 2;   // jump closure: 0 injection, 1 interpolated, 2 = 1 + second-order coarse side (default)
    int monitor = 10, max_steps = 0;
    bool poison = false;
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
        else if (s == "--closure") closure = std::atoi(next().c_str());
        else if (s == "--monitor") monitor = std::atoi(next().c_str());
        else if (s == "--steps") max_steps = std::atoi(next().c_str());
        else if (s == "--poison") poison = true;
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
    eng.interpolate_jumps = closure == 1 || closure == 2;
    eng.coarse_side_second_order = closure >= 2;
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
    if (root)
        std::printf("[cell-amr] %s: base %d x %d, %d levels, %d initial passes -> %d leaves, %d rank%s\n", name.c_str(), u.nx,
                    u.ny, levels, passes, mesh.num_leaves(), nranks, nranks > 1 ? "s" : "");

    // distributed run: the topology and the initial state are replicated, the state is distributed from here on
    CellComm comm;
    comm.rank = rank;
    comm.size = nranks;
#ifdef FV_USE_MPI
    if (nranks > 1) {
        comm.allgatherv = [](const double* send, int n, double* recv, const int* counts, const int* displs) {
            MPI_Allgatherv(send, n, MPI_DOUBLE, recv, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        };
        comm.allreduce_max = [](double v) {
            double g = v;
            MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            return g;
        };
        comm.exchange = [](const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv) {
            std::vector<MPI_Request> req;
            req.reserve(send.size() + recv.size());
            for (size_t r = 0; r < recv.size(); ++r)
                if (!recv[r].empty()) {
                    req.emplace_back();
                    MPI_Irecv(recv[r].data(), static_cast<int>(recv[r].size()), MPI_DOUBLE, static_cast<int>(r), 91, MPI_COMM_WORLD,
                              &req.back());
                }
            for (size_t r = 0; r < send.size(); ++r)
                if (!send[r].empty()) {
                    req.emplace_back();
                    MPI_Isend(send[r].data(), static_cast<int>(send[r].size()), MPI_DOUBLE, static_cast<int>(r), 91, MPI_COMM_WORLD,
                              &req.back());
                }
            MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE);
        };
    }
#endif
    CellParallel par(mesh, eng, comm);
    par.poison = poison;

    std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(sim.p.time_scheme), sim.p.cfl, 1.0, 5, 1e-3);
    CellAdvance adv(mesh, eng, *ts);
    if (nranks > 1) adv.par = &par;
    Field R;
    double S = eng.residual(mesh, mesh.U, R), t = 0.0, remesh_wall = 0.0, leaf_sum = 0.0;
    double t_gather = 0.0, t_speed = 0.0, t_partition = 0.0;   // parts of the remesh that CellAdaptor does not time
    if (nranks > 1) par.partition();
    int step = 0;
    // loads on the body: the net momentum flux into the solid leaves (sum of their residuals times the area), the same
    // definition as the monitor of the uniform solver; coefficients on q L, L the chord (airfoil) or the diameter. Needs the
    // state of every leaf: gathered first, summed on every rank in leaf order.
    std::vector<std::array<double, 3>> history;   // t, Cd, Cl
    const bool airfoil_body = sim.p.use_ibm;
    const double q_ref = 0.5 * sim.p.rho_in * sim.p.u_in * sim.p.u_in *
                         (sim.p.ibm_shape.find("NACA") != std::string::npos || sim.p.ibm_shape == "Airfoil" ? sim.p.ibm_chord
                                                                                                               : 2.0 * sim.p.ibm_radius);
    const auto sample_loads = [&](double time) {
        if (!airfoil_body || q_ref <= 0.0 || monitor <= 0) return;
        if (nranks > 1) par.gather_state();
        Field Rl;
        eng.residual(mesh, mesh.U, Rl);
        double fx = 0.0, fy = 0.0;
        for (int c = 0; c < mesh.num_leaves(); ++c)
            if (eng.solid()[c]) {
                fx += Rl[c][1] * mesh.area(c);
                fy += Rl[c][2] * mesh.area(c);
            }
        history.push_back({time, fx / q_ref, fy / q_ref});
    };
    const auto t0 = Clock::now();
    double step_wall = 0.0;
    while (t < tend && (max_steps == 0 || step < max_steps)) {
        const auto s0 = Clock::now();
        const double dt = std::min(sim.p.cfl * std::min(mesh.dx(0), mesh.dy(0)) / S, tend - t);
        S = adv.step(dt, t);
        if (eng.has_body()) {
            eng.solid_reset(mesh, mesh.U);
            if (nranks > 1) par.exchange_solid();
        }
        t += dt;
        ++step;
        leaf_sum += mesh.num_leaves();
        step_wall += seconds_since(s0);
        if (levels > 0 && step % every == 0) {
            const auto r0 = Clock::now();
            const auto g0 = Clock::now();
            if (nranks > 1) par.gather_state();   // the indicator needs the state of every leaf
            t_gather += seconds_since(g0);
            if (adaptor.remesh()) {
                const auto v0 = Clock::now();
                S = eng.residual(mesh, mesh.U, R);   // the wave-speed bound follows the new leaves
                t_speed += seconds_since(v0);
            }
            const auto q0 = Clock::now();
            if (nranks > 1) par.partition();
            t_partition += seconds_since(q0);
            remesh_wall += seconds_since(r0);
        }
        if (monitor > 0 && step % monitor == 0) sample_loads(t);
        if (root && step % 100 == 0) std::printf("step %6d  t = %.5f  leaves %d\n", step, t, mesh.num_leaves());
    }
    const double wall = seconds_since(t0);
    if (nranks > 1) par.gather_state();

    // balance report: owned leaves per level and rank, the weighted work sum 2^l N_l, the halo
    if (nranks > 1) {
        std::vector<double> mine = {static_cast<double>(par.held_count()), static_cast<double>(par.halo_count())};
        std::vector<double> all(2 * static_cast<size_t>(nranks), 0.0);
        std::vector<int> counts(static_cast<size_t>(nranks), 2), displs(static_cast<size_t>(nranks), 0);
        for (int r = 1; r < nranks; ++r) displs[static_cast<size_t>(r)] = 2 * r;
        comm.allgatherv(mine.data(), 2, all.data(), counts.data(), displs.data());
        if (root) {
            std::printf("[cell-amr] balance (leaves owned per level; work = sum 2^l N_l; held and halo leaves):\n");
            double wmax = 0.0, wsum = 0.0;
            for (int r = 0; r < nranks; ++r) {
                double work = 0.0;
                std::printf("  rank %d:", r);
                for (int l = 0; l <= levels; ++l) {
                    const int n = par.owned_end(r, l) - par.owned_begin(r, l);
                    work += static_cast<double>(n) * (1 << l);
                    std::printf(" L%d %d", l, n);
                }
                std::printf("  work %.0f  held %.0f  halo %.0f\n", work, all[2 * static_cast<size_t>(r)], all[2 * static_cast<size_t>(r) + 1]);
                wmax = std::max(wmax, work);
                wsum += work;
            }
            std::printf("  work max / mean = %.3f\n", wmax / (wsum / nranks));
        }
    }
    if (root) {
        std::printf("[cell-amr] remesh parts (rank 0): gather %.2f s, indicator %.2f s, adapt %.2f s, engine build %.2f s, "
                    "speed bound %.2f s, partition %.2f s\n",
                    t_gather, adaptor.t_indicator, adaptor.t_adapt, adaptor.t_build, t_speed, t_partition);
        std::printf("[cell-amr] churn: %lld leaves flagged for refinement, about %lld sibling groups joined over %d steps\n",
                    adaptor.n_refined, adaptor.n_joined, step);
        std::vector<int> per_level(static_cast<size_t>(levels) + 1, 0);
        for (int c = 0; c < mesh.num_leaves(); ++c) ++per_level[mesh.level[c]];
        std::printf("[cell-amr] %d steps, wall %.2f s (time stepping %.2f s, remesh incl. gather, indicator, build and partition %.2f s = "
                    "%.1f %%), mean leaves %.0f, final leaves %d (",
                    step, wall, step_wall, remesh_wall, 100.0 * remesh_wall / wall, leaf_sum / step, mesh.num_leaves());
        for (int l = 0; l <= levels; ++l) std::printf("%sL%d %d", l ? ", " : "", l, per_level[l]);
        std::printf(")\n");
        if (!history.empty()) {
            double cd = 0.0, cl = 0.0;
            int n = 0;
            for (const auto& h : history)
                if (h[0] > tend - 1.0) {
                    cd += h[1];
                    cl += h[2];
                    ++n;
                }
            std::printf("[cell-amr] loads: t = %.3f Cd %.4f Cl %.4f; mean over the last unit of time Cd %.4f Cl %.4f\n",
                        history.back()[0], history.back()[1], history.back()[2], cd / n, cl / n);
        }
        std::filesystem::create_directories(out);
        {   // the exact final state, for bit comparisons between rank counts
            std::ofstream b(out + "/final.bin", std::ios::binary);
            const int32_t n = mesh.num_leaves();
            b.write(reinterpret_cast<const char*>(&n), sizeof n);
            for (int c = 0; c < n; ++c) {
                const int32_t v[3] = {mesh.level[c], mesh.ci[c], mesh.cj[c]};
                b.write(reinterpret_cast<const char*>(v), sizeof v);
            }
            for (int c = 0; c < n; ++c) b.write(reinterpret_cast<const char*>(mesh.U[c].data()), sizeof(double) * kNVar);
        }
        std::ofstream f(out + "/leaves.csv");
        const std::vector<double> xi_end = ind.xi(mesh, mesh.U, sim.fluid);   // the indicator of the final leaves
        f << "x,y,dx,level,rho,u,v,p,solid,xi\n";
        for (int c = 0; c < mesh.num_leaves(); ++c) {
            const State W = sim.fluid.cons_to_prim(mesh.U[c]);
            f << mesh.xc(c) << ',' << mesh.yc(c) << ',' << mesh.dx(mesh.level[c]) << ',' << mesh.level[c] << ',' << W[0] << ','
              << W[1] << ',' << W[2] << ',' << W[3] << ',' << (eng.has_body() && eng.solid()[c] ? 1 : 0) << ',' << xi_end[c]
              << '\n';
        }
        std::printf("[cell-amr] leaves written to %s/leaves.csv and final.bin\n", out.c_str());
    }
#ifdef FV_USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
