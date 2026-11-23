#include "fv/Profile.hpp"
#include "fv/Switches.hpp"
#include "fv/Simulation.hpp"
#include "fv/ViscousBound.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <map>
#include <optional>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fv/Geometry.hpp"
#include "fv/SolutionMapping.hpp"
#ifdef FV_USE_MPI
#include "fv/MPIDecomposition.hpp"
#endif

#ifndef FV_PROJECT_ROOT
#define FV_PROJECT_ROOT "."
#endif

namespace fs = std::filesystem;

#include "SimulationInternal.hpp"
#include "SimulationPostprocess.hpp"

namespace fv {

using namespace sim_detail;

std::array<double, 3> Simulation::volume_averages(const Field& Up, long long n_global) const {
    double sums[3] = {0.0, 0.0, 0.0};
    for (const State& c : Up) {
        sums[0] += c[0];
        sums[1] += 0.5 * (c[1] * c[1] + c[2] * c[2] + c[4] * c[4]) / c[0];
        sums[2] += c[3];
    }
#ifdef FV_USE_MPI
    int init = 0;
    MPI_Initialized(&init);
    int size = 1;
    if (init) MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size > 1) {
        double g[3];
        MPI_Allreduce(sums, g, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (int q = 0; q < 3; ++q) sums[q] = g[q];
    }
#endif
    return {sums[0] / n_global, sums[1] / n_global, sums[2] / n_global};
}

void Simulation::arm_stop_file() const {
    std::error_code ec;
    fs::remove(fs::path(exporter.output_dir()) / "STOP", ec);
}

void Simulation::poll_stop_file() const {
    std::error_code ec;
    if (!g_stop_requested.load() && fs::exists(fs::path(exporter.output_dir()) / "STOP", ec)) g_stop_requested = true;
}

std::FILE* Simulation::open_monitor() const {
    if (p.monitor_interval <= 0) return nullptr;
    const std::string path = (fs::path(exporter.output_dir()) / (file_prefix + "_monitor.csv")).string();
    std::FILE* f = std::fopen(path.c_str(), start_step > 0 ? "a" : "w");
    if (f && start_step == 0)
        std::fprintf(f, "step,t,mass,kinetic_energy,total_energy%s,res_rho,res_mom,res_E\n", p.use_ibm ? ",Fx,Fy,Cd,Cl,Cm" : "");
    return f;
}

void Simulation::monitor_row(std::FILE* f, int step, const std::array<double, 3>& v, const std::array<double, 3>* loads,
                             const std::array<double, 3>& res) const {
    if (!f) return;
    std::fprintf(f, "%d,%.10e,%.15e,%.15e,%.15e", step, t, v[0], v[1], v[2]);
    if (loads) {
        // Coefficients on q A (2-D: per unit depth, A = L; 3-D: L Lz, or pi r^2 for a sphere);
        // L = chord (airfoil) or diameter; Cm about (ibm_cx, ibm_cy), nose-up positive
        const bool airfoil = !naca_digits(p.ibm_shape).empty() || lower(p.ibm_shape) == "airfoil";
        const double L = airfoil ? p.ibm_chord : 2.0 * p.ibm_radius;
        const double A =
            !mesh.is3d()
                ? L
                : (lower(p.ibm_shape) == "sphere" ? 3.14159265358979323846 * p.ibm_radius * p.ibm_radius : L * mesh.Lz);
        const double qA = 0.5 * p.rho_in * p.u_in * p.u_in * A;
        const double Fx = (*loads)[0], Fy = (*loads)[1], Mz = (*loads)[2];
        std::fprintf(f, ",%.10e,%.10e,%.10e,%.10e,%.10e", Fx, Fy, qA > 0 ? Fx / qA : 0.0, qA > 0 ? Fy / qA : 0.0,
                     qA > 0 ? -Mz / (qA * L) : 0.0);
    }
    std::fprintf(f, ",%.10e,%.10e,%.10e\n", res[0], res[1], res[2]);
    std::fflush(f);
}

namespace {
// Sums of the squared residual per quantity and of the weights: {rho, momentum, energy, weight}; the norm is sqrt(sum / weight).
using ResSums = std::array<double, 4>;

inline void add_residual(ResSums& s, const State& r, double weight) {
    s[0] += r[0] * r[0] * weight;
    s[1] += (r[1] * r[1] + r[2] * r[2] + r[4] * r[4]) * weight;
    s[2] += r[3] * r[3] * weight;
    s[3] += weight;
}

inline std::array<double, 3> residual_norms(const ResSums& s) {
    if (!(s[3] > 0.0)) return {0.0, 0.0, 0.0};
    return {std::sqrt(s[0] / s[3]), std::sqrt(s[1] / s[3]), std::sqrt(s[2] / s[3])};
}
}   // namespace

std::array<double, 3> Simulation::ibm_loads(const Mesh& m, const ImmersedBoundary& ib, const Field& U,
                                            const Field& R) const {
    const std::vector<char>& solid = ib.mask();
    const double vol = m.is3d() ? m.cell_volume() : m.cell_area();
    double fx = 0.0, fy = 0.0, mz = 0.0;
    for (int c = 0; c < m.num_cells(); ++c) {
        if (!solid[c]) continue;
        double rx = R[c][1], ry = R[c][2];
        if (body_force) {
            rx -= U[c][0] * body_force->gx;
            ry -= U[c][0] * body_force->gy;
        }   // not a load
        const double ex = rx * vol, ey = ry * vol;
        fx += ex;
        fy += ey;
        mz += (m.xc(c % m.nx) - p.ibm_cx) * ey - (m.yc((c / m.nx) % m.ny) - p.ibm_cy) * ex;
    }
    return {fx, fy, mz};
}

// ---------------------------------------------------------------------------
// Time loop shared by the engines
// ---------------------------------------------------------------------------

Simulation::LoopResult Simulation::run_loop(const LoopEngine& e) {
    LoopResult out;
    int step = start_step;
    int last_snapshot_step = step == 0 ? 0 : -1;
    double next_write_time = next_time_output(t, p.write_interval_time);
    const auto t0 = Clock::now();
    while (t < p.t_end) {
        std::optional<ProfileScope> iteration;
        if (e.profile_iteration) iteration.emplace(Section::Step);
        const StepResult r = e.advance(step + 1, next_write_time);
        ++step;
        const bool time_output = reached_time_output(t, next_write_time, p.write_interval_time);
        if (time_output) next_write_time += p.write_interval_time;
        e.after_step(step);

        if (step % p.save_interval == 0 || time_output) {
            e.save(step, r, seconds_since(t0));
            last_snapshot_step = step;
        }
        poll_stop_file();
        if (e.stop_requested ? e.stop_requested() : g_stop_requested.load()) {
            out.interrupted = true;
            break;
        }
        if (p.max_steps > 0 && step - start_step >= p.max_steps) break;
    }
    out.wall = seconds_since(t0);
    out.step = step;
    out.snapshot_due = out.interrupted || last_snapshot_step != step;
    return out;
}

ImmersedBoundary::BodyPose Simulation::prescribed_pose(double t) const {
    ImmersedBoundary::BodyPose pose;
    pose.x = p.ibm_vx * t;
    pose.y = p.ibm_vy * t;
    pose.angle = p.ibm_omega * t;
    pose.u = p.ibm_vx;
    pose.v = p.ibm_vy;
    pose.omega = p.ibm_omega;
    return pose;
}

// ---------------------------------------------------------------------------
// Serial time loop
// ---------------------------------------------------------------------------

void Simulation::solve() {
    std::printf("=== TIME-MARCHING  (t_end = %g s) ===\n", p.t_end);
    std::fflush(stdout);
    if (cells) {
        solve_amr();
        return;
    }

    const double min_delta = dt_length;
    int step = start_step;
    PostprocessWriter post(*test_case, exporter.output_dir(), file_prefix, step);
    if (step == 0) exporter.export_vtr(file_prefix, fluid, U, mesh, t, step);

    const ResidualFn res = [this](const Field& Uin, Field& R) { return residual(Uin, R); };
    std::FILE* mon = open_monitor();
    arm_stop_file();
    const long long n_all = mesh.num_cells();
    Field R_loads;
    // R_loads holds the residual of the current state at the current time (r_valid): the monitor row, the force output of the same step
    // and the first stage of the next step all use the one evaluation (offer_first_residual), the same bits as evaluating each again
    bool r_valid = false;
    double r_speed = 0.0;
    const bool moving = p.use_ibm && ibm->moving();
    // the body at time `tt` (constant velocities from its placement at t = 0); the pose is fixed during a step
    const auto place_body = [&](double tt) {
        if (ibm->move_to(mesh, prescribed_pose(tt))) {
            ibm->fill_fresh(U, fluid);   // the cells the body left take the state of the wall layer before any ghost fill
            hancock_sides_.clear();      // the solid-side flags of the MUSCL-Hancock predictor follow the mask
        }
    };
    const auto offer_residual = [&] {
        // a moving body changes the ghost table before the next step: the residual of the old pose is not the first stage of the next step
        if (r_valid && !moving) scheme->offer_first_residual(R_loads, r_speed);
        r_valid = false;
    };
    if (moving) place_body(t);
    auto row = [&](int st) {   // averages, plus the IBM loads from a residual of the current state
        scheme->stage_time = t;
        std::array<double, 3> loads{};
        r_speed = residual(U, R_loads);
        r_valid = true;
        if (p.use_ibm) loads = ibm_loads(mesh, *ibm, U, R_loads);
        ResSums rs{};
        for (int c = 0; c < mesh.num_cells(); ++c)
            if (!p.use_ibm || !ibm->mask()[c]) add_residual(rs, R_loads[c], 1.0);
        monitor_row(mon, st, volume_averages(U, n_all), p.use_ibm ? &loads : nullptr, residual_norms(rs));
    };
    if (mon && step == 0) row(step);
    if (!post.empty()) {
        if (step == 0) {
            post.write_uniform(mesh, fluid, U, step, t, true);
            if (post.has_averages()) post.observe_averages(step, t, volume_primitive_means(fluid, mesh, U), true);
        } else {
            post.skip_step(step);
            if (post.has_averages()) post.observe_averages(step, t, volume_primitive_means(fluid, mesh, U), false, false);
        }
        if (step == 0 && post.has_force_due(step, true)) {
            if (!r_valid) residual(U, R_loads);
            post.write_forces(step, t, ibm_loads(mesh, *ibm, U, R_loads), mesh, true);
        }
    }
    offer_residual();
    LoopEngine engine;
    engine.advance = [&](int st, double t_stop) {
        if (moving && st > start_step + 1) place_body(t);   // the first step of a run was placed above
        const StepResult r = scheme->step(U, res, min_delta, t, std::min(p.t_end, t_stop), p.dt_max);
        check_progress(t, r.dt, r.s_max, st, p.time_scheme.c_str());
        if (moving) {   // the body must not leave the ghost table behind: at most one cell per step
            const double travel = r.dt * std::max(std::abs(p.ibm_vx) / mesh.dx, std::abs(p.ibm_vy) / mesh.dy);
            if (travel > 1.0) throw std::runtime_error("the moving body travelled more than one cell in a step (lower the CFL number)");
        }
        if (p.use_ibm) ibm->apply_solid_reset(U, fluid);
        t += r.dt;
        scheme->stage_time = t;
        return r;
    };
    engine.after_step = [&](int st) {
        if (mon && st % p.monitor_interval == 0) row(st);
        if (!post.empty()) {
            post.write_uniform(mesh, fluid, U, st, t);
            if (post.has_averages()) post.observe_averages(st, t, volume_primitive_means(fluid, mesh, U));
            if (post.has_force_due(st)) {
                if (!r_valid) residual(U, R_loads);
                post.write_forces(st, t, ibm_loads(mesh, *ibm, U, R_loads), mesh);
            }
        }
        offer_residual();
    };
    engine.save = [&](int st, const StepResult& r, double wall) {
        {
            ProfileScope ps(Section::Snapshot);
            exporter.export_vtr(file_prefix, fluid, U, mesh, t, st);
        }
        {
            ProfileScope ps(Section::Checkpoint);
            save_checkpoint(st);
        }
        std::printf("Step %05d | t = %.5f s | dt = %.3e s | S_max = %.2f m/s | wall = %s | Checkpoint saved\n",
                    st, t, r.dt, r.s_max, format_elapsed(wall).c_str());
        std::fflush(stdout);
    };
    profile_reset();   // the profile counts the time loop only
    const LoopResult loop = run_loop(engine);
    step = loop.step;
    const double loop_wall = loop.wall;
    if (p.positivity_fallback)
        std::printf("[POSITIVITY] %lld repair passes, %lld cell flags over the run (0: no stage left the admissible set)\n", fb_repairs_, fb_flagged_);

    if (mon) {
        if (step % p.monitor_interval != 0) row(step);
        std::fclose(mon);
    }
    if (!post.empty()) {
        post.write_uniform(mesh, fluid, U, step, t, true);
        if (post.has_averages()) post.observe_averages(step, t, volume_primitive_means(fluid, mesh, U), true);
        if (post.has_force_due(step, true)) {
            residual(U, R_loads);
            post.write_forces(step, t, ibm_loads(mesh, *ibm, U, R_loads), mesh, true);
        }
    }
    if (loop.interrupted) std::printf("\n[INTERRUPTED] Stopping the simulation...\n");
    if (loop.snapshot_due) exporter.export_vtr(file_prefix, fluid, U, mesh, t, step);
    save_checkpoint(step);
    exporter.write_pvd(file_prefix);
    dump_final_state(step);
    std::printf("=== DONE  - open '%s.pvd' in ParaView ===  (time loop %.6f s, %d steps)\n", file_prefix.c_str(),
                loop_wall, step - start_step);
    if (profile_enabled()) print_profile(stdout, profile_values(), 1);
}

void Simulation::solve_amr() {
    CellMesh& cm = *cells;
    CellFluxEngine& eng = *cell_engine;
    const int nranks = cell_comm.size, rank = cell_comm.rank;
    const bool root = rank == 0;
    CellAdvance adv(cm, eng, *scheme);
    if (cell_par) adv.par = cell_par.get();
    if (p.positivity_fallback) {
        const double p_inf = fluid.type == EOS::Type::StiffenedGas ? fluid.p_inf : 0.0;
        adv.enable_positivity_fallback([this, p_inf](const State& u) {
            const double rho = u[0];
            if (!(rho > 0.0)) return false;
            const double ke = 0.5 * (u[1] * u[1] + u[2] * u[2] + u[4] * u[4]) / rho;
            return fluid.pressure_law().from_internal_energy_density(u[3] - ke) + p_inf > 0.0;
        });
    }
    adv.stage_hook = [this](int, double stage_time) { update_boundary_values(stage_time, boundaries); };
    // each rank keeps only its leaves and their halo (CellParallel::localize), unless FENICE_REPLICATED_REMESH=1
    const bool local_run = cell_par && cell_par->localized();
    // the leaves of the whole mesh and their state (collective; a localized run gives them to rank 0 only, the one that writes)
    std::vector<int> g_level, g_ci, g_cj, g_ck;
    Field g_U;
    const bool octree = cm.dim == 3;
    const auto leaves = [&]() -> LeafArrays {
        if (local_run) {
            cell_par->collect(g_level, g_ci, g_cj, g_ck, g_U, true);   // rank 0 only: it alone writes
            return {&g_level, &g_ci, &g_cj, &g_U, octree ? &g_ck : nullptr};
        }
        if (cell_par) cell_par->gather_state();
        return {&cm.level, &cm.ci, &cm.cj, &cm.U, octree ? &cm.ck : nullptr};
    };
    std::unique_ptr<PostprocessWriter> post;
    if (root) post = std::make_unique<PostprocessWriter>(*test_case, exporter.output_dir(), file_prefix, start_step);
    const auto post_state = [&](int st, bool force, bool reset_average, bool emit) {
        if ((!force && !amr_postprocess_state_due(*test_case, st)) ||
            (force && test_case->probes.empty() && test_case->samples.empty() && test_case->averages.empty()))
            return;
        if (!emit && test_case->averages.empty()) {
            if (root) post->skip_step(st);
            return;
        }
        const LeafArrays lf = leaves();
        if (!root) return;
        if (emit) post->write_amr(cm, fluid, *lf.level, *lf.ci, *lf.cj, *lf.U, st, t, force, lf.ck);
        else post->skip_step(st);
        if (post->has_averages()) {
            const std::array<double, 5> means = volume_primitive_means(fluid, cm, *lf.level, *lf.U);
            if (reset_average) post->reset_averages(t, means);
            post->observe_averages(st, t, means, force, emit);
        }
    };
    // number of leaves of the whole mesh (collective)
    const auto total_leaves = [&]() {
        double n = 0.0;
        for (int l = 0; l <= cm.max_level; ++l) n += eng.owned_end(l) - eng.owned_begin(l);
        return static_cast<int>(cell_par ? cell_par->reduce_sum(n) : n);
    };
    // largest wave speed of the leaves of the whole mesh (collective), from the forced state of the leaves a rank holds
    const auto wave_speed = [&]() {
        eng.force_state(cm, cm.U);
        const double s = eng.wave_speed(cm, cm.U);
        return cell_par ? cell_par->reduce_max(s) : s;
    };

    Field R;
    // wave-speed bound the time step is taken from, one step old (a restart reads it from the checkpoint)
    double S = start_step > 0 ? cell_speed_ : (local_run ? wave_speed() : eng.residual(cm, cm.U, R));
    cell_speed_ = S;
    int step = start_step;

    // Every rank writes the leaves it owns (the whole mesh on one rank)
    const auto export_state = [&](int st) {
        std::vector<int> owned;
        for (int l = 0; l <= cm.max_level; ++l)
            for (int c = eng.owned_begin(l); c < eng.owned_end(l); ++c) owned.push_back(c);
        exporter.export_cells(file_prefix, fluid, cm, owned, t, st, rank, nranks, [&] {
            if (cell_par) cell_par->reduce_max(0.0);   // a barrier
        });
    };

    // Monitor: area-weighted means over the leaves and the IBM loads. Every rank sums its owned leaves in leaf order and the
    // partial sums are added over the ranks (collective): one rank gives the serial bits, several agree with it to roundoff.
    std::FILE* mon = root ? open_monitor() : nullptr;
    arm_stop_file();
    const bool monitoring = p.monitor_interval > 0;
    const auto row = [&](int st, bool write_monitor, bool write_force, bool force_final) {
        scheme->stage_time = t;
        eng.stage_time = t;
        update_boundary_values(t, boundaries);
        double sums[4] = {0.0, 0.0, 0.0, 0.0};   // mass, kinetic energy, total energy, area
        for (int l = 0; l <= cm.max_level; ++l)
            for (int c = eng.owned_begin(l); c < eng.owned_end(l); ++c) {
                const State& u = cm.U[static_cast<size_t>(c)];
                const double a = cm.volume(c);   // (the area in 2-D)
                sums[0] += u[0] * a;
                sums[1] += 0.5 * (u[1] * u[1] + u[2] * u[2] + u[4] * u[4]) / u[0] * a;
                sums[2] += u[3] * a;
                sums[3] += a;
            }
        std::array<double, 3> loads{};
        ResSums rs{};
        {
            Field Rl;
            if (!cell_par) {
                eng.residual(cm, cm.U, Rl);
            } else {
                // the partitioned engine: the fluxes of each level, the coarse-fine ones handed to the coarse owners through the
                // registers (set to the flux itself, as a weight 1 gives), then R of the owned leaves
                Field Rown(static_cast<size_t>(cm.num_leaves()));
                eng.force_state(cm, cm.U);   // as the first pass of a step does
                for (int l = 0; l <= cm.max_level; ++l) {
                    if (!cell_par->level_present(l)) continue;
                    eng.compute_level_fluxes(cm, cm.U, l);
                    if (l == 0) continue;
                    eng.clear_registers(cm, l);
                    eng.accumulate_registers(cm, l, 1.0);
                    cell_par->exchange_registers(l);
                }
                for (int l = 0; l <= cm.max_level; ++l) {
                    if (!cell_par->level_present(l)) continue;
                    eng.gather_level(cm, l, true, Rl);
                    std::copy(Rl.begin(), Rl.end(), Rown.begin() + eng.owned_begin(l));
                }
                Rl.swap(Rown);
            }
            if (p.use_ibm) {
                loads = cell_loads(cm, Rl);
                if (cell_par)
                    for (double& v : loads) v = cell_par->reduce_sum(v);
            }
            if (write_force) {
                Field contributions(static_cast<size_t>(cm.num_leaves()));
                const std::vector<char>& solid = eng.solid();
                for (int l = 0; l <= cm.max_level; ++l)
                    for (int c = eng.owned_begin(l); c < eng.owned_end(l); ++c) {
                        if (!solid[static_cast<size_t>(c)]) continue;
                        double rx = Rl[static_cast<size_t>(c)][1], ry = Rl[static_cast<size_t>(c)][2];
                        if (body_force) {
                            rx -= cm.U[static_cast<size_t>(c)][0] * body_force->gx;
                            ry -= cm.U[static_cast<size_t>(c)][0] * body_force->gy;
                        }
                        const double ex = rx * cm.volume(c), ey = ry * cm.volume(c);   // (the area in 2-D; Fx, Fy, Mz as the uniform solver's)
                        contributions[static_cast<size_t>(c)] =
                            {ex, ey, (cm.xc(c) - p.ibm_cx) * ey - (cm.yc(c) - p.ibm_cy) * ex, 0.0, 0.0};
                    }
                Field ordered;
                if (cell_par) cell_par->collect_field(contributions, ordered, true);
                else ordered = std::move(contributions);
                if (root) {
                    std::array<double, 3> deterministic{};
                    for (const State& c : ordered) {
                        deterministic[0] += c[0];
                        deterministic[1] += c[1];
                        deterministic[2] += c[2];
                    }
                    post->write_forces(st, t, deterministic, mesh, force_final);
                }
            }
            const std::vector<char>& solid = eng.solid();
            for (int l = 0; l <= cm.max_level; ++l)
                for (int c = eng.owned_begin(l); c < eng.owned_end(l); ++c)
                    if (!p.use_ibm || !solid[static_cast<size_t>(c)]) add_residual(rs, Rl[static_cast<size_t>(c)], cm.volume(c));
        }
        if (cell_par) {
            for (double& v : sums) v = cell_par->reduce_sum(v);
            for (double& v : rs) v = cell_par->reduce_sum(v);
        }
        if (write_monitor)
            monitor_row(mon, st, {sums[0] / sums[3] * mesh.cell_volume(), sums[1] / sums[3] * mesh.cell_volume(), sums[2] / sums[3] * mesh.cell_volume()},
                        p.use_ibm ? &loads : nullptr, residual_norms(rs));
    };

    if (step == 0) export_state(step);
    post_state(step, step == 0, false, step == 0);
    if (monitoring && step == 0) row(step, true, postprocess_force_due(*test_case, step), false);
    else if (step == 0 && postprocess_force_due(*test_case, step)) row(step, false, true, false);

    double remesh_wall = 0.0, t_gather = 0.0, t_speed = 0.0, t_partition = 0.0;
    LoopEngine engine;
    engine.profile_iteration = false;   // Section::Step is the advance alone: the remesh and the output have their own sections
    engine.advance = [&](int st, double t_stop) {
        // the time-step length of the base mesh: min(dx, dy) in 2-D, 2 / sum 1/h in 3-D (Simulation::dt_length)
        const double h0 = octree ? step_length_3d(cm.dx(0), cm.dy(0), cm.dz(0)) : step_length_2d(cm.dx(0), cm.dy(0));
        double dt = std::min(scheme->cfl * h0 / std::max(S, 1e-10), p.t_end - t);
        if (p.dt_max > 0.0) dt = std::min(dt, p.dt_max);
        dt = std::min(dt, t_stop - t);
        {
            ProfileScope ps(Section::Step);
            S = adv.step(dt, t);
        }
        check_progress(t, dt, S, st, p.time_scheme.c_str());
        cell_speed_ = S;
        if (p.use_ibm) {
            if (!eng.ghost_ready())
                throw std::runtime_error("The ghost-cell wall of the immersed body is no longer set up: the cells around the body are "
                                         "not all at the finest level after a remesh");
            eng.solid_reset(cm, cm.U);
            if (cell_par) cell_par->exchange_solid();
        }
        t += dt;
        return StepResult{dt, S};
    };
    engine.after_step = [&](int st) {
        bool changed = false;
        if (st % p.amr_refine_every == 0) {
            const auto r0 = Clock::now();
            if (local_run) {
                // indicator on the owned leaves and their region, flags gathered, state of the new leaves migrated: no whole mesh
                changed = cell_par->remesh(*cell_adaptor);
            } else {
                // the indicator looks at every leaf: the whole mesh and its state on every rank (the replicated remesh)
                if (cell_par) cell_par->gather_state();
                t_gather += seconds_since(r0);
                changed = cell_adaptor->remesh();
                if (changed && cell_par) {
                    const auto r2 = Clock::now();
                    cell_par->partition();
                    t_partition += seconds_since(r2);
                }
            }
            if (changed) {
                const auto r1 = Clock::now();
                S = wave_speed();   // the wave-speed bound follows the new leaves (each rank's leaves, then the maximum)
                cell_speed_ = S;
                t_speed += seconds_since(r1);
            }
            remesh_wall += seconds_since(r0);
            profile_add(Section::Remesh, seconds_since(r0));
        }
        post_state(st, false, changed, true);
        const bool monitor_due = monitoring && st % p.monitor_interval == 0;
        const bool force_due = postprocess_force_due(*test_case, st);
        if (monitor_due || force_due) row(st, monitor_due, force_due, false);
    };
    // (the wave speed printed is the bound of the next step: S, which a remesh has updated, not the one the step returned)
    engine.save = [&](int st, const StepResult& r, double wall) {
        ProfileScope ps(Section::Output);
        export_state(st);
        // (several ranks: rank 0 writes the leaves of each rank as they arrive, the whole mesh is never gathered)
        if (cell_par) save_checkpoint_streamed(st);
        else save_checkpoint(st);
        const int nleaves = total_leaves();
        if (root)
            std::printf("Step %05d | t = %.5f s | dt = %.3e s | S_max = %.2f m/s | leaves = %d | wall = %s%s\n", st, t, r.dt, S, nleaves,
                        format_elapsed(wall).c_str(), " | Checkpoint saved");
        std::fflush(stdout);
    };
    engine.stop_requested = [&] {
        const double stop = g_stop_requested.load() ? 1.0 : 0.0;
        return (cell_par ? cell_par->reduce_max(stop) : stop) > 0.0;
    };
    profile_reset();   // the profile counts the time loop only
    const LoopResult loop = run_loop(engine);
    step = loop.step;
    const double loop_wall = loop.wall;
    const bool interrupted = loop.interrupted;

    if (monitoring || !test_case->forces.empty()) {
        const bool monitor_final = monitoring && step % p.monitor_interval != 0;
        const bool force_final = !test_case->forces.empty() && !postprocess_force_due(*test_case, step);
        if (monitor_final || force_final) row(step, monitor_final, force_final, force_final);
    }
    post_state(step, true, false, true);
    if (monitoring) {
        if (mon) std::fclose(mon);
    }
    if (root && interrupted) std::printf("\n[INTERRUPTED] Stopping the simulation...\n");
    if (loop.snapshot_due) export_state(step);
    // resident memory of every rank (collective), before the whole mesh is gathered for the files
    const MemoryUsage mem_end = memory_usage();
    std::vector<double> mem = {mem_setup_.current, mem_init_.current, mem_end.current, mem_end.peak};
    if (cell_par) {
        std::vector<double> all(static_cast<size_t>(4 * nranks));
        std::vector<int> counts(static_cast<size_t>(nranks), 4), displs(static_cast<size_t>(nranks));
        for (int r = 0; r < nranks; ++r) displs[static_cast<size_t>(r)] = 4 * r;
        cell_comm.allgatherv(mem.data(), 4, all.data(), counts.data(), displs.data());
        mem.swap(all);
    }
    // the profile of every rank (collective; empty, and the same on every rank, when FENICE_PROFILE is off)
    std::vector<double> profile = profile_values();
    if (!profile.empty() && cell_par) {
        const int n = static_cast<int>(profile.size());
        std::vector<double> all(profile.size() * static_cast<size_t>(nranks));
        std::vector<int> counts(static_cast<size_t>(nranks), n), displs(static_cast<size_t>(nranks));
        for (int r = 0; r < nranks; ++r) displs[static_cast<size_t>(r)] = n * r;
        cell_comm.allgatherv(profile.data(), n, all.data(), counts.data(), displs.data());
        profile.swap(all);
    }
    // leaves of the whole mesh at each level (collective)
    std::vector<long long> per_level(static_cast<size_t>(cm.max_level) + 1, 0);
    long long leaves_end = 0;
    for (int l = 0; l <= cm.max_level; ++l) {
        const double n = eng.owned_end(l) - eng.owned_begin(l);
        per_level[static_cast<size_t>(l)] = static_cast<long long>(cell_par ? cell_par->reduce_sum(n) : n);
        leaves_end += per_level[static_cast<size_t>(l)];
    }
    if (cell_par) save_checkpoint_streamed(step);
    else save_checkpoint(step);
    // on several ranks the whole mesh is gathered (on rank 0) only for the final dump, a test output
    const bool whole = !cell_par || !p.dump_state.empty();
    LeafArrays lf;
    if (whole) lf = leaves();
    if (!root) return;
    exporter.write_pvd(file_prefix);
    if (whole) dump_final_state(step, &lf);
    if (!local_run) print_halo_reach("final mesh");
    std::printf("[AMR] %lld leaves at the end (", leaves_end);
    for (int l = 0; l <= cm.max_level; ++l) std::printf("%slevel %d: %lld", l ? ", " : "", l, per_level[static_cast<size_t>(l)]);
    std::printf("); remesh %.2f s of the %.2f s time loop\n", remesh_wall, loop_wall);
    std::printf("[AMR] over the run: %lld leaves refined, %lld groups joined\n", static_cast<long long>(cell_adaptor->n_refined),
                static_cast<long long>(cell_adaptor->n_joined));
    // (a localized run has no gather: the "gather" is the region of the mesh the indicator reads and its state, the "indicator" the
    // flags of the owned leaves and their exchange, the "adapt" the adapt of the topology and the state of the new leaves, the
    // "partition" the partition, localize and halo fill after the engine build)
    if (local_run) {
        std::printf("[AMR] remesh parts: gather %.2f s, indicator %.2f s, adapt %.2f s, engine build %.2f s, wave-speed bound %.2f s, partition %.2f s\n",
                    cell_par->t_region, cell_par->t_flags, cell_par->t_migrate, cell_par->t_rebuild, t_speed, cell_par->t_repartition);
        std::printf("[AMR] of the adapt, the topology: %.2f s\n", cell_par->t_topo);
    } else
        std::printf("[AMR] remesh parts: gather %.2f s, indicator %.2f s, adapt %.2f s, engine build %.2f s, wave-speed bound %.2f s, partition %.2f s\n",
                    t_gather, cell_adaptor->t_indicator, cell_adaptor->t_adapt, cell_adaptor->t_build, t_speed, t_partition);
    if (root && p.positivity_fallback)
        std::printf("[POSITIVITY] %lld repair passes (the same on every rank), %lld leaves flagged on rank 0\n", adv.positivity_repairs(),
                    adv.positivity_flags());
    double lo[4], hi[4];
    for (int q = 0; q < 4; ++q) {
        lo[q] = hi[q] = mem[static_cast<size_t>(q)];
        for (int r = 1; r < nranks; ++r) {
            lo[q] = std::min(lo[q], mem[static_cast<size_t>(4 * r + q)]);
            hi[q] = std::max(hi[q], mem[static_cast<size_t>(4 * r + q)]);
        }
    }
    const double MB = 1024.0 * 1024.0;
    std::printf("[AMR] resident memory per rank [MB], min / max over %d rank%s: after setup %.1f / %.1f, after the initial "
                "adaptation %.1f / %.1f, at the end %.1f / %.1f, peak %.1f / %.1f\n",
                nranks, nranks > 1 ? "s" : "", lo[0] / MB, hi[0] / MB, lo[1] / MB, hi[1] / MB, lo[2] / MB, hi[2] / MB,
                lo[3] / MB, hi[3] / MB);
    double eng_leaf = 0.0, eng_uniform = 0.0;
    eng.memory_bytes(eng_leaf, eng_uniform);
    std::printf("[AMR] tables of rank 0 [MB]: mesh %.1f (leaves, state, faces, lookups), engine %.1f (stencils, work arrays), "
                "engine uniform-mesh %.1f (IBM masks, ghost table), body band %.1f, indicator %.1f\n",
                cm.memory_bytes() / MB, eng_leaf / MB, eng_uniform / MB, cell_adaptor->band_bytes() / MB,
                cell_adaptor->xi_bytes() / MB);
    if (!profile.empty()) print_profile(stdout, profile, nranks);
    std::printf("=== DONE  - open '%s.pvd' in ParaView ===  (time loop %.3f s, %d steps)\n", file_prefix.c_str(), loop_wall,
                step - start_step);
}

// ---------------------------------------------------------------------------
// MPI time loop
// ---------------------------------------------------------------------------

#ifdef FV_USE_MPI
void Simulation::run_mpi(MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    const bool root = (rank == 0);

    setup(p, root);
    MPIDecomposition decomp(p.nx, p.ny, comm, p.nz);

    // Layout table
    const char dir = decomp.dir;
    {
        const int start = dir == 'y' ? decomp.j_start : decomp.i_start;
        const int local = dir == 'y' ? decomp.ny_local : decomp.nx_local;
        int mine[4] = {rank, start, start + local - 1, local};
        std::vector<int> all(root ? 4 * decomp.n_ranks : 0);
        MPI_Gather(mine, 4, MPI_INT, root ? all.data() : nullptr, 4, MPI_INT, 0, comm);
        if (root) {
            std::printf("\n[MPI] %c-decomposition  -  %d ranks, global ny=%d, nx=%d\n", dir == 'y' ? 'Y' : 'X',
                        decomp.n_ranks, p.ny, p.nx);
            std::printf("  %5s  %8s  %6s  %9s  %s\n", "Rank", dir == 'y' ? "j_start" : "i_start",
                        dir == 'y' ? "j_end" : "i_end", dir == 'y' ? "ny_local" : "nx_local", "role");
            std::printf("  -----  --------  ------  ---------  ----------------\n");
            for (int r = 0; r < decomp.n_ranks; ++r) {
                std::string role;
                if (r == 0) role = dir == 'y' ? "bottom" : "left";
                if (r == decomp.n_ranks - 1) {
                    if (!role.empty()) role += ", ";
                    role += dir == 'y' ? "top" : "right";
                }
                if (role.empty()) role = "interior";
                std::printf("  %5d  %8d  %6d  %9d  %s\n", all[4 * r], all[4 * r + 1], all[4 * r + 2], all[4 * r + 3],
                            role.c_str());
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    }

    // Local mesh with its global origin (also positions the IBM mask correctly
    // for X-decomposition).
    Mesh lm;
    if (dir == 'y') {
        const double dy = p.Ly / p.ny;
        lm = generate_cartesian_mesh_3d(p.Lx, dy * decomp.ny_local, p.Lz, p.nx, decomp.ny_local, p.nz, 0.0,
                                        decomp.j_start * dy);
    } else {
        const double dx = p.Lx / p.nx;
        lm = generate_cartesian_mesh_3d(dx * decomp.nx_local, p.Ly, p.Lz, decomp.nx_local, p.ny, p.nz,
                                        decomp.i_start * dx, 0.0);
    }
    lm.dx = mesh.dx;   // exact global spacing (Ly_local/ny_local can differ from Ly/ny by an ulp)
    lm.dy = mesh.dy;
    lm.dz = mesh.dz;
    std::unique_ptr<MusclReconstruction> local_muscl;
    if (p.reconstructs())
        local_muscl = std::make_unique<MusclReconstruction>(fluid, parse_limiter(p.muscl_limiter), parse_reconstruction(p.reconstruction),
                                                            p.weno_epsilon);

    std::unique_ptr<ImmersedBoundary> local_ibm;   // the strip's solid cells (the loads and the residual norms)
    const auto build_local_ibm = [&] {   // again after the body moved
        const std::vector<char>& gm = ibm->mask();   // global mask: identical to the serial run
        auto gmask = [&](int k, int j, int i) {
            return gm[(static_cast<size_t>(k) * p.ny + (j + p.ny) % p.ny) * p.nx + (i + p.nx) % p.nx];
        };
        std::vector<char> lmask;
        for (int k = 0; k < lm.nz; ++k)
            for (int j = 0; j < lm.ny; ++j)
                for (int i = 0; i < lm.nx; ++i) lmask.push_back(gmask(k, decomp.j_start + j, decomp.i_start + i));
        local_ibm = std::make_unique<ImmersedBoundary>(lm.nx, lm.ny, std::move(lmask), p.rho_in, p.p_out, lm.nz);
    };
    if (p.use_ibm) {
        build_local_ibm();
        int mine = local_ibm->solid_count(), total = 0;
        MPI_Reduce(&mine, &total, 1, MPI_INT, MPI_SUM, 0, comm);
        if (root)
            std::printf("[IBM] %s at (%g, %g) r=%g -> %d solid cells (across %d ranks)\n", p.ibm_shape.c_str(),
                        p.ibm_cx, p.ibm_cy, p.ibm_radius, total, decomp.n_ranks);
    }

    // MPI-interface sides start transmissive (overridden by halos anyway)
    bool periodic = false;
    BCType l = parse_bc(p.bc_xmin), r = parse_bc(p.bc_xmax), b = parse_bc(p.bc_ymin), tp = parse_bc(p.bc_ymax);
    if (dir == 'y') {
        periodic = (b == BCType::Periodic || tp == BCType::Periodic);
        if (!(decomp.is_lo_rank && !periodic)) b = BCType::Transmissive;
        if (!(decomp.is_hi_rank && !periodic)) tp = BCType::Transmissive;
    } else {
        periodic = (l == BCType::Periodic || r == BCType::Periodic);
        if (!(decomp.is_lo_rank && !periodic)) l = BCType::Transmissive;
        if (!(decomp.is_hi_rank && !periodic)) r = BCType::Transmissive;
    }
    BoundaryManager lb(lm, fluid);
    if (p.positivity_fallback) {
        // every rank repeats a stage together: the "the set grew" decision is the OR over the ranks; the flags are those of the local strip
        fb_cells_.assign(static_cast<size_t>(lm.num_cells()), 0);
        fb_reduce_ = [&decomp](bool grew) { return decomp.allreduce_max(grew ? 1.0 : 0.0) > 0.0; };
    }
    lb.set_boundary_types(l, r, b, tp);
    lb.set_type(Face::Front, parse_bc(p.bc_zmin));   // z sides are never split
    lb.set_type(Face::Back, parse_bc(p.bc_zmax));
    lb.set_params(bc_params(p));
    {
        auto reducer = [&decomp](double& sum, double& count) {
            double in[2] = {sum, count}, out[2];
            MPI_Allreduce(in, out, 2, MPI_DOUBLE, MPI_SUM, decomp.comm);
            sum = out[0];
            count = out[1];
        };
        // The sides parallel to the strips are split across all ranks
        for (Face f :
             dir == 'y' ? std::array<Face, 2>{Face::Left, Face::Right} : std::array<Face, 2>{Face::Bottom, Face::Top})
            lb.set_mean_reducer(f, reducer);
        if (lm.is3d()) {   // front / back are split across every rank as well
            lb.set_mean_reducer(Face::Front, reducer);
            lb.set_mean_reducer(Face::Back, reducer);
        }
    }
    if (root) {
        // Domain BCs (rank 0's own l/r/b/tp have interface sides set to transmissive)
        std::printf("[MPI] Domain BCs  left=%s, right=%s, bottom=%s, top=%s\n", bc_name(parse_bc(p.bc_xmin)).c_str(),
                    bc_name(parse_bc(p.bc_xmax)).c_str(), bc_name(parse_bc(p.bc_ymin)).c_str(),
                    bc_name(parse_bc(p.bc_ymax)).c_str());
        if (dir == 'y') {
            std::printf("[MPI] Local meshes ready  -  each rank: nx=%d, ny_local varies\n", p.nx);
            if (periodic) std::printf("[MPI] Periodic Y detected - wrap-around halo exchange enabled\n");
        } else {
            std::printf("[MPI] Local meshes ready  -  each rank: ny=%d, nx_local varies\n", p.ny);
            if (periodic) std::printf("[MPI] Periodic X detected - wrap-around halo exchange enabled\n");
        }
    }

    // Initial field on rank 0
    Field U_global;
    int start = 0;
    if (root) {
        if (p.resume && fs::exists(checkpoint_path())) {
            start = load_checkpoint();
            U_global = U;
        } else {
            if (p.resume)
                std::printf("\n[WARNING] No checkpoint at %s. Starting from t=0.\n", checkpoint_path().c_str());
            U_global = p.use_hot_start ? load_hot_start(p.hot_start_file) : initial_field();
        }
    }
    MPI_Bcast(&t, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&start, 1, MPI_INT, 0, comm);
    {   // a resumed MUSCL-Hancock run: the lagged wave speed read by rank 0 is the step of every rank
        double lagged = scheme->lagged_speed();
        MPI_Bcast(&lagged, 1, MPI_DOUBLE, 0, comm);
        scheme->set_lagged_speed(lagged);
    }

    Field U_local = decomp.scatter(U_global);
    // Ghost-cell wall: the ghost table of the whole mesh (same on every rank); each rank fills its own ghost cells and fetches
    // the few cells of other strips their image points read (they may lie several strips away), so MPI gives the serial bits
    const bool ghost = p.use_ibm;
    const auto local_of = [&decomp](int c) { return decomp.local_index(c); };
    MPIDecomposition::CellFetch ghost_fetch;
    std::vector<State> ghost_remote;
    std::unordered_map<int, int> ghost_pos;
    const auto plan_ghost_fetch = [&] {   // collective; again after the body moved
        ghost_fetch = decomp.plan_fetch(ibm->ghost_inputs(local_of));
        ghost_pos.clear();
        for (size_t k = 0; k < ghost_fetch.want.size(); ++k) ghost_pos[ghost_fetch.want[k]] = static_cast<int>(k);
    };
    if (ghost) plan_ghost_fetch();
    const auto cells_of = [&decomp](const Field& Ul, const std::vector<State>& remote, const std::unordered_map<int, int>& pos) {
        return [&decomp, &Ul, &remote, &pos](int c) -> const State& {
            const int l = decomp.local_index(c);
            return l >= 0 ? Ul[static_cast<size_t>(l)] : remote[static_cast<size_t>(pos.at(c))];
        };
    };
    // ghost cells of this strip from Uin, written to Uout (also Uout = Uin: only solid cells are written)
    const auto fill_ghosts = [&](const Field& Uin, Field& Uout, bool reset) {
        decomp.fetch(ghost_fetch, Uin, ghost_remote);
        ibm->fill_ghosts_part(local_of, cells_of(Uin, ghost_remote, ghost_pos), Uout, fluid, reset);
    };
    // The cells fill_ghosts writes in this strip (the ghost cells and the deeper solid cells: every solid cell). The residual forces the
    // state in place and puts them back after the assembly (the way Simulation::assemble does: the same bits as forcing a copy of the strip,
    // without the copy): the cells the ghost fill reads are fluid cells, which it never writes.
    std::vector<int> forced_local;
    std::vector<State> forced_saved;
    const auto list_forced = [&] {
        forced_local.clear();
        std::vector<char> solid(ibm->mask().size(), 0);
        for (size_t c = 0; c < solid.size(); ++c) solid[c] = ibm->mask()[c] ? 1 : 0;
        for (const int c : ibm->ghost_cell_list()) solid[static_cast<size_t>(c)] = 1;
        for (size_t c = 0; c < solid.size(); ++c)
            if (solid[c] && local_of(static_cast<int>(c)) >= 0) forced_local.push_back(local_of(static_cast<int>(c)));
    };
    if (ghost) list_forced();
    if (ghost && start == 0) fill_ghosts(Field(U_local), U_local, false);
    if (root)
        std::printf("[MPI] Initial field scattered to all ranks  (total cells: %d, per-rank avg: %d)\n\n",
                    mesh.num_cells(), mesh.num_cells() / decomp.n_ranks);

    std::unique_ptr<PostprocessWriter> post;
    if (root) post = std::make_unique<PostprocessWriter>(*test_case, exporter.output_dir(), file_prefix, start);
    Field R_post;
    const auto post_state = [&](int st, bool force, bool emit) {
        if (!has_postprocess(*test_case) || (!force && !postprocess_state_due(*test_case, st))) return;
        if (!emit && test_case->averages.empty()) {
            if (root) post->skip_step(st);
            return;
        }
        Field global = decomp.gather(U_local);
        if (!root) return;
        if (emit) post->write_uniform(mesh, fluid, global, st, t, force);
        else post->skip_step(st);
        if (post->has_averages()) post->observe_averages(st, t, volume_primitive_means(fluid, mesh, global), force, emit);
        if (emit && post->has_force_due(st, force)) {
            residual(global, R_post);
            post->write_forces(st, t, ibm_loads(mesh, *ibm, global, R_post), mesh, force);
        }
    };

    Field halo_lo, halo_hi, flag_field, flag_lo, flag_hi;
    // MUSCL-Hancock with an immersed body: the sides of the strip's cells that face a solid cell, from the mask of the whole mesh (a cell next to
    // the interface sees the solid cells of the neighbouring strip, as in the serial run)
    std::vector<unsigned char> strip_sides;
    const auto build_strip_sides = [&] {
        if (p.use_ibm && dir == 'y') strip_sides = solid_side_mask(mesh, ibm->mask(), decomp.j_start, decomp.j_start + lm.ny);
        if (p.use_ibm && dir == 'x') strip_sides = solid_side_mask(mesh, ibm->mask(), 0, -1, decomp.i_start, decomp.i_start + lm.nx);
    };
    build_strip_sides();
    const bool hancock = dynamic_cast<const MusclHancock*>(scheme.get()) != nullptr;
    const ResidualFn res = [&](const Field& Uin, Field& R) {
        const Field* Uc = &Uin;
        if (ghost) {
            Field& Uw = const_cast<Field&>(Uin);   // restored below; every caller passes a state it owns
            forced_saved.resize(forced_local.size());
            for (size_t k = 0; k < forced_local.size(); ++k) forced_saved[k] = Uw[static_cast<size_t>(forced_local[k])];
            fill_ghosts(Uw, Uw, false);
        }
        bool has_lo, has_hi;
        decomp.exchange_halos(*Uc, periodic, halo_lo, halo_hi, has_lo, has_hi);
        StripHalos h{dir, has_lo ? &halo_lo : nullptr, has_hi ? &halo_hi : nullptr};
        if (hancock) {
            h.solid = p.use_ibm ? &local_ibm->mask() : nullptr;
            h.solid_sides = p.use_ibm ? &strip_sides : nullptr;
            h.exchange_faces = [&](const Field& to_lo, const Field& to_hi, Field& from_lo, Field& from_hi) {
                bool a = false, b = false;
                decomp.exchange_rows(to_lo, to_hi, periodic, from_lo, from_hi, a, b);
                if (!a) from_lo.clear();
                if (!b) from_hi.clear();
            };
        }
        if (fb_active_) {
            // positivity_fallback: the flags of the strip's border cells go to the neighbours the way the states do (slot 0 of a State)
            flag_field.assign(Uc->size(), State{});
            for (size_t c = 0; c < flag_field.size(); ++c) flag_field[c][0] = fb_cells_[c] ? 1.0 : 0.0;
            bool flag_has_lo, flag_has_hi;
            decomp.exchange_halos(flag_field, periodic, flag_lo, flag_hi, flag_has_lo, flag_has_hi);
            h.flag_lo = flag_has_lo ? &flag_lo : nullptr;
            h.flag_hi = flag_has_hi ? &flag_hi : nullptr;
        }
        update_boundary_values(scheme->stage_time, lb);
        const double s = assemble(lm, lb, nullptr, local_muscl.get(), *Uc, R, &h);
        if (ghost) {
            Field& Uw = const_cast<Field&>(Uin);
            for (size_t k = 0; k < forced_local.size(); ++k) Uw[static_cast<size_t>(forced_local[k])] = forced_saved[k];
        }
        return decomp.allreduce_max(s);
    };
    // Once per step: the ghost cells from the fluid, the deeper solid cells back to the reference state (the cells it reads are
    // fluid cells: no copy of the strip is needed)
    const auto solid_reset = [&](Field& Uloc) { fill_ghosts(Uloc, Uloc, true); };
    const double min_delta = dt_length;

    if (start == 0) {
        Field g = decomp.gather(U_local);
        if (root) exporter.export_vtr(file_prefix, fluid, g, mesh, t, 0);
    }
    post_state(start, start == 0, start == 0);
    if (root) {
        std::printf("=== MPI TIME-MARCHING  (%d ranks | t_end = %g s) ===\n", decomp.n_ranks, p.t_end);
        std::printf("[MPI] Output: full domain gathered to rank 0 before every VTK write\n");
        std::fflush(stdout);
    }

    int step = start;
    start_step = start;
    std::FILE* mon = root ? open_monitor() : nullptr;
    arm_stop_file();
    const bool monitoring = p.monitor_interval > 0;
    const long long n_all = static_cast<long long>(p.nx) * p.ny * p.nz;
    Field R_loads;
    bool r_valid = false;   // R_loads is the residual of the current state, which the first stage of the next step takes (offer_first_residual)
    double r_speed = 0.0;
    auto row = [&](int st) {   // collective: every rank calls it
        scheme->stage_time = t;
        update_boundary_values(t, lb);
        const auto v = volume_averages(U_local, n_all);
        std::array<double, 3> loads{};
        r_speed = res(U_local, R_loads);   // residual of the current state: the IBM loads and the norms, summed over the strips
        r_valid = true;
        if (p.use_ibm) {
            const std::array<double, 3> mine = ibm_loads(lm, *local_ibm, U_local, R_loads);
            MPI_Allreduce(mine.data(), loads.data(), 3, MPI_DOUBLE, MPI_SUM, comm);
        }
        ResSums mine_rs{}, rs{};
        for (int c = 0; c < lm.num_cells(); ++c)
            if (!p.use_ibm || !local_ibm->mask()[c]) add_residual(mine_rs, R_loads[c], 1.0);
        MPI_Allreduce(mine_rs.data(), rs.data(), 4, MPI_DOUBLE, MPI_SUM, comm);
        monitor_row(mon, st, v, p.use_ibm ? &loads : nullptr, residual_norms(rs));
    };
    const bool moving = p.use_ibm && ibm->moving();
    // Every rank places the body at the same time with the same bits (the pose is a function of t; the replicated ghost table is rebuilt
    // everywhere together). What depends on the table is made again: the strip's solid cells, the ghost fetch plan (collective), the list of forced
    // cells, the Hancock sides; the cells the body left take the wall-layer state from fluid cells fetched from the other strips.
    const auto place_body = [&](double tt) {
        if (!ibm->move_to(mesh, prescribed_pose(tt))) return;
        MPIDecomposition::CellFetch plan = decomp.plan_fetch(ibm->fresh_inputs(local_of));
        std::vector<State> remote;
        decomp.fetch(plan, U_local, remote);
        std::unordered_map<int, int> pos;
        for (size_t k = 0; k < plan.want.size(); ++k) pos[plan.want[k]] = static_cast<int>(k);
        ibm->fill_fresh_part(local_of, cells_of(U_local, remote, pos), U_local, fluid);
        build_local_ibm();
        plan_ghost_fetch();
        list_forced();
        build_strip_sides();
    };
    const auto offer_residual = [&] {
        if (r_valid && !moving) scheme->offer_first_residual(R_loads, r_speed);   // a moving body: the table changes before the next step
        r_valid = false;
    };
    if (moving) place_body(t);
    if (monitoring && step == 0) row(step);
    offer_residual();
    LoopEngine engine;
    engine.advance = [&](int st, double t_stop) {
        if (moving && st > start_step + 1) place_body(t);   // the first step of a run was placed above
        const StepResult sr = scheme->step(U_local, res, min_delta, t, std::min(p.t_end, t_stop), p.dt_max);
        check_progress(t, sr.dt, sr.s_max, st, p.time_scheme.c_str());
        if (moving) {   // as in the serial loop: at most one cell per step
            const double travel = sr.dt * std::max(std::abs(p.ibm_vx) / mesh.dx, std::abs(p.ibm_vy) / mesh.dy);
            if (travel > 1.0) throw std::runtime_error("the moving body travelled more than one cell in a step (lower the CFL number)");
        }
        if (p.use_ibm) solid_reset(U_local);
        t += sr.dt;
        scheme->stage_time = t;
        update_boundary_values(t, lb);
        return sr;
    };
    engine.after_step = [&](int st) {
        if (monitoring && st % p.monitor_interval == 0) row(st);
        post_state(st, false, true);
        offer_residual();
    };
    engine.save = [&](int st, const StepResult& sr, double wall) {
        Field g = decomp.gather(U_local);
        if (root) {
            exporter.export_vtr(file_prefix, fluid, g, mesh, t, st);
            U = std::move(g);
            save_checkpoint(st);
            std::printf("Step %05d | t = %.5f s | dt = %.3e s | S_max = %.2f m/s | wall = %s | Checkpoint saved\n",
                        st, t, sr.dt, sr.s_max, format_elapsed(wall).c_str());
            std::fflush(stdout);
        }
    };
    // Agree on Ctrl+C across ranks so nobody blocks in a collective
    engine.stop_requested = [&] { return decomp.allreduce_max(g_stop_requested ? 1.0 : 0.0) > 0.0; };
    const LoopResult loop = run_loop(engine);
    step = loop.step;
    const double loop_wall = loop.wall;
    const bool interrupted = loop.interrupted;
    if (root && p.positivity_fallback) std::printf("[POSITIVITY] %lld repair passes (the same on every rank)\n", fb_repairs_);

    if (monitoring && step % p.monitor_interval != 0) row(step);
    if (mon) std::fclose(mon);
    post_state(step, true, true);
    Field g = decomp.gather(U_local);
    if (root) {
        if (interrupted) std::printf("\n[INTERRUPTED] Stopping...\n");
        if (loop.snapshot_due) exporter.export_vtr(file_prefix, fluid, g, mesh, t, step);
        U = std::move(g);
        save_checkpoint(step);
        exporter.write_pvd(file_prefix);
        dump_final_state(step);
        std::printf("=== DONE  - open '%s.pvd' in ParaView ===  (time loop %.6f s, %d steps)\n", file_prefix.c_str(),
                    loop_wall, step - start);
    }
}
#endif   // FV_USE_MPI

}   // namespace fv
