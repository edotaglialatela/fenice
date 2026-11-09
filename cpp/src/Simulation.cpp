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

namespace fv {

using namespace sim_detail;

std::atomic<bool> g_stop_requested{false};


std::string format_elapsed(double s) {
    char buf[32];
    if (s < 60)
        std::snprintf(buf, sizeof(buf), "%.1fs", s);
    else
        std::snprintf(buf, sizeof(buf), "%d:%02d", static_cast<int>(s) / 60, static_cast<int>(s) % 60);
    return buf;
}

namespace {
// ANSI foreground colour: 24-bit, or the nearest entry of the xterm-256 colour cube.
std::string ansi_rgb(int colors, int r, int g, int b) {
    char buf[32];
    if (colors == 24) {
        std::snprintf(buf, sizeof(buf), "\x1b[38;2;%d;%d;%dm", r, g, b);
    } else {
        auto level = [](int v) { return v < 48 ? 0 : v < 115 ? 1 : (v - 35) / 40; };   // 0,95,135,175,215,255
        std::snprintf(buf, sizeof(buf), "\x1b[38;5;%dm", 16 + 36 * level(r) + 6 * level(g) + level(b));
    }
    return buf;
}
}   // namespace

void print_logo(int colors) {
    std::ifstream in(std::string(FV_PROJECT_ROOT) + "/cpp/logo.txt", std::ios::binary);
    if (!in) return;
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
    }
    // Phoenix gradient, top to bottom: halo gold, orange, red, oxblood (the GUI's brand colours)
    static const int stops[][3] = {{0xF2, 0xC6, 0x5C}, {0xE8, 0x8E, 0x3A}, {0xD2, 0x50, 0x3A}, {0xB2, 0x37, 0x44}};
    const int n = static_cast<int>(lines.size()), ns = 4;
    for (int r = 0; r < n; ++r) {
        if (colors) {
            const double t = n > 1 ? static_cast<double>(r) / (n - 1) * (ns - 1) : 0.0;
            const int k = std::min(static_cast<int>(t), ns - 2);
            const double f = t - k;
            int c[3];
            for (int q = 0; q < 3; ++q)
                c[q] = static_cast<int>(stops[k][q] + (stops[k + 1][q] - stops[k][q]) * f + 0.5);
            std::fputs(ansi_rgb(colors, c[0], c[1], c[2]).c_str(), stdout);
        }
        std::fputs(lines[r].c_str(), stdout);
        std::fputs(colors ? "\x1b[0m\n" : "\n", stdout);
    }
    if (colors) {
        const std::string gold = ansi_rgb(colors, stops[0][0], stops[0][1], stops[0][2]);
        std::printf("\n%s\x1b[1m%54s\x1b[0m\n\x1b[2m%67s\x1b[0m\n\n", gold.c_str(), "F E N I C E",
                    "compressible finite-volume solver");
    } else {
        std::printf("\n%54s\n%67s\n\n", "F E N I C E", "compressible finite-volume solver");
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

std::string Simulation::airfoil_file_path() const {
    const fs::path f(p.ibm_file);
    if (f.is_absolute()) return f.string();
    const fs::path in_case = fs::path(test_case->source).parent_path() / f;   // next to the case.toml
    if (fs::exists(in_case)) return in_case.string();
    if (fs::exists(f)) return f.string();
    throw std::runtime_error("Airfoil file '" + p.ibm_file + "' not found (looked in " +
                             in_case.parent_path().string() + " and the current folder)");
}

void Simulation::load_boundary_functions() {
    boundary_expressions_.clear();
    boundary_tables_.clear();
    const std::array<std::string, 4> keys = {"u_in", "v_in", "rho_in", "p_out"};
    for (const std::string& key : keys) {
        const std::string raw = p.source.str(key, key == "rho_in" || key == "p_out" ? "1.0" : "0.0");
        const std::string table_key = key + "_table";
        const std::string table_name = p.source.str(table_key, "");
        char* end = nullptr;
        std::strtod(raw.c_str(), &end);
        const bool numeric = end && end != raw.c_str() && *end == '\0';
        if (!table_name.empty()) {
            if (!numeric) throw std::invalid_argument(key + " expression and " + table_key + " cannot both be set");
            fs::path table_path(table_name);
            if (!table_path.is_absolute()) {
                const fs::path beside_case = fs::path(test_case->source).parent_path() / table_path;
                std::error_code ec;
                if (fs::exists(beside_case, ec)) table_path = beside_case;
                else if (!fs::exists(table_path, ec))
                    throw std::runtime_error("Boundary time table '" + table_name + "' not found next to " + test_case->source +
                                             " or in the current folder");
            }
            boundary_tables_[key] = read_time_table(table_path);
        } else if (!numeric) {
            boundary_expressions_[key] = compile_time_expression(p.source, mesh, raw, key);
        }
    }

    const auto initial = [&](const std::string& key, double fallback) {
        const auto table = boundary_tables_.find(key);
        if (table != boundary_tables_.end()) return table_value(table->second, 0.0);
        const auto expr = boundary_expressions_.find(key);
        return expr != boundary_expressions_.end() ? expr->second(0.0) : fallback;
    };
    p.u_in = initial("u_in", p.u_in);
    p.v_in = initial("v_in", p.v_in);
    p.rho_in = initial("rho_in", p.rho_in);
    p.p_out = initial("p_out", p.p_out);
    if (!(p.rho_in > 0.0) || !(p.p_out > 0.0))
        throw std::invalid_argument("time-dependent rho_in and p_out must be positive at t=0");
    if (p.reynolds > 0.0) {
        if (!(p.rho_in > 0.0) || !(p.u_in > 0.0))
            throw std::invalid_argument("reynolds needs positive u_in and rho_in at t=0 (including boundary tables)");
        const bool airfoil = !naca_digits(p.ibm_shape).empty() || lower(p.ibm_shape) == "airfoil";
        const double length = airfoil ? p.ibm_chord : 2.0 * p.ibm_radius;
        p.mu = p.rho_in * p.u_in * length / p.reynolds;
        char value[32];
        std::snprintf(value, sizeof(value), "%.17g", p.mu);
        p.source.set("mu", value);
    }
}

void Simulation::update_boundary_values(double stage_time, BoundaryManager& manager) const {
    if (boundary_expressions_.empty() && boundary_tables_.empty()) return;
    BCParams values = manager.params();
    const auto evaluate = [&](const std::string& key, double current) {
        const auto table = boundary_tables_.find(key);
        if (table != boundary_tables_.end()) return table_value(table->second, stage_time);
        const auto expression = boundary_expressions_.find(key);
        return expression == boundary_expressions_.end() ? current : expression->second(stage_time);
    };
    values.u_in = evaluate("u_in", values.u_in);
    values.v_in = evaluate("v_in", values.v_in);
    values.rho_in = evaluate("rho_in", values.rho_in);
    values.p_out = evaluate("p_out", values.p_out);
    if (!(values.rho_in > 0.0) || !(values.p_out > 0.0))
        throw std::runtime_error("time-dependent rho_in and p_out must stay positive (at t=" + std::to_string(stage_time) + ")");
    manager.set_params(values);
}

void Simulation::setup(const SimParams& params, bool verbose) {
    p = params;
    test_case = &get_test_case(p.test_case);
    if (!test_case->forces.empty() && !p.use_ibm)
        throw std::invalid_argument("[[force]] outputs need use_ibm = true");

    scheme = make_time_scheme(parse_time_scheme(p.time_scheme), p.cfl);

    file_prefix = to_prefix(test_case->key);   // the case name, also for a case given by path
    // AMR runs go to <case>_amr so they don't overwrite the uniform-mesh run.
    const std::string subdir = p.use_amr ? file_prefix + "_amr" : file_prefix;
    exporter = VTKExporter((fs::path(p.output_root) / subdir).string(), verbose, p.vtk_format != "ascii");

    mesh = generate_cartesian_mesh_3d(p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz);
    load_boundary_functions();
    const auto inside_axis = [](double x, double origin, double length) { return x >= origin && x <= origin + length; };
    for (const ProbeSpec& probe : test_case->probes) {
        if (!inside_axis(probe.x, mesh.x0, mesh.Lx) || !inside_axis(probe.y, mesh.y0, mesh.Ly) ||
            (mesh.is3d() && (!probe.has_z || !inside_axis(probe.z, mesh.z0, mesh.Lz))) ||
            (!mesh.is3d() && probe.has_z && probe.z != 0.0))
            throw std::invalid_argument("probe '" + probe.name + "' coordinates must lie inside the simulation domain");
    }
    for (const SampleSpec& sample : test_case->samples) {
        const size_t dimensions = mesh.is3d() ? 3 : 2;
        if (sample.from.size() != dimensions || sample.to.size() != dimensions)
            throw std::invalid_argument("sample '" + sample.name + "' needs " + std::to_string(dimensions) +
                                        " coordinates for this mesh");
        for (size_t d = 0; d < dimensions; ++d) {
            const double origin = d == 0 ? mesh.x0 : (d == 1 ? mesh.y0 : mesh.z0);
            const double length = d == 0 ? mesh.Lx : (d == 1 ? mesh.Ly : mesh.Lz);
            if (!inside_axis(sample.from[d], origin, length) || !inside_axis(sample.to[d], origin, length))
                throw std::invalid_argument("sample '" + sample.name + "' endpoints must lie inside the simulation domain");
        }
    }
    if (verbose && mesh.is3d()) std::printf("[MESH] 3-D: %d x %d x %d cells\n", p.nx, p.ny, p.nz);
    // CFL length: min(dx, dy) in 2-D. In 3-D the unsplit fluxes add three directional Courant
    // numbers; 2 / sum(1/h) bounds that sum by 2 CFL, the same bound min(dx, dy) gives a 2-D square cell.
    dt_length = step_length_2d(mesh.Lx / mesh.nx, mesh.Ly / mesh.ny);
    if (mesh.is3d())
        dt_length = step_length_3d(mesh.Lx / mesh.nx, mesh.Ly / mesh.ny, mesh.Lz / mesh.nz);
#ifdef _OPENMP
    if (verbose) std::printf("[OPENMP] Native kernels on %d thread(s)\n", omp_get_max_threads());
#endif

    if (lower(p.eos_type) == "ideal gas")
        fluid = EOS::ideal(p.gamma);
    else
        fluid = EOS::stiffened(p.gamma, p.p_inf);

    if (p.use_ibm) {
        const std::string digits = naca_digits(p.ibm_shape);
        if (!digits.empty() || lower(p.ibm_shape) == "airfoil") {
            const Polygon2D unit = digits.empty() ? load_airfoil(airfoil_file_path()) : naca4(digits);
            const Polygon2D placed = place_airfoil(unit, p.ibm_chord, p.ibm_aoa, p.ibm_cx, p.ibm_cy);
            ibm_factory = [placed, rho = p.rho_in, pout = p.p_out](const Mesh& m) {
                return ImmersedBoundary(m, placed, rho, pout);
            };
            ibm = std::make_unique<ImmersedBoundary>(ibm_factory(mesh));
            if (verbose)
                std::printf("[IBM] %s, chord %g, %g deg, quarter chord at (%g, %g) -> %d solid cells\n",
                            digits.empty() ? airfoil_file_path().c_str() : p.ibm_shape.c_str(), p.ibm_chord, p.ibm_aoa,
                            p.ibm_cx, p.ibm_cy, ibm->solid_count());
        } else {
            ibm_factory = [shape = p.ibm_shape, r = p.ibm_radius, cx = p.ibm_cx, cy = p.ibm_cy, rho = p.rho_in,
                           pout = p.p_out, cz = p.ibm_cz](const Mesh& m) {
                return ImmersedBoundary(m, shape, r, cx, cy, rho, pout, cz);
            };
            ibm = std::make_unique<ImmersedBoundary>(ibm_factory(mesh));
            if (verbose)
                std::printf("[IBM] %s at (%g, %g) r=%g -> %d solid cells\n", p.ibm_shape.c_str(), p.ibm_cx, p.ibm_cy,
                            p.ibm_radius, ibm->solid_count());
        }
        if (ibm->solid_count() == 0)
            std::printf("[WARNING] The IBM body covers no cell: check its size and position\n");
        if (p.ibm_method == "ghost") {
            ibm->enable_ghost_cells(mesh, !p.use_viscous, p.ibm_curvature);
            if (verbose)
                std::printf("[IBM] Ghost-cell wall (%s): %d ghost cells, loads on the true surface\n",
                            p.use_viscous ? "no-slip" : "slip", ibm->ghost_count());
        }
        if (verbose && p.reynolds > 0.0) std::printf("[PHYSICS] Re = %g -> mu = %g\n", p.reynolds, p.mu);
    }

    boundaries = BoundaryManager(mesh, fluid);
    boundaries.set_boundary_types(parse_bc(p.bc_left), parse_bc(p.bc_right), parse_bc(p.bc_bottom), parse_bc(p.bc_top));
    boundaries.set_type(Face::Front, parse_bc(p.bc_front));
    boundaries.set_type(Face::Back, parse_bc(p.bc_back));
    boundaries.set_params(bc_params(p));
    update_boundary_values(0.0, boundaries);

    const RiemannType rt = parse_riemann(p.riemann_solver);
    riemann = RiemannSolver(fluid, rt, 0.1, p.shock_blend);
    if (verbose && rt == RiemannType::LowMachHLLC) std::printf("[SOLVER] Low-Mach correction active (Thornber 2008)\n");
    if (verbose && rt == RiemannType::PreconditionedHLLC)
        std::printf("[SOLVER] Preconditioned HLLC active (Weiss-Smith 1995, Ma_cut=0.1)\n");

    if (verbose && p.use_muscl_alias)
        std::printf("[PARAMS] use_muscl is retired: read as reconstruction = %s (set reconstruction alone: none, muscl, weno3 "
                    "or weno5)\n",
                    p.reconstruction.c_str());
    if (p.reconstructs()) {
        muscl = std::make_unique<MusclReconstruction>(fluid, parse_limiter(p.muscl_limiter), parse_reconstruction(p.reconstruction), p.weno_epsilon);
        if (verbose && p.reconstruction == "muscl")
            std::printf("[SOLVER] 2nd-Order MUSCL active (Limiter: %s)\n", p.muscl_limiter.c_str());
        if (verbose && p.reconstruction != "muscl") std::printf("[SOLVER] %s reconstruction active\n", p.reconstruction.c_str());
        if (verbose && parse_time_scheme(p.time_scheme) == TimeSchemeType::ExplicitEuler)
            std::printf("[WARNING] MUSCL with Explicit Euler is unstable for smooth waves (amplitudes grow): "
                        "use SSP Runge-Kutta 2 or 3\n");
    } else {
        muscl.reset();
        if (verbose) std::printf("[SOLVER] 1st-Order Spatial (Godunov) active\n");
    }

    if (p.use_viscous) {
        ns = std::make_unique<NavierStokes>(fluid, p.mu, p.pr);
        ns->set_second_order_boundary(p.boundary_order == 2);
        if (verbose) std::printf("[PHYSICS] Viscous fluxes enabled (Navier-Stokes)\n");
    } else {
        ns.reset();
        if (verbose) std::printf("[PHYSICS] Inviscid flow (Euler)\n");
    }

    if (p.use_bodyforce) {
        body_force = std::make_unique<ConstantBodyForce>(ConstantBodyForce{p.gx, p.gy, mesh.is3d() ? p.gz : 0.0});
        if (verbose && mesh.is3d())
            std::printf("[PHYSICS] Body Forces enabled: gx=%g, gy=%g, gz=%g\n", p.gx, p.gy, p.gz);
        else if (verbose)
            std::printf("[PHYSICS] Body Forces enabled: gx=%g, gy=%g\n", p.gx, p.gy);
    } else {
        body_force.reset();
    }

    rhs_sources.clear();
    if (body_force) {
        RhsSourceTerm term;
        term.uniform = [this](const Mesh&, const Field& Uin, Field& R, double) {
            ProfileScope profile(Section::BodyForce);
            body_force->add_to(Uin, R);
        };
        term.adaptive = [this](const CellMesh&, const Field& Uin, Field& R, int begin, int end, double, const Field*) {
            ProfileScope profile(Section::BodyForce);
            const ConstantBodyForce& body = *body_force;
#pragma omp parallel for schedule(static) if (end - begin >= kOmpMinWork)
            for (int c = begin; c < end; ++c) {
                State& r = R[static_cast<size_t>(c - begin)];
                const State& u = Uin[static_cast<size_t>(c)];
                r[1] += u[0] * body.gx;
                r[2] += u[0] * body.gy;
                r[3] += u[1] * body.gx + u[2] * body.gy;
                if (body.gz != 0.0) {
                    r[3] += u[4] * body.gz;
                    r[4] += u[0] * body.gz;
                }
            }
        };
        rhs_sources.add(std::move(term));
    }
    if (p.use_source) {
        RhsSourceTerm term;
        term.uniform = [this](const Mesh& m, const Field&, Field& R, double stage_time) {
            ProfileScope profile(Section::Source);
            add_source(source_for(m, stage_time), R);
        };
        term.adaptive = [this](const CellMesh&, const Field&, Field& R, int begin, int end, double, const Field* cached) {
            ProfileScope profile(Section::Source);
            if (!cached) throw std::logic_error("expression source field is not available");
#pragma omp parallel for schedule(static) if (end - begin >= kOmpMinWork)
            for (int c = begin; c < end; ++c)
                for (int q = 0; q < kNVar; ++q)
                    R[static_cast<size_t>(c - begin)][static_cast<size_t>(q)] +=
                        (*cached)[static_cast<size_t>(c)][static_cast<size_t>(q)];
        };
        rhs_sources.add(std::move(term));
    }

    // AMR: the adaptive mesh and its engine; on several MPI ranks the leaves are distributed over them (CellParallel)
    cells.reset();
    cell_engine.reset();
    cell_adaptor.reset();
    cell_par.reset();
    hot_interp_.reset();
    hot_interp3_.reset();
    cell_comm = CellComm{};
    if (p.use_amr) {
        const bool per_x = boundaries.type(Face::Left) == BCType::Periodic;
        const bool per_y = boundaries.type(Face::Bottom) == BCType::Periodic;
        const bool per_z = boundaries.type(Face::Front) == BCType::Periodic;
        if (mesh.is3d())   // an octree (docs/design/amr-octree.md); not written yet: snapshots, post-processing, checkpoints
            cells = std::make_unique<CellMesh>(CellMesh::make_3d(p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz, p.amr_max_level, per_x, per_y,
                                                                 per_z, mesh.x0, mesh.y0, mesh.z0));
        else
            cells = std::make_unique<CellMesh>(p.Lx, p.Ly, p.nx, p.ny, p.amr_max_level, per_x, per_y);
        cell_physics = CellPhysics{};
        cell_physics.ns = ns.get();
        cell_physics.body = body_force.get();
        cell_physics.rhs_sources = rhs_sources.empty() ? nullptr : &rhs_sources;
        if (p.use_source) {
            cell_physics.source = &p.source;
            cell_physics.source_unsteady = p.source_unsteady;
            cell_physics.source_mesh = &mesh;
        }
        if (p.use_ibm) {
            cell_physics.ibm_factory = ibm_factory;
            cell_physics.ibm_ghost = p.ibm_method == "ghost";
            cell_physics.ibm_curvature = p.ibm_curvature;
            cell_physics.ibm_slip = !p.use_viscous;
        }
        cell_engine = std::make_unique<CellFluxEngine>(fluid, riemann, muscl.get(), boundaries);
        cell_engine->set_physics(cell_physics);
        cell_indicator = CellIndicator{};
        cell_indicator.criterion = p.amr_criterion;
        cell_indicator.split = p.amr_refine_threshold;
        cell_indicator.join = p.amr_derefine_threshold;
        cell_indicator.diffusion = p.amr_diffusion;
        cell_indicator.use_density = p.amr_indicator.find("density") != std::string::npos;
        cell_indicator.use_velocity = p.amr_indicator.find("velocity") != std::string::npos;
        cell_indicator.use_pressure = p.amr_indicator.find("pressure") != std::string::npos;
        cell_adaptor = std::make_unique<CellAdaptor>(*cells, *cell_engine, fluid, cell_indicator);
        if (!p.amr_force_box.empty()) {
            double b[6];
            const int got = std::sscanf(p.amr_force_box.c_str(), "%lf%*[ ,]%lf%*[ ,]%lf%*[ ,]%lf%*[ ,]%lf%*[ ,]%lf", &b[0], &b[1], &b[2], &b[3],
                                        &b[4], &b[5]);
            if (mesh.is3d() && got == 6)
                cell_adaptor->set_force_box3(b[0], b[1], b[2], b[3], b[4], b[5]);
            else if (got >= 4)
                cell_adaptor->set_force_box(b[0], b[1], b[2], b[3]);   // (3-D: every z)
            else
                throw std::invalid_argument("amr_force_box needs x0, x1, y0, y1 (and z0, z1 in 3-D)");
        }
        if (p.use_ibm && p.amr_refine_ibm && mesh.is3d()) {
            // 3-D: a shell of finest leaves around the surface (near_body_surface); the ghost-cell wall is ready with 6 cells
            const int margin = p.ibm_method == "ghost" ? 6 : 2;
            cell_adaptor->set_body(ibm_factory, margin);
            if (verbose) std::printf("[AMR] Body band: %d finest-level cells on both sides of the surface\n", margin);
        } else if (p.use_ibm && p.amr_refine_ibm) {
            int margin = 2;   // the fringe the solid reset treats
            if (p.ibm_method == "ghost") {   // every ghost cell and the cells it reads (and the load probes) at the finest level
                const int M = p.amr_max_level;
                const Mesh fine = generate_cartesian_mesh(p.Lx, p.Ly, p.nx << M, p.ny << M, mesh.x0, mesh.y0);
                ImmersedBoundary ib = ibm_factory(fine);
                ib.enable_ghost_cells(fine, !p.use_viscous, p.ibm_curvature);
                margin = std::max(margin, ib.ghost_reach());
                if (verbose) std::printf("[AMR] Ghost-cell wall: finest-level band of %d cells around the body\n", margin);
            }
            cell_adaptor->set_body(ibm_factory, margin);
        }
        if (verbose)
            std::printf("[AMR] Enabled - %d levels, indicator %s (variation %g, refine at %g, join below %g, %d smoothing "
                        "iterations), subcycling active\n",
                        p.amr_max_level, p.amr_indicator.c_str(), p.amr_criterion, p.amr_refine_threshold,
                        p.amr_derefine_threshold, p.amr_diffusion);
#ifdef FV_USE_MPI
        int init = 0, size = 1;
        MPI_Initialized(&init);
        if (init) MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size > 1) {
            cell_comm.size = size;
            MPI_Comm_rank(MPI_COMM_WORLD, &cell_comm.rank);
            cell_comm.allgatherv = [](const double* send, int n, double* recv, const int* counts, const int* displs) {
                MPI_Allgatherv(send, n, MPI_DOUBLE, recv, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
            };
            cell_comm.allreduce_max = [](double v) {
                double g = v;
                MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                return g;
            };
            cell_comm.allreduce_sum = [](double v) {
                double g = v;
                MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                return g;
            };
            cell_comm.exchange = [](const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv) {
                std::vector<MPI_Request> req;
                req.reserve(send.size() + recv.size());
                for (size_t r = 0; r < recv.size(); ++r)
                    if (!recv[r].empty()) {
                        req.emplace_back();
                        MPI_Irecv(recv[r].data(), static_cast<int>(recv[r].size()), MPI_DOUBLE, static_cast<int>(r), 91,
                                  MPI_COMM_WORLD, &req.back());
                    }
                for (size_t r = 0; r < send.size(); ++r)
                    if (!send[r].empty()) {
                        req.emplace_back();
                        MPI_Isend(send[r].data(), static_cast<int>(send[r].size()), MPI_DOUBLE, static_cast<int>(r), 91,
                                  MPI_COMM_WORLD, &req.back());
                    }
                MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE);
            };
            cell_par = std::make_unique<CellParallel>(*cells, *cell_engine, cell_comm);
        }
#endif
    }

    if (verbose)
        std::printf("Mesh: %d cells | Riemann: %s | Time: %s | CFL: %g\n", mesh.num_cells(), p.riemann_solver.c_str(),
                    p.time_scheme.c_str(), p.cfl);
    std::fflush(stdout);
}

// FENICE_MEM_TRACE=1: the resident and peak memory of the process at a step of the setup (stderr)
static void mem_trace(const char* where) {
    if (!switch_on("FENICE_MEM_TRACE")) return;
    const MemoryUsage m = memory_usage();
    std::fprintf(stderr, "MEMTRACE %s: resident %.0f MB, peak %.0f MB\n", where, m.current * 1e-6, m.peak * 1e-6);
}

void Simulation::initialize_field() {
    if (test_case->requires_viscosity && !ns)
        std::printf("[WARNING] This test case requires Navier-Stokes - viscous physics is off!\n");

    mem_setup_ = memory_usage();
    // An adaptive run needs a global field only for a hot start (interpolated onto the leaves)
    const bool global_field = !cells || p.use_hot_start;
    if (global_field) U = p.use_hot_start ? load_hot_start(p.hot_start_file) : initial_field();
    start_step = 0;
    if (p.resume) start_step = load_checkpoint();

    // A fresh start (also resume without a checkpoint) is forced; a restored state already is
    if (p.use_ibm && start_step == 0 && !cells) ibm->apply_forcing(U, fluid);
    if (!cells) return;

    if (p.use_hot_start) {
        // Seed every leaf by sampling the hot-start field at the leaf's cell centre (bilinear, extrapolating at the edges)
        // (a source of another size than the base mesh is sampled at its own resolution)
        const bool own = hot_src_nx_ > 0;
        const int nxs = own ? hot_src_nx_ : mesh.nx, nys = own ? hot_src_ny_ : mesh.ny;
        const double dx_s = mesh.Lx / nxs, dy_s = mesh.Ly / nys;
        if (cells->dim == 3) {
            const int nzs = own ? hot_src_nz_ : mesh.nz;
            const double dz_s = mesh.Lz / nzs;
            hot_interp3_ = std::make_unique<GridInterp3D>(linspace(dx_s / 2.0, mesh.Lx - dx_s / 2.0, nxs),
                                                          linspace(dy_s / 2.0, mesh.Ly - dy_s / 2.0, nys),
                                                          linspace(dz_s / 2.0, mesh.Lz - dz_s / 2.0, nzs), own ? hot_src_ : U);
        } else {
            hot_interp_ = std::make_unique<GridInterp2D>(linspace(dx_s / 2.0, mesh.Lx - dx_s / 2.0, nxs),
                                                         linspace(dy_s / 2.0, mesh.Ly - dy_s / 2.0, nys), own ? hot_src_ : U);
        }
    }
    // Several ranks with the shares kept separately: no rank ever holds the whole mesh (the topology is faceless, the share is built
    // from it, and the initial passes run through CellParallel::remesh with the initial condition filled on the share)
    const bool distributed_init = start_step == 0 && cell_par && !replicated_storage();
    if (distributed_init) {
        {   // the base topology is only needed to cut the first share
            CellMesh topo(*cells);
            topo.U = Field();
            topo.set_faceless(true);
            for (int k = 0; k < p.amr_init_levels; ++k)
                if (!topo.adapt(std::vector<char>(static_cast<size_t>(topo.num_leaves()), 1))) break;
            mem_trace("base topology");
            cell_par->init_adapt(topo, *cell_adaptor, [this](CellMesh& m) { fill_cells(m); });
        }
        if (p.use_ibm && p.ibm_method == "ghost")
            std::printf(cell_engine->ghost_ready() ? "[IBM] Ghost-cell wall active on the finest-level band\n"
                                                   : "[WARNING] Ghost-cell wall inactive: the cells around the body are not all at the "
                                                     "finest level; the mask forcing is used\n");
        if (p.use_ibm) {
            cell_engine->solid_reset(*cells, cells->U);
            cell_par->exchange_solid();
        }
        mem_trace("init_adapt done");
        cell_par->reset_timers();   // the report of the remesh counts the time loop only
        if (switch_on("FENICE_HALO_REACH") && cell_comm.rank == 0)
            std::printf("[AMR] FENICE_HALO_REACH needs the whole mesh: use a serial run or FENICE_REPLICATED_REMESH=1\n");
    } else if (start_step == 0) {
        // The initial condition is evaluated at each leaf's own resolution, so refined leaves see the exact initial
        // discontinuities rather than their interpolation from the base grid
        fill_cells(*cells);
        cell_engine->build(*cells);
        mem_trace("base built");
        for (int k = 0; k < p.amr_init_levels; ++k) {
            if (!cells->adapt(std::vector<char>(static_cast<size_t>(cells->num_leaves()), 1))) break;
            cell_engine->build(*cells);
            fill_cells(*cells);
        }
        mem_trace("init levels");
        cell_adaptor->init_adapt([this](CellMesh& m) { fill_cells(m); });
        if (p.use_ibm && p.ibm_method == "ghost")
            std::printf(cell_engine->ghost_ready() ? "[IBM] Ghost-cell wall active on the finest-level band\n"
                                                   : "[WARNING] Ghost-cell wall inactive: the cells around the body are not all at the "
                                                     "finest level; the mask forcing is used\n");
        if (p.use_ibm) cell_engine->solid_reset(*cells, cells->U);
        mem_trace("init_adapt done");
    }
    if (cell_par && !distributed_init && !share_resumed_) {
        cell_par->partition();
        print_halo_reach("initial mesh");
        if (!replicated_storage()) {
            mem_trace("before localize");
            cell_par->localize();   // each rank keeps only its leaves and their halo
            mem_trace("after localize");
        }
    }
    mem_trace("init end");
    mem_init_ = memory_usage();
}

bool Simulation::replicated_storage() {
    return switch_on("FENICE_REPLICATED_REMESH");
}

void Simulation::print_halo_reach(const char* when) const {
    if (!switch_on("FENICE_HALO_REACH") || cell_comm.rank != 0) return;
    const CellParallel::HaloReach hr = CellParallel::halo_reach(*cells, *cell_engine);
    // per kind: the largest ring, and per level difference (level of the driver - level of the leaf read: ring)
    std::printf("[AMR] halo reach (%s), rings of the driver's level [level difference driver - read: ring]:", when);
    const char* names[] = {"interior", "coarse-fine", "boundary", "ghost images", "solid reset"};
    for (int k = 0; k < CellParallel::HaloReach::NumKinds; ++k) {
        std::printf("%s %s %d [", k ? ";" : "", names[k], hr.max(k));
        bool first = true;
        for (size_t d = 0; d < hr.ring[k].size(); ++d)
            if (hr.ring[k][d] >= 0) {
                std::printf("%s%+d: %d", first ? "" : ", ", static_cast<int>(d) - cells->max_level, hr.ring[k][d]);
                first = false;
            }
        std::printf("]");
    }
    std::printf("; reads per ring:");
    for (size_t k = 0; k < hr.reads.size(); ++k) std::printf(" %zu: %lld", k, hr.reads[k]);
    std::printf("\n");
    std::fflush(stdout);
}

void Simulation::fill_cells(CellMesh& m) const {
    m.U.assign(static_cast<size_t>(m.num_leaves()), State{});
    if (hot_interp3_) {
        for (int c = 0; c < m.num_leaves(); ++c) m.U[static_cast<size_t>(c)] = (*hot_interp3_)(m.xc(c), m.yc(c), m.zc(c));
        return;
    }
    if (hot_interp_) {
        for (int c = 0; c < m.num_leaves(); ++c) m.U[static_cast<size_t>(c)] = (*hot_interp_)(m.xc(c), m.yc(c));
        return;
    }
    for (int l = 0; l <= m.max_level; ++l) {
        if (m.level_begin[l + 1] == m.level_begin[l]) continue;
        Mesh ml = m.dim == 3 ? generate_cartesian_mesh_3d(m.Lx, m.Ly, m.Lz, m.nx_at(l), m.ny_at(l), m.nz_at(l), m.x0, m.y0, m.z0)
                             : generate_cartesian_mesh(m.Lx, m.Ly, m.nx_at(l), m.ny_at(l), m.x0, m.y0);
        // only the box of cells that holds this level's leaves (the values are those of the whole level: Mesh window)
        int i0 = ml.nx, i1 = 0, j0 = ml.ny, j1 = 0;
        for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c) {
            i0 = std::min(i0, m.ci[static_cast<size_t>(c)]);
            i1 = std::max(i1, m.ci[static_cast<size_t>(c)] + 1);
            j0 = std::min(j0, m.cj[static_cast<size_t>(c)]);
            j1 = std::max(j1, m.cj[static_cast<size_t>(c)] + 1);
        }
        ml.win_i0 = i0;
        ml.win_i1 = i1;
        ml.win_j0 = j0;
        ml.win_j1 = j1;
        const Field F = test_case->initialize(ml, fluid, p.source);
        for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c)
            m.U[static_cast<size_t>(c)] =
                F[static_cast<size_t>(ml.window_cell(m.ci[static_cast<size_t>(c)], m.cj[static_cast<size_t>(c)], m.ck[static_cast<size_t>(c)]))];
    }
}

std::array<double, 3> Simulation::cell_loads(const CellMesh& m, const Field& R) const {
    const std::vector<char>& solid = cell_engine->solid();
    double fx = 0.0, fy = 0.0, mz = 0.0;
    for (int l = 0; l <= m.max_level; ++l)   // the leaves this rank owns, in leaf order (all of them without a partition)
        for (int c = cell_engine->owned_begin(l); c < cell_engine->owned_end(l); ++c) {
            if (!solid[static_cast<size_t>(c)]) continue;
            double rx = R[static_cast<size_t>(c)][1], ry = R[static_cast<size_t>(c)][2];
            if (body_force) {
                rx -= m.U[static_cast<size_t>(c)][0] * body_force->gx;
                ry -= m.U[static_cast<size_t>(c)][0] * body_force->gy;
            }   // not a load
            const double ex = rx * m.volume(c), ey = ry * m.volume(c);
            fx += ex;
            fy += ey;
            mz += (m.xc(c) - p.ibm_cx) * ey - (m.yc(c) - p.ibm_cy) * ex;
        }
    return {fx, fy, mz};
}


Field Simulation::initial_field() const {
    if (test_case->three_d) {
        if (!mesh.is3d()) throw std::invalid_argument("Test case '" + test_case->key + "' is 3-D: set nz > 1");
        return test_case->initialize(mesh, fluid, p.source);
    }
    if (!mesh.is3d()) return test_case->initialize(mesh, fluid, p.source);
    // 2-D case on a 3-D mesh: initialise one layer and extrude it along z
    const Mesh m2 = generate_cartesian_mesh(mesh.Lx, mesh.Ly, mesh.nx, mesh.ny);
    const Field layer = test_case->initialize(m2, fluid, p.source);
    Field U3(static_cast<size_t>(mesh.num_cells()));
    for (int k = 0; k < mesh.nz; ++k)
        std::copy(layer.begin(), layer.end(), U3.begin() + static_cast<std::ptrdiff_t>(k) * layer.size());
    return U3;
}

// ---------------------------------------------------------------------------
// Residuals
// ---------------------------------------------------------------------------

// Reconstruction and Riemann solver in one pass (MusclReconstruction::muscl_fluxes): the MUSCL scheme on the AVX2 kernels; the
// results are the same as with the two passes (FENICE_NO_FUSED=1 takes those, for A/B timing).
static bool fused_enabled() {
    static const bool off = switch_on("FENICE_NO_FUSED");
    return !off;
}
static bool fused_muscl(const MusclReconstruction& mu) { return fused_enabled() && mu.scheme() == Reconstruction::Muscl; }

double Simulation::assemble(const Mesh& m, const BoundaryManager& bm, const ImmersedBoundary* ib,
                            const MusclReconstruction* mu, const Field& U_in, Field& R, const StripHalos* halos) {
    if (m.is3d()) return assemble_3d(m, bm, ib, mu, U_in, R, halos);
    const Field* Uc = &U_in;
    if (ib) {
        ProfileScope profile(Section::IBMForcing);
        Uf_ = U_in;
        ib->apply_forcing(Uf_, fluid);
        Uc = &Uf_;
    }
    const int nx = m.nx, ny = m.ny, nxf = m.num_xfaces();
    constexpr int D = kHaloDepth;

    TangentialHalos th;
    if (halos) th = {halos->dir, D, halos->lo, halos->hi};
    // with a reconstruction every face between two cells is overwritten below: only the sides need their first-order states
    {
        ProfileScope ps(Section::Boundary);
        bm.apply_boundary_conditions(*Uc, UL_, UR_, halos ? &th : nullptr, !(mu ? true : fused_enabled()));
    }

    Halos mh;
    if (halos) {
        // MPI halos override the physical-BC ghost states at interface faces
        if (halos->dir == 'y') {
            if (halos->lo)
                for (int i = 0; i < nx; ++i) UL_[m.boundary_face(Face::Bottom, i)] = (*halos->lo)[(D - 1) * nx + i];
            if (halos->hi)
                for (int i = 0; i < nx; ++i) UR_[m.boundary_face(Face::Top, i)] = (*halos->hi)[i];
            mh.b = halos->lo;
            mh.t = halos->hi;
        } else {
            if (halos->lo)
                for (int j = 0; j < ny; ++j) UL_[m.boundary_face(Face::Left, j)] = (*halos->lo)[j * D + D - 1];
            if (halos->hi)
                for (int j = 0; j < ny; ++j) UR_[m.boundary_face(Face::Right, j)] = (*halos->hi)[j * D];
            mh.l = halos->lo;
            mh.r = halos->hi;
        }
    }
    // WENO5-Z reads three cells: its periodic halos and ghost layers are three deep (serial only)
    const int Dh = (mu && mu->scheme() == Reconstruction::Weno5) ? 3 : D;
    mh.depth = Dh;
    // Periodic axes inside the (sub)domain: wrap-around halos so MUSCL also reconstructs
    // the periodic boundary faces (otherwise they would be a 1st-order seam).
    const bool per_x = bm.type(Face::Left) == BCType::Periodic && bm.type(Face::Right) == BCType::Periodic;
    const bool per_y = bm.type(Face::Bottom) == BCType::Periodic && bm.type(Face::Top) == BCType::Periodic;
    if (mu && per_x && !mh.l && !mh.r && nx >= Dh) {
        perL_.resize(static_cast<size_t>(ny) * Dh);
        perR_.resize(static_cast<size_t>(ny) * Dh);
        for (int j = 0; j < ny; ++j)
            for (int d = 0; d < Dh; ++d) {
                perL_[j * Dh + d] = (*Uc)[j * nx + nx - Dh + d];
                perR_[j * Dh + d] = (*Uc)[j * nx + d];
            }
        mh.l = &perL_;
        mh.r = &perR_;
    }
    if (mu && per_y && !mh.b && !mh.t && ny >= Dh) {
        perB_.assign(Uc->end() - static_cast<std::ptrdiff_t>(Dh) * nx, Uc->end());
        perT_.assign(Uc->begin(), Uc->begin() + static_cast<std::ptrdiff_t>(Dh) * nx);
        mh.b = &perB_;
        mh.t = &perT_;
    }
    // boundary_order = 2: the physical sides get ghost layers (the BC's own even / odd extension), so that the faces on
    // the boundary are reconstructed like the interior ones. A side that is periodic, an MPI interface (halo already
    // set) or non-reflecting keeps its first-order face states.
    if (mu && p.boundary_order == 2) {
        const auto ghost = [&](Face side, const Field*& slot, Field& buf) {
            if (slot) return;
            const BCType bt = bm.type(side);
            if (bt == BCType::Periodic || bt == BCType::NonReflecting) return;
            if (bm.fill_ghost_slab(*Uc, side, Dh, buf)) slot = &buf;
        };
        ghost(Face::Left, mh.l, ghL_);
        ghost(Face::Right, mh.r, ghR_);
        ghost(Face::Bottom, mh.b, ghB_);
        ghost(Face::Top, mh.t, ghT_);
    }
    double S;
    if (mu && fused_muscl(*mu)) {
        // reconstruction, Riemann solver and flux balance in one pass over rows of faces: the face fluxes are not stored
        ProfileScope ps(Section::Flux);
        S = mu->muscl_residual_2d(riemann, UL_, UR_, *Uc, m, mh, R);
    } else if (!mu && fused_enabled()) {
        ProfileScope ps(Section::Flux);   // first order: the same pass, the states of the faces are the cells
        S = riemann.first_order_residual_2d(UL_, UR_, *Uc, m, R);
    } else {
        if (mu) {
            ProfileScope ps(Section::Reconstruct);
            mu->reconstruct(UL_, UR_, *Uc, nx, ny, mh);
        }
        {
            ProfileScope ps(Section::Flux);
            S = riemann.compute_flux(UL_, UR_, m, F_);
        }
        ProfileScope ps(Section::Divergence);
        R.resize(Uc->size());
        const double inv_area = 1.0 / m.cell_area(), dxl = m.dx, dyl = m.dy;
#pragma omp parallel for schedule(static) if (nx * ny >= kOmpMinWork)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const State& fl = F_[j * (nx + 1) + i];
                const State& fr = F_[j * (nx + 1) + i + 1];
                const State& fb = F_[nxf + j * nx + i];
                const State& ft = F_[nxf + (j + 1) * nx + i];
                State& r = R[j * nx + i];
                for (int q = 0; q < kNVar; ++q)
                    r[q] = ((0.0 - (fr[q] * dyl - fl[q] * dyl)) - (ft[q] * dxl - fb[q] * dxl)) * inv_area;
            }
    }
    if (ns) {
        ProfileScope ps(Section::Viscous);
        {
            // the ghost ring of the padded state only: the interior is the cell field, which the viscous sweep reads in place
            ProfileScope profile(Section::PadCopy);
            bm.get_padded_frame(*Uc, frame_);
            if (halos) {
                if (halos->dir == 'y') {
                    if (halos->lo)
                        for (int i = 0; i < nx; ++i) frame_.bottom[i + 1] = (*halos->lo)[(D - 1) * nx + i];
                    if (halos->hi)
                        for (int i = 0; i < nx; ++i) frame_.top[i + 1] = (*halos->hi)[i];
                } else {
                    if (halos->lo)
                        for (int j = 0; j < ny; ++j) frame_.left[j] = (*halos->lo)[j * D + D - 1];
                    if (halos->hi)
                        for (int j = 0; j < ny; ++j) frame_.right[j] = (*halos->hi)[j * D];
                }
            }
            bm.fix_frame_corners(frame_, halos ? (halos->dir == 'x' ? 0 : 1) : -1, halos && halos->lo, halos && halos->hi);
        }
        CrossGhosts cg;
        cg.periodic_x = per_x;
        cg.periodic_y = per_y;
        std::vector<double> lo_a, lo_b, hi_a, hi_b;
        if (halos) {
            cg.dir = halos->dir;
            if (halos->lo) {
                strip_tangential_derivs(fluid, bm, m, halos->dir, adjacent_strip(*halos->lo, halos->dir, true, nx, ny),
                                        lo_a, lo_b);
                cg.lo_a = lo_a.data();
                cg.lo_b = lo_b.data();
            }
            if (halos->hi) {
                strip_tangential_derivs(fluid, bm, m, halos->dir, adjacent_strip(*halos->hi, halos->dir, false, nx, ny),
                                        hi_a, hi_b);
                cg.hi_a = hi_a.data();
                cg.hi_b = hi_b.data();
            }
        }
        const double s_visc = ns->viscous_residual_rows(*Uc, frame_, m, R, &cg, true);
        S = std::max(S, s_visc);
    }

    if (!rhs_sources.empty()) rhs_sources.apply_uniform(m, *Uc, R, scheme ? scheme->stage_time : 0.0);
    return S;
}

double Simulation::assemble_3d(const Mesh& m, const BoundaryManager& bm, const ImmersedBoundary* ib,
                               const MusclReconstruction* mu, const Field& U_in, Field& R, const StripHalos* halos) {
    const Field* Uc = &U_in;
    if (ib) {
        ProfileScope profile(Section::IBMForcing);
        Uf_ = U_in;
        ib->apply_forcing(Uf_, fluid);
        Uc = &Uf_;
    }
    const Field& Uv = *Uc;
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    constexpr int D = kHaloDepth;

    TangentialHalos th;
    if (halos) th = {halos->dir, D, halos->lo, halos->hi};
    {
        ProfileScope ps(Section::Boundary);
        bm.apply_boundary_conditions(Uv, UL_, UR_, halos ? &th : nullptr, mu == nullptr);
    }

    // MPI halos (slab layouts, see Halos) override the BC ghosts at interface faces
    Halos mh;
    if (halos) {
        if (halos->dir == 'y') {
            for (int k = 0; k < nz; ++k)
                for (int i = 0; i < nx; ++i) {
                    if (halos->lo)
                        UL_[m.boundary_face(Face::Bottom, k * nx + i)] = (*halos->lo)[((D - 1) * nz + k) * nx + i];
                    if (halos->hi) UR_[m.boundary_face(Face::Top, k * nx + i)] = (*halos->hi)[k * nx + i];
                }
            mh.b = halos->lo;
            mh.t = halos->hi;
        } else {
            for (int s = 0; s < ny * nz; ++s) {
                if (halos->lo) UL_[m.boundary_face(Face::Left, s)] = (*halos->lo)[s * D + D - 1];
                if (halos->hi) UR_[m.boundary_face(Face::Right, s)] = (*halos->hi)[s * D];
            }
            mh.l = halos->lo;
            mh.r = halos->hi;
        }
    }
    const int Dh = (mu && mu->scheme() == Reconstruction::Weno5) ? 3 : D;   // WENO5-Z reads three cells
    mh.depth = Dh;
    // Periodic axes: wrap-around MUSCL halos (the boundary faces are then 2nd order too)
    const bool per[3] = {bm.type(Face::Left) == BCType::Periodic && bm.type(Face::Right) == BCType::Periodic,
                         bm.type(Face::Bottom) == BCType::Periodic && bm.type(Face::Top) == BCType::Periodic,
                         bm.type(Face::Front) == BCType::Periodic && bm.type(Face::Back) == BCType::Periodic};
    if (mu) {
        if (per[0] && !mh.l && !mh.r && nx >= Dh) {
            perL_.resize(static_cast<size_t>(ny) * nz * Dh);
            perR_.resize(perL_.size());
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int d = 0; d < Dh; ++d) {
                        perL_[(k * ny + j) * Dh + d] = Uv[m.cell(nx - Dh + d, j, k)];
                        perR_[(k * ny + j) * Dh + d] = Uv[m.cell(d, j, k)];
                    }
            mh.l = &perL_;
            mh.r = &perR_;
        }
        if (per[1] && !mh.b && !mh.t && ny >= Dh) {
            perB_.resize(static_cast<size_t>(Dh) * nz * nx);
            perT_.resize(perB_.size());
            for (int d = 0; d < Dh; ++d)
                for (int k = 0; k < nz; ++k)
                    for (int i = 0; i < nx; ++i) {
                        perB_[(d * nz + k) * nx + i] = Uv[m.cell(i, ny - Dh + d, k)];
                        perT_[(d * nz + k) * nx + i] = Uv[m.cell(i, d, k)];
                    }
            mh.b = &perB_;
            mh.t = &perT_;
        }
        if (per[2] && nz >= Dh) {
            perF_.resize(static_cast<size_t>(Dh) * ny * nx);
            perK_.resize(perF_.size());
            for (int d = 0; d < Dh; ++d)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        perF_[(d * ny + j) * nx + i] = Uv[m.cell(i, j, nz - Dh + d)];
                        perK_[(d * ny + j) * nx + i] = Uv[m.cell(i, j, d)];
                    }
            mh.f = &perF_;
            mh.bk = &perK_;
        }
        // boundary_order = 2: the physical sides get ghost layers (the BC's own even / odd extension), so that the faces on
        // the boundary are reconstructed like the interior ones. A side that is periodic, an MPI interface (halo already
        // set) or non-reflecting keeps its first-order face states.
        if (mu && p.boundary_order == 2) {
            const auto ghost = [&](Face side, const Field*& slot, Field& buf) {
                if (slot) return;
                const BCType bt = bm.type(side);
                if (bt == BCType::Periodic || bt == BCType::NonReflecting) return;
                if (bm.fill_ghost_slab(Uv, side, Dh, buf)) slot = &buf;
            };
            ghost(Face::Left, mh.l, ghL_);
            ghost(Face::Right, mh.r, ghR_);
            ghost(Face::Bottom, mh.b, ghB_);
            ghost(Face::Top, mh.t, ghT_);
            ghost(Face::Front, mh.f, ghF_);
            ghost(Face::Back, mh.bk, ghK_);
        }
    }

    double S;
    if (mu && fused_muscl(*mu)) {
        ProfileScope ps(Section::Flux);   // reconstruction and Riemann solver in one pass
        S = mu->muscl_fluxes(riemann, UL_, UR_, Uv, m, mh, F_);
    } else {
        if (mu) {
            ProfileScope ps(Section::Reconstruct);
            mu->reconstruct_3d(UL_, UR_, Uv, m, mh);
        }
        ProfileScope ps(Section::Flux);
        S = riemann.compute_flux(UL_, UR_, m, F_);
    }

    ProfileScope divergence_scope(Section::Divergence);
    R.resize(Uv.size());
    const double Ax = m.face_area(0), Ay = m.face_area(1), Az = m.face_area(2), inv_vol = 1.0 / m.cell_volume();
#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
    for (int kj = 0; kj < ny * nz; ++kj) {
        const int j = kj % ny, k = kj / ny;
        for (int i = 0; i < nx; ++i) {
            const State& fl = F_[m.xface(i, j, k)];
            const State& fr = F_[m.xface(i + 1, j, k)];
            const State& fb = F_[m.yface(i, j, k)];
            const State& ft = F_[m.yface(i, j + 1, k)];
            const State& ff = F_[m.zface(i, j, k)];
            const State& fk = F_[m.zface(i, j, k + 1)];
            State& r = R[m.cell(i, j, k)];
            for (int q = 0; q < kNVar; ++q)
                r[q] =
                    (((0.0 - (fr[q] * Ax - fl[q] * Ax)) - (ft[q] * Ay - fb[q] * Ay)) - (fk[q] * Az - ff[q] * Az)) * inv_vol;
        }
    }

    divergence_scope.stop();
    if (ns) {
        ProfileScope ps(Section::Viscous);
        {
            ProfileScope profile(Section::PadCopy);
            bm.get_padded_U(Uv, Upad_);
        }
        CrossGhosts cg;
        cg.periodic_x = per[0];
        cg.periodic_y = per[1];
        cg.periodic_z = per[2];
        std::vector<double> gh[8];   // MPI cross ghosts: lo a, b, c, d, hi a, b, c, d
        if (halos) {
            const int PX = nx + 2, PY = ny + 2;
            auto P = [&](int ii, int jj, int kk) -> State& {
                return Upad_[(static_cast<size_t>(kk) * PY + jj) * PX + ii];
            };
            if (halos->dir == 'y') {
                for (int k = 0; k < nz; ++k)
                    for (int i = 0; i < nx; ++i) {
                        if (halos->lo) P(i + 1, 0, k + 1) = (*halos->lo)[((D - 1) * nz + k) * nx + i];
                        if (halos->hi) P(i + 1, ny + 1, k + 1) = (*halos->hi)[k * nx + i];
                    }
            } else {
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j) {
                        if (halos->lo) P(0, j + 1, k + 1) = (*halos->lo)[(k * ny + j) * D + D - 1];
                        if (halos->hi) P(nx + 1, j + 1, k + 1) = (*halos->hi)[(k * ny + j) * D];
                    }
            }
            cg.dir = halos->dir;
            const double** slots[8] = {&cg.lo_a, &cg.lo_b, &cg.lo_c, &cg.lo_d, &cg.hi_a, &cg.hi_b, &cg.hi_c, &cg.hi_d};
            for (int side = 0; side < 2; ++side) {
                const Field* hal = side == 0 ? halos->lo : halos->hi;
                if (!hal) continue;
                slab_tangential_derivs(fluid, bm, m, halos->dir, adjacent_slab(*hal, halos->dir, side == 0, m),
                                       gh[4 * side], gh[4 * side + 1], gh[4 * side + 2], gh[4 * side + 3]);
                for (int q = 0; q < 4; ++q) *slots[4 * side + q] = gh[4 * side + q].data();
            }
        }
        {
            ProfileScope profile(Section::PadCornerFix);
            bm.fix_padded_corners(Upad_, halos ? (halos->dir == 'x' ? 0 : 1) : -1, halos && halos->lo, halos && halos->hi);
        }
        compute_gradients_3d(Upad_, m, gX_, gY_, gZ_);
        const double s_visc = ns->viscous_residual_3d(Uv, gX_, gY_, gZ_, m, Upad_, R, &cg, dt_length, true);
        S = std::max(S, s_visc);
    }

    if (!rhs_sources.empty()) rhs_sources.apply_uniform(m, Uv, R, scheme ? scheme->stage_time : 0.0);
    return S;
}
const Field& Simulation::source_for(const Mesh& m, double stage_time) {
    const std::array<double, 9> key = {static_cast<double>(m.nx), static_cast<double>(m.ny),
                                       static_cast<double>(m.nz), m.dx, m.dy, m.dz, m.x0, m.y0, m.z0};
    if (p.source_unsteady) {
        auto ut = unsteady_source_cache_.find(key);
        if (ut == unsteady_source_cache_.end())
            ut = unsteady_source_cache_.emplace(key, std::make_pair(std::numeric_limits<double>::quiet_NaN(), Field{}))
                     .first;
        if (!(ut->second.first == stage_time)) {   // also true for the NaN of a new entry
            ut->second.second = evaluate_source(m, p.source, stage_time);
            ut->second.first = stage_time;
        }
        return ut->second.second;
    }
    auto it = source_cache_.find(key);
    if (it == source_cache_.end()) it = source_cache_.emplace(key, evaluate_source(m, p.source)).first;
    return it->second;
}

double Simulation::residual(const Field& Uin, Field& R) {
    update_boundary_values(scheme ? scheme->stage_time : t, boundaries);
    return assemble(mesh, boundaries, p.use_ibm ? ibm.get() : nullptr, muscl.get(), Uin, R, nullptr);
}

// ---------------------------------------------------------------------------
// Checkpoints / hot start
// ---------------------------------------------------------------------------

}   // namespace fv
