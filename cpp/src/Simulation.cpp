#include "fv/Simulation.hpp"

#include <chrono>
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

namespace fv {

std::atomic<bool> g_stop_requested{false};

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

std::string to_prefix(std::string s) {
    for (auto& c : s) c = (c == ' ') ? '_' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Tangential velocity derivatives of a strip of MPI ghost cells, using the same
// padded central differences (and physical side BCs) as an interior row/column.
// dir 'y': strip is a row of nx cells -> du/dx, dv/dx; dir 'x': column of ny cells -> du/dy, dv/dy.
// R += S: the source terms of the case (source_*) enter the residual of dU/dt = R as + S
void add_source(const Field& S, Field& R) {
    const int n = static_cast<int>(R.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) R[c][q] += S[c][q];
}

void strip_tangential_derivs(const EOS& eos, const BoundaryManager& bm, const Mesh& m, char dir, const Field& strip,
                             std::vector<double>& a, std::vector<double>& b) {
    const bool row = (dir == 'y');
    const int n = row ? m.nx : m.ny;
    Mesh m1 = row ? generate_cartesian_mesh(m.Lx, m.dy, m.nx, 1) : generate_cartesian_mesh(m.dx, m.Ly, 1, m.ny);
    m1.dx = m.dx;
    m1.dy = m.dy;
    BoundaryManager b1(m1, eos);
    if (row)
        b1.set_boundary_types(bm.type(Face::Left), bm.type(Face::Right), BCType::Transmissive, BCType::Transmissive);
    else
        b1.set_boundary_types(BCType::Transmissive, BCType::Transmissive, bm.type(Face::Bottom), bm.type(Face::Top));
    b1.set_params(bm.params());
    Field P;
    b1.get_padded_U(strip, P);
    a.resize(n);
    b.resize(n);
    for (int k = 0; k < n; ++k) {
        // row: P is 3 x (nx+2), use the middle row; column: P is (ny+2) x 3, use the middle column
        const State& hi = row ? P[(m.nx + 2) + k + 2] : P[(k + 2) * 3 + 1];
        const State& lo = row ? P[(m.nx + 2) + k] : P[k * 3 + 1];
        const double h2 = row ? 2.0 * m.dx : 2.0 * m.dy;
        State g;
        for (int q = 0; q < kNVar; ++q) g[q] = (hi[q] - lo[q]) / h2;
        const State W = eos.cons_to_prim(strip[k]);
        a[k] = (g[1] - W[1] * g[0]) / W[0];
        b[k] = (g[2] - W[2] * g[0]) / W[0];
    }
}

// BC parameters of a run (the same on every MPI rank: global domain lengths).
BCParams bc_params(const SimParams& p) {
    BCParams b;
    b.u_in = p.u_in;
    b.rho_in = p.rho_in;
    b.p_out = p.p_out;
    b.nrbc_sigma = p.nrbc_sigma;
    b.nrbc_beta = p.nrbc_beta;
    b.length[0] = p.Lx;
    b.length[1] = p.Ly;
    b.length[2] = p.Lz;
    return b;
}

// Cells of the halo slice adjacent to the interface (lo: last slice, hi: first).
Field adjacent_strip(const Field& halo, char dir, bool lo, int nx, int ny) {
    constexpr int D = kHaloDepth;
    Field s;
    if (dir == 'y')
        for (int i = 0; i < nx; ++i) s.push_back(halo[(lo ? D - 1 : 0) * nx + i]);
    else
        for (int j = 0; j < ny; ++j) s.push_back(halo[j * D + (lo ? D - 1 : 0)]);
    return s;
}

// 3-D halo slab adjacent to the interface: y-decomposition -> nz*nx cells
// [k*nx + i]; x-decomposition -> nz*ny cells [k*ny + j].
Field adjacent_slab(const Field& halo, char dir, bool lo, const Mesh& m) {
    constexpr int D = kHaloDepth;
    Field s;
    if (dir == 'y') {
        for (int k = 0; k < m.nz; ++k)
            for (int i = 0; i < m.nx; ++i) s.push_back(halo[((lo ? D - 1 : 0) * m.nz + k) * m.nx + i]);
    } else {
        for (int k = 0; k < m.nz; ++k)
            for (int j = 0; j < m.ny; ++j) s.push_back(halo[(k * m.ny + j) * D + (lo ? D - 1 : 0)]);
    }
    return s;
}

// Tangential velocity derivatives of a 3-D ghost slab (CrossGhosts 3-D order), using
// the same padded central differences (and physical side BCs) as an interior slab:
//   dir 'y' (slab nx x nz): a = du/dx, b = dv/dx, c = dw/dz, d = dv/dz
//   dir 'x' (slab ny x nz): a = du/dy, b = dv/dy, c = dw/dz, d = du/dz
void slab_tangential_derivs(const EOS& eos, const BoundaryManager& bm, const Mesh& m, char dir, const Field& slab,
                            std::vector<double>& a, std::vector<double>& b, std::vector<double>& c,
                            std::vector<double>& d) {
    const bool ydec = (dir == 'y');
    // One-cell-thick mesh across the decomposition direction, same spacing and side BCs
    Mesh m1 = ydec ? generate_cartesian_mesh_3d(m.Lx, m.dy, m.Lz, m.nx, 1, m.nz)
                   : generate_cartesian_mesh_3d(m.dx, m.Ly, m.Lz, 1, m.ny, m.nz);
    m1.dx = m.dx;
    m1.dy = m.dy;
    m1.dz = m.dz;
    BoundaryManager b1(m1, eos);
    if (ydec)
        b1.set_boundary_types(bm.type(Face::Left), bm.type(Face::Right), BCType::Transmissive, BCType::Transmissive);
    else
        b1.set_boundary_types(BCType::Transmissive, BCType::Transmissive, bm.type(Face::Bottom), bm.type(Face::Top));
    b1.set_type(Face::Front, bm.type(Face::Front));
    b1.set_type(Face::Back, bm.type(Face::Back));
    b1.set_params(bm.params());
    Field P;
    b1.get_padded_U(slab, P);
    const int PX = m1.nx + 2, PY = m1.ny + 2;
    auto at = [&](int ii, int jj, int kk) -> const State& { return P[(static_cast<size_t>(kk) * PY + jj) * PX + ii]; };
    const int ns = static_cast<int>(slab.size());
    a.resize(ns);
    b.resize(ns);
    c.resize(ns);
    d.resize(ns);
    for (int k = 0; k < m.nz; ++k) {
        const int nt = ydec ? m.nx : m.ny;
        for (int t = 0; t < nt; ++t) {
            const int ii = ydec ? t + 1 : 1, jj = ydec ? 1 : t + 1, kk = k + 1;
            const int s = k * nt + t;
            const State W = eos.cons_to_prim(slab[s]);
            auto deriv = [&](int comp, int axis) {   // d(vel comp)/dx_axis via the conservative chain rule
                const State& hi =
                    axis == 0 ? at(ii + 1, jj, kk) : (axis == 1 ? at(ii, jj + 1, kk) : at(ii, jj, kk + 1));
                const State& lo =
                    axis == 0 ? at(ii - 1, jj, kk) : (axis == 1 ? at(ii, jj - 1, kk) : at(ii, jj, kk - 1));
                const double h2 = 2.0 * (axis == 0 ? m.dx : (axis == 1 ? m.dy : m.dz));
                const double g_rho = (hi[0] - lo[0]) / h2, g_mom = (hi[kMom[comp]] - lo[kMom[comp]]) / h2;
                return (g_mom - W[kMom[comp]] * g_rho) / W[0];
            };
            if (ydec) {
                a[s] = deriv(0, 0);
                b[s] = deriv(1, 0);
                c[s] = deriv(2, 2);
                d[s] = deriv(1, 2);
            } else {
                a[s] = deriv(0, 1);
                b[s] = deriv(1, 1);
                c[s] = deriv(2, 2);
                d[s] = deriv(0, 2);
            }
        }
    }
}
}   // namespace

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

void Simulation::setup(const SimParams& params, bool verbose) {
    p = params;
    test_case = &get_test_case(p.test_case);

    scheme = make_time_scheme(parse_time_scheme(p.time_scheme), p.cfl, p.dt_phys, p.max_inner_iter, p.inner_tol_rel);

    file_prefix = to_prefix(test_case->key);   // the case name, also for a case given by path
    // AMR runs go to <case>_amr so they don't overwrite the uniform-mesh run.
    const std::string subdir = p.use_amr ? file_prefix + "_amr" : file_prefix;
    exporter = VTKExporter((fs::path(p.output_root) / subdir).string(), verbose, p.vtk_format != "ascii");

    mesh = generate_cartesian_mesh_3d(p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz);
    if (verbose && mesh.is3d()) std::printf("[MESH] 3-D: %d x %d x %d cells\n", p.nx, p.ny, p.nz);
    // CFL length: min(dx, dy) in 2-D. In 3-D the unsplit fluxes add three directional Courant
    // numbers; 2 / sum(1/h) bounds that sum by 2 CFL, the same bound min(dx, dy) gives a 2-D square cell.
    dt_length = std::min(mesh.Lx / mesh.nx, mesh.Ly / mesh.ny);
    if (mesh.is3d())
        dt_length = 2.0 / (1.0 / (mesh.Lx / mesh.nx) + 1.0 / (mesh.Ly / mesh.ny) + 1.0 / (mesh.Lz / mesh.nz));
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
            ibm->enable_ghost_cells(mesh, !p.use_viscous);
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

    const RiemannType rt = parse_riemann(p.riemann_solver);
    riemann = RiemannSolver(fluid, rt, 0.1);
    if (verbose && rt == RiemannType::LowMachHLLC) std::printf("[SOLVER] Low-Mach correction active (Thornber 2008)\n");
    if (verbose && rt == RiemannType::PreconditionedHLLC)
        std::printf("[SOLVER] Preconditioned HLLC active (Weiss-Smith 1995, Ma_cut=0.1)\n");

    if (p.use_muscl) {
        muscl = std::make_unique<MusclReconstruction>(fluid, parse_limiter(p.muscl_limiter));
        if (verbose) std::printf("[SOLVER] 2nd-Order MUSCL active (Limiter: %s)\n", p.muscl_limiter.c_str());
        if (verbose && parse_time_scheme(p.time_scheme) == TimeSchemeType::ExplicitEuler)
            std::printf("[WARNING] MUSCL with Explicit Euler is unstable for smooth waves (amplitudes grow): "
                        "use SSP Runge-Kutta 2 or Runge-Kutta 4\n");
    } else {
        muscl.reset();
        if (verbose) std::printf("[SOLVER] 1st-Order Spatial (Godunov) active\n");
    }

    if (p.use_viscous) {
        ns = std::make_unique<NavierStokes>(fluid, p.mu, p.pr);
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

    // AMR: the adaptive mesh and its engine; on several MPI ranks the leaves are distributed over them (CellParallel)
    cells.reset();
    cell_engine.reset();
    cell_adaptor.reset();
    cell_par.reset();
    hot_interp_.reset();
    cell_comm = CellComm{};
    if (p.use_amr) {
        const bool per_x = boundaries.type(Face::Left) == BCType::Periodic;
        const bool per_y = boundaries.type(Face::Bottom) == BCType::Periodic;
        cells = std::make_unique<CellMesh>(p.Lx, p.Ly, p.nx, p.ny, p.amr_max_level, per_x, per_y);
        cell_physics = CellPhysics{};
        cell_physics.ns = ns.get();
        cell_physics.body = body_force.get();
        if (p.use_source) {
            cell_physics.source = &p.source;
            cell_physics.source_unsteady = p.source_unsteady;
            cell_physics.source_mesh = &mesh;
        }
        if (p.use_ibm) {
            cell_physics.ibm_factory = ibm_factory;
            cell_physics.ibm_ghost = p.ibm_method == "ghost";
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
        if (p.use_ibm && p.amr_refine_ibm) {
            int margin = 2;   // the fringe the solid reset treats
            if (p.ibm_method == "ghost") {   // every ghost cell and the cells it reads (and the load probes) at the finest level
                const int M = p.amr_max_level;
                const Mesh fine = generate_cartesian_mesh(p.Lx, p.Ly, p.nx << M, p.ny << M, mesh.x0, mesh.y0);
                ImmersedBoundary ib = ibm_factory(fine);
                ib.enable_ghost_cells(fine, !p.use_viscous);
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

void Simulation::initialize_field() {
    if (test_case->requires_viscosity && !ns)
        std::printf("[WARNING] This test case requires Navier-Stokes - viscous physics is off!\n");

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
        const double dx_s = mesh.Lx / mesh.nx, dy_s = mesh.Ly / mesh.ny;
        hot_interp_ = std::make_unique<GridInterp2D>(linspace(dx_s / 2.0, mesh.Lx - dx_s / 2.0, mesh.nx),
                                                     linspace(dy_s / 2.0, mesh.Ly - dy_s / 2.0, mesh.ny), U);
    }
    if (start_step == 0) {
        // The initial condition is evaluated at each leaf's own resolution, so refined leaves see the exact initial
        // discontinuities rather than their interpolation from the base grid
        fill_cells(*cells);
        cell_engine->build(*cells);
        for (int k = 0; k < p.amr_init_levels; ++k) {
            if (!cells->adapt(std::vector<char>(static_cast<size_t>(cells->num_leaves()), 1))) break;
            cell_engine->build(*cells);
            fill_cells(*cells);
        }
        cell_adaptor->init_adapt([this](CellMesh& m) { fill_cells(m); });
        if (p.use_ibm && p.ibm_method == "ghost")
            std::printf(cell_engine->ghost_ready() ? "[IBM] Ghost-cell wall active on the finest-level band\n"
                                                   : "[WARNING] Ghost-cell wall inactive: the cells around the body are not all at the "
                                                     "finest level; the mask forcing is used\n");
        if (p.use_ibm) cell_engine->solid_reset(*cells, cells->U);
    }
    if (cell_par) cell_par->partition();
}

void Simulation::fill_cells(CellMesh& m) const {
    m.U.assign(static_cast<size_t>(m.num_leaves()), State{});
    if (hot_interp_) {
        for (int c = 0; c < m.num_leaves(); ++c) m.U[static_cast<size_t>(c)] = (*hot_interp_)(m.xc(c), m.yc(c));
        return;
    }
    for (int l = 0; l <= m.max_level; ++l) {
        if (m.level_begin[l + 1] == m.level_begin[l]) continue;
        const Mesh ml = generate_cartesian_mesh(m.Lx, m.Ly, m.nx_at(l), m.ny_at(l), m.x0, m.y0);
        const Field F = test_case->initialize(ml, fluid, p.source);
        for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c)
            m.U[static_cast<size_t>(c)] = F[static_cast<size_t>(m.cj[static_cast<size_t>(c)]) * m.nx_at(l) + m.ci[static_cast<size_t>(c)]];
    }
}

std::array<double, 3> Simulation::cell_loads(const CellMesh& m, const Field& R) const {
    const std::vector<char>& solid = cell_engine->solid();
    double fx = 0.0, fy = 0.0, mz = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c) {
        if (!solid[static_cast<size_t>(c)]) continue;
        double rx = R[static_cast<size_t>(c)][1], ry = R[static_cast<size_t>(c)][2];
        if (body_force) {
            rx -= m.U[static_cast<size_t>(c)][0] * body_force->gx;
            ry -= m.U[static_cast<size_t>(c)][0] * body_force->gy;
        }   // not a load
        const double ex = rx * m.area(c), ey = ry * m.area(c);
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

double Simulation::assemble(const Mesh& m, const BoundaryManager& bm, const ImmersedBoundary* ib,
                            const MusclReconstruction* mu, const Field& U_in, Field& R, const StripHalos* halos) {
    if (m.is3d()) return assemble_3d(m, bm, ib, mu, U_in, R, halos);
    const Field* Uc = &U_in;
    if (ib) {
        Uf_ = U_in;
        ib->apply_forcing(Uf_, fluid);
        Uc = &Uf_;
    }
    const int nx = m.nx, ny = m.ny, nxf = m.num_xfaces();
    constexpr int D = kHaloDepth;

    TangentialHalos th;
    if (halos) th = {halos->dir, D, halos->lo, halos->hi};
    bm.apply_boundary_conditions(*Uc, UL_, UR_, halos ? &th : nullptr);

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
    // Periodic axes inside the (sub)domain: wrap-around halos so MUSCL also reconstructs
    // the periodic boundary faces (otherwise they would be a 1st-order seam).
    const bool per_x = bm.type(Face::Left) == BCType::Periodic && bm.type(Face::Right) == BCType::Periodic;
    const bool per_y = bm.type(Face::Bottom) == BCType::Periodic && bm.type(Face::Top) == BCType::Periodic;
    if (mu && per_x && !mh.l && !mh.r && nx >= D) {
        perL_.resize(static_cast<size_t>(ny) * D);
        perR_.resize(static_cast<size_t>(ny) * D);
        for (int j = 0; j < ny; ++j)
            for (int d = 0; d < D; ++d) {
                perL_[j * D + d] = (*Uc)[j * nx + nx - D + d];
                perR_[j * D + d] = (*Uc)[j * nx + d];
            }
        mh.l = &perL_;
        mh.r = &perR_;
    }
    if (mu && per_y && !mh.b && !mh.t && ny >= D) {
        perB_.assign(Uc->end() - static_cast<std::ptrdiff_t>(D) * nx, Uc->end());
        perT_.assign(Uc->begin(), Uc->begin() + static_cast<std::ptrdiff_t>(D) * nx);
        mh.b = &perB_;
        mh.t = &perT_;
    }
    if (mu) mu->reconstruct(UL_, UR_, *Uc, nx, ny, mh);

    double S = riemann.compute_flux(UL_, UR_, m, F_);

    R.resize(Uc->size());
    const double area = m.cell_area(), dxl = m.dx, dyl = m.dy;
#pragma omp parallel for schedule(static) if (nx * ny >= kOmpMinWork)
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const State& fl = F_[j * (nx + 1) + i];
            const State& fr = F_[j * (nx + 1) + i + 1];
            const State& fb = F_[nxf + j * nx + i];
            const State& ft = F_[nxf + (j + 1) * nx + i];
            State& r = R[j * nx + i];
            for (int q = 0; q < kNVar; ++q)
                r[q] = ((0.0 - (fr[q] * dyl - fl[q] * dyl)) - (ft[q] * dxl - fb[q] * dxl)) / area;
        }

    if (ns) {
        bm.get_padded_U(*Uc, Upad_);
        if (halos) {
            const int PW = nx + 2;
            if (halos->dir == 'y') {
                if (halos->lo)
                    for (int i = 0; i < nx; ++i) Upad_[i + 1] = (*halos->lo)[(D - 1) * nx + i];
                if (halos->hi)
                    for (int i = 0; i < nx; ++i) Upad_[(ny + 1) * PW + i + 1] = (*halos->hi)[i];
            } else {
                if (halos->lo)
                    for (int j = 0; j < ny; ++j) Upad_[(j + 1) * PW] = (*halos->lo)[j * D + D - 1];
                if (halos->hi)
                    for (int j = 0; j < ny; ++j) Upad_[(j + 1) * PW + nx + 1] = (*halos->hi)[j * D];
            }
        }
        compute_gradients(Upad_, nx, ny, m.dx, m.dy, gX_, gY_);
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
        const double s_visc = ns->viscous_residual(*Uc, gX_, gY_, m, Upad_, Rv_, &cg);
        const int n = static_cast<int>(R.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
        for (int c = 0; c < n; ++c)
            for (int q = 0; q < kNVar; ++q) R[c][q] += Rv_[c][q];
        S = std::max(S, s_visc);
    }

    if (body_force) body_force->add_to(*Uc, R);
    if (p.use_source) add_source(source_for(m), R);
    return S;
}

double Simulation::assemble_3d(const Mesh& m, const BoundaryManager& bm, const ImmersedBoundary* ib,
                               const MusclReconstruction* mu, const Field& U_in, Field& R, const StripHalos* halos) {
    const Field* Uc = &U_in;
    if (ib) {
        Uf_ = U_in;
        ib->apply_forcing(Uf_, fluid);
        Uc = &Uf_;
    }
    const Field& Uv = *Uc;
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    constexpr int D = kHaloDepth;

    TangentialHalos th;
    if (halos) th = {halos->dir, D, halos->lo, halos->hi};
    bm.apply_boundary_conditions(Uv, UL_, UR_, halos ? &th : nullptr);

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
    // Periodic axes: wrap-around MUSCL halos (the boundary faces are then 2nd order too)
    const bool per[3] = {bm.type(Face::Left) == BCType::Periodic && bm.type(Face::Right) == BCType::Periodic,
                         bm.type(Face::Bottom) == BCType::Periodic && bm.type(Face::Top) == BCType::Periodic,
                         bm.type(Face::Front) == BCType::Periodic && bm.type(Face::Back) == BCType::Periodic};
    if (mu) {
        if (per[0] && !mh.l && !mh.r && nx >= D) {
            perL_.resize(static_cast<size_t>(ny) * nz * D);
            perR_.resize(perL_.size());
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int d = 0; d < D; ++d) {
                        perL_[(k * ny + j) * D + d] = Uv[m.cell(nx - D + d, j, k)];
                        perR_[(k * ny + j) * D + d] = Uv[m.cell(d, j, k)];
                    }
            mh.l = &perL_;
            mh.r = &perR_;
        }
        if (per[1] && !mh.b && !mh.t && ny >= D) {
            perB_.resize(static_cast<size_t>(D) * nz * nx);
            perT_.resize(perB_.size());
            for (int d = 0; d < D; ++d)
                for (int k = 0; k < nz; ++k)
                    for (int i = 0; i < nx; ++i) {
                        perB_[(d * nz + k) * nx + i] = Uv[m.cell(i, ny - D + d, k)];
                        perT_[(d * nz + k) * nx + i] = Uv[m.cell(i, d, k)];
                    }
            mh.b = &perB_;
            mh.t = &perT_;
        }
        if (per[2] && nz >= D) {
            perF_.resize(static_cast<size_t>(D) * ny * nx);
            perK_.resize(perF_.size());
            for (int d = 0; d < D; ++d)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        perF_[(d * ny + j) * nx + i] = Uv[m.cell(i, j, nz - D + d)];
                        perK_[(d * ny + j) * nx + i] = Uv[m.cell(i, j, d)];
                    }
            mh.f = &perF_;
            mh.bk = &perK_;
        }
        mu->reconstruct_3d(UL_, UR_, Uv, m, mh);
    }

    double S = riemann.compute_flux(UL_, UR_, m, F_);

    R.resize(Uv.size());
    const double Ax = m.face_area(0), Ay = m.face_area(1), Az = m.face_area(2), vol = m.cell_volume();
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
                    (((0.0 - (fr[q] * Ax - fl[q] * Ax)) - (ft[q] * Ay - fb[q] * Ay)) - (fk[q] * Az - ff[q] * Az)) / vol;
        }
    }

    if (ns) {
        bm.get_padded_U(Uv, Upad_);
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
        compute_gradients_3d(Upad_, m, gX_, gY_, gZ_);
        const double s_visc = ns->viscous_residual_3d(Uv, gX_, gY_, gZ_, m, Upad_, Rv_, &cg, dt_length);
        const int n = static_cast<int>(R.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
        for (int cc = 0; cc < n; ++cc)
            for (int q = 0; q < kNVar; ++q) R[cc][q] += Rv_[cc][q];
        S = std::max(S, s_visc);
    }

    if (body_force) body_force->add_to(Uv, R);
    if (p.use_source) add_source(source_for(m), R);
    return S;
}
const Field& Simulation::source_for(const Mesh& m) {
    const std::array<double, 9> key = {static_cast<double>(m.nx), static_cast<double>(m.ny),
                                       static_cast<double>(m.nz), m.dx, m.dy, m.dz, m.x0, m.y0, m.z0};
    if (p.source_unsteady) {
        const double t = scheme ? scheme->stage_time : 0.0;
        auto ut = unsteady_source_cache_.find(key);
        if (ut == unsteady_source_cache_.end())
            ut = unsteady_source_cache_.emplace(key, std::make_pair(std::numeric_limits<double>::quiet_NaN(), Field{}))
                     .first;
        if (!(ut->second.first == t)) {   // also true for the NaN of a new entry
            ut->second.second = evaluate_source(m, p.source, t);
            ut->second.first = t;
        }
        return ut->second.second;
    }
    auto it = source_cache_.find(key);
    if (it == source_cache_.end()) it = source_cache_.emplace(key, evaluate_source(m, p.source)).first;
    return it->second;
}

double Simulation::residual(const Field& Uin, Field& R) {
    return assemble(mesh, boundaries, p.use_ibm ? ibm.get() : nullptr, muscl.get(), Uin, R, nullptr);
}

// ---------------------------------------------------------------------------
// Checkpoints / hot start
// ---------------------------------------------------------------------------

std::string Simulation::checkpoint_path() const {
    return (fs::path(exporter.output_dir()) / (file_prefix + "_restart.npz")).string();
}

void Simulation::save_checkpoint(int step) {
    const std::string path = checkpoint_path();
    const std::string tmp = path.substr(0, path.size() - 4) + "_tmp.npz";
    std::vector<double> times;
    std::vector<std::string> files;
    for (const auto& e : exporter.collection) {
        times.push_back(e.time);
        files.push_back(e.filename);
    }
    NpzArchive a;
    if (cells) {
        // an adaptive run: the leaves (level and integer coordinates) and their state, and the wave-speed bound the next
        // time step is taken from
        const size_t n = static_cast<size_t>(cells->num_leaves());
        std::vector<double> lev(n), li(n), lj(n);
        for (size_t c = 0; c < n; ++c) {
            lev[c] = cells->level[c];
            li[c] = cells->ci[c];
            lj[c] = cells->cj[c];
        }
        a["U"] = field_to_npy(cells->U);
        a["leaf_level"] = NpyArray::from_doubles(lev, {static_cast<int64_t>(n)});
        a["leaf_i"] = NpyArray::from_doubles(li, {static_cast<int64_t>(n)});
        a["leaf_j"] = NpyArray::from_doubles(lj, {static_cast<int64_t>(n)});
        a["S"] = NpyArray::from_double(cell_speed_);
    } else {
        a["U"] = field_to_npy(U, mesh.is3d() ? kNVar : 4);
    }
    a["t"] = NpyArray::from_double(t);
    a["step"] = NpyArray::from_int(step);
    a["nx"] = NpyArray::from_int(mesh.nx);
    a["ny"] = NpyArray::from_int(mesh.ny);
    a["Lx"] = NpyArray::from_double(mesh.Lx);
    a["Ly"] = NpyArray::from_double(mesh.Ly);
    a["nz"] = NpyArray::from_int(mesh.nz);
    a["Lz"] = NpyArray::from_double(mesh.Lz);
    a["collection_time"] = NpyArray::from_doubles(times, {static_cast<int64_t>(times.size())});
    a["collection_file"] = NpyArray::from_strings(files);
    write_npz(tmp, a);
    fs::rename(tmp, path);
}

int Simulation::load_checkpoint() {
    const std::string path = checkpoint_path();
    if (!fs::exists(path)) {
        std::printf("\n[WARNING] No checkpoint found in %s. Starting from zero.\n", path.c_str());
        return 0;
    }
    const NpzArchive a = read_npz(path);
    const bool adaptive_file = a.count("leaf_level") > 0;
    if (adaptive_file != (cells != nullptr))
        throw std::runtime_error(std::string("The checkpoint ") + path + " is from " +
                                 (adaptive_file ? "an adaptive (AMR)" : "a uniform-mesh") + " run: resume it with use_amr " +
                                 (adaptive_file ? "on" : "off") + " (a uniform checkpoint can seed an AMR run with a hot start)");
    if (cells) {
        const auto ints = [&](const char* key) {
            const std::vector<double> d = a.at(key).as_doubles();
            return std::vector<int>(d.begin(), d.end());
        };
        Field Uc = npy_to_field(a.at("U"));
        cells->assign_leaves(ints("leaf_level"), ints("leaf_i"), ints("leaf_j"), &Uc);
        cell_engine->build(*cells);
        cell_speed_ = a.count("S") ? a.at("S").scalar() : 0.0;
    } else {
        U = npy_to_field(a.at("U"));
    }
    t = a.at("t").scalar();
    const int step = static_cast<int>(a.at("step").scalar());
    exporter.collection.clear();
    if (a.count("collection_time") && a.count("collection_file")) {
        const auto times = a.at("collection_time").as_doubles();
        const auto files = a.at("collection_file").as_strings();
        for (size_t k = 0; k < std::min(times.size(), files.size()); ++k)
            exporter.collection.push_back({times[k], files[k]});
    } else if (a.count("collection")) {
        std::printf("[RESTART] Legacy checkpoint (pickled index): .pvd index restarts from this point\n");
    }
    std::printf("\n[RESTART] Simulation resumed from t = %.5f s (Step %d)\n", t, step);
    return step;
}

Field Simulation::load_hot_start(const std::string& path) {
    // If the source is this run's own checkpoint, back it up before the first save overwrites it.
    const std::string ck = checkpoint_path();
    std::error_code ec;
    if (fs::exists(path) && fs::exists(ck) && fs::equivalent(path, ck, ec)) {
        std::string backup = ck;
        backup.replace(backup.size() - std::string("_restart.npz").size(), std::string::npos, "_hotstart_source.npz");
        fs::copy_file(path, backup, fs::copy_options::overwrite_existing);
        std::printf("[Hot Start] Source checkpoint backed up to: %s\n", backup.c_str());
    }
    const NpzArchive a = read_npz(path);
    int nx_c, ny_c;
    double Lx_c, Ly_c;
    if (a.count("nx")) {
        nx_c = static_cast<int>(a.at("nx").scalar());
        ny_c = static_cast<int>(a.at("ny").scalar());
        Lx_c = a.at("Lx").scalar();
        Ly_c = a.at("Ly").scalar();
    } else {
        nx_c = p.hot_start_src_nx;
        ny_c = p.hot_start_src_ny;
        if (nx_c <= 0 || ny_c <= 0)
            throw std::runtime_error("Hot-start file '" + path +
                                     "' has no grid metadata and no manual Nx / Ny were provided "
                                     "(set hot_start_src_nx / hot_start_src_ny).");
        Lx_c = mesh.Lx;
        Ly_c = mesh.Ly;
        std::printf("[Hot Start] No metadata in file - using manual dims: Nx=%d, Ny=%d, Lx=%g, Ly=%g\n", nx_c, ny_c,
                    Lx_c, Ly_c);
    }
    if (std::abs(Lx_c - mesh.Lx) > 1e-10 || std::abs(Ly_c - mesh.Ly) > 1e-10)
        std::printf("[WARNING] Hot-start domain (%g x %g) differs from target (%g x %g). "
                    "Proceeding with coordinate rescaling.\n",
                    Lx_c, Ly_c, mesh.Lx, mesh.Ly);
    const int nz_c = a.count("nz") ? static_cast<int>(a.at("nz").scalar()) : 1;
    if (nz_c > 1 && !mesh.is3d()) throw std::runtime_error("Hot start from a 3-D checkpoint needs a 3-D run (nz > 1)");
    std::printf("[Hot Start] Mapping %d x %d x %d -> %d x %d x %d ...\n", nx_c, ny_c, nz_c, mesh.nx, mesh.ny, mesh.nz);
    const Field Uc = npy_to_field(a.at("U"));
    if (static_cast<int>(Uc.size()) != nx_c * ny_c * nz_c)
        throw std::runtime_error("Hot-start state has " + std::to_string(Uc.size()) + " cells, expected " +
                                 std::to_string(nx_c * ny_c * nz_c));
    if (nz_c > 1) return map_solution_3d(Uc, nx_c, ny_c, nz_c, mesh.nx, mesh.ny, mesh.nz, mesh.Lx, mesh.Ly, mesh.Lz);
    Field U2 = map_solution(Uc, nx_c, ny_c, mesh.nx, mesh.ny, mesh.Lx, mesh.Ly);
    if (!mesh.is3d()) return U2;
    Field U3(static_cast<size_t>(mesh.num_cells()));   // 2-D source on a 3-D mesh: extrude along z
    for (int k = 0; k < mesh.nz; ++k)
        std::copy(U2.begin(), U2.end(), U3.begin() + static_cast<std::ptrdiff_t>(k) * U2.size());
    return U3;
}

void Simulation::dump_final_state(int step) const {
    if (p.dump_state.empty()) return;
    NpzArchive a;
    a["t"] = NpyArray::from_double(t);
    a["step"] = NpyArray::from_int(step);
    if (cells) {
        std::vector<double> meta;
        meta.reserve(static_cast<size_t>(cells->num_leaves()) * 5);
        for (int c = 0; c < cells->num_leaves(); ++c) {
            const size_t k = static_cast<size_t>(c);
            const int l = cells->level[k];
            meta.insert(meta.end(), {cells->x0 + cells->ci[k] * cells->dx(l), cells->x0 + (cells->ci[k] + 1) * cells->dx(l),
                                     cells->y0 + cells->cj[k] * cells->dy(l), cells->y0 + (cells->cj[k] + 1) * cells->dy(l),
                                     static_cast<double>(l)});
        }
        a["U"] = field_to_npy(cells->U);
        a["leaf_meta"] = NpyArray::from_doubles(meta, {static_cast<int64_t>(meta.size() / 5), 5});
    } else {
        a["U"] = field_to_npy(U, mesh.is3d() ? kNVar : 4);
    }
    write_npz(p.dump_state, a);
}

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

std::FILE* Simulation::open_monitor() const {
    if (p.monitor_interval <= 0) return nullptr;
    const std::string path = (fs::path(exporter.output_dir()) / (file_prefix + "_monitor.csv")).string();
    std::FILE* f = std::fopen(path.c_str(), start_step > 0 ? "a" : "w");
    if (f && start_step == 0)
        std::fprintf(f, "step,t,mass,kinetic_energy,total_energy%s\n", p.use_ibm ? ",Fx,Fy,Cd,Cl,Cm" : "");
    return f;
}

void Simulation::monitor_row(std::FILE* f, int step, const std::array<double, 3>& v,
                             const std::array<double, 3>* loads) const {
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
    std::fprintf(f, "\n");
    std::fflush(f);
}

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
    if (step == 0) exporter.export_vtr(file_prefix, fluid, U, mesh, t, step);

    const ResidualFn res = [this](const Field& Uin, Field& R) { return residual(Uin, R); };
    std::FILE* mon = open_monitor();
    const long long n_all = mesh.num_cells();
    Field R_loads;
    auto row = [&](int st) {   // averages, plus the IBM loads from a residual of the current state
        std::array<double, 3> loads{};
        if (p.use_ibm) {
            residual(U, R_loads);
            loads = ibm_loads(mesh, *ibm, U, R_loads);
        }
        monitor_row(mon, st, volume_averages(U, n_all), p.use_ibm ? &loads : nullptr);
    };
    if (mon && step == 0) row(step);
    const auto t0 = Clock::now();
    bool interrupted = false;
    while (t < p.t_end) {
        const StepResult r = scheme->step(U, res, min_delta, t, p.t_end);
        if (p.use_ibm) ibm->apply_solid_reset(U, fluid);
        t += r.dt;
        ++step;
        if (mon && step % p.monitor_interval == 0) row(step);

        if (step % p.save_interval == 0) {
            exporter.export_vtr(file_prefix, fluid, U, mesh, t, step);
            save_checkpoint(step);
            char inner[32] = "";
            if (scheme->inner_iter_last() >= 0)
                std::snprintf(inner, sizeof(inner), " | inner=%d", scheme->inner_iter_last());
            std::printf("Step %05d | t = %.5f s | dt = %.3e s | S_max = %.2f m/s%s | wall = %s | Checkpoint saved\n",
                        step, t, r.dt, r.s_max, inner, format_elapsed(seconds_since(t0)).c_str());
            std::fflush(stdout);
        }
        if (g_stop_requested) {
            interrupted = true;
            break;
        }
        if (p.max_steps > 0 && step - start_step >= p.max_steps) break;
    }
    const double loop_wall = seconds_since(t0);

    if (mon) {
        if (step % p.monitor_interval != 0) row(step);
        std::fclose(mon);
    }
    if (interrupted) {
        std::printf("\n[INTERRUPTED] Stopping the simulation...\n");
        exporter.export_vtr(file_prefix, fluid, U, mesh, t, step);
    } else if (step % p.save_interval != 0) {
        exporter.export_vtr(file_prefix, fluid, U, mesh, t, step);
    }
    save_checkpoint(step);
    exporter.write_pvd(file_prefix);
    dump_final_state(step);
    std::printf("=== DONE  - open '%s.pvd' in ParaView ===  (time loop %.3f s, %d steps)\n", file_prefix.c_str(),
                loop_wall, step - start_step);
}

void Simulation::solve_amr() {
    CellMesh& cm = *cells;
    CellFluxEngine& eng = *cell_engine;
    const int nranks = cell_comm.size, rank = cell_comm.rank;
    const bool root = rank == 0;
    CellAdvance adv(cm, eng, *scheme);
    if (cell_par) adv.par = cell_par.get();
    // after this call every rank holds the state of every leaf (collective)
    const auto gather = [&]() {
        if (cell_par) cell_par->gather_state();
    };

    Field R;
    // wave-speed bound the time step is taken from, one step old (a restart reads it from the checkpoint)
    double S = start_step > 0 ? cell_speed_ : eng.residual(cm, cm.U, R);
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

    // Monitor: area-weighted means over the leaves and the IBM loads, summed in leaf order on the gathered state, so the
    // numbers do not depend on the number of ranks
    std::FILE* mon = root ? open_monitor() : nullptr;
    const bool monitoring = p.monitor_interval > 0;
    const auto row = [&](int st) {
        gather();
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, area = 0.0;
        for (int c = 0; c < cm.num_leaves(); ++c) {
            const State& u = cm.U[static_cast<size_t>(c)];
            const double a = cm.area(c);
            s0 += u[0] * a;
            s1 += 0.5 * (u[1] * u[1] + u[2] * u[2] + u[4] * u[4]) / u[0] * a;
            s2 += u[3] * a;
            area += a;
        }
        std::array<double, 3> loads{};
        if (p.use_ibm) {
            Field Rl;
            eng.residual(cm, cm.U, Rl);
            loads = cell_loads(cm, Rl);
        }
        monitor_row(mon, st, {s0 / area * mesh.cell_area(), s1 / area * mesh.cell_area(), s2 / area * mesh.cell_area()},
                    p.use_ibm ? &loads : nullptr);
    };

    if (step == 0) export_state(step);
    if (monitoring && step == 0) row(step);

    const auto t0 = Clock::now();
    double remesh_wall = 0.0, t_gather = 0.0, t_speed = 0.0, t_partition = 0.0;
    bool interrupted = false;
    while (t < p.t_end) {
        const double dt = std::min(scheme->cfl * std::min(cm.dx(0), cm.dy(0)) / std::max(S, 1e-10), p.t_end - t);
        S = adv.step(dt, t);
        cell_speed_ = S;
        if (p.use_ibm) {
            eng.solid_reset(cm, cm.U);
            if (cell_par) cell_par->exchange_solid();
        }
        t += dt;
        ++step;

        if (step % p.amr_refine_every == 0) {
            const auto r0 = Clock::now();
            gather();   // the indicator looks at every leaf
            t_gather += seconds_since(r0);
            if (cell_adaptor->remesh()) {
                const auto r2 = Clock::now();
                if (cell_par) cell_par->partition();
                t_partition += seconds_since(r2);
                const auto r1 = Clock::now();
                S = eng.wave_speed(cm, cm.U);   // the wave-speed bound follows the new leaves (each rank's leaves, then the maximum)
                if (cell_par) S = cell_par->reduce_max(S);
                cell_speed_ = S;
                t_speed += seconds_since(r1);
            }
            remesh_wall += seconds_since(r0);
        }
        if (monitoring && step % p.monitor_interval == 0) row(step);

        if (step % p.save_interval == 0) {
            export_state(step);
            gather();
            if (root) save_checkpoint(step);
            if (root)
                std::printf("Step %05d | t = %.5f s | dt = %.3e s | S_max = %.2f m/s | leaves = %d | wall = %s | Checkpoint saved\n",
                            step, t, dt, S, cm.num_leaves(), format_elapsed(seconds_since(t0)).c_str());
            std::fflush(stdout);
        }
        const double stop = g_stop_requested.load() ? 1.0 : 0.0;
        if ((cell_par ? cell_par->reduce_max(stop) : stop) > 0.0) {
            interrupted = true;
            break;
        }
        if (p.max_steps > 0 && step - start_step >= p.max_steps) break;
    }
    const double loop_wall = seconds_since(t0);

    if (monitoring) {
        if (step % p.monitor_interval != 0) row(step);
        if (mon) std::fclose(mon);
    }
    if (root && interrupted) std::printf("\n[INTERRUPTED] Stopping the simulation...\n");
    if (interrupted || step % p.save_interval != 0) export_state(step);
    gather();
    if (!root) return;
    save_checkpoint(step);
    exporter.write_pvd(file_prefix);
    dump_final_state(step);
    std::vector<int> per_level(static_cast<size_t>(cm.max_level) + 1, 0);
    for (int c = 0; c < cm.num_leaves(); ++c) ++per_level[static_cast<size_t>(cm.level[static_cast<size_t>(c)])];
    std::printf("[AMR] %d leaves at the end (", cm.num_leaves());
    for (int l = 0; l <= cm.max_level; ++l) std::printf("%slevel %d: %d", l ? ", " : "", l, per_level[static_cast<size_t>(l)]);
    std::printf("); remesh %.2f s of the %.2f s time loop\n", remesh_wall, loop_wall);
    std::printf("[AMR] remesh parts: gather %.2f s, indicator %.2f s, adapt %.2f s, engine build %.2f s, wave-speed bound %.2f s, partition %.2f s\n",
                t_gather, cell_adaptor->t_indicator, cell_adaptor->t_adapt, cell_adaptor->t_build, t_speed, t_partition);
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
    if (auto* bdf = dynamic_cast<BDF2DualTime*>(scheme.get()))
        bdf->global_norm = [&decomp](double v) { return decomp.allreduce_max(v); };

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
    if (p.use_muscl) local_muscl = std::make_unique<MusclReconstruction>(fluid, parse_limiter(p.muscl_limiter));

    std::unique_ptr<ImmersedBoundary> local_ibm;
    std::vector<char> ghost_mask_lo, ghost_mask_hi;
    if (p.use_ibm) {
        const std::vector<char>& gm = ibm->mask();   // global mask: identical to the serial run
        auto gmask = [&](int k, int j, int i) {
            return gm[(static_cast<size_t>(k) * p.ny + (j + p.ny) % p.ny) * p.nx + (i + p.nx) % p.nx];
        };
        std::vector<char> lmask;
        for (int k = 0; k < lm.nz; ++k)
            for (int j = 0; j < lm.ny; ++j)
                for (int i = 0; i < lm.nx; ++i) lmask.push_back(gmask(k, decomp.j_start + j, decomp.i_start + i));
        // Ghost masks across each interface, slab index k*nx + i (y) or k*ny + j (x)
        for (int k = 0; k < lm.nz; ++k) {
            if (dir == 'y')
                for (int i = 0; i < p.nx; ++i) {
                    ghost_mask_lo.push_back(gmask(k, decomp.j_start - 1, i));
                    ghost_mask_hi.push_back(gmask(k, decomp.j_start + decomp.ny_local, i));
                }
            else
                for (int j = 0; j < p.ny; ++j) {
                    ghost_mask_lo.push_back(gmask(k, j, decomp.i_start - 1));
                    ghost_mask_hi.push_back(gmask(k, j, decomp.i_start + decomp.nx_local));
                }
        }
        local_ibm = std::make_unique<ImmersedBoundary>(lm.nx, lm.ny, std::move(lmask), p.rho_in, p.p_out, lm.nz);
        int mine = local_ibm->solid_count(), total = 0;
        MPI_Reduce(&mine, &total, 1, MPI_INT, MPI_SUM, 0, comm);
        if (root)
            std::printf("[IBM] %s at (%g, %g) r=%g -> %d solid cells (across %d ranks)\n", p.ibm_shape.c_str(),
                        p.ibm_cx, p.ibm_cy, p.ibm_radius, total, decomp.n_ranks);
    }

    // MPI-interface sides start transmissive (overridden by halos anyway)
    bool periodic = false;
    BCType l = parse_bc(p.bc_left), r = parse_bc(p.bc_right), b = parse_bc(p.bc_bottom), tp = parse_bc(p.bc_top);
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
    lb.set_boundary_types(l, r, b, tp);
    lb.set_type(Face::Front, parse_bc(p.bc_front));   // z sides are never split
    lb.set_type(Face::Back, parse_bc(p.bc_back));
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
        std::printf("[MPI] Domain BCs  left=%s, right=%s, bottom=%s, top=%s\n", bc_name(parse_bc(p.bc_left)).c_str(),
                    bc_name(parse_bc(p.bc_right)).c_str(), bc_name(parse_bc(p.bc_bottom)).c_str(),
                    bc_name(parse_bc(p.bc_top)).c_str());
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

    Field U_local = decomp.scatter(U_global);
    // Ghost-cell wall: the ghost table of the whole mesh (same on every rank); each rank fills its own ghost cells and fetches
    // the few cells of other strips their image points read (they may lie several strips away), so MPI gives the serial bits
    const bool ghost = p.use_ibm && ibm && ibm->ghost_cells();
    const auto local_of = [&decomp](int c) { return decomp.local_index(c); };
    MPIDecomposition::CellFetch ghost_fetch;
    std::vector<State> ghost_remote;
    std::unordered_map<int, int> ghost_pos;
    if (ghost) {
        ghost_fetch = decomp.plan_fetch(ibm->ghost_inputs(local_of));
        for (size_t k = 0; k < ghost_fetch.want.size(); ++k) ghost_pos[ghost_fetch.want[k]] = static_cast<int>(k);
    }
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
    if (p.use_ibm && start == 0) {
        if (ghost)
            fill_ghosts(Field(U_local), U_local, false);
        else
            local_ibm->apply_forcing(U_local, fluid);
    }
    if (root)
        std::printf("[MPI] Initial field scattered to all ranks  (total cells: %d, per-rank avg: %d)\n\n",
                    mesh.num_cells(), mesh.num_cells() / decomp.n_ranks);

    Field halo_lo, halo_hi, forced;
    const ResidualFn res = [&](const Field& Uin, Field& R) {
        const Field* Uc = &Uin;
        if (local_ibm) {
            forced = Uin;
            if (ghost)
                fill_ghosts(Uin, forced, false);
            else
                local_ibm->apply_forcing(forced, fluid);
            Uc = &forced;
        }
        bool has_lo, has_hi;
        decomp.exchange_halos(*Uc, periodic, halo_lo, halo_hi, has_lo, has_hi);
        StripHalos h{dir, has_lo ? &halo_lo : nullptr, has_hi ? &halo_hi : nullptr};
        const double s = assemble(lm, lb, nullptr, local_muscl.get(), *Uc, R, &h);
        return decomp.allreduce_max(s);
    };
    // Solid reset with the neighbours across each interface (fringe detection)
    const auto solid_reset = [&](Field& Uloc) {
        if (ghost) {
            fill_ghosts(Field(Uloc), Uloc, true);
            return;
        }
        bool has_lo, has_hi;
        decomp.exchange_halos(Uloc, periodic, halo_lo, halo_hi, has_lo, has_hi);
        const Field slo = has_lo ? adjacent_slab(halo_lo, dir, true, lm) : Field{};
        const Field shi = has_hi ? adjacent_slab(halo_hi, dir, false, lm) : Field{};
        SolidResetHalo sh{dir, has_lo ? &slo : nullptr, has_hi ? &shi : nullptr, &ghost_mask_lo, &ghost_mask_hi};
        local_ibm->apply_solid_reset(Uloc, fluid, &sh);
    };
    const double min_delta = dt_length;

    if (start == 0) {
        Field g = decomp.gather(U_local);
        if (root) exporter.export_vtr(file_prefix, fluid, g, mesh, t, 0);
    }
    if (root) {
        std::printf("=== MPI TIME-MARCHING  (%d ranks | t_end = %g s) ===\n", decomp.n_ranks, p.t_end);
        std::printf("[MPI] Output: full domain gathered to rank 0 before every VTK write\n");
        std::fflush(stdout);
    }

    int step = start;
    start_step = start;
    std::FILE* mon = root ? open_monitor() : nullptr;
    const bool monitoring = p.monitor_interval > 0;
    const long long n_all = static_cast<long long>(p.nx) * p.ny * p.nz;
    Field R_loads;
    auto row = [&](int st) {   // collective: every rank calls it
        const auto v = volume_averages(U_local, n_all);
        std::array<double, 3> loads{};
        if (p.use_ibm) {   // IBM loads: residual of the current state (mask or ghost cells), summed over the strips
            res(U_local, R_loads);
            const std::array<double, 3> mine = ibm_loads(lm, *local_ibm, U_local, R_loads);
            MPI_Allreduce(mine.data(), loads.data(), 3, MPI_DOUBLE, MPI_SUM, comm);
        }
        monitor_row(mon, st, v, p.use_ibm ? &loads : nullptr);
    };
    if (monitoring && step == 0) row(step);
    const auto t0 = Clock::now();
    bool interrupted = false;
    while (t < p.t_end) {
        const StepResult sr = scheme->step(U_local, res, min_delta, t, p.t_end);
        if (p.use_ibm) solid_reset(U_local);
        t += sr.dt;
        ++step;
        if (monitoring && step % p.monitor_interval == 0) row(step);

        if (step % p.save_interval == 0) {
            Field g = decomp.gather(U_local);
            if (root) {
                exporter.export_vtr(file_prefix, fluid, g, mesh, t, step);
                U = std::move(g);
                save_checkpoint(step);
                std::printf("Step %05d | t = %.5f s | dt = %.3e s | S_max = %.2f m/s | wall = %s | Checkpoint saved\n",
                            step, t, sr.dt, sr.s_max, format_elapsed(seconds_since(t0)).c_str());
                std::fflush(stdout);
            }
        }
        // Agree on Ctrl+C across ranks so nobody blocks in a collective
        if (decomp.allreduce_max(g_stop_requested ? 1.0 : 0.0) > 0.0) {
            interrupted = true;
            break;
        }
        if (p.max_steps > 0 && step - start >= p.max_steps) break;
    }
    const double loop_wall = seconds_since(t0);

    if (monitoring && step % p.monitor_interval != 0) row(step);
    if (mon) std::fclose(mon);
    Field g = decomp.gather(U_local);
    if (root) {
        if (interrupted) std::printf("\n[INTERRUPTED] Stopping...\n");
        if (interrupted || step % p.save_interval != 0) exporter.export_vtr(file_prefix, fluid, g, mesh, t, step);
        U = std::move(g);
        save_checkpoint(step);
        exporter.write_pvd(file_prefix);
        dump_final_state(step);
        std::printf("=== DONE  - open '%s.pvd' in ParaView ===  (time loop %.3f s, %d steps)\n", file_prefix.c_str(),
                    loop_wall, step - start);
    }
}
#endif   // FV_USE_MPI

}   // namespace fv
