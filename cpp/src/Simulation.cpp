#include "fv/Simulation.hpp"

#include <chrono>
#include <cstdio>
#include <map>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

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

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

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
void strip_tangential_derivs(const EOS& eos, const BoundaryManager& bm, const Mesh& m, char dir, const Field& strip,
                             std::vector<double>& a, std::vector<double>& b) {
    const bool row = (dir == 'y');
    const int n = row ? m.nx : m.ny;
    Mesh m1 = row ? generate_cartesian_mesh(m.Lx, m.dy, m.nx, 1) : generate_cartesian_mesh(m.dx, m.Ly, 1, m.ny);
    m1.dx = m.dx;
    m1.dy = m.dy;
    BoundaryManager b1(m1, eos);
    if (row) b1.set_boundary_types(bm.type(Face::Left), bm.type(Face::Right), BCType::Transmissive, BCType::Transmissive);
    else     b1.set_boundary_types(BCType::Transmissive, BCType::Transmissive, bm.type(Face::Bottom), bm.type(Face::Top));
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
    b.length[0] = p.Lx; b.length[1] = p.Ly; b.length[2] = p.Lz;
    return b;
}

// Cells of the halo slice adjacent to the interface (lo: last slice, hi: first).
Field adjacent_strip(const Field& halo, char dir, bool lo, int nx, int ny) {
    constexpr int D = kHaloDepth;
    Field s;
    if (dir == 'y') for (int i = 0; i < nx; ++i) s.push_back(halo[(lo ? D - 1 : 0) * nx + i]);
    else            for (int j = 0; j < ny; ++j) s.push_back(halo[j * D + (lo ? D - 1 : 0)]);
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
    m1.dx = m.dx; m1.dy = m.dy; m1.dz = m.dz;
    BoundaryManager b1(m1, eos);
    if (ydec) b1.set_boundary_types(bm.type(Face::Left), bm.type(Face::Right), BCType::Transmissive, BCType::Transmissive);
    else      b1.set_boundary_types(BCType::Transmissive, BCType::Transmissive, bm.type(Face::Bottom), bm.type(Face::Top));
    b1.set_type(Face::Front, bm.type(Face::Front));
    b1.set_type(Face::Back, bm.type(Face::Back));
    b1.set_params(bm.params());
    Field P;
    b1.get_padded_U(slab, P);
    const int PX = m1.nx + 2, PY = m1.ny + 2;
    auto at = [&](int ii, int jj, int kk) -> const State& { return P[(static_cast<size_t>(kk) * PY + jj) * PX + ii]; };
    const int ns = static_cast<int>(slab.size());
    a.resize(ns); b.resize(ns); c.resize(ns); d.resize(ns);
    for (int k = 0; k < m.nz; ++k) {
        const int nt = ydec ? m.nx : m.ny;
        for (int t = 0; t < nt; ++t) {
            const int ii = ydec ? t + 1 : 1, jj = ydec ? 1 : t + 1, kk = k + 1;
            const int s = k * nt + t;
            const State W = eos.cons_to_prim(slab[s]);
            auto deriv = [&](int comp, int axis) {   // d(vel comp)/dx_axis via the conservative chain rule
                const State& hi = axis == 0 ? at(ii + 1, jj, kk) : (axis == 1 ? at(ii, jj + 1, kk) : at(ii, jj, kk + 1));
                const State& lo = axis == 0 ? at(ii - 1, jj, kk) : (axis == 1 ? at(ii, jj - 1, kk) : at(ii, jj, kk - 1));
                const double h2 = 2.0 * (axis == 0 ? m.dx : (axis == 1 ? m.dy : m.dz));
                const double g_rho = (hi[0] - lo[0]) / h2, g_mom = (hi[kMom[comp]] - lo[kMom[comp]]) / h2;
                return (g_mom - W[kMom[comp]] * g_rho) / W[0];
            };
            if (ydec) { a[s] = deriv(0, 0); b[s] = deriv(1, 0); c[s] = deriv(2, 2); d[s] = deriv(1, 2); }
            else      { a[s] = deriv(0, 1); b[s] = deriv(1, 1); c[s] = deriv(2, 2); d[s] = deriv(0, 2); }
        }
    }
}
}  // namespace

// State of the leaf cell containing (x, y), or nullptr outside every leaf.
const State* sample_leaf_state(const AMRTree& tree, double x, double y) {
    const AMRPatch* leaf = tree.leaf_at(x, y);
    if (!leaf) return nullptr;
    const int i = std::min(std::max(static_cast<int>((x - leaf->xmin) / leaf->mesh.dx), 0), leaf->nx() - 1);
    const int j = std::min(std::max(static_cast<int>((y - leaf->ymin) / leaf->mesh.dy), 0), leaf->ny() - 1);
    return &leaf->ghost_source()[j * leaf->nx() + i];
}

std::string format_elapsed(double s) {
    char buf[32];
    if (s < 60) std::snprintf(buf, sizeof(buf), "%.1fs", s);
    else        std::snprintf(buf, sizeof(buf), "%d:%02d", static_cast<int>(s) / 60, static_cast<int>(s) % 60);
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
}  // namespace

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
    if (mesh.is3d()) dt_length = 2.0 / (1.0 / (mesh.Lx / mesh.nx) + 1.0 / (mesh.Ly / mesh.ny) + 1.0 / (mesh.Lz / mesh.nz));
#ifdef _OPENMP
    if (verbose) std::printf("[OPENMP] Native kernels on %d thread(s)\n", omp_get_max_threads());
#endif

    if (lower(p.eos_type) == "ideal gas") fluid = EOS::ideal(p.gamma);
    else                                  fluid = EOS::stiffened(p.gamma, p.p_inf);

    if (p.use_ibm) {
        const std::string digits = naca_digits(p.ibm_shape);
        if (!digits.empty() || lower(p.ibm_shape) == "airfoil") {
            const Polygon2D unit = digits.empty() ? load_airfoil(airfoil_file_path()) : naca4(digits);
            const Polygon2D placed = place_airfoil(unit, p.ibm_chord, p.ibm_aoa, p.ibm_cx, p.ibm_cy);
            ibm_bbox = {1e300, -1e300, 1e300, -1e300};
            for (const auto& q : placed.pts) {
                ibm_bbox[0] = std::min(ibm_bbox[0], q[0]);  ibm_bbox[1] = std::max(ibm_bbox[1], q[0]);
                ibm_bbox[2] = std::min(ibm_bbox[2], q[1]);  ibm_bbox[3] = std::max(ibm_bbox[3], q[1]);
            }
            ibm_factory = [placed, rho = p.rho_in, pout = p.p_out](const Mesh& m) {
                return ImmersedBoundary(m, placed, rho, pout);
            };
            ibm = std::make_unique<ImmersedBoundary>(ibm_factory(mesh));
            if (verbose)
                std::printf("[IBM] %s, chord %g, %g deg, quarter chord at (%g, %g) -> %d solid cells\n",
                            digits.empty() ? airfoil_file_path().c_str() : p.ibm_shape.c_str(), p.ibm_chord,
                            p.ibm_aoa, p.ibm_cx, p.ibm_cy, ibm->solid_count());
        } else {
            ibm_bbox = {p.ibm_cx - p.ibm_radius, p.ibm_cx + p.ibm_radius, p.ibm_cy - p.ibm_radius, p.ibm_cy + p.ibm_radius};
            ibm_factory = [shape = p.ibm_shape, r = p.ibm_radius, cx = p.ibm_cx, cy = p.ibm_cy, rho = p.rho_in,
                           pout = p.p_out, cz = p.ibm_cz](const Mesh& m) {
                return ImmersedBoundary(m, shape, r, cx, cy, rho, pout, cz);
            };
            ibm = std::make_unique<ImmersedBoundary>(ibm_factory(mesh));
            if (verbose)
                std::printf("[IBM] %s at (%g, %g) r=%g -> %d solid cells\n", p.ibm_shape.c_str(), p.ibm_cx, p.ibm_cy,
                            p.ibm_radius, ibm->solid_count());
        }
        if (ibm->solid_count() == 0) std::printf("[WARNING] The IBM body covers no cell: check its size and position\n");
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
        if (verbose && mesh.is3d()) std::printf("[PHYSICS] Body Forces enabled: gx=%g, gy=%g, gz=%g\n", p.gx, p.gy, p.gz);
        else if (verbose)           std::printf("[PHYSICS] Body Forces enabled: gx=%g, gy=%g\n", p.gx, p.gy);
    } else {
        body_force.reset();
    }

    // AMR (serial only: AMR and MPI are mutually exclusive)
    amr_tree.reset();
    amr_indicator.reset();
    if (p.use_amr) {
        amr_tree = std::make_unique<AMRTree>(p.Lx, p.Ly, p.amr_patch_nx, p.amr_patch_ny, p.amr_max_level,
                                             p.nx / p.amr_patch_nx, p.ny / p.amr_patch_ny);
        amr_indicator = std::make_unique<AMRErrorIndicator>(p.amr_refine_threshold, p.amr_derefine_threshold,
                                                            p.amr_buffer_layers);
        amr_indicator->use_density = p.amr_indicator != "velocity";
        amr_indicator->use_speed   = p.amr_indicator != "density";
        if (p.use_ibm && p.amr_refine_ibm)
            amr_indicator->force_refine = [this](const AMRPatch& patch) { return patch_touches_body(patch); };
        if (verbose)
            std::printf("[AMR] Enabled - max_level=%d, patch_size=%dx%d, subcycling active\n", p.amr_max_level,
                        p.amr_patch_nx, p.amr_patch_ny);
    }

    if (verbose)
        std::printf("Mesh: %d cells | Riemann: %s | Time: %s | CFL: %g\n", mesh.num_cells(), p.riemann_solver.c_str(),
                    p.time_scheme.c_str(), p.cfl);
    std::fflush(stdout);
}

void Simulation::initialize_field() {
    if (test_case->requires_viscosity && !ns)
        std::printf("[WARNING] This test case requires Navier-Stokes - viscous physics is off!\n");

    U = p.use_hot_start ? load_hot_start(p.hot_start_file) : initial_field();
    start_step = 0;
    if (p.resume) start_step = load_checkpoint();

    // A fresh start (also resume without a checkpoint) is forced; a restored state already is
    if (p.use_ibm && start_step == 0) ibm->apply_forcing(U, fluid);

    if (amr_tree) {
        for (int k = 0; k < p.amr_init_levels; ++k)
            for (AMRPatch* patch : amr_tree->leaves())
                if (patch->level < amr_tree->max_level) amr_tree->refine_patch(patch);
        if (p.use_ibm && p.amr_refine_ibm) refine_body_patches();
        if (p.use_hot_start) {
            // Seed every leaf by sampling the hot-start field at the leaf's global cell centroids
            // (bilinear, extrapolating at the edges).
            const double dx_s = mesh.Lx / mesh.nx, dy_s = mesh.Ly / mesh.ny;
            const GridInterp2D interp(linspace(dx_s / 2.0, mesh.Lx - dx_s / 2.0, mesh.nx),
                                      linspace(dy_s / 2.0, mesh.Ly - dy_s / 2.0, mesh.ny), U);
            for (AMRPatch* patch : amr_tree->leaves())
                for (int j = 0; j < patch->ny(); ++j)
                    for (int i = 0; i < patch->nx(); ++i)
                        patch->U[j * patch->nx() + i] = interp(patch->mesh.xc(i), patch->mesh.yc(j));
        } else {
            // Evaluate the initial condition at each leaf's own resolution, so refined leaves see the
            // exact initial discontinuities rather than their interpolation from the base grid
            for (AMRPatch* patch : amr_tree->leaves()) {
                Mesh pm = patch->mesh;   // cell positions of the patch, but Lx / Ly of the domain (expressions like Lx/2)
                pm.Lx = mesh.Lx;  pm.Ly = mesh.Ly;
                patch->U = test_case->initialize(pm, fluid, p.source);
            }
        }
        if (p.use_ibm && start_step == 0)
            for (AMRPatch* patch : amr_tree->leaves()) {
                ensure_patch_masks(*patch);
                ImmersedBoundary(patch->nx(), patch->ny(), patch->solid, p.rho_in, p.p_out).apply_forcing(patch->U, fluid);
            }
    }
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
            if (halos->lo) for (int i = 0; i < nx; ++i) UL_[m.boundary_face(Face::Bottom, i)] = (*halos->lo)[(D - 1) * nx + i];
            if (halos->hi) for (int i = 0; i < nx; ++i) UR_[m.boundary_face(Face::Top, i)] = (*halos->hi)[i];
            mh.b = halos->lo; mh.t = halos->hi;
        } else {
            if (halos->lo) for (int j = 0; j < ny; ++j) UL_[m.boundary_face(Face::Left, j)] = (*halos->lo)[j * D + D - 1];
            if (halos->hi) for (int j = 0; j < ny; ++j) UR_[m.boundary_face(Face::Right, j)] = (*halos->hi)[j * D];
            mh.l = halos->lo; mh.r = halos->hi;
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
        mh.l = &perL_; mh.r = &perR_;
    }
    if (mu && per_y && !mh.b && !mh.t && ny >= D) {
        perB_.assign(Uc->end() - static_cast<std::ptrdiff_t>(D) * nx, Uc->end());
        perT_.assign(Uc->begin(), Uc->begin() + static_cast<std::ptrdiff_t>(D) * nx);
        mh.b = &perB_; mh.t = &perT_;
    }
    if (mu) mu->reconstruct(UL_, UR_, *Uc, nx, ny, mh);

    double S = riemann.compute_flux(UL_, UR_, m, F_);

    R.resize(Uc->size());
    const double area = m.cell_area(), dxl = m.dx, dyl = m.dy;
#pragma omp parallel for schedule(static) if(nx * ny >= kOmpMinWork)
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
                if (halos->lo) for (int i = 0; i < nx; ++i) Upad_[i + 1] = (*halos->lo)[(D - 1) * nx + i];
                if (halos->hi) for (int i = 0; i < nx; ++i) Upad_[(ny + 1) * PW + i + 1] = (*halos->hi)[i];
            } else {
                if (halos->lo) for (int j = 0; j < ny; ++j) Upad_[(j + 1) * PW] = (*halos->lo)[j * D + D - 1];
                if (halos->hi) for (int j = 0; j < ny; ++j) Upad_[(j + 1) * PW + nx + 1] = (*halos->hi)[j * D];
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
                strip_tangential_derivs(fluid, bm, m, halos->dir, adjacent_strip(*halos->lo, halos->dir, true, nx, ny), lo_a, lo_b);
                cg.lo_a = lo_a.data(); cg.lo_b = lo_b.data();
            }
            if (halos->hi) {
                strip_tangential_derivs(fluid, bm, m, halos->dir, adjacent_strip(*halos->hi, halos->dir, false, nx, ny), hi_a, hi_b);
                cg.hi_a = hi_a.data(); cg.hi_b = hi_b.data();
            }
        }
        const double dt_visc = ns->viscous_residual(*Uc, gX_, gY_, m, Upad_, Rv_, &cg);
        const int n = static_cast<int>(R.size());
#pragma omp parallel for schedule(static) if(n >= kOmpMinWork)
        for (int c = 0; c < n; ++c)
            for (int q = 0; q < kNVar; ++q) R[c][q] += Rv_[c][q];
        const double min_delta = std::min(m.dx, m.dy);
        S = std::max(S, p.cfl * min_delta / (dt_visc + 1e-12));
    }

    if (body_force) body_force->add_to(*Uc, R);
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
                    if (halos->lo) UL_[m.boundary_face(Face::Bottom, k * nx + i)] = (*halos->lo)[((D - 1) * nz + k) * nx + i];
                    if (halos->hi) UR_[m.boundary_face(Face::Top, k * nx + i)] = (*halos->hi)[k * nx + i];
                }
            mh.b = halos->lo; mh.t = halos->hi;
        } else {
            for (int s = 0; s < ny * nz; ++s) {
                if (halos->lo) UL_[m.boundary_face(Face::Left, s)] = (*halos->lo)[s * D + D - 1];
                if (halos->hi) UR_[m.boundary_face(Face::Right, s)] = (*halos->hi)[s * D];
            }
            mh.l = halos->lo; mh.r = halos->hi;
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
            mh.l = &perL_; mh.r = &perR_;
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
            mh.b = &perB_; mh.t = &perT_;
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
            mh.f = &perF_; mh.bk = &perK_;
        }
        mu->reconstruct_3d(UL_, UR_, Uv, m, mh);
    }

    double S = riemann.compute_flux(UL_, UR_, m, F_);

    R.resize(Uv.size());
    const double Ax = m.face_area(0), Ay = m.face_area(1), Az = m.face_area(2), vol = m.cell_volume();
#pragma omp parallel for schedule(static) if(nx * ny * nz >= kOmpMinWork)
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
                r[q] = (((0.0 - (fr[q] * Ax - fl[q] * Ax)) - (ft[q] * Ay - fb[q] * Ay)) - (fk[q] * Az - ff[q] * Az)) / vol;
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
            auto P = [&](int ii, int jj, int kk) -> State& { return Upad_[(static_cast<size_t>(kk) * PY + jj) * PX + ii]; };
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
        const double dt_visc = ns->viscous_residual_3d(Uv, gX_, gY_, gZ_, m, Upad_, Rv_, &cg);
        const int n = static_cast<int>(R.size());
#pragma omp parallel for schedule(static) if(n >= kOmpMinWork)
        for (int cc = 0; cc < n; ++cc)
            for (int q = 0; q < kNVar; ++q) R[cc][q] += Rv_[cc][q];
        S = std::max(S, p.cfl * dt_length / (dt_visc + 1e-12));
    }

    if (body_force) body_force->add_to(Uv, R);
    return S;
}
double Simulation::residual(const Field& Uin, Field& R) {
    return assemble(mesh, boundaries, p.use_ibm ? ibm.get() : nullptr, muscl.get(), Uin, R, nullptr);
}

bool Simulation::patch_touches_body(const AMRPatch& patch) const {
    const double mx = patch.mesh.dx, my = patch.mesh.dy;
    return patch.xmax + mx >= ibm_bbox[0] && patch.xmin - mx <= ibm_bbox[1] &&
           patch.ymax + my >= ibm_bbox[2] && patch.ymin - my <= ibm_bbox[3];
}

void Simulation::refine_body_patches() {
    for (int pass = 0; pass < amr_tree->max_level; ++pass)
        for (AMRPatch* patch : amr_tree->leaves())
            if (patch->level < amr_tree->max_level && patch_touches_body(*patch)) amr_tree->refine_patch(patch);
    amr_tree->enforce_2to1_balance();
}

void Simulation::ensure_patch_masks(AMRPatch& patch) const {
    if (!p.use_ibm || !patch.solid.empty()) return;
    const Mesh& m = patch.mesh;
    patch.solid = ibm_factory(m).mask();
    // Mask of the two-cell ghost ring from the same geometry; 0 outside the domain
    Mesh me = m;
    me.nx += 4;  me.ny += 4;
    me.x0 -= 2.0 * m.dx;  me.y0 -= 2.0 * m.dy;
    me.Lx = me.nx * m.dx;  me.Ly = me.ny * m.dy;
    patch.solid_ext = ibm_factory(me).mask();
    const int PW = m.nx + 4;
    for (int jj = 0; jj < m.ny + 4; ++jj)
        for (int ii = 0; ii < PW; ++ii) {
            const bool interior = ii >= 2 && ii < m.nx + 2 && jj >= 2 && jj < m.ny + 2;
            const double x = m.x0 + (ii - 1.5) * m.dx, y = m.y0 + (jj - 1.5) * m.dy;
            const bool inside = x > 0.0 && x < mesh.Lx && y > 0.0 && y < mesh.Ly;
            char& c = patch.solid_ext[jj * PW + ii];
            if (interior) c = patch.solid[(jj - 2) * m.nx + ii - 2];
            else if (!inside) c = 0;
        }
}

double Simulation::patch_residual(AMRPatch& patch, AMRTree& tree, const Field& Uc_in, Field& R,
                                  Field& flux_x, Field& flux_y) {
    // Physical BCs only where the patch touches the domain boundary; internal
    // faces start transmissive and are overwritten from the neighbouring leaf.
    constexpr double eps = 1e-10;
    const bool edge_l = patch.xmin <= eps, edge_r = patch.xmax >= mesh.Lx - eps;
    const bool edge_b = patch.ymin <= eps, edge_t = patch.ymax >= mesh.Ly - eps;
    BoundaryManager pb(patch.mesh, fluid);
    pb.set_boundary_types(edge_l ? boundaries.type(Face::Left) : BCType::Transmissive,
                          edge_r ? boundaries.type(Face::Right) : BCType::Transmissive,
                          edge_b ? boundaries.type(Face::Bottom) : BCType::Transmissive,
                          edge_t ? boundaries.type(Face::Top) : BCType::Transmissive);
    pb.set_params(boundaries.params());
    // Per-thread scratch: patches of one AMR level are advanced concurrently.
    thread_local Field pUL, pUR, pF, pUf;

    const Mesh& m = patch.mesh;
    const int nx = m.nx, ny = m.ny;

    // IBM: velocity of the solid cells is zeroed in every residual, as in assemble
    const Field* Up = &Uc_in;
    if (p.use_ibm) {
        ensure_patch_masks(patch);
        pUf = Uc_in;
        ImmersedBoundary(nx, ny, patch.solid, p.rho_in, p.p_out).apply_forcing(pUf, fluid);
        Up = &pUf;
    }
    const Field& Uc = *Up;

    const double dx = m.dx, dy = m.dy;
    const bool if_l = patch.xmin > eps, if_r = patch.xmax < mesh.Lx - eps;
    const bool if_b = patch.ymin > eps, if_t = patch.ymax < mesh.Ly - eps;

    std::map<const AMRPatch*, GridInterp2D> interps;
    auto leaf_at = [&](double x, double y) -> const AMRPatch* { return tree.leaf_at(x, y); };
    auto cell_at = [&](const AMRPatch* lf, double x, double y) -> const State& {
        const int i = std::min(std::max(static_cast<int>((x - lf->xmin) / lf->mesh.dx), 0), lf->nx() - 1);
        const int j = std::min(std::max(static_cast<int>((y - lf->ymin) / lf->mesh.dy), 0), lf->ny() - 1);
        return lf->ghost_source()[j * lf->nx() + i];
    };
    auto ghost_state = [&](double x, double y) -> const State* {
        static thread_local State out;
        const AMRPatch* lf = leaf_at(x, y);
        if (!lf) return nullptr;
        if (lf->level == patch.level) { out = cell_at(lf, x, y); return &out; }
        if (lf->level > patch.level) {   // finer: mean of the 2 x 2 fine cells under the ghost cell
            out = State{0, 0, 0, 0, 0};
            for (double sx : {-0.25, 0.25})
                for (double sy : {-0.25, 0.25}) {
                    const double fx = x + sx * dx, fy = y + sy * dy;
                    const AMRPatch* fl = leaf_at(fx, fy);
                    const State& c = cell_at(fl ? fl : lf, fx, fy);
                    for (int q = 0; q < kNVar; ++q) out[q] += 0.25 * c[q];
                }
            return &out;
        }
        auto it = interps.find(lf);   // coarser: linear interpolation
        if (it == interps.end()) {
            std::vector<double> xs(lf->nx()), ys(lf->ny());
            for (int i = 0; i < lf->nx(); ++i) xs[i] = lf->mesh.xc(i);
            for (int j = 0; j < lf->ny(); ++j) ys[j] = lf->mesh.yc(j);
            it = interps.emplace(lf, GridInterp2D(xs, ys, lf->ghost_source())).first;
        }
        out = it->second(x, y);
        return &out;
    };
    // Ghost-cell state of cell (ix, iy) outside this patch (solid cells at rest); false outside the domain
    auto ghost_cell = [&](int ix, int iy, State& c) {
        const double x = m.x0 + (ix + 0.5) * dx, y = m.y0 + (iy + 0.5) * dy;
        if (!(x > 0.0 && x < mesh.Lx && y > 0.0 && y < mesh.Ly)) return false;
        const State* g = ghost_state(x, y);
        if (!g) return false;
        c = *g;
        if (p.use_ibm && patch.ext_solid(ix, iy)) {
            State W = fluid.cons_to_prim(c);
            W[1] = 0.0; W[2] = 0.0; W[4] = 0.0;
            c = fluid.prim_to_cons(W);
        }
        return true;
    };

    // The non-reflecting outlet differentiates along its boundary: where that boundary continues into
    // another patch, hand it the neighbouring cells as the MPI strip halos do (one pass per direction)
    const auto nrbc = [&](Face f) { return pb.type(f) == BCType::NonReflecting; };
    const bool halo_y = (nrbc(Face::Left) || nrbc(Face::Right)) && (if_b || if_t);
    const bool halo_x = (nrbc(Face::Bottom) || nrbc(Face::Top)) && (if_l || if_r);
    if (!halo_y && !halo_x) {
        pb.apply_boundary_conditions(Uc, pUL, pUR);
    } else {
        constexpr int D = kHaloDepth;
        Field ylo(static_cast<size_t>(D) * nx), yhi(ylo.size()), xlo(static_cast<size_t>(D) * ny), xhi(xlo.size());
        State c;
        for (int i = 0; i < nx; ++i)
            for (int d = 0; d < D; ++d) {
                ylo[d * nx + i] = ghost_cell(i, -1, c) ? c : Uc[i];
                yhi[d * nx + i] = ghost_cell(i, ny, c) ? c : Uc[(ny - 1) * nx + i];
            }
        for (int j = 0; j < ny; ++j)
            for (int d = 0; d < D; ++d) {
                xlo[j * D + d] = ghost_cell(-1, j, c) ? c : Uc[j * nx];
                xhi[j * D + d] = ghost_cell(nx, j, c) ? c : Uc[j * nx + nx - 1];
            }
        const TangentialHalos thy{'y', D, if_b ? &ylo : nullptr, if_t ? &yhi : nullptr};
        const TangentialHalos thx{'x', D, if_l ? &xlo : nullptr, if_r ? &xhi : nullptr};
        pb.apply_boundary_conditions(Uc, pUL, pUR, halo_y ? &thy : &thx);
        if (halo_y && halo_x) {   // bottom / top faces from the x pass
            thread_local Field qL, qR;
            pb.apply_boundary_conditions(Uc, qL, qR, &thx);
            for (Face f : {Face::Bottom, Face::Top})
                for (int k = 0; k < m.boundary_count(f); ++k) {
                    pUL[m.boundary_face(f, k)] = qL[m.boundary_face(f, k)];
                    pUR[m.boundary_face(f, k)] = qR[m.boundary_face(f, k)];
                }
        }
    }

    // Interface faces: the Riemann ghost and the two MUSCL halo layers come from the same canonical
    // ghost cells (same level: the cell; finer: 2 x 2 mean; coarser: interpolation), so a seam between
    // equal-level patches is reconstructed as in the uniform solver
    constexpr int HD = kHaloDepth;
    thread_local Field hl, hr, hb, ht;
    Halos mh;
    auto nearest = [&](int ix, int iy, const State& own) { State c; return ghost_cell(ix, iy, c) ? c : own; };
    if (if_l) {
        hl.resize(static_cast<size_t>(ny) * HD);
        for (int j = 0; j < ny; ++j) for (int d = 0; d < HD; ++d) hl[j * HD + d] = nearest(d - HD, j, Uc[j * nx]);
        for (int j = 0; j < ny; ++j) pUL[m.boundary_face(Face::Left, j)] = hl[j * HD + HD - 1];
        mh.l = &hl;
    }
    if (if_r) {
        hr.resize(static_cast<size_t>(ny) * HD);
        for (int j = 0; j < ny; ++j) for (int d = 0; d < HD; ++d) hr[j * HD + d] = nearest(nx + d, j, Uc[j * nx + nx - 1]);
        for (int j = 0; j < ny; ++j) pUR[m.boundary_face(Face::Right, j)] = hr[j * HD];
        mh.r = &hr;
    }
    if (if_b) {
        hb.resize(static_cast<size_t>(HD) * nx);
        for (int d = 0; d < HD; ++d) for (int i = 0; i < nx; ++i) hb[d * nx + i] = nearest(i, d - HD, Uc[i]);
        for (int i = 0; i < nx; ++i) pUL[m.boundary_face(Face::Bottom, i)] = hb[(HD - 1) * nx + i];
        mh.b = &hb;
    }
    if (if_t) {
        ht.resize(static_cast<size_t>(HD) * nx);
        for (int d = 0; d < HD; ++d) for (int i = 0; i < nx; ++i) ht[d * nx + i] = nearest(i, ny + d, Uc[(ny - 1) * nx + i]);
        for (int i = 0; i < nx; ++i) pUR[m.boundary_face(Face::Top, i)] = ht[i];
        mh.t = &ht;
    }
    if (muscl) muscl->reconstruct(pUL, pUR, Uc, nx, ny, mh);
    double S = riemann.compute_flux(pUL, pUR, m, pF);

    const int nxf = m.num_xfaces(), nf = m.num_faces();
    flux_x.resize(nxf);
    flux_y.resize(nf - nxf);
    for (int f = 0; f < nxf; ++f)
        for (int q = 0; q < kNVar; ++q) flux_x[f][q] = pF[f][q] * dy;
    for (int f = nxf; f < nf; ++f)
        for (int q = 0; q < kNVar; ++q) flux_y[f - nxf][q] = pF[f][q] * dx;

    if (ns) {
        // Padded state: physical-BC ghosts from the BoundaryManager; across interfaces (corners
        // included) a ghost state that depends only on the ghost cell and the level, so both patches
        // sharing a face see the same cross-derivatives and compute the same viscous flux
        thread_local Field Pad, gX, gY, Rv;
        thread_local ViscousFaceFluxes vff;
        const NavierStokes nsl(fluid, ns->mu(), p.pr);   // own scratch: the shared ns is not thread-safe
        pb.get_padded_U(Uc, Pad);
        const int PW = nx + 2, PH = ny + 2;
        for (int jj = 0; jj < PH; ++jj)
            for (int ii = 0; ii < PW; ++ii) {
                State c;
                if (!(ii >= 1 && ii <= nx && jj >= 1 && jj <= ny) && ghost_cell(ii - 1, jj - 1, c)) Pad[jj * PW + ii] = c;
            }
        // Ghost cells at the corner of an interface and a physical boundary: the boundary condition
        // applied to the ghost cell next to it, as the neighbouring patch does for its own cell
        auto bc_pad = [&](const State& cell, bool along_y, bool high) {
            const Mesh m1 = generate_cartesian_mesh(dx, dy, 1, 1);
            BoundaryManager b1(m1, fluid);
            b1.set_boundary_types(along_y ? BCType::Transmissive : pb.type(Face::Left),
                                  along_y ? BCType::Transmissive : pb.type(Face::Right),
                                  along_y ? pb.type(Face::Bottom) : BCType::Transmissive,
                                  along_y ? pb.type(Face::Top) : BCType::Transmissive);
            b1.set_params(boundaries.params());
            Field one(1, cell), P1;
            b1.get_padded_U(one, P1);
            return P1[along_y ? (high ? 7 : 1) : (high ? 5 : 3)];
        };
        for (int jj : {0, ny + 1}) {
            if (!(jj == 0 ? edge_b : edge_t)) continue;
            for (int ii : {0, nx + 1})
                if (ii == 0 ? if_l : if_r) Pad[jj * PW + ii] = bc_pad(Pad[(jj == 0 ? 1 : ny) * PW + ii], true, jj != 0);
        }
        for (int ii : {0, nx + 1}) {
            if (!(ii == 0 ? edge_l : edge_r)) continue;
            for (int jj : {0, ny + 1})
                if (jj == 0 ? if_b : if_t) Pad[jj * PW + ii] = bc_pad(Pad[jj * PW + (ii == 0 ? 1 : nx)], false, ii != 0);
        }
        compute_gradients(Pad, nx, ny, m.dx, m.dy, gX, gY);

        // du/dy, dv/dy (column) or du/dx, dv/dx (row) of the ghost cells across interfaces
        thread_local std::vector<double> ga[4], gb[4];
        CrossGhosts cg;
        auto ghost_derivs = [&](bool column, int side, std::vector<double>& a, std::vector<double>& b) {
            const int n = column ? ny : nx;
            const double h2 = 2.0 * (column ? m.dy : m.dx);
            a.resize(n);  b.resize(n);
            for (int k = 0; k < n; ++k) {
                const int c = column ? (k + 1) * PW + (side == 0 ? 0 : nx + 1) : (side == 0 ? 0 : ny + 1) * PW + k + 1;
                const int step = column ? PW : 1;
                const State &hi = Pad[c + step], &lo = Pad[c - step];
                State g;
                for (int q = 0; q < kNVar; ++q) g[q] = (hi[q] - lo[q]) / h2;
                const State W = fluid.cons_to_prim(Pad[c]);
                a[k] = (g[1] - W[1] * g[0]) / W[0];
                b[k] = (g[2] - W[2] * g[0]) / W[0];
            }
        };
        if (if_l) { ghost_derivs(true, 0, ga[0], gb[0]);  cg.x_lo_a = ga[0].data(); cg.x_lo_b = gb[0].data(); }
        if (if_r) { ghost_derivs(true, 1, ga[1], gb[1]);  cg.x_hi_a = ga[1].data(); cg.x_hi_b = gb[1].data(); }
        if (if_b) { ghost_derivs(false, 0, ga[2], gb[2]); cg.y_lo_a = ga[2].data(); cg.y_lo_b = gb[2].data(); }
        if (if_t) { ghost_derivs(false, 1, ga[3], gb[3]); cg.y_hi_a = ga[3].data(); cg.y_hi_b = gb[3].data(); }

        const double dt_visc = nsl.viscous_residual(Uc, gX, gY, m, Pad, Rv, &cg, &vff);
        // Total flux = convective - viscous, so that R = -div(flux) and the reflux sees both
        for (size_t f = 0; f < flux_x.size(); ++f)
            for (int q = 0; q < kNVar; ++q) flux_x[f][q] -= vff.fx[f][q];
        for (size_t f = 0; f < flux_y.size(); ++f)
            for (int q = 0; q < kNVar; ++q) flux_y[f][q] -= vff.fy[f][q];
        S = std::max(S, p.cfl * std::min(m.dx, m.dy) / (dt_visc + 1e-12));
    }

    R.resize(Uc.size());
    const double area = m.cell_area();
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const State& fl = flux_x[j * (nx + 1) + i];
            const State& fr = flux_x[j * (nx + 1) + i + 1];
            const State& fb = flux_y[j * nx + i];
            const State& ft = flux_y[(j + 1) * nx + i];
            State& r = R[j * nx + i];
            for (int q = 0; q < kNVar; ++q) r[q] = ((0.0 - (fr[q] - fl[q])) - (ft[q] - fb[q])) / area;
        }
    if (body_force) body_force->add_to(Uc, R);
    return S;
}

void Simulation::amr_solid_reset(AMRTree& tree, const AMRParallel* par) {
    const auto all = tree.leaves();
    const bool dist = par && par->size > 1;
    std::vector<AMRPatch*> leaves;   // the leaves reset by this rank
    for (AMRPatch* l : all)
        if (!dist || l->owner == par->rank) leaves.push_back(l);
    // Snapshot first: every leaf reads its neighbours' pre-reset cells, whatever the order
    for (AMRPatch* l : all) {
        if (dist && !l->held) {   // not held: any read turns the result into NaN
            l->U_ghost.assign(static_cast<size_t>(l->nx()) * l->ny(), State{NAN, NAN, NAN, NAN, NAN});
        } else {
            if (!dist || l->owner == par->rank) ensure_patch_masks(*l);
            l->U_ghost = l->U;
        }
        l->has_ghost = true;
    }
    const int nl = static_cast<int>(leaves.size());
#pragma omp parallel for schedule(dynamic) if(nl > 1)
    for (int k = 0; k < nl; ++k) {
        AMRPatch& patch = *leaves[k];
        const Mesh& m = patch.mesh;
        const int nx = m.nx, ny = m.ny, PW = nx + 2, PH = ny + 2;
        Field E(static_cast<size_t>(PW) * PH);
        std::vector<char> mask(E.size(), 1);   // outside the domain: "solid", nothing to pull from
        for (int jj = 0; jj < PH; ++jj)
            for (int ii = 0; ii < PW; ++ii) {
                const int e = jj * PW + ii;
                if (ii >= 1 && ii <= nx && jj >= 1 && jj <= ny) {
                    E[e] = patch.U[(jj - 1) * nx + ii - 1];
                    mask[e] = patch.solid[(jj - 1) * nx + ii - 1];
                    continue;
                }
                const double x = m.x0 + (ii - 0.5) * m.dx, y = m.y0 + (jj - 0.5) * m.dy;
                const State* s = (x > 0.0 && x < mesh.Lx && y > 0.0 && y < mesh.Ly) ? sample_leaf_state(tree, x, y) : nullptr;
                if (s) {
                    E[e] = *s;
                    mask[e] = patch.ext_solid(ii - 1, jj - 1);
                } else {
                    const int i = std::min(std::max(ii - 1, 0), nx - 1), j = std::min(std::max(jj - 1, 0), ny - 1);
                    E[e] = patch.U[j * nx + i];
                }
            }
        ImmersedBoundary(PW, PH, std::move(mask), p.rho_in, p.p_out).apply_solid_reset(E, fluid);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                if (patch.solid[j * nx + i]) patch.U[j * nx + i] = E[(j + 1) * PW + i + 1];
    }
    for (AMRPatch* l : all) l->has_ghost = false;
    if (dist) refresh_halos(tree, *par);
}

// ---------------------------------------------------------------------------
// Checkpoints / hot start
// ---------------------------------------------------------------------------

std::string Simulation::checkpoint_path() const {
    return (fs::path(exporter.output_dir()) / (file_prefix + "_restart.npz")).string();
}

void Simulation::save_checkpoint(int step) {
    const std::string path = checkpoint_path();
    const std::string tmp  = path.substr(0, path.size() - 4) + "_tmp.npz";
    std::vector<double>      times;
    std::vector<std::string> files;
    for (const auto& e : exporter.collection) { times.push_back(e.time); files.push_back(e.filename); }
    NpzArchive a;
    a["U"]    = field_to_npy(U, mesh.is3d() ? kNVar : 4);
    a["t"]    = NpyArray::from_double(t);
    a["step"] = NpyArray::from_int(step);
    a["nx"]   = NpyArray::from_int(mesh.nx);
    a["ny"]   = NpyArray::from_int(mesh.ny);
    a["Lx"]   = NpyArray::from_double(mesh.Lx);
    a["Ly"]   = NpyArray::from_double(mesh.Ly);
    a["nz"]   = NpyArray::from_int(mesh.nz);
    a["Lz"]   = NpyArray::from_double(mesh.Lz);
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
    U = npy_to_field(a.at("U"));
    t = a.at("t").scalar();
    const int step = static_cast<int>(a.at("step").scalar());
    exporter.collection.clear();
    if (a.count("collection_time") && a.count("collection_file")) {
        const auto times = a.at("collection_time").as_doubles();
        const auto files = a.at("collection_file").as_strings();
        for (size_t k = 0; k < std::min(times.size(), files.size()); ++k) exporter.collection.push_back({times[k], files[k]});
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
        nx_c = static_cast<int>(a.at("nx").scalar());  ny_c = static_cast<int>(a.at("ny").scalar());
        Lx_c = a.at("Lx").scalar();                    Ly_c = a.at("Ly").scalar();
    } else {
        nx_c = p.hot_start_src_nx;  ny_c = p.hot_start_src_ny;
        if (nx_c <= 0 || ny_c <= 0)
            throw std::runtime_error("Hot-start file '" + path +
                                     "' has no grid metadata and no manual Nx / Ny were provided "
                                     "(set hot_start_src_nx / hot_start_src_ny).");
        Lx_c = mesh.Lx;  Ly_c = mesh.Ly;
        std::printf("[Hot Start] No metadata in file - using manual dims: Nx=%d, Ny=%d, Lx=%g, Ly=%g\n", nx_c, ny_c,
                    Lx_c, Ly_c);
    }
    if (std::abs(Lx_c - mesh.Lx) > 1e-10 || std::abs(Ly_c - mesh.Ly) > 1e-10)
        std::printf("[WARNING] Hot-start domain (%g x %g) differs from target (%g x %g). "
                    "Proceeding with coordinate rescaling.\n", Lx_c, Ly_c, mesh.Lx, mesh.Ly);
    const int nz_c = a.count("nz") ? static_cast<int>(a.at("nz").scalar()) : 1;
    if (nz_c > 1 && !mesh.is3d())
        throw std::runtime_error("Hot start from a 3-D checkpoint needs a 3-D run (nz > 1)");
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
    a["t"]    = NpyArray::from_double(t);
    a["step"] = NpyArray::from_int(step);
    if (amr_tree) {
        Field all;
        std::vector<double> meta;
        for (const AMRPatch* lf : amr_tree->leaves()) {
            all.insert(all.end(), lf->U.begin(), lf->U.end());
            meta.insert(meta.end(), {lf->xmin, lf->xmax, lf->ymin, lf->ymax, static_cast<double>(lf->level)});
        }
        a["U"] = field_to_npy(all);
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
        const double A = !mesh.is3d() ? L
                       : (lower(p.ibm_shape) == "sphere" ? 3.14159265358979323846 * p.ibm_radius * p.ibm_radius
                                                         : L * mesh.Lz);
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
        if (body_force) { rx -= U[c][0] * body_force->gx; ry -= U[c][0] * body_force->gy; }   // not a load
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
    if (amr_tree) { solve_amr(); return; }

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
            if (scheme->inner_iter_last() >= 0) std::snprintf(inner, sizeof(inner), " | inner=%d", scheme->inner_iter_last());
            std::printf("Step %05d | t = %.5f s | dt = %.3e s | S_max = %.2f m/s%s | wall = %s | Checkpoint saved\n",
                        step, t, r.dt, r.s_max, inner, format_elapsed(seconds_since(t0)).c_str());
            std::fflush(stdout);
        }
        if (g_stop_requested) { interrupted = true; break; }
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
    AMRTree& tree = *amr_tree;
    int step = start_step;
    const double min_delta_base = std::min(tree.roots[0]->mesh.dx, tree.roots[0]->mesh.dy);

    // Under MPI every rank holds the whole tree and shares each level's patches (see AMRParallel)
    AMRParallel par_storage;
    const AMRParallel* par = nullptr;
#ifdef FV_USE_MPI
    {
        int init = 0, size = 1;
        MPI_Initialized(&init);
        if (init) MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size > 1) {
            par_storage.size = size;
            MPI_Comm_rank(MPI_COMM_WORLD, &par_storage.rank);
            par_storage.allgatherv = [](const double* send, int n, double* recv, const int* counts, const int* displs) {
                MPI_Allgatherv(send, n, MPI_DOUBLE, recv, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
            };
            par_storage.exchange = [](const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv) {
                std::vector<MPI_Request> req;
                req.reserve(send.size() + recv.size());
                for (size_t r = 0; r < recv.size(); ++r)
                    if (!recv[r].empty()) {
                        req.emplace_back();
                        MPI_Irecv(recv[r].data(), static_cast<int>(recv[r].size()), MPI_DOUBLE, static_cast<int>(r), 77, MPI_COMM_WORLD, &req.back());
                    }
                for (size_t r = 0; r < send.size(); ++r)
                    if (!send[r].empty()) {
                        req.emplace_back();
                        MPI_Isend(send[r].data(), static_cast<int>(send[r].size()), MPI_DOUBLE, static_cast<int>(r), 77, MPI_COMM_WORLD, &req.back());
                    }
                MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE);
            };
            par_storage.allreduce_max = [](double v) {
                double g = v;
                MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                return g;
            };
            par = &par_storage;
        }
    }
#endif
    const bool root = !par || par->rank == 0;
    if (par) assign_ownership(tree, par->rank, par->size);

    if (step == 0 && root) exporter.export_amr(file_prefix, fluid, tree, t, step);

    const PatchResidualFn pres = [this](AMRPatch& patch, AMRTree& tr, const Field& Uin, Field& R, Field& fx, Field& fy) {
        return patch_residual(patch, tr, Uin, R, fx, fy);
    };

    // Bootstrap S_max over all initial leaves; afterwards advance_amr_level
    // returns the max over every leaf it advanced (one-step-lag estimate).
    double S_max = 0.0;
    {
        Field R, fx, fy;
        for (AMRPatch* leaf : tree.leaves()) S_max = std::max(S_max, patch_residual(*leaf, tree, leaf->U, R, fx, fy));
    }

    // Monitor: area-weighted means over the leaves; IBM loads summed over the leaves in tree order
    std::FILE* mon = root ? open_monitor() : nullptr;
    auto row = [&](int st) {
        double sums[3] = {0.0, 0.0, 0.0}, area_tot = 0.0;
        std::array<double, 3> loads{};
        Field R, fx, fy;
        for (AMRPatch* leaf : tree.leaves()) {
            const double a = leaf->mesh.cell_area();
            for (const State& c : leaf->U) {
                sums[0] += c[0] * a;
                sums[1] += 0.5 * (c[1] * c[1] + c[2] * c[2] + c[4] * c[4]) / c[0] * a;
                sums[2] += c[3] * a;
                area_tot += a;
            }
            if (p.use_ibm) {
                patch_residual(*leaf, tree, leaf->U, R, fx, fy);
                const auto l = ibm_loads(leaf->mesh, ImmersedBoundary(leaf->nx(), leaf->ny(), leaf->solid, p.rho_in, p.p_out),
                                         leaf->U, R);
                for (int q = 0; q < 3; ++q) loads[q] += l[q];
            }
        }
        monitor_row(mon, st, {sums[0] / area_tot * mesh.cell_area(), sums[1] / area_tot * mesh.cell_area(),
                              sums[2] / area_tot * mesh.cell_area()}, p.use_ibm ? &loads : nullptr);
    };
    if (mon && step == 0) {
        if (p.use_ibm) for (AMRPatch* leaf : tree.leaves()) ensure_patch_masks(*leaf);
        row(step);
    }

    const auto t0 = Clock::now();
    bool interrupted = false;
    std::vector<AMRPatch*> to_refine, to_derefine;
    while (t < p.t_end) {
        double dt = scheme->cfl * min_delta_base / (S_max + 1e-10);
        dt = std::min(dt, p.t_end - t);
        S_max = advance_amr_level(tree, 0, dt, t, p.t_end, pres, *scheme, par);
        if (p.use_ibm) amr_solid_reset(tree, par);
        if (par) gather_leaf_state(tree, *par);   // the remesh and the output still read every leaf
        t += dt;
        ++step;

        if (step % p.amr_refine_every == 0) {
            amr_indicator->mark(tree, to_refine, to_derefine);
            for (AMRPatch* patch : to_refine)
                if (patch->is_leaf() && patch->level < tree.max_level) tree.refine_patch(patch);
            for (AMRPatch* parent : to_derefine) {
                bool leaf_kids = !parent->is_leaf();
                for (AMRPatch* c : parent->children) leaf_kids = leaf_kids && c->is_leaf();
                if (leaf_kids) tree.derefine_patch(parent);
            }
            tree.enforce_2to1_balance();
            if (par) assign_ownership(tree, par->rank, par->size);
        }

        if (mon && step % p.monitor_interval == 0) row(step);

        if (step % p.save_interval == 0 && root) {
            exporter.export_amr(file_prefix, fluid, tree, t, step);
            std::printf("Step %05d | t = %.5f s | dt = %.3e s | S_max = %.2f m/s | patches = %zu | wall = %s\n", step, t,
                        dt, S_max, tree.leaves().size(), format_elapsed(seconds_since(t0)).c_str());
            std::fflush(stdout);
        }
        if (par ? par->allreduce_max(g_stop_requested ? 1.0 : 0.0) > 0.0 : g_stop_requested.load()) { interrupted = true; break; }
        if (p.max_steps > 0 && step - start_step >= p.max_steps) break;
    }
    const double loop_wall = seconds_since(t0);

    if (mon) {
        if (step % p.monitor_interval != 0) row(step);
        std::fclose(mon);
    }
    if (!root) return;
    if (interrupted) std::printf("\n[INTERRUPTED] Stopping AMR simulation...\n");
    exporter.export_amr(file_prefix, fluid, tree, t, step);
    exporter.write_pvd(file_prefix);
    dump_final_state(step);
    std::printf("=== DONE  - open '%s.pvd' in ParaView ===  (time loop %.3f s, %d steps)\n", file_prefix.c_str(),
                loop_wall, step - start_step);
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
            std::printf("  %5s  %8s  %6s  %9s  %s\n", "Rank", dir == 'y' ? "j_start" : "i_start", dir == 'y' ? "j_end" : "i_end",
                        dir == 'y' ? "ny_local" : "nx_local", "role");
            std::printf("  -----  --------  ------  ---------  ----------------\n");
            for (int r = 0; r < decomp.n_ranks; ++r) {
                std::string role;
                if (r == 0) role = dir == 'y' ? "bottom" : "left";
                if (r == decomp.n_ranks - 1) {
                    if (!role.empty()) role += ", ";
                    role += dir == 'y' ? "top" : "right";
                }
                if (role.empty()) role = "interior";
                std::printf("  %5d  %8d  %6d  %9d  %s\n", all[4 * r], all[4 * r + 1], all[4 * r + 2], all[4 * r + 3], role.c_str());
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
            std::printf("[IBM] %s at (%g, %g) r=%g -> %d solid cells (across %d ranks)\n", p.ibm_shape.c_str(), p.ibm_cx,
                        p.ibm_cy, p.ibm_radius, total, decomp.n_ranks);
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
        for (Face f : dir == 'y' ? std::array<Face, 2>{Face::Left, Face::Right} : std::array<Face, 2>{Face::Bottom, Face::Top})
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
            if (p.resume) std::printf("\n[WARNING] No checkpoint at %s. Starting from t=0.\n", checkpoint_path().c_str());
            U_global = p.use_hot_start ? load_hot_start(p.hot_start_file) : initial_field();
        }
    }
    MPI_Bcast(&t, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&start, 1, MPI_INT, 0, comm);

    Field U_local = decomp.scatter(U_global);
    if (p.use_ibm && start == 0) local_ibm->apply_forcing(U_local, fluid);
    if (root)
        std::printf("[MPI] Initial field scattered to all ranks  (total cells: %d, per-rank avg: %d)\n\n",
                    mesh.num_cells(), mesh.num_cells() / decomp.n_ranks);

    Field halo_lo, halo_hi, forced;
    const ResidualFn res = [&](const Field& Uin, Field& R) {
        const Field* Uc = &Uin;
        if (local_ibm) {
            forced = Uin;
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
        if (p.use_ibm) {   // IBM loads: residual of the current state, summed over the strips
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
        if (decomp.allreduce_max(g_stop_requested ? 1.0 : 0.0) > 0.0) { interrupted = true; break; }
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
#endif  // FV_USE_MPI

}  // namespace fv
