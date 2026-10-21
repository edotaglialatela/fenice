// Simulation orchestrator.
#pragma once

#include <array>
#include <atomic>
#include <cstdio>
#include <functional>
#include <memory>

#include "fv/AMR.hpp"
#include "fv/Boundaries.hpp"
#include "fv/IO.hpp"
#include "fv/Params.hpp"
#include "fv/Physics.hpp"
#include "fv/Schemes.hpp"
#include "fv/TestCases.hpp"
#include "fv/TimeScheme.hpp"

#ifdef FV_USE_MPI
#include <mpi.h>
#endif

namespace fv {

// Set by the SIGINT handler; the time loops stop cleanly (checkpoint + .pvd).
extern std::atomic<bool> g_stop_requested;

// MPI-interface halos for one strip (dir 'y': lo=bottom/hi=top, 'x': lo=left/hi=right).
struct StripHalos {
    char         dir = 'y';        // decomposition direction
    const Field* lo  = nullptr;    // halo from the lower-rank neighbour (nullptr at a physical boundary)
    const Field* hi  = nullptr;    // halo from the higher-rank neighbour
};

// Owns every solver object of a run and drives the time loop (serial, AMR or MPI).
class Simulation {
public:
    // Build mesh, EOS, BCs, schemes and optional physics from `params`.
    void setup(const SimParams& params, bool verbose = true);
    // Initial field: test-case initializer or hot start, IBM forcing, case BCs,
    // AMR seeding, and the checkpoint when resuming.
    void initialize_field();

    // Full spatial residual R(U) on the serial domain; returns S_max.
    double residual(const Field& U, Field& R);
    // Residual of one AMR leaf: convective flux, plus (when enabled) IBM forcing, viscous flux and
    // body force. Ghost states across patch interfaces come from the neighbouring leaves; the
    // viscous flux is added to flux_x / flux_y so coarse-fine refluxing also corrects it.
    double patch_residual(AMRPatch& patch, AMRTree& tree, const Field& U, Field& R, Field& flux_x, Field& flux_y);

    // Build the IBM masks of an AMR leaf from the body geometry (no-op if present or without IBM).
    // Touches only `patch`, so concurrent calls on different leaves are safe.
    void ensure_patch_masks(AMRPatch& patch) const;
    // Once per complete AMR step: fringe Neumann + deep-solid reset on every leaf, reading the
    // neighbour cells across patch interfaces from a snapshot taken before any leaf is reset.
    void amr_solid_reset(AMRTree& tree);

    // True if the box of `patch`, grown by one of its cells, overlaps the immersed body's bounding box.
    bool patch_touches_body(const AMRPatch& patch) const;
    // Refine leaves touching the body up to max_level (2:1 balanced), e.g. before seeding at t = 0.
    void refine_body_patches();

    void solve();   // serial uniform mesh or AMR
#ifdef FV_USE_MPI
    // Full MPI run (all ranks call it): decomposition, scatter, time loop, gather + output on rank 0.
    void run_mpi(MPI_Comm comm);
#endif

    // results/<case>/<case>_restart.npz
    std::string checkpoint_path() const;
    // ibm_file resolved: absolute, else next to the case.toml, else the current folder.
    std::string airfoil_file_path() const;
    // Atomic checkpoint: U, t, step, nx, ny, Lx, Ly and the .pvd index.
    void        save_checkpoint(int step);
    // Load the checkpoint (if any); returns the step to resume from (0 if missing).
    int         load_checkpoint();
    // Bilinear mapping of a checkpoint (any resolution) onto this run's mesh.
    Field       load_hot_start(const std::string& path);
    // Write the final state to `dump_state` (if set) for regression checks.
    void        dump_final_state(int step) const;
    // Volume averages of the field: mass, kinetic energy, total energy (all ranks under MPI).
    std::array<double, 3> volume_averages(const Field& U_part, long long n_global) const;
    // Loads of the fluid on the IBM body over the cells of `m`: {Fx, Fy, Mz}, Mz about
    // (ibm_cx, ibm_cy), counter-clockwise positive. The net momentum flux into the solid cells,
    // i.e. the sum of their residuals R (from residual(U, R)) times the cell volume, body force
    // excluded: pressure and viscous parts together, for any shape. Per unit depth in 2-D.
    std::array<double, 3> ibm_loads(const Mesh& m, const ImmersedBoundary& ib, const Field& U, const Field& R) const;

    // Shared residual assembly for a (sub)domain, optionally with MPI halos.
    double assemble(const Mesh& m, const BoundaryManager& bm, const ImmersedBoundary* ib,
                    const MusclReconstruction* mu, const Field& U_in, Field& R, const StripHalos* halos);
    // 3-D counterpart (called by assemble when m.is3d()).
    double assemble_3d(const Mesh& m, const BoundaryManager& bm, const ImmersedBoundary* ib,
                       const MusclReconstruction* mu, const Field& U_in, Field& R, const StripHalos* halos);
    // Initial field of the test case (2-D cases are extruded along z on a 3-D mesh).
    Field initial_field() const;

    SimParams                            p;              // run parameters
    EOS                                  fluid;          // equation of state
    Mesh                                 mesh;           // global mesh
    double                               dt_length = 0;  // CFL length scale (set by setup)
    BoundaryManager                      boundaries;     // global BCs
    RiemannSolver                        riemann;        // convective flux
    std::unique_ptr<MusclReconstruction> muscl;          // nullptr: 1st-order Godunov
    std::unique_ptr<NavierStokes>        ns;             // nullptr: Euler
    std::unique_ptr<ConstantBodyForce>   body_force;     // optional source term
    std::unique_ptr<ImmersedBoundary>    ibm;            // optional obstacle
    std::function<ImmersedBoundary(const Mesh&)> ibm_factory;   // same body on any (patch) mesh
    std::array<double, 4> ibm_bbox{};                           // body extent {xmin, xmax, ymin, ymax}
    std::unique_ptr<TimeScheme>          scheme;         // time integrator
    std::unique_ptr<AMRTree>             amr_tree;       // non-null when AMR is on
    std::unique_ptr<AMRErrorIndicator>   amr_indicator;  // AMR marking
    VTKExporter                          exporter;       // VTK + .pvd output
    const TestCase*                      test_case = nullptr;
    std::string                          file_prefix;    // test case name, lower case with underscores

    Field  U;                // global conservative state
    double t          = 0.0; // physical time [s]
    int    start_step = 0;   // step to resume from

private:
    void solve_amr();
    // Monitor CSV (rank 0): header on open, one row per call.
    std::FILE* open_monitor() const;
    // `loads`: {Fx, Fy, Mz} of the IBM body (adds Fx, Fy, Cd, Cl, Cm), nullptr without IBM.
    void monitor_row(std::FILE* f, int step, const std::array<double, 3>& v,
                     const std::array<double, 3>* loads = nullptr) const;

    // Scratch buffers reused across residual evaluations
    Field Uf_, UL_, UR_, F_, Upad_, gX_, gY_, Rv_;
    Field perL_, perR_, perB_, perT_, perF_, perK_;   // wrap-around MUSCL halos of periodic axes
    Field gZ_;
};

// AMR ghost-cell helpers (exposed for tests)
// Linear prolongation of a COARSER neighbour into patch's ghost layer at 'face'.
Field prolong_ghost_face(const AMRPatch& patch, const AMRPatch& nb, Face face);
// State of the leaf cell containing (x, y), or nullptr if no leaf covers it.
const State* sample_leaf_state(const std::vector<AMRPatch*>& leaves, double x, double y);
// Same, through the tree's spatial index (no scan over the leaves).
const State* sample_leaf_state(const AMRTree& tree, double x, double y);

// "12.3s" or "m:ss".
std::string format_elapsed(double seconds);
// Print aesthetics/logo.txt. `colors`: 0 plain, 256 (xterm-256 palette) or 24 (24-bit
// colour) for a gold-to-oxblood gradient; see terminal_colors() in the fenice driver.
void        print_logo(int colors = 0);

}  // namespace fv
