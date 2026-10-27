// Simulation orchestrator.
#pragma once

#include <array>
#include <atomic>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>

#include "fv/Boundaries.hpp"
#include "fv/CellAdvance.hpp"
#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"
#include "fv/CellParallel.hpp"
#include "fv/CellRemesh.hpp"
#include "fv/IO.hpp"
#include "fv/Params.hpp"
#include "fv/Physics.hpp"
#include "fv/Schemes.hpp"
#include "fv/SolutionMapping.hpp"
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
    char dir = 'y';              // decomposition direction
    const Field* lo = nullptr;   // halo from the lower-rank neighbour (nullptr at a physical boundary)
    const Field* hi = nullptr;   // halo from the higher-rank neighbour
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
    void solve();   // serial uniform mesh or AMR
#ifdef FV_USE_MPI
    // Full MPI run (all ranks call it): decomposition, scatter, time loop, gather + output on rank 0.
    void run_mpi(MPI_Comm comm);
#endif

    // results/<case>/<case>_restart.npz
    std::string checkpoint_path() const;
    // ibm_file resolved: absolute, else next to the case.toml, else the current folder.
    std::string airfoil_file_path() const;
    // The leaves of the whole AMR mesh and their state, in (level, key) order: the arrays of `cells`, or what a distributed run
    // gathered (CellParallel::collect).
    struct LeafArrays {
        const std::vector<int>* level = nullptr;
        const std::vector<int>* ci = nullptr;
        const std::vector<int>* cj = nullptr;
        const Field* U = nullptr;
    };
    // Atomic checkpoint: U, t, step, nx, ny, Lx, Ly and the .pvd index. `leaves`: the AMR leaves, if `cells` is not the whole mesh.
    void save_checkpoint(int step, const LeafArrays* leaves = nullptr);
    // Load the checkpoint (if any); returns the step to resume from (0 if missing).
    int load_checkpoint();
    // Bilinear mapping of a checkpoint (any resolution) onto this run's mesh.
    Field load_hot_start(const std::string& path);
    // Write the final state to `dump_state` (if set) for regression checks: U, and with AMR the leaves' bounds and level
    // (`leaf_meta` rows x0, x1, y0, y1, level). With AMR on several ranks the state of every leaf must be valid on rank 0.
    void dump_final_state(int step, const LeafArrays* leaves = nullptr) const;
    // Volume averages of the field: mass, kinetic energy, total energy (all ranks under MPI).
    std::array<double, 3> volume_averages(const Field& U_part, long long n_global) const;
    // Loads of the fluid on the IBM body over the cells of `m`: {Fx, Fy, Mz}, Mz about
    // (ibm_cx, ibm_cy), counter-clockwise positive. The net momentum flux into the solid cells,
    // i.e. the sum of their residuals R (from residual(U, R)) times the cell volume, body force
    // excluded: pressure and viscous parts together, for any shape. Per unit depth in 2-D.
    std::array<double, 3> ibm_loads(const Mesh& m, const ImmersedBoundary& ib, const Field& U, const Field& R) const;

    // Shared residual assembly for a (sub)domain, optionally with MPI halos.
    double assemble(const Mesh& m, const BoundaryManager& bm, const ImmersedBoundary* ib, const MusclReconstruction* mu,
                    const Field& U_in, Field& R, const StripHalos* halos);
    // 3-D counterpart (called by assemble when m.is3d()).
    double assemble_3d(const Mesh& m, const BoundaryManager& bm, const ImmersedBoundary* ib,
                       const MusclReconstruction* mu, const Field& U_in, Field& R, const StripHalos* halos);
    // Initial field of the test case (2-D cases are extruded along z on a 3-D mesh).
    Field initial_field() const;

    SimParams p;                                                // run parameters
    EOS fluid;                                                  // equation of state
    Mesh mesh;                                                  // global mesh
    double dt_length = 0;                                       // CFL length scale (set by setup)
    BoundaryManager boundaries;                                 // global BCs
    RiemannSolver riemann;                                      // convective flux
    std::unique_ptr<MusclReconstruction> muscl;                 // nullptr: 1st-order Godunov
    std::unique_ptr<NavierStokes> ns;                           // nullptr: Euler
    std::unique_ptr<ConstantBodyForce> body_force;              // optional source term
    // Source terms of the case (source_*), evaluated once per mesh the residual is assembled on (the whole mesh,
    // or an MPI strip); keyed by (nx, ny, nz, dx, dy, dz, x0, y0, z0)
    std::map<std::array<double, 9>, Field> source_cache_;
    // Unsteady sources (a source expression uses t): per mesh, the field of the last stage time it was asked for;
    // the inner iterations of BDF2 and the stages that share a time reuse it
    std::map<std::array<double, 9>, std::pair<double, Field>> unsteady_source_cache_;
    const Field& source_for(const Mesh& m);
    std::unique_ptr<ImmersedBoundary> ibm;                      // optional obstacle
    std::function<ImmersedBoundary(const Mesh&)> ibm_factory;   // same body on the uniform mesh of any level
    std::unique_ptr<TimeScheme> scheme;                         // time integrator
    // Adaptive mesh (use_amr): the leaves, the engine that computes their residual, the indicator, and on several MPI ranks
    // the distribution of the leaves (CellParallel). `cells` is null without AMR.
    std::unique_ptr<CellMesh> cells;
    std::unique_ptr<CellFluxEngine> cell_engine;
    CellPhysics cell_physics;
    CellIndicator cell_indicator;
    std::unique_ptr<CellAdaptor> cell_adaptor;
    CellComm cell_comm;                       // rank and size (1 without MPI)
    std::unique_ptr<CellParallel> cell_par;   // non-null on several ranks
    // Localized run: the topology of the whole mesh (leaves and faces, no state), kept by every rank for CellParallel::remesh
    std::unique_ptr<CellMesh> cell_topo;
    VTKExporter exporter;                                       // VTK + .pvd output
    const TestCase* test_case = nullptr;
    std::string file_prefix;   // test case name, lower case with underscores

    Field U;              // global conservative state
    double t = 0.0;       // physical time [s]
    int start_step = 0;   // step to resume from

private:
    void solve_amr();
    // Initial state of the leaves of `m` at their centres: the initial condition of the case evaluated on the uniform mesh of
    // each leaf's level, or the hot-start field interpolated.
    void fill_cells(CellMesh& m) const;
    // {Fx, Fy, Mz} of the immersed body from the residual of the leaves (the definition of ibm_loads).
    std::array<double, 3> cell_loads(const CellMesh& m, const Field& R) const;
    std::unique_ptr<GridInterp2D> hot_interp_;   // hot-start field of an AMR run
    double cell_speed_ = 0.0;                    // wave-speed bound the next AMR time step is taken from
    MemoryUsage mem_setup_, mem_init_;           // resident memory after setup and after the initial adaptation (AMR report)
    // FENICE_REPLICATED_REMESH=1: every rank keeps the whole AMR mesh (the storage before docs/design/amr-distributed.md phase 2)
    static bool replicated_storage();
    void print_halo_reach(const char* when) const;   // FENICE_HALO_REACH=1: CellParallel::halo_reach of the AMR mesh
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

// "12.3s" or "m:ss".
std::string format_elapsed(double seconds);
// Print cpp/logo.txt. `colors`: 0 plain, 256 (xterm-256 palette) or 24 (24-bit
// colour) for a gold-to-oxblood gradient; see terminal_colors() in the fenice driver.
void print_logo(int colors = 0);

}   // namespace fv
