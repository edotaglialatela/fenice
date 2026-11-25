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
    // positivity_fallback: the flags of the cells of those two halos (slot 0 of each State), exchanged like the states; nullptr when off
    const Field* flag_lo = nullptr;
    const Field* flag_hi = nullptr;
    // MUSCL-Hancock: the cells of an immersed body of the strip (not predicted) and the exchange of the predicted states of the edge rows
    const std::vector<char>* solid = nullptr;
    const std::vector<unsigned char>* solid_sides = nullptr;   // ... and the sides that face one (solid_side_mask), not predicted either
    FaceExchange exchange_faces{};
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
    // positivity_fallback: how many repair passes the stages needed so far (0: none, or the key is off).
    long long positivity_repairs() const { return fb_repairs_; }
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
        const std::vector<int>* ck = nullptr;   // 3-D adaptive mesh only
    };
    // Atomic checkpoint: U, t, step, nx, ny, Lx, Ly and the .pvd index. `leaves`: the AMR leaves, if `cells` is not the whole mesh.
    void save_checkpoint(int step, const LeafArrays* leaves = nullptr);
    // The same file for an adaptive run on several ranks, written by rank 0 as the leaves of each rank arrive
    // (CellParallel::stream_owned): no rank holds the whole mesh. Collective.
    void save_checkpoint_streamed(int step);
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
    // Pose of a body with a prescribed motion (ibm_vx, ibm_vy, ibm_omega) at time t: a pure function of t, so every MPI rank holds the same bits.
    ImmersedBoundary::BodyPose prescribed_pose(double t) const;

    // A body driven by the flow (ibm_mass > 0): translation of the body from its placement, its velocity, and what the second-order update needs.
    // Every rank of an MPI run advances the same copy from the same loads, so the poses agree bit for bit.
    struct BodyDynamics {
        double x = 0.0, y = 0.0, u = 0.0, v = 0.0;   // displacement and velocity at the current time
        double z = 0.0, w = 0.0, az = 0.0;           // the same along z (a 3-D body), and the acceleration of the previous step
        double ax = 0.0, ay = 0.0, dt = 0.0;         // acceleration of the previous step and its length (dt = 0: no previous step)
        double fx = 0.0, fy = 0.0, fz = 0.0;         // loads of the first stage of the current step (R of U^n at the pose of the step)
        bool captured = false;                       // fx, fy hold this step's loads
    };
    BodyDynamics body_;
    // The moving body in a checkpoint ("body_pose": the pose the geometry was built at, and for a body driven by the flow its state): written with
    // every checkpoint, read at setup of a resumed run on every rank, before anything is built from the body.
    std::vector<double> body_checkpoint() const;
    void restore_body_from_checkpoint();
    // The pose of the body for the step starting at time t: the state of body_ if the body is driven by the flow, else the prescribed one.
    ImmersedBoundary::BodyPose body_pose(double t) const;
    // Velocity of the body during the step (the pose's).
    std::array<double, 3> body_velocity() const;
    // Advance body_ by dt from the loads it holds (explicit, second order: Adams-Bashforth for the velocity, trapezoid for the position;
    // a first step is Euler). Springs and dampers act on the state at the start of the step; the weight m g acts with a body force.
    void advance_body(double dt);
    // The contribution of each solid cell of the part `m` of the whole mesh `g` (a strip placed at cell (i0, j0) of it; the whole mesh: m = g, 0, 0) to
    // the loads: its global index and {Fx, Fy, Mz} about the pivot, from global coordinates so a strip gives the serial bits. The sum is taken in
    // ascending global index (sum_load_terms), so it does not depend on how the mesh is cut.
    struct LoadTerm {
        long long cell;
        double fx, fy, mz, fz;
    };
    void ibm_load_terms(const Mesh& m, const Mesh& g, int i0, int j0, const std::vector<char>& solid, const Field& U, const Field& R, double pivot_x,
                        double pivot_y, std::vector<LoadTerm>& out) const;
    static std::array<double, 4> sum_load_terms(std::vector<LoadTerm>& terms);   // Fx, Fy, Mz, Fz

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
    RhsSourceTerms rhs_sources;                                  // ordered RHS contribution interface
    // Source terms of the case (source_*), evaluated once per mesh the residual is assembled on (the whole mesh,
    // or an MPI strip); keyed by (nx, ny, nz, dx, dy, dz, x0, y0, z0)
    std::map<std::array<double, 9>, Field> source_cache_;
    // Unsteady sources (a source expression uses t): per mesh, the fields of the two last stage times it was asked for
    // (most recent first); the stages that share a time reuse them (SSP-RK2: the last stage of a step is the first of
    // the next, SSP-RK3: the first stage of a step is its predecessor's second)
    std::map<std::array<double, 9>, std::array<std::pair<double, Field>, 2>> unsteady_source_cache_;
    const Field& source_for(const Mesh& m, double stage_time);
    std::map<std::string, std::function<double(double)>> boundary_expressions_;
    std::map<std::string, std::vector<std::pair<double, double>>> boundary_tables_;
    void load_boundary_functions();
    void update_boundary_values(double stage_time, BoundaryManager& manager) const;
    std::unique_ptr<ImmersedBoundary> ibm;                      // optional obstacle
    std::function<ImmersedBoundary(const Mesh&)> ibm_factory;
    // A prescribed motion on an adaptive mesh: the pose the factory builds the body at (geo, the geometry of the last body update; the engine, the
    // band and every leaf classification follow it) and the pivot. Shared with the factory and the velocity function of the engine.
    struct MovingBodyState {
        ImmersedBoundary::BodyPose geo;
        double px = 0.0, py = 0.0, pz = 0.0;
    };
    std::shared_ptr<MovingBodyState> amr_body_;   // same body on the uniform mesh of any level
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
    VTKExporter exporter;                                       // VTK + .pvd output
    const TestCase* test_case = nullptr;
    std::string file_prefix;   // test case name, lower case with underscores

    Field U;              // global conservative state
    double t = 0.0;       // physical time [s]
    int start_step = 0;   // step to resume from

private:
    void solve_amr();
    // The entries of a checkpoint other than the state and the leaves: time, step, mesh size, snapshot index, format version.
    NpzArchive checkpoint_header(int step) const;
    // What an engine (uniform, uniform on MPI strips, adaptive) gives the time loop. run_loop owns what the engines share: the
    // output times, when a snapshot is due, the stop requests, max_steps and the wall time of the loop.
    struct LoopEngine {
        // Step number `step` from time t, not beyond `t_stop` (the next output time): the state, the solid reset, t and the stage
        // time. Returns the time step taken and the largest wave speed.
        std::function<StepResult(int step, double t_stop)> advance;
        // After step `step`: what the engine does every step, in its own order (remesh, monitor row, post-processing outputs).
        std::function<void(int step)> after_step;
        // A step where output is due: snapshot, checkpoint and the progress line; `wall` = seconds since the loop began.
        std::function<void(int step, const StepResult& r, double wall)> save;
        // Whether a stop was asked for (Ctrl+C or the STOP file). Collective on several ranks, so that all of them leave the loop
        // at the same step; empty = the flag of this process.
        std::function<bool()> stop_requested;
        // Section::Step of the profile covers the whole iteration (false: the engine times its own step).
        bool profile_iteration = true;
    };
    // How the time loop ended.
    struct LoopResult {
        int step = 0;                // last step taken
        double wall = 0.0;           // seconds spent in the loop
        bool interrupted = false;    // left on a stop request
        bool snapshot_due = false;   // the last step has no snapshot yet (or the run was interrupted): write one
    };
    // The time loop from start_step (whose snapshot the engine wrote if it is step 0) to t_end, a stop request or max_steps.
    LoopResult run_loop(const LoopEngine& e);
    // Initial state of the leaves of `m` at their centres: the initial condition of the case evaluated on the uniform mesh of
    // each leaf's level, or the hot-start field interpolated.
    void fill_cells(CellMesh& m) const;
    // {Fx, Fy, Mz} of the immersed body from the residual of the leaves (the definition of ibm_loads).
    std::array<double, 3> cell_loads(const CellMesh& m, const Field& R) const;
    std::unique_ptr<GridInterp2D> hot_interp_;   // hot-start field of an AMR run
    std::unique_ptr<GridInterp3D> hot_interp3_;  // the same on an octree (3-D)
    Field hot_src_;                              // the hot-start source at its own resolution (set by load_hot_start)
    int hot_src_nx_ = 0, hot_src_ny_ = 0, hot_src_nz_ = 1;   // its size (nx 0: none)
    bool share_resumed_ = false;                 // the AMR state came from the checkpoint straight into each rank's share
    double cell_speed_ = 0.0;                    // wave-speed bound the next AMR time step is taken from
    MemoryUsage mem_setup_, mem_init_;           // resident memory after setup and after the initial adaptation (AMR report)
    // FENICE_REPLICATED_REMESH=1: every rank keeps the whole AMR mesh (the storage before the distributed one)
    static bool replicated_storage();
    void print_halo_reach(const char* when) const;   // FENICE_HALO_REACH=1: CellParallel::halo_reach of the AMR mesh
    // Monitor CSV (rank 0): header on open, one row per call.
    std::FILE* open_monitor() const;
    // A file named STOP in the output folder asks for the same clean stop as Ctrl+C (checkpoint + .pvd); this works under MPI,
    // where mpiexec aborts the job on Ctrl+C. arm_stop_file removes a stale one at the start of the time loop, poll_stop_file
    // (once per step) turns a new one into g_stop_requested.
    void arm_stop_file() const;
    void poll_stop_file() const;
    // `loads`: {Fx, Fy, Mz} of the IBM body (adds Fx, Fy, Cd, Cl, Cm), nullptr without IBM.
    // `res`: the residual norms {rho, momentum, energy} (columns res_rho, res_mom, res_E): the root-mean-square over the fluid
    // cells (weighted by cell area / volume) of dU/dt, the right-hand side of the semi-discrete equations, per conserved quantity
    // (momentum: the magnitude of the vector). They tend to zero as a flow becomes steady; the GUI normalises them by the first row.
    void monitor_row(std::FILE* f, int step, const std::array<double, 3>& v, const std::array<double, 3>* loads,
                     const std::array<double, 3>& res) const;

    // Scratch buffers reused across residual evaluations
    Field UL_, UR_, F_, Upad_;
    std::vector<State> Usaved_;   // states of the body cells while the ghost-wall forcing is applied in place
    struct InPlaceForcing;
    PaddedFrame frame_;   // 2-D viscous sweep: ghost ring of the padded state (Upad_ is the 3-D padded state)
    Field perL_, perR_, perB_, perT_, perF_, perK_;   // wrap-around MUSCL halos of periodic axes
    Field ghL_, ghR_, ghB_, ghT_, ghF_, ghK_;         // ghost layers of the physical sides (boundary_order = 2)
    // positivity_fallback: the flagged cells of the stage being repaired (the residual then takes the two-pass path), the guard that
    // fills them (installed in the time scheme) and the totals printed at the end of the run
    std::vector<char> fb_cells_;
    bool fb_active_ = false;
    StageGuard fb_guard_;
    std::function<bool(bool)> fb_reduce_;   // MPI: the OR of the ranks' "the set grew" (every rank repeats the stage together); empty: serial
    long long fb_repairs_ = 0, fb_flagged_ = 0;
    std::vector<unsigned char> hancock_sides_;   // MUSCL-Hancock with an immersed body: the sides of the cells that face a solid cell
    Field hsrc_;   // MUSCL-Hancock: body-force + source terms of the stage, computed before the predictor
    long long hancock_reverted_ = 0;   // MUSCL-Hancock: predicted face states left unpredicted because they were inadmissible (total of the run)
};

// "12.3s" or "m:ss".
std::string format_elapsed(double seconds);
// Print cpp/logo.txt. `colors`: 0 plain, 256 (xterm-256 palette) or 24 (24-bit
// colour) for a gold-to-oxblood gradient; see terminal_colors() in the fenice driver.
void print_logo(int colors = 0);

}   // namespace fv
