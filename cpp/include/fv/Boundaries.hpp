// Ghost-cell boundary conditions.
//
// Convective ghosts fill the U_L / U_R face arrays fed to the Riemann solver;
// diffusive ghosts fill a (ny+2) x (nx+2) padded array consumed by the gradient
// calculator and the viscous flux.
#pragma once

#include <functional>

#include "fv/Core.hpp"

namespace fv {

// Boundary-condition types (see the module page for the ghost-state formulas).
enum class BCType { Wall, NoSlip, MovingWall, VelInlet, MassflowInlet, Outlet, Transmissive, Periodic, NonReflecting };

// Accepts both internal names ("no_slip") and GUI labels ("Viscous Wall (No-Slip)").
BCType parse_bc(const std::string& s);
// Internal name of a BC type ("no_slip", "vel_inlet", ...).
std::string bc_name(BCType t);

// Parameters of the inlet / outlet / moving-wall / non-reflecting BCs.
struct BCParams {
    double u_in = 0.0, v_in = 0.0, rho_in = 1.0, p_out = 1.0;   // inlet/wall velocity, inlet density, outlet pressure
    double nrbc_sigma = 0.28;             // non-reflecting outlet: relaxation towards p_out (0: none)
    double nrbc_beta = -1.0;              // NRBC transverse weight (1: none, 0: full; < 0: local Mach number)
    double length[3] = {1.0, 1.0, 1.0};   // global domain size per axis (relaxation length of the NRBC)
    int boundary_order = 1;               // 2: the viscous ghost of a moving wall keeps the internal energy of the interior cell
};

// MPI strip halos, for the non-reflecting BC's derivatives along a boundary that crosses rank
// interfaces. Layouts as in the halo exchange: y-strips [(d*nz + k)*nx + i], x-strips
// [(k*ny + j)*depth + d]; lo's layer depth-1 and hi's layer 0 touch the strip.
struct TangentialHalos {
    char dir = 'y';              // decomposition direction
    int depth = 2;               // halo layers
    const Field* lo = nullptr;   // lower neighbour's cells (nullptr: physical end)
    const Field* hi = nullptr;   // upper neighbour's cells
};

// Builds the ghost states of one (sub)domain from its boundary types (four sides in
// 2-D, six in 3-D).  Defaults: transmissive left/right, slip walls bottom/top,
// periodic front/back.
class BoundaryManager {
public:
    BoundaryManager() = default;
    // Manager for `mesh`; ghost states use `eos` for primitive <-> conservative conversions.
    BoundaryManager(const Mesh& mesh, const EOS& eos) : mesh_(mesh), eos_(eos) {}

    // Set the BC type of each in-plane side (front / back keep theirs).
    void set_boundary_types(BCType left, BCType right, BCType bottom, BCType top);
    // Set the inlet / outlet / moving-wall parameters.
    void set_params(const BCParams& p) { params_ = p; }

    // BC type of side f.
    BCType type(Face f) const { return types_[static_cast<int>(f)]; }
    // Change the BC type of one side.
    void set_type(Face f, BCType t) { types_[static_cast<int>(f)] = t; }
    // Current BC parameters.
    const BCParams& params() const { return params_; }
    // Mesh the manager was built for.
    const Mesh& mesh() const { return mesh_; }

    // First-order face states for every face of the mesh (UL/UR resized to num_faces).
    // `th`: MPI strip halos (non-reflecting sides crossing rank interfaces). `interior = false` leaves the faces between
    // two cells unset (resized only) for a caller that overwrites every one of them, such as a MUSCL / WENO reconstruction.
    void apply_boundary_conditions(const Field& U, Field& UL, Field& UR, const TangentialHalos* th = nullptr,
                                   bool interior = true) const;

    // Ghost state across a physical boundary `side` for the interior state Ui (every type but periodic and
    // non-reflecting, which need more than one cell); `u_target` is the mass-flow inlet velocity. The same
    // function builds the ghosts of apply_boundary_conditions.
    State ghost_state(Face side, const State& Ui, double u_target = 0.0) const;

    // Tangential neighbours of a cell next to a non-reflecting side: for each tangential axis t (use[t]) the
    // primitive states at both ends of the central difference (the cell itself at a domain end, then span = 1) and
    // the number of cell widths between them (2, or 1 at an end).
    struct NrbcTangent {
        bool use[3] = {false, false, false};
        State Wlo[3], Whi[3];
        double span[3] = {2.0, 2.0, 2.0};
    };
    // Ghost state of the non-reflecting outlet for the interior state Ui; `hs` are the cell sizes per axis.
    State nonreflecting_ghost(Face side, const State& Ui, const NrbcTangent& tg, const double* hs) const;
    // Ghost layers of a physical side in the slab layout of the MPI halos (Halos in Schemes.hpp: b/t [(d*nz + k)*nx + i],
    // l/r [(k*ny + j)*depth + d], f/bk [(d*ny + j)*nx + i]; d grows with the coordinate): layer l (l = 0 next to the
    // boundary) is the ghost state of the interior cell l cells from the boundary, i.e. the even / odd extension of
    // the solution across the boundary that the first-order ghost states are the first layer of. Reconstructing
    // through them gives the faces on the boundary a second-order state. Not for periodic or non-reflecting sides;
    // needs the side's mass-flow target of the latest apply_boundary_conditions. Returns false (slab untouched)
    // when the side has fewer than `depth` cells.
    // boundary_order = 2, 2-D: rebuild the four corner entries of the padded state (rows 0 and ny+1, columns 0 and
    // nx+1) as the left / right boundary ghost of the cell in the ghost row, so the tangential differences of the
    // face values on a left / right boundary face see the wrapped (periodic), neighbour-rank (MPI) or mirrored value
    // instead of an edge copy. 3-D: the same for the edge entries (ghost along two axes). `mpi_axis` (0 = x, 1 = y,
    // -1 none) and `mpi_lo` / `mpi_hi` say that the ghost layer of that side was filled from a neighbour rank: the
    // entry then takes the boundary condition of the other axis, applied to the halo cell. Call after the MPI layers
    // are written; no effect with boundary_order = 1.
    void fix_padded_corners(Field& P, int mpi_axis = -1, bool mpi_lo = false, bool mpi_hi = false) const;
    bool fill_ghost_slab(const Field& U, Face side, int depth, Field& slab) const;

    // Ghost cell of the diffusive (viscous) stencil next to a physical side `side` for the interior state `in`:
    // the zero-gradient copy with the momentum rules of get_padded_U (2-D). Periodic sides use the opposite cell.
    State viscous_ghost(Face side, const State& in) const;

    // U padded with BC-consistent ghost cells: (ny+2)*(nx+2) in 2-D, (nz+2)*(ny+2)*(nx+2) in 3-D (row-major).
    void get_padded_U(const Field& U, Field& U_pad) const;

    // MPI: combine the mass-flow inlet's (sum, count) of interior density across
    // the ranks sharing side `f`, so the inlet average matches the serial run.
    void set_mean_reducer(Face f, std::function<void(double& sum, double& count)> r) {
        reducers_[static_cast<int>(f)] = std::move(r);
    }

private:
    void apply_side(const Field& U, Field& UL, Field& UR, Face side, const TangentialHalos* th) const;
    void padded_3d(const Field& U, Field& P) const;

    Mesh mesh_;
    EOS eos_;
    BCType types_[6] = {BCType::Transmissive, BCType::Transmissive, BCType::Wall,
                        BCType::Wall,         BCType::Periodic,     BCType::Periodic};
    BCParams params_;
    std::function<void(double&, double&)> reducers_[6];
    mutable double u_target_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // mass-flow inlet velocity per side (apply_side)
};

}   // namespace fv
