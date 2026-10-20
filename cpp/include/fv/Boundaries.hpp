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
enum class BCType { Wall, NoSlip, MovingWall, VelInlet, MassflowInlet, Outlet, Transmissive, Periodic };

// Accepts both internal names ("no_slip") and GUI labels ("Viscous Wall (No-Slip)").
BCType      parse_bc(const std::string& s);
// Internal name of a BC type ("no_slip", "vel_inlet", ...).
std::string bc_name(BCType t);

// Parameters of the inlet / outlet / moving-wall BCs.
struct BCParams {
    double u_in = 0.0, v_in = 0.0, rho_in = 1.0, p_out = 1.0;   // inlet/wall velocity, inlet density, outlet pressure
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
    BCType          type(Face f) const { return types_[static_cast<int>(f)]; }
    // Change the BC type of one side.
    void            set_type(Face f, BCType t) { types_[static_cast<int>(f)] = t; }
    // Current BC parameters.
    const BCParams& params() const { return params_; }
    // Mesh the manager was built for.
    const Mesh&     mesh() const { return mesh_; }

    // First-order face states for every face of the mesh (UL/UR resized to num_faces).
    void apply_boundary_conditions(const Field& U, Field& UL, Field& UR) const;

    // U padded with BC-consistent ghost cells: (ny+2)*(nx+2) in 2-D, (nz+2)*(ny+2)*(nx+2) in 3-D (row-major).
    void get_padded_U(const Field& U, Field& U_pad) const;

    // MPI: combine the mass-flow inlet's (sum, count) of interior density across
    // the ranks sharing side `f`, so the inlet average matches the serial run.
    void set_mean_reducer(Face f, std::function<void(double& sum, double& count)> r) {
        reducers_[static_cast<int>(f)] = std::move(r);
    }

private:
    void apply_side(const Field& U, Field& UL, Field& UR, Face side) const;
    void padded_3d(const Field& U, Field& P) const;

    Mesh     mesh_;
    EOS      eos_;
    BCType   types_[6] = {BCType::Transmissive, BCType::Transmissive, BCType::Wall, BCType::Wall,
                          BCType::Periodic, BCType::Periodic};
    BCParams params_;
    std::function<void(double&, double&)> reducers_[6];
};

}  // namespace fv
