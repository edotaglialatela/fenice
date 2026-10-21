// Additional physics: cell gradients, Navier-Stokes viscous residual, body
// forces and the immersed boundary method (ports of gradientCalculation.py,
// addPhys.py, sourceTerms.py, IBM.py).
#pragma once

#include "fv/Core.hpp"
#include "fv/Geometry.hpp"

namespace fv {

// Central-difference cell-centre gradients from a (ny+2)*(nx+2) padded state.
void compute_gradients(const Field& U_pad, int nx, int ny, double dx, double dy,
                       Field& grad_x, Field& grad_y);
// 3-D version from a (nz+2)*(ny+2)*(nx+2) padded state.
void compute_gradients_3d(const Field& U_pad, const Mesh& mesh, Field& grad_x, Field& grad_y, Field& grad_z);

// Tangential velocity derivatives in the ghost cells across MPI interfaces, so
// the cross-derivative face averages match the serial stencil.
//   dir 'y': a = du/dx, b = dv/dx, c = dw/dz, d = dv/dz of the bottom (lo) / top (hi)
//            ghost slab, index k*nx + i  (2-D: a, b only, index i)
//   dir 'x': a = du/dy, b = dv/dy, c = dw/dz, d = du/dz of the left (lo) / right (hi)
//            ghost slab, index k*ny + j  (2-D: a, b only, index j)
struct CrossGhosts {
    char dir = 'y';                 // decomposition direction
    const double* lo_a = nullptr;   // low side (see above)
    const double* lo_b = nullptr;
    const double* lo_c = nullptr;   // 3-D only
    const double* lo_d = nullptr;   // 3-D only
    const double* hi_a = nullptr;   // high side
    const double* hi_b = nullptr;
    const double* hi_c = nullptr;
    const double* hi_d = nullptr;
    // Periodic axes inside this (sub)domain: boundary faces take the opposite edge cell.
    bool periodic_x = false, periodic_y = false, periodic_z = false;
};

// Viscous (parabolic) part of the compressible Navier-Stokes residual: Newtonian
// stress tensor (Stokes hypothesis) and Fourier heat flux, k = mu gamma / (Pr (gamma-1)).
class NavierStokes {
public:
    // Dynamic viscosity `mu` [Pa s] and Prandtl number `Pr`.
    NavierStokes(const EOS& eos, double mu, double Pr) : eos_(eos), mu_(mu), Pr_(Pr) {}

    // Writes the viscous residual into R_visc and returns the parabolic
    // time-scale limit dt_visc.  U_pad is the BC-aware padded state.
    double viscous_residual(const Field& U, const Field& grad_x, const Field& grad_y,
                            const Mesh& mesh, const Field& U_pad, Field& R_visc,
                            const CrossGhosts* cross = nullptr) const;
    // 3-D version: full stress tensor tau_ij = mu (du_i/dx_j + du_j/dx_i) - 2/3 mu delta_ij div u.
    double viscous_residual_3d(const Field& U, const Field& grad_x, const Field& grad_y, const Field& grad_z,
                               const Mesh& mesh, const Field& U_pad, Field& R_visc,
                               const CrossGhosts* cross = nullptr) const;

    // Dynamic viscosity.
    double mu() const { return mu_; }

private:
    EOS    eos_;
    double mu_, Pr_;
    mutable Field W_, W_pad_;
    mutable std::vector<double> scratch_;
};

// Uniform body-force acceleration (gravity or imposed pressure gradient).
struct ConstantBodyForce {
    double gx = 0.0, gy = 0.0;   // acceleration [m/s^2]; gy < 0 for downward gravity
    double gz = 0.0;             // z-acceleration (3-D)
    // Add S = [0, rho gx, rho gy, rho u.g, rho gz] to the residual R.
    void add_to(const Field& U, Field& R) const;
};

// Neighbour cells across the MPI interfaces of a strip, for the fringe test of
// the solid reset.  dir 'y': lo/hi rows, index k*nx + i; dir 'x': columns, index k*ny + j.
struct SolidResetHalo {
    char dir = 'y';                              // decomposition direction
    const Field*             lo = nullptr;       // neighbour states, low side
    const Field*             hi = nullptr;       // neighbour states, high side
    const std::vector<char>* mask_lo = nullptr;  // solid mask of those neighbours
    const std::vector<char>* mask_hi = nullptr;
};

// Direct-forcing immersed boundary with a circular obstacle.
class ImmersedBoundary {
public:
    ImmersedBoundary() = default;
    // Obstacle of `radius` centred at (cx, cy[, cz]); deep solid cells are frozen to
    // (rho_ref, p_ref).  2-D: "Cylinder" and "Sphere" are both a circle.  3-D: "Sphere"
    // is a sphere, "Cylinder" an infinite cylinder along z.  Case-insensitive; other shapes throw.
    ImmersedBoundary(const Mesh& mesh, const std::string& shape, double radius, double cx, double cy,
                     double rho_ref, double p_ref, double cz = 0.0);
    // Body bounded by a closed polygon in the x-y plane (an airfoil section): a cell is solid
    // when its centre is inside. 3-D: the section extruded along z (a wing of infinite span).
    ImmersedBoundary(const Mesh& mesh, const Polygon2D& section, double rho_ref, double p_ref);
    // From a precomputed mask (MPI strips cut from the global mask).
    ImmersedBoundary(int nx, int ny, std::vector<char> mask, double rho_ref, double p_ref, int nz = 1)
        : nx_(nx), ny_(ny), nz_(nz), rho_ref_(rho_ref), p_ref_(p_ref), mask_(std::move(mask)) {}

    // Zero the velocity in solid cells (safe inside RK stages).
    void apply_forcing(Field& U, const EOS& eos) const;
    // Once per complete step: fringe Neumann + deep-solid reset.
    void apply_solid_reset(Field& U, const EOS& eos, const SolidResetHalo* halo = nullptr) const;

    // Number of solid cells.
    int  solid_count() const;
    // Solid mask, one entry per cell (1 = solid).
    const std::vector<char>& mask() const { return mask_; }

private:
    int nx_ = 0, ny_ = 0, nz_ = 1;
    double rho_ref_ = 1.0, p_ref_ = 1.0;
    std::vector<char> mask_;
};

}  // namespace fv
