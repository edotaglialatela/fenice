// Additional physics: cell gradients, Navier-Stokes viscous residual, body
// forces and the immersed boundary method (ports of gradientCalculation.py,
// addPhys.py, sourceTerms.py, IBM.py).
#pragma once

#include <array>
#include <functional>

#include "fv/Core.hpp"
#include "fv/Geometry.hpp"

namespace fv {

// Central-difference cell-centre gradients from a (ny+2)*(nx+2) padded state.
void compute_gradients(const Field& U_pad, int nx, int ny, double dx, double dy, Field& grad_x, Field& grad_y);
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

    // Writes the viscous residual into R_visc and returns the wave speed that bounds the explicit time
    // step: the maximum over cells of |u| + c + L / dt_visc, with dt_visc the parabolic limit of the
    // cell. The solver takes S = max(S_convective, this), so the convective and the viscous eigenvalues
    // share one stability budget. U_pad is the BC-aware padded state.
    double viscous_residual(const Field& U, const Field& grad_x, const Field& grad_y, const Mesh& mesh,
                            const Field& U_pad, Field& R_visc, const CrossGhosts* cross = nullptr) const;
    // 3-D version: full stress tensor tau_ij = mu (du_i/dx_j + du_j/dx_i) - 2/3 mu delta_ij div u. `length`
    // is the time-step length L of the solver (0: 2 / sum(1/h)).
    double viscous_residual_3d(const Field& U, const Field& grad_x, const Field& grad_y, const Field& grad_z,
                               const Mesh& mesh, const Field& U_pad, Field& R_visc,
                               const CrossGhosts* cross = nullptr, double length = 0.0) const;

    // Dynamic viscosity.
    double mu() const { return mu_; }
    // Prandtl number.
    double prandtl() const { return Pr_; }

private:
    EOS eos_;
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
    char dir = 'y';                               // decomposition direction
    const Field* lo = nullptr;                    // neighbour states, low side
    const Field* hi = nullptr;                    // neighbour states, high side
    const std::vector<char>* mask_lo = nullptr;   // solid mask of those neighbours
    const std::vector<char>* mask_hi = nullptr;
};

// Direct-forcing immersed boundary with a circular obstacle.
class ImmersedBoundary {
public:
    ImmersedBoundary() = default;
    // Obstacle of `radius` centred at (cx, cy[, cz]); deep solid cells are frozen to
    // (rho_ref, p_ref).  2-D: "Cylinder" and "Sphere" are both a circle.  3-D: "Sphere"
    // is a sphere, "Cylinder" an infinite cylinder along z.  Case-insensitive; other shapes throw.
    ImmersedBoundary(const Mesh& mesh, const std::string& shape, double radius, double cx, double cy, double rho_ref,
                     double p_ref, double cz = 0.0);
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

    // Reference density / pressure the deep solid cells are reset to.
    double rho_ref() const { return rho_ref_; }
    double p_ref() const { return p_ref_; }

    // Number of solid cells.
    int solid_count() const;
    // Solid mask, one entry per cell (1 = solid).
    const std::vector<char>& mask() const { return mask_; }

    // Sharp-interface (ghost-cell) wall, 2-D, for a body built from its geometry (cylinder or polygon). The solid cells that
    // the fluid stencils read (a fluid cell within two cells) become ghost cells: each gets the state mirrored across the true
    // wall from an image point in the fluid (bilinear in the fluid cells around it): density and pressure with zero normal
    // gradient, the velocity reflected (no-slip) or only its normal component (slip), so the wall condition holds on the
    // body's surface, not on a staircase. From then on apply_forcing fills the ghost cells (and stops the deeper solid cells)
    // and apply_solid_reset does the same plus the deep-solid reset (energy from the EOS). Throws for a mask-only body or 3-D.
    void enable_ghost_cells(const Mesh& mesh, bool slip);
    // True after enable_ghost_cells.
    bool ghost_cells() const { return ghost_on_; }
    // Number of ghost cells.
    int ghost_count() const { return static_cast<int>(ghost_.size()); }
    // Ghost-cell wall: force (x, y) and moment about (cx, cy), per unit span, on the body, integrated over its true surface from
    // two probes at d and 2 d (d >= 1.5 cells) along the normal: wall pressure extrapolated linearly (2 p1 - p2) and, with
    // mu > 0, the wall shear mu du_t/dn = mu (4 u_t1 - u_t2) / (2 d) of the quadratic through zero at the wall (no-slip).
    std::array<double, 3> surface_loads(const Field& U, const EOS& eos, double mu, double cx, double cy) const;
    // Ghost-cell wall on a part of the mesh (MPI strips), on an object built for the whole mesh. local(c): index of global
    // cell c in the part's field, -1 outside it; in(c): input state of global cell c (any cell an owned ghost reads, see
    // ghost_inputs). Fills the ghost cells of the part and stops (reset = false) or resets (reset = true) its deeper solid
    // cells, as apply_forcing / apply_solid_reset do on the whole mesh, with the same bits.
    using CellIndex = std::function<int(int)>;
    using CellState = std::function<const State&(int)>;
    void fill_ghosts_part(const CellIndex& local, const CellState& in, Field& U_part, const EOS& eos, bool reset) const;
    // Only the ghost cells of fill_ghosts_part (the deeper solid cells untouched).
    void fill_ghost_cells(const CellIndex& local, const CellState& in, Field& U_part, const EOS& eos) const;
    // The global cells the ghost cells of the part read.
    std::vector<int> ghost_inputs(const CellIndex& local) const;
    // (ghost cell, cell it reads) pairs, the ghost cells, and every cell the ghost fill and the surface loads read.
    std::vector<std::pair<int, int>> ghost_reads() const;
    std::vector<int> ghost_cell_list() const;
    std::vector<int> referenced_cells() const;
    // Largest distance, in cells (max norm), from the solid cells to a cell the ghost fill or the surface loads read, plus one:
    // the width a band of uniform cells around the body needs (AMR: the finest-level band).
    int ghost_reach() const;
    // Surface loads of the surface samples k with k % parts == part (sum the parts for the total), and the cells they read.
    std::array<double, 3> surface_loads_part(int part, int parts, const CellState& in, const EOS& eos, double mu, double cx,
                                             double cy) const;
    std::vector<int> surface_inputs(int part, int parts) const;
    // Closest point of the body surface to (x, y) and the outward unit normal there (2-D; geometry constructors only).
    void closest_wall_point(double x, double y, double& wx, double& wy, double& nx, double& ny) const;

private:
    int nx_ = 0, ny_ = 0, nz_ = 1;
    double rho_ref_ = 1.0, p_ref_ = 1.0;
    std::vector<char> mask_;
    // geometry (constructors from a shape or a polygon)
    enum class Shape { None, Circle, Ball, Polygon } shape_ = Shape::None;
    double cx_ = 0.0, cy_ = 0.0, cz_ = 0.0, r_ = 0.0;
    Polygon2D poly_;
    // ghost-cell wall: interpolation of the primitive state at a point from up to four fluid cells
    struct Probe {
        int cell[4] = {0, 0, 0, 0};
        double w[4] = {0.0, 0.0, 0.0, 0.0};
        State at(const CellState& in, const EOS& eos) const;
    };
    struct Ghost {
        int cell = 0;
        double ratio = 0.0;   // distance ghost-wall / distance image-wall (Dirichlet values scale by -ratio)
        double nx = 0.0, ny = 0.0;
        Probe image;
    };
    struct SurfacePoint {
        double x = 0.0, y = 0.0, nx = 0.0, ny = 0.0, ds = 0.0, d = 0.0;   // probes at d and 2 d along the normal
        Probe probe, probe2;
    };
    bool ghost_on_ = false, slip_ = false;
    std::vector<Ghost> ghost_;
    std::vector<SurfacePoint> surface_;
    std::vector<char> ghost_mask_;
    bool make_probe(const Mesh& m, double x, double y, Probe& pr) const;   // false: no fluid cell around (x, y)
};

}   // namespace fv
