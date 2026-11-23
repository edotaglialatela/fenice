// Additional physics: cell gradients, Navier-Stokes viscous residual, body
// forces and the immersed boundary method (ports of gradientCalculation.py,
// addPhys.py, sourceTerms.py, IBM.py).
#pragma once

#include <array>
#include <functional>
#include <utility>
#include <unordered_set>
#include <vector>

#include "fv/Core.hpp"
#include "fv/Geometry.hpp"

namespace fv {

class CellMesh;

// Contribution to the right-hand side of dU/dt = R.  A term is called once per mesh/range; its implementation owns the cell loop,
// so adding new source types does not add virtual dispatch inside a face or cell kernel.
struct RhsSourceTerm {
    using UniformApply = std::function<void(const Mesh&, const Field&, Field&, double stage_time)>;
    using AdaptiveApply =
        std::function<void(const CellMesh&, const Field&, Field&, int begin, int end, double stage_time, const Field* cached)>;
    UniformApply uniform;
    AdaptiveApply adaptive;
};

class RhsSourceTerms {
public:
    void add(RhsSourceTerm term) { terms_.push_back(std::move(term)); }
    void clear() { terms_.clear(); }
    bool empty() const { return terms_.empty(); }

    void apply_uniform(const Mesh& mesh, const Field& U, Field& R, double stage_time) const {
        for (const RhsSourceTerm& term : terms_)
            if (term.uniform) term.uniform(mesh, U, R, stage_time);
    }
    void apply_adaptive(const CellMesh& mesh, const Field& U, Field& R, int begin, int end, double stage_time,
                        const Field* cached = nullptr) const {
        for (const RhsSourceTerm& term : terms_)
            if (term.adaptive) term.adaptive(mesh, U, R, begin, end, stage_time, cached);
    }

private:
    std::vector<RhsSourceTerm> terms_;
};

// Central-difference cell-centre gradients from a (ny+2)*(nx+2) padded state.
void compute_gradients(const Field& U_pad, int nx, int ny, double dx, double dy, Field& grad_x, Field& grad_y);

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
    // step: the maximum over cells of |u| + c + 2 L D sum 1/h^2, with D = max(4/3 nu, kappa / (rho c_v))
    // the diffusivity of the viscous stencil's fastest mode. The solver takes S = max(S_convective, this), so the convective and the viscous eigenvalues
    // share one stability budget. U_pad is the BC-aware padded state. `accumulate`: R_visc already holds the other terms of
    // the residual (one entry per cell) and the viscous residual is added to it instead of replacing it.
    double viscous_residual(const Field& U, const Field& grad_x, const Field& grad_y, const Mesh& mesh,
                            const Field& U_pad, Field& R_visc, const CrossGhosts* cross = nullptr,
                            bool accumulate = false) const;
    // The same residual and wave speed as viscous_residual, bit for bit, without the gradient field: the cell quantities
    // are computed row by row in cache as the faces need them, from the cell field U and the ghost ring `frame` alone
    // (BoundaryManager::get_padded_frame): the padded state and its primitive copy are never built, a cell's primitive
    // state is converted with its row. The 2-D solver uses this one; the gradient version is the reference of the tests.
    double viscous_residual_rows(const Field& U, const PaddedFrame& frame, const Mesh& mesh, Field& R_visc,
                                 const CrossGhosts* cross = nullptr, bool accumulate = false) const;
    // 3-D version: full stress tensor tau_ij = mu (du_i/dx_j + du_j/dx_i) - 2/3 mu delta_ij div u. `length`
    // is the time-step length L of the solver (0: 2 / sum(1/h)). `U_pad` is the (nz+2)*(ny+2)*(nx+2) padded state; the
    // cell-centre gradients are the central differences of it, computed in the sweep.
    double viscous_residual_3d(const Field& U, const Mesh& mesh, const Field& U_pad, Field& R_visc,
                               const CrossGhosts* cross = nullptr, double length = 0.0, bool accumulate = false) const;

    // Second-order physical-boundary faces (boundary_order = 2): the cross derivatives on a face on a physical
    // boundary are the tangential differences of the face values (ghost and cell average) instead of the
    // first-cell value, which is O(h) away from the wall.
    void set_second_order_boundary(bool on) { bface2_ = on; }

    // The 3-D sweep by planes (default on): the fluxes of the faces of a plane at a time, balanced at once, instead of one pass over
    // the whole mesh per axis. The same bits; off gives the passes (the reference of the tests, FENICE_NO_FUSED = 1).
    void set_fused_sweep(bool on) { fused3d_ = on; }

    // Dynamic viscosity.
    double mu() const { return mu_; }
    // Prandtl number.
    double prandtl() const { return Pr_; }

private:
    EOS eos_;
    double mu_, Pr_;
    bool bface2_ = false;
    bool fused3d_ = true;
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

// Immersed body with a sharp ghost-cell wall: a cell is solid when its centre lies inside the body, and the solid cells the fluid
// stencils read are ghost cells that carry the wall condition of the true surface (enable_ghost_cells).
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
    // From a precomputed mask (an MPI strip cut from the global mask): the solid cells only, for the loads; no wall.
    ImmersedBoundary(int nx, int ny, std::vector<char> mask, double rho_ref, double p_ref, int nz = 1)
        : nx_(nx), ny_(ny), nz_(nz), rho_ref_(rho_ref), p_ref_(p_ref), mask_(std::move(mask)) {}

    // Inside every residual (safe in RK stages): the ghost cells from the fluid, the deeper solid cells stopped. After
    // enable_ghost_cells only.
    void apply_forcing(Field& U, const EOS& eos) const;
    // apply_forcing on U itself, after saving the states of the cells it writes (ghost cells and deeper solid cells) in `saved`;
    // restore_forced puts them back. The same bits as forcing a copy of U, without the copy.
    void force_in_place(Field& U, const EOS& eos, std::vector<State>& saved) const;
    void restore_forced(Field& U, const std::vector<State>& saved) const;
    // Once per complete step: the ghost cells from the fluid, the deeper solid cells reset to the reference state (energy from the EOS).
    void apply_solid_reset(Field& U, const EOS& eos) const;

    // Reference density / pressure the deep solid cells are reset to.
    double rho_ref() const { return rho_ref_; }
    double p_ref() const { return p_ref_; }

    // Number of solid cells.
    int solid_count() const;
    // Solid mask, one entry per cell (1 = solid).
    const std::vector<char>& mask() const { return mask_; }

    // Sharp-interface (ghost-cell) wall, for a body built from its geometry (cylinder, sphere or polygon). The solid cells that
    // the fluid stencils read (a fluid cell within two cells) become ghost cells: each gets the state mirrored across the true
    // wall from an image point in the fluid (bilinear in the fluid cells around it): density and pressure with zero normal
    // gradient, the velocity reflected (no-slip) or only its normal component (slip), so the wall condition holds on the
    // body's surface, not on a staircase. apply_forcing fills the ghost cells (and stops the deeper solid cells)
    // and apply_solid_reset does the same plus the deep-solid reset (energy from the EOS). Throws for a mask-only body.
    // A ghost cell with no fluid around any image point takes the nearest fluid cell (ghost_fallback_count).
    // 3-D: a sphere (trilinear image points over eight cells), or the infinite cylinder / extruded section (the 2-D image point of the
    // ghost cell's own layer, so each layer equals the 2-D run bit for bit); no surface samples (surface_loads is zero), the monitor loads
    // come from the momentum flux into the solid cells; ibm_curvature is 2-D only.
    // curvature (slip wall only): the ghost pressure follows the normal momentum equation at the wall, dp/dn = rho kappa u_t^2 (kappa:
    // curvature of the wall, positive for a convex body), and the density follows constant entropy along the normal, instead of a zero
    // normal gradient of both; without it a curved wall carries a first-order pressure error.
    void enable_ghost_cells(const Mesh& mesh, bool slip, bool curvature = false);
    // The same wall on a mesh `fine` that is never stored (the finest level of an adaptive mesh): solidity is solid_at at the cell
    // centres (the mask by construction) and only the `candidates` are visited (cell indices of `fine` in increasing order, e.g. the
    // solid finest leaves); where they include every ghost cell of `fine` the table is that of enable_ghost_cells(fine) bit for bit.
    // No array over `fine`; for a body built from its geometry; `fine` must have fewer than 2^31 cells.
    void enable_ghost_cells_sparse(const Mesh& fine, bool slip, bool curvature, const std::vector<int>& candidates);
    // True after enable_ghost_cells.
    bool ghost_cells() const { return ghost_on_; }
    // Number of ghost cells.
    int ghost_count() const { return static_cast<int>(ghost_.size()); }
    // Ghost cells with no fluid cell around any image point (a slot, or a part of the body thinner than the stencil): they take the
    // density and pressure of the nearest fluid cell and stay at rest (no-slip) or keep its tangential velocity (slip).
    int ghost_fallback_count() const { return fallback_; }
    // Ghost-cell wall: force (x, y) and moment about (cx, cy), per unit span, on the body, integrated over its true surface from
    // two probes at d and 2 d (d >= 1.5 cells) along the normal: wall pressure extrapolated linearly (2 p1 - p2) and, with
    // mu > 0, the wall shear mu du_t/dn = mu (4 u_t1 - u_t2) / (2 d) of the quadratic through zero at the wall (no-slip).
    // For surface distributions; the monitor's loads are the momentum flux into the solid cells (Simulation::ibm_loads), which
    // this integral underestimates by 1-3 % (cylinder at Re = 40, 20 cells per diameter; 1 % at 40), the truncation error of the
    // wall-shear fit, which converges with the mesh.
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
    // Per ghost cell (same order): distance ghost-wall over distance image-wall (the ghost state scales the image state by minus this).
    std::vector<double> ghost_ratios() const;
    std::vector<int> referenced_cells() const;
    // Bytes of the mask and the ghost-cell tables.
    double memory_bytes() const;
    // Largest distance, in cells (max norm), from the solid cells to a cell the ghost fill or the surface loads read, plus one:
    // the width a band of uniform cells around the body needs (AMR: the finest-level band).
    int ghost_reach() const;
    // Surface loads of the surface samples k with k % parts == part (sum the parts for the total), and the cells they read.
    std::array<double, 3> surface_loads_part(int part, int parts, const CellState& in, const EOS& eos, double mu, double cx,
                                             double cy) const;
    std::vector<int> surface_inputs(int part, int parts) const;
    // Closest point of the body surface to (x, y) and the outward unit normal there (2-D; geometry constructors only).
    void closest_wall_point(double x, double y, double& wx, double& wy, double& nx, double& ny) const;
    // The same in 3-D: a sphere, or a cylinder / section extruded along z (then wz = z and the normal has no z-component).
    void closest_wall_point3(double x, double y, double z, double& wx, double& wy, double& wz, double& nx, double& ny, double& nz) const;
    // True for a body built from its geometry (a shape or a polygon), false for a mask-only one.
    bool has_geometry() const { return shape_ != Shape::None; }
    // Bounding box of a 2-D body (circle or polygon; false for the others).
    bool bounds(double& x0, double& x1, double& y0, double& y1) const;
    // The solid cells of ine (ascending index) found inside the bounding box, with no array over the mesh: the candidates of
    // enable_ghost_cells_sparse for a 2-D body.
    std::vector<int> solid_cells_near(const Mesh& fine) const;

    // Pose of a moving body relative to the geometry it was built with: translation (x, y, z), rotation `angle` [rad, counter-clockwise
    // in the x-y plane] about the pivot given to enable_motion, and the velocities of the pivot (u, v, w) and the angular velocity omega.
    struct BodyPose {
        double x = 0.0, y = 0.0, z = 0.0;
        double angle = 0.0;
        double u = 0.0, v = 0.0, w = 0.0;
        double omega = 0.0;
    };
    // Switch the wall to a moving one (after enable_ghost_cells; uniform meshes, geometry bodies, 2-D or a sphere). The geometry at
    // this moment is the reference; (px, py) is the pivot of the rotation. Until this call nothing changes: a static body runs the
    // static code and its bits are unchanged.
    void enable_motion(double px, double py);
    // True after enable_motion.
    bool moving() const { return moving_; }
    // Place the body at `pose`. The geometry (mask, ghost table) is rebuilt only when the body has moved by more than a quarter cell
    // since the last build (translation, or rotation times the largest distance from the pivot); in between the wall stays where it was
    // built and only the velocities (of the pose) are refreshed. The ghost cells carry the wall velocity of the pose, rigid-body motion
    // at the closest wall point; the deeper solid cells carry the body velocity at the reference state. Returns true if it rebuilt.
    // The cells the body left (fresh cells) hold stale solid states until fill_fresh.
    bool move_to(const Mesh& mesh, const BodyPose& pose);
    // Change the translational velocity of the pose without moving the geometry: the wall velocities of the ghost table follow (a body driven by the
    // flow on an adaptive mesh, whose table stays between geometry updates).
    void set_velocity(double u, double v);
    // Place the body (the mask only, no ghost table, no fresh cells) at the pose: for a body made by a factory at the pose of the moment, before
    // its ghost wall is built (the adaptive mesh); the ghost cells then carry the wall velocity of the pose.
    void set_pose(const Mesh& mesh, const BodyPose& pose);
    // After a rebuilding move_to: the fresh cells take the state of the wall layer from an image point in the settled fluid (density and
    // pressure with zero normal gradient, the velocity linear from the wall velocity at the wall to the image velocity). Call before any
    // ghost fill; reads fluid cells only.
    void fill_fresh(Field& U, const EOS& eos);
    // The same on a part of the mesh (MPI strips; local, in as in fill_ghosts_part): fills the fresh cells the part owns, with the bits of fill_fresh,
    // and forgets the fresh cells. fresh_inputs: the global cells they read (the part's own are among them).
    void fill_fresh_part(const CellIndex& local, const CellState& in, Field& U_part, const EOS& eos);
    std::vector<int> fresh_inputs(const CellIndex& local) const;
    // The fresh cells of a sparse table (the adaptive mesh's finest level), after enable_ghost_cells_sparse of the new pose: the cells that were solid
    // in `previous` (the same body at its last pose) and are fluid now, with their image points in settled fluid. fill_fresh_part fills them.
    void make_fresh_sparse(const ImmersedBoundary& previous, const Mesh& fine);
    // (fresh cell, cell its fill reads) pairs, the cell itself included (the leaves a share must hold; not part of the wall's readiness).
    std::vector<std::pair<int, int>> fresh_reads() const;
    // Number of fresh cells of the last rebuild, and how many of them had no fluid around any image point (they copy the nearest settled
    // fluid cell at the wall velocity).
    int fresh_count() const { return static_cast<int>(fresh_.size()); }
    int fresh_fallback_count() const { return fresh_fallback_; }
    // The pose of the last move_to (the identity before it).
    const BodyPose& pose() const { return pose_; }
    // Largest distance of the body from its pivot (a rotation by an angle moves the body by at most angle times this; 0 for a circle).
    double rotation_reach() const { return reach_; }
    // The test the mask is made of: whether a cell centred at (x, y, z) is solid (z is read by the sphere only); the mask of
    // any mesh is this test at its cell centres (the adaptive mesh classifies its leaves with it). Throws for a mask-only body.
    bool solid_at(double x, double y, double z) const;

private:
    int nx_ = 0, ny_ = 0, nz_ = 1;
    double rho_ref_ = 1.0, p_ref_ = 1.0;
    std::vector<char> mask_;
    // geometry (constructors from a shape or a polygon)
    enum class Shape { None, Circle, Ball, Polygon } shape_ = Shape::None;
    double cx_ = 0.0, cy_ = 0.0, cz_ = 0.0, r_ = 0.0;
    Polygon2D poly_;
    double px0_ = 0.0, px1_ = 0.0, py0_ = 0.0, py1_ = 0.0;   // bounding box of the polygon (the far cells skip the inside test)
    // ghost-cell wall: interpolation of the primitive state at a point from up to four fluid cells
    struct Probe {
        int cell[8] = {0, 0, 0, 0, 0, 0, 0, 0};       // 2-D (and bodies extruded along z): the first four, bilinear in the layer
        double w[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // sphere: all eight, trilinear
        State at(const CellState& in, const EOS& eos) const;
    };
    struct Ghost {
        int cell = 0;
        double ratio = 0.0;   // distance ghost-wall / distance image-wall (Dirichlet values scale by -ratio)
        double nx = 0.0, ny = 0.0, nz = 0.0;
        double kappa = 0.0, span = 0.0;   // wall curvature (convex > 0) and distance image-ghost along the normal
        double wu = 0.0, wv = 0.0, ww = 0.0;   // wall velocity at the closest wall point (moving body only)
        double wx = 0.0, wy = 0.0, wz = 0.0;   // the closest wall point
        Probe image;
    };
    struct FreshCell {
        int cell = 0;
        double ratio = 0.0;   // distance cell-wall / distance image-wall
        double wu = 0.0, wv = 0.0;
        Probe image;
    };
    std::vector<FreshCell> fresh_;
    std::unordered_set<int> fresh_solid_;   // sparse table: the fresh cells count as solid while their probes are made
    void build_fresh(const Mesh& m, const std::vector<int>& cells);   // fresh_ for these cells (they must be fluid now)
    int fresh_fallback_ = 0;
    BodyPose built_;                 // the pose of the last geometry build
    double reach_ = 0.0;             // largest distance of the body from the pivot (rotation moves the polygon by angle * reach_)
    void refresh_wall_velocity();
    void place_geometry(const Mesh& m, const BodyPose& pose);   // the shape and the mask at the pose    // the ghost wall velocities from the current pose on the built geometry
    struct SurfacePoint {
        double x = 0.0, y = 0.0, nx = 0.0, ny = 0.0, ds = 0.0, d = 0.0;   // probes at d and 2 d along the normal
        Probe probe, probe2;
    };
    // motion (enable_motion / move_to): the reference geometry, the pivot, the current pose and the mesh of the last placement
    bool moving_ = false;
    double ref_cx_ = 0.0, ref_cy_ = 0.0, ref_cz_ = 0.0, pivot_x_ = 0.0, pivot_y_ = 0.0;
    Polygon2D ref_poly_;
    BodyPose pose_;
    Mesh mesh_;
    void body_velocity(double x, double y, double z, double& bu, double& bv, double& bw) const;   // rigid velocity of the pose at a point
    bool ghost_on_ = false, slip_ = false, curvature_ = false;
    int fallback_ = 0;      // ghost cells without an image point (ghost_fallback_count)
    // sparse ghost table of a polygon: the solidity of the cells around the body, tested once (the point-in-polygon test costs the number of edges)
    std::vector<char> sbox_;
    int sb_i0_ = 0, sb_j0_ = 0, sb_w_ = 0, sb_h_ = 0;
    bool sbox_on_ = false;
    bool sparse_ = false;   // the ghost wall of enable_ghost_cells_sparse: solidity from solid_at, no arrays over the mesh
    bool is_solid(const Mesh& m, int i, int j, int k) const;   // mask_ (dense) or solid_at of the centre (sparse)
    void build_ghosts(const Mesh& m, bool slip, bool curvature, const std::vector<int>* candidates);
    std::vector<Ghost> ghost_;
    std::vector<SurfacePoint> surface_;
    std::vector<char> ghost_mask_;
    mutable std::vector<int> forced_cells_;   // the cells apply_forcing writes with the ghost wall (built on first use)
    mutable bool forced_built_ = false;
    bool make_probe(const Mesh& m, double x, double y, Probe& pr, int layer = 0) const;   // bilinear in the layer `layer`
    bool make_probe3(const Mesh& m, double x, double y, double z, Probe& pr) const;       // trilinear (sphere)
    double wall_curvature(double wx, double wy, double nx, double ny, double w) const;   // signed, convex > 0   // false: no fluid cell around (x, y)
};

}   // namespace fv
