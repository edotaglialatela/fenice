// Residual of a CellMesh: one flux per face (convective and viscous), gathered by the cells.
//
// The stencil of every face is fixed when the mesh changes (build), so the residual reads index arrays only.
// A stencil is (c0, c1, c2, c3) along the face normal: c1 / c2 are the leaves left / right of the face, c0 / c3
// the cells beyond them.  A cell beyond that is coarser is read by injection (its value), a cell beyond that is
// finer by the mean of the leaves under it (a "virtual cell", built as ((a+b)+(c+d))*0.25, recursively), the
// edge of the domain by replication of c1 / c2 and a periodic wrap by the cell on the other side.  At a
// coarse-fine face the fine side is reconstructed against the injected coarse value and the coarse side keeps
// its cell value (first order), so a uniform flow is preserved exactly across level jumps.
// The viscous terms use the same neighbour codes: the cell-centre gradients (central differences of the four
// neighbouring values, virtual cells across a jump, the diffusive ghost at a physical edge) feed the cross
// derivatives, the normal derivative of a face is the two-point difference over the distance of its two cell centres.
// With level-0 leaves only the residual is bit-identical to Simulation::assemble.
#pragma once

#include <array>
#include <functional>
#include <vector>

#include "fv/Boundaries.hpp"
#include "fv/CaseFile.hpp"
#include "fv/CellMesh.hpp"
#include "fv/Physics.hpp"
#include "fv/Schemes.hpp"

namespace fv {

// Optional physics of the residual (all pointers / functions must outlive the engine).
struct CellPhysics {
    const NavierStokes* ns = nullptr;          // viscous terms (needs the cell sizes of the leaves)
    const ConstantBodyForce* body = nullptr;   // body force
    const Params* source = nullptr;            // source_* expressions (null: none)
    bool source_unsteady = false;              // the expressions use t: evaluated at CellFluxEngine::stage_time
    const Mesh* source_mesh = nullptr;         // base mesh the expression variables (Lx, nx, ...) come from
    // Immersed body: the mask of a uniform mesh; masks per level are cut from it (empty: no body)
    std::function<ImmersedBoundary(const Mesh&)> ibm_factory;
};

// Stencils, boundary closure and face fluxes of a CellMesh; see the file header.
class CellFluxEngine {
public:
    // `muscl` = nullptr: first-order Godunov.  `bc` supplies the boundary types, their parameters and the ghost
    // states.  The referenced objects must outlive the engine.
    CellFluxEngine(const EOS& eos, const RiemannSolver& riemann, const MusclReconstruction* muscl,
                   const BoundaryManager& bc)
        : eos_(eos), riemann_(riemann), muscl_(muscl), bc_(bc) {}

    // Viscous terms, forces, sources and the immersed body; call before build().
    void set_physics(const CellPhysics& p) { phys_ = p; }

    // Stencils, virtual cells, solid masks and steady sources of `mesh`; call again after every CellMesh::adapt.
    // Throws if the periodic flags of the mesh and the boundary types disagree.
    void build(const CellMesh& mesh);

    // R = dU/dt for the state U (one entry per leaf); returns the largest wave speed (convective, or the viscous
    // bound of the explicit time step when larger).
    double residual(const CellMesh& mesh, const Field& U, Field& R);

    // --- per-level pieces for the subcycled advance (CellAdvance) ---
    // Fluxes of the faces of level l (all of them: interior, coarse-fine whose fine side is level l, boundary), read
    // from U, the state of every leaf (the other levels as they are); returns their largest wave speed.
    double compute_level_fluxes(const CellMesh& mesh, const Field& U, int l);
    // R of the leaves of level l (Rl[k] is leaf level_begin[l] + k).  A face of level l + 1 on a side of such a leaf
    // is a coarse-fine face seen from its coarse side: with `replace` it contributes the time-averaged flux of the
    // register, without it the flux of the last compute_level_fluxes (test hook: breaks conservation).
    void gather_level(const CellMesh& mesh, int l, bool replace, Field& Rl);
    // Zero the registers of the faces of level l.
    void clear_registers(const CellMesh& mesh, int l);
    // reg += w * F for the coarse-fine faces of level l (call after compute_level_fluxes of that level).
    void accumulate_registers(const CellMesh& mesh, int l, double w);

    // --- immersed body (no-ops without one) ---
    // Take the body forcing of the whole state U (velocity of the solid cells zeroed); the level pieces read the
    // forced copy, so call it before the first level pass and again for a level whose state was just updated.
    void force_state(const CellMesh& mesh, const Field& U);
    void force_level(const CellMesh& mesh, const Field& U, int l);
    // Once per complete step: fringe Neumann condition and deep-solid reset of the solid leaves, in place.
    void solid_reset(const CellMesh& mesh, Field& U) const;
    // True with an immersed body.
    bool has_body() const { return static_cast<bool>(phys_.ibm_factory); }
    // Solid flag of every leaf (valid after build; empty without a body).
    const std::vector<char>& solid() const { return solid_; }

    // Time the unsteady sources are evaluated at (CellAdvance sets it from the scheme's stage time).
    double stage_time = 0.0;

    // Number of virtual cells (coarse averages over finer neighbours, and the interpolated coarse values) of the stencils.
    int num_virtual() const { return static_cast<int>(vroot_.size() + interp_.size()); }

    // With MUSCL: the cell a fine leaf sees across a side towards a coarser leaf is the coarse value interpolated (limited
    // slopes of the coarse leaf, in primitive variables) to the centre of the fine cell that would sit there, instead of the
    // injected coarse value. False (set before build) restores the injection (the test control; the first-order scheme
    // always injects).
    bool interpolate_jumps = true;
    // The coarse side of a coarse-fine face is reconstructed too (set before build; default on): the standard limited MUSCL
    // formula on the coarse leaf, its neighbour across the face being the mean of the fine leaves there, so both sides of a
    // jump are second order. False keeps the coarse leaf's cell state there (first order, the test control).
    bool coarse_side_second_order = true;
    // One interpolated virtual cell: the fine leaf, the side of it facing the coarser leaf, and the coarse leaf.
    struct InterpVirtual {
        int fine = -1, coarse = -1;
        Face side = Face::Left;
        double dn = 0.0, dt = 0.0;   // offsets of the virtual centre from the coarse centre, in coarse widths, along the
        int axis = 0;                // normal axis `axis` (of `side`) and the transverse one
    };
    const std::vector<InterpVirtual>& interpolated() const { return interp_; }
    // Primitive state of interpolated virtual cell k as of the last residual / level pass.
    const State& interpolated_state(int k) const { return Wint_[static_cast<size_t>(k)]; }

private:
    State eval(const Field& U, int code) const;
    State cell_state(int code) const;   // conservative state of a leaf or virtual cell (of the forced state)
    void prepare(const CellMesh& m, const Field& U, int level);   // level < 0: everything
    void refresh_all(int N);
    void refresh_some(int N, const std::vector<int>& idx);
    void mass_flow_targets(const CellMesh& m, double* u_target) const;
    double fluxes(const CellMesh& m, int fb, int fe, const double* u_target);
    void cell_quantities(const CellMesh& m, const std::vector<int>& cells);
    void viscous_fluxes(const CellMesh& m, int fb, int fe);
    double viscous_speed(const CellMesh& m, int cb, int ce) const;
    void gather(const CellMesh& m, int cb, int ce, bool replace, Field& R);
    void update_source(const CellMesh& m, int cb, int ce);

    EOS eos_;
    const RiemannSolver& riemann_;
    const MusclReconstruction* muscl_;
    const BoundaryManager& bc_;
    CellPhysics phys_;

    int nleaf_ = 0;
    // Per face: indices into the primitive array W_ (a leaf, or nleaf_ + virtual cell); unused for boundary faces
    std::vector<std::array<int, 4>> stencil_;
    // Per leaf and side: the neighbour across it as a leaf index, nleaf_ + virtual cell, or -1 at a physical edge
    std::vector<int> nbr_;
    // Virtual cells: root code (>= 0 a leaf, < 0 the node -1 - code) and the nodes (four children codes)
    std::vector<int> vroot_;
    std::vector<std::array<int, 4>> vnode_;
    std::vector<InterpVirtual> interp_;   // interpolated virtual cells, numbered after the mean-type ones
    Field Wint_;                          // their primitive state
    std::vector<int> nbr_raw_;            // like nbr_ but injecting coarser neighbours (the slopes of the coarse leaves)
    void eval_interp();
    const Field* us_ = nullptr;   // state the pieces read: the input, or the forced copy
    Field forced_;                // body-forced state (with a body)
    Field Vc_;                    // conservative state of the virtual cells
    // primitive state of leaves + virtual cells; face fluxes (convective per unit length, viscous times length);
    // coarse-fine time-averaged fluxes (convective, viscous)
    Field W_, F_, Fv_, reg_, regv_;
    std::vector<std::vector<int>> need_;     // per level: the entries of W_ read by the stencils of its interior faces
    std::vector<std::vector<int>> vcells_;   // per level: its leaves and the coarse cells of its coarse-fine faces
    // viscous cell quantities: u, v, T, du/dx, du/dy, dv/dx, dv/dy
    std::vector<double> su_, sv_, sT_, dudx_, dudy_, dvdx_, dvdy_;
    double s_visc_ = 0.0;
    std::vector<char> solid_;
    double ibm_rho_ref_ = 1.0, ibm_p_ref_ = 1.0;   // deep-solid reset state
    std::vector<std::vector<char>> level_mask_;   // cache: mask of the uniform mesh of each level
    Field src_;                                    // source terms per leaf
    // Mass-flow inlet: boundary faces of each side in increasing transverse position, and their weights
    // (1 << (finest level of the side - face level)), so a uniform side sums exactly like the uniform solver
    std::vector<int> mf_face_[4];
    std::vector<double> mf_weight_[4];
};

}   // namespace fv
