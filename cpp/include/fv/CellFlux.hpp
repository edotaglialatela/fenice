// Convective residual of a CellMesh: one Riemann flux per face, gathered by the cells.
//
// The stencil of every face is fixed when the mesh changes (build), so the residual reads index arrays only.
// A stencil is (c0, c1, c2, c3) along the face normal: c1 / c2 are the leaves left / right of the face, c0 / c3
// the cells beyond them.  A cell beyond that is coarser is read by injection (its value), a cell beyond that is
// finer by the mean of the leaves under it (a "virtual cell", built as ((a+b)+(c+d))*0.25, recursively), the
// edge of the domain by replication of c1 / c2 and a periodic wrap by the cell on the other side.  At a
// coarse-fine face the fine side is reconstructed against the injected coarse value and the coarse side keeps
// its cell value (first order), so a uniform flow is preserved exactly across level jumps.
// With level-0 leaves only the residual is bit-identical to Simulation::assemble.
#pragma once

#include <array>
#include <vector>

#include "fv/Boundaries.hpp"
#include "fv/CellMesh.hpp"
#include "fv/Schemes.hpp"

namespace fv {

// Stencils, boundary closure and face fluxes of a CellMesh; see the file header.
class CellFluxEngine {
public:
    // `muscl` = nullptr: first-order Godunov.  `bc` supplies the boundary types, their parameters and the ghost
    // states (all types but non-reflecting, which build() rejects).  The referenced objects must outlive the engine.
    CellFluxEngine(const EOS& eos, const RiemannSolver& riemann, const MusclReconstruction* muscl,
                   const BoundaryManager& bc)
        : eos_(eos), riemann_(riemann), muscl_(muscl), bc_(bc) {}

    // Stencils and virtual cells of `mesh`; call again after every CellMesh::adapt.  Throws if the periodic
    // flags of the mesh and the boundary types disagree.
    void build(const CellMesh& mesh);

    // R = -div F for the state U (one entry per leaf); returns the largest wave speed of any face.
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

    // Number of virtual cells (coarse averages over finer neighbours) of the current stencils.
    int num_virtual() const { return static_cast<int>(vroot_.size()); }

private:
    State eval(const Field& U, int code) const;
    void refresh_all(const Field& U, int N);
    void refresh_some(const Field& U, int N, const std::vector<int>& idx);
    void mass_flow_targets(const CellMesh& m, const Field& U, double* u_target) const;
    double fluxes(const CellMesh& m, const Field& U, int fb, int fe, const double* u_target);
    void gather(const CellMesh& m, int cb, int ce, bool replace, Field& R);

    EOS eos_;
    const RiemannSolver& riemann_;
    const MusclReconstruction* muscl_;
    const BoundaryManager& bc_;

    int nleaf_ = 0;
    // Per face: indices into the primitive array W_ (a leaf, or nleaf_ + virtual cell); unused for boundary faces
    std::vector<std::array<int, 4>> stencil_;
    // Virtual cells: root code (>= 0 a leaf, < 0 the node -1 - code) and the nodes (four children codes)
    std::vector<int> vroot_;
    std::vector<std::array<int, 4>> vnode_;
    Field W_, F_, reg_;   // primitive state of leaves + virtual cells; face fluxes; coarse-fine time-averaged fluxes
    std::vector<std::vector<int>> need_;   // per level: the entries of W_ read by the stencils of its interior faces
    // Mass-flow inlet: boundary faces of each side in increasing transverse position, and their weights
    // (1 << (finest level of the side - face level)), so a uniform side sums exactly like the uniform solver
    std::vector<int> mf_face_[4];
    std::vector<double> mf_weight_[4];
};

}   // namespace fv
