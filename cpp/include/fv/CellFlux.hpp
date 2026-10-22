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

    // Number of virtual cells (coarse averages over finer neighbours) of the current stencils.
    int num_virtual() const { return static_cast<int>(vroot_.size()); }

private:
    State eval(const Field& U, int code) const;

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
    Field W_, F_;
    // Mass-flow inlet: boundary faces of each side in increasing transverse position, and their weights
    // (1 << (finest level of the side - face level)), so a uniform side sums exactly like the uniform solver
    std::vector<int> mf_face_[4];
    std::vector<double> mf_weight_[4];
};

}   // namespace fv
