// Adaptation of a CellMesh from the solution: the face-variation indicator, the flags it gives, the
// finest-level band around an immersed body, and the remesh / initial adaptation drivers.
//
// Indicator (docs/design/amr-cell-based.md section 11):
// It is evaluated level by level on the cells of each level (the leaves of that level and the split ancestors of the deeper
// leaves, a split cell holding the mean of its children), so a parent has its own xi at the parent level:
//   1. xi = 0 in every cell of the level.
//   2. For every interface of the level (two cells of the level, or a cell and a coarser leaf) and every enabled variable
//      (density, pressure, velocity magnitude), with c_l, c_r the values on its two sides: m = max(min(|c_l|, |c_r|), floor),
//      variation = |c_r - c_l| / m; if variation >= criterion, the cells of the level on both sides get xi = 1.
//   3. `diffusion` iterations: every interface of two cells with one of them above `diffusion_trigger`, or of a cell above it
//      against a coarser leaf, adds `diffusion_coeff` to the cell(s) of the level; after all interfaces xi += added.
//   4. A leaf is refined if its xi >= split; four sibling leaves are joined if the xi of their PARENT cell is < join.
#pragma once

#include <array>
#include <functional>
#include <vector>

#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"

namespace fv {

// Parameters of the indicator.  The defaults are a starting point for the user to fix (section 11).
struct CellIndicator {
    bool use_density = true, use_pressure = false, use_velocity = false;   // variables whose variation is tested
    double criterion = 0.1;          // relative variation across a face that flags both leaves
    double value_floor = 1e-2;       // floor of min(|c_l|, |c_r|); the velocity magnitude uses velocity_floor
    double velocity_floor = 0.1;
    int diffusion = 2;               // iterations of the smoothing
    double diffusion_coeff = 0.1;    // added to both leaves of a face per iteration
    double diffusion_trigger = 0.05; // a face spreads only if a leaf has xi above this
    double split = 0.15;             // refine at xi >= split
    double join = 0.10;              // join siblings when all have xi < join

    // Test switch: false joins four siblings when all four have xi < join at their own level (the rule of the first
    // version, which flickers at the edge of a refined region) instead of testing the parent's xi.
    bool parent_join = true;

    // The smoothing of step 3: interfaces {a, b} between two cells of the level, or {a, -1} between a cell and a coarser leaf.
    void diffuse(std::vector<double>& xi, const std::vector<std::array<int, 2>>& interfaces) const;
    // The same smoothing gathered per cell (and parallel): nbr[4 c + s] is the cell across side s of cell c in the same level,
    // -1 if a coarser leaf lies there, -2 if nothing does. Gives the same bits as diffuse() on the interfaces of that table.
    void diffuse_cells(std::vector<double>& xi, const std::vector<int>& nbr) const;
    // Per leaf of `mesh`: its xi at its own level, and the xi of its parent cell at the parent level (1e300 at level 0).
    struct Result {
        std::vector<double> xi, xi_parent;
    };
    // On a partial mesh (a rank's region: whole sibling groups are not required) the values of a leaf are right as long as every
    // leaf listed by region() for it is present with its state; the others are not to be used.
    Result evaluate(const CellMesh& mesh, const Field& U, const EOS& eos) const;
    // The leaves of the whole mesh `mesh` that xi and xi_parent of the leaves `leaves` read: the leaves of the cells within
    // diffusion + 1 cells of the leaf (at its level) and of its parent (at the parent level), with the leaves under a split cell.
    // Sets need[c] = 1 for them (need has one entry per leaf; entries already set stay).
    void region(const CellMesh& mesh, const std::vector<int>& leaves, std::vector<char>& need) const;
    // xi of every leaf (evaluate(...).xi).
    std::vector<double> xi(const CellMesh& mesh, const Field& U, const EOS& eos) const;
};

// Remesh driver: indicator -> flags -> CellMesh::adapt -> CellFluxEngine::build.
class CellAdaptor {
public:
    // The mesh (with its state U) and engine must outlive the object.
    CellAdaptor(CellMesh& mesh, CellFluxEngine& engine, const EOS& eos, const CellIndicator& ind)
        : mesh_(mesh), engine_(engine), eos_(eos), ind_(ind) {}

    // Keep the immersed body, and `margin` finest-level cells around it (the fringe of the solid reset), on the finest
    // level: `factory` gives the body mask on any uniform mesh. Call before remesh / init_adapt.
    void set_body(std::function<ImmersedBoundary(const Mesh&)> factory, int margin = 2);

    // One adaptation of the leaves from the current state; the engine is rebuilt when the leaves changed.
    // Returns whether they did.  xi of the call is kept in last_xi (per leaf of the mesh BEFORE the call).
    bool remesh();

    // Initial adaptation: `initial(mesh)` fills mesh.U at the leaf centres (each leaf at its own level); repeat
    // indicator + adapt until nothing changes (at most max_level + 1 passes), then fill once more. Returns the passes made.
    int init_adapt(const std::function<void(CellMesh&)>& initial);

    const CellIndicator& indicator() const { return ind_; }

    // True if the leaf lies in the finest-level band of the body.
    bool in_body_band(int leaf);

    // The decisions of remesh() (before the ripple and the merge test of CellMesh::adapt) for the leaves `which` of `m`, from
    // the state U of `m`: refine[k] / coarsen[k] for leaf which[k]. `m` may be a partial mesh (see CellIndicator::evaluate).
    void flags(const CellMesh& m, const Field& U, const std::vector<int>& which, std::vector<char>& refine,
               std::vector<char>& coarsen);

    std::vector<double> last_xi;
    // Bytes of the body band (finest uniform mesh) and of last_xi.
    double band_bytes() const;
    double xi_bytes() const;
    // Accumulated wall time [s] of the three parts of remesh(): indicator + flags, CellMesh::adapt, CellFluxEngine::build.
    double t_indicator = 0.0, t_adapt = 0.0, t_build = 0.0;
    // Leaves flagged for refinement and the leaves removed by joins, accumulated over the remeshes (churn of the mesh).
    long long n_refined = 0, n_joined = 0;

private:
    void build_band();
    bool in_body_band(int level, int i, int j);
    // refine / coarsen of one leaf from its indicator values and the body band
    void decide(int max_level, int level, int i, int j, double xi, double xi_parent, char& refine, char& coarsen);

    CellMesh& mesh_;
    CellFluxEngine& engine_;
    EOS eos_;
    CellIndicator ind_;
    std::function<ImmersedBoundary(const Mesh&)> body_;
    int margin_ = 2;
    std::vector<char> band_;   // body + margin on the uniform mesh of the finest level
};

}   // namespace fv
