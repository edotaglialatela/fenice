// Adaptation of a CellMesh from the solution: the face-variation indicator, the flags it gives, the
// finest-level band around an immersed body, and the remesh / initial adaptation drivers.
//
// Indicator:
// It is evaluated level by level on the cells of each level (the leaves of that level and the split ancestors of the deeper
// leaves, a split cell holding the mean of its children: 4 in 2-D, 8 in 3-D, which has 6 sides and a periodic z), so a parent has
// its own xi at the parent level:
//   1. xi = 0 in every cell of the level.
//   2. For every interface of the level (two cells of the level, or a cell and a coarser leaf) and every enabled variable
//      (density, pressure, velocity magnitude), with c_l, c_r the values on its two sides: m = max(min(|c_l|, |c_r|), floor),
//      variation = |c_r - c_l| / m; if variation >= criterion, the cells of the level on both sides get xi = 1.
//   3. `diffusion` iterations: every interface of two cells with one of them above `diffusion_trigger`, or of a cell above it
//      against a coarser leaf, adds `diffusion_coeff` to the cell(s) of the level; after all interfaces xi += added.
//   4. A leaf is refined if its xi >= split; four sibling leaves are joined if the xi of their PARENT cell is < join, where that
//      xi is computed (steps 2-3) with criterion * join_band, so a join needs a variation clearly below the refine trigger.
#pragma once

#include <array>
#include <functional>
#include <vector>

#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"

namespace fv {

// The cells of a level (NX x NY, periodic axes wrap) within r cells, along each axis, of the `seeds` (i, j): a bitmap over the box
// [i0, i0 + w) x [j0, j0 + h) that holds the seeds and the margin (the whole axis where the margin wraps a periodic one).
CellBox dilate_cells(const std::vector<std::array<int, 2>>& seeds, int NX, int NY, int r, bool periodic_x, bool periodic_y);
// The same in 3-D (Chebyshev distance r along each of the three axes): a bitmap over [i0, i0 + w) x [j0, j0 + h) x [k0, k0 + d).
CellBox dilate_cells3(const std::vector<std::array<int, 3>>& seeds, int NX, int NY, int NZ, int r, bool periodic_x, bool periodic_y,
                      bool periodic_z);

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
    // The parent xi of the join test is evaluated with criterion * join_band (a dead band against the refine test: a flow whose
    // variation sits at the criterion would otherwise be joined and refined again every few remeshes). Fixed, not a parameter.
    static constexpr double join_band = 0.9;

    // Test switch: false joins four siblings when all four have xi < join at their own level (the rule of the first
    // version, which flickers at the edge of a refined region) instead of testing the parent's xi.
    bool parent_join = true;

    // The smoothing of step 3: interfaces {a, b} between two cells of the level, or {a, -1} between a cell and a coarser leaf.
    void diffuse(std::vector<double>& xi, const std::vector<std::array<int, 2>>& interfaces) const;
    // The same smoothing gathered per cell (and parallel): nbr[sides c + s] is the cell across side s of cell c in the same level,
    // -1 if a coarser leaf lies there, -2 if nothing does (sides: 4 in 2-D, 6 in 3-D). Gives the same bits as diffuse() on the
    // interfaces of that table.
    void diffuse_cells(std::vector<double>& xi, const std::vector<int>& nbr, int sides = 4) const;
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
    // 3-D: no mask of the finest level; a leaf is in the band when a finest cell inside it has its centre within `margin` finest widths
    // (the largest of dx, dy, dz) of the body's surface (near_body_surface): a shell of `margin` cells on both sides of the surface (the
    // deep interior of the body is not kept at the finest level). The ghost-cell wall is ready with margin 6 (its own rule).
    void set_body(std::function<ImmersedBoundary(const Mesh&)> factory, int margin = 2);
    // Keep every leaf that overlaps the box [x0, x1] x [y0, y1] at the finest level, whatever the indicator says (a static refined
    // region: a test hook for orders of accuracy across a level jump, `amr_force_box`). Call before remesh / init_adapt.
    void set_force_box(double x0, double x1, double y0, double y1) {
        box_ = {x0, x1, y0, y1, -1e300, 1e300};
        has_box_ = true;
    }
    // The same in 3-D with the z range [z0, z1].
    void set_force_box3(double x0, double x1, double y0, double y1, double z0, double z1) {
        box_ = {x0, x1, y0, y1, z0, z1};
        has_box_ = true;
    }

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
    bool in_body_band(int level, int i, int j, int k);
    // refine / coarsen of one leaf from its indicator values and the body band
    void decide(int max_level, int level, int i, int j, int k, double xi, double xi_parent, char& refine, char& coarsen);

    CellMesh& mesh_;
    CellFluxEngine& engine_;
    EOS eos_;
    CellIndicator ind_;
    std::function<ImmersedBoundary(const Mesh&)> body_;
    int margin_ = 2;
    std::array<double, 6> box_ = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    bool has_box_ = false;
    std::vector<char> band_;   // body + margin on the uniform mesh of the finest level (2-D)
    std::shared_ptr<ImmersedBoundary> band_geo_;   // 3-D: the body built once on the base mesh (its geometry)
};

}   // namespace fv
