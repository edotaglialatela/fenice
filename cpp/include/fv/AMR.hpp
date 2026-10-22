// Block-structured quad-tree AMR: patches, tree operations, Lohner error indicator,
// recursive subcycling with PARAMESH-style refluxing.
#pragma once

#include <functional>
#include <map>
#include <memory>

#include "fv/Core.hpp"
#include "fv/TimeScheme.hpp"

namespace fv {

// One block of the quad-tree: a uniform sub-mesh with its own state.  All patches
// share the same cell count; the cell size halves per level.
struct AMRPatch {
    double xmin = 0, xmax = 0, ymin = 0, ymax = 0;   // physical extent
    int level = 0;                                   // 0 = root
    int id = 0;                                      // unique, creation order
    AMRPatch* parent = nullptr;                      // nullptr for the root
    std::vector<AMRPatch*> children;                 // [BL, BR, TL, TR] when refined
    Mesh mesh;                                       // global coordinates (x0 = xmin, y0 = ymin)
    Field U;                                         // conservative state, row-major

    // Immersed-boundary masks (built lazily from the body geometry, which never changes).
    std::vector<char> solid;       // nx*ny, 1 = solid cell
    std::vector<char> solid_ext;   // (nx+4)*(ny+4) with a two-cell ghost ring; 0 outside the domain

    // Distribution (see assign_ownership): the rank that advances this leaf, and whether this rank
    // holds its data at all (owned, or a halo copy of a leaf next to an owned one)
    int owner = 0;
    bool held = true;
    std::vector<int> viewers;   // other ranks that hold a halo copy of this leaf (sorted)
    std::vector<int> peers;     // those of them that advance a leaf of the same level next to it

    // Transient state used by advance_amr_level
    bool has_ghost = false;   // U_ghost holds the frozen neighbour snapshot
    Field U_ghost;
    Field last_flux_x, last_flux_y;   // face flux * face length of the last stage
    bool has_reflux = false;          // reflux registers allocated (abuts a coarser leaf)
    Field reflux_fx, reflux_fy;       // sum of dt_fine * fine face fluxes over the sub-steps
    std::vector<Face> reflux_faces;   // faces touching a coarser neighbour

    // True if the patch has no children (an active computational block).
    bool is_leaf() const { return children.empty(); }
    // Cells in x.
    int nx() const { return mesh.nx; }
    // Cells in y.
    int ny() const { return mesh.ny; }
    // Solid flag of cell (ix, iy), -2 <= ix < nx+2, -2 <= iy < ny+2 (ghost ring included).
    char ext_solid(int ix, int iy) const { return solid_ext[(iy + 2) * (nx() + 4) + ix + 2]; }
    // State neighbours must read: the frozen snapshot during a level advance.
    const Field& ghost_source() const { return has_ghost ? U_ghost : U; }
};

// Quad-tree manager: refinement, derefinement, 2:1 balance and neighbour lookup.
class AMRTree {
public:
    // Level-0 patches tile [0, Lx] x [0, Ly] as nroot_x x nroot_y blocks of nx_patch x ny_patch cells.
    // Patch dimensions must be even when max_level > 0 (2 fine cells must tile 1 coarse cell);
    // throws otherwise.
    AMRTree(double Lx, double Ly, int nx_patch, int ny_patch, int max_level = 2, int nroot_x = 1, int nroot_y = 1);

    std::vector<AMRPatch*> leaves() const;   // root by root (row-major), DFS inside each
    // Leaf covering the point (nullptr outside the domain). Uses an index rebuilt after every tree
    // change; the first call after a change (leaves() makes one) must not be inside a parallel region.
    AMRPatch* leaf_at(double x, double y) const;
    std::vector<AMRPatch*> all_patches() const;   // by creation id

    // Split a leaf into 4 children [BL, BR, TL, TR] by conservative injection; throws at max_level.
    std::vector<AMRPatch*> refine_patch(AMRPatch* patch);
    // Collapse 4 leaf children into `patch` by conservative averaging.
    void derefine_patch(AMRPatch* patch);
    // Refine until no two face-neighbouring leaves differ by more than one level.
    void enforce_2to1_balance();

    // First leaf across `face` (nullptr at the domain boundary).
    AMRPatch* get_neighbour(const AMRPatch* patch, Face face) const;
    // Every leaf across `face` (up to 2 finer leaves under 2:1 balance).
    std::vector<AMRPatch*> neighbour_leaves(const AMRPatch* patch, Face face) const;
    // Every other leaf touching the box of `patch` grown by `cells` of its own cells on all sides
    // (corners included): the leaves whose data a residual of `patch` may read.
    std::vector<AMRPatch*> leaves_near(const AMRPatch* patch, int cells) const;

    AMRPatch* root = nullptr;       // roots[0]; the level-0 patch when the tree has a single root
    std::vector<AMRPatch*> roots;   // level-0 patches, row-major (x fastest)
    // False in a distributed run: a patch is created without state, which only the ranks holding it
    // allocate (refine / derefine skip the arithmetic where the source has no data).
    bool store_state = true;
    double Lx, Ly;                       // domain size
    int nx_patch, ny_patch, max_level;   // block size and refinement depth
    int nroot_x, nroot_y;                // level-0 patches per axis

private:
    AMRPatch* new_patch(double x0, double x1, double y0, double y1, int level, AMRPatch* parent);
    static void collect_leaves(AMRPatch* p, std::vector<AMRPatch*>& out);
    void rebuild_index() const;
    // Slot-grid index range [i0, i1) x [j0, j1) of a patch
    void slots_of(const AMRPatch* p, int& i0, int& i1, int& j0, int& j1) const;

    // Leaf list and slot grid (one slot per finest-level patch footprint), valid unless dirty_
    mutable std::vector<AMRPatch*> leaf_cache_;
    mutable std::vector<AMRPatch*> slot_;
    mutable bool dirty_ = true;
    int nslot_x_ = 1, nslot_y_ = 1;
    double slot_w_ = 1.0, slot_h_ = 1.0;

    std::map<int, std::unique_ptr<AMRPatch>> patches_;
    int next_id_ = 0;
};

// Lohner second-derivative density sensor and refine / derefine marking.
class AMRErrorIndicator {
public:
    // Refine above `refine_threshold`, derefine below `derefine_threshold`, with
    // `buffer_layers` rings of neighbours added around flagged patches.
    AMRErrorIndicator(double refine_threshold = 0.3, double derefine_threshold = 0.1, int buffer_layers = 1)
        : refine_threshold(refine_threshold), derefine_threshold(derefine_threshold), buffer_layers(buffer_layers) {}

    // eta = max over cells of |second difference| / (|first differences| + 0.01 max|f|), in [0, 1], for
    // f = density and / or the speed |u| (the largest of the enabled ones).
    double indicator(const AMRPatch& patch) const;
    // Leaves to refine (flagged + buffer, sorted by id) and parents to derefine.
    void mark(AMRTree& tree, std::vector<AMRPatch*>& to_refine, std::vector<AMRPatch*>& to_derefine) const;
    // The same, from the indicator of every leaf (by patch id) computed elsewhere.
    void mark(AMRTree& tree, const std::map<int, double>& eta, std::vector<AMRPatch*>& to_refine,
              std::vector<AMRPatch*>& to_derefine) const;

    double refine_threshold, derefine_threshold;   // eta thresholds
    int buffer_layers;                             // halo rings around flagged patches
    bool use_density = true, use_speed = false;    // fields the ratio is taken of
    // Optional: leaves for which this is true are refined to max_level whatever eta says, and their
    // parents are never derefined (an immersed body kept at the finest level).
    std::function<bool(const AMRPatch&)> force_refine;
};

// Residual of one leaf patch: fills R (cells) and flux_x / flux_y (face flux *
// face length) and returns S_max.
using PatchResidualFn =
    std::function<double(AMRPatch& patch, AMRTree& tree, const Field& U, Field& R, Field& flux_x, Field& flux_y)>;

// Distributed advance: every rank holds the whole tree (so every rank takes the same remesh decisions); the
// leaves of each level are split over the ranks (assign_ownership), each rank advances its own, and the
// owner of a leaf sends its data to the ranks that hold a halo copy of it: the same-level neighbours after
// every stage, everything after a level has been advanced (and refluxed). The arithmetic of a patch does
// not depend on its rank: results equal serial.
struct AMRParallel {
    int rank = 0, size = 1;
    // Gather the `n_send` doubles of every rank into `recv` (rank r's block has counts[r], at displs[r]).
    std::function<void(const double* send, int n_send, double* recv, const int* counts, const int* displs)> allgatherv;
    std::function<double(double)> allreduce_max;   // max over the ranks
    // Sparse all-to-all: send[r] goes to rank r, recv[r] (pre-sized by the caller) is filled from rank r;
    // an empty buffer means no message. Returns when all of them are complete.
    std::function<void(const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv)> exchange;
};

// After assign_ownership on a tree without state: give the held leaves their (zeroed) storage.
void allocate_held(AMRTree& tree);
// Every owner sends the state of its leaves to the ranks that hold a halo copy (after the owners changed
// them outside advance_amr_level, e.g. the solid reset).
void refresh_halos(AMRTree& tree, const AMRParallel& par);
// `width` doubles per leaf, computed by the owner of each leaf with `fn(leaf, out)`; the result holds every
// leaf's values in leaf order on all ranks. With par == nullptr this rank computes them all.
std::vector<double> allgather_leaf_values(AMRTree& tree, const AMRParallel* par, int width,
                                          const std::function<void(AMRPatch&, double*)>& fn);
// The state of every leaf concatenated in leaf order, on rank 0 only (empty on the others).
Field gather_leaves_on_root(AMRTree& tree, const AMRParallel& par);

// Distribute the leaves over `size` ranks: each level is cut into contiguous blocks along the Morton
// order of its slot-grid position (patches of one level have equal size, so equal counts). Sets
// `owner` and `held` (owned, or within the 2-cell stencil of an owned leaf) as seen from `rank`.
// Deterministic from the tree alone, so every rank computes the same partition.
void assign_ownership(AMRTree& tree, int rank, int size);

// Recursive subcycled advance of level `level` by dt; returns max wave speed. With `par` the patches of a
// level are shared out over the ranks (contiguous blocks in leaf order).
double advance_amr_level(AMRTree& tree, int level, double dt, double current_time, double max_time,
                         const PatchResidualFn& residual, TimeScheme& scheme, const AMRParallel* par = nullptr);

// Mark, refine, derefine and rebalance the tree. With `par` every rank computes the indicator of its own
// leaves, the values are shared, and all ranks then take the same decisions on their copy of the tree.
// The data follows: a rank that holds a patch refines it, the owner of the first child collects and
// averages the four children of a derefined parent (they are in its halo), and after the new partition
// the leaves move to the ranks that now need them. Throws if the ranks' trees diverge.
void remesh(AMRTree& tree, const AMRErrorIndicator& indicator, const AMRParallel* par = nullptr);

// Test hook: disables the coarse-fine reflux correction (negative control).
extern bool g_amr_disable_reflux;

}   // namespace fv
