// Cell-based adaptive mesh: a flat list of leaf cells (level, i, j[, k]) sorted by (level, Morton key), the face list
// between them and a cell -> face gather list, in two dimensions (quadtree leaves) or three (octree leaves, CellMesh::make_3d).
// Topology and data transfer; no physics.
//
// Level l has (nx << l) x (ny << l) [x (nz << l)] cells of size (Lx / nx / 2^l, Ly / ny / 2^l[, Lz / nz / 2^l]); level 0 is the
// user's base mesh, so a mesh with level-0 leaves only carries exactly the geometry of `Mesh` (same expressions, bit for bit).
// Vertex 2:1 balance: two leaves that touch, even at a single corner (or edge), differ by at most one level.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "fv/CellKey.hpp"
#include "fv/Core.hpp"

namespace fv {

// What a face separates.
enum class CellFaceKind : uint8_t {
    Interior,     // two leaves of the same level
    CoarseFine,   // a fine leaf and a leaf one level coarser (the coarse side is covered by 2 such faces in 2-D, 4 in 3-D)
    Boundary,     // one leaf and the edge of the domain
};

// A leaf that is not in a partial mesh (a face end, a neighbour); never -1, which is the missing side of a boundary face.
constexpr int kOutside = -2;

// One face between two leaves, or between a leaf and a domain edge.  `left` / `right` are the leaves on the
// lower / higher side along `axis` (-1 on the missing side of a boundary face, kOutside in a partial mesh for a leaf that is not
// kept); across a periodic wrap `left` is the last column (row, layer) and `right` the first.  The face belongs to the level of
// the finer of its two cells.
struct CellFace {
    int left = -1, right = -1;
    int level = 0;                          // level of the finer cell
    int tpos = 0;                           // index, at `level`, of the finer cell along the first transverse axis (transverse1)
    int tpos2 = 0;                          // 3-D: the same along the second transverse axis (transverse2); 0 in 2-D
    uint8_t axis = 0;                       // 0: normal to x, 1: normal to y, 2: normal to z (3-D)
    CellFaceKind kind = CellFaceKind::Interior;
    Face side = Face::Left;                 // boundary faces only: the domain edge
    double length = 0.0;                    // measure of the finer cell's side: a length in 2-D, an area in 3-D (Mesh::face_area)
};
static_assert(sizeof(CellFace) == 32, "CellFace is read by every face loop of the engine: keep it at 32 bytes");

// A leaf's neighbourhood across one side (a summary of the face list).
struct CellNeighbour {
    // Outside: a partial mesh does not know this side (a leaf across it, or one of the finer ones, is not kept)
    enum class Kind { Boundary, Same, Coarser, Finer, Outside } kind = Kind::Boundary;
    int a = -1;   // the neighbour leaf; for Finer the one at the lowest transverse position(s)
    int b = -1;   // Finer only: the next one along the first transverse axis (the upper one in 2-D)
    int c = -1;   // Finer in 3-D only: the next ones along the second transverse axis, (t1, t2 + 1) ...
    int d = -1;   // ... and (t1 + 1, t2 + 1)
};

// A box of cells of one level: a bitmap over [i0, i0 + w) x [j0, j0 + h) (see dilate_cells in CellRemesh.hpp). 2-D.
struct CellBox {
    int i0 = 0, j0 = 0, w = 0, h = 0;
    int k0 = 0, d = 1;     // 3-D: the z range [k0, k0 + d) (2-D: one layer)
    std::vector<char> g;   // g[((k - k0) * h + (j - j0)) * w + (i - i0)]
};

// Leaf cells of an adaptive mesh, their faces and the gather lists; see the file header.
class CellMesh {
public:
    // 2-D base mesh nx x ny on [x0, x0+Lx] x [y0, y0+Ly]; levels 0..max_level.  A periodic axis wraps its two edges.
    // Throws if the finest level does not fit the keys (nx << max_level must stay below 2^29).
    CellMesh(double Lx, double Ly, int nx, int ny, int max_level = 0, bool periodic_x = false,
             bool periodic_y = false, double x0 = 0.0, double y0 = 0.0);
    // 3-D base mesh nx x ny x nz on [x0, x0+Lx] x [y0, y0+Ly] x [z0, z0+Lz] (octree leaves). The finest level must have fewer than
    // 2^21 cells per axis.
    static CellMesh make_3d(double Lx, double Ly, double Lz, int nx, int ny, int nz, int max_level = 0, bool periodic_x = false,
                            bool periodic_y = false, bool periodic_z = false, double x0 = 0.0, double y0 = 0.0, double z0 = 0.0);
    // A mesh with the geometry of `like` (2-D or 3-D) and no leaf, to be filled by assign_leaves / assign_partial_leaves: the base
    // mesh is not built (its sort and its faces cost as much as a remesh of a share).
    static CellMesh make_empty(const CellMesh& like);

    // --- leaves (parallel arrays, sorted by (level, Morton key) of (ci, cj[, ck])) ---
    // Number of leaves.
    int num_leaves() const { return static_cast<int>(level.size()); }
    // First leaf of level l (level_begin[max_level + 1] == num_leaves()); each level is one contiguous run.
    std::vector<int> level_begin;
    std::vector<int> level, ci, cj;   // level and integer coordinates (at that level) of every leaf
    std::vector<int> ck;              // the z coordinate: 0 for every leaf of a 2-D mesh
    // Conservative state of every leaf; empty = no state (adapt then skips the transfer).
    Field U;

    // Dimension: 2 or 3.
    int dim = 2;
    // Sides of a leaf (4 or 6, in Face order): the gather lists are indexed sides() * c + s.
    int sides() const { return 2 * dim; }
    // Children of a cell (4 or 8).
    int children() const { return 1 << dim; }
    // Faces on a side towards finer leaves (2 or 4).
    int fine_faces() const { return 1 << (dim - 1); }

    // Cell size in x at level l (exact halving of the base size).
    double dx(int l) const { return dx0_ / static_cast<double>(1 << l); }
    // Cell size in y at level l.
    double dy(int l) const { return dy0_ / static_cast<double>(1 << l); }
    // Cell size in z at level l (3-D).
    double dz(int l) const { return dz0_ / static_cast<double>(1 << l); }
    // Area of a leaf (2-D; the cell measure of a 2-D mesh).
    double area(int leaf) const { return dx(level[leaf]) * dy(level[leaf]); }
    // Volume of a leaf: dx dy dz in 3-D, the area in 2-D (as Mesh::cell_volume).
    double volume(int leaf) const {
        return dim == 3 ? dx(level[leaf]) * dy(level[leaf]) * dz(level[leaf]) : area(leaf);
    }
    // Global x of the centre of a leaf; at level 0 equal to Mesh::xc.
    double xc(int leaf) const { return (ci[leaf] + 0.5) * dx(level[leaf]) + x0; }
    // Same for y.
    double yc(int leaf) const { return (cj[leaf] + 0.5) * dy(level[leaf]) + y0; }
    // Same for z (3-D).
    double zc(int leaf) const { return (ck[leaf] + 0.5) * dz(level[leaf]) + z0; }
    // Cells in x at level l.
    int nx_at(int l) const { return nx << l; }
    // Cells in y at level l.
    int ny_at(int l) const { return ny << l; }
    // Cells in z at level l (1 in 2-D).
    int nz_at(int l) const { return dim == 3 ? nz << l : 1; }
    // Morton key of the cell (i, j, k) of a level (k ignored in 2-D).
    uint64_t key_of(int i, int j, int k = 0) const { return cell_key(dim, i, j, k); }

    // Leaf with exactly these coordinates, or -1.  (A lookup, never used in a time loop.)
    int find(int l, int i, int j, int k = 0) const;
    // Leaf equal to or containing cell (l, i, j[, k]) (coordinates inside the domain), or -1 if that cell is subdivided.
    int covering(int l, int i, int j, int k = 0) const;

    // --- faces (rebuilt after every adapt) ---
    // All faces, non-decreasing in level; face_level_begin[l] is the first of level l (size max_level + 2).
    std::vector<CellFace> faces;
    // First face of each level.
    std::vector<int> face_level_begin;
    // Cell -> faces gather list: the faces on side s (Face::Left..Top, ..Back in 3-D) of leaf c are
    // side_face[side_ptr[sides() c + s] .. side_ptr[sides() c + s + 1]): one, or fine_faces() (finer neighbours) ordered by the
    // transverse position (tpos2 major, tpos minor); the single entry of a boundary side is the boundary face.
    std::vector<int> side_ptr, side_face;
    // Neighbourhood of a leaf across a side.
    CellNeighbour neighbour(int leaf, Face side) const;

    // --- adaptation ---
    // Split the leaves with refine[c] != 0 (not at max_level) into children() children by conservative injection, then
    // refine more leaves, as far as needed, until the vertex 2:1 balance holds.  Merge the children() sibling leaves into
    // their parent (their mean) when all have coarsen[c] != 0, none is refined, and the parent would keep the balance with every
    // neighbour as it is after this call's refinements; a merge that would not is skipped (conservative: decided on the state
    // before the call, so the outcome does not depend on the order of the groups).  Empty vectors mean "none".  Rebuilds the
    // faces (only if something changed) and returns whether the leaves changed.
    bool adapt(const std::vector<char>& refine, const std::vector<char>& coarsen = {});

    // Replace the leaves by the given ones (a restart): level and integer coordinates of every leaf, in any order; `state`,
    // if given, holds the state in the same order and is reordered with them. Throws if the leaves do not tile the domain or
    // break the vertex 2:1 balance. Rebuilds the faces. `ks` is empty for a 2-D mesh.
    void assign_leaves(const std::vector<int>& levels, const std::vector<int>& is, const std::vector<int>& js,
                       Field* state = nullptr);
    void assign_leaves(const std::vector<int>& levels, const std::vector<int>& is, const std::vector<int>& js,
                       const std::vector<int>& ks, Field* state = nullptr);

    // --- partial mesh (one rank's share of a distributed run, CellParallel::localize) ---
    // Keep only the leaves with keep_leaf != 0 and the faces with keep_face != 0 (call on a whole mesh), in the same order: the
    // leaf and face indices become local ones, leaf_map / face_map give the new index of every old leaf / face (-1: dropped).
    // A face keeps its end as kOutside if that leaf is dropped; a side whose faces are not all kept has an empty gather list
    // (neighbour() says Outside). The state U goes with the leaves. Frees the incremental bookkeeping.
    void keep_partial(const std::vector<char>& keep_leaf, const std::vector<char>& keep_face, std::vector<int>& leaf_map,
                      std::vector<int>& face_map);
    // The same kind of partial mesh built directly from the whole mesh `whole` (same geometry) and the leaves to keep, without
    // a face list of the whole mesh: the kept leaves in their order, the faces they make (a leaf that is not kept is kOutside
    // at the other end of a face), a side whose faces are not all there empty. The state of a leaf comes from whole_state (one
    // entry per leaf of `whole`) if given.
    void build_from(const CellMesh& whole, const std::vector<char>& keep_leaf, const Field* whole_state = nullptr);
    // The leaves given (any subset of a mesh, in any order) as a partial mesh without faces: sorted by (level, key), with the lookups
    // (find, covering). The state, if given, follows the leaves. For a fetched region (CellParallel::fetch_leaves). `ks` empty in 2-D.
    void assign_partial_leaves(const std::vector<int>& levels, const std::vector<int>& is, const std::vector<int>& js, Field* state);
    void assign_partial_leaves(const std::vector<int>& levels, const std::vector<int>& is, const std::vector<int>& js,
                               const std::vector<int>& ks, Field* state);
    // True if the leaves, faces, gather lists and level runs are the same as those of `other` (tests).
    bool same_arrays(const CellMesh& other) const;
    // A faceless mesh keeps the leaves and the lookups (find, covering, the level runs) but no face list or side lists, so adapt()
    // costs no face work: the topology a rank keeps next to its share (CellParallel::remesh). Turning it off rebuilds the faces.
    void set_faceless(bool on);
    // True after keep_partial / build_from (until the leaves are assigned again).
    bool partial = false;

    // --- last change of the leaves (lets dependent tables be updated instead of rebuilt) ---
    // Counter of the changes of the leaves (construction, adapt, assign_leaves).
    long long version = 0;
    // After an adapt that changed the leaves (empty after any other change, or with FENICE_FULL_REBUILD set): for every leaf
    // its index before the call (-1: a new leaf, a child or a parent)...
    std::vector<int> prev_leaf;
    // ... whether its neighbourhood may have changed: a new leaf, or one of the squares (cubes) of its size across its sides holds
    // a new leaf. The faces, neighbours and stencils of a clean leaf are its old ones with the indices renumbered...
    std::vector<char> topo_dirty;
    // ... and for every face its index before the call (-1: made anew; the faces a clean leaf makes are copied).
    std::vector<int> prev_face;
    // With record_sources set (a distributed run, which moves the state by itself), an adapt that changed the leaves also sets, for
    // every leaf, where its state comes from: {c, -1, ...}: the state of the old leaf c (a leaf that stayed, or the injection
    // into a child of a refined leaf); {a, b, c, d[, e, f, g, h]}: the mean of the old leaves of a merged group, in child order
    // (child_offset: x fastest, then y, then z); the unused entries are -1.
    // The decisions of adapt on any mesh with lookups: a whole mesh, or the partial mesh of a fetched region (CellParallel::remesh).
    // forced_by_finer: leaf b, not flagged itself, must be refined because a finer leaf around its block is (R: one flag per leaf).
    // merge_ok: the sibling group anchored at the leaf a (even coordinates) can be merged (coarsen, R: one flag per leaf). With
    // `check`, a cell that holds no leaf at all (a region that was fetched too small) throws std::logic_error instead of counting
    // as subdivided.
    bool forced_by_finer(int b, const std::vector<char>& R, bool check) const;
    bool merge_ok(int a, const std::vector<char>& coarsen, const std::vector<char>& R, bool check) const;
    // True if the cell (l, i, j[, k]) is covered by a leaf or holds leaves.
    bool has_leaf_under(int l, int i, int j, int k = 0) const;
    bool record_sources = false;
    std::vector<std::array<int, 8>> source;

    // Bytes of the leaf arrays, the state, the faces, the gather lists, the lookup tables and the incremental bookkeeping.
    double memory_bytes() const;

    // --- mesh data ---
    double Lx = 0.0, Ly = 0.0, Lz = 1.0;
    int nx = 1, ny = 1, nz = 1, max_level = 0;
    bool periodic_x = false, periodic_y = false, periodic_z = false;
    double x0 = 0.0, y0 = 0.0, z0 = 0.0;

private:
    CellMesh() = default;   // make_3d fills the fields
    double dx0_ = 0.0, dy0_ = 0.0, dz0_ = 1.0;
    // Lookup (position -> leaf): a dense table per level where the level's cell count is small compared with the leaf count, else
    // a binary search of the Morton keys of the level's leaves (mkey_, parallel to the leaf arrays).
    std::vector<std::vector<int>> grid_;
    std::vector<uint64_t> mkey_;

    std::vector<int> face_off_;   // faces made by leaf c: faces[face_off_[c] .. face_off_[c + 1])
    bool faceless_ = false;
    std::vector<CellFace> face_spare_;   // the face list before last, reused as the buffer of the next rebuild

    // Leaves of the base mesh in Morton order, then rebuild (both constructors).
    void init_base();
    // Lookup tables, faces and gather lists of the current leaves. `incremental`: prev_leaf is set (adapt), the faces of the clean
    // leaves are copied from the old ones.
    void rebuild(bool incremental = false);
    // Level runs and the position -> leaf lookup of the current leaves.
    void build_lookups();
    // Wrap (i, j, k) of level l into the domain on periodic axes; false if it falls outside a non-periodic edge.
    bool wrap(int l, int& i, int& j, int& k) const;
    // Face that leaf c makes on side s, looked up in `in` (this mesh, or the whole mesh of build_from with the leaf's coordinates
    // given): 0 none (a finer neighbour, or the same-level neighbour makes it), 1 a face (f filled; its leaf ends are indices of
    // `in`), -1 the side faces finer leaves (none made).
    int face_of(const CellMesh& in, int c, int l, int i, int j, int k, int s, CellFace& f) const;
    // Leaves sorted by (level, key) from the given arrays (assign_leaves / assign_partial_leaves).
    void set_sorted_leaves(const std::vector<int>& lv, const std::vector<int>& ii, const std::vector<int>& jj,
                           const std::vector<int>& kk, Field* state, const char* who);
};

}   // namespace fv
