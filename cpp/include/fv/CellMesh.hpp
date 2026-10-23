// Cell-based adaptive mesh: a flat list of leaf cells (level, i, j) sorted by (level, Morton key), the face list
// between them and a cell -> face gather list.  2-D only.  Topology and data transfer; no physics.
//
// Level l has (nx << l) x (ny << l) cells of size (Lx / nx / 2^l, Ly / ny / 2^l); level 0 is the user's base mesh,
// so a mesh with level-0 leaves only carries exactly the geometry of `Mesh` (same expressions, bit for bit).
// Corner (vertex) 2:1 balance: two leaves that touch, even at a single corner, differ by at most one level.
#pragma once

#include <cstdint>
#include <vector>

#include "fv/Core.hpp"

namespace fv {

// Interleaved bits of (i, j): the Morton (Z-order) key of a cell inside its level.
uint64_t morton_key(uint32_t i, uint32_t j);

// What a face separates.
enum class CellFaceKind : uint8_t {
    Interior,     // two leaves of the same level
    CoarseFine,   // a fine leaf and a leaf one level coarser (the coarse side is covered by two such faces)
    Boundary,     // one leaf and the edge of the domain
};

// One face between two leaves, or between a leaf and a domain edge.  `left` / `right` are the leaves on the
// lower / higher side along `axis` (-1 on the missing side of a boundary face); across a periodic wrap `left`
// is the last column (row) and `right` the first.  The face belongs to the level of the finer of its two cells.
struct CellFace {
    int left = -1, right = -1;
    int level = 0;                          // level of the finer cell
    int tpos = 0;                           // index, at `level`, of the transverse row / column of the finer cell
    uint8_t axis = 0;                       // 0: normal to x (a vertical face), 1: normal to y
    CellFaceKind kind = CellFaceKind::Interior;
    Face side = Face::Left;                 // boundary faces only: the domain edge
    double length = 0.0;                    // size of the finer cell along the face
};

// A leaf's neighbourhood across one side (a summary of the face list).
struct CellNeighbour {
    enum class Kind { Boundary, Same, Coarser, Finer } kind = Kind::Boundary;
    int a = -1;   // the neighbour leaf; for Finer the one with the lower transverse position
    int b = -1;   // Finer only: the upper one
};

// Leaf cells of a 2-D adaptive mesh, their faces and the gather lists; see the file header.
class CellMesh {
public:
    // Base mesh nx x ny on [x0, x0+Lx] x [y0, y0+Ly]; levels 0..max_level.  A periodic axis wraps its two edges.
    // Throws if the finest level does not fit the keys (nx << max_level must stay below 2^29).
    CellMesh(double Lx, double Ly, int nx, int ny, int max_level = 0, bool periodic_x = false,
             bool periodic_y = false, double x0 = 0.0, double y0 = 0.0);

    // --- leaves (parallel arrays, sorted by (level, Morton key) of (ci, cj)) ---
    // Number of leaves.
    int num_leaves() const { return static_cast<int>(level.size()); }
    // First leaf of level l (level_begin[max_level + 1] == num_leaves()); each level is one contiguous run.
    std::vector<int> level_begin;
    std::vector<int> level, ci, cj;   // level and integer coordinates (at that level) of every leaf
    // Conservative state of every leaf; empty = no state (adapt then skips the transfer).
    Field U;

    // Cell size in x at level l (exact halving of the base size).
    double dx(int l) const { return dx0_ / static_cast<double>(1 << l); }
    // Cell size in y at level l.
    double dy(int l) const { return dy0_ / static_cast<double>(1 << l); }
    // Area of a leaf.
    double area(int leaf) const { return dx(level[leaf]) * dy(level[leaf]); }
    // Global x of the centre of a leaf; at level 0 equal to Mesh::xc.
    double xc(int leaf) const { return (ci[leaf] + 0.5) * dx(level[leaf]) + x0; }
    // Same for y.
    double yc(int leaf) const { return (cj[leaf] + 0.5) * dy(level[leaf]) + y0; }
    // Cells in x at level l.
    int nx_at(int l) const { return nx << l; }
    // Cells in y at level l.
    int ny_at(int l) const { return ny << l; }

    // Leaf with exactly these coordinates, or -1.  (A lookup, never used in a time loop.)
    int find(int l, int i, int j) const;
    // Leaf equal to or containing cell (l, i, j) (coordinates inside the domain), or -1 if that cell is subdivided.
    int covering(int l, int i, int j) const;

    // --- faces (rebuilt after every adapt) ---
    // All faces, non-decreasing in level; face_level_begin[l] is the first of level l (size max_level + 2).
    std::vector<CellFace> faces;
    // First face of each level.
    std::vector<int> face_level_begin;
    // Cell -> faces gather list: the faces on side s (Face::Left..Top) of leaf c are
    // side_face[side_ptr[4 c + s] .. side_ptr[4 c + s + 1]): one, or two (finer neighbours) ordered by increasing
    // transverse position; the single entry of a boundary side is the boundary face.
    std::vector<int> side_ptr, side_face;
    // Neighbourhood of a leaf across a side.
    CellNeighbour neighbour(int leaf, Face side) const;

    // --- adaptation ---
    // Split the leaves with refine[c] != 0 (not at max_level) into four children by conservative injection, then
    // refine more leaves, as far as needed, until the corner 2:1 balance holds.  Merge four sibling leaves into
    // their parent (mean of the four) when all four have coarsen[c] != 0, none is refined, and the parent
    // would keep the balance with every corner neighbour as it is after this call's refinements; a merge that
    // would not is skipped (conservative: decided on the state before the call, so the outcome does not depend on
    // the order of the groups).  Empty vectors mean "none".  Rebuilds the faces (only if something changed) and
    // returns whether the leaves changed.
    bool adapt(const std::vector<char>& refine, const std::vector<char>& coarsen = {});

    // Replace the leaves by the given ones (a restart): level and integer coordinates of every leaf, in any order; `state`,
    // if given, holds the state in the same order and is reordered with them. Throws if the leaves do not tile the domain or
    // break the corner 2:1 balance. Rebuilds the faces.
    void assign_leaves(const std::vector<int>& levels, const std::vector<int>& is, const std::vector<int>& js,
                       Field* state = nullptr);

    // --- last change of the leaves (lets dependent tables be updated instead of rebuilt) ---
    // Counter of the changes of the leaves (construction, adapt, assign_leaves).
    long long version = 0;
    // After an adapt that changed the leaves (empty after any other change, or with FENICE_FULL_REBUILD set): for every leaf
    // its index before the call (-1: a new leaf, a child or a parent)...
    std::vector<int> prev_leaf;
    // ... whether its neighbourhood may have changed: a new leaf, or one of the four squares of its size across its sides holds
    // a new leaf. The faces, neighbours and stencils of a clean leaf are its old ones with the indices renumbered...
    std::vector<char> topo_dirty;
    // ... and for every face its index before the call (-1: made anew; the faces a clean leaf makes are copied).
    std::vector<int> prev_face;

    // --- mesh data ---
    double Lx, Ly;
    int nx, ny, max_level;
    bool periodic_x, periodic_y;
    double x0, y0;

private:
    double dx0_, dy0_;
    // Lookup (position -> leaf): a dense table per level where nx_l * ny_l is small compared with the leaf count, else a
    // binary search of the Morton keys of the level's leaves (mkey_, parallel to the leaf arrays).
    std::vector<std::vector<int>> grid_;
    std::vector<uint64_t> mkey_;

    std::vector<int> face_off_;   // faces made by leaf c: faces[face_off_[c] .. face_off_[c + 1])

    // Lookup tables, faces and gather lists of the current leaves. `incremental`: prev_leaf is set (adapt), the faces of the clean
    // leaves are copied from the old ones.
    void rebuild(bool incremental = false);
    // Wrap (i, j) of level l into the domain on periodic axes; false if it falls outside a non-periodic edge.
    bool wrap(int l, int& i, int& j) const;
};

}   // namespace fv
