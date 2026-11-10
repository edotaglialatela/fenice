// Internals shared by the CellParallel*.cpp files.
#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "fv/CellMesh.hpp"
#include "fv/CellRemesh.hpp"

namespace fv {
namespace detail {

constexpr uint64_t kNoKey = ~static_cast<uint64_t>(0);   // above every Morton key of the mesh

// The Morton key of a cell of a level (morton_key in 2-D, morton_key3 in 3-D)
inline uint64_t cell_morton(int dim, int i, int j, int k) {
    return dim == 3 ? morton_key3(static_cast<uint32_t>(i), static_cast<uint32_t>(j), static_cast<uint32_t>(k))
                    : morton_key(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
}

inline uint64_t leaf_key(const CellMesh& m, int c) {
    return cell_morton(m.dim, m.ci[static_cast<size_t>(c)], m.cj[static_cast<size_t>(c)], m.ck[static_cast<size_t>(c)]);
}

// The cells of level `level` within r cells of the seeds (i, j, k) (k = 0 in 2-D): dilate_cells, or dilate_cells3 in 3-D
inline CellBox dilate_seeds(const CellMesh& m, int level, const std::vector<std::array<int, 3>>& seeds, int r) {
    if (m.dim == 3)
        return dilate_cells3(seeds, m.nx_at(level), m.ny_at(level), m.nz_at(level), r, m.periodic_x, m.periodic_y, m.periodic_z);
    std::vector<std::array<int, 2>> s2(seeds.size());
    for (size_t q = 0; q < seeds.size(); ++q) s2[q] = {seeds[q][0], seeds[q][1]};
    return dilate_cells(s2, m.nx_at(level), m.ny_at(level), r, m.periodic_x, m.periodic_y);
}

// Visit the cells (i, j, k) set in a box
template <class F>
inline void for_box(const CellBox& b, const F& f) {
    for (int k = 0; k < b.d; ++k)
        for (int j = 0; j < b.h; ++j)
            for (int i = 0; i < b.w; ++i)
                if (b.g[(static_cast<size_t>(k) * b.h + j) * b.w + i]) f(b.i0 + i, b.j0 + j, b.k0 + k);
}

// An empty mesh with the geometry of `m` (2-D or 3-D), for the partial meshes of the distributed layer (no leaf: the base mesh is
// not built, the caller assigns the leaves)
inline CellMesh empty_like(const CellMesh& m) { return CellMesh::make_empty(m); }

}   // namespace detail
}   // namespace fv
