// Key and neighbourhood arithmetic of the leaf engine in two and three dimensions (the dimension layer of
// docs/design/amr-octree.md): Morton keys, children of a cell, the 3^D - 1 neighbour offsets, the transverse axes of a face.
// Header-only; CellMesh picks the dimension at run time.
#pragma once

#include <cstdint>

namespace fv {

namespace cellkey_detail {
inline uint64_t spread2(uint32_t v) {   // one zero bit between the bits of v (2-D)
    uint64_t x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
    x = (x | (x << 2)) & 0x3333333333333333ull;
    x = (x | (x << 1)) & 0x5555555555555555ull;
    return x;
}
inline uint64_t spread3(uint32_t v) {   // two zero bits between the bits of v (3-D, 21 bits)
    uint64_t x = v & 0x1FFFFFu;
    x = (x | (x << 32)) & 0x1F00000000FFFFull;
    x = (x | (x << 16)) & 0x1F0000FF0000FFull;
    x = (x | (x << 8)) & 0x100F00F00F00F00Full;
    x = (x | (x << 4)) & 0x10C30C30C30C30C3ull;
    x = (x | (x << 2)) & 0x1249249249249249ull;
    return x;
}
}   // namespace cellkey_detail

// Interleaved bits of (i, j): the Morton (Z-order) key of a cell inside its level (2-D).
inline uint64_t morton_key(uint32_t i, uint32_t j) { return cellkey_detail::spread2(i) | (cellkey_detail::spread2(j) << 1); }

// Interleaved bits of (i, j, k), i lowest: the Morton key of a 3-D cell (21 bits per axis). key >> 3 is the parent's key.
inline uint64_t morton_key3(uint32_t i, uint32_t j, uint32_t k) {
    return cellkey_detail::spread3(i) | (cellkey_detail::spread3(j) << 1) | (cellkey_detail::spread3(k) << 2);
}

// Bits per axis of the key of a D-dimensional cell: the finest level must have fewer than 2^bits cells per axis.
constexpr int cell_key_bits(int dim) { return dim == 3 ? 21 : 29; }

// Key of cell (i, j, k) of a D-dimensional mesh (k ignored in 2-D).
inline uint64_t cell_key(int dim, int i, int j, int k) {
    return dim == 3 ? morton_key3(static_cast<uint32_t>(i), static_cast<uint32_t>(j), static_cast<uint32_t>(k))
                    : morton_key(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
}

// Child q (0 .. 2^D - 1) of a cell sits at offset (q & 1, (q >> 1) & 1, (q >> 2) & 1) in the children's level: the children in
// increasing key order (x fastest, then y, then z).
inline void child_offset(int q, int o[3]) {
    o[0] = q & 1;
    o[1] = (q >> 1) & 1;
    o[2] = (q >> 2) & 1;
}

// The 3^D - 1 neighbour offsets (faces, edges, corners) in the order dz outer, dy, dx inner (the 2-D order dy outer, dx inner).
constexpr int cell_neighbour_count(int dim) { return dim == 3 ? 26 : 8; }
namespace cellkey_detail {
struct NeighbourTable {
    signed char o2[8][3], o3[26][3];
    constexpr NeighbourTable() : o2(), o3() {
        for (int r = 0, q2 = 0, q3 = 0; r < 27; ++r) {
            if (r == 13) continue;   // the centre
            const signed char x = static_cast<signed char>(r % 3 - 1), y = static_cast<signed char>((r / 3) % 3 - 1),
                              z = static_cast<signed char>(r / 9 - 1);
            o3[q3][0] = x;
            o3[q3][1] = y;
            o3[q3][2] = z;
            ++q3;
            if (z == 0) {
                o2[q2][0] = x;
                o2[q2][1] = y;
                o2[q2][2] = 0;
                ++q2;
            }
        }
    }
};
constexpr NeighbourTable kNeighbours{};
}   // namespace cellkey_detail
inline void neighbour_offset(int dim, int q, int o[3]) {
    const signed char* t = dim == 3 ? cellkey_detail::kNeighbours.o3[q] : cellkey_detail::kNeighbours.o2[q];
    o[0] = t[0];
    o[1] = t[1];
    o[2] = t[2];
}

// The two transverse axes of a face normal to `axis`: (t1, t2) = (y, z) for x, (x, z) for y, (x, y) for z. In 2-D the second one
// is z, whose coordinate is always 0.
inline int transverse1(int axis) { return axis == 0 ? 1 : 0; }
inline int transverse2(int axis) { return axis == 2 ? 1 : 2; }

}   // namespace fv
