// Internals shared by the CellParallel*.cpp files.
#pragma once
#include <cstdint>

#include "fv/CellMesh.hpp"

namespace fv {
namespace detail {

constexpr uint64_t kNoKey = ~static_cast<uint64_t>(0);   // above every Morton key of the mesh

inline uint64_t leaf_key(const CellMesh& m, int c) {
    return morton_key(static_cast<uint32_t>(m.ci[static_cast<size_t>(c)]), static_cast<uint32_t>(m.cj[static_cast<size_t>(c)]));
}

}   // namespace detail
}   // namespace fv
