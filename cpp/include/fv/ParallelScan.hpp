// Parallel building blocks of the remesh: prefix sums, merges of sorted runs and the compaction of marks. They work on
// integers and on keys that are all distinct, so the result is the serial one for any thread count (no reordering of
// floating-point sums happens here). Short inputs run serially.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <vector>

#include "fv/Core.hpp"

namespace fv {

// Blocks of the parallel passes: a fixed number, so the split does not depend on the thread count.
constexpr int kScanBlocks = 64;

// a[k] += a[k - 1] for k = 1 .. n - 1 (inclusive prefix sum in place).
inline void inclusive_scan_inplace(int* a, int n) {
    if (n < 4 * kOmpMinWork) {
        for (int k = 1; k < n; ++k) a[k] += a[k - 1];
        return;
    }
    const int chunk = (n + kScanBlocks - 1) / kScanBlocks;
    int carry[kScanBlocks + 1];
#pragma omp parallel for schedule(static)
    for (int b = 0; b < kScanBlocks; ++b) {
        const int lo = std::min(n, b * chunk), hi = std::min(n, lo + chunk);
        for (int k = lo + 1; k < hi; ++k) a[k] += a[k - 1];
    }
    carry[0] = 0;
    for (int b = 0; b < kScanBlocks; ++b) {
        const int hi = std::min(n, (b + 1) * chunk);
        carry[b + 1] = carry[b] + (hi > b * chunk ? a[hi - 1] : 0);
    }
#pragma omp parallel for schedule(static)
    for (int b = 1; b < kScanBlocks; ++b) {
        const int lo = std::min(n, b * chunk), hi = std::min(n, lo + chunk);
        for (int k = lo; k < hi; ++k) a[k] += carry[b];
    }
}

// out[0 .. na + nb) = std::merge(a, a + na, b, b + nb, out, less), each output block placed by a co-rank search.
template <class T, class Less>
void parallel_merge(const T* a, int na, const T* b, int nb, T* out, Less less) {
    const int n = na + nb;
    if (n < 4 * kOmpMinWork || na == 0 || nb == 0) {
        std::merge(a, a + na, b, b + nb, out, less);
        return;
    }
    const int chunk = (n + kScanBlocks - 1) / kScanBlocks;
    // number of entries of a among the first k of the merge (std::merge takes a first on ties)
    const auto corank = [&](int k) {
        int lo = std::max(0, k - nb), hi = std::min(k, na);
        while (lo < hi) {
            const int i = (lo + hi) / 2, j = k - i;
            if (j > 0 && !less(b[j - 1], a[i])) lo = i + 1;
            else hi = i;
        }
        return lo;
    };
#pragma omp parallel for schedule(static)
    for (int blk = 0; blk < kScanBlocks; ++blk) {
        const int k0 = std::min(n, blk * chunk), k1 = std::min(n, k0 + chunk);
        if (k0 == k1) continue;
        const int i0 = corank(k0), i1 = corank(k1);
        std::merge(a + i0, a + i1, b + (k0 - i0), b + (k1 - i1), out + k0, less);
    }
}

// Marks that several threads may set at once (relaxed byte stores).
using MarkArray = std::vector<std::atomic<unsigned char>>;

// The indices k in [lo, hi) with mark[k] set, in increasing order, appended to out; the marks are cleared.
inline void take_marked(MarkArray& mark, int lo, int hi, std::vector<int>& out) {
    const int n = hi - lo, base = static_cast<int>(out.size());
    if (n < 4 * kOmpMinWork) {
        for (int k = lo; k < hi; ++k)
            if (mark[static_cast<size_t>(k)].load(std::memory_order_relaxed)) {
                out.push_back(k);
                mark[static_cast<size_t>(k)].store(0, std::memory_order_relaxed);
            }
        return;
    }
    const int chunk = (n + kScanBlocks - 1) / kScanBlocks;
    int pos[kScanBlocks + 1];
    pos[0] = 0;
#pragma omp parallel for schedule(static)
    for (int b = 0; b < kScanBlocks; ++b) {
        const int k0 = lo + std::min(n, b * chunk), k1 = lo + std::min(n, (b + 1) * chunk);
        int cnt = 0;
        for (int k = k0; k < k1; ++k) cnt += mark[static_cast<size_t>(k)].load(std::memory_order_relaxed) ? 1 : 0;
        pos[b + 1] = cnt;
    }
    for (int b = 0; b < kScanBlocks; ++b) pos[b + 1] += pos[b];
    out.resize(static_cast<size_t>(base) + pos[kScanBlocks]);
#pragma omp parallel for schedule(static)
    for (int b = 0; b < kScanBlocks; ++b) {
        const int k0 = lo + std::min(n, b * chunk), k1 = lo + std::min(n, (b + 1) * chunk);
        int w = base + pos[b];
        for (int k = k0; k < k1; ++k)
            if (mark[static_cast<size_t>(k)].load(std::memory_order_relaxed)) {
                out[static_cast<size_t>(w++)] = k;
                mark[static_cast<size_t>(k)].store(0, std::memory_order_relaxed);
            }
    }
}

}   // namespace fv
