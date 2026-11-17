// Internal: the vector type of the kernels of SchemesSimd.cpp and the few operations they are written with, for each instruction
// set: four doubles per value (lanes 0 .. 3), as one AVX2 register on x86-64 or two NEON registers on ARM64.
//
// The contract of the kernels is bit-identity with the scalar code, so the operations are only those every backend computes as
// the same IEEE operation per lane: add / sub / mul / div / sqrt, ordered comparisons that give a mask (all ones or all zeros
// per lane; false if an operand is NaN), bit logic, and the selection of a lane by a mask. No fused multiply-add and no vector
// min / max (written as a comparison and a selection, as std::min / std::max are). A backend is one block of this file; the
// kernels use nothing else, so a new instruction set is a new block here and a line in the build.
#pragma once

#include <cstdint>

#include "SchemesSimd.hpp"   // FV_SIMD and the instruction set of the build

#if defined(FV_SIMD_BACKEND_AVX2)
#include <immintrin.h>
#elif defined(FV_SIMD_BACKEND_NEON)
#include <arm_neon.h>
#endif

#ifdef FV_SIMD

namespace fv {
namespace detail {
namespace simd {

#if defined(FV_SIMD_BACKEND_AVX2)

using V = __m256d;

inline V set1(double x) { return _mm256_set1_pd(x); }
// The value with lanes 0 .. 3 = a0 .. a3
inline V set4(double a0, double a1, double a2, double a3) { return _mm256_set_pd(a3, a2, a1, a0); }
// Lanes 0 .. 3 from / to p[0] .. p[3] (any alignment)
inline V load(const double* p) { return _mm256_loadu_pd(p); }
inline void store(double* p, V a) { _mm256_storeu_pd(p, a); }
// Lanes 0 .. 3 to p[0] .. p[3] past the caches (p aligned to 32 bytes); stream_fence() makes them visible to the other threads
inline void store_stream(double* p, V a) { _mm256_stream_pd(p, a); }
inline void stream_fence() { _mm_sfence(); }
inline V add(V a, V b) { return _mm256_add_pd(a, b); }
inline V sub(V a, V b) { return _mm256_sub_pd(a, b); }
inline V mul(V a, V b) { return _mm256_mul_pd(a, b); }
inline V dv(V a, V b) { return _mm256_div_pd(a, b); }
inline V vsqrt(V a) { return _mm256_sqrt_pd(a); }
inline V lt(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
inline V ge(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_GE_OQ); }
inline V le(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); }
inline V band(V a, V b) { return _mm256_and_pd(a, b); }
inline V bor(V a, V b) { return _mm256_or_pd(a, b); }
// (not a) and b
inline V andnot(V a, V b) { return _mm256_andnot_pd(a, b); }
// mask ? b : a, lane by lane (`mask` all ones or all zeros per lane)
inline V blend(V a, V b, V mask) { return _mm256_blendv_pd(a, b, mask); }
// The mask that is set in every lane, or in none
inline V mask_all(bool on) { return _mm256_castsi256_pd(_mm256_set1_epi64x(on ? -1 : 0)); }
// The mask of the lanes i with a[i] == value
inline V mask_int_eq(const int* a, int value) {
    const __m256i a64 = _mm256_cvtepi32_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a)));
    return _mm256_castsi256_pd(_mm256_cmpeq_epi64(a64, _mm256_set1_epi64x(value)));
}
// At the end of a kernel, before scalar code runs again (the upper halves of the registers are cleared: no transition penalty)
inline void end_batch() { _mm256_zeroupper(); }

#elif defined(FV_SIMD_BACKEND_NEON)

struct V {
    float64x2_t lo, hi;   // lanes 0, 1 and 2, 3
};

namespace neon {
inline float64x2_t f(uint64x2_t m) { return vreinterpretq_f64_u64(m); }
inline uint64x2_t u(float64x2_t x) { return vreinterpretq_u64_f64(x); }
}   // namespace neon

inline V set1(double x) { return {vdupq_n_f64(x), vdupq_n_f64(x)}; }
// The value with lanes 0 .. 3 = a0 .. a3
inline V set4(double a0, double a1, double a2, double a3) {
    const double lo[2] = {a0, a1}, hi[2] = {a2, a3};
    return {vld1q_f64(lo), vld1q_f64(hi)};
}
// Lanes 0 .. 3 from / to p[0] .. p[3] (any alignment)
inline V load(const double* p) { return {vld1q_f64(p), vld1q_f64(p + 2)}; }
inline void store(double* p, V a) {
    vst1q_f64(p, a.lo);
    vst1q_f64(p + 2, a.hi);
}
// No store past the caches on this instruction set: an ordinary store
inline void store_stream(double* p, V a) { store(p, a); }
inline void stream_fence() {}
inline V add(V a, V b) { return {vaddq_f64(a.lo, b.lo), vaddq_f64(a.hi, b.hi)}; }
inline V sub(V a, V b) { return {vsubq_f64(a.lo, b.lo), vsubq_f64(a.hi, b.hi)}; }
inline V mul(V a, V b) { return {vmulq_f64(a.lo, b.lo), vmulq_f64(a.hi, b.hi)}; }
inline V dv(V a, V b) { return {vdivq_f64(a.lo, b.lo), vdivq_f64(a.hi, b.hi)}; }
inline V vsqrt(V a) { return {vsqrtq_f64(a.lo), vsqrtq_f64(a.hi)}; }
inline V lt(V a, V b) { return {neon::f(vcltq_f64(a.lo, b.lo)), neon::f(vcltq_f64(a.hi, b.hi))}; }
inline V ge(V a, V b) { return {neon::f(vcgeq_f64(a.lo, b.lo)), neon::f(vcgeq_f64(a.hi, b.hi))}; }
inline V le(V a, V b) { return {neon::f(vcleq_f64(a.lo, b.lo)), neon::f(vcleq_f64(a.hi, b.hi))}; }
inline V band(V a, V b) {
    return {neon::f(vandq_u64(neon::u(a.lo), neon::u(b.lo))), neon::f(vandq_u64(neon::u(a.hi), neon::u(b.hi)))};
}
inline V bor(V a, V b) {
    return {neon::f(vorrq_u64(neon::u(a.lo), neon::u(b.lo))), neon::f(vorrq_u64(neon::u(a.hi), neon::u(b.hi)))};
}
// (not a) and b
inline V andnot(V a, V b) {
    return {neon::f(vbicq_u64(neon::u(b.lo), neon::u(a.lo))), neon::f(vbicq_u64(neon::u(b.hi), neon::u(a.hi)))};
}
// mask ? b : a, lane by lane (`mask` all ones or all zeros per lane)
inline V blend(V a, V b, V mask) { return {vbslq_f64(neon::u(mask.lo), b.lo, a.lo), vbslq_f64(neon::u(mask.hi), b.hi, a.hi)}; }
// The mask that is set in every lane, or in none
inline V mask_all(bool on) {
    const uint64x2_t m = vdupq_n_u64(on ? ~static_cast<uint64_t>(0) : 0);
    return {neon::f(m), neon::f(m)};
}
// The mask of the lanes i with a[i] == value
inline V mask_int_eq(const int* a, int value) {
    uint64_t m[4];
    for (int i = 0; i < 4; ++i) m[i] = a[i] == value ? ~static_cast<uint64_t>(0) : 0;
    return {neon::f(vld1q_u64(m)), neon::f(vld1q_u64(m + 2))};
}
// At the end of a kernel, before scalar code runs again (nothing to do with this instruction set)
inline void end_batch() {}

#endif

// std::max(a, b) = (a < b) ? b : a   and   std::min(a, b) = (b < a) ? b : a
inline V vmax(V a, V b) { return blend(a, b, lt(a, b)); }
inline V vmin(V a, V b) { return blend(a, b, lt(b, a)); }
inline V vabs(V a) { return andnot(set1(-0.0), a); }
inline V vcopysign(V x, V y) {
    const V sign = set1(-0.0);
    return bor(andnot(sign, x), band(sign, y));
}

}   // namespace simd
}   // namespace detail
}   // namespace fv

#endif   // FV_SIMD
