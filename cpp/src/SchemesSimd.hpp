// Internal: the AVX2 batch of the Riemann solvers (SchemesAvx2.cpp).  The batch computes, lane by lane, the same
// sequence of IEEE operations as RiemannSolver::face_flux, so its results are bit-identical to the scalar ones.
#pragma once

#include "fv/Schemes.hpp"

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(FV_NO_SIMD)
#define FV_SIMD_AVX2 1
#endif

namespace fv {
namespace detail {

#ifdef FV_SIMD_AVX2
// True if this CPU and operating system run AVX2 code and the user has not switched it off (FENICE_NO_SIMD).
bool avx2_available();
// Fluxes and wave speeds of the faces 0 .. n-1 (n a multiple of 4).
void riemann_batch_avx2(const EOS& eos, RiemannType type, double ma_cut, const State* UL, const State* UR, const int* axis,
                        int n, State* F, double* smax);
// The same for faces whose states come from the MUSCL reconstruction: face k reads the primitive states W[st[4k]] .. W[st[4k+3]]
// (the four points of its stencil) and applies the limiter `lim` before the Riemann solver, as the scalar reconstruction does.
void muscl_riemann_batch_avx2(const EOS& eos, RiemannType type, double ma_cut, Limiter lim, const State* W, const int* st,
                              const int* axis, int n, State* F, double* smax);
#else
inline bool avx2_available() { return false; }
#endif

}   // namespace detail
}   // namespace fv
