// Internal: the AVX2 kernels (SchemesAvx2.cpp): the batch of the Riemann solvers and the viscous cell kernels of the AMR engine.
// They compute, lane by lane, the same sequence of IEEE operations as the scalar code, so their results are bit-identical.
#pragma once

#include "fv/Schemes.hpp"

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(FV_NO_SIMD)
#define FV_SIMD_AVX2 1
#endif
// The viscous cell kernels of the AMR engine are built with gcc / clang only: with MSVC they made the stepping slower than the scalar
// code on the same CPU (Ryzen 7 7800X3D, airfoil 720x360: +0.42 s with MSVC, -0.53 s with gcc; docs/design/amr-optimisation.md).
#if defined(FV_SIMD_AVX2) && !defined(_MSC_VER)
#define FV_SIMD_VISCOUS 1
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
#ifdef FV_SIMD_VISCOUS
// Viscous cell kernels of the AMR engine for the leaves c0 .. c0+n-1 (n a multiple of 4), bit-identical to the scalar functions of
// CellFlux.cpp; the arguments are those of viscous_cell_quantities_run and viscous_speed_range (CellFlux.hpp), with `out` the seven
// arrays of ViscousCellArrays in their order (u, v, T, du/dx, du/dy, dv/dx, dv/dy).
void viscous_cells_avx2(const EOS& eos, const State* U, const int* nbr, const int* level, const double* dxl, const double* dyl,
                        int c0, int n, double* const* out);
double viscous_speed_avx2(const EOS& eos, double mu, double k_cond, const State* U, const int* level, const double* dxl,
                          const double* dyl, int c0, int n);
#endif
#else
inline bool avx2_available() { return false; }
#endif

}   // namespace detail
}   // namespace fv
