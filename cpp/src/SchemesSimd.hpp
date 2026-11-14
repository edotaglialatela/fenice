// Internal: the AVX2 kernels (SchemesAvx2.cpp): the batch of the Riemann solvers and the viscous cell kernels of the AMR engine.
// They compute, lane by lane, the same sequence of IEEE operations as the scalar code, so their results are bit-identical.
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
void riemann_batch_avx2(const EOS& eos, RiemannType type, double ma_cut, double sb, const State* UL, const State* UR, const int* axis,
                        int n, State* F, double* smax);
// The same for faces whose states come from the MUSCL reconstruction: face k reads the primitive states W[st[4k]] .. W[st[4k+3]]
// (the four points of its stencil) and applies the limiter `lim` before the Riemann solver, as the scalar reconstruction does.
void muscl_riemann_batch_avx2(const EOS& eos, RiemannType type, double ma_cut, double sb, Limiter lim, const State* W, const int* st,
                              const int* axis, int n, State* F, double* smax, bool planar);
// Viscous cell kernels of the AMR engine for the leaves c0 .. c0+n-1 (n a multiple of 4), bit-identical to the scalar functions of
// CellFlux.cpp; the arguments are those of viscous_cell_quantities_run and viscous_speed_range (CellFlux.hpp), with `out` the seven
// arrays of ViscousCellArrays in their order (u, v, T, du/dx, du/dy, dv/dx, dv/dy).
void viscous_cells_avx2(const EOS& eos, const State* U, const int* nbr, const int* level, const double* dxl, const double* dyl,
                        int c0, int n, double* const* out);
double viscous_speed_avx2(const EOS& eos, double mu, double k_cond, const State* U, const int* level, const double* dxl,
                          const double* dyl, int c0, int n);
// Cell quantities of the uniform 2-D viscous row sweep (NavierStokes::viscous_residual_rows) for the columns i0 .. i0+n-1 of one row
// (n a multiple of 4, 1 <= i0, i0+n <= nx-1: the east / west neighbours are cells of the row): the seven arrays of `out` (u, v, T,
// du/dx, du/dy, dv/dx, dv/dy) at index i, and the largest wave-speed bound of the cells (0 if none is positive). `north` / `south`
// point at the cells above / below column 0. Bit-identical to the scalar loop of the row sweep.
double viscous_row_cells_avx2(const EOS& eos, double mu, double k_cond, double L, double inv_h2, double dx, double dy, const State* cur,
                              const State* north, const State* south, int i0, int n, double* const* out);
// Interior viscous face fluxes of the same sweep, components 1..3 (the others are zero), bit-identical to its lambdas.  X faces
// i0 .. i0+n-1 (n a multiple of 4, 1 <= i0, i0+n <= nx) of one row from the arrays of the row (u, v, T, du/dy, dv/dy), written at
// index i; y faces i0 .. i0+n-1 between two rows, `B` / `T` = their (u, v, T, du/dx, dv/dx) arrays.
void viscous_row_xfaces_avx2(double mu, double k_cond, double dx, double dy, const double* u, const double* v, const double* T,
                             const double* dudy, const double* dvdy, int i0, int n, double* f1, double* f2, double* f3);
void viscous_row_yfaces_avx2(double mu, double k_cond, double dx, double dy, const double* const* B, const double* const* T, int i0,
                             int n, double* g1, double* g2, double* g3);
#else
inline bool avx2_available() { return false; }
#endif

}   // namespace detail
}   // namespace fv
