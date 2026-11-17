// Internal: the vector kernels (SchemesSimd.cpp; AVX2 on x86-64, NEON on ARM64): the batch of the Riemann solvers and the viscous
// kernels of the uniform and the AMR engine.
// They compute, lane by lane, the same sequence of IEEE operations as the scalar code, so their results are bit-identical.
#pragma once

#include "fv/Schemes.hpp"

// The instruction set of the build (none with FV_NO_SIMD or on another processor): the kernels are written once, on the operations
// of SimdOps.hpp, which has one block per set.
#if !defined(FV_NO_SIMD) && (defined(__x86_64__) || defined(_M_X64))
#define FV_SIMD 1
#define FV_SIMD_BACKEND_AVX2 1
#elif !defined(FV_NO_SIMD) && defined(__aarch64__)
#define FV_SIMD 1
#define FV_SIMD_BACKEND_NEON 1
#endif

namespace fv {
namespace detail {

#ifdef FV_SIMD
// True if this CPU and operating system run the vector kernels (AVX2 is checked at run time, NEON is part of every ARM64 CPU) and
// the user has not switched them off (FENICE_NO_SIMD).
bool simd_available();
// Fluxes and wave speeds of the faces 0 .. n-1 (n a multiple of 4).
void riemann_batch_simd(const EOS& eos, RiemannType type, double sb, const State* UL, const State* UR, const int* axis,
                        int n, State* F, double* smax);
// The same for faces whose states come from the MUSCL reconstruction: face k reads the primitive states W[st[4k]] .. W[st[4k+3]]
// (the four points of its stencil) and applies the limiter `lim` before the Riemann solver, as the scalar reconstruction does.
void muscl_riemann_batch_simd(const EOS& eos, RiemannType type, double sb, Limiter lim, const State* W, const int* st,
                              const int* axis, int n, State* F, double* smax, bool planar);
// The same for the faces of one line of the uniform mesh, normal to `axis`: the primitive state is stored per variable (plane q at
// W + q * plane), face k of the batch has its stencil at the indices base + k + point * stride, point 0 .. 3. n a multiple of 4.
void muscl_riemann_line_simd(const EOS& eos, RiemannType type, double sb, Limiter lim, const double* W, size_t plane, size_t base,
                             size_t stride, int axis, int n, State* F, double* smax, bool planar);
// Primitive states (EOS::cons_to_prim) of the cells U[0 .. n-1] (n a multiple of 4), the variable q at out[q][k].
void cons_to_prim_planes_simd(const EOS& eos, const State* U, int n, double* const* out);
// The same for WENO3-Z (weno3z of Schemes.cpp): four cells per face, planes[q][base + k + s * stride], s = 0 .. 3.
void weno3_line_simd(const EOS& eos, double eps_rel, const double* const* planes, size_t base, size_t stride, int n, State* UL,
                     State* UR);
// Conservative WENO5-Z states of the faces of one line (MusclReconstruction::reconstruct_weno): face k of the batch (n a multiple of 4)
// has its six cells at planes[q][base + k + s * stride], s = 0 .. 5, one plane per primitive variable q = 0 .. 4; UL[k] / UR[k] receive
// the left / right state, bit-identical to eos.prim_to_cons of weno5z(c0 .. c4) and weno5z(c5 .. c1) per variable (Schemes.cpp).
void weno5_line_simd(const EOS& eos, double eps_rel, const double* const* planes, size_t base, size_t stride, int n, State* UL,
                     State* UR);
// WENO3-Z (`five` false: four cells per face) or WENO5-Z (`five`: six) conservative states of n faces (a multiple of 4) of the adaptive
// engine: face k has its primitive cells at W[st[ns * k + p]], p = 0 .. ns - 1 (ns = 4 or 6, the cells in order along the normal);
// UL[k] / UR[k] are bit-identical to MusclReconstruction::weno_face_states' scalar loop (weno3_value / weno5_value, prim_to_cons).
void weno_gather_simd(const EOS& eos, double eps_rel, bool five, const State* W, const int* st, int n, State* UL, State* UR);
// Viscous cell kernels of the AMR engine for the leaves c0 .. c0+n-1 (n a multiple of 4), bit-identical to the scalar functions of
// CellFlux.cpp; the arguments are those of viscous_cell_quantities_run and viscous_speed_range (CellFlux.hpp), with `out` the seven
// arrays of ViscousCellArrays in their order (u, v, T, du/dx, du/dy, dv/dx, dv/dy).
void viscous_cells_simd(const EOS& eos, const State* U, const int* nbr, const int* level, const double* dxl, const double* dyl,
                        int c0, int n, double* const* out);
double viscous_speed_simd(const EOS& eos, double mu, double k_cond, const State* U, const int* level, const double* dxl,
                          const double* dyl, int c0, int n);
// Cell quantities of the uniform 2-D viscous row sweep (NavierStokes::viscous_residual_rows) for the columns i0 .. i0+n-1 of one row
// (n a multiple of 4, 1 <= i0, i0+n <= nx-1: the east / west neighbours are cells of the row): the seven arrays of `out` (u, v, T,
// du/dx, du/dy, dv/dx, dv/dy) at index i, and the largest wave-speed bound of the cells (0 if none is positive). `north` / `south`
// point at the cells above / below column 0. Bit-identical to the scalar loop of the row sweep.
double viscous_row_cells_simd(const EOS& eos, double mu, double k_cond, double L, double inv_h2, double dx, double dy, const State* cur,
                              const State* north, const State* south, int i0, int n, double* const* out);
// Interior viscous face fluxes of the same sweep, components 1..3 (the others are zero), bit-identical to its lambdas.  X faces
// i0 .. i0+n-1 (n a multiple of 4, 1 <= i0, i0+n <= nx) of one row from the arrays of the row (u, v, T, du/dy, dv/dy), written at
// index i; y faces i0 .. i0+n-1 between two rows, `B` / `T` = their (u, v, T, du/dx, dv/dx) arrays.
void viscous_row_xfaces_simd(double mu, double k_cond, double dx, double dy, const double* u, const double* v, const double* T,
                             const double* dudy, const double* dvdy, int i0, int n, double* f1, double* f2, double* f3);
void viscous_row_yfaces_simd(double mu, double k_cond, double dx, double dy, const double* const* B, const double* const* T, int i0,
                             int n, double* g1, double* g2, double* g3);
// Interior viscous face fluxes of the uniform 3-D sweep (NavierStokes::viscous_residual_3d) for n faces (a multiple of 4) normal to axis d,
// consecutive along x, bit-identical to its scalar face code. Every pointer addresses the LEFT cell of the first face (velocity
// components, temperature, the four cross-derivative arrays in CrossGhosts order); the right cell of a face is `delta` cells further.
void viscous_faces3d_simd(int d, double mu, double k_cond, double h, double area, const double* const* vel, const double* T,
                          const double* const* gc, long delta, int n, State* out);
// Cell quantities of the uniform 3-D viscous sweep for n cells (a multiple of 4) consecutive along x: velocity (3 arrays), temperature
// (`out`, 4 arrays) and the nine G[a][b] = (d(rho u_a)/dx_b - u_a d(rho)/dx_b) / rho (`g`, index 3 a + b) from the central differences
// of the conservative state. `W`: primitive state of the first cell; `xm`, `xp`, `ym`, `yp`, `zm`, `zp`: the conservative
// neighbour cell of the first cell on each side. Bit-identical to the scalar loop of NavierStokes::viscous_residual_3d.
void viscous_cells3d_simd(double dx, double dy, double dz, const State* W, const State* xm, const State* xp, const State* ym,
                          const State* yp, const State* zm, const State* zp, int n, double* const* out, double* const* g);
// out[i] = a[i] + s * b[i] for i = 0 .. n-1, written past the caches (non-temporal stores: no read of the old `out`, which is a pure
// output of the time update). `out` aligned to 32 bytes, n a multiple of 4; `out` must not overlap `a` or `b`. Bit-identical to the
// scalar loop.
void axpy_stream_simd(double* out, const double* a, double s, const double* b, size_t n);
#else
inline bool simd_available() { return false; }
#endif

}   // namespace detail
}   // namespace fv
