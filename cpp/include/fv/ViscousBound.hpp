#pragma once
// The viscous wave-speed bound of one cell and the lengths it is made with, written once for every scalar copy (uniform 2-D and 3-D, fused rows, AMR 2-D and 3-D).
// The operation order is part of the bit-reproducibility rule: do not reorder.  Do NOT include this header from SchemesAvx2.cpp
// (that file is compiled for AVX2; its twin `viscous_speed_avx2` is written by hand and guarded by
// `viscous_cell_kernels_equal_scalar_bit_for_bit`).
#include <algorithm>

namespace fv {

// D = max(4/3 mu/rho, k/(rho c_v)) + 1e-12, the diffusivity of the viscous stencil (k_cond (gamma - 1) / rho = k / (rho c_v) for the
// ideal gas).
inline double viscous_nu_eff(double mu, double k_cond, double gamma, double rho) {
    return std::max((4.0 / 3.0) * (mu / rho), (k_cond * (gamma - 1.0)) / rho) + 1e-12;
}

// Speed bound of a cell: convective part (largest |velocity component| + sound speed) plus the viscous eigenvalue, which counts twice
// (2 L sum(1/h^2) D) in the shared stability budget; L = dt_length of the cell.
inline double viscous_speed_bound(double max_abs_velocity, double sound_speed, double L, double inv_h2, double nu_eff) {
    return max_abs_velocity + sound_speed + 2.0 * L * inv_h2 * nu_eff;
}

// The length of the time step, L in dt = CFL L / S: min(dx, dy) in 2-D; in 3-D 2 / sum(1/h), so that the summed directional Courant
// number stays at 2 CFL as for a 2-D square cell (Simulation::dt_length).
inline double step_length_2d(double dx, double dy) { return std::min(dx, dy); }
inline double step_length_3d(double dx, double dy, double dz) { return 2.0 / (1.0 / dx + 1.0 / dy + 1.0 / dz); }

// sum of 1/h^2 over the axes: the viscous stencil of the cell.
inline double inv_h2_2d(double dx, double dy) { return 1.0 / (dx * dx) + 1.0 / (dy * dy); }
inline double inv_h2_3d(double dx, double dy, double dz) { return 1.0 / (dx * dx) + 1.0 / (dy * dy) + 1.0 / (dz * dz); }

}  // namespace fv
