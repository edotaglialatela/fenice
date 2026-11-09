#pragma once
// The viscous wave-speed bound of one cell, written once for every scalar copy (uniform 2-D and 3-D, fused rows, AMR 2-D and 3-D).
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

}  // namespace fv
