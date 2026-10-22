// Mesh-to-mesh interpolation (scipy RegularGridInterpolator semantics) and
// conservative restriction.
#pragma once

#include "fv/Core.hpp"

namespace fv {

// numpy.linspace(a, b, n) (last point set exactly to b).
std::vector<double> linspace(double a, double b, int n);

// Bilinear interpolation on a rectilinear grid of cell values, with linear
// extrapolation outside the grid (RegularGridInterpolator, method='linear',
// bounds_error=False, fill_value=None).  values[j*xs.size() + i] sits at (xs[i], ys[j]).
class GridInterp2D {
public:
    // Grid coordinates and the values on it (kept by reference: must outlive the interpolator).
    GridInterp2D(std::vector<double> xs, std::vector<double> ys, const Field& values)
        : xs_(std::move(xs)), ys_(std::move(ys)), v_(values) {}
    // Interpolated state at (x, y).
    State operator()(double x, double y) const;

private:
    std::vector<double> xs_, ys_;
    const Field& v_;
};

// Bilinear coarse -> fine mapping over the same [0,Lx]x[0,Ly] domain.
Field map_solution(const Field& U_coarse, int nx_c, int ny_c, int nx_f, int ny_f, double Lx, double Ly);

// Trilinear coarse -> fine mapping over the same [0,Lx]x[0,Ly]x[0,Lz] box (3-D hot start).
Field map_solution_3d(const Field& U_coarse, int nx_c, int ny_c, int nz_c, int nx_f, int ny_f, int nz_f, double Lx,
                      double Ly, double Lz);

}   // namespace fv
