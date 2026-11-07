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

// Trilinear interpolation of cell values on a rectilinear grid (the arithmetic of map_solution_3d: clamped interval, extrapolating
// weight), values[(k * ny + j) * nx + i] at (xs[i], ys[j], zs[k]): the 3-D hot start of an adaptive mesh samples it at the leaf centres.
class GridInterp3D {
public:
    GridInterp3D(std::vector<double> xs, std::vector<double> ys, std::vector<double> zs, const Field& values)
        : xs_(std::move(xs)), ys_(std::move(ys)), zs_(std::move(zs)), v_(values) {}
    // Interpolated state at (x, y, z).
    State operator()(double x, double y, double z) const;

private:
    std::vector<double> xs_, ys_, zs_;
    const Field& v_;
};

// 3-D version of leaves_to_uniform below: leaves (level, i, j, k) of an nx_base x ny_base x nz_base base mesh onto the uniform mesh of
// the finest level, (k * ny_out + j) * nx_out + i.
Field leaves_to_uniform_3d(const Field& U, const std::vector<int>& level, const std::vector<int>& ci, const std::vector<int>& cj,
                           const std::vector<int>& ck, int nx_base, int ny_base, int nz_base, int& nx_out, int& ny_out, int& nz_out);

// The state of an adaptive run on the uniform mesh of its finest level: every leaf (level, integer coordinates on its level's
// grid of nx_base * 2^level by ny_base * 2^level cells) fills the block of finest cells it covers with its value. nx_out /
// ny_out return the size of that uniform mesh (the base mesh times 2^finest level).
Field leaves_to_uniform(const Field& U, const std::vector<int>& level, const std::vector<int>& ci, const std::vector<int>& cj,
                        int nx_base, int ny_base, int& nx_out, int& ny_out);

// Bilinear coarse -> fine mapping over the same [0,Lx]x[0,Ly] domain.
Field map_solution(const Field& U_coarse, int nx_c, int ny_c, int nx_f, int ny_f, double Lx, double Ly);

// Trilinear coarse -> fine mapping over the same [0,Lx]x[0,Ly]x[0,Lz] box (3-D hot start).
Field map_solution_3d(const Field& U_coarse, int nx_c, int ny_c, int nz_c, int nx_f, int ny_f, int nz_f, double Lx,
                      double Ly, double Lz);

}   // namespace fv
