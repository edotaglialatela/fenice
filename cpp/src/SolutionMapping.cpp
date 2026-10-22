#include "fv/SolutionMapping.hpp"

#include <stdexcept>

namespace fv {

std::vector<double> linspace(double a, double b, int n) {
    std::vector<double> v(std::max(n, 0));
    if (n == 1) {
        v[0] = a;
        return v;
    }
    const double step = (b - a) / (n - 1);
    for (int k = 0; k < n; ++k) v[k] = k * step + a;
    if (n > 1) v[n - 1] = b;
    return v;
}

namespace {
// scipy _find_indices: interval index (searchsorted left - 1, clamped) and
// normalised distance (unclamped -> linear extrapolation).
inline void find_index(const std::vector<double>& g, double x, int& i, double& t) {
    const int n = static_cast<int>(g.size());
    if (n < 2) {
        i = 0;
        t = 0.0;
        return;
    }
    i = static_cast<int>(std::lower_bound(g.begin(), g.end(), x) - g.begin()) - 1;
    if (i < 0) i = 0;
    if (i > n - 2) i = n - 2;
    t = (x - g[i]) / (g[i + 1] - g[i]);
}
}   // namespace

State GridInterp2D::operator()(double x, double y) const {
    const int nx = static_cast<int>(xs_.size());
    const int ny = static_cast<int>(ys_.size());
    int i, j;
    double tx, ty;
    find_index(xs_, x, i, tx);
    find_index(ys_, y, j, ty);
    const int i1 = (nx > 1) ? i + 1 : i;
    const int j1 = (ny > 1) ? j + 1 : j;
    const State& a = v_[j * nx + i];
    const State& b = v_[j * nx + i1];
    const State& c = v_[j1 * nx + i];
    const State& d = v_[j1 * nx + i1];
    State out;
    for (int q = 0; q < kNVar; ++q)
        out[q] = a[q] * (1 - ty) * (1 - tx) + b[q] * (1 - ty) * tx + c[q] * ty * (1 - tx) + d[q] * ty * tx;
    return out;
}

Field map_solution(const Field& U_coarse, int nx_c, int ny_c, int nx_f, int ny_f, double Lx, double Ly) {
    const double dx_c = Lx / nx_c, dy_c = Ly / ny_c;
    const double dx_f = Lx / nx_f, dy_f = Ly / ny_f;
    GridInterp2D interp(linspace(dx_c / 2.0, Lx - dx_c / 2.0, nx_c), linspace(dy_c / 2.0, Ly - dy_c / 2.0, ny_c),
                        U_coarse);
    const auto x_f = linspace(dx_f / 2.0, Lx - dx_f / 2.0, nx_f);
    const auto y_f = linspace(dy_f / 2.0, Ly - dy_f / 2.0, ny_f);
    Field U_fine(static_cast<size_t>(nx_f) * ny_f);
#pragma omp parallel for schedule(static) if (nx_f * ny_f >= kOmpMinWork)
    for (int j = 0; j < ny_f; ++j)
        for (int i = 0; i < nx_f; ++i) U_fine[j * nx_f + i] = interp(x_f[i], y_f[j]);
    return U_fine;
}

Field map_solution_3d(const Field& U_c, int nx_c, int ny_c, int nz_c, int nx_f, int ny_f, int nz_f, double Lx,
                      double Ly, double Lz) {
    auto centres = [](double L, int n) {
        const double h = L / n;
        return linspace(h / 2.0, L - h / 2.0, n);
    };
    const auto xc = centres(Lx, nx_c), yc = centres(Ly, ny_c), zc = centres(Lz, nz_c);
    const auto xf = centres(Lx, nx_f), yf = centres(Ly, ny_f), zf = centres(Lz, nz_f);
    // Interval index and weight per axis (scipy RegularGridInterpolator: clamped index, extrapolating weight)
    auto locate = [](const std::vector<double>& g, double x, int& i, double& t) {
        const int n = static_cast<int>(g.size());
        if (n < 2) {
            i = 0;
            t = 0.0;
            return;
        }
        i = static_cast<int>(std::lower_bound(g.begin(), g.end(), x) - g.begin()) - 1;
        i = std::min(std::max(i, 0), n - 2);
        t = (x - g[i]) / (g[i + 1] - g[i]);
    };
    Field U_f(static_cast<size_t>(nx_f) * ny_f * nz_f);
#pragma omp parallel for schedule(static) if (nx_f * ny_f * nz_f >= kOmpMinWork)
    for (int k = 0; k < nz_f; ++k)
        for (int j = 0; j < ny_f; ++j)
            for (int i = 0; i < nx_f; ++i) {
                int ix, iy, iz;
                double tx, ty, tz;
                locate(xc, xf[i], ix, tx);
                locate(yc, yf[j], iy, ty);
                locate(zc, zf[k], iz, tz);
                const int ix1 = nx_c > 1 ? ix + 1 : ix, iy1 = ny_c > 1 ? iy + 1 : iy, iz1 = nz_c > 1 ? iz + 1 : iz;
                State out{};
                for (int c = 0; c < 8; ++c) {
                    const int a = (c & 4) ? iz1 : iz, b = (c & 2) ? iy1 : iy, d = (c & 1) ? ix1 : ix;
                    const double w = ((c & 4) ? tz : 1 - tz) * ((c & 2) ? ty : 1 - ty) * ((c & 1) ? tx : 1 - tx);
                    const State& v = U_c[(static_cast<size_t>(a) * ny_c + b) * nx_c + d];
                    for (int q = 0; q < kNVar; ++q) out[q] += v[q] * w;
                }
                U_f[(static_cast<size_t>(k) * ny_f + j) * nx_f + i] = out;
            }
    return U_f;
}

Field restrict_patch(const Field& U_fine, int nx_f, int ny_f, int nx_c, int ny_c) {
    const int rx = nx_f / nx_c, ry = ny_f / ny_c;
    if (rx * nx_c != nx_f || ry * ny_c != ny_f)
        throw std::invalid_argument("Fine grid is not an integer multiple of coarse grid.");
    const double inv = 1.0 / (rx * ry);
    Field U_c(static_cast<size_t>(nx_c) * ny_c);
    for (int jc = 0; jc < ny_c; ++jc)
        for (int ic = 0; ic < nx_c; ++ic) {
            State s = {0, 0, 0, 0};
            for (int jj = jc * ry; jj < (jc + 1) * ry; ++jj)
                for (int ii = ic * rx; ii < (ic + 1) * rx; ++ii)
                    for (int q = 0; q < kNVar; ++q) s[q] += U_fine[jj * nx_f + ii][q];
            for (int q = 0; q < kNVar; ++q) U_c[jc * nx_c + ic][q] = inv * s[q];
        }
    return U_c;
}

}   // namespace fv
