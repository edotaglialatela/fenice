#pragma once
// Internal helpers of the Simulation translation units (Simulation.cpp, SimulationRestart.cpp, SimulationRun.cpp): moved out of the
// anonymous namespaces of the old single file, unchanged. Not part of the public API.

#include "fv/Simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <map>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "fv/Geometry.hpp"

namespace fs = std::filesystem;

namespace fv {
namespace sim_detail {


using Clock = std::chrono::steady_clock;

inline double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

inline std::string to_prefix(std::string s) {
    for (auto& c : s) c = (c == ' ') ? '_' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline double next_time_output(double time, double interval) {
    if (!(interval > 0.0)) return std::numeric_limits<double>::infinity();
    const double quotient = time / interval;
    const double tolerance = 8.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(quotient));
    return (std::floor(quotient + tolerance) + 1.0) * interval;
}

inline bool reached_time_output(double time, double target, double interval) {
    if (!std::isfinite(target)) return false;
    const double tolerance = 8.0 * std::numeric_limits<double>::epsilon() *
                             std::max({std::abs(time), std::abs(target), std::abs(interval)});
    return std::abs(time - target) <= tolerance;
}

inline std::vector<std::pair<double, double>> read_time_table(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot read boundary time table " + path.string());
    std::vector<std::pair<double, double>> values;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        if (line_number == 1 && line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
            static_cast<unsigned char>(line[1]) == 0xBB && static_cast<unsigned char>(line[2]) == 0xBF)
            line.erase(0, 3);   // UTF-8 BOM, commonly emitted by spreadsheet CSV exports
        const size_t first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        for (char& c : line)
            if (c == ',') c = ' ';
        std::istringstream row(line);
        double time = 0.0, value = 0.0;
        if (!(row >> time >> value)) {
            if (values.empty()) {
                std::istringstream header(line);
                std::string first, second, extra;
                if ((header >> first >> second) && !(header >> extra) &&
                    (lower(first) == "t" || lower(first) == "time") && lower(second) == "value")
                    continue;   // optional CSV header, also accepted after blank/comment lines
            }
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) + " must contain t,value");
        }
        std::string extra;
        if (row >> extra) throw std::runtime_error(path.string() + ":" + std::to_string(line_number) + " has more than two columns");
        if (!std::isfinite(time) || !std::isfinite(value))
            throw std::runtime_error(path.string() + ":" + std::to_string(line_number) + " values must be finite");
        if (!values.empty() && !(time > values.back().first))
            throw std::runtime_error(path.string() + ": times must be strictly increasing");
        values.push_back({time, value});
    }
    if (values.size() < 2) throw std::runtime_error(path.string() + " must contain at least two (t,value) rows");
    return values;
}

inline double table_value(const std::vector<std::pair<double, double>>& values, double time) {
    if (!std::isfinite(time)) throw std::runtime_error("boundary table time must be finite");
    if (time <= values.front().first) return values.front().second;
    if (time >= values.back().first) return values.back().second;
    const auto hi = std::upper_bound(values.begin(), values.end(), time,
                                     [](double t, const auto& item) { return t < item.first; });
    const auto lo = hi - 1;
    const double f = (time - lo->first) / (hi->first - lo->first);
    return lo->second + f * (hi->second - lo->second);
}

// Tangential velocity derivatives of a strip of MPI ghost cells, using the same
// padded central differences (and physical side BCs) as an interior row/column.
// dir 'y': strip is a row of nx cells -> du/dx, dv/dx; dir 'x': column of ny cells -> du/dy, dv/dy.
// R += S: the source terms of the case (source_*) enter the residual of dU/dt = R as + S
inline void add_source(const Field& S, Field& R) {
    const int n = static_cast<int>(R.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) R[c][q] += S[c][q];
}

inline void strip_tangential_derivs(const EOS& eos, const BoundaryManager& bm, const Mesh& m, char dir, const Field& strip,
                             std::vector<double>& a, std::vector<double>& b) {
    const bool row = (dir == 'y');
    const int n = row ? m.nx : m.ny;
    Mesh m1 = row ? generate_cartesian_mesh(m.Lx, m.dy, m.nx, 1) : generate_cartesian_mesh(m.dx, m.Ly, 1, m.ny);
    m1.dx = m.dx;
    m1.dy = m.dy;
    BoundaryManager b1(m1, eos);
    if (row)
        b1.set_boundary_types(bm.type(Face::Left), bm.type(Face::Right), BCType::Transmissive, BCType::Transmissive);
    else
        b1.set_boundary_types(BCType::Transmissive, BCType::Transmissive, bm.type(Face::Bottom), bm.type(Face::Top));
    b1.set_params(bm.params());
    Field P;
    b1.get_padded_U(strip, P);
    a.resize(n);
    b.resize(n);
    for (int k = 0; k < n; ++k) {
        // row: P is 3 x (nx+2), use the middle row; column: P is (ny+2) x 3, use the middle column
        const State& hi = row ? P[(m.nx + 2) + k + 2] : P[(k + 2) * 3 + 1];
        const State& lo = row ? P[(m.nx + 2) + k] : P[k * 3 + 1];
        const double h2 = row ? 2.0 * m.dx : 2.0 * m.dy;
        State g;
        for (int q = 0; q < kNVar; ++q) g[q] = (hi[q] - lo[q]) / h2;
        const State W = eos.cons_to_prim(strip[k]);
        a[k] = (g[1] - W[1] * g[0]) / W[0];
        b[k] = (g[2] - W[2] * g[0]) / W[0];
    }
}

// BC parameters of a run (the same on every MPI rank: global domain lengths).
inline BCParams bc_params(const SimParams& p) {
    BCParams b;
    b.boundary_order = p.boundary_order;
    b.u_in = p.u_in;
    b.v_in = p.v_in;
    b.rho_in = p.rho_in;
    b.p_out = p.p_out;
    b.nrbc_sigma = p.nrbc_sigma;
    b.nrbc_beta = p.nrbc_beta;
    b.length[0] = p.Lx;
    b.length[1] = p.Ly;
    b.length[2] = p.Lz;
    return b;
}

// Cells of the halo slice adjacent to the interface (lo: last slice, hi: first).
inline Field adjacent_strip(const Field& halo, char dir, bool lo, int nx, int ny) {
    constexpr int D = kHaloDepth;
    Field s;
    if (dir == 'y')
        for (int i = 0; i < nx; ++i) s.push_back(halo[(lo ? D - 1 : 0) * nx + i]);
    else
        for (int j = 0; j < ny; ++j) s.push_back(halo[j * D + (lo ? D - 1 : 0)]);
    return s;
}

// 3-D halo slab adjacent to the interface: y-decomposition -> nz*nx cells
// [k*nx + i]; x-decomposition -> nz*ny cells [k*ny + j].
inline Field adjacent_slab(const Field& halo, char dir, bool lo, const Mesh& m) {
    constexpr int D = kHaloDepth;
    Field s;
    if (dir == 'y') {
        for (int k = 0; k < m.nz; ++k)
            for (int i = 0; i < m.nx; ++i) s.push_back(halo[((lo ? D - 1 : 0) * m.nz + k) * m.nx + i]);
    } else {
        for (int k = 0; k < m.nz; ++k)
            for (int j = 0; j < m.ny; ++j) s.push_back(halo[(k * m.ny + j) * D + (lo ? D - 1 : 0)]);
    }
    return s;
}

// Tangential velocity derivatives of a 3-D ghost slab (CrossGhosts 3-D order), using
// the same padded central differences (and physical side BCs) as an interior slab:
//   dir 'y' (slab nx x nz): a = du/dx, b = dv/dx, c = dw/dz, d = dv/dz
//   dir 'x' (slab ny x nz): a = du/dy, b = dv/dy, c = dw/dz, d = du/dz
inline void slab_tangential_derivs(const EOS& eos, const BoundaryManager& bm, const Mesh& m, char dir, const Field& slab,
                            std::vector<double>& a, std::vector<double>& b, std::vector<double>& c,
                            std::vector<double>& d) {
    const bool ydec = (dir == 'y');
    // One-cell-thick mesh across the decomposition direction, same spacing and side BCs
    Mesh m1 = ydec ? generate_cartesian_mesh_3d(m.Lx, m.dy, m.Lz, m.nx, 1, m.nz)
                   : generate_cartesian_mesh_3d(m.dx, m.Ly, m.Lz, 1, m.ny, m.nz);
    m1.dx = m.dx;
    m1.dy = m.dy;
    m1.dz = m.dz;
    BoundaryManager b1(m1, eos);
    if (ydec)
        b1.set_boundary_types(bm.type(Face::Left), bm.type(Face::Right), BCType::Transmissive, BCType::Transmissive);
    else
        b1.set_boundary_types(BCType::Transmissive, BCType::Transmissive, bm.type(Face::Bottom), bm.type(Face::Top));
    b1.set_type(Face::Front, bm.type(Face::Front));
    b1.set_type(Face::Back, bm.type(Face::Back));
    b1.set_params(bm.params());
    Field P;
    b1.get_padded_U(slab, P);
    const int PX = m1.nx + 2, PY = m1.ny + 2;
    auto at = [&](int ii, int jj, int kk) -> const State& { return P[(static_cast<size_t>(kk) * PY + jj) * PX + ii]; };
    const int ns = static_cast<int>(slab.size());
    a.resize(ns);
    b.resize(ns);
    c.resize(ns);
    d.resize(ns);
    for (int k = 0; k < m.nz; ++k) {
        const int nt = ydec ? m.nx : m.ny;
        for (int t = 0; t < nt; ++t) {
            const int ii = ydec ? t + 1 : 1, jj = ydec ? 1 : t + 1, kk = k + 1;
            const int s = k * nt + t;
            const State W = eos.cons_to_prim(slab[s]);
            auto deriv = [&](int comp, int axis) {   // d(vel comp)/dx_axis via the conservative chain rule
                const State& hi =
                    axis == 0 ? at(ii + 1, jj, kk) : (axis == 1 ? at(ii, jj + 1, kk) : at(ii, jj, kk + 1));
                const State& lo =
                    axis == 0 ? at(ii - 1, jj, kk) : (axis == 1 ? at(ii, jj - 1, kk) : at(ii, jj, kk - 1));
                const double h2 = 2.0 * (axis == 0 ? m.dx : (axis == 1 ? m.dy : m.dz));
                const double g_rho = (hi[0] - lo[0]) / h2, g_mom = (hi[kMom[comp]] - lo[kMom[comp]]) / h2;
                return (g_mom - W[kMom[comp]] * g_rho) / W[0];
            };
            if (ydec) {
                a[s] = deriv(0, 0);
                b[s] = deriv(1, 0);
                c[s] = deriv(2, 2);
                d[s] = deriv(1, 2);
            } else {
                a[s] = deriv(0, 1);
                b[s] = deriv(1, 1);
                c[s] = deriv(2, 2);
                d[s] = deriv(0, 2);
            }
        }
    }
}

// A step whose wave speed is not finite, or whose time step is below the resolution of t, means the solution has diverged
// (a non-finite state, or a speed so large that t no longer advances: the loop would never end). Every rank sees the same
// speed (it is reduced), so under MPI all ranks stop together.
inline void check_progress(double t, double dt, double smax, int step, const char* scheme) {
    if (std::isfinite(smax) && std::isfinite(dt) && t + dt > t) return;
    char msg[256];
    std::snprintf(msg, sizeof msg,
                  "the solution diverged at step %d (t = %.6g, dt = %.3g, largest wave speed = %.3g) with %s: lower the CFL number, "
                  "or use a more robust scheme (reconstruction, time integrator) for this problem",
                  step, t, dt, smax, scheme);
    throw std::runtime_error(msg);
}

}  // namespace sim_detail
}  // namespace fv
