#include "fv/Profile.hpp"
#include "fv/Switches.hpp"
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
#include "fv/SolutionMapping.hpp"
#ifdef FV_USE_MPI
#include "fv/MPIDecomposition.hpp"
#endif

#ifndef FV_PROJECT_ROOT
#define FV_PROJECT_ROOT "."
#endif

namespace fs = std::filesystem;

namespace fv {

std::atomic<bool> g_stop_requested{false};

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

std::string to_prefix(std::string s) {
    for (auto& c : s) c = (c == ' ') ? '_' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

double next_time_output(double time, double interval) {
    if (!(interval > 0.0)) return std::numeric_limits<double>::infinity();
    const double quotient = time / interval;
    const double tolerance = 8.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(quotient));
    return (std::floor(quotient + tolerance) + 1.0) * interval;
}

bool reached_time_output(double time, double target, double interval) {
    if (!std::isfinite(target)) return false;
    const double tolerance = 8.0 * std::numeric_limits<double>::epsilon() *
                             std::max({std::abs(time), std::abs(target), std::abs(interval)});
    return std::abs(time - target) <= tolerance;
}

std::vector<std::pair<double, double>> read_time_table(const fs::path& path) {
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

double table_value(const std::vector<std::pair<double, double>>& values, double time) {
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
void add_source(const Field& S, Field& R) {
    const int n = static_cast<int>(R.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c)
        for (int q = 0; q < kNVar; ++q) R[c][q] += S[c][q];
}

void strip_tangential_derivs(const EOS& eos, const BoundaryManager& bm, const Mesh& m, char dir, const Field& strip,
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
BCParams bc_params(const SimParams& p) {
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
Field adjacent_strip(const Field& halo, char dir, bool lo, int nx, int ny) {
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
Field adjacent_slab(const Field& halo, char dir, bool lo, const Mesh& m) {
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
void slab_tangential_derivs(const EOS& eos, const BoundaryManager& bm, const Mesh& m, char dir, const Field& slab,
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
}   // namespace

std::string format_elapsed(double s) {
    char buf[32];
    if (s < 60)
        std::snprintf(buf, sizeof(buf), "%.1fs", s);
    else
        std::snprintf(buf, sizeof(buf), "%d:%02d", static_cast<int>(s) / 60, static_cast<int>(s) % 60);
    return buf;
}

namespace {
// ANSI foreground colour: 24-bit, or the nearest entry of the xterm-256 colour cube.
std::string ansi_rgb(int colors, int r, int g, int b) {
    char buf[32];
    if (colors == 24) {
        std::snprintf(buf, sizeof(buf), "\x1b[38;2;%d;%d;%dm", r, g, b);
    } else {
        auto level = [](int v) { return v < 48 ? 0 : v < 115 ? 1 : (v - 35) / 40; };   // 0,95,135,175,215,255
        std::snprintf(buf, sizeof(buf), "\x1b[38;5;%dm", 16 + 36 * level(r) + 6 * level(g) + level(b));
    }
    return buf;
}
}   // namespace

void print_logo(int colors) {
    std::ifstream in(std::string(FV_PROJECT_ROOT) + "/cpp/logo.txt", std::ios::binary);
    if (!in) return;
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
    }
    // Phoenix gradient, top to bottom: halo gold, orange, red, oxblood (the GUI's brand colours)
    static const int stops[][3] = {{0xF2, 0xC6, 0x5C}, {0xE8, 0x8E, 0x3A}, {0xD2, 0x50, 0x3A}, {0xB2, 0x37, 0x44}};
    const int n = static_cast<int>(lines.size()), ns = 4;
    for (int r = 0; r < n; ++r) {
        if (colors) {
            const double t = n > 1 ? static_cast<double>(r) / (n - 1) * (ns - 1) : 0.0;
            const int k = std::min(static_cast<int>(t), ns - 2);
            const double f = t - k;
            int c[3];
            for (int q = 0; q < 3; ++q)
                c[q] = static_cast<int>(stops[k][q] + (stops[k + 1][q] - stops[k][q]) * f + 0.5);
            std::fputs(ansi_rgb(colors, c[0], c[1], c[2]).c_str(), stdout);
        }
        std::fputs(lines[r].c_str(), stdout);
        std::fputs(colors ? "\x1b[0m\n" : "\n", stdout);
    }
    if (colors) {
        const std::string gold = ansi_rgb(colors, stops[0][0], stops[0][1], stops[0][2]);
        std::printf("\n%s\x1b[1m%54s\x1b[0m\n\x1b[2m%67s\x1b[0m\n\n", gold.c_str(), "F E N I C E",
                    "compressible finite-volume solver");
    } else {
        std::printf("\n%54s\n%67s\n\n", "F E N I C E", "compressible finite-volume solver");
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

std::string Simulation::airfoil_file_path() const {
    const fs::path f(p.ibm_file);
    if (f.is_absolute()) return f.string();
    const fs::path in_case = fs::path(test_case->source).parent_path() / f;   // next to the case.toml
    if (fs::exists(in_case)) return in_case.string();
    if (fs::exists(f)) return f.string();
    throw std::runtime_error("Airfoil file '" + p.ibm_file + "' not found (looked in " +
                             in_case.parent_path().string() + " and the current folder)");
}

void Simulation::load_boundary_functions() {
    boundary_expressions_.clear();
    boundary_tables_.clear();
    const std::array<std::string, 4> keys = {"u_in", "v_in", "rho_in", "p_out"};
    for (const std::string& key : keys) {
        const std::string raw = p.source.str(key, key == "rho_in" || key == "p_out" ? "1.0" : "0.0");
        const std::string table_key = key + "_table";
        const std::string table_name = p.source.str(table_key, "");
        char* end = nullptr;
        std::strtod(raw.c_str(), &end);
        const bool numeric = end && end != raw.c_str() && *end == '\0';
        if (!table_name.empty()) {
            if (!numeric) throw std::invalid_argument(key + " expression and " + table_key + " cannot both be set");
            fs::path table_path(table_name);
            if (!table_path.is_absolute()) {
                const fs::path beside_case = fs::path(test_case->source).parent_path() / table_path;
                std::error_code ec;
                if (fs::exists(beside_case, ec)) table_path = beside_case;
                else if (!fs::exists(table_path, ec))
                    throw std::runtime_error("Boundary time table '" + table_name + "' not found next to " + test_case->source +
                                             " or in the current folder");
            }
            boundary_tables_[key] = read_time_table(table_path);
        } else if (!numeric) {
            boundary_expressions_[key] = compile_time_expression(p.source, mesh, raw, key);
        }
    }

    const auto initial = [&](const std::string& key, double fallback) {
        const auto table = boundary_tables_.find(key);
        if (table != boundary_tables_.end()) return table_value(table->second, 0.0);
        const auto expr = boundary_expressions_.find(key);
        return expr != boundary_expressions_.end() ? expr->second(0.0) : fallback;
    };
    p.u_in = initial("u_in", p.u_in);
    p.v_in = initial("v_in", p.v_in);
    p.rho_in = initial("rho_in", p.rho_in);
    p.p_out = initial("p_out", p.p_out);
    if (!(p.rho_in > 0.0) || !(p.p_out > 0.0))
        throw std::invalid_argument("time-dependent rho_in and p_out must be positive at t=0");
    if (p.reynolds > 0.0) {
        if (!(p.rho_in > 0.0) || !(p.u_in > 0.0))
            throw std::invalid_argument("reynolds needs positive u_in and rho_in at t=0 (including boundary tables)");
        const bool airfoil = !naca_digits(p.ibm_shape).empty() || lower(p.ibm_shape) == "airfoil";
        const double length = airfoil ? p.ibm_chord : 2.0 * p.ibm_radius;
        p.mu = p.rho_in * p.u_in * length / p.reynolds;
        char value[32];
        std::snprintf(value, sizeof(value), "%.17g", p.mu);
        p.source.set("mu", value);
    }
}

void Simulation::update_boundary_values(double stage_time, BoundaryManager& manager) const {
    if (boundary_expressions_.empty() && boundary_tables_.empty()) return;
    BCParams values = manager.params();
    const auto evaluate = [&](const std::string& key, double current) {
        const auto table = boundary_tables_.find(key);
        if (table != boundary_tables_.end()) return table_value(table->second, stage_time);
        const auto expression = boundary_expressions_.find(key);
        return expression == boundary_expressions_.end() ? current : expression->second(stage_time);
    };
    values.u_in = evaluate("u_in", values.u_in);
    values.v_in = evaluate("v_in", values.v_in);
    values.rho_in = evaluate("rho_in", values.rho_in);
    values.p_out = evaluate("p_out", values.p_out);
    if (!(values.rho_in > 0.0) || !(values.p_out > 0.0))
        throw std::runtime_error("time-dependent rho_in and p_out must stay positive (at t=" + std::to_string(stage_time) + ")");
    manager.set_params(values);
}

void Simulation::setup(const SimParams& params, bool verbose) {
    p = params;
    test_case = &get_test_case(p.test_case);
    if (!test_case->forces.empty() && !p.use_ibm)
        throw std::invalid_argument("[[force]] outputs need use_ibm = true");

    scheme = make_time_scheme(parse_time_scheme(p.time_scheme), p.cfl);

    file_prefix = to_prefix(test_case->key);   // the case name, also for a case given by path
    // AMR runs go to <case>_amr so they don't overwrite the uniform-mesh run.
    const std::string subdir = p.use_amr ? file_prefix + "_amr" : file_prefix;
    exporter = VTKExporter((fs::path(p.output_root) / subdir).string(), verbose, p.vtk_format != "ascii");

    mesh = generate_cartesian_mesh_3d(p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz);
    load_boundary_functions();
    const auto inside_axis = [](double x, double origin, double length) { return x >= origin && x <= origin + length; };
    for (const ProbeSpec& probe : test_case->probes) {
        if (!inside_axis(probe.x, mesh.x0, mesh.Lx) || !inside_axis(probe.y, mesh.y0, mesh.Ly) ||
            (mesh.is3d() && (!probe.has_z || !inside_axis(probe.z, mesh.z0, mesh.Lz))) ||
            (!mesh.is3d() && probe.has_z && probe.z != 0.0))
            throw std::invalid_argument("probe '" + probe.name + "' coordinates must lie inside the simulation domain");
    }
    for (const SampleSpec& sample : test_case->samples) {
        const size_t dimensions = mesh.is3d() ? 3 : 2;
        if (sample.from.size() != dimensions || sample.to.size() != dimensions)
            throw std::invalid_argument("sample '" + sample.name + "' needs " + std::to_string(dimensions) +
                                        " coordinates for this mesh");
        for (size_t d = 0; d < dimensions; ++d) {
            const double origin = d == 0 ? mesh.x0 : (d == 1 ? mesh.y0 : mesh.z0);
            const double length = d == 0 ? mesh.Lx : (d == 1 ? mesh.Ly : mesh.Lz);
            if (!inside_axis(sample.from[d], origin, length) || !inside_axis(sample.to[d], origin, length))
                throw std::invalid_argument("sample '" + sample.name + "' endpoints must lie inside the simulation domain");
        }
    }
    if (verbose && mesh.is3d()) std::printf("[MESH] 3-D: %d x %d x %d cells\n", p.nx, p.ny, p.nz);
    // CFL length: min(dx, dy) in 2-D. In 3-D the unsplit fluxes add three directional Courant
    // numbers; 2 / sum(1/h) bounds that sum by 2 CFL, the same bound min(dx, dy) gives a 2-D square cell.
    dt_length = std::min(mesh.Lx / mesh.nx, mesh.Ly / mesh.ny);
    if (mesh.is3d())
        dt_length = 2.0 / (1.0 / (mesh.Lx / mesh.nx) + 1.0 / (mesh.Ly / mesh.ny) + 1.0 / (mesh.Lz / mesh.nz));
#ifdef _OPENMP
    if (verbose) std::printf("[OPENMP] Native kernels on %d thread(s)\n", omp_get_max_threads());
#endif

    if (lower(p.eos_type) == "ideal gas")
        fluid = EOS::ideal(p.gamma);
    else
        fluid = EOS::stiffened(p.gamma, p.p_inf);

    if (p.use_ibm) {
        const std::string digits = naca_digits(p.ibm_shape);
        if (!digits.empty() || lower(p.ibm_shape) == "airfoil") {
            const Polygon2D unit = digits.empty() ? load_airfoil(airfoil_file_path()) : naca4(digits);
            const Polygon2D placed = place_airfoil(unit, p.ibm_chord, p.ibm_aoa, p.ibm_cx, p.ibm_cy);
            ibm_factory = [placed, rho = p.rho_in, pout = p.p_out](const Mesh& m) {
                return ImmersedBoundary(m, placed, rho, pout);
            };
            ibm = std::make_unique<ImmersedBoundary>(ibm_factory(mesh));
            if (verbose)
                std::printf("[IBM] %s, chord %g, %g deg, quarter chord at (%g, %g) -> %d solid cells\n",
                            digits.empty() ? airfoil_file_path().c_str() : p.ibm_shape.c_str(), p.ibm_chord, p.ibm_aoa,
                            p.ibm_cx, p.ibm_cy, ibm->solid_count());
        } else {
            ibm_factory = [shape = p.ibm_shape, r = p.ibm_radius, cx = p.ibm_cx, cy = p.ibm_cy, rho = p.rho_in,
                           pout = p.p_out, cz = p.ibm_cz](const Mesh& m) {
                return ImmersedBoundary(m, shape, r, cx, cy, rho, pout, cz);
            };
            ibm = std::make_unique<ImmersedBoundary>(ibm_factory(mesh));
            if (verbose)
                std::printf("[IBM] %s at (%g, %g) r=%g -> %d solid cells\n", p.ibm_shape.c_str(), p.ibm_cx, p.ibm_cy,
                            p.ibm_radius, ibm->solid_count());
        }
        if (ibm->solid_count() == 0)
            std::printf("[WARNING] The IBM body covers no cell: check its size and position\n");
        if (p.ibm_method == "ghost") {
            ibm->enable_ghost_cells(mesh, !p.use_viscous, p.ibm_curvature);
            if (verbose)
                std::printf("[IBM] Ghost-cell wall (%s): %d ghost cells, loads on the true surface\n",
                            p.use_viscous ? "no-slip" : "slip", ibm->ghost_count());
        }
        if (verbose && p.reynolds > 0.0) std::printf("[PHYSICS] Re = %g -> mu = %g\n", p.reynolds, p.mu);
    }

    boundaries = BoundaryManager(mesh, fluid);
    boundaries.set_boundary_types(parse_bc(p.bc_left), parse_bc(p.bc_right), parse_bc(p.bc_bottom), parse_bc(p.bc_top));
    boundaries.set_type(Face::Front, parse_bc(p.bc_front));
    boundaries.set_type(Face::Back, parse_bc(p.bc_back));
    boundaries.set_params(bc_params(p));
    update_boundary_values(0.0, boundaries);

    const RiemannType rt = parse_riemann(p.riemann_solver);
    riemann = RiemannSolver(fluid, rt, 0.1, p.shock_blend);
    if (verbose && rt == RiemannType::LowMachHLLC) std::printf("[SOLVER] Low-Mach correction active (Thornber 2008)\n");
    if (verbose && rt == RiemannType::PreconditionedHLLC)
        std::printf("[SOLVER] Preconditioned HLLC active (Weiss-Smith 1995, Ma_cut=0.1)\n");

    if (verbose && p.use_muscl_alias)
        std::printf("[PARAMS] use_muscl is retired: read as reconstruction = %s (set reconstruction alone: none, muscl, weno3 "
                    "or weno5)\n",
                    p.reconstruction.c_str());
    if (p.reconstructs()) {
        muscl = std::make_unique<MusclReconstruction>(fluid, parse_limiter(p.muscl_limiter), parse_reconstruction(p.reconstruction), p.weno_epsilon);
        if (verbose && p.reconstruction == "muscl")
            std::printf("[SOLVER] 2nd-Order MUSCL active (Limiter: %s)\n", p.muscl_limiter.c_str());
        if (verbose && p.reconstruction != "muscl") std::printf("[SOLVER] %s reconstruction active\n", p.reconstruction.c_str());
        if (verbose && parse_time_scheme(p.time_scheme) == TimeSchemeType::ExplicitEuler)
            std::printf("[WARNING] MUSCL with Explicit Euler is unstable for smooth waves (amplitudes grow): "
                        "use SSP Runge-Kutta 2 or 3\n");
    } else {
        muscl.reset();
        if (verbose) std::printf("[SOLVER] 1st-Order Spatial (Godunov) active\n");
    }

    if (p.use_viscous) {
        ns = std::make_unique<NavierStokes>(fluid, p.mu, p.pr);
        ns->set_second_order_boundary(p.boundary_order == 2);
        if (verbose) std::printf("[PHYSICS] Viscous fluxes enabled (Navier-Stokes)\n");
    } else {
        ns.reset();
        if (verbose) std::printf("[PHYSICS] Inviscid flow (Euler)\n");
    }

    if (p.use_bodyforce) {
        body_force = std::make_unique<ConstantBodyForce>(ConstantBodyForce{p.gx, p.gy, mesh.is3d() ? p.gz : 0.0});
        if (verbose && mesh.is3d())
            std::printf("[PHYSICS] Body Forces enabled: gx=%g, gy=%g, gz=%g\n", p.gx, p.gy, p.gz);
        else if (verbose)
            std::printf("[PHYSICS] Body Forces enabled: gx=%g, gy=%g\n", p.gx, p.gy);
    } else {
        body_force.reset();
    }

    rhs_sources.clear();
    if (body_force) {
        RhsSourceTerm term;
        term.uniform = [this](const Mesh&, const Field& Uin, Field& R, double) {
            ProfileScope profile(Section::BodyForce);
            body_force->add_to(Uin, R);
        };
        term.adaptive = [this](const CellMesh&, const Field& Uin, Field& R, int begin, int end, double, const Field*) {
            ProfileScope profile(Section::BodyForce);
            const ConstantBodyForce& body = *body_force;
#pragma omp parallel for schedule(static) if (end - begin >= kOmpMinWork)
            for (int c = begin; c < end; ++c) {
                State& r = R[static_cast<size_t>(c - begin)];
                const State& u = Uin[static_cast<size_t>(c)];
                r[1] += u[0] * body.gx;
                r[2] += u[0] * body.gy;
                r[3] += u[1] * body.gx + u[2] * body.gy;
                if (body.gz != 0.0) {
                    r[3] += u[4] * body.gz;
                    r[4] += u[0] * body.gz;
                }
            }
        };
        rhs_sources.add(std::move(term));
    }
    if (p.use_source) {
        RhsSourceTerm term;
        term.uniform = [this](const Mesh& m, const Field&, Field& R, double stage_time) {
            ProfileScope profile(Section::Source);
            add_source(source_for(m, stage_time), R);
        };
        term.adaptive = [this](const CellMesh&, const Field&, Field& R, int begin, int end, double, const Field* cached) {
            ProfileScope profile(Section::Source);
            if (!cached) throw std::logic_error("expression source field is not available");
#pragma omp parallel for schedule(static) if (end - begin >= kOmpMinWork)
            for (int c = begin; c < end; ++c)
                for (int q = 0; q < kNVar; ++q)
                    R[static_cast<size_t>(c - begin)][static_cast<size_t>(q)] +=
                        (*cached)[static_cast<size_t>(c)][static_cast<size_t>(q)];
        };
        rhs_sources.add(std::move(term));
    }

    // AMR: the adaptive mesh and its engine; on several MPI ranks the leaves are distributed over them (CellParallel)
    cells.reset();
    cell_engine.reset();
    cell_adaptor.reset();
    cell_par.reset();
    hot_interp_.reset();
    hot_interp3_.reset();
    cell_comm = CellComm{};
    if (p.use_amr) {
        const bool per_x = boundaries.type(Face::Left) == BCType::Periodic;
        const bool per_y = boundaries.type(Face::Bottom) == BCType::Periodic;
        const bool per_z = boundaries.type(Face::Front) == BCType::Periodic;
        if (mesh.is3d())   // an octree (docs/design/amr-octree.md); not written yet: snapshots, post-processing, checkpoints
            cells = std::make_unique<CellMesh>(CellMesh::make_3d(p.Lx, p.Ly, p.Lz, p.nx, p.ny, p.nz, p.amr_max_level, per_x, per_y,
                                                                 per_z, mesh.x0, mesh.y0, mesh.z0));
        else
            cells = std::make_unique<CellMesh>(p.Lx, p.Ly, p.nx, p.ny, p.amr_max_level, per_x, per_y);
        cell_physics = CellPhysics{};
        cell_physics.ns = ns.get();
        cell_physics.body = body_force.get();
        cell_physics.rhs_sources = rhs_sources.empty() ? nullptr : &rhs_sources;
        if (p.use_source) {
            cell_physics.source = &p.source;
            cell_physics.source_unsteady = p.source_unsteady;
            cell_physics.source_mesh = &mesh;
        }
        if (p.use_ibm) {
            cell_physics.ibm_factory = ibm_factory;
            cell_physics.ibm_ghost = p.ibm_method == "ghost";
            cell_physics.ibm_curvature = p.ibm_curvature;
            cell_physics.ibm_slip = !p.use_viscous;
        }
        cell_engine = std::make_unique<CellFluxEngine>(fluid, riemann, muscl.get(), boundaries);
        cell_engine->set_physics(cell_physics);
        cell_indicator = CellIndicator{};
        cell_indicator.criterion = p.amr_criterion;
        cell_indicator.split = p.amr_refine_threshold;
        cell_indicator.join = p.amr_derefine_threshold;
        cell_indicator.diffusion = p.amr_diffusion;
        cell_indicator.use_density = p.amr_indicator.find("density") != std::string::npos;
        cell_indicator.use_velocity = p.amr_indicator.find("velocity") != std::string::npos;
        cell_indicator.use_pressure = p.amr_indicator.find("pressure") != std::string::npos;
        cell_adaptor = std::make_unique<CellAdaptor>(*cells, *cell_engine, fluid, cell_indicator);
        if (!p.amr_force_box.empty()) {
            double b[6];
            const int got = std::sscanf(p.amr_force_box.c_str(), "%lf%*[ ,]%lf%*[ ,]%lf%*[ ,]%lf%*[ ,]%lf%*[ ,]%lf", &b[0], &b[1], &b[2], &b[3],
                                        &b[4], &b[5]);
            if (mesh.is3d() && got == 6)
                cell_adaptor->set_force_box3(b[0], b[1], b[2], b[3], b[4], b[5]);
            else if (got >= 4)
                cell_adaptor->set_force_box(b[0], b[1], b[2], b[3]);   // (3-D: every z)
            else
                throw std::invalid_argument("amr_force_box needs x0, x1, y0, y1 (and z0, z1 in 3-D)");
        }
        if (p.use_ibm && p.amr_refine_ibm && mesh.is3d()) {
            // 3-D: a shell of finest leaves around the surface (near_body_surface); the ghost-cell wall is ready with 6 cells
            const int margin = p.ibm_method == "ghost" ? 6 : 2;
            cell_adaptor->set_body(ibm_factory, margin);
            if (verbose) std::printf("[AMR] Body band: %d finest-level cells on both sides of the surface\n", margin);
        } else if (p.use_ibm && p.amr_refine_ibm) {
            int margin = 2;   // the fringe the solid reset treats
            if (p.ibm_method == "ghost") {   // every ghost cell and the cells it reads (and the load probes) at the finest level
                const int M = p.amr_max_level;
                const Mesh fine = generate_cartesian_mesh(p.Lx, p.Ly, p.nx << M, p.ny << M, mesh.x0, mesh.y0);
                ImmersedBoundary ib = ibm_factory(fine);
                ib.enable_ghost_cells(fine, !p.use_viscous, p.ibm_curvature);
                margin = std::max(margin, ib.ghost_reach());
                if (verbose) std::printf("[AMR] Ghost-cell wall: finest-level band of %d cells around the body\n", margin);
            }
            cell_adaptor->set_body(ibm_factory, margin);
        }
        if (verbose)
            std::printf("[AMR] Enabled - %d levels, indicator %s (variation %g, refine at %g, join below %g, %d smoothing "
                        "iterations), subcycling active\n",
                        p.amr_max_level, p.amr_indicator.c_str(), p.amr_criterion, p.amr_refine_threshold,
                        p.amr_derefine_threshold, p.amr_diffusion);
#ifdef FV_USE_MPI
        int init = 0, size = 1;
        MPI_Initialized(&init);
        if (init) MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size > 1) {
            cell_comm.size = size;
            MPI_Comm_rank(MPI_COMM_WORLD, &cell_comm.rank);
            cell_comm.allgatherv = [](const double* send, int n, double* recv, const int* counts, const int* displs) {
                MPI_Allgatherv(send, n, MPI_DOUBLE, recv, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
            };
            cell_comm.allreduce_max = [](double v) {
                double g = v;
                MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                return g;
            };
            cell_comm.allreduce_sum = [](double v) {
                double g = v;
                MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                return g;
            };
            cell_comm.exchange = [](const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv) {
                std::vector<MPI_Request> req;
                req.reserve(send.size() + recv.size());
                for (size_t r = 0; r < recv.size(); ++r)
                    if (!recv[r].empty()) {
                        req.emplace_back();
                        MPI_Irecv(recv[r].data(), static_cast<int>(recv[r].size()), MPI_DOUBLE, static_cast<int>(r), 91,
                                  MPI_COMM_WORLD, &req.back());
                    }
                for (size_t r = 0; r < send.size(); ++r)
                    if (!send[r].empty()) {
                        req.emplace_back();
                        MPI_Isend(send[r].data(), static_cast<int>(send[r].size()), MPI_DOUBLE, static_cast<int>(r), 91,
                                  MPI_COMM_WORLD, &req.back());
                    }
                MPI_Waitall(static_cast<int>(req.size()), req.data(), MPI_STATUSES_IGNORE);
            };
            cell_par = std::make_unique<CellParallel>(*cells, *cell_engine, cell_comm);
        }
#endif
    }

    if (verbose)
        std::printf("Mesh: %d cells | Riemann: %s | Time: %s | CFL: %g\n", mesh.num_cells(), p.riemann_solver.c_str(),
                    p.time_scheme.c_str(), p.cfl);
    std::fflush(stdout);
}

// FENICE_MEM_TRACE=1: the resident and peak memory of the process at a step of the setup (stderr)
// A step whose wave speed is not finite, or whose time step is below the resolution of t, means the solution has diverged
// (a non-finite state, or a speed so large that t no longer advances: the loop would never end). Every rank sees the same
// speed (it is reduced), so under MPI all ranks stop together.
static void check_progress(double t, double dt, double smax, int step, const char* scheme) {
    if (std::isfinite(smax) && std::isfinite(dt) && t + dt > t) return;
    char msg[256];
    std::snprintf(msg, sizeof msg,
                  "the solution diverged at step %d (t = %.6g, dt = %.3g, largest wave speed = %.3g) with %s: lower the CFL number, "
                  "or use a more robust scheme (reconstruction, time integrator) for this problem",
                  step, t, dt, smax, scheme);
    throw std::runtime_error(msg);
}

static void mem_trace(const char* where) {
    if (!switch_on("FENICE_MEM_TRACE")) return;
    const MemoryUsage m = memory_usage();
    std::fprintf(stderr, "MEMTRACE %s: resident %.0f MB, peak %.0f MB\n", where, m.current * 1e-6, m.peak * 1e-6);
}

void Simulation::initialize_field() {
    if (test_case->requires_viscosity && !ns)
        std::printf("[WARNING] This test case requires Navier-Stokes - viscous physics is off!\n");

    mem_setup_ = memory_usage();
    // An adaptive run needs a global field only for a hot start (interpolated onto the leaves)
    const bool global_field = !cells || p.use_hot_start;
    if (global_field) U = p.use_hot_start ? load_hot_start(p.hot_start_file) : initial_field();
    start_step = 0;
    if (p.resume) start_step = load_checkpoint();

    // A fresh start (also resume without a checkpoint) is forced; a restored state already is
    if (p.use_ibm && start_step == 0 && !cells) ibm->apply_forcing(U, fluid);
    if (!cells) return;

    if (p.use_hot_start) {
        // Seed every leaf by sampling the hot-start field at the leaf's cell centre (bilinear, extrapolating at the edges)
        // (a source of another size than the base mesh is sampled at its own resolution)
        const bool own = hot_src_nx_ > 0;
        const int nxs = own ? hot_src_nx_ : mesh.nx, nys = own ? hot_src_ny_ : mesh.ny;
        const double dx_s = mesh.Lx / nxs, dy_s = mesh.Ly / nys;
        if (cells->dim == 3) {
            const int nzs = own ? hot_src_nz_ : mesh.nz;
            const double dz_s = mesh.Lz / nzs;
            hot_interp3_ = std::make_unique<GridInterp3D>(linspace(dx_s / 2.0, mesh.Lx - dx_s / 2.0, nxs),
                                                          linspace(dy_s / 2.0, mesh.Ly - dy_s / 2.0, nys),
                                                          linspace(dz_s / 2.0, mesh.Lz - dz_s / 2.0, nzs), own ? hot_src_ : U);
        } else {
            hot_interp_ = std::make_unique<GridInterp2D>(linspace(dx_s / 2.0, mesh.Lx - dx_s / 2.0, nxs),
                                                         linspace(dy_s / 2.0, mesh.Ly - dy_s / 2.0, nys), own ? hot_src_ : U);
        }
    }
    // Several ranks with the shares kept separately: no rank ever holds the whole mesh (the topology is faceless, the share is built
    // from it, and the initial passes run through CellParallel::remesh with the initial condition filled on the share)
    const bool distributed_init = start_step == 0 && cell_par && !replicated_storage();
    if (distributed_init) {
        {   // the base topology is only needed to cut the first share
            CellMesh topo(*cells);
            topo.U = Field();
            topo.set_faceless(true);
            for (int k = 0; k < p.amr_init_levels; ++k)
                if (!topo.adapt(std::vector<char>(static_cast<size_t>(topo.num_leaves()), 1))) break;
            mem_trace("base topology");
            cell_par->init_adapt(topo, *cell_adaptor, [this](CellMesh& m) { fill_cells(m); });
        }
        if (p.use_ibm && p.ibm_method == "ghost")
            std::printf(cell_engine->ghost_ready() ? "[IBM] Ghost-cell wall active on the finest-level band\n"
                                                   : "[WARNING] Ghost-cell wall inactive: the cells around the body are not all at the "
                                                     "finest level; the mask forcing is used\n");
        if (p.use_ibm) {
            cell_engine->solid_reset(*cells, cells->U);
            cell_par->exchange_solid();
        }
        mem_trace("init_adapt done");
        cell_par->reset_timers();   // the report of the remesh counts the time loop only
        if (switch_on("FENICE_HALO_REACH") && cell_comm.rank == 0)
            std::printf("[AMR] FENICE_HALO_REACH needs the whole mesh: use a serial run or FENICE_REPLICATED_REMESH=1\n");
    } else if (start_step == 0) {
        // The initial condition is evaluated at each leaf's own resolution, so refined leaves see the exact initial
        // discontinuities rather than their interpolation from the base grid
        fill_cells(*cells);
        cell_engine->build(*cells);
        mem_trace("base built");
        for (int k = 0; k < p.amr_init_levels; ++k) {
            if (!cells->adapt(std::vector<char>(static_cast<size_t>(cells->num_leaves()), 1))) break;
            cell_engine->build(*cells);
            fill_cells(*cells);
        }
        mem_trace("init levels");
        cell_adaptor->init_adapt([this](CellMesh& m) { fill_cells(m); });
        if (p.use_ibm && p.ibm_method == "ghost")
            std::printf(cell_engine->ghost_ready() ? "[IBM] Ghost-cell wall active on the finest-level band\n"
                                                   : "[WARNING] Ghost-cell wall inactive: the cells around the body are not all at the "
                                                     "finest level; the mask forcing is used\n");
        if (p.use_ibm) cell_engine->solid_reset(*cells, cells->U);
        mem_trace("init_adapt done");
    }
    if (cell_par && !distributed_init && !share_resumed_) {
        cell_par->partition();
        print_halo_reach("initial mesh");
        if (!replicated_storage()) {
            mem_trace("before localize");
            cell_par->localize();   // each rank keeps only its leaves and their halo
            mem_trace("after localize");
        }
    }
    mem_trace("init end");
    mem_init_ = memory_usage();
}

bool Simulation::replicated_storage() {
    return switch_on("FENICE_REPLICATED_REMESH");
}

void Simulation::print_halo_reach(const char* when) const {
    if (!switch_on("FENICE_HALO_REACH") || cell_comm.rank != 0) return;
    const CellParallel::HaloReach hr = CellParallel::halo_reach(*cells, *cell_engine);
    // per kind: the largest ring, and per level difference (level of the driver - level of the leaf read: ring)
    std::printf("[AMR] halo reach (%s), rings of the driver's level [level difference driver - read: ring]:", when);
    const char* names[] = {"interior", "coarse-fine", "boundary", "ghost images", "solid reset"};
    for (int k = 0; k < CellParallel::HaloReach::NumKinds; ++k) {
        std::printf("%s %s %d [", k ? ";" : "", names[k], hr.max(k));
        bool first = true;
        for (size_t d = 0; d < hr.ring[k].size(); ++d)
            if (hr.ring[k][d] >= 0) {
                std::printf("%s%+d: %d", first ? "" : ", ", static_cast<int>(d) - cells->max_level, hr.ring[k][d]);
                first = false;
            }
        std::printf("]");
    }
    std::printf("; reads per ring:");
    for (size_t k = 0; k < hr.reads.size(); ++k) std::printf(" %zu: %lld", k, hr.reads[k]);
    std::printf("\n");
    std::fflush(stdout);
}

void Simulation::fill_cells(CellMesh& m) const {
    m.U.assign(static_cast<size_t>(m.num_leaves()), State{});
    if (hot_interp3_) {
        for (int c = 0; c < m.num_leaves(); ++c) m.U[static_cast<size_t>(c)] = (*hot_interp3_)(m.xc(c), m.yc(c), m.zc(c));
        return;
    }
    if (hot_interp_) {
        for (int c = 0; c < m.num_leaves(); ++c) m.U[static_cast<size_t>(c)] = (*hot_interp_)(m.xc(c), m.yc(c));
        return;
    }
    for (int l = 0; l <= m.max_level; ++l) {
        if (m.level_begin[l + 1] == m.level_begin[l]) continue;
        Mesh ml = m.dim == 3 ? generate_cartesian_mesh_3d(m.Lx, m.Ly, m.Lz, m.nx_at(l), m.ny_at(l), m.nz_at(l), m.x0, m.y0, m.z0)
                             : generate_cartesian_mesh(m.Lx, m.Ly, m.nx_at(l), m.ny_at(l), m.x0, m.y0);
        // only the box of cells that holds this level's leaves (the values are those of the whole level: Mesh window)
        int i0 = ml.nx, i1 = 0, j0 = ml.ny, j1 = 0;
        for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c) {
            i0 = std::min(i0, m.ci[static_cast<size_t>(c)]);
            i1 = std::max(i1, m.ci[static_cast<size_t>(c)] + 1);
            j0 = std::min(j0, m.cj[static_cast<size_t>(c)]);
            j1 = std::max(j1, m.cj[static_cast<size_t>(c)] + 1);
        }
        ml.win_i0 = i0;
        ml.win_i1 = i1;
        ml.win_j0 = j0;
        ml.win_j1 = j1;
        const Field F = test_case->initialize(ml, fluid, p.source);
        for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c)
            m.U[static_cast<size_t>(c)] =
                F[static_cast<size_t>(ml.window_cell(m.ci[static_cast<size_t>(c)], m.cj[static_cast<size_t>(c)], m.ck[static_cast<size_t>(c)]))];
    }
}

std::array<double, 3> Simulation::cell_loads(const CellMesh& m, const Field& R) const {
    const std::vector<char>& solid = cell_engine->solid();
    double fx = 0.0, fy = 0.0, mz = 0.0;
    for (int l = 0; l <= m.max_level; ++l)   // the leaves this rank owns, in leaf order (all of them without a partition)
        for (int c = cell_engine->owned_begin(l); c < cell_engine->owned_end(l); ++c) {
            if (!solid[static_cast<size_t>(c)]) continue;
            double rx = R[static_cast<size_t>(c)][1], ry = R[static_cast<size_t>(c)][2];
            if (body_force) {
                rx -= m.U[static_cast<size_t>(c)][0] * body_force->gx;
                ry -= m.U[static_cast<size_t>(c)][0] * body_force->gy;
            }   // not a load
            const double ex = rx * m.volume(c), ey = ry * m.volume(c);
            fx += ex;
            fy += ey;
            mz += (m.xc(c) - p.ibm_cx) * ey - (m.yc(c) - p.ibm_cy) * ex;
        }
    return {fx, fy, mz};
}

namespace {

std::string output_token(const std::string& name) {
    std::string out;
    for (unsigned char c : name) {
        if (std::isalnum(c) || c == '-' || c == '_') out.push_back(static_cast<char>(c));
        else if (!out.empty() && out.back() != '_') out.push_back('_');
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out.empty() ? "output" : out;
}

// A resumed run appends to the file of the stopped one. Rows past the restart step (written after the last checkpoint of a run that
// did not stop gracefully) are dropped first, so that no step appears twice.
void drop_rows_after(const fs::path& path, int step) {
    std::ifstream in(path, std::ios::binary);   // binary: the line ends (CR LF on Windows) are kept as written
    if (!in) return;
    std::string line, kept;
    bool first = true, dropped = false;
    while (std::getline(in, line)) {
        if (!first && std::atoi(line.c_str()) > step) {
            dropped = true;
            continue;
        }
        first = false;
        kept += line;
        kept.push_back(static_cast<char>(10));   // LF
    }
    in.close();
    if (!dropped) return;
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary);
        out << kept;
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) throw std::runtime_error("Cannot rewrite post-processing output " + path.string() + ": " + ec.message());
}

std::FILE* open_post_csv(const fs::path& path, int start_step, const std::string& header) {
    std::error_code ec;
    const bool existed = start_step > 0 && fs::exists(path, ec);
    if (existed) drop_rows_after(path, start_step);
    std::FILE* f = std::fopen(path.string().c_str(), start_step > 0 ? "a" : "w");
    if (!f) throw std::runtime_error("Cannot open post-processing output " + path.string());
    if (!existed) std::fprintf(f, "%s\n", header.c_str());
    return f;
}

int bounded_cell(double x, double origin, double spacing, int count) {
    int i = static_cast<int>(std::floor((x - origin) / spacing));
    return std::max(0, std::min(count - 1, i));
}

int leaf_at_point(const CellMesh& cm, const std::vector<int>& levels, const std::vector<int>& ci,
                  const std::vector<int>& cj, double x, double y, const std::vector<int>* ck = nullptr, double z = 0.0) {
    for (int l = cm.max_level; l >= 0; --l) {
        const int i = bounded_cell(x, cm.x0, cm.dx(l), cm.nx_at(l));
        const int j = bounded_cell(y, cm.y0, cm.dy(l), cm.ny_at(l));
        const int k = ck ? bounded_cell(z, cm.z0, cm.dz(l), cm.nz_at(l)) : 0;
        const uint64_t key = ck ? cm.key_of(i, j, k) : morton_key(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
        auto first = std::lower_bound(levels.begin(), levels.end(), l);
        auto last = std::upper_bound(first, levels.end(), l);
        size_t lo = static_cast<size_t>(first - levels.begin());
        size_t hi = static_cast<size_t>(last - levels.begin());
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            const uint64_t mid_key = ck ? cm.key_of(ci[mid], cj[mid], (*ck)[mid])
                                        : morton_key(static_cast<uint32_t>(ci[mid]), static_cast<uint32_t>(cj[mid]));
            if (mid_key < key)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo < static_cast<size_t>(last - levels.begin()) && ci[lo] == i && cj[lo] == j && (!ck || (*ck)[lo] == k))
            return static_cast<int>(lo);
    }
    return -1;
}

class PostprocessWriter {
public:
    PostprocessWriter(const TestCase& tc, const std::string& output_dir, const std::string& prefix, int start_step)
        : probes_(tc.probes), samples_(tc.samples), forces_(tc.forces), averages_(tc.averages),
          avg_state_(tc.averages.size()), last_probe_(tc.probes.size(), -1), last_sample_(tc.samples.size(), -1),
          last_force_(tc.forces.size(), -1), dir_(output_dir), prefix_(prefix), start_step_(start_step) {
        try {
            for (const ProbeSpec& s : probes_)
                probe_files_.push_back(open_csv(s.name + "_probe", "step,t,x,y,z,cell_x,cell_y,cell_z,rho,u,v,p,w"));
            for (const SampleSpec& s : samples_)
                sample_files_.push_back(open_csv(s.name + "_sample", "step,t,point,x,y,z,cell_x,cell_y,cell_z,rho,u,v,p,w"));
            for (const ForceSpec& s : forces_)
                force_files_.push_back(open_csv(s.name + "_force", "step,t,Fx,Fy,Mz,Cd,Cl,Cm"));
            for (const AverageSpec& s : averages_) {
                std::string header = "step,t";
                for (const std::string& var : s.variables) header += "," + var + "_time_average";
                average_files_.push_back(open_csv(s.name + "_average", header));
            }
        } catch (...) {
            close_all();
            throw;
        }
    }

    ~PostprocessWriter() { close_all(); }
    PostprocessWriter(const PostprocessWriter&) = delete;
    PostprocessWriter& operator=(const PostprocessWriter&) = delete;

    bool empty() const { return probes_.empty() && samples_.empty() && forces_.empty() && averages_.empty(); }
    bool has_averages() const { return !averages_.empty(); }
    bool has_force_due(int step, bool force = false) const {
        for (size_t i = 0; i < forces_.size(); ++i)
            if ((force || step % forces_[i].interval == 0) && last_force_[i] != step) return true;
        return false;
    }

    void skip_step(int step) {
        std::fill(last_probe_.begin(), last_probe_.end(), step);
        std::fill(last_sample_.begin(), last_sample_.end(), step);
        std::fill(last_force_.begin(), last_force_.end(), step);
        for (AverageState& state : avg_state_) state.last_written_step = step;
    }

    void write_uniform(const Mesh& mesh, const EOS& eos, const Field& U, int step, double time, bool force = false) {
        for (size_t n = 0; n < probes_.size(); ++n) {
            if (!due(step, probes_[n].interval, force, last_probe_[n])) continue;
            const ProbeSpec& p = probes_[n];
            const int i = bounded_cell(p.x, mesh.x0, mesh.dx, mesh.nx);
            const int j = bounded_cell(p.y, mesh.y0, mesh.dy, mesh.ny);
            const int k = mesh.is3d() ? bounded_cell(p.z, mesh.z0, mesh.dz, mesh.nz) : 0;
            const State w = eos.cons_to_prim(U[static_cast<size_t>(mesh.cell(i, j, k))]);
            write_point(probe_files_[n], step, time, -1, p.x, p.y, p.z, mesh.xc(i), mesh.yc(j),
                        mesh.is3d() ? mesh.zc(k) : 0.0, w);
            last_probe_[n] = step;
        }
        for (size_t n = 0; n < samples_.size(); ++n) {
            if (!due(step, samples_[n].interval, force, last_sample_[n])) continue;
            const SampleSpec& s = samples_[n];
            for (int point = 0; point < s.points; ++point) {
                const double f = static_cast<double>(point) / static_cast<double>(s.points - 1);
                const double x = s.from[0] + f * (s.to[0] - s.from[0]);
                const double y = s.from[1] + f * (s.to[1] - s.from[1]);
                const double z = s.from.size() == 3 ? s.from[2] + f * (s.to[2] - s.from[2]) : 0.0;
                const int i = bounded_cell(x, mesh.x0, mesh.dx, mesh.nx);
                const int j = bounded_cell(y, mesh.y0, mesh.dy, mesh.ny);
                const int k = mesh.is3d() ? bounded_cell(z, mesh.z0, mesh.dz, mesh.nz) : 0;
                const State w = eos.cons_to_prim(U[static_cast<size_t>(mesh.cell(i, j, k))]);
                write_point(sample_files_[n], step, time, point, x, y, z, mesh.xc(i), mesh.yc(j),
                            mesh.is3d() ? mesh.zc(k) : 0.0, w);
            }
            last_sample_[n] = step;
        }
    }

    // `ck`: the k of the leaves of an octree (null in 2-D)
    void write_amr(const CellMesh& mesh, const EOS& eos, const std::vector<int>& levels, const std::vector<int>& ci,
                   const std::vector<int>& cj, const Field& U, int step, double time, bool force = false,
                   const std::vector<int>* ck = nullptr) {
        for (size_t n = 0; n < probes_.size(); ++n) {
            if (!due(step, probes_[n].interval, force, last_probe_[n])) continue;
            const ProbeSpec& p = probes_[n];
            const int c = leaf_at_point(mesh, levels, ci, cj, p.x, p.y, ck, p.z);
            if (c < 0) throw std::logic_error("post-processing probe did not map to an AMR leaf");
            const State w = eos.cons_to_prim(U[static_cast<size_t>(c)]);
            const int l = levels[static_cast<size_t>(c)];
            const double cx = mesh.x0 + (ci[static_cast<size_t>(c)] + 0.5) * mesh.dx(l);
            const double cy = mesh.y0 + (cj[static_cast<size_t>(c)] + 0.5) * mesh.dy(l);
            const double cz = ck ? mesh.z0 + ((*ck)[static_cast<size_t>(c)] + 0.5) * mesh.dz(l) : 0.0;
            write_point(probe_files_[n], step, time, -1, p.x, p.y, ck ? p.z : 0.0, cx, cy, cz, w);
            last_probe_[n] = step;
        }
        for (size_t n = 0; n < samples_.size(); ++n) {
            if (!due(step, samples_[n].interval, force, last_sample_[n])) continue;
            const SampleSpec& s = samples_[n];
            for (int point = 0; point < s.points; ++point) {
                const double f = static_cast<double>(point) / static_cast<double>(s.points - 1);
                const double x = s.from[0] + f * (s.to[0] - s.from[0]);
                const double y = s.from[1] + f * (s.to[1] - s.from[1]);
                const double z = ck && s.from.size() == 3 ? s.from[2] + f * (s.to[2] - s.from[2]) : 0.0;
                const int c = leaf_at_point(mesh, levels, ci, cj, x, y, ck, z);
                if (c < 0) throw std::logic_error("post-processing sample did not map to an AMR leaf");
                const State w = eos.cons_to_prim(U[static_cast<size_t>(c)]);
                const int l = levels[static_cast<size_t>(c)];
                const double cx = mesh.x0 + (ci[static_cast<size_t>(c)] + 0.5) * mesh.dx(l);
                const double cy = mesh.y0 + (cj[static_cast<size_t>(c)] + 0.5) * mesh.dy(l);
                const double cz = ck ? mesh.z0 + ((*ck)[static_cast<size_t>(c)] + 0.5) * mesh.dz(l) : 0.0;
                write_point(sample_files_[n], step, time, point, x, y, z, cx, cy, cz, w);
            }
            last_sample_[n] = step;
        }
    }

    void write_forces(int step, double time, const std::array<double, 3>& loads, const Mesh& mesh, bool force = false) {
        for (size_t n = 0; n < forces_.size(); ++n) {
            if (!due(step, forces_[n].interval, force, last_force_[n])) continue;
            const ForceSpec& s = forces_[n];
            const double q = 0.5 * s.reference_density * s.reference_velocity * s.reference_velocity;
            const double area = s.reference_length * (mesh.is3d() ? mesh.Lz : 1.0);
            const double scale = q * area;
            std::fprintf(force_files_[n], "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n", step, time, loads[0],
                         loads[1], loads[2], scale > 0.0 ? loads[0] / scale : 0.0,
                         scale > 0.0 ? loads[1] / scale : 0.0,
                         scale > 0.0 ? -loads[2] / (scale * s.reference_length) : 0.0);
            std::fflush(force_files_[n]);
            last_force_[n] = step;
        }
    }

    void observe_averages(int step, double time, const std::array<double, 5>& means, bool force = false, bool emit = true) {
        for (size_t n = 0; n < averages_.size(); ++n) {
            const AverageSpec& spec = averages_[n];
            AverageState& state = avg_state_[n];
            std::vector<double> current;
            current.reserve(spec.variables.size());
            for (const std::string& name : spec.variables) current.push_back(means[primitive_index(name)]);
            if (!state.initialized) {
                state.start = state.previous_time = time;
                state.previous = current;
                state.integral.assign(current.size(), 0.0);
                state.initialized = true;
            } else if (time > state.previous_time) {
                const double dt = time - state.previous_time;
                for (size_t q = 0; q < current.size(); ++q)
                    state.integral[q] += 0.5 * (state.previous[q] + current[q]) * dt;
                state.previous_time = time;
                state.previous = current;
            }
            if (!emit) {
                state.last_written_step = step;
            } else if ((force || step % spec.interval == 0) && state.last_written_step != step) {
                std::fprintf(average_files_[n], "%d,%.17e", step, time);
                const double elapsed = time - state.start;
                for (size_t q = 0; q < current.size(); ++q)
                    std::fprintf(average_files_[n], ",%.17e", elapsed > 0.0 ? state.integral[q] / elapsed : current[q]);
                std::fprintf(average_files_[n], "\n");
                std::fflush(average_files_[n]);
                state.last_written_step = step;
            }
        }
    }

    void reset_averages(double time, const std::array<double, 5>& means) {
        for (size_t n = 0; n < averages_.size(); ++n) {
            const AverageSpec& spec = averages_[n];
            AverageState& state = avg_state_[n];
            state = AverageState{};
            for (const std::string& name : spec.variables) state.previous.push_back(means[primitive_index(name)]);
            state.integral.assign(state.previous.size(), 0.0);
            state.start = state.previous_time = time;
            state.initialized = true;
        }
    }

private:
    struct AverageState {
        bool initialized = false;
        double start = 0.0, previous_time = 0.0;
        int last_written_step = -1;
        std::vector<double> previous, integral;
    };

    std::FILE* open_csv(const std::string& suffix, const std::string& header) {
        const fs::path path = fs::path(dir_) / (prefix_ + "_" + output_token(suffix) + ".csv");
        std::FILE* f = open_post_csv(path, start_step_, header);
        files_.push_back(f);
        return f;
    }

    static bool due(int step, int interval, bool force, int last) { return (force || step % interval == 0) && last != step; }

    static int primitive_index(const std::string& name) {
        if (name == "rho") return 0;
        if (name == "u") return 1;
        if (name == "v") return 2;
        if (name == "p") return 3;
        return 4;
    }

    static void write_point(std::FILE* f, int step, double time, int point, double x, double y, double z, double cx,
                            double cy, double cz, const State& w) {
        if (point < 0)
            std::fprintf(f, "%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n", step,
                         time, x, y, z, cx, cy, cz, w[0], w[1], w[2], w[3], w[4]);
        else
            std::fprintf(f, "%d,%.17e,%d,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e,%.17e\n", step,
                         time, point, x, y, z, cx, cy, cz, w[0], w[1], w[2], w[3], w[4]);
        std::fflush(f);
    }

    void close_all() {
        for (std::FILE* f : files_)
            if (f) std::fclose(f);
        files_.clear();
    }

    std::vector<ProbeSpec> probes_;
    std::vector<SampleSpec> samples_;
    std::vector<ForceSpec> forces_;
    std::vector<AverageSpec> averages_;
    std::vector<AverageState> avg_state_;
    std::vector<int> last_probe_, last_sample_, last_force_;
    std::vector<std::FILE*> files_, probe_files_, sample_files_, force_files_, average_files_;
    std::string dir_, prefix_;
    int start_step_;
};

std::array<double, 5> volume_primitive_means(const EOS& eos, const Mesh& mesh, const Field& U) {
    std::array<double, 5> sums{};
    const double volume = mesh.cell_volume();
    for (const State& u : U) {
        const State w = eos.cons_to_prim(u);
        for (int q = 0; q < 5; ++q) sums[static_cast<size_t>(q)] += w[static_cast<size_t>(q)] * volume;
    }
    const double total = static_cast<double>(U.size()) * volume;
    for (double& x : sums) x /= total;
    return sums;
}

std::array<double, 5> volume_primitive_means(const EOS& eos, const CellMesh& mesh, const std::vector<int>& levels,
                                             const Field& U) {
    std::array<double, 5> sums{};
    double total = 0.0;
    for (size_t c = 0; c < levels.size(); ++c) {
        const double area = mesh.dim == 3 ? mesh.dx(levels[c]) * mesh.dy(levels[c]) * mesh.dz(levels[c])   // (the volume)
                                          : mesh.dx(levels[c]) * mesh.dy(levels[c]);
        const State w = eos.cons_to_prim(U[c]);
        for (int q = 0; q < 5; ++q) sums[static_cast<size_t>(q)] += w[static_cast<size_t>(q)] * area;
        total += area;
    }
    for (double& x : sums) x /= total;
    return sums;
}

bool has_postprocess(const TestCase& tc) {
    return !tc.probes.empty() || !tc.samples.empty() || !tc.forces.empty() || !tc.averages.empty();
}

bool postprocess_state_due(const TestCase& tc, int step) {
    for (const ProbeSpec& p : tc.probes) if (step % p.interval == 0) return true;
    for (const SampleSpec& s : tc.samples) if (step % s.interval == 0) return true;
    for (const ForceSpec& f : tc.forces) if (step % f.interval == 0) return true;
    return !tc.averages.empty();   // a time average is integrated at every completed step
}

bool amr_postprocess_state_due(const TestCase& tc, int step) {
    for (const ProbeSpec& p : tc.probes) if (step % p.interval == 0) return true;
    for (const SampleSpec& s : tc.samples) if (step % s.interval == 0) return true;
    return !tc.averages.empty();   // force outputs are computed from the adaptive residual separately
}

bool postprocess_force_due(const TestCase& tc, int step) {
    for (const ForceSpec& f : tc.forces) if (step % f.interval == 0) return true;
    return false;
}

}   // namespace

Field Simulation::initial_field() const {
    if (test_case->three_d) {
        if (!mesh.is3d()) throw std::invalid_argument("Test case '" + test_case->key + "' is 3-D: set nz > 1");
        return test_case->initialize(mesh, fluid, p.source);
    }
    if (!mesh.is3d()) return test_case->initialize(mesh, fluid, p.source);
    // 2-D case on a 3-D mesh: initialise one layer and extrude it along z
    const Mesh m2 = generate_cartesian_mesh(mesh.Lx, mesh.Ly, mesh.nx, mesh.ny);
    const Field layer = test_case->initialize(m2, fluid, p.source);
    Field U3(static_cast<size_t>(mesh.num_cells()));
    for (int k = 0; k < mesh.nz; ++k)
        std::copy(layer.begin(), layer.end(), U3.begin() + static_cast<std::ptrdiff_t>(k) * layer.size());
    return U3;
}

// ---------------------------------------------------------------------------
// Residuals
// ---------------------------------------------------------------------------

// Reconstruction and Riemann solver in one pass (MusclReconstruction::muscl_fluxes): the MUSCL scheme on the AVX2 kernels; the
// results are the same as with the two passes (FENICE_NO_FUSED=1 takes those, for A/B timing).
static bool fused_enabled() {
    static const bool off = switch_on("FENICE_NO_FUSED");
    return !off;
}
static bool fused_muscl(const MusclReconstruction& mu) { return fused_enabled() && mu.scheme() == Reconstruction::Muscl; }

double Simulation::assemble(const Mesh& m, const BoundaryManager& bm, const ImmersedBoundary* ib,
                            const MusclReconstruction* mu, const Field& U_in, Field& R, const StripHalos* halos) {
    if (m.is3d()) return assemble_3d(m, bm, ib, mu, U_in, R, halos);
    const Field* Uc = &U_in;
    if (ib) {
        ProfileScope profile(Section::IBMForcing);
        Uf_ = U_in;
        ib->apply_forcing(Uf_, fluid);
        Uc = &Uf_;
    }
    const int nx = m.nx, ny = m.ny, nxf = m.num_xfaces();
    constexpr int D = kHaloDepth;

    TangentialHalos th;
    if (halos) th = {halos->dir, D, halos->lo, halos->hi};
    // with a reconstruction every face between two cells is overwritten below: only the sides need their first-order states
    {
        ProfileScope ps(Section::Boundary);
        bm.apply_boundary_conditions(*Uc, UL_, UR_, halos ? &th : nullptr, !(mu ? true : fused_enabled()));
    }

    Halos mh;
    if (halos) {
        // MPI halos override the physical-BC ghost states at interface faces
        if (halos->dir == 'y') {
            if (halos->lo)
                for (int i = 0; i < nx; ++i) UL_[m.boundary_face(Face::Bottom, i)] = (*halos->lo)[(D - 1) * nx + i];
            if (halos->hi)
                for (int i = 0; i < nx; ++i) UR_[m.boundary_face(Face::Top, i)] = (*halos->hi)[i];
            mh.b = halos->lo;
            mh.t = halos->hi;
        } else {
            if (halos->lo)
                for (int j = 0; j < ny; ++j) UL_[m.boundary_face(Face::Left, j)] = (*halos->lo)[j * D + D - 1];
            if (halos->hi)
                for (int j = 0; j < ny; ++j) UR_[m.boundary_face(Face::Right, j)] = (*halos->hi)[j * D];
            mh.l = halos->lo;
            mh.r = halos->hi;
        }
    }
    // WENO5-Z reads three cells: its periodic halos and ghost layers are three deep (serial only)
    const int Dh = (mu && mu->scheme() == Reconstruction::Weno5) ? 3 : D;
    mh.depth = Dh;
    // Periodic axes inside the (sub)domain: wrap-around halos so MUSCL also reconstructs
    // the periodic boundary faces (otherwise they would be a 1st-order seam).
    const bool per_x = bm.type(Face::Left) == BCType::Periodic && bm.type(Face::Right) == BCType::Periodic;
    const bool per_y = bm.type(Face::Bottom) == BCType::Periodic && bm.type(Face::Top) == BCType::Periodic;
    if (mu && per_x && !mh.l && !mh.r && nx >= Dh) {
        perL_.resize(static_cast<size_t>(ny) * Dh);
        perR_.resize(static_cast<size_t>(ny) * Dh);
        for (int j = 0; j < ny; ++j)
            for (int d = 0; d < Dh; ++d) {
                perL_[j * Dh + d] = (*Uc)[j * nx + nx - Dh + d];
                perR_[j * Dh + d] = (*Uc)[j * nx + d];
            }
        mh.l = &perL_;
        mh.r = &perR_;
    }
    if (mu && per_y && !mh.b && !mh.t && ny >= Dh) {
        perB_.assign(Uc->end() - static_cast<std::ptrdiff_t>(Dh) * nx, Uc->end());
        perT_.assign(Uc->begin(), Uc->begin() + static_cast<std::ptrdiff_t>(Dh) * nx);
        mh.b = &perB_;
        mh.t = &perT_;
    }
    // boundary_order = 2: the physical sides get ghost layers (the BC's own even / odd extension), so that the faces on
    // the boundary are reconstructed like the interior ones. A side that is periodic, an MPI interface (halo already
    // set) or non-reflecting keeps its first-order face states.
    if (mu && p.boundary_order == 2) {
        const auto ghost = [&](Face side, const Field*& slot, Field& buf) {
            if (slot) return;
            const BCType bt = bm.type(side);
            if (bt == BCType::Periodic || bt == BCType::NonReflecting) return;
            if (bm.fill_ghost_slab(*Uc, side, Dh, buf)) slot = &buf;
        };
        ghost(Face::Left, mh.l, ghL_);
        ghost(Face::Right, mh.r, ghR_);
        ghost(Face::Bottom, mh.b, ghB_);
        ghost(Face::Top, mh.t, ghT_);
    }
    double S;
    if (mu && fused_muscl(*mu)) {
        // reconstruction, Riemann solver and flux balance in one pass over rows of faces: the face fluxes are not stored
        ProfileScope ps(Section::Flux);
        S = mu->muscl_residual_2d(riemann, UL_, UR_, *Uc, m, mh, R);
    } else if (!mu && fused_enabled()) {
        ProfileScope ps(Section::Flux);   // first order: the same pass, the states of the faces are the cells
        S = riemann.first_order_residual_2d(UL_, UR_, *Uc, m, R);
    } else {
        if (mu) {
            ProfileScope ps(Section::Reconstruct);
            mu->reconstruct(UL_, UR_, *Uc, nx, ny, mh);
        }
        {
            ProfileScope ps(Section::Flux);
            S = riemann.compute_flux(UL_, UR_, m, F_);
        }
        ProfileScope ps(Section::Divergence);
        R.resize(Uc->size());
        const double inv_area = 1.0 / m.cell_area(), dxl = m.dx, dyl = m.dy;
#pragma omp parallel for schedule(static) if (nx * ny >= kOmpMinWork)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const State& fl = F_[j * (nx + 1) + i];
                const State& fr = F_[j * (nx + 1) + i + 1];
                const State& fb = F_[nxf + j * nx + i];
                const State& ft = F_[nxf + (j + 1) * nx + i];
                State& r = R[j * nx + i];
                for (int q = 0; q < kNVar; ++q)
                    r[q] = ((0.0 - (fr[q] * dyl - fl[q] * dyl)) - (ft[q] * dxl - fb[q] * dxl)) * inv_area;
            }
    }
    if (ns) {
        ProfileScope ps(Section::Viscous);
        {
            // the ghost ring of the padded state only: the interior is the cell field, which the viscous sweep reads in place
            ProfileScope profile(Section::PadCopy);
            bm.get_padded_frame(*Uc, frame_);
            if (halos) {
                if (halos->dir == 'y') {
                    if (halos->lo)
                        for (int i = 0; i < nx; ++i) frame_.bottom[i + 1] = (*halos->lo)[(D - 1) * nx + i];
                    if (halos->hi)
                        for (int i = 0; i < nx; ++i) frame_.top[i + 1] = (*halos->hi)[i];
                } else {
                    if (halos->lo)
                        for (int j = 0; j < ny; ++j) frame_.left[j] = (*halos->lo)[j * D + D - 1];
                    if (halos->hi)
                        for (int j = 0; j < ny; ++j) frame_.right[j] = (*halos->hi)[j * D];
                }
            }
            bm.fix_frame_corners(frame_, halos ? (halos->dir == 'x' ? 0 : 1) : -1, halos && halos->lo, halos && halos->hi);
        }
        CrossGhosts cg;
        cg.periodic_x = per_x;
        cg.periodic_y = per_y;
        std::vector<double> lo_a, lo_b, hi_a, hi_b;
        if (halos) {
            cg.dir = halos->dir;
            if (halos->lo) {
                strip_tangential_derivs(fluid, bm, m, halos->dir, adjacent_strip(*halos->lo, halos->dir, true, nx, ny),
                                        lo_a, lo_b);
                cg.lo_a = lo_a.data();
                cg.lo_b = lo_b.data();
            }
            if (halos->hi) {
                strip_tangential_derivs(fluid, bm, m, halos->dir, adjacent_strip(*halos->hi, halos->dir, false, nx, ny),
                                        hi_a, hi_b);
                cg.hi_a = hi_a.data();
                cg.hi_b = hi_b.data();
            }
        }
        const double s_visc = ns->viscous_residual_rows(*Uc, frame_, m, R, &cg, true);
        S = std::max(S, s_visc);
    }

    if (!rhs_sources.empty()) rhs_sources.apply_uniform(m, *Uc, R, scheme ? scheme->stage_time : 0.0);
    return S;
}

double Simulation::assemble_3d(const Mesh& m, const BoundaryManager& bm, const ImmersedBoundary* ib,
                               const MusclReconstruction* mu, const Field& U_in, Field& R, const StripHalos* halos) {
    const Field* Uc = &U_in;
    if (ib) {
        ProfileScope profile(Section::IBMForcing);
        Uf_ = U_in;
        ib->apply_forcing(Uf_, fluid);
        Uc = &Uf_;
    }
    const Field& Uv = *Uc;
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    constexpr int D = kHaloDepth;

    TangentialHalos th;
    if (halos) th = {halos->dir, D, halos->lo, halos->hi};
    {
        ProfileScope ps(Section::Boundary);
        bm.apply_boundary_conditions(Uv, UL_, UR_, halos ? &th : nullptr, mu == nullptr);
    }

    // MPI halos (slab layouts, see Halos) override the BC ghosts at interface faces
    Halos mh;
    if (halos) {
        if (halos->dir == 'y') {
            for (int k = 0; k < nz; ++k)
                for (int i = 0; i < nx; ++i) {
                    if (halos->lo)
                        UL_[m.boundary_face(Face::Bottom, k * nx + i)] = (*halos->lo)[((D - 1) * nz + k) * nx + i];
                    if (halos->hi) UR_[m.boundary_face(Face::Top, k * nx + i)] = (*halos->hi)[k * nx + i];
                }
            mh.b = halos->lo;
            mh.t = halos->hi;
        } else {
            for (int s = 0; s < ny * nz; ++s) {
                if (halos->lo) UL_[m.boundary_face(Face::Left, s)] = (*halos->lo)[s * D + D - 1];
                if (halos->hi) UR_[m.boundary_face(Face::Right, s)] = (*halos->hi)[s * D];
            }
            mh.l = halos->lo;
            mh.r = halos->hi;
        }
    }
    const int Dh = (mu && mu->scheme() == Reconstruction::Weno5) ? 3 : D;   // WENO5-Z reads three cells
    mh.depth = Dh;
    // Periodic axes: wrap-around MUSCL halos (the boundary faces are then 2nd order too)
    const bool per[3] = {bm.type(Face::Left) == BCType::Periodic && bm.type(Face::Right) == BCType::Periodic,
                         bm.type(Face::Bottom) == BCType::Periodic && bm.type(Face::Top) == BCType::Periodic,
                         bm.type(Face::Front) == BCType::Periodic && bm.type(Face::Back) == BCType::Periodic};
    if (mu) {
        if (per[0] && !mh.l && !mh.r && nx >= Dh) {
            perL_.resize(static_cast<size_t>(ny) * nz * Dh);
            perR_.resize(perL_.size());
            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int d = 0; d < Dh; ++d) {
                        perL_[(k * ny + j) * Dh + d] = Uv[m.cell(nx - Dh + d, j, k)];
                        perR_[(k * ny + j) * Dh + d] = Uv[m.cell(d, j, k)];
                    }
            mh.l = &perL_;
            mh.r = &perR_;
        }
        if (per[1] && !mh.b && !mh.t && ny >= Dh) {
            perB_.resize(static_cast<size_t>(Dh) * nz * nx);
            perT_.resize(perB_.size());
            for (int d = 0; d < Dh; ++d)
                for (int k = 0; k < nz; ++k)
                    for (int i = 0; i < nx; ++i) {
                        perB_[(d * nz + k) * nx + i] = Uv[m.cell(i, ny - Dh + d, k)];
                        perT_[(d * nz + k) * nx + i] = Uv[m.cell(i, d, k)];
                    }
            mh.b = &perB_;
            mh.t = &perT_;
        }
        if (per[2] && nz >= Dh) {
            perF_.resize(static_cast<size_t>(Dh) * ny * nx);
            perK_.resize(perF_.size());
            for (int d = 0; d < Dh; ++d)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i) {
                        perF_[(d * ny + j) * nx + i] = Uv[m.cell(i, j, nz - Dh + d)];
                        perK_[(d * ny + j) * nx + i] = Uv[m.cell(i, j, d)];
                    }
            mh.f = &perF_;
            mh.bk = &perK_;
        }
        // boundary_order = 2: the physical sides get ghost layers (the BC's own even / odd extension), so that the faces on
        // the boundary are reconstructed like the interior ones. A side that is periodic, an MPI interface (halo already
        // set) or non-reflecting keeps its first-order face states.
        if (mu && p.boundary_order == 2) {
            const auto ghost = [&](Face side, const Field*& slot, Field& buf) {
                if (slot) return;
                const BCType bt = bm.type(side);
                if (bt == BCType::Periodic || bt == BCType::NonReflecting) return;
                if (bm.fill_ghost_slab(Uv, side, Dh, buf)) slot = &buf;
            };
            ghost(Face::Left, mh.l, ghL_);
            ghost(Face::Right, mh.r, ghR_);
            ghost(Face::Bottom, mh.b, ghB_);
            ghost(Face::Top, mh.t, ghT_);
            ghost(Face::Front, mh.f, ghF_);
            ghost(Face::Back, mh.bk, ghK_);
        }
    }

    double S;
    if (mu && fused_muscl(*mu)) {
        ProfileScope ps(Section::Flux);   // reconstruction and Riemann solver in one pass
        S = mu->muscl_fluxes(riemann, UL_, UR_, Uv, m, mh, F_);
    } else {
        if (mu) {
            ProfileScope ps(Section::Reconstruct);
            mu->reconstruct_3d(UL_, UR_, Uv, m, mh);
        }
        ProfileScope ps(Section::Flux);
        S = riemann.compute_flux(UL_, UR_, m, F_);
    }

    ProfileScope divergence_scope(Section::Divergence);
    R.resize(Uv.size());
    const double Ax = m.face_area(0), Ay = m.face_area(1), Az = m.face_area(2), inv_vol = 1.0 / m.cell_volume();
#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
    for (int kj = 0; kj < ny * nz; ++kj) {
        const int j = kj % ny, k = kj / ny;
        for (int i = 0; i < nx; ++i) {
            const State& fl = F_[m.xface(i, j, k)];
            const State& fr = F_[m.xface(i + 1, j, k)];
            const State& fb = F_[m.yface(i, j, k)];
            const State& ft = F_[m.yface(i, j + 1, k)];
            const State& ff = F_[m.zface(i, j, k)];
            const State& fk = F_[m.zface(i, j, k + 1)];
            State& r = R[m.cell(i, j, k)];
            for (int q = 0; q < kNVar; ++q)
                r[q] =
                    (((0.0 - (fr[q] * Ax - fl[q] * Ax)) - (ft[q] * Ay - fb[q] * Ay)) - (fk[q] * Az - ff[q] * Az)) * inv_vol;
        }
    }

    divergence_scope.stop();
    if (ns) {
        ProfileScope ps(Section::Viscous);
        {
            ProfileScope profile(Section::PadCopy);
            bm.get_padded_U(Uv, Upad_);
        }
        CrossGhosts cg;
        cg.periodic_x = per[0];
        cg.periodic_y = per[1];
        cg.periodic_z = per[2];
        std::vector<double> gh[8];   // MPI cross ghosts: lo a, b, c, d, hi a, b, c, d
        if (halos) {
            const int PX = nx + 2, PY = ny + 2;
            auto P = [&](int ii, int jj, int kk) -> State& {
                return Upad_[(static_cast<size_t>(kk) * PY + jj) * PX + ii];
            };
            if (halos->dir == 'y') {
                for (int k = 0; k < nz; ++k)
                    for (int i = 0; i < nx; ++i) {
                        if (halos->lo) P(i + 1, 0, k + 1) = (*halos->lo)[((D - 1) * nz + k) * nx + i];
                        if (halos->hi) P(i + 1, ny + 1, k + 1) = (*halos->hi)[k * nx + i];
                    }
            } else {
                for (int k = 0; k < nz; ++k)
                    for (int j = 0; j < ny; ++j) {
                        if (halos->lo) P(0, j + 1, k + 1) = (*halos->lo)[(k * ny + j) * D + D - 1];
                        if (halos->hi) P(nx + 1, j + 1, k + 1) = (*halos->hi)[(k * ny + j) * D];
                    }
            }
            cg.dir = halos->dir;
            const double** slots[8] = {&cg.lo_a, &cg.lo_b, &cg.lo_c, &cg.lo_d, &cg.hi_a, &cg.hi_b, &cg.hi_c, &cg.hi_d};
            for (int side = 0; side < 2; ++side) {
                const Field* hal = side == 0 ? halos->lo : halos->hi;
                if (!hal) continue;
                slab_tangential_derivs(fluid, bm, m, halos->dir, adjacent_slab(*hal, halos->dir, side == 0, m),
                                       gh[4 * side], gh[4 * side + 1], gh[4 * side + 2], gh[4 * side + 3]);
                for (int q = 0; q < 4; ++q) *slots[4 * side + q] = gh[4 * side + q].data();
            }
        }
        {
            ProfileScope profile(Section::PadCornerFix);
            bm.fix_padded_corners(Upad_, halos ? (halos->dir == 'x' ? 0 : 1) : -1, halos && halos->lo, halos && halos->hi);
        }
        compute_gradients_3d(Upad_, m, gX_, gY_, gZ_);
        const double s_visc = ns->viscous_residual_3d(Uv, gX_, gY_, gZ_, m, Upad_, R, &cg, dt_length, true);
        S = std::max(S, s_visc);
    }

    if (!rhs_sources.empty()) rhs_sources.apply_uniform(m, Uv, R, scheme ? scheme->stage_time : 0.0);
    return S;
}
const Field& Simulation::source_for(const Mesh& m, double stage_time) {
    const std::array<double, 9> key = {static_cast<double>(m.nx), static_cast<double>(m.ny),
                                       static_cast<double>(m.nz), m.dx, m.dy, m.dz, m.x0, m.y0, m.z0};
    if (p.source_unsteady) {
        auto ut = unsteady_source_cache_.find(key);
        if (ut == unsteady_source_cache_.end())
            ut = unsteady_source_cache_.emplace(key, std::make_pair(std::numeric_limits<double>::quiet_NaN(), Field{}))
                     .first;
        if (!(ut->second.first == stage_time)) {   // also true for the NaN of a new entry
            ut->second.second = evaluate_source(m, p.source, stage_time);
            ut->second.first = stage_time;
        }
        return ut->second.second;
    }
    auto it = source_cache_.find(key);
    if (it == source_cache_.end()) it = source_cache_.emplace(key, evaluate_source(m, p.source)).first;
    return it->second;
}

double Simulation::residual(const Field& Uin, Field& R) {
    update_boundary_values(scheme ? scheme->stage_time : t, boundaries);
    return assemble(mesh, boundaries, p.use_ibm ? ibm.get() : nullptr, muscl.get(), Uin, R, nullptr);
}

// ---------------------------------------------------------------------------
// Checkpoints / hot start
// ---------------------------------------------------------------------------

std::string Simulation::checkpoint_path() const {
    return (fs::path(exporter.output_dir()) / (file_prefix + "_restart.npz")).string();
}

namespace {
// Version of the checkpoint layout (1 = the keys written below). A file without the key is older than the key and is read as
// version 1 (missing keys have defaults); a file with a higher number is from a newer solver and is refused.
constexpr int kCheckpointFormat = 1;

void require_known_format(const std::string& path, bool present, double version) {
    if (present && version > kCheckpointFormat)
        throw std::runtime_error("The checkpoint " + path + " has format_version " + std::to_string(static_cast<int>(version)) +
                                 ", newer than this solver reads (" + std::to_string(kCheckpointFormat) + "): use a newer fenice");
}
}  // namespace

void Simulation::save_checkpoint(int step, const LeafArrays* leaves) {
    const std::string path = checkpoint_path();
    const std::string tmp = path.substr(0, path.size() - 4) + "_tmp.npz";
    std::vector<double> times;
    std::vector<std::string> files;
    for (const auto& e : exporter.collection) {
        times.push_back(e.time);
        files.push_back(e.filename);
    }
    NpzArchive a;
    if (cells) {
        // an adaptive run: the leaves (level and integer coordinates) and their state, and the wave-speed bound the next
        // time step is taken from
        // (an octree also stores k and the fifth state component rho w; the 2-D file is unchanged)
        const bool octree = cells->dim == 3;
        const LeafArrays lf = leaves ? *leaves : LeafArrays{&cells->level, &cells->ci, &cells->cj, &cells->U, octree ? &cells->ck : nullptr};
        const size_t n = lf.level->size();
        std::vector<double> lev(n), li(n), lj(n), lk(octree ? n : 0);
        for (size_t c = 0; c < n; ++c) {
            lev[c] = (*lf.level)[c];
            li[c] = (*lf.ci)[c];
            lj[c] = (*lf.cj)[c];
            if (octree) lk[c] = (*lf.ck)[c];
        }
        a["U"] = field_to_npy(*lf.U, octree ? kNVar : 4);
        a["leaf_level"] = NpyArray::from_doubles(lev, {static_cast<int64_t>(n)});
        a["leaf_i"] = NpyArray::from_doubles(li, {static_cast<int64_t>(n)});
        a["leaf_j"] = NpyArray::from_doubles(lj, {static_cast<int64_t>(n)});
        if (octree) a["leaf_k"] = NpyArray::from_doubles(lk, {static_cast<int64_t>(n)});
        a["S"] = NpyArray::from_double(cell_speed_);
    } else {
        a["U"] = field_to_npy(U, mesh.is3d() ? kNVar : 4);
    }
    a["format_version"] = NpyArray::from_int(kCheckpointFormat);
    a["t"] = NpyArray::from_double(t);
    a["step"] = NpyArray::from_int(step);
    a["nx"] = NpyArray::from_int(mesh.nx);
    a["ny"] = NpyArray::from_int(mesh.ny);
    a["Lx"] = NpyArray::from_double(mesh.Lx);
    a["Ly"] = NpyArray::from_double(mesh.Ly);
    a["nz"] = NpyArray::from_int(mesh.nz);
    a["Lz"] = NpyArray::from_double(mesh.Lz);
    a["collection_time"] = NpyArray::from_doubles(times, {static_cast<int64_t>(times.size())});
    a["collection_file"] = NpyArray::from_strings(files);
    write_npz(tmp, a);
    rename_replace(tmp, path);
}

int Simulation::load_checkpoint() {
    const std::string path = checkpoint_path();
    if (!fs::exists(path)) {
        std::printf("\n[WARNING] No checkpoint found in %s. Starting from zero.\n", path.c_str());
        return 0;
    }
    const NpzFile file(path);
    require_known_format(path, file.has("format_version"), file.has("format_version") ? file.read("format_version").scalar() : 0);
    const bool adaptive_file = file.has("leaf_level");
    // Several ranks keeping only their shares: the leaves (integers) give the topology every rank cuts its share from, and each rank
    // reads only the state rows of its own leaves and their halo; the whole state and the whole mesh are never held
    const bool share = adaptive_file && cells && cell_par && !replicated_storage();
    NpzArchive a;
    if (share) {
        for (const char* key : {"t", "step", "S", "collection_time", "collection_file", "collection"})
            if (file.has(key)) a[key] = file.read(key);
    } else {
        a = read_npz(path);
    }
    if (adaptive_file != (cells != nullptr))
        throw std::runtime_error(std::string("The checkpoint ") + path + " is from " +
                                 (adaptive_file ? "an adaptive (AMR)" : "a uniform-mesh") + " run: resume it with use_amr " +
                                 (adaptive_file ? "on" : "off") + " (a uniform checkpoint can seed an AMR run with a hot start)");
    if (adaptive_file) {
        // a 2-D checkpoint has no leaf_k; an octree checkpoint is resumed on the mesh it was written for
        const bool file_3d = file.has("leaf_k");
        if (file_3d != (cells->dim == 3))
            throw std::runtime_error(std::string("The checkpoint ") + path + " is from a " + (file_3d ? "3-D" : "2-D") +
                                     " adaptive run: resume it with nz " + (file_3d ? "> 1" : "= 1"));
        if (file_3d) {
            const auto num = [&](const char* key) { return file.read(key).scalar(); };
            if (static_cast<int>(num("nx")) != mesh.nx || static_cast<int>(num("ny")) != mesh.ny || static_cast<int>(num("nz")) != mesh.nz ||
                std::abs(num("Lx") - mesh.Lx) > 1e-10 || std::abs(num("Ly") - mesh.Ly) > 1e-10 || std::abs(num("Lz") - mesh.Lz) > 1e-10)
                throw std::runtime_error("The checkpoint " + path + " is from a base mesh of another size or domain: use_hot_start maps it onto this one");
        }
    }
    if (share) {
        const auto ints = [&](const char* key) {
            const std::vector<double> d = file.read(key).as_doubles();
            return std::vector<int>(d.begin(), d.end());
        };
        {
            const bool octree = cells->dim == 3;
            std::vector<int> lv = ints("leaf_level"), li = ints("leaf_i"), lj = ints("leaf_j"), lk = octree ? ints("leaf_k") : std::vector<int>();
            CellMesh topo(*cells);
            topo.U = Field();
            topo.set_faceless(true);
            if (octree) topo.assign_leaves(lv, li, lj, lk, nullptr);
            else topo.assign_leaves(lv, li, lj, nullptr);   // sorts by (level, Morton) and checks that the leaves tile the domain
            std::vector<int> file_row(static_cast<size_t>(topo.num_leaves()), -1);   // row of the checkpoint of each leaf of the topology
            for (size_t r = 0; r < lv.size(); ++r)
                file_row[static_cast<size_t>(topo.find(lv[r], li[r], lj[r], octree ? lk[r] : 0))] = static_cast<int>(r);
            lv = li = lj = lk = std::vector<int>();
            cell_par->resume_share(topo, [&](CellMesh& m) {
                const size_t n = static_cast<size_t>(m.num_leaves());
                std::vector<int64_t> rows(n);
                for (size_t c = 0; c < n; ++c)
                    rows[c] = file_row[static_cast<size_t>(topo.find(m.level[c], m.ci[c], m.cj[c], octree ? m.ck[c] : 0))];
                int ncol = 0;
                const std::vector<double> d = file.read_rows("U", rows, ncol);
                if (ncol != 4 && ncol != kNVar) throw std::runtime_error("Expected an (n, 4) or (n, 5) state array");
                if (octree && ncol != kNVar) throw std::runtime_error("The checkpoint of a 3-D adaptive run needs an (n, 5) state array");
                m.U.assign(n, State{});
                for (size_t c = 0; c < n; ++c)
                    for (int q = 0; q < ncol; ++q) m.U[c][static_cast<size_t>(q)] = d[c * static_cast<size_t>(ncol) + static_cast<size_t>(q)];
            });
        }
        share_resumed_ = true;
        cell_speed_ = a.count("S") ? a.at("S").scalar() : 0.0;
    } else if (cells) {
        const auto ints = [&](const char* key) {
            const std::vector<double> d = a.at(key).as_doubles();
            return std::vector<int>(d.begin(), d.end());
        };
        Field Uc = npy_to_field(a.at("U"));
        if (cells->dim == 3) {
            if (a.at("U").shape.size() != 2 || a.at("U").shape[1] != kNVar)
                throw std::runtime_error("The checkpoint of a 3-D adaptive run needs an (n, 5) state array");
            cells->assign_leaves(ints("leaf_level"), ints("leaf_i"), ints("leaf_j"), ints("leaf_k"), &Uc);
        } else {
            cells->assign_leaves(ints("leaf_level"), ints("leaf_i"), ints("leaf_j"), &Uc);
        }
        cell_engine->build(*cells);
        cell_speed_ = a.count("S") ? a.at("S").scalar() : 0.0;
    } else {
        U = npy_to_field(a.at("U"));
    }
    t = a.at("t").scalar();
    const int step = static_cast<int>(a.at("step").scalar());
    exporter.collection.clear();
    if (a.count("collection_time") && a.count("collection_file")) {
        const auto times = a.at("collection_time").as_doubles();
        const auto files = a.at("collection_file").as_strings();
        for (size_t k = 0; k < std::min(times.size(), files.size()); ++k)
            exporter.collection.push_back({times[k], files[k]});
    } else if (a.count("collection")) {
        std::printf("[RESTART] Legacy checkpoint (pickled index): .pvd index restarts from this point\n");
    }
    std::printf("\n[RESTART] Simulation resumed from t = %.5f s (Step %d)\n", t, step);
    return step;
}

Field Simulation::load_hot_start(const std::string& path) {
    // If the source is this run's own checkpoint, back it up before the first save overwrites it.
    const std::string ck = checkpoint_path();
    std::error_code ec;
    if (fs::exists(path) && fs::exists(ck) && fs::equivalent(path, ck, ec)) {
        std::string backup = ck;
        backup.replace(backup.size() - std::string("_restart.npz").size(), std::string::npos, "_hotstart_source.npz");
        fs::copy_file(path, backup, fs::copy_options::overwrite_existing);
        std::printf("[Hot Start] Source checkpoint backed up to: %s\n", backup.c_str());
    }
    const NpzArchive a = read_npz(path);
    require_known_format(path, a.count("format_version") > 0, a.count("format_version") ? a.at("format_version").scalar() : 0);
    int nx_c, ny_c;
    double Lx_c, Ly_c;
    if (a.count("nx")) {
        nx_c = static_cast<int>(a.at("nx").scalar());
        ny_c = static_cast<int>(a.at("ny").scalar());
        Lx_c = a.at("Lx").scalar();
        Ly_c = a.at("Ly").scalar();
    } else {
        nx_c = p.hot_start_src_nx;
        ny_c = p.hot_start_src_ny;
        if (nx_c <= 0 || ny_c <= 0)
            throw std::runtime_error("Hot-start file '" + path +
                                     "' has no grid metadata and no manual Nx / Ny were provided "
                                     "(set hot_start_src_nx / hot_start_src_ny).");
        Lx_c = mesh.Lx;
        Ly_c = mesh.Ly;
        std::printf("[Hot Start] No metadata in file - using manual dims: Nx=%d, Ny=%d, Lx=%g, Ly=%g\n", nx_c, ny_c,
                    Lx_c, Ly_c);
    }
    if (std::abs(Lx_c - mesh.Lx) > 1e-10 || std::abs(Ly_c - mesh.Ly) > 1e-10)
        std::printf("[WARNING] Hot-start domain (%g x %g) differs from target (%g x %g). "
                    "Proceeding with coordinate rescaling.\n",
                    Lx_c, Ly_c, mesh.Lx, mesh.Ly);
    int nz_c = a.count("nz") ? static_cast<int>(a.at("nz").scalar()) : 1;
    if (nz_c > 1 && !mesh.is3d()) throw std::runtime_error("Hot start from a 3-D checkpoint needs a 3-D run (nz > 1)");
    std::printf("[Hot Start] Mapping %d x %d x %d -> %d x %d x %d ...\n", nx_c, ny_c, nz_c, mesh.nx, mesh.ny, mesh.nz);
    Field Uc = npy_to_field(a.at("U"));
    hot_src_.clear();
    hot_src_nx_ = hot_src_ny_ = 0;
    hot_src_nz_ = 1;
    if (a.count("leaf_level")) {
        // the checkpoint of an adaptive run: its leaves on the uniform mesh of their finest level (nx, ny, nz are the base mesh)
        const bool src_3d = a.count("leaf_k") > 0;
        if (src_3d != (nz_c > 1)) throw std::runtime_error("The adaptive checkpoint has inconsistent leaves (leaf_k and nz disagree)");
        if (mesh.is3d() != src_3d)
            throw std::runtime_error(std::string("Hot start from a ") + (src_3d ? "3-D" : "2-D") + " adaptive (AMR) checkpoint needs a " +
                                     (src_3d ? "3-D" : "2-D") + " run");
        const auto ints = [&](const char* key) {
            const std::vector<double> d = a.at(key).as_doubles();
            return std::vector<int>(d.begin(), d.end());
        };
        int nxf = 0, nyf = 0, nzf = 1;
        if (src_3d) {
            Uc = leaves_to_uniform_3d(Uc, ints("leaf_level"), ints("leaf_i"), ints("leaf_j"), ints("leaf_k"), nx_c, ny_c, nz_c, nxf, nyf, nzf);
            std::printf("[Hot Start] Adaptive checkpoint: %d x %d x %d base mesh, finest level as %d x %d x %d cells\n", nx_c, ny_c, nz_c,
                        nxf, nyf, nzf);
        } else {
            Uc = leaves_to_uniform(Uc, ints("leaf_level"), ints("leaf_i"), ints("leaf_j"), nx_c, ny_c, nxf, nyf);
            std::printf("[Hot Start] Adaptive checkpoint: %d x %d base mesh, finest level as %d x %d cells\n", nx_c, ny_c, nxf, nyf);
        }
        nx_c = nxf;
        ny_c = nyf;
        nz_c = nzf;
    }
    if (cells && a.count("leaf_level") && (nx_c != mesh.nx || ny_c != mesh.ny || nz_c != mesh.nz)) {
        // an adaptive target samples the finest-level field of an adaptive source at the leaf centres: keep it at its own
        // resolution, not on the base mesh (a uniform source is mapped onto the base mesh first)
        hot_src_ = Uc;
        hot_src_nx_ = nx_c;
        hot_src_ny_ = ny_c;
        hot_src_nz_ = nz_c;
    }
    if (static_cast<int>(Uc.size()) != nx_c * ny_c * nz_c)
        throw std::runtime_error("Hot-start state has " + std::to_string(Uc.size()) + " cells, expected " +
                                 std::to_string(nx_c * ny_c * nz_c));
    if (nz_c > 1) return map_solution_3d(Uc, nx_c, ny_c, nz_c, mesh.nx, mesh.ny, mesh.nz, mesh.Lx, mesh.Ly, mesh.Lz);
    Field U2 = map_solution(Uc, nx_c, ny_c, mesh.nx, mesh.ny, mesh.Lx, mesh.Ly);
    if (!mesh.is3d()) return U2;
    Field U3(static_cast<size_t>(mesh.num_cells()));   // 2-D source on a 3-D mesh: extrude along z
    for (int k = 0; k < mesh.nz; ++k)
        std::copy(U2.begin(), U2.end(), U3.begin() + static_cast<std::ptrdiff_t>(k) * U2.size());
    return U3;
}

void Simulation::dump_final_state(int step, const LeafArrays* leaves) const {
    if (p.dump_state.empty()) return;
    NpzArchive a;
    a["t"] = NpyArray::from_double(t);
    a["step"] = NpyArray::from_int(step);
    if (cells) {
        const bool octree = cells->dim == 3;
        const LeafArrays lf = leaves ? *leaves : LeafArrays{&cells->level, &cells->ci, &cells->cj, &cells->U, octree ? &cells->ck : nullptr};
        const size_t n = lf.level->size(), cols = octree ? 7 : 5;   // x0, x1, y0, y1[, z0, z1], level
        std::vector<double> meta;
        meta.reserve(n * cols);
        for (size_t k = 0; k < n; ++k) {
            const int l = (*lf.level)[k], i = (*lf.ci)[k], j = (*lf.cj)[k];
            meta.insert(meta.end(), {cells->x0 + i * cells->dx(l), cells->x0 + (i + 1) * cells->dx(l),
                                     cells->y0 + j * cells->dy(l), cells->y0 + (j + 1) * cells->dy(l)});
            if (octree) {
                const int z = (*lf.ck)[k];
                meta.insert(meta.end(), {cells->z0 + z * cells->dz(l), cells->z0 + (z + 1) * cells->dz(l)});
            }
            meta.push_back(static_cast<double>(l));
        }
        a["U"] = field_to_npy(*lf.U, octree ? kNVar : 4);   // (3-D: rho w too)
        a["leaf_meta"] = NpyArray::from_doubles(meta, {static_cast<int64_t>(n), static_cast<int64_t>(cols)});
    } else {
        a["U"] = field_to_npy(U, mesh.is3d() ? kNVar : 4);
    }
    write_npz(p.dump_state, a);
}

std::array<double, 3> Simulation::volume_averages(const Field& Up, long long n_global) const {
    double sums[3] = {0.0, 0.0, 0.0};
    for (const State& c : Up) {
        sums[0] += c[0];
        sums[1] += 0.5 * (c[1] * c[1] + c[2] * c[2] + c[4] * c[4]) / c[0];
        sums[2] += c[3];
    }
#ifdef FV_USE_MPI
    int init = 0;
    MPI_Initialized(&init);
    int size = 1;
    if (init) MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size > 1) {
        double g[3];
        MPI_Allreduce(sums, g, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (int q = 0; q < 3; ++q) sums[q] = g[q];
    }
#endif
    return {sums[0] / n_global, sums[1] / n_global, sums[2] / n_global};
}

void Simulation::arm_stop_file() const {
    std::error_code ec;
    fs::remove(fs::path(exporter.output_dir()) / "STOP", ec);
}

void Simulation::poll_stop_file() const {
    std::error_code ec;
    if (!g_stop_requested.load() && fs::exists(fs::path(exporter.output_dir()) / "STOP", ec)) g_stop_requested = true;
}

std::FILE* Simulation::open_monitor() const {
    if (p.monitor_interval <= 0) return nullptr;
    const std::string path = (fs::path(exporter.output_dir()) / (file_prefix + "_monitor.csv")).string();
    std::FILE* f = std::fopen(path.c_str(), start_step > 0 ? "a" : "w");
    if (f && start_step == 0)
        std::fprintf(f, "step,t,mass,kinetic_energy,total_energy%s,res_rho,res_mom,res_E\n", p.use_ibm ? ",Fx,Fy,Cd,Cl,Cm" : "");
    return f;
}

void Simulation::monitor_row(std::FILE* f, int step, const std::array<double, 3>& v, const std::array<double, 3>* loads,
                             const std::array<double, 3>& res) const {
    if (!f) return;
    std::fprintf(f, "%d,%.10e,%.15e,%.15e,%.15e", step, t, v[0], v[1], v[2]);
    if (loads) {
        // Coefficients on q A (2-D: per unit depth, A = L; 3-D: L Lz, or pi r^2 for a sphere);
        // L = chord (airfoil) or diameter; Cm about (ibm_cx, ibm_cy), nose-up positive
        const bool airfoil = !naca_digits(p.ibm_shape).empty() || lower(p.ibm_shape) == "airfoil";
        const double L = airfoil ? p.ibm_chord : 2.0 * p.ibm_radius;
        const double A =
            !mesh.is3d()
                ? L
                : (lower(p.ibm_shape) == "sphere" ? 3.14159265358979323846 * p.ibm_radius * p.ibm_radius : L * mesh.Lz);
        const double qA = 0.5 * p.rho_in * p.u_in * p.u_in * A;
        const double Fx = (*loads)[0], Fy = (*loads)[1], Mz = (*loads)[2];
        std::fprintf(f, ",%.10e,%.10e,%.10e,%.10e,%.10e", Fx, Fy, qA > 0 ? Fx / qA : 0.0, qA > 0 ? Fy / qA : 0.0,
                     qA > 0 ? -Mz / (qA * L) : 0.0);
    }
    std::fprintf(f, ",%.10e,%.10e,%.10e\n", res[0], res[1], res[2]);
    std::fflush(f);
}

namespace {
// Sums of the squared residual per quantity and of the weights: {rho, momentum, energy, weight}; the norm is sqrt(sum / weight).
using ResSums = std::array<double, 4>;

inline void add_residual(ResSums& s, const State& r, double weight) {
    s[0] += r[0] * r[0] * weight;
    s[1] += (r[1] * r[1] + r[2] * r[2] + r[4] * r[4]) * weight;
    s[2] += r[3] * r[3] * weight;
    s[3] += weight;
}

inline std::array<double, 3> residual_norms(const ResSums& s) {
    if (!(s[3] > 0.0)) return {0.0, 0.0, 0.0};
    return {std::sqrt(s[0] / s[3]), std::sqrt(s[1] / s[3]), std::sqrt(s[2] / s[3])};
}
}   // namespace

std::array<double, 3> Simulation::ibm_loads(const Mesh& m, const ImmersedBoundary& ib, const Field& U,
                                            const Field& R) const {
    const std::vector<char>& solid = ib.mask();
    const double vol = m.is3d() ? m.cell_volume() : m.cell_area();
    double fx = 0.0, fy = 0.0, mz = 0.0;
    for (int c = 0; c < m.num_cells(); ++c) {
        if (!solid[c]) continue;
        double rx = R[c][1], ry = R[c][2];
        if (body_force) {
            rx -= U[c][0] * body_force->gx;
            ry -= U[c][0] * body_force->gy;
        }   // not a load
        const double ex = rx * vol, ey = ry * vol;
        fx += ex;
        fy += ey;
        mz += (m.xc(c % m.nx) - p.ibm_cx) * ey - (m.yc((c / m.nx) % m.ny) - p.ibm_cy) * ex;
    }
    return {fx, fy, mz};
}

// ---------------------------------------------------------------------------
// Serial time loop
// ---------------------------------------------------------------------------

void Simulation::solve() {
    std::printf("=== TIME-MARCHING  (t_end = %g s) ===\n", p.t_end);
    std::fflush(stdout);
    if (cells) {
        solve_amr();
        return;
    }

    const double min_delta = dt_length;
    int step = start_step;
    int last_snapshot_step = -1;
    double next_write_time = next_time_output(t, p.write_interval_time);
    PostprocessWriter post(*test_case, exporter.output_dir(), file_prefix, step);
    if (step == 0) {
        exporter.export_vtr(file_prefix, fluid, U, mesh, t, step);
        last_snapshot_step = step;
    }

    const ResidualFn res = [this](const Field& Uin, Field& R) { return residual(Uin, R); };
    std::FILE* mon = open_monitor();
    arm_stop_file();
    const long long n_all = mesh.num_cells();
    Field R_loads;
    auto row = [&](int st) {   // averages, plus the IBM loads from a residual of the current state
        scheme->stage_time = t;
        std::array<double, 3> loads{};
        residual(U, R_loads);
        if (p.use_ibm) loads = ibm_loads(mesh, *ibm, U, R_loads);
        ResSums rs{};
        for (int c = 0; c < mesh.num_cells(); ++c)
            if (!p.use_ibm || !ibm->mask()[c]) add_residual(rs, R_loads[c], 1.0);
        monitor_row(mon, st, volume_averages(U, n_all), p.use_ibm ? &loads : nullptr, residual_norms(rs));
    };
    if (mon && step == 0) row(step);
    if (!post.empty()) {
        if (step == 0) {
            post.write_uniform(mesh, fluid, U, step, t, true);
            if (post.has_averages()) post.observe_averages(step, t, volume_primitive_means(fluid, mesh, U), true);
        } else {
            post.skip_step(step);
            if (post.has_averages()) post.observe_averages(step, t, volume_primitive_means(fluid, mesh, U), false, false);
        }
        if (step == 0 && post.has_force_due(step, true)) {
            residual(U, R_loads);
            post.write_forces(step, t, ibm_loads(mesh, *ibm, U, R_loads), mesh, true);
        }
    }
    profile_reset();   // the profile counts the time loop only
    const auto t0 = Clock::now();
    bool interrupted = false;
    while (t < p.t_end) {
        ProfileScope step_profile(Section::Step);
        const StepResult r = scheme->step(U, res, min_delta, t, std::min(p.t_end, next_write_time), p.dt_max);
        check_progress(t, r.dt, r.s_max, step + 1, p.time_scheme.c_str());
        if (p.use_ibm) ibm->apply_solid_reset(U, fluid);
        t += r.dt;
        scheme->stage_time = t;
        ++step;
        const bool time_output = reached_time_output(t, next_write_time, p.write_interval_time);
        if (time_output) next_write_time += p.write_interval_time;
        if (mon && step % p.monitor_interval == 0) row(step);
        if (!post.empty()) {
            post.write_uniform(mesh, fluid, U, step, t);
            if (post.has_averages()) post.observe_averages(step, t, volume_primitive_means(fluid, mesh, U));
            if (post.has_force_due(step)) {
                residual(U, R_loads);
                post.write_forces(step, t, ibm_loads(mesh, *ibm, U, R_loads), mesh);
            }
        }

        if (step % p.save_interval == 0 || time_output) {
            {
                ProfileScope ps(Section::Snapshot);
                exporter.export_vtr(file_prefix, fluid, U, mesh, t, step);
            }
            {
                ProfileScope ps(Section::Checkpoint);
                save_checkpoint(step);
            }
            last_snapshot_step = step;
            std::printf("Step %05d | t = %.5f s | dt = %.3e s | S_max = %.2f m/s | wall = %s | Checkpoint saved\n",
                        step, t, r.dt, r.s_max, format_elapsed(seconds_since(t0)).c_str());
            std::fflush(stdout);
        }
        poll_stop_file();
        if (g_stop_requested) {
            interrupted = true;
            break;
        }
        if (p.max_steps > 0 && step - start_step >= p.max_steps) break;
    }
    const double loop_wall = seconds_since(t0);

    if (mon) {
        if (step % p.monitor_interval != 0) row(step);
        std::fclose(mon);
    }
    if (!post.empty()) {
        post.write_uniform(mesh, fluid, U, step, t, true);
        if (post.has_averages()) post.observe_averages(step, t, volume_primitive_means(fluid, mesh, U), true);
        if (post.has_force_due(step, true)) {
            residual(U, R_loads);
            post.write_forces(step, t, ibm_loads(mesh, *ibm, U, R_loads), mesh, true);
        }
    }
    if (interrupted) {
        std::printf("\n[INTERRUPTED] Stopping the simulation...\n");
        exporter.export_vtr(file_prefix, fluid, U, mesh, t, step);
    } else if (last_snapshot_step != step) {
        exporter.export_vtr(file_prefix, fluid, U, mesh, t, step);
    }
    save_checkpoint(step);
    exporter.write_pvd(file_prefix);
    dump_final_state(step);
    std::printf("=== DONE  - open '%s.pvd' in ParaView ===  (time loop %.6f s, %d steps)\n", file_prefix.c_str(),
                loop_wall, step - start_step);
    if (profile_enabled()) print_profile(stdout, profile_values(), 1);
}

void Simulation::solve_amr() {
    CellMesh& cm = *cells;
    CellFluxEngine& eng = *cell_engine;
    const int nranks = cell_comm.size, rank = cell_comm.rank;
    const bool root = rank == 0;
    CellAdvance adv(cm, eng, *scheme);
    if (cell_par) adv.par = cell_par.get();
    adv.stage_hook = [this](int, double stage_time) { update_boundary_values(stage_time, boundaries); };
    // each rank keeps only its leaves and their halo (CellParallel::localize), unless FENICE_REPLICATED_REMESH=1
    const bool local_run = cell_par && cell_par->localized();
    // the leaves of the whole mesh and their state (collective; a localized run gives them to rank 0 only, the one that writes)
    std::vector<int> g_level, g_ci, g_cj, g_ck;
    Field g_U;
    const bool octree = cm.dim == 3;
    const auto leaves = [&]() -> LeafArrays {
        if (local_run) {
            cell_par->collect(g_level, g_ci, g_cj, g_ck, g_U, true);   // rank 0 only: it alone writes
            return {&g_level, &g_ci, &g_cj, &g_U, octree ? &g_ck : nullptr};
        }
        if (cell_par) cell_par->gather_state();
        return {&cm.level, &cm.ci, &cm.cj, &cm.U, octree ? &cm.ck : nullptr};
    };
    // (a 3-D adaptive run writes no force rows: the load of a body is a 2-D quantity here)
    if (octree && root && !test_case->forces.empty())
        std::printf("[AMR] 3-D adaptive mesh: the [[force]] rows are not written (probes, sample lines and averages are)\n");
    std::unique_ptr<PostprocessWriter> post;
    if (root) post = std::make_unique<PostprocessWriter>(*test_case, exporter.output_dir(), file_prefix, start_step);
    const auto post_state = [&](int st, bool force, bool reset_average, bool emit) {
        if ((!force && !amr_postprocess_state_due(*test_case, st)) ||
            (force && test_case->probes.empty() && test_case->samples.empty() && test_case->averages.empty()))
            return;
        if (!emit && test_case->averages.empty()) {
            if (root) post->skip_step(st);
            return;
        }
        const LeafArrays lf = leaves();
        if (!root) return;
        if (emit) post->write_amr(cm, fluid, *lf.level, *lf.ci, *lf.cj, *lf.U, st, t, force, lf.ck);
        else post->skip_step(st);
        if (post->has_averages()) {
            const std::array<double, 5> means = volume_primitive_means(fluid, cm, *lf.level, *lf.U);
            if (reset_average) post->reset_averages(t, means);
            post->observe_averages(st, t, means, force, emit);
        }
    };
    // number of leaves of the whole mesh (collective)
    const auto total_leaves = [&]() {
        double n = 0.0;
        for (int l = 0; l <= cm.max_level; ++l) n += eng.owned_end(l) - eng.owned_begin(l);
        return static_cast<int>(cell_par ? cell_par->reduce_sum(n) : n);
    };
    // largest wave speed of the leaves of the whole mesh (collective), from the forced state of the leaves a rank holds
    const auto wave_speed = [&]() {
        eng.force_state(cm, cm.U);
        const double s = eng.wave_speed(cm, cm.U);
        return cell_par ? cell_par->reduce_max(s) : s;
    };

    Field R;
    // wave-speed bound the time step is taken from, one step old (a restart reads it from the checkpoint)
    double S = start_step > 0 ? cell_speed_ : (local_run ? wave_speed() : eng.residual(cm, cm.U, R));
    cell_speed_ = S;
    int step = start_step;
    int last_snapshot_step = -1;
    double next_write_time = next_time_output(t, p.write_interval_time);

    // Every rank writes the leaves it owns (the whole mesh on one rank)
    const auto export_state = [&](int st) {
        std::vector<int> owned;
        for (int l = 0; l <= cm.max_level; ++l)
            for (int c = eng.owned_begin(l); c < eng.owned_end(l); ++c) owned.push_back(c);
        exporter.export_cells(file_prefix, fluid, cm, owned, t, st, rank, nranks, [&] {
            if (cell_par) cell_par->reduce_max(0.0);   // a barrier
        });
    };

    // Monitor: area-weighted means over the leaves and the IBM loads. Every rank sums its owned leaves in leaf order and the
    // partial sums are added over the ranks (collective): one rank gives the serial bits, several agree with it to roundoff.
    std::FILE* mon = root ? open_monitor() : nullptr;
    arm_stop_file();
    const bool monitoring = p.monitor_interval > 0;
    const auto row = [&](int st, bool write_monitor, bool write_force, bool force_final) {
        scheme->stage_time = t;
        eng.stage_time = t;
        update_boundary_values(t, boundaries);
        double sums[4] = {0.0, 0.0, 0.0, 0.0};   // mass, kinetic energy, total energy, area
        for (int l = 0; l <= cm.max_level; ++l)
            for (int c = eng.owned_begin(l); c < eng.owned_end(l); ++c) {
                const State& u = cm.U[static_cast<size_t>(c)];
                const double a = cm.volume(c);   // (the area in 2-D)
                sums[0] += u[0] * a;
                sums[1] += 0.5 * (u[1] * u[1] + u[2] * u[2] + u[4] * u[4]) / u[0] * a;
                sums[2] += u[3] * a;
                sums[3] += a;
            }
        std::array<double, 3> loads{};
        ResSums rs{};
        {
            Field Rl;
            if (!cell_par) {
                eng.residual(cm, cm.U, Rl);
            } else {
                // the partitioned engine: the fluxes of each level, the coarse-fine ones handed to the coarse owners through the
                // registers (set to the flux itself, as a weight 1 gives), then R of the owned leaves
                Field Rown(static_cast<size_t>(cm.num_leaves()));
                eng.force_state(cm, cm.U);   // as the first pass of a step does
                for (int l = 0; l <= cm.max_level; ++l) {
                    if (!cell_par->level_present(l)) continue;
                    eng.compute_level_fluxes(cm, cm.U, l);
                    if (l == 0) continue;
                    eng.clear_registers(cm, l);
                    eng.accumulate_registers(cm, l, 1.0);
                    cell_par->exchange_registers(l);
                }
                for (int l = 0; l <= cm.max_level; ++l) {
                    if (!cell_par->level_present(l)) continue;
                    eng.gather_level(cm, l, true, Rl);
                    std::copy(Rl.begin(), Rl.end(), Rown.begin() + eng.owned_begin(l));
                }
                Rl.swap(Rown);
            }
            if (p.use_ibm) {
                loads = cell_loads(cm, Rl);
                if (cell_par)
                    for (double& v : loads) v = cell_par->reduce_sum(v);
            }
            if (write_force && !octree) {
                Field contributions(static_cast<size_t>(cm.num_leaves()));
                const std::vector<char>& solid = eng.solid();
                for (int l = 0; l <= cm.max_level; ++l)
                    for (int c = eng.owned_begin(l); c < eng.owned_end(l); ++c) {
                        if (!solid[static_cast<size_t>(c)]) continue;
                        double rx = Rl[static_cast<size_t>(c)][1], ry = Rl[static_cast<size_t>(c)][2];
                        if (body_force) {
                            rx -= cm.U[static_cast<size_t>(c)][0] * body_force->gx;
                            ry -= cm.U[static_cast<size_t>(c)][0] * body_force->gy;
                        }
                        const double ex = rx * cm.area(c), ey = ry * cm.area(c);
                        contributions[static_cast<size_t>(c)] =
                            {ex, ey, (cm.xc(c) - p.ibm_cx) * ey - (cm.yc(c) - p.ibm_cy) * ex, 0.0, 0.0};
                    }
                Field ordered;
                if (cell_par) cell_par->collect_field(contributions, ordered, true);
                else ordered = std::move(contributions);
                if (root) {
                    std::array<double, 3> deterministic{};
                    for (const State& c : ordered) {
                        deterministic[0] += c[0];
                        deterministic[1] += c[1];
                        deterministic[2] += c[2];
                    }
                    post->write_forces(st, t, deterministic, mesh, force_final);
                }
            }
            const std::vector<char>& solid = eng.solid();
            for (int l = 0; l <= cm.max_level; ++l)
                for (int c = eng.owned_begin(l); c < eng.owned_end(l); ++c)
                    if (!p.use_ibm || !solid[static_cast<size_t>(c)]) add_residual(rs, Rl[static_cast<size_t>(c)], cm.volume(c));
        }
        if (cell_par) {
            for (double& v : sums) v = cell_par->reduce_sum(v);
            for (double& v : rs) v = cell_par->reduce_sum(v);
        }
        if (write_monitor)
            monitor_row(mon, st, {sums[0] / sums[3] * mesh.cell_volume(), sums[1] / sums[3] * mesh.cell_volume(), sums[2] / sums[3] * mesh.cell_volume()},
                        p.use_ibm ? &loads : nullptr, residual_norms(rs));
    };

    if (step == 0) {
        export_state(step);
        last_snapshot_step = step;
    }
    post_state(step, step == 0, false, step == 0);
    if (monitoring && step == 0) row(step, true, postprocess_force_due(*test_case, step), false);
    else if (step == 0 && postprocess_force_due(*test_case, step)) row(step, false, true, false);

    const auto t0 = Clock::now();
    profile_reset();   // the profile counts the time loop only
    double remesh_wall = 0.0, t_gather = 0.0, t_speed = 0.0, t_partition = 0.0;
    bool interrupted = false;
    while (t < p.t_end) {
        // the time-step length of the base mesh: min(dx, dy) in 2-D, 2 / sum 1/h in 3-D (Simulation::dt_length)
        const double h0 = octree ? 2.0 / (1.0 / cm.dx(0) + 1.0 / cm.dy(0) + 1.0 / cm.dz(0)) : std::min(cm.dx(0), cm.dy(0));
        double dt = std::min(scheme->cfl * h0 / std::max(S, 1e-10), p.t_end - t);
        if (p.dt_max > 0.0) dt = std::min(dt, p.dt_max);
        dt = std::min(dt, next_write_time - t);
        {
            ProfileScope ps(Section::Step);
            S = adv.step(dt, t);
        }
        check_progress(t, dt, S, step + 1, p.time_scheme.c_str());
        cell_speed_ = S;
        if (p.use_ibm) {
            eng.solid_reset(cm, cm.U);
            if (cell_par) cell_par->exchange_solid();
        }
        t += dt;
        ++step;
        const bool time_output = reached_time_output(t, next_write_time, p.write_interval_time);
        if (time_output) next_write_time += p.write_interval_time;

        bool changed = false;
        if (step % p.amr_refine_every == 0) {
            const auto r0 = Clock::now();
            if (local_run) {
                // indicator on the owned leaves and their region, flags gathered, state of the new leaves migrated: no whole mesh
                changed = cell_par->remesh(*cell_adaptor);
            } else {
                // the indicator looks at every leaf: the whole mesh and its state on every rank (the replicated remesh)
                if (cell_par) cell_par->gather_state();
                t_gather += seconds_since(r0);
                changed = cell_adaptor->remesh();
                if (changed && cell_par) {
                    const auto r2 = Clock::now();
                    cell_par->partition();
                    t_partition += seconds_since(r2);
                }
            }
            if (changed) {
                const auto r1 = Clock::now();
                S = wave_speed();   // the wave-speed bound follows the new leaves (each rank's leaves, then the maximum)
                cell_speed_ = S;
                t_speed += seconds_since(r1);
            }
            remesh_wall += seconds_since(r0);
            profile_add(Section::Remesh, seconds_since(r0));
        }
        post_state(step, false, changed, true);
        const bool monitor_due = monitoring && step % p.monitor_interval == 0;
        const bool force_due = postprocess_force_due(*test_case, step);
        if (monitor_due || force_due) row(step, monitor_due, force_due, false);

        if (step % p.save_interval == 0 || time_output) {
            ProfileScope ps(Section::Output);
            export_state(step);
            const LeafArrays lf = leaves();
            const int nleaves = total_leaves();
            if (root) save_checkpoint(step, &lf);
            last_snapshot_step = step;
            if (root)
                std::printf("Step %05d | t = %.5f s | dt = %.3e s | S_max = %.2f m/s | leaves = %d | wall = %s%s\n", step, t, dt, S, nleaves,
                            format_elapsed(seconds_since(t0)).c_str(), " | Checkpoint saved");
            std::fflush(stdout);
        }
        poll_stop_file();
        const double stop = g_stop_requested.load() ? 1.0 : 0.0;
        if ((cell_par ? cell_par->reduce_max(stop) : stop) > 0.0) {
            interrupted = true;
            break;
        }
        if (p.max_steps > 0 && step - start_step >= p.max_steps) break;
    }
    const double loop_wall = seconds_since(t0);

    if (monitoring || !test_case->forces.empty()) {
        const bool monitor_final = monitoring && step % p.monitor_interval != 0;
        const bool force_final = !test_case->forces.empty() && !postprocess_force_due(*test_case, step);
        if (monitor_final || force_final) row(step, monitor_final, force_final, force_final);
    }
    post_state(step, true, false, true);
    if (monitoring) {
        if (mon) std::fclose(mon);
    }
    if (root && interrupted) std::printf("\n[INTERRUPTED] Stopping the simulation...\n");
    if (interrupted || last_snapshot_step != step) export_state(step);
    // resident memory of every rank (collective), before the whole mesh is gathered for the files
    const MemoryUsage mem_end = memory_usage();
    std::vector<double> mem = {mem_setup_.current, mem_init_.current, mem_end.current, mem_end.peak};
    if (cell_par) {
        std::vector<double> all(static_cast<size_t>(4 * nranks));
        std::vector<int> counts(static_cast<size_t>(nranks), 4), displs(static_cast<size_t>(nranks));
        for (int r = 0; r < nranks; ++r) displs[static_cast<size_t>(r)] = 4 * r;
        cell_comm.allgatherv(mem.data(), 4, all.data(), counts.data(), displs.data());
        mem.swap(all);
    }
    // the profile of every rank (collective; empty, and the same on every rank, when FENICE_PROFILE is off)
    std::vector<double> profile = profile_values();
    if (!profile.empty() && cell_par) {
        const int n = static_cast<int>(profile.size());
        std::vector<double> all(profile.size() * static_cast<size_t>(nranks));
        std::vector<int> counts(static_cast<size_t>(nranks), n), displs(static_cast<size_t>(nranks));
        for (int r = 0; r < nranks; ++r) displs[static_cast<size_t>(r)] = n * r;
        cell_comm.allgatherv(profile.data(), n, all.data(), counts.data(), displs.data());
        profile.swap(all);
    }
    const LeafArrays lf = leaves();
    if (!root) return;
    save_checkpoint(step, &lf);
    exporter.write_pvd(file_prefix);
    dump_final_state(step, &lf);
    if (!local_run) print_halo_reach("final mesh");
    std::vector<int> per_level(static_cast<size_t>(cm.max_level) + 1, 0);
    for (const int l : *lf.level) ++per_level[static_cast<size_t>(l)];
    std::printf("[AMR] %d leaves at the end (", static_cast<int>(lf.level->size()));
    for (int l = 0; l <= cm.max_level; ++l) std::printf("%slevel %d: %d", l ? ", " : "", l, per_level[static_cast<size_t>(l)]);
    std::printf("); remesh %.2f s of the %.2f s time loop\n", remesh_wall, loop_wall);
    // (a localized run has no gather: the "gather" is the region of the mesh the indicator reads and its state, the "indicator" the
    // flags of the owned leaves and their exchange, the "adapt" the adapt of the topology and the state of the new leaves, the
    // "partition" the partition, localize and halo fill after the engine build)
    if (local_run) {
        std::printf("[AMR] remesh parts: gather %.2f s, indicator %.2f s, adapt %.2f s, engine build %.2f s, wave-speed bound %.2f s, partition %.2f s\n",
                    cell_par->t_region, cell_par->t_flags, cell_par->t_migrate, cell_par->t_rebuild, t_speed, cell_par->t_repartition);
        std::printf("[AMR] of the adapt, the topology: %.2f s\n", cell_par->t_topo);
    } else
        std::printf("[AMR] remesh parts: gather %.2f s, indicator %.2f s, adapt %.2f s, engine build %.2f s, wave-speed bound %.2f s, partition %.2f s\n",
                    t_gather, cell_adaptor->t_indicator, cell_adaptor->t_adapt, cell_adaptor->t_build, t_speed, t_partition);
    double lo[4], hi[4];
    for (int q = 0; q < 4; ++q) {
        lo[q] = hi[q] = mem[static_cast<size_t>(q)];
        for (int r = 1; r < nranks; ++r) {
            lo[q] = std::min(lo[q], mem[static_cast<size_t>(4 * r + q)]);
            hi[q] = std::max(hi[q], mem[static_cast<size_t>(4 * r + q)]);
        }
    }
    const double MB = 1024.0 * 1024.0;
    std::printf("[AMR] resident memory per rank [MB], min / max over %d rank%s: after setup %.1f / %.1f, after the initial "
                "adaptation %.1f / %.1f, at the end %.1f / %.1f, peak %.1f / %.1f\n",
                nranks, nranks > 1 ? "s" : "", lo[0] / MB, hi[0] / MB, lo[1] / MB, hi[1] / MB, lo[2] / MB, hi[2] / MB,
                lo[3] / MB, hi[3] / MB);
    double eng_leaf = 0.0, eng_uniform = 0.0;
    eng.memory_bytes(eng_leaf, eng_uniform);
    std::printf("[AMR] tables of rank 0 [MB]: mesh %.1f (leaves, state, faces, lookups), engine %.1f (stencils, work arrays), "
                "engine uniform-mesh %.1f (IBM masks, ghost table), body band %.1f, indicator %.1f\n",
                cm.memory_bytes() / MB, eng_leaf / MB, eng_uniform / MB, cell_adaptor->band_bytes() / MB,
                cell_adaptor->xi_bytes() / MB);
    if (!profile.empty()) print_profile(stdout, profile, nranks);
    std::printf("=== DONE  - open '%s.pvd' in ParaView ===  (time loop %.3f s, %d steps)\n", file_prefix.c_str(), loop_wall,
                step - start_step);
}

// ---------------------------------------------------------------------------
// MPI time loop
// ---------------------------------------------------------------------------

#ifdef FV_USE_MPI
void Simulation::run_mpi(MPI_Comm comm) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    const bool root = (rank == 0);

    setup(p, root);
    MPIDecomposition decomp(p.nx, p.ny, comm, p.nz);

    // Layout table
    const char dir = decomp.dir;
    {
        const int start = dir == 'y' ? decomp.j_start : decomp.i_start;
        const int local = dir == 'y' ? decomp.ny_local : decomp.nx_local;
        int mine[4] = {rank, start, start + local - 1, local};
        std::vector<int> all(root ? 4 * decomp.n_ranks : 0);
        MPI_Gather(mine, 4, MPI_INT, root ? all.data() : nullptr, 4, MPI_INT, 0, comm);
        if (root) {
            std::printf("\n[MPI] %c-decomposition  -  %d ranks, global ny=%d, nx=%d\n", dir == 'y' ? 'Y' : 'X',
                        decomp.n_ranks, p.ny, p.nx);
            std::printf("  %5s  %8s  %6s  %9s  %s\n", "Rank", dir == 'y' ? "j_start" : "i_start",
                        dir == 'y' ? "j_end" : "i_end", dir == 'y' ? "ny_local" : "nx_local", "role");
            std::printf("  -----  --------  ------  ---------  ----------------\n");
            for (int r = 0; r < decomp.n_ranks; ++r) {
                std::string role;
                if (r == 0) role = dir == 'y' ? "bottom" : "left";
                if (r == decomp.n_ranks - 1) {
                    if (!role.empty()) role += ", ";
                    role += dir == 'y' ? "top" : "right";
                }
                if (role.empty()) role = "interior";
                std::printf("  %5d  %8d  %6d  %9d  %s\n", all[4 * r], all[4 * r + 1], all[4 * r + 2], all[4 * r + 3],
                            role.c_str());
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    }

    // Local mesh with its global origin (also positions the IBM mask correctly
    // for X-decomposition).
    Mesh lm;
    if (dir == 'y') {
        const double dy = p.Ly / p.ny;
        lm = generate_cartesian_mesh_3d(p.Lx, dy * decomp.ny_local, p.Lz, p.nx, decomp.ny_local, p.nz, 0.0,
                                        decomp.j_start * dy);
    } else {
        const double dx = p.Lx / p.nx;
        lm = generate_cartesian_mesh_3d(dx * decomp.nx_local, p.Ly, p.Lz, decomp.nx_local, p.ny, p.nz,
                                        decomp.i_start * dx, 0.0);
    }
    lm.dx = mesh.dx;   // exact global spacing (Ly_local/ny_local can differ from Ly/ny by an ulp)
    lm.dy = mesh.dy;
    lm.dz = mesh.dz;
    std::unique_ptr<MusclReconstruction> local_muscl;
    if (p.reconstructs())
        local_muscl = std::make_unique<MusclReconstruction>(fluid, parse_limiter(p.muscl_limiter), parse_reconstruction(p.reconstruction),
                                                            p.weno_epsilon);

    std::unique_ptr<ImmersedBoundary> local_ibm;
    std::vector<char> ghost_mask_lo, ghost_mask_hi;
    if (p.use_ibm) {
        const std::vector<char>& gm = ibm->mask();   // global mask: identical to the serial run
        auto gmask = [&](int k, int j, int i) {
            return gm[(static_cast<size_t>(k) * p.ny + (j + p.ny) % p.ny) * p.nx + (i + p.nx) % p.nx];
        };
        std::vector<char> lmask;
        for (int k = 0; k < lm.nz; ++k)
            for (int j = 0; j < lm.ny; ++j)
                for (int i = 0; i < lm.nx; ++i) lmask.push_back(gmask(k, decomp.j_start + j, decomp.i_start + i));
        // Ghost masks across each interface, slab index k*nx + i (y) or k*ny + j (x)
        for (int k = 0; k < lm.nz; ++k) {
            if (dir == 'y')
                for (int i = 0; i < p.nx; ++i) {
                    ghost_mask_lo.push_back(gmask(k, decomp.j_start - 1, i));
                    ghost_mask_hi.push_back(gmask(k, decomp.j_start + decomp.ny_local, i));
                }
            else
                for (int j = 0; j < p.ny; ++j) {
                    ghost_mask_lo.push_back(gmask(k, j, decomp.i_start - 1));
                    ghost_mask_hi.push_back(gmask(k, j, decomp.i_start + decomp.nx_local));
                }
        }
        local_ibm = std::make_unique<ImmersedBoundary>(lm.nx, lm.ny, std::move(lmask), p.rho_in, p.p_out, lm.nz);
        int mine = local_ibm->solid_count(), total = 0;
        MPI_Reduce(&mine, &total, 1, MPI_INT, MPI_SUM, 0, comm);
        if (root)
            std::printf("[IBM] %s at (%g, %g) r=%g -> %d solid cells (across %d ranks)\n", p.ibm_shape.c_str(),
                        p.ibm_cx, p.ibm_cy, p.ibm_radius, total, decomp.n_ranks);
    }

    // MPI-interface sides start transmissive (overridden by halos anyway)
    bool periodic = false;
    BCType l = parse_bc(p.bc_left), r = parse_bc(p.bc_right), b = parse_bc(p.bc_bottom), tp = parse_bc(p.bc_top);
    if (dir == 'y') {
        periodic = (b == BCType::Periodic || tp == BCType::Periodic);
        if (!(decomp.is_lo_rank && !periodic)) b = BCType::Transmissive;
        if (!(decomp.is_hi_rank && !periodic)) tp = BCType::Transmissive;
    } else {
        periodic = (l == BCType::Periodic || r == BCType::Periodic);
        if (!(decomp.is_lo_rank && !periodic)) l = BCType::Transmissive;
        if (!(decomp.is_hi_rank && !periodic)) r = BCType::Transmissive;
    }
    BoundaryManager lb(lm, fluid);
    lb.set_boundary_types(l, r, b, tp);
    lb.set_type(Face::Front, parse_bc(p.bc_front));   // z sides are never split
    lb.set_type(Face::Back, parse_bc(p.bc_back));
    lb.set_params(bc_params(p));
    {
        auto reducer = [&decomp](double& sum, double& count) {
            double in[2] = {sum, count}, out[2];
            MPI_Allreduce(in, out, 2, MPI_DOUBLE, MPI_SUM, decomp.comm);
            sum = out[0];
            count = out[1];
        };
        // The sides parallel to the strips are split across all ranks
        for (Face f :
             dir == 'y' ? std::array<Face, 2>{Face::Left, Face::Right} : std::array<Face, 2>{Face::Bottom, Face::Top})
            lb.set_mean_reducer(f, reducer);
        if (lm.is3d()) {   // front / back are split across every rank as well
            lb.set_mean_reducer(Face::Front, reducer);
            lb.set_mean_reducer(Face::Back, reducer);
        }
    }
    if (root) {
        // Domain BCs (rank 0's own l/r/b/tp have interface sides set to transmissive)
        std::printf("[MPI] Domain BCs  left=%s, right=%s, bottom=%s, top=%s\n", bc_name(parse_bc(p.bc_left)).c_str(),
                    bc_name(parse_bc(p.bc_right)).c_str(), bc_name(parse_bc(p.bc_bottom)).c_str(),
                    bc_name(parse_bc(p.bc_top)).c_str());
        if (dir == 'y') {
            std::printf("[MPI] Local meshes ready  -  each rank: nx=%d, ny_local varies\n", p.nx);
            if (periodic) std::printf("[MPI] Periodic Y detected - wrap-around halo exchange enabled\n");
        } else {
            std::printf("[MPI] Local meshes ready  -  each rank: ny=%d, nx_local varies\n", p.ny);
            if (periodic) std::printf("[MPI] Periodic X detected - wrap-around halo exchange enabled\n");
        }
    }

    // Initial field on rank 0
    Field U_global;
    int start = 0;
    if (root) {
        if (p.resume && fs::exists(checkpoint_path())) {
            start = load_checkpoint();
            U_global = U;
        } else {
            if (p.resume)
                std::printf("\n[WARNING] No checkpoint at %s. Starting from t=0.\n", checkpoint_path().c_str());
            U_global = p.use_hot_start ? load_hot_start(p.hot_start_file) : initial_field();
        }
    }
    MPI_Bcast(&t, 1, MPI_DOUBLE, 0, comm);
    MPI_Bcast(&start, 1, MPI_INT, 0, comm);

    Field U_local = decomp.scatter(U_global);
    // Ghost-cell wall: the ghost table of the whole mesh (same on every rank); each rank fills its own ghost cells and fetches
    // the few cells of other strips their image points read (they may lie several strips away), so MPI gives the serial bits
    const bool ghost = p.use_ibm && ibm && ibm->ghost_cells();
    const auto local_of = [&decomp](int c) { return decomp.local_index(c); };
    MPIDecomposition::CellFetch ghost_fetch;
    std::vector<State> ghost_remote;
    std::unordered_map<int, int> ghost_pos;
    if (ghost) {
        ghost_fetch = decomp.plan_fetch(ibm->ghost_inputs(local_of));
        for (size_t k = 0; k < ghost_fetch.want.size(); ++k) ghost_pos[ghost_fetch.want[k]] = static_cast<int>(k);
    }
    const auto cells_of = [&decomp](const Field& Ul, const std::vector<State>& remote, const std::unordered_map<int, int>& pos) {
        return [&decomp, &Ul, &remote, &pos](int c) -> const State& {
            const int l = decomp.local_index(c);
            return l >= 0 ? Ul[static_cast<size_t>(l)] : remote[static_cast<size_t>(pos.at(c))];
        };
    };
    // ghost cells of this strip from Uin, written to Uout (also Uout = Uin: only solid cells are written)
    const auto fill_ghosts = [&](const Field& Uin, Field& Uout, bool reset) {
        decomp.fetch(ghost_fetch, Uin, ghost_remote);
        ibm->fill_ghosts_part(local_of, cells_of(Uin, ghost_remote, ghost_pos), Uout, fluid, reset);
    };
    if (p.use_ibm && start == 0) {
        if (ghost)
            fill_ghosts(Field(U_local), U_local, false);
        else
            local_ibm->apply_forcing(U_local, fluid);
    }
    if (root)
        std::printf("[MPI] Initial field scattered to all ranks  (total cells: %d, per-rank avg: %d)\n\n",
                    mesh.num_cells(), mesh.num_cells() / decomp.n_ranks);

    std::unique_ptr<PostprocessWriter> post;
    if (root) post = std::make_unique<PostprocessWriter>(*test_case, exporter.output_dir(), file_prefix, start);
    Field R_post;
    const auto post_state = [&](int st, bool force, bool emit) {
        if (!has_postprocess(*test_case) || (!force && !postprocess_state_due(*test_case, st))) return;
        if (!emit && test_case->averages.empty()) {
            if (root) post->skip_step(st);
            return;
        }
        Field global = decomp.gather(U_local);
        if (!root) return;
        if (emit) post->write_uniform(mesh, fluid, global, st, t, force);
        else post->skip_step(st);
        if (post->has_averages()) post->observe_averages(st, t, volume_primitive_means(fluid, mesh, global), force, emit);
        if (emit && post->has_force_due(st, force)) {
            residual(global, R_post);
            post->write_forces(st, t, ibm_loads(mesh, *ibm, global, R_post), mesh, force);
        }
    };

    Field halo_lo, halo_hi, forced;
    const ResidualFn res = [&](const Field& Uin, Field& R) {
        const Field* Uc = &Uin;
        if (local_ibm) {
            forced = Uin;
            if (ghost)
                fill_ghosts(Uin, forced, false);
            else
                local_ibm->apply_forcing(forced, fluid);
            Uc = &forced;
        }
        bool has_lo, has_hi;
        decomp.exchange_halos(*Uc, periodic, halo_lo, halo_hi, has_lo, has_hi);
        StripHalos h{dir, has_lo ? &halo_lo : nullptr, has_hi ? &halo_hi : nullptr};
        update_boundary_values(scheme->stage_time, lb);
        const double s = assemble(lm, lb, nullptr, local_muscl.get(), *Uc, R, &h);
        return decomp.allreduce_max(s);
    };
    // Solid reset with the neighbours across each interface (fringe detection)
    const auto solid_reset = [&](Field& Uloc) {
        if (ghost) {
            fill_ghosts(Field(Uloc), Uloc, true);
            return;
        }
        bool has_lo, has_hi;
        decomp.exchange_halos(Uloc, periodic, halo_lo, halo_hi, has_lo, has_hi);
        const Field slo = has_lo ? adjacent_slab(halo_lo, dir, true, lm) : Field{};
        const Field shi = has_hi ? adjacent_slab(halo_hi, dir, false, lm) : Field{};
        SolidResetHalo sh{dir, has_lo ? &slo : nullptr, has_hi ? &shi : nullptr, &ghost_mask_lo, &ghost_mask_hi};
        local_ibm->apply_solid_reset(Uloc, fluid, &sh);
    };
    const double min_delta = dt_length;
    double next_write_time = next_time_output(t, p.write_interval_time);
    int last_snapshot_step = -1;

    if (start == 0) {
        Field g = decomp.gather(U_local);
        if (root) exporter.export_vtr(file_prefix, fluid, g, mesh, t, 0);
        last_snapshot_step = 0;
    }
    post_state(start, start == 0, start == 0);
    if (root) {
        std::printf("=== MPI TIME-MARCHING  (%d ranks | t_end = %g s) ===\n", decomp.n_ranks, p.t_end);
        std::printf("[MPI] Output: full domain gathered to rank 0 before every VTK write\n");
        std::fflush(stdout);
    }

    int step = start;
    start_step = start;
    std::FILE* mon = root ? open_monitor() : nullptr;
    arm_stop_file();
    const bool monitoring = p.monitor_interval > 0;
    const long long n_all = static_cast<long long>(p.nx) * p.ny * p.nz;
    Field R_loads;
    auto row = [&](int st) {   // collective: every rank calls it
        scheme->stage_time = t;
        update_boundary_values(t, lb);
        const auto v = volume_averages(U_local, n_all);
        std::array<double, 3> loads{};
        res(U_local, R_loads);   // residual of the current state: the IBM loads (mask or ghost cells) and the norms, summed over the strips
        if (p.use_ibm) {
            const std::array<double, 3> mine = ibm_loads(lm, *local_ibm, U_local, R_loads);
            MPI_Allreduce(mine.data(), loads.data(), 3, MPI_DOUBLE, MPI_SUM, comm);
        }
        ResSums mine_rs{}, rs{};
        for (int c = 0; c < lm.num_cells(); ++c)
            if (!p.use_ibm || !local_ibm->mask()[c]) add_residual(mine_rs, R_loads[c], 1.0);
        MPI_Allreduce(mine_rs.data(), rs.data(), 4, MPI_DOUBLE, MPI_SUM, comm);
        monitor_row(mon, st, v, p.use_ibm ? &loads : nullptr, residual_norms(rs));
    };
    if (monitoring && step == 0) row(step);
    const auto t0 = Clock::now();
    bool interrupted = false;
    while (t < p.t_end) {
        ProfileScope step_profile(Section::Step);
        const StepResult sr = scheme->step(U_local, res, min_delta, t, std::min(p.t_end, next_write_time), p.dt_max);
        check_progress(t, sr.dt, sr.s_max, step + 1, p.time_scheme.c_str());
        if (p.use_ibm) solid_reset(U_local);
        t += sr.dt;
        scheme->stage_time = t;
        update_boundary_values(t, lb);
        ++step;
        const bool time_output = reached_time_output(t, next_write_time, p.write_interval_time);
        if (time_output) next_write_time += p.write_interval_time;
        if (monitoring && step % p.monitor_interval == 0) row(step);
        post_state(step, false, true);

        if (step % p.save_interval == 0 || time_output) {
            Field g = decomp.gather(U_local);
            if (root) {
                exporter.export_vtr(file_prefix, fluid, g, mesh, t, step);
                U = std::move(g);
                save_checkpoint(step);
                std::printf("Step %05d | t = %.5f s | dt = %.3e s | S_max = %.2f m/s | wall = %s | Checkpoint saved\n",
                            step, t, sr.dt, sr.s_max, format_elapsed(seconds_since(t0)).c_str());
                std::fflush(stdout);
            }
            last_snapshot_step = step;
        }
        // Agree on Ctrl+C across ranks so nobody blocks in a collective
        poll_stop_file();
        if (decomp.allreduce_max(g_stop_requested ? 1.0 : 0.0) > 0.0) {
            interrupted = true;
            break;
        }
        if (p.max_steps > 0 && step - start >= p.max_steps) break;
    }
    const double loop_wall = seconds_since(t0);

    if (monitoring && step % p.monitor_interval != 0) row(step);
    if (mon) std::fclose(mon);
    post_state(step, true, true);
    Field g = decomp.gather(U_local);
    if (root) {
        if (interrupted) std::printf("\n[INTERRUPTED] Stopping...\n");
        if (interrupted || last_snapshot_step != step) exporter.export_vtr(file_prefix, fluid, g, mesh, t, step);
        U = std::move(g);
        save_checkpoint(step);
        exporter.write_pvd(file_prefix);
        dump_final_state(step);
        std::printf("=== DONE  - open '%s.pvd' in ParaView ===  (time loop %.6f s, %d steps)\n", file_prefix.c_str(),
                    loop_wall, step - start);
    }
}
#endif   // FV_USE_MPI

}   // namespace fv
