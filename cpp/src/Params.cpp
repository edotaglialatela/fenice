#include "fv/Params.hpp"

#include "fv/Geometry.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <utility>

#ifndef FV_PROJECT_ROOT
#define FV_PROJECT_ROOT "."
#endif

namespace fv {

namespace {
std::string trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
// Whole-string number ("1e-3" yes, "0.5x" or "1,5" no).
bool parse_number(const std::string& s, double& v) {
    try {
        size_t used = 0;
        v = std::stod(s, &used);
        return trim(s.substr(used)).empty();
    } catch (const std::exception&) { return false; }
}
}   // namespace

void Params::merge(const Params& other) {
    for (const auto& kv : other.values_) values_[kv.first] = kv.second;
}

std::string Params::str(const std::string& key, const std::string& def) const {
    auto it = values_.find(key);
    return it == values_.end() ? def : it->second;
}

double Params::num(const std::string& key, double def) const {
    auto it = values_.find(key);
    if (it == values_.end() || it->second.empty()) return def;
    double v = 0.0;
    if (!parse_number(it->second, v))
        throw std::invalid_argument("Parameter '" + key + "' is not a number: '" + it->second + "'");
    return v;
}

int Params::integer(const std::string& key, int def) const {
    auto it = values_.find(key);
    if (it == values_.end() || it->second.empty()) return def;
    double v = 0.0;   // "100" and "100.0" (as Python may write it) are both accepted
    if (!parse_number(it->second, v) || v != std::floor(v) || std::abs(v) > 2147483647.0)
        throw std::invalid_argument("Parameter '" + key + "' is not an integer: '" + it->second + "'");
    return static_cast<int>(v);
}

bool Params::flag(const std::string& key, bool def) const {
    auto it = values_.find(key);
    if (it == values_.end() || it->second.empty()) return def;
    const std::string v = lower(it->second);
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    throw std::invalid_argument("Parameter '" + key + "' is not a boolean: '" + it->second + "'");
}

void Params::set_assignment(const std::string& kv) {
    const size_t eq = kv.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("Expected key=value, got '" + kv + "'");
    set(trim(kv.substr(0, eq)), trim(kv.substr(eq + 1)));
}

void Params::load_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open config file '" + path + "'");
    std::string line;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        set_assignment(line);
    }
}

Params baseline_defaults() {
    Params p;
    const char* kv[][2] = {
        {"test_case", "Sod Shock Tube"},
        {"Lx", "1.0"},
        {"Ly", "1.0"},
        {"nx", "100"},
        {"ny", "100"},
        {"Lz", "1.0"},
        {"nz", "1"},
        {"eos_type", "Ideal Gas"},
        {"p_inf", "0.0"},
        {"gamma", "1.4"},
        {"mu", "1e-3"},
        {"pr", "0.72"},
        {"cfl", "0.5"},
        {"t_end", "0.005"},
        {"save_interval", "1"},
        {"use_viscous", "false"},
        {"riemann_solver", "HLLC"},
        {"time_scheme", "Explicit Euler"},
        {"bc_left", "wall"},
        {"bc_right", "wall"},
        {"bc_bottom", "wall"},
        {"bc_top", "wall"},
        {"bc_front", "periodic"},
        {"bc_back", "periodic"},
        {"u_in", "0.0"},
        {"rho_in", "1.0"},
        {"p_out", "1.0"},
        {"nrbc_sigma", "0.28"},
        {"nrbc_beta", "-1"},
        {"use_bodyforce", "false"},
        {"gx", "0.0"},
        {"gy", "-9.81"},
        {"gz", "0.0"},
        {"source_rho", "0"},
        {"source_mx", "0"},
        {"source_my", "0"},
        {"source_mz", "0"},
        {"source_E", "0"},
        {"resume", "false"},
        {"use_mpi", "false"},
        {"n_cpus", "2"},
        {"use_ibm", "false"},
        {"ibm_shape", "Cylinder"},
        {"ibm_radius", "0.1"},
        {"ibm_cx", "0.5"},
        {"ibm_cy", "0.5"},
        {"ibm_cz", "0.5"},
        {"ibm_file", ""},
        {"ibm_chord", "1.0"},
        {"ibm_aoa", "0.0"},
        {"reynolds", "0"},
        {"use_muscl", "false"},
        {"muscl_limiter", "Van Leer"},
        {"dt_phys", "0.1"},
        {"max_inner_iter", "50"},
        {"inner_tol_rel", "1e-3"},
        {"use_hot_start", "false"},
        {"hot_start_file", ""},
        {"hot_start_src_nx", "0"},
        {"hot_start_src_ny", "0"},
        {"use_amr", "false"},
        {"amr_max_level", "2"},
        {"amr_criterion", "0.1"},
        {"amr_refine_threshold", "0.15"},
        {"amr_derefine_threshold", "0.1"},
        {"amr_diffusion", "2"},
        {"amr_refine_every", "10"},
        {"amr_init_levels", "0"},
        {"amr_refine_ibm", "true"},
        {"amr_indicator", "density+velocity"},
        {"output_root", FV_PROJECT_ROOT "/results"},
        {"dump_state", ""},
        {"max_steps", "0"},
        {"monitor_interval", "0"},
        {"vtk_format", "binary"},
    };
    for (const auto& e : kv) p.set(e[0], e[1]);
    return p;
}

SimParams SimParams::from(const Params& p) {
    SimParams s;
    s.source = p;
    s.test_case = p.str("test_case");
    s.Lx = p.num("Lx");
    s.Ly = p.num("Ly");
    s.nx = p.integer("nx");
    s.ny = p.integer("ny");
    s.Lz = p.num("Lz", 1.0);
    s.nz = p.integer("nz", 1);
    s.eos_type = p.str("eos_type", "Ideal Gas");
    s.gamma = p.num("gamma", 1.4);
    s.p_inf = (lower(s.eos_type) == "stiffened gas") ? p.num("p_inf", 0.0) : 0.0;
    s.bc_left = p.str("bc_left");
    s.bc_right = p.str("bc_right");
    s.bc_bottom = p.str("bc_bottom");
    s.bc_top = p.str("bc_top");
    s.bc_front = p.str("bc_front", "periodic");
    s.bc_back = p.str("bc_back", "periodic");
    s.riemann_solver = p.str("riemann_solver", "HLLC");
    s.time_scheme = p.str("time_scheme", "Explicit Euler");
    s.cfl = p.num("cfl", 0.5);
    s.t_end = p.num("t_end");
    s.save_interval = std::max(1, p.integer("save_interval", 1));
    s.use_viscous = p.flag("use_viscous");
    s.mu = p.num("mu", 1e-3);
    s.pr = p.num("pr", 0.72);
    s.u_in = p.num("u_in", 0.0);
    s.rho_in = p.num("rho_in", 1.0);
    s.p_out = p.num("p_out", 1.0);
    s.nrbc_sigma = p.num("nrbc_sigma", 0.28);
    s.nrbc_beta = p.num("nrbc_beta", -1.0);
    s.use_bodyforce = p.flag("use_bodyforce");
    s.gx = p.num("gx", 0.0);
    s.gy = p.num("gy", 0.0);
    s.gz = p.num("gz", 0.0);
    s.use_source = false;
    s.source_unsteady = false;
    for (const char* key : {"source_rho", "source_mx", "source_my", "source_mz", "source_E"}) {
        const std::string v = p.str(key, "0");
        char* end = nullptr;
        const double d = std::strtod(v.c_str(), &end);
        if (!(end && *end == '\0' && d == 0.0)) s.use_source = true;   // an expression, or a non-zero number
        // The identifier `t` (not the t of sqrt, t_end or tan: whole identifiers only; numbers like 1e-3 are skipped)
        for (size_t i = 0; i < v.size();) {
            if (std::isdigit(static_cast<unsigned char>(v[i])) || v[i] == '.') {
                while (i < v.size() && (std::isalnum(static_cast<unsigned char>(v[i])) || v[i] == '.' ||
                                        ((v[i] == '+' || v[i] == '-') && (v[i - 1] == 'e' || v[i - 1] == 'E'))))
                    ++i;
            } else if (std::isalpha(static_cast<unsigned char>(v[i])) || v[i] == '_') {
                size_t j = i;
                while (j < v.size() && (std::isalnum(static_cast<unsigned char>(v[j])) || v[j] == '_')) ++j;
                if (j - i == 1 && v[i] == 't') s.source_unsteady = true;
                i = j;
            } else {
                ++i;
            }
        }
    }
    s.resume = p.flag("resume");
    s.use_mpi = p.flag("use_mpi");
    s.n_cpus = p.integer("n_cpus", 1);
    s.use_ibm = p.flag("use_ibm");
    s.ibm_shape = p.str("ibm_shape", "Cylinder");
    s.ibm_radius = p.num("ibm_radius", 0.1);
    s.ibm_cx = p.num("ibm_cx", 0.5);
    s.ibm_cy = p.num("ibm_cy", 0.5);
    s.ibm_cz = p.num("ibm_cz", 0.5);
    s.ibm_file = p.str("ibm_file", "");
    s.ibm_chord = p.num("ibm_chord", 1.0);
    s.ibm_aoa = p.num("ibm_aoa", 0.0);
    s.reynolds = p.num("reynolds", 0.0);
    s.use_muscl = p.flag("use_muscl");
    s.muscl_limiter = p.str("muscl_limiter", "Van Leer");
    s.dt_phys = p.num("dt_phys", s.t_end / 100.0);
    s.max_inner_iter = p.integer("max_inner_iter", 50);
    s.inner_tol_rel = p.num("inner_tol_rel", 1e-3);
    s.use_hot_start = p.flag("use_hot_start");
    s.hot_start_file = p.str("hot_start_file");
    s.hot_start_src_nx = p.integer("hot_start_src_nx", 0);
    s.hot_start_src_ny = p.integer("hot_start_src_ny", 0);
    s.use_amr = p.flag("use_amr");
    s.amr_max_level = p.integer("amr_max_level", 2);
    s.amr_criterion = p.num("amr_criterion", 0.1);
    s.amr_refine_threshold = p.num("amr_refine_threshold", 0.15);
    s.amr_derefine_threshold = p.num("amr_derefine_threshold", 0.1);
    s.amr_diffusion = p.integer("amr_diffusion", 2);
    s.amr_refine_every = std::max(1, p.integer("amr_refine_every", 10));
    s.amr_init_levels = p.integer("amr_init_levels", 0);
    s.amr_refine_ibm = p.flag("amr_refine_ibm");
    s.amr_indicator = lower(p.str("amr_indicator", "density+velocity"));
    s.output_root = p.str("output_root", "results");
    s.dump_state = p.str("dump_state");
    s.max_steps = p.integer("max_steps", 0);
    s.monitor_interval = p.integer("monitor_interval", 0);
    s.vtk_format = lower(p.str("vtk_format", "binary"));

    if (s.nx <= 0 || s.ny <= 0 || s.nz <= 0) throw std::invalid_argument("nx, ny and nz must be positive");
    if (s.Lx <= 0 || s.Ly <= 0 || s.Lz <= 0) throw std::invalid_argument("Lx, Ly and Lz must be positive");
    if (s.use_amr && s.nz > 1) throw std::invalid_argument("AMR is 2-D only (nz must be 1 when use_amr is on)");
    if (s.use_source && s.use_ibm) throw std::invalid_argument("source terms (source_*) are not supported with the IBM");
    {   // amr_indicator: variables joined by '+', each one of density, velocity, pressure
        const std::string& ind = s.amr_indicator;
        size_t b = 0;
        bool ok = !ind.empty();
        while (ok && b <= ind.size()) {
            size_t e = ind.find('+', b);
            if (e == std::string::npos) e = ind.size();
            const std::string v = ind.substr(b, e - b);
            ok = v == "density" || v == "velocity" || v == "pressure";
            b = e + 1;
        }
        if (!ok)
            throw std::invalid_argument("amr_indicator must be density, velocity or pressure, or several of them joined by "
                                        "'+' (e.g. density+velocity); got '" + ind + "'");
    }
    if (s.use_amr && !(s.amr_criterion > 0.0)) throw std::invalid_argument("amr_criterion must be positive");
    if (!(s.amr_derefine_threshold <= s.amr_refine_threshold))
        throw std::invalid_argument("amr_derefine_threshold must not exceed amr_refine_threshold (else a refined cell "
                                    "would be joined and refined again)");
    if (s.use_amr && s.amr_diffusion < 0) throw std::invalid_argument("amr_diffusion must be >= 0");
    if (s.use_amr) {
        const std::string ts = lower(s.time_scheme);
        if (ts == "bdf2 dual time" || ts == "bdf2")
            throw std::invalid_argument("AMR does not support the BDF2 dual-time scheme (use Explicit Euler, SSP "
                                        "Runge-Kutta 2 or Runge-Kutta 4)");
    }
    const std::string eos = lower(s.eos_type);
    if (eos != "ideal gas" && eos != "stiffened gas")
        throw std::invalid_argument("Unknown eos_type: '" + s.eos_type + "' (use Ideal Gas or Stiffened Gas)");
    const std::pair<const std::string*, const std::string*> sides[] = {
        {&s.bc_left, &s.bc_right}, {&s.bc_bottom, &s.bc_top}, {&s.bc_front, &s.bc_back}};
    const char* axis = "xyz";
    for (int a = 0; a < 3; ++a)
        if ((lower(*sides[a].first) == "periodic") != (lower(*sides[a].second) == "periodic"))
            throw std::invalid_argument(std::string("Periodic boundaries come in pairs: both ") + axis[a] +
                                        " sides must be periodic, or neither (got '" + *sides[a].first + "' and '" +
                                        *sides[a].second + "')");
    if (s.vtk_format != "binary" && s.vtk_format != "ascii")
        throw std::invalid_argument("Unknown vtk_format: '" + s.vtk_format + "' (use binary or ascii)");
    if (!(s.nrbc_sigma >= 0.0)) throw std::invalid_argument("nrbc_sigma must be >= 0");
    if (!(s.nrbc_beta <= 1.0)) throw std::invalid_argument("nrbc_beta must be <= 1 (< 0: local Mach number)");
    if (!(s.cfl > 0.0)) throw std::invalid_argument("cfl must be positive");
    if (!(s.t_end > 0.0)) throw std::invalid_argument("t_end must be positive");
    if (!(s.gamma > 1.0)) throw std::invalid_argument("gamma must be greater than 1");
    const std::string shape = lower(s.ibm_shape);
    const bool airfoil = shape == "airfoil" || !naca_digits(s.ibm_shape).empty();
    if (shape != "cylinder" && shape != "sphere" && !airfoil)
        throw std::invalid_argument("Unknown ibm_shape: '" + s.ibm_shape +
                                    "' (use Cylinder, Sphere, Airfoil or NACA dddd)");
    if (s.use_ibm && shape == "airfoil" && s.ibm_file.empty())
        throw std::invalid_argument("ibm_shape = Airfoil needs ibm_file (a Selig or Lednicer coordinate file)");
    if (s.use_ibm && airfoil && !(s.ibm_chord > 0.0)) throw std::invalid_argument("ibm_chord must be positive");
    if (s.reynolds < 0.0) throw std::invalid_argument("reynolds must be >= 0 (0: use mu)");
    if (s.reynolds > 0.0) {
        // mu from the Reynolds number on the body: rho_in u_in L / mu, L = chord or diameter
        if (!s.use_ibm) throw std::invalid_argument("reynolds needs an IBM body for its length scale (or set mu)");
        if (!(s.u_in > 0.0) || !(s.rho_in > 0.0))
            throw std::invalid_argument("reynolds needs positive u_in and rho_in");
        const double L = airfoil ? s.ibm_chord : 2.0 * s.ibm_radius;
        s.mu = s.rho_in * s.u_in * L / s.reynolds;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.17g", s.mu);
        s.source.set("mu", buf);   // expressions in the case file see the same mu
    }
    if (s.use_viscous && (!(s.mu >= 0.0) || !(s.pr > 0.0)))
        throw std::invalid_argument("mu must be >= 0 and pr > 0 with use_viscous");
    return s;
}

}   // namespace fv
