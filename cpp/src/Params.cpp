#include "fv/Params.hpp"

#include <cctype>
#include <fstream>
#include <stdexcept>

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
}  // namespace

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
    try {
        return std::stod(it->second);
    } catch (const std::exception&) {
        throw std::invalid_argument("Parameter '" + key + "' is not a number: '" + it->second + "'");
    }
}

int Params::integer(const std::string& key, int def) const {
    auto it = values_.find(key);
    if (it == values_.end() || it->second.empty()) return def;
    try {
        size_t used = 0;
        const double v = std::stod(it->second, &used);
        return static_cast<int>(v);
    } catch (const std::exception&) {
        throw std::invalid_argument("Parameter '" + key + "' is not an integer: '" + it->second + "'");
    }
}

bool Params::flag(const std::string& key, bool def) const {
    auto it = values_.find(key);
    if (it == values_.end() || it->second.empty()) return def;
    const std::string v = lower(it->second);
    if (v == "1" || v == "true" || v == "yes" || v == "on")  return true;
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
        {"Lx", "1.0"}, {"Ly", "1.0"}, {"nx", "100"}, {"ny", "100"}, {"Lz", "1.0"}, {"nz", "1"},
        {"eos_type", "Ideal Gas"}, {"p_inf", "0.0"}, {"gamma", "1.4"},
        {"mu", "1e-3"}, {"pr", "0.72"}, {"cfl", "0.5"},
        {"t_end", "0.005"}, {"save_interval", "1"}, {"use_viscous", "false"},
        {"riemann_solver", "HLLC"}, {"time_scheme", "Explicit Euler"},
        {"bc_left", "wall"}, {"bc_right", "wall"}, {"bc_bottom", "wall"}, {"bc_top", "wall"},
        {"bc_front", "periodic"}, {"bc_back", "periodic"},
        {"u_in", "0.0"}, {"rho_in", "1.0"}, {"p_out", "1.0"},
        {"use_bodyforce", "false"}, {"gx", "0.0"}, {"gy", "-9.81"}, {"gz", "0.0"},
        {"resume", "false"}, {"use_mpi", "false"}, {"n_cpus", "2"},
        {"use_ibm", "false"}, {"ibm_shape", "Cylinder"}, {"ibm_radius", "0.1"},
        {"ibm_cx", "0.5"}, {"ibm_cy", "0.5"}, {"ibm_cz", "0.5"},
        {"use_muscl", "false"}, {"muscl_limiter", "Van Leer"},
        {"dt_phys", "0.1"}, {"max_inner_iter", "50"}, {"inner_tol_rel", "1e-3"},
        {"use_hot_start", "false"}, {"hot_start_file", ""},
        {"hot_start_src_nx", "0"}, {"hot_start_src_ny", "0"},
        {"use_amr", "false"}, {"amr_max_level", "2"}, {"amr_patch_nx", "8"}, {"amr_patch_ny", "8"},
        {"amr_refine_threshold", "0.3"}, {"amr_derefine_threshold", "0.1"},
        {"amr_refine_every", "10"}, {"amr_init_levels", "0"}, {"amr_buffer_layers", "1"},
        {"output_root", FV_PROJECT_ROOT "/results"}, {"dump_state", ""}, {"max_steps", "0"},
        {"monitor_interval", "0"},
    };
    for (const auto& e : kv) p.set(e[0], e[1]);
    return p;
}

SimParams SimParams::from(const Params& p) {
    SimParams s;
    s.source = p;
    s.test_case      = p.str("test_case");
    s.Lx             = p.num("Lx");
    s.Ly             = p.num("Ly");
    s.nx             = p.integer("nx");
    s.ny             = p.integer("ny");
    s.Lz             = p.num("Lz", 1.0);
    s.nz             = p.integer("nz", 1);
    s.eos_type       = p.str("eos_type", "Ideal Gas");
    s.gamma          = p.num("gamma", 1.4);
    s.p_inf          = (lower(s.eos_type) == "stiffened gas") ? p.num("p_inf", 0.0) : 0.0;
    s.bc_left        = p.str("bc_left");
    s.bc_right       = p.str("bc_right");
    s.bc_bottom      = p.str("bc_bottom");
    s.bc_top         = p.str("bc_top");
    s.bc_front       = p.str("bc_front", "periodic");
    s.bc_back        = p.str("bc_back", "periodic");
    s.riemann_solver = p.str("riemann_solver", "HLLC");
    s.time_scheme    = p.str("time_scheme", "Explicit Euler");
    s.cfl            = p.num("cfl", 0.5);
    s.t_end          = p.num("t_end");
    s.save_interval  = std::max(1, p.integer("save_interval", 1));
    s.use_viscous    = p.flag("use_viscous");
    s.mu             = p.num("mu", 1e-3);
    s.pr             = p.num("pr", 0.72);
    s.u_in           = p.num("u_in", 0.0);
    s.rho_in         = p.num("rho_in", 1.0);
    s.p_out          = p.num("p_out", 1.0);
    s.use_bodyforce  = p.flag("use_bodyforce");
    s.gx             = p.num("gx", 0.0);
    s.gy             = p.num("gy", 0.0);
    s.gz             = p.num("gz", 0.0);
    s.resume         = p.flag("resume");
    s.use_mpi        = p.flag("use_mpi");
    s.n_cpus         = p.integer("n_cpus", 1);
    s.use_ibm        = p.flag("use_ibm");
    s.ibm_shape      = p.str("ibm_shape", "Cylinder");
    s.ibm_radius     = p.num("ibm_radius", 0.1);
    s.ibm_cx         = p.num("ibm_cx", 0.5);
    s.ibm_cy         = p.num("ibm_cy", 0.5);
    s.ibm_cz         = p.num("ibm_cz", 0.5);
    s.use_muscl      = p.flag("use_muscl");
    s.muscl_limiter  = p.str("muscl_limiter", "Van Leer");
    s.dt_phys        = p.num("dt_phys", s.t_end / 100.0);
    s.max_inner_iter = p.integer("max_inner_iter", 50);
    s.inner_tol_rel  = p.num("inner_tol_rel", 1e-3);
    s.use_hot_start  = p.flag("use_hot_start");
    s.hot_start_file = p.str("hot_start_file");
    s.hot_start_src_nx = p.integer("hot_start_src_nx", 0);
    s.hot_start_src_ny = p.integer("hot_start_src_ny", 0);
    s.use_amr        = p.flag("use_amr");
    s.amr_max_level  = p.integer("amr_max_level", 2);
    s.amr_patch_nx   = p.integer("amr_patch_nx", 8);
    s.amr_patch_ny   = p.integer("amr_patch_ny", 8);
    s.amr_refine_threshold   = p.num("amr_refine_threshold", 0.3);
    s.amr_derefine_threshold = p.num("amr_derefine_threshold", 0.1);
    s.amr_refine_every  = std::max(1, p.integer("amr_refine_every", 10));
    s.amr_init_levels   = p.integer("amr_init_levels", 0);
    s.amr_buffer_layers = p.integer("amr_buffer_layers", 1);
    s.output_root    = p.str("output_root", "results");
    s.dump_state     = p.str("dump_state");
    s.max_steps      = p.integer("max_steps", 0);
    s.monitor_interval = p.integer("monitor_interval", 0);

    if (s.nx <= 0 || s.ny <= 0 || s.nz <= 0) throw std::invalid_argument("nx, ny and nz must be positive");
    if (s.Lx <= 0 || s.Ly <= 0 || s.Lz <= 0) throw std::invalid_argument("Lx, Ly and Lz must be positive");
    if (s.use_amr && s.nz > 1) throw std::invalid_argument("AMR is 2-D only (nz must be 1 when use_amr is on)");
    return s;
}

}  // namespace fv
