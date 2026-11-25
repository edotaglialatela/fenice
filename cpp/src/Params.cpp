#include "fv/Params.hpp"

#include "fv/CaseFile.hpp"
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

// The one table of the parameters: key, case-file section, baseline default. A new parameter is added here (in the order a case file
// writes it) and in docs/source/guide/parameters.md (docs/build.py fails when a key is missing there); the typed view is SimParams::from.
const std::vector<ParamSpec>& parameter_table() {
    static const std::vector<ParamSpec> t = {
    // test_case is the case itself (no section, never in a file); a null default means no baseline value
    {"test_case", "", "Sod Shock Tube"},

    {"Lx", "mesh", "1.0"},
    {"Ly", "mesh", "1.0"},
    {"Lz", "mesh", "1.0"},
    {"nx", "mesh", "100"},
    {"ny", "mesh", "100"},
    {"nz", "mesh", "1"},

    {"eos_type", "thermo", "Ideal Gas"},
    {"gamma", "thermo", "1.4"},
    {"p_inf", "thermo", "0.0"},

    {"use_viscous", "physics", "false"},
    {"mu", "physics", "1e-3"},
    {"reynolds", "physics", "0"},
    {"pr", "physics", "0.72"},
    {"use_bodyforce", "physics", "false"},
    {"gx", "physics", "0.0"},
    {"gy", "physics", "-9.81"},
    {"gz", "physics", "0.0"},

    {"source_rho", "source", "0"},
    {"source_mx", "source", "0"},
    {"source_my", "source", "0"},
    {"source_mz", "source", "0"},
    {"source_E", "source", "0"},

    {"bc_xmin", "boundaries", "wall"},
    {"bc_xmax", "boundaries", "wall"},
    {"bc_ymin", "boundaries", "wall"},
    {"bc_ymax", "boundaries", "wall"},
    {"bc_zmin", "boundaries", "periodic"},
    {"bc_zmax", "boundaries", "periodic"},
    {"u_in", "boundaries", "0.0"},
    {"v_in", "boundaries", "0.0"},
    {"rho_in", "boundaries", "1.0"},
    {"p_out", "boundaries", "1.0"},
    {"u_in_table", "boundaries", ""},
    {"v_in_table", "boundaries", ""},
    {"rho_in_table", "boundaries", ""},
    {"p_out_table", "boundaries", ""},
    {"nrbc_sigma", "boundaries", "0.28"},
    {"nrbc_beta", "boundaries", "-1"},

    {"riemann_solver", "numerics", "HLLC"},
    {"reconstruction", "numerics", "none"},
    {"flux_quadrature", "numerics", "midpoint"},
    {"muscl_limiter", "numerics", "Van Leer"},
    {"weno_epsilon", "numerics", "1e-6"},
    {"positivity_fallback", "numerics", "false"},
    {"boundary_order", "numerics", "2"},
    {"time_scheme", "numerics", "auto"},
    {"cfl", "numerics", "0.5"},
    {"shock_blend", "numerics", "0.2"},
    // use_muscl: the retired switch of the second-order path, still read as an alias of reconstruction (SimParams::from)
    {"use_muscl", "numerics", nullptr},

    {"t_end", "run", "0.005"},
    {"save_interval", "run", "1"},
    {"write_interval_time", "run", "0"},
    {"dt_max", "run", "0"},
    {"monitor_interval", "run", "0"},
    {"max_steps", "run", "0"},
    {"resume", "run", "false"},
    {"output_root", "run", FV_PROJECT_ROOT "/results"},
    {"dump_state", "run", ""},
    {"vtk_format", "run", "binary"},

    {"use_mpi", "parallel", "false"},
    {"n_cpus", "parallel", "2"},

    {"use_ibm", "ibm", "false"},
    {"ibm_shape", "ibm", "Cylinder"},
    {"ibm_radius", "ibm", "0.1"},
    {"ibm_cx", "ibm", "0.5"},
    {"ibm_cy", "ibm", "0.5"},
    {"ibm_cz", "ibm", "0.5"},
    {"ibm_file", "ibm", ""},
    {"ibm_chord", "ibm", "1.0"},
    {"ibm_aoa", "ibm", "0.0"},
    // ibm_method: retired (the ghost-cell wall is the only wall); auto and ghost are still read, mask is an error (SimParams::from)
    {"ibm_method", "ibm", nullptr},
    {"ibm_curvature", "ibm", "false"},
    {"ibm_vx", "ibm", "0.0"},
    {"ibm_vy", "ibm", "0.0"},
    {"ibm_vz", "ibm", "0.0"},
    {"ibm_omega", "ibm", "0.0"},
    {"ibm_omega_x", "ibm", "0.0"},
    {"ibm_omega_y", "ibm", "0.0"},
    {"ibm_omega_from", "ibm", "0.0"},
    {"ibm_omega_until", "ibm", "0.0"},
    {"ibm_mass", "ibm", "0.0"},
    {"ibm_inertia", "ibm", "0.0"},
    {"ibm_stiff_theta", "ibm", "0.0"},
    {"ibm_damp_theta", "ibm", "0.0"},
    {"ibm_stiff_x", "ibm", "0.0"},
    {"ibm_stiff_y", "ibm", "0.0"},
    {"ibm_stiff_z", "ibm", "0.0"},
    {"ibm_damp_x", "ibm", "0.0"},
    {"ibm_damp_y", "ibm", "0.0"},
    {"ibm_damp_z", "ibm", "0.0"},

    {"use_hot_start", "hot_start", "false"},
    {"hot_start_file", "hot_start", ""},
    {"hot_start_src_nx", "hot_start", "0"},
    {"hot_start_src_ny", "hot_start", "0"},

    {"use_amr", "amr", "false"},
    {"amr_max_level", "amr", "2"},
    {"amr_indicator", "amr", "density+velocity"},
    {"amr_criterion", "amr", "0.1"},
    {"amr_refine_threshold", "amr", "0.15"},
    {"amr_derefine_threshold", "amr", "0.1"},
    {"amr_diffusion", "amr", "2"},
    {"amr_refine_every", "amr", "10"},
    {"amr_init_levels", "amr", "0"},
    // amr_refine_ibm: retired (the body is always kept at the finest level); true is still read, false is an error
    {"amr_refine_ibm", "amr", nullptr},
    {"amr_force_box", "amr", ""},
    };
    return t;
}

Params baseline_defaults() {
    Params p;
    for (const ParamSpec& s : parameter_table())
        if (s.default_value) p.set(s.key, s.default_value);
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
    s.bc_xmin = p.str("bc_xmin");
    s.bc_xmax = p.str("bc_xmax");
    s.bc_ymin = p.str("bc_ymin");
    s.bc_ymax = p.str("bc_ymax");
    s.bc_zmin = p.str("bc_zmin", "periodic");
    s.bc_zmax = p.str("bc_zmax", "periodic");
    s.riemann_solver = p.str("riemann_solver", "HLLC");
    s.time_scheme = p.str("time_scheme", "auto");
    s.cfl = p.num("cfl", 0.5);
    s.t_end = p.num("t_end");
    s.save_interval = std::max(1, p.integer("save_interval", 1));
    s.write_interval_time = p.num("write_interval_time", 0.0);
    s.dt_max = p.num("dt_max", 0.0);
    s.use_viscous = p.flag("use_viscous");
    s.mu = p.num("mu", 1e-3);
    s.pr = p.num("pr", 0.72);
    Mesh expression_mesh;
    expression_mesh.Lx = s.Lx;
    expression_mesh.Ly = s.Ly;
    expression_mesh.Lz = s.Lz;
    expression_mesh.nx = s.nx;
    expression_mesh.ny = s.ny;
    expression_mesh.nz = s.nz;
    s.u_in = compile_time_expression(p, expression_mesh, p.str("u_in", "0.0"), "u_in")(0.0);
    s.v_in = compile_time_expression(p, expression_mesh, p.str("v_in", "0.0"), "v_in")(0.0);
    s.rho_in = compile_time_expression(p, expression_mesh, p.str("rho_in", "1.0"), "rho_in")(0.0);
    s.p_out = compile_time_expression(p, expression_mesh, p.str("p_out", "1.0"), "p_out")(0.0);
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
    if (p.has("ibm_method")) {   // retired: the ghost-cell wall is the only wall
        const std::string method = lower(p.str("ibm_method", "auto"));
        if (method != "auto" && method != "ghost")
            throw std::invalid_argument("ibm_method = " + method + ": the key is retired, the immersed body always has the ghost-cell "
                                        "wall (remove the key)");
    }
    s.ibm_curvature = p.flag("ibm_curvature");
    s.ibm_vx = p.num("ibm_vx", 0.0);
    s.ibm_vy = p.num("ibm_vy", 0.0);
    s.ibm_vz = p.num("ibm_vz", 0.0);
    s.ibm_omega = p.num("ibm_omega", 0.0);
    s.ibm_omega_x = p.num("ibm_omega_x", 0.0);
    s.ibm_omega_y = p.num("ibm_omega_y", 0.0);
    s.ibm_omega_from = p.num("ibm_omega_from", 0.0);
    s.ibm_omega_until = p.num("ibm_omega_until", 0.0);
    s.ibm_mass = p.num("ibm_mass", 0.0);
    s.ibm_inertia = p.num("ibm_inertia", 0.0);
    s.ibm_stiff_theta = p.num("ibm_stiff_theta", 0.0);
    s.ibm_damp_theta = p.num("ibm_damp_theta", 0.0);
    s.ibm_stiff_x = p.num("ibm_stiff_x", 0.0);
    s.ibm_stiff_y = p.num("ibm_stiff_y", 0.0);
    s.ibm_stiff_z = p.num("ibm_stiff_z", 0.0);
    s.ibm_damp_x = p.num("ibm_damp_x", 0.0);
    s.ibm_damp_y = p.num("ibm_damp_y", 0.0);
    s.ibm_damp_z = p.num("ibm_damp_z", 0.0);
    s.reynolds = p.num("reynolds", 0.0);
    s.muscl_limiter = p.str("muscl_limiter", "Van Leer");
    s.reconstruction = lower(p.str("reconstruction", "none"));
    s.flux_quadrature = lower(p.str("flux_quadrature", "midpoint"));
    if (p.has("use_muscl")) {   // the retired switch of the second-order path, read with its old meaning
        s.use_muscl_alias = true;
        if (!p.flag("use_muscl")) {
            if (s.reconstruction == "weno3" || s.reconstruction == "weno5")
                throw std::invalid_argument("use_muscl = false contradicts reconstruction = " + s.reconstruction +
                                            " (use_muscl is retired: remove it and set reconstruction alone)");
            s.reconstruction = "none";   // use_muscl = false with reconstruction = muscl (its old default) was first order
        } else if (s.reconstruction == "none") {
            s.reconstruction = "muscl";
        }
    }
    // auto: MUSCL-Hancock with the MUSCL reconstruction, Explicit Euler otherwise (the first-order default)
    if (lower(s.time_scheme) == "auto") s.time_scheme = s.reconstruction == "muscl" ? "MUSCL-Hancock" : "Explicit Euler";
    s.weno_epsilon = p.num("weno_epsilon", 1e-6);
    s.positivity_fallback = p.flag("positivity_fallback");
    s.boundary_order = p.integer("boundary_order", 2);
    s.shock_blend = p.num("shock_blend", 0.2);
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
    if (p.has("amr_refine_ibm") && !p.flag("amr_refine_ibm"))   // retired: the wall needs the body at the finest level
        throw std::invalid_argument("amr_refine_ibm = false: the key is retired, an immersed body is always kept at the finest level "
                                    "(remove the key)");
    s.amr_force_box = p.str("amr_force_box", "");
    if (!s.amr_force_box.empty()) {
        double b[6];
        char extra = 0;
        const int got = std::sscanf(s.amr_force_box.c_str(), "%lf%*[ ,]%lf%*[ ,]%lf%*[ ,]%lf%*[ ,]%lf%*[ ,]%lf %c", &b[0], &b[1], &b[2],
                                    &b[3], &b[4], &b[5], &extra);
        const bool box3 = s.nz > 1 && got == 6;   // 3-D adaptive mesh: x0 x1 y0 y1 z0 z1
        if ((got != 4 && !box3) || !(b[1] > b[0]) || !(b[3] > b[2]) || (box3 && !(b[5] > b[4])))
            throw std::invalid_argument("amr_force_box must be 'x0 x1 y0 y1' (and 'z0 z1' for a 3-D mesh) with each lower bound below the upper; got '" +
                                        s.amr_force_box + "'");
    }
    s.amr_indicator = lower(p.str("amr_indicator", "density+velocity"));
    s.output_root = p.str("output_root", "results");
    s.dump_state = p.str("dump_state");
    s.max_steps = p.integer("max_steps", 0);
    s.monitor_interval = p.integer("monitor_interval", 0);
    s.vtk_format = lower(p.str("vtk_format", "binary"));

    if (s.nx <= 0 || s.ny <= 0 || s.nz <= 0) throw std::invalid_argument("nx, ny and nz must be positive");
    {   // cell and face indices are 32-bit: refuse a mesh whose face count (the largest index range) does not fit
        // (in double: the product of three ints can exceed 64 bits, and the comparison needs no exact count)
        const double nx = s.nx, ny = s.ny, nz = s.nz;
        const double faces = (nx + 1) * ny * nz + nx * (ny + 1) * nz + (s.nz > 1 ? nx * ny * (nz + 1) : 0.0);
        if (faces > 2147483647.0) {
            char count[32];
            std::snprintf(count, sizeof count, "%.4g", faces);
            throw std::invalid_argument(std::string("nx, ny, nz: the mesh has about ") + count +
                                        " faces, more than the 2147483647 a 32-bit index holds (about 700 million cells in 3-D)");
        }
    }
    if (s.Lx <= 0 || s.Ly <= 0 || s.Lz <= 0) throw std::invalid_argument("Lx, Ly and Lz must be positive");
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
    if (!(s.weno_epsilon >= 0.0)) throw std::invalid_argument("weno_epsilon must be >= 0");
    if (s.reconstruction != "none" && s.reconstruction != "muscl" && s.reconstruction != "weno3" && s.reconstruction != "weno5")
        throw std::invalid_argument("reconstruction must be none, muscl, weno3 or weno5; got '" + s.reconstruction + "'");
    if (s.reconstruction == "weno3" || s.reconstruction == "weno5") {
        const std::string ts = lower(s.time_scheme);
        if (s.reconstruction == "weno5" && s.use_mpi && !s.use_amr)
            throw std::invalid_argument("reconstruction = weno5 is serial on the uniform mesh (its MPI halo is two cells deep, WENO5 "
                                        "reads three): use weno3 with MPI, or AMR (weno5 runs on MPI ranks there)");
        if (s.reconstruction == "weno5" && s.use_amr) {
            // the adaptive stencil wraps around a periodic axis; the uniform solver replicates when the axis has fewer than three cells
            const std::pair<const std::string*, int> axes[] = {{&s.bc_xmin, s.nx}, {&s.bc_ymin, s.ny}, {&s.bc_zmin, s.nz}};
            for (int a = 0; a < (s.nz > 1 ? 3 : 2); ++a)
                if (lower(*axes[a].first) == "periodic" && axes[a].second < 3)
                    throw std::invalid_argument("reconstruction = weno5 with AMR needs at least three base cells along a periodic axis");
        }
        if (ts == "ssp runge-kutta 2" || ts == "ssprk2" || ts == "rk2")
            throw std::invalid_argument("reconstruction = " + s.reconstruction + " is not used with SSP Runge-Kutta 2 (use SSP "
                                        "Runge-Kutta 3)");
    }
    if (s.flux_quadrature != "midpoint" && s.flux_quadrature != "fourth")
        throw std::invalid_argument("flux_quadrature must be midpoint or fourth; got '" + s.flux_quadrature + "'");
    if (s.fourth_flux()) {
        if (s.reconstruction != "weno3" && s.reconstruction != "weno5")
            throw std::invalid_argument("flux_quadrature = fourth needs reconstruction = weno3 or weno5");
        if (s.use_amr) throw std::invalid_argument("flux_quadrature = fourth is not available with AMR yet");
        if (s.use_mpi) throw std::invalid_argument("flux_quadrature = fourth is serial yet (its corrections read faces two cells across a rank interface)");
        if (s.use_ibm) throw std::invalid_argument("flux_quadrature = fourth is not available with an immersed body yet");
    }
    if (const std::string ts = lower(s.time_scheme); ts == "muscl-hancock" || ts == "hancock") {
        const char* why = s.reconstruction != "muscl" ? "needs reconstruction = muscl"
                          : s.use_amr && s.positivity_fallback ? "is not available with positivity_fallback"
                                                      : nullptr;
        if (why) throw std::invalid_argument(std::string("time_scheme = MUSCL-Hancock ") + why);
    }
    if (s.boundary_order != 1 && s.boundary_order != 2) throw std::invalid_argument("boundary_order must be 1 or 2");
    if (s.positivity_fallback) {
        if (s.reconstruction != "weno3" && s.reconstruction != "weno5")
            throw std::invalid_argument("positivity_fallback needs reconstruction = weno3 or weno5 (the other schemes keep their reconstructed states between the cells)");
        if (s.flux_quadrature != "midpoint") throw std::invalid_argument("positivity_fallback needs flux_quadrature = midpoint");
    }
    // a first-order scheme has no reconstruction to extend to the boundary faces: the order is then 1 whatever was asked
    if (s.boundary_order == 2 && !s.reconstructs()) s.boundary_order = 1;
    if (s.use_amr && !(s.amr_criterion > 0.0)) throw std::invalid_argument("amr_criterion must be positive");
    if (!(s.amr_derefine_threshold <= s.amr_refine_threshold))
        throw std::invalid_argument("amr_derefine_threshold must not exceed amr_refine_threshold (else a refined cell "
                                    "would be joined and refined again)");
    if (s.use_amr && s.amr_diffusion < 0) throw std::invalid_argument("amr_diffusion must be >= 0");
    const std::string eos = lower(s.eos_type);
    if (eos != "ideal gas" && eos != "stiffened gas")
        throw std::invalid_argument("Unknown eos_type: '" + s.eos_type + "' (use Ideal Gas or Stiffened Gas)");
    const std::pair<const std::string*, const std::string*> sides[] = {
        {&s.bc_xmin, &s.bc_xmax}, {&s.bc_ymin, &s.bc_ymax}, {&s.bc_zmin, &s.bc_zmax}};
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
    if (!(s.write_interval_time >= 0.0) || !std::isfinite(s.write_interval_time))
        throw std::invalid_argument("write_interval_time must be finite and >= 0 (0: off)");
    if (!(s.dt_max >= 0.0) || !std::isfinite(s.dt_max)) throw std::invalid_argument("dt_max must be finite and >= 0 (0: off)");
    if (!(s.gamma > 1.0)) throw std::invalid_argument("gamma must be greater than 1");
    const std::string shape = lower(s.ibm_shape);
    const bool airfoil = shape == "airfoil" || !naca_digits(s.ibm_shape).empty();
    if (shape != "cylinder" && shape != "sphere" && !airfoil)
        throw std::invalid_argument("Unknown ibm_shape: '" + s.ibm_shape +
                                    "' (use Cylinder, Sphere, Airfoil or NACA dddd)");
    if (s.use_ibm && shape == "airfoil" && s.ibm_file.empty())
        throw std::invalid_argument("ibm_shape = Airfoil needs ibm_file (a Selig or Lednicer coordinate file)");
    if (s.use_ibm && airfoil && !(s.ibm_chord > 0.0)) throw std::invalid_argument("ibm_chord must be positive");
    if (s.use_ibm && s.nz > 1 && s.ibm_curvature) throw std::invalid_argument("ibm_curvature is 2-D only (for now)");
    if (!(s.ibm_omega_from >= 0.0) || !(s.ibm_omega_until >= 0.0)) throw std::invalid_argument("ibm_omega_from and ibm_omega_until must be >= 0");
    if ((s.ibm_omega_until > 0.0 || s.ibm_omega_from > 0.0) && s.ibm_omega == 0.0 && s.ibm_omega_x == 0.0 && s.ibm_omega_y == 0.0)
        throw std::invalid_argument("ibm_omega_from / ibm_omega_until need a rotation (ibm_omega)");
    if (s.ibm_omega_until > 0.0 && s.ibm_omega_until <= s.ibm_omega_from) throw std::invalid_argument("ibm_omega_until must be later than ibm_omega_from");
    if (s.use_ibm && s.ibm_moving() && s.ibm_curvature) throw std::invalid_argument("ibm_curvature is not available for a moving body");
    if (s.use_ibm && s.ibm_moving() && s.nz > 1) {   // in 3-D: a sphere that translates and spins
        if (shape != "sphere") throw std::invalid_argument("a moving body in 3-D is a sphere (for now)");
    }
    if (s.use_ibm && s.nz == 1 && (s.ibm_omega_x != 0.0 || s.ibm_omega_y != 0.0)) throw std::invalid_argument("ibm_omega_x and ibm_omega_y need a 3-D mesh (nz > 1)");
    if (s.use_ibm && s.ibm_moving() && s.nz == 1 && s.ibm_vz != 0.0) throw std::invalid_argument("ibm_vz needs a 3-D mesh (nz > 1)");
    if (!(s.ibm_mass >= 0.0) || !(s.ibm_stiff_x >= 0.0) || !(s.ibm_stiff_y >= 0.0) || !(s.ibm_stiff_z >= 0.0) || !(s.ibm_damp_x >= 0.0) ||
        !(s.ibm_damp_y >= 0.0) || !(s.ibm_damp_z >= 0.0))
        throw std::invalid_argument("ibm_mass, ibm_stiff_x/y/z and ibm_damp_x/y/z must be >= 0");
    if (!(s.ibm_inertia >= 0.0) || !(s.ibm_stiff_theta >= 0.0) || !(s.ibm_damp_theta >= 0.0))
        throw std::invalid_argument("ibm_inertia, ibm_stiff_theta and ibm_damp_theta must be >= 0");
    if (s.use_ibm && !s.ibm_drives_translation() &&
        (s.ibm_stiff_x != 0.0 || s.ibm_stiff_y != 0.0 || s.ibm_stiff_z != 0.0 || s.ibm_damp_x != 0.0 || s.ibm_damp_y != 0.0 || s.ibm_damp_z != 0.0))
        throw std::invalid_argument("ibm_stiff_* and ibm_damp_* act on a body driven by the flow: set ibm_mass > 0");
    if (s.use_ibm && !s.ibm_drives_rotation() && (s.ibm_stiff_theta != 0.0 || s.ibm_damp_theta != 0.0))
        throw std::invalid_argument("ibm_stiff_theta and ibm_damp_theta act on a body whose rotation is driven by the flow: set ibm_inertia > 0");
    if (s.use_ibm && s.ibm_dynamic()) {
        if (!s.ibm_drives_rotation() && (s.ibm_omega != 0.0 || s.ibm_omega_x != 0.0 || s.ibm_omega_y != 0.0))
            throw std::invalid_argument("a body driven by the flow translates only unless its rotation is driven too (ibm_omega must be 0, or set ibm_inertia > 0: then it is the initial angular velocity)");
        if (s.ibm_drives_rotation()) {
            if (s.nz > 1) throw std::invalid_argument("a driven rotation is 2-D for now (the sphere follows)");
            if (s.ibm_omega_from > 0.0 || s.ibm_omega_until > 0.0) throw std::invalid_argument("ibm_omega_from / ibm_omega_until act on a prescribed rotation, not on a driven one");
            if (!s.ibm_drives_translation() && (s.ibm_vx != 0.0 || s.ibm_vy != 0.0 || s.ibm_vz != 0.0))
                throw std::invalid_argument("a driven rotation with a prescribed translation is not available: the body is pinned at its placement (ibm_vx = ibm_vy = 0), or ibm_mass > 0 frees its translation too");
            if (s.ibm_drives_translation() && shape != "cylinder")
                throw std::invalid_argument("free translation and free rotation together are available for a cylinder only (the pivot is its centre of mass); a body whose pivot is not its centre of mass couples the two");
        }
        if (const std::string ts = lower(s.time_scheme); ts == "muscl-hancock" || ts == "hancock")
            throw std::invalid_argument("a body driven by the flow needs a Runge-Kutta or Euler time scheme (its loads come from the first stage)");
        if (s.positivity_fallback) throw std::invalid_argument("a body driven by the flow cannot use positivity_fallback");
    }
    if (s.reynolds < 0.0) throw std::invalid_argument("reynolds must be >= 0 (0: use mu)");
    if (s.reynolds > 0.0) {
        // mu from the Reynolds number on the body: rho_in u_in L / mu, L = chord or diameter
        if (!s.use_ibm) throw std::invalid_argument("reynolds needs an IBM body for its length scale (or set mu)");
        // A boundary table is resolved next to the case file in Simulation::setup; defer the
        // positive-reference check and derived mu until then if the numeric t=0 values are absent.
        if (s.u_in > 0.0 && s.rho_in > 0.0) {
            const double L = airfoil ? s.ibm_chord : 2.0 * s.ibm_radius;
            s.mu = s.rho_in * s.u_in * L / s.reynolds;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.17g", s.mu);
            s.source.set("mu", buf);   // expressions in the case file see the same mu
        } else {
            s.mu = 0.0;   // placeholder: Simulation::setup replaces it after resolving boundary tables
            s.source.set("mu", "0.0");
        }
    }
    if (s.use_viscous && (!(s.mu >= 0.0) || !(s.pr > 0.0)))
        throw std::invalid_argument("mu must be >= 0 and pr > 0 with use_viscous");
    return s;
}

}   // namespace fv
