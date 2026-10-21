#include "fv/CaseFile.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

#include <toml.hpp>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4200)   // TinyExpr's flexible array member (valid C)
#endif
#include "tinyexpr.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

namespace fs = std::filesystem;

namespace fv {

namespace {

// Canonical section of every parameter (test_case is the case itself, so never in a file).
const std::vector<std::pair<std::string, std::vector<std::string>>>& section_table() {
    static const std::vector<std::pair<std::string, std::vector<std::string>>> t = {
        {"mesh", {"Lx", "Ly", "Lz", "nx", "ny", "nz"}},
        {"thermo", {"eos_type", "gamma", "p_inf"}},
        {"physics", {"use_viscous", "mu", "reynolds", "pr", "use_bodyforce", "gx", "gy", "gz"}},
        {"boundaries", {"bc_left", "bc_right", "bc_bottom", "bc_top", "bc_front", "bc_back", "u_in", "rho_in", "p_out",
                        "nrbc_sigma", "nrbc_beta"}},
        {"numerics", {"riemann_solver", "use_muscl", "muscl_limiter", "time_scheme", "cfl", "dt_phys", "max_inner_iter",
                      "inner_tol_rel"}},
        {"run", {"t_end", "save_interval", "monitor_interval", "max_steps", "resume", "output_root", "dump_state", "vtk_format"}},
        {"parallel", {"use_mpi", "n_cpus"}},
        {"ibm", {"use_ibm", "ibm_shape", "ibm_radius", "ibm_cx", "ibm_cy", "ibm_cz", "ibm_file", "ibm_chord",
                 "ibm_aoa"}},
        {"hot_start", {"use_hot_start", "hot_start_file", "hot_start_src_nx", "hot_start_src_ny"}},
        {"amr", {"use_amr", "amr_max_level", "amr_patch_nx", "amr_patch_ny", "amr_refine_threshold",
                 "amr_derefine_threshold", "amr_refine_every", "amr_init_levels", "amr_buffer_layers", "amr_refine_ibm", "amr_indicator"}},
    };
    return t;
}

std::vector<std::string> all_param_keys() {
    std::vector<std::string> keys;
    for (const auto& s : section_table()) keys.insert(keys.end(), s.second.begin(), s.second.end());
    return keys;
}

const std::vector<std::string> kStateKeys = {"rho", "u", "v", "w", "p"};
const std::vector<std::string> kRegionKeys = {"shape", "min", "max", "point", "normal", "center", "radius", "axis", "state"};
const std::vector<std::string> kShapes = {"box", "halfspace", "sphere", "cylinder"};
const std::vector<std::string> kCaseKeys = {"name", "description", "order", "requires_viscous", "requires_3d"};

std::string hint(const std::string& word, const std::vector<std::string>& options) {
    const std::string m = closest_match(word, options);
    return m.empty() ? "" : " (did you mean '" + m + "'?)";
}

[[noreturn]] void fail(const std::string& file, const toml::node* n, const std::string& msg) {
    std::string where = file;
    if (n && n->source().begin.line) where += ":" + std::to_string(n->source().begin.line);
    throw std::runtime_error(where + ": " + msg);
}

// Shortest decimal that reads back to exactly `v` ("1.0", "0.0025", "6.283185307179586").
std::string format_double(double v) {
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof(buf), v);
    std::string s(buf, r.ptr);
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
    return s;
}

// A TOML scalar as a parameter string (numbers keep their exact value).
std::string scalar_string(const std::string& file, const toml::node& n, const std::string& what) {
    if (auto v = n.as_string()) return v->get();
    if (auto v = n.as_boolean()) return v->get() ? "true" : "false";
    if (auto v = n.as_integer()) return std::to_string(v->get());
    if (auto v = n.as_floating_point()) return format_double(v->get());
    fail(file, &n, what + " must be a number, a string or a boolean");
}

// A number or an expression string (geometry entries and state values).
std::string number_or_expr(const std::string& file, const toml::node& n, const std::string& what) {
    if (n.is_string() || n.is_integer() || n.is_floating_point()) return scalar_string(file, n, what);
    fail(file, &n, what + " must be a number or an expression string");
}

std::vector<std::string> vector_of(const std::string& file, const toml::node& n, const std::string& what) {
    const toml::array* a = n.as_array();
    if (!a || a->size() < 2 || a->size() > 3) fail(file, &n, what + " must be an array of 2 or 3 entries");
    std::vector<std::string> out;
    for (const auto& e : *a) out.push_back(number_or_expr(file, e, what + " entries"));
    return out;
}

std::map<std::string, std::string> state_of(const std::string& file, const toml::node& n, const std::string& what) {
    const toml::table* t = n.as_table();
    if (!t) fail(file, &n, what + " must be a table, e.g. { rho = 1.0, p = 1.0 }");
    std::map<std::string, std::string> out;
    for (const auto& [k, v] : *t) {
        const std::string key(k.str());
        if (std::find(kStateKeys.begin(), kStateKeys.end(), key) == kStateKeys.end())
            fail(file, &v, "unknown variable '" + key + "' in " + what + hint(key, kStateKeys) +
                               "; use rho, u, v, w, p");
        out[key] = number_or_expr(file, v, what + "." + key);
    }
    return out;
}

InitialRegion region_of(const std::string& file, const toml::node& n, int index) {
    const std::string what = "region " + std::to_string(index);
    const toml::table* t = n.as_table();
    if (!t) fail(file, &n, "[[initial.region]] entries must be tables");
    InitialRegion r;
    for (const auto& [k, v] : *t) {
        const std::string key(k.str());
        if (key == "shape") r.shape = scalar_string(file, v, what + ".shape");
        else if (key == "min") r.min = vector_of(file, v, what + ".min");
        else if (key == "max") r.max = vector_of(file, v, what + ".max");
        else if (key == "point") r.point = vector_of(file, v, what + ".point");
        else if (key == "normal") r.normal = vector_of(file, v, what + ".normal");
        else if (key == "center") r.center = vector_of(file, v, what + ".center");
        else if (key == "radius") r.radius = number_or_expr(file, v, what + ".radius");
        else if (key == "axis") r.axis = scalar_string(file, v, what + ".axis");
        else if (key == "state") r.state = state_of(file, v, what + ".state");
        else fail(file, &v, "unknown key '" + key + "' in " + what + hint(key, kRegionKeys));
    }
    auto need = [&](bool ok, const std::string& msg) { if (!ok) fail(file, &n, what + ": " + msg); };
    need(!r.shape.empty(), "missing 'shape' (box, halfspace, sphere or cylinder)");
    need(std::find(kShapes.begin(), kShapes.end(), r.shape) != kShapes.end(),
         "unknown shape '" + r.shape + "'" + hint(r.shape, kShapes));
    need(!r.state.empty(), "missing 'state', e.g. state = { rho = 1.0, p = 1.0 }");
    if (r.shape == "box") need(!r.min.empty() && !r.max.empty(), "a box needs 'min' and 'max'");
    if (r.shape == "halfspace") need(!r.point.empty() && !r.normal.empty(), "a halfspace needs 'point' and 'normal'");
    if (r.shape == "sphere" || r.shape == "cylinder")
        need(!r.center.empty() && !r.radius.empty(), "a " + r.shape + " needs 'center' and 'radius'");
    need(r.axis == "x" || r.axis == "y" || r.axis == "z", "axis must be \"x\", \"y\" or \"z\"");
    return r;
}

// ---------------------------------------------------------------------------
// Expressions (TinyExpr) over the cell centre and the numeric run parameters
// ---------------------------------------------------------------------------

bool parse_number(const std::string& s, double& out) {
    if (s.empty()) return false;
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return end && *end == '\0';
}

class Evaluator {
public:
    Evaluator(const Params& run, const Mesh& m) {
        // Variable storage must not move: te_compile keeps the addresses
        names_.reserve(run.values().size() + 8);
        values_.reserve(run.values().size() + 8);
        auto add = [&](const std::string& name, double v) {
            for (size_t i = 0; i < names_.size(); ++i)
                if (names_[i] == name) { values_[i] = v; return; }
            names_.push_back(name);
            values_.push_back(v);
        };
        add("x", 0.0); add("y", 0.0); add("z", 0.0);
        for (const auto& [k, v] : run.values()) {
            double d;
            if (parse_number(v, d)) add(k, d);
        }
        // The mesh is authoritative for the domain (e.g. one z-layer of an extruded case)
        add("Lx", m.Lx); add("Ly", m.Ly); add("Lz", m.Lz);
        add("nx", m.nx); add("ny", m.ny); add("nz", m.nz);
        for (size_t i = 0; i < names_.size(); ++i)
            vars_.push_back({names_[i].c_str(), &values_[i], 0, nullptr});   // 0 = TE_VARIABLE
    }

    // Compile once; numbers are kept exact (no parsing through the expression engine).
    int compile(const std::string& src, const std::string& what) {
        Item it;
        if (parse_number(src, it.constant)) { items_.push_back(std::move(it)); return static_cast<int>(items_.size()) - 1; }
        int err = 0;
        te_expr* e = te_compile(src.c_str(), vars_.data(), static_cast<int>(vars_.size()), &err);
        if (!e) {
            std::string msg = what + ": cannot evaluate \"" + src + "\" (error at character " + std::to_string(err) + ")";
            // Name a likely misspelt identifier at the error position
            size_t b = static_cast<size_t>(std::max(err - 1, 0));
            while (b > 0 && (std::isalnum(static_cast<unsigned char>(src[b - 1])) || src[b - 1] == '_')) --b;
            size_t e2 = b;
            while (e2 < src.size() && (std::isalnum(static_cast<unsigned char>(src[e2])) || src[e2] == '_')) ++e2;
            if (e2 > b && std::isalpha(static_cast<unsigned char>(src[b]))) {
                std::vector<std::string> known(names_.begin(), names_.end());
                for (const char* f : {"sin", "cos", "tan", "exp", "sqrt", "pow", "abs", "ln", "log", "atan2", "pi", "e"})
                    known.push_back(f);
                msg += hint(src.substr(b, e2 - b), known);
            }
            throw std::runtime_error(msg);
        }
        it.expr.reset(e);
        items_.push_back(std::move(it));
        return static_cast<int>(items_.size()) - 1;
    }

    void set_position(double x, double y, double z) { values_[0] = x; values_[1] = y; values_[2] = z; }

    double eval(int i) const { const Item& it = items_[static_cast<size_t>(i)]; return it.expr ? te_eval(it.expr.get()) : it.constant; }

private:
    struct Free { void operator()(te_expr* e) const { te_free(e); } };
    struct Item { double constant = 0.0; std::unique_ptr<te_expr, Free> expr; };
    std::vector<std::string> names_;
    std::vector<double> values_;
    std::vector<te_variable> vars_;
    std::vector<Item> items_;
};

int state_index(const std::string& var) {   // primitive layout [rho, u, v, p, w]
    if (var == "rho") return 0;
    if (var == "u") return 1;
    if (var == "v") return 2;
    if (var == "p") return 3;
    return 4;   // w
}

// Region geometry evaluated once (it may use the run parameters but not x, y, z)
struct Geometry {
    const InitialRegion* r = nullptr;
    double a[3] = {0, 0, 0}, b[3] = {0, 0, 0}, radius = 0.0;
    int nb = 0;   // number of given components of a / b
    std::vector<std::pair<int, int>> state;   // (primitive index, evaluator item)
};

void eval_vec(Evaluator& ev, const std::vector<std::string>& v, double* out, double fill, const std::string& what) {
    for (int d = 0; d < 3; ++d) out[d] = fill;
    for (size_t d = 0; d < v.size(); ++d) out[d] = ev.eval(ev.compile(v[d], what));
}

bool inside(const Geometry& g, const double c[3], bool three_d) {
    const InitialRegion& r = *g.r;
    const int dims = three_d ? 3 : 2;
    if (r.shape == "box") {
        for (int d = 0; d < dims; ++d)
            if (c[d] < g.a[d] || c[d] > g.b[d]) return false;
        return true;
    }
    if (r.shape == "halfspace") {
        double dot = 0.0;
        for (int d = 0; d < dims; ++d) dot += (c[d] - g.a[d]) * g.b[d];
        return dot >= 0.0;
    }
    if (r.shape == "sphere") {
        double s = 0.0;
        for (int d = 0; d < dims; ++d) s += (c[d] - g.a[d]) * (c[d] - g.a[d]);
        return s <= g.radius * g.radius;
    }
    // cylinder: distance from the axis through `center`
    const int ax = r.axis == "x" ? 0 : (r.axis == "y" ? 1 : 2);
    double s = 0.0;
    for (int d = 0; d < dims; ++d)
        if (d != ax) s += (c[d] - g.a[d]) * (c[d] - g.a[d]);
    return s <= g.radius * g.radius;
}

// ---------------------------------------------------------------------------
// TOML output
// ---------------------------------------------------------------------------

std::string toml_string(const std::string& s) {
    std::string out = "\"";
    for (char ch : s) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x", ch);
                    out += b;
                } else {
                    out += ch;
                }
        }
    }
    return out + "\"";
}

// A parameter string as a typed TOML value: bool, integer, float or string.
std::string toml_value(const std::string& v) {
    std::string low = v;
    std::transform(low.begin(), low.end(), low.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (low == "true" || low == "false") return low;
    if (!v.empty() && v.find_first_not_of("+-0123456789") == std::string::npos && v.find_first_of("0123456789") != std::string::npos)
        return v;
    double d;
    if (parse_number(v, d) && std::isfinite(d)) return format_double(d);
    return toml_string(v);
}

// Numbers stay numbers, anything else is an expression string.
std::string toml_number_or_expr(const std::string& v) {
    double d;
    return parse_number(v, d) && std::isfinite(d) ? format_double(d) : toml_string(v);
}

std::string toml_vector(const std::vector<std::string>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + toml_number_or_expr(v[i]);
    return s + "]";
}

std::string toml_state(const std::map<std::string, std::string>& st) {
    std::string s = "{ ";
    bool first = true;
    for (const std::string& k : kStateKeys) {
        auto it = st.find(k);
        if (it == st.end()) continue;
        s += (first ? "" : ", ") + k + " = " + toml_number_or_expr(it->second);
        first = false;
    }
    return s + " }";
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::string param_section(const std::string& key) {
    for (const auto& s : section_table())
        if (std::find(s.second.begin(), s.second.end(), key) != s.second.end()) return s.first;
    return "";
}

void check_parameter_keys(const Params& p, const std::string& where) {
    for (const auto& kv : p.values()) {
        if (kv.first == "test_case" || !param_section(kv.first).empty()) continue;
        throw std::invalid_argument(where + ": unknown parameter '" + kv.first + "'" + hint(kv.first, all_param_keys()));
    }
}

const std::vector<std::string>& param_sections() {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> n;
        for (const auto& s : section_table()) n.push_back(s.first);
        return n;
    }();
    return names;
}

std::string closest_match(const std::string& word, const std::vector<std::string>& candidates) {
    auto lower = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    };
    const std::string w = lower(word);
    std::string best;
    size_t best_d = 4;
    for (const std::string& c : candidates) {
        const std::string cl = lower(c);
        // Levenshtein distance
        std::vector<size_t> prev(cl.size() + 1), cur(cl.size() + 1);
        for (size_t j = 0; j <= cl.size(); ++j) prev[j] = j;
        for (size_t i = 1; i <= w.size(); ++i) {
            cur[0] = i;
            for (size_t j = 1; j <= cl.size(); ++j)
                cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (w[i - 1] == cl[j - 1] ? 0 : 1)});
            std::swap(prev, cur);
        }
        if (prev[cl.size()] < best_d) { best_d = prev[cl.size()]; best = c; }
    }
    return best;
}

CaseFile load_case_file(const std::string& path_in) {
    fs::path path(path_in);
    if (fs::is_directory(path)) path /= "case.toml";
    const std::string file = path.generic_string();
    if (!fs::exists(path)) throw std::runtime_error("Case file not found: " + file);

    toml::table tbl;
    try {
        tbl = toml::parse_file(file);
    } catch (const toml::parse_error& e) {
        throw std::runtime_error(file + ":" + std::to_string(e.source().begin.line) + ": " + std::string(e.description()));
    }

    CaseFile cf;
    cf.path = file;
    const std::vector<std::string> keys = all_param_keys();
    std::vector<std::string> top = param_sections();
    top.insert(top.begin(), "case");
    top.push_back("initial");

    for (const auto& [sk, sv] : tbl) {
        const std::string section(sk.str());
        const toml::table* st = sv.as_table();
        if (!st || std::find(top.begin(), top.end(), section) == top.end())
            fail(file, &sv, "unknown section [" + section + "]" + hint(section, top));

        if (section == "case") {
            for (const auto& [k, v] : *st) {
                const std::string key(k.str());
                if (key == "name") cf.name = scalar_string(file, v, "case.name");
                else if (key == "description") cf.description = scalar_string(file, v, "case.description");
                else if (key == "order") {
                    if (!v.is_integer()) fail(file, &v, "case.order must be an integer");
                    cf.order = static_cast<int>(v.as_integer()->get());
                } else if (key == "requires_viscous" || key == "requires_3d") {
                    if (!v.is_boolean()) fail(file, &v, "case." + key + " must be true or false");
                    (key == "requires_viscous" ? cf.requires_viscous : cf.requires_3d) = v.as_boolean()->get();
                } else {
                    fail(file, &v, "unknown key '" + key + "' in [case]" + hint(key, kCaseKeys));
                }
            }
            continue;
        }

        if (section == "initial") {
            for (const auto& [k, v] : *st) {
                const std::string key(k.str());
                if (key == "profile") cf.initial.profile = scalar_string(file, v, "initial.profile");
                else if (key == "state") cf.initial.state = state_of(file, v, "initial.state");
                else if (key == "region") {
                    const toml::array* arr = v.as_array();
                    if (!arr) fail(file, &v, "write regions as [[initial.region]] tables");
                    int i = 1;
                    for (const auto& r : *arr) cf.initial.regions.push_back(region_of(file, r, i++));
                } else {
                    fail(file, &v, "unknown key '" + key + "' in [initial]" +
                                       hint(key, {"profile", "state", "region"}));
                }
            }
            continue;
        }

        for (const auto& [k, v] : *st) {
            const std::string key(k.str());
            const std::string home = param_section(key);
            if (home.empty()) {
                if (key == "test_case") fail(file, &v, "'test_case' is the case itself; it cannot be set in a case file");
                fail(file, &v, "unknown parameter '" + key + "'" + hint(key, keys));
            }
            if (home != section) fail(file, &v, "'" + key + "' belongs in [" + home + "], not [" + section + "]");
            cf.params.set(key, scalar_string(file, v, key));
        }
    }

    if (cf.name.empty()) fail(file, nullptr, "missing [case] name");
    const InitialSpec& in = cf.initial;
    if (in.profile.empty()) {
        if (!in.state.count("rho") || !in.state.count("p"))
            fail(file, nullptr, "[initial] needs a background state with at least rho and p, "
                                "e.g. state = { rho = 1.0, u = 0.0, v = 0.0, p = 1.0 }, or a profile");
    } else if (!in.state.empty()) {
        fail(file, nullptr, "[initial] takes a profile or a background state, not both (use regions to modify a profile)");
    }

    // Check every expression now, so a typo is reported with the file rather than at run time
    Params vars = baseline_defaults();
    vars.merge(cf.params);
    Evaluator ev(vars, generate_cartesian_mesh(1.0, 1.0, 2, 2));
    try {
        for (const auto& [var, src] : in.state) ev.compile(src, "initial.state." + var);
        for (size_t i = 0; i < in.regions.size(); ++i) {
            const InitialRegion& r = in.regions[i];
            const std::string what = "region " + std::to_string(i + 1);
            for (const auto* vec : {&r.min, &r.max, &r.point, &r.normal, &r.center})
                for (const std::string& s : *vec) ev.compile(s, what + " geometry");
            if (!r.radius.empty()) ev.compile(r.radius, what + ".radius");
            for (const auto& [var, src] : r.state) ev.compile(src, what + ".state." + var);
        }
    } catch (const std::runtime_error& e) {
        fail(file, nullptr, e.what());
    }
    return cf;
}

void save_case_file(const CaseFile& cf, const std::string& path_in) {
    fs::path path(path_in);
    if (path.extension() != ".toml") path /= "case.toml";
    if (path.has_parent_path()) fs::create_directories(path.parent_path());

    std::ostringstream o;
    o << "# Fenice case file - see the \"Writing a test case\" page of the documentation.\n\n";
    o << "[case]\nname = " << toml_string(cf.name) << "\n";
    if (!cf.description.empty()) o << "description = " << toml_string(cf.description) << "\n";
    o << "order = " << cf.order << "\n";
    if (cf.requires_viscous) o << "requires_viscous = true\n";
    if (cf.requires_3d) o << "requires_3d = true\n";

    for (const auto& [section, skeys] : section_table()) {
        bool header = false;
        for (const std::string& k : skeys) {
            if (!cf.params.has(k)) continue;
            if (!header) { o << "\n[" << section << "]\n"; header = true; }
            o << k << " = " << toml_value(cf.params.str(k)) << "\n";
        }
    }

    o << "\n[initial]\n";
    if (!cf.initial.profile.empty()) o << "profile = " << toml_string(cf.initial.profile) << "\n";
    if (!cf.initial.state.empty()) o << "state = " << toml_state(cf.initial.state) << "\n";
    for (const InitialRegion& r : cf.initial.regions) {
        o << "\n[[initial.region]]\nshape = " << toml_string(r.shape) << "\n";
        if (!r.min.empty()) o << "min = " << toml_vector(r.min) << "\n";
        if (!r.max.empty()) o << "max = " << toml_vector(r.max) << "\n";
        if (!r.point.empty()) o << "point = " << toml_vector(r.point) << "\n";
        if (!r.normal.empty()) o << "normal = " << toml_vector(r.normal) << "\n";
        if (!r.center.empty()) o << "center = " << toml_vector(r.center) << "\n";
        if (!r.radius.empty()) o << "radius = " << toml_number_or_expr(r.radius) << "\n";
        if (r.shape == "cylinder" && r.axis != "z") o << "axis = " << toml_string(r.axis) << "\n";
        o << "state = " << toml_state(r.state) << "\n";
    }

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write " + path.generic_string());
    f << o.str();
}

Field evaluate_initial(const InitialSpec& spec, const Mesh& m, const EOS& eos, const Params& run, const Field* base) {
    Evaluator ev(run, m);

    std::vector<std::pair<int, int>> background;   // (primitive index, item)
    for (const auto& [var, src] : spec.state) background.push_back({state_index(var), ev.compile(src, "initial.state." + var)});

    std::vector<Geometry> geo(spec.regions.size());
    for (size_t i = 0; i < spec.regions.size(); ++i) {
        const InitialRegion& r = spec.regions[i];
        const std::string what = "region " + std::to_string(i + 1);
        Geometry& g = geo[i];
        g.r = &r;
        const double inf = std::numeric_limits<double>::infinity();
        if (r.shape == "box") {
            eval_vec(ev, r.min, g.a, -inf, what + ".min");
            eval_vec(ev, r.max, g.b, inf, what + ".max");
        } else if (r.shape == "halfspace") {
            eval_vec(ev, r.point, g.a, 0.0, what + ".point");
            eval_vec(ev, r.normal, g.b, 0.0, what + ".normal");
        } else {
            if (r.shape == "sphere" && m.is3d() && r.center.size() < 3)
                throw std::runtime_error(what + ": a sphere on a 3-D mesh needs a 3-component center");
            eval_vec(ev, r.center, g.a, 0.0, what + ".center");
            g.radius = ev.eval(ev.compile(r.radius, what + ".radius"));
        }
        for (const auto& [var, src] : r.state) g.state.push_back({state_index(var), ev.compile(src, what + ".state." + var)});
    }

    Field U(static_cast<size_t>(m.num_cells()));
    const bool three_d = m.is3d();
    for (int k = 0; k < m.nz; ++k)
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                const int cell = m.cell(i, j, k);
                const double c[3] = {m.xc(i), m.yc(j), three_d ? m.zc(k) : 0.0};
                ev.set_position(c[0], c[1], c[2]);
                bool hit = false;
                State W{};
                for (const Geometry& g : geo)
                    if (inside(g, c, three_d)) { hit = true; break; }
                if (base && !hit) {   // untouched profile cell: keep it bit for bit
                    U[cell] = (*base)[cell];
                    continue;
                }
                if (base) W = eos.cons_to_prim((*base)[cell]);
                for (const auto& [q, item] : background) W[q] = ev.eval(item);
                for (const Geometry& g : geo)
                    if (inside(g, c, three_d))
                        for (const auto& [q, item] : g.state) W[q] = ev.eval(item);
                U[cell] = eos.prim_to_cons(W);
            }
    return U;
}

}  // namespace fv
