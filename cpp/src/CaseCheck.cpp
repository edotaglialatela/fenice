// Case check: keys that restate the baseline, keys without effect, and the origin of each value.
#include "fv/CaseCheck.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <map>
#include <set>

namespace fv {

namespace {

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// True if the two values are the same number, or the same text ignoring case.
bool same_value(const std::string& a, const std::string& b) {
    try {
        size_t ua = 0, ub = 0;
        const double x = std::stod(a, &ua), y = std::stod(b, &ub);
        if (ua == a.size() && ub == b.size()) return x == y;
    } catch (const std::exception&) {}
    return lower(a) == lower(b);
}

// A group of keys that only act while a condition on the resolved run holds.
struct Dependency {
    std::vector<std::string> keys;
    std::function<bool(const Params&)> active;
    std::string reason;   // why the keys do nothing when `active` is false
};

const std::vector<Dependency>& dependencies() {
    static const std::vector<Dependency> d = {
        {{"mu", "pr", "reynolds"}, [](const Params& p) { return p.flag("use_viscous"); }, "use_viscous is false"},
        {{"gx", "gy", "gz"}, [](const Params& p) { return p.flag("use_bodyforce"); }, "use_bodyforce is false"},
        {{"gz"}, [](const Params& p) { return p.integer("nz") > 1; }, "the run is 2-D (nz = 1)"},
        {{"Lz", "bc_zmin", "bc_zmax", "ibm_cz"}, [](const Params& p) { return p.integer("nz") > 1; }, "the run is 2-D (nz = 1)"},
        {{"ibm_shape", "ibm_radius", "ibm_cx", "ibm_cy", "ibm_cz", "ibm_file", "ibm_chord", "ibm_aoa", "ibm_curvature", "ibm_vx", "ibm_vy",
          "ibm_omega"},
         [](const Params& p) { return p.flag("use_ibm"); }, "use_ibm is false"},
        {{"ibm_file", "ibm_chord", "ibm_aoa"},
         [](const Params& p) {
             const std::string s = lower(p.str("ibm_shape"));
             return s == "airfoil" || s.rfind("naca", 0) == 0;
         },
         "ibm_shape is not an airfoil"},
        {{"amr_max_level", "amr_indicator", "amr_criterion", "amr_refine_threshold", "amr_derefine_threshold", "amr_diffusion",
          "amr_refine_every", "amr_init_levels", "amr_force_box"},
         [](const Params& p) { return p.flag("use_amr"); }, "use_amr is false"},
        {{"hot_start_file", "hot_start_src_nx", "hot_start_src_ny"}, [](const Params& p) { return p.flag("use_hot_start"); },
         "use_hot_start is false"},
        {{"n_cpus"}, [](const Params& p) { return p.flag("use_mpi"); }, "use_mpi is false (threads come from --threads)"},
        {{"muscl_limiter"}, [](const Params& p) { return lower(p.str("reconstruction")) == "muscl"; },
         "reconstruction is not muscl"},
        {{"weno_epsilon"},
         [](const Params& p) {
             const std::string r = lower(p.str("reconstruction"));
             return r == "weno3" || r == "weno5";
         },
         "reconstruction is not weno3 or weno5"},
        {{"boundary_order"}, [](const Params& p) { return lower(p.str("reconstruction")) != "none"; },
         "reconstruction is none (first order: boundary faces always keep the ghost states)"},
    };
    return d;
}

// A numeric parameter, or NaN when it is an expression (or unset).
double number_or_nan(const Params& p, const std::string& key) {
    try {
        return p.has(key) ? p.num(key) : std::nan("");
    } catch (const std::exception&) { return std::nan(""); }
}

}   // namespace

std::string param_origin(const ParamLayers& layers, const std::string& key) {
    if (layers.overrides.has(key)) return "--set";
    if (layers.config.has(key)) return "config";
    if (layers.case_file.has(key)) return "case";
    if (layers.baseline.has(key)) return "baseline";
    return "";
}

std::vector<CaseFinding> check_case(const ParamLayers& layers) {
    Params run = layers.baseline;
    run.merge(layers.case_file);
    run.merge(layers.config);
    run.merge(layers.overrides);

    std::vector<CaseFinding> out;
    std::map<std::string, bool> flagged;   // keys already given a warning

    if (layers.requires_3d && run.integer("nz") <= 1)
        out.push_back({CaseFinding::Error, "nz", "the case needs a 3-D mesh: set nz > 1 (and Lz)"});
    if (layers.requires_viscous && !run.flag("use_viscous"))
        out.push_back({CaseFinding::Warning, "use_viscous", "the case is meant for Navier-Stokes: use_viscous is false"});
    if (run.num("cfl") > 1.0)
        out.push_back({CaseFinding::Warning, "cfl", "cfl = " + run.str("cfl") + " is above 1: the explicit update is unstable"});

    // an inlet with nothing flowing through it
    {
        const bool three_d = run.integer("nz") > 1;
        bool inlet = false;
        for (const char* side : {"bc_xmin", "bc_xmax", "bc_ymin", "bc_ymax", "bc_zmin", "bc_zmax"})
            if (lower(run.str(side)) == "vel_inlet" && (three_d || (std::string(side) != "bc_zmin" && std::string(side) != "bc_zmax")))
                inlet = true;
        const bool tabled = !run.str("u_in_table").empty() || !run.str("v_in_table").empty();
        const double u = number_or_nan(run, "u_in"), v = number_or_nan(run, "v_in");
        if (inlet && !tabled && u == 0.0 && v == 0.0)
            out.push_back({CaseFinding::Warning, "u_in", "a vel_inlet boundary is set but u_in and v_in are both 0: nothing flows in"});
    }

    // an immersed body: inside the domain, resolved
    if (run.flag("use_ibm")) {
        const double Lx = run.num("Lx"), Ly = run.num("Ly"), Lz = run.num("Lz");
        const bool three_d = run.integer("nz") > 1;
        const double cx = number_or_nan(run, "ibm_cx"), cy = number_or_nan(run, "ibm_cy"), cz = number_or_nan(run, "ibm_cz");
        const std::string shape = lower(run.str("ibm_shape"));
        const bool round_body = shape == "cylinder" || shape == "sphere";
        const double r = round_body ? number_or_nan(run, "ibm_radius") : 0.0;
        auto outside = [&](double c, double L) { return c < 0.0 || c > L; };
        auto across = [&](double c, double L) { return c - r < 0.0 || c + r > L; };
        if (outside(cx, Lx) || outside(cy, Ly) || (three_d && outside(cz, Lz)))
            out.push_back({CaseFinding::Warning, "ibm_cx", "the immersed body is centred outside the domain (ibm_cx, ibm_cy, ibm_cz)"});
        else if (round_body && (across(cx, Lx) || across(cy, Ly) || (three_d && across(cz, Lz))))
            out.push_back({CaseFinding::Warning, "ibm_radius", "the immersed body crosses the domain edge: shrink ibm_radius or move the centre"});
        if (round_body && r > 0.0) {
            double h = Lx / run.integer("nx");   // cell size at the finest level around the body
            if (run.flag("use_amr")) h /= std::pow(2.0, run.integer("amr_max_level"));
            const double per_diameter = 2.0 * r / h;
            if (per_diameter < 10.0)
                out.push_back({CaseFinding::Note, "ibm_radius",
                               "the body spans " + std::to_string(static_cast<int>(per_diameter)) +
                                   " cells per diameter at the finest level; the validation cases use 20 or more"});
        }
    }

    // reynolds sets mu: an explicit mu next to it is overridden
    if (run.num("reynolds", 0.0) > 0.0 && run.flag("use_viscous") && param_origin(layers, "mu") != "baseline") {
        out.push_back({CaseFinding::Unused, "mu", "mu is set, but reynolds > 0 computes mu from it: remove one of the two"});
        flagged["mu"] = true;
    }

    for (const Dependency& dep : dependencies()) {
        if (dep.active(run)) continue;
        for (const std::string& k : dep.keys) {
            const std::string origin = param_origin(layers, k);
            if (origin == "baseline" || origin.empty() || flagged[k]) continue;
            out.push_back({CaseFinding::Unused, k, k + " is set (" + origin + ") but has no effect: " + dep.reason});
            flagged[k] = true;
        }
    }

    for (const Params* layer : {&layers.case_file, &layers.config, &layers.overrides})
        for (const auto& kv : layer->values()) {
            const std::string& k = kv.first;
            if (k == "test_case" || k == "output_root" || flagged[k] || !layers.baseline.has(k)) continue;
            if (param_origin(layers, k) == "baseline") continue;
            if (!same_value(run.str(k), layers.baseline.str(k))) continue;
            // a key set by a lower layer and changed by a higher one is not a restatement
            out.push_back({CaseFinding::Note, k, k + " = " + run.str(k) + " is the baseline default: it can be left out"});
            flagged[k] = true;
        }
    return out;
}

}   // namespace fv
