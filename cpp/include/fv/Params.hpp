// Simulation parameters: a string key/value store (config file + CLI
// overrides) and its typed view.
#pragma once

#include <map>
#include <string>

namespace fv {

// String key/value store: parameters as read from the config file and the command line.
class Params {
public:
    // Set one parameter.
    void set(const std::string& key, const std::string& value) { values_[key] = value; }
    // Copy every entry of `other`, overriding existing keys.
    void merge(const Params& other);
    // True if `key` is set.
    bool has(const std::string& key) const { return values_.count(key) > 0; }

    // String value, or `def` when unset.
    std::string str(const std::string& key, const std::string& def = "") const;
    // Numeric value; throws if the value is not a number.
    double num(const std::string& key, double def = 0.0) const;
    // Integer value ("100" or "100.0").
    int integer(const std::string& key, int def = 0) const;
    // Boolean value: true/false, 1/0, yes/no, on/off (case-insensitive).
    bool flag(const std::string& key, bool def = false) const;

    // "key = value" lines; '#' starts a comment.  Throws on unreadable files.
    void load_file(const std::string& path);
    // "key=value" (CLI --set form).
    void set_assignment(const std::string& kv);

    // Every entry, sorted by key.
    const std::map<std::string, std::string>& values() const { return values_; }

private:
    std::map<std::string, std::string> values_;
};

// Baseline defaults, applied before the test-case defaults.
Params baseline_defaults();

// Typed view of the parameters.  Field names are the config keys; see the
// parameter reference above for meanings and defaults.
struct SimParams {
    std::string test_case;
    double Lx, Ly, Lz;   // Lz: 3-D only
    int nx, ny, nz;      // nz = 1: 2-D run
    std::string eos_type;
    double gamma, p_inf;
    std::string bc_left, bc_right, bc_bottom, bc_top;
    std::string bc_front, bc_back;   // z-min / z-max sides (3-D)
    std::string riemann_solver, time_scheme;
    double cfl, t_end;
    int save_interval;
    bool use_viscous;
    double mu, pr;
    double u_in, rho_in, p_out;
    double nrbc_sigma;   // non-reflecting outlet: relaxation of p towards p_out (0: none)
    double nrbc_beta;    // NRBC transverse weight in [0, 1] (1: LODI only); < 0: local Mach number
    bool use_bodyforce;
    double gx, gy, gz;
    bool use_source;   // any of source_rho / source_mx / source_my / source_mz / source_E is not 0 (see SimParams::source)
    bool source_unsteady;   // a source expression uses the time `t`: it is evaluated at every stage time
    bool resume;
    bool use_mpi;
    int n_cpus;
    bool use_ibm;
    std::string ibm_shape;                       // Cylinder, Sphere, Airfoil (ibm_file) or NACA dddd
    double ibm_radius, ibm_cx, ibm_cy, ibm_cz;   // (cx, cy): centre, or an airfoil's quarter chord
    std::string ibm_file;        // Airfoil: Selig / Lednicer coordinate file (relative: to the case folder)
    double ibm_chord, ibm_aoa;   // airfoil chord and angle of attack [deg, nose up]
    double reynolds;             // > 0: sets mu = rho_in u_in L / reynolds (L: chord or diameter)
    bool use_muscl;
    std::string muscl_limiter;
    double dt_phys;
    int max_inner_iter;
    double inner_tol_rel;
    bool use_hot_start;
    std::string hot_start_file;
    int hot_start_src_nx, hot_start_src_ny;
    bool use_amr;
    int amr_max_level, amr_patch_nx, amr_patch_ny;
    double amr_refine_threshold, amr_derefine_threshold;
    int amr_refine_every, amr_init_levels, amr_buffer_layers;
    bool amr_refine_ibm = true;              // keep the immersed body at the finest level
    std::string amr_indicator = "density";   // fields of the refinement indicator: density, velocity, density+velocity

    // C++-only extras
    std::string output_root;   // parent of the per-case results folder
    std::string dump_state;    // optional .npz with the final state (regression tests)
    int max_steps;             // stop after this many steps (<= 0: unlimited)
    int monitor_interval;      // write volume averages to <case>_monitor.csv every N steps (0: off)
    std::string vtk_format;    // "binary" (inline base64, default) or "ascii" VTK data arrays
    Params source;             // the key/value parameters this view was built from (case-file expressions)

    // Parse and validate (positive mesh sizes and domain lengths).
    static SimParams from(const Params& p);
};

}   // namespace fv
