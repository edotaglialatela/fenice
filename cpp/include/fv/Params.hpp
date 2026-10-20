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
    double      num(const std::string& key, double def = 0.0) const;
    // Integer value ("100" or "100.0").
    int         integer(const std::string& key, int def = 0) const;
    // Boolean value: true/false, 1/0, yes/no, on/off (case-insensitive).
    bool        flag(const std::string& key, bool def = false) const;

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
    double Lx, Ly, Lz;          // Lz: 3-D only
    int    nx, ny, nz;          // nz = 1: 2-D run
    std::string eos_type;
    double gamma, p_inf;
    std::string bc_left, bc_right, bc_bottom, bc_top;
    std::string bc_front, bc_back;   // z-min / z-max sides (3-D)
    std::string riemann_solver, time_scheme;
    double cfl, t_end;
    int    save_interval;
    bool   use_viscous;
    double mu, pr;
    double u_in, rho_in, p_out;
    bool   use_bodyforce;
    double gx, gy, gz;
    bool   resume;
    bool   use_mpi;
    int    n_cpus;
    bool   use_ibm;
    std::string ibm_shape;
    double ibm_radius, ibm_cx, ibm_cy, ibm_cz;
    bool   use_muscl;
    std::string muscl_limiter;
    double dt_phys;
    int    max_inner_iter;
    double inner_tol_rel;
    bool   use_hot_start;
    std::string hot_start_file;
    int    hot_start_src_nx, hot_start_src_ny;
    bool   use_amr;
    int    amr_max_level, amr_patch_nx, amr_patch_ny;
    double amr_refine_threshold, amr_derefine_threshold;
    int    amr_refine_every, amr_init_levels, amr_buffer_layers;

    // C++-only extras
    std::string output_root;   // parent of the per-case results folder
    std::string dump_state;    // optional .npz with the final state (regression tests)
    int    max_steps;          // stop after this many steps (<= 0: unlimited)
    int    monitor_interval;   // write volume averages to <case>_monitor.csv every N steps (0: off)
    Params source;             // the key/value parameters this view was built from (case-file expressions)

    // Parse and validate (positive mesh sizes and domain lengths).
    static SimParams from(const Params& p);
};

}  // namespace fv
