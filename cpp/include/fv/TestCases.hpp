// Test cases: the case files under cases/ (one folder per case, holding case.toml), plus
// the built-in initial-condition profiles that case files can refer to.
#pragma once

#include <functional>

#include "fv/CaseFile.hpp"
#include "fv/Params.hpp"

namespace fv {

// A test case, as loaded from its case file.
struct TestCase {
    std::string key;                 // registry key ([case] name), e.g. "Sod Shock Tube"
    std::string description;
    std::string source;              // the case.toml it came from
    bool        requires_viscosity = false;   // warn when run without Navier-Stokes
    bool        three_d = false;     // initialize() expects a 3-D mesh (otherwise 2-D, extruded along z)
    Params      defaults;            // case parameters (override baseline_defaults)
    InitialSpec initial;             // as written in the case file
    // Initial conservative field on `mesh`; `run` holds the resolved run parameters.
    std::function<Field(const Mesh& mesh, const EOS& eos, const Params& run)> initialize;
};

// Folder holding the case folders: $FENICE_CASES if set, otherwise <repository>/cases.
std::string cases_directory();

// All cases of cases_directory(), sorted by [case] order, then name (the GUI order).
// A case file that fails to load is reported on stderr and skipped.
const std::vector<TestCase>& test_registry();

// A case by [case] name, by folder name, or by path to a case folder / case.toml.
// Throws std::invalid_argument for an unknown case (with a "did you mean" hint) and
// std::runtime_error for a case file that does not load.
const TestCase& get_test_case(const std::string& key);

// Folder name for a case name: lower case, other characters as '_' ("Sod Shock Tube" ->
// "sod_shock_tube").
std::string case_folder_name(const std::string& name);

// Save the resolved run `run` as a new case.  `target` is a case name (written to
// cases_directory()/case_folder_name(target)) or a path; `name` is the [case] name ("" ->
// derived from `target`).  The initial conditions are copied from `source`.  Keeps the
// parameters the source case sets plus those that differ from the baseline defaults.
// Refuses to overwrite an existing case.  Returns the written case.toml.
std::string save_case(const TestCase& source, const Params& run, const std::string& target, const std::string& name);

// Built-in initial-condition profile, for [initial] profile = "NAME" in a case file.
using Profile = std::function<Field(const Mesh& mesh, const EOS& eos, const Params& run)>;
// The profile called `name`, or nullptr.
const Profile* find_profile(const std::string& name);
// Names of all built-in profiles.
std::vector<std::string> profile_names();

// Exact Taylor-Green vortex [rho, u, v, p] on the periodic square [0, L]^2 (k = 2 pi / L,
// U0 = 1, rho0 = 1, p0 = 100), decaying as exp(-2 nu k^2 t) with nu = mu / rho0.
State taylor_green_exact(double x, double y, double t, double L, double mu);

// Initial 3-D Taylor-Green vortex [rho, u, v, p, w] on [0, 2 pi]^3 (U0 = 1, rho0 = 1, Ma = 0.1):
// u = sin x cos y cos z, v = -cos x sin y cos z, w = 0.
State taylor_green_3d_initial(double x, double y, double z, double gamma);

// Exact isentropic vortex [rho, u, v, p] at time t (free stream (1, 1), strength beta = 5,
// centred at (Lx/2, Ly/2) at t = 0, periodic images handled by minimum-image distance).
State isentropic_vortex_exact(double x, double y, double t, double Lx, double Ly, double gamma);

}  // namespace fv
