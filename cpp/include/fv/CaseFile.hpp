// Case files: a test case as a folder holding a case.toml (mesh, physics, numerics,
// boundaries, run control and initial conditions), read at run time, so adding a case
// needs no C++ and no recompilation.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "fv/Core.hpp"
#include "fv/Params.hpp"

namespace fv {

// One region of the initial condition: the cells whose centre lies inside take the
// region's `state` values (later regions override earlier ones).  Geometry entries are
// numbers or expressions of the run parameters (e.g. "Lx/2"); vectors have 2 or 3 entries.
struct InitialRegion {
    std::string shape;                   // "box", "halfspace", "sphere" or "cylinder"
    std::vector<std::string> min, max;   // box: inside where min <= centre <= max (missing z: unbounded)
    std::vector<std::string> point;      // halfspace: inside where (centre - point) . normal >= 0
    std::vector<std::string> normal;
    std::vector<std::string> center;            // sphere / cylinder centre
    std::string radius;                         // sphere / cylinder radius
    std::string axis = "z";                     // cylinder axis: "x", "y" or "z"
    std::map<std::string, std::string> state;   // primitive overrides: rho, u, v, w, p
};

// Initial condition: a built-in profile and/or a background state, then regions in order.
// State values are numbers or expressions of x, y, z (cell centre), pi, e and every
// numeric run parameter by name (Lx, mu, gamma, u_in, ...).
struct InitialSpec {
    std::string profile;                        // built-in C++ initialiser (see TestCases.hpp)
    std::map<std::string, std::string> state;   // background: rho, u, v, w, p
    std::vector<InitialRegion> regions;
};

// A parsed case file.
struct CaseFile {
    std::string name;   // registry key, e.g. "Sod Shock Tube"
    std::string description;
    std::string path;                // the case.toml it was read from
    int order = 1000;                // position in --list-cases and the GUI
    bool requires_viscous = false;   // warn when run without Navier-Stokes
    bool requires_3d = false;        // needs nz > 1 (otherwise a 2-D case, extruded on 3-D meshes)
    Params params;                   // parameter overrides, flattened from the sections
    InitialSpec initial;
};

// Read a case (a folder containing case.toml, or the .toml file itself).  Throws
// std::runtime_error naming the file and the problem: syntax error, unknown or misplaced
// key (with a "did you mean" hint), wrong value type, incomplete region.
CaseFile load_case_file(const std::string& path);

// Write `cf` to `path` (a folder gets a case.toml; parent folders are created), with the
// parameters grouped in their canonical sections.
void save_case_file(const CaseFile& cf, const std::string& path);

// Section of the case file a parameter belongs in ("mesh", "numerics", ...); "" if unknown.
std::string param_section(const std::string& key);

// Throw std::invalid_argument (prefixed with `where`, with a "did you mean" hint) if `p`
// holds a key that is neither a parameter nor test_case: catches --set / --config typos.
void check_parameter_keys(const Params& p, const std::string& where);

// Parameter sections of a case file, in canonical order.
const std::vector<std::string>& param_sections();

// Closest of `candidates` to `word` (edit distance <= 3), or "": for "did you mean" hints.
std::string closest_match(const std::string& word, const std::vector<std::string>& candidates);

// Initial field on `mesh` from `spec`: start from `base` (a profile's field) or from the
// background state, then apply the regions.  `run` supplies the expression variables.
Field evaluate_initial(const InitialSpec& spec, const Mesh& mesh, const EOS& eos, const Params& run,
                       const Field* base = nullptr);

}   // namespace fv
