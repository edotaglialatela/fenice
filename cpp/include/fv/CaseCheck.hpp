// Case check: what a resolved run says beyond what SimParams::from rejects (keys that restate the baseline, keys whose
// switch is off) and where each value comes from.
#pragma once

#include <string>
#include <vector>

#include "fv/Params.hpp"

namespace fv {

// The layers a run is resolved from, lowest first: baseline defaults, the case file, a --config file, --set overrides.
struct ParamLayers {
    Params baseline, case_file, config, overrides;
    bool requires_3d = false;        // the case file's [case] requires_3d
    bool requires_viscous = false;   // the case file's [case] requires_viscous
};

// Where the value of `key` in the resolved run comes from: "baseline", "case", "config" or "--set" ("" if no layer sets it).
std::string param_origin(const ParamLayers& layers, const std::string& key);

// One remark about a run.
struct CaseFinding {
    enum Level { Note, Unused, Warning, Error };
    Level level;
    std::string key;       // the parameter concerned
    std::string message;   // what is wrong and what to do about it
};

// Remarks on the run `layers` resolve to.  An error is a combination the run would stop on, or that cannot work as written; a warning
// is a setup that is probably not meant (an inlet at rest, a body across the domain edge); `Unused` is a key that has no effect (its
// switch is off, or another key overrides it); a note is a key that restates its baseline default or a resolution remark.
std::vector<CaseFinding> check_case(const ParamLayers& layers);

}   // namespace fv
