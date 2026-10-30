// The environment switches of the program: every FENICE_* variable the C++ code reads is listed here, with its effect, and
// is read through switch_on / switch_value, which throw for a name that is not in the table. `fenice --print-switches`
// prints the table; the page "Environment switches" of the documentation is made from it.
#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace fv {

// One environment switch: its name, what kind of value it takes ("flag": on when set to anything but "" or "0"; "path": a
// folder or file), who reads it, and what it does.
struct EnvSwitch {
    const char* name;
    const char* kind;
    const char* reader;   // "solver" (read by the C++ code) or "python" (read by the Python package, listed here for completeness)
    const char* effect;
};

// All the switches, in the order they are documented.
const std::vector<EnvSwitch>& env_switches();

// Whether the flag `name` is on. Throws std::logic_error if `name` is not a flag of the table.
bool switch_on(const char* name);

// The value of the path switch `name`, or "" if it is unset. Throws std::logic_error if `name` is not in the table.
std::string switch_value(const char* name);

// Print the table to `out`, one switch per paragraph (what `fenice --print-switches` shows).
void print_env_switches(std::FILE* out);

}   // namespace fv
