#include "fv/Switches.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace fv {

const std::vector<EnvSwitch>& env_switches() {
    static const std::vector<EnvSwitch> table = {
        {"FENICE_NO_SIMD", "flag", "solver",
         "Run the scalar Riemann and viscous kernels instead of the AVX2 ones. The results are identical bit for bit; use it to time "
         "the vector kernels or to check the equivalence."},
        {"FENICE_FULL_REBUILD", "flag", "solver",
         "AMR: rebuild the faces and the engine tables of every leaf after each adaptation instead of copying those of the leaves "
         "that did not change. The tables are identical; for A/B timing and bisecting."},
        {"FENICE_REPLICATED_REMESH", "flag", "solver",
         "AMR on MPI: every rank keeps the whole AMR mesh instead of its share of the leaves (the storage before the distributed "
         "one; for A/B comparison)."},
        {"FENICE_REPLICATED_TOPOLOGY", "flag", "solver",
         "AMR on MPI: every rank holds and adapts the topology of the whole mesh at a remesh. Faster than the default, in which no "
         "rank does, and it uses more memory per rank; the results are identical bit for bit."},
        {"FENICE_MEM_TRACE", "flag", "solver",
         "Print the resident and the peak memory of the process at the steps of the setup (standard error)."},
        {"FENICE_PROFILE", "flag", "solver",
         "AMR: print, at the end of the run, the wall time of the time step, the output and each part of the remesh (min / max over "
         "the ranks). Costs nothing when off."},
        {"FENICE_HALO_REACH", "flag", "solver",
         "AMR: print the halo reach of the mesh (how many rings of leaves the face lists read across a rank boundary). Needs the "
         "whole mesh: a serial run, or FENICE_REPLICATED_REMESH=1 on MPI."},
        {"FENICE_DEBUG_TABLES", "flag", "solver",
         "AMR: when the engine tables of an incremental and of a full build are compared and differ, print which table differs "
         "(standard error)."},
        {"FENICE_CASES", "path", "solver",
         "The folder that holds the test cases (one folder per case with a case.toml). Default: the cases folder of the repository."},
        {"FENICE_EXE", "path", "python",
         "The solver executable that fenice.exe() and fenice.run() use. Default: build/release/fenice."},
    };
    return table;
}

namespace {

const EnvSwitch& find(const char* name, const char* kind) {
    for (const EnvSwitch& s : env_switches())
        if (std::strcmp(s.name, name) == 0) {
            if (std::strcmp(s.kind, kind) != 0)
                throw std::logic_error(std::string("environment switch ") + name + " is a " + s.kind + ", not a " + kind);
            return s;
        }
    throw std::logic_error(std::string("environment switch ") + name + " is not in the table of fv::env_switches()");
}

}   // namespace

bool switch_on(const char* name) {
    find(name, "flag");
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && std::strcmp(v, "0") != 0;
}

std::string switch_value(const char* name) {
    find(name, "path");
    const char* v = std::getenv(name);
    return v ? v : "";
}

void print_env_switches(std::FILE* out) {
    std::fprintf(out, "Environment switches (a flag is on when set to anything but empty or 0):\n");
    for (const EnvSwitch& s : env_switches())
        std::fprintf(out, "\n  %s  (%s, read by the %s)\n      %s\n", s.name, s.kind, s.reader, s.effect);
}

}   // namespace fv
