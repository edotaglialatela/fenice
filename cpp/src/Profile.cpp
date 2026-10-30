#include "fv/Profile.hpp"

#include <algorithm>
#include <array>

#include "fv/Switches.hpp"

namespace fv {

namespace {

constexpr size_t kN = static_cast<size_t>(Section::kCount);

// thread_local: the tests emulate ranks with threads, and a rank is one thread (all sections are timed outside OpenMP regions)
struct Totals {
    std::array<double, kN> seconds{};
    std::array<long long, kN> calls{};
};
thread_local Totals g_totals;

}   // namespace

const char* section_name(Section s) {
    static const char* const names[kN] = {"time step",         "output",           "remesh",          "remesh: region",
                                          "remesh: flags",     "remesh: topology", "remesh: migrate", "remesh: rebuild",
                                          "remesh: partition", "fetch_leaves",     "share: region",   "share: mesh",
                                          "share: engine",     "share: partition", "share: localize"};
    return names[static_cast<size_t>(s)];
}

bool profile_enabled() {
    static const bool on = switch_on("FENICE_PROFILE");
    return on;
}

void profile_add(Section s, double seconds) {
    if (!profile_enabled()) return;
    g_totals.seconds[static_cast<size_t>(s)] += seconds;
    ++g_totals.calls[static_cast<size_t>(s)];
}

std::vector<double> profile_values() {
    if (!profile_enabled()) return {};
    std::vector<double> v(2 * kN);
    for (size_t k = 0; k < kN; ++k) {
        v[k] = g_totals.seconds[k];
        v[kN + k] = static_cast<double>(g_totals.calls[k]);
    }
    return v;
}

void print_profile(std::FILE* out, const std::vector<double>& all, int nranks) {
    if (all.size() != 2 * kN * static_cast<size_t>(nranks)) return;
    std::fprintf(out, "[PROFILE] wall time by section, min / max over %d rank%s (a nested section is counted in the one around it)\n",
                 nranks, nranks > 1 ? "s" : "");
    for (size_t k = 0; k < kN; ++k) {
        double lo = all[k], hi = all[k];
        for (int r = 1; r < nranks; ++r) {
            lo = std::min(lo, all[2 * kN * static_cast<size_t>(r) + k]);
            hi = std::max(hi, all[2 * kN * static_cast<size_t>(r) + k]);
        }
        const long long calls = static_cast<long long>(all[kN + k]);   // rank 0
        if (calls == 0 && hi == 0.0) continue;
        std::fprintf(out, "[PROFILE]   %-18s %6lld calls  %8.3f / %8.3f s\n", section_name(static_cast<Section>(k)), calls, lo, hi);
    }
}

void profile_reset() {
    g_totals = Totals();
}

}   // namespace fv
