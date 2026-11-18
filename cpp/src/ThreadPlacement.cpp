#include "fv/ThreadPlacement.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>

#include "fv/Switches.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>

#include <fstream>
#include <sstream>
#endif

namespace fv {

std::vector<int> choose_thread_cpus(const std::vector<CoreInfo>& cores, const std::vector<int>& allowed, int threads) {
    if (threads <= 1) return {};
    struct Usable {
        int efficiency;
        std::vector<int> cpus;
    };
    std::vector<Usable> usable;
    bool any_smt = false;
    for (const CoreInfo& c : cores) {
        Usable u{c.efficiency, {}};
        for (int cpu : c.cpus)
            if (std::find(allowed.begin(), allowed.end(), cpu) != allowed.end()) u.cpus.push_back(cpu);
        if (u.cpus.empty()) continue;
        if (u.cpus.size() > 1) any_smt = true;
        usable.push_back(std::move(u));
    }
    if (!any_smt || threads > static_cast<int>(usable.size())) return {};
    std::stable_sort(usable.begin(), usable.end(), [](const Usable& a, const Usable& b) { return a.efficiency > b.efficiency; });
    std::vector<int> out;
    for (int t = 0; t < threads; ++t) out.push_back(*std::min_element(usable[t].cpus.begin(), usable[t].cpus.end()));
    return out;
}

namespace {

[[maybe_unused]] bool env_set(const char* name) {
    const char* v = std::getenv(name);
    return v && *v;
}

#ifdef _WIN32

// The cores and the allowed CPUs of the process, in the processor group 0. False (do not pin) with more than one group (more than 64
// logical CPUs): a thread would need a group as well.
bool read_topology(std::vector<CoreInfo>& cores, std::vector<int>& allowed) {
    if (GetActiveProcessorGroupCount() != 1) return false;
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return false;
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
                                          &len))
        return false;
    for (DWORD off = 0; off < len;) {
        const auto* p = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + off);
        if (p->Relationship == RelationProcessorCore) {
            CoreInfo c;
            c.efficiency = p->Processor.EfficiencyClass;
            for (int b = 0; b < static_cast<int>(8 * sizeof(KAFFINITY)); ++b)
                if (p->Processor.GroupMask[0].Mask & (static_cast<KAFFINITY>(1) << b)) c.cpus.push_back(b);
            cores.push_back(std::move(c));
        }
        off += p->Size;
    }
    DWORD_PTR process = 0, system = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process, &system)) return false;
    for (int b = 0; b < static_cast<int>(8 * sizeof(DWORD_PTR)); ++b)
        if (process & (static_cast<DWORD_PTR>(1) << b)) allowed.push_back(b);
    return !cores.empty() && !allowed.empty();
}

bool pin_this_thread(int cpu) { return SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << cpu) != 0; }

#elif defined(__linux__)

// "0,24" or "0-1,8" -> the CPU numbers
std::vector<int> parse_cpu_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) {
        const size_t dash = part.find('-');
        if (dash == std::string::npos) {
            out.push_back(std::atoi(part.c_str()));
        } else {
            const int a = std::atoi(part.substr(0, dash).c_str()), b = std::atoi(part.substr(dash + 1).c_str());
            for (int i = a; i <= b; ++i) out.push_back(i);
        }
    }
    return out;
}

bool read_topology(std::vector<CoreInfo>& cores, std::vector<int>& allowed) {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return false;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
        if (CPU_ISSET(cpu, &set)) allowed.push_back(cpu);
    std::map<std::string, CoreInfo> by_siblings;   // the sibling list names the core
    std::vector<std::string> order;
    for (int cpu : allowed) {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        std::ifstream f(base + "/topology/thread_siblings_list");
        std::string list;
        if (!f || !std::getline(f, list) || list.empty()) return false;
        if (!by_siblings.count(list)) {
            CoreInfo c;
            c.cpus = parse_cpu_list(list);
            std::ifstream cap(base + "/cpu_capacity");   // present on some machines with fast and slow cores
            int v = 0;
            if (cap >> v) c.efficiency = v;
            by_siblings[list] = std::move(c);
            order.push_back(list);
        }
    }
    for (const std::string& k : order) cores.push_back(by_siblings[k]);
    return !cores.empty() && !allowed.empty();
}

bool pin_this_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;   // pid 0: the calling thread
}

#else

bool read_topology(std::vector<CoreInfo>&, std::vector<int>&) { return false; }
bool pin_this_thread(int) { return false; }

#endif

}   // namespace

Placement place_threads(int threads) {
    Placement r;
#ifdef _OPENMP
    if (threads <= 1 || switch_on("FENICE_NO_PIN")) return r;
    for (const char* name : {"OMP_PLACES", "OMP_PROC_BIND", "GOMP_CPU_AFFINITY", "KMP_AFFINITY"})
        if (env_set(name)) return r;
    std::vector<CoreInfo> cores;
    std::vector<int> allowed;
    if (!read_topology(cores, allowed)) return r;
    const std::vector<int> cpus = choose_thread_cpus(cores, allowed, threads);
    if (cpus.empty()) return r;
    std::vector<char> ok(cpus.size(), 0);
    bool team_ok = true;
#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num();
        if (omp_get_num_threads() != threads) {
#pragma omp critical(fv_place_threads)
            team_ok = false;
        } else {
            ok[static_cast<size_t>(t)] = pin_this_thread(cpus[static_cast<size_t>(t)]) ? 1 : 0;
        }
    }
    if (!team_ok || std::find(ok.begin(), ok.end(), 0) != ok.end()) {
        r.note = "[PARALLEL] the threads could not all be placed on distinct cores (left to the scheduler)";
        return r;
    }
    r.pinned = true;
    r.note = "[PARALLEL] " + std::to_string(threads) +
             " threads placed on distinct physical cores (FENICE_NO_PIN=1 leaves them to the scheduler)";
#else
    (void)threads;
#endif
    return r;
}

}   // namespace fv
