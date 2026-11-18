// Placing the OpenMP threads on distinct physical cores. A machine with SMT runs two logical CPUs on one core; two solver threads on the
// siblings of one core compete for the same vector units (division, square root) while other cores stay idle, which costs about 10 % of a
// time step at 8 threads (the flux and viscous passes scale 4.7x instead of 5.5x). The results never depend on the placement.
//
// The cores come from the topology the operating system reports at run time (Windows: GetLogicalProcessorInformationEx, Linux: the
// sibling lists in sysfs); nothing names a processor. Nothing is done when the process is already restricted to one logical CPU per core,
// when the threads outnumber the allowed cores, when the machine has no SMT, under MPI (the ranks of a node would overlap), when
// OMP_PLACES / OMP_PROC_BIND / GOMP_CPU_AFFINITY / KMP_AFFINITY is set, or with FENICE_NO_PIN=1.
#pragma once

#include <string>
#include <vector>

namespace fv {

// One physical core: its logical CPUs (the SMT siblings) and its efficiency class (on a machine with fast and slow cores the fast ones
// have the higher class; otherwise all are 0).
struct CoreInfo {
    std::vector<int> cpus;
    int efficiency = 0;
};

// The logical CPU for each of `threads` threads: one per physical core, the cores of the highest efficiency class first and then in the
// order given, using only the CPUs in `allowed`. Empty (do not pin) when threads <= 1, when the threads outnumber the cores that have an
// allowed CPU, when no allowed core has more than one allowed CPU (nothing to gain), or when `cores` is empty.
std::vector<int> choose_thread_cpus(const std::vector<CoreInfo>& cores, const std::vector<int>& allowed, int threads);

// What place_threads did: whether the threads are pinned and one line for the console (empty when nothing was done).
struct Placement {
    bool pinned = false;
    std::string note;
};

// Pin the first `threads` OpenMP threads of the process (the calling thread is thread 0) to distinct physical cores when that helps
// (see above). Call once, after the number of threads is set and before the first parallel region of the solver.
Placement place_threads(int threads);

}   // namespace fv
