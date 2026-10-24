#include "fv/Core.hpp"

#include <cstdio>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#elif !defined(__linux__)
#include <sys/resource.h>
#endif

namespace fv {

Mesh generate_cartesian_mesh(double Lx, double Ly, int nx, int ny, double x0, double y0) {
    Mesh m;
    m.nx = nx;
    m.ny = ny;
    m.Lx = Lx;
    m.Ly = Ly;
    m.dx = Lx / nx;
    m.dy = Ly / ny;
    m.x0 = x0;
    m.y0 = y0;
    return m;
}

Mesh generate_cartesian_mesh_3d(double Lx, double Ly, double Lz, int nx, int ny, int nz, double x0, double y0,
                                double z0) {
    Mesh m = generate_cartesian_mesh(Lx, Ly, nx, ny, x0, y0);
    m.nz = nz;
    m.Lz = Lz;
    m.dz = Lz / nz;
    m.z0 = z0;
    return m;
}

void cons_to_prim(const EOS& eos, const Field& U, Field& W) {
    const int n = static_cast<int>(U.size());
    W.resize(U.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c) W[c] = eos.cons_to_prim(U[c]);
}

void prim_to_cons(const EOS& eos, const Field& W, Field& U) {
    const int n = static_cast<int>(W.size());
    U.resize(W.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int c = 0; c < n; ++c) U[c] = eos.prim_to_cons(W[c]);
}

MemoryUsage memory_usage() {
    MemoryUsage m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        m.current = static_cast<double>(pmc.WorkingSetSize);
        m.peak = static_cast<double>(pmc.PeakWorkingSetSize);
    }
#elif defined(__linux__)
    if (std::FILE* f = std::fopen("/proc/self/status", "r")) {   // VmRSS / VmHWM in kB
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            double kb = 0.0;
            if (std::sscanf(line, "VmRSS: %lf", &kb) == 1) m.current = kb * 1024.0;
            if (std::sscanf(line, "VmHWM: %lf", &kb) == 1) m.peak = kb * 1024.0;
        }
        std::fclose(f);
    }
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) m.peak = static_cast<double>(ru.ru_maxrss);   // bytes on macOS
#endif
    return m;
}

}   // namespace fv
