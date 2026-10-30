// Harness of the unit tests, shared by the test_*.cpp files: the registry, CHECK / CHECK_NEAR / TEST, and a few helpers.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <functional>
#include <condition_variable>
#include <iterator>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mutex>
#include <numeric>
#include <random>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "fv/CellAdvance.hpp"
#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"
#include "fv/CellParallel.hpp"
#include "fv/CellRemesh.hpp"
#include "fv/ParallelScan.hpp"
#include "fv/Simulation.hpp"
#include "fv/SolutionMapping.hpp"
#include "fv/Switches.hpp"
#include "../src/SchemesSimd.hpp"   // detail::avx2_available: which path the bit-for-bit tests exercised

using namespace fv;
namespace fs = std::filesystem;

extern int g_fail, g_checks;   // defined in test_main.cpp

struct TestCaseReg {
    const char* name;
    void (*fn)();
};
std::vector<TestCaseReg>& registry();   // defined in test_main.cpp
struct Registrar {
    Registrar(const char* n, void (*f)()) { registry().push_back({n, f}); }
};

#define TEST(name)                                                                                                     \
    static void name();                                                                                                \
    static Registrar reg_##name(#name, &name);                                                                         \
    static void name()

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        ++g_checks;                                                                                                    \
        if (!(cond)) {                                                                                                 \
            ++g_fail;                                                                                                  \
            std::printf("    FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                                            \
        }                                                                                                              \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                          \
    do {                                                                                                               \
        ++g_checks;                                                                                                    \
        const double va_ = (a), vb_ = (b);                                                                             \
        if (!(std::abs(va_ - vb_) <= (tol))) {                                                                         \
            ++g_fail;                                                                                                  \
            std::printf("    FAIL %s:%d  %s = %.15g, expected %.15g (tol %g)\n", __FILE__, __LINE__, #a, va_, vb_,     \
                        (double)(tol));                                                                                \
        }                                                                                                              \
    } while (0)

inline fs::path tmp_dir() {
    fs::path d = fs::temp_directory_path() / "fv_cpp_tests";
    fs::create_directories(d);
    return d;
}

inline SimParams params_for(const std::string& tc, std::initializer_list<std::pair<const char*, const char*>> kv) {
    Params p = baseline_defaults();
    p.merge(get_test_case(tc).defaults);
    p.set("test_case", tc);
    p.set("output_root", tmp_dir().string());
    for (const auto& e : kv) p.set(e.first, e.second);
    return SimParams::from(p);
}

inline Field uniform_field(const EOS& eos, int n, double rho, double u, double v, double p) {
    return Field(static_cast<size_t>(n), eos.prim_to_cons({rho, u, v, p}));
}

// Smooth, non-trivial state with nonzero velocity everywhere (also inside an immersed body)
inline State test_state(const EOS& eos, double x, double y) {
    return eos.prim_to_cons({1.0 + 0.1 * std::sin(1.3 * x) * std::cos(0.9 * y), 1.0 + 0.2 * std::sin(0.7 * y),
                             0.15 * std::cos(1.1 * x), 17.0 + 0.5 * std::cos(0.6 * x + 0.4 * y)});
}

inline SimParams airfoil_params(std::initializer_list<std::pair<const char*, const char*>> kv) {
    Params pr = baseline_defaults();
    pr.merge(get_test_case("IBM Airfoil").defaults);
    pr.set("test_case", "IBM Airfoil");
    pr.set("output_root", tmp_dir().string());
    pr.set("use_mpi", "false");
    for (const auto& e : kv) pr.set(e.first, e.second);
    return SimParams::from(pr);
}

inline double max_abs(const Field& R) {
    double m = 0.0;
    for (const State& r : R)
        for (double x : r) m = std::max(m, std::abs(x));
    return m;
}
