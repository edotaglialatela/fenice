// Unit tests for the C++ solver, main(). The tests are in test_*.cpp; test_common.hpp has the harness.
// Self-contained harness: run `fenice_tests` (exit code = number of failed checks).
#include "test_common.hpp"

int g_fail = 0, g_checks = 0;

std::vector<TestCaseReg>& registry() {
    static std::vector<TestCaseReg> r;   // function-local: registration from several files does not depend on their initialisation order
    return r;
}

// fenice_tests                 all tests (what CI runs)
// fenice_tests --list          the names, one per line
// fenice_tests NAME [NAME ..]  the tests whose name contains any of the arguments; a filter that matches nothing is an error
int main(int argc, char** argv) {
    std::vector<std::string> filters;
    for (int a = 1; a < argc; ++a) {
        if (std::string(argv[a]) == "--list") {
            for (const auto& t : registry()) std::printf("%s\n", t.name);
            return 0;
        }
        filters.emplace_back(argv[a]);
    }
    auto selected = [&](const char* name) {
        if (filters.empty()) return true;
        for (const std::string& f : filters)
            if (std::string(name).find(f) != std::string::npos) return true;
        return false;
    };
    int failed_tests = 0;
    size_t ran = 0;
    for (const auto& t : registry()) {
        if (!selected(t.name)) continue;
        ++ran;
        const int before = g_fail;
        std::printf("[ RUN  ] %s\n", t.name);
        std::fflush(stdout);
        try {
            t.fn();
        } catch (const std::exception& e) {
            ++g_fail;
            std::printf("    EXCEPTION: %s\n", e.what());
        }
        const bool ok = g_fail == before;
        failed_tests += ok ? 0 : 1;
        std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", t.name);
    }
    if (ran == 0) {
        std::printf("no test matches the filter (fenice_tests --list shows the names)\n");
        return 2;
    }
    std::printf("\n%zu tests, %d checks, %d failed checks, %d failed tests\n", ran, g_checks, g_fail, failed_tests);
    return failed_tests == 0 ? 0 : 1;
}
