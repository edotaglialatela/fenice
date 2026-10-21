// fenice - command-line driver for the C++ FV solver.
//
// Parameters resolve in this order (later wins):
//   baseline defaults -> test-case DEFAULTS -> --config file -> --set overrides
//
// Examples
//   fenice --case "Sod Shock Tube"
//   fenice --case "Shu-Osher Test" --set nx=2000 --set use_muscl=true
//   fenice --config run.cfg               (as written by the GUI launcher)
//   mpiexec -n 8 fenice --case "IBM Supersonic Flow over Cylinder"
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <io.h>
#include <windows.h>
#define FV_ISATTY_STDOUT() _isatty(_fileno(stdout))
#else
#include <unistd.h>
#define FV_ISATTY_STDOUT() isatty(fileno(stdout))
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef FV_USE_MPI
#include <mpi.h>
#endif

#include "fv/Simulation.hpp"

namespace {

void on_sigint(int) {
    if (fv::g_stop_requested.exchange(true)) std::_Exit(130);   // second Ctrl+C: hard stop
    std::signal(SIGINT, on_sigint);
}

void usage() {
    std::printf(
        "Usage: fenice [options]\n"
        "  --case NAME|PATH   test case: a name from --list-cases, or a case folder / case.toml\n"
        "  --config FILE      'key = value' parameter file\n"
        "  --set KEY=VALUE    override one parameter (repeatable)\n"
        "  --threads N        OpenMP threads (default: all cores; 1 per rank under MPI)\n"
        "  --list-cases       list the built-in test cases\n"
        "  --print-config     print the resolved parameters and exit\n"
        "  --print-case-defaults  print only the test case's own defaults and exit\n"
        "  --no-logo          do not print the banner\n"
        "  --new-case NAME    create cases/<name>/case.toml from --from CASE (default: --case), then exit\n"
        "  --from CASE        source case for --new-case\n"
        "  --save-case TARGET save the resolved run (case + --config + --set) as a new case and exit;\n"
        "                     TARGET is a case name (written under cases/) or a folder path\n"
        "  --case-name NAME   [case] name for --save-case (default: from TARGET)\n"
        "Keys: nx, ny, t_end, use_muscl, bc_left, ... (see the Parameters page of the docs),\n"
        "plus output_root, dump_state and max_steps.\n");
}

bool launched_by_mpiexec() {
    for (const char* v : {"PMI_SIZE", "PMI_RANK", "OMPI_COMM_WORLD_SIZE", "MPI_LOCALNRANKS"})
        if (std::getenv(v)) return true;
    return false;
}

std::string self_path(const char* argv0) {
#ifdef _WIN32
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n > 0 && n < MAX_PATH) return std::string(buf, n);
#endif
    return argv0;
}

// Colour depth for the banner: 0 (plain), 256 or 24 (24-bit).  Colour only on a terminal
// (or with FORCE_COLOR, which the MPI relaunch sets for its piped ranks); NO_COLOR and
// TERM=dumb turn it off (https://no-color.org).
int terminal_colors() {
    if (std::getenv("NO_COLOR")) return 0;
    const char* term = std::getenv("TERM");
    if (term && std::string(term) == "dumb") return 0;
    const bool tty = FV_ISATTY_STDOUT() != 0;
    if (!tty && !std::getenv("FORCE_COLOR")) return 0;
#ifdef _WIN32
    if (tty) {   // the console interprets escape sequences only with VT processing on
        const HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode = 0;
        if (!GetConsoleMode(h, &mode) || !SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) return 0;
    }
    return 24;   // Windows Terminal and conhost (Windows 10+) both render 24-bit colour
#else
    const char* ct = std::getenv("COLORTERM");
    if (ct && (std::string(ct) == "truecolor" || std::string(ct) == "24bit")) return 24;
    return 256;   // e.g. macOS Terminal.app
#endif
}

std::string quote(const std::string& s) { return "\"" + s + "\""; }

// Re-launch this executable under mpiexec with n ranks.  Returns
// the exit code, or -1 if no MPI launcher is available.
int relaunch_under_mpiexec(int n, const std::string& exe, const std::vector<std::string>& args) {
    std::string mpiexec;
    if (const char* bin = std::getenv("MSMPI_BIN")) mpiexec = std::string(bin) + "mpiexec.exe";
    else mpiexec = "mpiexec";
    std::string cmd = quote(mpiexec) + " -n " + std::to_string(n) + " " + quote(exe);
    for (const auto& a : args) cmd += " " + quote(a);
    cmd += " --mpi-child";
    std::printf("[MPI] Launching %d workers: %s\n", n, cmd.c_str());
    std::fflush(stdout);
#ifdef _WIN32
    cmd = "\"" + cmd + "\"";   // cmd.exe strips one level of outer quotes
#endif
    const int rc = std::system(cmd.c_str());
    if (rc != 0) std::printf("[MPI] Workers exited with code %d.\n", rc);
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::vector<std::string> raw(argv + 1, argv + argc);
    std::string case_name, config_file, save_target, save_name, from_case;
    std::vector<std::string> sets;
    bool list_cases = false, print_config = false, print_case_defaults = false, logo = true, mpi_child = false;
    int threads = 0;

    for (size_t k = 0; k < raw.size(); ++k) {
        const std::string& a = raw[k];
        auto next = [&]() -> std::string {
            if (k + 1 >= raw.size()) { std::fprintf(stderr, "Missing value after %s\n", a.c_str()); std::exit(2); }
            return raw[++k];
        };
        if (a == "--case") case_name = next();
        else if (a == "--config") config_file = next();
        else if (a == "--set") sets.push_back(next());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--list-cases") list_cases = true;
        else if (a == "--print-config") print_config = true;
        else if (a == "--print-case-defaults") print_case_defaults = true;
        else if (a == "--no-logo") logo = false;
        else if (a == "--mpi-child") mpi_child = true;
        else if (a == "--save-case") save_target = next();
        else if (a == "--new-case") save_target = save_name = next();
        else if (a == "--from") from_case = next();
        else if (a == "--case-name") save_name = next();
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "Unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    std::vector<std::string> forward;   // arguments to pass to MPI children
    for (const auto& a : raw) if (a != "--mpi-child") forward.push_back(a);

    if (list_cases) {
        for (const auto& tc : fv::test_registry()) std::printf("%s\n", tc.key.c_str());
        return 0;
    }

    if (!from_case.empty()) {
        if (save_target.empty()) { std::fprintf(stderr, "--from needs --new-case NAME\n"); return 2; }
        case_name = from_case;
    }

    fv::SimParams sp;
    fv::Params merged;
    try {
        fv::Params cfg, ov;
        if (!config_file.empty()) cfg.load_file(config_file);
        for (const auto& s : sets) ov.set_assignment(s);
        fv::check_parameter_keys(cfg, config_file);
        fv::check_parameter_keys(ov, "--set");
        merged = fv::baseline_defaults();
        if (case_name.empty()) case_name = ov.str("test_case", cfg.str("test_case", merged.str("test_case")));
        merged.merge(fv::get_test_case(case_name).defaults);
        merged.merge(cfg);
        merged.merge(ov);
        merged.set("test_case", case_name);
        sp = fv::SimParams::from(merged);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ERROR] %s\n", e.what());
        return 2;
    }
    if (!save_target.empty()) {
        try {
            const fv::TestCase& src = fv::get_test_case(case_name);
            const std::string file = fv::save_case(src, merged, save_target, save_name);
            std::printf("[CASE] Wrote %s (from '%s')\n", file.c_str(), src.key.c_str());
            std::printf("[CASE] Edit it, then run: fenice --case \"%s\"\n", fv::load_case_file(file).name.c_str());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[ERROR] %s\n", e.what());
            return 2;
        }
        return 0;
    }
    if (print_case_defaults) {
        for (const auto& kv : fv::get_test_case(case_name).defaults.values())
            std::printf("%s = %s\n", kv.first.c_str(), kv.second.c_str());
        return 0;
    }
    if (print_config) {
        for (const auto& kv : merged.values()) std::printf("%s = %s\n", kv.first.c_str(), kv.second.c_str());
        return 0;
    }

    const bool under_mpi = mpi_child || launched_by_mpiexec();
#ifdef FV_USE_MPI
    if (sp.use_mpi && sp.n_cpus > 1 && !under_mpi) {
        // The ranks write to mpiexec's pipe: pass the colour decision down to them
        if (logo && terminal_colors() && !std::getenv("FORCE_COLOR")) {
#ifdef _WIN32
            _putenv_s("FORCE_COLOR", "1");
#else
            setenv("FORCE_COLOR", "1", 0);
#endif
        }
        return relaunch_under_mpiexec(sp.n_cpus, self_path(argv[0]), forward);
    }
#else
    if (sp.use_mpi && sp.n_cpus > 1 && !under_mpi)
        std::printf("[MPI] This build has no MPI support - running on one process with OpenMP threads instead.\n");
#endif

    int rank = 0, size = 1;
#ifdef FV_USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
    else if (size > 1 && !std::getenv("OMP_NUM_THREADS")) omp_set_num_threads(1);   // avoid oversubscription
#endif
    std::signal(SIGINT, on_sigint);
    if (logo && rank == 0) fv::print_logo(terminal_colors());

    int rc = 0;
    try {
        fv::Simulation sim;
        if (size > 1 && !sp.use_amr) {
#ifdef FV_USE_MPI
            sim.p = sp;
            sim.run_mpi(MPI_COMM_WORLD);
#endif
        } else {   // serial, or AMR: every rank holds the tree and shares the patches (solve_amr)
            sim.setup(sp, rank == 0);
            sim.initialize_field();
            sim.solve();
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ERROR] %s\n", e.what());
        rc = 1;
#ifdef FV_USE_MPI
        if (size > 1) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    }
#ifdef FV_USE_MPI
    MPI_Finalize();
#endif
    return rc;
}
