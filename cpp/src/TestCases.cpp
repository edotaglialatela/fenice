#include "fv/TestCases.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>

#ifndef FV_PROJECT_ROOT
#define FV_PROJECT_ROOT "."
#endif

namespace fs = std::filesystem;

namespace fv {

namespace {

// --- Built-in profiles: analytic fields kept in C++ (bit-identical to the validated runs) ---

Field taylor_green_profile(const Mesh& m, const EOS& eos, const Params& run) {
    const double mu = run.num("mu", 0.0);
    Field U(static_cast<size_t>(m.window_cells()));
    for (int j = m.win_j0; j < m.window_j1(); ++j)
        for (int i = m.win_i0; i < m.window_i1(); ++i)
            U[m.window_cell(i, j)] = eos.prim_to_cons(taylor_green_exact(m.xc(i), m.yc(j), 0.0, m.Lx, mu));
    return U;
}

Field isentropic_vortex_profile(const Mesh& m, const EOS& eos, const Params&) {
    Field U(static_cast<size_t>(m.window_cells()));
    for (int j = m.win_j0; j < m.window_j1(); ++j)
        for (int i = m.win_i0; i < m.window_i1(); ++i)
            U[m.window_cell(i, j)] = eos.prim_to_cons(isentropic_vortex_exact(m.xc(i), m.yc(j), 0.0, m.Lx, m.Ly, eos.gamma));
    return U;
}

Field taylor_green_3d_profile(const Mesh& m, const EOS& eos, const Params&) {
    Field U(static_cast<size_t>(m.num_cells()));
    for (int k = 0; k < m.nz; ++k)
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i)
                U[m.cell(i, j, k)] = eos.prim_to_cons(taylor_green_3d_initial(m.xc(i), m.yc(j), m.zc(k), eos.gamma));
    return U;
}

const std::map<std::string, Profile>& profiles() {
    static const std::map<std::string, Profile> p = {
        {"taylor_green", taylor_green_profile},
        {"isentropic_vortex", isentropic_vortex_profile},
        {"taylor_green_3d", taylor_green_3d_profile},
    };
    return p;
}

TestCase make_test_case(const CaseFile& cf) {
    TestCase tc;
    tc.key = cf.name;
    tc.description = cf.description;
    tc.source = cf.path;
    tc.requires_viscosity = cf.requires_viscous;
    tc.three_d = cf.requires_3d;
    tc.defaults = cf.params;
    tc.initial = cf.initial;
    const Profile* prof = nullptr;
    if (!cf.initial.profile.empty()) {
        prof = find_profile(cf.initial.profile);
        if (!prof) {
            const std::string m = closest_match(cf.initial.profile, profile_names());
            throw std::runtime_error(cf.path + ": unknown profile '" + cf.initial.profile + "'" +
                                     (m.empty() ? "" : " (did you mean '" + m + "'?)"));
        }
    }
    const std::string name = cf.name;
    const InitialSpec spec = cf.initial;
    tc.initialize = [name, spec, prof](const Mesh& m, const EOS& eos, const Params& run) {
        std::printf("[TEST CASE] Initialising %s...\n", name.c_str());
        if (!prof) return evaluate_initial(spec, m, eos, run);
        const Field base = (*prof)(m, eos, run);
        return spec.regions.empty() ? base : evaluate_initial(spec, m, eos, run, &base);
    };
    return tc;
}

std::vector<TestCase> load_registry() {
    std::vector<TestCase> reg;
    const fs::path dir(cases_directory());
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        std::fprintf(stderr, "[CASES] Case folder not found: %s\n", dir.generic_string().c_str());
        return reg;
    }
    std::vector<std::pair<int, TestCase>> found;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_directory() || !fs::exists(entry.path() / "case.toml")) continue;
        try {
            const CaseFile cf = load_case_file(entry.path().string());
            found.push_back({cf.order, make_test_case(cf)});
        } catch (const std::exception& e) { std::fprintf(stderr, "[CASES] Skipping %s\n", e.what()); }
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second.key < b.second.key;
    });
    for (auto& f : found) {
        for (const TestCase& prev : reg)
            if (prev.key == f.second.key)
                throw std::runtime_error("Two case files use the name '" + f.second.key + "': " + prev.source +
                                         " and " + f.second.source);
        reg.push_back(std::move(f.second));
    }
    return reg;
}

}   // namespace

std::string cases_directory() {
    if (const char* env = std::getenv("FENICE_CASES"))
        if (*env) return env;
    return std::string(FV_PROJECT_ROOT) + "/cases";
}

const std::vector<TestCase>& test_registry() {
    static const std::vector<TestCase> reg = load_registry();
    return reg;
}

const TestCase& get_test_case(const std::string& key) {
    for (const auto& tc : test_registry())
        if (tc.key == key) return tc;
    for (const auto& tc : test_registry())
        if (fs::path(tc.source).parent_path().filename().string() == key) return tc;

    // A case folder / case.toml outside the registry
    std::error_code ec;
    if (fs::exists(fs::path(key), ec)) {
        static std::map<std::string, std::unique_ptr<TestCase>> loaded;
        const std::string id = fs::weakly_canonical(fs::path(key), ec).generic_string();
        auto it = loaded.find(id);
        if (it == loaded.end())
            it = loaded.emplace(id, std::make_unique<TestCase>(make_test_case(load_case_file(key)))).first;
        return *it->second;
    }

    std::vector<std::string> names;
    for (const auto& tc : test_registry()) names.push_back(tc.key);
    const std::string m = closest_match(key, names);
    throw std::invalid_argument("Unknown test case: '" + key + "'" + (m.empty() ? "" : " (did you mean '" + m + "'?)") +
                                ". Run --list-cases for the available cases, or give a path to a case folder.");
}

std::string case_folder_name(const std::string& name) {
    std::string s;
    for (unsigned char c : name) {
        if (std::isalnum(c))
            s += static_cast<char>(std::tolower(c));
        else if (!s.empty() && s.back() != '_')
            s += '_';
    }
    while (!s.empty() && s.back() == '_') s.pop_back();
    return s.empty() ? "case" : s;
}

std::string save_case(const TestCase& source, const Params& run, const std::string& target, const std::string& name) {
    const bool is_path = target.find_first_of("/\\") != std::string::npos || fs::path(target).extension() == ".toml";
    fs::path dir = is_path ? fs::path(target) : fs::path(cases_directory()) / case_folder_name(target);
    if (dir.extension() == ".toml") dir = dir.parent_path();
    const fs::path file = dir / "case.toml";
    if (fs::exists(file)) throw std::runtime_error("A case already exists at " + file.generic_string());

    CaseFile cf;
    cf.name = !name.empty() ? name : (is_path ? dir.filename().string() : target);
    for (const auto& tc : test_registry())
        if (tc.key == cf.name)
            throw std::runtime_error("A case named '" + cf.name + "' already exists (" + tc.source + ")");
    cf.description = "Created from '" + source.key + "'";
    cf.requires_viscous = source.requires_viscosity;
    cf.requires_3d = source.three_d;
    cf.initial = source.initial;

    // Parameters: those the source case sets, plus anything that differs from the baseline
    const Params base = baseline_defaults();
    auto same = [](const std::string& a, const std::string& b) {
        if (a == b) return true;
        char* ea = nullptr;
        char* eb = nullptr;
        const double da = std::strtod(a.c_str(), &ea), db = std::strtod(b.c_str(), &eb);
        return !a.empty() && !b.empty() && *ea == '\0' && *eb == '\0' && da == db;
    };
    for (const auto& [k, v] : run.values()) {
        if (param_section(k).empty() || k == "output_root" || k == "dump_state" || k == "max_steps" || k == "resume")
            continue;
        if (source.defaults.has(k) || !base.has(k) || !same(v, base.str(k))) cf.params.set(k, v);
    }
    save_case_file(cf, file.string());
    return file.generic_string();
}

const Profile* find_profile(const std::string& name) {
    auto it = profiles().find(name);
    return it == profiles().end() ? nullptr : &it->second;
}

std::vector<std::string> profile_names() {
    std::vector<std::string> n;
    for (const auto& kv : profiles()) n.push_back(kv.first);
    return n;
}

State taylor_green_3d_initial(double x, double y, double z, double gamma) {
    constexpr double U0 = 1.0, rho0 = 1.0, Ma = 0.1;
    const double p0 = rho0 * U0 * U0 / (gamma * Ma * Ma);
    const double u = U0 * std::sin(x) * std::cos(y) * std::cos(z);
    const double v = -U0 * std::cos(x) * std::sin(y) * std::cos(z);
    const double p = p0 + rho0 * U0 * U0 / 16.0 * (std::cos(2.0 * x) + std::cos(2.0 * y)) * (std::cos(2.0 * z) + 2.0);
    return {rho0, u, v, p, 0.0};
}

State taylor_green_exact(double x, double y, double t, double L, double mu) {
    constexpr double PI = 3.14159265358979323846, U0 = 1.0, rho0 = 1.0, p0 = 100.0;
    const double k = 2.0 * PI / L;
    const double F = std::exp(-2.0 * (mu / rho0) * k * k * t);
    const double u = -U0 * std::cos(k * x) * std::sin(k * y) * F;
    const double v = U0 * std::sin(k * x) * std::cos(k * y) * F;
    const double p = p0 - 0.25 * rho0 * U0 * U0 * (std::cos(2.0 * k * x) + std::cos(2.0 * k * y)) * F * F;
    return {rho0, u, v, p};
}

State isentropic_vortex_exact(double x, double y, double t, double Lx, double Ly, double gamma) {
    constexpr double PI = 3.14159265358979323846, beta = 5.0, u_inf = 1.0, v_inf = 1.0;
    // Vortex centre advected with the free stream; minimum-image distance on the torus
    auto wrap = [](double d, double L) { return d - L * std::round(d / L); };
    const double dx = wrap(x - (0.5 * Lx + u_inf * t), Lx);
    const double dy = wrap(y - (0.5 * Ly + v_inf * t), Ly);
    const double r2 = dx * dx + dy * dy;
    const double g = beta / (2.0 * PI) * std::exp(0.5 * (1.0 - r2));
    const double T = 1.0 - (gamma - 1.0) * beta * beta / (8.0 * gamma * PI * PI) * std::exp(1.0 - r2);
    const double rho = std::pow(T, 1.0 / (gamma - 1.0));
    return {rho, u_inf - g * dy, v_inf + g * dx, rho * T};
}

}   // namespace fv
