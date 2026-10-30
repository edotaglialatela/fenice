// Case files
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_common.hpp"

// ---------------------------------------------------------------------------
// Case files
// ---------------------------------------------------------------------------

namespace {
// Write `text` as <tmp>/<name>/case.toml and return the folder.
std::string write_case(const std::string& name, const std::string& text) {
    const fs::path d = tmp_dir() / "casefiles" / name;
    fs::create_directories(d);
    std::ofstream(d / "case.toml", std::ios::binary) << text;
    return d.string();
}

// The error message load_case_file throws for `text`, or "" if it loads.
std::string case_error(const std::string& name, const std::string& text) {
    try {
        load_case_file(write_case(name, text));
    } catch (const std::exception& e) { return e.what(); }
    return "";
}

const char* kMinimalCase = R"toml(
[case]
name = "Minimal"
[initial]
state = { rho = 1.0, p = 1.0 }
)toml";
}   // namespace

TEST(every_parameter_has_a_case_section) {
    const Params base = baseline_defaults();
    for (const auto& [k, v] : base.values()) {
        if (k == "test_case") continue;
        const bool ok = !param_section(k).empty();
        if (!ok) std::printf("    parameter '%s' has no case-file section\n", k.c_str());
        CHECK(ok);
    }
}

// Bad --set / --config values fail up front instead of being silently ignored or truncated.
TEST(parameter_validation) {
    auto error_of = [](std::initializer_list<std::pair<const char*, const char*>> sets) -> std::string {
        Params p = baseline_defaults();
        for (const auto& kv : sets) p.set(kv.first, kv.second);
        try {
            check_parameter_keys(p, "--set");
            SimParams::from(p);
        } catch (const std::invalid_argument& e) { return e.what(); }
        return "";
    };
    CHECK(error_of({}).empty());
    CHECK(error_of({{"nx", "200.0"}, {"cfl", " 0.4 "}, {"eos_type", "stiffened gas"}}).empty());
    CHECK(error_of({{"nX", "200"}}).find("did you mean 'nx'") != std::string::npos);
    CHECK(error_of({{"nx", "200.5"}}).find("not an integer") != std::string::npos);
    CHECK(error_of({{"cfl", "0.5x"}}).find("not a number") != std::string::npos);
    CHECK(error_of({{"cfl", "0"}}).find("cfl") != std::string::npos);
    CHECK(error_of({{"eos_type", "ideal"}}).find("eos_type") != std::string::npos);
    CHECK(error_of({{"bc_left", "periodic"}}).find("in pairs") != std::string::npos);
    CHECK(error_of({{"bc_left", "Periodic"}, {"bc_right", "periodic"}}).empty());
    // adaptive mesh: any base mesh, resume, periodic and mass-flow boundaries and sources are fine now
    CHECK(error_of({{"use_amr", "true"}, {"resume", "true"}}).empty());
    CHECK(error_of({{"use_amr", "true"}, {"nx", "100"}}).empty());
    CHECK(error_of({{"use_amr", "true"}, {"nx", "96"}, {"ny", "96"}, {"bc_left", "periodic"}, {"bc_right", "periodic"}}).empty());
    CHECK(error_of({{"use_amr", "true"}, {"bc_left", "massflow_inlet"}}).empty());
    CHECK(error_of({{"amr_patch_nx", "16"}}).find("removed") != std::string::npos);
    CHECK(error_of({{"amr_patch_ny", "16"}}).find("level-0 grid") != std::string::npos);
    CHECK(error_of({{"amr_buffer_layers", "1"}}).find("amr_diffusion") != std::string::npos);
    CHECK(error_of({{"amr_indicator", "density+vorticity"}}).find("amr_indicator") != std::string::npos);
    CHECK(error_of({{"amr_indicator", "pressure+density+velocity"}}).empty());
    CHECK(error_of({{"amr_derefine_threshold", "0.3"}, {"amr_refine_threshold", "0.2"}}).find("derefine") != std::string::npos);
    CHECK(error_of({{"use_amr", "true"}, {"amr_criterion", "0"}}).find("amr_criterion") != std::string::npos);
    CHECK(error_of({{"use_amr", "true"}, {"nx", "96"}, {"ny", "96"}, {"time_scheme", "BDF2 Dual Time"}}).find("BDF2") !=
          std::string::npos);
    CHECK(error_of({{"use_amr", "true"}, {"nx", "96"}, {"ny", "96"}}).empty());
    bool threw = false;
    try {
        parse_limiter("superbee");
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

TEST(case_file_parsing) {
    const CaseFile cf = load_case_file(write_case("parse", R"toml(
[case]
name = "Parse test"
description = "all value types"
order = 7
requires_viscous = true

[mesh]
Lx = 2.5
nx = 40

[physics]
use_viscous = true
mu = 2.5e-3

[numerics]
riemann_solver = "HLLC (Low-Mach Thornber)"

[initial]
state = { rho = 1.0, u = "0.5*x", p = 1.0 }

[[initial.region]]
shape = "sphere"
center = ["Lx/2", 0.5]
radius = 0.25
state = { rho = 0.1 }
)toml"));
    CHECK(cf.name == "Parse test");
    CHECK(cf.description == "all value types");
    CHECK(cf.order == 7);
    CHECK(cf.requires_viscous);
    CHECK(!cf.requires_3d);
    CHECK(cf.params.str("Lx") == "2.5");
    CHECK(cf.params.str("nx") == "40");
    CHECK(cf.params.str("use_viscous") == "true");
    CHECK(cf.params.num("mu") == 2.5e-3);
    CHECK(cf.params.str("riemann_solver") == "HLLC (Low-Mach Thornber)");
    CHECK(cf.initial.state.at("u") == "0.5*x");
    CHECK(cf.initial.regions.size() == 1);
    CHECK(cf.initial.regions[0].center[0] == "Lx/2");
    CHECK(cf.initial.regions[0].state.at("rho") == "0.1");
}

TEST(case_file_errors_are_helpful) {
    const std::string base = kMinimalCase;
    auto has = [](const std::string& msg, const char* needle) {
        const bool ok = msg.find(needle) != std::string::npos;
        if (!ok) std::printf("    expected '%s' in: %s\n", needle, msg.c_str());
        return ok;
    };
    CHECK(has(case_error("e1", base + "[thermo]\ngama = 1.4\n"), "did you mean 'gamma'"));
    CHECK(has(case_error("e2", base + "[run]\nnx = 10\n"), "'nx' belongs in [mesh], not [run]"));
    CHECK(has(case_error("e3", base + "[numeric]\ncfl = 0.5\n"), "did you mean 'numerics'"));
    CHECK(has(
        case_error("e4",
                   base + "[[initial.region]]\nshape = \"sphear\"\ncenter = [0, 0]\nradius = 1\nstate = { rho = 2 }\n"),
        "did you mean 'sphere'"));
    CHECK(has(case_error("e5", base + "[[initial.region]]\nshape = \"box\"\nmin = [0, 0]\nstate = { rho = 2 }\n"),
              "a box needs 'min' and 'max'"));
    CHECK(has(case_error("e6", "[case]\nname = \"x\"\n[initial]\nstate = { rho = 1.0 }\n"), "at least rho and p"));
    CHECK(has(case_error("e7", "[case]\nname = \"x\"\n[initial]\nstate = { rho = \"1 + sin(xx)\", p = 1.0 }\n"),
              "did you mean 'x'"));
    CHECK(has(case_error("e8", "[case]\nname = \"x\"\n[initial]\nstate = { rh = 1.0, p = 1.0 }\n"),
              "did you mean 'rho'"));
    CHECK(has(case_error("e9", base + "[mesh]\nnx = = 3\n"), "case.toml:"));   // TOML syntax, with a line
    CHECK(has(case_error("e10", base + "[mesh]\ntest_case = \"Sod\"\n"), "cannot be set in a case file"));
    CHECK(has(case_error("e11", "[initial]\nstate = { rho = 1.0, p = 1.0 }\n"), "missing [case] name"));
    CHECK(case_error("ok", base).empty());
}

TEST(initial_regions_and_expressions) {
    // 10 x 10 cells on [0, 1]^2: centres at 0.05, 0.15, ..., 0.95
    const Mesh m = generate_cartesian_mesh(1.0, 1.0, 10, 10);
    const EOS eos = EOS::ideal(1.4);
    Params run = baseline_defaults();
    run.set("u_in", "0.75");
    InitialSpec spec;
    spec.state = {{"rho", "1 + 0.5*x"}, {"u", "u_in"}, {"v", "0.0"}, {"p", "2.0"}};
    InitialRegion box;   // later regions override earlier ones
    box.shape = "box";
    box.min = {"0.0", "0.0"};
    box.max = {"0.5", "0.5"};
    box.state = {{"rho", "3.0"}};
    InitialRegion half;
    half.shape = "halfspace";
    half.point = {"0.0", "0.8"};
    half.normal = {"0.0", "1.0"};   // y >= 0.8
    half.state = {{"p", "5.0"}, {"v", "-1.0"}};
    InitialRegion circle;
    circle.shape = "sphere";
    circle.center = {"0.3", "0.3"};   // a grid vertex: the 4 cells around it are inside
    circle.radius = "0.08";
    circle.state = {{"rho", "9.0"}};
    InitialRegion cyl;
    cyl.shape = "cylinder";
    cyl.center = {"0.7", "0.2"};
    cyl.radius = "Lx/12";
    cyl.state = {{"rho", "7.0"}};
    spec.regions = {box, half, circle, cyl};
    const Field U = evaluate_initial(spec, m, eos, run);

    bool ok = true;
    for (int j = 0; j < 10; ++j)
        for (int i = 0; i < 10; ++i) {
            const double x = m.xc(i), y = m.yc(j);
            State W{1.0 + 0.5 * x, 0.75, 0.0, 2.0, 0.0};
            if (x <= 0.5 && y <= 0.5) W[0] = 3.0;
            if (y >= 0.8) {
                W[3] = 5.0;
                W[2] = -1.0;
            }
            if ((x - 0.3) * (x - 0.3) + (y - 0.3) * (y - 0.3) <= 0.08 * 0.08) W[0] = 9.0;
            if ((x - 0.7) * (x - 0.7) + (y - 0.2) * (y - 0.2) <= (1.0 / 12) * (1.0 / 12)) W[0] = 7.0;
            const State E = eos.prim_to_cons(W);
            for (int q = 0; q < kNVar; ++q) ok = ok && U[j * 10 + i][q] == E[q];
        }
    CHECK(ok);
    // The circle and the cylinder each cover the 4 cells around their centre
    int n9 = 0, n7 = 0;
    for (const State& s : U) {
        n9 += s[0] == 9.0;
        n7 += s[0] == 7.0;
    }
    CHECK(n9 == 4);
    CHECK(n7 == 4);

    // A profile with a region on top: untouched cells keep the profile's state bit for bit
    InitialSpec ps;
    InitialRegion bubble = circle;
    bubble.state = {{"rho", "0.5"}};
    ps.regions = {bubble};
    Field base(static_cast<size_t>(m.num_cells()));
    for (int c = 0; c < m.num_cells(); ++c) base[c] = eos.prim_to_cons({1.0 + 0.01 * c, 0.1, 0.2, 3.0, 0.0});
    const Field P = evaluate_initial(ps, m, eos, run, &base);
    int kept = 0, changed = 0;
    for (int c = 0; c < m.num_cells(); ++c) {
        if (P[c] == base[c])
            ++kept;
        else if (std::abs(eos.cons_to_prim(P[c])[0] - 0.5) < 1e-14)
            ++changed;
    }
    CHECK(kept == 96);
    CHECK(changed == 4);

    // Usual precedence: -a^2 = -(a^2) (a Gaussian must not become exp(+x^2)), a^b^c = a^(b^c)
    InitialSpec pr;
    pr.state = {{"rho", "2^3^2"}, {"u", "-2^2"}, {"v", "exp(-x^2)"}, {"p", "1.0"}};
    const Field Q = evaluate_initial(pr, m, eos, run);
    const State Wq = eos.cons_to_prim(Q[3]);   // cell (3, 0): x = 0.35
    CHECK(Wq[0] == 512.0);
    CHECK_NEAR(Wq[1], -4.0, 1e-12);
    CHECK_NEAR(Wq[2], std::exp(-0.35 * 0.35), 1e-12);
}

TEST(case_file_save_load_round_trip) {
    CaseFile cf = load_case_file(write_case("rt_src", R"toml(
[case]
name = "Round trip"
description = "quote \" and backslash \\ survive"
order = 3
requires_3d = true

[mesh]
Lx = 6.283185307179586
nz = 8

[boundaries]
bc_front = "wall"
p_out = 0.7142857142857143

[numerics]
inner_tol_rel = 1e-3

[initial]
state = { rho = 1.0, u = "sin(x)*cos(y)", w = 0.25, p = "1/(gamma*0.1*0.1)" }

[[initial.region]]
shape = "cylinder"
center = ["Lx/2", 1.0, 0.0]
radius = 0.5
axis = "x"
state = { rho = 2.0 }

[[initial.region]]
shape = "halfspace"
point = [0.0, 0.0, "Lz/2"]
normal = [0.0, 0.0, 1.0]
state = { p = 9.0 }
)toml"));
    const std::string out = (tmp_dir() / "casefiles" / "rt_saved").string();
    save_case_file(cf, out);
    const CaseFile back = load_case_file(out);
    CHECK(back.name == cf.name);
    CHECK(back.description == cf.description);
    CHECK(back.order == 3 && back.requires_3d && !back.requires_viscous);
    CHECK(back.params.values().size() == cf.params.values().size());
    for (const auto& [k, v] : cf.params.values()) {
        const std::string w = back.params.str(k);
        char* e1 = nullptr;
        char* e2 = nullptr;
        const double a = std::strtod(v.c_str(), &e1), b = std::strtod(w.c_str(), &e2);
        CHECK(w == v || (*e1 == '\0' && *e2 == '\0' && a == b));
    }
    CHECK(back.initial.state == cf.initial.state);
    CHECK(back.initial.regions.size() == 2);
    CHECK(back.initial.regions[0].axis == "x");
    CHECK(back.initial.regions[1].point == cf.initial.regions[1].point);
    // Same initial field, bit for bit
    const Mesh m = generate_cartesian_mesh_3d(6.283185307179586, 2.0, 1.0, 12, 6, 8);
    Params run = baseline_defaults();
    run.merge(cf.params);
    const Field a = evaluate_initial(cf.initial, m, EOS::ideal(1.4), run);
    const Field b = evaluate_initial(back.initial, m, EOS::ideal(1.4), run);
    CHECK(a == b);
}

TEST(builtin_cases_and_save_case) {
    // The shipped case files: names, order and flags
    const std::vector<std::string> names = {
        "Sod Shock Tube",      "Poiseuille Flow",   "Lid Driven Cavity",     "IBM Flow over Cylinder",
        "IBM Airfoil",         "IBM Airfoil AMR",   "Shu-Osher Test",        "IBM Supersonic Flow over Cylinder",
        "Taylor-Green Vortex", "Isentropic Vortex", "Taylor-Green Vortex 3D"};
    std::vector<std::string> got;
    for (const TestCase& tc : test_registry()) got.push_back(tc.key);
    CHECK(std::search(got.begin(), got.end(), names.begin(), names.end()) == got.begin());
    CHECK(get_test_case("Taylor-Green Vortex 3D").three_d);
    CHECK(get_test_case("Lid Driven Cavity").requires_viscosity);
    CHECK(!get_test_case("Sod Shock Tube").requires_viscosity);
    CHECK(&get_test_case("sod_shock_tube") == &get_test_case("Sod Shock Tube"));   // folder name
    bool threw = false;
    try {
        get_test_case("Sod Shok Tube");
    } catch (const std::invalid_argument& e) {
        threw = std::string(e.what()).find("did you mean 'Sod Shock Tube'") != std::string::npos;
    }
    CHECK(threw);
    CHECK(case_folder_name("IBM Flow over Cylinder") == "ibm_flow_over_cylinder");
    CHECK(case_folder_name("Re = 1600 (3-D)") == "re_1600_3_d");

    // save_case keeps the source's parameters and anything changed; it never overwrites
    Params run = baseline_defaults();
    run.merge(get_test_case("Lid Driven Cavity").defaults);
    run.set("nx", "48");
    run.set("output_root", "somewhere");
    const std::string dir = (tmp_dir() / "casefiles" / "saved_cavity").string();
    fs::remove_all(dir);
    const std::string file = save_case(get_test_case("Lid Driven Cavity"), run, dir, "Saved cavity");
    const TestCase& saved = get_test_case(dir);
    CHECK(saved.key == "Saved cavity");
    CHECK(saved.defaults.str("nx") == "48");
    CHECK(saved.defaults.str("bc_top") == "moving_wall");
    CHECK(!saved.defaults.has("output_root"));
    CHECK(saved.requires_viscosity);
    bool refused = false;
    try {
        save_case(get_test_case("Lid Driven Cavity"), run, dir, "Saved cavity 2");
    } catch (const std::runtime_error&) { refused = true; }
    CHECK(refused);
    CHECK(fs::exists(file));
}

