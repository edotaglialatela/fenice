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

// boundary_order is 2 by default, is 1 with a first-order scheme (nothing to extend) whatever was asked, and an explicit 1 is kept.
TEST(boundary_order_default) {
    auto order_of = [](std::initializer_list<std::pair<const char*, const char*>> sets) {
        Params p = baseline_defaults();
        for (const auto& kv : sets) p.set(kv.first, kv.second);
        return SimParams::from(p).boundary_order;
    };
    CHECK(order_of({{"reconstruction", "muscl"}}) == 2);
    CHECK(order_of({{"reconstruction", "weno3"}, {"time_scheme", "SSP Runge-Kutta 3"}}) == 2);
    CHECK(order_of({{"reconstruction", "weno5"}, {"time_scheme", "SSP Runge-Kutta 3"}}) == 2);
    CHECK(order_of({{"reconstruction", "muscl"}, {"use_amr", "true"}}) == 2);
    CHECK(order_of({{"reconstruction", "muscl"}, {"boundary_order", "1"}}) == 1);
    CHECK(order_of({{"reconstruction", "none"}}) == 1);
    CHECK(order_of({{"reconstruction", "none"}, {"boundary_order", "2"}}) == 1);
    CHECK(order_of({}) == 1);   // the baseline scheme is first order
    bool threw = false;
    try {
        order_of({{"reconstruction", "muscl"}, {"boundary_order", "3"}});
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// reconstruction is the one key of the face reconstruction (none, muscl, weno3, weno5); the retired use_muscl is read as an alias
// with its old meaning, so every old input means what it meant: false (with reconstruction absent or its old default muscl) is first
// order, true with reconstruction absent is muscl, true with a reconstruction keeps it, false with a WENO scheme is an error as before.
TEST(use_muscl_is_read_as_reconstruction) {
    auto rec = [](std::initializer_list<std::pair<const char*, const char*>> sets, bool* alias = nullptr) {
        Params p = baseline_defaults();
        p.set("time_scheme", "SSP Runge-Kutta 3");
        for (const auto& kv : sets) p.set(kv.first, kv.second);
        const SimParams s = SimParams::from(p);
        if (alias) *alias = s.use_muscl_alias;
        return s.reconstruction;
    };
    bool alias = true;
    CHECK(rec({}, &alias) == "none" && !alias);
    CHECK(rec({{"reconstruction", "WENO3"}}, &alias) == "weno3" && !alias);
    CHECK(rec({{"use_muscl", "false"}}, &alias) == "none" && alias);
    CHECK(rec({{"use_muscl", "false"}, {"reconstruction", "muscl"}}) == "none");
    CHECK(rec({{"use_muscl", "true"}}) == "muscl");
    CHECK(rec({{"use_muscl", "true"}, {"reconstruction", "weno5"}}) == "weno5");
    CHECK(rec({{"use_muscl", "true"}, {"reconstruction", "muscl"}}) == "muscl");
    const auto throws = [&](std::initializer_list<std::pair<const char*, const char*>> sets) {
        try {
            rec(sets);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    CHECK(throws({{"use_muscl", "false"}, {"reconstruction", "weno3"}}));
    CHECK(throws({{"reconstruction", "muscl3"}}));
    CHECK(!throws({{"reconstruction", "none"}}));
    // the reconstruction (not the alias) decides the scheme the solver builds
    Simulation sim;
    Params p = baseline_defaults();
    p.merge(get_test_case("Sod Shock Tube").defaults);
    p.set("test_case", "Sod Shock Tube");
    p.set("output_root", tmp_dir().string());
    p.set("use_mpi", "false");
    p.set("use_muscl", "true");
    sim.setup(SimParams::from(p), false);
    CHECK(sim.muscl != nullptr && sim.p.reconstruction == "muscl");
    p.set("use_muscl", "false");
    Simulation first;
    first.setup(SimParams::from(p), false);
    CHECK(first.muscl == nullptr && first.p.reconstruction == "none");
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
cfl = 0.4

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


// ---------------------------------------------------------------------------
// Post-processing tables and time-dependent boundary values
// ---------------------------------------------------------------------------

namespace {
const std::string kPostTables = R"toml(
[[probe]]
name = "p1"
x = 0.25
y = 0.5
interval = 3

[[sample]]
name = "line"
from = [0.0, 0.5]
to = [1.0, 0.5]
points = 5

[[force]]
name = "body"
reference_length = 0.2
reference_density = 1.5
reference_velocity = 2.0
interval = 4

[[average]]
name = "mean"
variables = ["rho", "u", "u", "p"]
interval = 2
)toml";
}   // namespace

TEST(post_processing_tables_parse_and_round_trip) {
    const CaseFile cf = load_case_file(write_case("post_ok", std::string(kMinimalCase) + kPostTables));
    CHECK(cf.probes.size() == 1 && cf.samples.size() == 1 && cf.forces.size() == 1 && cf.averages.size() == 1);
    CHECK(cf.probes[0].name == "p1" && cf.probes[0].x == 0.25 && cf.probes[0].interval == 3 && !cf.probes[0].has_z);
    CHECK(cf.samples[0].points == 5 && cf.samples[0].interval == 1 && cf.samples[0].to[0] == 1.0);
    CHECK(cf.forces[0].reference_density == 1.5 && cf.forces[0].interval == 4);
    CHECK(cf.averages[0].variables == std::vector<std::string>({"rho", "u", "p"}));   // a repeated variable is kept once
    // the generated names of unnamed entries
    const CaseFile anon =
        load_case_file(write_case("post_anon", std::string(kMinimalCase) + "\n[[probe]]\nx = 0.1\ny = 0.2\n"));
    CHECK(anon.probes[0].name == "probe_1" && anon.probes[0].interval == 1);

    const std::string saved = (tmp_dir() / "casefiles" / "post_saved").string();
    save_case_file(cf, saved);
    const CaseFile back = load_case_file(saved);
    CHECK(back.probes.size() == 1 && back.probes[0].name == "p1" && back.probes[0].y == 0.5 && back.probes[0].interval == 3);
    CHECK(back.samples.size() == 1 && back.samples[0].from == cf.samples[0].from && back.samples[0].to == cf.samples[0].to &&
          back.samples[0].points == 5);
    CHECK(back.forces.size() == 1 && back.forces[0].reference_velocity == 2.0 && back.forces[0].interval == 4);
    CHECK(back.averages.size() == 1 && back.averages[0].variables == cf.averages[0].variables &&
          back.averages[0].interval == 2);
}

TEST(post_processing_tables_reject_bad_input) {
    const auto rejected = [&](const char* name, const std::string& table, const char* what) {
        const std::string e = case_error(name, std::string(kMinimalCase) + table);
        if (e.find(what) == std::string::npos) std::printf("    %s: got '%s', wanted '%s'\n", name, e.c_str(), what);
        return e.find(what) != std::string::npos;
    };
    CHECK(rejected("bad_interval0", "\n[[probe]]\nx = 0.1\ny = 0.2\ninterval = 0\n", "positive integer"));
    CHECK(rejected("bad_interval_f", "\n[[probe]]\nx = 0.1\ny = 0.2\ninterval = 2.5\n", "positive integer"));
    CHECK(rejected("bad_probe_y", "\n[[probe]]\nx = 0.1\n", "needs both x and y"));
    CHECK(rejected("bad_probe_key", "\n[[probe]]\nx = 0.1\ny = 0.2\nintervall = 2\n", "unknown key"));
    CHECK(rejected("bad_points", "\n[[sample]]\nfrom = [0.0, 0.0]\nto = [1.0, 0.0]\npoints = 1\n", "points"));
    CHECK(rejected("bad_sample_dim", "\n[[sample]]\nfrom = [0.0, 0.0]\nto = [1.0, 0.0, 0.0]\n",
                   "same number of coordinates"));
    CHECK(rejected("bad_sample_dim1", "\n[[sample]]\nfrom = [0.0]\nto = [1.0]\n", "2 or 3 entries"));
    CHECK(rejected("bad_variable", "\n[[average]]\nvariables = [\"rh0\"]\n", "unknown average variable"));
    CHECK(rejected("bad_variables_empty", "\n[[average]]\nvariables = []\n", "non-empty array"));
    CHECK(rejected("bad_name", "\n[[probe]]\nname = \"a b\"\nx = 0.1\ny = 0.2\n", "may contain only"));
    CHECK(rejected("bad_name_path", "\n[[probe]]\nname = \"../x\"\nx = 0.1\ny = 0.2\n", "filename-safe"));
    CHECK(rejected("bad_duplicate",
                   "\n[[probe]]\nname = \"a\"\nx = 0.1\ny = 0.2\n[[probe]]\nname = \"a\"\nx = 0.3\ny = 0.2\n",
                   "duplicate probe name"));
    CHECK(rejected("bad_force_ref",
                   "\n[[force]]\nreference_length = 1.0\nreference_density = 0.0\nreference_velocity = 1.0\n",
                   "must be positive"));
    CHECK(rejected("bad_force_missing", "\n[[force]]\nreference_length = 1.0\n", "needs reference_length"));
}

TEST(post_processing_domain_checks_at_setup) {
    const auto refused = [&](const char* name, const std::string& tables, const char* what) {
        const std::string dir = write_case(name, std::string(R"toml(
[case]
name = "Post domain"
[mesh]
nx = 8
ny = 4
Ly = 0.5
[initial]
state = { rho = 1.0, p = 1.0 }
)toml") + tables);
        try {
            Simulation sim;
            Params p = baseline_defaults();
            p.merge(load_case_file(dir).params);
            p.set("test_case", dir);
            p.set("output_root", tmp_dir().string());
            sim.setup(SimParams::from(p), false);
        } catch (const std::exception& e) { return std::string(e.what()).find(what) != std::string::npos; }
        return false;
    };
    CHECK(refused("dom_probe_x", "\n[[probe]]\nx = 5.0\ny = 0.1\n", "inside the simulation domain"));
    CHECK(refused("dom_probe_z", "\n[[probe]]\nx = 0.5\ny = 0.1\nz = 3.0\n", "inside the simulation domain"));
    CHECK(refused("dom_sample", "\n[[sample]]\nfrom = [0.0, 0.1]\nto = [1.0, 0.9]\npoints = 3\n", "inside the simulation domain"));
    CHECK(refused("dom_sample_dim", "\n[[sample]]\nfrom = [0.0, 0.1, 0.0]\nto = [1.0, 0.2, 0.0]\npoints = 3\n", "coordinates"));
    CHECK(refused("dom_force", "\n[[force]]\nreference_length = 1.0\nreference_density = 1.0\nreference_velocity = 1.0\n",
                  "use_ibm"));
}

TEST(time_dependent_boundary_values) {
    const auto lid = [&](std::initializer_list<std::pair<const char*, const char*>> kv) {
        Params p = baseline_defaults();
        p.merge(get_test_case("Lid Driven Cavity").defaults);
        p.set("test_case", "Lid Driven Cavity");
        p.set("output_root", tmp_dir().string());
        p.set("nx", "8");
        p.set("ny", "8");
        for (const auto& e : kv) p.set(e.first, e.second);
        return SimParams::from(p);
    };
    const auto write_table = [&](const char* name, const std::string& text) {
        const fs::path f = tmp_dir() / "casefiles" / name;
        fs::create_directories(f.parent_path());
        std::ofstream(f, std::ios::binary) << text;
        return f.string();
    };
    // expression of t: evaluated at the stage time
    {
        Simulation sim;
        sim.setup(lid({{"u_in", "1 + 2*t"}}), false);
        CHECK(sim.boundaries.params().u_in == 1.0);
        sim.update_boundary_values(0.5, sim.boundaries);
        CHECK_NEAR(sim.boundaries.params().u_in, 2.0, 1e-14);
    }
    // table: BOM, header and comments accepted, linear inside, the end values held outside
    {
        const std::string table = write_table("bc_table.csv", "\xEF\xBB\xBFt,value\n0.0,1.0\n# a comment\n1.0,3.0\n2.0,3.0\n");
        Simulation sim;
        sim.setup(lid({{"u_in_table", table.c_str()}}), false);
        CHECK(sim.boundaries.params().u_in == 1.0);
        sim.update_boundary_values(0.25, sim.boundaries);
        CHECK_NEAR(sim.boundaries.params().u_in, 1.5, 1e-14);
        sim.update_boundary_values(9.0, sim.boundaries);
        CHECK(sim.boundaries.params().u_in == 3.0);
        sim.update_boundary_values(-1.0, sim.boundaries);
        CHECK(sim.boundaries.params().u_in == 1.0);
    }
    // an expression and a table for the same value; bad tables; a missing file
    const auto refused = [&](std::initializer_list<std::pair<const char*, const char*>> kv, const char* what) {
        try {
            Simulation sim;
            sim.setup(lid(kv), false);
        } catch (const std::exception& e) { return std::string(e.what()).find(what) != std::string::npos; }
        return false;
    };
    const std::string good = write_table("bc_good.csv", "0,1\n1,2\n");
    const std::string down = write_table("bc_down.csv", "0,1\n1,2\n1,3\n");
    const std::string one = write_table("bc_one.csv", "0,1\n");
    const std::string three = write_table("bc_three.csv", "0,1,2\n1,2,3\n");
    CHECK(refused({{"u_in", "1+t"}, {"u_in_table", good.c_str()}}, "cannot both be set"));
    CHECK(refused({{"u_in_table", down.c_str()}}, "strictly increasing"));
    CHECK(refused({{"u_in_table", one.c_str()}}, "at least two"));
    CHECK(refused({{"u_in_table", three.c_str()}}, "more than two columns"));
    CHECK(refused({{"u_in_table", "no_such_file.csv"}}, "not found"));
    // a density that is positive at t = 0 is accepted, and refused once it has crossed zero
    {
        Simulation sim;
        sim.setup(lid({{"rho_in", "1 - t"}}), false);
        bool threw = false;
        try {
            sim.update_boundary_values(2.0, sim.boundaries);
        } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    }
}

TEST(save_case_copies_boundary_tables) {
    // a relative table (here found from the current folder) is copied next to the saved case file; a missing one is refused
    const fs::path work = tmp_dir() / "save_tables";
    fs::remove_all(work);
    fs::create_directories(work);
    std::ofstream(work / "lid.csv") << "t,value\n0,1\n1,2\n";
    const fs::path previous = fs::current_path();
    fs::current_path(work);
    Params run = baseline_defaults();
    run.merge(get_test_case("Lid Driven Cavity").defaults);
    run.set("u_in_table", "lid.csv");
    const std::string saved = save_case(get_test_case("Lid Driven Cavity"), run, (work / "saved").string(), "Saved With Table");
    CHECK(fs::exists(work / "saved" / "lid.csv"));
    const CaseFile back = load_case_file((work / "saved").string());
    CHECK(back.params.str("u_in_table") == "lid.csv");
    run.set("u_in_table", "absent.csv");
    bool refused = false;
    try {
        save_case(get_test_case("Lid Driven Cavity"), run, (work / "saved2").string(), "Saved With Table 2");
    } catch (const std::runtime_error& e) { refused = std::string(e.what()).find("not found") != std::string::npos; }
    CHECK(refused);
    CHECK(!fs::exists(work / "saved2" / "case.toml"));
    fs::current_path(previous);
    (void)saved;
}
