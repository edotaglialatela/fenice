// AMR in three dimensions, phase Q7 (docs/design/amr-octree.md): the files of an octree run. VTK hexahedra, the checkpoint (leaf_k,
// five state columns), the guards of a resume, the hot start from a uniform and from an adaptive checkpoint.
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_amr_common.hpp"

#include <fstream>
#include <sstream>
#include <string>

namespace {

using BCs = std::array<const char*, 6>;
using KV = std::vector<std::pair<const char*, const char*>>;

const BCs kPeriodic = {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"};

std::string slurp(const fs::path& p) {
    std::ifstream in(p.string());
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The numbers of the ASCII <DataArray> called `name`
std::vector<double> data_of(const std::string& xml, const std::string& name) {
    const size_t a = xml.find("Name=\"" + name + "\"");
    const size_t b = xml.find('>', a) + 1;
    const size_t e = xml.find("</DataArray>", b);
    std::istringstream in(xml.substr(b, e - b));
    std::vector<double> v;
    double x;
    while (in >> x) v.push_back(x);
    return v;
}

// A 3-D adaptive run set up (not run) on the Sod defaults: base `n` cells per axis, three levels, periodic everywhere
void setup_oct(Simulation& sim, const fs::path& root, KV kv = {}) {
    const std::string r = root.string();
    KV all = {{"use_amr", "true"}, {"nx", "6"}, {"ny", "6"}, {"nz", "6"}, {"Lx", "1.2"}, {"Ly", "1.2"}, {"Lz", "1.2"},
              {"amr_max_level", "2"}, {"amr_criterion", "1e9"}, {"output_root", r.c_str()}, {"reconstruction", "muscl"}};
    all.insert(all.end(), kv.begin(), kv.end());
    setup_3d(sim, kPeriodic, all);
}

// The leaves of an octree after a few random adapts, with a state whose five components are all nonzero and vary along every axis
void fill_adapted(Simulation& sim, unsigned seed) {
    CellMesh& m = *sim.cells;
    m.U.assign(static_cast<size_t>(m.num_leaves()), State{});
    random_adapt(m, seed, 3, 25, 10);
    m.U.resize(static_cast<size_t>(m.num_leaves()));
    for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state3(sim.fluid, m.xc(c), m.yc(c), m.zc(c));
}

}   // namespace

TEST(cellio3d_vtu_hexahedra_have_the_geometry_and_the_fields_of_the_leaves) {
    const EOS eos = EOS::ideal(1.4);
    CellMesh m = CellMesh::make_3d(1.2, 0.8, 0.6, 3, 2, 2, 2, false, false, false, 0.1, -0.2, 0.3);
    random_adapt(m, 7, 3, 30, 5);
    CHECK(m.max_level == 2 && m.num_leaves() > 12);
    m.U.resize(static_cast<size_t>(m.num_leaves()));
    for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state3(eos, m.xc(c), m.yc(c), m.zc(c));
    std::vector<int> all;
    for (int c = 0; c < m.num_leaves(); ++c) all.push_back(c);
    const fs::path dir = tmp_dir() / "vtu3d";
    fs::remove_all(dir);
    VTKExporter ex(dir.string(), false, false);
    ex.export_cells("oct", eos, m, all, 0.0, 4);
    const std::string xml = slurp(dir / "oct_00004.vtu");
    const int n = m.num_leaves();
    CHECK(xml.find("NumberOfPoints=\"" + std::to_string(8 * n) + "\" NumberOfCells=\"" + std::to_string(n) + "\"") != std::string::npos);
    const std::vector<double> pts = data_of(xml, "Points"), types = data_of(xml, "types"), offs = data_of(xml, "offsets"),
                              conn = data_of(xml, "connectivity");
    CHECK(static_cast<int>(pts.size()) == 24 * n && static_cast<int>(types.size()) == n && static_cast<int>(conn.size()) == 8 * n);
    bool ok_types = true, ok_off = true, ok_conn = true, ok_geo = true;
    double volume = 0.0;
    for (int c = 0; c < n; ++c) {
        ok_types = ok_types && types[c] == 12;   // VTK_HEXAHEDRON
        ok_off = ok_off && offs[c] == 8.0 * (c + 1);
        for (int q = 0; q < 8; ++q) ok_conn = ok_conn && conn[8 * c + q] == 8.0 * c + q;
        const int l = m.level[c];
        const double x0 = m.x0 + m.ci[c] * m.dx(l), x1 = x0 + m.dx(l), y0 = m.y0 + m.cj[c] * m.dy(l), y1 = y0 + m.dy(l),
                     z0 = m.z0 + m.ck[c] * m.dz(l), z1 = z0 + m.dz(l);
        // the bottom face counter-clockwise seen from +z, then the same corners on top (a twisted order would still bound the same box)
        const double want[24] = {x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1};
        for (int q = 0; q < 24; ++q) ok_geo = ok_geo && std::abs(pts[24 * c + q] - want[q]) < 1e-6;
        volume += (pts[24 * c + 3] - pts[24 * c]) * (pts[24 * c + 7] - pts[24 * c + 1]) * (pts[24 * c + 14] - pts[24 * c + 2]);
    }
    CHECK(ok_types && ok_off && ok_conn && ok_geo);
    CHECK_NEAR(volume, 1.2 * 0.8 * 0.6, 1e-5);   // the hexahedra tile the domain
    // the cell fields: the third velocity component is w, the Mach number includes it
    const std::vector<double> vel = data_of(xml, "Velocity"), mach = data_of(xml, "Mach"), rho = data_of(xml, "Density");
    bool ok_field = static_cast<int>(vel.size()) == 3 * n;
    for (int c = 0; c < n && ok_field; ++c) {
        const State w = eos.cons_to_prim(m.U[c]);
        const double cs = eos.sound_speed(w[3], w[0]);
        ok_field = std::abs(vel[3 * c + 2] - w[4]) < 1e-4 * (1 + std::abs(w[4])) && std::abs(vel[3 * c] - w[1]) < 1e-4 * (1 + std::abs(w[1])) &&
                   std::abs(mach[c] - std::sqrt(w[1] * w[1] + w[2] * w[2] + w[4] * w[4]) / cs) < 1e-4 && std::abs(rho[c] - w[0]) < 1e-4;
    }
    CHECK(ok_field);
    CHECK(xml.find("Name=\"AMR_Level\"") != std::string::npos);
}

TEST(cellio3d_checkpoint_round_trip_and_guards) {
    const fs::path root = tmp_dir() / "oct_ck";
    fs::remove_all(root);
    Simulation a;
    setup_oct(a, root);
    a.initialize_field();
    fill_adapted(a, 11);
    CHECK(a.cells->max_level == 2 && a.cells->level_begin[3] > a.cells->level_begin[2]);
    a.t = 0.125;
    a.save_checkpoint(7);
    const NpzArchive z = read_npz(a.checkpoint_path());
    CHECK(z.count("leaf_k") && z.at("U").shape.size() == 2 && z.at("U").shape[1] == kNVar);
    CHECK(z.at("nz").scalar() == 6.0 && z.at("Lz").scalar() == 1.2);
    // the resumed run holds the same leaves, the same state (all five components), the time, the step and the speed bound
    Simulation b;
    setup_oct(b, root, {{"resume", "true"}});
    b.initialize_field();
    CHECK(b.start_step == 7 && b.t == 0.125);
    CHECK(b.cells->level == a.cells->level && b.cells->ci == a.cells->ci && b.cells->cj == a.cells->cj && b.cells->ck == a.cells->ck);
    CHECK(b.cells->U == a.cells->U);
    double wmax = 0.0;
    for (const State& u : b.cells->U) wmax = std::max(wmax, std::abs(u[4]));
    CHECK(wmax > 0.01);   // rho w really travelled
    // a resume on another base mesh is refused (it is a hot start), and so is a 2-D run on this file
    Simulation c;
    setup_oct(c, root, {{"resume", "true"}, {"nx", "8"}});
    bool threw = false;
    try {
        c.initialize_field();
    } catch (const std::runtime_error& e) { threw = std::string(e.what()).find("another size") != std::string::npos; }
    CHECK(threw);
    // a 2-D adaptive file is not resumed by a 3-D run, nor the 3-D file by a 2-D run
    const fs::path root2 = tmp_dir() / "oct_ck2d";
    fs::remove_all(root2);
    const std::string r2 = root2.string();
    Simulation d;
    d.setup(params_for("Sod Shock Tube", {{"use_amr", "true"}, {"nx", "6"}, {"ny", "6"}, {"amr_max_level", "2"}, {"output_root", r2.c_str()}}), false);
    d.initialize_field();
    d.save_checkpoint(3);
    Simulation e3;
    setup_oct(e3, root2, {{"resume", "true"}});
    threw = false;
    try {
        e3.initialize_field();
    } catch (const std::runtime_error& e) { threw = std::string(e.what()).find("2-D adaptive run") != std::string::npos; }
    CHECK(threw);
    Simulation f;
    f.setup(params_for("Sod Shock Tube", {{"use_amr", "true"}, {"nx", "6"}, {"ny", "6"}, {"amr_max_level", "2"}, {"resume", "true"}, {"output_root", r2.c_str()}}), false);
    std::error_code ec;
    fs::copy_file(a.checkpoint_path(), f.checkpoint_path(), fs::copy_options::overwrite_existing, ec);
    threw = false;
    try {
        f.initialize_field();
    } catch (const std::runtime_error& e) { threw = std::string(e.what()).find("3-D adaptive run") != std::string::npos; }
    CHECK(threw);
}

TEST(cellio3d_hot_start_from_a_uniform_and_from_an_adaptive_checkpoint) {
    // (1) a uniform 3-D checkpoint on the base mesh: every level-0 leaf takes the value of its cell (the trilinear weights are 0)
    const fs::path root = tmp_dir() / "oct_hot";
    fs::remove_all(root);
    const std::string r = root.string();
    Simulation u;
    {
        KV kv = {{"use_amr", "false"}, {"nx", "6"}, {"ny", "6"}, {"nz", "6"}, {"Lx", "1.2"}, {"Ly", "1.2"}, {"Lz", "1.2"}, {"output_root", r.c_str()}};
        setup_3d(u, kPeriodic, kv);
        u.initialize_field();
        for (int k = 0; k < 6; ++k)
            for (int j = 0; j < 6; ++j)
                for (int i = 0; i < 6; ++i) u.U[static_cast<size_t>(u.mesh.cell(i, j, k))] = test_state3(u.fluid, u.mesh.xc(i), u.mesh.yc(j), u.mesh.zc(k));
        u.save_checkpoint(0);
    }
    const std::string src = u.checkpoint_path();
    const fs::path root_t = tmp_dir() / "oct_hot_t";
    fs::remove_all(root_t);
    Simulation t;
    setup_oct(t, root_t, {{"use_hot_start", "true"}, {"hot_start_file", src.c_str()}});
    t.initialize_field();
    CHECK(t.cells->num_leaves() == 216 && t.cells->max_level == 2);
    bool same = true;
    for (int c = 0; c < t.cells->num_leaves(); ++c) {
        CHECK(t.cells->level[c] == 0);
        same = same && t.cells->U[c] == u.U[static_cast<size_t>(u.mesh.cell(t.cells->ci[c], t.cells->cj[c], t.cells->ck[c]))];
    }
    CHECK(same);
    // (2) an adaptive checkpoint: the finest-level field is sampled at the leaf centres, so a target leaf of the finest level that is
    // also a leaf of the source takes the source's value
    const fs::path root_a = tmp_dir() / "oct_hot_a";
    fs::remove_all(root_a);
    Simulation a;
    setup_oct(a, root_a);
    a.initialize_field();
    fill_adapted(a, 5);
    a.save_checkpoint(2);
    const fs::path root_b = tmp_dir() / "oct_hot_b";
    fs::remove_all(root_b);
    Simulation b;
    setup_oct(b, root_b, {{"use_hot_start", "true"}, {"hot_start_file", a.checkpoint_path().c_str()}, {"amr_init_levels", "2"}, {"amr_criterion", "1e-9"}});
    b.initialize_field();
    int shared = 0;
    bool equal = true;
    for (int c = 0; c < b.cells->num_leaves(); ++c) {
        if (b.cells->level[c] != 2) continue;
        const int s = a.cells->find(2, b.cells->ci[c], b.cells->cj[c], b.cells->ck[c]);
        if (s < 0) continue;
        ++shared;
        for (int q = 0; q < kNVar; ++q) equal = equal && std::abs(b.cells->U[c][q] - a.cells->U[s][q]) <= 1e-12 * (1.0 + std::abs(a.cells->U[s][q]));
    }
    CHECK(shared > 20 && equal);
    // an adaptive 3-D source cannot seed a 2-D run
    Simulation d;
    d.setup(params_for("Sod Shock Tube", {{"nx", "6"}, {"ny", "6"}, {"use_hot_start", "true"}, {"hot_start_file", a.checkpoint_path().c_str()},
                                        {"output_root", r.c_str()}}), false);
    bool threw = false;
    try {
        d.initialize_field();
    } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}
