// AMR in three dimensions, phase Q2 (docs/design/amr-octree.md): the convective residual of CellFluxEngine on octree leaves.
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_amr_common.hpp"

#include <algorithm>
#include <string>

namespace {

// Smooth 3-D state with every velocity component nonzero (primitive [rho, u, v, p, w])
State test_state3(const EOS& eos, double x, double y, double z) {
    return eos.prim_to_cons({1.0 + 0.1 * std::sin(1.3 * x) * std::cos(0.9 * y) * std::cos(0.7 * z + 0.2),
                             1.0 + 0.2 * std::sin(0.7 * y + 0.3 * z), 0.15 * std::cos(1.1 * x - 0.4 * z),
                             17.0 + 0.5 * std::cos(0.6 * x + 0.4 * y + 0.5 * z), 0.12 * std::sin(0.8 * x + 0.5 * y)});
}

// A 3-D Simulation of the face-engine rig (Sod defaults, 9 x 7 x 5 cells unless overridden) with the six boundary types.
void setup_3d(Simulation& sim, const std::array<const char*, 6>& bc, std::vector<std::pair<const char*, const char*>> kv) {
    std::vector<std::pair<const char*, const char*>> all = {{"nz", "5"},          {"Lz", "0.9"},        {"bc_left", bc[0]},
                                                           {"bc_right", bc[1]},  {"bc_bottom", bc[2]}, {"bc_top", bc[3]},
                                                           {"bc_front", bc[4]},  {"bc_back", bc[5]}};
    all.insert(all.end(), kv.begin(), kv.end());
    Params p = baseline_defaults();
    p.merge(get_test_case("Sod Shock Tube").defaults);
    p.set("test_case", "Sod Shock Tube");
    p.set("output_root", tmp_dir().string());
    p.set("use_mpi", "false");
    for (const auto& e : std::vector<std::pair<const char*, const char*>>{{"nx", "9"}, {"ny", "7"}, {"Lx", "1.7"}, {"Ly", "1.1"},
                                                                         {"u_in", "0.7"}, {"v_in", "0.2"}, {"rho_in", "1.2"},
                                                                         {"p_out", "1.3"}})
        p.set(e.first, e.second);
    for (const auto& e : all) p.set(e.first, e.second);
    sim.setup(SimParams::from(p), false);
}

bool periodic_axis(const Simulation& sim, int a) { return sim.boundaries.type(static_cast<Face>(2 * a)) == BCType::Periodic; }

CellMesh octree_of(const Simulation& sim, int max_level) {
    const Mesh& u = sim.mesh;
    return CellMesh::make_3d(u.Lx, u.Ly, u.Lz, u.nx, u.ny, u.nz, max_level, periodic_axis(sim, 0), periodic_axis(sim, 1),
                             periodic_axis(sim, 2), u.x0, u.y0, u.z0);
}

void random_adapt(CellMesh& m, unsigned seed, int rounds, int pr = 15, int pc = 30) {
    std::mt19937 rng(seed);
    for (int round = 0; round < rounds; ++round) {
        std::vector<char> r(static_cast<size_t>(m.num_leaves())), c(static_cast<size_t>(m.num_leaves()));
        for (int k = 0; k < m.num_leaves(); ++k) {
            r[k] = static_cast<int>(rng() % 100) < pr;
            c[k] = static_cast<int>(rng() % 100) < pc;
        }
        m.adapt(r, c);
    }
}

double max_abs(const Field& R, int q) {
    double s = 0.0;
    for (const State& r : R) s = std::max(s, std::abs(r[q]));
    return s;
}

}   // namespace

TEST(cellflux3d_level0_matches_uniform_residual_bitwise) {
    using BCs = std::array<const char*, 6>;
    const BCs sets[] = {{"wall", "wall", "wall", "wall", "wall", "wall"},
                        {"transmissive", "outlet", "no_slip", "moving_wall", "wall", "transmissive"},
                        {"vel_inlet", "outlet", "wall", "transmissive", "periodic", "periodic"},
                        {"periodic", "periodic", "wall", "wall", "transmissive", "no_slip"},
                        {"wall", "transmissive", "periodic", "periodic", "wall", "wall"},
                        {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"}};
    struct Scheme {
        const char *muscl, *limiter, *riemann, *recon;
    };
    const Scheme schemes[] = {{"false", "Van Leer", "HLLC", "none"},     {"true", "Van Leer", "HLLC", "muscl"},
                              {"true", "Minmod", "HLLC", "muscl"},        {"true", "MC", "lowmach", "muscl"},
                              {"true", "Koren", "preconditioned", "muscl"}, {"true", "Van Leer", "HLLC", "weno3"}};
    int n_cases = 0;
    for (const BCs& bc : sets)
        for (const Scheme& sc : schemes)
            for (const char* bo : {"1", "2"}) {
                if (std::string(sc.muscl) == "false" && std::string(bo) == "2") continue;
                Simulation sim;
                setup_3d(sim, bc,
                         {{"muscl_limiter", sc.limiter}, {"riemann_solver", sc.riemann},
                          {"reconstruction", sc.recon}, {"boundary_order", bo}});
                const Mesh& u = sim.mesh;
                CellMesh cm = octree_of(sim, 0);
                Field Uu(static_cast<size_t>(u.num_cells()));
                for (int k = 0; k < u.nz; ++k)
                    for (int j = 0; j < u.ny; ++j)
                        for (int i = 0; i < u.nx; ++i) Uu[u.cell(i, j, k)] = test_state3(sim.fluid, u.xc(i), u.yc(j), u.zc(k));
                cm.U.resize(static_cast<size_t>(cm.num_leaves()));
                for (int c = 0; c < cm.num_leaves(); ++c) cm.U[c] = Uu[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
                Field Ru, Rc;
                const double Su = sim.residual(Uu, Ru);
                CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
                eng.build(cm);
                const double Sc = eng.residual(cm, cm.U, Rc);
                Field Rperm(Ru.size());
                for (int c = 0; c < cm.num_leaves(); ++c) Rperm[c] = Ru[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
                const bool same = bits_equal(Rperm, Rc) && Su == Sc;
                CHECK(same);
                if (!same)
                    std::printf("    case bc %s/%s/%s/%s/%s/%s  muscl %s %s %s %s bo %s\n", bc[0], bc[1], bc[2], bc[3], bc[4], bc[5],
                                sc.muscl, sc.limiter, sc.riemann, sc.recon, bo);
                ++n_cases;
            }
    CHECK(n_cases == 66);

    // above kOmpMinWork the loops run in parallel: same bits
    for (const BCs& bc : {BCs{"wall", "outlet", "no_slip", "moving_wall", "transmissive", "wall"},
                          BCs{"periodic", "periodic", "wall", "wall", "periodic", "periodic"}}) {
        Simulation sim;
        setup_3d(sim, bc, {{"nx", "24"}, {"ny", "20"}, {"nz", "10"}, {"reconstruction", "muscl"}, {"muscl_limiter", "MC"}});
        const Mesh& u = sim.mesh;
        CellMesh cm = octree_of(sim, 0);
        CHECK(cm.num_leaves() >= kOmpMinWork);
        Field Uu(static_cast<size_t>(u.num_cells()));
        for (int k = 0; k < u.nz; ++k)
            for (int j = 0; j < u.ny; ++j)
                for (int i = 0; i < u.nx; ++i) Uu[u.cell(i, j, k)] = test_state3(sim.fluid, u.xc(i), u.yc(j), u.zc(k));
        cm.U.resize(static_cast<size_t>(cm.num_leaves()));
        for (int c = 0; c < cm.num_leaves(); ++c) cm.U[c] = Uu[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
        Field Ru, Rc, Rperm(Uu.size());
        const double Su = sim.residual(Uu, Ru);
        CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        eng.build(cm);
        const double Sc = eng.residual(cm, cm.U, Rc);
        for (int c = 0; c < cm.num_leaves(); ++c) Rperm[c] = Ru[u.cell(cm.ci[c], cm.cj[c], cm.ck[c])];
        CHECK(bits_equal(Rperm, Rc) && Su == Sc);
    }
}

TEST(cellflux3d_invariant_axis_equals_the_2d_engine) {
    // A field that does not depend on axis a, on an octree that is the 2-D tree of the other two axes extruded along a (periodic):
    // the 3-D residual equals the 2-D engine's on the 2-D tree, to round-off (the 3-D gather multiplies by the face area and divides
    // by the volume, and sums four fine faces in pairs). Exercises the 3-D jump closure across x-, y- and z-faces and both
    // transverse offsets of the interpolated cells; the a-momentum residual is exactly zero.
    struct Scheme {
        const char *muscl, *recon, *bo;
    };
    const Scheme schemes[] = {{"false", "none", "1"}, {"true", "muscl", "1"}, {"true", "muscl", "2"}, {"true", "weno3", "2"}};
    int checked_jumps = 0;
    for (int a = 0; a < 3; ++a) {
        const int p = a == 0 ? 1 : 0, q = a == 2 ? 1 : 2;   // the in-plane axes (2-D x <- p, 2-D y <- q)
        const int n2[2] = {9, 7};
        const double L2[2] = {1.7, 1.1}, La = 0.6;
        const char* bc2[4] = {"transmissive", "wall", "wall", "outlet"};   // left, right, bottom, top of the 2-D problem
        for (const Scheme& sc : schemes) {
            Simulation s2;
            s2.setup(cell_flux_params({{"bc_left", bc2[0]}, {"bc_right", bc2[1]}, {"bc_bottom", bc2[2]}, {"bc_top", bc2[3]},
                                       {"reconstruction", sc.recon}, {"boundary_order", sc.bo}}),
                     false);
            // the 3-D problem: axis p carries the 2-D x data, q the 2-D y data, a is periodic
            std::array<const char*, 6> bc3;
            bc3[2 * p] = bc2[0];
            bc3[2 * p + 1] = bc2[1];
            bc3[2 * q] = bc2[2];
            bc3[2 * q + 1] = bc2[3];
            bc3[2 * a] = bc3[2 * a + 1] = "periodic";
            std::string n3[3], L3[3];
            n3[p] = std::to_string(n2[0]);
            n3[q] = std::to_string(n2[1]);
            n3[a] = "2";
            char buf[3][32];
            for (int ax = 0; ax < 3; ++ax) {
                const double L = ax == p ? L2[0] : (ax == q ? L2[1] : La);
                std::snprintf(buf[ax], sizeof(buf[ax]), "%.17g", L);
                L3[ax] = buf[ax];
            }
            Simulation s3;
            setup_3d(s3, bc3,
                     {{"nx", n3[0].c_str()}, {"ny", n3[1].c_str()}, {"nz", n3[2].c_str()}, {"Lx", L3[0].c_str()}, {"Ly", L3[1].c_str()},
                      {"Lz", L3[2].c_str()}, {"reconstruction", sc.recon}, {"boundary_order", sc.bo}});
            // the 2-D tree, then its extrusion
            CellMesh m2(L2[0], L2[1], n2[0], n2[1], 2, false, false, s2.mesh.x0, s2.mesh.y0);
            random_adapt(m2, 41u + static_cast<unsigned>(a), 4, 18, 20);
            CellMesh m3 = octree_of(s3, 2);
            std::vector<int> lv, ii, jj, kk;
            for (int c = 0; c < m2.num_leaves(); ++c) {
                const int l = m2.level[c], layers = 2 << l;
                for (int t = 0; t < layers; ++t) {
                    int x[3];
                    x[p] = m2.ci[c];
                    x[q] = m2.cj[c];
                    x[a] = t;
                    lv.push_back(l);
                    ii.push_back(x[0]);
                    jj.push_back(x[1]);
                    kk.push_back(x[2]);
                }
            }
            m3.assign_leaves(lv, ii, jj, kk);
            check_cell_mesh_nd(m3);
            // states: the 2-D field, its in-plane velocity on axes p, q, none along a
            m2.U.resize(static_cast<size_t>(m2.num_leaves()));
            for (int c = 0; c < m2.num_leaves(); ++c) m2.U[c] = test_state(s2.fluid, m2.xc(c), m2.yc(c));
            m3.U.resize(static_cast<size_t>(m3.num_leaves()));
            std::vector<int> to2(static_cast<size_t>(m3.num_leaves()));
            for (int c = 0; c < m3.num_leaves(); ++c) {
                const int x[3] = {m3.ci[c], m3.cj[c], m3.ck[c]};
                const int c2 = m2.find(m3.level[c], x[p], x[q]);
                CHECK(c2 >= 0);
                to2[static_cast<size_t>(c)] = c2;
                const State W2 = s2.fluid.cons_to_prim(m2.U[c2]);
                State W3{};
                W3[0] = W2[0];
                W3[3] = W2[3];
                W3[kMom[p]] = W2[1];
                W3[kMom[q]] = W2[2];
                W3[kMom[a]] = 0.0;
                m3.U[c] = s3.fluid.prim_to_cons(W3);
            }
            CellFluxEngine e2(s2.fluid, s2.riemann, s2.muscl.get(), s2.boundaries), e3(s3.fluid, s3.riemann, s3.muscl.get(), s3.boundaries);
            e2.build(m2);
            e3.build(m3);
            Field R2, R3;
            const double S2 = e2.residual(m2, m2.U, R2), S3 = e3.residual(m3, m3.U, R3);
            CHECK(std::abs(S2 - S3) <= 1e-14 * S2);
            const int q3[4] = {0, kMom[p], kMom[q], 3}, q2[4] = {0, 1, 2, 3};
            double worst = 0.0;
            for (int k = 0; k < 4; ++k) {
                const double scale = max_abs(R2, q2[k]);
                for (int c = 0; c < m3.num_leaves(); ++c) {
                    const double d = std::abs(R3[c][q3[k]] - R2[static_cast<size_t>(to2[c])][q2[k]]);
                    worst = std::max(worst, d / scale);
                }
            }
            CHECK(worst <= 1e-12);
            if (worst > 1e-12) std::printf("    axis %d scheme %s %s bo %s: worst relative difference %.3e\n", a, sc.muscl, sc.recon, sc.bo, worst);
            for (int c = 0; c < m3.num_leaves(); ++c) CHECK(R3[c][kMom[a]] == 0.0);
            for (const CellFace& f : m3.faces) checked_jumps += f.kind == CellFaceKind::CoarseFine && f.axis != a;
        }
    }
    CHECK(checked_jumps > 1000);
}

TEST(cellflux3d_free_stream_is_exact_across_level_jumps) {
    using BCs = std::array<const char*, 6>;
    for (const BCs& bc : {BCs{"transmissive", "transmissive", "transmissive", "transmissive", "transmissive", "transmissive"},
                          BCs{"periodic", "periodic", "transmissive", "transmissive", "periodic", "periodic"},
                          BCs{"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"}})
        for (const char* recon : {"none", "muscl", "weno3"})
            for (const char* bo : {"1", "2"}) {
                const bool muscl = std::string(recon) != "none";
                if (!muscl && std::string(bo) == "2") continue;
                Simulation sim;
                setup_3d(sim, bc, {{"nx", "4"}, {"ny", "3"}, {"nz", "3"}, {"reconstruction", recon}, {"muscl_limiter", "Koren"}, {"boundary_order", bo}});
                CellMesh m = octree_of(sim, 2);
                random_adapt(m, 900u, 4);
                int jumps = 0;
                for (const CellFace& f : m.faces) jumps += f.kind == CellFaceKind::CoarseFine;
                CHECK(jumps > 50);
                CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
                eng.build(m);
                CHECK(eng.num_virtual() > 0);
                m.U.assign(static_cast<size_t>(m.num_leaves()), sim.fluid.prim_to_cons({1.3, 0.8, -0.45, 2.1, 0.35}));
                Field R;
                eng.residual(m, m.U, R);
                // WENO3-Z does not reproduce a constant to the last bit (its candidates are not exact) while a first-order boundary
                // face uses the round-tripped cell state: the leaves on a physical side then differ by round-off, as on a uniform mesh.
                // Every other leaf, the level jumps included, is exact.
                const bool skip_edge = std::string(recon) == "weno3" && std::string(bo) == "1";
                double worst = 0.0;
                int interior = 0;
                for (int c = 0; c < m.num_leaves(); ++c) {
                    bool edge = false;
                    for (int s = 0; s < m.sides(); ++s) edge = edge || m.neighbour(c, static_cast<Face>(s)).kind == CellNeighbour::Kind::Boundary;
                    if (skip_edge && edge) continue;
                    ++interior;
                    for (int q = 0; q < kNVar; ++q) worst = std::max(worst, std::abs(R[c][q]));
                }
                CHECK(interior > m.num_leaves() / 3);
                CHECK(worst == 0.0);
                if (worst != 0.0) std::printf("    bc %s/%s/%s recon %s bo %s: worst %.3e\n", bc[0], bc[2], bc[4], recon, bo, worst);
            }
}

TEST(cellflux3d_residual_is_conservative_on_adapted_meshes) {
    using BCs = std::array<const char*, 6>;
    const BCs per = {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"};
    for (int cfg = 0; cfg < 4; ++cfg) {
        Simulation sim;
        setup_3d(sim, per, {{"nx", "4"}, {"ny", "3"}, {"nz", "3"}, {"reconstruction", cfg == 2 ? "weno3" : (cfg < 3 ? "muscl" : "none")}, {"muscl_limiter", "MC"},
                            {"riemann_solver", cfg % 2 ? "lowmach" : "HLLC"}});
        CellMesh m = octree_of(sim, 2);
        random_adapt(m, 300u + static_cast<unsigned>(cfg), 4);
        CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        eng.build(m);
        m.U.resize(static_cast<size_t>(m.num_leaves()));
        for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state3(sim.fluid, m.xc(c), m.yc(c), m.zc(c));
        Field R;
        CHECK(eng.residual(m, m.U, R) > 0.5);
        for (int q = 0; q < kNVar; ++q) {
            double sum = 0.0, mag = 0.0;
            for (int c = 0; c < m.num_leaves(); ++c) {
                sum += R[c][q] * m.volume(c);
                mag += std::abs(R[c][q]) * m.volume(c);
            }
            CHECK(mag > 1e-6);
            CHECK(std::abs(sum) <= 1e-13 * mag);   // each face flux is used by both sides with the same area
        }
    }
}

TEST(cellflux3d_incremental_build_equals_full_build_and_threads_agree) {
    using BCs = std::array<const char*, 6>;
    Simulation sim;
    setup_3d(sim, BCs{"transmissive", "wall", "periodic", "periodic", "wall", "outlet"},
             {{"nx", "6"}, {"ny", "6"}, {"nz", "5"}, {"reconstruction", "muscl"}, {"boundary_order", "2"}});
    CellMesh m = octree_of(sim, 2);
    CellFluxEngine inc(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    inc.build(m);
    std::mt19937 rng(77);
    for (int round = 0; round < 6; ++round) {
        std::vector<char> r(static_cast<size_t>(m.num_leaves())), c(static_cast<size_t>(m.num_leaves()));
        for (int k = 0; k < m.num_leaves(); ++k) {
            r[k] = rng() % 100 < 12;
            c[k] = rng() % 100 < 40;
        }
        if (!m.adapt(r, c)) continue;
        inc.build(m);   // incremental: right after the adapt of the mesh it was built for
        CellFluxEngine full(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        full.build(m);
        CHECK(inc.same_tables(full));
    }
    CHECK(m.num_leaves() >= kOmpMinWork);   // the loops of the residual run in parallel
    m.U.resize(static_cast<size_t>(m.num_leaves()));
    for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state3(sim.fluid, m.xc(c), m.yc(c), m.zc(c));
    Field R1, Rn;
    omp_set_num_threads(1);
    const double s1 = inc.residual(m, m.U, R1);
    omp_set_num_threads(omp_get_num_procs());
    const double sn = inc.residual(m, m.U, Rn);
    CHECK(bits_equal(R1, Rn) && s1 == sn);
}

namespace {

// Monotone polynomial field on [0, 4]^d whose derivatives along every axis stay away from 0 (no extremum, so the limiters do not
// clip; a first version with u_x = 0.1 x vanishing at x = 0 made the limiter clip along the domain edges and the far-field order
// drop to 1): primitive [rho, u, v, p, w] (z = 0 in 2-D)
State poly_prim(double x, double y, double z) {
    const double X = x + 1.0, Z = z + 1.0;
    return {1.0 + 0.2 * x + 0.1 * y + 0.05 * x * y + 0.03 * Z * Z, 0.5 + 0.1 * y + 0.05 * X * X + 0.02 * z,
            0.3 + 0.1 * x + 0.04 * Z * Z + 0.03 * y, 10.0 + 0.5 * x + 0.3 * y + 0.2 * z + 0.1 * y * y, -0.2 + 0.1 * y + 0.05 * X * Z};
}

// Physical Euler flux along `axis` of the primitive state W (2-D: w is ignored by the 2-D flux, it is 0 there)
State euler_flux(const EOS& eos, const State& W, int axis) {
    const State U = eos.prim_to_cons(W);
    const double un = W[kMom[axis]];
    State F;
    for (int q = 0; q < kNVar; ++q) F[q] = U[q] * un;
    F[kMom[axis]] += W[3];
    F[3] += W[3] * un;
    return F;
}

// -div F at a point by central differences of the physical flux (step 1e-5: error far below the truncation errors measured)
State exact_residual(const EOS& eos, int dim, double x, double y, double z) {
    const double d = 1e-5;
    State r{};
    for (int a = 0; a < dim; ++a) {
        const double e[3] = {a == 0 ? d : 0.0, a == 1 ? d : 0.0, a == 2 ? d : 0.0};
        State Wp = poly_prim(x + e[0], y + e[1], z + e[2]), Wm = poly_prim(x - e[0], y - e[1], z - e[2]);
        if (dim == 2) Wp[4] = Wm[4] = 0.0;
        const State Fp = euler_flux(eos, Wp, a), Fm = euler_flux(eos, Wm, a);
        for (int q = 0; q < kNVar; ++q) r[q] -= (Fp[q] - Fm[q]) / (2.0 * d);
    }
    return r;
}

// Mean normalised truncation error of the residual near the jump (|signed max-norm distance to the refined box surface| < 2 h0)
// and far from it (coarse leaves outside the box, > 3 h0 from it and from the domain edge), on the base mesh n^d of [0, 4]^d with the box [1, 2]^d one level finer.
// With refine = false the same measures on the uniform base mesh (the far errors of the two agree: the jump's influence is local).
void truncation_errors(int dim, int n, double& near, double& far, bool refine = true) {
    Simulation sim;
    const std::string ns = std::to_string(n);
    if (dim == 2) {
        sim.setup(cell_flux_params({{"nx", ns.c_str()}, {"ny", ns.c_str()}, {"Lx", "4"}, {"Ly", "4"}, {"bc_left", "transmissive"},
                                    {"bc_right", "transmissive"}, {"bc_bottom", "transmissive"}, {"bc_top", "transmissive"},
                                    {"reconstruction", "muscl"}, {"muscl_limiter", "Van Leer"}}),
                  false);
    } else {
        setup_3d(sim, {"transmissive", "transmissive", "transmissive", "transmissive", "transmissive", "transmissive"},
                 {{"nx", ns.c_str()}, {"ny", ns.c_str()}, {"nz", ns.c_str()}, {"Lx", "4"}, {"Ly", "4"}, {"Lz", "4"},
                  {"reconstruction", "muscl"}, {"muscl_limiter", "Van Leer"}});
    }
    CellMesh m = dim == 2 ? CellMesh(4.0, 4.0, n, n, 1, false, false, sim.mesh.x0, sim.mesh.y0) : octree_of(sim, 1);
    const auto centre = [&](int c, double x[3]) {   // relative to the origin of the mesh: the field and the box live on [0, 4]^d
        x[0] = m.xc(c) - m.x0;
        x[1] = m.yc(c) - m.y0;
        x[2] = dim == 3 ? m.zc(c) - m.z0 : 0.0;
    };
    const auto signed_dist = [&](const double x[3]) {   // max-norm signed distance to the box surface (< 0 inside)
        double s = -1e300;
        for (int a = 0; a < dim; ++a) s = std::max(s, std::max(1.0 - x[a], x[a] - 2.0));
        return s;
    };
    std::vector<char> r(static_cast<size_t>(m.num_leaves()));
    for (int c = 0; c < m.num_leaves(); ++c) {
        double x[3];
        centre(c, x);
        r[c] = signed_dist(x) < 0.0;
    }
    if (refine) m.adapt(r);
    m.U.resize(static_cast<size_t>(m.num_leaves()));
    for (int c = 0; c < m.num_leaves(); ++c) {
        double x[3];
        centre(c, x);
        State W = poly_prim(x[0], x[1], x[2]);
        if (dim == 2) W[4] = 0.0;
        m.U[c] = sim.fluid.prim_to_cons(W);
    }
    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    eng.build(m);
    Field R;
    eng.residual(m, m.U, R);
    Field Rex(R.size());
    State scale{};
    for (int c = 0; c < m.num_leaves(); ++c) {
        double x[3];
        centre(c, x);
        Rex[c] = exact_residual(sim.fluid, dim, x[0], x[1], x[2]);
        for (int q = 0; q < kNVar; ++q) scale[q] = std::max(scale[q], std::abs(Rex[c][q]));
    }
    const double h0 = 4.0 / n;
    double sn = 0.0, sf = 0.0;
    int cn = 0, cf = 0;
    for (int c = 0; c < m.num_leaves(); ++c) {
        double x[3];
        centre(c, x);
        bool inner = true;
        for (int a = 0; a < dim; ++a) inner = inner && x[a] > 3.0 * h0 && x[a] < 4.0 - 3.0 * h0;
        if (!inner) continue;
        double e = 0.0;
        for (int q = 0; q < (dim == 3 ? kNVar : 4); ++q)
            if (scale[q] > 0.0) e += std::abs(R[c][q] - Rex[c][q]) / scale[q];
        const double s = signed_dist(x);
        if (std::abs(s) < 2.0 * h0) {
            sn += e;
            ++cn;
        } else if (s > 3.0 * h0) {   // coarse leaves outside the box only (the same kind of cell at every n)
            sf += e;
            ++cf;
        }
    }
    near = sn / std::max(cn, 1);
    far = sf / std::max(cf, 1);
}

}   // namespace

TEST(cellflux3d_near_jump_truncation_table) {
    // Bands, fixed before the first run: (1) far from the jump the order (n = 32 -> 64) is >= 1.8 in both dimensions (the measure
    // itself); (2) the 3-D near-jump order is >= the 2-D one - 0.3; (3) the near / far ratio at n = 64 in 3-D is <= 3x the 2-D one.
    const int ns[3] = {16, 32, 64};
    double near[2][3], far[2][3];
    for (int d = 0; d < 2; ++d)
        for (int k = 0; k < 3; ++k) truncation_errors(d + 2, ns[k], near[d][k], far[d][k]);
    std::printf("    residual truncation error (mean, normalised), base n^d, box [1,2]^d of [0,4]^d one level finer (orders 16 -> 32, 32 -> 64):\n");
    for (int d = 0; d < 2; ++d) {
        std::printf("    %d-D  near jump:", d + 2);
        for (int k = 0; k < 3; ++k) std::printf(" %.3e", near[d][k]);
        std::printf("  orders %.2f %.2f   far:", std::log2(near[d][0] / near[d][1]), std::log2(near[d][1] / near[d][2]));
        for (int k = 0; k < 3; ++k) std::printf(" %.3e", far[d][k]);
        std::printf("  orders %.2f %.2f   near/far at 64: %.2f\n", std::log2(far[d][0] / far[d][1]), std::log2(far[d][1] / far[d][2]),
                    near[d][2] / far[d][2]);
    }
    for (int d = 0; d < 2; ++d) CHECK(std::log2(far[d][1] / far[d][2]) >= 1.8);
    CHECK(std::log2(near[1][1] / near[1][2]) >= std::log2(near[0][1] / near[0][2]) - 0.3);
    CHECK(near[1][2] / far[1][2] <= 3.0 * near[0][2] / far[0][2]);
}

TEST(cellflux3d_level_pieces_equal_the_whole_residual) {
    // The subcycled advance (Q3) uses the per-level pieces, which read the per-level lists (need_, lface_, the owned ranges):
    // compute_level_fluxes of every level + gather_level must give the residual's bits, and wave_speed its speed.
    using BCs = std::array<const char*, 6>;
    for (const char* recon : {"muscl", "weno3"}) {
        Simulation sim;
        setup_3d(sim, BCs{"transmissive", "wall", "periodic", "periodic", "wall", "outlet"},
                 {{"nx", "5"}, {"ny", "4"}, {"nz", "4"}, {"reconstruction", recon}, {"boundary_order", "2"}});
        CellMesh m = octree_of(sim, 2);
        for (const double frac : {0.6, 0.3}) {   // a level-1 region and a level-2 one inside it: all three levels present
            std::vector<char> r(static_cast<size_t>(m.num_leaves()));
            for (int c = 0; c < m.num_leaves(); ++c) r[c] = m.xc(c) - m.x0 < frac * m.Lx && m.zc(c) - m.z0 < 0.7 * m.Lz;
            m.adapt(r);
        }
        CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        eng.build(m);
        m.U.resize(static_cast<size_t>(m.num_leaves()));
        for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state3(sim.fluid, m.xc(c), m.yc(c), m.zc(c));
        Field R;
        const double S = eng.residual(m, m.U, R);
        int levels = 0;
        for (int l = 0; l <= m.max_level; ++l) {
            if (m.level_begin[l + 1] == m.level_begin[l]) continue;
            ++levels;
            eng.compute_level_fluxes(m, m.U, l);
            Field Rl;
            eng.gather_level(m, l, false, Rl);
            const Field slice(R.begin() + m.level_begin[l], R.begin() + m.level_begin[l + 1]);
            CHECK(bits_equal(Rl, slice));
        }
        CHECK(levels == 3);
        CHECK(eng.wave_speed(m, m.U) == S);
    }
}

TEST(cellflux3d_rejects_what_it_does_not_do_yet) {
    using BCs = std::array<const char*, 6>;
    const auto throws = [](const BCs& bc, std::vector<std::pair<const char*, const char*>> kv) {
        Simulation sim;
        setup_3d(sim, bc, kv);
        CellMesh m = octree_of(sim, 1);
        CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        if (sim.p.use_viscous) {
            CellPhysics ph;
            ph.ns = sim.ns.get();
            eng.set_physics(ph);
        }
        try {
            eng.build(m);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    const BCs walls = {"wall", "wall", "wall", "wall", "wall", "wall"};
    CHECK(!throws(walls, {{"reconstruction", "muscl"}}));
    CHECK(throws(walls, {{"reconstruction", "muscl"}, {"use_viscous", "true"}, {"mu", "0.01"}}));
    CHECK(throws(BCs{"massflow_inlet", "outlet", "wall", "wall", "wall", "wall"}, {{"reconstruction", "muscl"}}));
    CHECK(throws(BCs{"wall", "nonreflecting", "wall", "wall", "wall", "wall"}, {{"reconstruction", "muscl"}}));
    CHECK(throws(walls, {{"reconstruction", "weno5"}}));
}
