// Helpers shared by the AMR test files: the mesh checks, the random adapted meshes, the parity rig between the uniform solver and the
// face-list engine. Functions are inline (several files include this).
#pragma once
#include "test_common.hpp"


// Independent oracle: leaf boxes in integer coordinates of the finest level, every touching pair found by brute
// force (periodic axes by shifted copies); the mesh's own lookup tables are never used.
struct LeafBox {
    long long x0, x1, y0, y1;
    int level;
};

// axis, left, right, start of the segment (finer cell's own transverse start), kind (0 interior, 1 coarse-fine,
// 2 boundary), level, boundary side
using FaceTuple = std::tuple<int, int, int, long long, int, int, int>;

inline void check_cell_mesh(const CellMesh& m) {
    const int N = m.num_leaves(), M = m.max_level;
    const long long W = static_cast<long long>(m.nx) << M, H = static_cast<long long>(m.ny) << M;
    std::vector<LeafBox> box(N);
    long long area = 0;
    for (int c = 0; c < N; ++c) {
        const long long s = 1ll << (M - m.level[c]);
        box[c] = {m.ci[c] * s, (m.ci[c] + 1) * s, m.cj[c] * s, (m.cj[c] + 1) * s, m.level[c]};
        area += s * s;
    }
    CHECK(area == W * H);   // with the pairwise test below: the leaves tile the domain

    // order: level runs, Morton inside a level
    for (int c = 1; c < N; ++c) {
        CHECK(m.level[c] >= m.level[c - 1]);
        if (m.level[c] == m.level[c - 1])
            CHECK(morton_key(m.ci[c], m.cj[c]) > morton_key(m.ci[c - 1], m.cj[c - 1]));
    }
    CHECK(static_cast<int>(m.level_begin.size()) == M + 2 && m.level_begin[M + 1] == N);
    for (int l = 0; l <= M; ++l)
        for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c) CHECK(m.level[c] == l);

    const std::vector<long long> shx = m.periodic_x ? std::vector<long long>{-W, 0, W} : std::vector<long long>{0};
    const std::vector<long long> shy = m.periodic_y ? std::vector<long long>{-H, 0, H} : std::vector<long long>{0};
    std::vector<FaceTuple> want;
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b)
            for (long long sx : shx)
                for (long long sy : shy) {
                    if (a == b && sx == 0 && sy == 0) continue;
                    const LeafBox& A = box[a];
                    const LeafBox B = {box[b].x0 + sx, box[b].x1 + sx, box[b].y0 + sy, box[b].y1 + sy, box[b].level};
                    const long long ox = std::min(A.x1, B.x1) - std::max(A.x0, B.x0);   // overlap; 0 = touching
                    const long long oy = std::min(A.y1, B.y1) - std::max(A.y0, B.y0);
                    if (ox < 0 || oy < 0) continue;
                    CHECK(!(ox > 0 && oy > 0));   // no overlap
                    CHECK(std::abs(A.level - B.level) <= 1);   // corner 2:1 balance for every touching pair
                    const int kind = A.level == B.level ? 0 : 1, lev = std::max(A.level, B.level);
                    const LeafBox& fine = A.level >= B.level ? box[a] : box[b];
                    if (A.x1 == B.x0 && oy > 0) want.emplace_back(0, a, b, fine.y0, kind, lev, 0);
                    if (A.y1 == B.y0 && ox > 0) want.emplace_back(1, a, b, fine.x0, kind, lev, 0);
                }
    for (int a = 0; a < N; ++a) {
        const LeafBox& A = box[a];
        if (!m.periodic_x && A.x0 == 0) want.emplace_back(0, -1, a, A.y0, 2, A.level, static_cast<int>(Face::Left));
        if (!m.periodic_x && A.x1 == W) want.emplace_back(0, a, -1, A.y0, 2, A.level, static_cast<int>(Face::Right));
        if (!m.periodic_y && A.y0 == 0) want.emplace_back(1, -1, a, A.x0, 2, A.level, static_cast<int>(Face::Bottom));
        if (!m.periodic_y && A.y1 == H) want.emplace_back(1, a, -1, A.x0, 2, A.level, static_cast<int>(Face::Top));
    }
    std::vector<FaceTuple> got;
    for (const CellFace& f : m.faces) {
        const long long s = 1ll << (M - f.level);
        const int kind = f.kind == CellFaceKind::Interior ? 0 : (f.kind == CellFaceKind::CoarseFine ? 1 : 2);
        got.emplace_back(f.axis, f.left, f.right, f.tpos * s, kind, f.level, kind == 2 ? static_cast<int>(f.side) : 0);
        CHECK(f.length == (f.axis == 0 ? m.dy(f.level) : m.dx(f.level)));
    }
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    CHECK(want == got);   // exactly one face per shared segment, with the right orientation, kind and level

    for (size_t f = 1; f < m.faces.size(); ++f) CHECK(m.faces[f].level >= m.faces[f - 1].level);
    CHECK(static_cast<int>(m.face_level_begin.size()) == M + 2 &&
          m.face_level_begin[M + 1] == static_cast<int>(m.faces.size()));
    for (int l = 0; l <= M; ++l)
        for (int f = m.face_level_begin[l]; f < m.face_level_begin[l + 1]; ++f) CHECK(m.faces[f].level == l);

    // gather lists: every side covered exactly once by one or two faces, lower transverse position first
    for (int c = 0; c < N; ++c)
        for (int s = 0; s < 4; ++s) {
            const int b = m.side_ptr[4 * c + s], e = m.side_ptr[4 * c + s + 1];
            CHECK(e - b == 1 || e - b == 2);
            long long covered = 0;
            for (int k = b; k < e; ++k) {
                const CellFace& f = m.faces[m.side_face[k]];
                covered += 1ll << (M - f.level);
                CHECK(f.axis == s / 2);
                CHECK((s & 1 ? f.left : f.right) == c);
                if (k > b) CHECK(f.tpos > m.faces[m.side_face[k - 1]].tpos);
            }
            CHECK(covered == 1ll << (M - m.level[c]));
            const CellNeighbour nb = m.neighbour(c, static_cast<Face>(s));
            if (nb.kind == CellNeighbour::Kind::Boundary) CHECK(nb.a < 0);
            if (nb.kind == CellNeighbour::Kind::Same) CHECK(m.level[nb.a] == m.level[c]);
            if (nb.kind == CellNeighbour::Kind::Coarser) CHECK(m.level[nb.a] == m.level[c] - 1);
            if (nb.kind == CellNeighbour::Kind::Finer)
                CHECK(m.level[nb.a] == m.level[c] + 1 && m.level[nb.b] == m.level[c] + 1 && nb.a != nb.b);
        }
}

// The same oracle for a mesh of either dimension (boxes in finest-level integer coordinates, z = [0, 1) in 2-D): tiling, order,
// vertex 2:1 balance for every touching pair (faces, edges, corners), exactly one face per shared square (segment in 2-D) with its
// orientation, kind, level, transverse positions and measure, the gather lists (one face, or one per finer leaf in tpos2-major
// order, covering the side) and the neighbour summaries. Brute force over all pairs: keep the meshes to a few thousand leaves.
inline void check_cell_mesh_nd(const CellMesh& m) {
    const int N = m.num_leaves(), M = m.max_level, D = m.dim, ns = m.sides();
    const long long E[3] = {static_cast<long long>(m.nx) << M, static_cast<long long>(m.ny) << M,
                            D == 3 ? static_cast<long long>(m.nz) << M : 1};
    struct Box {
        long long lo[3], hi[3];
        int level;
    };
    std::vector<Box> box(static_cast<size_t>(N));
    long long volume = 0;
    for (int c = 0; c < N; ++c) {
        const long long s = 1ll << (M - m.level[c]);
        const long long x[3] = {m.ci[c], m.cj[c], m.ck[c]};
        Box& b = box[static_cast<size_t>(c)];
        for (int a = 0; a < 3; ++a) {
            b.lo[a] = a < D ? x[a] * s : 0;
            b.hi[a] = a < D ? (x[a] + 1) * s : 1;
        }
        b.level = m.level[c];
        long long v = 1;
        for (int a = 0; a < D; ++a) v *= s;
        volume += v;
        CHECK(D == 3 || m.ck[c] == 0);
    }
    CHECK(volume == E[0] * E[1] * E[2]);   // with the pairwise test below: the leaves tile the domain
    for (int c = 1; c < N; ++c) {
        CHECK(m.level[c] >= m.level[c - 1]);
        if (m.level[c] == m.level[c - 1]) CHECK(m.key_of(m.ci[c], m.cj[c], m.ck[c]) > m.key_of(m.ci[c - 1], m.cj[c - 1], m.ck[c - 1]));
    }
    CHECK(static_cast<int>(m.level_begin.size()) == M + 2 && m.level_begin[M + 1] == N);
    for (int l = 0; l <= M; ++l)
        for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c) CHECK(m.level[c] == l);

    const bool per[3] = {m.periodic_x, m.periodic_y, D == 3 && m.periodic_z};
    std::vector<long long> sh[3];
    for (int a = 0; a < 3; ++a) sh[a] = per[a] ? std::vector<long long>{-E[a], 0, E[a]} : std::vector<long long>{0};
    // axis, left, right, t1 start, t2 start (of the finer cell), kind (0 interior, 1 coarse-fine, 2 boundary), level, boundary side
    using Tup = std::tuple<int, int, int, long long, long long, int, int, int>;
    std::vector<Tup> want;
    for (int a = 0; a < N; ++a)
        for (int b = 0; b < N; ++b)
            for (long long sx : sh[0])
                for (long long sy : sh[1])
                    for (long long sz : sh[2]) {
                        if (a == b && sx == 0 && sy == 0 && sz == 0) continue;
                        const Box& A = box[static_cast<size_t>(a)];
                        Box B = box[static_cast<size_t>(b)];
                        const long long s3[3] = {sx, sy, sz};
                        for (int q = 0; q < 3; ++q) {
                            B.lo[q] += s3[q];
                            B.hi[q] += s3[q];
                        }
                        long long ov[3];
                        bool touch = true, inside = true;
                        for (int q = 0; q < 3; ++q) {
                            ov[q] = std::min(A.hi[q], B.hi[q]) - std::max(A.lo[q], B.lo[q]);
                            touch = touch && ov[q] >= 0;
                            inside = inside && ov[q] > 0;
                        }
                        if (!touch) continue;
                        CHECK(!inside);                               // no overlap
                        CHECK(std::abs(A.level - B.level) <= 1);      // vertex 2:1 balance for every touching pair
                        const int kind = A.level == B.level ? 0 : 1, lev = std::max(A.level, B.level);
                        const Box& fine = A.level >= B.level ? box[static_cast<size_t>(a)] : box[static_cast<size_t>(b)];
                        for (int ax = 0; ax < D; ++ax) {
                            const int t1 = transverse1(ax), t2 = transverse2(ax);
                            if (A.hi[ax] == B.lo[ax] && ov[t1] > 0 && ov[t2] > 0)
                                want.emplace_back(ax, a, b, fine.lo[t1], fine.lo[t2], kind, lev, 0);
                        }
                    }
    for (int a = 0; a < N; ++a) {
        const Box& A = box[static_cast<size_t>(a)];
        for (int ax = 0; ax < D; ++ax) {
            if (per[ax]) continue;
            const int t1 = transverse1(ax), t2 = transverse2(ax);
            if (A.lo[ax] == 0) want.emplace_back(ax, -1, a, A.lo[t1], A.lo[t2], 2, A.level, 2 * ax);
            if (A.hi[ax] == E[ax]) want.emplace_back(ax, a, -1, A.lo[t1], A.lo[t2], 2, A.level, 2 * ax + 1);
        }
    }
    std::vector<Tup> got;
    for (const CellFace& f : m.faces) {
        const long long s = 1ll << (M - f.level);
        const int kind = f.kind == CellFaceKind::Interior ? 0 : (f.kind == CellFaceKind::CoarseFine ? 1 : 2);
        got.emplace_back(f.axis, f.left, f.right, f.tpos * s, f.tpos2 * s, kind, f.level, kind == 2 ? static_cast<int>(f.side) : 0);
        const int l = f.level;
        const double meas = D == 3 ? (f.axis == 0 ? m.dy(l) * m.dz(l) : (f.axis == 1 ? m.dx(l) * m.dz(l) : m.dx(l) * m.dy(l)))
                                   : (f.axis == 0 ? m.dy(l) : m.dx(l));
        CHECK(f.length == meas);
        CHECK(D == 3 || f.tpos2 == 0);
    }
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    CHECK(want == got);   // exactly one face per shared square, with the right orientation, kind, level and position

    for (size_t f = 1; f < m.faces.size(); ++f) CHECK(m.faces[f].level >= m.faces[f - 1].level);
    CHECK(static_cast<int>(m.face_level_begin.size()) == M + 2 && m.face_level_begin[M + 1] == static_cast<int>(m.faces.size()));
    for (int l = 0; l <= M; ++l)
        for (int f = m.face_level_begin[l]; f < m.face_level_begin[l + 1]; ++f) CHECK(m.faces[f].level == l);

    // gather lists: every side covered exactly once by one face or one per finer leaf, (tpos2, tpos) increasing
    CHECK(static_cast<int>(m.side_ptr.size()) == ns * N + 1);
    for (int c = 0; c < N; ++c)
        for (int s = 0; s < ns; ++s) {
            const int b = m.side_ptr[ns * c + s], e = m.side_ptr[ns * c + s + 1];
            CHECK(e - b == 1 || e - b == m.fine_faces());
            long long covered = 0;
            for (int k = b; k < e; ++k) {
                const CellFace& f = m.faces[m.side_face[k]];
                long long a = 1;
                for (int q = 1; q < D; ++q) a *= 1ll << (M - f.level);
                covered += a;
                CHECK(f.axis == s / 2);
                CHECK((s & 1 ? f.left : f.right) == c);
                if (k > b) {
                    const CellFace& g = m.faces[m.side_face[k - 1]];
                    CHECK(std::make_pair(f.tpos2, f.tpos) > std::make_pair(g.tpos2, g.tpos));
                }
            }
            long long full = 1;
            for (int q = 1; q < D; ++q) full *= 1ll << (M - m.level[c]);
            CHECK(covered == full);
            const CellNeighbour nb = m.neighbour(c, static_cast<Face>(s));
            if (nb.kind == CellNeighbour::Kind::Boundary) CHECK(nb.a < 0);
            if (nb.kind == CellNeighbour::Kind::Same) CHECK(m.level[nb.a] == m.level[c]);
            if (nb.kind == CellNeighbour::Kind::Coarser) CHECK(m.level[nb.a] == m.level[c] - 1);
            if (nb.kind == CellNeighbour::Kind::Finer) {
                CHECK(m.level[nb.a] == m.level[c] + 1 && m.level[nb.b] == m.level[c] + 1 && nb.a != nb.b);
                if (D == 3)
                    CHECK(m.level[nb.c] == m.level[c] + 1 && m.level[nb.d] == m.level[c] + 1 && nb.c != nb.d && nb.c != nb.a &&
                          nb.d != nb.b);
                else
                    CHECK(nb.c < 0 && nb.d < 0);
            }
        }
}

inline double cell_mesh_total(const CellMesh& m, int q) {
    double s = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c) s += m.U[c][q] * m.area(c);
    return s;
}

inline void random_state(CellMesh& m, std::mt19937& rng) {
    m.U.resize(m.num_leaves());
    for (State& u : m.U)
        for (double& v : u) v = 0.001 * static_cast<double>(rng() % 100000) + 1.0;
}



inline SimParams cell_flux_params(std::initializer_list<std::pair<const char*, const char*>> kv) {
    std::vector<std::pair<const char*, const char*>> all = {{"nx", "9"},          {"ny", "7"},          {"Lx", "1.7"},
                                                           {"Ly", "1.1"},        {"u_in", "0.7"},      {"v_in", "0.2"},
                                                           {"rho_in", "1.2"},    {"p_out", "1.3"}};
    all.insert(all.end(), kv.begin(), kv.end());
    Params p = baseline_defaults();
    p.merge(get_test_case("Sod Shock Tube").defaults);
    p.set("test_case", "Sod Shock Tube");
    p.set("output_root", tmp_dir().string());
    p.set("use_mpi", "false");
    for (const auto& e : all) p.set(e.first, e.second);
    return SimParams::from(p);
}

inline bool bits_equal(const Field& a, const Field& b) {
    if (a.size() != b.size()) return false;
    for (size_t c = 0; c < a.size(); ++c)
        for (int q = 0; q < kNVar; ++q)
            if (!(a[c][q] == b[c][q])) return false;
    return true;
}



inline CellMesh random_adapted_mesh(bool px, bool py, unsigned seed, int rounds = 5) {
    CellMesh m(1.5, 1.0, 6, 5, 3, px, py);
    std::mt19937 rng(seed);
    for (int round = 0; round < rounds; ++round) {
        std::vector<char> r(m.num_leaves()), c(m.num_leaves());
        for (int k = 0; k < m.num_leaves(); ++k) {
            r[k] = rng() % 100 < 15;
            c[k] = rng() % 100 < 30;
        }
        m.adapt(r, c);
    }
    return m;
}



// Exact density of the Sod problem (rho, u, p = 1, 0, 1 | 0.125, 0, 0.1, gamma 1.4, diaphragm at x = 0.5) at time t.
inline double sod_exact_density(double x, double t) {
    const double g = 1.4, rL = 1.0, pL = 1.0, rR = 0.125, pR = 0.1;
    const double cL = std::sqrt(g * pL / rL), cR = std::sqrt(g * pR / rR);
    const auto fK = [&](double p, double rK, double pK, double cK) {
        if (p > pK) return (p - pK) * std::sqrt((2.0 / ((g + 1.0) * rK)) / (p + (g - 1.0) / (g + 1.0) * pK));
        return 2.0 * cK / (g - 1.0) * (std::pow(p / pK, (g - 1.0) / (2.0 * g)) - 1.0);
    };
    double p = 0.3;
    for (int it = 0; it < 100; ++it) {
        const double f = fK(p, rL, pL, cL) + fK(p, rR, pR, cR);
        const double h = 1e-7 * p;
        const double df = (fK(p + h, rL, pL, cL) + fK(p + h, rR, pR, cR) - f) / h;
        const double pn = std::max(p - f / df, 1e-6);
        if (std::abs(pn - p) < 1e-14) {
            p = pn;
            break;
        }
        p = pn;
    }
    const double us = 0.5 * (fK(p, rR, pR, cR) - fK(p, rL, pL, cL));
    const double S = (x - 0.5) / t;
    const double rsL = rL * std::pow(p / pL, 1.0 / g);
    const double rsR = rR * ((p / pR + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * p / pR + 1.0));
    const double SR = cR * std::sqrt((g + 1.0) / (2.0 * g) * p / pR + (g - 1.0) / (2.0 * g));
    const double cs = cL * std::pow(p / pL, (g - 1.0) / (2.0 * g));
    if (S < -cL) return rL;
    if (S < us - cs) {
        const double c = 2.0 / (g + 1.0) * (cL - (g - 1.0) / 2.0 * S);
        return rL * std::pow(c / cL, 2.0 / (g - 1.0));
    }
    if (S < us) return rsL;
    if (S < SR) return rsR;
    return rR;
}

struct CellSetup {
    EOS eos = EOS::ideal(1.4);
    RiemannSolver rs;
    MusclReconstruction mu;
    BoundaryManager bm;
    CellSetup(bool periodic, Limiter lim = Limiter::VanLeer, Reconstruction rec = Reconstruction::Muscl)
        : rs(eos, RiemannType::HLLC), mu(eos, lim, rec), bm(generate_cartesian_mesh(1.0, 1.0, 4, 4), eos) {
        if (periodic)
            bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Periodic, BCType::Periodic);
        else
            bm.set_boundary_types(BCType::Transmissive, BCType::Transmissive, BCType::Wall, BCType::Wall);
    }
};

inline const char* const kSchemeNames[3] = {"Explicit Euler", "SSP Runge-Kutta 2", "SSP Runge-Kutta 3"};

inline double cell_total(const CellMesh& m, int q) {
    double s = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c) s += m.U[c][q] * m.area(c);
    return s;
}



// A uniform-solver simulation and a level-0 cell mesh holding the same state, with the physics of the run.
struct ParityRig {
    Simulation sim;
    std::unique_ptr<CellMesh> cm;
    CellPhysics ph;
    std::unique_ptr<CellFluxEngine> eng;
    Field Uu;

    explicit ParityRig(std::initializer_list<std::pair<const char*, const char*>> kv, int max_level = 0) {
        sim.setup(cell_flux_params(kv), false);
        const Mesh& u = sim.mesh;
        ph.ns = sim.ns.get();
        ph.body = sim.body_force.get();
        if (sim.p.use_source) {
            ph.source = &sim.p.source;
            ph.source_unsteady = sim.p.source_unsteady;
            ph.source_mesh = &sim.mesh;
        }
        if (sim.p.use_ibm) ph.ibm_factory = sim.ibm_factory;
        const bool px = sim.boundaries.type(Face::Left) == BCType::Periodic;
        const bool py = sim.boundaries.type(Face::Bottom) == BCType::Periodic;
        cm = std::make_unique<CellMesh>(u.Lx, u.Ly, u.nx, u.ny, max_level, px, py, u.x0, u.y0);
        Uu.resize(static_cast<size_t>(u.num_cells()));
        for (int j = 0; j < u.ny; ++j)
            for (int i = 0; i < u.nx; ++i) Uu[j * u.nx + i] = test_state(sim.fluid, u.xc(i), u.yc(j));
        cm->U.resize(cm->num_leaves());
        for (int c = 0; c < cm->num_leaves(); ++c) cm->U[c] = Uu[cm->cj[c] * u.nx + cm->ci[c]];
        eng = std::make_unique<CellFluxEngine>(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
        eng->set_physics(ph);
        eng->build(*cm);
    }

    // residual and S_max of both solvers at stage time t, equal bit for bit?
    bool residual_equal(double t = 0.0) {
        sim.scheme->stage_time = t;
        eng->stage_time = t;
        Field Ru, Rc;
        const double Su = sim.residual(Uu, Ru);
        const double Sc = eng->residual(*cm, cm->U, Rc);
        Field perm(Ru.size());
        const Mesh& u = sim.mesh;
        for (int c = 0; c < cm->num_leaves(); ++c) perm[c] = Ru[cm->cj[c] * u.nx + cm->ci[c]];
        return bits_equal(perm, Rc) && Su == Sc;
    }
};



// Initial state of a test case at the leaf centres (evaluated on the uniform mesh of each level, so a level-0 mesh
// gets exactly the uniform initial field).
inline void init_cell_case(CellMesh& cm, const Simulation& sim, const char* name) {
    cm.U.assign(static_cast<size_t>(cm.num_leaves()), State{});
    for (int l = 0; l <= cm.max_level; ++l) {
        if (cm.level_begin[l + 1] == cm.level_begin[l]) continue;
        const Mesh ml = generate_cartesian_mesh(cm.Lx, cm.Ly, cm.nx_at(l), cm.ny_at(l), cm.x0, cm.y0);
        const Field F = get_test_case(name).initialize(ml, sim.fluid, sim.p.source);
        for (int c = cm.level_begin[l]; c < cm.level_begin[l + 1]; ++c) cm.U[c] = F[cm.cj[c] * cm.nx_at(l) + cm.ci[c]];
    }
}

inline CellPhysics physics_of(Simulation& sim) {
    CellPhysics ph;
    ph.ns = sim.ns.get();
    ph.body = sim.body_force.get();
    if (sim.p.use_source) {
        ph.source = &sim.p.source;
        ph.source_unsteady = sim.p.source_unsteady;
        ph.source_mesh = &sim.mesh;
    }
    if (sim.p.use_ibm) ph.ibm_factory = sim.ibm_factory;
    return ph;
}

// Run the cell mesh to t_end with the scheme and CFL of the simulation.
inline void run_cell_mesh(CellMesh& cm, Simulation& sim, double t_end, int nsteps_max = 100000, int closure = 2) {
    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
    eng.interpolate_jumps = closure == 1 || closure == 2;   // 0 injection, 1 interpolated coarse value, 2 = 1 + a second-order coarse side, 3 = injection + second-order coarse side
    eng.coarse_side_second_order = closure >= 2;
    eng.set_physics(physics_of(sim));
    eng.build(cm);
    std::unique_ptr<TimeScheme> ts = make_time_scheme(parse_time_scheme(sim.p.time_scheme), sim.p.cfl);
    CellAdvance adv(cm, eng, *ts);
    Field R;
    double S = eng.residual(cm, cm.U, R), t = 0.0;
    for (int k = 0; k < nsteps_max && t < t_end; ++k) {
        const double dt = std::min(sim.p.cfl * std::min(cm.dx(0), cm.dy(0)) / S, t_end - t);
        S = adv.step(dt, t);
        t += dt;
    }
}

// L1 difference of the density of `cm` against a level-0 reference mesh `ref` (mean of the reference cells under a leaf)
inline double density_error(const CellMesh& cm, const CellMesh& ref) {
    std::vector<double> grid(static_cast<size_t>(ref.nx) * ref.ny);
    for (int c = 0; c < ref.num_leaves(); ++c) grid[static_cast<size_t>(ref.cj[c]) * ref.nx + ref.ci[c]] = ref.U[c][0];
    double err = 0.0, area = 0.0;
    for (int c = 0; c < cm.num_leaves(); ++c) {
        const int r = ref.nx / cm.nx_at(cm.level[c]);   // reference cells per leaf side
        double mean = 0.0;
        for (int j = 0; j < r; ++j)
            for (int i = 0; i < r; ++i) mean += grid[static_cast<size_t>(cm.cj[c] * r + j) * ref.nx + cm.ci[c] * r + i];
        mean /= r * r;
        err += std::abs(cm.U[c][0] - mean) * cm.area(c);
        area += cm.area(c);
    }
    return err / area;
}

// ---------------------------------------------------------------------------
// Octree (3-D) helpers, shared by test_amr_octree_flux.cpp and test_amr_octree_advance.cpp
// ---------------------------------------------------------------------------

// Smooth 3-D state with every velocity component nonzero (primitive [rho, u, v, p, w])
inline State test_state3(const EOS& eos, double x, double y, double z) {
    return eos.prim_to_cons({1.0 + 0.1 * std::sin(1.3 * x) * std::cos(0.9 * y) * std::cos(0.7 * z + 0.2),
                             1.0 + 0.2 * std::sin(0.7 * y + 0.3 * z), 0.15 * std::cos(1.1 * x - 0.4 * z),
                             17.0 + 0.5 * std::cos(0.6 * x + 0.4 * y + 0.5 * z), 0.12 * std::sin(0.8 * x + 0.5 * y)});
}

// A 3-D Simulation of the face-engine rig (Sod defaults, 9 x 7 x 5 cells unless overridden) with the six boundary types.
inline void setup_3d(Simulation& sim, const std::array<const char*, 6>& bc, std::vector<std::pair<const char*, const char*>> kv) {
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

inline bool periodic_axis(const Simulation& sim, int a) { return sim.boundaries.type(static_cast<Face>(2 * a)) == BCType::Periodic; }

// The level-0 octree of the 3-D uniform mesh of `sim` (same origin, sizes and periodic axes)
inline CellMesh octree_of(const Simulation& sim, int max_level) {
    const Mesh& u = sim.mesh;
    return CellMesh::make_3d(u.Lx, u.Ly, u.Lz, u.nx, u.ny, u.nz, max_level, periodic_axis(sim, 0), periodic_axis(sim, 1),
                             periodic_axis(sim, 2), u.x0, u.y0, u.z0);
}

// `rounds` adapts with random refine (pr %) and join (pc %) flags
inline void random_adapt(CellMesh& m, unsigned seed, int rounds, int pr = 15, int pc = 30) {
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

// Total of component q over the leaves (volume-weighted in 3-D, area-weighted in 2-D), and the same of |U_q|
inline double cell_total_nd(const CellMesh& m, int q, bool absolute = false) {
    double s = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c) s += (absolute ? std::abs(m.U[c][q]) : m.U[c][q]) * m.volume(c);
    return s;
}

