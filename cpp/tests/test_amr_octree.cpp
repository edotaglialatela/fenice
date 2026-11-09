// AMR in three dimensions (octree leaves): the key layer and the CellMesh topology (phase Q1), the
// convective residual of the face engine (phase Q2).
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_amr_common.hpp"

#include <algorithm>
#include <numeric>
#include <set>

namespace {

// The bits of v spread with `gap` zero bits between them, by the plain definition (the oracle of the key functions).
uint64_t spread_naive(uint32_t v, int gap, int bits) {
    uint64_t x = 0;
    for (int b = 0; b < bits; ++b)
        if (v & (1u << b)) x |= 1ull << (b * (gap + 1));
    return x;
}

double mesh_total_3d(const CellMesh& m, int q) {
    double s = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c) s += m.U[c][q] * m.volume(c);
    return s;
}

void random_flags(const CellMesh& m, std::mt19937& rng, int pr, int pc, std::vector<char>& r, std::vector<char>& c) {
    r.assign(static_cast<size_t>(m.num_leaves()), 0);
    c.assign(static_cast<size_t>(m.num_leaves()), 0);
    for (int k = 0; k < m.num_leaves(); ++k) {
        r[k] = static_cast<int>(rng() % 100) < pr;
        c[k] = static_cast<int>(rng() % 100) < pc;
    }
}

CellMesh random_octree(bool px, bool py, bool pz, unsigned seed, int rounds = 4) {
    CellMesh m = CellMesh::make_3d(1.5, 1.0, 0.75, 3, 2, 2, 2, px, py, pz, -0.25, 0.5, 0.125);
    std::mt19937 rng(seed);
    for (int round = 0; round < rounds; ++round) {
        std::vector<char> r, c;
        random_flags(m, rng, 15, 35, r, c);
        m.adapt(r, c);
    }
    return m;
}

// The same leaves assigned to a fresh mesh: the full rebuild, against which the incremental one is compared.
CellMesh rebuilt(const CellMesh& m) {
    CellMesh f = m.dim == 3 ? CellMesh::make_3d(m.Lx, m.Ly, m.Lz, m.nx, m.ny, m.nz, m.max_level, m.periodic_x, m.periodic_y,
                                                m.periodic_z, m.x0, m.y0, m.z0)
                            : CellMesh(m.Lx, m.Ly, m.nx, m.ny, m.max_level, m.periodic_x, m.periodic_y, m.x0, m.y0);
    if (m.dim == 3) f.assign_leaves(m.level, m.ci, m.cj, m.ck);
    else f.assign_leaves(m.level, m.ci, m.cj);
    return f;
}

}   // namespace

TEST(cellkey_morton_keys_match_the_bit_definition) {
    std::mt19937 rng(17);
    for (int t = 0; t < 2000; ++t) {
        const uint32_t i = rng() & 0x1FFFFF, j = rng() & 0x1FFFFF, k = rng() & 0x1FFFFF;
        CHECK(morton_key3(i, j, k) == (spread_naive(i, 2, 21) | (spread_naive(j, 2, 21) << 1) | (spread_naive(k, 2, 21) << 2)));
        CHECK(morton_key3(i, j, k) >> 3 == morton_key3(i >> 1, j >> 1, k >> 1));   // the parent's key
        const uint32_t a = rng() & 0x1FFFFFFF, b = rng() & 0x1FFFFFFF;
        CHECK(morton_key(a, b) == (spread_naive(a, 1, 29) | (spread_naive(b, 1, 29) << 1)));
        CHECK(cell_key(2, static_cast<int>(a), static_cast<int>(b), 5) == morton_key(a, b));
    }
    // children in key order, and the neighbour offsets: 3^D - 1 distinct, none zero, dx fastest
    for (int D = 2; D <= 3; ++D) {
        for (int q = 0; q < (1 << D); ++q) {
            int o[3];
            child_offset(q, o);
            CHECK(cell_key(D, 4 + o[0], 6 + o[1], 2 + o[2]) == (cell_key(D, 2, 3, 1) << D) + static_cast<uint64_t>(q));
        }
        std::set<std::array<int, 3>> seen;
        int prev = -1000;
        for (int q = 0; q < cell_neighbour_count(D); ++q) {
            int o[3];
            neighbour_offset(D, q, o);
            CHECK(!(o[0] == 0 && o[1] == 0 && o[2] == 0));
            CHECK(D == 3 || o[2] == 0);
            const int lin = (o[2] * 3 + o[1]) * 3 + o[0];
            CHECK(lin > prev);
            prev = lin;
            seen.insert({o[0], o[1], o[2]});
        }
        CHECK(static_cast<int>(seen.size()) == cell_neighbour_count(D));
    }
    // the key limit of 3-D meshes: 21 bits per axis at the finest level
    bool threw = false;
    try {
        CellMesh::make_3d(1.0, 1.0, 1.0, 1 << 19, 1, 1, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(cellmesh3d_level0_matches_uniform_mesh) {
    for (int per = 0; per < 8; ++per) {
        const bool px = per & 1, py = per & 2, pz = per & 4;
        CellMesh m = CellMesh::make_3d(3.7, 1.9, 1.3, 5, 4, 3, 2, px, py, pz, -0.3, 0.45, 0.2);
        const Mesh u = generate_cartesian_mesh_3d(3.7, 1.9, 1.3, 5, 4, 3, -0.3, 0.45, 0.2);
        CHECK(m.dim == 3 && m.sides() == 6 && m.children() == 8 && m.fine_faces() == 4);
        CHECK(m.num_leaves() == 60);
        for (int c = 0; c < m.num_leaves(); ++c) {
            CHECK(m.dx(0) == u.dx && m.dy(0) == u.dy && m.dz(0) == u.dz);
            CHECK(m.xc(c) == u.xc(m.ci[c]) && m.yc(c) == u.yc(m.cj[c]) && m.zc(c) == u.zc(m.ck[c]));
            CHECK(m.volume(c) == u.cell_volume());
        }
        for (const CellFace& f : m.faces) CHECK(f.length == u.face_area(f.axis));
        CHECK(m.dz(2) == u.dz / 4.0);
        const int nxf = px ? 5 : 6, nyf = py ? 4 : 5, nzf = pz ? 3 : 4;
        CHECK(static_cast<int>(m.faces.size()) == nxf * 4 * 3 + 5 * nyf * 3 + 5 * 4 * nzf);
        check_cell_mesh_nd(m);
    }
    // the generic oracle agrees with the 2-D one on 2-D meshes
    for (int cfg = 0; cfg < 4; ++cfg) {
        const CellMesh m = random_adapted_mesh(cfg & 1, cfg & 2, 300u + static_cast<unsigned>(cfg), 4);
        CHECK(m.dim == 2 && m.sides() == 4);
        check_cell_mesh(m);
        check_cell_mesh_nd(m);
    }
}

TEST(cellmesh3d_small_periodic_domains) {
    // one cell per axis (a leaf is its own neighbour) and two (two faces between the same pair of leaves)
    for (const auto& n : std::vector<std::array<int, 3>>{{1, 1, 1}, {1, 2, 1}, {2, 2, 2}, {2, 1, 3}}) {
        CellMesh m = CellMesh::make_3d(1.0, 1.0, 1.0, n[0], n[1], n[2], 2, true, true, true);
        check_cell_mesh_nd(m);
        std::mt19937 rng(11u + static_cast<unsigned>(n[0] * 100 + n[1] * 10 + n[2]));
        for (int round = 0; round < 4; ++round) {
            std::vector<char> r, c;
            random_flags(m, rng, 25, 50, r, c);
            m.adapt(r, c);
            check_cell_mesh_nd(m);
        }
    }
    const CellMesh two = CellMesh::make_3d(1.0, 1.0, 1.0, 2, 2, 2, 0, true, true, true);
    CHECK(two.faces.size() == 24);
}

TEST(cellmesh3d_random_adapt_invariants) {
    for (int cfg = 0; cfg < 8; ++cfg) {
        CellMesh m = CellMesh::make_3d(2.0, 1.5, 1.0, 3, 2, 2, 2, cfg & 1, cfg & 2, cfg & 4, 0.5, -1.0, 0.25);
        std::mt19937 rng(4321u + static_cast<unsigned>(cfg));
        random_state(m, rng);
        for (int round = 0; round < 7; ++round) {
            double before[kNVar];
            for (int q = 0; q < kNVar; ++q) before[q] = mesh_total_3d(m, q);
            std::vector<char> r, c;
            random_flags(m, rng, 12, 60, r, c);
            m.adapt(r, c);
            check_cell_mesh_nd(m);
            CHECK(static_cast<int>(m.U.size()) == m.num_leaves());
            for (int q = 0; q < kNVar; ++q) CHECK_NEAR(mesh_total_3d(m, q), before[q], 1e-12 * before[q]);
            // the incremental rebuild of adapt equals a full rebuild of the same leaves
            CHECK(m.same_arrays(rebuilt(m)));
        }
    }
    // and in 2-D, through the generalised code
    for (int cfg = 0; cfg < 4; ++cfg) {
        CellMesh m(2.0, 1.5, 6, 5, 3, cfg & 1, cfg & 2);
        std::mt19937 rng(77u + static_cast<unsigned>(cfg));
        for (int round = 0; round < 8; ++round) {
            std::vector<char> r, c;
            random_flags(m, rng, 12, 60, r, c);
            m.adapt(r, c);
            CHECK(m.same_arrays(rebuilt(m)));
        }
    }
}

TEST(cellmesh3d_refine_then_coarsen_restores_state) {
    CellMesh m = CellMesh::make_3d(1.0, 1.0, 1.0, 3, 2, 2, 2);
    std::mt19937 rng(99);
    random_state(m, rng);
    const Field U0 = m.U;
    const std::vector<int> ci0 = m.ci, cj0 = m.cj, ck0 = m.ck;
    std::vector<char> r(static_cast<size_t>(m.num_leaves()));
    for (char& v : r) v = rng() % 100 < 40;
    m.adapt(r);
    CHECK(m.num_leaves() > 12);
    check_cell_mesh_nd(m);
    std::vector<char> c(static_cast<size_t>(m.num_leaves()));
    for (int k = 0; k < m.num_leaves(); ++k) c[k] = m.level[k] == 1;
    m.adapt({}, c);
    CHECK(m.num_leaves() == 12 && m.ci == ci0 && m.cj == cj0 && m.ck == ck0);
    CHECK(m.U == U0);   // bitwise: injection, then the mean of eight equal values
    check_cell_mesh_nd(m);
}

TEST(cellmesh3d_ripple_across_edges_and_corners_and_merge_refusal) {
    CellMesh m = CellMesh::make_3d(4.0, 4.0, 4.0, 4, 4, 4, 3);
    const auto flag = [](const CellMesh& mesh, int l, int i, int j, int k) {
        std::vector<char> r(static_cast<size_t>(mesh.num_leaves()), 0);
        const int c = mesh.find(l, i, j, k);
        CHECK(c >= 0);
        if (c >= 0) r[c] = 1;
        return r;
    };
    m.adapt(flag(m, 0, 1, 1, 1));
    CHECK(m.num_leaves() == 71 && m.find(1, 2, 2, 2) >= 0 && m.find(1, 3, 3, 3) >= 0);
    // refine the child at the corner (2,2,2) of the level-0 lattice: the seven level-0 cells around that vertex must follow,
    // including the one that touches it only at the corner and the three that touch it only along an edge
    m.adapt(flag(m, 1, 3, 3, 3));
    for (int k = 1; k <= 2; ++k)
        for (int j = 1; j <= 2; ++j)
            for (int i = 1; i <= 2; ++i) CHECK(m.find(0, i, j, k) < 0);
    CHECK(m.find(2, 6, 6, 6) >= 0 && m.find(1, 4, 4, 4) >= 0 && m.find(0, 3, 3, 3) >= 0 && m.find(0, 0, 0, 0) >= 0);
    CHECK(m.num_leaves() == 71 + 7 + 7 * 7);
    check_cell_mesh_nd(m);
    {
        // the group of (0;2,2,2) cannot merge back: its corner neighbour block (1,1,1) is two levels deeper at that vertex
        std::vector<char> c(static_cast<size_t>(m.num_leaves()), 0);
        for (int q = 0; q < m.num_leaves(); ++q) c[q] = m.level[q] == 1 && m.ci[q] / 2 == 2 && m.cj[q] / 2 == 2 && m.ck[q] / 2 == 2;
        CHECK(!m.adapt({}, c));
        // the block (0;2,1,1) touches (1;3,3,3) across a face (x = 4 in level-1 units): refused
        for (int q = 0; q < m.num_leaves(); ++q) c[q] = m.level[q] == 1 && m.ci[q] / 2 == 2 && m.cj[q] / 2 == 1 && m.ck[q] / 2 == 1;
        CHECK(!m.adapt({}, c));
        // the block (0;2,2,1) touches it along an edge only (x = y = 4, z in [3, 4)): refused as well
        for (int q = 0; q < m.num_leaves(); ++q) c[q] = m.level[q] == 1 && m.ci[q] / 2 == 2 && m.cj[q] / 2 == 2 && m.ck[q] / 2 == 1;
        CHECK(!m.adapt({}, c));
        // coarsen the level-2 leaves first; then every group can merge back
        std::vector<char> c2(static_cast<size_t>(m.num_leaves()), 0);
        for (int q = 0; q < m.num_leaves(); ++q) c2[q] = m.level[q] == 2;
        CHECK(m.adapt({}, c2));
        check_cell_mesh_nd(m);
        std::vector<char> c1(static_cast<size_t>(m.num_leaves()), 0);
        for (int q = 0; q < m.num_leaves(); ++q) c1[q] = m.level[q] == 1;
        CHECK(m.adapt({}, c1));
        CHECK(m.num_leaves() == 64);
        check_cell_mesh_nd(m);
    }
    {
        // four finer neighbours across a side, in tpos2-major order
        CellMesh q = CellMesh::make_3d(4.0, 4.0, 4.0, 4, 4, 4, 2);
        q.adapt(flag(q, 0, 1, 1, 1));
        const CellNeighbour nb = q.neighbour(q.find(0, 0, 1, 1), Face::Right);
        CHECK(nb.kind == CellNeighbour::Kind::Finer);
        CHECK(nb.a == q.find(1, 2, 2, 2) && nb.b == q.find(1, 2, 3, 2) && nb.c == q.find(1, 2, 2, 3) && nb.d == q.find(1, 2, 3, 3));
        const CellNeighbour up = q.neighbour(q.find(1, 2, 2, 2), Face::Back);
        CHECK(up.kind == CellNeighbour::Kind::Same && up.a == q.find(1, 2, 2, 3));
        const CellNeighbour top = q.neighbour(q.find(1, 3, 3, 3), Face::Back);
        CHECK(top.kind == CellNeighbour::Kind::Coarser && top.a == q.find(0, 1, 1, 2));
        const CellNeighbour back = q.neighbour(nb.a, Face::Left);
        CHECK(back.kind == CellNeighbour::Kind::Coarser && back.a == q.find(0, 0, 1, 1));
    }
}

TEST(cellmesh3d_forced_by_finer_closure_equals_the_ripple) {
    int forced = 0;
    for (unsigned seed = 0; seed < 12; ++seed) {
        CellMesh m = random_octree(seed & 1, seed & 2, seed & 4, 500u + seed, 3);
        const int N = m.num_leaves(), M = m.max_level;
        std::mt19937 rng(900u + seed);
        std::vector<char> rf(static_cast<size_t>(N), 0);
        for (int c = 0; c < N; ++c) rf[c] = m.level[c] < M && rng() % 100 < 8;
        std::vector<char> closure = rf;
        for (bool again = true; again;) {
            again = false;
            for (int c = 0; c < N; ++c)
                if (!closure[c] && m.forced_by_finer(c, closure, true)) closure[c] = 1, again = true;
        }
        CellMesh ref = m;
        ref.adapt(rf);
        for (int c = 0; c < N; ++c) {
            const bool refined = ref.find(m.level[c], m.ci[c], m.cj[c], m.ck[c]) < 0;
            CHECK(refined == static_cast<bool>(closure[c]));
            forced += closure[c] && !rf[c];
        }
        check_cell_mesh_nd(ref);
    }
    CHECK(forced > 0);   // the ripple was exercised
}

TEST(cellmesh3d_partial_mesh_built_directly_equals_the_cut_one) {
    for (unsigned seed = 0; seed < 8; ++seed) {
        CellMesh m = random_octree(seed & 1, seed & 2, seed & 4, 700u + seed, 4);
        const int N = m.num_leaves();
        std::mt19937 rng(40u + seed);
        random_state(m, rng);
        std::vector<char> keep(static_cast<size_t>(N), 0);
        const int lo = static_cast<int>(rng() % static_cast<unsigned>(N)), n = N / 3;
        for (int k = 0; k < n; ++k) keep[static_cast<size_t>((lo + k) % N)] = 1;
        // the faces a kept leaf makes: the lower leaf of a same-level pair, the fine leaf of a coarse-fine pair, the boundary leaf
        std::vector<char> keep_face(m.faces.size(), 0);
        for (size_t f = 0; f < m.faces.size(); ++f) {
            const CellFace& g = m.faces[f];
            const int maker = g.kind == CellFaceKind::Boundary ? (g.left >= 0 ? g.left : g.right)
                              : g.kind == CellFaceKind::Interior ? g.left
                                                                  : (m.level[g.left] == g.level ? g.left : g.right);
            keep_face[f] = keep[static_cast<size_t>(maker)];
        }
        CellMesh cut = m, direct = m;
        std::vector<int> leaf_map, face_map;
        cut.keep_partial(keep, keep_face, leaf_map, face_map);
        direct.build_from(m, keep, &m.U);
        CHECK(cut.same_arrays(direct));
        CHECK(cut.U == direct.U);
        CHECK(cut.partial && direct.partial);
        // a partial mesh of the leaves alone keeps the lookups
        std::vector<int> lv, ii, jj, kk;
        for (int c = 0; c < N; ++c)
            if (keep[c]) {
                lv.push_back(m.level[c]);
                ii.push_back(m.ci[c]);
                jj.push_back(m.cj[c]);
                kk.push_back(m.ck[c]);
            }
        CellMesh leaves = m;
        leaves.assign_partial_leaves(lv, ii, jj, kk, nullptr);
        CHECK(leaves.level == cut.level && leaves.ci == cut.ci && leaves.cj == cut.cj && leaves.ck == cut.ck);
        for (int c = 0; c < leaves.num_leaves(); ++c)
            CHECK(leaves.find(leaves.level[c], leaves.ci[c], leaves.cj[c], leaves.ck[c]) == c);
    }
}

TEST(cellmesh3d_assign_leaves_checks_the_tiling) {
    const CellMesh m = random_octree(false, true, false, 31u, 3);
    CellMesh f = CellMesh::make_3d(m.Lx, m.Ly, m.Lz, m.nx, m.ny, m.nz, m.max_level, m.periodic_x, m.periodic_y, m.periodic_z, m.x0,
                                   m.y0, m.z0);
    // shuffled input, same mesh
    std::vector<int> order(static_cast<size_t>(m.num_leaves()));
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(3));
    std::vector<int> lv, ii, jj, kk;
    for (const int c : order) {
        lv.push_back(m.level[c]);
        ii.push_back(m.ci[c]);
        jj.push_back(m.cj[c]);
        kk.push_back(m.ck[c]);
    }
    f.assign_leaves(lv, ii, jj, kk);
    CHECK(f.level == m.level && f.ci == m.ci && f.cj == m.cj && f.ck == m.ck);
    check_cell_mesh_nd(f);
    bool threw = false;
    lv.pop_back();
    ii.pop_back();
    jj.pop_back();
    kk.pop_back();
    try {
        f.assign_leaves(lv, ii, jj, kk);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);   // a missing leaf leaves a hole
}
