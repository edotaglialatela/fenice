// AMR: indicator and remesh
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_amr_common.hpp"

// ---------------------------------------------------------------------------
// Indicator and remesh (P5)
// ---------------------------------------------------------------------------

TEST(cellindicator_matches_a_structured_reference) {
    // Level-0 mesh: the indicator against a plain 4-neighbour implementation on the structured grid.
    const EOS eos = EOS::ideal(1.4);
    const int nx = 40, ny = 12;
    CellMesh m(1.0, 0.3, nx, ny, 0);
    std::mt19937 rng(3);
    std::vector<double> rho(static_cast<size_t>(nx) * ny);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            rho[static_cast<size_t>(j) * nx + i] = (i < 17 ? 1.0 : 0.4) + (j == 6 && i > 30 ? 0.35 : 0.0) + 0.001 * (rng() % 10);
    m.U.resize(m.num_leaves());
    for (int c = 0; c < m.num_leaves(); ++c)
        m.U[c] = eos.prim_to_cons({rho[static_cast<size_t>(m.cj[c]) * nx + m.ci[c]], 0.3, -0.1, 1.0});
    CellIndicator ind;
    ind.criterion = 0.1;
    const std::vector<double> got = ind.xi(m, m.U, eos);
    // reference
    std::vector<double> x(rho.size(), 0.0);
    const auto test = [&](int a, int b) {
        const double vmin = std::max(std::min(std::abs(rho[a]), std::abs(rho[b])), 1e-2);
        if (std::abs(rho[b] - rho[a]) / vmin >= 0.1) x[a] = x[b] = 1.0;
    };
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            if (i + 1 < nx) test(j * nx + i, j * nx + i + 1);
            if (j + 1 < ny) test(j * nx + i, (j + 1) * nx + i);
        }
    for (int it = 0; it < 2; ++it) {
        std::vector<double> add(x.size(), 0.0);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const int a = j * nx + i;
                for (const int b : {i + 1 < nx ? a + 1 : -1, j + 1 < ny ? a + nx : -1})
                    if (b >= 0 && (x[a] > 0.05 || x[b] > 0.05)) add[a] += 0.1, add[b] += 0.1;
            }
        for (size_t k = 0; k < x.size(); ++k) x[k] += add[k];
    }
    double worst = 0.0;
    int flagged = 0;
    for (int c = 0; c < m.num_leaves(); ++c) {
        worst = std::max(worst, std::abs(got[c] - x[static_cast<size_t>(m.cj[c]) * nx + m.ci[c]]));
        flagged += got[c] >= ind.split;
    }
    CHECK(worst < 1e-12);
    CHECK(flagged > 12 && flagged < 150);   // the step columns with their ring, and the bump
}

TEST(cellindicator_diffusion_rule_at_level_jumps) {
    // Interfaces of a level: {a, b} two cells of the level (either one above the trigger spreads to both), {a, -1} a cell of
    // the level against a coarser leaf (only the cell decides and only it receives).
    CellIndicator ind;
    ind.diffusion = 1;
    const std::vector<std::array<int, 2>> ifaces = {{0, 1}, {1, 2}, {2, -1}, {3, -1}};
    std::vector<double> xi = {1.0, 0.0, 0.0, 0.0};
    ind.diffuse(xi, ifaces);
    CHECK_NEAR(xi[0], 1.1, 1e-12);
    CHECK_NEAR(xi[1], 0.1, 1e-12);   // cell 0 spreads to cell 1; cell 1 starts at 0, so nothing reaches cell 2 in this iteration
    CHECK(xi[2] == 0.0 && xi[3] == 0.0);
    xi = {0.0, 0.0, 1.0, 0.0};
    ind.diffuse(xi, ifaces);
    CHECK_NEAR(xi[2], 1.2, 1e-12);   // cell 2 spreads to cell 1 (pair) and to itself once more through {2, -1}
    CHECK_NEAR(xi[1], 0.1, 1e-12);
    CHECK(xi[0] == 0.0 && xi[3] == 0.0);
    xi = {0.0, 0.0, 0.0, 0.0};
    ind.diffuse(xi, ifaces);
    CHECK(xi == std::vector<double>({0.0, 0.0, 0.0, 0.0}));
}

TEST(riemann_batch_equals_scalar_bit_for_bit) {
    // face_flux_batch (AVX2 where the CPU has it) must give the bits of face_flux for every solver, both equations of state,
    // every axis, a batch length that is not a multiple of four, and the awkward states: densities and pressures under the
    // floor, negative internal energy, identical and mirrored states, speeds equal to zero to the last bits, signed zeros, NaN
    // (all region tests false: zero flux, a NaN wave speed). Two NaN count as equal.
    unsigned long long seed = 0x9E3779B97F4A7C15ull;
    const auto rnd = [&]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(seed >> 11) * (1.0 / 9007199254740992.0);
    };
    const auto same = [](double a, double b) {
        unsigned long long x, y;
        std::memcpy(&x, &a, 8);
        std::memcpy(&y, &b, 8);
        return x == y || (std::isnan(a) && std::isnan(b));
    };
    for (int kind = 0; kind < 2; ++kind) {
        const EOS eos = kind == 0 ? EOS::ideal(1.4) : EOS::stiffened(4.4, 6.0e8);
        const double p0 = kind == 0 ? 1.0 : 1.0e8;
        const int n = 4099;   // not a multiple of four: the tail runs the scalar code
        std::vector<State> UL(n), UR(n);
        std::vector<int> axis(n);
        for (int k = 0; k < n; ++k) {
            const bool strong = rnd() < 0.15;
            const double jump = strong ? 4.0 : 0.02;
            State a = {0.5 + rnd(), 2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0, p0 * (0.5 + rnd()), 2.0 * rnd() - 1.0};
            State b = a;
            b[0] *= 1.0 + jump * rnd();
            b[1] += jump * (rnd() - 0.5);
            b[2] += jump * (rnd() - 0.5);
            b[3] *= 1.0 + jump * rnd();
            b[4] += 0.1 * jump * (rnd() - 0.5);
            UL[k] = eos.prim_to_cons(a);
            UR[k] = eos.prim_to_cons(b);
            axis[k] = static_cast<int>(rnd() * 3.0) % 3;
        }
        // awkward states in the first entries
        int k = 0;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const auto put = [&](State l, State r, int ax) {
            UL[k] = l;
            UR[k] = r;
            axis[k] = ax;
            ++k;
        };
        for (int ax = 0; ax < 3; ++ax) {
            const State a = eos.prim_to_cons({1.0, 0.3, -0.2, p0, 0.1});
            put(a, a, ax);
            put(a, eos.prim_to_cons({1.0, -0.3, -0.2, p0, 0.1}), ax);
            put(eos.prim_to_cons({1.0, 0.0, 0.0, p0, 0.0}), eos.prim_to_cons({0.125, -0.0, 0.0, 0.1 * p0, -0.0}), ax);
            put({1e-20, 0.0, 0.0, 1e-20, 0.0}, {1.0, 0.0, 0.0, p0, 0.0}, ax);
            put({1.0, 5.0, 0.0, 0.1, 0.0}, {1.0, 5.0, 0.0, 0.1, 0.0}, ax);
            put({-1.0, 1.0, 0.0, 1.0, 0.0}, {1.0, 0.0, 0.0, p0, 0.0}, ax);
            put({0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, ax);
            put({nan, 0.0, 0.0, p0, 0.0}, {1.0, 0.0, 0.0, p0, 0.0}, ax);
            put(a, {1.0, nan, 0.0, p0, 0.0}, ax);
            put(a, {1.0, 0.0, 0.0, nan, 0.0}, ax);
            for (int ulp = -3; ulp <= 3; ++ulp) {   // normal velocity equal to the sound speed to the ulp
                const double c = std::sqrt(eos.gamma * (p0 + (kind == 0 ? 0.0 : eos.p_inf)) / 1.0);
                double u = c;
                for (int q = 0; q < std::abs(ulp); ++q) u = std::nextafter(u, ulp < 0 ? 0.0 : 1e300);
                State W = {1.0, 0.0, 0.0, p0, 0.0};
                W[ax == 0 ? 1 : (ax == 1 ? 2 : 4)] = u;
                State Wr = W;
                Wr[3] *= 1.0 + 1e-3;
                put(eos.prim_to_cons(W), eos.prim_to_cons(Wr), ax);
                put(eos.prim_to_cons(Wr), eos.prim_to_cons(W), ax);
            }
        }
        for (const RiemannType type : {RiemannType::HLLC, RiemannType::LowMachHLLC, RiemannType::PreconditionedHLLC}) {
            const RiemannSolver rs(eos, type, 0.1);
            std::vector<State> Fb(n);
            std::vector<double> sb(n);
            rs.face_flux_batch(UL.data(), UR.data(), axis.data(), n, Fb.data(), sb.data());
            int bad = 0;
            for (int q = 0; q < n; ++q) {
                double ss;
                const State Fs = rs.face_flux(UL[q], UR[q], axis[q], ss);
                bool ok = same(ss, sb[q]);
                for (int c = 0; c < kNVar; ++c) ok = ok && same(Fs[c], Fb[q][c]);
                bad += ok ? 0 : 1;
            }
            CHECK(bad == 0);
            if (bad) std::printf("    eos %d solver %d: %d of %d faces differ\n", kind, static_cast<int>(type), bad, n);
        }
    }
}

TEST(viscous_cell_kernels_equal_scalar_bit_for_bit) {
    // The viscous cell kernels of the AMR engine (AVX2 where the CPU has it: viscous_cell_quantities_run, viscous_speed_range) must
    // give the bits of the one-cell functions for both equations of state, runs and ranges that start anywhere and whose length is
    // not a multiple of four, and the awkward states (read as a cell's own state and as its neighbours): densities under the floor
    // and negative, negative pressure, zero states, signed zeros, mirrored (wall / no-slip / moving-wall ghost) neighbours, a cell
    // that is its own neighbour, NaN in single components and whole NaN states. Two NaN count as equal.
#ifdef FV_SIMD_AVX2
    std::printf("    vector path: %s\n", fv::detail::avx2_available() ? "AVX2" : "scalar only (no AVX2 or FENICE_NO_SIMD)");
#else
    std::printf("    vector path: not built (non-x86 or FV_NO_SIMD): scalar against scalar\n");
#endif
    unsigned long long seed = 0x2545F4914F6CDD1Dull;
    const auto rnd = [&]() {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(seed >> 11) * (1.0 / 9007199254740992.0);
    };
    const auto same = [](double a, double b) {
        unsigned long long x, y;
        std::memcpy(&x, &a, 8);
        std::memcpy(&y, &b, 8);
        return x == y || (std::isnan(a) && std::isnan(b));
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double mu = 0.02, k_cond = 0.02 * 1.4 / (0.72 * 0.4);
    const double dxl[4] = {0.05, 0.025, 0.0125, 0.00625}, dyl[4] = {0.03, 0.015, 0.0075, 0.00375};
    for (int kind = 0; kind < 2; ++kind) {
        const EOS eos = kind == 0 ? EOS::ideal(1.4) : EOS::stiffened(4.4, 6.0e8);
        const double p0 = kind == 0 ? 1.0 : 1.0e8;
        const int n = 4099;
        std::vector<State> U(n);
        std::vector<int> nbr(static_cast<size_t>(4) * n), level(n);
        for (int c = 0; c < n; ++c) {
            U[c] = eos.prim_to_cons({0.5 + rnd(), 2.0 * rnd() - 1.0, 2.0 * rnd() - 1.0, p0 * (0.5 + rnd()), 0.0});
            level[c] = static_cast<int>(rnd() * 4.0) % 4;
        }
        // awkward states in the first entries; every cell's neighbours are random leaves (often one of these, itself or a repeat)
        const State a = eos.prim_to_cons({1.0, 0.3, -0.2, p0, 0.0});
        const State awk[] = {a,
                             {a[0], -a[1], -a[2], a[3], 0.0},                  // no-slip ghost
                             {a[0], -a[1], a[2], a[3], 0.0},                   // wall ghost
                             {a[0], 2.0 * a[0] * 1.0 - a[1], -a[2], a[3], 0.0},   // moving-wall ghost
                             {1e-20, 0.0, 0.0, 1e-20, 0.0},
                             {-1.0, 1.0, 0.0, 1.0, 0.0},
                             {0.0, 0.0, 0.0, 0.0, 0.0},
                             {1.0, 5.0, 0.0, 0.1, 0.0},   // negative pressure: floored
                             {1.0, -0.0, 0.0, p0, 0.0},
                             {1.0, 0.0, -0.0, p0, 0.0},
                             {nan, 0.0, 0.0, p0, 0.0},
                             {1.0, nan, 0.0, p0, 0.0},
                             {1.0, 0.0, nan, p0, 0.0},
                             {1.0, 0.0, 0.0, nan, 0.0},
                             {nan, nan, nan, nan, 0.0}};
        const int nawk = static_cast<int>(sizeof(awk) / sizeof(awk[0]));
        for (int c = 0; c < nawk; ++c) U[c] = awk[c];
        for (int c = 0; c < n; ++c)
            for (int s = 0; s < 4; ++s) {
                const double r = rnd();
                nbr[static_cast<size_t>(4) * c + s] = r < 0.3 ? static_cast<int>(rnd() * nawk) % nawk
                                                      : r < 0.4 ? c
                                                                : static_cast<int>(rnd() * n) % n;
            }
        std::vector<std::vector<double>> qb(7, std::vector<double>(n, -7.0)), qs = qb;
        const auto view = [](std::vector<std::vector<double>>& q) {
            return ViscousCellArrays{q[0].data(), q[1].data(), q[2].data(), q[3].data(), q[4].data(), q[5].data(), q[6].data()};
        };
        const std::pair<int, int> runs[] = {{0, n}, {0, 0}, {0, 1}, {1, 3}, {3, 4}, {5, 5}, {2, 7}, {9, 8}, {6, 9}, {13, 2000}};
        int bad = 0;
        for (const auto& r : runs) {
            viscous_cell_quantities_run(eos, U.data(), nbr.data(), level.data(), dxl, dyl, r.first, r.second, view(qb));
            for (int c = r.first; c < r.first + r.second; ++c) {
                const int* nb = &nbr[static_cast<size_t>(4) * c];
                viscous_cell_quantities(eos, U[c], U[nb[0]], U[nb[1]], U[nb[2]], U[nb[3]], dxl[level[c]], dyl[level[c]], view(qs), c);
                bool ok = true;
                for (int j = 0; j < 7; ++j) ok = ok && same(qs[j][c], qb[j][c]);
                bad += ok ? 0 : 1;
            }
        }
        CHECK(bad == 0);
        if (bad) std::printf("    eos %d cell quantities: %d cells differ\n", kind, bad);

        // wave-speed bound: the running std::max of the scalar code over any range, NaN states ignored, all-NaN ranges give 0
        const auto serial = [&](int b, int e) {
            double s = 0.0;
            for (int c = b; c < e; ++c) s = std::max(s, viscous_cell_speed(eos, mu, k_cond, U[c], dxl[level[c]], dyl[level[c]]));
            return s;
        };
        bad = 0;
        for (const auto& r : runs) {
            const double sb = viscous_speed_range(eos, mu, k_cond, U.data(), level.data(), dxl, dyl, r.first, r.first + r.second);
            bad += same(sb, serial(r.first, r.first + r.second)) ? 0 : 1;
        }
        std::vector<State> Un(9, State{nan, nan, nan, nan, 0.0});
        std::vector<int> ln(9, 1);
        for (int e : {4, 8, 9}) bad += viscous_speed_range(eos, mu, k_cond, Un.data(), ln.data(), dxl, dyl, 0, e) == 0.0 ? 0 : 1;
        CHECK(bad == 0);
        if (bad) std::printf("    eos %d speed: %d ranges differ\n", kind, bad);
    }
}

TEST(cellindicator_cell_gather_diffusion_equals_the_interface_rule_bit_for_bit) {
    // diffuse_cells (gather per cell, parallel) must give the bits of diffuse (scatter over the interface list) on any table:
    // random same-level neighbours (also a cell twice, as on a 1-cell periodic axis, and itself), coarser sides and edges.
    unsigned seed = 12345u;
    const auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) & 0xFFFFu;
    };
    for (int rep = 0; rep < 6; ++rep) {
        const int n = rep < 3 ? 40 : 6000;   // above kOmpMinWork the loops run in parallel
        std::vector<int> nbr(static_cast<size_t>(4) * n, -2);
        std::vector<std::array<int, 2>> ifaces;
        for (int c = 0; c < n; ++c)
            for (int s = 0; s < 4; ++s) {
                const unsigned r = rnd() % 8;
                if (r < 5) {
                    const int b = static_cast<int>(rnd() % static_cast<unsigned>(n));
                    nbr[static_cast<size_t>(4) * c + s] = b;
                } else if (r == 5) {
                    nbr[static_cast<size_t>(4) * c + s] = -1;
                }
            }
        // the interface list the table stands for: a same-level entry c -> b is an interface {c, b} that gives to both cells once
        // per listing, which is what each cell counts in its own row only for the pairs listed from its side.  The gather rule counts
        // a pair for a cell from the cell's own sides, so build the list from the rows of the lower index and mirror the rest.
        std::vector<int> sym(nbr.size(), -2);
        for (int c = 0; c < n; ++c)
            for (int s = 0; s < 4; ++s) {
                const int b = nbr[static_cast<size_t>(4) * c + s];
                if (b == -1) {
                    sym[static_cast<size_t>(4) * c + s] = -1;
                    ifaces.push_back({c, -1});
                }
            }
        // same-level pairs: make the table symmetric by construction (each listed pair occupies one free side in each cell)
        std::vector<int> used(static_cast<size_t>(n), 0);
        for (int c = 0; c < n; ++c)
            for (int s = 0; s < 4; ++s) {
                const int b = nbr[static_cast<size_t>(4) * c + s];
                if (b < 0 || used[static_cast<size_t>(c)] >= 4 || used[static_cast<size_t>(b)] >= 4) continue;
                int sc = -1, sb = -1;
                for (int q = 0; q < 4 && sc < 0; ++q)
                    if (sym[static_cast<size_t>(4) * c + q] == -2) sc = q;
                if (b == c) {
                    if (sc < 0) continue;
                    int sc2 = -1;
                    for (int q = sc + 1; q < 4 && sc2 < 0; ++q)
                        if (sym[static_cast<size_t>(4) * c + q] == -2) sc2 = q;
                    if (sc2 < 0) continue;
                    sym[static_cast<size_t>(4) * c + sc] = c;
                    sym[static_cast<size_t>(4) * c + sc2] = c;
                    ifaces.push_back({c, c});
                    used[static_cast<size_t>(c)] += 2;
                    continue;
                }
                for (int q = 0; q < 4 && sb < 0; ++q)
                    if (sym[static_cast<size_t>(4) * b + q] == -2) sb = q;
                if (sc < 0 || sb < 0) continue;
                sym[static_cast<size_t>(4) * c + sc] = b;
                sym[static_cast<size_t>(4) * b + sb] = c;
                ifaces.push_back({c, b});
                ++used[static_cast<size_t>(c)];
                ++used[static_cast<size_t>(b)];
            }
        for (int it = 1; it <= 3; ++it) {
            CellIndicator ind;
            ind.diffusion = it;
            ind.diffusion_coeff = 0.1;
            ind.diffusion_trigger = 0.05;
            std::vector<double> xa(static_cast<size_t>(n)), xb;
            for (int c = 0; c < n; ++c) xa[c] = (rnd() % 5 == 0) ? 1.0 : (rnd() % 7 == 0 ? 0.07 : 0.0);
            xb = xa;
            ind.diffuse(xa, ifaces);
            ind.diffuse_cells(xb, sym);
            CHECK(xa == xb);
        }
    }
}

TEST(cellflux_wave_speed_equals_the_residual_bound) {
    // the level-by-level pass returns the value residual() does (convective and viscous), on random adapted meshes
    for (int sc = 0; sc < 3; ++sc) {
        CellSetup su(true);
        CellMesh m = random_adapted_mesh(true, true, 90u + static_cast<unsigned>(sc), 6);
        CHECK(m.level_begin[3] < m.num_leaves());
        m.U.resize(m.num_leaves());
        for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state(su.eos, m.xc(c), m.yc(c));
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        eng.build(m);
        Field R;
        const double a = eng.residual(m, m.U, R);
        const double b = eng.wave_speed(m, m.U);
        CHECK(a == b);
        CHECK(a > 0.0);
    }
}

TEST(cellindicator_levels_see_only_their_own_interfaces) {
    // A density jump far inside the coarse region must not flag the fine leaves of a refined block (they only see their own
    // level's interfaces and those to a coarser leaf), and a jump next to the block flags the fine cells through the
    // coarse-fine interface.
    const EOS eos = EOS::ideal(1.4);
    CellMesh m(8.0, 8.0, 8, 8, 2);
    std::vector<char> r(static_cast<size_t>(m.num_leaves()), 0);
    r[m.find(0, 1, 1)] = 1;
    m.adapt(r);
    CellIndicator ind;
    m.U.resize(m.num_leaves());
    for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = eos.prim_to_cons({m.xc(c) > 5.0 ? 3.0 : 1.0, 0.0, 0.0, 1.0});
    CellIndicator::Result far = ind.evaluate(m, m.U, eos);
    double fine_xi = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c)
        if (m.level[c] == 1) fine_xi = std::max(fine_xi, far.xi[c]);
    CHECK(fine_xi == 0.0);   // the jump at x = 5 is three coarse cells away
    for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = eos.prim_to_cons({m.xc(c) > 2.0 ? 3.0 : 1.0, 0.0, 0.0, 1.0});
    CellIndicator::Result near = ind.evaluate(m, m.U, eos);
    double fine_near = 0.0;
    for (int c = 0; c < m.num_leaves(); ++c)
        if (m.level[c] == 1 && m.xc(c) > 1.6) fine_near = std::max(fine_near, near.xi[c]);
    CHECK(fine_near >= 1.0);   // the jump sits on the interface of the block
    // the parent cell of the block at level 0 has its own xi: it is next to the jump too
    int parent_flagged = 0;
    for (int c = 0; c < m.num_leaves(); ++c)
        if (m.level[c] == 1 && near.xi_parent[c] >= 1.0) ++parent_flagged;
    CHECK(parent_flagged == 4);
}

TEST(cellremesh_a_marginal_ramp_reaches_a_fixed_point_with_the_parent_rule_and_flickers_without) {
    // A smooth density ramp whose variation per coarse face (0.14 at x = 0) is above the criterion 0.1 and per fine face
    // (0.07) below it. Expectation fixed beforehand: with the parent's xi the mesh stops changing within max_level + 1
    // remeshes of the static field and stays; with the children's xi it keeps changing (refine, join, refine ...).
    CellSetup su(false);
    const auto changes = [&](bool parent_rule, int remeshes, std::vector<int>& counts) {
        CellMesh m(1.0, 0.25, 32, 8, 2);
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        CellIndicator ind;
        ind.parent_join = parent_rule;
        CellAdaptor ad(m, eng, su.eos, ind);
        const auto fill = [&](CellMesh& mm) {
            mm.U.resize(mm.num_leaves());
            for (int c = 0; c < mm.num_leaves(); ++c) mm.U[c] = su.eos.prim_to_cons({1.0 + 4.5 * mm.xc(c), 0.0, 0.0, 1.0});
        };
        fill(m);
        eng.build(m);
        counts.clear();
        int n_changed = 0;
        for (int k = 0; k < remeshes; ++k) {
            const bool ch = ad.remesh();
            fill(m);   // the static field again, exactly, at the new leaves
            counts.push_back(m.num_leaves());
            if (ch) ++n_changed;
        }
        return n_changed;
    };
    std::vector<int> with_parent, with_children;
    const int ch_parent = changes(true, 10, with_parent);
    const int ch_children = changes(false, 10, with_children);
    std::printf("    marginal ramp, leaves after each of 10 remeshes: parent rule");
    for (int n : with_parent) std::printf(" %d", n);
    std::printf(" | children rule");
    for (int n : with_children) std::printf(" %d", n);
    std::printf("\n");
    CHECK(ch_parent <= 3);                                                   // max_level + 1 remeshes, then nothing
    CHECK(with_parent[4] == with_parent[9] && with_parent[3] == with_parent[9]);
    CHECK(ch_children >= 8);                                                 // the old rule keeps changing
    CHECK(with_children[8] != with_children[9] || with_children[7] != with_children[8]);
}

TEST(cellremesh_step_is_stable_conservative_and_deterministic) {
    CellSetup su(false);
    const auto build = [&](CellMesh& m, CellFluxEngine& eng, CellAdaptor& ad) {
        const auto initial = [&](CellMesh& mm) {
            mm.U.resize(mm.num_leaves());
            for (int c = 0; c < mm.num_leaves(); ++c)
                mm.U[c] = su.eos.prim_to_cons({mm.xc(c) < 0.5 ? 1.0 : 0.3, 0.2, 0.0, 1.0});
        };
        initial(m);
        eng.build(m);
        return ad.init_adapt(initial);
    };
    CellMesh m1(1.0, 0.25, 64, 16, 2), m2(1.0, 0.25, 64, 16, 2);
    CellFluxEngine e1(su.eos, su.rs, &su.mu, su.bm), e2(su.eos, su.rs, &su.mu, su.bm);
    CellIndicator ind;
    CellAdaptor a1(m1, e1, su.eos, ind), a2(m2, e2, su.eos, ind);
    const int p1 = build(m1, e1, a1), p2 = build(m2, e2, a2);
    CHECK(p1 >= 2 && p1 == p2);
    CHECK(m1.ci == m2.ci && m1.cj == m2.cj && m1.level == m2.level && m1.U == m2.U);   // deterministic
    CHECK(m1.num_leaves() > 64 * 16 && m1.num_leaves() < 256 * 64);
    // the finest level sits at the step (x = 0.5): its leaves are within the diffusion band of it
    int fine_far = 0;
    for (int c = 0; c < m1.num_leaves(); ++c)
        if (m1.level[c] == 2 && std::abs(m1.xc(c) - 0.5) > 0.1) ++fine_far;
    CHECK(fine_far == 0);
    // remeshing a static field again changes nothing, and the leaves stay bitwise the same
    const std::vector<int> ci = m1.ci, cj = m1.cj, lv = m1.level;
    const Field U = m1.U;
    CHECK(!a1.remesh());
    CHECK(m1.ci == ci && m1.cj == cj && m1.level == lv && m1.U == U);
    // the step moves: remeshing follows it and conserves the totals
    double before[4];
    for (int q = 0; q < 4; ++q) before[q] = cell_total(m1, q);
    for (int c = 0; c < m1.num_leaves(); ++c)
        m1.U[c] = su.eos.prim_to_cons({m1.xc(c) < 0.62 ? 1.0 : 0.3, 0.2, 0.0, 1.0});
    for (int q = 0; q < 4; ++q) before[q] = cell_total(m1, q);
    bool changed = false;
    for (int k = 0; k < 4; ++k) changed = a1.remesh() || changed;
    CHECK(changed);
    for (int q = 0; q < 4; ++q) CHECK_NEAR(cell_total(m1, q), before[q], 1e-12 * std::abs(before[q]));
    check_cell_mesh(m1);
}

TEST(cellremesh_keeps_the_immersed_body_on_the_finest_level) {
    ParityRig rig({{"nx", "36"}, {"ny", "24"}, {"use_ibm", "true"}, {"ibm_shape", "Cylinder"}, {"ibm_radius", "0.25"},
                   {"ibm_cx", "0.8"}, {"ibm_cy", "0.55"}, {"use_muscl", "true"}},
                  2);
    CellMesh& m = *rig.cm;
    CellIndicator ind;
    CellAdaptor ad(m, *rig.eng, rig.sim.fluid, ind);
    ad.set_body(rig.sim.ibm_factory, 2);
    const auto initial = [&](CellMesh& mm) {
        mm.U.assign(static_cast<size_t>(mm.num_leaves()), rig.sim.fluid.prim_to_cons({1.0, 0.5, 0.0, 1.0}));
    };
    ad.init_adapt(initial);
    const auto check_body = [&]() {
        const ImmersedBoundary fine = rig.sim.ibm_factory(generate_cartesian_mesh(m.Lx, m.Ly, m.nx_at(2), m.ny_at(2), m.x0, m.y0));
        int body_leaves = 0, coarse_leaves = 0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            const int r = 1 << (2 - m.level[c]);
            bool has_solid = false;
            for (int j = m.cj[c] * r; j < (m.cj[c] + 1) * r; ++j)
                for (int i = m.ci[c] * r; i < (m.ci[c] + 1) * r; ++i) has_solid = has_solid || fine.mask()[static_cast<size_t>(j) * m.nx_at(2) + i];
            if (has_solid) {
                ++body_leaves;
                CHECK(m.level[c] == 2);
            }
            coarse_leaves += m.level[c] == 0;
            if (rig.eng->solid()[c]) CHECK(m.level[c] == 2);
        }
        CHECK(body_leaves > 20);
        CHECK(coarse_leaves > 100);   // a uniform flow refines nothing else
    };
    check_body();
    CHECK(!ad.remesh());
    check_body();
    check_cell_mesh(m);
}

TEST(cellremesh_dynamic_sod_follows_the_waves) {
    // Bands fixed before the first run: base 50 + two levels with the indicator, remeshed every 4 coarse steps,
    // against the exact density: error within 1.6 x the uniform-200 one, a mean leaf count below 0.6 x 3200, the
    // shock region on the finest level at the end, mass conserved to 1e-12.
    CellSetup su(false);
    const auto init = [&](CellMesh& mm) {
        mm.U.resize(mm.num_leaves());
        for (int c = 0; c < mm.num_leaves(); ++c)
            mm.U[c] = mm.xc(c) < 0.5 ? su.eos.prim_to_cons({1.0, 0.0, 0.0, 1.0}) : su.eos.prim_to_cons({0.125, 0.0, 0.0, 0.1});
    };
    const auto run = [&](int nx, int max_level, double& mean_leaves, int& shock_level) {
        CellMesh m(1.0, 0.08, nx, std::max(4, nx / 12), max_level);
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        CellIndicator ind;
        CellAdaptor ad(m, eng, su.eos, ind);
        init(m);
        eng.build(m);
        if (max_level > 0) ad.init_adapt(init);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4, 1.0, 5, 1e-3);
        CellAdvance adv(m, eng, *ts);
        const double mass0 = cell_total(m, 0);
        Field R;
        double S = eng.residual(m, m.U, R), t = 0.0, leaf_sum = 0.0;
        int steps = 0;
        while (t < 0.2) {
            const double dt = std::min(0.4 * m.dx(0) / S, 0.2 - t);
            S = adv.step(dt, t);
            t += dt;
            ++steps;
            leaf_sum += m.num_leaves();
            if (max_level > 0 && steps % 4 == 0 && ad.remesh()) S = eng.residual(m, m.U, R);   // the speed bound follows the new leaves
        }
        CHECK_NEAR(cell_total(m, 0), mass0, 1e-12 * mass0);
        mean_leaves = leaf_sum / steps;
        shock_level = 0;
        for (int c = 0; c < m.num_leaves(); ++c)
            if (std::abs(m.xc(c) - 0.8528) < 0.02) shock_level = std::max(shock_level, m.level[c]);
        double err = 0.0, area = 0.0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            err += std::abs(m.U[c][0] - sod_exact_density(m.xc(c), 0.2)) * m.area(c);
            area += m.area(c);
        }
        return err / area;
    };
    double leaves_c = 0, leaves_f = 0, leaves_a = 0;
    int sl_c = 0, sl_f = 0, sl_a = 0;
    const double e_coarse = run(50, 0, leaves_c, sl_c), e_fine = run(200, 0, leaves_f, sl_f), e_amr = run(50, 2, leaves_a, sl_a);
    std::printf("    dynamic Sod L1(rho): uniform 50 %.4e, uniform 200 %.4e, adaptive %.4e (mean %.0f leaves, shock level %d)\n",
                e_coarse, e_fine, e_amr, leaves_a, sl_a);
    CHECK(e_amr < 1.6 * e_fine);
    CHECK(e_amr < e_coarse);
    CHECK(leaves_a < 0.6 * 3200);
    CHECK(sl_a == 2);
}

TEST(cellremesh_sod_indicator_study) {
    // How the indicator settings change the adaptive Sod run (base 50 + 2 levels, remesh every 4 steps): L1(rho) against the
    // exact solution and the mean number of leaves; uniform 50 and 200 for scale. Prints a table; the guards only catch breakage.
    CellSetup su(false);
    const auto init = [&](CellMesh& mm) {
        mm.U.resize(mm.num_leaves());
        for (int c = 0; c < mm.num_leaves(); ++c)
            mm.U[c] = mm.xc(c) < 0.5 ? su.eos.prim_to_cons({1.0, 0.0, 0.0, 1.0}) : su.eos.prim_to_cons({0.125, 0.0, 0.0, 0.1});
    };
    const auto run = [&](const CellIndicator& ind, int nx, int max_level, double& mean_leaves) {
        CellMesh m(1.0, 0.08, nx, std::max(4, nx / 12), max_level);
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        CellAdaptor ad(m, eng, su.eos, ind);
        init(m);
        eng.build(m);
        if (max_level > 0) ad.init_adapt(init);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4, 1.0, 5, 1e-3);
        CellAdvance adv(m, eng, *ts);
        Field R;
        double S = eng.residual(m, m.U, R), t = 0.0, sum = 0.0;
        int steps = 0;
        while (t < 0.2) {
            const double dt = std::min(0.4 * m.dx(0) / S, 0.2 - t);
            S = adv.step(dt, t);
            t += dt;
            ++steps;
            sum += m.num_leaves();
            if (max_level > 0 && steps % 4 == 0 && ad.remesh()) S = eng.residual(m, m.U, R);   // the speed bound follows the new leaves
        }
        mean_leaves = sum / steps;
        double err = 0.0, area = 0.0;
        for (int c = 0; c < m.num_leaves(); ++c) {
            err += std::abs(m.U[c][0] - sod_exact_density(m.xc(c), 0.2)) * m.area(c);
            area += m.area(c);
        }
        return err / area;
    };
    double leaves = 0.0;
    CellIndicator base;
    std::printf("    Sod indicator study (L1 rho, mean leaves): uniform 50 %.3e, uniform 200 %.3e\n", run(base, 50, 0, leaves), run(base, 200, 0, leaves));
    for (const double crit : {0.02, 0.05, 0.1})
        for (const int vars : {0, 1, 2}) {   // density; density + pressure; density + velocity
            CellIndicator ind;
            ind.criterion = crit;
            ind.use_pressure = vars == 1;
            ind.use_velocity = vars == 2;
            const double e = run(ind, 50, 2, leaves);
            std::printf("      criterion %.2f, %-16s: %.3e, %6.0f leaves\n", crit, vars == 0 ? "density" : (vars == 1 ? "density+pressure" : "density+velocity"), e, leaves);
            CHECK(e > 0.0 && e < 0.02);
        }
}

