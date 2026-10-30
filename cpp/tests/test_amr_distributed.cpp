// AMR: distributed run (ranks emulated by threads), topology-free remesh
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_amr_common.hpp"

// ---------------------------------------------------------------------------
// Distributed run (P6): ranks emulated by threads
// ---------------------------------------------------------------------------

namespace {

// In-process stand-in for MPI: `size` threads call the same collectives in the same order.
class FakeHub {
public:
    explicit FakeHub(int n) : n_(n), mail_(static_cast<size_t>(n), std::vector<std::vector<double>>(static_cast<size_t>(n))),
                              slot_(static_cast<size_t>(n)), dslot_(static_cast<size_t>(n), 0.0) {}

    CellComm comm(int rank) {
        CellComm c;
        c.rank = rank;
        c.size = n_;
        c.allgatherv = [this, rank](const double* send, int n, double* recv, const int* counts, const int* displs) {
            slot_[static_cast<size_t>(rank)].assign(send, send + n);
            barrier();
            for (int r = 0; r < n_; ++r) {
                if (static_cast<int>(slot_[static_cast<size_t>(r)].size()) != counts[r]) throw std::logic_error("allgatherv: counts differ");
                std::copy(slot_[static_cast<size_t>(r)].begin(), slot_[static_cast<size_t>(r)].end(), recv + displs[r]);
            }
            barrier();
        };
        c.allreduce_max = [this, rank](double v) {
            dslot_[static_cast<size_t>(rank)] = v;
            barrier();
            double g = dslot_[0];
            for (int r = 1; r < n_; ++r) g = std::max(g, dslot_[static_cast<size_t>(r)]);
            barrier();
            return g;
        };
        c.allreduce_sum = [this, rank](double v) {
            dslot_[static_cast<size_t>(rank)] = v;
            barrier();
            double g = dslot_[0];
            for (int r = 1; r < n_; ++r) g += dslot_[static_cast<size_t>(r)];
            barrier();
            return g;
        };
        c.exchange = [this, rank](const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv) {
            for (int r = 0; r < n_; ++r) mail_[static_cast<size_t>(rank)][static_cast<size_t>(r)] = send[static_cast<size_t>(r)];
            barrier();
            bool bad = false;
            for (int r = 0; r < n_; ++r) {
                const std::vector<double>& in = mail_[static_cast<size_t>(r)][static_cast<size_t>(rank)];
                if (in.size() != recv[static_cast<size_t>(r)].size()) bad = true;   // the lists of the two ranks disagree
                else recv[static_cast<size_t>(r)] = in;
            }
            barrier();
            if (bad) throw std::logic_error("exchange: a message does not match what the receiver expects");
        };
        return c;
    }

private:
    void barrier() {
        std::unique_lock<std::mutex> lock(m_);
        const int gen = gen_;
        if (++count_ == n_) {
            count_ = 0;
            ++gen_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [&] { return gen != gen_; });
        }
    }
    int n_;
    std::mutex m_;
    std::condition_variable cv_;
    int count_ = 0, gen_ = 0;
    std::vector<std::vector<std::vector<double>>> mail_;
    std::vector<std::vector<double>> slot_;
    std::vector<double> dslot_;
};

struct DistSpec {
    const char* name;
    bool px, py;
    BCType bc[4];   // left, right, bottom, top (periodic axes: both sides periodic)
    bool viscous, force, source, unsteady_source, body;
    unsigned seed;
    bool remesh;
};

// Everything one rank (or the serial reference) needs, built from the spec.
struct DistEnv {
    EOS eos = EOS::ideal(1.4);
    RiemannSolver rs;
    MusclReconstruction mu;
    BoundaryManager bm;
    NavierStokes ns;
    ConstantBodyForce bf;
    Params src;
    Mesh base;
    CellPhysics ph;
    explicit DistEnv(const DistSpec& sp)
        : rs(eos, RiemannType::HLLC), mu(eos, Limiter::VanLeer), bm(generate_cartesian_mesh(1.5, 1.0, 6, 5), eos),
          ns(eos, 0.03, 0.72), bf{0.3, -0.5, 0.0}, base(generate_cartesian_mesh(1.5, 1.0, 6, 5)) {
        bm.set_boundary_types(sp.bc[0], sp.bc[1], sp.bc[2], sp.bc[3]);
        BCParams bp;
        bp.u_in = 0.6;
        bp.v_in = 0.1;
        bp.rho_in = 1.1;
        bp.p_out = 17.0;
        bm.set_params(bp);
        if (sp.viscous) ph.ns = &ns;
        if (sp.force) ph.body = &bf;
        if (sp.source) {
            src = baseline_defaults();
            src.set("source_rho", "0.1*sin(3*x)*cos(2*y)");
            src.set("source_mx", sp.unsteady_source ? "0.2*x*y*cos(t)" : "0.2*x*y");
            src.set("source_E", "0.5*cos(x+y)");
            ph.source = &src;
            ph.source_unsteady = sp.unsteady_source;
            ph.source_mesh = &base;
        }
        if (sp.body)
            ph.ibm_factory = [](const Mesh& m) { return ImmersedBoundary(m, "Cylinder", 0.18, 0.7, 0.5, 1.0, 17.0); };
    }
};

// Run the case for `steps` steps on `ranks` ranks (0: the serial path without CellParallel); returns the final state as every
// rank sees it after gather_state (all identical, checked), and the held / halo counts of the last rank.
Field run_distributed(const DistSpec& sp, int ranks, int steps, std::vector<int>* counts = nullptr, bool local = false,
                      bool dist_remesh = false, bool no_topo = false) {
    CellMesh base = random_adapted_mesh(sp.px, sp.py, sp.seed, 6);
    const int Nranks = std::max(ranks, 1);
    FakeHub hub(Nranks);
    std::vector<Field> result(static_cast<size_t>(Nranks));
    std::vector<std::string> error(static_cast<size_t>(Nranks));
    std::vector<int> held(static_cast<size_t>(Nranks), 0), halo(static_cast<size_t>(Nranks), 0), empty(static_cast<size_t>(Nranks), 0);
    const auto body = [&](int rank) {
        try {
            DistEnv env(sp);
            const bool ghost = std::strstr(sp.name, "ghost") != nullptr;
            env.ph.ibm_ghost = ghost;
            CellMesh m = base;
            m.U.resize(m.num_leaves());
            for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state(env.eos, m.xc(c), m.yc(c));
            CellFluxEngine eng(env.eos, env.rs, &env.mu, env.bm);
            eng.set_physics(env.ph);
            eng.build(m);
            CellIndicator ind;
            ind.criterion = 0.02;
            CellAdaptor ad(m, eng, env.eos, ind);
            if (sp.body && ghost) {   // the body band wide enough for the ghost cells and their images
                const Mesh fine = generate_cartesian_mesh(m.Lx, m.Ly, m.nx_at(m.max_level), m.ny_at(m.max_level));
                ImmersedBoundary ib = env.ph.ibm_factory(fine);
                ib.enable_ghost_cells(fine, false);
                ad.set_body(env.ph.ibm_factory, std::max(2, ib.ghost_reach()));
            } else if (sp.body) {
                ad.set_body(env.ph.ibm_factory, 1);
            }
            std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4, 1.0, 5, 1e-3);
            std::unique_ptr<CellParallel> par;
            if (ranks > 0) {
                par = std::make_unique<CellParallel>(m, eng, hub.comm(rank));
                par->poison = true;
                par->first_extra = dist_remesh ? -5 : 1;   // the first reach is too small: the retry runs at every remesh
                par->shadow = dist_remesh && !no_topo;   // the directly built share against the extracted one, at every remesh
            }
            CellAdvance adv(m, eng, *ts);
            adv.par = par.get();
            Field R;
            double S = eng.residual(m, m.U, R), t = 0.0;
            // the wave-speed bound after a remesh, as Simulation takes it (each rank's leaves, then the maximum)
            const auto wave_speed = [&]() {
                eng.force_state(m, m.U);
                const double s = eng.wave_speed(m, m.U);
                return par ? par->reduce_max(s) : s;
            };
            std::unique_ptr<CellMesh> topo;   // the whole mesh's topology, kept by every rank for the distributed remesh
            if (dist_remesh && !no_topo) {
                topo = std::make_unique<CellMesh>(m);
                topo->U.clear();
                topo->set_faceless(true);
            }
            if (par) par->partition();
            if (par && local) par->localize();
            for (int k = 0; k < steps; ++k) {
                const double dt = 0.4 * m.dx(0) / S;
                S = adv.step(dt, t);
                t += dt;
                if (sp.body) {
                    eng.solid_reset(m, m.U);
                    if (par) par->exchange_solid();
                }
                if (sp.remesh && (k + 1) % 2 == 0 && no_topo) {
                    if (par->remesh(ad)) S = wave_speed();
                } else if (sp.remesh && (k + 1) % 2 == 0 && dist_remesh) {
                    if (par->remesh(*topo, ad)) S = wave_speed();
                } else if (sp.remesh && (k + 1) % 2 == 0) {
                    if (par && local) par->globalize();
                    else if (par) par->gather_state();
                    const bool changed = ad.remesh();
                    if (changed) S = wave_speed();   // every leaf is valid here
                    else if (par && local) eng.build(m);        // the localized tables are gone
                    if (par) par->partition();
                    if (par && local) par->localize();
                }
            }
            if (par) {
                held[static_cast<size_t>(rank)] = par->held_count();
                halo[static_cast<size_t>(rank)] = par->halo_count();
                if (local) {
                    CHECK(m.partial);
                    CHECK(m.num_leaves() == par->held_count());
                    for (int l = 0; l <= m.max_level; ++l)   // a level of the whole mesh this rank keeps no leaf of
                        if (par->level_present(l) && m.level_begin[l + 1] == m.level_begin[l]) ++empty[static_cast<size_t>(rank)];
                    par->globalize();
                } else {
                    par->gather_state();
                }
            }
            if (par && dist_remesh && Nranks > 1) CHECK(par->region_retries > 0);
            result[static_cast<size_t>(rank)] = m.U;
        } catch (const std::exception& e) {
            error[static_cast<size_t>(rank)] = e.what();
        }
    };
    std::vector<std::thread> threads;
    for (int r = 0; r < Nranks; ++r) threads.emplace_back(body, r);
    for (std::thread& th : threads) th.join();
    for (int r = 0; r < Nranks; ++r)
        if (!error[static_cast<size_t>(r)].empty()) std::printf("    rank %d: %s\n", r, error[static_cast<size_t>(r)].c_str());
    for (int r = 1; r < Nranks; ++r) CHECK(bits_equal(result[0], result[static_cast<size_t>(r)]));   // all ranks agree
    if (counts) *counts = {held[static_cast<size_t>(Nranks - 1)], halo[static_cast<size_t>(Nranks - 1)], std::accumulate(empty.begin(), empty.end(), 0)};
    return result[0];
}

}   // namespace

TEST(cellparallel_equals_serial_bitwise_for_any_rank_count) {
    const BCType per = BCType::Periodic;
    const DistSpec specs[] = {
        // name, px, py, bcs, viscous, force, source, unsteady, body, seed, remesh
        {"euler periodic", true, true, {per, per, per, per}, false, false, false, false, false, 11u, true},
        {"viscous + force + source periodic", true, true, {per, per, per, per}, true, true, true, false, false, 12u, true},
        {"unsteady source, mixed", true, false, {per, per, BCType::NoSlip, BCType::MovingWall}, true, false, true, true, false, 13u, false},
        {"walls, outlet, nrbc, viscous", false, false, {BCType::Wall, BCType::NonReflecting, BCType::NoSlip, BCType::Outlet}, true, false, false, false, false, 14u, true},
        {"inlet + mass flow", false, false, {BCType::MassflowInlet, BCType::Transmissive, BCType::Wall, BCType::VelInlet}, true, false, false, false, false, 15u, false},
        {"immersed body, viscous", false, false, {BCType::VelInlet, BCType::Outlet, BCType::Wall, BCType::Wall}, true, false, false, false, true, 16u, true},
    };
    for (const DistSpec& sp : specs) {
        const Field serial = run_distributed(sp, 0, 6);
        for (const int ranks : {1, 2, 3, 5, 7}) {
            std::vector<int> counts;
            const Field dist = run_distributed(sp, ranks, 6, &counts);
            const bool same = bits_equal(serial, dist);
            CHECK(same);
            if (!same) std::printf("    %s on %d ranks differs from the serial run\n", sp.name, ranks);
            if (ranks >= 5) CHECK(counts[0] < static_cast<int>(serial.size()));   // a rank does not hold everything
            if (ranks >= 2) CHECK(counts[1] > 0);                                  // and it does have a halo
        }
    }
}

TEST(cellparallel_localized_run_equals_serial_bitwise_and_keeps_only_its_halo) {
    // Phase 2 of docs/design/amr-distributed.md: after partition() each rank keeps only its owned leaves and the halo
    // (CellParallel::localize), steps and remeshes on them (globalize() before the replicated remesh), and the final state is the
    // serial one bit for bit, with every physics of the replicated test.
    const BCType per = BCType::Periodic;
    const DistSpec specs[] = {
        {"euler periodic", true, true, {per, per, per, per}, false, false, false, false, false, 11u, true},
        {"viscous + force + source periodic", true, true, {per, per, per, per}, true, true, true, false, false, 12u, true},
        {"unsteady source, mixed", true, false, {per, per, BCType::NoSlip, BCType::MovingWall}, true, false, true, true, false, 13u, false},
        {"walls, outlet, nrbc, viscous", false, false, {BCType::Wall, BCType::NonReflecting, BCType::NoSlip, BCType::Outlet}, true, false, false, false, false, 14u, true},
        {"inlet + mass flow", false, false, {BCType::MassflowInlet, BCType::Transmissive, BCType::Wall, BCType::VelInlet}, true, false, false, false, false, 15u, false},
        {"immersed body, viscous", false, false, {BCType::VelInlet, BCType::Outlet, BCType::Wall, BCType::Wall}, true, false, false, false, true, 16u, true},
    };
    int empty_levels = 0;
    for (const DistSpec& sp : specs) {
        const Field serial = run_distributed(sp, 0, 6);
        for (const int ranks : {1, 2, 3, 5, 7, 16}) {
            std::vector<int> counts;
            const Field dist = run_distributed(sp, ranks, 6, &counts, true);
            empty_levels += counts[2];
            const bool same = bits_equal(serial, dist);
            CHECK(same);
            if (!same) std::printf("    %s on %d ranks (localized) differs from the serial run\n", sp.name, ranks);
            if (ranks >= 5) CHECK(counts[0] < static_cast<int>(serial.size()));
        }
    }
    // The case that hung the hot-start regression on 4 ranks: a rank that keeps no leaf of a level other ranks have must still
    // make that level's collective calls (CellParallel::level_present). The test must reach it, or it proves nothing.
    std::printf("    ranks with an empty local level (rank x level, summed over the runs): %d\n", empty_levels);
    CHECK(empty_levels > 0);
}

TEST(cellparallel_distributed_remesh_equals_serial_bitwise) {
    // Phase 3 of docs/design/amr-distributed.md: the remesh of a localized run takes the indicator of the owned leaves from a
    // region of the mesh, gathers the flags, adapts the topology on every rank and builds the state of the new owned leaves
    // from the old ones (CellParallel::remesh): no gather of the mesh or of its state. The final state is the serial one bit for
    // bit, with every physics of the other tests, on 1-16 emulated ranks.
    const BCType per = BCType::Periodic;
    const DistSpec specs[] = {
        {"euler periodic", true, true, {per, per, per, per}, false, false, false, false, false, 11u, true},
        {"viscous + force + source periodic", true, true, {per, per, per, per}, true, true, true, false, false, 12u, true},
        {"walls, outlet, nrbc, viscous", false, false, {BCType::Wall, BCType::NonReflecting, BCType::NoSlip, BCType::Outlet}, true, false, false, false, false, 14u, true},
        {"immersed body, viscous", false, false, {BCType::VelInlet, BCType::Outlet, BCType::Wall, BCType::Wall}, true, false, false, false, true, 16u, true},
        {"immersed body (ghost cells)", false, false, {BCType::VelInlet, BCType::Outlet, BCType::Wall, BCType::Wall}, true, false, false, false, true, 26u, true},
        {"inlet + mass flow", false, false, {BCType::MassflowInlet, BCType::Transmissive, BCType::Wall, BCType::VelInlet}, true, false, false, false, false, 15u, true},
        {"unsteady source, mixed", true, false, {per, per, BCType::NoSlip, BCType::MovingWall}, true, false, true, true, false, 13u, true},
    };
    for (const DistSpec& sp : specs) {
        const Field serial = run_distributed(sp, 0, 8);
        for (const int ranks : {1, 2, 3, 5, 7, 16}) {
            const Field dist = run_distributed(sp, ranks, 8, nullptr, true, true);
            const bool same = bits_equal(serial, dist);
            CHECK(same);
            if (!same) std::printf("    %s on %d ranks (distributed remesh) differs from the serial run\n", sp.name, ranks);
        }
    }
}

TEST(cellparallel_distributed_init_equals_serial_init_bitwise) {
    // The adaptation of a fresh start (CellParallel::init_adapt): every rank builds its share of the base mesh from the faceless
    // topology and the passes run through remesh() with the initial condition filled on the share; the leaves and their state are
    // those of CellAdaptor::init_adapt on the whole mesh, bit for bit.
    const BCType per = BCType::Periodic;
    const DistSpec specs[] = {
        {"euler periodic", true, true, {per, per, per, per}, false, false, false, false, false, 11u, true},
        {"walls, outlet, nrbc, viscous", false, false, {BCType::Wall, BCType::NonReflecting, BCType::NoSlip, BCType::Outlet}, true, false, false, false, false, 14u, true},
        {"immersed body, viscous", false, false, {BCType::VelInlet, BCType::Outlet, BCType::Wall, BCType::Wall}, true, false, false, false, true, 16u, true},
        {"immersed body (ghost cells)", false, false, {BCType::VelInlet, BCType::Outlet, BCType::Wall, BCType::Wall}, true, false, false, false, true, 26u, true},
        {"inlet + mass flow", false, false, {BCType::MassflowInlet, BCType::Transmissive, BCType::Wall, BCType::VelInlet}, true, false, false, false, false, 15u, true},
    };
    struct Out {
        std::vector<int> level, ci, cj;
        Field U;
        int passes = 0;
    };
    const auto run = [&](const DistSpec& sp, int ranks, int init_levels) {
        const int Nranks = std::max(ranks, 1);
        FakeHub hub(Nranks);
        std::vector<Out> out(static_cast<size_t>(Nranks));
        std::vector<std::string> error(static_cast<size_t>(Nranks));
        const auto body = [&](int rank) {
            try {
                DistEnv env(sp);
                const bool ghost = std::strstr(sp.name, "ghost") != nullptr;
                env.ph.ibm_ghost = ghost;
                CellMesh m(1.5, 1.0, 12, 8, 2, sp.px, sp.py);
                const auto fill = [&](CellMesh& mm) {
                    mm.U.resize(static_cast<size_t>(mm.num_leaves()));
                    for (int c = 0; c < mm.num_leaves(); ++c) mm.U[static_cast<size_t>(c)] = test_state(env.eos, mm.xc(c), mm.yc(c));
                };
                CellFluxEngine eng(env.eos, env.rs, &env.mu, env.bm);
                eng.set_physics(env.ph);
                CellIndicator ind;
                ind.criterion = 0.02;
                CellAdaptor ad(m, eng, env.eos, ind);
                if (sp.body && ghost) {
                    const Mesh fine = generate_cartesian_mesh(m.Lx, m.Ly, m.nx_at(m.max_level), m.ny_at(m.max_level));
                    ImmersedBoundary ib = env.ph.ibm_factory(fine);
                    ib.enable_ghost_cells(fine, false);
                    ad.set_body(env.ph.ibm_factory, std::max(2, ib.ghost_reach()));
                } else if (sp.body) {
                    ad.set_body(env.ph.ibm_factory, 1);
                }
                Out& o = out[static_cast<size_t>(rank)];
                if (ranks == 0) {
                    for (int k = 0; k < init_levels; ++k) m.adapt(std::vector<char>(static_cast<size_t>(m.num_leaves()), 1));   // amr_init_levels
                    eng.build(m);
                    o.passes = ad.init_adapt(fill);
                } else {
                    CellParallel par(m, eng, hub.comm(rank));
                    par.poison = true;
                    par.first_extra = -5;
                    CellMesh topo = m;
                    topo.U.clear();
                    topo.set_faceless(true);
                    for (int k = 0; k < init_levels; ++k) topo.adapt(std::vector<char>(static_cast<size_t>(topo.num_leaves()), 1));
                    o.passes = par.init_adapt(topo, ad, fill);
                    CHECK(m.partial);
                    {   // the gather to rank 0 gives it the arrays of the gather to every rank, and the others nothing
                        std::vector<int> l1, i1, j1, l2, i2, j2;
                        Field u1, u2;
                        par.collect(l1, i1, j1, u1);
                        par.collect(l2, i2, j2, u2, true);
                        if (rank == 0) CHECK(l1 == l2 && i1 == i2 && j1 == j2 && bits_equal(u1, u2));
                        else CHECK(l2.empty() && u2.empty());
                    }
                    par.globalize();
                }
                o.level = m.level;
                o.ci = m.ci;
                o.cj = m.cj;
                o.U = m.U;
            } catch (const std::exception& e) {
                error[static_cast<size_t>(rank)] = e.what();
            }
        };
        std::vector<std::thread> threads;
        for (int r = 0; r < Nranks; ++r) threads.emplace_back(body, r);
        for (std::thread& th : threads) th.join();
        for (int r = 0; r < Nranks; ++r)
            if (!error[static_cast<size_t>(r)].empty()) std::printf("    rank %d: %s\n", r, error[static_cast<size_t>(r)].c_str());
        return out;
    };
    int adapted = 0;
    for (const DistSpec& sp : specs)
      for (const int init_levels : {0, 1}) {
        const Out serial = run(sp, 0, init_levels)[0];
        adapted += serial.passes > 1;
        for (const int ranks : {1, 2, 3, 5, 8}) {
            const std::vector<Out> d = run(sp, ranks, init_levels);
            for (const Out& o : d) {
                const bool same = o.level == serial.level && o.ci == serial.ci && o.cj == serial.cj && bits_equal(o.U, serial.U) &&
                                  o.passes == serial.passes;
                CHECK(same);
                if (!same) std::printf("    %s on %d ranks (distributed init) differs from the serial init\n", sp.name, ranks);
            }
        }
    }
    CHECK(adapted >= 3);   // the initial mesh does adapt in most of the specs
}

TEST(fetch_leaves_returns_exactly_the_leaves_of_a_region_from_the_owners) {
    // Phase 6 of docs/design/amr-distributed.md: the leaves of the mesh that intersect a region (boxes of cells of one level, a range
    // of leaf levels) are fetched from the owners, each answering from its own leaves only: the set and the state are those a brute-force
    // search of the whole mesh finds, for any partition, periodicity and region.
    const BCType per = BCType::Periodic;
    int checked = 0, remote = 0;
    for (int pp = 0; pp < 4; ++pp)
        for (unsigned seed = 0; seed < 3; ++seed) {
            const bool px = pp & 1, py = pp & 2;
            CellMesh m = random_adapted_mesh(px, py, 60u + seed + 10u * static_cast<unsigned>(pp), 6);
            const int N = m.num_leaves(), M = m.max_level;
            m.U.resize(static_cast<size_t>(N));
            for (int c = 0; c < N; ++c) m.U[static_cast<size_t>(c)] = {1.0 + c, 0.5 * c, -0.25 * c, 2.0 + c, 0.125 * c};
            for (const int R : {1, 2, 3, 5, 7}) {
                std::vector<std::vector<uint64_t>> split(static_cast<size_t>(M) + 1, std::vector<uint64_t>(static_cast<size_t>(R) + 1, ~static_cast<uint64_t>(0)));
                for (int l = 0; l <= M; ++l) {
                    const long long b = m.level_begin[l], n = m.level_begin[l + 1] - b;
                    for (int r = 0; r < R; ++r) {
                        const long long pos = b + n * r / R;
                        if (pos < b + n)
                            split[static_cast<size_t>(l)][static_cast<size_t>(r)] =
                                morton_key(static_cast<uint32_t>(m.ci[static_cast<size_t>(pos)]), static_cast<uint32_t>(m.cj[static_cast<size_t>(pos)]));
                    }
                }
                FakeHub hub(R);
                std::vector<std::string> error(static_cast<size_t>(R));
                std::vector<int> nfound(static_cast<size_t>(R), 0), nown(static_cast<size_t>(R), 0);
                const auto body = [&](int rank) {
                    try {
                        const DistSpec sp{"euler periodic", px, py, {per, per, per, per}, false, false, false, false, false, 11u, false};
                        DistEnv env(sp);
                        CellMesh mine = m;
                        CellFluxEngine eng(env.eos, env.rs, &env.mu, env.bm);
                        CellParallel par(mine, eng, hub.comm(rank));
                        std::vector<int> ol, oi, oj;
                        Field ou;
                        std::vector<int> owned;
                        for (int l = 0; l <= M; ++l) {
                            const long long b = m.level_begin[l], n = m.level_begin[l + 1] - b;
                            for (long long c = b + n * rank / R; c < b + n * (rank + 1) / R; ++c) owned.push_back(static_cast<int>(c));
                        }
                        for (const int c : owned) {
                            ol.push_back(m.level[static_cast<size_t>(c)]);
                            oi.push_back(m.ci[static_cast<size_t>(c)]);
                            oj.push_back(m.cj[static_cast<size_t>(c)]);
                            ou.push_back(m.U[static_cast<size_t>(c)]);
                        }
                        CellMesh own(m.Lx, m.Ly, m.nx, m.ny, M, px, py, m.x0, m.y0);
                        own.assign_partial_leaves(ol, oi, oj, &ou);   // the responder holds its own leaves only
                        std::vector<RegionItem> items;
                        std::mt19937 rng(7u + static_cast<unsigned>(rank));
                        for (int o = 0; o <= M; ++o) {
                            std::vector<std::array<int, 2>> seeds;
                            for (const int c : owned)
                                if (m.level[static_cast<size_t>(c)] == o) seeds.push_back({m.ci[static_cast<size_t>(c)], m.cj[static_cast<size_t>(c)]});
                            if (seeds.empty()) continue;
                            RegionItem it;
                            it.level = o;
                            it.box = dilate_cells(seeds, m.nx_at(o), m.ny_at(o), 1 + static_cast<int>(rng() % 4), px, py);
                            it.tmin = static_cast<int>(rng() % (M + 1));
                            it.tmax = it.tmin + static_cast<int>(rng() % (M + 1 - it.tmin));
                            items.push_back(std::move(it));
                        }
                        CellMesh out(m.Lx, m.Ly, m.nx, m.ny, M, px, py, m.x0, m.y0);
                        par.fetch_leaves(split, own, &ou, items, out);
                        std::set<std::array<int, 3>> expect;   // brute force on the whole mesh
                        for (int c = 0; c < N; ++c) {
                            const int t = m.level[static_cast<size_t>(c)], i = m.ci[static_cast<size_t>(c)], j = m.cj[static_cast<size_t>(c)];
                            for (const RegionItem& it : items) {
                                if (t < it.tmin || t > it.tmax) continue;
                                const auto in_box = [&](int a, int b) {
                                    return a >= it.box.i0 && a < it.box.i0 + it.box.w && b >= it.box.j0 && b < it.box.j0 + it.box.h &&
                                           it.box.g[static_cast<size_t>(b - it.box.j0) * it.box.w + (a - it.box.i0)];
                                };
                                bool hit = false;
                                if (t > it.level) {
                                    hit = in_box(i >> (t - it.level), j >> (t - it.level));
                                } else {
                                    const int n = 1 << (it.level - t);
                                    for (int b = 0; b < n && !hit; ++b)
                                        for (int a = 0; a < n && !hit; ++a) hit = in_box(i * n + a, j * n + b);
                                }
                                if (hit) {
                                    expect.insert({t, i, j});
                                    break;
                                }
                            }
                        }
                        std::set<std::array<int, 3>> got;
                        bool states_ok = true;
                        for (int c = 0; c < out.num_leaves(); ++c) {
                            got.insert({out.level[static_cast<size_t>(c)], out.ci[static_cast<size_t>(c)], out.cj[static_cast<size_t>(c)]});
                            const int w = m.find(out.level[static_cast<size_t>(c)], out.ci[static_cast<size_t>(c)], out.cj[static_cast<size_t>(c)]);
                            states_ok = states_ok && w >= 0 && bits_equal(Field{out.U[static_cast<size_t>(c)]}, Field{m.U[static_cast<size_t>(w)]});
                        }
                        CHECK(got == expect);
                        CHECK(states_ok);
                        nfound[static_cast<size_t>(rank)] = static_cast<int>(got.size());
                        nown[static_cast<size_t>(rank)] = static_cast<int>(owned.size());
                    } catch (const std::exception& e) {
                        error[static_cast<size_t>(rank)] = e.what();
                    }
                };
                std::vector<std::thread> threads;
                for (int r = 0; r < R; ++r) threads.emplace_back(body, r);
                for (std::thread& th : threads) th.join();
                for (int r = 0; r < R; ++r) {
                    if (!error[static_cast<size_t>(r)].empty()) std::printf("    rank %d: %s\n", r, error[static_cast<size_t>(r)].c_str());
                    CHECK(error[static_cast<size_t>(r)].empty());
                    ++checked;
                    if (nfound[static_cast<size_t>(r)] > nown[static_cast<size_t>(r)]) ++remote;
                }
            }
        }
    CHECK(checked > 0);
    CHECK(remote > checked / 2);   // most ranks do fetch leaves of other ranks
}

TEST(global_facts_by_bisection_equal_the_splitters_of_partition) {
    // The splitters and the counts of the levels from the leaves as they happen to be spread over the ranks (not the owners' blocks),
    // found without any rank holding the order of a level, are the ones partition() cuts on the whole mesh: same keys, same empty ranks.
    const BCType per = BCType::Periodic;
    int levels_compared = 0;
    for (int pp = 0; pp < 4; ++pp)
        for (unsigned seed = 0; seed < 3; ++seed) {
            const bool px = pp & 1, py = pp & 2;
            CellMesh m = random_adapted_mesh(px, py, 80u + seed + 10u * static_cast<unsigned>(pp), 6);
            const int M = m.max_level;
            for (const int R : {1, 2, 3, 5, 7, 16, 40}) {
                FakeHub hub(R);
                std::vector<std::string> error(static_cast<size_t>(R));
                const auto body = [&](int rank) {
                    try {
                        const BCType wall = BCType::Wall;
                        const DistSpec sp{"euler", px, py, {px ? per : wall, px ? per : wall, py ? per : wall, py ? per : wall}, false, false, false, false, false, 11u, false};
                        DistEnv env(sp);
                        CellMesh mine = m;
                        mine.U.assign(static_cast<size_t>(mine.num_leaves()), State{1.0, 0.0, 0.0, 2.5, 0.0});
                        CellFluxEngine eng(env.eos, env.rs, &env.mu, env.bm);
                        eng.build(mine);
                        CellParallel par(mine, eng, hub.comm(rank));
                        par.partition();   // the oracle: the splitters on the whole mesh
                        std::vector<std::vector<uint64_t>> keys(static_cast<size_t>(M) + 1);
                        for (int c = 0; c < m.num_leaves(); ++c)   // the leaves spread over the ranks by a hash
                            if (static_cast<int>((static_cast<unsigned>(c) * 2654435761u + 12345u) % static_cast<unsigned>(R)) == rank)
                                keys[static_cast<size_t>(m.level[static_cast<size_t>(c)])].push_back(
                                    morton_key(static_cast<uint32_t>(m.ci[static_cast<size_t>(c)]), static_cast<uint32_t>(m.cj[static_cast<size_t>(c)])));
                        for (std::vector<uint64_t>& k : keys) std::sort(k.begin(), k.end());
                        const GlobalFacts gf = par.global_facts(keys);
                        bool ok = gf.split == par.splitters();
                        for (int l = 0; l <= M; ++l) ok = ok && gf.count[static_cast<size_t>(l)] == m.level_begin[l + 1] - m.level_begin[l];
                        CHECK(ok);
                        if (rank == 0) levels_compared += M + 1;
                    } catch (const std::exception& e) {
                        error[static_cast<size_t>(rank)] = e.what();
                    }
                };
                std::vector<std::thread> threads;
                for (int r = 0; r < R; ++r) threads.emplace_back(body, r);
                for (std::thread& th : threads) th.join();
                for (int r = 0; r < R; ++r) {
                    if (!error[static_cast<size_t>(r)].empty()) std::printf("    rank %d: %s\n", r, error[static_cast<size_t>(r)].c_str());
                    CHECK(error[static_cast<size_t>(r)].empty());
                }
            }
        }
    CHECK(levels_compared > 0);
}

// Hash of a leaf and a seed, the same on every rank: the random decisions of the direct test of the topology-free remesh.
static unsigned leaf_hash(int l, int i, int j, unsigned seed) {
    unsigned h = seed * 2654435761u + static_cast<unsigned>(l) * 40503u;
    h = (h ^ (h >> 15)) * 2246822519u + static_cast<unsigned>(i) * 3266489917u;
    h = (h ^ (h >> 13)) * 668265263u + static_cast<unsigned>(j) * 374761393u;
    return h ^ (h >> 16);
}

TEST(topology_free_adapt_equals_whole_mesh_adapt_on_random_flags) {
    // docs/design/amr-distributed.md section 14: random refine / coarsen flags (a hash of the leaf, the same for any partition) on random
    // balanced meshes: the leaves and the state the ranks make, with no rank holding the whole topology, are those of CellMesh::adapt
    // on the whole mesh, bit for bit. Counted, so that the test cannot pass without meeting a ripple that crosses a rank boundary
    // twice, a sibling group split over ranks, and a merge another rank's refinement blocks.
    const BCType per = BCType::Periodic, wall = BCType::Wall;
    long long multihop = 0, split_groups = 0, blocked = 0, merged = 0, forced = 0;
    for (int pp = 0; pp < 4; ++pp)
        for (unsigned seed = 0; seed < 8; ++seed) {
            const bool px = pp & 1, py = pp & 2;
            CellMesh m = random_adapted_mesh(px, py, 120u + seed + 10u * static_cast<unsigned>(pp), seed % 2 ? 3 : 5);
            const int N = m.num_leaves(), M = m.max_level;
            m.U.resize(static_cast<size_t>(N));
            for (int c = 0; c < N; ++c) m.U[static_cast<size_t>(c)] = {1.0 + 0.01 * c, 0.5 + 0.003 * c, -0.25 * c, 2.0 + 0.1 * c, 0.125};
            const unsigned fs = 31u + seed;
            std::vector<char> rf(static_cast<size_t>(N), 0), cf(static_cast<size_t>(N), 0);
            for (int c = 0; c < N; ++c) {
                const int l = m.level[static_cast<size_t>(c)], i = m.ci[static_cast<size_t>(c)], j = m.cj[static_cast<size_t>(c)];
                rf[static_cast<size_t>(c)] = l < M && leaf_hash(l, i, j, fs) % 9 == 0;
                cf[static_cast<size_t>(c)] = leaf_hash(l, i, j, fs + 77u) % 10 < 8;
            }
            CellMesh ref = m;   // the whole-mesh adapt, and the ripple closure to count what the test meets
            ref.record_sources = true;
            ref.adapt(rf, cf);
            std::vector<char> closure = rf;
            for (bool again = true; again;) {
                again = false;
                for (int c = 0; c < N; ++c)
                    if (!closure[static_cast<size_t>(c)] && m.forced_by_finer(c, closure, false)) closure[static_cast<size_t>(c)] = 1, again = true;
            }
            for (const int R : {1, 2, 3, 5, 7, 16}) {
                const auto owner_of_leaf = [&](int c) {   // the equal-count blocks of every level
                    const int l = m.level[static_cast<size_t>(c)];
                    const long long b = m.level_begin[l], n = m.level_begin[l + 1] - b, k = c - b;
                    for (int r = 0; r < R; ++r)
                        if (k >= n * r / R && k < n * (r + 1) / R) return r;
                    return R - 1;
                };
                if (R >= 2) {
                    for (int c = 0; c < N; ++c) {
                        const int l = m.level[static_cast<size_t>(c)];
                        if (!closure[static_cast<size_t>(c)] || rf[static_cast<size_t>(c)]) continue;
                        ++forced;
                        for (int dj = -1; dj <= 2; ++dj)
                            for (int di = -1; di <= 2; ++di) {
                                if (l >= M || (di >= 0 && di <= 1 && dj >= 0 && dj <= 1)) continue;
                                int i = 2 * m.ci[static_cast<size_t>(c)] + di, j = 2 * m.cj[static_cast<size_t>(c)] + dj;
                                if (i < 0 || j < 0 || i >= m.nx_at(l + 1) || j >= m.ny_at(l + 1)) continue;
                                const int a = m.find(l + 1, i, j);
                                if (a >= 0 && closure[static_cast<size_t>(a)] && !rf[static_cast<size_t>(a)] && owner_of_leaf(a) != owner_of_leaf(c)) ++multihop;
                            }
                    }
                    for (int a = 0; a < N; ++a) {   // groups that merge or are blocked
                        const int l = m.level[static_cast<size_t>(a)];
                        if (l == 0 || (m.ci[static_cast<size_t>(a)] & 1) || (m.cj[static_cast<size_t>(a)] & 1)) continue;
                        const int s[4] = {a, m.find(l, m.ci[static_cast<size_t>(a)] + 1, m.cj[static_cast<size_t>(a)]),
                                          m.find(l, m.ci[static_cast<size_t>(a)], m.cj[static_cast<size_t>(a)] + 1),
                                          m.find(l, m.ci[static_cast<size_t>(a)] + 1, m.cj[static_cast<size_t>(a)] + 1)};
                        bool sib = true;
                        for (const int q : s) sib = sib && q >= 0 && cf[static_cast<size_t>(q)] && !closure[static_cast<size_t>(q)];
                        if (!sib) continue;
                        bool cross = false;
                        for (const int q : s) cross = cross || owner_of_leaf(q) != owner_of_leaf(a);
                        if (m.merge_ok(a, cf, closure, false)) {
                            ++merged;
                            split_groups += cross;
                        } else {   // blocked: by a refined child next to the group owned elsewhere
                            for (int q = 0; q < N; ++q) {
                                if (!closure[static_cast<size_t>(q)] || m.level[static_cast<size_t>(q)] != l) continue;
                                const int di = m.ci[static_cast<size_t>(q)] - m.ci[static_cast<size_t>(a)], dj = m.cj[static_cast<size_t>(q)] - m.cj[static_cast<size_t>(a)];
                                if (di >= -2 && di <= 3 && dj >= -2 && dj <= 3 && owner_of_leaf(q) != owner_of_leaf(a)) {
                                    ++blocked;
                                    break;
                                }
                            }
                        }
                    }
                }
                FakeHub hub(R);
                std::vector<std::string> error(static_cast<size_t>(R));
                std::mutex mu;
                std::vector<std::array<double, 8>> got;   // level, i, j and the state of the leaves the ranks own after the remesh
                const auto body = [&](int rank) {
                    try {
                        const DistSpec sp{"euler", px, py, {px ? per : wall, px ? per : wall, py ? per : wall, py ? per : wall}, false, false, false, false, false, 11u, false};
                        DistEnv env(sp);
                        CellMesh mine = m;
                        CellFluxEngine eng(env.eos, env.rs, &env.mu, env.bm);
                        eng.build(mine);
                        CellIndicator ind;
                        CellAdaptor ad(mine, eng, env.eos, ind);
                        CellParallel par(mine, eng, hub.comm(rank));
                        par.poison = true;
                        par.partition();
                        par.localize();
                        par.flag_hook = [&](const CellMesh& W, const std::vector<int>& which, std::vector<char>& r, std::vector<char>& c) {
                            r.assign(which.size(), 0);
                            c.assign(which.size(), 0);
                            for (size_t k = 0; k < which.size(); ++k) {
                                const int w = which[k], l = W.level[static_cast<size_t>(w)], i = W.ci[static_cast<size_t>(w)], j = W.cj[static_cast<size_t>(w)];
                                r[k] = l < M && leaf_hash(l, i, j, fs) % 9 == 0;
                                c[k] = leaf_hash(l, i, j, fs + 77u) % 10 < 8;
                            }
                        };
                                                const bool changed = par.remesh(ad);
                        (void)changed;
                        std::lock_guard<std::mutex> lock(mu);
                        for (int l = 0; l <= M; ++l)
                            for (int c = eng.owned_begin(l); c < eng.owned_end(l); ++c) {
                                const State& u = mine.U[static_cast<size_t>(c)];
                                got.push_back({static_cast<double>(l), static_cast<double>(mine.ci[static_cast<size_t>(c)]), static_cast<double>(mine.cj[static_cast<size_t>(c)]),
                                               u[0], u[1], u[2], u[3], u[4]});
                            }
                    } catch (const std::exception& e) {
                        error[static_cast<size_t>(rank)] = e.what();
                    }
                };
                std::vector<std::thread> threads;
                for (int r = 0; r < R; ++r) threads.emplace_back(body, r);
                for (std::thread& th : threads) th.join();
                for (int r = 0; r < R; ++r) {
                    if (!error[static_cast<size_t>(r)].empty()) std::printf("    rank %d: %s\n", r, error[static_cast<size_t>(r)].c_str());
                    CHECK(error[static_cast<size_t>(r)].empty());
                }
                // the leaves and the state against the whole-mesh adapt
                std::map<std::array<int, 3>, State> want;
                for (int c = 0; c < ref.num_leaves(); ++c) want[{ref.level[static_cast<size_t>(c)], ref.ci[static_cast<size_t>(c)], ref.cj[static_cast<size_t>(c)]}] = ref.U[static_cast<size_t>(c)];
                bool same = got.size() == want.size();
                for (const std::array<double, 8>& g : got) {
                    const auto it = want.find({static_cast<int>(g[0]), static_cast<int>(g[1]), static_cast<int>(g[2])});
                    if (it == want.end()) {
                        if (same) std::printf("    leaf (%d,%d,%d) not in the whole-mesh result\n", static_cast<int>(g[0]), static_cast<int>(g[1]), static_cast<int>(g[2]));
                        same = false;
                        continue;
                    }
                    for (int q = 0; q < kNVar; ++q) same = same && std::memcmp(&g[3 + static_cast<size_t>(q)], &it->second[static_cast<size_t>(q)], sizeof(double)) == 0;
                }
                if (!same) std::printf("    %d ranks: the leaves or their state differ from the whole-mesh adapt (%zu vs %zu leaves)\n", R, got.size(), want.size());
                CHECK(same);
            }
        }
    std::printf("    ripple-forced leaves %lld, across a boundary through a forced finer leaf %lld; merged groups %lld, split over ranks %lld; blocked by another rank %lld\n",
                forced, multihop, merged, split_groups, blocked);
    CHECK(multihop > 0);
    CHECK(split_groups > 0);
    CHECK(blocked > 0);
}

TEST(cellparallel_topology_free_remesh_equals_serial_bitwise) {
    // The remesh with no rank holding the topology of the whole mesh (CellParallel::remesh(ad)) on every physics of the other tests, with
    // the real indicator: the final state is the serial run's, bit for bit, on 1-16 emulated ranks (NaN-poisoned halos, a first reach
    // too small so that the retry runs).
    const BCType per = BCType::Periodic;
    const DistSpec specs[] = {
        {"euler periodic", true, true, {per, per, per, per}, false, false, false, false, false, 11u, true},
        {"viscous + force + source periodic", true, true, {per, per, per, per}, true, true, true, false, false, 12u, true},
        {"walls, outlet, nrbc, viscous", false, false, {BCType::Wall, BCType::NonReflecting, BCType::NoSlip, BCType::Outlet}, true, false, false, false, false, 14u, true},
        {"immersed body, viscous", false, false, {BCType::VelInlet, BCType::Outlet, BCType::Wall, BCType::Wall}, true, false, false, false, true, 16u, true},
        {"immersed body (ghost cells)", false, false, {BCType::VelInlet, BCType::Outlet, BCType::Wall, BCType::Wall}, true, false, false, false, true, 26u, true},
        {"inlet + mass flow", false, false, {BCType::MassflowInlet, BCType::Transmissive, BCType::Wall, BCType::VelInlet}, true, false, false, false, false, 15u, true},
        {"unsteady source, mixed", true, false, {per, per, BCType::NoSlip, BCType::MovingWall}, true, false, true, true, false, 13u, true},
    };
    for (const DistSpec& sp : specs) {
        const Field serial = run_distributed(sp, 0, 8);
        for (const int ranks : {1, 2, 3, 5, 7, 16}) {
            const Field dist = run_distributed(sp, ranks, 8, nullptr, true, true, true);
            const bool same = bits_equal(serial, dist);
            CHECK(same);
            if (!same) std::printf("    %s on %d ranks (topology-free remesh) differs from the serial run\n", sp.name, ranks);
        }
    }
}

TEST(topology_free_remesh_fails_loudly_when_the_region_is_too_small) {
    // Mutation check: with the region shrunk to the owned cells themselves, the decisions would silently read "no finer leaf, no
    // blocked merge" for every cell not fetched. They throw instead, on every rank.
    const BCType per = BCType::Periodic;
    CellMesh m = random_adapted_mesh(true, true, 140u, 5);
    const int N = m.num_leaves();
    m.U.assign(static_cast<size_t>(N), State{1.0, 0.0, 0.0, 2.5, 0.0});
    const int R = 3;
    FakeHub hub(R);
    std::vector<std::string> error(static_cast<size_t>(R));
    const auto body = [&](int rank) {
        try {
            const DistSpec sp{"euler periodic", true, true, {per, per, per, per}, false, false, false, false, false, 11u, false};
            DistEnv env(sp);
            CellMesh mine = m;
            CellFluxEngine eng(env.eos, env.rs, &env.mu, env.bm);
            eng.build(mine);
            CellIndicator ind;
            CellAdaptor ad(mine, eng, env.eos, ind);
            CellParallel par(mine, eng, hub.comm(rank));
            par.partition();
            par.localize();
            par.flag_hook = [](const CellMesh&, const std::vector<int>& which, std::vector<char>& r, std::vector<char>& c) {
                r.assign(which.size(), 0);
                c.assign(which.size(), 1);
            };
            par.region_shrink = -(ind.diffusion + 1);
            par.remesh(ad);
        } catch (const std::exception& e) {
            error[static_cast<size_t>(rank)] = e.what();
        }
    };
    std::vector<std::thread> threads;
    for (int r = 0; r < R; ++r) threads.emplace_back(body, r);
    for (std::thread& th : threads) th.join();
    for (int r = 0; r < R; ++r) CHECK(!error[static_cast<size_t>(r)].empty());
}

TEST(a_partial_mesh_throws_when_a_face_reads_a_leaf_it_does_not_keep) {
    // CellFluxEngine::localize checks that every read of every face the rank computes is a kept leaf: a halo that is one leaf
    // short is an error at once, not a wrong number later.
    const BCType per = BCType::Periodic;
    const DistSpec sp{"euler periodic", true, true, {per, per, per, per}, false, false, false, false, false, 31u, false};
    for (const bool drop : {false, true}) {
        DistEnv env(sp);
        CellMesh m = random_adapted_mesh(true, true, 31u, 6);
        CellFluxEngine eng(env.eos, env.rs, &env.mu, env.bm);
        eng.set_physics(env.ph);
        eng.build(m);
        const int N = m.num_leaves();
        CellFluxEngine::Partition p;   // this rank owns the first half of every level
        p.leaf_owned.assign(static_cast<size_t>(N), 0);
        for (int l = 0; l <= m.max_level; ++l)
            for (int c = m.level_begin[l]; c < (m.level_begin[l] + m.level_begin[l + 1]) / 2; ++c) p.leaf_owned[static_cast<size_t>(c)] = 1;
        p.face_active.assign(m.faces.size(), 0);
        p.leaf_held = p.leaf_owned;
        std::vector<int> reads;
        for (size_t f = 0; f < m.faces.size(); ++f) {
            const CellFace& face = m.faces[f];
            bool mine;
            if (face.kind == CellFaceKind::Boundary)
                mine = p.leaf_owned[static_cast<size_t>(face.left >= 0 ? face.left : face.right)] != 0;
            else if (face.kind == CellFaceKind::CoarseFine)
                mine = p.leaf_owned[static_cast<size_t>(m.level[face.left] > m.level[face.right] ? face.left : face.right)] != 0;
            else
                mine = p.leaf_owned[static_cast<size_t>(face.left)] || p.leaf_owned[static_cast<size_t>(face.right)];
            if (!mine) continue;
            p.face_active[f] = 1;
            reads.clear();
            eng.face_dependencies(m, static_cast<int>(f), reads);
            for (const int c : reads) p.leaf_held[static_cast<size_t>(c)] = 1;
        }
        if (drop) {   // one halo leaf missing
            for (int c = N - 1; c >= 0; --c)
                if (p.leaf_held[static_cast<size_t>(c)] && !p.leaf_owned[static_cast<size_t>(c)]) {
                    p.leaf_held[static_cast<size_t>(c)] = 0;
                    break;
                }
        }
        eng.set_partition(m, p);
        std::vector<char> keep_face = p.face_active;
        for (size_t f = 0; f < keep_face.size(); ++f) {
            const CellFace& face = m.faces[f];
            if ((face.left >= 0 && p.leaf_owned[static_cast<size_t>(face.left)]) || (face.right >= 0 && p.leaf_owned[static_cast<size_t>(face.right)]))
                keep_face[f] = 1;
        }
        std::vector<int> leaf_map, face_map;
        m.keep_partial(p.leaf_held, keep_face, leaf_map, face_map);
        bool threw = false;
        try {
            eng.localize(m, leaf_map, face_map);
        } catch (const std::logic_error&) {
            threw = true;
        }
        CHECK(threw == drop);
        if (!drop) {
            CHECK(m.partial && m.num_leaves() < N);
            bool outside = false;   // some side of some kept leaf looks outside, and asking for it is safe
            for (int c = 0; c < m.num_leaves() && !outside; ++c)
                for (int s = 0; s < 4; ++s) outside = outside || m.neighbour(c, static_cast<Face>(s)).kind == CellNeighbour::Kind::Outside;
            CHECK(outside);
        }
    }
}

TEST(halo_reach_bounds_the_halo_of_any_partition) {
    // Phase 0 of docs/design/amr-distributed.md: CellParallel::halo_reach measures every read of a distributed run from the leaf
    // whose owner makes it, independently of the partition. The real halo of partition() on 1-16 emulated ranks must lie in
    // the geometric halo it defines (every halo leaf within the reach, for its level difference, of an owned leaf), the mass-flow
    // side leaves apart (read on every rank). Guards the reach a halo by region (phase 4) will use: if a new read is added to
    // partition() but not to face_dependencies / halo_reach, this fails. The reach itself is pinned to the measured values.
    const BCType per = BCType::Periodic;
    const DistSpec specs[] = {
        {"euler periodic", true, true, {per, per, per, per}, false, false, false, false, false, 21u, false},
        {"viscous periodic", true, true, {per, per, per, per}, true, true, true, false, false, 22u, false},
        {"walls, outlet, nrbc, viscous", false, false, {BCType::Wall, BCType::NonReflecting, BCType::NoSlip, BCType::Outlet}, true, false, false, false, false, 23u, false},
        {"inlet + mass flow", false, false, {BCType::MassflowInlet, BCType::Transmissive, BCType::Wall, BCType::VelInlet}, true, false, false, false, false, 24u, false},
        {"immersed body (mask)", false, false, {BCType::VelInlet, BCType::Outlet, BCType::Wall, BCType::Wall}, true, false, false, false, true, 25u, false},
        {"immersed body (ghost cells)", false, false, {BCType::VelInlet, BCType::Outlet, BCType::Wall, BCType::Wall}, true, false, false, false, true, 26u, false},
    };
    int worst[CellParallel::HaloReach::NumKinds] = {};
    for (const DistSpec& sp : specs) {
        const bool ghost = std::strstr(sp.name, "ghost") != nullptr;
        DistEnv env(sp);
        env.ph.ibm_ghost = ghost;
        CellMesh m = random_adapted_mesh(sp.px, sp.py, sp.seed, 6);
        m.U.assign(static_cast<size_t>(m.num_leaves()), State{});
        for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state(env.eos, m.xc(c), m.yc(c));
        CellFluxEngine eng(env.eos, env.rs, &env.mu, env.bm);
        eng.set_physics(env.ph);
        eng.build(m);
        if (ghost) {   // the body band at the finest level, wide enough for the ghost cells and their images
            const Mesh fine = generate_cartesian_mesh(m.Lx, m.Ly, m.nx_at(m.max_level), m.ny_at(m.max_level));
            ImmersedBoundary ib = env.ph.ibm_factory(fine);
            ib.enable_ghost_cells(fine, false);
            CellAdaptor ad(m, eng, env.eos, CellIndicator{});
            ad.set_body(env.ph.ibm_factory, std::max(2, ib.ghost_reach()));
            ad.init_adapt([&](CellMesh& mm) {
                mm.U.assign(static_cast<size_t>(mm.num_leaves()), State{});
                for (int c = 0; c < mm.num_leaves(); ++c) mm.U[c] = test_state(env.eos, mm.xc(c), mm.yc(c));
            });
            CHECK(eng.ghost_ready());
        }
        const CellParallel::HaloReach hr = CellParallel::halo_reach(m, eng);
        for (int k = 0; k < CellParallel::HaloReach::NumKinds; ++k) worst[k] = std::max(worst[k], hr.max(k));
        std::vector<int> bound(static_cast<size_t>(2 * m.max_level) + 1, -1);   // per level difference, over all kinds
        for (int k = 0; k < CellParallel::HaloReach::NumKinds; ++k)
            for (size_t d = 0; d < bound.size(); ++d) bound[d] = std::max(bound[d], hr.ring[k][d]);
        std::vector<char> mf(static_cast<size_t>(m.num_leaves()), 0);
        eng.mass_flow_dependencies(m, mf);
        int outside = 0, halo = 0;
        for (const int R : {1, 2, 3, 5, 7, 16}) {
            // partition() exchanges its requests: one thread per rank, each with its own copy of the mesh and engine
            FakeHub hub(R);
            std::vector<std::vector<char>> held(static_cast<size_t>(R));
            std::vector<std::vector<int>> owner(static_cast<size_t>(R));
            std::vector<std::string> error(static_cast<size_t>(R));
            std::vector<std::thread> threads;
            for (int r = 0; r < R; ++r)
                threads.emplace_back([&, r]() {
                    try {
                        CellMesh mr = m;
                        CellFluxEngine er(env.eos, env.rs, &env.mu, env.bm);
                        er.set_physics(env.ph);
                        er.build(mr);
                        CellParallel par(mr, er, hub.comm(r));
                        par.partition();
                        for (int c = 0; c < mr.num_leaves(); ++c) {
                            held[static_cast<size_t>(r)].push_back(par.holds(c) ? 1 : 0);
                            owner[static_cast<size_t>(r)].push_back(par.owner(c));
                        }
                    } catch (const std::exception& e) {
                        error[static_cast<size_t>(r)] = e.what();
                    }
                });
            for (std::thread& th : threads) th.join();
            for (int r = 0; r < R; ++r) {
                if (!error[static_cast<size_t>(r)].empty()) {
                    std::printf("    rank %d: %s\n", r, error[static_cast<size_t>(r)].c_str());
                    ++outside;
                    continue;
                }
                const std::vector<int>& own = owner[static_cast<size_t>(r)];
                for (int c = 0; c < m.num_leaves(); ++c) {
                    if (!held[static_cast<size_t>(r)][static_cast<size_t>(c)] || own[static_cast<size_t>(c)] == r || mf[static_cast<size_t>(c)]) continue;
                    ++halo;
                    bool in = false;
                    for (int o = 0; o < m.num_leaves() && !in; ++o) {
                        if (own[static_cast<size_t>(o)] != r) continue;
                        const int b = bound[static_cast<size_t>(m.max_level + m.level[o] - m.level[c])];
                        in = b >= 0 && CellParallel::ring(m, o, c) <= b;
                    }
                    if (!in) ++outside;
                }
            }
        }
        CHECK(halo > 0);
        CHECK(outside == 0);
        if (outside) std::printf("    %s: %d halo leaves outside the reach\n", sp.name, outside);
        if (ghost) CHECK(hr.max(CellParallel::HaloReach::GhostImage) > hr.max(CellParallel::HaloReach::Interior));
    }
    // the reach measured on these meshes (four levels; rings of the driver's level, docs/design/amr-distributed.md section 3.2):
    // a stencil that reaches further must be accounted for in the halo
    CHECK(worst[CellParallel::HaloReach::Interior] <= 5);
    CHECK(worst[CellParallel::HaloReach::CoarseFine] <= 10);
    CHECK(worst[CellParallel::HaloReach::Boundary] <= 4);
    CHECK(worst[CellParallel::HaloReach::GhostImage] <= 7);
    CHECK(worst[CellParallel::HaloReach::SolidReset] <= 5);
    std::printf("    reach: interior %d, coarse-fine %d, boundary %d, ghost images %d, solid reset %d\n", worst[0], worst[1],
                worst[2], worst[3], worst[4]);
}

TEST(local_indicator_equals_the_whole_mesh_indicator_for_any_partition) {
    // docs/design/amr-distributed.md phase 3: a rank evaluates the indicator on its owned leaves and the leaves
    // CellIndicator::region lists for them (a partial mesh, with the state of those leaves only): xi and xi_parent of its leaves
    // are the whole mesh's, bit for bit, for any partition, periodicity and smoothing depth.
    const EOS eos = EOS::ideal(1.4);
    int compared = 0, active = 0, smaller = 0, regions = 0;
    for (int per = 0; per < 4; ++per)
        for (unsigned seed = 0; seed < 3; ++seed) {
            const bool px = per & 1, py = per & 2;
            CellMesh m = random_adapted_mesh(px, py, 40u + seed + 10u * static_cast<unsigned>(per), 6);
            std::mt19937 rng(900u + seed);
            const int N = m.num_leaves();
            m.U.resize(static_cast<size_t>(N));
            for (int c = 0; c < N; ++c) {   // steps of a few percent to a factor, so that the flags and the smoothing are active
                const double x = m.xc(c), y = m.yc(c);
                const int s = (static_cast<int>(4.0 * x) + static_cast<int>(3.0 * y)) % 3;
                const double noise = 0.02 * static_cast<double>(rng() % 100) / 100.0;
                State w = {1.0 + 0.12 * s + noise, 0.3 * s, -0.2 * (s & 1), 1.0 + 0.1 * s + noise, 0.0};
                m.U[static_cast<size_t>(c)] = eos.prim_to_cons(w);
            }
            CellIndicator ind;
            ind.use_pressure = ind.use_velocity = true;
            ind.diffusion = 1 + static_cast<int>(seed);
            ind.criterion = 0.08;
            const CellIndicator::Result whole = ind.evaluate(m, m.U, eos);
            for (const int R : {1, 2, 3, 5, 7, 16}) {
                for (int r = 0; r < R; ++r) {
                    std::vector<int> owned;
                    for (int l = 0; l <= m.max_level; ++l) {
                        const long long b = m.level_begin[l], n = m.level_begin[l + 1] - b;
                        for (long long c = b + n * r / R; c < b + n * (r + 1) / R; ++c) owned.push_back(static_cast<int>(c));
                    }
                    std::vector<char> need(static_cast<size_t>(N), 0);
                    ind.region(m, owned, need);
                    for (const int c : owned) CHECK(need[static_cast<size_t>(c)]);
                    CellMesh sub(m.Lx, m.Ly, m.nx, m.ny, m.max_level, px, py, m.x0, m.y0);
                    sub.level.clear();
                    sub.ci.clear();
                    sub.cj.clear();
                    sub.U.clear();
                    std::vector<int> to_sub(static_cast<size_t>(N), -1);
                    sub.level_begin.assign(static_cast<size_t>(m.max_level) + 2, 0);
                    for (int c = 0; c < N; ++c)
                        if (need[static_cast<size_t>(c)]) {
                            to_sub[static_cast<size_t>(c)] = static_cast<int>(sub.level.size());
                            sub.level.push_back(m.level[static_cast<size_t>(c)]);
                            sub.ci.push_back(m.ci[static_cast<size_t>(c)]);
                            sub.cj.push_back(m.cj[static_cast<size_t>(c)]);
                            sub.U.push_back(m.U[static_cast<size_t>(c)]);
                            ++sub.level_begin[static_cast<size_t>(m.level[static_cast<size_t>(c)]) + 1];
                        }
                    for (int l = 0; l <= m.max_level; ++l) sub.level_begin[static_cast<size_t>(l) + 1] += sub.level_begin[static_cast<size_t>(l)];
                    const CellIndicator::Result part = ind.evaluate(sub, sub.U, eos);
                    ++regions;
                    smaller += sub.num_leaves() < N;
                    for (const int c : owned) {
                        const int k = to_sub[static_cast<size_t>(c)];
                        ++compared;
                        active += whole.xi[static_cast<size_t>(c)] > 0.0;
                        CHECK(part.xi[static_cast<size_t>(k)] == whole.xi[static_cast<size_t>(c)]);
                        CHECK(part.xi_parent[static_cast<size_t>(k)] == whole.xi_parent[static_cast<size_t>(c)]);
                    }
                }
            }
        }
    CHECK(compared > 0);
    CHECK(active > compared / 20);   // the indicator is not trivially zero
    CHECK(smaller > regions / 4);    // and a rank's region is mostly a part of the mesh
    std::printf("    %d leaves compared in %d regions (%d smaller than the mesh), %d with xi > 0\n", compared, regions, smaller, active);
}

TEST(cellparallel_ring_counts_cells_of_the_first_leaf) {
    CellMesh m(4.0, 4.0, 4, 4, 2, true, false);
    std::vector<char> r(static_cast<size_t>(m.num_leaves()), 0);
    r[static_cast<size_t>(m.find(0, 0, 0))] = 1;
    m.adapt(r);
    r.assign(static_cast<size_t>(m.num_leaves()), 0);
    r[static_cast<size_t>(m.find(1, 0, 0))] = 1;
    m.adapt(r);
    // finest-level units (4 per base cell): f = [0,1) x [0,1), s = [14,16) x [0,2) (refined by the ripple across the seam),
    // a = [8,12) x [0,4), b = [4,8) x [0,4), g = [2,4) x [2,4), h = [8,12) x [8,12)
    const int f = m.find(2, 0, 0), s = m.find(1, 7, 0), a = m.find(0, 2, 0), b = m.find(0, 1, 0), g = m.find(1, 1, 1),
              h = m.find(0, 2, 2);
    CHECK(f >= 0 && s >= 0 && a >= 0 && b >= 0 && g >= 0 && h >= 0);
    CHECK(CellParallel::ring(m, a, a) == 0);
    CHECK(CellParallel::ring(m, a, b) == 1);   // neighbours
    CHECK(CellParallel::ring(m, f, s) == 1);   // touching across the periodic seam
    CHECK(CellParallel::ring(m, s, f) == 1);
    CHECK(CellParallel::ring(m, f, a) == 5);   // 4 finest cells between them across the seam (7 the other way)
    CHECK(CellParallel::ring(m, a, f) == 2);   // ... one cell of a's level
    CHECK(CellParallel::ring(m, g, a) == 3);
    CHECK(CellParallel::ring(m, a, g) == 2);
    CHECK(CellParallel::ring(m, f, h) == 8);   // y is not periodic: 7 finest cells
    CHECK(CellParallel::ring(m, h, f) == 2);
}

TEST(cell_engine_result_does_not_depend_on_the_thread_count) {
    // Adapted mesh above kOmpMinWork (every parallel loop of the engine runs in parallel), physics on: the result of three
    // steps with 1 thread and with several threads is bit-identical (all reductions are maxima, no sums across threads).
#ifdef _OPENMP
    const EOS eos = EOS::ideal(1.4);
    NavierStokes ns(eos, 0.02, 0.72);
    ConstantBodyForce bf{0.2, -0.3, 0.0};
    Params src = baseline_defaults();
    src.set("source_rho", "0.05*sin(5*x)*cos(3*y)");
    src.set("source_E", "0.3*cos(x+2*y)");
    const Mesh base = generate_cartesian_mesh(1.0, 0.75, 64, 48);
    CellSetup su(true);
    CellPhysics ph;
    ph.ns = &ns;
    ph.body = &bf;
    ph.source = &src;
    ph.source_mesh = &base;
    const auto run = [&](int threads) {
        omp_set_num_threads(threads);
        CellMesh m(1.0, 0.75, 64, 48, 2, true, true);
        for (int pass = 0; pass < 2; ++pass) {
            std::vector<char> r(static_cast<size_t>(m.num_leaves()), 0);
            for (int c = 0; c < m.num_leaves(); ++c) r[c] = std::abs(m.xc(c) - 0.4) < 0.15 && m.yc(c) < 0.5;
            m.adapt(r);
        }
        m.U.resize(m.num_leaves());
        for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state(su.eos, m.xc(c), m.yc(c));
        CellFluxEngine eng(su.eos, su.rs, &su.mu, su.bm);
        eng.set_physics(ph);
        eng.build(m);
        std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4, 1.0, 5, 1e-3);
        CellAdvance adv(m, eng, *ts);
        Field R;
        double S = eng.residual(m, m.U, R), t = 0.0;
        for (int k = 0; k < 3; ++k) {
            const double dt = 0.4 * m.dx(0) / S;
            S = adv.step(dt, t);
            t += dt;
        }
        return std::make_pair(m.U, m.num_leaves());
    };
    const auto one = run(1), many = run(std::max(2, std::min(8, omp_get_num_procs())));
    omp_set_num_threads(omp_get_num_procs());
    CHECK(one.second > 4096 && one.second == many.second);
    CHECK(bits_equal(one.first, many.first));
#endif
}

TEST(parallel_scan_helpers_equal_the_serial_loops) {
    // The blocked paths of the remesh helpers (taken from 4 * kOmpMinWork entries on, whatever the thread count) against the
    // serial loops: sizes around the switch and the block count, empty, short and lopsided inputs, sparse and dense marks.
    std::mt19937 gen(7u);
    const int t = 4 * kOmpMinWork;
    int bad = 0;
    for (const int n : {0, 1, 63, t - 1, t, t + 1, 64 * 300 - 1, 64 * 300, 64 * 300 + 1, 99991}) {
        std::vector<int> a(static_cast<size_t>(n)), ref;
        for (int& x : a) x = static_cast<int>(gen() % 5u);
        ref = a;
        for (int k = 1; k < n; ++k) ref[k] += ref[k - 1];
        inclusive_scan_inplace(a.data(), n);
        bad += a == ref ? 0 : 1;
        for (const unsigned density : {0u, 1u, 50u, 100u}) {   // percent of the entries set
            std::vector<char> on(static_cast<size_t>(n));
            for (char& x : on) x = gen() % 100u < density;
            std::vector<int> want, got = {-7};   // appended after what out holds
            for (int k = 0; k < n; ++k)
                if (on[k]) want.push_back(k);
            collect_indices(0, n, [&](int k) { return on[k] != 0; }, got);
            bad += got.size() == want.size() + 1 && got[0] == -7 && std::equal(want.begin(), want.end(), got.begin() + 1) ? 0 : 1;
            MarkArray mark(static_cast<size_t>(n) + 1);
            for (int k = 0; k < n; ++k)
                if (on[k]) mark[static_cast<size_t>(k)].store(1);
            std::vector<int> taken;
            take_marked(mark, 0, n, taken);
            bad += taken == want ? 0 : 1;
            for (int k = 0; k < n; ++k) bad += mark[static_cast<size_t>(k)].load() ? 1 : 0;
        }
    }
    // merges: distinct keys split between the two runs at random, or one run entirely below the other
    for (const int n : {2, 100, t - 1, t, t + 1, 64 * 400 + 37, 100003}) {
        for (int mode = 0; mode < 6; ++mode) {
            std::vector<long long> keys(static_cast<size_t>(n));
            for (int k = 0; k < n; ++k) keys[k] = 3ll * k + static_cast<long long>(gen() % 3u);
            std::vector<long long> a, b;
            for (int k = 0; k < n; ++k) {
                bool to_a;
                switch (mode) {
                case 0: to_a = gen() % 2u == 0u; break;     // even split
                case 1: to_a = k != n / 2; break;           // b holds one entry
                case 2: to_a = k % 997 != 3; break;         // b sparse
                case 3: to_a = k % 997 == 3; break;         // a sparse
                case 4: to_a = k < n / 3; break;            // a entirely below b
                default: to_a = k >= n / 3; break;          // a entirely above b
                }
                (to_a ? a : b).push_back(keys[k]);
            }
            std::vector<long long> want(static_cast<size_t>(n)), got(static_cast<size_t>(n), -1);
            const auto less = [](long long x, long long y) { return x < y; };
            std::merge(a.begin(), a.end(), b.begin(), b.end(), want.begin(), less);
            parallel_merge(a.data(), static_cast<int>(a.size()), b.data(), static_cast<int>(b.size()), got.data(), less);
            bad += got == want ? 0 : 1;
        }
    }
    CHECK(bad == 0);
}

TEST(remesh_tables_do_not_depend_on_the_thread_count) {
    // A remesh above kOmpMinWork (every parallel pass of the indicator, the adaptation, the face and side lists and the engine
    // build runs in parallel): x periodic, y walled, an immersed body with its finest-level band, steady sources. After each of
    // 12 remeshes (a moving front: refinements and joins) the leaves, the state, the faces, the gather lists, the incremental
    // bookkeeping, the indicator and every engine table must be the same bits with 1, 3 and 7 threads.
#ifdef _OPENMP
    const EOS eos = EOS::ideal(1.4);
    const int nx = 128, ny = 64, maxl = 2;
    const double Lx = 2.0, Ly = 1.0;
    const Mesh base = generate_cartesian_mesh(Lx, Ly, nx, ny);
    BoundaryManager bm(base, eos);
    bm.set_boundary_types(BCType::Periodic, BCType::Periodic, BCType::Wall, BCType::Wall);
    RiemannSolver rs(eos, RiemannType::HLLC);
    MusclReconstruction mu(eos, Limiter::VanLeer);
    Params src;
    src.set("source_rho", "0.1*sin(x)*cos(y)");
    CellPhysics ph;
    ph.source = &src;
    ph.source_mesh = &base;
    const auto body = [](const Mesh& m) { return ImmersedBoundary(m, "Cylinder", 0.12, 1.2, 1.0, 1.0, 1.0); };
    ph.ibm_factory = body;
    CellIndicator ind;
    ind.use_velocity = true;
    struct Snap {
        std::vector<int> level, ci, cj, side_ptr, side_face, prev_leaf, prev_face, face_level_begin;
        std::vector<char> topo_dirty;
        std::vector<CellFace> faces;
        Field U;
        std::vector<double> xi, xi_parent;
        std::unique_ptr<CellFluxEngine> eng;
    };
    long long joins = 0;
    const auto run = [&](int threads) {
        omp_set_num_threads(threads);
        std::vector<Snap> out;
        CellMesh m(Lx, Ly, nx, ny, maxl, true, false);
        auto eng = std::make_unique<CellFluxEngine>(eos, rs, &mu, bm);
        eng->set_physics(ph);
        eng->build(m);
        CellAdaptor ad(m, *eng, eos, ind);
        ad.set_body(body, 2);
        for (int it = 0; it < 12; ++it) {
            // a density front and a vortex that move across the domain
            const double fx = 0.2 + 0.13 * it, vx = 1.6 - 0.11 * it, vy = 0.5 + 0.2 * std::sin(0.9 * it);
            m.U.resize(static_cast<size_t>(m.num_leaves()));
            for (int c = 0; c < m.num_leaves(); ++c) {
                const double x = m.xc(c), y = m.yc(c), r2 = (x - vx) * (x - vx) + (y - vy) * (y - vy);
                const double rho = 1.0 + 0.5 * std::tanh((x - fx) / 0.03), g = 2.0 * std::exp(-r2 / 0.01);
                m.U[c] = eos.prim_to_cons(State{rho, -g * (y - vy), g * (x - vx), 1.0, 0.0});
            }
            Snap s;
            const CellIndicator::Result ev = ind.evaluate(m, m.U, eos);
            s.xi = ev.xi;
            s.xi_parent = ev.xi_parent;
            ad.remesh();
            s.level = m.level;
            s.ci = m.ci;
            s.cj = m.cj;
            s.U = m.U;
            s.faces = m.faces;
            s.side_ptr = m.side_ptr;
            s.side_face = m.side_face;
            s.prev_leaf = m.prev_leaf;
            s.prev_face = m.prev_face;
            s.topo_dirty = m.topo_dirty;
            s.face_level_begin = m.face_level_begin;
            s.eng = std::make_unique<CellFluxEngine>(*eng);
            out.push_back(std::move(s));
        }
        joins = ad.n_joined;
        return out;
    };
    const std::vector<Snap> one = run(1);
    int bad = 0, big = 0;
    for (const int threads : {3, 7}) {
        const std::vector<Snap> many = run(threads);
        for (size_t k = 0; k < one.size(); ++k) {
            const Snap &a = one[k], &b = many[k];
            bool same = a.level == b.level && a.ci == b.ci && a.cj == b.cj && bits_equal(a.U, b.U) && a.side_ptr == b.side_ptr &&
                        a.side_face == b.side_face && a.prev_leaf == b.prev_leaf && a.prev_face == b.prev_face &&
                        a.topo_dirty == b.topo_dirty && a.face_level_begin == b.face_level_begin &&
                        a.faces.size() == b.faces.size() && a.xi.size() == b.xi.size() &&
                        std::memcmp(a.xi.data(), b.xi.data(), a.xi.size() * sizeof(double)) == 0 &&
                        std::memcmp(a.xi_parent.data(), b.xi_parent.data(), a.xi.size() * sizeof(double)) == 0;
            for (size_t f = 0; same && f < a.faces.size(); ++f) {
                const CellFace &p = a.faces[f], &q = b.faces[f];
                same = p.left == q.left && p.right == q.right && p.level == q.level && p.tpos == q.tpos && p.axis == q.axis &&
                       p.kind == q.kind && p.side == q.side && std::memcmp(&p.length, &q.length, sizeof(double)) == 0;
            }
            same = same && a.eng->same_tables(*b.eng);
            bad += same ? 0 : 1;
            if (!same) std::printf("    %d threads: remesh %d differs\n", threads, static_cast<int>(k));
        }
    }
    for (const Snap& k : one) big += k.level.size() > 4096 ? 1 : 0;
    omp_set_num_threads(omp_get_num_procs());
    CHECK(big == static_cast<int>(one.size()));
    CHECK(joins > 0);
    CHECK(bad == 0);
#endif
}
