// AMR in three dimensions, phase Q6 (docs/design/amr-octree.md): the distributed run of an octree (ranks emulated by threads).
// Part of the unit tests: see test_common.hpp for the harness; test_main.cpp holds main().
#include "test_amr_common.hpp"

#include <algorithm>
#include <condition_variable>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace {

using BCs = std::array<const char*, 6>;
using KV = std::vector<std::pair<const char*, const char*>>;

// In-process stand-in for MPI (the one of test_amr_distributed.cpp): `size` threads call the same collectives in the same order.
class FakeHub3 {
public:
    explicit FakeHub3(int n) : n_(n), mail_(static_cast<size_t>(n), std::vector<std::vector<double>>(static_cast<size_t>(n))),
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
                if (in.size() != recv[static_cast<size_t>(r)].size()) bad = true;
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

struct Spec3 {
    const char* name;
    BCs bc;
    KV kv;   // parameters (viscosity, body force, sources, IBM, boundary order)
    unsigned seed;
    bool remesh;
};

// The case on `ranks` ranks (0: the serial path without CellParallel) for `steps` steps, remeshed every 2 steps the replicated way
// (the whole mesh back on every rank, CellAdaptor::remesh, partition again); `local`: each rank keeps only its share between remeshes.
// Returns the final state (checked equal on every rank); counts = {held, halo} of the last rank.
Field run3(const Spec3& sp, int ranks, int steps, bool local, std::vector<int>* counts = nullptr, bool dist_remesh = false,
           int* retries = nullptr) {
    const int R = std::max(ranks, 1);
    std::vector<std::unique_ptr<Simulation>> sims;
    for (int r = 0; r < R; ++r) {   // (built here: the case registry is read once, not from the rank threads)
        sims.push_back(std::make_unique<Simulation>());
        KV kv = {{"nx", "8"}, {"ny", "6"}, {"nz", "5"}, {"reconstruction", "muscl"}, {"time_scheme", "SSP Runge-Kutta 2"}};
        kv.insert(kv.end(), sp.kv.begin(), sp.kv.end());
        setup_3d(*sims.back(), sp.bc, kv);
    }
    CellMesh base = octree_of(*sims[0], 2);
    random_adapt(base, sp.seed, 3, 6, 30);
    FakeHub3 hub(R);
    std::vector<Field> result(static_cast<size_t>(R));
    std::vector<std::string> error(static_cast<size_t>(R));
    std::vector<int> held(static_cast<size_t>(R), 0), halo(static_cast<size_t>(R), 0);
    const auto body = [&](int rank) {
        try {
            Simulation& sim = *sims[static_cast<size_t>(rank)];
            CellMesh m = base;
            m.U.resize(static_cast<size_t>(m.num_leaves()));
            for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state3(sim.fluid, m.xc(c), m.yc(c), m.zc(c));
            CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
            CellPhysics ph = physics_of(sim);
            eng.set_physics(ph);
            eng.build(m);
            CellIndicator ind;
            ind.criterion = 0.02;
            CellAdaptor ad(m, eng, sim.fluid, ind);
            if (sim.p.use_ibm) ad.set_body(sim.ibm_factory, 2);
            std::unique_ptr<TimeScheme> ts = make_time_scheme(TimeSchemeType::SSPRK2, 0.4);
            std::unique_ptr<CellParallel> par;
            if (ranks > 0) {
                par = std::make_unique<CellParallel>(m, eng, hub.comm(rank));
                par->poison = true;
                par->shadow = dist_remesh;                 // the share built directly against the one cut from the whole new mesh
                par->first_extra = dist_remesh ? -10 : 1;   // a first reach too small (the 3-D table less 10 rings): the retry runs
            }
            CellAdvance adv(m, eng, *ts);
            adv.par = par.get();
            const double L = 2.0 / (1.0 / m.dx(0) + 1.0 / m.dy(0) + 1.0 / m.dz(0));
            const auto wave_speed = [&]() {
                eng.force_state(m, m.U);
                const double s = eng.wave_speed(m, m.U);
                return par ? par->reduce_max(s) : s;
            };
            double S = wave_speed(), t = 0.0;
            if (par) par->partition();
            if (par && local) par->localize();
            for (int k = 0; k < steps; ++k) {
                const double dt = 0.4 * L / S;
                S = adv.step(dt, t);
                t += dt;
                if (sim.p.use_ibm) {
                    eng.solid_reset(m, m.U);
                    if (par) par->exchange_solid();
                }
                if (sp.remesh && (k + 1) % 2 == 0 && par && dist_remesh) {
                    if (par->remesh(ad)) S = wave_speed();   // the topology-free remesh of the shares
                } else if (sp.remesh && (k + 1) % 2 == 0) {
                    if (par && local) par->globalize();
                    else if (par) par->gather_state();
                    const bool changed = ad.remesh();
                    if (changed) S = wave_speed();
                    else if (par && local) eng.build(m);   // the localized tables are gone
                    if (par) par->partition();
                    if (par && local) par->localize();
                }
            }
            if (par && retries && rank == 0) *retries = par->region_retries;
            if (par) {
                held[static_cast<size_t>(rank)] = par->held_count();
                halo[static_cast<size_t>(rank)] = par->halo_count();
                if (local) {
                    CHECK(m.partial && m.num_leaves() == par->held_count());
                    par->globalize();
                } else {
                    par->gather_state();
                }
            }
            result[static_cast<size_t>(rank)] = m.U;
        } catch (const std::exception& e) {
            error[static_cast<size_t>(rank)] = e.what();
        }
    };
    std::vector<std::thread> threads;
    for (int r = 0; r < R; ++r) threads.emplace_back(body, r);
    for (std::thread& th : threads) th.join();
    for (int r = 0; r < R; ++r) {
        if (!error[static_cast<size_t>(r)].empty()) std::printf("    %s, rank %d: %s\n", sp.name, r, error[static_cast<size_t>(r)].c_str());
        CHECK(error[static_cast<size_t>(r)].empty());
    }
    for (int r = 1; r < R; ++r) CHECK(bits_equal(result[0], result[static_cast<size_t>(r)]));
    if (counts) *counts = {held[static_cast<size_t>(R - 1)], halo[static_cast<size_t>(R - 1)]};
    return result[0];
}

const Spec3 kSpecs[] = {
    {"euler periodic", {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"}, {}, 11u, true},
    {"viscous + body force + source, periodic", {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"},
     {{"use_viscous", "true"}, {"mu", "0.03"}, {"use_bodyforce", "true"}, {"gx", "0.3"}, {"gz", "-0.5"}, {"source_rho", "0.1*sin(3*x)*cos(2*z)"},
      {"source_mz", "0.2*x*y*cos(t)"}},
     12u, true},
    {"walls, nrbc, mass flow, viscous, boundary_order 2", {"massflow_inlet", "nonreflecting", "no_slip", "moving_wall", "wall", "outlet"},
     {{"use_viscous", "true"}, {"mu", "0.03"}, {"boundary_order", "2"}, {"rho_in", "1.1"}}, 13u, true},
    {"immersed sphere (mask), viscous", {"vel_inlet", "outlet", "wall", "wall", "periodic", "periodic"},
     {{"use_viscous", "true"}, {"mu", "0.02"}, {"use_ibm", "true"}, {"ibm_method", "mask"}, {"ibm_shape", "Sphere"}, {"ibm_radius", "0.27"},
      {"ibm_cx", "0.83"}, {"ibm_cy", "0.52"}, {"ibm_cz", "0.47"}},
     14u, true},
};

}   // namespace

TEST(cellparallel3d_equals_serial_bitwise_for_any_rank_count) {
    // The distributed run of an octree (owners per level, halos from face_dependencies with six sides, registers of four finer faces,
    // the solid reset's neighbours), with the halos poisoned (NaN): the final state equals the serial run's bit for bit on 1 - 7 ranks,
    // replicated and localized (each rank keeps only its share), with the remesh of the whole mesh every 2 steps.
    for (const Spec3& sp : kSpecs) {
        const Field serial = run3(sp, 0, 4, false);
        for (const bool local : {false, true})
            for (const int ranks : {1, 2, 3, 5, 7}) {
                std::vector<int> counts;
                const Field dist = run3(sp, ranks, 4, local, &counts);
                const bool same = bits_equal(serial, dist);
                CHECK(same);
                if (!same) std::printf("    %s on %d ranks (%s) differs from the serial run\n", sp.name, ranks, local ? "localized" : "replicated");
                if (ranks >= 5) CHECK(counts[0] < static_cast<int>(serial.size()));
                if (ranks >= 2) CHECK(counts[1] > 0);
            }
    }
}

TEST(cellparallel3d_topology_free_remesh_equals_serial_bitwise) {
    // The remesh of the localized shares with no rank holding the whole mesh (fetch of the region, the decisions of the owned leaves,
    // the ripple rounds, the merges of 8, the migration of the new leaves, the share built from a fetched region), with the shadow
    // check at every remesh and a first reach too small (the retry runs): the final state equals the serial run's bit for bit.
    for (const Spec3& sp : kSpecs) {
        if (!sp.remesh) continue;
        const Field serial = run3(sp, 0, 4, false);
        for (const int ranks : {1, 2, 3, 5, 7}) {
            int retries = 0;
            const Field dist = run3(sp, ranks, 4, true, nullptr, true, &retries);
            const bool same = bits_equal(serial, dist);
            CHECK(same);
            if (!same) std::printf("    %s on %d ranks (topology-free remesh) differs from the serial run\n", sp.name, ranks);
            if (ranks > 1) CHECK(retries > 0);
        }
    }
}

TEST(cellparallel3d_halo_reach_lies_within_the_region_rings) {
    // How far the reads of a distributed 3-D run reach (CellParallel::halo_reach: every read from the leaf whose owner makes it, in
    // rings of that leaf's level, by level difference), with every physics of the 3-D engine, on octrees of three and four levels.
    // The region of a rank's share (CellShare.cpp, reach_rings) must hold them: d = owner level - level read, d < 0: 8 rings, 0: 11,
    // 1: 10, 2: 9, >= 3: 11. A stencil that reaches further must be added there (the retry of a share that is too small would hide it:
    // the remesh tests also run with the retry forced). Prints the measured table. History: the first run used the 2-D table
    // (4, 6, 8, 9, 11) and failed: in 3-D the reads reach 7 rings at d = -1, 10 at d = 0 and 9 at d = 1 (coarse-fine faces two levels
    // apart, the immersed body); the 3-D table is that measurement plus one ring.
    const auto table = [](int d) { return d < 0 ? 8 : (d == 0 ? 11 : (d == 1 ? 10 : (d == 2 ? 9 : 11))); };
    std::map<int, int> measured;   // level difference -> largest ring
    for (const Spec3& sp : kSpecs)
        for (const int max_level : {2, 3}) {
            Simulation sim;
            KV kv = {{"nx", "8"}, {"ny", "6"}, {"nz", "5"}, {"reconstruction", "muscl"}};
            kv.insert(kv.end(), sp.kv.begin(), sp.kv.end());
            setup_3d(sim, sp.bc, kv);
            CellMesh m = octree_of(sim, max_level);
            random_adapt(m, sp.seed + 100u * static_cast<unsigned>(max_level), max_level + 1, 5, 25);
            m.U.resize(static_cast<size_t>(m.num_leaves()));
            for (int c = 0; c < m.num_leaves(); ++c) m.U[c] = test_state3(sim.fluid, m.xc(c), m.yc(c), m.zc(c));
            CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
            eng.set_physics(physics_of(sim));
            eng.build(m);
            const CellParallel::HaloReach hr = CellParallel::halo_reach(m, eng);
            for (int k = 0; k < CellParallel::HaloReach::NumKinds; ++k)
                for (int idx = 0; idx <= 2 * max_level; ++idx) {
                    const int ring = hr.ring[k][static_cast<size_t>(idx)];
                    if (ring < 0) continue;
                    const int d = idx - max_level;
                    measured[d] = std::max(measured[d], ring);
                    CHECK(ring <= table(d));
                    if (ring > table(d)) std::printf("    %s, %d levels: kind %d, level difference %d reaches %d rings (table %d)\n", sp.name, max_level + 1, k, d, ring, table(d));
                }
        }
    std::printf("    measured reach (rings of the owner's level) by level difference:");
    for (const auto& e : measured) std::printf("  %d: %d (table %d)", e.first, e.second, table(e.first));
    std::printf("\n");
    CHECK(measured.count(0) && measured.count(1) && measured.count(-1));
}

TEST(cellparallel3d_distributed_init_equals_serial_init_bitwise) {
    // The adaptation of a fresh start on an octree (CellParallel::init_adapt): every rank builds its share of the faceless base topology
    // and the passes run through the topology-free remesh with the initial condition filled on the share; the leaves, their state and
    // the number of passes are those of CellAdaptor::init_adapt on the whole mesh, bit for bit.
    struct Out {
        std::vector<int> level, ci, cj, ck;
        Field U;
        int passes = 0;
    };
    int adapted = 0;
    for (const Spec3& sp : kSpecs) {
        const auto run = [&](int ranks) {
            const int R = std::max(ranks, 1);
            std::vector<std::unique_ptr<Simulation>> sims;
            for (int r = 0; r < R; ++r) {
                sims.push_back(std::make_unique<Simulation>());
                KV kv = {{"nx", "8"}, {"ny", "6"}, {"nz", "5"}, {"reconstruction", "muscl"}};
                kv.insert(kv.end(), sp.kv.begin(), sp.kv.end());
                setup_3d(*sims.back(), sp.bc, kv);
            }
            FakeHub3 hub(R);
            std::vector<Out> out(static_cast<size_t>(R));
            std::vector<std::string> error(static_cast<size_t>(R));
            const auto body = [&](int rank) {
                try {
                    Simulation& sim = *sims[static_cast<size_t>(rank)];
                    CellMesh m = octree_of(sim, 2);
                    const auto fill = [&](CellMesh& mm) {
                        mm.U.resize(static_cast<size_t>(mm.num_leaves()));
                        for (int c = 0; c < mm.num_leaves(); ++c) mm.U[static_cast<size_t>(c)] = test_state3(sim.fluid, 3.0 * mm.xc(c), 2.0 * mm.yc(c), 2.0 * mm.zc(c));
                    };
                    CellFluxEngine eng(sim.fluid, sim.riemann, sim.muscl.get(), sim.boundaries);
                    eng.set_physics(physics_of(sim));
                    CellIndicator ind;
                    ind.criterion = 0.02;
                    CellAdaptor ad(m, eng, sim.fluid, ind);
                    if (sim.p.use_ibm) ad.set_body(sim.ibm_factory, 2);
                    Out& o = out[static_cast<size_t>(rank)];
                    if (ranks == 0) {
                        eng.build(m);
                        o.passes = ad.init_adapt(fill);
                    } else {
                        CellParallel par(m, eng, hub.comm(rank));
                        par.poison = true;
                        par.first_extra = -10;
                        CellMesh topo = m;
                        topo.U.clear();
                        topo.set_faceless(true);
                        o.passes = par.init_adapt(topo, ad, fill);
                        CHECK(m.partial);
                        par.globalize();
                    }
                    o.level = m.level;
                    o.ci = m.ci;
                    o.cj = m.cj;
                    o.ck = m.ck;
                    o.U = m.U;
                } catch (const std::exception& e) {
                    error[static_cast<size_t>(rank)] = e.what();
                }
            };
            std::vector<std::thread> threads;
            for (int r = 0; r < R; ++r) threads.emplace_back(body, r);
            for (std::thread& th : threads) th.join();
            for (int r = 0; r < R; ++r) {
                if (!error[static_cast<size_t>(r)].empty()) std::printf("    %s, rank %d: %s\n", sp.name, r, error[static_cast<size_t>(r)].c_str());
                CHECK(error[static_cast<size_t>(r)].empty());
            }
            return out;
        };
        const Out serial = run(0)[0];
        adapted += serial.passes > 1;
        for (const int ranks : {1, 2, 3, 5, 8})
            for (const Out& o : run(ranks)) {
                const bool same = o.level == serial.level && o.ci == serial.ci && o.cj == serial.cj && o.ck == serial.ck &&
                                  bits_equal(o.U, serial.U) && o.passes == serial.passes;
                CHECK(same);
                if (!same) std::printf("    %s on %d ranks (distributed init) differs from the serial init\n", sp.name, ranks);
            }
    }
    CHECK(adapted >= 3);
}

TEST(cellparallel3d_ring_and_collect) {
    // ring() in 3-D counts the cells of the first leaf's level along z too (periodic wrap included); collect() on a localized 3-D run
    // returns the leaves of the whole mesh (with k) in (level, key) order with their states.
    CellMesh m = CellMesh::make_3d(1.0, 1.0, 1.0, 4, 4, 4, 1, false, false, true);
    const int a = m.find(0, 1, 1, 0), b = m.find(0, 1, 1, 3), c = m.find(0, 1, 1, 2), d = m.find(0, 3, 1, 0);
    CHECK(CellParallel::ring(m, a, a) == 0);
    CHECK(CellParallel::ring(m, a, b) == 1);   // across the periodic z seam
    CHECK(CellParallel::ring(m, a, c) == 2);
    CHECK(CellParallel::ring(m, a, d) == 2);
    const Spec3& sp = kSpecs[1];
    const int R = 3;
    std::vector<std::unique_ptr<Simulation>> sims;
    for (int r = 0; r < R; ++r) {
        sims.push_back(std::make_unique<Simulation>());
        KV kv = {{"nx", "8"}, {"ny", "6"}, {"nz", "5"}, {"reconstruction", "muscl"}};
        kv.insert(kv.end(), sp.kv.begin(), sp.kv.end());
        setup_3d(*sims.back(), sp.bc, kv);
    }
    CellMesh base = octree_of(*sims[0], 2);
    random_adapt(base, 5u, 3, 6, 30);
    base.U.resize(static_cast<size_t>(base.num_leaves()));
    for (int q = 0; q < base.num_leaves(); ++q) base.U[q] = test_state3(sims[0]->fluid, base.xc(q), base.yc(q), base.zc(q));
    FakeHub3 hub(R);
    std::vector<int> ok(R, 0);
    std::vector<std::thread> th;
    for (int r = 0; r < R; ++r)
        th.emplace_back([&, r]() {
            try {
                CellMesh mm = base;
                CellFluxEngine eng(sims[static_cast<size_t>(r)]->fluid, sims[static_cast<size_t>(r)]->riemann, sims[static_cast<size_t>(r)]->muscl.get(),
                                   sims[static_cast<size_t>(r)]->boundaries);
                eng.set_physics(physics_of(*sims[static_cast<size_t>(r)]));
                eng.build(mm);
                CellParallel par(mm, eng, hub.comm(r));
                par.partition();
                par.localize();
                std::vector<int> lv, ci, cj, ck;
                Field U;
                par.collect(lv, ci, cj, ck, U);
                ok[static_cast<size_t>(r)] = lv == base.level && ci == base.ci && cj == base.cj && ck == base.ck && U == base.U;
            } catch (const std::exception& e) {
                std::printf("    rank %d: %s\n", r, e.what());
            }
        });
    for (std::thread& t : th) t.join();
    for (int r = 0; r < R; ++r) CHECK(ok[static_cast<size_t>(r)] == 1);
}
