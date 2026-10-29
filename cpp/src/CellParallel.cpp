#include "fv/CellParallel.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "fv/CellRemesh.hpp"
#include "fv/CellFlux.hpp"

namespace fv {

namespace {

constexpr uint64_t kNoKey = ~static_cast<uint64_t>(0);   // above every Morton key of the mesh

uint64_t leaf_key(const CellMesh& m, int c) {
    return morton_key(static_cast<uint32_t>(m.ci[static_cast<size_t>(c)]), static_cast<uint32_t>(m.cj[static_cast<size_t>(c)]));
}

}   // namespace

int CellParallel::owner_of(int level, uint64_t key) const {
    const std::vector<uint64_t>& sp = split_[static_cast<size_t>(level)];
    // the last rank whose first key is <= key (ranks with an empty block share the first key of the next one)
    return static_cast<int>(std::upper_bound(sp.begin(), sp.end() - 1, key) - sp.begin()) - 1;
}

void CellParallel::partition(const CellMesh* whole, const GlobalFacts* facts) {
    if (!partition_local(whole, facts))
        throw std::logic_error("CellParallel::partition: the leaves this rank keeps do not cover what its faces read");
    partition_exchange();
}

GlobalFacts CellParallel::global_facts(const std::vector<std::vector<uint64_t>>& keys) const {
    const int R = comm_.size, M = mesh_.max_level;
    const size_t L = static_cast<size_t>(M) + 1;
    GlobalFacts gf;
    gf.count.assign(L, 0);
    gf.split.assign(L, std::vector<uint64_t>(static_cast<size_t>(R) + 1, kNoKey));
    // all ranks' values of a vector of doubles, in rank order
    const auto gather = [&](const std::vector<double>& mine, std::vector<double>& all) {
        const int n = static_cast<int>(mine.size());
        all.resize(static_cast<size_t>(n) * R);
        std::vector<int> counts(static_cast<size_t>(R), n), displs(static_cast<size_t>(R));
        for (int r = 0; r < R; ++r) displs[static_cast<size_t>(r)] = r * n;
        if (R > 1) comm_.allgatherv(mine.data(), n, all.data(), counts.data(), displs.data());
        else all = mine;
    };
    std::vector<double> mine(L), all;
    for (size_t l = 0; l < L; ++l) mine[l] = static_cast<double>(keys[l].size());
    gather(mine, all);
    for (int r = 0; r < R; ++r)
        for (size_t l = 0; l < L; ++l) gf.count[l] += static_cast<long long>(all[static_cast<size_t>(r) * L + l]);
    // the target position of every (level, rank): the key found is the one of that position in the level's order
    struct Target {
        size_t l;
        int r;
        long long pos;
        uint64_t lo, hi;
    };
    std::vector<Target> tg;
    int rounds = 0;
    for (size_t l = 0; l < L; ++l) {
        int bits = 1;
        while ((1 << bits) < std::max(mesh_.nx_at(static_cast<int>(l)), mesh_.ny_at(static_cast<int>(l)))) ++bits;
        rounds = std::max(rounds, 2 * bits);
        for (int r = 0; r < R; ++r) {
            const long long pos = gf.count[l] * r / R;
            if (pos < gf.count[l]) tg.push_back({l, r, pos, 0, (static_cast<uint64_t>(1) << (2 * bits)) - 1});
        }
    }
    std::vector<double> cnt(tg.size()), tot;
    for (int round = 0; round <= rounds; ++round) {
        for (size_t k = 0; k < tg.size(); ++k) {   // the keys <= mid of this rank
            const uint64_t mid = tg[k].lo + (tg[k].hi - tg[k].lo) / 2;
            const std::vector<uint64_t>& kl = keys[tg[k].l];
            cnt[k] = static_cast<double>(std::upper_bound(kl.begin(), kl.end(), mid) - kl.begin());
        }
        gather(cnt, tot);
        for (size_t k = 0; k < tg.size(); ++k) {
            long long below = 0;
            for (int r = 0; r < R; ++r) below += static_cast<long long>(tot[static_cast<size_t>(r) * tg.size() + k]);
            const uint64_t mid = tg[k].lo + (tg[k].hi - tg[k].lo) / 2;
            if (below >= tg[k].pos + 1) tg[k].hi = mid;
            else tg[k].lo = mid + 1;
        }
    }
    for (const Target& t : tg) gf.split[t.l][static_cast<size_t>(t.r)] = t.lo;
    return gf;
}

bool CellParallel::partition_local(const CellMesh* whole, const GlobalFacts* facts) {
    if (localized_) throw std::logic_error("CellParallel::partition: the mesh is localized (globalize() first)");
    // `whole`: the mesh is a partial one (CellMesh::build_from) and the splitters, the levels present and the global indices of the
    // blocks are those of the whole mesh; a face that reads what the rank does not keep makes this return false (nothing
    // communicated yet), on a whole mesh it is an error
    const CellMesh& g = whole ? *whole : mesh_;
    const int N = mesh_.num_leaves(), R = comm_.size, M = mesh_.max_level, me = comm_.rank;
    // splitters: each level cut into contiguous blocks of equal count along its Morton order; rank r owns at level l the keys
    // [split[l][r], split[l][r + 1])
    level_present_.assign(static_cast<size_t>(M) + 1, 0);
    split_.assign(static_cast<size_t>(M) + 1, std::vector<uint64_t>(static_cast<size_t>(R) + 1, kNoKey));
    ob_.assign(static_cast<size_t>(R), std::vector<int>(static_cast<size_t>(M) + 1, 0));
    oe_ = ob_;
    long long b = 0;   // the index of the first leaf of the level in the whole mesh
    for (int l = 0; l <= M; ++l) {
        const long long n = facts ? facts->count[static_cast<size_t>(l)] : g.level_begin[l + 1] - g.level_begin[l];
        level_present_[static_cast<size_t>(l)] = n > 0;
        for (int r = 0; r < R; ++r) {
            const int s0 = static_cast<int>(b + n * r / R), s1 = static_cast<int>(b + n * (r + 1) / R);
            ob_[static_cast<size_t>(r)][static_cast<size_t>(l)] = s0;
            oe_[static_cast<size_t>(r)][static_cast<size_t>(l)] = s1;
            if (facts) split_[static_cast<size_t>(l)][static_cast<size_t>(r)] = facts->split[static_cast<size_t>(l)][static_cast<size_t>(r)];
            else if (s0 < b + n) split_[static_cast<size_t>(l)][static_cast<size_t>(r)] = leaf_key(g, s0);
        }
        b += n;
    }
    owner_.resize(static_cast<size_t>(N));
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
    for (int c = 0; c < N; ++c) owner_[static_cast<size_t>(c)] = owner_of(mesh_.level[static_cast<size_t>(c)], leaf_key(mesh_, c));

    try {
        // what this rank computes and reads: `held` every leaf it reads, `stage` those read in a stage of their own level (by the
        // faces of that level, the ghost fill, the mass-flow averages)
        stage_mask_.assign(static_cast<size_t>(N), 0);
        std::vector<char>& stage = stage_mask_;
        std::vector<char> held(static_cast<size_t>(N), 0);
        CellFluxEngine::Partition part;
        part.face_active.assign(mesh_.faces.size(), 0);
        part.leaf_owned.assign(static_cast<size_t>(N), 0);
        for (int c = 0; c < N; ++c)
            if (owner_[static_cast<size_t>(c)] == me) part.leaf_owned[static_cast<size_t>(c)] = held[static_cast<size_t>(c)] = 1;
        if (whole) {   // the owned leaves must see every side: a partial mesh that is too small shows here, before anything is sent
            for (int c = 0; c < N; ++c)
                if (owner_[static_cast<size_t>(c)] == me)
                    for (int s = 0; s < 4; ++s)
                        if (mesh_.neighbour(c, static_cast<Face>(s)).kind == CellNeighbour::Kind::Outside) return false;
        }
        std::vector<int> reads;
        for (size_t f = 0; f < mesh_.faces.size(); ++f) {
            const CellFace& face = mesh_.faces[f];
            bool mine;
            if (face.kind == CellFaceKind::Boundary) {
                mine = owner_[static_cast<size_t>(face.left >= 0 ? face.left : face.right)] == me;
            } else if (face.left == kOutside || face.right == kOutside) {
                // a face at the edge of what the rank keeps: it must not be one the rank computes
                const int in = face.left == kOutside ? face.right : face.left;
                if (in == kOutside) continue;
                // (a coarse-fine face is computed by the owner of its finer leaf only)
                if (owner_[static_cast<size_t>(in)] == me && (face.kind == CellFaceKind::Interior || mesh_.level[static_cast<size_t>(in)] >= face.level))
                    return false;
                continue;
            } else if (face.kind == CellFaceKind::CoarseFine) {   // the owner of the finer leaf
                mine = owner_[static_cast<size_t>(mesh_.level[face.left] > mesh_.level[face.right] ? face.left : face.right)] == me;
            } else {   // both owners
                mine = owner_[static_cast<size_t>(face.left)] == me || owner_[static_cast<size_t>(face.right)] == me;
            }
            if (!mine) continue;
            part.face_active[f] = 1;
            reads.clear();
            engine_.face_dependencies(mesh_, static_cast<int>(f), reads);
            for (const int c : reads) {
                held[static_cast<size_t>(c)] = 1;
                if (mesh_.level[static_cast<size_t>(c)] == face.level) stage[static_cast<size_t>(c)] = 1;
            }
        }
        // the ghost-cell wall: a rank filling a ghost leaf (finest level) reads the fluid leaves of its image point
        for (const auto& gr : engine_.ghost_reads())
            if (stage[static_cast<size_t>(gr.first)]) stage[static_cast<size_t>(gr.second)] = held[static_cast<size_t>(gr.second)] = 1;
        for (int c = 0; c < N; ++c)   // (a ghost leaf with an image leaf this rank does not keep: the images are not complete)
            if (engine_.ghost_incomplete(c) && (stage[static_cast<size_t>(c)] || owner_[static_cast<size_t>(c)] == me)) return false;
        // the mass-flow averages read their side leaves on every rank, at every stage
        std::vector<char> mf(static_cast<size_t>(N), 0);
        engine_.mass_flow_dependencies(mesh_, mf);
        for (int c = 0; c < N; ++c)
            if (mf[static_cast<size_t>(c)]) stage[static_cast<size_t>(c)] = held[static_cast<size_t>(c)] = 1;
        // the solid reset reads the neighbours of the solid leaves it updates (ghost-cell wall: the image leaves of the ghost leaves)
        for (const auto& gr : engine_.ghost_reads())
            if (owner_[static_cast<size_t>(gr.first)] == me) held[static_cast<size_t>(gr.second)] = 1;
        if (engine_.has_body())
            for (int c = 0; c < N; ++c)
                if (owner_[static_cast<size_t>(c)] == me && engine_.solid()[static_cast<size_t>(c)])
                    for (int s = 0; s < 4; ++s) {
                        const CellNeighbour nb = mesh_.neighbour(c, static_cast<Face>(s));
                        for (const int n : {nb.a, nb.b})
                            if (n >= 0) held[static_cast<size_t>(n)] = 1;
                    }
        part.leaf_held = held;
        face_active_ = part.face_active;
        held_.swap(held);
        engine_.set_partition(mesh_, part);
    } catch (const std::logic_error&) {
        if (!whole) throw;
        return false;
    }

    // receive lists (in the order of the leaves, which is the order of (level, key)) and the requests to the owners: (2 level +
    // stage, i, j) per leaf
    const std::vector<char> no_body;
    const std::vector<char>& solid = engine_.has_body() ? engine_.solid() : no_body;
    stage_.assign(static_cast<size_t>(M) + 1, {});
    final_.assign(static_cast<size_t>(M) + 1, {});
    reg_.assign(static_cast<size_t>(M) + 1, {});
    const auto size_lists = [R](Lists& li) {
        li.send.assign(static_cast<size_t>(R), {});
        li.recv.assign(static_cast<size_t>(R), {});
    };
    for (int l = 0; l <= M; ++l) {
        size_lists(stage_[static_cast<size_t>(l)]);
        size_lists(final_[static_cast<size_t>(l)]);
        size_lists(reg_[static_cast<size_t>(l)]);
    }
    size_lists(solid_);
    req_.assign(static_cast<size_t>(R), {});
    for (int c = 0; c < N; ++c) {
        const int o = owner_[static_cast<size_t>(c)];
        if (!held_[static_cast<size_t>(c)] || o == me) continue;
        const int l = mesh_.level[static_cast<size_t>(c)];
        const bool st = stage_mask_[static_cast<size_t>(c)] != 0;
        final_[static_cast<size_t>(l)].recv[static_cast<size_t>(o)].push_back(c);
        if (st) stage_[static_cast<size_t>(l)].recv[static_cast<size_t>(o)].push_back(c);
        if (!solid.empty() && solid[static_cast<size_t>(c)]) solid_.recv[static_cast<size_t>(o)].push_back(c);
        std::vector<double>& q = req_[static_cast<size_t>(o)];
        q.push_back(2.0 * l + (st ? 1.0 : 0.0));
        q.push_back(static_cast<double>(mesh_.ci[static_cast<size_t>(c)]));
        q.push_back(static_cast<double>(mesh_.cj[static_cast<size_t>(c)]));
    }
    return true;
}

void CellParallel::partition_exchange() {
    const int N = mesh_.num_leaves(), R = comm_.size, M = mesh_.max_level, me = comm_.rank;
    const std::vector<char> no_body;
    const std::vector<char>& solid = engine_.has_body() ? engine_.solid() : no_body;
    if (R > 1) {
        // the number of requests each rank sends to each other rank, then the requests
        std::vector<std::vector<double>> cnt_send(static_cast<size_t>(R)), cnt_recv(static_cast<size_t>(R));
        for (int r = 0; r < R; ++r)
            if (r != me) {
                cnt_send[static_cast<size_t>(r)].assign(1, static_cast<double>(req_[static_cast<size_t>(r)].size()));
                cnt_recv[static_cast<size_t>(r)].assign(1, 0.0);
            }
        comm_.exchange(cnt_send, cnt_recv);
        std::vector<std::vector<double>> got(static_cast<size_t>(R));
        for (int r = 0; r < R; ++r)
            if (r != me) got[static_cast<size_t>(r)].resize(static_cast<size_t>(cnt_recv[static_cast<size_t>(r)][0]));
        comm_.exchange(req_, got);
        for (int r = 0; r < R; ++r) {
            const std::vector<double>& q = got[static_cast<size_t>(r)];
            for (size_t k = 0; k + 2 < q.size(); k += 3) {
                const int code = static_cast<int>(q[k]), l = code / 2;
                const int c = mesh_.find(l, static_cast<int>(q[k + 1]), static_cast<int>(q[k + 2]));
                if (c < 0 || owner_[static_cast<size_t>(c)] != me) throw std::logic_error("CellParallel: a request for a leaf this rank does not own");
                final_[static_cast<size_t>(l)].send[static_cast<size_t>(r)].push_back(c);
                if (code & 1) stage_[static_cast<size_t>(l)].send[static_cast<size_t>(r)].push_back(c);
                if (!solid.empty() && solid[static_cast<size_t>(c)]) solid_.send[static_cast<size_t>(r)].push_back(c);
            }
        }
    }
    std::vector<std::vector<double>>().swap(req_);
    std::vector<char>().swap(stage_mask_);

    // registers of the coarse-fine faces whose two leaves have different owners, listed by (fine leaf, its side) on both ranks
    std::vector<int> ob(static_cast<size_t>(M) + 1, 0), oe(static_cast<size_t>(M) + 1, 0);   // this rank's owned leaves of each level
    for (int l = 0; l <= M; ++l) {
        ob[static_cast<size_t>(l)] = oe[static_cast<size_t>(l)] = mesh_.level_begin[l];
        bool first = true;
        for (int c = mesh_.level_begin[l]; c < mesh_.level_begin[l + 1]; ++c)
            if (owner_[static_cast<size_t>(c)] == me) {
                if (first) ob[static_cast<size_t>(l)] = c;
                first = false;
                oe[static_cast<size_t>(l)] = c + 1;
            }
    }
    for (int l = 1; l <= M; ++l) {
        Lists& rl = reg_[static_cast<size_t>(l)];
        for (int c = ob[static_cast<size_t>(l)]; c < oe[static_cast<size_t>(l)]; ++c)
            for (int s = 0; s < 4; ++s) {   // owned fine leaves: send to the owner of a coarser neighbour
                const CellNeighbour nb = mesh_.neighbour(c, static_cast<Face>(s));
                if (nb.kind != CellNeighbour::Kind::Coarser) continue;
                const int oc = owner_[static_cast<size_t>(nb.a)];
                if (oc != me) rl.send[static_cast<size_t>(oc)].push_back(mesh_.side_face[static_cast<size_t>(mesh_.side_ptr[static_cast<size_t>(4) * c + s])]);
            }
        std::vector<std::vector<std::array<int, 3>>> in(static_cast<size_t>(R));   // (fine leaf, side, face) per fine owner
        for (int C = ob[static_cast<size_t>(l) - 1]; C < oe[static_cast<size_t>(l) - 1]; ++C)
            for (int s = 0; s < 4; ++s) {   // owned coarse leaves: receive from the owners of the finer neighbours
                const CellNeighbour nb = mesh_.neighbour(C, static_cast<Face>(s));
                if (nb.kind != CellNeighbour::Kind::Finer) continue;
                const int fs = static_cast<int>(opposite(static_cast<Face>(s)));
                for (const int fl : {nb.a, nb.b}) {
                    const int of = owner_[static_cast<size_t>(fl)];
                    if (of != me)
                        in[static_cast<size_t>(of)].push_back({fl, fs, mesh_.side_face[static_cast<size_t>(mesh_.side_ptr[static_cast<size_t>(4) * fl + fs])]});
                }
            }
        for (int r = 0; r < R; ++r) {
            std::vector<std::array<int, 3>>& v = in[static_cast<size_t>(r)];
            std::sort(v.begin(), v.end());
            for (const std::array<int, 3>& e : v) rl.recv[static_cast<size_t>(r)].push_back(e[2]);
        }
    }

    if (poison)
        for (int c = 0; c < N; ++c)
            if (!held_[static_cast<size_t>(c)]) mesh_.U[static_cast<size_t>(c)].fill(std::numeric_limits<double>::quiet_NaN());
}

void CellParallel::push_leaves(const Lists& lists) {
    if (comm_.size <= 1) return;
    std::vector<std::vector<double>> send(static_cast<size_t>(comm_.size)), recv(static_cast<size_t>(comm_.size));
    for (int r = 0; r < comm_.size; ++r) {
        for (const int c : lists.send[static_cast<size_t>(r)])
            send[static_cast<size_t>(r)].insert(send[static_cast<size_t>(r)].end(), mesh_.U[static_cast<size_t>(c)].begin(),
                                                mesh_.U[static_cast<size_t>(c)].end());
        recv[static_cast<size_t>(r)].resize(lists.recv[static_cast<size_t>(r)].size() * kNVar);
    }
    comm_.exchange(send, recv);
    for (int r = 0; r < comm_.size; ++r) {
        size_t k = 0;
        for (const int c : lists.recv[static_cast<size_t>(r)])
            for (int q = 0; q < kNVar; ++q) mesh_.U[static_cast<size_t>(c)][static_cast<size_t>(q)] = recv[static_cast<size_t>(r)][k++];
    }
}

void CellParallel::exchange_stage(int level) { push_leaves(stage_[static_cast<size_t>(level)]); }

void CellParallel::exchange_final(int level) { push_leaves(final_[static_cast<size_t>(level)]); }

void CellParallel::exchange_solid() {
    push_leaves(solid_);
    for (int l = 0; l <= mesh_.max_level; ++l) engine_.force_level(mesh_, mesh_.U, l);
}

void CellParallel::exchange_registers(int level) {
    if (comm_.size <= 1) return;
    const Lists& rl = reg_[static_cast<size_t>(level)];
    std::vector<std::vector<double>> send(static_cast<size_t>(comm_.size)), recv(static_cast<size_t>(comm_.size));
    for (int r = 0; r < comm_.size; ++r) {
        for (const int f : rl.send[static_cast<size_t>(r)]) {
            State conv, visc;
            engine_.get_register(f, conv, visc);
            send[static_cast<size_t>(r)].insert(send[static_cast<size_t>(r)].end(), conv.begin(), conv.end());
            send[static_cast<size_t>(r)].insert(send[static_cast<size_t>(r)].end(), visc.begin(), visc.end());
        }
        recv[static_cast<size_t>(r)].resize(rl.recv[static_cast<size_t>(r)].size() * 2 * kNVar);
    }
    comm_.exchange(send, recv);
    for (int r = 0; r < comm_.size; ++r) {
        size_t k = 0;
        for (const int f : rl.recv[static_cast<size_t>(r)]) {
            State conv, visc;
            for (int q = 0; q < kNVar; ++q) conv[static_cast<size_t>(q)] = recv[static_cast<size_t>(r)][k++];
            for (int q = 0; q < kNVar; ++q) visc[static_cast<size_t>(q)] = recv[static_cast<size_t>(r)][k++];
            engine_.set_register(f, conv, visc);
        }
    }
}

void CellParallel::localize() {
    if (localized_) throw std::logic_error("CellParallel::localize: already localized");
    if (face_active_.size() != mesh_.faces.size() || held_.size() != static_cast<size_t>(mesh_.num_leaves()))
        throw std::logic_error("CellParallel::localize: partition() was not called for this mesh");
    const int me = comm_.rank;
    // the faces to keep: those this rank computes, every face of an owned leaf (the gather of its residual, the registers) and
    // the faces of the mass-flow inlets
    std::vector<char> keep_face = face_active_;
    for (size_t f = 0; f < keep_face.size(); ++f) {
        const CellFace& face = mesh_.faces[f];
        if ((face.left >= 0 && owner_[static_cast<size_t>(face.left)] == me) || (face.right >= 0 && owner_[static_cast<size_t>(face.right)] == me))
            keep_face[f] = 1;
    }
    engine_.mass_flow_faces(keep_face);
    std::vector<int> leaf_map, face_map;
    mesh_.keep_partial(held_, keep_face, leaf_map, face_map);
    engine_.localize(mesh_, leaf_map, face_map);
    // this object's lists: leaf indices (stage, final, solid) and face indices (registers)
    const auto remap = [](std::vector<std::vector<int>>& v, const std::vector<int>& map) {
        for (std::vector<int>& row : v)
            for (int& k : row) {
                k = map[static_cast<size_t>(k)];
                if (k < 0) throw std::logic_error("CellParallel::localize: a listed leaf or face is not kept");
            }
    };
    for (std::vector<Lists>* ls : {&stage_, &final_})
        for (Lists& li : *ls) {
            remap(li.send, leaf_map);
            remap(li.recv, leaf_map);
        }
    remap(solid_.send, leaf_map);
    remap(solid_.recv, leaf_map);
    for (Lists& li : reg_) {
        remap(li.send, face_map);
        remap(li.recv, face_map);
    }
    std::vector<int> owner(static_cast<size_t>(mesh_.num_leaves()));
    for (size_t c = 0; c < leaf_map.size(); ++c)
        if (leaf_map[c] >= 0) owner[static_cast<size_t>(leaf_map[c])] = owner_[c];
    owner_.swap(owner);
    held_.assign(static_cast<size_t>(mesh_.num_leaves()), 1);
    for (std::vector<int>& row : ob_) row.assign(row.size(), 0);
    oe_ = ob_;
    for (int l = 0; l <= mesh_.max_level; ++l) {   // this rank's own owned runs (the other ranks' are not known any more)
        ob_[static_cast<size_t>(me)][static_cast<size_t>(l)] = engine_.owned_begin(l);
        oe_[static_cast<size_t>(me)][static_cast<size_t>(l)] = engine_.owned_end(l);
    }
    std::vector<char>().swap(face_active_);
    localized_ = true;
}

void CellParallel::fetch_states(const std::vector<std::array<int, 3>>& want, std::vector<State>& out) const {
    const int R = comm_.size, me = comm_.rank;
    out.assign(want.size(), State{});
    std::vector<std::vector<size_t>> slot(static_cast<size_t>(R));   // per owner: the entries of `want` it answers
    std::vector<std::vector<double>> req(static_cast<size_t>(R));
    for (size_t k = 0; k < want.size(); ++k) {
        const int c = mesh_.find(want[k][0], want[k][1], want[k][2]);
        if (c >= 0) {
            out[k] = mesh_.U[static_cast<size_t>(c)];
            continue;
        }
        const int o = owner_of(want[k][0], morton_key(static_cast<uint32_t>(want[k][1]), static_cast<uint32_t>(want[k][2])));
        if (o == me || R == 1) throw std::logic_error("CellParallel::fetch_states: a leaf of this rank is missing");
        slot[static_cast<size_t>(o)].push_back(k);
        for (const int v : want[k]) req[static_cast<size_t>(o)].push_back(static_cast<double>(v));
    }
    if (R == 1) return;
    std::vector<std::vector<double>> cnt_send(static_cast<size_t>(R)), cnt_recv(static_cast<size_t>(R));
    for (int r = 0; r < R; ++r)
        if (r != me) {
            cnt_send[static_cast<size_t>(r)].assign(1, static_cast<double>(req[static_cast<size_t>(r)].size()));
            cnt_recv[static_cast<size_t>(r)].assign(1, 0.0);
        }
    comm_.exchange(cnt_send, cnt_recv);
    std::vector<std::vector<double>> asked(static_cast<size_t>(R)), reply(static_cast<size_t>(R)), back(static_cast<size_t>(R));
    for (int r = 0; r < R; ++r)
        if (r != me) {
            asked[static_cast<size_t>(r)].resize(static_cast<size_t>(cnt_recv[static_cast<size_t>(r)][0]));
            back[static_cast<size_t>(r)].resize(slot[static_cast<size_t>(r)].size() * kNVar);
        }
    comm_.exchange(req, asked);
    for (int r = 0; r < R; ++r) {
        const std::vector<double>& q = asked[static_cast<size_t>(r)];
        for (size_t k = 0; k + 2 < q.size(); k += 3) {
            const int c = mesh_.find(static_cast<int>(q[k]), static_cast<int>(q[k + 1]), static_cast<int>(q[k + 2]));
            if (c < 0) throw std::logic_error("CellParallel::fetch_states: a request for a leaf this rank does not hold");
            reply[static_cast<size_t>(r)].insert(reply[static_cast<size_t>(r)].end(), mesh_.U[static_cast<size_t>(c)].begin(),
                                                 mesh_.U[static_cast<size_t>(c)].end());
        }
    }
    comm_.exchange(reply, back);
    for (int r = 0; r < R; ++r)
        for (size_t k = 0; k < slot[static_cast<size_t>(r)].size(); ++k)
            for (int q = 0; q < kNVar; ++q)
                out[slot[static_cast<size_t>(r)][k]][static_cast<size_t>(q)] = back[static_cast<size_t>(r)][k * kNVar + static_cast<size_t>(q)];
}

void CellParallel::region_mask(const CellMesh& topo, const std::vector<int>& owned, int extra, std::vector<char>& keep) const {
    const int N = topo.num_leaves(), M = topo.max_level;
    keep.assign(static_cast<size_t>(N), 0);
    std::vector<std::vector<std::array<int, 2>>> seeds(static_cast<size_t>(M) + 1);
    for (const int c : owned) {
        keep[static_cast<size_t>(c)] = 1;
        seeds[static_cast<size_t>(topo.level[static_cast<size_t>(c)])].push_back({topo.ci[static_cast<size_t>(c)], topo.cj[static_cast<size_t>(c)]});
    }
    // rings of the owned leaf's level within which a leaf of another level is read (docs/design/amr-distributed.md section 8.2, a
    // ring more than the airfoil needs; `extra` more rings on a retry), by level difference d = owner level - level read
    const auto rings = [](int d) { return d < 0 ? 4 : (d == 0 ? 6 : (d == 1 ? 8 : (d == 2 ? 9 : 11))); };
    const bool body = engine_.has_body();
    for (int o = 0; o <= M; ++o) {
        if (seeds[static_cast<size_t>(o)].empty()) continue;
        for (int t = 0; t <= M; ++t) {
            if (topo.level_begin[t + 1] == topo.level_begin[t]) continue;
            int r = std::max(0, rings(o - t) + extra);
            if (body && t == o && o == M) r = std::max(r, 8 + extra);   // the images of the ghost cells (7 rings)
            const CellBox b = dilate_cells(seeds[static_cast<size_t>(o)], topo.nx_at(o), topo.ny_at(o), r, topo.periodic_x, topo.periodic_y);
            for (int j = 0; j < b.h; ++j)
                for (int i = 0; i < b.w; ++i) {
                    if (!b.g[static_cast<size_t>(j) * b.w + i]) continue;
                    const int ci = b.i0 + i, cj = b.j0 + j;
                    if (t <= o) {
                        const int c = topo.find(t, ci >> (o - t), cj >> (o - t));
                        if (c >= 0) keep[static_cast<size_t>(c)] = 1;
                    } else {   // the leaves of level t under this cell
                        const int n = 1 << (t - o);
                        for (int qj = 0; qj < n; ++qj)
                            for (int qi = 0; qi < n; ++qi) {
                                const int c = topo.find(t, ci * n + qi, cj * n + qj);
                                if (c >= 0) keep[static_cast<size_t>(c)] = 1;
                            }
                    }
                }
        }
    }
    // the leaves along a mass-flow inlet (its average reads them on every rank)
    for (int s = 0; s < 4; ++s) {
        if (!engine_.mass_flow_side(s)) continue;
        for (int c = 0; c < N; ++c) {
            const int l = topo.level[static_cast<size_t>(c)];
            const bool on = s == 0 ? topo.ci[static_cast<size_t>(c)] == 0
                                   : (s == 1 ? topo.ci[static_cast<size_t>(c)] == topo.nx_at(l) - 1
                                             : (s == 2 ? topo.cj[static_cast<size_t>(c)] == 0 : topo.cj[static_cast<size_t>(c)] == topo.ny_at(l) - 1));
            if (on) keep[static_cast<size_t>(c)] = 1;
        }
    }
}

bool CellParallel::same_lists(const CellParallel& o) const {
    const auto same = [](const std::vector<Lists>& a, const std::vector<Lists>& b) {
        if (a.size() != b.size()) return false;
        for (size_t l = 0; l < a.size(); ++l)
            if (a[l].send != b[l].send || a[l].recv != b[l].recv) return false;
        return true;
    };
    return same(stage_, o.stage_) && same(final_, o.final_) && same(reg_, o.reg_) && solid_.send == o.solid_.send &&
           solid_.recv == o.solid_.recv && owner_ == o.owner_ && held_ == o.held_ && split_ == o.split_ &&
           level_present_ == o.level_present_ && ob_ == o.ob_ && oe_ == o.oe_;
}

void CellParallel::owned_leaves(const CellMesh& topo, std::vector<int>& out) const {
    const int R = comm_.size, me = comm_.rank;
    out.clear();
    for (int l = 0; l <= topo.max_level; ++l) {
        const long long lb = topo.level_begin[static_cast<size_t>(l)], n = topo.level_begin[static_cast<size_t>(l) + 1] - lb;
        for (long long c = lb + n * me / R; c < lb + n * (me + 1) / R; ++c) out.push_back(static_cast<int>(c));
    }
}

void CellParallel::build_share(const CellMesh& topo, const std::vector<int>& mine, const Field* nU,
                               std::chrono::steady_clock::time_point* built) {
    // the leaves within the reach of the owned ones: a partial mesh, its engine tables, the partition; a rank whose share does
    // not cover what its faces read (before anything is sent) makes every rank try again with a wider reach
    for (int extra = first_extra;; extra += 2) {
        bool ok = true;
        std::vector<char> keep;
        region_mask(topo, mine, extra, keep);
        try {
            mesh_.build_from(topo, keep, nU);
            engine_.build(mesh_, &topo);
            ok = partition_local(&topo);
        } catch (const std::logic_error&) {
            ok = false;
        }
        if (reduce_max(ok ? 0.0 : 1.0) == 0.0) break;
        ++region_retries;
        if (extra > 15) throw std::logic_error("CellParallel: no reach of the leaves covers what the faces read");
    }
    if (built) *built = std::chrono::steady_clock::now();
    partition_exchange();
    localize();
}

int CellParallel::init_adapt(CellMesh& topo, CellAdaptor& ad, const std::function<void(CellMesh&)>& fill) {
    localized_ = false;
    std::vector<int> mine;
    owned_leaves(topo, mine);
    build_share(topo, mine, nullptr, nullptr);
    int passes = 0;
    for (; passes < topo.max_level + 1; ++passes) {
        fill(mesh_);
        if (!remesh(topo, ad)) {
            ++passes;
            break;
        }
    }
    fill(mesh_);
    return passes;
}

void CellParallel::exchange_sized(const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv) const {
    const int R = comm_.size, me = comm_.rank;
    recv.assign(static_cast<size_t>(R), {});
    if (R == 1) return;
    std::vector<std::vector<double>> cs(static_cast<size_t>(R)), cr(static_cast<size_t>(R));
    for (int r = 0; r < R; ++r)
        if (r != me) {
            cs[static_cast<size_t>(r)].assign(1, static_cast<double>(send[static_cast<size_t>(r)].size()));
            cr[static_cast<size_t>(r)].assign(1, 0.0);
        }
    comm_.exchange(cs, cr);
    for (int r = 0; r < R; ++r)
        if (r != me) recv[static_cast<size_t>(r)].resize(static_cast<size_t>(cr[static_cast<size_t>(r)][0]));
    comm_.exchange(send, recv);
}

int CellParallel::owner_in(const std::vector<std::vector<uint64_t>>& split, int level, uint64_t key) {
    const std::vector<uint64_t>& sp = split[static_cast<size_t>(level)];
    // (a cell below the first key of the level has no leaf: rank 0 answers that there is none)
    return std::max(0, static_cast<int>(std::upper_bound(sp.begin(), sp.end() - 1, key) - sp.begin()) - 1);
}

void CellParallel::fetch_leaves(const std::vector<std::vector<uint64_t>>& split, const CellMesh& responder, const Field* resp_state,
                                const std::vector<RegionItem>& items, CellMesh& out, FetchPlan* plan) const {
    const int R = comm_.size, me = comm_.rank, M = responder.max_level;
    const bool states = resp_state != nullptr;
    const size_t rec = states ? 2 + kNVar : 2;
    if (plan) {
        plan->from.assign(static_cast<size_t>(R), {});
        plan->served.assign(static_cast<size_t>(R), {});
    }
    std::vector<int> lv, ii, jj;
    Field U;
    std::vector<std::vector<long long>> found(static_cast<size_t>(M) + 1);   // sorted cell numbers (j * nx + i) of the leaves found, per level
    const auto cell_no = [&](int t, int i, int j) { return static_cast<long long>(j) * responder.nx_at(t) + i; };
    for (int t = 0; t <= M; ++t) {
        // the cells of level t the region asks about, minus those a coarser leaf already found covers
        std::vector<long long> cand;
        for (const RegionItem& it : items) {
            if (t < it.tmin || t > it.tmax) continue;
            const CellBox& b = it.box;
            for (int j = 0; j < b.h; ++j)
                for (int i = 0; i < b.w; ++i) {
                    if (!b.g[static_cast<size_t>(j) * b.w + i]) continue;
                    const int ci = b.i0 + i, cj = b.j0 + j;
                    if (t <= it.level) {
                        cand.push_back(cell_no(t, ci >> (it.level - t), cj >> (it.level - t)));
                    } else {
                        const int n = 1 << (t - it.level);
                        for (int qj = 0; qj < n; ++qj)
                            for (int qi = 0; qi < n; ++qi) cand.push_back(cell_no(t, ci * n + qi, cj * n + qj));
                    }
                }
        }
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        std::vector<std::vector<double>> ask(static_cast<size_t>(R));
        std::vector<long long> mine;
        for (const long long no : cand) {
            const int i = static_cast<int>(no % responder.nx_at(t)), j = static_cast<int>(no / responder.nx_at(t));
            bool covered = false;
            for (int s = 0; s < t && !covered; ++s)
                covered = std::binary_search(found[static_cast<size_t>(s)].begin(), found[static_cast<size_t>(s)].end(),
                                             cell_no(s, i >> (t - s), j >> (t - s)));
            if (covered) continue;
            const int o = owner_in(split, t, morton_key(static_cast<uint32_t>(i), static_cast<uint32_t>(j)));
            if (o == me) {
                mine.push_back(no);
            } else {
                ask[static_cast<size_t>(o)].push_back(i);
                ask[static_cast<size_t>(o)].push_back(j);
            }
        }
        std::vector<std::vector<double>> got_ask, reply(static_cast<size_t>(R)), got_reply;
        exchange_sized(ask, got_ask);
        for (int r = 0; r < R; ++r) {   // the leaves asked of this rank that it has
            if (r == me) continue;
            const std::vector<double>& q = got_ask[static_cast<size_t>(r)];
            for (size_t k = 0; k + 1 < q.size(); k += 2) {
                const int c = responder.find(t, static_cast<int>(q[k]), static_cast<int>(q[k + 1]));
                if (c < 0) continue;
                reply[static_cast<size_t>(r)].push_back(q[k]);
                reply[static_cast<size_t>(r)].push_back(q[k + 1]);
                if (states)
                    for (int v = 0; v < kNVar; ++v) reply[static_cast<size_t>(r)].push_back((*resp_state)[static_cast<size_t>(c)][static_cast<size_t>(v)]);
                if (plan) plan->served[static_cast<size_t>(r)].push_back(c);
            }
        }
        exchange_sized(reply, got_reply);
        std::vector<long long>& fl = found[static_cast<size_t>(t)];
        for (const long long no : mine) {
            const int i = static_cast<int>(no % responder.nx_at(t)), j = static_cast<int>(no / responder.nx_at(t));
            const int c = responder.find(t, i, j);
            if (c < 0) continue;
            fl.push_back(no);
            lv.push_back(t);
            ii.push_back(i);
            jj.push_back(j);
            if (states) U.push_back((*resp_state)[static_cast<size_t>(c)]);
        }
        for (int r = 0; r < R; ++r) {
            if (r == me) continue;
            const std::vector<double>& a = got_reply[static_cast<size_t>(r)];
            for (size_t k = 0; k + rec <= a.size(); k += rec) {
                const int i = static_cast<int>(a[k]), j = static_cast<int>(a[k + 1]);
                fl.push_back(cell_no(t, i, j));
                lv.push_back(t);
                ii.push_back(i);
                jj.push_back(j);
                if (states) {
                    State u;
                    for (int v = 0; v < kNVar; ++v) u[static_cast<size_t>(v)] = a[k + 2 + static_cast<size_t>(v)];
                    U.push_back(u);
                }
                if (plan) plan->from[static_cast<size_t>(r)].push_back({t, i, j});
            }
        }
        std::sort(fl.begin(), fl.end());
    }
    out.assign_partial_leaves(lv, ii, jj, states ? &U : nullptr);
}

bool CellParallel::remesh(CellMesh& topo, CellAdaptor& ad) {
    if (!localized_) throw std::logic_error("CellParallel::remesh: the mesh is not localized");
    using Clock = std::chrono::steady_clock;
    const auto sec = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    const int R = comm_.size, me = comm_.rank, M = topo.max_level;
    const auto owned_run = [&](const CellMesh& t, int l, int r, int& b, int& e) {   // the equal-count block of partition()
        const long long lb = t.level_begin[static_cast<size_t>(l)], n = t.level_begin[static_cast<size_t>(l) + 1] - lb;
        b = static_cast<int>(lb + n * r / R);
        e = static_cast<int>(lb + n * (r + 1) / R);
    };
    const auto t0 = Clock::now();
    // 1) this rank's owned leaves (topo indices), the leaves its indicator reads, and their state (held here, or from the owners)
    int N = topo.num_leaves();
    std::vector<int> owned;
    for (int l = 0; l <= M; ++l) {
        int b, e;
        owned_run(topo, l, me, b, e);
        for (int c = b; c < e; ++c) owned.push_back(c);
    }
    std::vector<char> need(static_cast<size_t>(N), 0);
    ad.indicator().region(topo, owned, need);
    for (const int c : owned) need[static_cast<size_t>(c)] = 1;
    if (!sub_)
        sub_ = std::make_unique<CellMesh>(topo.Lx, topo.Ly, topo.nx, topo.ny, topo.max_level, topo.periodic_x, topo.periodic_y,
                                          topo.x0, topo.y0);
    CellMesh& sub = *sub_;
    sub.level.clear();
    sub.ci.clear();
    sub.cj.clear();
    sub.level_begin.assign(static_cast<size_t>(M) + 2, 0);
    std::vector<std::array<int, 3>> want;
    for (int c = 0; c < N; ++c)
        if (need[static_cast<size_t>(c)]) {
            const int l = topo.level[static_cast<size_t>(c)];
            sub.level.push_back(l);
            sub.ci.push_back(topo.ci[static_cast<size_t>(c)]);
            sub.cj.push_back(topo.cj[static_cast<size_t>(c)]);
            ++sub.level_begin[static_cast<size_t>(l) + 1];
            want.push_back({l, topo.ci[static_cast<size_t>(c)], topo.cj[static_cast<size_t>(c)]});
        }
    for (int l = 0; l <= M; ++l) sub.level_begin[static_cast<size_t>(l) + 1] += sub.level_begin[static_cast<size_t>(l)];
    fetch_states(want, sub.U);
    const auto t1 = Clock::now();
    // 2) the decisions for the owned leaves, then everybody's (one number per leaf)
    std::vector<int> owned_sub;
    {
        int k = 0;
        size_t o = 0;
        for (int c = 0; c < N && o < owned.size(); ++c) {
            if (!need[static_cast<size_t>(c)]) continue;
            if (c == owned[o]) {
                owned_sub.push_back(k);
                ++o;
            }
            ++k;
        }
    }
    std::vector<char> rf, cf;
    ad.flags(sub, sub.U, owned_sub, rf, cf);
    std::vector<double> mine(owned.size());
    for (size_t k = 0; k < owned.size(); ++k) mine[k] = static_cast<double>(rf[k] + 2 * cf[k]);
    std::vector<int> counts(static_cast<size_t>(R), 0), displs(static_cast<size_t>(R), 0);
    for (int r = 0; r < R; ++r) {
        for (int l = 0; l <= M; ++l) {
            int b, e;
            owned_run(topo, l, r, b, e);
            counts[static_cast<size_t>(r)] += e - b;
        }
        if (r > 0) displs[static_cast<size_t>(r)] = displs[static_cast<size_t>(r) - 1] + counts[static_cast<size_t>(r) - 1];
    }
    std::vector<double> all(static_cast<size_t>(N));
    if (R > 1) comm_.allgatherv(mine.data(), static_cast<int>(mine.size()), all.data(), counts.data(), displs.data());
    else all = mine;
    std::vector<char> refine(static_cast<size_t>(N), 0), coarsen(static_cast<size_t>(N), 0);
    long long nr = 0;
    for (int r = 0, k = 0; r < R; ++r)
        for (int l = 0; l <= M; ++l) {
            int b, e;
            owned_run(topo, l, r, b, e);
            for (int c = b; c < e; ++c, ++k) {
                const int code = static_cast<int>(all[static_cast<size_t>(k)]);
                refine[static_cast<size_t>(c)] = static_cast<char>(code & 1);
                coarsen[static_cast<size_t>(c)] = static_cast<char>(code >> 1);
                nr += code & 1;
            }
        }
    const auto t2 = Clock::now();
    // 3) the same adapt on every rank (ripple, merges), which also says where the state of every new leaf comes from
    const std::vector<int> old_level = topo.level, old_ci = topo.ci, old_cj = topo.cj;
    topo.record_sources = true;
    const int before = N;
    const bool changed = topo.adapt(refine, coarsen);
    topo.record_sources = false;
    t_topo += sec(t2, Clock::now());
    t_region += sec(t0, t1);
    t_flags += sec(t1, t2);
    ad.t_indicator += sec(t0, t2);
    if (!changed) return false;
    N = topo.num_leaves();
    ad.n_refined += nr;
    ad.n_joined += std::max(0, before + 3 * static_cast<int>(nr) - N) / 3;
    // 4) the state of the new owned leaves: the old leaves they come from (held here, or fetched), then injection / mean
    std::vector<int> mine_new;
    for (int l = 0; l <= M; ++l) {
        int b, e;
        owned_run(topo, l, me, b, e);
        for (int c = b; c < e; ++c) mine_new.push_back(c);
    }
    std::vector<int> from;   // the old leaves read
    for (const int c : mine_new)
        for (const int s : topo.source[static_cast<size_t>(c)])
            if (s >= 0) from.push_back(s);
    std::sort(from.begin(), from.end());
    from.erase(std::unique(from.begin(), from.end()), from.end());
    std::vector<std::array<int, 3>> ask(from.size());
    for (size_t k = 0; k < from.size(); ++k)
        ask[k] = {old_level[static_cast<size_t>(from[k])], old_ci[static_cast<size_t>(from[k])], old_cj[static_cast<size_t>(from[k])]};
    std::vector<State> got;
    fetch_states(ask, got);
    const auto state_of = [&](int old_leaf) -> const State& {
        return got[static_cast<size_t>(std::lower_bound(from.begin(), from.end(), old_leaf) - from.begin())];
    };
    Field nU(static_cast<size_t>(N));
    for (State& u : nU) u.fill(std::numeric_limits<double>::quiet_NaN());
    for (const int c : mine_new) {
        const std::array<int, 4>& s = topo.source[static_cast<size_t>(c)];
        if (s[1] < 0) {
            nU[static_cast<size_t>(c)] = state_of(s[0]);   // stays, or the injection into a child
        } else {
            const State &a = state_of(s[0]), &b = state_of(s[1]), &cc = state_of(s[2]), &d = state_of(s[3]);
            State m;
            for (size_t q = 0; q < static_cast<size_t>(kNVar); ++q) m[q] = ((a[q] + b[q]) + (cc[q] + d[q])) * 0.25;
            nU[static_cast<size_t>(c)] = m;
        }
    }
    std::vector<std::array<int, 4>>().swap(topo.source);
    const auto t3 = Clock::now();
    // 5) this rank's mesh and engine for the new leaves, the new partition, and the halo
    localized_ = false;
    const auto build_whole = [&](CellMesh& m, CellFluxEngine& e) {   // the whole mesh and engine, then the share is extracted
        m = topo;
        m.set_faceless(false);
        m.U = nU;
        m.prev_leaf.clear();   // the engine's tables are of the localized mesh: a full build
        m.topo_dirty.clear();
        m.prev_face.clear();
        e.build(m);
    };
    std::unique_ptr<CellMesh> shadow_mesh;
    std::unique_ptr<CellFluxEngine> shadow_engine;
    std::unique_ptr<CellParallel> shadow_par;
    if (!direct_build || shadow) {
        CellMesh& m = direct_build ? *(shadow_mesh = std::make_unique<CellMesh>(topo)) : mesh_;
        CellFluxEngine copy(engine_);
        CellFluxEngine& e = direct_build ? *(shadow_engine = std::make_unique<CellFluxEngine>(copy)) : engine_;
        build_whole(m, e);
        if (direct_build) {
            shadow_par = std::make_unique<CellParallel>(m, e, comm_);
            shadow_par->partition();
            shadow_par->localize();
        }
    }
    auto t4 = Clock::now();
    if (direct_build) {
        build_share(topo, mine_new, &nU, &t4);
        if (shadow) {
            // the same share made the old way must be this one
            for (int l = 0; l <= M; ++l) shadow_par->exchange_final(l);
            const bool sm = mesh_.same_arrays(*shadow_mesh), se = engine_.same_tables(*shadow_engine, &mesh_), sl = same_lists(*shadow_par);
            if (!sm || !se || !sl) std::fprintf(stderr, "shadow rank %d: mesh %d engine %d lists %d\n", comm_.rank, sm, se, sl);
            // every rank throws together (a rank that threw alone would leave the others waiting in a collective)
            if (reduce_max(sm && se && sl ? 0.0 : 1.0) != 0.0)
                throw std::logic_error("CellParallel::remesh: the directly built share differs from the extracted one");
        }
    } else {
        partition();
        localize();
    }
    for (int l = 0; l <= M; ++l) exchange_final(l);
    const auto t5 = Clock::now();
    t_migrate += sec(t2, t3);
    t_rebuild += sec(t3, t4);
    t_repartition += sec(t4, t5);
    ad.t_adapt += sec(t2, t3);
    ad.t_build += sec(t3, t4);
    return true;
}

void CellParallel::gather_owned(std::vector<double>& all, bool root_only) const {
    const int R = comm_.size, M = mesh_.max_level;
    std::vector<double> send;
    for (int l = 0; l <= M; ++l)
        for (int c = engine_.owned_begin(l); c < engine_.owned_end(l); ++c) {
            send.push_back(l);
            send.push_back(mesh_.ci[static_cast<size_t>(c)]);
            send.push_back(mesh_.cj[static_cast<size_t>(c)]);
            send.insert(send.end(), mesh_.U[static_cast<size_t>(c)].begin(), mesh_.U[static_cast<size_t>(c)].end());
        }
    std::vector<int> counts(static_cast<size_t>(R), 1), displs(static_cast<size_t>(R));
    std::vector<double> mine(1, static_cast<double>(send.size())), sizes(static_cast<size_t>(R));
    for (int r = 0; r < R; ++r) displs[static_cast<size_t>(r)] = r;
    if (R > 1) comm_.allgatherv(mine.data(), 1, sizes.data(), counts.data(), displs.data());
    else sizes[0] = mine[0];
    long long total = 0;
    for (int r = 0; r < R; ++r) {
        counts[static_cast<size_t>(r)] = static_cast<int>(sizes[static_cast<size_t>(r)]);
        displs[static_cast<size_t>(r)] = static_cast<int>(total);
        total += counts[static_cast<size_t>(r)];
    }
    if (R > 1 && root_only) {   // every rank sends its records to rank 0
        const int me = comm_.rank;
        std::vector<std::vector<double>> out(static_cast<size_t>(R)), in(static_cast<size_t>(R));
        if (me != 0) out[0].swap(send);
        else
            for (int r = 1; r < R; ++r) in[static_cast<size_t>(r)].resize(static_cast<size_t>(counts[static_cast<size_t>(r)]));
        comm_.exchange(out, in);
        all.clear();
        if (me == 0) {
            all.reserve(static_cast<size_t>(total));
            all.insert(all.end(), send.begin(), send.end());
            for (int r = 1; r < R; ++r) {
                all.insert(all.end(), in[static_cast<size_t>(r)].begin(), in[static_cast<size_t>(r)].end());
                std::vector<double>().swap(in[static_cast<size_t>(r)]);
            }
        }
        return;
    }
    all.resize(static_cast<size_t>(total));
    if (R > 1) comm_.allgatherv(send.data(), static_cast<int>(send.size()), all.data(), counts.data(), displs.data());
    else all = send;
}

void CellParallel::collect(std::vector<int>& level, std::vector<int>& ci, std::vector<int>& cj, Field& U, bool root_only) const {
    constexpr size_t rec = 3 + kNVar;
    std::vector<double> all;
    gather_owned(all, root_only);
    const size_t n = all.size() / rec;
    std::vector<size_t> order(n);
    for (size_t k = 0; k < n; ++k) order[k] = k;
    std::vector<uint64_t> key(n);
    for (size_t k = 0; k < n; ++k)
        key[k] = morton_key(static_cast<uint32_t>(all[rec * k + 1]), static_cast<uint32_t>(all[rec * k + 2]));
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const int la = static_cast<int>(all[rec * a]), lb = static_cast<int>(all[rec * b]);
        return la != lb ? la < lb : key[a] < key[b];
    });
    level.resize(n);
    ci.resize(n);
    cj.resize(n);
    U.resize(n);
    for (size_t k = 0; k < n; ++k) {
        const double* r = &all[rec * order[k]];
        level[k] = static_cast<int>(r[0]);
        ci[k] = static_cast<int>(r[1]);
        cj[k] = static_cast<int>(r[2]);
        for (int q = 0; q < kNVar; ++q) U[k][static_cast<size_t>(q)] = r[3 + q];
    }
}

void CellParallel::globalize() {
    if (!localized_) return;
    constexpr size_t rec = 3 + kNVar;
    std::vector<double> all;
    gather_owned(all);
    const size_t n = all.size() / rec;
    std::vector<int> lv(n), ii(n), jj(n);
    Field U(n);
    for (size_t k = 0; k < n; ++k) {
        const double* r = &all[rec * k];
        lv[k] = static_cast<int>(r[0]);
        ii[k] = static_cast<int>(r[1]);
        jj[k] = static_cast<int>(r[2]);
        for (int q = 0; q < kNVar; ++q) U[k][static_cast<size_t>(q)] = r[3 + q];
    }
    mesh_.assign_leaves(lv, ii, jj, &U);   // sorts by (level, key) and rebuilds the faces; the engine and the partition are stale
    owner_.clear();
    held_.clear();
    for (std::vector<Lists>* ls : {&stage_, &final_, &reg_}) ls->clear();
    solid_ = Lists();
    localized_ = false;
}

void CellParallel::gather_state() {
    if (localized_) throw std::logic_error("CellParallel::gather_state: the mesh is localized (use collect() or globalize())");
    if (comm_.size <= 1) return;
    const int R = comm_.size, M = mesh_.max_level;
    std::vector<int> counts(static_cast<size_t>(R), 0), displs(static_cast<size_t>(R), 0);
    for (int r = 0; r < R; ++r) {
        for (int l = 0; l <= M; ++l) counts[static_cast<size_t>(r)] += (oe_[static_cast<size_t>(r)][static_cast<size_t>(l)] - ob_[static_cast<size_t>(r)][static_cast<size_t>(l)]) * kNVar;
        if (r > 0) displs[static_cast<size_t>(r)] = displs[static_cast<size_t>(r) - 1] + counts[static_cast<size_t>(r) - 1];
    }
    std::vector<double> send, recv(static_cast<size_t>(displs.back() + counts.back()));
    send.reserve(static_cast<size_t>(counts[static_cast<size_t>(comm_.rank)]));
    for (int l = 0; l <= M; ++l)
        for (int c = ob_[static_cast<size_t>(comm_.rank)][static_cast<size_t>(l)]; c < oe_[static_cast<size_t>(comm_.rank)][static_cast<size_t>(l)]; ++c)
            send.insert(send.end(), mesh_.U[static_cast<size_t>(c)].begin(), mesh_.U[static_cast<size_t>(c)].end());
    comm_.allgatherv(send.data(), static_cast<int>(send.size()), recv.data(), counts.data(), displs.data());
    for (int r = 0; r < R; ++r) {
        size_t k = static_cast<size_t>(displs[static_cast<size_t>(r)]);
        for (int l = 0; l <= M; ++l)
            for (int c = ob_[static_cast<size_t>(r)][static_cast<size_t>(l)]; c < oe_[static_cast<size_t>(r)][static_cast<size_t>(l)]; ++c)
                for (int q = 0; q < kNVar; ++q) mesh_.U[static_cast<size_t>(c)][static_cast<size_t>(q)] = recv[k++];
    }
}

int CellParallel::held_count() const {
    int n = 0;
    for (const char h : held_) n += h != 0;
    return n;
}

int CellParallel::halo_count() const {
    int n = 0;
    for (size_t c = 0; c < held_.size(); ++c) n += held_[c] && owner_[c] != comm_.rank;
    return n;
}

int CellParallel::HaloReach::max(int kind) const {
    int r = 0;
    for (const int k : ring[kind]) r = std::max(r, k);
    return r;
}

int CellParallel::HaloReach::max() const {
    int r = 0;
    for (int k = 0; k < NumKinds; ++k) r = std::max(r, max(k));
    return r;
}

int CellParallel::ring(const CellMesh& m, int a, int b) {
    if (a == b) return 0;
    const int M = m.max_level, sa = 1 << (M - m.level[a]), sb = 1 << (M - m.level[b]);
    // gap between the two squares along one axis, in finest-level cells (0: they touch or overlap), wrapped if periodic
    const auto gap = [](long long a0, long long a1, long long b0, long long b1, long long period) {
        long long g = std::max(0ll, std::max(b0 - a1, a0 - b1));
        if (period > 0) {
            g = std::min(g, std::max(0ll, std::max(b0 - period - a1, a0 - (b1 - period))));
            g = std::min(g, std::max(0ll, std::max(b0 + period - a1, a0 - (b1 + period))));
        }
        return g;
    };
    const long long ax = static_cast<long long>(m.ci[a]) * sa, ay = static_cast<long long>(m.cj[a]) * sa;
    const long long bx = static_cast<long long>(m.ci[b]) * sb, by = static_cast<long long>(m.cj[b]) * sb;
    const long long g = std::max(gap(ax, ax + sa, bx, bx + sb, m.periodic_x ? m.nx_at(M) : 0),
                                 gap(ay, ay + sa, by, by + sb, m.periodic_y ? m.ny_at(M) : 0));
    return static_cast<int>(g / sa) + 1;
}

CellParallel::HaloReach CellParallel::halo_reach(const CellMesh& m, const CellFluxEngine& eng) {
    HaloReach hr;
    for (std::vector<int>& r : hr.ring) r.assign(static_cast<size_t>(2 * m.max_level) + 1, -1);
    const auto note = [&](int kind, int driver, int read) {
        const int k = ring(m, driver, read);
        if (static_cast<int>(hr.reads.size()) <= k) hr.reads.resize(static_cast<size_t>(k) + 1, 0);
        ++hr.reads[static_cast<size_t>(k)];
        int& slot = hr.ring[kind][static_cast<size_t>(m.max_level + m.level[driver] - m.level[read])];
        slot = std::max(slot, k);
    };
    // the image leaves of every ghost leaf
    std::vector<std::vector<int>> images(static_cast<size_t>(m.num_leaves()));
    for (const auto& gr : eng.ghost_reads()) images[static_cast<size_t>(gr.first)].push_back(gr.second);
    std::vector<int> reads;
    for (int f = 0; f < static_cast<int>(m.faces.size()); ++f) {
        const CellFace& face = m.faces[static_cast<size_t>(f)];
        int drivers[2] = {-1, -1};
        int kind = HaloReach::Interior;
        if (face.kind == CellFaceKind::Boundary) {
            drivers[0] = face.left >= 0 ? face.left : face.right;
            kind = HaloReach::Boundary;
        } else if (face.kind == CellFaceKind::CoarseFine) {
            drivers[0] = m.level[face.left] > m.level[face.right] ? face.left : face.right;
            kind = HaloReach::CoarseFine;
        } else {
            drivers[0] = face.left;
            drivers[1] = face.right;
        }
        reads.clear();
        eng.face_dependencies(m, f, reads);
        for (const int d : drivers) {
            if (d < 0) continue;
            for (const int c : reads) {
                note(kind, d, c);
                if (face.level == m.max_level)   // a rank holding a ghost leaf at the finest level fills it from its images
                    for (const int g : images[static_cast<size_t>(c)]) note(HaloReach::GhostImage, d, g);
            }
        }
    }
    for (const auto& gr : eng.ghost_reads()) note(HaloReach::SolidReset, gr.first, gr.second);
    if (eng.has_body())
        for (int c = 0; c < m.num_leaves(); ++c)
            if (eng.solid()[static_cast<size_t>(c)])
                for (int s = 0; s < 4; ++s) {
                    const CellNeighbour nb = m.neighbour(c, static_cast<Face>(s));
                    for (const int n : {nb.a, nb.b})
                        if (n >= 0) note(HaloReach::SolidReset, c, n);
                }
    return hr;
}

}   // namespace fv
