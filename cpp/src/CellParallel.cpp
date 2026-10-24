#include "fv/CellParallel.hpp"

#include <algorithm>
#include <limits>

namespace fv {

void CellParallel::partition() {
    const int N = mesh_.num_leaves(), R = comm_.size, M = mesh_.max_level, me = comm_.rank;
    // owners: each level cut into contiguous blocks of equal count along its Morton order
    owner_.assign(static_cast<size_t>(N), 0);
    ob_.assign(static_cast<size_t>(R), std::vector<int>(static_cast<size_t>(M) + 1, 0));
    oe_ = ob_;
    for (int l = 0; l <= M; ++l) {
        const long long b = mesh_.level_begin[l], n = mesh_.level_begin[l + 1] - b;
        for (int r = 0; r < R; ++r) {
            const int s0 = static_cast<int>(b + n * r / R), s1 = static_cast<int>(b + n * (r + 1) / R);
            ob_[static_cast<size_t>(r)][static_cast<size_t>(l)] = s0;
            oe_[static_cast<size_t>(r)][static_cast<size_t>(l)] = s1;
            for (int c = s0; c < s1; ++c) owner_[static_cast<size_t>(c)] = r;
        }
    }

    // who computes which face, and the leaves each rank reads
    std::vector<std::vector<std::vector<char>>> need_stage(
        static_cast<size_t>(R), std::vector<std::vector<char>>(static_cast<size_t>(M) + 1, std::vector<char>(static_cast<size_t>(N), 0)));
    std::vector<std::vector<char>> need_final(static_cast<size_t>(R), std::vector<char>(static_cast<size_t>(N), 0));
    CellFluxEngine::Partition part;
    part.face_active.assign(mesh_.faces.size(), 0);
    part.leaf_owned.assign(static_cast<size_t>(N), 0);
    for (int c = 0; c < N; ++c) {
        part.leaf_owned[static_cast<size_t>(c)] = owner_[static_cast<size_t>(c)] == me;
        need_final[static_cast<size_t>(owner_[static_cast<size_t>(c)])][static_cast<size_t>(c)] = 1;
    }
    for (size_t f = 0; f < mesh_.faces.size(); ++f) {
        const CellFace& face = mesh_.faces[f];
        int ranks[2] = {-1, -1};
        if (face.kind == CellFaceKind::Boundary) {
            ranks[0] = owner_[static_cast<size_t>(face.left >= 0 ? face.left : face.right)];
        } else if (face.kind == CellFaceKind::CoarseFine) {   // the owner of the finer leaf
            ranks[0] = owner_[static_cast<size_t>(mesh_.level[face.left] > mesh_.level[face.right] ? face.left : face.right)];
        } else {
            ranks[0] = owner_[static_cast<size_t>(face.left)];
            ranks[1] = owner_[static_cast<size_t>(face.right)];
            if (ranks[1] == ranks[0]) ranks[1] = -1;
        }
        for (const int r : ranks) {
            if (r < 0) continue;
            engine_.face_dependencies(mesh_, static_cast<int>(f), need_stage[static_cast<size_t>(r)][static_cast<size_t>(face.level)]);
            if (r == me) part.face_active[f] = 1;
        }
    }
    // the ghost-cell wall: a rank filling a ghost leaf (finest level) reads the fluid leaves of its image point
    for (const auto& gr : engine_.ghost_reads())
        for (int r = 0; r < R; ++r) {
            std::vector<char>& nm = need_stage[static_cast<size_t>(r)][static_cast<size_t>(M)];
            if (nm[static_cast<size_t>(gr.first)]) nm[static_cast<size_t>(gr.second)] = 1;
        }
    // the mass-flow averages read their side leaves on every rank, at every stage
    std::vector<char> mf(static_cast<size_t>(N), 0);
    engine_.mass_flow_dependencies(mesh_, mf);
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < N; ++c)
            if (mf[static_cast<size_t>(c)]) need_stage[static_cast<size_t>(r)][static_cast<size_t>(mesh_.level[c])][static_cast<size_t>(c)] = 1;
    // the solid reset reads the neighbours of the solid leaves it updates (ghost-cell wall: the image leaves of the ghost leaves)
    for (const auto& gr : engine_.ghost_reads())
        need_final[static_cast<size_t>(owner_[static_cast<size_t>(gr.first)])][static_cast<size_t>(gr.second)] = 1;
    if (engine_.has_body())
        for (int c = 0; c < N; ++c)
            if (engine_.solid()[static_cast<size_t>(c)])
                for (int s = 0; s < 4; ++s) {
                    const CellNeighbour nb = mesh_.neighbour(c, static_cast<Face>(s));
                    for (const int n : {nb.a, nb.b})
                        if (n >= 0) need_final[static_cast<size_t>(owner_[static_cast<size_t>(c)])][static_cast<size_t>(n)] = 1;
                }
    for (int r = 0; r < R; ++r)
        for (int l = 0; l <= M; ++l)
            for (int c = 0; c < N; ++c)
                if (need_stage[static_cast<size_t>(r)][static_cast<size_t>(l)][static_cast<size_t>(c)]) need_final[static_cast<size_t>(r)][static_cast<size_t>(c)] = 1;
    part.leaf_held = need_final[static_cast<size_t>(me)];
    held_ = part.leaf_held;
    engine_.set_partition(mesh_, part);

    // send / receive lists
    stage_.assign(static_cast<size_t>(M) + 1, {});
    final_.assign(static_cast<size_t>(M) + 1, {});
    reg_.assign(static_cast<size_t>(M) + 1, {});
    for (int l = 0; l <= M; ++l) {
        std::vector<std::vector<char>> ns(static_cast<size_t>(R));
        for (int r = 0; r < R; ++r) ns[static_cast<size_t>(r)] = need_stage[static_cast<size_t>(r)][static_cast<size_t>(l)];
        build_lists(stage_[static_cast<size_t>(l)], ns, mesh_.level_begin[l], mesh_.level_begin[l + 1], nullptr);
        build_lists(final_[static_cast<size_t>(l)], need_final, mesh_.level_begin[l], mesh_.level_begin[l + 1], nullptr);
        if (l == 0) continue;
        Lists& rl = reg_[static_cast<size_t>(l)];
        rl.send.assign(static_cast<size_t>(R), {});
        rl.recv.assign(static_cast<size_t>(R), {});
        for (int f = mesh_.face_level_begin[l]; f < mesh_.face_level_begin[l + 1]; ++f) {
            const CellFace& face = mesh_.faces[static_cast<size_t>(f)];
            if (face.kind != CellFaceKind::CoarseFine) continue;
            const bool fine_left = mesh_.level[face.left] > mesh_.level[face.right];
            const int of = owner_[static_cast<size_t>(fine_left ? face.left : face.right)];
            const int oc = owner_[static_cast<size_t>(fine_left ? face.right : face.left)];
            if (of == oc) continue;
            if (of == me) rl.send[static_cast<size_t>(oc)].push_back(f);
            if (oc == me) rl.recv[static_cast<size_t>(of)].push_back(f);
        }
    }
    std::vector<char> solid_only(static_cast<size_t>(N), 0);
    if (engine_.has_body()) solid_only = engine_.solid();
    build_lists(solid_, need_final, 0, N, &solid_only);

    if (poison)
        for (int c = 0; c < N; ++c)
            if (!held_[static_cast<size_t>(c)]) mesh_.U[static_cast<size_t>(c)].fill(std::numeric_limits<double>::quiet_NaN());
}

void CellParallel::build_lists(Lists& out, const std::vector<std::vector<char>>& need, int b, int e,
                               const std::vector<char>* only) const {
    const int R = comm_.size, me = comm_.rank;
    out.send.assign(static_cast<size_t>(R), {});
    out.recv.assign(static_cast<size_t>(R), {});
    for (int c = b; c < e; ++c) {
        if (only && !(*only)[static_cast<size_t>(c)]) continue;
        const int o = owner_[static_cast<size_t>(c)];
        if (o == me) {
            for (int r = 0; r < R; ++r)
                if (r != me && need[static_cast<size_t>(r)][static_cast<size_t>(c)]) out.send[static_cast<size_t>(r)].push_back(c);
        } else if (need[static_cast<size_t>(me)][static_cast<size_t>(c)]) {
            out.recv[static_cast<size_t>(o)].push_back(c);
        }
    }
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

void CellParallel::gather_state() {
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

}   // namespace fv
