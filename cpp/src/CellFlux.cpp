#include "fv/CellFlux.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace fv {

namespace {

struct Builder {
    const CellMesh& m;
    std::vector<int>& vroot;
    std::vector<std::array<int, 4>>& vnode;
    std::map<std::array<int, 3>, int> vmap;   // block (level, i, j) -> virtual cell

    // Code of the block (l, i, j): a leaf, or a node whose children are the four sub-blocks
    int block_code(int l, int i, int j) {
        const int leaf = m.find(l, i, j);
        if (leaf >= 0) return leaf;
        std::array<int, 4> ch{};
        for (int oj = 0; oj < 2; ++oj)
            for (int oi = 0; oi < 2; ++oi) ch[2 * oj + oi] = block_code(l + 1, 2 * i + oi, 2 * j + oj);
        vnode.push_back(ch);
        return -static_cast<int>(vnode.size());   // node k <-> code -1 - k
    }

    // Index into W for the cell beyond leaf c across `side`
    int beyond(int c, Face side) {
        const CellNeighbour nb = m.neighbour(c, side);
        switch (nb.kind) {
        case CellNeighbour::Kind::Boundary: return c;                       // edge replication
        case CellNeighbour::Kind::Same:
        case CellNeighbour::Kind::Coarser: return nb.a;                     // the cell, or injection of the coarse one
        default: {
            const std::array<int, 3> key = {m.level[nb.a] - 1, m.ci[nb.a] >> 1, m.cj[nb.a] >> 1};
            const auto it = vmap.find(key);
            if (it != vmap.end()) return m.num_leaves() + it->second;
            const int id = static_cast<int>(vroot.size());
            vroot.push_back(block_code(key[0], key[1], key[2]));
            vmap[key] = id;
            return m.num_leaves() + id;
        }
        }
    }
};

}   // namespace

State CellFluxEngine::eval(const Field& U, int code) const {
    if (code >= 0) return U[code];
    const std::array<int, 4>& ch = vnode_[static_cast<size_t>(-1 - code)];
    const State a = eval(U, ch[0]), b = eval(U, ch[1]), c = eval(U, ch[2]), d = eval(U, ch[3]);
    State r;
    for (int k = 0; k < kNVar; ++k) r[k] = ((a[k] + b[k]) + (c[k] + d[k])) * 0.25;
    return r;
}

void CellFluxEngine::build(const CellMesh& m) {
    const auto per = [&](Face lo, Face hi) {
        const bool a = bc_.type(lo) == BCType::Periodic, b = bc_.type(hi) == BCType::Periodic;
        if (a != b) throw std::invalid_argument("CellFluxEngine: both sides of a periodic axis must be periodic");
        return a;
    };
    if (per(Face::Left, Face::Right) != m.periodic_x || per(Face::Bottom, Face::Top) != m.periodic_y)
        throw std::invalid_argument("CellFluxEngine: the periodic axes of the mesh and the boundary types differ");
    for (Face s : kAllFaces)
        if (bc_.type(s) == BCType::NonReflecting)
            throw std::invalid_argument("CellFluxEngine: the non-reflecting outlet is not supported yet");

    nleaf_ = m.num_leaves();
    vroot_.clear();
    vnode_.clear();
    Builder b{m, vroot_, vnode_, {}};
    stencil_.assign(m.faces.size(), {-1, -1, -1, -1});
    for (size_t f = 0; f < m.faces.size(); ++f) {
        const CellFace& face = m.faces[f];
        if (face.kind == CellFaceKind::Boundary) continue;
        const Face back = face.axis == 0 ? Face::Left : Face::Bottom, fwd = face.axis == 0 ? Face::Right : Face::Top;
        const int l = face.left, r = face.right;
        std::array<int, 4>& s = stencil_[f];
        s[1] = l;
        s[2] = r;
        const int ll = m.level[l], lr = m.level[r];
        s[0] = ll >= lr ? b.beyond(l, back) : l;   // a coarse c1 keeps the first-order value
        s[3] = lr >= ll ? b.beyond(r, fwd) : r;
    }

    need_.assign(static_cast<size_t>(m.max_level) + 1, {});
    if (muscl_)
        for (int l = 0; l <= m.max_level; ++l) {
            std::vector<int>& v = need_[static_cast<size_t>(l)];
            for (int f = m.face_level_begin[l]; f < m.face_level_begin[l + 1]; ++f)
                if (m.faces[f].kind != CellFaceKind::Boundary) v.insert(v.end(), stencil_[f].begin(), stencil_[f].end());
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
    reg_.assign(m.faces.size(), State{});

    for (int s = 0; s < 4; ++s) {
        mf_face_[s].clear();
        mf_weight_[s].clear();
        if (bc_.type(static_cast<Face>(s)) != BCType::MassflowInlet) continue;
        std::vector<std::pair<long long, int>> order;
        int fine = 0;
        for (size_t f = 0; f < m.faces.size(); ++f) {
            const CellFace& face = m.faces[f];
            if (face.kind != CellFaceKind::Boundary || face.side != static_cast<Face>(s)) continue;
            order.push_back({static_cast<long long>(face.tpos) << (m.max_level - face.level), static_cast<int>(f)});
            fine = std::max(fine, face.level);
        }
        std::sort(order.begin(), order.end());
        for (const auto& o : order) {
            mf_face_[s].push_back(o.second);
            mf_weight_[s].push_back(static_cast<double>(1 << (fine - m.faces[o.second].level)));
        }
    }
}

void CellFluxEngine::refresh_all(const Field& U, int N) {
    W_.resize(static_cast<size_t>(N) + vroot_.size());
    const EOS& eos = eos_;
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
    for (int c = 0; c < N; ++c) W_[c] = eos.cons_to_prim(U[c]);
    for (size_t v = 0; v < vroot_.size(); ++v) W_[N + v] = eos.cons_to_prim(eval(U, vroot_[v]));
}

void CellFluxEngine::refresh_some(const Field& U, int N, const std::vector<int>& idx) {
    W_.resize(static_cast<size_t>(N) + vroot_.size());
    const EOS& eos = eos_;
    const int n = static_cast<int>(idx.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int k = 0; k < n; ++k) {
        const int i = idx[k];
        W_[i] = i < N ? eos.cons_to_prim(U[i]) : eos.cons_to_prim(eval(U, vroot_[static_cast<size_t>(i - N)]));
    }
}

void CellFluxEngine::mass_flow_targets(const CellMesh& m, const Field& U, double* u_target) const {
    const BCParams& bp = bc_.params();
    for (int s = 0; s < 4; ++s) {
        u_target[s] = 0.0;
        if (mf_face_[s].empty()) continue;
        double sum = 0.0, count = 0.0;
        for (size_t k = 0; k < mf_face_[s].size(); ++k) {
            const CellFace& f = m.faces[mf_face_[s][k]];
            sum += eos_.cons_to_prim(U[f.left >= 0 ? f.left : f.right])[0] * mf_weight_[s][k];
            count += mf_weight_[s][k];
        }
        const double rho_avg = std::max(sum / count, 1e-5);
        u_target[s] = (bp.rho_in * bp.u_in) / rho_avg;
    }
}

double CellFluxEngine::fluxes(const CellMesh& m, const Field& U, int fb, int fe, const double* u_target) {
    const EOS& eos = eos_;
    const Limiter lim = muscl_ ? muscl_->limiter() : Limiter::Minmod;
    if (F_.size() != m.faces.size()) F_.resize(m.faces.size());
    double smax_all = 0.0;
    auto roundtrip = [&](State& u) { u = eos.prim_to_cons(eos.cons_to_prim(u)); };
#pragma omp parallel if (fe - fb >= kOmpMinWork)
    {
        double local = 0.0;
#pragma omp for schedule(static) nowait
        for (int f = fb; f < fe; ++f) {
            const CellFace& face = m.faces[f];
            State UL, UR;
            if (face.kind == CellFaceKind::Boundary) {
                const bool ghost_right = face.left >= 0;
                const int c = ghost_right ? face.left : face.right;
                const State Ui = U[c];
                const State Ug = bc_.ghost_state(face.side, Ui, u_target[static_cast<int>(face.side)]);
                UL = ghost_right ? Ui : Ug;
                UR = ghost_right ? Ug : Ui;
                if (muscl_) {
                    roundtrip(UL);
                    roundtrip(UR);
                }
            } else if (!muscl_) {
                UL = U[face.left];
                UR = U[face.right];
            } else {
                const std::array<int, 4>& s = stencil_[f];
                const State &w0 = W_[s[0]], &w1 = W_[s[1]], &w2 = W_[s[2]], &w3 = W_[s[3]];
                State wl, wr;
                for (int q = 0; q < kNVar; ++q) {
                    const double c0 = w0[q], c1 = w1[q], c2 = w2[q], c3 = w3[q];
                    const double bq = c2 - c1;
                    wl[q] = c1 + 0.5 * apply_limiter(lim, c1 - c0, bq);
                    wr[q] = c2 - 0.5 * apply_limiter(lim, c3 - c2, bq);
                }
                UL = eos.prim_to_cons(wl);
                UR = eos.prim_to_cons(wr);
            }
            double sm;
            F_[f] = riemann_.face_flux(UL, UR, face.axis, sm);
            if (sm > local) local = sm;
        }
#pragma omp critical(fv_cellflux_max)
        if (local > smax_all) smax_all = local;
    }
    return smax_all;
}

void CellFluxEngine::gather(const CellMesh& m, int cb, int ce, bool replace, Field& R) {
    R.resize(static_cast<size_t>(ce - cb));
#pragma omp parallel for schedule(static) if (ce - cb >= kOmpMinWork)
    for (int c = cb; c < ce; ++c) {
        // side integrals: sum of (face flux * face length), one or two faces per side
        State G[4];
        for (int s = 0; s < 4; ++s) {
            const int b = m.side_ptr[4 * c + s], e = m.side_ptr[4 * c + s + 1];
            for (int k = b; k < e; ++k) {
                const int fi = m.side_face[k];
                const CellFace& f = m.faces[fi];
                // a finer face on the side of this leaf: its coarse side uses the registered time average
                const State& Fk = (replace && f.level > m.level[c]) ? reg_[fi] : F_[fi];
                if (k == b)
                    for (int q = 0; q < kNVar; ++q) G[s][q] = Fk[q] * f.length;
                else
                    for (int q = 0; q < kNVar; ++q) G[s][q] += Fk[q] * f.length;
            }
        }
        const double area = m.area(c);
        for (int q = 0; q < kNVar; ++q) R[c - cb][q] = ((0.0 - (G[1][q] - G[0][q])) - (G[3][q] - G[2][q])) / area;
    }
}

double CellFluxEngine::residual(const CellMesh& m, const Field& U, Field& R) {
    const int N = m.num_leaves(), nf = static_cast<int>(m.faces.size());
    if (N != nleaf_ || static_cast<int>(stencil_.size()) != nf)
        throw std::logic_error("CellFluxEngine::residual: build() was not called for this mesh");
    if (muscl_) refresh_all(U, N);
    double u_target[4];
    mass_flow_targets(m, U, u_target);
    const double s = fluxes(m, U, 0, nf, u_target);
    gather(m, 0, N, false, R);
    return s;
}

double CellFluxEngine::compute_level_fluxes(const CellMesh& m, const Field& U, int l) {
    if (m.num_leaves() != nleaf_ || static_cast<int>(stencil_.size()) != static_cast<int>(m.faces.size()))
        throw std::logic_error("CellFluxEngine: build() was not called for this mesh");
    if (muscl_) refresh_some(U, nleaf_, need_[static_cast<size_t>(l)]);
    double u_target[4];
    mass_flow_targets(m, U, u_target);
    return fluxes(m, U, m.face_level_begin[l], m.face_level_begin[l + 1], u_target);
}

void CellFluxEngine::gather_level(const CellMesh& m, int l, bool replace, Field& Rl) {
    gather(m, m.level_begin[l], m.level_begin[l + 1], replace, Rl);
}

void CellFluxEngine::clear_registers(const CellMesh& m, int l) {
    if (reg_.size() != m.faces.size()) reg_.assign(m.faces.size(), State{});
    for (int f = m.face_level_begin[l]; f < m.face_level_begin[l + 1]; ++f) reg_[f] = State{};
}

void CellFluxEngine::accumulate_registers(const CellMesh& m, int l, double w) {
    const int fb = m.face_level_begin[l], fe = m.face_level_begin[l + 1];
#pragma omp parallel for schedule(static) if (fe - fb >= kOmpMinWork)
    for (int f = fb; f < fe; ++f)
        if (m.faces[f].kind == CellFaceKind::CoarseFine)
            for (int q = 0; q < kNVar; ++q) reg_[f][q] += w * F_[f][q];
}

}   // namespace fv
