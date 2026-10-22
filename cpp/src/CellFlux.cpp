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

    // Index into W for the cell across `side` of leaf c (the leaf itself at a physical edge: replication)
    int beyond(int c, Face side) {
        const CellNeighbour nb = m.neighbour(c, side);
        switch (nb.kind) {
        case CellNeighbour::Kind::Boundary: return c;   // edge replication
        case CellNeighbour::Kind::Same:
        case CellNeighbour::Kind::Coarser: return nb.a;   // the cell, or injection of the coarse one
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

constexpr double kMuDiag = 4.0 / 3.0;
constexpr double kMuOff = 2.0 / 3.0;

}   // namespace

State CellFluxEngine::eval(const Field& U, int code) const {
    if (code >= 0) return U[code];
    const std::array<int, 4>& ch = vnode_[static_cast<size_t>(-1 - code)];
    const State a = eval(U, ch[0]), b = eval(U, ch[1]), c = eval(U, ch[2]), d = eval(U, ch[3]);
    State r;
    for (int k = 0; k < kNVar; ++k) r[k] = ((a[k] + b[k]) + (c[k] + d[k])) * 0.25;
    return r;
}

State CellFluxEngine::cell_state(int code) const { return code < nleaf_ ? (*us_)[code] : Vc_[code - nleaf_]; }

void CellFluxEngine::build(const CellMesh& m) {
    const auto per = [&](Face lo, Face hi) {
        const bool a = bc_.type(lo) == BCType::Periodic, b = bc_.type(hi) == BCType::Periodic;
        if (a != b) throw std::invalid_argument("CellFluxEngine: both sides of a periodic axis must be periodic");
        return a;
    };
    if (per(Face::Left, Face::Right) != m.periodic_x || per(Face::Bottom, Face::Top) != m.periodic_y)
        throw std::invalid_argument("CellFluxEngine: the periodic axes of the mesh and the boundary types differ");

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
    // neighbour codes of every leaf (gradients, non-reflecting tangential differences)
    nbr_.assign(static_cast<size_t>(4) * nleaf_, -1);
    for (int c = 0; c < nleaf_; ++c)
        for (int s = 0; s < 4; ++s)
            if (m.neighbour(c, static_cast<Face>(s)).kind != CellNeighbour::Kind::Boundary)
                nbr_[static_cast<size_t>(4) * c + s] = b.beyond(c, static_cast<Face>(s));

    need_.assign(static_cast<size_t>(m.max_level) + 1, {});
    if (muscl_)
        for (int l = 0; l <= m.max_level; ++l) {
            std::vector<int>& v = need_[static_cast<size_t>(l)];
            for (int f = m.face_level_begin[l]; f < m.face_level_begin[l + 1]; ++f)
                if (m.faces[f].kind != CellFaceKind::Boundary) v.insert(v.end(), stencil_[f].begin(), stencil_[f].end());
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        }
    vcells_.assign(static_cast<size_t>(m.max_level) + 1, {});
    for (int l = 0; l <= m.max_level; ++l) {
        std::vector<int>& v = vcells_[static_cast<size_t>(l)];
        for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c) v.push_back(c);
        for (int f = m.face_level_begin[l]; f < m.face_level_begin[l + 1]; ++f)
            if (m.faces[f].kind == CellFaceKind::CoarseFine)
                v.push_back(m.level[m.faces[f].left] < l ? m.faces[f].left : m.faces[f].right);
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }
    reg_.assign(m.faces.size(), State{});
    regv_.assign(m.faces.size(), State{});
    Fv_.assign(m.faces.size(), State{});
    su_.assign(static_cast<size_t>(nleaf_), 0.0);
    sv_ = sT_ = dudx_ = dudy_ = dvdx_ = dvdy_ = su_;

    // immersed body: mask of every leaf from the uniform mask of its level
    solid_.clear();
    if (has_body()) {
        level_mask_.resize(static_cast<size_t>(m.max_level) + 1);
        solid_.assign(static_cast<size_t>(nleaf_), 0);
        for (int l = 0; l <= m.max_level; ++l) {
            if (m.level_begin[l + 1] == m.level_begin[l]) continue;
            if (level_mask_[static_cast<size_t>(l)].empty()) {
                const ImmersedBoundary ib =
                    phys_.ibm_factory(generate_cartesian_mesh(m.Lx, m.Ly, m.nx_at(l), m.ny_at(l), m.x0, m.y0));
                level_mask_[static_cast<size_t>(l)] = ib.mask();
                ibm_rho_ref_ = ib.rho_ref();
                ibm_p_ref_ = ib.p_ref();
            }
            const std::vector<char>& mk = level_mask_[static_cast<size_t>(l)];
            for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c)
                solid_[c] = mk[static_cast<size_t>(m.cj[c]) * m.nx_at(l) + m.ci[c]];
        }
        forced_.resize(static_cast<size_t>(nleaf_));
    }
    // steady sources at the leaf centres
    src_.assign(static_cast<size_t>(nleaf_), State{});
    if (phys_.source && !phys_.source_unsteady) update_source(m, 0, nleaf_);

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

void CellFluxEngine::update_source(const CellMesh& m, int cb, int ce) {
    if (!phys_.source) return;
    std::vector<double> xs(static_cast<size_t>(ce - cb)), ys(xs.size());
    for (int c = cb; c < ce; ++c) {
        xs[static_cast<size_t>(c - cb)] = m.xc(c);
        ys[static_cast<size_t>(c - cb)] = m.yc(c);
    }
    evaluate_source_points(*phys_.source_mesh, *phys_.source, phys_.source_unsteady ? stage_time : 0.0, xs.data(),
                           ys.data(), ce - cb, &src_[static_cast<size_t>(cb)]);
}

void CellFluxEngine::force_state(const CellMesh& m, const Field& U) {
    if (!has_body()) return;
    forced_.resize(static_cast<size_t>(m.num_leaves()));
    for (int c = 0; c < m.num_leaves(); ++c) {
        State W = eos_.cons_to_prim(U[c]);
        if (solid_[c]) {
            W[1] = 0.0;
            W[2] = 0.0;
            W[4] = 0.0;
        }
        forced_[c] = eos_.prim_to_cons(W);
    }
}

void CellFluxEngine::force_level(const CellMesh& m, const Field& U, int l) {
    if (!has_body()) return;
    const int b = m.level_begin[l], e = m.level_begin[l + 1];
#pragma omp parallel for schedule(static) if (e - b >= kOmpMinWork)
    for (int c = b; c < e; ++c) {
        State W = eos_.cons_to_prim(U[c]);
        if (solid_[c]) {
            W[1] = 0.0;
            W[2] = 0.0;
            W[4] = 0.0;
        }
        forced_[c] = eos_.prim_to_cons(W);
    }
}

void CellFluxEngine::prepare(const CellMesh& m, const Field& U, int level) {
    if (has_body()) {
        if (level < 0)
            force_state(m, U);
        else
            force_level(m, U, level);
        us_ = &forced_;
    } else {
        us_ = &U;
    }
    Vc_.resize(vroot_.size());
    for (size_t v = 0; v < vroot_.size(); ++v) Vc_[v] = eval(*us_, vroot_[v]);
    if (muscl_) {
        if (level < 0)
            refresh_all(nleaf_);
        else
            refresh_some(nleaf_, need_[static_cast<size_t>(level)]);
    }
}

void CellFluxEngine::refresh_all(int N) {
    W_.resize(static_cast<size_t>(N) + vroot_.size());
    const EOS& eos = eos_;
    const Field& U = *us_;
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
    for (int c = 0; c < N; ++c) W_[c] = eos.cons_to_prim(U[c]);
    for (size_t v = 0; v < vroot_.size(); ++v) W_[N + v] = eos.cons_to_prim(Vc_[v]);
}

void CellFluxEngine::refresh_some(int N, const std::vector<int>& idx) {
    W_.resize(static_cast<size_t>(N) + vroot_.size());
    const EOS& eos = eos_;
    const Field& U = *us_;
    const int n = static_cast<int>(idx.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int k = 0; k < n; ++k) {
        const int i = idx[k];
        W_[i] = i < N ? eos.cons_to_prim(U[i]) : eos.cons_to_prim(Vc_[static_cast<size_t>(i - N)]);
    }
}

void CellFluxEngine::mass_flow_targets(const CellMesh& m, double* u_target) const {
    const BCParams& bp = bc_.params();
    const Field& U = *us_;
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

double CellFluxEngine::fluxes(const CellMesh& m, int fb, int fe, const double* u_target) {
    const EOS& eos = eos_;
    const Field& U = *us_;
    const Limiter lim = muscl_ ? muscl_->limiter() : Limiter::Minmod;
    if (F_.size() != m.faces.size()) F_.resize(m.faces.size());
    double smax_all = 0.0;
    auto roundtrip = [&](State& u) { u = eos.prim_to_cons(eos.cons_to_prim(u)); };
    // non-reflecting ghost of leaf c across `side`: central differences along the other axis from the neighbour codes
    auto nrbc = [&](int c, Face side, const State& Ui) {
        BoundaryManager::NrbcTangent tg;
        const int axis = face_axis(side), l = m.level[c];
        for (int t = 0; t < 2; ++t) {
            if (t == axis) continue;
            const int cl = nbr_[static_cast<size_t>(4) * c + 2 * t], ch = nbr_[static_cast<size_t>(4) * c + 2 * t + 1];
            double span = 2.0;
            if (cl < 0) span -= 1.0;
            if (ch < 0) span -= 1.0;
            if (span <= 0.0) continue;
            tg.use[t] = true;
            tg.Wlo[t] = cl < 0 ? eos.cons_to_prim(Ui) : eos.cons_to_prim(cell_state(cl));
            tg.Whi[t] = ch < 0 ? eos.cons_to_prim(Ui) : eos.cons_to_prim(cell_state(ch));
            tg.span[t] = span;
        }
        const double hs[3] = {m.dx(l), m.dy(l), 0.0};
        return bc_.nonreflecting_ghost(side, Ui, tg, hs);
    };
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
                const State Ug = bc_.type(face.side) == BCType::NonReflecting
                                     ? nrbc(c, face.side, Ui)
                                     : bc_.ghost_state(face.side, Ui, u_target[static_cast<int>(face.side)]);
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

void CellFluxEngine::cell_quantities(const CellMesh& m, const std::vector<int>& cells) {
    const EOS& eos = eos_;
    const Field& U = *us_;
    const int n = static_cast<int>(cells.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int k = 0; k < n; ++k) {
        const int c = cells[k];
        const int l = m.level[c];
        const State& Ui = U[c];
        const State W = eos.cons_to_prim(Ui);
        State nb[4];   // left, right, bottom, top values (the diffusive ghost at a physical edge)
        for (int s = 0; s < 4; ++s) {
            const int code = nbr_[static_cast<size_t>(4) * c + s];
            nb[s] = code >= 0 ? cell_state(code) : bc_.viscous_ghost(static_cast<Face>(s), Ui);
        }
        const double dx = m.dx(l), dy = m.dy(l);
        double gx[3], gy[3];
        for (int q = 0; q < 3; ++q) {
            gx[q] = (nb[1][q] - nb[0][q]) / (2.0 * dx);
            gy[q] = (nb[3][q] - nb[2][q]) / (2.0 * dy);
        }
        const double rho = W[0], u = W[1], v = W[2];
        su_[c] = u;
        sv_[c] = v;
        sT_[c] = W[3] / rho;
        dudx_[c] = (gx[1] - u * gx[0]) / rho;
        dudy_[c] = (gy[1] - u * gy[0]) / rho;
        dvdx_[c] = (gx[2] - v * gx[0]) / rho;
        dvdy_[c] = (gy[2] - v * gy[0]) / rho;
    }
}

void CellFluxEngine::viscous_fluxes(const CellMesh& m, int fb, int fe) {
    const NavierStokes& ns = *phys_.ns;
    const double mu = ns.mu(), gamma = eos_.gamma;
    const double k_cond = mu * gamma / (ns.prandtl() * (gamma - 1.0));
    const Field& U = *us_;
    if (Fv_.size() != m.faces.size()) Fv_.resize(m.faces.size());
#pragma omp parallel for schedule(static) if (fe - fb >= kOmpMinWork)
    for (int f = fb; f < fe; ++f) {
        const CellFace& face = m.faces[f];
        const int ax = face.axis, lv = face.level;
        double uL, vL, TL, uR, vR, TR, h;
        double ca, cb;   // cross derivatives of u and v along the other axis
        if (face.kind == CellFaceKind::Boundary) {
            const bool ghost_left = face.left < 0;
            const int c = ghost_left ? face.right : face.left;
            const State g = eos_.cons_to_prim(bc_.viscous_ghost(face.side, U[c]));
            const double gu = g[1], gv = g[2], gT = g[3] / g[0];
            if (ghost_left) {
                uL = gu;
                vL = gv;
                TL = gT;
                uR = su_[c];
                vR = sv_[c];
                TR = sT_[c];
            } else {
                uL = su_[c];
                vL = sv_[c];
                TL = sT_[c];
                uR = gu;
                vR = gv;
                TR = gT;
            }
            h = ax == 0 ? m.dx(lv) : m.dy(lv);
            const double a = ax == 0 ? dudy_[c] : dudx_[c], b = ax == 0 ? dvdy_[c] : dvdx_[c];
            ca = 0.5 * (a + a);
            cb = 0.5 * (b + b);
        } else {
            const int L = face.left, R = face.right;
            uL = su_[L];
            vL = sv_[L];
            TL = sT_[L];
            uR = su_[R];
            vR = sv_[R];
            TR = sT_[R];
            h = ax == 0 ? 0.5 * (m.dx(m.level[L]) + m.dx(m.level[R])) : 0.5 * (m.dy(m.level[L]) + m.dy(m.level[R]));
            ca = ax == 0 ? 0.5 * (dudy_[L] + dudy_[R]) : 0.5 * (dudx_[L] + dudx_[R]);
            cb = ax == 0 ? 0.5 * (dvdy_[L] + dvdy_[R]) : 0.5 * (dvdx_[L] + dvdx_[R]);
        }
        const double u_f = 0.5 * (uL + uR), v_f = 0.5 * (vL + vR);
        const double du_n = (uR - uL) / h, dv_n = (vR - vL) / h, dT_n = (TR - TL) / h;
        const double len = face.length;
        if (ax == 0) {
            const double tau_xx = mu * (kMuDiag * du_n - kMuOff * cb);
            const double tau_xy = mu * (ca + dv_n);
            const double q_x = -k_cond * dT_n;
            Fv_[f] = {0.0, tau_xx * len, tau_xy * len, (u_f * tau_xx + v_f * tau_xy - q_x) * len};
        } else {
            const double tau_yy = mu * (kMuDiag * dv_n - kMuOff * ca);
            const double tau_yx = mu * (du_n + cb);
            const double q_y = -k_cond * dT_n;
            Fv_[f] = {0.0, tau_yx * len, tau_yy * len, (u_f * tau_yx + v_f * tau_yy - q_y) * len};
        }
    }
}

double CellFluxEngine::viscous_speed(const CellMesh& m, int cb, int ce) const {
    const NavierStokes& ns = *phys_.ns;
    const double mu = ns.mu(), gamma = eos_.gamma;
    const double k_cond = mu * gamma / (ns.prandtl() * (gamma - 1.0));
    const Field& U = *us_;
    double s_max = 0.0;
#pragma omp parallel if (ce - cb >= kOmpMinWork)
    {
        double local = 0.0;
#pragma omp for schedule(static) nowait
        for (int c = cb; c < ce; ++c) {
            const int l = m.level[c];
            const double dx = m.dx(l), dy = m.dy(l);
            const double L = std::min(dx, dy), inv_h2 = 1.0 / (dx * dx) + 1.0 / (dy * dy);
            const State W = eos_.cons_to_prim(U[c]);
            const double rho = W[0];
            const double nu_eff = std::max(mu / rho, k_cond / (rho * (gamma / (gamma - 1.0)))) + 1e-12;
            const double s =
                std::max(std::abs(W[1]), std::abs(W[2])) + eos_.sound_speed(W[3], rho) + 2.0 * L * inv_h2 * nu_eff;
            local = std::max(local, s);
        }
#pragma omp critical(fv_cellvisc_speed)
        s_max = std::max(s_max, local);
    }
    return s_max;
}

void CellFluxEngine::gather(const CellMesh& m, int cb, int ce, bool replace, Field& R) {
    R.resize(static_cast<size_t>(ce - cb));
    const bool visc = phys_.ns != nullptr;
    const Field& U = *us_;
    if (phys_.source && phys_.source_unsteady) update_source(m, cb, ce);
#pragma omp parallel for schedule(static) if (ce - cb >= kOmpMinWork)
    for (int c = cb; c < ce; ++c) {
        // side integrals: sum of (face flux * face length), one or two faces per side
        State G[4], H[4];
        for (int s = 0; s < 4; ++s) {
            const int b = m.side_ptr[4 * c + s], e = m.side_ptr[4 * c + s + 1];
            for (int k = b; k < e; ++k) {
                const int fi = m.side_face[k];
                const CellFace& f = m.faces[fi];
                // a finer face on the side of this leaf: its coarse side uses the registered time average
                const bool reg = replace && f.level > m.level[c];
                const State& Fk = reg ? reg_[fi] : F_[fi];
                if (k == b)
                    for (int q = 0; q < kNVar; ++q) G[s][q] = Fk[q] * f.length;
                else
                    for (int q = 0; q < kNVar; ++q) G[s][q] += Fk[q] * f.length;
                if (visc) {
                    const State& Vk = reg ? regv_[fi] : Fv_[fi];
                    if (k == b)
                        H[s] = Vk;
                    else
                        for (int q = 0; q < kNVar; ++q) H[s][q] += Vk[q];
                }
            }
        }
        const double area = m.area(c);
        State& r = R[c - cb];
        for (int q = 0; q < kNVar; ++q) r[q] = ((0.0 - (G[1][q] - G[0][q])) - (G[3][q] - G[2][q])) / area;
        if (visc)
            for (int q = 0; q < kNVar; ++q) r[q] += ((0.0 + (H[1][q] - H[0][q])) + (H[3][q] - H[2][q])) / area;
        if (phys_.body) {
            const ConstantBodyForce& bf = *phys_.body;
            r[1] += U[c][0] * bf.gx;
            r[2] += U[c][0] * bf.gy;
            r[3] += U[c][1] * bf.gx + U[c][2] * bf.gy;
            if (bf.gz != 0.0) {
                r[3] += U[c][4] * bf.gz;
                r[4] += U[c][0] * bf.gz;
            }
        }
        if (phys_.source)
            for (int q = 0; q < kNVar; ++q) r[q] += src_[c][q];
    }
}

double CellFluxEngine::residual(const CellMesh& m, const Field& U, Field& R) {
    const int N = m.num_leaves(), nf = static_cast<int>(m.faces.size());
    if (N != nleaf_ || static_cast<int>(stencil_.size()) != nf)
        throw std::logic_error("CellFluxEngine::residual: build() was not called for this mesh");
    prepare(m, U, -1);
    double u_target[4];
    mass_flow_targets(m, u_target);
    double s = fluxes(m, 0, nf, u_target);
    if (phys_.ns) {
        std::vector<int> all(static_cast<size_t>(N));
        for (int c = 0; c < N; ++c) all[c] = c;
        cell_quantities(m, all);
        viscous_fluxes(m, 0, nf);
        s = std::max(s, viscous_speed(m, 0, N));
    }
    gather(m, 0, N, false, R);
    return s;
}

double CellFluxEngine::compute_level_fluxes(const CellMesh& m, const Field& U, int l) {
    if (m.num_leaves() != nleaf_ || static_cast<int>(stencil_.size()) != static_cast<int>(m.faces.size()))
        throw std::logic_error("CellFluxEngine: build() was not called for this mesh");
    prepare(m, U, l);
    double u_target[4];
    mass_flow_targets(m, u_target);
    double s = fluxes(m, m.face_level_begin[l], m.face_level_begin[l + 1], u_target);
    if (phys_.ns) {
        cell_quantities(m, vcells_[static_cast<size_t>(l)]);
        viscous_fluxes(m, m.face_level_begin[l], m.face_level_begin[l + 1]);
        s = std::max(s, viscous_speed(m, m.level_begin[l], m.level_begin[l + 1]));
    }
    return s;
}

void CellFluxEngine::gather_level(const CellMesh& m, int l, bool replace, Field& Rl) {
    gather(m, m.level_begin[l], m.level_begin[l + 1], replace, Rl);
}

void CellFluxEngine::clear_registers(const CellMesh& m, int l) {
    for (int f = m.face_level_begin[l]; f < m.face_level_begin[l + 1]; ++f) {
        reg_[f] = State{};
        regv_[f] = State{};
    }
}

void CellFluxEngine::accumulate_registers(const CellMesh& m, int l, double w) {
    const int fb = m.face_level_begin[l], fe = m.face_level_begin[l + 1];
    const bool visc = phys_.ns != nullptr;
#pragma omp parallel for schedule(static) if (fe - fb >= kOmpMinWork)
    for (int f = fb; f < fe; ++f)
        if (m.faces[f].kind == CellFaceKind::CoarseFine)
            for (int q = 0; q < kNVar; ++q) {
                reg_[f][q] += w * F_[f][q];
                if (visc) regv_[f][q] += w * Fv_[f][q];
            }
}

void CellFluxEngine::solid_reset(const CellMesh& m, Field& U) const {
    if (!has_body()) return;
    const int N = m.num_leaves();
    auto pressure = [&](const State& c) {
        return (eos_.gamma - 1.0) * (c[3] - 0.5 * (c[1] * c[1] + c[2] * c[2] + c[4] * c[4]) / c[0]);
    };
    std::vector<double> rho(static_cast<size_t>(N)), p(static_cast<size_t>(N));
    for (int c = 0; c < N; ++c) {
        rho[c] = U[c][0];
        p[c] = pressure(U[c]);
    }
    std::vector<double> new_rho = rho, new_p = p;
    std::vector<char> fringe(static_cast<size_t>(N), 0);
    // Neumann condition: a solid leaf next to fluid copies rho / p from it; passes right, left, top, bottom (later
    // passes win), each reading the original rho / p; no pull across a domain edge, periodic or not. A finer
    // neighbour contributes the mean of its fluid leaves.
    for (const Face side : {Face::Right, Face::Left, Face::Top, Face::Bottom}) {
        for (int c = 0; c < N; ++c) {
            if (!solid_[c]) continue;
            const int l = m.level[c];
            const bool edge = (side == Face::Right && m.ci[c] + 1 == m.nx_at(l)) ||
                              (side == Face::Left && m.ci[c] == 0) ||
                              (side == Face::Top && m.cj[c] + 1 == m.ny_at(l)) || (side == Face::Bottom && m.cj[c] == 0);
            if (edge) continue;
            const CellNeighbour nb = m.neighbour(c, side);
            double sr = 0.0, sp = 0.0;
            int cnt = 0;
            for (const int n : {nb.a, nb.b}) {
                if (n < 0 || solid_[n]) continue;
                sr += rho[n];
                sp += p[n];
                ++cnt;
            }
            if (cnt == 0) continue;
            new_rho[c] = cnt == 1 ? sr : sr / cnt;
            new_p[c] = cnt == 1 ? sp : sp / cnt;
            fringe[c] = 1;
        }
    }
    for (int c = 0; c < N; ++c) {
        if (!solid_[c]) continue;
        if (!fringe[c]) {
            new_rho[c] = ibm_rho_ref_;
            new_p[c] = ibm_p_ref_;
        }   // deep solid
        U[c] = {new_rho[c], 0.0, 0.0, new_p[c] / (eos_.gamma - 1.0), 0.0};
    }
}

}   // namespace fv
