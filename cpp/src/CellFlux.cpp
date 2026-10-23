#include "fv/CellFlux.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>

#include "SchemesSimd.hpp"

namespace fv {

namespace {

// Code of the block (l, i, j): a leaf, or a node whose children are the four sub-blocks (appended to vnode)
int block_code(const CellMesh& m, std::vector<std::array<int, 4>>& vnode, int l, int i, int j) {
    const int leaf = m.find(l, i, j);
    if (leaf >= 0) return leaf;
    std::array<int, 4> ch{};
    for (int oj = 0; oj < 2; ++oj)
        for (int oi = 0; oi < 2; ++oi) ch[2 * oj + oi] = block_code(m, vnode, l + 1, 2 * i + oi, 2 * j + oj);
    vnode.push_back(ch);
    return -static_cast<int>(vnode.size());   // node k <-> code -1 - k
}

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

    const bool ip = muscl_ && interpolate_jumps;
    // incremental: right after an adapt of the mesh of the last build (CellMesh::prev_leaf), same interpolation setting
    const bool inc = !m.prev_leaf.empty() && built_mesh_ == &m && built_version_ == m.version - 1 && built_ip_ == ip &&
                     static_cast<int>(m.prev_leaf.size()) == m.num_leaves();
    const int old_n = nleaf_;
    std::vector<int> old_raw, old_nbr;
    std::vector<std::array<int, 4>> old_st, old_ikey;
    std::vector<std::array<int, 3>> old_vkey;
    std::vector<char> old_special;
    if (inc) {
        old_raw.swap(nbr_raw_);
        old_nbr.swap(nbr_);
        old_st.swap(stencil_);
        old_vkey.swap(vkey_);
        old_ikey.swap(ikey_);
        old_special.swap(special_);
    }
    nleaf_ = m.num_leaves();
    const auto clean = [&](int c) { return inc && !m.topo_dirty[static_cast<size_t>(c)]; };
    // 1) Virtual cells. Only a leaf with a finer side (the mean of the block across it) or, with interpolated jumps, a coarser
    // side (the interpolated cell next to it) needs one; they are numbered by key (block, resp. fine leaf and side), so the
    // numbering depends on the mesh only, and every table below is filled in parallel by lookups (or copied).
    special_.assign(static_cast<size_t>(nleaf_), 0);
#pragma omp parallel for schedule(static) if (nleaf_ >= kOmpMinWork)
    for (int c = 0; c < nleaf_; ++c) {
        if (clean(c)) {
            special_[c] = old_special[static_cast<size_t>(m.prev_leaf[c])];
            continue;
        }
        for (int s = 0; s < 4; ++s) {
            const CellNeighbour::Kind k = m.neighbour(c, static_cast<Face>(s)).kind;
            if (k == CellNeighbour::Kind::Finer || k == CellNeighbour::Kind::Coarser) special_[c] = 1;
        }
    }
    vkey_.clear();
    ikey_.clear();
    for (int c = 0; c < nleaf_; ++c) {
        if (!special_[c]) continue;
        for (int s = 0; s < 4; ++s) {
            const CellNeighbour nb = m.neighbour(c, static_cast<Face>(s));
            if (nb.kind == CellNeighbour::Kind::Finer)
                vkey_.push_back({m.level[nb.a] - 1, m.ci[nb.a] >> 1, m.cj[nb.a] >> 1});
            else if (nb.kind == CellNeighbour::Kind::Coarser && ip)
                ikey_.push_back({m.level[c], m.ci[c], m.cj[c], s});
        }
    }
    std::sort(vkey_.begin(), vkey_.end());
    vkey_.erase(std::unique(vkey_.begin(), vkey_.end()), vkey_.end());
    std::sort(ikey_.begin(), ikey_.end());
    make_virtual_cells(m);
    // 2) injecting neighbour codes of every leaf (slopes of the coarse leaves), stencils, neighbour codes with the interpolated
    // virtual cells where enabled (gradients, non-reflecting tangential differences). A clean leaf, and a face between two clean
    // leaves made by one of them, keep their old codes, renumbered: leaves by the adapt, virtual cells by key.
    nbr_raw_.assign(static_cast<size_t>(4) * nleaf_, -1);
    nbr_.assign(static_cast<size_t>(4) * nleaf_, -1);
    stencil_.assign(m.faces.size(), {-1, -1, -1, -1});
    if (inc) {
        std::vector<int> leaf_map(static_cast<size_t>(old_n), -1);
        for (int c = 0; c < nleaf_; ++c)
            if (m.prev_leaf[c] >= 0) leaf_map[static_cast<size_t>(m.prev_leaf[c])] = c;
        const int nmean = static_cast<int>(vkey_.size());
        std::vector<int> virt_map(old_vkey.size() + old_ikey.size(), -1);   // old virtual cell -> new code (-1: gone)
        for (size_t v = 0; v < old_vkey.size(); ++v) {
            const auto it = std::lower_bound(vkey_.begin(), vkey_.end(), old_vkey[v]);
            if (it != vkey_.end() && *it == old_vkey[v]) virt_map[v] = nleaf_ + static_cast<int>(it - vkey_.begin());
        }
        for (size_t v = 0; v < old_ikey.size(); ++v) {
            const auto it = std::lower_bound(ikey_.begin(), ikey_.end(), old_ikey[v]);
            if (it != ikey_.end() && *it == old_ikey[v])
                virt_map[old_vkey.size() + v] = nleaf_ + nmean + static_cast<int>(it - ikey_.begin());
        }
        bool lost = false;
        const auto remap = [&](int code) {
            if (code < 0) return code;
            const int r = code < old_n ? leaf_map[static_cast<size_t>(code)] : virt_map[static_cast<size_t>(code - old_n)];
            if (r < 0) lost = true;
            return r;
        };
#pragma omp parallel for schedule(static) if (nleaf_ >= kOmpMinWork)
        for (int c = 0; c < nleaf_; ++c) {
            if (!clean(c)) {
                leaf_codes(m, c);
                continue;
            }
            const size_t o = static_cast<size_t>(4) * m.prev_leaf[c], n = static_cast<size_t>(4) * c;
            for (int s = 0; s < 4; ++s) {
                nbr_raw_[n + s] = remap(old_raw[o + s]);
                nbr_[n + s] = remap(old_nbr[o + s]);
            }
        }
#pragma omp parallel for schedule(static) if (m.faces.size() >= static_cast<size_t>(kOmpMinWork))
        for (int f = 0; f < static_cast<int>(m.faces.size()); ++f) {
            const CellFace& face = m.faces[static_cast<size_t>(f)];
            if (face.kind == CellFaceKind::Boundary) continue;
            const int pf = m.prev_face[static_cast<size_t>(f)];
            if (pf < 0 || !clean(face.left) || !clean(face.right)) {
                face_stencil(m, f);
                continue;
            }
            for (int k = 0; k < 4; ++k) stencil_[static_cast<size_t>(f)][k] = remap(old_st[static_cast<size_t>(pf)][k]);
        }
        if (lost) throw std::logic_error("CellFluxEngine: a clean leaf refers to a cell that is gone (incremental build)");
    } else {
#pragma omp parallel for schedule(static) if (nleaf_ >= kOmpMinWork)
        for (int c = 0; c < nleaf_; ++c) leaf_codes(m, c);
#pragma omp parallel for schedule(static) if (m.faces.size() >= static_cast<size_t>(kOmpMinWork))
        for (int f = 0; f < static_cast<int>(m.faces.size()); ++f) face_stencil(m, f);
    }

    reg_.assign(m.faces.size(), State{});
    regv_.assign(m.faces.size(), State{});
    Fv_.assign(m.faces.size(), State{});
    su_.assign(static_cast<size_t>(nleaf_), 0.0);
    sv_ = sT_ = dudx_ = dudy_ = dvdx_ = dvdy_ = su_;

    // immersed body: mask of every leaf from the uniform mask of its level (a leaf kept by the adapt keeps its value)
    std::vector<char> old_solid;
    old_solid.swap(solid_);
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
            const int cb = m.level_begin[l], ce = m.level_begin[l + 1];
#pragma omp parallel for schedule(static) if (ce - cb >= kOmpMinWork)
            for (int c = cb; c < ce; ++c)
                solid_[c] = inc && m.prev_leaf[c] >= 0 ? old_solid[static_cast<size_t>(m.prev_leaf[c])]
                                                       : mk[static_cast<size_t>(m.cj[c]) * m.nx_at(l) + m.ci[c]];
        }
        forced_.resize(static_cast<size_t>(nleaf_));
    }
    rebuild_lists(m, nullptr);

    // steady sources at the leaf centres (a leaf kept by the adapt keeps its value)
    if (inc && phys_.source && !phys_.source_unsteady && src_.size() == static_cast<size_t>(old_n)) {
        Field old_src;
        old_src.swap(src_);
        src_.assign(static_cast<size_t>(nleaf_), State{});
        std::vector<int> fresh;
        for (int c = 0; c < nleaf_; ++c) {
            if (m.prev_leaf[c] >= 0)
                src_[c] = old_src[static_cast<size_t>(m.prev_leaf[c])];
            else
                fresh.push_back(c);
        }
        std::vector<double> xs(fresh.size()), ys(fresh.size());
        for (size_t k = 0; k < fresh.size(); ++k) {
            xs[k] = m.xc(fresh[k]);
            ys[k] = m.yc(fresh[k]);
        }
        Field sv(fresh.size());
        evaluate_source_points(*phys_.source_mesh, *phys_.source, 0.0, xs.data(), ys.data(), static_cast<int>(fresh.size()),
                               sv.data());
        for (size_t k = 0; k < fresh.size(); ++k) src_[static_cast<size_t>(fresh[k])] = sv[k];
    } else {
        src_.assign(static_cast<size_t>(nleaf_), State{});
        if (phys_.source && !phys_.source_unsteady) update_source(m, 0, nleaf_);
    }

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
    built_mesh_ = &m;
    built_version_ = m.version;
    built_ip_ = ip;
}

bool CellFluxEngine::same_tables(const CellFluxEngine& o) const {
    const auto same_field = [](const Field& a, const Field& b) {
        return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(State)) == 0);
    };
    bool ok = nleaf_ == o.nleaf_ && vkey_ == o.vkey_ && ikey_ == o.ikey_ && vroot_ == o.vroot_ && vnode_ == o.vnode_ &&
              nbr_raw_ == o.nbr_raw_ && nbr_ == o.nbr_ && stencil_ == o.stencil_ && special_ == o.special_ && lface_ == o.lface_ &&
              lheld_ == o.lheld_ && need_ == o.need_ && vcells_ == o.vcells_ && lown_b_ == o.lown_b_ && lown_e_ == o.lown_e_ &&
              solid_ == o.solid_ && same_field(src_, o.src_) && interp_.size() == o.interp_.size();
    for (size_t k = 0; ok && k < interp_.size(); ++k) {
        const InterpVirtual &a = interp_[k], &b = o.interp_[k];
        ok = a.fine == b.fine && a.coarse == b.coarse && a.side == b.side && a.axis == b.axis &&
             std::memcmp(&a.dn, &b.dn, sizeof(double)) == 0 && std::memcmp(&a.dt, &b.dt, sizeof(double)) == 0;
    }
    for (int s = 0; ok && s < 4; ++s) ok = mf_face_[s] == o.mf_face_[s] && mf_weight_[s] == o.mf_weight_[s];
    return ok;
}

void CellFluxEngine::make_virtual_cells(const CellMesh& m) {
    vroot_.clear();
    vnode_.clear();
    for (const std::array<int, 3>& k : vkey_) vroot_.push_back(block_code(m, vnode_, k[0], k[1], k[2]));
    interp_.clear();
    for (const std::array<int, 4>& k : ikey_) {
        InterpVirtual iv;
        iv.fine = m.find(k[0], k[1], k[2]);
        iv.side = static_cast<Face>(k[3]);
        iv.coarse = m.neighbour(iv.fine, iv.side).a;
        iv.axis = face_axis(iv.side);
        iv.dn = (k[3] & 1) ? -0.25 : 0.25;   // the virtual centre lies nearer to the fine leaf
        const int idx_t = iv.axis == 0 ? k[2] : k[1];   // transverse half of the coarse leaf the fine leaf is in
        iv.dt = (idx_t & 1) ? 0.25 : -0.25;
        interp_.push_back(iv);
    }
}

int CellFluxEngine::beyond(const CellMesh& m, int c, Face side, bool interpolate) const {
    const CellNeighbour nb = m.neighbour(c, side);
    switch (nb.kind) {
    case CellNeighbour::Kind::Boundary: return c;   // edge replication
    case CellNeighbour::Kind::Same: return nb.a;
    case CellNeighbour::Kind::Coarser: {
        if (!interpolate) return nb.a;   // injection of the coarse value
        const std::array<int, 4> key = {m.level[c], m.ci[c], m.cj[c], static_cast<int>(side)};
        const auto it = std::lower_bound(ikey_.begin(), ikey_.end(), key);
        if (it == ikey_.end() || *it != key) throw std::logic_error("CellFluxEngine: missing interpolated virtual cell");
        return nleaf_ + static_cast<int>(vkey_.size()) + static_cast<int>(it - ikey_.begin());
    }
    default: {
        const std::array<int, 3> key = {m.level[nb.a] - 1, m.ci[nb.a] >> 1, m.cj[nb.a] >> 1};
        const auto it = std::lower_bound(vkey_.begin(), vkey_.end(), key);
        if (it == vkey_.end() || *it != key) throw std::logic_error("CellFluxEngine: missing virtual cell");
        return nleaf_ + static_cast<int>(it - vkey_.begin());
    }
    }
}

void CellFluxEngine::leaf_codes(const CellMesh& m, int c) {
    const bool ip = muscl_ && interpolate_jumps;
    for (int s = 0; s < 4; ++s) {
        if (m.neighbour(c, static_cast<Face>(s)).kind == CellNeighbour::Kind::Boundary) continue;
        nbr_raw_[static_cast<size_t>(4) * c + s] = beyond(m, c, static_cast<Face>(s), false);
        nbr_[static_cast<size_t>(4) * c + s] = beyond(m, c, static_cast<Face>(s), ip);
    }
}

void CellFluxEngine::face_stencil(const CellMesh& m, int f) {
    const CellFace& face = m.faces[static_cast<size_t>(f)];
    if (face.kind == CellFaceKind::Boundary) return;
    const bool ip = muscl_ && interpolate_jumps;
    const Face back = face.axis == 0 ? Face::Left : Face::Bottom, fwd = face.axis == 0 ? Face::Right : Face::Top;
    const int l = face.left, r = face.right;
    std::array<int, 4>& s = stencil_[static_cast<size_t>(f)];
    const int ll = m.level[l], lr = m.level[r];
    if (ll > lr) {   // fine leaf on the left: reconstructed against the virtual cell, the coarse state (s[3]) is not
        s[0] = beyond(m, l, back, ip);
        s[1] = l;
        s[2] = beyond(m, l, fwd, ip);
        s[3] = r;
    } else if (lr > ll) {
        s[0] = l;
        s[1] = beyond(m, r, back, ip);
        s[2] = r;
        s[3] = beyond(m, r, fwd, ip);
    } else {
        s[0] = beyond(m, l, back, ip);
        s[1] = l;
        s[2] = r;
        s[3] = beyond(m, r, fwd, ip);
    }
}

namespace {
// Sort v and drop the repeats. `seen` is a zeroed scratch array covering every value of v; it is zero again on return.
// A pass over the marks replaces a comparison sort, which dominated the rebuild after a remesh.
void sort_unique(std::vector<int>& v, std::vector<char>& seen) {
    if (v.size() < 64) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return;
    }
    int lo = v[0], hi = v[0];
    for (const int x : v) {
        seen[static_cast<size_t>(x)] = 1;
        lo = std::min(lo, x);
        hi = std::max(hi, x);
    }
    v.clear();
    for (int x = lo; x <= hi; ++x)
        if (seen[static_cast<size_t>(x)]) {
            v.push_back(x);
            seen[static_cast<size_t>(x)] = 0;
        }
}
}   // namespace

void CellFluxEngine::rebuild_lists(const CellMesh& m, const Partition* part) {
    std::vector<char> seen(static_cast<size_t>(nleaf_) + vroot_.size() + interp_.size() + 1, 0);
    const size_t L = static_cast<size_t>(m.max_level) + 1;
    lface_.assign(L, {});
    lheld_.assign(L, {});
    need_.assign(L, {});
    vcells_.assign(L, {});
    lown_b_.assign(L, 0);
    lown_e_.assign(L, 0);
    owned_mask_ = part ? part->leaf_owned : std::vector<char>();
    all_faces_.resize(m.faces.size());
    for (size_t f = 0; f < m.faces.size(); ++f) all_faces_[f] = static_cast<int>(f);
    for (int l = 0; l <= m.max_level; ++l) {
        const size_t sl = static_cast<size_t>(l);
        lown_b_[sl] = lown_e_[sl] = m.level_begin[l];
        bool first = true;
        for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c) {
            if (!part || part->leaf_owned[c]) {
                if (first) lown_b_[sl] = c;
                first = false;
                lown_e_[sl] = c + 1;
            }
            if (!part || part->leaf_held[c]) lheld_[sl].push_back(c);
        }
        for (int f = m.face_level_begin[l]; f < m.face_level_begin[l + 1]; ++f)
            if (!part || part->face_active[f]) lface_[sl].push_back(f);
        // W entries read by the stencils of the faces of this level
        if (muscl_) {
            std::vector<int>& v = need_[sl];
            for (const int f : lface_[sl]) {
                if (m.faces[f].kind == CellFaceKind::Boundary) continue;
                v.insert(v.end(), stencil_[f].begin(), stencil_[f].end());
                if (coarse_side_second_order && m.faces[f].kind == CellFaceKind::CoarseFine) {
                    const int C = m.level[m.faces[f].left] > m.level[m.faces[f].right] ? stencil_[f][3] : stencil_[f][0];
                    for (int s = 0; s < 4; ++s)
                        if (nbr_raw_[static_cast<size_t>(4) * C + s] >= 0) v.push_back(nbr_raw_[static_cast<size_t>(4) * C + s]);
                }
            }
            sort_unique(v, seen);
        }
        // cells whose viscous quantities the faces of this level read
        std::vector<int>& v = vcells_[sl];
        if (!part) {
            for (int c = m.level_begin[l]; c < m.level_begin[l + 1]; ++c) v.push_back(c);
            for (const int f : lface_[sl])
                if (m.faces[f].kind == CellFaceKind::CoarseFine)
                    v.push_back(m.level[m.faces[f].left] < l ? m.faces[f].left : m.faces[f].right);
        } else {
            for (const int f : lface_[sl]) {
                if (m.faces[f].left >= 0) v.push_back(m.faces[f].left);
                if (m.faces[f].right >= 0) v.push_back(m.faces[f].right);
            }
        }
        sort_unique(v, seen);
    }
}

void CellFluxEngine::set_partition(const CellMesh& m, const Partition& p) { rebuild_lists(m, &p); }

void CellFluxEngine::clear_partition(const CellMesh& m) { rebuild_lists(m, nullptr); }

void CellFluxEngine::mark_tree(int code, std::vector<char>& mask) const {
    if (code >= 0) {
        mask[static_cast<size_t>(code)] = 1;
        return;
    }
    for (const int ch : vnode_[static_cast<size_t>(-1 - code)]) mark_tree(ch, mask);
}

void CellFluxEngine::mark_code(int code, std::vector<char>& mask) const {
    if (code < 0) return;
    if (code < nleaf_) {
        mask[static_cast<size_t>(code)] = 1;
        return;
    }
    const size_t v = static_cast<size_t>(code - nleaf_);
    if (v < vroot_.size()) {
        mark_tree(vroot_[v], mask);
        return;
    }
    const InterpVirtual& iv = interp_[v - vroot_.size()];
    mask[static_cast<size_t>(iv.coarse)] = 1;
    for (int s = 0; s < 4; ++s) mark_code(nbr_raw_[static_cast<size_t>(4) * iv.coarse + s], mask);
}

void CellFluxEngine::face_dependencies(const CellMesh& m, int f, std::vector<char>& mask) const {
    const CellFace& face = m.faces[static_cast<size_t>(f)];
    const auto cell = [&](int c) {
        mask[static_cast<size_t>(c)] = 1;
        for (int s = 0; s < 4; ++s) mark_code(nbr_[static_cast<size_t>(4) * c + s], mask);   // gradients, tangential differences
    };
    if (face.kind == CellFaceKind::Boundary) {
        cell(face.left >= 0 ? face.left : face.right);
        return;
    }
    for (int k = 0; k < 4; ++k) mark_code(stencil_[static_cast<size_t>(f)][k], mask);
    cell(face.left);
    cell(face.right);
    if (face.kind == CellFaceKind::CoarseFine) {   // the coarse leaf's own neighbours (second-order coarse side)
        const int C = m.level[face.left] > m.level[face.right] ? face.right : face.left;
        for (int s = 0; s < 4; ++s) mark_code(nbr_raw_[static_cast<size_t>(4) * C + s], mask);
    }
}

void CellFluxEngine::mass_flow_dependencies(const CellMesh& m, std::vector<char>& mask) const {
    for (int s = 0; s < 4; ++s)
        for (const int f : mf_face_[s]) mask[static_cast<size_t>(m.faces[static_cast<size_t>(f)].left >= 0 ? m.faces[static_cast<size_t>(f)].left : m.faces[static_cast<size_t>(f)].right)] = 1;
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
    for (int l = 0; l <= m.max_level; ++l) force_level(m, U, l);   // the leaves this rank holds
}

void CellFluxEngine::force_all(const CellMesh& m, const Field& U) {
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
    const std::vector<int>& held = lheld_[static_cast<size_t>(l)];
    const int nh = static_cast<int>(held.size());
#pragma omp parallel for schedule(static) if (nh >= kOmpMinWork)
    for (int k = 0; k < nh; ++k) {
        const int c = held[static_cast<size_t>(k)];
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
            force_all(m, U);
        else
            force_level(m, U, level);
        us_ = &forced_;
    } else {
        us_ = &U;
    }
    Vc_.resize(vroot_.size() + interp_.size());
    for (size_t v = 0; v < vroot_.size(); ++v) Vc_[v] = eval(*us_, vroot_[v]);
    eval_interp();
    if (muscl_) {
        if (level < 0)
            refresh_all(nleaf_);
        else
            refresh_some(nleaf_, need_[static_cast<size_t>(level)]);
    }
}

void CellFluxEngine::eval_interp() {
    if (interp_.empty()) return;
    const EOS& eos = eos_;
    const Limiter lim = muscl_->limiter();
    const size_t nm = vroot_.size();
    Wint_.resize(interp_.size());
    for (size_t k = 0; k < interp_.size(); ++k) {
        const InterpVirtual& iv = interp_[k];
        const int C = iv.coarse;
        const State Wc = eos.cons_to_prim((*us_)[C]);
        const auto prim_at = [&](int code) { return code < 0 ? Wc : eos.cons_to_prim(cell_state(code)); };
        State sn{}, st{};
        for (int pass = 0; pass < 2; ++pass) {
            const int ax = pass == 0 ? iv.axis : 1 - iv.axis;
            const State Wb = prim_at(nbr_raw_[static_cast<size_t>(4) * C + 2 * ax]);
            const State Wf = prim_at(nbr_raw_[static_cast<size_t>(4) * C + 2 * ax + 1]);
            State& sl = pass == 0 ? sn : st;
            for (int q = 0; q < kNVar; ++q) sl[q] = apply_limiter(lim, Wc[q] - Wb[q], Wf[q] - Wc[q]);
        }
        State w;
        for (int q = 0; q < kNVar; ++q) w[q] = Wc[q] + iv.dn * sn[q] + iv.dt * st[q];
        Wint_[k] = w;
        Vc_[nm + k] = eos.prim_to_cons(w);
    }
}

void CellFluxEngine::refresh_all(int N) {
    W_.resize(static_cast<size_t>(N) + vroot_.size() + interp_.size());
    const EOS& eos = eos_;
    const Field& U = *us_;
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
    for (int c = 0; c < N; ++c) W_[c] = eos.cons_to_prim(U[c]);
    for (size_t v = 0; v < vroot_.size(); ++v) W_[N + v] = eos.cons_to_prim(Vc_[v]);
    for (size_t k = 0; k < interp_.size(); ++k) W_[N + vroot_.size() + k] = Wint_[k];
}

void CellFluxEngine::refresh_some(int N, const std::vector<int>& idx) {
    W_.resize(static_cast<size_t>(N) + vroot_.size() + interp_.size());
    const int nm = static_cast<int>(vroot_.size());
    const EOS& eos = eos_;
    const Field& U = *us_;
    const int n = static_cast<int>(idx.size());
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int k = 0; k < n; ++k) {
        const int i = idx[k];
        W_[i] = i < N ? eos.cons_to_prim(U[i])
                      : (i < N + nm ? eos.cons_to_prim(Vc_[static_cast<size_t>(i - N)]) : Wint_[static_cast<size_t>(i - N - nm)]);
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

double CellFluxEngine::fluxes(const CellMesh& m, const std::vector<int>& faces, const double* u_target) {
    const int nlist = static_cast<int>(faces.size());
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
#pragma omp parallel if (nlist >= kOmpMinWork)
    {
        double local = 0.0;
        // The Riemann problems of the faces go to the solver in chunks (four at a time with AVX2); the results are the same as
        // one call per face.
        constexpr int kChunk = 256;
        State bUL[kChunk], bUR[kChunk], bF[kChunk];
        int bax[kChunk], bface[kChunk];
        double bsm[kChunk];
        int nb = 0;
        const auto flush = [&]() {
            riemann_.face_flux_batch(bUL, bUR, bax, nb, bF, bsm);
            for (int k = 0; k < nb; ++k) {
                F_[bface[k]] = bF[k];
                if (bsm[k] > local) local = bsm[k];
            }
            nb = 0;
        };
        // faces with an ordinary four-point MUSCL stencil: reconstruction and Riemann solver together
        int mst[4 * kChunk], max_[kChunk], mface[kChunk];
        State mF[kChunk];
        double msm[kChunk];
        int nm = 0;
        const auto flush_muscl = [&]() {
            riemann_.muscl_face_flux_batch(lim, W_.data(), mst, max_, nm, mF, msm);
            for (int k = 0; k < nm; ++k) {
                F_[mface[k]] = mF[k];
                if (msm[k] > local) local = msm[k];
            }
            nm = 0;
        };
#pragma omp for schedule(static) nowait
        for (int fi = 0; fi < nlist; ++fi) {
            const int f = faces[static_cast<size_t>(fi)];
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
                if (face.kind == CellFaceKind::CoarseFine) {
                    // the fine side is reconstructed against the (virtual) cell across the face, the coarse side is its cell state
                    const bool fine_left = m.level[face.left] > m.level[face.right];
                    for (int q = 0; q < kNVar; ++q) {
                        // coarse leaf C of the face: its state and its neighbours across the two sides along the normal
                        const int C = fine_left ? s[3] : s[0];
                        const int ax = face.axis;
                        const int code_b = nbr_raw_[static_cast<size_t>(4) * C + 2 * ax];
                        const int code_f = nbr_raw_[static_cast<size_t>(4) * C + 2 * ax + 1];
                        const double wc = coarse_side_second_order ? W_[C][q] : 0.0;
                        const double wb = !coarse_side_second_order ? 0.0 : (code_b < 0 ? wc : W_[code_b][q]);
                        const double wf = !coarse_side_second_order ? 0.0 : (code_f < 0 ? wc : W_[code_f][q]);
                        if (fine_left) {
                            const double c0 = w0[q], c1 = w1[q], c2 = w2[q];
                            wl[q] = c1 + 0.5 * apply_limiter(lim, c1 - c0, c2 - c1);
                            wr[q] = coarse_side_second_order ? wc - 0.5 * apply_limiter(lim, wf - wc, wc - wb) : w3[q];
                        } else {
                            const double c1 = w1[q], c2 = w2[q], c3 = w3[q];
                            wl[q] = coarse_side_second_order ? wc + 0.5 * apply_limiter(lim, wc - wb, wf - wc) : w0[q];
                            wr[q] = c2 - 0.5 * apply_limiter(lim, c3 - c2, c2 - c1);
                        }
                    }
                } else {
                    for (int k = 0; k < 4; ++k) mst[4 * nm + k] = s[k];
                    max_[nm] = face.axis;
                    mface[nm] = f;
                    if (++nm == kChunk) flush_muscl();
                    continue;
                }
                UL = eos.prim_to_cons(wl);
                UR = eos.prim_to_cons(wr);
            }
            bUL[nb] = UL;
            bUR[nb] = UR;
            bax[nb] = face.axis;
            bface[nb] = f;
            if (++nb == kChunk) flush();
        }
        flush();
        flush_muscl();
#pragma omp critical(fv_cellflux_max)
        if (local > smax_all) smax_all = local;
    }
    return smax_all;
}

void viscous_cell_quantities(const EOS& eos, const State& Ui, const State& n0, const State& n1, const State& n2, const State& n3,
                             double dx, double dy, const ViscousCellArrays& out, int c) {
    const State W = eos.cons_to_prim(Ui);
    double gx[3], gy[3];
    for (int q = 0; q < 3; ++q) {
        gx[q] = (n1[q] - n0[q]) / (2.0 * dx);
        gy[q] = (n3[q] - n2[q]) / (2.0 * dy);
    }
    const double rho = W[0], u = W[1], v = W[2];
    out.u[c] = u;
    out.v[c] = v;
    out.T[c] = W[3] / rho;
    out.dudx[c] = (gx[1] - u * gx[0]) / rho;
    out.dudy[c] = (gy[1] - u * gy[0]) / rho;
    out.dvdx[c] = (gx[2] - v * gx[0]) / rho;
    out.dvdy[c] = (gy[2] - v * gy[0]) / rho;
}

void viscous_cell_quantities_run(const EOS& eos, const State* U, const int* nbr, const int* level, const double* dx_of_level,
                                 const double* dy_of_level, int c0, int n, const ViscousCellArrays& out) {
    int done = 0;
#ifdef FV_SIMD_AVX2
    if (detail::avx2_available()) {
        done = n & ~3;
        if (done > 0) {
            double* const o[7] = {out.u, out.v, out.T, out.dudx, out.dudy, out.dvdx, out.dvdy};
            detail::viscous_cells_avx2(eos, U, nbr, level, dx_of_level, dy_of_level, c0, done, o);
        }
    }
#endif
    for (int c = c0 + done; c < c0 + n; ++c) {
        const int* nb = nbr + static_cast<size_t>(4) * c;
        viscous_cell_quantities(eos, U[c], U[nb[0]], U[nb[1]], U[nb[2]], U[nb[3]], dx_of_level[level[c]], dy_of_level[level[c]], out,
                                c);
    }
}

double viscous_cell_speed(const EOS& eos, double mu, double k_cond, const State& U, double dx, double dy) {
    const double gamma = eos.gamma;
    const double L = std::min(dx, dy), inv_h2 = 1.0 / (dx * dx) + 1.0 / (dy * dy);
    const State W = eos.cons_to_prim(U);
    const double rho = W[0];
    const double nu_eff = std::max(mu / rho, k_cond / (rho * (gamma / (gamma - 1.0)))) + 1e-12;
    return std::max(std::abs(W[1]), std::abs(W[2])) + eos.sound_speed(W[3], rho) + 2.0 * L * inv_h2 * nu_eff;
}

double viscous_speed_range(const EOS& eos, double mu, double k_cond, const State* U, const int* level, const double* dx_of_level,
                           const double* dy_of_level, int cb, int ce) {
    double s_max = 0.0;
    int done = 0;
#ifdef FV_SIMD_AVX2
    if (detail::avx2_available()) {
        done = (ce - cb) & ~3;
        if (done > 0) s_max = detail::viscous_speed_avx2(eos, mu, k_cond, U, level, dx_of_level, dy_of_level, cb, done);
    }
#endif
    for (int c = cb + done; c < ce; ++c)
        s_max = std::max(s_max, viscous_cell_speed(eos, mu, k_cond, U[c], dx_of_level[level[c]], dy_of_level[level[c]]));
    return s_max;
}

namespace {

// Cell sizes per level (CellMesh::dx / dy) for the viscous kernels
struct LevelSizes {
    std::vector<double> dx, dy;
    explicit LevelSizes(const CellMesh& m) : dx(static_cast<size_t>(m.max_level) + 1), dy(dx.size()) {
        for (int l = 0; l <= m.max_level; ++l) {
            dx[static_cast<size_t>(l)] = m.dx(l);
            dy[static_cast<size_t>(l)] = m.dy(l);
        }
    }
};

constexpr int kViscBlock = 256;   // list entries per block of the viscous loops (the unit of the OpenMP schedule)

}   // namespace

void CellFluxEngine::cell_quantities(const CellMesh& m, const std::vector<int>& cells) {
    const EOS& eos = eos_;
    const Field& U = *us_;
    const int n = static_cast<int>(cells.size());
    const LevelSizes hs(m);
    const ViscousCellArrays out{su_.data(), sv_.data(), sT_.data(), dudx_.data(), dudy_.data(), dvdx_.data(), dvdy_.data()};
    const int nblock = (n + kViscBlock - 1) / kViscBlock;
    // Runs of consecutive leaves whose four neighbours are leaves go to the range kernel (four at a time with AVX2); a cell with a
    // virtual neighbour or a physical edge is done on its own.
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
    for (int b = 0; b < nblock; ++b) {
        const int kb = b * kViscBlock, ke = std::min(n, kb + kViscBlock);
        int run_c = 0, run_n = 0;
        for (int k = kb; k < ke; ++k) {
            const int c = cells[k];
            const int* code = &nbr_[static_cast<size_t>(4) * c];
            const auto leaf = [&](int s) { return static_cast<unsigned>(code[s]) < static_cast<unsigned>(nleaf_); };
            const bool leaves = leaf(0) && leaf(1) && leaf(2) && leaf(3);
            if (leaves && run_n > 0 && c == run_c + run_n) {
                ++run_n;
                continue;
            }
            if (run_n > 0) viscous_cell_quantities_run(eos, U.data(), nbr_.data(), m.level.data(), hs.dx.data(), hs.dy.data(), run_c,
                                                       run_n, out);
            run_n = 0;
            if (leaves) {
                run_c = c;
                run_n = 1;
                continue;
            }
            const int l = m.level[c];
            const State& Ui = U[c];
            State nb[4];   // left, right, bottom, top values (the diffusive ghost at a physical edge)
            for (int s = 0; s < 4; ++s) nb[s] = code[s] >= 0 ? cell_state(code[s]) : bc_.viscous_ghost(static_cast<Face>(s), Ui);
            viscous_cell_quantities(eos, Ui, nb[0], nb[1], nb[2], nb[3], m.dx(l), m.dy(l), out, c);
        }
        if (run_n > 0)
            viscous_cell_quantities_run(eos, U.data(), nbr_.data(), m.level.data(), hs.dx.data(), hs.dy.data(), run_c, run_n, out);
    }
}

void CellFluxEngine::viscous_fluxes(const CellMesh& m, const std::vector<int>& faces) {
    const int nlist = static_cast<int>(faces.size());
    const NavierStokes& ns = *phys_.ns;
    const double mu = ns.mu(), gamma = eos_.gamma;
    const double k_cond = mu * gamma / (ns.prandtl() * (gamma - 1.0));
    const Field& U = *us_;
    if (Fv_.size() != m.faces.size()) Fv_.resize(m.faces.size());
#pragma omp parallel for schedule(static) if (nlist >= kOmpMinWork)
    for (int fi = 0; fi < nlist; ++fi) {
        const int f = faces[static_cast<size_t>(fi)];
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
    const LevelSizes hs(m);
    const int nblock = (ce - cb + kViscBlock - 1) / kViscBlock;
    double s_max = 0.0;
#pragma omp parallel if (ce - cb >= kOmpMinWork)
    {
        double local = 0.0;
        // blocks of consecutive leaves, four at a time with AVX2: a maximum does not depend on the grouping (every speed is > 0 or
        // NaN, which std::max(local, s) ignores)
#pragma omp for schedule(static) nowait
        for (int b = 0; b < nblock; ++b) {
            const int c0 = cb + b * kViscBlock, c1 = std::min(ce, c0 + kViscBlock);
            local = std::max(local, viscous_speed_range(eos_, mu, k_cond, U.data(), m.level.data(), hs.dx.data(), hs.dy.data(), c0, c1));
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
    double s = fluxes(m, all_faces_, u_target);
    if (phys_.ns) {
        std::vector<int> all(static_cast<size_t>(N));
        for (int c = 0; c < N; ++c) all[c] = c;
        cell_quantities(m, all);
        viscous_fluxes(m, all_faces_);
        s = std::max(s, viscous_speed(m, 0, N));
    }
    gather(m, 0, N, false, R);
    return s;
}

double CellFluxEngine::wave_speed(const CellMesh& m, const Field& U) {
    double s = 0.0;
    for (int l = 0; l <= m.max_level; ++l)
        if (m.level_begin[l + 1] > m.level_begin[l]) s = std::max(s, compute_level_fluxes(m, U, l));
    return s;
}

double CellFluxEngine::compute_level_fluxes(const CellMesh& m, const Field& U, int l) {
    if (m.num_leaves() != nleaf_ || static_cast<int>(stencil_.size()) != static_cast<int>(m.faces.size()))
        throw std::logic_error("CellFluxEngine: build() was not called for this mesh");
    prepare(m, U, l);
    double u_target[4];
    mass_flow_targets(m, u_target);
    double s = fluxes(m, lface_[static_cast<size_t>(l)], u_target);
    if (phys_.ns) {
        cell_quantities(m, vcells_[static_cast<size_t>(l)]);
        viscous_fluxes(m, lface_[static_cast<size_t>(l)]);
        s = std::max(s, viscous_speed(m, lown_b_[static_cast<size_t>(l)], lown_e_[static_cast<size_t>(l)]));
    }
    return s;
}

void CellFluxEngine::gather_level(const CellMesh& m, int l, bool replace, Field& Rl) {
    gather(m, lown_b_[static_cast<size_t>(l)], lown_e_[static_cast<size_t>(l)], replace, Rl);
}

void CellFluxEngine::clear_registers(const CellMesh& m, int l) {
    for (int f = m.face_level_begin[l]; f < m.face_level_begin[l + 1]; ++f) {
        reg_[f] = State{};
        regv_[f] = State{};
    }
}

void CellFluxEngine::accumulate_registers(const CellMesh& m, int l, double w) {
    const std::vector<int>& faces = lface_[static_cast<size_t>(l)];
    const int nlist = static_cast<int>(faces.size());
    const bool visc = phys_.ns != nullptr;
#pragma omp parallel for schedule(static) if (nlist >= kOmpMinWork)
    for (int fi = 0; fi < nlist; ++fi) {
        const int f = faces[static_cast<size_t>(fi)];
        if (m.faces[f].kind == CellFaceKind::CoarseFine)
            for (int q = 0; q < kNVar; ++q) {
                reg_[f][q] += w * F_[f][q];
                if (visc) regv_[f][q] += w * Fv_[f][q];
            }
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
            if (!solid_[c] || (!owned_mask_.empty() && !owned_mask_[c])) continue;
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
        if (!solid_[c] || (!owned_mask_.empty() && !owned_mask_[c])) continue;
        if (!fringe[c]) {
            new_rho[c] = ibm_rho_ref_;
            new_p[c] = ibm_p_ref_;
        }   // deep solid
        U[c] = {new_rho[c], 0.0, 0.0, new_p[c] / (eos_.gamma - 1.0), 0.0};
    }
}

}   // namespace fv
