#include "fv/Switches.hpp"
#include "fv/CellFlux.hpp"
#include "fv/ViscousBound.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

#include "SchemesSimd.hpp"
#include "fv/MemoryBytes.hpp"
#include "fv/ParallelScan.hpp"

namespace fv {

namespace {

// A virtual cell whose block is not all there (a partial mesh): not a code of a leaf or a node, and not kOutside (-2 is the node 1)
constexpr int kIncomplete = std::numeric_limits<int>::min();

// Code of the block (l, i, j, k): a leaf, or a node whose children are the 2^D sub-blocks in key order (appended to vnode);
// kIncomplete if a part of the block is not in the mesh (a partial mesh)
int block_code(const CellMesh& m, std::vector<std::array<int, 8>>& vnode, int l, int i, int j, int k) {
    if (l > m.max_level) return kIncomplete;
    const int leaf = m.find(l, i, j, k);
    if (leaf >= 0) return leaf;
    std::array<int, 8> ch;
    ch.fill(kIncomplete);
    bool whole = true;
    for (int q = 0; q < m.children(); ++q) {
        int o[3];
        child_offset(q, o);
        ch[static_cast<size_t>(q)] = block_code(m, vnode, l + 1, 2 * i + o[0], 2 * j + o[1], 2 * k + o[2]);
        whole = whole && ch[static_cast<size_t>(q)] != kIncomplete;
    }
    if (!whole) return kIncomplete;
    vnode.push_back(ch);
    return -static_cast<int>(vnode.size());   // node k <-> code -1 - k
}

}   // namespace

State CellFluxEngine::eval(const Field& U, int code) const {
    if (code >= 0) return U[code];
    const std::array<int, 8>& ch = vnode_[static_cast<size_t>(-1 - code)];
    const State a = eval(U, ch[0]), b = eval(U, ch[1]), c = eval(U, ch[2]), d = eval(U, ch[3]);
    State r;
    if (nch_ == 4) {
        for (int k = 0; k < kNVar; ++k) r[k] = ((a[k] + b[k]) + (c[k] + d[k])) * 0.25;
        return r;
    }
    // 3-D: each z-plane as in 2-D, then the two planes (the order of CellMesh's merge)
    const State e = eval(U, ch[4]), f = eval(U, ch[5]), g = eval(U, ch[6]), h = eval(U, ch[7]);
    for (int k = 0; k < kNVar; ++k) r[k] = (((a[k] + b[k]) + (c[k] + d[k])) + ((e[k] + f[k]) + (g[k] + h[k]))) * 0.125;
    return r;
}

State CellFluxEngine::cell_state(int code) const { return code < nleaf_ ? (*us_)[code] : Vc_[code - nleaf_]; }

void CellFluxEngine::size_work_arrays(const CellMesh& m) {
    // Work arrays, sized only: every entry is written before it is read in a step (fluxes and cell quantities by the pass of
    // their level, registers by clear_registers before the subcycles), as for F_; zero-filling them took ~2 % of the loop.
    // Registers only for the coarse-fine faces (slot per face, consecutive in face order, so each level's are one run); the
    // viscous face fluxes only with viscous terms.
    reg_slot_.assign(m.faces.size(), -1);
    reg_level_begin_.assign(static_cast<size_t>(m.max_level) + 2, 0);
    int nreg = 0;
    for (int l = 0; l <= m.max_level; ++l) {
        reg_level_begin_[static_cast<size_t>(l)] = nreg;
        for (int f = m.face_level_begin[l]; f < m.face_level_begin[l + 1]; ++f)
            if (m.faces[static_cast<size_t>(f)].kind == CellFaceKind::CoarseFine) reg_slot_[static_cast<size_t>(f)] = nreg++;
    }
    reg_level_begin_[static_cast<size_t>(m.max_level) + 1] = nreg;
    reg_.resize(static_cast<size_t>(nreg));
    regv_.resize(static_cast<size_t>(nreg));
    if (phys_.ns) Fv_.resize(m.faces.size());
    else Fv_ = Field();
    for (std::vector<double>* a : {&su_, &sv_, &sT_, &dudx_, &dudy_, &dvdx_, &dvdy_}) a->resize(static_cast<size_t>(nleaf_));
    for (std::vector<double>* a : {&sw_, &dudz_, &dvdz_, &dwdx_, &dwdy_, &dwdz_}) {
        if (dim_ == 3 && phys_.ns)
            a->resize(static_cast<size_t>(nleaf_));
        else
            std::vector<double>().swap(*a);
    }
}

void CellFluxEngine::build(const CellMesh& m, const CellMesh* whole) {
    const auto per = [&](Face lo, Face hi) {
        const bool a = bc_.type(lo) == BCType::Periodic, b = bc_.type(hi) == BCType::Periodic;
        if (a != b) throw std::invalid_argument("CellFluxEngine: both sides of a periodic axis must be periodic");
        return a;
    };
    if (per(Face::Left, Face::Right) != m.periodic_x || per(Face::Bottom, Face::Top) != m.periodic_y ||
        (m.dim == 3 && per(Face::Front, Face::Back) != m.periodic_z))
        throw std::invalid_argument("CellFluxEngine: the periodic axes of the mesh and the boundary types differ");
    if (m.dim == 3) {   // what the octree engine does not do yet (docs/design/amr-octree.md, phase Q4)
        if (muscl_ && muscl_->scheme() == Reconstruction::Weno5)
            throw std::invalid_argument("CellFluxEngine: WENO5-Z is not available on an adaptive mesh");
    }
    if (built_mesh_ != &m || dim_ != m.dim) built_mesh_ = nullptr;   // no incremental build across a change of dimension
    dim_ = m.dim;
    ns_ = m.sides();
    nch_ = m.children();

    const bool ip = muscl_ && interpolate_jumps;
    // incremental: right after an adapt of the mesh of the last build (CellMesh::prev_leaf), same interpolation setting
    const bool inc = !m.prev_leaf.empty() && built_mesh_ == &m && built_version_ == m.version - 1 && built_ip_ == ip &&
                     static_cast<int>(m.prev_leaf.size()) == m.num_leaves();
    const int old_n = nleaf_;
    std::vector<int> old_raw, old_nbr;
    std::vector<std::array<int, 4>> old_st;
    std::vector<std::array<int, 5>> old_ikey;
    std::vector<std::array<int, 4>> old_vkey;
    std::vector<char> old_special;
    if (inc) {   // the tables of the build before last become the buffers of the new ones
        old_raw.swap(nbr_raw_);
        old_nbr.swap(nbr_);
        old_st.swap(stencil_);
        nbr_raw_.swap(nbr_raw_spare_);
        nbr_.swap(nbr_spare_);
        stencil_.swap(stencil_spare_);
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
        for (int s = 0; s < ns_; ++s) {
            const CellNeighbour::Kind k = m.neighbour(c, static_cast<Face>(s)).kind;
            if (k == CellNeighbour::Kind::Finer || k == CellNeighbour::Kind::Coarser) special_[c] = 1;
        }
    }
    vkey_.clear();
    ikey_.clear();
    std::vector<int> special_leaves;
    collect_indices(0, nleaf_, [&](int c) { return special_[c] != 0; }, special_leaves);
    for (const int c : special_leaves) {
        for (int s = 0; s < ns_; ++s) {
            const CellNeighbour nb = m.neighbour(c, static_cast<Face>(s));
            if (nb.kind == CellNeighbour::Kind::Finer)
                vkey_.push_back({m.level[nb.a] - 1, m.ci[nb.a] >> 1, m.cj[nb.a] >> 1, m.ck[nb.a] >> 1});
            else if (nb.kind == CellNeighbour::Kind::Coarser && ip)
                ikey_.push_back({m.level[c], m.ci[c], m.cj[c], m.ck[c], s});
        }
    }
    std::sort(vkey_.begin(), vkey_.end());
    vkey_.erase(std::unique(vkey_.begin(), vkey_.end()), vkey_.end());
    std::sort(ikey_.begin(), ikey_.end());
    make_virtual_cells(m);
    // 2) injecting neighbour codes of every leaf (slopes of the coarse leaves), stencils, neighbour codes with the interpolated
    // virtual cells where enabled (gradients, non-reflecting tangential differences). A clean leaf, and a face between two clean
    // leaves made by one of them, keep their old codes, renumbered: leaves by the adapt, virtual cells by key.
    // (every entry is written below: leaf_codes and face_stencil write -1 on the boundary)
    nbr_raw_.resize(static_cast<size_t>(ns_) * nleaf_);
    nbr_.resize(static_cast<size_t>(ns_) * nleaf_);
    stencil_.resize(m.faces.size());
    if (inc) {
        std::vector<int> leaf_map(static_cast<size_t>(old_n), -1);
#pragma omp parallel for schedule(static) if (nleaf_ >= kOmpMinWork)
        for (int c = 0; c < nleaf_; ++c)   // prev_leaf is one-to-one: distinct entries
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
            const size_t o = static_cast<size_t>(ns_) * m.prev_leaf[c], n = static_cast<size_t>(ns_) * c;
            for (int s = 0; s < ns_; ++s) {
                nbr_raw_[n + s] = remap(old_raw[o + s]);
                nbr_[n + s] = remap(old_nbr[o + s]);
            }
        }
#pragma omp parallel for schedule(static) if (m.faces.size() >= static_cast<size_t>(kOmpMinWork))
        for (int f = 0; f < static_cast<int>(m.faces.size()); ++f) {
            const CellFace& face = m.faces[static_cast<size_t>(f)];
            const int pf = m.prev_face[static_cast<size_t>(f)];
            if (face.kind == CellFaceKind::Boundary || pf < 0 || !clean(face.left) || !clean(face.right)) {
                face_stencil(m, f);
                continue;
            }
            for (int k = 0; k < 4; ++k) stencil_[static_cast<size_t>(f)][k] = remap(old_st[static_cast<size_t>(pf)][k]);
        }
        if (lost) throw std::logic_error("CellFluxEngine: a clean leaf refers to a cell that is gone (incremental build)");
        nbr_raw_spare_.swap(old_raw);
        nbr_spare_.swap(old_nbr);
        stencil_spare_.swap(old_st);
    } else {
#pragma omp parallel for schedule(static) if (nleaf_ >= kOmpMinWork)
        for (int c = 0; c < nleaf_; ++c) leaf_codes(m, c);
#pragma omp parallel for schedule(static) if (m.faces.size() >= static_cast<size_t>(kOmpMinWork))
        for (int f = 0; f < static_cast<int>(m.faces.size()); ++f) face_stencil(m, f);
    }

    size_work_arrays(m);

    // immersed body: mask of every leaf from the uniform mask of its level (a leaf kept by the adapt keeps its value)
    std::vector<char> old_solid;
    old_solid.swap(solid_);
    if (has_body() && m.dim == 3) {
        // per-leaf classification by the centre test of the body's geometry (the uniform mask is that test at the cell centres)
        if (!ibm_geo_) {
            // the body's shape follows the mesh it is built on (a sphere on a mesh with nz = 1 is a cylinder): an octree with a single
            // base cell along z still has 3-D levels, so the geometry is built on a mesh with at least two
            ibm_geo_ = std::make_shared<ImmersedBoundary>(
                phys_.ibm_factory(generate_cartesian_mesh_3d(m.Lx, m.Ly, m.Lz, m.nx, m.ny, std::max(m.nz, 2), m.x0, m.y0, m.z0)));
            if (!ibm_geo_->has_geometry())
                throw std::invalid_argument("CellFluxEngine: a 3-D adaptive mesh needs a body built from its geometry (a mask-only body)");
            ibm_rho_ref_ = ibm_geo_->rho_ref();
            ibm_p_ref_ = ibm_geo_->p_ref();
        }
        solid_.assign(static_cast<size_t>(nleaf_), 0);
        const ImmersedBoundary& ib = *ibm_geo_;
#pragma omp parallel for schedule(static) if (nleaf_ >= kOmpMinWork)
        for (int c = 0; c < nleaf_; ++c)
            solid_[c] = inc && m.prev_leaf[c] >= 0 ? old_solid[static_cast<size_t>(m.prev_leaf[c])]
                                                   : static_cast<char>(ib.solid_at(m.xc(c), m.yc(c), m.zc(c)));
        forced_.resize(static_cast<size_t>(nleaf_));
    } else if (has_body()) {
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
    if (whole) {   // a partial mesh: the lists come with the partition
        for (auto* v : {&lface_, &lheld_, &need_, &vcells_}) v->clear();
    } else {
        rebuild_lists(m, nullptr);
    }
    map_ghost_cells(m, whole);

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
        std::vector<double> xs(fresh.size()), ys(fresh.size()), zs(m.dim == 3 ? fresh.size() : 0);
        for (size_t k = 0; k < fresh.size(); ++k) {
            xs[k] = m.xc(fresh[k]);
            ys[k] = m.yc(fresh[k]);
            if (m.dim == 3) zs[k] = m.zc(fresh[k]);
        }
        Field sv(fresh.size());
        evaluate_source_points(*phys_.source_mesh, *phys_.source, 0.0, xs.data(), ys.data(), static_cast<int>(fresh.size()),
                               sv.data(), m.dim == 3 ? zs.data() : nullptr);
        for (size_t k = 0; k < fresh.size(); ++k) src_[static_cast<size_t>(fresh[k])] = sv[k];
    } else {
        src_.resize(static_cast<size_t>(nleaf_));
#pragma omp parallel for schedule(static) if (nleaf_ >= kOmpMinWork)
        for (int c = 0; c < nleaf_; ++c) src_[c] = State{};
        if (phys_.source && !phys_.source_unsteady) update_source(m, 0, nleaf_);
    }

    for (int s = 0; s < 6; ++s) {
        mf_face_[s].clear();
        mf_weight_[s].clear();
        if (s >= ns_) continue;
        if (bc_.type(static_cast<Face>(s)) != BCType::MassflowInlet) continue;
        // in the order of the uniform solver's boundary cells: the second transverse position (3-D) major, the first minor
        std::vector<std::pair<std::pair<long long, long long>, int>> order;
        int fine = 0;
        for (size_t f = 0; f < m.faces.size(); ++f) {
            const CellFace& face = m.faces[f];
            if (face.kind != CellFaceKind::Boundary || face.side != static_cast<Face>(s)) continue;
            const int sh = m.max_level - face.level;
            order.push_back({{static_cast<long long>(face.tpos2) << sh, static_cast<long long>(face.tpos) << sh}, static_cast<int>(f)});
            fine = std::max(fine, face.level);
        }
        std::sort(order.begin(), order.end());
        for (const auto& o : order) {
            mf_face_[s].push_back(o.second);
            // the finer faces a face covers: 2 per level in 2-D, 4 in 3-D
            mf_weight_[s].push_back(static_cast<double>(1 << ((m.dim - 1) * (fine - m.faces[o.second].level))));
        }
    }
    built_mesh_ = &m;
    built_version_ = m.version;
    built_ip_ = ip;
}

// special_ of a leaf with a side that is not known (a partial mesh's edge) is not defined; nothing active reads it
bool CellFluxEngine::same_special(const CellFluxEngine& o, const CellMesh* edge_mesh) const {
    if (special_.size() != o.special_.size()) return false;
    for (size_t c = 0; c < special_.size(); ++c) {
        if (special_[c] == o.special_[c]) continue;
        if (!edge_mesh) return false;
        // a leaf with a side the partial mesh does not know has no flag worth comparing: no active face reads it (nbr_ and stencil_ are compared)
        bool edge = false;
        for (int s = 0; s < ns_; ++s)
            edge = edge || edge_mesh->neighbour(static_cast<int>(c), static_cast<Face>(s)).kind == CellNeighbour::Kind::Outside;
        if (!edge) return false;
    }
    return true;
}

// The finest-level cells -> leaves of the wall. In 2-D the table is the whole mesh's, cut to the kept leaves: equal. In 3-D a rank's
// table is made from its own solid leaves, so it lacks the cells that only the ghost cells of other ranks read: its entries must be
// among the other's, with the same leaf (the entries it has are the ones its fill and its solid reset look up).
bool CellFluxEngine::same_fine_leaves(const CellFluxEngine& o) const {
    if (dim_ != 3) return fine_leaf_ == o.fine_leaf_;
    for (const auto& e : fine_leaf_) {
        const auto it = o.fine_leaf_.find(e.first);
        if (it == o.fine_leaf_.end() || it->second != e.second) return false;
    }
    return true;
}

bool CellFluxEngine::same_tables(const CellFluxEngine& o, const CellMesh* edge_mesh) const {
    const auto same_field = [](const Field& a, const Field& b) {
        return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(State)) == 0);
    };
    const std::pair<const char*, bool> checks[] = {
        {"nleaf", nleaf_ == o.nleaf_}, {"vkey", vkey_ == o.vkey_}, {"ikey", ikey_ == o.ikey_}, {"vroot", vroot_ == o.vroot_},
        {"vnode", vnode_ == o.vnode_}, {"nbr_raw", nbr_raw_ == o.nbr_raw_}, {"nbr", nbr_ == o.nbr_}, {"stencil", stencil_ == o.stencil_},
        {"special", same_special(o, edge_mesh)}, {"lface", lface_ == o.lface_}, {"lheld", lheld_ == o.lheld_}, {"need", need_ == o.need_},
        {"vcells", vcells_ == o.vcells_}, {"lown", lown_b_ == o.lown_b_ && lown_e_ == o.lown_e_}, {"solid", solid_ == o.solid_},
        {"src", same_field(src_, o.src_)}, {"interp", interp_.size() == o.interp_.size()}};
    bool ok = true;
    for (const auto& c : checks)
        if (!c.second) {
            if (switch_on("FENICE_DEBUG_TABLES")) std::fprintf(stderr, "same_tables: %s differs\n", c.first);
            ok = false;
        }
    for (size_t k = 0; ok && k < interp_.size(); ++k) {
        const InterpVirtual &a = interp_[k], &b = o.interp_[k];
        ok = a.fine == b.fine && a.coarse == b.coarse && a.side == b.side && a.axis == b.axis &&
             std::memcmp(&a.dn, &b.dn, sizeof(double)) == 0 && std::memcmp(&a.dt, &b.dt, sizeof(double)) == 0 &&
             std::memcmp(&a.dt2, &b.dt2, sizeof(double)) == 0;
    }
    ok = ok && dim_ == o.dim_;
    for (int s = 0; ok && s < 6; ++s) ok = mf_face_[s] == o.mf_face_[s] && mf_weight_[s] == o.mf_weight_[s];
    ok = ok && owned_mask_ == o.owned_mask_ && held_mask_ == o.held_mask_ && all_faces_ == o.all_faces_ && ghost_ready_ == o.ghost_ready_ &&
         (!ghost_ready_ || (same_fine_leaves(o) && ghost_reads_leaf_ == o.ghost_reads_leaf_ && ghost_leaf_mask_ == o.ghost_leaf_mask_));
    return ok;
}

void CellFluxEngine::make_virtual_cells(const CellMesh& m) {
    vroot_.clear();
    vnode_.clear();
    for (const std::array<int, 4>& k : vkey_) vroot_.push_back(block_code(m, vnode_, k[0], k[1], k[2], k[3]));
    interp_.clear();
    for (const std::array<int, 5>& k : ikey_) {
        InterpVirtual iv;
        iv.fine = m.find(k[0], k[1], k[2], k[3]);
        iv.side = static_cast<Face>(k[4]);
        iv.coarse = m.neighbour(iv.fine, iv.side).a;
        iv.axis = face_axis(iv.side);
        iv.dn = (k[4] & 1) ? -0.25 : 0.25;   // the virtual centre lies nearer to the fine leaf
        // transverse halves of the coarse leaf the fine leaf is in (coordinates k[1..3] = i, j, k)
        const int idx_t = k[1 + transverse1(iv.axis)];
        iv.dt = (idx_t & 1) ? 0.25 : -0.25;
        if (m.dim == 3) iv.dt2 = (k[1 + transverse2(iv.axis)] & 1) ? 0.25 : -0.25;
        interp_.push_back(iv);
    }
}

int CellFluxEngine::beyond(const CellMesh& m, int c, Face side, bool interpolate) const {
    const CellNeighbour nb = m.neighbour(c, side);
    switch (nb.kind) {
    case CellNeighbour::Kind::Outside: return kOutside;   // a partial mesh: not known here
    case CellNeighbour::Kind::Boundary: return c;   // edge replication
    case CellNeighbour::Kind::Same: return nb.a;
    case CellNeighbour::Kind::Coarser: {
        if (!interpolate) return nb.a;   // injection of the coarse value
        const std::array<int, 5> key = {m.level[c], m.ci[c], m.cj[c], m.ck[c], static_cast<int>(side)};
        const auto it = std::lower_bound(ikey_.begin(), ikey_.end(), key);
        if (it == ikey_.end() || *it != key) throw std::logic_error("CellFluxEngine: missing interpolated virtual cell");
        return nleaf_ + static_cast<int>(vkey_.size()) + static_cast<int>(it - ikey_.begin());
    }
    default: {
        const std::array<int, 4> key = {m.level[nb.a] - 1, m.ci[nb.a] >> 1, m.cj[nb.a] >> 1, m.ck[nb.a] >> 1};
        const auto it = std::lower_bound(vkey_.begin(), vkey_.end(), key);
        if (it == vkey_.end() || *it != key) throw std::logic_error("CellFluxEngine: missing virtual cell");
        return nleaf_ + static_cast<int>(it - vkey_.begin());
    }
    }
}

void CellFluxEngine::leaf_codes(const CellMesh& m, int c) {
    const bool ip = muscl_ && interpolate_jumps;
    for (int s = 0; s < ns_; ++s) {
        if (m.neighbour(c, static_cast<Face>(s)).kind == CellNeighbour::Kind::Boundary) {
            nbr_raw_[static_cast<size_t>(ns_) * c + s] = nbr_[static_cast<size_t>(ns_) * c + s] = -1;
            continue;
        }
        nbr_raw_[static_cast<size_t>(ns_) * c + s] = beyond(m, c, static_cast<Face>(s), false);
        nbr_[static_cast<size_t>(ns_) * c + s] = beyond(m, c, static_cast<Face>(s), ip);
    }
}

void CellFluxEngine::face_stencil(const CellMesh& m, int f) {
    const CellFace& face = m.faces[static_cast<size_t>(f)];
    if (face.kind == CellFaceKind::Boundary) {
        stencil_[static_cast<size_t>(f)] = {-1, -1, -1, -1};
        return;
    }
    if (face.left == kOutside || face.right == kOutside) {   // a partial mesh: a face at the edge of what the rank keeps
        stencil_[static_cast<size_t>(f)] = {kOutside, kOutside, kOutside, kOutside};
        return;
    }
    const bool ip = muscl_ && interpolate_jumps;
    const Face back = static_cast<Face>(2 * face.axis), fwd = static_cast<Face>(2 * face.axis + 1);
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
    if (!part) {
        rebuild_lists_whole(m);
        return;
    }
    std::vector<char> seen(static_cast<size_t>(nleaf_) + vroot_.size() + interp_.size() + 1, 0);
    const size_t L = static_cast<size_t>(m.max_level) + 1;
    lface_.assign(L, {});
    lheld_.assign(L, {});
    need_.assign(L, {});
    vcells_.assign(L, {});
    lown_b_.assign(L, 0);
    lown_e_.assign(L, 0);
    owned_mask_ = part ? part->leaf_owned : std::vector<char>();
    held_mask_ = part ? part->leaf_held : std::vector<char>();
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
                    for (int s = 0; s < ns_; ++s)
                        if (nbr_raw_[static_cast<size_t>(ns_) * C + s] >= 0) v.push_back(nbr_raw_[static_cast<size_t>(ns_) * C + s]);
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

void CellFluxEngine::rebuild_lists_whole(const CellMesh& m) {
    // rebuild_lists without a partition, in parallel: every leaf is owned and held, every face active, so the leaf and face
    // lists of a level are its ranges; need_ and vcells_ are sets (sorted, distinct), marked from the faces and compacted.
    const int ncode = nleaf_ + static_cast<int>(vroot_.size() + interp_.size());
    const size_t L = static_cast<size_t>(m.max_level) + 1;
    lface_.assign(L, {});
    lheld_.assign(L, {});
    need_.assign(L, {});
    vcells_.assign(L, {});
    lown_b_.assign(L, 0);
    lown_e_.assign(L, 0);
    owned_mask_.clear();
    held_mask_.clear();
    const int NF = static_cast<int>(m.faces.size());
    all_faces_.resize(m.faces.size());
#pragma omp parallel for schedule(static) if (NF >= kOmpMinWork)
    for (int f = 0; f < NF; ++f) all_faces_[static_cast<size_t>(f)] = f;
    const auto range = [](std::vector<int>& v, int b, int e) {
        v.resize(static_cast<size_t>(e - b));
#pragma omp parallel for schedule(static) if (e - b >= kOmpMinWork)
        for (int k = b; k < e; ++k) v[static_cast<size_t>(k - b)] = k;
    };
    MarkArray seen(static_cast<size_t>(ncode) + 1);
    const auto mark = [&](int x) { seen[static_cast<size_t>(x)].store(1, std::memory_order_relaxed); };
    for (int l = 0; l <= m.max_level; ++l) {
        const size_t sl = static_cast<size_t>(l);
        const int cb = m.level_begin[l], ce = m.level_begin[l + 1], fb = m.face_level_begin[l], fe = m.face_level_begin[l + 1];
        lown_b_[sl] = cb;
        lown_e_[sl] = ce;
        range(lheld_[sl], cb, ce);
        range(lface_[sl], fb, fe);
        // W entries read by the stencils of the faces of this level
        if (muscl_) {
#pragma omp parallel for schedule(static) if (fe - fb >= kOmpMinWork)
            for (int f = fb; f < fe; ++f) {
                const CellFace& face = m.faces[static_cast<size_t>(f)];
                if (face.kind == CellFaceKind::Boundary) continue;
                const std::array<int, 4>& st = stencil_[static_cast<size_t>(f)];
                for (int k = 0; k < 4; ++k) mark(st[k]);
                if (coarse_side_second_order && face.kind == CellFaceKind::CoarseFine) {
                    const int C = m.level[face.left] > m.level[face.right] ? st[3] : st[0];
                    for (int s = 0; s < ns_; ++s)
                        if (nbr_raw_[static_cast<size_t>(ns_) * C + s] >= 0) mark(nbr_raw_[static_cast<size_t>(ns_) * C + s]);
                }
            }
            take_marked(seen, 0, ncode, need_[sl]);
        }
        // cells whose viscous quantities the faces of this level read: the coarse leaves of its coarse-fine faces (of lower
        // levels, so before the level's own leaves), then the level's leaves
        std::vector<int>& v = vcells_[sl];
        if (l > 0) {
#pragma omp parallel for schedule(static) if (fe - fb >= kOmpMinWork)
            for (int f = fb; f < fe; ++f) {
                const CellFace& face = m.faces[static_cast<size_t>(f)];
                if (face.kind == CellFaceKind::CoarseFine) mark(m.level[face.left] < l ? face.left : face.right);
            }
            take_marked(seen, 0, cb, v);
        }
        const size_t nc = v.size();
        v.resize(nc + static_cast<size_t>(ce - cb));
#pragma omp parallel for schedule(static) if (ce - cb >= kOmpMinWork)
        for (int c = cb; c < ce; ++c) v[nc + static_cast<size_t>(c - cb)] = c;
    }
}

void CellFluxEngine::set_partition(const CellMesh& m, const Partition& p) { rebuild_lists(m, &p); }

void CellFluxEngine::mass_flow_faces(std::vector<char>& mask) const {
    for (int s = 0; s < 6; ++s)
        for (const int f : mf_face_[s]) mask[static_cast<size_t>(f)] = 1;
}

namespace {
// Is every leaf under `code` (a leaf, or a node of vnode) kept?
bool tree_kept(const std::vector<std::array<int, 8>>& vnode, int nch, const std::vector<int>& leaf_map, int code) {
    if (code == kIncomplete) return false;
    if (code >= 0) return leaf_map[static_cast<size_t>(code)] >= 0;
    const std::array<int, 8>& ch = vnode[static_cast<size_t>(-1 - code)];
    for (int k = 0; k < nch; ++k)
        if (!tree_kept(vnode, nch, leaf_map, ch[static_cast<size_t>(k)])) return false;
    return true;
}
// Copy the tree of `code` into `out` with the leaves renumbered (children before their node, like block_code).
int copy_tree(const std::vector<std::array<int, 8>>& vnode, int nch, const std::vector<int>& leaf_map,
              std::vector<std::array<int, 8>>& out, int code) {
    if (code >= 0) return leaf_map[static_cast<size_t>(code)];
    std::array<int, 8> ch;
    ch.fill(kIncomplete);
    const std::array<int, 8>& old = vnode[static_cast<size_t>(-1 - code)];
    for (int k = 0; k < nch; ++k) ch[static_cast<size_t>(k)] = copy_tree(vnode, nch, leaf_map, out, old[static_cast<size_t>(k)]);
    out.push_back(ch);
    return -static_cast<int>(out.size());
}
}   // namespace

void CellFluxEngine::localize(const CellMesh& m, const std::vector<int>& leaf_map, const std::vector<int>& face_map) {
    const int old_n = nleaf_, new_n = m.num_leaves();
    if (static_cast<int>(leaf_map.size()) != old_n || face_map.size() != stencil_.size() || lface_.empty())
        throw std::logic_error("CellFluxEngine::localize: the maps do not match the build / the partition");
    const int nmean = static_cast<int>(vroot_.size()), nip = static_cast<int>(interp_.size());
    // virtual cells: the mean-type ones whose leaves are all kept, then the interpolated ones whose fine leaf, coarse leaf and
    // the codes the coarse leaf's slopes read are all kept; vcode = the new code of each old virtual cell (kOutside: gone)
    std::vector<int> vcode(static_cast<size_t>(nmean + nip), kOutside);
    std::vector<int> new_vroot;
    std::vector<std::array<int, 8>> new_vnode;
    std::vector<std::array<int, 4>> new_vkey;
    for (int v = 0; v < nmean; ++v)
        if (tree_kept(vnode_, nch_, leaf_map, vroot_[static_cast<size_t>(v)])) {
            vcode[static_cast<size_t>(v)] = new_n + static_cast<int>(new_vroot.size());
            new_vroot.push_back(copy_tree(vnode_, nch_, leaf_map, new_vnode, vroot_[static_cast<size_t>(v)]));
            new_vkey.push_back(vkey_[static_cast<size_t>(v)]);
        }
    const int new_nmean = static_cast<int>(new_vroot.size());
    const auto kept_code = [&](int code) {   // an old code of a leaf or a mean-type virtual cell
        if (code == kOutside) return false;
        if (code < 0) return true;
        return code < old_n ? leaf_map[static_cast<size_t>(code)] >= 0 : vcode[static_cast<size_t>(code - old_n)] != kOutside;
    };
    std::vector<InterpVirtual> new_interp;
    std::vector<std::array<int, 5>> new_ikey;
    for (int k = 0; k < nip; ++k) {
        const InterpVirtual& iv = interp_[static_cast<size_t>(k)];
        bool ok = leaf_map[static_cast<size_t>(iv.fine)] >= 0 && leaf_map[static_cast<size_t>(iv.coarse)] >= 0;
        for (int s = 0; ok && s < ns_; ++s) ok = kept_code(nbr_raw_[static_cast<size_t>(ns_) * iv.coarse + s]);
        if (!ok) continue;
        vcode[static_cast<size_t>(nmean + k)] = new_n + new_nmean + static_cast<int>(new_interp.size());
        InterpVirtual n = iv;
        n.fine = leaf_map[static_cast<size_t>(iv.fine)];
        n.coarse = leaf_map[static_cast<size_t>(iv.coarse)];
        new_interp.push_back(n);
        new_ikey.push_back(ikey_[static_cast<size_t>(k)]);
    }
    const auto remap = [&](int code) {
        if (code < 0) return code;   // the edge of the domain (-1), or kOutside
        if (code < old_n) return leaf_map[static_cast<size_t>(code)] >= 0 ? leaf_map[static_cast<size_t>(code)] : kOutside;
        return vcode[static_cast<size_t>(code - old_n)];
    };
    // tables per leaf
    std::vector<int> nraw(static_cast<size_t>(ns_) * new_n), nnbr(static_cast<size_t>(ns_) * new_n);
    std::vector<char> nspecial(static_cast<size_t>(new_n), 0), nsolid, nowned(static_cast<size_t>(new_n), 0),
        nghost(static_cast<size_t>(new_n), 0);
    Field nsrc;
    const bool has_src = src_.size() == static_cast<size_t>(old_n);
    if (!solid_.empty()) nsolid.resize(static_cast<size_t>(new_n));
    if (has_src) nsrc.resize(static_cast<size_t>(new_n));
    for (int c = 0; c < old_n; ++c) {
        const int k = leaf_map[static_cast<size_t>(c)];
        if (k < 0) continue;
        for (int s = 0; s < ns_; ++s) {
            nraw[static_cast<size_t>(ns_) * k + s] = remap(nbr_raw_[static_cast<size_t>(ns_) * c + s]);
            nnbr[static_cast<size_t>(ns_) * k + s] = remap(nbr_[static_cast<size_t>(ns_) * c + s]);
        }
        nspecial[static_cast<size_t>(k)] = special_[static_cast<size_t>(c)];
        if (!nsolid.empty()) nsolid[static_cast<size_t>(k)] = solid_[static_cast<size_t>(c)];
        if (has_src) nsrc[static_cast<size_t>(k)] = src_[static_cast<size_t>(c)];
        nowned[static_cast<size_t>(k)] = owned_mask_.empty() || owned_mask_[static_cast<size_t>(c)];
        if (!ghost_leaf_mask_.empty()) nghost[static_cast<size_t>(k)] = ghost_leaf_mask_[static_cast<size_t>(c)];
    }
    // stencils of the kept faces
    std::vector<std::array<int, 4>> nst(m.faces.size());
    for (size_t f = 0; f < face_map.size(); ++f) {
        if (face_map[f] < 0) continue;
        for (size_t q = 0; q < 4; ++q) nst[static_cast<size_t>(face_map[f])][q] = remap(stencil_[f][q]);
    }
    // the lists of the partition
    const size_t L = static_cast<size_t>(m.max_level) + 1;
    for (size_t l = 0; l < L; ++l) {
        for (std::vector<int>* v : {&lheld_[l], &vcells_[l]})
            for (int& c : *v) {
                c = leaf_map[static_cast<size_t>(c)];
                if (c < 0) throw std::logic_error("CellFluxEngine::localize: a listed leaf is not kept");
            }
        for (int& f : lface_[l]) {
            f = face_map[static_cast<size_t>(f)];
            if (f < 0) throw std::logic_error("CellFluxEngine::localize: a face this rank computes is not kept");
        }
        for (int& code : need_[l]) {
            code = remap(code);
            if (code < 0) throw std::logic_error("CellFluxEngine::localize: a face this rank computes reads a cell that is not kept");
        }
    }
    for (int s = 0; s < 6; ++s)
        for (int& f : mf_face_[s]) {
            f = face_map[static_cast<size_t>(f)];
            if (f < 0) throw std::logic_error("CellFluxEngine::localize: a mass-flow face is not kept");
        }
    // the ghost-cell wall: finest-level cells -> kept leaves (the table of the cells stays whole, like the IBM mask of each level)
    std::unordered_map<int, int> nfine;
    for (const auto& e : fine_leaf_)
        if (leaf_map[static_cast<size_t>(e.second)] >= 0) nfine[e.first] = leaf_map[static_cast<size_t>(e.second)];
    fine_leaf_.swap(nfine);
    std::vector<std::pair<int, int>> nreads;
    for (const auto& pr : ghost_reads_leaf_)
        if (leaf_map[static_cast<size_t>(pr.first)] >= 0 && leaf_map[static_cast<size_t>(pr.second)] >= 0)
            nreads.push_back({leaf_map[static_cast<size_t>(pr.first)], leaf_map[static_cast<size_t>(pr.second)]});
    ghost_reads_leaf_.swap(nreads);
    // commit
    nleaf_ = new_n;
    nbr_raw_.swap(nraw);
    nbr_.swap(nnbr);
    stencil_.swap(nst);
    special_.swap(nspecial);
    solid_.swap(nsolid);
    if (has_src) src_.swap(nsrc);
    else Field().swap(src_);
    ghost_leaf_mask_.swap(nghost);
    std::vector<char>().swap(ghost_incomplete_);
    vroot_.swap(new_vroot);
    vnode_.swap(new_vnode);
    vkey_.swap(new_vkey);
    interp_.swap(new_interp);
    ikey_.swap(new_ikey);
    owned_mask_.swap(nowned);
    held_mask_.assign(static_cast<size_t>(new_n), 1);
    all_faces_.resize(m.faces.size());
    for (size_t f = 0; f < m.faces.size(); ++f) all_faces_[f] = static_cast<int>(f);
    // the owned range of every level must be one run of the local leaves
    for (size_t l = 0; l < L; ++l) {
        const int lv = static_cast<int>(l);
        int b = m.level_begin[lv], e = b;
        bool first = true;
        for (int c = m.level_begin[lv]; c < m.level_begin[lv + 1]; ++c)
            if (owned_mask_[static_cast<size_t>(c)]) {
                if (first) b = c;
                else if (e != c) throw std::logic_error("CellFluxEngine::localize: the owned leaves of a level are not contiguous");
                first = false;
                e = c + 1;
            }
        lown_b_[l] = b;
        lown_e_[l] = e;
    }
    for (std::vector<int>* a : {&nbr_raw_spare_, &nbr_spare_}) std::vector<int>().swap(*a);
    std::vector<std::array<int, 4>>().swap(stencil_spare_);
    for (Field* a : {&W_, &F_, &Fv_, &Vc_, &Wint_, &forced_}) Field().swap(*a);
    if (has_body()) forced_.resize(static_cast<size_t>(new_n));
    size_work_arrays(m);
    built_mesh_ = &m;
    built_version_ = m.version;
    // every read of every face this rank computes is a kept leaf or a kept virtual cell (kOutside throws in mark_code)
    std::vector<int> reads;
    for (size_t l = 0; l < L; ++l)
        for (const int f : lface_[l]) {
            reads.clear();
            face_dependencies(m, f, reads);
            for (const int c : reads)
                if (c < 0 || c >= new_n) throw std::logic_error("CellFluxEngine::localize: a face reads a leaf that is not kept");
        }
}

void CellFluxEngine::clear_partition(const CellMesh& m) { rebuild_lists(m, nullptr); }

template <class Mark>
void CellFluxEngine::mark_tree(int code, const Mark& mark) const {
    if (code == kIncomplete) throw std::logic_error("CellFluxEngine: a face reads a virtual cell that is not whole in this rank's partial mesh");
    if (code >= 0) {
        mark(code);
        return;
    }
    const std::array<int, 8>& ch = vnode_[static_cast<size_t>(-1 - code)];
    for (int k = 0; k < nch_; ++k) mark_tree(ch[static_cast<size_t>(k)], mark);
}

template <class Mark>
void CellFluxEngine::mark_code(int code, const Mark& mark) const {
    if (code == kOutside) throw std::logic_error("CellFluxEngine: a face reads a leaf that is not in this rank's partial mesh");
    if (code < 0) return;
    if (code < nleaf_) {
        mark(code);
        return;
    }
    const size_t v = static_cast<size_t>(code - nleaf_);
    if (v < vroot_.size()) {
        mark_tree(vroot_[v], mark);
        return;
    }
    const InterpVirtual& iv = interp_[v - vroot_.size()];
    mark(iv.coarse);
    for (int s = 0; s < ns_; ++s) mark_code(nbr_raw_[static_cast<size_t>(ns_) * iv.coarse + s], mark);
}

template <class Mark>
void CellFluxEngine::mark_face(const CellMesh& m, int f, const Mark& mark) const {
    const CellFace& face = m.faces[static_cast<size_t>(f)];
    const auto cell = [&](int c) {
        mark(c);
        for (int s = 0; s < ns_; ++s) mark_code(nbr_[static_cast<size_t>(ns_) * c + s], mark);   // gradients, tangential differences
    };
    if (face.kind == CellFaceKind::Boundary) {
        cell(face.left >= 0 ? face.left : face.right);
        return;
    }
    for (int k = 0; k < 4; ++k) mark_code(stencil_[static_cast<size_t>(f)][k], mark);
    cell(face.left);
    cell(face.right);
    if (face.kind == CellFaceKind::CoarseFine) {   // the coarse leaf's own neighbours (second-order coarse side)
        const int C = m.level[face.left] > m.level[face.right] ? face.right : face.left;
        for (int s = 0; s < ns_; ++s) mark_code(nbr_raw_[static_cast<size_t>(ns_) * C + s], mark);
    }
}

void CellFluxEngine::memory_bytes(double& per_leaf, double& uniform) const {
    per_leaf = container_bytes(stencil_) + container_bytes(nbr_) + container_bytes(vroot_) + container_bytes(vnode_) +
               container_bytes(interp_) + container_bytes(Wint_) + container_bytes(nbr_raw_) + container_bytes(nbr_raw_spare_) +
               container_bytes(nbr_spare_) + container_bytes(stencil_spare_) + container_bytes(vkey_) + container_bytes(ikey_) +
               container_bytes(special_) + container_bytes(lface_) + container_bytes(lheld_) + container_bytes(lown_b_) +
               container_bytes(lown_e_) + container_bytes(all_faces_) + container_bytes(owned_mask_) + container_bytes(forced_) +
               container_bytes(Vc_) + container_bytes(W_) + container_bytes(F_) + container_bytes(Fv_) + container_bytes(reg_) +
               container_bytes(regv_) + container_bytes(need_) + container_bytes(vcells_) + container_bytes(su_) +
               container_bytes(sv_) + container_bytes(sT_) + container_bytes(dudx_) + container_bytes(dudy_) + container_bytes(dvdx_) +
               container_bytes(dvdy_) + container_bytes(sw_) + container_bytes(dudz_) + container_bytes(dvdz_) + container_bytes(dwdx_) +
               container_bytes(dwdy_) + container_bytes(dwdz_) + container_bytes(solid_) + container_bytes(fine_leaf_) + container_bytes(ghost_reads_leaf_) +
               container_bytes(ghost_leaf_mask_) + container_bytes(held_mask_) + container_bytes(src_);
    for (int s = 0; s < 6; ++s) per_leaf += container_bytes(mf_face_[s]) + container_bytes(mf_weight_[s]);
    uniform = container_bytes(level_mask_) + (ghost_ib_ ? ghost_ib_->memory_bytes() : 0.0) + (ibm_geo_ ? ibm_geo_->memory_bytes() : 0.0);
}

void CellFluxEngine::face_dependencies(const CellMesh& m, int f, std::vector<char>& mask) const {
    mark_face(m, f, [&](int c) { mask[static_cast<size_t>(c)] = 1; });
}

void CellFluxEngine::face_dependencies(const CellMesh& m, int f, std::vector<int>& out) const {
    mark_face(m, f, [&](int c) { out.push_back(c); });
}

void CellFluxEngine::mass_flow_dependencies(const CellMesh& m, std::vector<char>& mask) const {
    for (int s = 0; s < 6; ++s)
        for (const int f : mf_face_[s]) mask[static_cast<size_t>(m.faces[static_cast<size_t>(f)].left >= 0 ? m.faces[static_cast<size_t>(f)].left : m.faces[static_cast<size_t>(f)].right)] = 1;
}

void CellFluxEngine::update_source(const CellMesh& m, int cb, int ce) {
    if (!phys_.source) return;
    std::vector<double> xs(static_cast<size_t>(ce - cb)), ys(xs.size()), zs(m.dim == 3 ? xs.size() : 0);
    for (int c = cb; c < ce; ++c) {
        xs[static_cast<size_t>(c - cb)] = m.xc(c);
        ys[static_cast<size_t>(c - cb)] = m.yc(c);
        if (m.dim == 3) zs[static_cast<size_t>(c - cb)] = m.zc(c);
    }
    evaluate_source_points(*phys_.source_mesh, *phys_.source, phys_.source_unsteady ? stage_time : 0.0, xs.data(),
                           ys.data(), ce - cb, &src_[static_cast<size_t>(cb)], m.dim == 3 ? zs.data() : nullptr);
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
    }    if (ghost_ready_) fill_ghost_leaves(U, forced_, nullptr);
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
    if (ghost_ready_ && l == m.max_level) fill_ghost_leaves(U, forced_, held_mask_.empty() ? nullptr : &held_mask_);
}

std::vector<int> CellFluxEngine::ghost_referenced_cells(const CellMesh& m) {
    if (!has_body() || !phys_.ibm_ghost) return {};
    if (!ghost_ib_) {
        const int M = m.max_level;
        const Mesh fine = generate_cartesian_mesh(m.Lx, m.Ly, m.nx_at(M), m.ny_at(M), m.x0, m.y0);
        ghost_ib_ = std::make_shared<ImmersedBoundary>(phys_.ibm_factory(fine));
        ghost_ib_->enable_ghost_cells(fine, phys_.ibm_slip, phys_.ibm_curvature);
    }
    const auto& cells = ghost_ib_->referenced_cells();
    return std::vector<int>(cells.begin(), cells.end());
}

void CellFluxEngine::map_ghost_cells(const CellMesh& m, const CellMesh* whole) {
    ghost_ready_ = false;
    fine_leaf_.clear();
    ghost_reads_leaf_.clear();
    ghost_incomplete_.clear();
    ghost_leaf_mask_.assign(static_cast<size_t>(nleaf_), 0);
    if (!has_body() || !phys_.ibm_ghost) return;
    if (m.dim == 3) {
        map_ghost_cells_3d(m, whole);
        return;
    }
    const int M = m.max_level, NX = m.nx_at(M);
    ghost_referenced_cells(m);
    // whether the wall is active is a fact of the whole mesh (given, or looked up in `whole`); a rank keeps the cells whose leaf it has
    const bool known = ghost_whole_ready_ >= 0;
    if (known && ghost_whole_ready_ == 0) return;   // not (yet) all finest-level leaves: the mask forcing stays
    const CellMesh& all = whole ? *whole : m;
    for (const int fc : ghost_ib_->referenced_cells()) {
        if (!known && all.find(M, fc % NX, fc / NX) < 0) return;   // not (yet) a finest-level leaf: the mask forcing stays
        const int leaf = m.find(M, fc % NX, fc / NX);
        if (leaf >= 0) fine_leaf_[fc] = leaf;
    }
    for (const int g : ghost_ib_->ghost_cell_list()) {
        const auto it = fine_leaf_.find(g);
        if (it != fine_leaf_.end()) ghost_leaf_mask_[static_cast<size_t>(it->second)] = 1;
    }
    for (const auto& pr : ghost_ib_->ghost_reads()) {
        const auto a = fine_leaf_.find(pr.first), b = fine_leaf_.find(pr.second);
        if (a != fine_leaf_.end() && b != fine_leaf_.end()) {
            ghost_reads_leaf_.push_back({a->second, b->second});
        } else if (a != fine_leaf_.end()) {
            ghost_incomplete_.resize(static_cast<size_t>(nleaf_), 0);
            ghost_incomplete_[static_cast<size_t>(a->second)] = 1;
        }
    }
    ghost_ready_ = true;
}

bool near_body_surface(const ImmersedBoundary& body, const CellMesh& m, int level, int i, int j, int k, double reach) {
    const int M = m.max_level;
    const double dx = m.dx(level), dy = m.dy(level), dz = m.dz(level);
    const double x = (i + 0.5) * dx + m.x0, y = (j + 0.5) * dy + m.y0, z = (k + 0.5) * dz + m.z0;
    double wx, wy, wz, nx, ny, nz;
    body.closest_wall_point3(x, y, z, wx, wy, wz, nx, ny, nz);
    const double d = std::sqrt((x - wx) * (x - wx) + (y - wy) * (y - wy) + (z - wz) * (z - wz));
    if (level == M) return d <= reach;
    // the finest centres inside lie within (half diagonal of the cell - half diagonal of a finest cell) of its centre
    const double span = 0.5 * std::sqrt(dx * dx + dy * dy + dz * dz) -
                        0.5 * std::sqrt(m.dx(M) * m.dx(M) + m.dy(M) * m.dy(M) + m.dz(M) * m.dz(M));
    if (d - span > reach) return false;
    for (int q = 0; q < 8; ++q)
        if (near_body_surface(body, m, level + 1, 2 * i + (q & 1), 2 * j + ((q >> 1) & 1), 2 * k + ((q >> 2) & 1), reach)) return true;
    return false;
}

bool CellFluxEngine::ghost_band_violated(const CellMesh& m) const {
    if (!has_body() || !phys_.ibm_ghost) return false;
    if (!ibm_geo_) throw std::logic_error("CellFluxEngine::ghost_band_violated: no engine build has made the body's geometry yet");
    const int M = m.max_level;
    const double reach = 6.0 * std::max({m.dx(M), m.dy(M), m.dz(M)});
    for (int c = 0; c < m.num_leaves(); ++c)
        if (m.level[static_cast<size_t>(c)] < M && near_body_surface(*ibm_geo_, m, m.level[static_cast<size_t>(c)], m.ci[static_cast<size_t>(c)], m.cj[static_cast<size_t>(c)], m.ck[static_cast<size_t>(c)], reach))
            return true;
    return false;
}

void CellFluxEngine::map_ghost_cells_3d(const CellMesh& m, const CellMesh* whole) {
    // The ghost-cell wall on octree leaves (docs/design/amr-octree.md 15.6): the table of the uniform finest level built sparsely
    // from the solid finest leaves, active when it is that table bit for bit and every cell it reads is a finest leaf.
    const int M = m.max_level, NX = m.nx_at(M), NY = m.ny_at(M), NZ = m.nz_at(M);
    const double hf[3] = {m.dx(M), m.dy(M), m.dz(M)};
    const ImmersedBoundary& geo = *ibm_geo_;
    // 1) no coarser leaf near the surface: a ghost cell (solid, a fluid cell within two finest cells) or a cell its image point reads
    // could lie in it. Every cell the wall reads has its centre within 6 finest widths of the wall (a ghost cell within 2 sqrt 3, an
    // image point at most 4 cells out and its interpolation cells within sqrt 3 of it): a coarser leaf must hold no finest cell whose
    // centre is that near (near_body_surface, the rule of the body band with margin 6).
    // A rank's share (the readiness a collective fact of the whole mesh, given as `ghost_whole_ready_`, or looked up in `whole`) keeps
    // the table entries of the leaves it has; a ghost leaf whose image leaves it lacks is marked incomplete (as in 2-D).
    const bool known = ghost_whole_ready_ >= 0, part = known || whole;
    if (known && ghost_whole_ready_ == 0) return;   // not (yet) refined: the mask stays
    if (!known) {
        const CellMesh& all = whole ? *whole : m;
        const double reach = 6.0 * std::max({hf[0], hf[1], hf[2]});
        for (int c = 0; c < all.level_begin[M]; ++c)
            if (near_body_surface(geo, all, all.level[c], all.ci[c], all.cj[c], all.ck[c], reach)) return;   // not (yet) refined: the mask stays
    }
    // 2) the table from the solid finest leaves, in cell order of the finest level
    if (static_cast<long long>(NX) * NY * NZ > 2147483647LL) return;
    std::vector<int> cand;
    for (int c = m.level_begin[M]; c < m.level_begin[M + 1]; ++c)
        if (solid_[c]) cand.push_back((m.ck[c] * NY + m.cj[c]) * NX + m.ci[c]);
    std::sort(cand.begin(), cand.end());
    const Mesh fine = generate_cartesian_mesh_3d(m.Lx, m.Ly, m.Lz, NX, NY, NZ, m.x0, m.y0, m.z0);
    auto ib = std::make_shared<ImmersedBoundary>(geo);
    ib->enable_ghost_cells_sparse(fine, phys_.ibm_slip, phys_.ibm_curvature, cand);
    ghost_ib_ = ib;
    // 3) every cell the wall reads is a finest leaf
    for (const int fc : ghost_ib_->referenced_cells()) {
        const int leaf = m.find(M, fc % NX, (fc / NX) % NY, fc / (NX * NY));
        if (leaf < 0) {
            if (part) continue;   // a rank's share does not hold every leaf
            fine_leaf_.clear();
            return;
        }
        fine_leaf_[fc] = leaf;
    }
    for (const int g : ghost_ib_->ghost_cell_list()) {
        const auto it = fine_leaf_.find(g);
        if (it != fine_leaf_.end()) ghost_leaf_mask_[static_cast<size_t>(it->second)] = 1;
    }
    for (const auto& pr : ghost_ib_->ghost_reads()) {
        const auto a = fine_leaf_.find(pr.first), b = fine_leaf_.find(pr.second);
        if (a != fine_leaf_.end() && b != fine_leaf_.end()) {
            ghost_reads_leaf_.push_back({a->second, b->second});
        } else if (a != fine_leaf_.end()) {
            ghost_incomplete_.resize(static_cast<size_t>(nleaf_), 0);
            ghost_incomplete_[static_cast<size_t>(a->second)] = 1;
        }
    }
    ghost_ready_ = true;
}

void CellFluxEngine::fill_ghost_leaves(const Field& U, Field& out, const std::vector<char>* only) const {
    // the ghost leaves (of `only`, if given) from the fluid leaves their image points read; same bits as the uniform solver
    // In a rank's partial mesh fine_leaf_ holds only the kept leaves: a cell that is not there is not filled (the first function),
    // and one read as an image point is NaN (as the state of a leaf outside the halo is in a replicated run with poisoned halos)
    static const State nan_state = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN(),
                                    std::numeric_limits<double>::quiet_NaN()};
    ghost_ib_->fill_ghost_cells(
        [&](int fc) {
            const auto it = fine_leaf_.find(fc);
            if (it == fine_leaf_.end()) return -1;
            const int leaf = it->second;
            return only && !(*only)[static_cast<size_t>(leaf)] ? -1 : leaf;
        },
        [&](int fc) -> const State& {
            const auto it = fine_leaf_.find(fc);
            return it == fine_leaf_.end() ? nan_state : U[static_cast<size_t>(it->second)];
        },
        out, eos_);
}

std::array<double, 3> CellFluxEngine::ghost_loads(const Field& U, double mu, double cx, double cy) const {
    if (!ghost_ready_) return {0.0, 0.0, 0.0};
    return ghost_ib_->surface_loads_part(0, 1, [&](int fc) -> const State& { return U[static_cast<size_t>(fine_leaf_.at(fc))]; },
                                         eos_, mu, cx, cy);
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
    double u_target[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // mass-flow inlets: the ghost of a coarse leaf on such a side
    if (!interp_.empty() && muscl_ && bc_.params().boundary_order == 2) mass_flow_targets(m, u_target);
    eval_interp(u_target);
    if (muscl_) {
        if (level < 0)
            refresh_all(nleaf_);
        else
            refresh_some(nleaf_, need_[static_cast<size_t>(level)]);
    }
}

void CellFluxEngine::eval_interp(const double* u_target) {
    if (interp_.empty()) return;
    const EOS& eos = eos_;
    const bool bo2 = bc_.params().boundary_order == 2;
    const Limiter lim = muscl_->limiter();
    const size_t nm = vroot_.size();
    Wint_.resize(interp_.size());
    for (size_t k = 0; k < interp_.size(); ++k) {
        const InterpVirtual& iv = interp_[k];
        const int C = iv.coarse;
        const State Wc = eos.cons_to_prim((*us_)[C]);
        // a side of the coarse leaf on a physical boundary: its own state (first order), or with boundary_order = 2 its ghost layer
        const auto prim_at = [&](int code, int side) {
            if (code >= 0) return eos.cons_to_prim(cell_state(code));
            const Face fs = static_cast<Face>(side);
            if (bo2 && bc_.type(fs) != BCType::Periodic && bc_.type(fs) != BCType::NonReflecting)
                return eos.cons_to_prim(bc_.ghost_state(fs, (*us_)[C], u_target[side]));
            return Wc;
        };
        // limited slopes of the coarse leaf: along the normal, the first transverse axis and (3-D) the second one
        State sl3[3] = {};
        for (int pass = 0; pass < dim_; ++pass) {
            const int ax = pass == 0 ? iv.axis : (pass == 1 ? transverse1(iv.axis) : transverse2(iv.axis));
            const State Wb = prim_at(nbr_raw_[static_cast<size_t>(ns_) * C + 2 * ax], 2 * ax);
            const State Wf = prim_at(nbr_raw_[static_cast<size_t>(ns_) * C + 2 * ax + 1], 2 * ax + 1);
            State& sl = sl3[pass];
            for (int q = 0; q < kNVar; ++q) sl[q] = apply_limiter(lim, Wc[q] - Wb[q], Wf[q] - Wc[q]);
        }
        const State &sn = sl3[0], &st = sl3[1], &st2 = sl3[2];
        State w;
        if (dim_ == 2)
            for (int q = 0; q < kNVar; ++q) w[q] = Wc[q] + iv.dn * sn[q] + iv.dt * st[q];
        else
            for (int q = 0; q < kNVar; ++q) w[q] = Wc[q] + iv.dn * sn[q] + iv.dt * st[q] + iv.dt2 * st2[q];
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
    for (int s = 0; s < 6; ++s) {
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

}   // namespace fv
