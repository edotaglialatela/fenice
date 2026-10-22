#include "fv/AMR.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>

#include "fv/SolutionMapping.hpp"

namespace fv {

bool g_amr_disable_reflux = false;

// ---------------------------------------------------------------------------
// AMRTree
// ---------------------------------------------------------------------------

AMRTree::AMRTree(double Lx_, double Ly_, int nxp, int nyp, int max_lvl, int nrx, int nry)
    : Lx(Lx_), Ly(Ly_), nx_patch(nxp), ny_patch(nyp), max_level(max_lvl), nroot_x(nrx), nroot_y(nry) {
    // Children have the same cell count over half the extent, so 2 fine cells
    // must tile 1 coarse cell: odd dimensions break conservation.
    if (max_level > 0 && (nxp % 2 != 0 || nyp % 2 != 0))
        throw std::invalid_argument(
            "AMR patch dimensions must be EVEN for refinement to align 2:1 (got nx_patch=" + std::to_string(nxp) +
            ", ny_patch=" + std::to_string(nyp) + "). Odd dimensions break conservation.");
    if (nrx < 1 || nry < 1) throw std::invalid_argument("AMR needs at least one root patch per axis");
    nslot_x_ = nrx << max_level;
    nslot_y_ = nry << max_level;
    slot_w_ = Lx / nslot_x_;
    slot_h_ = Ly / nslot_y_;
    for (int j = 0; j < nry; ++j)
        for (int i = 0; i < nrx; ++i)
            roots.push_back(new_patch(Lx * i / nrx, Lx * (i + 1) / nrx, Ly * j / nry, Ly * (j + 1) / nry, 0, nullptr));
    root = roots[0];
}

AMRPatch* AMRTree::new_patch(double x0, double x1, double y0, double y1, int level, AMRPatch* parent) {
    auto p = std::make_unique<AMRPatch>();
    p->xmin = x0;
    p->xmax = x1;
    p->ymin = y0;
    p->ymax = y1;
    p->level = level;
    p->parent = parent;
    p->id = next_id_++;
    p->mesh = generate_cartesian_mesh(x1 - x0, y1 - y0, nx_patch, ny_patch, x0, y0);
    // Cell size from the base grid by exact halving, so tiled patches share the uniform mesh's dx
    p->mesh.dx = std::ldexp(Lx / (nroot_x * nx_patch), -level);
    p->mesh.dy = std::ldexp(Ly / (nroot_y * ny_patch), -level);
    if (store_state) p->U.assign(p->mesh.num_cells(), State{0, 0, 0, 0});
    AMRPatch* raw = p.get();
    patches_[raw->id] = std::move(p);
    dirty_ = true;
    return raw;
}

void AMRTree::collect_leaves(AMRPatch* p, std::vector<AMRPatch*>& out) {
    if (p->is_leaf()) {
        out.push_back(p);
        return;
    }
    for (AMRPatch* c : p->children) collect_leaves(c, out);
}

void AMRTree::slots_of(const AMRPatch* p, int& i0, int& i1, int& j0, int& j1) const {
    i0 = static_cast<int>(std::lround(p->xmin / slot_w_));
    i1 = static_cast<int>(std::lround(p->xmax / slot_w_));
    j0 = static_cast<int>(std::lround(p->ymin / slot_h_));
    j1 = static_cast<int>(std::lround(p->ymax / slot_h_));
}

void AMRTree::rebuild_index() const {
    leaf_cache_.clear();
    for (AMRPatch* r : roots) collect_leaves(r, leaf_cache_);
    slot_.assign(static_cast<size_t>(nslot_x_) * nslot_y_, nullptr);
    for (AMRPatch* lf : leaf_cache_) {
        int i0, i1, j0, j1;
        slots_of(lf, i0, i1, j0, j1);
        for (int j = j0; j < j1; ++j)
            for (int i = i0; i < i1; ++i) slot_[static_cast<size_t>(j) * nslot_x_ + i] = lf;
    }
    dirty_ = false;
}

std::vector<AMRPatch*> AMRTree::leaves() const {
    if (dirty_) rebuild_index();
    return leaf_cache_;
}

AMRPatch* AMRTree::leaf_at(double x, double y) const {
    constexpr double eps = 1e-9;
    if (dirty_) rebuild_index();
    if (x < -eps || x > Lx + eps || y < -eps || y > Ly + eps) return nullptr;
    const int i = std::min(std::max(static_cast<int>(x / slot_w_), 0), nslot_x_ - 1);
    const int j = std::min(std::max(static_cast<int>(y / slot_h_), 0), nslot_y_ - 1);
    return slot_[static_cast<size_t>(j) * nslot_x_ + i];
}

std::vector<AMRPatch*> AMRTree::all_patches() const {
    std::vector<AMRPatch*> out;
    out.reserve(patches_.size());
    for (const auto& kv : patches_) out.push_back(kv.second.get());
    return out;
}

std::vector<AMRPatch*> AMRTree::refine_patch(AMRPatch* patch) {
    if (!patch->is_leaf()) throw std::invalid_argument("Cannot refine non-leaf patch");
    if (patch->level >= max_level) throw std::invalid_argument("Patch already at max_level");

    const double xm = 0.5 * (patch->xmin + patch->xmax);
    const double ym = 0.5 * (patch->ymin + patch->ymax);
    const double quads[4][4] = {
        {patch->xmin, xm, patch->ymin, ym},
        {xm, patch->xmax, patch->ymin, ym},
        {patch->xmin, xm, ym, patch->ymax},
        {xm, patch->xmax, ym, patch->ymax},
    };

    // Conservative piecewise-constant injection: each child cell copies the
    // parent cell containing its centre.
    const Mesh& mp = patch->mesh;
    const double dx_p = mp.dx, dy_p = mp.dy;
    const double x0_p = mp.xc(0) - 0.5 * dx_p;
    const double y0_p = mp.yc(0) - 0.5 * dy_p;

    std::vector<AMRPatch*> kids;
    for (const auto& q : quads) {
        AMRPatch* child = new_patch(q[0], q[1], q[2], q[3], patch->level + 1, patch);
        const Mesh& mc = child->mesh;
        if (patch->U.empty()) {   // data-less rank: only the tree changes
            patch->children.push_back(child);
            kids.push_back(child);
            continue;
        }
        if (child->U.empty()) child->U.assign(mc.num_cells(), State{0, 0, 0, 0});
        for (int j = 0; j < mc.ny; ++j) {
            const int jp = std::min(std::max(static_cast<int>((mc.yc(j) - y0_p) / dy_p), 0), mp.ny - 1);
            for (int i = 0; i < mc.nx; ++i) {
                const int ip = std::min(std::max(static_cast<int>((mc.xc(i) - x0_p) / dx_p), 0), mp.nx - 1);
                child->U[j * mc.nx + i] = patch->U[jp * mp.nx + ip];
            }
        }
        patch->children.push_back(child);
        kids.push_back(child);
    }
    Field().swap(patch->U);   // an interior patch has no state of its own: derefine_patch rebuilds it
    return kids;
}

void AMRTree::derefine_patch(AMRPatch* patch) {
    if (patch->is_leaf()) throw std::invalid_argument("Patch has no children");
    for (AMRPatch* c : patch->children)
        if (!c->is_leaf()) throw std::invalid_argument("Can only derefine if all children are leaves");

    bool all_data = true;
    for (AMRPatch* c : patch->children) all_data = all_data && !c->U.empty();
    if (!all_data) {   // data-less rank: only the tree changes
        for (AMRPatch* c : patch->children) patches_.erase(c->id);
        patch->children.clear();
        dirty_ = true;
        return;
    }
    // Conservative area-weighted average of the 2x2 fine cells per coarse cell.
    const Mesh& mp = patch->mesh;
    const int nx_p = mp.nx, ny_p = mp.ny;
    Field Uc(static_cast<size_t>(nx_p) * ny_p, State{0, 0, 0, 0});
    for (AMRPatch* child : patch->children) {
        const double dx_p = mp.dx, dy_p = mp.dy;
        const double x_off = mp.xc(0) - 0.5 * mp.dx;
        const double y_off = mp.yc(0) - 0.5 * mp.dy;
        const int i_off = static_cast<int>(std::lround((child->xmin - x_off) / dx_p));
        const int j_off = static_cast<int>(std::lround((child->ymin - y_off) / dy_p));
        for (int jc = 0; jc < child->ny(); ++jc)
            for (int ic = 0; ic < child->nx(); ++ic) {
                const int jp = j_off + jc / 2, ip = i_off + ic / 2;
                if (jp >= 0 && jp < ny_p && ip >= 0 && ip < nx_p)
                    for (int q = 0; q < kNVar; ++q) Uc[jp * nx_p + ip][q] += 0.25 * child->U[jc * child->nx() + ic][q];
            }
    }
    patch->U = std::move(Uc);
    for (AMRPatch* c : patch->children) patches_.erase(c->id);
    patch->children.clear();
    dirty_ = true;
}

void AMRTree::enforce_2to1_balance() {
    bool changed = true;
    while (changed) {
        changed = false;
        const auto snapshot = leaves();
        for (AMRPatch* patch : snapshot) {
            AMRPatch* nbs[4];
            for (int f = 0; f < 4; ++f) nbs[f] = get_neighbour(patch, kAllFaces[f]);
            for (AMRPatch* nb : nbs) {
                if (nb && nb->is_leaf() && patch->level - nb->level > 1 && nb->level < max_level) {
                    refine_patch(nb);
                    changed = true;
                }
            }
        }
    }
}

AMRPatch* AMRTree::get_neighbour(const AMRPatch* patch, Face face) const {
    const auto nb = neighbour_leaves(patch, face);
    return nb.empty() ? nullptr : nb.front();
}

std::vector<AMRPatch*> AMRTree::neighbour_leaves(const AMRPatch* p, Face face) const {
    std::vector<AMRPatch*> out;
    if (dirty_) rebuild_index();
    int i0, i1, j0, j1;
    slots_of(p, i0, i1, j0, j1);
    // Slot range just outside the patch along `face`
    switch (face) {
    case Face::Left:
        if (i0 <= 0) return out;
        i1 = i0;
        i0 = i0 - 1;
        break;
    case Face::Right:
        if (i1 >= nslot_x_) return out;
        i0 = i1;
        i1 = i1 + 1;
        break;
    case Face::Bottom:
        if (j0 <= 0) return out;
        j1 = j0;
        j0 = j0 - 1;
        break;
    case Face::Top:
        if (j1 >= nslot_y_) return out;
        j0 = j1;
        j1 = j1 + 1;
        break;
    case Face::Front:
    case Face::Back: return out;   // AMR is 2-D only
    }
    for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {
            AMRPatch* nb = slot_[static_cast<size_t>(j) * nslot_x_ + i];
            if (nb && nb != p && (out.empty() || out.back() != nb)) out.push_back(nb);
        }
    return out;
}

std::vector<AMRPatch*> AMRTree::leaves_near(const AMRPatch* p, int cells) const {
    std::vector<AMRPatch*> out;
    if (dirty_) rebuild_index();
    constexpr double eps = 1e-9;
    const double gx = cells * p->mesh.dx, gy = cells * p->mesh.dy;
    const int i0 = std::max(static_cast<int>(std::floor((p->xmin - gx) / slot_w_ + eps)), 0);
    const int i1 = std::min(static_cast<int>(std::ceil((p->xmax + gx) / slot_w_ - eps)), nslot_x_);
    const int j0 = std::max(static_cast<int>(std::floor((p->ymin - gy) / slot_h_ + eps)), 0);
    const int j1 = std::min(static_cast<int>(std::ceil((p->ymax + gy) / slot_h_ - eps)), nslot_y_);
    std::set<const AMRPatch*> seen;
    for (int j = j0; j < j1; ++j)
        for (int i = i0; i < i1; ++i) {
            AMRPatch* nb = slot_[static_cast<size_t>(j) * nslot_x_ + i];
            if (nb && nb != p && seen.insert(nb).second) out.push_back(nb);
        }
    return out;
}

// ---------------------------------------------------------------------------
// Distribution
// ---------------------------------------------------------------------------

namespace {
// Interleave the bits of two slot coordinates (x in the even bits)
std::uint64_t morton_key(std::uint32_t x, std::uint32_t y) {
    auto spread = [](std::uint64_t v) {
        v = (v | (v << 16)) & 0x0000FFFF0000FFFFull;
        v = (v | (v << 8)) & 0x00FF00FF00FF00FFull;
        v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0Full;
        v = (v | (v << 2)) & 0x3333333333333333ull;
        v = (v | (v << 1)) & 0x5555555555555555ull;
        return v;
    };
    return spread(x) | (spread(y) << 1);
}
}   // namespace

void assign_ownership(AMRTree& tree, int rank, int size) {
    constexpr int kStencil = 2;   // cells of the owner's own level a residual reads across an interface
    const auto lv = tree.leaves();
    const double sw = tree.Lx / (tree.nroot_x << tree.max_level), sh = tree.Ly / (tree.nroot_y << tree.max_level);
    for (int level = 0; level <= tree.max_level; ++level) {
        std::vector<std::pair<std::uint64_t, AMRPatch*>> at;
        for (AMRPatch* p : lv)
            if (p->level == level)
                at.emplace_back(morton_key(static_cast<std::uint32_t>(std::lround(p->xmin / sw)),
                                           static_cast<std::uint32_t>(std::lround(p->ymin / sh))),
                                p);
        std::sort(at.begin(), at.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        const long long n = static_cast<long long>(at.size());
        for (long long k = 0; k < n; ++k) at[k].second->owner = static_cast<int>(k * size / n);
    }
    for (AMRPatch* p : lv) {
        p->viewers.clear();
        p->peers.clear();
    }
    auto add = [](std::vector<int>& v, int r) {
        const auto it = std::lower_bound(v.begin(), v.end(), r);
        if (it == v.end() || *it != r) v.insert(it, r);
    };
    // Every rank derives the halo sets of all ranks from the (replicated) tree: no communication
    for (AMRPatch* y : lv)
        for (AMRPatch* nb : tree.leaves_near(y, kStencil))
            if (nb->owner != y->owner) {
                add(nb->viewers, y->owner);
                if (nb->level == y->level) add(nb->peers, y->owner);
            }
    for (AMRPatch* p : lv) p->held = p->owner == rank || std::binary_search(p->viewers.begin(), p->viewers.end(), rank);
}

// ---------------------------------------------------------------------------
// Lohner second-derivative error indicator
// ---------------------------------------------------------------------------

namespace {
// Lohner ratio of the scalar `value(cell)` over a patch
template <class F> double lohner_eta(const AMRPatch& patch, F value) {
    const int nx = patch.nx(), ny = patch.ny();
    const int PW = nx + 2;
    // Anti-symmetric (odd) reflection: ghost = 2*edge - interior, so a linear
    // density ramp gives zero second difference.
    std::vector<double> rp(static_cast<size_t>(PW) * (ny + 2), 0.0);
    double rho_max = 0.0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double r = value(patch.U[j * nx + i]);
            rp[(j + 1) * PW + i + 1] = r;
            rho_max = std::max(rho_max, std::abs(r));
        }
    auto R = [&](int jj, int ii) -> double& { return rp[jj * PW + ii]; };
    for (int j = 1; j <= ny; ++j) {
        R(j, 0) = 2.0 * R(j, 1) - (nx > 1 ? R(j, 2) : R(j, 1));
        R(j, nx + 1) = 2.0 * R(j, nx) - (nx > 1 ? R(j, nx - 1) : R(j, nx));
    }
    for (int i = 1; i <= nx; ++i) {
        R(0, i) = 2.0 * R(1, i) - (ny > 1 ? R(2, i) : R(1, i));
        R(ny + 1, i) = 2.0 * R(ny, i) - (ny > 1 ? R(ny - 1, i) : R(ny, i));
    }
    const double floor = 0.01 * (rho_max + 1e-12);
    double eta = 0.0;
    for (int j = 1; j <= ny; ++j)
        for (int i = 1; i <= nx; ++i) {
            const double c = R(j, i);
            const double d2x = R(j, i + 1) - 2.0 * c + R(j, i - 1);
            const double d2y = R(j + 1, i) - 2.0 * c + R(j - 1, i);
            const double f1x = std::abs(R(j, i + 1) - c) + std::abs(c - R(j, i - 1));
            const double f1y = std::abs(R(j + 1, i) - c) + std::abs(c - R(j - 1, i));
            eta = std::max(eta, std::sqrt(d2x * d2x + d2y * d2y) / (f1x + f1y + floor));
        }
    return eta;
}
}   // namespace

double AMRErrorIndicator::indicator(const AMRPatch& patch) const {
    double eta = 0.0;
    if (use_density) eta = lohner_eta(patch, [](const State& c) { return c[0]; });
    if (use_speed)
        eta = std::max(eta, lohner_eta(patch, [](const State& c) {
                           return std::sqrt(c[1] * c[1] + c[2] * c[2] + c[4] * c[4]) / c[0];
                       }));
    return eta;
}

void AMRErrorIndicator::mark(AMRTree& tree, std::vector<AMRPatch*>& to_refine,
                             std::vector<AMRPatch*>& to_derefine) const {
    std::map<int, double> eta;
    for (AMRPatch* p : tree.leaves()) eta[p->id] = indicator(*p);
    mark(tree, eta, to_refine, to_derefine);
}

void AMRErrorIndicator::mark(AMRTree& tree, const std::map<int, double>& eta_in, std::vector<AMRPatch*>& to_refine,
                             std::vector<AMRPatch*>& to_derefine) const {
    to_refine.clear();
    to_derefine.clear();
    const auto lv = tree.leaves();
    const std::map<int, double>& eta = eta_in;

    std::set<int> forced;
    if (force_refine)
        for (AMRPatch* p : lv)
            if (force_refine(*p)) forced.insert(p->id);

    std::vector<AMRPatch*> flagged;
    for (AMRPatch* p : lv) {
        const double e = eta.at(p->id);
        if ((e > refine_threshold || forced.count(p->id)) && p->level < tree.max_level) {
            flagged.push_back(p);
        } else if (e < derefine_threshold && p->parent && p->level > 0) {
            // All siblings must be leaves below the threshold to collapse the parent.
            bool ok = true;
            for (AMRPatch* s : p->parent->children) {
                auto it = eta.find(s->id);
                if (!s->is_leaf() || (it == eta.end() ? 999.0 : it->second) >= derefine_threshold) {
                    ok = false;
                    break;
                }
            }
            if (ok && std::find(to_derefine.begin(), to_derefine.end(), p->parent) == to_derefine.end())
                to_derefine.push_back(p->parent);
        }
    }

    // Refinement buffer: `buffer_layers` rings of face neighbours (same level or coarser).
    std::map<int, AMRPatch*> buffered;
    for (AMRPatch* p : flagged) buffered[p->id] = p;
    std::vector<AMRPatch*> frontier = flagged;
    for (int layer = 0; layer < buffer_layers; ++layer) {
        std::vector<AMRPatch*> next;
        for (AMRPatch* p : frontier)
            for (Face f : kAllFaces)
                for (AMRPatch* nb : tree.neighbour_leaves(p, f))
                    if (nb->is_leaf() && nb->level < tree.max_level && nb->level <= p->level &&
                        buffered.find(nb->id) == buffered.end()) {
                        buffered[nb->id] = nb;
                        next.push_back(nb);
                    }
        frontier = std::move(next);
    }

    // Anti-thrash: never collapse a parent whose child is buffered for refinement.
    std::vector<AMRPatch*> kept;
    for (AMRPatch* par : to_derefine) {
        bool any = false;
        for (AMRPatch* c : par->children) any = any || buffered.count(c->id) > 0 || forced.count(c->id) > 0;
        if (!any) kept.push_back(par);
    }
    to_derefine = std::move(kept);
    for (const auto& kv : buffered) to_refine.push_back(kv.second);   // sorted by id
}

// ---------------------------------------------------------------------------
// Subcycling + refluxing
// ---------------------------------------------------------------------------

namespace {

// Flat indices into flux_x (which=='x') or flux_y for the faces along `face`,
// ordered by the transverse cell index.
void face_edge_indices(int nx, int ny, Face face, char& which, std::vector<int>& idx) {
    idx.clear();
    switch (face) {
    case Face::Left:
        which = 'x';
        for (int j = 0; j < ny; ++j) idx.push_back(j * (nx + 1));
        break;
    case Face::Right:
        which = 'x';
        for (int j = 0; j < ny; ++j) idx.push_back(j * (nx + 1) + nx);
        break;
    case Face::Bottom:
        which = 'y';
        for (int i = 0; i < nx; ++i) idx.push_back(i);
        break;
    case Face::Top:
        which = 'y';
        for (int i = 0; i < nx; ++i) idx.push_back(ny * nx + i);
        break;
    case Face::Front:
    case Face::Back: break;   // AMR is 2-D only
    }
}

// Registers live on the leaves this rank holds (own fine leaves accumulate, halo ones receive)
void init_reflux_registers(AMRTree& tree, int level, bool all_held) {
    for (AMRPatch* f : tree.leaves()) {
        if (f->level != level + 1) continue;
        f->reflux_faces.clear();
        for (Face face : kAllFaces) {
            bool coarser = false;
            for (AMRPatch* nb : tree.neighbour_leaves(f, face)) coarser = coarser || nb->level < f->level;
            if (coarser) f->reflux_faces.push_back(face);
        }
        f->has_reflux = !f->reflux_faces.empty();
        if (f->has_reflux && (all_held || f->held)) {
            f->reflux_fx.assign(static_cast<size_t>(f->nx() + 1) * f->ny(), State{0, 0, 0, 0});
            f->reflux_fy.assign(static_cast<size_t>(f->nx()) * (f->ny() + 1), State{0, 0, 0, 0});
        } else {
            f->reflux_fx.clear();
            f->reflux_fy.clear();
        }
    }
}

void accumulate_fine_face_fluxes(AMRTree& tree, int level, double dt_fine, int rank, bool dist) {
    std::vector<int> idx;
    char which;
    for (AMRPatch* f : tree.leaves()) {
        if (f->level != level + 1 || !f->has_reflux || (dist && f->owner != rank)) continue;
        for (Face face : f->reflux_faces) {
            face_edge_indices(f->nx(), f->ny(), face, which, idx);
            const Field& src = (which == 'x') ? f->last_flux_x : f->last_flux_y;
            Field& reg = (which == 'x') ? f->reflux_fx : f->reflux_fy;
            for (int k : idx)
                for (int q = 0; q < kNVar; ++q) reg[k][q] += dt_fine * src[k][q];
        }
    }
}

AMRPatch* fine_leaf_at(const std::vector<AMRPatch*>& fines, double coord, bool vertical) {
    constexpr double eps = 1e-9;
    for (AMRPatch* g : fines) {
        const double lo = vertical ? g->ymin : g->xmin;
        const double hi = vertical ? g->ymax : g->xmax;
        if (lo - eps <= coord && coord <= hi + eps) return g;
    }
    return nullptr;
}

// Correct coarse cells adjacent to coarse-fine faces by +-(sum F_fine - F_coarse dt)/A.
void apply_reflux(AMRTree& tree, int level, double dt, int rank, bool dist) {
    std::vector<int> idx_c, idx_g;
    char which_c, which_g;
    for (AMRPatch* c : tree.leaves()) {
        if (c->level != level || (dist && c->owner != rank)) continue;
        const int nx_c = c->nx(), ny_c = c->ny();
        const double dx_c = c->mesh.dx, dy_c = c->mesh.dy;
        const double area = c->mesh.cell_area();
        for (Face face : kAllFaces) {
            std::vector<AMRPatch*> fines;
            for (AMRPatch* g : tree.neighbour_leaves(c, face))
                if (g->level == level + 1) fines.push_back(g);
            if (fines.empty()) continue;
            face_edge_indices(nx_c, ny_c, face, which_c, idx_c);
            const Field& flux_c = (which_c == 'x') ? c->last_flux_x : c->last_flux_y;
            const Face opp = opposite(face);
            const bool vertical = (face == Face::Left || face == Face::Right);
            const double sign = (face == Face::Left || face == Face::Bottom) ? 1.0 : -1.0;
            const int n_along = vertical ? ny_c : nx_c;

            for (int k = 0; k < n_along; ++k) {
                State sumF = {0, 0, 0, 0};
                for (double frac : {0.25, 0.75}) {
                    const double coord = vertical ? c->ymin + (k + frac) * dy_c : c->xmin + (k + frac) * dx_c;
                    AMRPatch* g = fine_leaf_at(fines, coord, vertical);
                    if (!g || !g->has_reflux || g->reflux_fx.empty()) continue;
                    face_edge_indices(g->nx(), g->ny(), opp, which_g, idx_g);
                    const Field& reg = (which_g == 'x') ? g->reflux_fx : g->reflux_fy;
                    int kf;
                    if (vertical)
                        kf = std::min(std::max(static_cast<int>((coord - g->ymin) / g->mesh.dy), 0), g->ny() - 1);
                    else
                        kf = std::min(std::max(static_cast<int>((coord - g->xmin) / g->mesh.dx), 0), g->nx() - 1);
                    for (int q = 0; q < kNVar; ++q) sumF[q] += reg[idx_g[kf]][q];
                }
                const int ic = vertical ? (face == Face::Left ? 0 : nx_c - 1) : k;
                const int jc = vertical ? k : (face == Face::Bottom ? 0 : ny_c - 1);
                const int cell = jc * nx_c + ic;
                for (int q = 0; q < kNVar; ++q) {
                    const double delta = (sumF[q] - dt * flux_c[idx_c[k]][q]) / area;
                    c->U[cell][q] += sign * delta;
                }
            }
        }
    }
}

// What a message carries: the neighbour snapshot of a leaf, its state, or its reflux registers
enum class Kind { Ghost, State, Reflux };

// Marks the snapshot of a leaf this rank does not hold: any read of it turns the result into NaN
constexpr double kPoison = std::numeric_limits<double>::quiet_NaN();

std::vector<Field*> fields_of(AMRPatch& p, Kind k) {
    if (k == Kind::Ghost) return {&p.U_ghost};
    if (k == Kind::State) return {&p.U};
    return {&p.reflux_fx, &p.reflux_fy};
}

std::vector<size_t> sizes_of(const AMRPatch& p, Kind k) {
    const size_t nx = p.nx(), ny = p.ny();
    if (k == Kind::Reflux) return {(nx + 1) * ny, nx * (ny + 1)};
    return {nx * ny};
}

bool has_rank(const std::vector<int>& v, int r) { return std::find(v.begin(), v.end(), r) != v.end(); }

// Send the data of every leaf of `list` from its owner to the ranks dest[k] names (the receivers
// resize their copy to the expected size). Every rank calls this with the same arguments; a rank
// with nothing to send or receive just passes empty buffers.
void push_fields(const AMRParallel& par, const std::vector<AMRPatch*>& list, const std::vector<std::vector<int>>& dest,
                 Kind kind, const std::vector<int>* src = nullptr) {
    std::vector<std::vector<double>> send(par.size), recv(par.size);
    auto from = [&](size_t k) { return src ? (*src)[k] : list[k]->owner; };
    for (size_t k = 0; k < list.size(); ++k) {
        AMRPatch& p = *list[k];
        if (dest[k].empty()) continue;
        if (from(k) == par.rank) {
            for (int v : dest[k])
                for (Field* f : fields_of(p, kind))
                    for (const State& st : *f) send[v].insert(send[v].end(), st.begin(), st.end());
        } else if (has_rank(dest[k], par.rank)) {
            size_t n = 0;
            for (size_t m : sizes_of(p, kind)) n += m;
            recv[from(k)].resize(recv[from(k)].size() + n * kNVar);
        }
    }
    par.exchange(send, recv);
    std::vector<size_t> pos(par.size, 0);
    for (size_t k = 0; k < list.size(); ++k) {
        AMRPatch& p = *list[k];
        if (from(k) == par.rank || !has_rank(dest[k], par.rank)) continue;
        const int owner = from(k);
        const auto n = sizes_of(p, kind);
        auto f = fields_of(p, kind);
        for (size_t q = 0; q < f.size(); ++q) {
            f[q]->resize(n[q]);
            for (State& st : *f[q]) {
                std::copy(recv[owner].begin() + pos[owner], recv[owner].begin() + pos[owner] + kNVar, st.begin());
                pos[owner] += kNVar;
            }
        }
    }
}

// dest lists for push_fields: the halo holders (or the same-level peers) of each leaf
std::vector<std::vector<int>> dest_of(const std::vector<AMRPatch*>& list, bool peers_only) {
    std::vector<std::vector<int>> d;
    d.reserve(list.size());
    for (const AMRPatch* p : list) d.push_back(peers_only ? p->peers : p->viewers);
    return d;
}

// Fine owner -> coarse owner: the registers of the fine leaves whose coarse neighbour lives elsewhere
void exchange_reflux(AMRTree& tree, int level, const AMRParallel& par) {
    std::vector<AMRPatch*> list;
    std::vector<std::vector<int>> dest;
    for (AMRPatch* f : tree.leaves()) {
        if (f->level != level + 1 || !f->has_reflux) continue;
        std::vector<int> to;
        for (Face face : f->reflux_faces)
            for (AMRPatch* nb : tree.neighbour_leaves(f, face))
                if (nb->level < f->level && nb->owner != f->owner && !has_rank(to, nb->owner)) to.push_back(nb->owner);
        std::sort(to.begin(), to.end());
        list.push_back(f);
        dest.push_back(std::move(to));
    }
    push_fields(par, list, dest, Kind::Reflux);
}

}   // namespace

std::vector<double> allgather_leaf_values(AMRTree& tree, const AMRParallel* par, int width,
                                          const std::function<void(AMRPatch&, double*)>& fn) {
    const auto lv = tree.leaves();
    std::vector<double> out(lv.size() * width, 0.0);
    if (!par || par->size == 1) {
        for (size_t k = 0; k < lv.size(); ++k) fn(*lv[k], out.data() + k * width);
        return out;
    }
    std::vector<int> counts(par->size, 0), displs(par->size, 0);
    for (const AMRPatch* p : lv) counts[p->owner] += width;
    for (int r = 1; r < par->size; ++r) displs[r] = displs[r - 1] + counts[r - 1];
    std::vector<double> send, recv(out.size());
    for (AMRPatch* p : lv)
        if (p->owner == par->rank) {
            send.resize(send.size() + width);
            fn(*p, send.data() + send.size() - width);
        }
    par->allgatherv(send.data(), static_cast<int>(send.size()), recv.data(), counts.data(), displs.data());
    std::vector<size_t> pos(displs.begin(), displs.end());
    for (size_t k = 0; k < lv.size(); ++k) {
        std::copy(recv.begin() + pos[lv[k]->owner], recv.begin() + pos[lv[k]->owner] + width, out.begin() + k * width);
        pos[lv[k]->owner] += width;
    }
    return out;
}

Field gather_leaves_on_root(AMRTree& tree, const AMRParallel& par) {
    const auto lv = tree.leaves();
    std::vector<std::vector<double>> send(par.size), recv(par.size);
    for (const AMRPatch* p : lv) {
        const size_t n = static_cast<size_t>(p->nx()) * p->ny() * kNVar;
        if (p->owner == par.rank && par.rank != 0)
            for (const State& st : p->U) send[0].insert(send[0].end(), st.begin(), st.end());
        else if (par.rank == 0 && p->owner != 0)
            recv[p->owner].resize(recv[p->owner].size() + n);
    }
    par.exchange(send, recv);
    Field all;
    if (par.rank != 0) return all;
    std::vector<size_t> pos(par.size, 0);
    for (const AMRPatch* p : lv) {
        if (p->owner == 0) {
            all.insert(all.end(), p->U.begin(), p->U.end());
            continue;
        }
        const size_t n = static_cast<size_t>(p->nx()) * p->ny();
        for (size_t c = 0; c < n; ++c) {
            State st;
            std::copy(recv[p->owner].begin() + pos[p->owner], recv[p->owner].begin() + pos[p->owner] + kNVar,
                      st.begin());
            pos[p->owner] += kNVar;
            all.push_back(st);
        }
    }
    return all;
}

void refresh_halos(AMRTree& tree, const AMRParallel& par) {
    const auto lv = tree.leaves();
    push_fields(par, lv, dest_of(lv, false), Kind::State);
}

namespace {
// Free everything a leaf carries that is not the plain tree
void release_leaf(AMRPatch& l) {
    Field().swap(l.U);
    Field().swap(l.U_ghost);
    Field().swap(l.last_flux_x);
    Field().swap(l.last_flux_y);
    Field().swap(l.reflux_fx);
    Field().swap(l.reflux_fy);
    l.has_reflux = false;
    std::vector<char>().swap(l.solid);
    std::vector<char>().swap(l.solid_ext);
}

// The ranks must hold the same tree: a divergence would show up later as a hang in a message exchange
void check_same_tree(AMRTree& tree, const AMRParallel& par) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](std::uint64_t v) { h = (h ^ v) * 1099511628211ull; };
    for (const AMRPatch* l : tree.leaves()) {
        mix(static_cast<std::uint64_t>(l->id));
        mix(static_cast<std::uint64_t>(l->level));
        mix(static_cast<std::uint64_t>(l->owner));
    }
    const double v = static_cast<double>(h & ((1ull << 52) - 1));
    if (par.allreduce_max(v) != v || par.allreduce_max(-v) != -v)
        throw std::runtime_error("AMR: the ranks' trees diverged after the remesh");
}
}   // namespace

void allocate_held(AMRTree& tree) {
    for (AMRPatch* l : tree.leaves()) {
        if (l->held) {
            if (l->U.empty()) l->U.assign(static_cast<size_t>(l->nx()) * l->ny(), State{0, 0, 0, 0});
        } else
            release_leaf(*l);
    }
}

void remesh(AMRTree& tree, const AMRErrorIndicator& ind, const AMRParallel* par) {
    const bool dist = par && par->size > 1;
    std::vector<AMRPatch*> to_refine, to_derefine;
    if (dist) {
        const auto lv = tree.leaves();
        const auto eta =
            allgather_leaf_values(tree, par, 1, [&](AMRPatch& leaf, double* o) { *o = ind.indicator(leaf); });
        std::map<int, double> by_id;
        for (size_t k = 0; k < lv.size(); ++k) by_id[lv[k]->id] = eta[k];
        ind.mark(tree, by_id, to_refine, to_derefine);
    } else {
        ind.mark(tree, to_refine, to_derefine);
    }

    // Ranks that hold the data of a patch (owner + halo copies, valid until the new partition); a patch
    // created below inherits those of the patch it came from
    std::map<int, std::vector<int>> hold;
    if (dist)
        for (AMRPatch* l : tree.leaves()) {
            std::vector<int> v = l->viewers;
            v.push_back(l->owner);
            std::sort(v.begin(), v.end());
            hold[l->id] = std::move(v);
        }

    for (AMRPatch* patch : to_refine)
        if (patch->is_leaf() && patch->level < tree.max_level) tree.refine_patch(patch);
    for (AMRPatch* parent : to_derefine) {
        bool leaf_kids = !parent->is_leaf();
        for (AMRPatch* c : parent->children) leaf_kids = leaf_kids && c->is_leaf();
        if (!leaf_kids) continue;
        if (!dist) {
            tree.derefine_patch(parent);
            continue;
        }
        // The owner of the first child averages: its halo contains the siblings, so no message is needed
        const int collector = parent->children[0]->owner;
        for (const AMRPatch* c : parent->children)
            if (!has_rank(hold.at(c->id), collector))
                throw std::logic_error("AMR derefine: the collector does not hold all four children");
        tree.derefine_patch(parent);
        hold[parent->id] = {collector};
        if (par->rank != collector) Field().swap(parent->U);
    }
    tree.enforce_2to1_balance();
    if (!dist) return;

    assign_ownership(tree, par->rank, par->size);
    check_same_tree(tree, *par);   // before any exchange that relies on every rank having the same tree

    // Move the data: every rank that now needs a leaf (owner or halo) and does not hold it gets it from
    // the lowest rank that does
    const auto lv = tree.leaves();
    std::vector<AMRPatch*> moved;
    std::vector<std::vector<int>> dest;
    std::vector<int> src;
    for (AMRPatch* l : lv) {
        const std::vector<int>* have = nullptr;
        for (const AMRPatch* q = l; q && !have; q = q->parent) {
            const auto it = hold.find(q->id);
            if (it != hold.end()) have = &it->second;
        }
        if (!have) throw std::logic_error("AMR remesh: a leaf has no known holder");
        std::vector<int> need = l->viewers, missing;
        need.push_back(l->owner);
        for (int r : need)
            if (!has_rank(*have, r)) missing.push_back(r);
        if (missing.empty()) continue;
        std::sort(missing.begin(), missing.end());
        moved.push_back(l);
        dest.push_back(std::move(missing));
        src.push_back(have->front());
    }
    push_fields(*par, moved, dest, Kind::State, &src);
    for (AMRPatch* l : lv) {
        if (!l->held)
            release_leaf(*l);
        else if (l->owner != par->rank) {   // a halo copy: no masks, they follow the owner
            std::vector<char>().swap(l->solid);
            std::vector<char>().swap(l->solid_ext);
        }
    }
}

double advance_amr_level(AMRTree& tree, int level, double dt, double current_time, double max_time,
                         const PatchResidualFn& residual, TimeScheme& scheme, const AMRParallel* par) {
    dt = std::min(dt, max_time - current_time);
    if (dt <= 0.0) return 0.0;

    const bool dist = par && par->size > 1;
    const int rank = dist ? par->rank : 0;
    const auto lv = tree.leaves();
    std::vector<AMRPatch*> at_level;
    bool any_finer = false;
    for (AMRPatch* p : lv) {
        if (p->level == level) at_level.push_back(p);
        if (p->level > level) any_finer = true;
    }
    const bool has_finer = level < tree.max_level && any_finer;

    double S_max = 0.0;
    if (has_finer) {
        init_reflux_registers(tree, level, !dist);
        for (int sub = 0; sub < 2; ++sub) {
            const double t_sub = current_time + sub * (dt / 2);
            S_max = std::max(S_max, advance_amr_level(tree, level + 1, dt / 2, t_sub, max_time, residual, scheme, par));
            accumulate_fine_face_fluxes(tree, level, dt / 2, rank, dist);
        }
    }

    if (!at_level.empty()) {
        // Freeze every leaf so patches of other levels read one neighbour snapshot
        // (guardcell-then-advance); required for flux conservation.
        for (AMRPatch* p : lv) {
            if (!dist || p->held)
                p->U_ghost = p->U;
            else
                p->U_ghost.assign(p->U.size(),
                                  State{kPoison, kPoison, kPoison, kPoison, kPoison});   // not held: must not be read
            p->has_ghost = true;
        }
        // The patches of this level are one ODE system: concatenate them and advance with a single
        // call, so every stage sees the current stage of the same-level neighbours (as the uniform
        // solver does) while coarser / finer neighbours stay on the snapshot.
        std::vector<AMRPatch*> mine;   // the patches this rank advances
        for (AMRPatch* p : at_level)
            if (!dist || p->owner == rank) mine.push_back(p);
        const auto stage_dest = dist ? dest_of(at_level, true) : std::vector<std::vector<int>>();
        const int n_at = static_cast<int>(mine.size());
        std::vector<size_t> off(n_at + 1, 0);
        for (int k = 0; k < n_at; ++k) off[k + 1] = off[k] + mine[k]->U.size();
        Field Ucat(off[n_at]);
        for (int k = 0; k < n_at; ++k) std::copy(mine[k]->U.begin(), mine[k]->U.end(), Ucat.begin() + off[k]);
        const ResidualFn res = [&](const Field& Uin, Field& R) {
            R.resize(Uin.size());
            for (int k = 0; k < n_at; ++k)
                std::copy(Uin.begin() + off[k], Uin.begin() + off[k + 1], mine[k]->U_ghost.begin());
            if (dist) push_fields(*par, at_level, stage_dest, Kind::Ghost);   // same-level neighbours at this stage
            double s_max = 0.0;
#pragma omp parallel if (n_at > 1)
            {
                Field Rk;
                double s_local = 0.0;
#pragma omp for schedule(dynamic)
                for (int k = 0; k < n_at; ++k) {
                    AMRPatch* patch = mine[k];
                    s_local = std::max(
                        s_local, residual(*patch, tree, patch->U_ghost, Rk, patch->last_flux_x, patch->last_flux_y));
                    std::copy(Rk.begin(), Rk.end(), R.begin() + off[k]);
                }
#pragma omp critical(fv_amr_smax)
                s_max = std::max(s_max, s_local);
            }
            return s_max;
        };
        S_max = std::max(S_max, scheme.advance_fixed_dt(Ucat, res, dt));
        for (int k = 0; k < n_at; ++k) std::copy(Ucat.begin() + off[k], Ucat.begin() + off[k + 1], mine[k]->U.begin());
        if (dist) S_max = par->allreduce_max(S_max);
        for (AMRPatch* p : lv) p->has_ghost = false;   // keep capacity for the next step
    }

    if (has_finer) {
        if (dist) exchange_reflux(tree, level, *par);
        if (!g_amr_disable_reflux) apply_reflux(tree, level, dt, rank, dist);
    }
    // The state of this level is final (refluxed): the halo copies on other ranks follow
    if (dist && !at_level.empty()) push_fields(*par, at_level, dest_of(at_level, false), Kind::State);
    return S_max;
}

}   // namespace fv
