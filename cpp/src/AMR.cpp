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
            "AMR patch dimensions must be EVEN for refinement to align 2:1 (got nx_patch=" +
            std::to_string(nxp) + ", ny_patch=" + std::to_string(nyp) +
            "). Odd dimensions break conservation.");
    if (nrx < 1 || nry < 1) throw std::invalid_argument("AMR needs at least one root patch per axis");
    nslot_x_ = nrx << max_level;
    nslot_y_ = nry << max_level;
    slot_w_  = Lx / nslot_x_;
    slot_h_  = Ly / nslot_y_;
    for (int j = 0; j < nry; ++j)
        for (int i = 0; i < nrx; ++i)
            roots.push_back(new_patch(Lx * i / nrx, Lx * (i + 1) / nrx, Ly * j / nry, Ly * (j + 1) / nry, 0, nullptr));
    root = roots[0];
}

AMRPatch* AMRTree::new_patch(double x0, double x1, double y0, double y1, int level, AMRPatch* parent) {
    auto p = std::make_unique<AMRPatch>();
    p->xmin = x0; p->xmax = x1; p->ymin = y0; p->ymax = y1;
    p->level  = level;
    p->parent = parent;
    p->id     = next_id_++;
    p->mesh   = generate_cartesian_mesh(x1 - x0, y1 - y0, nx_patch, ny_patch, x0, y0);
    // Cell size from the base grid by exact halving, so tiled patches share the uniform mesh's dx
    p->mesh.dx = std::ldexp(Lx / (nroot_x * nx_patch), -level);
    p->mesh.dy = std::ldexp(Ly / (nroot_y * ny_patch), -level);
    p->U.assign(p->mesh.num_cells(), State{0, 0, 0, 0});
    AMRPatch* raw = p.get();
    patches_[raw->id] = std::move(p);
    dirty_ = true;
    return raw;
}

void AMRTree::collect_leaves(AMRPatch* p, std::vector<AMRPatch*>& out) {
    if (p->is_leaf()) { out.push_back(p); return; }
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
    return kids;
}

void AMRTree::derefine_patch(AMRPatch* patch) {
    if (patch->is_leaf()) throw std::invalid_argument("Patch has no children");
    for (AMRPatch* c : patch->children)
        if (!c->is_leaf()) throw std::invalid_argument("Can only derefine if all children are leaves");

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
        case Face::Left:   if (i0 <= 0) return out;           i1 = i0;     i0 = i0 - 1; break;
        case Face::Right:  if (i1 >= nslot_x_) return out;    i0 = i1;     i1 = i1 + 1; break;
        case Face::Bottom: if (j0 <= 0) return out;           j1 = j0;     j0 = j0 - 1; break;
        case Face::Top:    if (j1 >= nslot_y_) return out;    j0 = j1;     j1 = j1 + 1; break;
        case Face::Front:
        case Face::Back:   return out;   // AMR is 2-D only
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
        v = (v | (v << 8))  & 0x00FF00FF00FF00FFull;
        v = (v | (v << 4))  & 0x0F0F0F0F0F0F0F0Full;
        v = (v | (v << 2))  & 0x3333333333333333ull;
        v = (v | (v << 1))  & 0x5555555555555555ull;
        return v;
    };
    return spread(x) | (spread(y) << 1);
}
}  // namespace

void assign_ownership(AMRTree& tree, int rank, int size) {
    constexpr int kStencil = 2;   // cells of the owner's own level a residual reads across an interface
    const auto lv = tree.leaves();
    const double sw = tree.Lx / (tree.nroot_x << tree.max_level), sh = tree.Ly / (tree.nroot_y << tree.max_level);
    for (int level = 0; level <= tree.max_level; ++level) {
        std::vector<std::pair<std::uint64_t, AMRPatch*>> at;
        for (AMRPatch* p : lv)
            if (p->level == level)
                at.emplace_back(morton_key(static_cast<std::uint32_t>(std::lround(p->xmin / sw)),
                                           static_cast<std::uint32_t>(std::lround(p->ymin / sh))), p);
        std::sort(at.begin(), at.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        const long long n = static_cast<long long>(at.size());
        for (long long k = 0; k < n; ++k) at[k].second->owner = static_cast<int>(k * size / n);
    }
    for (AMRPatch* p : lv) p->held = (p->owner == rank);
    for (AMRPatch* p : lv)
        if (p->owner == rank)
            for (AMRPatch* nb : tree.leaves_near(p, kStencil)) nb->held = true;
}

// ---------------------------------------------------------------------------
// Lohner second-derivative error indicator
// ---------------------------------------------------------------------------

namespace {
// Lohner ratio of the scalar `value(cell)` over a patch
template <class F>
double lohner_eta(const AMRPatch& patch, F value) {
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
        R(j, 0)      = 2.0 * R(j, 1) - (nx > 1 ? R(j, 2) : R(j, 1));
        R(j, nx + 1) = 2.0 * R(j, nx) - (nx > 1 ? R(j, nx - 1) : R(j, nx));
    }
    for (int i = 1; i <= nx; ++i) {
        R(0, i)      = 2.0 * R(1, i) - (ny > 1 ? R(2, i) : R(1, i));
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
}  // namespace

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
    to_refine.clear();
    to_derefine.clear();
    const auto lv = tree.leaves();
    std::map<int, double> eta;
    for (AMRPatch* p : lv) eta[p->id] = indicator(*p);

    std::set<int> forced;
    if (force_refine)
        for (AMRPatch* p : lv)
            if (force_refine(*p)) forced.insert(p->id);

    std::vector<AMRPatch*> flagged;
    for (AMRPatch* p : lv) {
        const double e = eta[p->id];
        if ((e > refine_threshold || forced.count(p->id)) && p->level < tree.max_level) {
            flagged.push_back(p);
        } else if (e < derefine_threshold && p->parent && p->level > 0) {
            // All siblings must be leaves below the threshold to collapse the parent.
            bool ok = true;
            for (AMRPatch* s : p->parent->children) {
                auto it = eta.find(s->id);
                if (!s->is_leaf() || (it == eta.end() ? 999.0 : it->second) >= derefine_threshold) { ok = false; break; }
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
        case Face::Left:   which = 'x'; for (int j = 0; j < ny; ++j) idx.push_back(j * (nx + 1)); break;
        case Face::Right:  which = 'x'; for (int j = 0; j < ny; ++j) idx.push_back(j * (nx + 1) + nx); break;
        case Face::Bottom: which = 'y'; for (int i = 0; i < nx; ++i) idx.push_back(i); break;
        case Face::Top:    which = 'y'; for (int i = 0; i < nx; ++i) idx.push_back(ny * nx + i); break;
        case Face::Front:
        case Face::Back:   break;   // AMR is 2-D only
    }
}

void init_reflux_registers(AMRTree& tree, int level) {
    for (AMRPatch* f : tree.leaves()) {
        if (f->level != level + 1) continue;
        f->reflux_faces.clear();
        for (Face face : kAllFaces) {
            bool coarser = false;
            for (AMRPatch* nb : tree.neighbour_leaves(f, face)) coarser = coarser || nb->level < f->level;
            if (coarser) f->reflux_faces.push_back(face);
        }
        f->has_reflux = !f->reflux_faces.empty();
        if (f->has_reflux) {
            f->reflux_fx.assign(static_cast<size_t>(f->nx() + 1) * f->ny(), State{0, 0, 0, 0});
            f->reflux_fy.assign(static_cast<size_t>(f->nx()) * (f->ny() + 1), State{0, 0, 0, 0});
        } else {
            f->reflux_fx.clear();
            f->reflux_fy.clear();
        }
    }
}

void accumulate_fine_face_fluxes(AMRTree& tree, int level, double dt_fine) {
    std::vector<int> idx;
    char which;
    for (AMRPatch* f : tree.leaves()) {
        if (f->level != level + 1 || !f->has_reflux) continue;
        for (Face face : f->reflux_faces) {
            face_edge_indices(f->nx(), f->ny(), face, which, idx);
            const Field& src = (which == 'x') ? f->last_flux_x : f->last_flux_y;
            Field&       reg = (which == 'x') ? f->reflux_fx : f->reflux_fy;
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
void apply_reflux(AMRTree& tree, int level, double dt) {
    std::vector<int> idx_c, idx_g;
    char which_c, which_g;
    for (AMRPatch* c : tree.leaves()) {
        if (c->level != level) continue;
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
                    if (!g || !g->has_reflux) continue;
                    face_edge_indices(g->nx(), g->ny(), opp, which_g, idx_g);
                    const Field& reg = (which_g == 'x') ? g->reflux_fx : g->reflux_fy;
                    int kf;
                    if (vertical) kf = std::min(std::max(static_cast<int>((coord - g->ymin) / g->mesh.dy), 0), g->ny() - 1);
                    else          kf = std::min(std::max(static_cast<int>((coord - g->xmin) / g->mesh.dx), 0), g->nx() - 1);
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

// Average children back onto their parent (keeps parents valid for derefinement).
void restrict_fine_to_coarse(AMRTree& tree, int coarse_level) {
    for (AMRPatch* patch : tree.all_patches()) {
        if (patch->level != coarse_level || patch->is_leaf()) continue;
        const int nx_f = patch->children[0]->nx(), ny_f = patch->children[0]->ny();
        const int nf_x = 2 * nx_f, nf_y = 2 * ny_f;
        Field full(static_cast<size_t>(nf_x) * nf_y, State{0, 0, 0, 0});
        const double half_x = (patch->xmax - patch->xmin) / 2.0;
        const double half_y = (patch->ymax - patch->ymin) / 2.0;
        for (AMRPatch* child : patch->children) {
            const int i_off = static_cast<int>(std::lround((child->xmin - patch->xmin) / half_x));
            const int j_off = static_cast<int>(std::lround((child->ymin - patch->ymin) / half_y));
            for (int j = 0; j < ny_f; ++j)
                for (int i = 0; i < nx_f; ++i)
                    full[(j_off * ny_f + j) * nf_x + i_off * nx_f + i] = child->U[j * nx_f + i];
        }
        patch->U = restrict_patch(full, nf_x, nf_y, patch->nx(), patch->ny());
    }
}

enum class Share { Ghost, State, Fluxes };

// Marks the snapshot of a leaf this rank does not hold: any read of it turns the result into NaN
constexpr double kPoison = std::numeric_limits<double>::quiet_NaN();

// Replace the copies of the patches that a rank does not own by the owner's data.
void share_level(const AMRParallel& par, const std::vector<AMRPatch*>& pat, const std::vector<int>& owner, Share what) {
    auto fields = [&](AMRPatch& p) {
        std::vector<Field*> f;
        if (what == Share::Ghost) f = {&p.U_ghost};
        else if (what == Share::State) f = {&p.U};
        else f = {&p.last_flux_x, &p.last_flux_y};
        return f;
    };
    auto sizes = [&](const AMRPatch& p) {
        std::vector<size_t> n;
        const size_t nx = p.nx(), ny = p.ny();
        if (what == Share::Fluxes) n = {(nx + 1) * ny, nx * (ny + 1)};
        else n = {nx * ny};
        return n;
    };
    std::vector<int> counts(par.size, 0), displs(par.size, 0);
    for (size_t k = 0; k < pat.size(); ++k) {
        const auto n = sizes(*pat[k]);
        auto f = fields(*pat[k]);
        for (size_t q = 0; q < f.size(); ++q) {
            if (owner[k] != par.rank) f[q]->resize(n[q]);
            counts[owner[k]] += static_cast<int>(n[q] * kNVar);
        }
    }
    for (int r = 1; r < par.size; ++r) displs[r] = displs[r - 1] + counts[r - 1];
    std::vector<double> send, recv(static_cast<size_t>(displs[par.size - 1] + counts[par.size - 1]));
    for (size_t k = 0; k < pat.size(); ++k)
        if (owner[k] == par.rank)
            for (Field* f : fields(*pat[k]))
                for (const State& st : *f) send.insert(send.end(), st.begin(), st.end());
    par.allgatherv(send.data(), static_cast<int>(send.size()), recv.data(), counts.data(), displs.data());
    std::vector<size_t> pos(displs.begin(), displs.end());
    for (size_t k = 0; k < pat.size(); ++k) {
        const int r = owner[k];
        for (Field* f : fields(*pat[k])) {
            if (r != par.rank && what == Share::Ghost && !pat[k]->held)
                pos[r] += f->size() * kNVar;   // not a halo of this rank: leave the poisoned snapshot
            else if (r != par.rank)
                for (State& st : *f) { std::copy(recv.begin() + pos[r], recv.begin() + pos[r] + kNVar, st.begin()); pos[r] += kNVar; }
            else
                pos[r] += f->size() * kNVar;
        }
    }
}

}  // namespace

double advance_amr_level(AMRTree& tree, int level, double dt, double current_time, double max_time,
                         const PatchResidualFn& residual, TimeScheme& scheme, const AMRParallel* par) {
    dt = std::min(dt, max_time - current_time);
    if (dt <= 0.0) return 0.0;

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
        init_reflux_registers(tree, level);
        for (int sub = 0; sub < 2; ++sub) {
            const double t_sub = current_time + sub * (dt / 2);
            S_max = std::max(S_max, advance_amr_level(tree, level + 1, dt / 2, t_sub, max_time, residual, scheme, par));
            accumulate_fine_face_fluxes(tree, level, dt / 2);
        }
    }

    if (!at_level.empty()) {
        // Freeze every leaf so patches of other levels read one neighbour snapshot
        // (guardcell-then-advance); required for flux conservation.
        const bool dist = par && par->size > 1;
        for (AMRPatch* p : lv) {
            if (!dist || p->held) p->U_ghost = p->U;
            else p->U_ghost.assign(p->U.size(), State{kPoison, kPoison, kPoison, kPoison, kPoison});   // not held: must not be read
            p->has_ghost = true;
        }
        // The patches of this level are one ODE system: concatenate them and advance with a single
        // call, so every stage sees the current stage of the same-level neighbours (as the uniform
        // solver does) while coarser / finer neighbours stay on the snapshot.
        const int n_all = static_cast<int>(at_level.size());
        std::vector<int> owner(n_all, 0);
        if (dist) for (int k = 0; k < n_all; ++k) owner[k] = at_level[k]->owner;
        std::vector<AMRPatch*> mine;   // the patches this rank advances
        for (int k = 0; k < n_all; ++k)
            if (!dist || owner[k] == par->rank) mine.push_back(at_level[k]);
        const int n_at = static_cast<int>(mine.size());
        std::vector<size_t> off(n_at + 1, 0);
        for (int k = 0; k < n_at; ++k) off[k + 1] = off[k] + mine[k]->U.size();
        Field Ucat(off[n_at]);
        for (int k = 0; k < n_at; ++k) std::copy(mine[k]->U.begin(), mine[k]->U.end(), Ucat.begin() + off[k]);
        const ResidualFn res = [&](const Field& Uin, Field& R) {
            R.resize(Uin.size());
            for (int k = 0; k < n_at; ++k) std::copy(Uin.begin() + off[k], Uin.begin() + off[k + 1], mine[k]->U_ghost.begin());
            if (dist) share_level(*par, at_level, owner, Share::Ghost);   // same-level neighbours at this stage
            double s_max = 0.0;
#pragma omp parallel if(n_at > 1)
            {
                Field Rk;
                double s_local = 0.0;
#pragma omp for schedule(dynamic)
                for (int k = 0; k < n_at; ++k) {
                    AMRPatch* patch = mine[k];
                    s_local = std::max(s_local, residual(*patch, tree, patch->U_ghost, Rk, patch->last_flux_x, patch->last_flux_y));
                    std::copy(Rk.begin(), Rk.end(), R.begin() + off[k]);
                }
#pragma omp critical(fv_amr_smax)
                s_max = std::max(s_max, s_local);
            }
            return s_max;
        };
        S_max = std::max(S_max, scheme.advance_fixed_dt(Ucat, res, dt));
        for (int k = 0; k < n_at; ++k) std::copy(Ucat.begin() + off[k], Ucat.begin() + off[k + 1], mine[k]->U.begin());
        if (dist) {
            share_level(*par, at_level, owner, Share::State);
            share_level(*par, at_level, owner, Share::Fluxes);
            S_max = par->allreduce_max(S_max);
        }
        for (AMRPatch* p : lv) p->has_ghost = false;   // keep capacity for the next step
    }

    if (has_finer) {
        if (!g_amr_disable_reflux) apply_reflux(tree, level, dt);
        restrict_fine_to_coarse(tree, level);
    }
    return S_max;
}

}  // namespace fv
