#include "fv/AMR.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "fv/SolutionMapping.hpp"

namespace fv {

bool g_amr_disable_reflux = false;

// ---------------------------------------------------------------------------
// AMRTree
// ---------------------------------------------------------------------------

AMRTree::AMRTree(double Lx_, double Ly_, int nxp, int nyp, int max_lvl)
    : Lx(Lx_), Ly(Ly_), nx_patch(nxp), ny_patch(nyp), max_level(max_lvl) {
    // Children have the same cell count over half the extent, so 2 fine cells
    // must tile 1 coarse cell: odd dimensions break conservation.
    if (max_level > 0 && (nxp % 2 != 0 || nyp % 2 != 0))
        throw std::invalid_argument(
            "AMR patch dimensions must be EVEN for refinement to align 2:1 (got nx_patch=" +
            std::to_string(nxp) + ", ny_patch=" + std::to_string(nyp) +
            "). Odd dimensions break conservation.");
    root = new_patch(0.0, Lx, 0.0, Ly, 0, nullptr);
}

AMRPatch* AMRTree::new_patch(double x0, double x1, double y0, double y1, int level, AMRPatch* parent) {
    auto p = std::make_unique<AMRPatch>();
    p->xmin = x0; p->xmax = x1; p->ymin = y0; p->ymax = y1;
    p->level  = level;
    p->parent = parent;
    p->id     = next_id_++;
    p->mesh   = generate_cartesian_mesh(x1 - x0, y1 - y0, nx_patch, ny_patch, x0, y0);
    p->U.assign(p->mesh.num_cells(), State{0, 0, 0, 0});
    AMRPatch* raw = p.get();
    patches_[raw->id] = std::move(p);
    return raw;
}

void AMRTree::collect_leaves(AMRPatch* p, std::vector<AMRPatch*>& out) {
    if (p->is_leaf()) { out.push_back(p); return; }
    for (AMRPatch* c : p->children) collect_leaves(c, out);
}

std::vector<AMRPatch*> AMRTree::leaves() const {
    std::vector<AMRPatch*> out;
    collect_leaves(root, out);
    return out;
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
    const double dx_p = mp.Lx / mp.nx, dy_p = mp.Ly / mp.ny;
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
        const double dx_p = mp.Lx / nx_p, dy_p = mp.Ly / ny_p;
        const double x_off = mp.xc(0) - 0.5 * (mp.Lx / nx_p);
        const double y_off = mp.yc(0) - 0.5 * (mp.Ly / ny_p);
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
}

void AMRTree::enforce_2to1_balance() {
    bool changed = true;
    while (changed) {
        changed = false;
        const auto snapshot = leaves();
        for (AMRPatch* patch : snapshot) {
            const auto lv = leaves();
            AMRPatch* nbs[4];
            for (int f = 0; f < 4; ++f) nbs[f] = find_neighbour_leaf(patch, kAllFaces[f], lv);
            for (AMRPatch* nb : nbs) {
                if (nb && nb->is_leaf() && patch->level - nb->level > 1 && nb->level < max_level) {
                    refine_patch(nb);
                    changed = true;
                }
            }
        }
    }
}

AMRPatch* AMRTree::find_neighbour_leaf(const AMRPatch* p, Face face, const std::vector<AMRPatch*>& lv) const {
    constexpr double eps = 1e-10;
    for (AMRPatch* nb : lv) {
        if (nb == p) continue;
        bool hit = false;
        switch (face) {
            case Face::Left:
                if (p->xmin <= eps) return nullptr;
                hit = std::abs(nb->xmax - p->xmin) < eps && nb->ymin < p->ymax - eps && nb->ymax > p->ymin + eps;
                break;
            case Face::Right:
                if (p->xmax >= Lx - eps) return nullptr;
                hit = std::abs(nb->xmin - p->xmax) < eps && nb->ymin < p->ymax - eps && nb->ymax > p->ymin + eps;
                break;
            case Face::Bottom:
                if (p->ymin <= eps) return nullptr;
                hit = std::abs(nb->ymax - p->ymin) < eps && nb->xmin < p->xmax - eps && nb->xmax > p->xmin + eps;
                break;
            case Face::Top:
                if (p->ymax >= Ly - eps) return nullptr;
                hit = std::abs(nb->ymin - p->ymax) < eps && nb->xmin < p->xmax - eps && nb->xmax > p->xmin + eps;
                break;
        }
        if (hit) return nb;
    }
    return nullptr;
}

AMRPatch* AMRTree::get_neighbour(const AMRPatch* patch, Face face) const {
    return find_neighbour_leaf(patch, face, leaves());
}

std::vector<AMRPatch*> AMRTree::neighbour_leaves(const AMRPatch* p, Face face) const {
    constexpr double eps = 1e-10;
    std::vector<AMRPatch*> out;
    if ((face == Face::Left && p->xmin <= eps) || (face == Face::Right && p->xmax >= Lx - eps) ||
        (face == Face::Bottom && p->ymin <= eps) || (face == Face::Top && p->ymax >= Ly - eps))
        return out;
    for (AMRPatch* nb : leaves()) {
        if (nb == p) continue;
        bool hit = false;
        switch (face) {
            case Face::Left:
                hit = std::abs(nb->xmax - p->xmin) < eps && nb->ymin < p->ymax - eps && nb->ymax > p->ymin + eps; break;
            case Face::Right:
                hit = std::abs(nb->xmin - p->xmax) < eps && nb->ymin < p->ymax - eps && nb->ymax > p->ymin + eps; break;
            case Face::Bottom:
                hit = std::abs(nb->ymax - p->ymin) < eps && nb->xmin < p->xmax - eps && nb->xmax > p->xmin + eps; break;
            case Face::Top:
                hit = std::abs(nb->ymin - p->ymax) < eps && nb->xmin < p->xmax - eps && nb->xmax > p->xmin + eps; break;
        }
        if (hit) out.push_back(nb);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Lohner second-derivative error indicator
// ---------------------------------------------------------------------------

double AMRErrorIndicator::indicator(const AMRPatch& patch) const {
    const int nx = patch.nx(), ny = patch.ny();
    const int PW = nx + 2;
    // Anti-symmetric (odd) reflection: ghost = 2*edge - interior, so a linear
    // density ramp gives zero second difference.
    std::vector<double> rp(static_cast<size_t>(PW) * (ny + 2), 0.0);
    double rho_max = 0.0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double r = patch.U[j * nx + i][0];
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

void AMRErrorIndicator::mark(AMRTree& tree, std::vector<AMRPatch*>& to_refine,
                             std::vector<AMRPatch*>& to_derefine) const {
    to_refine.clear();
    to_derefine.clear();
    const auto lv = tree.leaves();
    std::map<int, double> eta;
    for (AMRPatch* p : lv) eta[p->id] = indicator(*p);

    std::vector<AMRPatch*> flagged;
    for (AMRPatch* p : lv) {
        const double e = eta[p->id];
        if (e > refine_threshold && p->level < tree.max_level) {
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
        for (AMRPatch* c : par->children) any = any || buffered.count(c->id) > 0;
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
        const double dx_c = c->mesh.Lx / nx_c, dy_c = c->mesh.Ly / ny_c;
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
                    if (vertical) kf = std::min(std::max(static_cast<int>((coord - g->ymin) / (g->mesh.Ly / g->ny())), 0), g->ny() - 1);
                    else          kf = std::min(std::max(static_cast<int>((coord - g->xmin) / (g->mesh.Lx / g->nx())), 0), g->nx() - 1);
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

}  // namespace

double advance_amr_level(AMRTree& tree, int level, double dt, double current_time, double max_time,
                         const PatchResidualFn& residual, TimeScheme& scheme) {
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
            S_max = std::max(S_max, advance_amr_level(tree, level + 1, dt / 2, t_sub, max_time, residual, scheme));
            accumulate_fine_face_fluxes(tree, level, dt / 2);
        }
    }

    if (!at_level.empty()) {
        // Freeze every leaf so all patches read the same neighbour snapshot
        // (guardcell-then-advance); required for flux conservation.
        for (AMRPatch* p : lv) { p->U_ghost = p->U; p->has_ghost = true; }
        // With the snapshot frozen, patches at this level are independent:
        // advance them concurrently, one scheme clone (scratch buffers) per thread.
        const int n_at = static_cast<int>(at_level.size());
#pragma omp parallel if(n_at > 1)
        {
            std::unique_ptr<TimeScheme> local = scheme.clone();
            double s_local = 0.0;
#pragma omp for schedule(dynamic)
            for (int k = 0; k < n_at; ++k) {
                AMRPatch* patch = at_level[k];
                ResidualFn res = [&](const Field& U, Field& R) {
                    return residual(*patch, tree, U, R, patch->last_flux_x, patch->last_flux_y);
                };
                s_local = std::max(s_local, local->advance_fixed_dt(patch->U, res, dt));
            }
#pragma omp critical(fv_amr_smax)
            S_max = std::max(S_max, s_local);
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
