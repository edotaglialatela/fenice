#include "fv/CellRemesh.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>

namespace fv {

namespace {

// The cells of one level of the hierarchy: the leaves of that level and the split ancestors of the deeper leaves (whose
// value is the mean of their four children), sorted by Morton key. A cell at level l exists wherever a leaf of level >= l
// lies; coarser leaves are not cells of this level.
struct LevelCells {
    std::vector<uint64_t> key;   // morton_key(i, j), ascending
    std::vector<int> ci, cj;
    std::vector<State> val;      // conservative state
    std::vector<int> leaf;       // leaf index, -1 for a split cell

    int find(int i, int j) const {
        const uint64_t k = morton_key(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
        const auto it = std::lower_bound(key.begin(), key.end(), k);
        return (it != key.end() && *it == k) ? static_cast<int>(it - key.begin()) : -1;
    }
    int size() const { return static_cast<int>(key.size()); }
};

}   // namespace

void CellIndicator::diffuse(std::vector<double>& xi, const std::vector<std::array<int, 2>>& interfaces) const {
    std::vector<double> add(xi.size());
    for (int it = 0; it < diffusion; ++it) {
        std::fill(add.begin(), add.end(), 0.0);
        for (const std::array<int, 2>& f : interfaces) {
            const int a = f[0], b = f[1];
            if (b < 0) {   // one cell of this level against a coarser leaf: the cell decides, only it receives
                if (xi[a] > diffusion_trigger) add[a] += diffusion_coeff;
            } else if (xi[a] > diffusion_trigger || xi[b] > diffusion_trigger) {
                add[a] += diffusion_coeff;
                add[b] += diffusion_coeff;
            }
        }
        for (size_t c = 0; c < xi.size(); ++c) xi[c] += add[c];
    }
}

CellIndicator::Result CellIndicator::evaluate(const CellMesh& m, const Field& U, const EOS& eos) const {
    const int N = m.num_leaves(), M = m.max_level;
    // 1) the cells of every level, finest first
    std::vector<LevelCells> lev(static_cast<size_t>(M) + 1);
    for (int l = M; l >= 0; --l) {
        LevelCells& L = lev[static_cast<size_t>(l)];
        // parents of the cells one level down (sorted, each preceded by its four children there)
        std::vector<uint64_t> pkey;
        std::vector<State> pval;
        std::vector<int> pi, pj;
        if (l < M) {
            const LevelCells& D = lev[static_cast<size_t>(l) + 1];
            for (int c = 0; c < D.size(); c += 4) {   // four consecutive children share a parent
                pkey.push_back(D.key[c] >> 2);
                pi.push_back(D.ci[c] >> 1);
                pj.push_back(D.cj[c] >> 1);
                State s;
                for (int q = 0; q < kNVar; ++q)
                    s[q] = ((D.val[c][q] + D.val[c + 1][q]) + (D.val[c + 2][q] + D.val[c + 3][q])) * 0.25;
                pval.push_back(s);
            }
        }
        // merge with the leaves of level l (already in Morton order in the leaf arrays)
        int a = m.level_begin[l], ea = m.level_begin[l + 1];
        size_t b = 0;
        while (a < ea || b < pkey.size()) {
            const uint64_t ka = a < ea ? morton_key(static_cast<uint32_t>(m.ci[a]), static_cast<uint32_t>(m.cj[a])) : ~0ull;
            const uint64_t kb = b < pkey.size() ? pkey[b] : ~0ull;
            if (ka < kb) {
                L.key.push_back(ka);
                L.ci.push_back(m.ci[a]);
                L.cj.push_back(m.cj[a]);
                L.val.push_back(U[a]);
                L.leaf.push_back(a);
                ++a;
            } else {
                L.key.push_back(kb);
                L.ci.push_back(pi[b]);
                L.cj.push_back(pj[b]);
                L.val.push_back(pval[b]);
                L.leaf.push_back(-1);
                ++b;
            }
        }
    }
    // leaf lookup by position, for a coarser leaf covering a position of level l
    const auto coarser_leaf_state = [&](int l, int i, int j, State& out) {
        for (int k = l - 1; k >= 0; --k) {
            i >>= 1;
            j >>= 1;
            const LevelCells& K = lev[static_cast<size_t>(k)];
            const int idx = K.find(i, j);
            if (idx >= 0 && K.leaf[idx] >= 0) {
                out = K.val[idx];
                return true;
            }
        }
        return false;
    };
    // 2) xi of every level
    std::vector<std::vector<double>> xi(static_cast<size_t>(M) + 1);
    for (int l = 0; l <= M; ++l) {
        const LevelCells& L = lev[static_cast<size_t>(l)];
        std::vector<double>& x = xi[static_cast<size_t>(l)];
        x.assign(static_cast<size_t>(L.size()), 0.0);
        std::vector<std::array<int, 2>> ifaces;
        const int NX = m.nx_at(l), NY = m.ny_at(l);
        const auto flag = [&](const State& sa, const State& sb) {
            const auto test = [&](double ca, double cb, double floor_value) {
                double vmin = std::min(std::abs(ca), std::abs(cb));
                if (vmin < value_floor) vmin = floor_value;
                return std::abs(cb - ca) / vmin >= criterion;
            };
            bool hit = false;
            if (use_density) hit = hit || test(sa[0], sb[0], value_floor);
            if (use_pressure || use_velocity) {
                const State wa = eos.cons_to_prim(sa), wb = eos.cons_to_prim(sb);
                if (use_pressure) hit = hit || test(wa[3], wb[3], value_floor);
                if (use_velocity) {
                    const double va = std::sqrt(wa[1] * wa[1] + wa[2] * wa[2] + wa[4] * wa[4]);
                    const double vb = std::sqrt(wb[1] * wb[1] + wb[2] * wb[2] + wb[4] * wb[4]);
                    hit = hit || test(va, vb, velocity_floor);
                }
            }
            return hit;
        };
        for (int c = 0; c < L.size(); ++c)
            for (int s = 0; s < 4; ++s) {
                int i = L.ci[c] + (s == 1 ? 1 : (s == 0 ? -1 : 0)), j = L.cj[c] + (s == 3 ? 1 : (s == 2 ? -1 : 0));
                if (i < 0 || i >= NX) {
                    if (!m.periodic_x) continue;
                    i = (i + NX) % NX;
                }
                if (j < 0 || j >= NY) {
                    if (!m.periodic_y) continue;
                    j = (j + NY) % NY;
                }
                const int nb = L.find(i, j);
                if (nb >= 0) {
                    if (s == 1 || s == 3) {   // each same-level pair once, from its lower cell
                        ifaces.push_back({c, nb});
                        if (flag(L.val[c], L.val[nb])) x[c] = x[nb] = 1.0;
                    }
                } else {
                    State coarse;
                    if (!coarser_leaf_state(l, i, j, coarse)) continue;   // cannot happen in a complete tree
                    ifaces.push_back({c, -1});
                    if (flag(L.val[c], coarse)) x[c] = 1.0;
                }
            }
        diffuse(x, ifaces);
    }
    // 3) per leaf: its own xi, and the xi of its parent at the parent level
    Result r;
    r.xi.assign(static_cast<size_t>(N), 0.0);
    r.xi_parent.assign(static_cast<size_t>(N), 1e300);
    for (int c = 0; c < N; ++c) {
        const int l = m.level[c];
        const LevelCells& L = lev[static_cast<size_t>(l)];
        r.xi[c] = xi[static_cast<size_t>(l)][static_cast<size_t>(L.find(m.ci[c], m.cj[c]))];
        if (l > 0) {
            const LevelCells& P = lev[static_cast<size_t>(l) - 1];
            r.xi_parent[c] = xi[static_cast<size_t>(l) - 1][static_cast<size_t>(P.find(m.ci[c] >> 1, m.cj[c] >> 1))];
        }
    }
    return r;
}

std::vector<double> CellIndicator::xi(const CellMesh& m, const Field& U, const EOS& eos) const {
    return evaluate(m, U, eos).xi;
}

void CellAdaptor::set_body(std::function<ImmersedBoundary(const Mesh&)> factory, int margin) {
    body_ = std::move(factory);
    margin_ = margin;
    band_.clear();
}

void CellAdaptor::build_band() {
    const int M = mesh_.max_level, NX = mesh_.nx_at(M), NY = mesh_.ny_at(M);
    const ImmersedBoundary ib = body_(generate_cartesian_mesh(mesh_.Lx, mesh_.Ly, NX, NY, mesh_.x0, mesh_.y0));
    const std::vector<char>& mk = ib.mask();
    // dilation by `margin_` cells, separably
    std::vector<char> tmp(mk.size(), 0);
    band_.assign(mk.size(), 0);
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
            if (mk[static_cast<size_t>(j) * NX + i])
                for (int d = -margin_; d <= margin_; ++d)
                    if (i + d >= 0 && i + d < NX) tmp[static_cast<size_t>(j) * NX + i + d] = 1;
    for (int j = 0; j < NY; ++j)
        for (int i = 0; i < NX; ++i)
            if (tmp[static_cast<size_t>(j) * NX + i])
                for (int d = -margin_; d <= margin_; ++d)
                    if (j + d >= 0 && j + d < NY) band_[static_cast<size_t>(j + d) * NX + i] = 1;
}

bool CellAdaptor::in_body_band(int c) {
    if (!body_) return false;
    if (band_.empty()) build_band();
    const int M = mesh_.max_level, NX = mesh_.nx_at(M), r = 1 << (M - mesh_.level[c]);
    for (int j = mesh_.cj[c] * r; j < (mesh_.cj[c] + 1) * r; ++j)
        for (int i = mesh_.ci[c] * r; i < (mesh_.ci[c] + 1) * r; ++i)
            if (band_[static_cast<size_t>(j) * NX + i]) return true;
    return false;
}

bool CellAdaptor::remesh() {
    using Clock = std::chrono::steady_clock;
    const auto sec = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    const auto t0 = Clock::now();
    const int N = mesh_.num_leaves();
    const CellIndicator::Result ev = ind_.evaluate(mesh_, mesh_.U, eos_);
    last_xi = ev.xi;
    std::vector<char> refine(static_cast<size_t>(N), 0), coarsen(static_cast<size_t>(N), 0);
    for (int c = 0; c < N; ++c) {
        refine[c] = mesh_.level[c] < mesh_.max_level && last_xi[c] >= ind_.split;
        // a group is joined on the xi of its parent cell at the parent level (the children's own xi with parent_join off)
        coarsen[c] = mesh_.level[c] > 0 && (ind_.parent_join ? ev.xi_parent[c] : last_xi[c]) < ind_.join;
        if (in_body_band(c)) {
            refine[c] = mesh_.level[c] < mesh_.max_level;
            coarsen[c] = 0;
        }
    }
    const auto t1 = Clock::now();
    const int before = mesh_.num_leaves();
    long long nr = 0;
    for (int c = 0; c < N; ++c) nr += refine[c];
    const bool changed = mesh_.adapt(refine, coarsen);
    n_refined += nr;
    n_joined += std::max(0, before + 3 * static_cast<int>(nr) - mesh_.num_leaves()) / 3;   // each refined leaf adds 3; each join removes 3
    const auto t2 = Clock::now();
    if (changed) engine_.build(mesh_);
    const auto t3 = Clock::now();
    t_indicator += sec(t0, t1);
    t_adapt += sec(t1, t2);
    t_build += sec(t2, t3);
    return changed;
}

int CellAdaptor::init_adapt(const std::function<void(CellMesh&)>& initial) {
    int passes = 0;
    for (; passes < mesh_.max_level + 1; ++passes) {
        initial(mesh_);
        if (!remesh()) {
            ++passes;
            break;
        }
    }
    initial(mesh_);
    return passes;
}

}   // namespace fv
