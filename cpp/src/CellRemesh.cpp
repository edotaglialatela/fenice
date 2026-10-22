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
    std::vector<int> grid;       // dense table j * NX + i -> cell (-1: none), when the level is small enough; else empty
    int NX = 0;
    // the quantities the tests look at, per cell (the same cons_to_prim as the pair test would do)
    std::vector<double> rho, p, speed;

    int find(int i, int j) const {
        if (!grid.empty()) return grid[static_cast<size_t>(j) * NX + i];
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

void CellIndicator::diffuse_cells(std::vector<double>& xi, const std::vector<int>& nbr) const {
    const int n = static_cast<int>(xi.size());
    std::vector<double> add(xi.size());
    for (int it = 0; it < diffusion; ++it) {
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
        for (int c = 0; c < n; ++c) {
            double a = 0.0;
            for (int s = 0; s < 4; ++s) {
                const int b = nbr[static_cast<size_t>(4) * c + s];
                if (b == -2) continue;
                // a same-level pair counts for both cells if either is above the trigger; a cell against a coarser leaf only
                // for itself. Every addend is the same number, so the order of the sum cannot change its value.
                if (b == -1 ? xi[c] > diffusion_trigger : (xi[c] > diffusion_trigger || xi[b] > diffusion_trigger))
                    a += diffusion_coeff;
            }
            add[c] = a;
        }
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
        for (int c = 0; c < n; ++c) xi[c] += add[c];
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
            const int np = D.size() / 4;
            pkey.resize(static_cast<size_t>(np));
            pi.resize(static_cast<size_t>(np));
            pj.resize(static_cast<size_t>(np));
            pval.resize(static_cast<size_t>(np));
#pragma omp parallel for schedule(static) if (np >= kOmpMinWork)
            for (int q = 0; q < np; ++q) {   // four consecutive children share a parent
                const int c = 4 * q;
                pkey[q] = D.key[c] >> 2;
                pi[q] = D.ci[c] >> 1;
                pj[q] = D.cj[c] >> 1;
                State s;
                for (int k = 0; k < kNVar; ++k)
                    s[k] = ((D.val[c][k] + D.val[c + 1][k]) + (D.val[c + 2][k] + D.val[c + 3][k])) * 0.25;
                pval[q] = s;
            }
        }
        // merge with the leaves of level l (already in Morton order in the leaf arrays)
        int a = m.level_begin[l], ea = m.level_begin[l + 1];
        size_t b = 0;
        const size_t total = static_cast<size_t>(ea - a) + pkey.size();
        L.key.reserve(total);
        L.ci.reserve(total);
        L.cj.reserve(total);
        L.val.reserve(total);
        L.leaf.reserve(total);
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
        L.NX = m.nx_at(l);
        const long long cells = static_cast<long long>(L.NX) * m.ny_at(l);
        if (cells <= 4ll * L.size() + 65536) {
            L.grid.assign(static_cast<size_t>(cells), -1);
            for (int c = 0; c < L.size(); ++c) L.grid[static_cast<size_t>(L.cj[c]) * L.NX + L.ci[c]] = c;
        }
        const int n = L.size();
        L.rho.resize(static_cast<size_t>(n));
        L.p.resize(static_cast<size_t>(n));
        L.speed.resize(static_cast<size_t>(n));
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
        for (int c = 0; c < n; ++c) {
            L.rho[c] = L.val[c][0];
            if (use_pressure || use_velocity) {
                const State w = eos.cons_to_prim(L.val[c]);
                L.p[c] = w[3];
                L.speed[c] = std::sqrt(w[1] * w[1] + w[2] * w[2] + w[4] * w[4]);
            }
        }
    }
    // the cell of a coarser leaf covering a position of level l: (level, cell index), or level -1
    const auto coarser_leaf = [&](int l, int i, int j, int& kl) {
        for (int k = l - 1; k >= 0; --k) {
            i >>= 1;
            j >>= 1;
            const LevelCells& K = lev[static_cast<size_t>(k)];
            const int idx = K.find(i, j);
            if (idx >= 0 && K.leaf[idx] >= 0) {
                kl = k;
                return idx;
            }
        }
        kl = -1;
        return -1;
    };
    // 2) xi of every level
    std::vector<std::vector<double>> xi(static_cast<size_t>(M) + 1);
    for (int l = 0; l <= M; ++l) {
        const LevelCells& L = lev[static_cast<size_t>(l)];
        std::vector<double>& x = xi[static_cast<size_t>(l)];
        const int n = L.size();
        x.assign(static_cast<size_t>(n), 0.0);
        // per cell and side: the same-level cell across it (>= 0), -1 against a coarser leaf, -2 at an edge
        std::vector<int> nbr(static_cast<size_t>(4) * n, -2);
        const int NX = m.nx_at(l), NY = m.ny_at(l);
        const auto test = [&](double ca, double cb, double floor_value) {
            double vmin = std::min(std::abs(ca), std::abs(cb));
            if (vmin < value_floor) vmin = floor_value;
            return std::abs(cb - ca) / vmin >= criterion;
        };
        const auto flag = [&](const LevelCells& A, int ia, const LevelCells& B, int ib) {
            bool hit = false;
            if (use_density) hit = hit || test(A.rho[ia], B.rho[ib], value_floor);
            if (use_pressure) hit = hit || test(A.p[ia], B.p[ib], value_floor);
            if (use_velocity) hit = hit || test(A.speed[ia], B.speed[ib], velocity_floor);
            return hit;
        };
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
        for (int c = 0; c < n; ++c) {
            bool hit = false;
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
                    nbr[static_cast<size_t>(4) * c + s] = nb;
                    hit = hit || flag(L, c, L, nb);
                } else {
                    int kl;
                    const int idx = coarser_leaf(l, i, j, kl);
                    if (idx < 0) continue;   // cannot happen in a complete tree
                    nbr[static_cast<size_t>(4) * c + s] = -1;
                    hit = hit || flag(L, c, lev[static_cast<size_t>(kl)], idx);
                }
            }
            if (hit) x[c] = 1.0;
        }
        diffuse_cells(x, nbr);
    }
    // 3) per leaf: its own xi, and the xi of its parent at the parent level
    Result r;
    r.xi.assign(static_cast<size_t>(N), 0.0);
    r.xi_parent.assign(static_cast<size_t>(N), 1e300);
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
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
