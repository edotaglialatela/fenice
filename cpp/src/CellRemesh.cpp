#include "fv/CellRemesh.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <memory>

#include "fv/MemoryBytes.hpp"
#include "fv/ParallelScan.hpp"

namespace fv {

namespace {

// The cells of one level of the hierarchy: the leaves of that level and the split ancestors of the deeper leaves (whose
// value is the mean of their four children), sorted by Morton key. A cell at level l exists wherever a leaf of level >= l
// lies; coarser leaves are not cells of this level.
struct LevelCells {
    std::vector<uint64_t> key;   // morton_key(i, j), ascending
    std::vector<int> ci, cj;
    std::unique_ptr<State[]> val;   // conservative state
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
            // On a whole mesh the cells one level down come in groups of four consecutive siblings. On a partial one (the leaves
            // a rank needs for its own indicator) a group may be incomplete at the edge: it makes no parent.
            bool whole = D.size() % 4 == 0;
            for (int q = 0; whole && q < D.size() / 4; ++q) whole = (D.key[4 * q] >> 2) == (D.key[4 * q + 3] >> 2);
            std::vector<int> first;   // first cell of every complete group (partial mesh only)
            if (!whole)
                for (int c = 0; c < D.size();) {
                    int e = c + 1;
                    while (e < D.size() && (D.key[e] >> 2) == (D.key[c] >> 2)) ++e;
                    if (e - c == 4) first.push_back(c);
                    c = e;
                }
            const int np = whole ? D.size() / 4 : static_cast<int>(first.size());
            pkey.resize(static_cast<size_t>(np));
            pi.resize(static_cast<size_t>(np));
            pj.resize(static_cast<size_t>(np));
            pval.resize(static_cast<size_t>(np));
#pragma omp parallel for schedule(static) if (np >= kOmpMinWork)
            for (int q = 0; q < np; ++q) {   // four consecutive children share a parent
                const int c = whole ? 4 * q : first[static_cast<size_t>(q)];
                pkey[q] = D.key[c] >> 2;
                pi[q] = D.ci[c] >> 1;
                pj[q] = D.cj[c] >> 1;
                State s;
                for (int k = 0; k < kNVar; ++k)
                    s[k] = ((D.val[c][k] + D.val[c + 1][k]) + (D.val[c + 2][k] + D.val[c + 3][k])) * 0.25;
                pval[q] = s;
            }
        }
        // merge with the leaves of level l (already in Morton order in the leaf arrays): the sources of the cells in order,
        // leaf a >= 0 or parent -1 - b (distinct keys)
        const int la = m.level_begin[l], na = m.level_begin[l + 1] - la, nb = static_cast<int>(pkey.size()), n = na + nb;
        std::vector<uint64_t> lkey(static_cast<size_t>(na));
        std::vector<int> sa(static_cast<size_t>(na)), sb(static_cast<size_t>(nb)), src(static_cast<size_t>(n));
#pragma omp parallel for schedule(static) if (na >= kOmpMinWork)
        for (int a = 0; a < na; ++a) {
            lkey[a] = morton_key(static_cast<uint32_t>(m.ci[la + a]), static_cast<uint32_t>(m.cj[la + a]));
            sa[a] = a;
        }
        for (int b = 0; b < nb; ++b) sb[b] = -1 - b;
        const auto key_of = [&](int x) { return x >= 0 ? lkey[x] : pkey[-1 - x]; };
        parallel_merge(sa.data(), na, sb.data(), nb, src.data(), [&](int x, int y) { return key_of(x) < key_of(y); });
        L.key.resize(static_cast<size_t>(n));
        L.ci.resize(static_cast<size_t>(n));
        L.cj.resize(static_cast<size_t>(n));
        L.val.reset(new State[static_cast<size_t>(n) + 1]);
        L.leaf.resize(static_cast<size_t>(n));
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
        for (int k = 0; k < n; ++k) {
            const int x = src[k];
            L.key[k] = key_of(x);
            if (x >= 0) {
                L.ci[k] = m.ci[la + x];
                L.cj[k] = m.cj[la + x];
                L.val[k] = U[la + x];
                L.leaf[k] = la + x;
            } else {
                L.ci[k] = pi[-1 - x];
                L.cj[k] = pj[-1 - x];
                L.val[k] = pval[-1 - x];
                L.leaf[k] = -1;
            }
        }
        L.NX = m.nx_at(l);
        const long long cells = static_cast<long long>(L.NX) * m.ny_at(l);
        if (cells <= 4ll * L.size() + 65536) {
            const int ncell = static_cast<int>(cells);
            L.grid.resize(static_cast<size_t>(ncell));
#pragma omp parallel for schedule(static) if (ncell >= kOmpMinWork)
            for (int k = 0; k < ncell; ++k) L.grid[k] = -1;
#pragma omp parallel for schedule(static) if (n >= kOmpMinWork)
            for (int c = 0; c < n; ++c) L.grid[static_cast<size_t>(L.cj[c]) * L.NX + L.ci[c]] = c;
        }
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
        // (a leaf at the edge of a partial mesh may have no parent cell: its values are not used)
        const int own = L.find(m.ci[c], m.cj[c]);
        r.xi[c] = own >= 0 ? xi[static_cast<size_t>(l)][static_cast<size_t>(own)] : 0.0;
        if (l > 0) {
            const LevelCells& P = lev[static_cast<size_t>(l) - 1];
            const int par = P.find(m.ci[c] >> 1, m.cj[c] >> 1);
            if (par >= 0) r.xi_parent[c] = xi[static_cast<size_t>(l) - 1][static_cast<size_t>(par)];
        }
    }
    return r;
}

std::vector<double> CellIndicator::xi(const CellMesh& m, const Field& U, const EOS& eos) const {
    return evaluate(m, U, eos).xi;
}

namespace {

// every leaf under the cell (t, i, j) of the whole mesh m (the leaf covering it, or the leaves of the cells it is split into)
void mark_cell(const CellMesh& m, int t, int i, int j, std::vector<char>& need) {
    const int c = m.covering(t, i, j);
    if (c >= 0) {
        need[static_cast<size_t>(c)] = 1;
        return;
    }
    for (int q = 0; q < 4; ++q) mark_cell(m, t + 1, 2 * i + (q & 1), 2 * j + (q >> 1), need);
}

// The cells of a level (NX x NY, periodic axes wrap) within r cells, along each axis, of the seed cells, as a bitmap over the
// box [i0, i1] x [j0, j1] that holds the seeds and the margin; returns the box (the whole axis if the margin wraps).
struct Box {
    int i0, i1, j0, j1;
    int w() const { return i1 - i0 + 1; }
    int h() const { return j1 - j0 + 1; }
};

Box dilated_box(const std::vector<std::array<int, 2>>& seeds, int NX, int NY, int r, bool px, bool py) {
    int lo[2] = {NX, NY}, hi[2] = {-1, -1};
    for (const std::array<int, 2>& s : seeds)
        for (int a = 0; a < 2; ++a) {
            lo[a] = std::min(lo[a], s[static_cast<size_t>(a)]);
            hi[a] = std::max(hi[a], s[static_cast<size_t>(a)]);
        }
    const int N[2] = {NX, NY};
    const bool per[2] = {px, py};
    for (int a = 0; a < 2; ++a) {
        lo[a] -= r;
        hi[a] += r;
        if (per[a] ? (lo[a] < 0 || hi[a] >= N[a]) : false) {
            lo[a] = 0;
            hi[a] = N[a] - 1;
        }
        lo[a] = std::max(lo[a], 0);
        hi[a] = std::min(hi[a], N[a] - 1);
    }
    return {lo[0], hi[0], lo[1], hi[1]};
}

}   // namespace

void CellIndicator::region(const CellMesh& m, const std::vector<int>& leaves, std::vector<char>& need) const {
    const int M = m.max_level, r = diffusion + 1;
    // per level t the cells whose neighbourhood matters: the leaves of level t and the parents of the leaves of level t + 1
    std::vector<std::vector<std::array<int, 2>>> seeds(static_cast<size_t>(M) + 1);
    for (const int c : leaves) {
        const int l = m.level[static_cast<size_t>(c)];
        seeds[static_cast<size_t>(l)].push_back({m.ci[static_cast<size_t>(c)], m.cj[static_cast<size_t>(c)]});
        if (l > 0) seeds[static_cast<size_t>(l) - 1].push_back({m.ci[static_cast<size_t>(c)] >> 1, m.cj[static_cast<size_t>(c)] >> 1});
    }
    for (int t = 0; t <= M; ++t) {
        const std::vector<std::array<int, 2>>& sd = seeds[static_cast<size_t>(t)];
        if (sd.empty()) continue;
        const int NX = m.nx_at(t), NY = m.ny_at(t);
        const Box b = dilated_box(sd, NX, NY, r, m.periodic_x, m.periodic_y);
        const int w = b.w(), h = b.h();
        const bool wrap_x = m.periodic_x && w == NX, wrap_y = m.periodic_y && h == NY;
        std::vector<char> g(static_cast<size_t>(w) * h, 0), tmp(g.size(), 0);
        for (const std::array<int, 2>& s : sd) g[static_cast<size_t>(s[1] - b.j0) * w + (s[0] - b.i0)] = 1;
        // dilate along x (shifted copies of every row), then along y
        for (int j = 0; j < h; ++j) {
            const char* in = &g[static_cast<size_t>(j) * w];
            char* out = &tmp[static_cast<size_t>(j) * w];
            for (int d = -r; d <= r; ++d)
                for (int i = 0; i < w; ++i) {
                    int k = i + d;
                    if (k < 0 || k >= w) {
                        if (!wrap_x) continue;
                        k = (k + w) % w;
                    }
                    if (in[k]) out[i] = 1;
                }
        }
        std::fill(g.begin(), g.end(), 0);
        for (int j = 0; j < h; ++j)
            for (int d = -r; d <= r; ++d) {
                int k = j + d;
                if (k < 0 || k >= h) {
                    if (!wrap_y) continue;
                    k = (k + h) % h;
                }
                const char* in = &tmp[static_cast<size_t>(k) * w];
                char* out = &g[static_cast<size_t>(j) * w];
                for (int i = 0; i < w; ++i) out[i] = out[i] | in[i];
            }
        for (int j = 0; j < h; ++j)
            for (int i = 0; i < w; ++i)
                if (g[static_cast<size_t>(j) * w + i]) mark_cell(m, t, b.i0 + i, b.j0 + j, need);
    }
}

double CellAdaptor::band_bytes() const { return container_bytes(band_); }

double CellAdaptor::xi_bytes() const { return container_bytes(last_xi); }

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

bool CellAdaptor::in_body_band(int c) { return in_body_band(mesh_.level[c], mesh_.ci[c], mesh_.cj[c]); }

bool CellAdaptor::in_body_band(int level, int ci, int cj) {
    if (!body_) return false;
    if (band_.empty()) build_band();
    const int M = mesh_.max_level, NX = mesh_.nx_at(M), r = 1 << (M - level);
    for (int j = cj * r; j < (cj + 1) * r; ++j)
        for (int i = ci * r; i < (ci + 1) * r; ++i)
            if (band_[static_cast<size_t>(j) * NX + i]) return true;
    return false;
}

void CellAdaptor::decide(int max_level, int level, int i, int j, double xi, double xi_parent, char& refine, char& coarsen) {
    refine = level < max_level && xi >= ind_.split;
    // a group is joined on the xi of its parent cell at the parent level (the children's own xi with parent_join off)
    coarsen = level > 0 && (ind_.parent_join ? xi_parent : xi) < ind_.join;
    if (in_body_band(level, i, j)) {
        refine = level < max_level;
        coarsen = 0;
    }
}

void CellAdaptor::flags(const CellMesh& m, const Field& U, const std::vector<int>& which, std::vector<char>& refine,
                        std::vector<char>& coarsen) {
    const CellIndicator::Result ev = ind_.evaluate(m, U, eos_);
    if (body_ && band_.empty()) build_band();   // decide only reads it in the loop
    const int n = static_cast<int>(which.size());
    refine.assign(static_cast<size_t>(n), 0);
    coarsen.assign(static_cast<size_t>(n), 0);
    for (int k = 0; k < n; ++k) {
        const int c = which[static_cast<size_t>(k)];
        decide(m.max_level, m.level[static_cast<size_t>(c)], m.ci[static_cast<size_t>(c)], m.cj[static_cast<size_t>(c)],
               ev.xi[static_cast<size_t>(c)], ev.xi_parent[static_cast<size_t>(c)], refine[static_cast<size_t>(k)],
               coarsen[static_cast<size_t>(k)]);
    }
}

bool CellAdaptor::remesh() {
    using Clock = std::chrono::steady_clock;
    const auto sec = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    const auto t0 = Clock::now();
    const int N = mesh_.num_leaves();
    CellIndicator::Result ev = ind_.evaluate(mesh_, mesh_.U, eos_);
    last_xi.swap(ev.xi);
    std::vector<char> refine(static_cast<size_t>(N), 0), coarsen(static_cast<size_t>(N), 0);
    if (body_ && band_.empty()) build_band();   // in_body_band only reads it in the loop
    long long nr = 0;
#pragma omp parallel for schedule(static) reduction(+ : nr) if (N >= kOmpMinWork)
    for (int c = 0; c < N; ++c) {
        decide(mesh_.max_level, mesh_.level[c], mesh_.ci[c], mesh_.cj[c], last_xi[c], ev.xi_parent[c], refine[c], coarsen[c]);
        nr += refine[c];
    }
    const auto t1 = Clock::now();
    const int before = mesh_.num_leaves();
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
