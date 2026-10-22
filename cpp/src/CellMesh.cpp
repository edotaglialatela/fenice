#include "fv/CellMesh.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace fv {

namespace {

uint64_t spread_bits(uint32_t v) {
    uint64_t x = v;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFull;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FFull;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0Full;
    x = (x | (x << 2)) & 0x3333333333333333ull;
    x = (x | (x << 1)) & 0x5555555555555555ull;
    return x;
}

constexpr int kKeyBits = 29;

uint64_t pack(int l, int i, int j) {
    return (static_cast<uint64_t>(l) << (2 * kKeyBits)) | (static_cast<uint64_t>(i) << kKeyBits) |
           static_cast<uint64_t>(j);
}

// Sort key of a leaf inside the whole list: level first, Morton key second.
struct SortKey {
    int level;
    uint64_t morton;
    bool operator<(const SortKey& o) const { return level != o.level ? level < o.level : morton < o.morton; }
};

}   // namespace

uint64_t morton_key(uint32_t i, uint32_t j) { return spread_bits(i) | (spread_bits(j) << 1); }

CellMesh::CellMesh(double Lx_, double Ly_, int nx_, int ny_, int max_level_, bool per_x, bool per_y, double x0_,
                   double y0_)
    : Lx(Lx_), Ly(Ly_), nx(nx_), ny(ny_), max_level(max_level_), periodic_x(per_x), periodic_y(per_y), x0(x0_),
      y0(y0_) {
    if (nx < 1 || ny < 1) throw std::invalid_argument("CellMesh: the base mesh needs at least one cell per axis");
    if (max_level < 0 || max_level > 20) throw std::invalid_argument("CellMesh: max_level must be in [0, 20]");
    if ((static_cast<long long>(std::max(nx, ny)) << max_level) >= (1ll << kKeyBits))
        throw std::invalid_argument("CellMesh: the finest level has too many cells per axis");
    dx0_ = Lx / nx;
    dy0_ = Ly / ny;
    level.assign(static_cast<size_t>(nx) * ny, 0);
    ci.resize(level.size());
    cj.resize(level.size());
    std::vector<int> order(level.size());
    std::vector<uint64_t> key(level.size());
    for (int j = 0, c = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i, ++c) {
            ci[c] = i;
            cj[c] = j;
            key[c] = morton_key(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
        }
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });
    std::vector<int> i2(level.size()), j2(level.size());
    for (size_t c = 0; c < order.size(); ++c) {
        i2[c] = ci[order[c]];
        j2[c] = cj[order[c]];
    }
    ci.swap(i2);
    cj.swap(j2);
    rebuild();
}

int CellMesh::find(int l, int i, int j) const {
    if (l < 0 || l > max_level) return -1;
    const auto it = index_.find(pack(l, i, j));
    return it == index_.end() ? -1 : it->second;
}

int CellMesh::covering(int l, int i, int j) const {
    for (; l >= 0; --l, i >>= 1, j >>= 1) {
        const int c = find(l, i, j);
        if (c >= 0) return c;
    }
    return -1;
}

bool CellMesh::wrap(int l, int& i, int& j) const {
    const int NX = nx_at(l), NY = ny_at(l);
    if (i < 0 || i >= NX) {
        if (!periodic_x) return false;
        i = (i % NX + NX) % NX;
    }
    if (j < 0 || j >= NY) {
        if (!periodic_y) return false;
        j = (j % NY + NY) % NY;
    }
    return true;
}

CellNeighbour CellMesh::neighbour(int leaf, Face side) const {
    const int s = static_cast<int>(side);
    const int b = side_ptr[4 * leaf + s], e = side_ptr[4 * leaf + s + 1];
    CellNeighbour n;
    const auto other = [&](int f) { return faces[f].left == leaf ? faces[f].right : faces[f].left; };
    if (faces[side_face[b]].kind == CellFaceKind::Boundary) return n;
    n.a = other(side_face[b]);
    if (e - b == 2) {
        n.kind = CellNeighbour::Kind::Finer;
        n.b = other(side_face[b + 1]);
    } else {
        n.kind = level[n.a] == level[leaf] ? CellNeighbour::Kind::Same : CellNeighbour::Kind::Coarser;
    }
    return n;
}

void CellMesh::rebuild() {
    const int N = num_leaves();
    // lookup table and level runs
    index_.clear();
    index_.reserve(static_cast<size_t>(N) * 2);
    level_begin.assign(static_cast<size_t>(max_level) + 2, N);
    for (int c = N - 1; c >= 0; --c) {
        index_[pack(level[c], ci[c], cj[c])] = c;
        level_begin[level[c]] = c;
    }
    for (int l = max_level; l >= 0; --l) level_begin[l] = std::min(level_begin[l], level_begin[l + 1]);

    // faces: each leaf creates the faces it is responsible for
    //   same level  -> the leaf on the lower side creates it (its Right / Top side)
    //   coarser nbr -> the fine leaf creates it
    //   finer nbr   -> nothing (the finer leaves create theirs)
    //   no nbr      -> a boundary face
    faces.clear();
    faces.reserve(static_cast<size_t>(N) * 2 + 8);
    for (int c = 0; c < N; ++c) {
        const int l = level[c];
        for (int s = 0; s < 4; ++s) {
            const int axis = s / 2, dir = (s & 1) ? 1 : -1;
            int ni = ci[c] + (axis == 0 ? dir : 0), nj = cj[c] + (axis == 1 ? dir : 0);
            CellFace f;
            f.level = l;
            f.axis = static_cast<uint8_t>(axis);
            f.tpos = axis == 0 ? cj[c] : ci[c];
            f.length = axis == 0 ? dy(l) : dx(l);
            if (!wrap(l, ni, nj)) {
                f.kind = CellFaceKind::Boundary;
                f.side = static_cast<Face>(s);
                f.left = dir > 0 ? c : -1;
                f.right = dir > 0 ? -1 : c;
                faces.push_back(f);
                continue;
            }
            const int same = find(l, ni, nj);
            if (same >= 0) {
                if (dir > 0) {
                    f.left = c;
                    f.right = same;
                    faces.push_back(f);
                }
                continue;
            }
            const int coarse = l > 0 ? find(l - 1, ni >> 1, nj >> 1) : -1;
            if (coarse >= 0) {
                f.kind = CellFaceKind::CoarseFine;
                f.left = dir > 0 ? c : coarse;
                f.right = dir > 0 ? coarse : c;
                faces.push_back(f);
            }
        }
    }
    face_level_begin.assign(static_cast<size_t>(max_level) + 2, static_cast<int>(faces.size()));
    for (int f = static_cast<int>(faces.size()) - 1; f >= 0; --f) face_level_begin[faces[f].level] = f;
    for (int l = max_level; l >= 0; --l) face_level_begin[l] = std::min(face_level_begin[l], face_level_begin[l + 1]);

    // cell -> face lists by side
    side_ptr.assign(static_cast<size_t>(4) * N + 1, 0);
    const auto side_of_left = [](const CellFace& f) { return f.axis == 0 ? 1 : 3; };    // Right / Top of `left`
    const auto side_of_right = [](const CellFace& f) { return f.axis == 0 ? 0 : 2; };   // Left / Bottom of `right`
    for (const CellFace& f : faces) {
        if (f.left >= 0) ++side_ptr[4 * f.left + side_of_left(f) + 1];
        if (f.right >= 0) ++side_ptr[4 * f.right + side_of_right(f) + 1];
    }
    for (size_t k = 0; k + 1 < side_ptr.size(); ++k) {
        const int n = side_ptr[k + 1];
        if (n < 1 || n > 2) throw std::logic_error("CellMesh: a cell side must carry one or two faces (2:1 violated?)");
        side_ptr[k + 1] += side_ptr[k];
    }
    side_face.assign(static_cast<size_t>(side_ptr.back()), -1);
    std::vector<int> fill(side_ptr.begin(), side_ptr.end() - 1);
    for (int fi = 0; fi < static_cast<int>(faces.size()); ++fi) {
        const CellFace& f = faces[fi];
        if (f.left >= 0) side_face[fill[4 * f.left + side_of_left(f)]++] = fi;
        if (f.right >= 0) side_face[fill[4 * f.right + side_of_right(f)]++] = fi;
    }
    for (int k = 0; k < 4 * N; ++k)
        if (side_ptr[k + 1] - side_ptr[k] == 2 && faces[side_face[side_ptr[k]]].tpos > faces[side_face[side_ptr[k] + 1]].tpos)
            std::swap(side_face[side_ptr[k]], side_face[side_ptr[k] + 1]);
}

bool CellMesh::adapt(const std::vector<char>& refine_in, const std::vector<char>& coarsen_in) {
    const int N = num_leaves();
    if ((!refine_in.empty() && static_cast<int>(refine_in.size()) != N) ||
        (!coarsen_in.empty() && static_cast<int>(coarsen_in.size()) != N))
        throw std::invalid_argument("CellMesh::adapt: the flags need one entry per leaf");
    std::vector<char> R(static_cast<size_t>(N), 0);
    for (int c = 0; c < N; ++c) R[c] = !refine_in.empty() && refine_in[c] && level[c] < max_level;

    // Ripple: a leaf that becomes level l+1 needs every corner neighbour at level >= l.  The mesh is balanced
    // before the call, so a lower neighbour is exactly one level lower and must be refined as well.
    std::vector<int> work;
    for (int c = 0; c < N; ++c)
        if (R[c]) work.push_back(c);
    while (!work.empty()) {
        const int a = work.back();
        work.pop_back();
        const int l = level[a];
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di) {
                if (di == 0 && dj == 0) continue;
                int i = ci[a] + di, j = cj[a] + dj;
                if (!wrap(l, i, j)) continue;
                const int b = covering(l, i, j);
                if (b >= 0 && level[b] < l && !R[b]) {
                    R[b] = 1;
                    work.push_back(b);
                }
            }
    }

    // Merges: sibling groups anchored at their (even, even) member
    std::vector<char> merge_anchor(static_cast<size_t>(N), 0), absorbed(static_cast<size_t>(N), 0);
    if (!coarsen_in.empty()) {
        for (int a = 0; a < N; ++a) {
            const int l = level[a];
            if (l == 0 || (ci[a] & 1) || (cj[a] & 1)) continue;
            int sib[4] = {a, find(l, ci[a] + 1, cj[a]), find(l, ci[a], cj[a] + 1), find(l, ci[a] + 1, cj[a] + 1)};
            bool ok = true;
            for (int s : sib) ok = ok && s >= 0 && coarsen_in[s] && !R[s];
            if (!ok) continue;
            // the parent (level l-1) must stay within one level of every corner neighbour block
            const int pi = ci[a] >> 1, pj = cj[a] >> 1;
            for (int dj = -1; dj <= 1 && ok; ++dj)
                for (int di = -1; di <= 1 && ok; ++di) {
                    if (di == 0 && dj == 0) continue;
                    int ni = pi + di, nj = pj + dj;
                    if (!wrap(l - 1, ni, nj)) continue;
                    if (covering(l - 1, ni, nj) >= 0) continue;   // same level or coarser: no conflict
                    // the neighbour block is subdivided: the children touching the parent must be leaves
                    // that stay at level l
                    for (int oj = 0; oj < 2; ++oj)
                        for (int oi = 0; oi < 2; ++oi) {
                            if ((di < 0 && oi == 0) || (di > 0 && oi == 1) || (dj < 0 && oj == 0) || (dj > 0 && oj == 1))
                                continue;
                            const int k = find(l, 2 * ni + oi, 2 * nj + oj);
                            if (k < 0 || R[k]) ok = false;
                        }
                }
            if (!ok) continue;
            merge_anchor[a] = 1;
            for (int s = 1; s < 4; ++s) absorbed[sib[s]] = 1;
        }
    }

    bool any = false;
    for (int c = 0; c < N && !any; ++c) any = R[c] || merge_anchor[c];
    if (!any) return false;

    // New leaf list
    struct NewLeaf {
        int level, i, j;
        State u;
    };
    const bool has_u = static_cast<int>(U.size()) == N;
    std::vector<NewLeaf> out;
    out.reserve(static_cast<size_t>(N) + 3 * std::count(R.begin(), R.end(), char(1)));
    for (int c = 0; c < N; ++c) {
        if (absorbed[c]) continue;
        const int l = level[c], i = ci[c], j = cj[c];
        if (R[c]) {
            for (int oj = 0; oj < 2; ++oj)
                for (int oi = 0; oi < 2; ++oi) out.push_back({l + 1, 2 * i + oi, 2 * j + oj, has_u ? U[c] : State{}});
        } else if (merge_anchor[c]) {
            State m{};
            if (has_u) {
                const State &a = U[c], &b = U[find(l, i + 1, j)], &cc = U[find(l, i, j + 1)],
                            &d = U[find(l, i + 1, j + 1)];
                for (int k = 0; k < kNVar; ++k) m[k] = ((a[k] + b[k]) + (cc[k] + d[k])) * 0.25;
            }
            out.push_back({l - 1, i >> 1, j >> 1, m});
        } else {
            out.push_back({l, i, j, has_u ? U[c] : State{}});
        }
    }
    std::vector<SortKey> keys(out.size());
    for (size_t k = 0; k < out.size(); ++k)
        keys[k] = {out[k].level, morton_key(static_cast<uint32_t>(out[k].i), static_cast<uint32_t>(out[k].j))};
    std::vector<int> order(out.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] < keys[b]; });
    level.resize(out.size());
    ci.resize(out.size());
    cj.resize(out.size());
    if (has_u) U.resize(out.size());
    for (size_t k = 0; k < out.size(); ++k) {
        const NewLeaf& n = out[order[k]];
        level[k] = n.level;
        ci[k] = n.i;
        cj[k] = n.j;
        if (has_u) U[k] = n.u;
    }
    rebuild();
    return true;
}

}   // namespace fv
