#include "fv/CellMesh.hpp"

#include <algorithm>
#include <cstdlib>
#include <array>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "fv/MemoryBytes.hpp"
#include "fv/ParallelScan.hpp"

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
    if (l < 0 || l > max_level || i < 0 || j < 0 || i >= nx_at(l) || j >= ny_at(l)) return -1;
    const std::vector<int>& g = grid_[static_cast<size_t>(l)];
    if (!g.empty()) return g[static_cast<size_t>(j) * nx_at(l) + i];
    const uint64_t k = morton_key(static_cast<uint32_t>(i), static_cast<uint32_t>(j));
    const auto b = mkey_.begin() + level_begin[l], e = mkey_.begin() + level_begin[l + 1];
    const auto it = std::lower_bound(b, e, k);
    return (it != e && *it == k) ? static_cast<int>(it - mkey_.begin()) : -1;
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
    if (faceless_) throw std::logic_error("CellMesh::neighbour: a faceless mesh has no face lists");
    const int s = static_cast<int>(side);
    const int b = side_ptr[4 * leaf + s], e = side_ptr[4 * leaf + s + 1];
    CellNeighbour n;
    if (b == e) {   // a partial mesh: this side looks outside what the rank keeps
        n.kind = CellNeighbour::Kind::Outside;
        return n;
    }
    const auto other = [&](int f) { return faces[f].left == leaf ? faces[f].right : faces[f].left; };
    if (faces[side_face[b]].kind == CellFaceKind::Boundary) return n;
    n.a = other(side_face[b]);
    if (e - b == 2) {
        n.kind = CellNeighbour::Kind::Finer;
        n.b = other(side_face[b + 1]);
        if (n.a == kOutside || n.b == kOutside) n.kind = CellNeighbour::Kind::Outside;
    } else if (n.a == kOutside) {
        n.kind = CellNeighbour::Kind::Outside;
    } else {
        n.kind = level[n.a] == level[leaf] ? CellNeighbour::Kind::Same : CellNeighbour::Kind::Coarser;
    }
    return n;
}

void CellMesh::build_lookups() {
    const int N = num_leaves();
    // lookup table and level runs
    level_begin.assign(static_cast<size_t>(max_level) + 2, N);
    for (int c = N - 1; c >= 0; --c) level_begin[level[c]] = c;
    for (int l = max_level; l >= 0; --l) level_begin[l] = std::min(level_begin[l], level_begin[l + 1]);
    mkey_.resize(static_cast<size_t>(N));
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
    for (int c = 0; c < N; ++c) mkey_[c] = morton_key(static_cast<uint32_t>(ci[c]), static_cast<uint32_t>(cj[c]));
    grid_.resize(static_cast<size_t>(max_level) + 1);
    for (int l = 0; l <= max_level; ++l) {
        const long long cells = static_cast<long long>(nx_at(l)) * ny_at(l);
        std::vector<int>& g = grid_[static_cast<size_t>(l)];
        if (cells > 4ll * N + 65536) {
            std::vector<int>().swap(g);
            continue;
        }
        g.resize(static_cast<size_t>(cells));   // the table of the last rebuild, overwritten
        const int nc = static_cast<int>(cells), cb = level_begin[l], ce = level_begin[l + 1], NX = nx_at(l);
#pragma omp parallel for schedule(static) if (nc >= kOmpMinWork)
        for (int k = 0; k < nc; ++k) g[static_cast<size_t>(k)] = -1;
#pragma omp parallel for schedule(static) if (ce - cb >= kOmpMinWork)
        for (int c = cb; c < ce; ++c) g[static_cast<size_t>(cj[c]) * NX + ci[c]] = c;
    }
}

void CellMesh::set_faceless(bool on) {
    if (faceless_ == on) return;
    faceless_ = on;
    rebuild();
}

void CellMesh::rebuild(bool incremental) {
    const int N = num_leaves();
    ++version;
    partial = false;
    static const bool force_full = std::getenv("FENICE_FULL_REBUILD") != nullptr;
    if (!incremental || force_full) prev_leaf.clear();
    topo_dirty.clear();
    prev_face.clear();
    build_lookups();
    if (faceless_) {
        for (std::vector<int>* a : {&side_ptr, &side_face, &face_level_begin, &face_off_}) std::vector<int>().swap(*a);
        std::vector<CellFace>().swap(faces);
        std::vector<CellFace>().swap(face_spare_);
        return;
    }

    // faces: each leaf creates the faces it is responsible for
    //   same level  -> the leaf on the lower side creates it (its Right / Top side)
    //   coarser nbr -> the fine leaf creates it
    //   finer nbr   -> nothing (the finer leaves create theirs)
    //   no nbr      -> a boundary face
    // Each leaf makes at most one face per side; the faces keep the leaf order, so the list is the serial one.
    //   same level  -> the leaf on the lower side creates it (its Right / Top side)
    //   coarser nbr -> the fine leaf creates it;  finer nbr -> nothing;  no nbr -> a boundary face
    const auto make_face = [&](int c, int s, CellFace& f) {
        const int l = level[c];
        const int axis = s / 2, dir = (s & 1) ? 1 : -1;
        int ni = ci[c] + (axis == 0 ? dir : 0), nj = cj[c] + (axis == 1 ? dir : 0);
        f = CellFace();
        f.level = l;
        f.axis = static_cast<uint8_t>(axis);
        f.tpos = axis == 0 ? cj[c] : ci[c];
        f.length = axis == 0 ? dy(l) : dx(l);
        if (!wrap(l, ni, nj)) {
            f.kind = CellFaceKind::Boundary;
            f.side = static_cast<Face>(s);
            f.left = dir > 0 ? c : -1;
            f.right = dir > 0 ? -1 : c;
            return true;
        }
        const int same = find(l, ni, nj);
        if (same >= 0) {
            if (dir <= 0) return false;
            f.left = c;
            f.right = same;
            return true;
        }
        const int coarse = l > 0 ? find(l - 1, ni >> 1, nj >> 1) : -1;
        if (coarse < 0) return false;
        f.kind = CellFaceKind::CoarseFine;
        f.left = dir > 0 ? c : coarse;
        f.right = dir > 0 ? coarse : c;
        return true;
    };
    // After an adapt: a kept leaf is clean if none of the four squares of its size across its sides holds a new leaf; then its
    // neighbours are the old ones (renumbered) and so are the faces it makes. The leaves keep their relative order, so the copied
    // faces land in the order a full rebuild gives.
    const bool inc = !prev_leaf.empty();
    std::vector<int> prev2new;
    if (inc) {
        const int old_n = static_cast<int>(face_off_.size()) - 1;
        prev2new.assign(static_cast<size_t>(old_n), -1);
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
        for (int c = 0; c < N; ++c)   // prev_leaf is one-to-one: distinct entries
            if (prev_leaf[c] >= 0) prev2new[static_cast<size_t>(prev_leaf[c])] = c;
        // Marked from the new leaves outward: the leaves c whose square of their size across a side holds a new leaf n are the
        // leaves of level l(n) - 2 .. l(n) next to the block of their level that holds n, and the finer leaves on the sides of n
        // (2:1 balance: a square next to c holds leaves of level l(c) + 2 at most, a face neighbour of n is one level finer at most).
        topo_dirty.assign(static_cast<size_t>(N), 0);
        std::vector<int> fresh;   // the new leaves (a few per thousand)
        collect_indices(0, N, [&](int c) { return prev_leaf[c] < 0; }, fresh);
        for (const int n : fresh) {
            topo_dirty[n] = 1;
            const int ln = level[n];
            for (int s = 0; s < 4; ++s) {
                const int axis = s / 2, dir = (s & 1) ? 1 : -1;
                const int di = axis == 0 ? dir : 0, dj = axis == 1 ? dir : 0;
                for (int lc = std::max(0, ln - 2); lc <= ln; ++lc) {
                    int bi = (ci[n] >> (ln - lc)) + di, bj = (cj[n] >> (ln - lc)) + dj;
                    if (!wrap(lc, bi, bj)) continue;
                    const int k = find(lc, bi, bj);
                    if (k >= 0) topo_dirty[k] = 1;
                }
                if (ln == max_level) continue;
                int bi = ci[n] + di, bj = cj[n] + dj;
                if (!wrap(ln, bi, bj)) continue;
                for (int t = 0; t < 2; ++t) {   // the two children of the block across side s that touch n
                    const int oi = axis == 0 ? (dir > 0 ? 0 : 1) : t, oj = axis == 1 ? (dir > 0 ? 0 : 1) : t;
                    const int k = find(ln + 1, 2 * bi + oi, 2 * bj + oj);
                    if (k >= 0) topo_dirty[k] = 1;
                }
            }
        }
    }
    std::vector<int> off(static_cast<size_t>(N) + 1, 0);
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
    for (int c = 0; c < N; ++c) {
        int k = 0;
        if (inc && !topo_dirty[c]) {
            k = face_off_[static_cast<size_t>(prev_leaf[c]) + 1] - face_off_[static_cast<size_t>(prev_leaf[c])];
        } else {
            CellFace f;
            for (int s = 0; s < 4; ++s) k += make_face(c, s, f) ? 1 : 0;
        }
        off[static_cast<size_t>(c) + 1] = k;
    }
    inclusive_scan_inplace(off.data(), N + 1);
    // the face list of the call before last as the buffer: every entry is written below
    std::vector<CellFace> nf;
    nf.swap(face_spare_);
    nf.resize(static_cast<size_t>(off[static_cast<size_t>(N)]));
    if (inc) prev_face.resize(nf.size());
    bool broken = false;
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
    for (int c = 0; c < N; ++c) {
        int k = off[static_cast<size_t>(c)];
        if (inc && !topo_dirty[c]) {
            const int o = prev_leaf[c];
            for (int g = face_off_[static_cast<size_t>(o)]; g < face_off_[static_cast<size_t>(o) + 1]; ++g, ++k) {
                CellFace f = faces[static_cast<size_t>(g)];
                if (f.left >= 0) f.left = prev2new[static_cast<size_t>(f.left)];
                if (f.right >= 0) f.right = prev2new[static_cast<size_t>(f.right)];
                if ((f.left < 0 && f.kind != CellFaceKind::Boundary) || (f.right < 0 && f.kind != CellFaceKind::Boundary)) broken = true;
                nf[static_cast<size_t>(k)] = f;
                prev_face[static_cast<size_t>(k)] = g;
            }
            continue;
        }
        CellFace f;
        for (int s = 0; s < 4; ++s)
            if (make_face(c, s, f)) {
                if (inc) prev_face[static_cast<size_t>(k)] = -1;
                nf[static_cast<size_t>(k++)] = f;
            }
    }
    if (broken) throw std::logic_error("CellMesh: a clean leaf's face lost a leaf (incremental rebuild)");
    faces.swap(nf);
    face_spare_.swap(nf);
    face_off_.swap(off);
    // the faces are non-decreasing in level (a face has the level of the leaf that made it, the finer one)
    face_level_begin.resize(static_cast<size_t>(max_level) + 2);
    for (int l = 0; l <= max_level + 1; ++l)
        face_level_begin[static_cast<size_t>(l)] = static_cast<int>(
            std::partition_point(faces.begin(), faces.end(), [l](const CellFace& f) { return f.level < l; }) - faces.begin());

    // cell -> face lists by side
    const int NF = static_cast<int>(faces.size()), NS = 4 * N;
    side_ptr.resize(static_cast<size_t>(NS) + 1);
#pragma omp parallel for schedule(static) if (NS >= kOmpMinWork)
    for (int k = 0; k <= NS; ++k) side_ptr[static_cast<size_t>(k)] = 0;
    const auto side_of_left = [](const CellFace& f) { return f.axis == 0 ? 1 : 3; };    // Right / Top of `left`
    const auto side_of_right = [](const CellFace& f) { return f.axis == 0 ? 0 : 2; };   // Left / Bottom of `right`
#pragma omp parallel for schedule(static) if (NF >= kOmpMinWork)
    for (int fi = 0; fi < NF; ++fi) {
        const CellFace& f = faces[static_cast<size_t>(fi)];
        if (f.left >= 0) {
            int& n = side_ptr[static_cast<size_t>(4) * f.left + side_of_left(f) + 1];
#pragma omp atomic
            ++n;
        }
        if (f.right >= 0) {
            int& n = side_ptr[static_cast<size_t>(4) * f.right + side_of_right(f) + 1];
#pragma omp atomic
            ++n;
        }
    }
    bool bad_count = false;
#pragma omp parallel for schedule(static) if (NS >= kOmpMinWork)
    for (int k = 1; k <= NS; ++k)
        if (side_ptr[static_cast<size_t>(k)] < 1 || side_ptr[static_cast<size_t>(k)] > 2) bad_count = true;
    if (bad_count) throw std::logic_error("CellMesh: a cell side must carry one or two faces (2:1 violated?)");
    inclusive_scan_inplace(side_ptr.data(), NS + 1);
    // A side with two faces faces two finer leaves, the children 2t and 2t + 1 of its transverse position t: the face of the
    // even one goes first (increasing transverse position), so every face has its own slot and the order is the serial one.
    side_face.resize(static_cast<size_t>(side_ptr.back()));
    const int NSF = static_cast<int>(side_face.size());
#pragma omp parallel for schedule(static) if (NSF >= kOmpMinWork)
    for (int k = 0; k < NSF; ++k) side_face[static_cast<size_t>(k)] = -1;
    const auto slot = [&](int k, const CellFace& f) {
        const int b = side_ptr[static_cast<size_t>(k)];
        return side_ptr[static_cast<size_t>(k) + 1] - b == 2 ? b + (f.tpos & 1) : b;
    };
#pragma omp parallel for schedule(static) if (NF >= kOmpMinWork)
    for (int fi = 0; fi < NF; ++fi) {
        const CellFace& f = faces[static_cast<size_t>(fi)];
        if (f.left >= 0) side_face[static_cast<size_t>(slot(4 * f.left + side_of_left(f), f))] = fi;
        if (f.right >= 0) side_face[static_cast<size_t>(slot(4 * f.right + side_of_right(f), f))] = fi;
    }
    bool bad_slot = false;
#pragma omp parallel for schedule(static) if (NS >= kOmpMinWork)
    for (int k = 0; k < NS; ++k) {
        const int b = side_ptr[static_cast<size_t>(k)], e = side_ptr[static_cast<size_t>(k) + 1];
        if (side_face[static_cast<size_t>(b)] < 0 || side_face[static_cast<size_t>(e) - 1] < 0 ||
            (e - b == 2 && faces[static_cast<size_t>(side_face[static_cast<size_t>(b)])].tpos + 1 !=
                               faces[static_cast<size_t>(side_face[static_cast<size_t>(b) + 1])].tpos))
            bad_slot = true;
    }
    if (bad_slot) throw std::logic_error("CellMesh: the two faces of a side are not two sibling finer leaves");
}

void CellMesh::assign_leaves(const std::vector<int>& lv, const std::vector<int>& ii, const std::vector<int>& jj, Field* state) {
    const size_t n = lv.size();
    if (ii.size() != n || jj.size() != n || (state && state->size() != n))
        throw std::invalid_argument("CellMesh::assign_leaves: the arrays differ in length");
    long long area = 0;   // in cells of the finest level
    for (size_t k = 0; k < n; ++k) {
        if (lv[k] < 0 || lv[k] > max_level || ii[k] < 0 || ii[k] >= nx_at(lv[k]) || jj[k] < 0 || jj[k] >= ny_at(lv[k]))
            throw std::invalid_argument("CellMesh::assign_leaves: a leaf is outside the mesh or deeper than max_level");
        area += 1ll << (2 * (max_level - lv[k]));
    }
    if (area != static_cast<long long>(nx) * ny * (1ll << (2 * max_level)))
        throw std::invalid_argument("CellMesh::assign_leaves: the leaves do not tile the domain");
    std::vector<SortKey> keys(n);
    for (size_t k = 0; k < n; ++k) keys[k] = {lv[k], morton_key(static_cast<uint32_t>(ii[k]), static_cast<uint32_t>(jj[k]))};
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] < keys[b]; });
    for (size_t k = 1; k < n; ++k)
        if (!(keys[order[k - 1]] < keys[order[k]])) throw std::invalid_argument("CellMesh::assign_leaves: a leaf is given twice");
    level.resize(n);
    ci.resize(n);
    cj.resize(n);
    Field reordered;
    if (state) reordered.resize(n);
    for (size_t k = 0; k < n; ++k) {
        level[k] = lv[order[k]];
        ci[k] = ii[order[k]];
        cj[k] = jj[order[k]];
        if (state) reordered[k] = (*state)[static_cast<size_t>(order[k])];
    }
    if (state) U = std::move(reordered);
    else U.clear();
    rebuild();
}

void CellMesh::build_from(const CellMesh& whole, const std::vector<char>& keep_leaf, const Field* whole_state) {
    const int NW = whole.num_leaves();
    if (static_cast<int>(keep_leaf.size()) != NW)
        throw std::invalid_argument("CellMesh::build_from: one flag per leaf of the whole mesh");
    if (whole_state && static_cast<int>(whole_state->size()) != NW)
        throw std::invalid_argument("CellMesh::build_from: one state per leaf of the whole mesh");
    std::vector<int> to_local(static_cast<size_t>(NW), -1), from_whole;
    for (int c = 0; c < NW; ++c)
        if (keep_leaf[static_cast<size_t>(c)]) {
            to_local[static_cast<size_t>(c)] = static_cast<int>(from_whole.size());
            from_whole.push_back(c);
        }
    const int N = static_cast<int>(from_whole.size());
    level.resize(static_cast<size_t>(N));
    ci.resize(static_cast<size_t>(N));
    cj.resize(static_cast<size_t>(N));
    U.assign(static_cast<size_t>(N), State{});   // (the state of a leaf comes from whole_state, or is set by the caller)
    for (int k = 0; k < N; ++k) {
        const size_t c = static_cast<size_t>(from_whole[static_cast<size_t>(k)]);
        level[static_cast<size_t>(k)] = whole.level[c];
        ci[static_cast<size_t>(k)] = whole.ci[c];
        cj[static_cast<size_t>(k)] = whole.cj[c];
        if (whole_state) U[static_cast<size_t>(k)] = (*whole_state)[c];
    }
    for (std::vector<int>* a : {&prev_leaf, &prev_face, &face_off_}) std::vector<int>().swap(*a);
    std::vector<char>().swap(topo_dirty);
    std::vector<CellFace>().swap(face_spare_);
    ++version;
    partial = true;
    build_lookups();
    // The faces the kept leaves make (make_face of rebuild(), the neighbours looked up in the whole mesh), in the order of the
    // leaves; the other end is kOutside if that leaf is not kept. expected[4 k + s]: the faces side s has in the whole mesh (two
    // towards two finer leaves, which make them), so a side is listed only if all of them are there.
    std::vector<CellFace> fs;
    std::vector<char> expected(static_cast<size_t>(4) * N, 1);
    const auto local = [&](int whole_leaf) {
        return to_local[static_cast<size_t>(whole_leaf)] >= 0 ? to_local[static_cast<size_t>(whole_leaf)] : kOutside;
    };
    for (int k = 0; k < N; ++k) {
        const int l = level[static_cast<size_t>(k)];
        for (int s = 0; s < 4; ++s) {
            const int axis = s / 2, dir = (s & 1) ? 1 : -1;
            int ni = ci[static_cast<size_t>(k)] + (axis == 0 ? dir : 0), nj = cj[static_cast<size_t>(k)] + (axis == 1 ? dir : 0);
            CellFace f;
            f.level = l;
            f.axis = static_cast<uint8_t>(axis);
            f.tpos = axis == 0 ? cj[static_cast<size_t>(k)] : ci[static_cast<size_t>(k)];
            f.length = axis == 0 ? dy(l) : dx(l);
            if (!wrap(l, ni, nj)) {
                f.kind = CellFaceKind::Boundary;
                f.side = static_cast<Face>(s);
                f.left = dir > 0 ? k : -1;
                f.right = dir > 0 ? -1 : k;
                fs.push_back(f);
                continue;
            }
            const int same = whole.find(l, ni, nj);
            if (same >= 0) {
                if (dir > 0) {
                    f.left = k;
                    f.right = local(same);
                    fs.push_back(f);
                }
                continue;   // (towards the lower leaf: it makes the face)
            }
            const int coarse = l > 0 ? whole.find(l - 1, ni >> 1, nj >> 1) : -1;
            if (coarse < 0) {
                expected[static_cast<size_t>(4) * k + s] = 2;
                continue;
            }
            f.kind = CellFaceKind::CoarseFine;
            f.left = dir > 0 ? k : local(coarse);
            f.right = dir > 0 ? local(coarse) : k;
            fs.push_back(f);
        }
    }
    faces.swap(fs);
    face_level_begin.resize(static_cast<size_t>(max_level) + 2);
    for (int l = 0; l <= max_level + 1; ++l)
        face_level_begin[static_cast<size_t>(l)] = static_cast<int>(
            std::partition_point(faces.begin(), faces.end(), [l](const CellFace& f) { return f.level < l; }) - faces.begin());
    // gather lists: the faces on each side of a kept leaf; a side without all its faces stays empty (Outside)
    const int NF = static_cast<int>(faces.size());
    std::vector<int> count(static_cast<size_t>(4) * N, 0);
    const auto side_of_left = [](const CellFace& f) { return f.axis == 0 ? 1 : 3; };    // Right / Top of `left`
    const auto side_of_right = [](const CellFace& f) { return f.axis == 0 ? 0 : 2; };   // Left / Bottom of `right`
    for (const CellFace& f : faces) {
        if (f.left >= 0) ++count[static_cast<size_t>(4) * f.left + side_of_left(f)];
        if (f.right >= 0) ++count[static_cast<size_t>(4) * f.right + side_of_right(f)];
    }
    side_ptr.assign(static_cast<size_t>(4) * N + 1, 0);
    for (int q = 0; q < 4 * N; ++q)
        side_ptr[static_cast<size_t>(q) + 1] = side_ptr[static_cast<size_t>(q)] + (count[static_cast<size_t>(q)] == expected[static_cast<size_t>(q)] ? count[static_cast<size_t>(q)] : 0);
    side_face.assign(static_cast<size_t>(side_ptr.back()), -1);
    const auto place = [&](int q, int fi, const CellFace& f) {
        if (count[static_cast<size_t>(q)] != expected[static_cast<size_t>(q)]) return;
        const int b = side_ptr[static_cast<size_t>(q)];
        side_face[static_cast<size_t>(count[static_cast<size_t>(q)] == 2 ? b + (f.tpos & 1) : b)] = fi;
    };
    for (int fi = 0; fi < NF; ++fi) {
        const CellFace& f = faces[static_cast<size_t>(fi)];
        if (f.left >= 0) place(4 * f.left + side_of_left(f), fi, f);
        if (f.right >= 0) place(4 * f.right + side_of_right(f), fi, f);
    }
}

bool CellMesh::same_arrays(const CellMesh& o) const {
    if (level != o.level || ci != o.ci || cj != o.cj || level_begin != o.level_begin || side_ptr != o.side_ptr ||
        side_face != o.side_face || face_level_begin != o.face_level_begin || faces.size() != o.faces.size() || partial != o.partial)
        return false;
    for (size_t f = 0; f < faces.size(); ++f) {
        const CellFace &a = faces[f], &b = o.faces[f];
        if (a.left != b.left || a.right != b.right || a.level != b.level || a.tpos != b.tpos || a.axis != b.axis || a.kind != b.kind ||
            a.side != b.side || a.length != b.length)
            return false;
    }
    return true;
}

void CellMesh::keep_partial(const std::vector<char>& keep_leaf, const std::vector<char>& keep_face, std::vector<int>& leaf_map,
                            std::vector<int>& face_map) {
    const int N = num_leaves(), NF = static_cast<int>(faces.size());
    if (static_cast<int>(keep_leaf.size()) != N || static_cast<int>(keep_face.size()) != NF)
        throw std::invalid_argument("CellMesh::keep_partial: the flags do not match the leaves / faces");
    leaf_map.assign(static_cast<size_t>(N), -1);
    face_map.assign(static_cast<size_t>(NF), -1);
    int nl = 0, nf = 0;
    for (int c = 0; c < N; ++c)
        if (keep_leaf[static_cast<size_t>(c)]) leaf_map[static_cast<size_t>(c)] = nl++;
    for (int f = 0; f < NF; ++f)
        if (keep_face[static_cast<size_t>(f)]) face_map[static_cast<size_t>(f)] = nf++;
    // the leaves, in their order (so the (level, key) order and the level runs stay)
    std::vector<int> lv(static_cast<size_t>(nl)), ii(static_cast<size_t>(nl)), jj(static_cast<size_t>(nl));
    Field us;
    if (!U.empty()) us.resize(static_cast<size_t>(nl));
    for (int c = 0; c < N; ++c) {
        const int k = leaf_map[static_cast<size_t>(c)];
        if (k < 0) continue;
        lv[static_cast<size_t>(k)] = level[static_cast<size_t>(c)];
        ii[static_cast<size_t>(k)] = ci[static_cast<size_t>(c)];
        jj[static_cast<size_t>(k)] = cj[static_cast<size_t>(c)];
        if (!us.empty()) us[static_cast<size_t>(k)] = U[static_cast<size_t>(c)];
    }
    // the faces; a leaf that is not kept becomes kOutside (a boundary keeps -1)
    std::vector<CellFace> fs(static_cast<size_t>(nf));
    for (int f = 0; f < NF; ++f) {
        const int k = face_map[static_cast<size_t>(f)];
        if (k < 0) continue;
        CellFace g = faces[static_cast<size_t>(f)];
        if (g.left >= 0) g.left = leaf_map[static_cast<size_t>(g.left)] >= 0 ? leaf_map[static_cast<size_t>(g.left)] : kOutside;
        if (g.right >= 0) g.right = leaf_map[static_cast<size_t>(g.right)] >= 0 ? leaf_map[static_cast<size_t>(g.right)] : kOutside;
        fs[static_cast<size_t>(k)] = g;
    }
    // the gather lists of the kept leaves: a side whose faces are not all kept is empty (kOutside)
    std::vector<int> sp(static_cast<size_t>(4) * nl + 1, 0), sf;
    for (int c = 0; c < N; ++c) {
        const int k = leaf_map[static_cast<size_t>(c)];
        if (k < 0) continue;
        for (int s = 0; s < 4; ++s) {
            const int b = side_ptr[static_cast<size_t>(4) * c + s], e = side_ptr[static_cast<size_t>(4) * c + s + 1];
            bool all = true;
            for (int q = b; q < e; ++q) all = all && face_map[static_cast<size_t>(side_face[static_cast<size_t>(q)])] >= 0;
            if (all)
                for (int q = b; q < e; ++q) sf.push_back(face_map[static_cast<size_t>(side_face[static_cast<size_t>(q)])]);
            sp[static_cast<size_t>(4) * k + s + 1] = static_cast<int>(sf.size());
        }
    }
    level.swap(lv);
    ci.swap(ii);
    cj.swap(jj);
    U.swap(us);
    faces.swap(fs);
    side_ptr.swap(sp);
    side_face.swap(sf);
    for (std::vector<int>* a : {&prev_leaf, &prev_face, &face_off_}) std::vector<int>().swap(*a);
    std::vector<char>().swap(topo_dirty);
    std::vector<CellFace>().swap(face_spare_);
    ++version;
    partial = true;
    build_lookups();
    face_level_begin.resize(static_cast<size_t>(max_level) + 2);
    for (int l = 0; l <= max_level + 1; ++l)
        face_level_begin[static_cast<size_t>(l)] = static_cast<int>(
            std::partition_point(faces.begin(), faces.end(), [l](const CellFace& f) { return f.level < l; }) - faces.begin());
}

double CellMesh::memory_bytes() const {
    return container_bytes(level_begin) + container_bytes(level) + container_bytes(ci) + container_bytes(cj) + container_bytes(U) +
           container_bytes(faces) + container_bytes(face_level_begin) + container_bytes(side_ptr) + container_bytes(side_face) +
           container_bytes(prev_leaf) + container_bytes(topo_dirty) + container_bytes(prev_face) + container_bytes(grid_) +
           container_bytes(mkey_) + container_bytes(face_off_) + container_bytes(face_spare_);
}

bool CellMesh::adapt(const std::vector<char>& refine_in, const std::vector<char>& coarsen_in) {
    const int N = num_leaves();
    if ((!refine_in.empty() && static_cast<int>(refine_in.size()) != N) ||
        (!coarsen_in.empty() && static_cast<int>(coarsen_in.size()) != N))
        throw std::invalid_argument("CellMesh::adapt: the flags need one entry per leaf");
    std::vector<char> R(static_cast<size_t>(N), 0);
    if (!refine_in.empty()) {
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
        for (int c = 0; c < N; ++c) R[c] = refine_in[c] && level[c] < max_level;
    }

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
        // each group is tested by its anchor and writes its own four entries
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
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

    // New leaf list.  Per target level three streams are each already in Morton order: the leaves that stay, the children
    // of the refined leaves of the level above, the parents of the merged groups of the level below.  A stream entry
    // carries where its state comes from (src >= 0: that leaf; src < 0: the group anchored at leaf -1 - src, a parent
    // mean or a child's injection), so the state is written once, into its final place.  The streams are filled in
    // parallel (blocks of the leaves of each level, counted first) and merged in parallel (their keys are distinct).
    struct NewLeaf {
        uint64_t key;
        int i, j, src;
        bool kept;
    };
    const bool has_u = static_cast<int>(U.size()) == N;
    const int nlev = max_level + 1;
    // per target level t: [keep | kids | parents] of t, then the merged level, at nb[t] in the new list
    std::vector<int> nk(static_cast<size_t>(nlev), 0), nc(static_cast<size_t>(nlev), 0), np(static_cast<size_t>(nlev), 0);
    // per source level l and block b: the keep / kids / parent entries before the block, within the level
    std::vector<std::array<int, 3>> before(static_cast<size_t>(nlev) * (kScanBlocks + 1));
    const auto block = [&](int l, int b, int& lo, int& hi) {
        const int cb = level_begin[l], n = level_begin[l + 1] - cb, chunk = (n + kScanBlocks - 1) / kScanBlocks;
        lo = cb + std::min(n, b * chunk);
        hi = cb + std::min(n, (b + 1) * chunk);
    };
    for (int l = 0; l < nlev; ++l) {
        std::array<int, 3>* bl = &before[static_cast<size_t>(l) * (kScanBlocks + 1)];
#pragma omp parallel for schedule(static) if (level_begin[l + 1] - level_begin[l] >= 4 * kOmpMinWork)
        for (int b = 0; b < kScanBlocks; ++b) {
            int lo, hi;
            block(l, b, lo, hi);
            std::array<int, 3> n = {0, 0, 0};
            for (int c = lo; c < hi; ++c) {
                if (absorbed[c]) continue;
                n[R[c] ? 1 : (merge_anchor[c] ? 2 : 0)] += R[c] ? 4 : 1;
            }
            bl[b + 1] = n;
        }
        bl[0] = {0, 0, 0};
        for (int b = 0; b < kScanBlocks; ++b)
            for (int q = 0; q < 3; ++q) bl[b + 1][q] += bl[b][q];
        nk[static_cast<size_t>(l)] = bl[kScanBlocks][0];
        if (l + 1 < nlev) nc[static_cast<size_t>(l) + 1] = bl[kScanBlocks][1];
        if (l > 0) np[static_cast<size_t>(l) - 1] = bl[kScanBlocks][2];
    }
    std::vector<int> nb(static_cast<size_t>(nlev) + 1, 0), sb(static_cast<size_t>(nlev) + 1, 0);
    for (int t = 0; t < nlev; ++t) {
        const int n = nk[static_cast<size_t>(t)] + nc[static_cast<size_t>(t)] + np[static_cast<size_t>(t)];
        nb[static_cast<size_t>(t) + 1] = nb[static_cast<size_t>(t)] + n;
    }
    const size_t nn = static_cast<size_t>(nb[static_cast<size_t>(nlev)]);
    // stream entries of level t at streams[nb[t] ..]: keep, then kids, then parents
    std::unique_ptr<NewLeaf[]> streams(new NewLeaf[nn + 1]), order_buf(new NewLeaf[nn + 1]), tmp(new NewLeaf[nn + 1]);
    for (int l = 0; l < nlev; ++l) {
        const std::array<int, 3>* bl = &before[static_cast<size_t>(l) * (kScanBlocks + 1)];
        NewLeaf* keep = streams.get() + nb[static_cast<size_t>(l)];
        NewLeaf* kids = l + 1 < nlev ? streams.get() + nb[static_cast<size_t>(l) + 1] + nk[static_cast<size_t>(l) + 1] : nullptr;
        NewLeaf* parents = l > 0 ? streams.get() + nb[static_cast<size_t>(l) - 1] + nk[static_cast<size_t>(l) - 1] +
                                       nc[static_cast<size_t>(l) - 1]
                                 : nullptr;
#pragma omp parallel for schedule(static) if (level_begin[l + 1] - level_begin[l] >= 4 * kOmpMinWork)
        for (int b = 0; b < kScanBlocks; ++b) {
            int lo, hi;
            block(l, b, lo, hi);
            int k0 = bl[b][0], k1 = bl[b][1], k2 = bl[b][2];
            for (int c = lo; c < hi; ++c) {
                if (absorbed[c]) continue;
                const int i = ci[c], j = cj[c];
                if (R[c]) {
                    for (int oj = 0; oj < 2; ++oj)
                        for (int oi = 0; oi < 2; ++oi)
                            kids[k1++] = {morton_key(static_cast<uint32_t>(2 * i + oi), static_cast<uint32_t>(2 * j + oj)), 2 * i + oi,
                                          2 * j + oj, c, false};
                } else if (merge_anchor[c]) {
                    parents[k2++] = {morton_key(static_cast<uint32_t>(i >> 1), static_cast<uint32_t>(j >> 1)), i >> 1, j >> 1, -1 - c,
                                     false};
                } else {
                    keep[k0++] = {mkey_[c], i, j, c, true};
                }
            }
        }
    }
    const auto by_key = [](const NewLeaf& x, const NewLeaf& y) { return x.key < y.key; };
    for (int t = 0; t < nlev; ++t) {
        const size_t b = static_cast<size_t>(nb[static_cast<size_t>(t)]);
        const int a = nk[static_cast<size_t>(t)], k = nc[static_cast<size_t>(t)], p = np[static_cast<size_t>(t)];
        parallel_merge(streams.get() + b, a, streams.get() + b + a, k, tmp.get() + b, by_key);
        parallel_merge(tmp.get() + b, a + k, streams.get() + b + a + k, p, order_buf.get() + b, by_key);
    }
    const NewLeaf* order = order_buf.get();   // all new leaves in order
    std::vector<int> nlevel(nn), ni(nn), nj(nn);
    for (int t = 0; t < nlev; ++t) {
        const int b = nb[static_cast<size_t>(t)], e = nb[static_cast<size_t>(t) + 1];
#pragma omp parallel for schedule(static) if (e - b >= kOmpMinWork)
        for (int k = b; k < e; ++k) nlevel[static_cast<size_t>(k)] = t;
    }
    prev_leaf.resize(nn);
    if (record_sources) source.resize(nn);
    Field nU;
    if (has_u) nU.resize(nn);
#pragma omp parallel for schedule(static) if (nn >= static_cast<size_t>(kOmpMinWork))
    for (long long k = 0; k < static_cast<long long>(nn); ++k) {
        const NewLeaf& n = order[static_cast<size_t>(k)];
        ni[static_cast<size_t>(k)] = n.i;
        nj[static_cast<size_t>(k)] = n.j;
        prev_leaf[static_cast<size_t>(k)] = n.kept ? n.src : -1;
        if (record_sources) {
            if (n.src >= 0) {
                source[static_cast<size_t>(k)] = {n.src, -1, -1, -1};
            } else {
                const int c = -1 - n.src, l = level[c];
                source[static_cast<size_t>(k)] = {c, find(l, ci[c] + 1, cj[c]), find(l, ci[c], cj[c] + 1), find(l, ci[c] + 1, cj[c] + 1)};
            }
        }
        if (!has_u) continue;
        if (n.src >= 0) {
            nU[static_cast<size_t>(k)] = U[n.src];   // stays, or the injection into a child
        } else {
            const int c = -1 - n.src, l = level[c];
            const State &a = U[c], &b = U[find(l, ci[c] + 1, cj[c])], &cc = U[find(l, ci[c], cj[c] + 1)],
                        &d = U[find(l, ci[c] + 1, cj[c] + 1)];
            State m;
            for (int q = 0; q < kNVar; ++q) m[q] = ((a[q] + b[q]) + (cc[q] + d[q])) * 0.25;
            nU[static_cast<size_t>(k)] = m;
        }
    }
    level.swap(nlevel);
    ci.swap(ni);
    cj.swap(nj);
    if (has_u) U.swap(nU);
    rebuild(true);
    return true;
}

}   // namespace fv
