#include "fv/Switches.hpp"
#include "fv/CellMesh.hpp"

#include <algorithm>
#include <cstdlib>
#include <array>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#include "fv/MemoryBytes.hpp"
#include "fv/ParallelScan.hpp"

namespace fv {

namespace {

// Sort key of a leaf inside the whole list: level first, Morton key second.
struct SortKey {
    int level;
    uint64_t morton;
    bool operator<(const SortKey& o) const { return level != o.level ? level < o.level : morton < o.morton; }
};

}   // namespace

CellMesh::CellMesh(double Lx_, double Ly_, int nx_, int ny_, int max_level_, bool per_x, bool per_y, double x0_,
                   double y0_)
    : Lx(Lx_), Ly(Ly_), nx(nx_), ny(ny_), max_level(max_level_), periodic_x(per_x), periodic_y(per_y), x0(x0_),
      y0(y0_) {
    if (nx < 1 || ny < 1) throw std::invalid_argument("CellMesh: the base mesh needs at least one cell per axis");
    if (max_level < 0 || max_level > 20) throw std::invalid_argument("CellMesh: max_level must be in [0, 20]");
    if ((static_cast<long long>(std::max(nx, ny)) << max_level) >= (1ll << cell_key_bits(2)))
        throw std::invalid_argument("CellMesh: the finest level has too many cells per axis");
    dx0_ = Lx / nx;
    dy0_ = Ly / ny;
    init_base();
}

CellMesh CellMesh::make_3d(double Lx_, double Ly_, double Lz_, int nx_, int ny_, int nz_, int max_level_, bool per_x, bool per_y,
                           bool per_z, double x0_, double y0_, double z0_) {
    if (nx_ < 1 || ny_ < 1 || nz_ < 1) throw std::invalid_argument("CellMesh: the base mesh needs at least one cell per axis");
    if (max_level_ < 0 || max_level_ > 20) throw std::invalid_argument("CellMesh: max_level must be in [0, 20]");
    if ((static_cast<long long>(std::max(nx_, std::max(ny_, nz_))) << max_level_) >= (1ll << cell_key_bits(3)))
        throw std::invalid_argument("CellMesh: the finest level has too many cells per axis (3-D keys hold 21 bits per axis)");
    CellMesh m;
    m.dim = 3;
    m.Lx = Lx_;
    m.Ly = Ly_;
    m.Lz = Lz_;
    m.nx = nx_;
    m.ny = ny_;
    m.nz = nz_;
    m.max_level = max_level_;
    m.periodic_x = per_x;
    m.periodic_y = per_y;
    m.periodic_z = per_z;
    m.x0 = x0_;
    m.y0 = y0_;
    m.z0 = z0_;
    m.dx0_ = Lx_ / nx_;
    m.dy0_ = Ly_ / ny_;
    m.dz0_ = Lz_ / nz_;
    m.init_base();
    return m;
}

void CellMesh::init_base() {
    const int NZ = nz_at(0);
    const size_t n = static_cast<size_t>(nx) * ny * NZ;
    level.assign(n, 0);
    ci.resize(n);
    cj.resize(n);
    ck.resize(n);
    std::vector<int> order(n);
    std::vector<uint64_t> key(n);
    size_t c = 0;
    for (int k = 0; k < NZ; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i, ++c) {
                ci[c] = i;
                cj[c] = j;
                ck[c] = k;
                key[c] = key_of(i, j, k);
            }
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });
    std::vector<int> i2(n), j2(n), k2(n);
    for (size_t q = 0; q < n; ++q) {
        i2[q] = ci[order[q]];
        j2[q] = cj[order[q]];
        k2[q] = ck[order[q]];
    }
    ci.swap(i2);
    cj.swap(j2);
    ck.swap(k2);
    rebuild();
}

int CellMesh::find(int l, int i, int j, int k) const {
    if (l < 0 || l > max_level || i < 0 || j < 0 || k < 0 || i >= nx_at(l) || j >= ny_at(l) || k >= nz_at(l)) return -1;
    const std::vector<int>& g = grid_[static_cast<size_t>(l)];
    if (!g.empty()) return g[(static_cast<size_t>(k) * ny_at(l) + j) * nx_at(l) + i];
    const uint64_t key = key_of(i, j, k);
    const auto b = mkey_.begin() + level_begin[l], e = mkey_.begin() + level_begin[l + 1];
    const auto it = std::lower_bound(b, e, key);
    return (it != e && *it == key) ? static_cast<int>(it - mkey_.begin()) : -1;
}

int CellMesh::covering(int l, int i, int j, int k) const {
    for (; l >= 0; --l, i >>= 1, j >>= 1, k >>= 1) {
        const int c = find(l, i, j, k);
        if (c >= 0) return c;
    }
    return -1;
}

bool CellMesh::wrap(int l, int& i, int& j, int& k) const {
    const int NX = nx_at(l), NY = ny_at(l), NZ = nz_at(l);
    if (i < 0 || i >= NX) {
        if (!periodic_x) return false;
        i = (i % NX + NX) % NX;
    }
    if (j < 0 || j >= NY) {
        if (!periodic_y) return false;
        j = (j % NY + NY) % NY;
    }
    if (k < 0 || k >= NZ) {
        if (!periodic_z) return false;
        k = (k % NZ + NZ) % NZ;
    }
    return true;
}

CellNeighbour CellMesh::neighbour(int leaf, Face side) const {
    if (faceless_) throw std::logic_error("CellMesh::neighbour: a faceless mesh has no face lists");
    const int s = static_cast<int>(side), ns = sides();
    const int b = side_ptr[ns * leaf + s], e = side_ptr[ns * leaf + s + 1];
    CellNeighbour n;
    if (b == e) {   // a partial mesh: this side looks outside what the rank keeps
        n.kind = CellNeighbour::Kind::Outside;
        return n;
    }
    const auto other = [&](int f) { return faces[f].left == leaf ? faces[f].right : faces[f].left; };
    if (faces[side_face[b]].kind == CellFaceKind::Boundary) return n;
    n.a = other(side_face[b]);
    if (e - b > 1) {
        n.kind = CellNeighbour::Kind::Finer;
        n.b = other(side_face[b + 1]);
        bool out = n.a == kOutside || n.b == kOutside;
        if (e - b == 4) {
            n.c = other(side_face[b + 2]);
            n.d = other(side_face[b + 3]);
            out = out || n.c == kOutside || n.d == kOutside;
        }
        if (out) n.kind = CellNeighbour::Kind::Outside;
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
    for (int c = 0; c < N; ++c) mkey_[c] = key_of(ci[c], cj[c], ck[c]);
    grid_.resize(static_cast<size_t>(max_level) + 1);
    for (int l = 0; l <= max_level; ++l) {
        const long long cells = static_cast<long long>(nx_at(l)) * ny_at(l) * nz_at(l);
        std::vector<int>& g = grid_[static_cast<size_t>(l)];
        if (cells > 4ll * N + 65536) {
            std::vector<int>().swap(g);
            continue;
        }
        g.resize(static_cast<size_t>(cells));   // the table of the last rebuild, overwritten
        const int nc = static_cast<int>(cells), cb = level_begin[l], ce = level_begin[l + 1], NX = nx_at(l), NY = ny_at(l);
#pragma omp parallel for schedule(static) if (nc >= kOmpMinWork)
        for (int k = 0; k < nc; ++k) g[static_cast<size_t>(k)] = -1;
#pragma omp parallel for schedule(static) if (ce - cb >= kOmpMinWork)
        for (int c = cb; c < ce; ++c) g[(static_cast<size_t>(ck[c]) * NY + cj[c]) * NX + ci[c]] = c;
    }
}

void CellMesh::set_faceless(bool on) {
    if (faceless_ == on) return;
    faceless_ = on;
    rebuild();
}

int CellMesh::face_of(const CellMesh& in, int c, int l, int i, int j, int k, int s, CellFace& f) const {
    // same level  -> the leaf on the lower side creates it (its Right / Top / Back side)
    // coarser nbr -> the fine leaf creates it;  finer nbr -> nothing (-1);  no nbr -> a boundary face
    const int axis = s / 2, dir = (s & 1) ? 1 : -1;
    const int x[3] = {i, j, k};
    int n[3] = {i, j, k};
    n[axis] += dir;
    f = CellFace();
    f.level = l;
    f.axis = static_cast<uint8_t>(axis);
    f.tpos = x[transverse1(axis)];
    f.tpos2 = x[transverse2(axis)];
    if (dim == 3)   // Mesh::face_area: dy dz, dx dz, dx dy
        f.length = axis == 0 ? dy(l) * dz(l) : (axis == 1 ? dx(l) * dz(l) : dx(l) * dy(l));
    else
        f.length = axis == 0 ? dy(l) : dx(l);
    if (!wrap(l, n[0], n[1], n[2])) {
        f.kind = CellFaceKind::Boundary;
        f.side = static_cast<Face>(s);
        f.left = dir > 0 ? c : -1;
        f.right = dir > 0 ? -1 : c;
        return 1;
    }
    const int same = in.find(l, n[0], n[1], n[2]);
    if (same >= 0) {
        if (dir <= 0) return 0;
        f.left = c;
        f.right = same;
        return 1;
    }
    const int coarse = l > 0 ? in.find(l - 1, n[0] >> 1, n[1] >> 1, n[2] >> 1) : -1;
    if (coarse < 0) return -1;
    f.kind = CellFaceKind::CoarseFine;
    f.left = dir > 0 ? c : coarse;
    f.right = dir > 0 ? coarse : c;
    return 1;
}

void CellMesh::rebuild(bool incremental) {
    const int N = num_leaves(), ns = sides();
    ++version;
    partial = false;
    static const bool force_full = switch_on("FENICE_FULL_REBUILD");
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

    // faces: each leaf creates the faces it is responsible for (face_of), at most one per side; the faces keep the leaf order,
    // so the list is the serial one.
    const auto make_face = [&](int c, int s, CellFace& f) { return face_of(*this, c, level[c], ci[c], cj[c], ck[c], s, f) == 1; };
    // After an adapt: a kept leaf is clean if none of the squares (cubes) of its size across its sides holds a new leaf; then its
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
            const int x[3] = {ci[n], cj[n], ck[n]};
            for (int s = 0; s < ns; ++s) {
                const int axis = s / 2, dir = (s & 1) ? 1 : -1;
                int d[3] = {0, 0, 0};
                d[axis] = dir;
                for (int lc = std::max(0, ln - 2); lc <= ln; ++lc) {
                    int b[3];
                    for (int a = 0; a < 3; ++a) b[a] = (x[a] >> (ln - lc)) + d[a];
                    if (!wrap(lc, b[0], b[1], b[2])) continue;
                    const int k = find(lc, b[0], b[1], b[2]);
                    if (k >= 0) topo_dirty[k] = 1;
                }
                if (ln == max_level) continue;
                int b[3] = {x[0] + d[0], x[1] + d[1], x[2] + d[2]};
                if (!wrap(ln, b[0], b[1], b[2])) continue;
                const int t1 = transverse1(axis), t2 = transverse2(axis);
                for (int t = 0; t < fine_faces(); ++t) {   // the children of the block across side s that touch n
                    int o[3] = {0, 0, 0};
                    o[axis] = dir > 0 ? 0 : 1;
                    o[t1] = t & 1;
                    o[t2] = t >> 1;
                    const int k = find(ln + 1, 2 * b[0] + o[0], 2 * b[1] + o[1], 2 * b[2] + o[2]);
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
            for (int s = 0; s < ns; ++s) k += make_face(c, s, f) ? 1 : 0;
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
        for (int s = 0; s < ns; ++s)
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
    const int NF = static_cast<int>(faces.size()), NS = ns * N, FF = fine_faces();
    side_ptr.resize(static_cast<size_t>(NS) + 1);
#pragma omp parallel for schedule(static) if (NS >= kOmpMinWork)
    for (int k = 0; k <= NS; ++k) side_ptr[static_cast<size_t>(k)] = 0;
    const auto side_of_left = [](const CellFace& f) { return 2 * f.axis + 1; };   // Right / Top / Back of `left`
    const auto side_of_right = [](const CellFace& f) { return 2 * f.axis; };      // Left / Bottom / Front of `right`
#pragma omp parallel for schedule(static) if (NF >= kOmpMinWork)
    for (int fi = 0; fi < NF; ++fi) {
        const CellFace& f = faces[static_cast<size_t>(fi)];
        if (f.left >= 0) {
            int& n = side_ptr[static_cast<size_t>(ns) * f.left + side_of_left(f) + 1];
#pragma omp atomic
            ++n;
        }
        if (f.right >= 0) {
            int& n = side_ptr[static_cast<size_t>(ns) * f.right + side_of_right(f) + 1];
#pragma omp atomic
            ++n;
        }
    }
    bool bad_count = false;
#pragma omp parallel for schedule(static) if (NS >= kOmpMinWork)
    for (int k = 1; k <= NS; ++k)
        if (side_ptr[static_cast<size_t>(k)] != 1 && side_ptr[static_cast<size_t>(k)] != FF) bad_count = true;
    if (bad_count) throw std::logic_error("CellMesh: a cell side must carry one face or one per finer leaf (2:1 violated?)");
    inclusive_scan_inplace(side_ptr.data(), NS + 1);
    // A side with several faces faces the finer leaves at the transverse positions (2 t1 + a, 2 t2 + b): the slot is a + 2 b, so
    // every face has its own slot (tpos2 major, tpos minor) and the order is the serial one.
    side_face.resize(static_cast<size_t>(side_ptr.back()));
    const int NSF = static_cast<int>(side_face.size());
#pragma omp parallel for schedule(static) if (NSF >= kOmpMinWork)
    for (int k = 0; k < NSF; ++k) side_face[static_cast<size_t>(k)] = -1;
    const auto slot = [&](int k, const CellFace& f) {
        const int b = side_ptr[static_cast<size_t>(k)];
        return side_ptr[static_cast<size_t>(k) + 1] - b > 1 ? b + (f.tpos & 1) + 2 * (f.tpos2 & 1) : b;
    };
#pragma omp parallel for schedule(static) if (NF >= kOmpMinWork)
    for (int fi = 0; fi < NF; ++fi) {
        const CellFace& f = faces[static_cast<size_t>(fi)];
        if (f.left >= 0) side_face[static_cast<size_t>(slot(ns * f.left + side_of_left(f), f))] = fi;
        if (f.right >= 0) side_face[static_cast<size_t>(slot(ns * f.right + side_of_right(f), f))] = fi;
    }
    bool bad_slot = false;
#pragma omp parallel for schedule(static) if (NS >= kOmpMinWork)
    for (int k = 0; k < NS; ++k) {
        const int b = side_ptr[static_cast<size_t>(k)], e = side_ptr[static_cast<size_t>(k) + 1];
        bool ok = true;
        for (int q = b; q < e; ++q) ok = ok && side_face[static_cast<size_t>(q)] >= 0;
        if (ok && e - b > 1) {
            const CellFace& f0 = faces[static_cast<size_t>(side_face[static_cast<size_t>(b)])];
            for (int q = 1; q < e - b; ++q) {
                const CellFace& fq = faces[static_cast<size_t>(side_face[static_cast<size_t>(b + q)])];
                ok = ok && fq.tpos == f0.tpos + (q & 1) && fq.tpos2 == f0.tpos2 + (q >> 1);
            }
        }
        if (!ok) bad_slot = true;
    }
    if (bad_slot) throw std::logic_error("CellMesh: the faces of a side are not sibling finer leaves");
}

void CellMesh::set_sorted_leaves(const std::vector<int>& lv, const std::vector<int>& ii, const std::vector<int>& jj,
                                 const std::vector<int>& kk, Field* state, const char* who) {
    const size_t n = lv.size();
    if (ii.size() != n || jj.size() != n || (dim == 3 ? kk.size() != n : (!kk.empty() && kk.size() != n)) ||
        (state && state->size() != n))
        throw std::invalid_argument(std::string(who) + ": the arrays differ in length");
    const auto kat = [&](size_t q) { return kk.empty() ? 0 : kk[q]; };
    // (key, input index) records sorted in place: no indirect comparison through an index array
    struct Rec {
        SortKey key;
        int index;
    };
    std::vector<Rec> recs(n);
    for (size_t q = 0; q < n; ++q) {
        if (lv[q] < 0 || lv[q] > max_level || ii[q] < 0 || ii[q] >= nx_at(lv[q]) || jj[q] < 0 || jj[q] >= ny_at(lv[q]) || kat(q) < 0 ||
            kat(q) >= nz_at(lv[q]))
            throw std::invalid_argument(std::string(who) + ": a leaf is outside the mesh or deeper than max_level");
        recs[q] = {{lv[q], key_of(ii[q], jj[q], kat(q))}, static_cast<int>(q)};
    }
    std::sort(recs.begin(), recs.end(), [](const Rec& a, const Rec& b) { return a.key < b.key; });
    for (size_t q = 1; q < n; ++q)
        if (!(recs[q - 1].key < recs[q].key)) throw std::invalid_argument(std::string(who) + ": a leaf is given twice");
    std::vector<int> order(n);
    for (size_t q = 0; q < n; ++q) order[q] = recs[q].index;
    level.resize(n);
    ci.resize(n);
    cj.resize(n);
    ck.resize(n);
    Field reordered;
    if (state) reordered.resize(n);
    for (size_t q = 0; q < n; ++q) {
        level[q] = lv[order[q]];
        ci[q] = ii[order[q]];
        cj[q] = jj[order[q]];
        ck[q] = kat(static_cast<size_t>(order[q]));
        if (state) reordered[q] = (*state)[static_cast<size_t>(order[q])];
    }
    if (state) U = std::move(reordered);
    else U.clear();
}

void CellMesh::assign_leaves(const std::vector<int>& lv, const std::vector<int>& ii, const std::vector<int>& jj, Field* state) {
    assign_leaves(lv, ii, jj, {}, state);
}

void CellMesh::assign_leaves(const std::vector<int>& lv, const std::vector<int>& ii, const std::vector<int>& jj,
                             const std::vector<int>& kk, Field* state) {
    const size_t n = lv.size();
    if (ii.size() != n || jj.size() != n || (dim == 3 ? kk.size() != n : !kk.empty()) || (state && state->size() != n))
        throw std::invalid_argument("CellMesh::assign_leaves: the arrays differ in length");
    long long volume = 0;   // in cells of the finest level
    for (size_t q = 0; q < n; ++q) {
        const int k = kk.empty() ? 0 : kk[q];
        if (lv[q] < 0 || lv[q] > max_level || ii[q] < 0 || ii[q] >= nx_at(lv[q]) || jj[q] < 0 || jj[q] >= ny_at(lv[q]) || k < 0 ||
            k >= nz_at(lv[q]))
            throw std::invalid_argument("CellMesh::assign_leaves: a leaf is outside the mesh or deeper than max_level");
        volume += 1ll << (dim * (max_level - lv[q]));
    }
    if (volume != static_cast<long long>(nx) * ny * nz_at(0) * (1ll << (dim * max_level)))
        throw std::invalid_argument("CellMesh::assign_leaves: the leaves do not tile the domain");
    set_sorted_leaves(lv, ii, jj, kk, state, "CellMesh::assign_leaves");
    rebuild();
}

void CellMesh::build_from(const CellMesh& whole, const std::vector<char>& keep_leaf, const Field* whole_state) {
    const int NW = whole.num_leaves(), ns = sides();
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
    ck.resize(static_cast<size_t>(N));
    U.assign(static_cast<size_t>(N), State{});   // (the state of a leaf comes from whole_state, or is set by the caller)
    for (int k = 0; k < N; ++k) {
        const size_t c = static_cast<size_t>(from_whole[static_cast<size_t>(k)]);
        level[static_cast<size_t>(k)] = whole.level[c];
        ci[static_cast<size_t>(k)] = whole.ci[c];
        cj[static_cast<size_t>(k)] = whole.cj[c];
        ck[static_cast<size_t>(k)] = whole.ck[c];
        if (whole_state) U[static_cast<size_t>(k)] = (*whole_state)[c];
    }
    for (std::vector<int>* a : {&prev_leaf, &prev_face, &face_off_}) std::vector<int>().swap(*a);
    std::vector<char>().swap(topo_dirty);
    std::vector<CellFace>().swap(face_spare_);
    ++version;
    partial = true;
    build_lookups();
    // The faces the kept leaves make (face_of, the neighbours looked up in the whole mesh), in the order of the leaves; the other
    // end is kOutside if that leaf is not kept. expected[ns k + s]: the faces side s has in the whole mesh (one per finer leaf,
    // which make them), so a side is listed only if all of them are there.
    std::vector<CellFace> fs;
    std::vector<char> expected(static_cast<size_t>(ns) * N, 1);
    const auto local = [&](int whole_leaf) {
        return to_local[static_cast<size_t>(whole_leaf)] >= 0 ? to_local[static_cast<size_t>(whole_leaf)] : kOutside;
    };
    for (int k = 0; k < N; ++k) {
        const size_t q = static_cast<size_t>(k);
        for (int s = 0; s < ns; ++s) {
            CellFace f;
            const int r = face_of(whole, k, level[q], ci[q], cj[q], ck[q], s, f);
            if (r < 0) {
                expected[static_cast<size_t>(ns) * k + s] = static_cast<char>(fine_faces());
                continue;
            }
            if (r == 0) continue;   // (towards the lower leaf: it makes the face)
            if (f.kind != CellFaceKind::Boundary) {   // the other end is a leaf of the whole mesh
                if (s & 1) f.right = local(f.right);
                else f.left = local(f.left);
            }
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
    std::vector<int> count(static_cast<size_t>(ns) * N, 0);
    const auto side_of_left = [](const CellFace& f) { return 2 * f.axis + 1; };   // Right / Top / Back of `left`
    const auto side_of_right = [](const CellFace& f) { return 2 * f.axis; };      // Left / Bottom / Front of `right`
    for (const CellFace& f : faces) {
        if (f.left >= 0) ++count[static_cast<size_t>(ns) * f.left + side_of_left(f)];
        if (f.right >= 0) ++count[static_cast<size_t>(ns) * f.right + side_of_right(f)];
    }
    side_ptr.assign(static_cast<size_t>(ns) * N + 1, 0);
    for (int q = 0; q < ns * N; ++q)
        side_ptr[static_cast<size_t>(q) + 1] =
            side_ptr[static_cast<size_t>(q)] + (count[static_cast<size_t>(q)] == expected[static_cast<size_t>(q)] ? count[static_cast<size_t>(q)] : 0);
    side_face.assign(static_cast<size_t>(side_ptr.back()), -1);
    const auto place = [&](int q, int fi, const CellFace& f) {
        if (count[static_cast<size_t>(q)] != expected[static_cast<size_t>(q)]) return;
        const int b = side_ptr[static_cast<size_t>(q)];
        side_face[static_cast<size_t>(count[static_cast<size_t>(q)] > 1 ? b + (f.tpos & 1) + 2 * (f.tpos2 & 1) : b)] = fi;
    };
    for (int fi = 0; fi < NF; ++fi) {
        const CellFace& f = faces[static_cast<size_t>(fi)];
        if (f.left >= 0) place(ns * f.left + side_of_left(f), fi, f);
        if (f.right >= 0) place(ns * f.right + side_of_right(f), fi, f);
    }
}

void CellMesh::assign_partial_leaves(const std::vector<int>& lv, const std::vector<int>& ii, const std::vector<int>& jj, Field* state) {
    assign_partial_leaves(lv, ii, jj, {}, state);
}

void CellMesh::assign_partial_leaves(const std::vector<int>& lv, const std::vector<int>& ii, const std::vector<int>& jj,
                                     const std::vector<int>& kk, Field* state) {
    set_sorted_leaves(lv, ii, jj, kk, state, "CellMesh::assign_partial_leaves");
    for (std::vector<int>* a : {&prev_leaf, &prev_face, &face_off_, &side_ptr, &side_face, &face_level_begin}) std::vector<int>().swap(*a);
    std::vector<char>().swap(topo_dirty);
    std::vector<CellFace>().swap(faces);
    std::vector<CellFace>().swap(face_spare_);
    ++version;
    partial = true;
    build_lookups();
}

bool CellMesh::same_arrays(const CellMesh& o) const {
    if (dim != o.dim || level != o.level || ci != o.ci || cj != o.cj || ck != o.ck || level_begin != o.level_begin ||
        side_ptr != o.side_ptr || side_face != o.side_face || face_level_begin != o.face_level_begin || faces.size() != o.faces.size() ||
        partial != o.partial)
        return false;
    for (size_t f = 0; f < faces.size(); ++f) {
        const CellFace &a = faces[f], &b = o.faces[f];
        if (a.left != b.left || a.right != b.right || a.level != b.level || a.tpos != b.tpos || a.tpos2 != b.tpos2 || a.axis != b.axis ||
            a.kind != b.kind || a.side != b.side || a.length != b.length)
            return false;
    }
    return true;
}

void CellMesh::keep_partial(const std::vector<char>& keep_leaf, const std::vector<char>& keep_face, std::vector<int>& leaf_map,
                            std::vector<int>& face_map) {
    const int N = num_leaves(), NF = static_cast<int>(faces.size()), ns = sides();
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
    std::vector<int> lv(static_cast<size_t>(nl)), ii(static_cast<size_t>(nl)), jj(static_cast<size_t>(nl)), kk(static_cast<size_t>(nl));
    Field us;
    if (!U.empty()) us.resize(static_cast<size_t>(nl));
    for (int c = 0; c < N; ++c) {
        const int k = leaf_map[static_cast<size_t>(c)];
        if (k < 0) continue;
        lv[static_cast<size_t>(k)] = level[static_cast<size_t>(c)];
        ii[static_cast<size_t>(k)] = ci[static_cast<size_t>(c)];
        jj[static_cast<size_t>(k)] = cj[static_cast<size_t>(c)];
        kk[static_cast<size_t>(k)] = ck[static_cast<size_t>(c)];
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
    std::vector<int> sp(static_cast<size_t>(ns) * nl + 1, 0), sf;
    for (int c = 0; c < N; ++c) {
        const int k = leaf_map[static_cast<size_t>(c)];
        if (k < 0) continue;
        for (int s = 0; s < ns; ++s) {
            const int b = side_ptr[static_cast<size_t>(ns) * c + s], e = side_ptr[static_cast<size_t>(ns) * c + s + 1];
            bool all = true;
            for (int q = b; q < e; ++q) all = all && face_map[static_cast<size_t>(side_face[static_cast<size_t>(q)])] >= 0;
            if (all)
                for (int q = b; q < e; ++q) sf.push_back(face_map[static_cast<size_t>(side_face[static_cast<size_t>(q)])]);
            sp[static_cast<size_t>(ns) * k + s + 1] = static_cast<int>(sf.size());
        }
    }
    level.swap(lv);
    ci.swap(ii);
    cj.swap(jj);
    ck.swap(kk);
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
    return container_bytes(level_begin) + container_bytes(level) + container_bytes(ci) + container_bytes(cj) + container_bytes(ck) +
           container_bytes(U) + container_bytes(faces) + container_bytes(face_level_begin) + container_bytes(side_ptr) +
           container_bytes(side_face) + container_bytes(prev_leaf) + container_bytes(topo_dirty) + container_bytes(prev_face) +
           container_bytes(grid_) + container_bytes(mkey_) + container_bytes(face_off_) + container_bytes(face_spare_);
}

bool CellMesh::has_leaf_under(int l, int i, int j, int k) const {
    if (covering(l, i, j, k) >= 0) return true;
    if (l >= max_level) return false;
    for (int q = 0; q < children(); ++q) {
        int o[3];
        child_offset(q, o);
        if (has_leaf_under(l + 1, 2 * i + o[0], 2 * j + o[1], 2 * k + o[2])) return true;
    }
    return false;
}

bool CellMesh::forced_by_finer(int b, const std::vector<char>& R, bool check) const {
    // the level l+1 cells of the ring around the 2^D block of b: a leaf there is a vertex neighbour of a leaf that b covers
    const int l = level[b];
    if (l >= max_level) return false;
    const int kz = dim == 3 ? 2 : 0;
    for (int dk = dim == 3 ? -1 : 0; dk <= kz; ++dk)
        for (int dj = -1; dj <= 2; ++dj)
            for (int di = -1; di <= 2; ++di) {
                if (di >= 0 && di <= 1 && dj >= 0 && dj <= 1 && (dim == 2 || (dk >= 0 && dk <= 1))) continue;
                int i = 2 * ci[b] + di, j = 2 * cj[b] + dj, k = 2 * ck[b] + dk;
                if (!wrap(l + 1, i, j, k)) continue;
                const int a = find(l + 1, i, j, k);
                if (a >= 0) {
                    if (R[a]) return true;
                } else if (check && !has_leaf_under(l + 1, i, j, k)) {
                    throw std::logic_error("CellMesh::forced_by_finer: a cell next to the leaf is not in the mesh (the region fetched is too small)");
                }
            }
    return false;
}

bool CellMesh::merge_ok(int a, const std::vector<char>& coarsen, const std::vector<char>& R, bool check) const {
    const int l = level[a];
    if (l == 0 || (ci[a] & 1) || (cj[a] & 1) || (ck[a] & 1)) return false;
    const int x[3] = {ci[a], cj[a], ck[a]}, nch = children();
    int sib[8];
    for (int q = 0; q < nch; ++q) {
        int o[3];
        child_offset(q, o);
        sib[q] = q == 0 ? a : find(l, x[0] + o[0], x[1] + o[1], x[2] + o[2]);
    }
    bool ok = true;
    for (int q = 1; q < nch && ok; ++q) {
        int o[3];
        child_offset(q, o);
        if (sib[q] < 0 && check && !has_leaf_under(l, x[0] + o[0], x[1] + o[1], x[2] + o[2]))
            throw std::logic_error("CellMesh::merge_ok: a sibling cell is not in the mesh (the region fetched is too small)");
    }
    for (int q = 0; q < nch; ++q) ok = ok && sib[q] >= 0 && coarsen[sib[q]] && !R[sib[q]];
    if (!ok) return false;
    // the parent (level l-1) must stay within one level of every vertex neighbour block
    const int p[3] = {x[0] >> 1, x[1] >> 1, x[2] >> 1};
    for (int q = 0; q < cell_neighbour_count(dim) && ok; ++q) {
        int d[3];
        neighbour_offset(dim, q, d);
        int n[3] = {p[0] + d[0], p[1] + d[1], p[2] + d[2]};
        if (!wrap(l - 1, n[0], n[1], n[2])) continue;
        if (covering(l - 1, n[0], n[1], n[2]) >= 0) continue;   // same level or coarser: no conflict
        if (check && !has_leaf_under(l - 1, n[0], n[1], n[2]))
            throw std::logic_error("CellMesh::merge_ok: a block next to the group is not in the mesh (the region fetched is too small)");
        // the neighbour block is subdivided: the children touching the parent must be leaves that stay at level l
        for (int r = 0; r < nch; ++r) {
            int o[3];
            child_offset(r, o);
            bool touch = true;
            for (int ax = 0; ax < 3; ++ax) touch = touch && !((d[ax] < 0 && o[ax] == 0) || (d[ax] > 0 && o[ax] == 1));
            if (!touch) continue;
            const int k = find(l, 2 * n[0] + o[0], 2 * n[1] + o[1], 2 * n[2] + o[2]);
            if (k < 0 && check && !has_leaf_under(l, 2 * n[0] + o[0], 2 * n[1] + o[1], 2 * n[2] + o[2]))
                throw std::logic_error("CellMesh::merge_ok: a child of the block next to the group is not in the mesh (the region fetched is too small)");
            if (k < 0 || R[k]) ok = false;
        }
    }
    return ok;
}

bool CellMesh::adapt(const std::vector<char>& refine_in, const std::vector<char>& coarsen_in) {
    const int N = num_leaves(), nch = children(), nnb = cell_neighbour_count(dim);
    if ((!refine_in.empty() && static_cast<int>(refine_in.size()) != N) ||
        (!coarsen_in.empty() && static_cast<int>(coarsen_in.size()) != N))
        throw std::invalid_argument("CellMesh::adapt: the flags need one entry per leaf");
    std::vector<char> R(static_cast<size_t>(N), 0);
    if (!refine_in.empty()) {
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
        for (int c = 0; c < N; ++c) R[c] = refine_in[c] && level[c] < max_level;
    }

    // Ripple: a leaf that becomes level l+1 needs every vertex neighbour at level >= l.  The mesh is balanced
    // before the call, so a lower neighbour is exactly one level lower and must be refined as well.
    std::vector<int> work;
    for (int c = 0; c < N; ++c)
        if (R[c]) work.push_back(c);
    while (!work.empty()) {
        const int a = work.back();
        work.pop_back();
        const int l = level[a];
        for (int q = 0; q < nnb; ++q) {
            int d[3];
            neighbour_offset(dim, q, d);
            int i = ci[a] + d[0], j = cj[a] + d[1], k = ck[a] + d[2];
            if (!wrap(l, i, j, k)) continue;
            const int b = covering(l, i, j, k);
            if (b >= 0 && level[b] < l && !R[b]) {
                R[b] = 1;
                work.push_back(b);
            }
        }
    }

    // Merges: sibling groups anchored at their (even, even[, even]) member
    std::vector<char> merge_anchor(static_cast<size_t>(N), 0), absorbed(static_cast<size_t>(N), 0);
    if (!coarsen_in.empty()) {
        // each group is tested by its anchor and writes its own entries
#pragma omp parallel for schedule(static) if (N >= kOmpMinWork)
        for (int a = 0; a < N; ++a) {
            if (!merge_ok(a, coarsen_in, R, false)) continue;
            const int l = level[a];
            merge_anchor[a] = 1;
            for (int q = 1; q < nch; ++q) {
                int o[3];
                child_offset(q, o);
                absorbed[find(l, ci[a] + o[0], cj[a] + o[1], ck[a] + o[2])] = 1;
            }
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
        int i, j, k, src;
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
                n[R[c] ? 1 : (merge_anchor[c] ? 2 : 0)] += R[c] ? nch : 1;
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
                const int i = ci[c], j = cj[c], k = ck[c];
                if (R[c]) {
                    for (int q = 0; q < nch; ++q) {   // in key order
                        int o[3];
                        child_offset(q, o);
                        kids[k1++] = {key_of(2 * i + o[0], 2 * j + o[1], 2 * k + o[2]), 2 * i + o[0], 2 * j + o[1], 2 * k + o[2], c, false};
                    }
                } else if (merge_anchor[c]) {
                    parents[k2++] = {key_of(i >> 1, j >> 1, k >> 1), i >> 1, j >> 1, k >> 1, -1 - c, false};
                } else {
                    keep[k0++] = {mkey_[c], i, j, k, c, true};
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
    std::vector<int> nlevel(nn), ni(nn), nj(nn), nkz(nn);
    for (int t = 0; t < nlev; ++t) {
        const int b = nb[static_cast<size_t>(t)], e = nb[static_cast<size_t>(t) + 1];
#pragma omp parallel for schedule(static) if (e - b >= kOmpMinWork)
        for (int k = b; k < e; ++k) nlevel[static_cast<size_t>(k)] = t;
    }
    prev_leaf.resize(nn);
    if (record_sources) source.resize(nn);
    Field nU;
    if (has_u) nU.resize(nn);
    // the siblings of the group anchored at c, in child order
    const auto group = [&](int c, int* g) {
        const int l = level[c];
        for (int q = 0; q < nch; ++q) {
            int o[3];
            child_offset(q, o);
            g[q] = q == 0 ? c : find(l, ci[c] + o[0], cj[c] + o[1], ck[c] + o[2]);
        }
    };
#pragma omp parallel for schedule(static) if (nn >= static_cast<size_t>(kOmpMinWork))
    for (long long k = 0; k < static_cast<long long>(nn); ++k) {
        const NewLeaf& n = order[static_cast<size_t>(k)];
        ni[static_cast<size_t>(k)] = n.i;
        nj[static_cast<size_t>(k)] = n.j;
        nkz[static_cast<size_t>(k)] = n.k;
        prev_leaf[static_cast<size_t>(k)] = n.kept ? n.src : -1;
        int g[8];
        if (n.src < 0) group(-1 - n.src, g);
        if (record_sources) {
            std::array<int, 8> s;
            s.fill(-1);
            if (n.src >= 0) {
                s[0] = n.src;
            } else {
                for (int q = 0; q < nch; ++q) s[static_cast<size_t>(q)] = g[q];
            }
            source[static_cast<size_t>(k)] = s;
        }
        if (!has_u) continue;
        if (n.src >= 0) {
            nU[static_cast<size_t>(k)] = U[n.src];   // stays, or the injection into a child
        } else if (nch == 4) {
            const State &a = U[g[0]], &b = U[g[1]], &cc = U[g[2]], &d = U[g[3]];
            State m;
            for (int q = 0; q < kNVar; ++q) m[q] = ((a[q] + b[q]) + (cc[q] + d[q])) * 0.25;
            nU[static_cast<size_t>(k)] = m;
        } else {   // 3-D: each z-plane as in 2-D, then the two planes
            State m;
            for (int q = 0; q < kNVar; ++q) {
                const double p0 = (U[g[0]][q] + U[g[1]][q]) + (U[g[2]][q] + U[g[3]][q]);
                const double p1 = (U[g[4]][q] + U[g[5]][q]) + (U[g[6]][q] + U[g[7]][q]);
                m[q] = (p0 + p1) * 0.125;
            }
            nU[static_cast<size_t>(k)] = m;
        }
    }
    level.swap(nlevel);
    ci.swap(ni);
    cj.swap(nj);
    ck.swap(nkz);
    if (has_u) U.swap(nU);
    rebuild(true);
    return true;
}

}   // namespace fv
