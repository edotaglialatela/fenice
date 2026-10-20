#ifdef FV_USE_MPI

#include "fv/MPIDecomposition.hpp"

#include <stdexcept>
#include <string>

namespace fv {

namespace {
constexpr int kTagUp   = 10;   // lower rank -> higher rank
constexpr int kTagDown = 20;   // higher rank -> lower rank
constexpr int kDoublesPerCell = kNVar;
}  // namespace

MPIDecomposition::MPIDecomposition(int nx_, int ny_, MPI_Comm c, int nz_) : comm(c), nx(nx_), ny(ny_), nz(nz_) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &n_ranks);
    if (n_ranks > std::max(nx, ny))
        throw std::invalid_argument("n_ranks (" + std::to_string(n_ranks) + ") exceeds both nx (" +
                                    std::to_string(nx) + ") and ny (" + std::to_string(ny) +
                                    "). Reduce the number of MPI ranks.");
    // Prefer row strips; each strip needs >= kHaloDepth cells for the halo exchange.
    // (Choosing 'y' whenever ny >= n_ranks would allow 1-row strips, whose halo
    // exchange reads uninitialised memory.)
    if (ny >= kHaloDepth * n_ranks || (n_ranks == 1)) dir = 'y';
    else if (nx >= kHaloDepth * n_ranks)              dir = 'x';
    else dir = (ny >= nx) ? 'y' : 'x';   // rejected below
    const int n = (dir == 'y') ? ny : nx;
    const int base = n / n_ranks, rem = n % n_ranks;
    counts.resize(n_ranks);
    offsets.resize(n_ranks);
    for (int r = 0, off = 0; r < n_ranks; ++r) {
        counts[r]  = base + (r < rem ? 1 : 0);
        offsets[r] = off;
        off += counts[r];
    }
    // The halo exchange sends kHaloDepth slices, so every strip needs at least that many.
    if (n_ranks > 1 && counts.back() < kHaloDepth)
        throw std::invalid_argument("Each MPI rank needs at least " + std::to_string(kHaloDepth) +
                                    " cells across its strip (" + std::to_string(n) + " cells over " +
                                    std::to_string(n_ranks) + " ranks). Reduce the number of MPI ranks.");
    if (dir == 'y') {
        ny_local = counts[rank]; nx_local = nx; j_start = offsets[rank];
    } else {
        nx_local = counts[rank]; ny_local = ny; i_start = offsets[rank];
    }
    is_lo_rank = (rank == 0);
    is_hi_rank = (rank == n_ranks - 1);
}

Field MPIDecomposition::scatter(const Field& U_global) const {
    Field local(static_cast<size_t>(nx_local) * ny_local * nz);
    std::vector<int> sc(n_ranks), so(n_ranks);
    Field packed;
    if (rank == 0) {
        // Pack each rank's block contiguously in its local order (k, j, i)
        packed.reserve(U_global.size());
        for (int r = 0; r < n_ranks; ++r) {
            so[r] = static_cast<int>(packed.size()) * kDoublesPerCell;
            for (int k = 0; k < nz; ++k) {
                if (dir == 'y') {
                    for (int j = offsets[r]; j < offsets[r] + counts[r]; ++j)
                        for (int i = 0; i < nx; ++i) packed.push_back(U_global[(k * ny + j) * nx + i]);
                } else {
                    for (int j = 0; j < ny; ++j)
                        for (int i = offsets[r]; i < offsets[r] + counts[r]; ++i) packed.push_back(U_global[(k * ny + j) * nx + i]);
                }
            }
            sc[r] = static_cast<int>(packed.size()) * kDoublesPerCell - so[r];
        }
    }
    MPI_Scatterv(rank == 0 ? packed.data()->data() : nullptr, sc.data(), so.data(), MPI_DOUBLE,
                 local.data()->data(), static_cast<int>(local.size()) * kDoublesPerCell, MPI_DOUBLE, 0, comm);
    return local;
}

Field MPIDecomposition::gather(const Field& U_local) const {
    std::vector<int> rc(n_ranks), ro(n_ranks);
    for (int r = 0, off = 0; r < n_ranks; ++r) {
        const int cells = ((dir == 'y') ? counts[r] * nx : counts[r] * ny) * nz;
        rc[r] = cells * kDoublesPerCell;
        ro[r] = off;
        off += rc[r];
    }
    Field packed(rank == 0 ? static_cast<size_t>(nx) * ny * nz : 0);
    MPI_Gatherv(U_local.data()->data(), static_cast<int>(U_local.size()) * kDoublesPerCell, MPI_DOUBLE,
                rank == 0 ? packed.data()->data() : nullptr, rc.data(), ro.data(), MPI_DOUBLE, 0, comm);
    if (rank != 0) return {};
    if (dir == 'y' && nz == 1) return packed;   // 2-D row strips are already in global order
    Field U(static_cast<size_t>(nx) * ny * nz);
    size_t c = 0;
    for (int r = 0; r < n_ranks; ++r)
        for (int k = 0; k < nz; ++k) {
            if (dir == 'y') {
                for (int j = offsets[r]; j < offsets[r] + counts[r]; ++j)
                    for (int i = 0; i < nx; ++i) U[(k * ny + j) * nx + i] = packed[c++];
            } else {
                for (int j = 0; j < ny; ++j)
                    for (int i = offsets[r]; i < offsets[r] + counts[r]; ++i) U[(k * ny + j) * nx + i] = packed[c++];
            }
        }
    return U;
}

void MPIDecomposition::exchange_halos(const Field& U, bool periodic, Field& lo, Field& hi,
                                      bool& has_lo, bool& has_hi) const {
    has_lo = has_hi = false;
    if (n_ranks == 1) return;
    constexpr int D = kHaloDepth;

    // Contiguous send buffers: the D slices nearest each interface.
    // Layouts (see Halos): y-strips [(d*nz + k)*nx + i], x-strips [(k*ny + j)*D + d]
    Field send_lo, send_hi;
    if (dir == 'y') {
        const int nl = ny_local;
        send_lo.resize(static_cast<size_t>(D) * nz * nx);
        send_hi.resize(send_lo.size());
        for (int d = 0; d < D; ++d)
            for (int k = 0; k < nz; ++k)
                for (int i = 0; i < nx; ++i) {
                    send_lo[(d * nz + k) * nx + i] = U[(k * nl + d) * nx + i];
                    send_hi[(d * nz + k) * nx + i] = U[(k * nl + nl - D + d) * nx + i];
                }
    } else {
        const int nl = nx_local;
        send_lo.resize(static_cast<size_t>(ny) * nz * D);
        send_hi.resize(send_lo.size());
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int d = 0; d < D; ++d) {
                    send_lo[(k * ny + j) * D + d] = U[(k * ny + j) * nl + d];
                    send_hi[(k * ny + j) * D + d] = U[(k * ny + j) * nl + nl - D + d];
                }
    }
    lo.resize(send_lo.size());
    hi.resize(send_lo.size());
    const int cnt = static_cast<int>(send_lo.size()) * kDoublesPerCell;

    MPI_Request reqs[4];
    int nreq = 0;
    // Upward / rightward (tag 10)
    if (!is_hi_rank)    MPI_Isend(send_hi.data()->data(), cnt, MPI_DOUBLE, rank + 1, kTagUp, comm, &reqs[nreq++]);
    else if (periodic)  MPI_Isend(send_hi.data()->data(), cnt, MPI_DOUBLE, 0, kTagUp, comm, &reqs[nreq++]);
    if (!is_lo_rank)    MPI_Irecv(lo.data()->data(), cnt, MPI_DOUBLE, rank - 1, kTagUp, comm, &reqs[nreq++]);
    else if (periodic)  MPI_Irecv(lo.data()->data(), cnt, MPI_DOUBLE, n_ranks - 1, kTagUp, comm, &reqs[nreq++]);
    // Downward / leftward (tag 20)
    if (!is_lo_rank)    MPI_Isend(send_lo.data()->data(), cnt, MPI_DOUBLE, rank - 1, kTagDown, comm, &reqs[nreq++]);
    else if (periodic)  MPI_Isend(send_lo.data()->data(), cnt, MPI_DOUBLE, n_ranks - 1, kTagDown, comm, &reqs[nreq++]);
    if (!is_hi_rank)    MPI_Irecv(hi.data()->data(), cnt, MPI_DOUBLE, rank + 1, kTagDown, comm, &reqs[nreq++]);
    else if (periodic)  MPI_Irecv(hi.data()->data(), cnt, MPI_DOUBLE, 0, kTagDown, comm, &reqs[nreq++]);
    MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);

    has_lo = !is_lo_rank || periodic;
    has_hi = !is_hi_rank || periodic;
}

double MPIDecomposition::allreduce_max(double v) const {
    double out = v;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_MAX, comm);
    return out;
}

}  // namespace fv

#endif  // FV_USE_MPI
