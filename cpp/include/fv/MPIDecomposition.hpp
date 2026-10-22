// MPI 1-D strip decomposition with halo exchange.
// Y-direction (row strips) when every rank gets >= kHaloDepth rows, otherwise
// X-direction (column strips).
#pragma once

#ifdef FV_USE_MPI

#include <mpi.h>

#include "fv/Core.hpp"
#include "fv/Schemes.hpp"

namespace fv {

// Splits the global nx x ny mesh into contiguous strips, one per rank.
class MPIDecomposition {
public:
    // Decomposition over `comm`; throws if no direction gives every rank kHaloDepth cells.
    MPIDecomposition(int nx, int ny, MPI_Comm comm, int nz = 1);

    // Rank 0's global field -> each rank's local strip.
    Field scatter(const Field& U_global) const;
    // Local strips -> global field on rank 0 (empty elsewhere).
    Field gather(const Field& U_local) const;

    // Exchange kHaloDepth slices; has_lo/has_hi report whether a neighbour
    // (or periodic wrap) exists.  Layouts match Halos (Schemes.hpp).
    void exchange_halos(const Field& U_local, bool periodic, Field& halo_lo, Field& halo_hi, bool& has_lo,
                        bool& has_hi) const;

    // Global maximum of v over all ranks.
    double allreduce_max(double v) const;

    MPI_Comm comm;                               // communicator
    int rank = 0, n_ranks = 1;                   // this rank, number of ranks
    int nx, ny, nz = 1;                          // global mesh size (strips span all z layers)
    char dir = 'y';                              // 'y': row strips, 'x': column strips
    int nx_local = 0, ny_local = 0;              // size of this rank's strip
    int i_start = 0, j_start = 0;                // global index of the strip's first column / row
    bool is_lo_rank = true, is_hi_rank = true;   // bottom/top (y) or left/right (x)
    std::vector<int> counts, offsets;            // rows (y) or columns (x) per rank
};

}   // namespace fv

#endif   // FV_USE_MPI
