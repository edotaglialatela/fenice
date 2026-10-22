// Distributed run of a CellMesh: replicated topology, distributed state.
//
// Every rank holds the whole mesh (leaf arrays, faces, stencils: the metadata), so every rank takes the same remesh
// decisions; the state of a leaf is valid only on the rank that owns it and on the ranks that read it (the halo).
//  * Ownership: each level is cut into contiguous blocks of equal leaf count along its Morton order.
//  * Work: a rank advances its owned leaves and computes every face it needs for them: a same-level face is computed by the
//    owners of both its leaves (redundantly, with identical inputs, hence identical bits), a coarse-fine face and a boundary
//    face by the owner of the finer leaf. The coarse owner receives the time-averaged registers of its coarse-fine faces
//    from the fine owners once per coarse step.
//  * Halo = exactly the leaves the engine reads for those faces (CellFluxEngine::face_dependencies). Per stage of a level the
//    owners push the stage state of the leaves other ranks read at that level; after a level has advanced they push its final
//    state to every rank that reads it.
//  * Results are bit-identical to the serial run for any number of ranks: the arithmetic of a face and of a cell does not
//    depend on the rank, the maxima are order independent, and nothing is summed across ranks.
#pragma once

#include <functional>
#include <vector>

#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"

namespace fv {

// Communication hooks (the same shape as for MPI; tests provide an in-process version).
struct CellComm {
    int rank = 0, size = 1;
    // Gather the `n` doubles of every rank into `recv` (rank r's block has counts[r], at displs[r]).
    std::function<void(const double* send, int n, double* recv, const int* counts, const int* displs)> allgatherv;
    std::function<double(double)> allreduce_max;   // max over the ranks
    // Sparse all-to-all: send[r] goes to rank r, recv[r] (pre-sized by the caller) is filled from rank r; an empty buffer means
    // no message. Every rank must call it at the same points.
    std::function<void(const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv)> exchange;
};

class CellParallel {
public:
    // The mesh (with its state U) and the engine (built for the mesh) must outlive the object.
    CellParallel(CellMesh& mesh, CellFluxEngine& engine, CellComm comm) : mesh_(mesh), engine_(engine), comm_(std::move(comm)) {}

    // Owners, the engine's partition, and the send / receive lists: call after every CellFluxEngine::build (that is, after
    // every adapt) and once at the start; the state of every leaf must be valid on every rank at that point.
    void partition();

    // Test switch: after partition() the state of the leaves this rank does not hold is set to NaN, so that a missing halo
    // cell cannot go unnoticed.
    bool poison = false;

    int rank() const { return comm_.rank; }
    int size() const { return comm_.size; }
    // Rank that owns a leaf.
    int owner(int leaf) const { return owner_[static_cast<size_t>(leaf)]; }
    bool holds(int leaf) const { return held_[static_cast<size_t>(leaf)] != 0; }

    // Push the stage state of the owned leaves of `level` that other ranks read in the faces of that level.
    void exchange_stage(int level);
    // Push the final state of the owned leaves of `level` to every rank that reads them.
    void exchange_final(int level);
    // Push the owned solid leaves (after the solid reset).
    void exchange_solid();
    // Fine owners send the registers of the coarse-fine faces of `level` (level >= 1) to the owners of the coarse leaves.
    void exchange_registers(int level);
    // Every rank gets the state of every leaf (before the indicator, and to write results).
    void gather_state();
    double reduce_max(double v) const { return comm_.size > 1 ? comm_.allreduce_max(v) : v; }

    // Leaves of level l owned by rank r: [owned_begin, owned_end).
    int owned_begin(int r, int l) const { return ob_[static_cast<size_t>(r)][static_cast<size_t>(l)]; }
    int owned_end(int r, int l) const { return oe_[static_cast<size_t>(r)][static_cast<size_t>(l)]; }
    // Number of leaves held (owned + halo) on this rank.
    int held_count() const;
    // Number of leaves read by this rank but owned elsewhere (the halo).
    int halo_count() const;

private:
    struct Lists {
        std::vector<std::vector<int>> send, recv;   // per rank: leaf (or face) indices, in increasing order
    };
    void push_leaves(const Lists& lists);
    void build_lists(Lists& out, const std::vector<std::vector<char>>& need, int level_begin, int level_end,
                     const std::vector<char>* only) const;

    CellMesh& mesh_;
    CellFluxEngine& engine_;
    CellComm comm_;
    std::vector<int> owner_;
    std::vector<char> held_;
    std::vector<std::vector<int>> ob_, oe_;   // [rank][level]
    std::vector<Lists> stage_, final_, reg_;  // per level
    Lists solid_;
};

}   // namespace fv
