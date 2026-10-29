// Distributed run of a CellMesh: replicated topology, distributed state.
//
// Every rank holds the whole mesh (leaf arrays, faces, stencils: the metadata), so every rank takes the same remesh
// decisions; the state of a leaf is valid only on the rank that owns it and on the ranks that read it (the halo).
//  * Ownership: each level is cut into contiguous blocks of equal leaf count along its Morton order, described by per-level
//    splitters (the first key of each rank's block), so the owner of a leaf follows from its (level, key).
//  * Work: a rank advances its owned leaves and computes every face it needs for them: a same-level face is computed by the
//    owners of both its leaves (redundantly, with identical inputs, hence identical bits), a coarse-fine face and a boundary
//    face by the owner of the finer leaf. The coarse owner receives the time-averaged registers of its coarse-fine faces
//    from the fine owners once per coarse step.
//  * Halo = exactly the leaves the engine reads for those faces (CellFluxEngine::face_dependencies). Per stage of a level the
//    owners push the stage state of the leaves other ranks read at that level; after a level has advanced they push its final
//    state to every rank that reads it. Each rank works out only its own reads and sends them to the owners as (level, i, j)
//    requests; nothing that crosses ranks is a leaf index, and no rank works out what the others read.
//  * Results are bit-identical to the serial run for any number of ranks: the arithmetic of a face and of a cell does not
//    depend on the rank, the maxima are order independent, and nothing is summed across ranks.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"

namespace fv {

class CellAdaptor;

// Communication hooks (the same shape as for MPI; tests provide an in-process version).
struct CellComm {
    int rank = 0, size = 1;
    // Gather the `n` doubles of every rank into `recv` (rank r's block has counts[r], at displs[r]).
    std::function<void(const double* send, int n, double* recv, const int* counts, const int* displs)> allgatherv;
    std::function<double(double)> allreduce_max;   // max over the ranks
    std::function<double(double)> allreduce_sum;   // sum over the ranks (in rank order: every rank gets the same value)
    // Sparse all-to-all: send[r] goes to rank r, recv[r] (pre-sized by the caller) is filled from rank r; an empty buffer means
    // no message. Every rank must call it at the same points.
    std::function<void(const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv)> exchange;
};

// What every rank knows about the whole mesh without holding it: the number of leaves of each level and the splitters (rank r owns at
// level l the keys [split[l][r], split[l][r + 1]); the position floor(n r / R) of the level, as partition() cuts it).
struct GlobalFacts {
    std::vector<long long> count;
    std::vector<std::vector<uint64_t>> split;
};

// A part of the mesh asked of the owners: the leaves with a level in [tmin, tmax] that intersect the cells of `box` (cells of `level`).
struct RegionItem {
    int level = 0;
    CellBox box;
    int tmin = 0, tmax = 0;
};

// What fetch_leaves exchanged with each rank, in the same order on both sides: the leaves received from rank r, and the indices
// (in the responder's mesh) of the leaves sent to it.
struct FetchPlan {
    std::vector<std::vector<std::array<int, 3>>> from;
    std::vector<std::vector<int>> served;
};

class CellParallel {
public:
    // The mesh (with its state U) and the engine (built for the mesh) must outlive the object.
    CellParallel(CellMesh& mesh, CellFluxEngine& engine, CellComm comm) : mesh_(mesh), engine_(engine), comm_(std::move(comm)) {}

    // Owners, the engine's partition, and the send / receive lists: call after every CellFluxEngine::build (that is, after
    // every adapt) and once at the start; the state of every leaf must be valid on every rank at that point. Collective (two
    // exchanges: the request counts, then the requests).
    // With `whole` (the remesh of a localized run) the mesh is a partial one made by CellMesh::build_from and `whole` supplies the
    // whole mesh's splitters and levels.
    void partition(const CellMesh* whole = nullptr, const GlobalFacts* facts = nullptr);
    // The facts of the whole mesh from the leaves each rank has (`keys[level]`: the Morton keys of this rank's leaves of the level,
    // sorted; the leaves of all ranks tile the mesh once, wherever they are). The splitters are found by a bisection on the key, all
    // levels and boundaries together (collective): the k-th smallest key of a level, exactly.
    GlobalFacts global_facts(const std::vector<std::vector<uint64_t>>& keys) const;

    // Keep only what this rank holds (call right after partition()): the mesh becomes a partial mesh of the held leaves (owned and
    // halo) and the faces it needs, the engine's tables and the lists of this object are re-indexed to it, and everything else is
    // freed. Nothing is recomputed, so the results are bit for bit those of the replicated run. A leaf index is local from then on
    // (the leaves keep their (level, key) order); until globalize() only the time stepping, the exchanges and the monitor sums
    // may be used: not partition(), gather_state() or an adapt.
    void localize();
    // Undo localize() (collective): every rank gets the whole mesh back from the owned leaves of all ranks (assign_leaves, so
    // the engine must be built again, then partition() and localize()) with the state of every leaf.
    void globalize();
    bool localized() const { return localized_; }
    // The remesh of a localized run (collective), without gathering the mesh or its state: `topo` is the whole mesh's topology
    // (leaves and faces, no state) that every rank keeps. Each rank takes the indicator of its owned leaves from the leaves
    // CellIndicator::region lists for them (those it does not hold are fetched from their owners), the flags of all ranks are
    // gathered (one byte per leaf), every rank applies them to `topo` (CellMesh::adapt: ripple and merges, identical everywhere),
    // and the state of the new owned leaves is built from the old leaves it comes from (fetched where not held). If the leaves
    // changed, the mesh and the engine are rebuilt, then partition() and localize(), and the halo is filled; returns whether
    // they did (if not, nothing was touched). The decisions and the state are those of ad.remesh() on the whole mesh, bit for bit.
    bool remesh(CellMesh& topo, CellAdaptor& ad);
    // The adaptation of a fresh start without a whole mesh on any rank: `topo` is the faceless topology of the base mesh (already
    // refined to the initial levels); every rank builds its share of it directly, then the passes of CellAdaptor::init_adapt run
    // through remesh(), `fill` giving the initial condition on the share (all its leaves). Returns the number of passes.
    int init_adapt(CellMesh& topo, CellAdaptor& ad, const std::function<void(CellMesh&)>& fill);
    // Test switches of remesh(): `direct_build` false makes the mesh and the engine for the whole mesh and extracts the share
    // (the way of phase 2); `shadow` also does that and throws if the direct build gave anything else (mesh, tables, lists).
    // `first_extra`: rings added to the first reach tried (a negative one makes the retry run); `region_retries` counts the retries.
    bool direct_build = true, shadow = false;
    int first_extra = 1, region_retries = 0;
    // The leaves of the mesh that intersect the region `items` (collective). The requests go to the ranks that own the leaves under
    // the splitters `split` (split[level][rank] = first key of the rank, then the end); each answers from `responder`, a mesh that holds
    // at least its own leaves (with their state `resp_state`, or none: then the result has no state). `out` gets the leaves found,
    // the rank's own among them, as a partial mesh. Level by level, coarse to fine; a cell inside a leaf already found is not asked.
    void fetch_leaves(const std::vector<std::vector<uint64_t>>& split, const CellMesh& responder, const Field* resp_state,
                      const std::vector<RegionItem>& items, CellMesh& out, FetchPlan* plan = nullptr) const;
    // The rank that owns the cell (level, key) under the splitters `split`.
    static int owner_in(const std::vector<std::vector<uint64_t>>& split, int level, uint64_t key);
    const std::vector<std::vector<uint64_t>>& splitters() const { return split_; }
    // True if the lists and the ownership of the two objects are the same (the shadow check).
    bool same_lists(const CellParallel& other) const;
    // Accumulated wall time [s] of remesh(): region + fetch + indicator, flags + adapt of topo, state migration, engine build,
    // partition + localize + halo.
    // The timers of the remesh since the last reset (the initial passes are reset out of them).
    void reset_timers() { t_topo = t_region = t_flags = t_migrate = t_rebuild = t_repartition = 0.0; region_retries = 0; }
    double t_topo = 0.0, t_region = 0.0, t_flags = 0.0, t_migrate = 0.0, t_rebuild = 0.0, t_repartition = 0.0;
    // True if the whole mesh (not only this rank's share) has leaves at `level`. Every rank must run the collective calls of a level
    // that exists anywhere, also when it holds no leaf of it (a localized mesh may have an empty level that others do not).
    bool level_present(int level) const { return level_present_[static_cast<size_t>(level)] != 0; }
    // Every rank gets the leaves of all ranks, in (level, key) order: the arrays of a whole mesh and its state (collective; the
    // mesh may be localized or not). For the checkpoint and the dump. With `root_only` only rank 0 gets them (the other ranks' arrays
    // are empty): the other ranks never hold a copy of the whole state.
    void collect(std::vector<int>& level, std::vector<int>& ci, std::vector<int>& cj, Field& U, bool root_only = false) const;

    // Test switch: after partition() the state of the leaves this rank does not hold is set to NaN, so that a missing halo
    // cell cannot go unnoticed.
    bool poison = false;

    int rank() const { return comm_.rank; }
    int size() const { return comm_.size; }
    // Rank that owns a leaf.
    int owner(int leaf) const { return owner_[static_cast<size_t>(leaf)]; }
    // Rank that owns the leaf (level, Morton key) of the current partition (a binary search of the level's splitters).
    int owner_of(int level, uint64_t key) const;
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
    // Sum of the ranks' values (collective). Across rank counts it agrees to roundoff only (the partial sums are added in another order).
    double reduce_sum(double v) const { return comm_.size > 1 ? comm_.allreduce_sum(v) : v; }

    // How far the reads of a distributed run reach (docs/design/amr-distributed.md, phase 0), independent of the partition: every
    // read is measured from the leaf whose owner makes it (the driver), in rings of the driver's level (see ring()).
    struct HaloReach {
        enum Kind { Interior, CoarseFine, Boundary, GhostImage, SolidReset, NumKinds };
        // Interior: reads of a same-level face, from either of its leaves (both owners compute it); CoarseFine: of a
        // coarse-fine face, from its fine leaf; Boundary: of a boundary face, from its leaf; GhostImage: the image leaves of a
        // ghost leaf that a finest-level face reads, from that face's driver; SolidReset: the reads of the solid reset
        // (neighbours, ghost images), from the solid leaf.
        // Largest ring per kind and level difference: ring[kind][max_level + level(driver) - level(read)] (max_level: the same
        // level, one more: the read leaf one level coarser, one less: one level finer); -1 where no such read was seen.
        std::vector<int> ring[NumKinds];
        std::vector<long long> reads;   // number of reads (with repetitions) per ring, all kinds together
        int max(int kind) const;        // largest ring of a kind (0 if none)
        int max() const;                // of all kinds
    };
    static HaloReach halo_reach(const CellMesh& mesh, const CellFluxEngine& engine);
    // Ring of leaf b seen from leaf a: 0 if b == a, else the smallest k >= 1 such that b overlaps the square of a dilated by k
    // cells of a's level (1: b touches a, even at a corner only); periodic axes wrap.
    static int ring(const CellMesh& mesh, int a, int b);

    // Leaves of level l owned by rank r: [owned_begin, owned_end).
    int owned_begin(int r, int l) const { return ob_[static_cast<size_t>(r)][static_cast<size_t>(l)]; }
    int owned_end(int r, int l) const { return oe_[static_cast<size_t>(r)][static_cast<size_t>(l)]; }
    // Number of leaves held (owned + halo) on this rank.
    int held_count() const;
    // Number of leaves read by this rank but owned elsewhere (the halo).
    int halo_count() const;

private:
    struct Lists {
        // per rank: leaf indices in increasing order, i.e. by (level, key); face indices of the registers ordered by their
        // (fine leaf, side)
        std::vector<std::vector<int>> send, recv;
    };
    void push_leaves(const Lists& lists);
    // partition() in two halves: the first communicates nothing and, with `whole`, returns false if the partial mesh does not
    // cover what this rank's faces read (the caller then makes a larger one on every rank before going on); the second is the
    // exchange of the requests and the lists that need the other ranks.
    bool partition_local(const CellMesh* whole, const GlobalFacts* facts = nullptr);
    void partition_exchange();
    // The share of the leaves `mine` (indices in `topo`): the reach of region_mask with retries, then partition_exchange and
    // localize. `nU` is the state of `topo` (entries of the leaves in the share), or null. `built` gets the time the tables were done.
    void build_share(const CellMesh& topo, const std::vector<int>& mine, const Field* nU,
                     std::chrono::steady_clock::time_point* built);
    // The share of the new owned leaves (arrays, state, the facts of the new mesh): the region around them is fetched from the other
    // owners (fetch_leaves), the partial mesh, the engine tables and the partition are made from it, retrying with a wider reach if a
    // face reads what the share lacks; then partition_exchange and localize.
    void build_share_fetched(const std::vector<int>& lv, const std::vector<int>& ii, const std::vector<int>& jj, const Field& U,
                             const GlobalFacts& gf, std::chrono::steady_clock::time_point* built);
    // The leaves of `topo` this rank owns under the equal-count split of each level.
    void owned_leaves(const CellMesh& topo, std::vector<int>& out) const;
    // The leaves a rank keeps to build its share directly: those within the reach of its owned leaves of `topo` (the whole mesh
    // after the adapt), `extra` rings more than the measured reach, and the leaves along the mass-flow inlets.
    void region_mask(const CellMesh& topo, const std::vector<int>& owned, int extra, std::vector<char>& keep) const;
    // The states of the leaves `want` ((level, i, j), each owned by some rank under the current partition): from this rank's leaves
    // if it holds them, else from their owners (collective).
    void fetch_states(const std::vector<std::array<int, 3>>& want, std::vector<State>& out) const;
    // Sparse all-to-all of vectors of any length: the sizes first, then the data (recv is sized here). send[own rank] must be empty.
    void exchange_sized(const std::vector<std::vector<double>>& send, std::vector<std::vector<double>>& recv) const;
    // The owned leaves of all ranks as records (level, ci, cj, state), 3 + kNVar doubles each, in rank order
    // (with `root_only`, on rank 0 only).
    void gather_owned(std::vector<double>& all, bool root_only = false) const;

    std::unique_ptr<CellMesh> sub_;   // the partial mesh of remesh(), kept to avoid rebuilding the base mesh each time
    bool localized_ = false;
    std::vector<char> level_present_;   // per level: the whole mesh has leaves there (set by partition())
    std::vector<char> face_active_;   // the faces this rank computes (of the last partition(); dropped by localize())
    CellMesh& mesh_;
    CellFluxEngine& engine_;
    CellComm comm_;
    std::vector<std::vector<uint64_t>> split_;   // [level][rank]: first Morton key of each rank's block (R + 1 entries)
    std::vector<int> owner_;
    std::vector<char> held_;
    std::vector<std::vector<int>> ob_, oe_;   // [rank][level]
    std::vector<std::vector<double>> req_;    // between the two halves of partition(): the requests to each owner
    std::vector<char> stage_mask_;            // ... the leaves read in a stage of their own level
    std::vector<Lists> stage_, final_, reg_;  // per level
    Lists solid_;
};

}   // namespace fv
