// A coarse profile of an AMR run, off by default (FENICE_PROFILE=1 prints it at the end): the wall time of the time step, the output
// and each part of the remesh, one number per part and rank. Sections are timed once per step or per remesh, never per face or
// cell; a section nested in another (Fetch and the Share* parts in RemeshRebuild) is counted in both.
#pragma once

#include <chrono>
#include <cstdio>
#include <vector>

namespace fv {

enum class Section {
    Step,               // one time step of all levels (the recursion with subcycling)
    Output,             // the snapshot and checkpoint written at a save interval
    Remesh,             // the whole remesh, with the wave-speed bound recomputed after it
    RemeshRegion,       // the region of the mesh the indicator reads, fetched with its state
    RemeshFlags,        // the indicator on the owned leaves and the exchange of the flags
    RemeshTopology,     // the adaptation of the topology (ripple, merges)
    RemeshMigrate,      // the state of the new leaves and their move to their owners
    RemeshRebuild,      // the share of the new mesh and its engine tables (contains the Share* parts)
    RemeshPartition,    // the partition, the localisation and the halo lists after the build
    Fetch,              // fetch_leaves: the leaves around a region asked from their owners
    ShareRegion,        // the mask of the leaves a share keeps
    ShareMesh,          // the partial mesh built from the kept leaves
    ShareEngine,        // CellFluxEngine::build on the partial mesh
    SharePartition,     // partition_local: the owners, the splitters and the lists
    ShareLocalize,      // the re-indexing of the engine and the halo to the share
    kCount
};

// The name printed for a section.
const char* section_name(Section s);

// Whether FENICE_PROFILE is on (read once). When it is off, nothing below costs more than this test.
bool profile_enabled();

// Add `seconds` to a section (one call).
void profile_add(Section s, double seconds);

// Times its own scope into a section.
class ProfileScope {
public:
    explicit ProfileScope(Section s) : section_(s), on_(profile_enabled()) {
        if (on_) start_ = std::chrono::steady_clock::now();
    }
    ~ProfileScope() {
        if (on_) profile_add(section_, std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count());
    }
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;

private:
    Section section_;
    bool on_;
    std::chrono::steady_clock::time_point start_;
};

// The accumulated numbers of this rank, flat: seconds of every section, then the number of calls of every section
// (2 * kCount values), in the order of Section. Empty when the profile is off.
std::vector<double> profile_values();

// Print the profile of all ranks: `all` holds profile_values() of every rank one after the other (what an all-gather of them
// gives), `nranks` is their count. One line per section that ran: calls on rank 0, seconds min / max over the ranks.
void print_profile(std::FILE* out, const std::vector<double>& all, int nranks);

// Forget the numbers (the profile of a run starts after the setup).
void profile_reset();

}   // namespace fv
