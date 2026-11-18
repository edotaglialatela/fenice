// Recursive subcycled time stepping of a CellMesh with flux replacement at the coarse-fine faces.
//
// Level l advances with dt_l = dt_0 / 2^l; before a level advances, the level above it (finer) does its two
// substeps, so the coarse level reads the final fine state and the fine level reads its coarse neighbours at the
// start of the coarse step (no time interpolation; the coarse side of a jump is first order and uses the
// registered flux, see below).  The leaves of one level are one contiguous run of the leaf arrays and form one
// ODE system for TimeScheme::advance_fixed_dt.
//
// Flux replacement: every stage of a fine level computes the flux of its coarse-fine faces and adds
// (dt_fine / dt_coarse) * b_i * F_i to the face's register (b_i = TimeScheme::stage_weights).  The coarse cell
// then uses the registered time-average in every stage of its own advance, so what it loses through the face is
// exactly what the fine cells gained: conservation to roundoff for every explicit scheme.
#pragma once

#include <array>
#include <functional>
#include <vector>

#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"
#include "fv/CellParallel.hpp"
#include "fv/TimeScheme.hpp"

namespace fv {

// Subcycled time stepping of a CellMesh; see the file header.
class CellAdvance {
public:
    // The mesh, engine (built for this mesh) and scheme must outlive the object.  Explicit schemes only.
    CellAdvance(CellMesh& mesh, CellFluxEngine& engine, TimeScheme& scheme)
        : mesh_(mesh), engine_(engine), scheme_(scheme) {}

    // Advance the whole hierarchy by dt from time t (dt is never split unevenly: clamp it before the call);
    // returns the largest wave speed seen by the first stage of every level.
    double step(double dt, double t);

    // Distributed run: set before the first step (after CellParallel::partition, which must be repeated after every
    // adapt). The recursion is the same on every rank: each level advances the owned leaves, the owners exchange the stage
    // states, and the fine owners send the registers before the coarse level advances.
    CellParallel* par = nullptr;
    // Test hook: false makes a coarse cell read the last computed flux of a coarse-fine face instead of the
    // register (the conservation test must then fail).
    bool flux_replacement = true;
    // positivity_fallback (opt-in): after each stage of a level the new state of the owned leaves is scanned with `admissible`; where
    // a leaf is not admissible, it joins the flagged set and the stage is repeated with first-order faces around the set (the face
    // kernels of CellFluxEngine), until no new leaf turns up (TimeScheme::kMaxRepairs). A repeated stage replaces its registers, it
    // does not add to them, so the flux replacement stays conservative. Without the call nothing is scanned.
    // A leaf flagged in its own stage repairs its same-level and boundary faces; a coarse-fine face is computed by the finer level
    // and enters the coarse leaf through the register, so it keeps the flux of the fine stage.
    void enable_positivity_fallback(std::function<bool(const State&)> admissible);
    long long positivity_repairs() const { return repairs_; }   // repeated passes, counted on every rank (the same number)
    long long positivity_flags() const { return flagged_; }     // leaves flagged over the run on this rank (owned leaves)
    ~CellAdvance();
    // Test hook, called before every stage residual with the level and the stage time set in the scheme.
    std::function<void(int level, double stage_time)> stage_hook;

private:
    double advance(int level, double dt, double t);
    double advance_own(int level, double dt, double t);

    CellMesh& mesh_;
    CellFluxEngine& engine_;
    TimeScheme& scheme_;
    // per level: the state of the owned leaves being advanced and the work fields of the scheme (kept between the steps: a
    // level's fields keep their size, nothing is allocated or filled in a step)
    std::vector<Field> ul_;
    std::vector<std::array<Field, 5>> work_;
    // positivity_fallback
    StageGuard guard_;
    std::function<bool(const State&)> admissible_;
    bool repeat_ = false;     // the next residual call repeats the stage just scanned
    int cur_level_ = 0, cur_b_ = 0;
    Field reg_conv_, reg_visc_;   // the registers of the level before its current stage
    long long repairs_ = 0, flagged_ = 0;
};

}   // namespace fv
