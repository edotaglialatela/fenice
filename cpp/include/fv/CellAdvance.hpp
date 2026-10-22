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

#include <functional>

#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"
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

    // Test hook: false makes a coarse cell read the last computed flux of a coarse-fine face instead of the
    // register (the conservation test must then fail).
    bool flux_replacement = true;
    // Test hook, called before every stage residual with the level and the stage time set in the scheme.
    std::function<void(int level, double stage_time)> stage_hook;

private:
    double advance(int level, double dt, double t);
    double advance_own(int level, double dt, double t);

    CellMesh& mesh_;
    CellFluxEngine& engine_;
    TimeScheme& scheme_;
};

}   // namespace fv
