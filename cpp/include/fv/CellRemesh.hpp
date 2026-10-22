// Adaptation of a CellMesh from the solution: the face-variation indicator, the flags it gives, the
// finest-level band around an immersed body, and the remesh / initial adaptation drivers.
//
// Indicator (docs/design/amr-cell-based.md section 11):
//   1. xi = 0 in every leaf.
//   2. For every interior face and every enabled variable (density, pressure, velocity magnitude), with c_l, c_r the
//      values of its two leaves: m = max(min(|c_l|, |c_r|), floor), variation = |c_r - c_l| / m; if
//      variation >= criterion, both leaves get xi = 1.
//   3. `diffusion` iterations: every face whose leaves are at the same level and one of them has xi > trigger, or whose
//      finer leaf has xi > trigger (coarse-fine face), adds `diffusion_coeff` to both leaves; after all faces xi += added.
//   4. A leaf is refined if xi >= split, four siblings are joined if every one has xi < join.
#pragma once

#include <functional>
#include <vector>

#include "fv/CellFlux.hpp"
#include "fv/CellMesh.hpp"

namespace fv {

// Parameters of the indicator.  The defaults are a starting point for the user to fix (section 11).
struct CellIndicator {
    bool use_density = true, use_pressure = false, use_velocity = false;   // variables whose variation is tested
    double criterion = 0.1;          // relative variation across a face that flags both leaves
    double value_floor = 1e-2;       // floor of min(|c_l|, |c_r|); the velocity magnitude uses velocity_floor
    double velocity_floor = 0.1;
    int diffusion = 2;               // iterations of the smoothing
    double diffusion_coeff = 0.1;    // added to both leaves of a face per iteration
    double diffusion_trigger = 0.05; // a face spreads only if a leaf has xi above this
    double split = 0.15;             // refine at xi >= split
    double join = 0.10;              // join siblings when all have xi < join

    // The smoothing of step 3 applied to a given xi.
    void diffuse(const CellMesh& mesh, std::vector<double>& xi) const;
    // xi of every leaf of `mesh` for the state `U`.
    std::vector<double> xi(const CellMesh& mesh, const Field& U, const EOS& eos) const;
};

// Remesh driver: indicator -> flags -> CellMesh::adapt -> CellFluxEngine::build.
class CellAdaptor {
public:
    // The mesh (with its state U) and engine must outlive the object.
    CellAdaptor(CellMesh& mesh, CellFluxEngine& engine, const EOS& eos, const CellIndicator& ind)
        : mesh_(mesh), engine_(engine), eos_(eos), ind_(ind) {}

    // Keep the immersed body, and `margin` finest-level cells around it (the fringe of the solid reset), on the finest
    // level: `factory` gives the body mask on any uniform mesh. Call before remesh / init_adapt.
    void set_body(std::function<ImmersedBoundary(const Mesh&)> factory, int margin = 2);

    // One adaptation of the leaves from the current state; the engine is rebuilt when the leaves changed.
    // Returns whether they did.  xi of the call is kept in last_xi (per leaf of the mesh BEFORE the call).
    bool remesh();

    // Initial adaptation: `initial(mesh)` fills mesh.U at the leaf centres (each leaf at its own level); repeat
    // indicator + adapt until nothing changes (at most max_level + 1 passes), then fill once more. Returns the passes made.
    int init_adapt(const std::function<void(CellMesh&)>& initial);

    // True if the leaf lies in the finest-level band of the body.
    bool in_body_band(int leaf);

    std::vector<double> last_xi;
    // Accumulated wall time [s] of the three parts of remesh(): indicator + flags, CellMesh::adapt, CellFluxEngine::build.
    double t_indicator = 0.0, t_adapt = 0.0, t_build = 0.0;

private:
    void build_band();

    CellMesh& mesh_;
    CellFluxEngine& engine_;
    EOS eos_;
    CellIndicator ind_;
    std::function<ImmersedBoundary(const Mesh&)> body_;
    int margin_ = 2;
    std::vector<char> band_;   // body + margin on the uniform mesh of the finest level
};

}   // namespace fv
