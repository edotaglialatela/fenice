// Spatial schemes: HLLC-family Riemann solvers, TVD limiters, MUSCL reconstruction.
#pragma once

#include "fv/Core.hpp"

namespace fv {

// ---------------------------------------------------------------------------
// Riemann solvers
// ---------------------------------------------------------------------------

// Approximate Riemann solver variants.
enum class RiemannType { HLLC, LowMachHLLC, PreconditionedHLLC };

// Accepts GUI labels ("HLLC (Low-Mach Thornber)") and short aliases ("lowmach").
RiemannType parse_riemann(const std::string& s);

// HLLC-family face-flux solver (stateless, thread-safe).
class RiemannSolver {
public:
    RiemannSolver() = default;
    // Solver of the given type; `ma_cut` is the cut-off Mach number of the preconditioned variant.
    RiemannSolver(const EOS& eos, RiemannType type, double ma_cut = 0.1) : eos_(eos), type_(type), ma_cut_(ma_cut) {}

    // Numerical flux at every face of `mesh`; returns the maximum wave speed.
    double compute_flux(const Field& UL, const Field& UR, const Mesh& mesh, Field& F) const;

    // Flux through one face normal to xis (0 = x, 1 = y, 2 = z); smax receives max(|S_L|, |S_R|).
    State face_flux(const State& UL, const State& UR, int axis, double& smax) const;

    // Fluxes of n faces: face k lies between UL[k] and UR[k] and is normal to axis[k] (0, 1, 2); F[k] and smax[k] get
    // what face_flux gives, bit for bit (four faces at a time with AVX2 where the CPU has it, else one by one).
    void face_flux_batch(const State* UL, const State* UR, const int* axis, int n, State* F, double* smax) const;

    // Solver variant.
    RiemannType type() const { return type_; }

private:
    State hllc(const State& UL, const State& UR, int axis, double& smax) const;

    EOS eos_;
    RiemannType type_ = RiemannType::HLLC;
    double ma_cut_ = 0.1;
};

// ---------------------------------------------------------------------------
// TVD limiters  phi(a, b): a = upwind delta, b = downwind delta
// ---------------------------------------------------------------------------

// TVD slope limiters, from most diffusive (Minmod) to most compressive (MC).
enum class Limiter { Minmod = 0, VanLeer = 1, MC = 2, Koren = 3 };

// Limiter from its name (case-insensitive); unknown names throw std::invalid_argument.
Limiter parse_limiter(const std::string& s);

// Sign of x (0 for 0).
inline double sgn(double x) { return (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0); }

// Minmod: 0.5 (sgn a + sgn b) min(|a|, |b|).
inline double lim_minmod(double a, double b) { return 0.5 * (sgn(a) + sgn(b)) * std::min(std::abs(a), std::abs(b)); }
// Van Leer: harmonic mean 2ab/(a+b) when ab > 0, else 0.
inline double lim_vanleer(double a, double b) {
    const double ab = a * b;
    return (ab > 0) ? 2.0 * ab / (a + b) : 0.0;
}
// Monotonized central: minmod(2a, (a+b)/2, 2b).
inline double lim_mc(double a, double b) {
    if (!(a * b > 0)) return 0.0;
    return std::min(std::min(2.0 * std::abs(a), 0.5 * std::abs(a + b)), 2.0 * std::abs(b)) * sgn(a);
}
// Koren: minmod(2a, (2a+b)/3, 2b), third order in smooth regions.
inline double lim_koren(double a, double b) {
    if (!(a * b > 0)) return 0.0;
    return std::fmin(std::fmin(2.0 * std::abs(a), std::abs(2.0 * a + b) / 3.0), 2.0 * std::abs(b)) * sgn(a);
}
// Dispatch phi(a, b) to the chosen limiter.
inline double apply_limiter(Limiter l, double a, double b) {
    switch (l) {
    case Limiter::Minmod: return lim_minmod(a, b);
    case Limiter::VanLeer: return lim_vanleer(a, b);
    case Limiter::MC: return lim_mc(a, b);
    default: return lim_koren(a, b);
    }
}

// ---------------------------------------------------------------------------
// MUSCL reconstruction
// ---------------------------------------------------------------------------

// Ghost-cell depth required by the MUSCL stencil (== MPI halo depth).
constexpr int kHaloDepth = 2;

// Optional halos (MPI neighbours or periodic wrap-around) in *conservative* variables,
// kHaloDepth slices deep (d = 0 is the slice farthest from the interface):
//   b / t  : below / above  [ (d*nz + k)*nx + i ]
//   l / r  : left / right   [ (k*ny + j)*kHaloDepth + d ]
//   f / bk : front / back   [ (d*ny + j)*nx + i ]          (3-D only)
// With nz = 1 these reduce to the 2-D layouts [d*nx + i] and [j*kHaloDepth + d].
struct Halos {
    const Field* b = nullptr;    // below
    const Field* t = nullptr;    // above
    const Field* l = nullptr;    // left
    const Field* r = nullptr;    // right
    const Field* f = nullptr;    // front (z-min)
    const Field* bk = nullptr;   // back (z-max)
};
class MusclReconstruction {
public:
    // Reconstruction with limiter `lim`.
    MusclReconstruction(const EOS& eos, Limiter lim) : eos_(eos), lim_(lim) {}

    // Upgrade first-order face states (in place) to 2nd-order MUSCL states.
    // Physical-boundary faces keep their 1st-order ghost states; MPI-interface
    // faces are upgraded when the matching halo is supplied.
    void reconstruct(Field& UL, Field& UR, const Field& U, int nx, int ny, const Halos& h = {}) const;
    // 3-D version (z-sweep added); m gives nx, ny, nz.
    void reconstruct_3d(Field& UL, Field& UR, const Field& U, const Mesh& m, const Halos& h = {}) const;

    // Active limiter.
    Limiter limiter() const { return lim_; }

private:
    EOS eos_;
    Limiter lim_;
};

}   // namespace fv
