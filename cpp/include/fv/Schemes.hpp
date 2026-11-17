// Spatial schemes: HLLC-family Riemann solvers, TVD limiters, MUSCL reconstruction.
#pragma once

#include "fv/Core.hpp"

namespace fv {

// ---------------------------------------------------------------------------
// Riemann solvers
// ---------------------------------------------------------------------------

// Approximate Riemann solver variants.
enum class RiemannType { HLLC, LowMachHLLC };

// Accepts GUI labels ("HLLC (Low-Mach Thornber)") and short aliases ("lowmach").
RiemannType parse_riemann(const std::string& s);

enum class Limiter;   // defined below

// HLLC-family face-flux solver (stateless, thread-safe).
class RiemannSolver {
public:
    RiemannSolver() = default;
    // Solver of the given type. `shock_blend` > 0 switches on the shock-face dissipation (see face_flux).
    RiemannSolver(const EOS& eos, RiemannType type, double shock_blend = 0.0) : eos_(eos), type_(type), shock_blend_(shock_blend) {}

    // Numerical flux at every face of `mesh`; returns the maximum wave speed.
    double compute_flux(const Field& UL, const Field& UR, const Mesh& mesh, Field& F) const;

    // Flux through one face normal to xis (0 = x, 1 = y, 2 = z); smax receives max(|S_L|, |S_R|).
    State face_flux(const State& UL, const State& UR, int axis, double& smax) const;

    // Fluxes of n faces: face k lies between UL[k] and UR[k] and is normal to axis[k] (0, 1, 2); F[k] and smax[k] get
    // what face_flux gives, bit for bit (four faces at a time with AVX2 where the CPU has it, else one by one).
    void face_flux_batch(const State* UL, const State* UR, const int* axis, int n, State* F, double* smax) const;

    // First-order scheme on a 2-D mesh (nz = 1) in one pass over rows of faces: R gets the flux balance of the cells,
    // ((0 - (Fr dy - Fl dy)) - (Ft dx - Fb dx)) / area, of the fluxes between the cell states U; the faces of the sides take UL / UR
    // (BoundaryManager::apply_boundary_conditions; the faces between two cells need not be set) and the face fluxes are never
    // stored. Same values as compute_flux followed by the divergence. Returns the maximum wave speed.
    double first_order_residual_2d(const Field& UL, const Field& UR, const Field& U, const Mesh& m, Field& R) const;

    // Like face_flux_batch for faces whose left / right states are the MUSCL reconstruction from the primitive states
    // W[st[4k]] .. W[st[4k+3]] (two cells each side of face k) with limiter `lim`: wl = c1 + 0.5 phi(c1 - c0, c2 - c1),
    // wr = c2 - 0.5 phi(c3 - c2, c2 - c1) per variable, converted to conservative form. Bit-identical to doing so one face at a time.
    // `planar`: a 2-D run (rho w = 0 everywhere): the fifth variable is not reconstructed, its face states are +0 (the same bits).
    void muscl_face_flux_batch(Limiter lim, const State* W, const int* st, const int* axis, int n, State* F, double* smax,
                               bool planar = false) const;

    // The same for the faces of one line of the uniform mesh, normal to `axis`: the primitive state is stored per variable (plane q
    // at W + q * plane) and face k of the batch has its four stencil points at base + k + p * stride, p = 0 .. 3.
    void muscl_line_flux_batch(Limiter lim, const double* W, size_t plane, size_t base, size_t stride, int axis, int n, State* F,
                               double* smax, bool planar) const;

    // Solver variant.
    RiemannType type() const { return type_; }
    // The shock-face dissipation threshold (0: off).
    double shock_blend() const { return shock_blend_; }

private:
    State hllc(const State& UL, const State& UR, int axis, double& smax) const;
    State hllc_core(const State& WL, const State& WR, const State& UL, const State& UR, double inv_L, double inv_R, double c_L,
                    double c_R, int axis, double& smax) const;
    State lowmach_flux(const State& WL, const State& WR, double inv_L, double inv_R, int axis, double& smax) const;

    EOS eos_;
    RiemannType type_ = RiemannType::HLLC;
    double shock_blend_ = 0.0;
};

// ---------------------------------------------------------------------------
// TVD limiters  phi(a, b): a = upwind delta, b = downwind delta
// ---------------------------------------------------------------------------

// TVD slope limiters, from most diffusive (Minmod) to most compressive (MC).
enum class Limiter { Minmod = 0, VanLeer = 1, MC = 2, Koren = 3 };

// Limiter from its name (case-insensitive); unknown names throw std::invalid_argument.
Limiter parse_limiter(const std::string& s);

// Face reconstruction of the second-order path: the limited linear MUSCL profile, or a WENO reconstruction (3 or 5 point
// stencil per side).
enum class Reconstruction { Muscl, Weno3, Weno5 };
// Scheme from its parameter value ("muscl", "weno3", "weno5"; case-insensitive).
Reconstruction parse_reconstruction(const std::string& s);

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
// With nz = 1 these reduce to the 2-D layouts [d*nx + i] and [j*kHaloDepth + d]. `depth` is the number of slices of
// every halo supplied (kHaloDepth for MUSCL and WENO3-Z, 3 for WENO5-Z, which reads its ghost cells from them).
struct Halos {
    int depth = kHaloDepth;
    const Field* b = nullptr;    // below
    const Field* t = nullptr;    // above
    const Field* l = nullptr;    // left
    const Field* r = nullptr;    // right
    const Field* f = nullptr;    // front (z-min)
    const Field* bk = nullptr;   // back (z-max)
    // WENO5-Z only: cells (row-major, 1 = solid cell of an immersed body). A face with such a cell among its six takes the
    // four-cell WENO3-Z stencil: the wall's ghost cells are two deep, the third cell would be a cell at rest inside the body.
    const std::vector<char>* narrow = nullptr;
};
class MusclReconstruction {
public:
    // Reconstruction with limiter `lim`.
    // `weno_eps`: relative regularisation of the WENO smoothness indicators (WENO only).
    MusclReconstruction(const EOS& eos, Limiter lim, Reconstruction scheme = Reconstruction::Muscl, double weno_eps = 1e-6)
        : eos_(eos), lim_(lim), scheme_(scheme), weno_eps_(weno_eps) {}

    // Upgrade first-order face states (in place) to 2nd-order MUSCL states.
    // Physical-boundary faces keep their 1st-order ghost states; MPI-interface
    // faces are upgraded when the matching halo is supplied.
    void reconstruct(Field& UL, Field& UR, const Field& U, int nx, int ny, const Halos& h = {}) const;
    // 3-D version (z-sweep added); m gives nx, ny, nz.
    void reconstruct_3d(Field& UL, Field& UR, const Field& U, const Mesh& m, const Halos& h = {}) const;

    // The fluxes of every face of `m` in one pass, for any reconstruction: faces with a full stencil are reconstructed and
    // sent to the Riemann solver together (the vector kernels), without storing their left / right states.
    // The other faces (physical sides without a ghost layer) take the first-order states in UL / UR, which must hold the
    // faces of the sides (BoundaryManager::apply_boundary_conditions), through the conservative-primitive round trip.
    // F gets what reconstruct (or reconstruct_3d) followed by RiemannSolver::compute_flux gives, bit for bit; returns the
    // maximum wave speed. With WENO5-Z the halos are three cells deep and `h.narrow` marks the solid cells (four-cell stencil).
    double muscl_fluxes(const RiemannSolver& riemann, const Field& UL, const Field& UR, const Field& U, const Mesh& m,
                        const Halos& h, Field& F) const;

    // The same for a 2-D mesh (nz = 1), with the flux balance of the cells taken in the same pass: R gets what muscl_fluxes
    // followed by the divergence of the face fluxes, ((0 - (Fr dy - Fl dy)) - (Ft dx - Fb dx)) / area, gives, but the face
    // fluxes never go to memory (a row of them at a time). Returns the maximum wave speed.
    double muscl_residual_2d(const RiemannSolver& riemann, const Field& UL, const Field& UR, const Field& U, const Mesh& m,
                             const Halos& h, Field& R) const;

    // The same for a 3-D mesh (nz > 1): the balance ((0 - (Fr Ax - Fl Ax)) - (Ft Ay - Fb Ay)) - (Fk Az - Ff Az)) * (1 / volume) in the
    // same pass, a plane of cells at a time (the face fluxes of a plane are kept, not the whole face array).
    double muscl_residual_3d(const RiemannSolver& riemann, const Field& UL, const Field& UR, const Field& U, const Mesh& m,
                             const Halos& h, Field& R) const;

    // WENO3-Z / WENO5-Z of the CONSERVED cell averages (not the primitives) on the faces with a full stencil of a uniform mesh (2-D when nz = 1): the
    // face-averaged states of the fourth-order flux (flux_quadrature = fourth). The other faces keep what UL / UR hold. Halos as for the other
    // reconstructions (three cells deep with WENO5-Z). Scalar.
    void reconstruct_conserved(Field& UL, Field& UR, const Field& U, const Mesh& m, const Halos& h) const;

    // The WENO3-Z value at the face on the right of c1 from the cells (c0, c1, c2) (the right-hand state of the same face is
    // weno3_value(c3, c2, c1)): the very function the uniform reconstruction calls, so another engine (the adaptive face
    // list) gets the same bits. `weno_eps`: the relative regularisation of the scheme.
    double weno3_value(double c0, double c1, double c2) const;
    // The WENO5-Z value at the face on the right of c2 from the cells (c0 .. c4); the right-hand state of the same face is
    // weno5_value(c5, c4, c3, c2, c1) of the six cells c0 .. c5. The function the uniform reconstruction calls, as weno3_value.
    double weno5_value(double c0, double c1, double c2, double c3, double c4) const;
    // The conservative WENO3-Z (`five` false: the four primitive cells W[st[4k .. 4k+3]] of face k) or WENO5-Z (`five`: the six cells
    // W[st[6k .. 6k+5]]) states of n faces of the adaptive engine, in order along the normal: UL[k] = prim_to_cons of weno3_value(c0, c1, c2)
    // (weno5_value(c0 .. c4)) of every variable, UR[k] that of weno3_value(c3, c2, c1) (weno5_value(c5, c4, c3, c2, c1)). Vector kernels
    // where the CPU has them; the same bits as one scalar call per face.
    void weno_face_states(bool five, const State* W, const int* st, int n, State* UL, State* UR) const;
    // Active limiter.
    Limiter limiter() const { return lim_; }
    // Active face reconstruction.
    Reconstruction scheme() const { return scheme_; }

private:
    // WENO3-Z (`five` false: four cells per face, a primitive state padded by 2 cells) and WENO5-Z (six cells, padded by 3) on the
    // uniform mesh: a side with a halo (periodic wrap, physical ghost layers or an MPI neighbour) reads it and its boundary faces
    // are reconstructed, a side without one replicates the edge cell (first-order boundary face). WENO5-Z is serial only: an MPI
    // halo is two cells deep. Faces of the 2-D layout when nz = 1. Vector kernels (SchemesSimd.cpp) where the CPU has them.
    void reconstruct_weno(Field& UL, Field& UR, const Field& U, int nx, int ny, int nz, const Halos& h, bool five) const;

    EOS eos_;
    Limiter lim_;
    Reconstruction scheme_;
    double weno_eps_;
};

// The fluxes of the faces of a uniform mesh from the face-averaged left / right states UL / UR (conserved), with the fourth-order transverse
// correction: point values U - (1/24) sum_t d2_t U, the Riemann solver on them, F + (1/24) sum_t d2_t F (d2_t: second difference over the
// neighbouring faces of the same family along the transverse axis t; periodic[t] wraps). A face keeps the flux of the averaged states where its
// full correction cannot be made: a face that is not reconstructed (no full stencil), a transverse neighbour or neighbour of it that does not exist,
// the shock sensor of the Riemann solver firing on it or on them, or a non-positive density or pressure of a point value. Returns the maximum wave
// speed. `h` tells which sides have a halo (their boundary faces are reconstructed).
double fourth_order_fluxes(const RiemannSolver& riemann, const EOS& eos, const Field& UL, const Field& UR, const Mesh& m, const Halos& h,
                           const bool periodic[3], Field& F);

}   // namespace fv
