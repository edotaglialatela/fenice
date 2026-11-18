#include "fv/Switches.hpp"
#include "fv/Schemes.hpp"

#include "SchemesSimd.hpp"

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef FV_SIMD_BACKEND_AVX2
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace fv {

#ifdef FV_SIMD
namespace detail {

// Here, not in SchemesSimd.cpp: on x86-64 that file is compiled for AVX2, and the check must run on CPUs without it.
bool simd_available() {
    static const bool ok = []() {
        if (switch_on("FENICE_NO_SIMD")) return false;
#if !defined(FV_SIMD_BACKEND_AVX2)
        return true;   // NEON is part of every ARM64 processor
#elif defined(_MSC_VER)
        int r[4];
        __cpuid(r, 0);
        if (r[0] < 7) return false;
        __cpuid(r, 1);
        if (!((r[2] >> 27) & 1) || !((r[2] >> 28) & 1)) return false;   // OSXSAVE and AVX
        if ((_xgetbv(0) & 6) != 6) return false;                         // the OS saves the XMM and YMM state
        __cpuidex(r, 7, 0);
        return ((r[1] >> 5) & 1) != 0;
#else
        unsigned a, b, c, d;
        if (!__get_cpuid(0, &a, &b, &c, &d) || a < 7) return false;
        __get_cpuid(1, &a, &b, &c, &d);
        if (!((c >> 27) & 1) || !((c >> 28) & 1)) return false;
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        if ((lo & 6) != 6) return false;
        __cpuid_count(7, 0, a, b, c, d);
        return ((b >> 5) & 1) != 0;
#endif
    }();
    return ok;
}

}   // namespace detail
#endif

namespace {

// Guard against near-zero denominators in wave-speed / starred-state formulas.
constexpr double kDenomEps = 1e-14;
// Floor for the Mach-number speed (LowMachHLLC).
constexpr double kMachEps = 1e-12;

inline double safe_denom(double x) { return std::copysign(std::max(std::abs(x), kDenomEps), x); }

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

}   // namespace

RiemannType parse_riemann(const std::string& s_in) {
    const std::string s = lower(s_in);
    if (s == "hllc") return RiemannType::HLLC;
    if (s == "hllc (low-mach thornber)" || s == "lowmach" || s == "lowmachhllc" || s == "low-mach")
        return RiemannType::LowMachHLLC;
    if (s == "hllc (preconditioned)" || s == "preconditioned" || s == "preconditionedhllc")
        throw std::invalid_argument("The preconditioned Riemann solver was removed: use 'HLLC (Low-Mach Thornber)' (or 'HLLC')");
    throw std::invalid_argument("Unknown Riemann solver: '" + s_in + "'");
}

Limiter parse_limiter(const std::string& s_in) {
    const std::string s = lower(s_in);
    if (s == "minmod") return Limiter::Minmod;
    if (s == "van leer" || s == "vanleer") return Limiter::VanLeer;
    if (s == "mc") return Limiter::MC;
    if (s == "koren") return Limiter::Koren;
    throw std::invalid_argument("Unknown MUSCL limiter: '" + s_in + "' (use Minmod, Van Leer, MC or Koren)");
}

Reconstruction parse_reconstruction(const std::string& s_in) {
    const std::string s = lower(s_in);
    if (s == "muscl") return Reconstruction::Muscl;
    if (s == "weno3") return Reconstruction::Weno3;
    if (s == "weno5") return Reconstruction::Weno5;
    throw std::invalid_argument("Unknown reconstruction: '" + s_in + "' (use muscl, weno3 or weno5)");
}

// ---------------------------------------------------------------------------
// HLLC (Toro et al., 1994) with Einfeldt wave-speed bounds
// ---------------------------------------------------------------------------

State RiemannSolver::hllc(const State& UL, const State& UR, int axis, double& smax) const {
    double inv_L, inv_R;
    const State WL = eos_.cons_to_prim(UL, &inv_L);
    const State WR = eos_.cons_to_prim(UR, &inv_R);
    return hllc_core(WL, WR, UL, UR, inv_L, inv_R, eos_.sound_speed_inv(WL[3], inv_L), eos_.sound_speed_inv(WR[3], inv_R), axis, smax);
}

// HLLC from the primitive and conservative states of both sides (WL, WR floored as by cons_to_prim, inv_* = 1 / rho, c_* = sound speed).
State RiemannSolver::hllc_core(const State& WL, const State& WR, const State& UL, const State& UR, double inv_L, double inv_R,
                               double c_L, double c_R, int axis, double& smax) const {
    const double rho_L = WL[0], p_L = WL[3];
    const double rho_R = WR[0], p_R = WR[3];
    const double vel_L[3] = {WL[1], WL[2], WL[4]};
    const double vel_R[3] = {WR[1], WR[2], WR[4]};
    const double E_L = UL[3] * inv_L, E_R = UR[3] * inv_R;

    // Axis-aligned face: the normal velocity is the `axis` component, the other two are tangential
    const double Vn_L = vel_L[axis];
    const double Vn_R = vel_R[axis];
    const int mn = kMom[axis];

    State FL, FR;
    FL[0] = rho_L * Vn_L;
    FR[0] = rho_R * Vn_R;
    for (int a = 0; a < 3; ++a) {
        FL[kMom[a]] = rho_L * vel_L[a] * Vn_L;
        FR[kMom[a]] = rho_R * vel_R[a] * Vn_R;
    }
    FL[mn] = FL[mn] + p_L;
    FR[mn] = FR[mn] + p_R;
    FL[3] = Vn_L * (rho_L * E_L + p_L);
    FR[3] = Vn_R * (rho_R * E_R + p_R);

    // Einfeldt bounds
    const double S_L = std::min(Vn_L - c_L, Vn_R - c_R);
    const double S_R = std::max(Vn_L + c_L, Vn_R + c_R);

    // Contact wave speed (Toro 2009, eq. 10.37)
    const double num = rho_R * Vn_R * (S_R - Vn_R) - rho_L * Vn_L * (S_L - Vn_L) + p_L - p_R;
    const double den = rho_R * (S_R - Vn_R) - rho_L * (S_L - Vn_L);
    const double S_M = num / safe_denom(den);

    const double den_star_L = safe_denom(S_L - S_M);
    const double den_star_R = safe_denom(S_R - S_M);
    const double den_pL = safe_denom(S_L - Vn_L);
    const double den_pR = safe_denom(S_R - Vn_R);

    // Starred states: normal momentum f*S_M, tangential momenta f*u_t
    const double fL = rho_L * (S_L - Vn_L) / den_star_L;
    const double fR = rho_R * (S_R - Vn_R) / den_star_R;
    State UsL, UsR;
    UsL[0] = fL;
    UsR[0] = fR;
    for (int a = 0; a < 3; ++a) {
        UsL[kMom[a]] = fL * (a == axis ? S_M : vel_L[a]);
        UsR[kMom[a]] = fR * (a == axis ? S_M : vel_R[a]);
    }
    UsL[3] = fL * (E_L + (S_M - Vn_L) * (S_M + p_L / (rho_L * den_pL)));
    UsR[3] = fR * (E_R + (S_M - Vn_R) * (S_M + p_R / (rho_R * den_pR)));

    // Flux selection by wave ordering
    State F = {0, 0, 0, 0, 0};
    if (S_L >= 0) F = FL;
    if (S_L < 0 && S_M >= 0)
        for (int q = 0; q < kNVar; ++q) F[q] = FL[q] + S_L * (UsL[q] - UL[q]);
    if (S_M < 0 && S_R >= 0)
        for (int q = 0; q < kNVar; ++q) F[q] = FR[q] + S_R * (UsR[q] - UR[q]);
    if (S_R < 0) F = FR;

    // Shock-face dissipation (carbuncle / odd-even cure): on a face with a strong pressure jump the contact and shear
    // restoration of HLLC is blended out toward HLL, which has no middle wave. The sensor reads this face's two states only
    // (no neighbour faces), so the flux stays a function of (UL, UR): no new halo, same stencil in the AMR face list.
    if (shock_blend_ > 0.0) {
        const double s = std::abs(p_R - p_L) / std::min(p_L, p_R);
        if (s > shock_blend_) {
            const double phi = std::min((s - shock_blend_) / shock_blend_, 1.0);
            State Fh;
            if (S_L >= 0) {
                Fh = FL;
            } else if (S_R <= 0) {
                Fh = FR;
            } else {
                const double inv = 1.0 / (S_R - S_L);
                for (int q = 0; q < kNVar; ++q)
                    Fh[q] = (S_R * FL[q] - S_L * FR[q] + S_L * S_R * (UR[q] - UL[q])) * inv;
            }
            for (int q = 0; q < kNVar; ++q) F[q] = (1.0 - phi) * F[q] + phi * Fh[q];
        }
    }

    smax = std::max(std::abs(S_L), std::abs(S_R));
    return F;
}

State RiemannSolver::face_flux(const State& UL, const State& UR, int axis, double& smax) const {
    if (type_ != RiemannType::LowMachHLLC) return hllc(UL, UR, axis, smax);
    double inv_L, inv_R;
    const State WL = eos_.cons_to_prim(UL, &inv_L);
    const State WR = eos_.cons_to_prim(UR, &inv_R);
    return lowmach_flux(WL, WR, inv_L, inv_R, axis, smax);
}

// Thornber (2008) low-Mach velocity-jump scaling, then standard HLLC, from the primitive states of both sides (floored as by
// cons_to_prim; inv_* = 1 / rho). The scaling changes the velocities only: the density, the pressure and so 1 / rho and the sound
// speed are those of the entry states, and the conservative states are built once from the scaled ones.
State RiemannSolver::lowmach_flux(const State& W0L, const State& W0R, double inv_L, double inv_R, int axis, double& smax) const {
    State WL = W0L, WR = W0R;
    const double c_L = eos_.sound_speed_inv(WL[3], inv_L);
    const double c_R = eos_.sound_speed_inv(WR[3], inv_R);
    const double M_L = std::sqrt(WL[1] * WL[1] + WL[2] * WL[2] + WL[4] * WL[4]) / (c_L + kMachEps);
    const double M_R = std::sqrt(WR[1] * WR[1] + WR[2] * WR[2] + WR[4] * WR[4]) / (c_R + kMachEps);
    const double z = std::min(std::max(std::max(M_L, M_R), 1e-4), 1.0);

    for (int a = 0; a < 3; ++a) {
        const int m = kMom[a];
        const double avg = 0.5 * (WL[m] + WR[m]);
        const double half = 0.5 * (WL[m] - WR[m]);
        WL[m] = avg + z * half;
        WR[m] = avg - z * half;
    }
    return hllc_core(WL, WR, eos_.prim_to_cons(WL), eos_.prim_to_cons(WR), inv_L, inv_R, c_L, c_R, axis, smax);
}

void RiemannSolver::face_flux_batch(const State* UL, const State* UR, const int* axis, int n, State* F, double* smax) const {
    int done = 0;
#ifdef FV_SIMD
    if (detail::simd_available()) {
        done = n & ~3;
        if (done > 0) detail::riemann_batch_simd(eos_, type_, shock_blend_, UL, UR, axis, done, F, smax);
    }
#endif
    for (int k = done; k < n; ++k) F[k] = face_flux(UL[k], UR[k], axis[k], smax[k]);
}

void RiemannSolver::muscl_face_flux_batch(Limiter lim, const State* W, const int* st, const int* axis, int n, State* F,
                                          double* smax, bool planar) const {
    int done = 0;
#ifdef FV_SIMD
    if (detail::simd_available()) {
        done = n & ~3;
        if (done > 0) detail::muscl_riemann_batch_simd(eos_, type_, shock_blend_, lim, W, st, axis, done, F, smax, planar);
    }
#endif
    for (int k = done; k < n; ++k) {
        const State &w0 = W[st[4 * k]], &w1 = W[st[4 * k + 1]], &w2 = W[st[4 * k + 2]], &w3 = W[st[4 * k + 3]];
        State wl, wr;
        for (int q = 0; q < kNVar; ++q) {
            if (planar && q == 4) {   // rho w = 0 in 2-D: the limited reconstruction of +0 is +0
                wl[q] = wr[q] = 0.0;
                continue;
            }
            const double c0 = w0[q], c1 = w1[q], c2 = w2[q], c3 = w3[q];
            const double bq = c2 - c1;
            wl[q] = c1 + 0.5 * apply_limiter(lim, c1 - c0, bq);
            wr[q] = c2 - 0.5 * apply_limiter(lim, c3 - c2, bq);
        }
        F[k] = face_flux(eos_.prim_to_cons(wl), eos_.prim_to_cons(wr), axis[k], smax[k]);
    }
}

void RiemannSolver::muscl_line_flux_batch(Limiter lim, const double* W, size_t plane, size_t base, size_t stride, int axis, int n,
                                          State* F, double* smax, bool planar) const {
    int done = 0;
#ifdef FV_SIMD
    if (detail::simd_available()) {
        done = n & ~3;
        if (done > 0) detail::muscl_riemann_line_simd(eos_, type_, shock_blend_, lim, W, plane, base, stride, axis, done, F, smax, planar);
    }
#endif
    for (int k = done; k < n; ++k) {
        State wl, wr;
        for (int q = 0; q < kNVar; ++q) {
            if (planar && q == 4) {   // rho w = 0 in 2-D: the limited reconstruction of +0 is +0
                wl[q] = wr[q] = 0.0;
                continue;
            }
            const double* p = W + q * plane + base + k;
            const double c0 = p[0], c1 = p[stride], c2 = p[2 * stride], c3 = p[3 * stride];
            const double bq = c2 - c1;
            wl[q] = c1 + 0.5 * apply_limiter(lim, c1 - c0, bq);
            wr[q] = c2 - 0.5 * apply_limiter(lim, c3 - c2, bq);
        }
        F[k] = face_flux(eos_.prim_to_cons(wl), eos_.prim_to_cons(wr), axis, smax[k]);
    }
}

double RiemannSolver::compute_flux(const Field& UL, const Field& UR, const Mesh& mesh, Field& F) const {
    const int nf = mesh.num_faces();
    const int nxf = mesh.num_xfaces(), nxyf = nxf + mesh.num_yfaces();
    F.resize(nf);
    double smax_all = 0.0;
    constexpr int kChunk = 512;
    const int nchunk = (nf + kChunk - 1) / kChunk;
#pragma omp parallel if (nf >= kOmpMinWork)
    {
        double local = 0.0;
        int ax[kChunk];
        double sm[kChunk];
#pragma omp for schedule(static) nowait
        for (int c = 0; c < nchunk; ++c) {
            const int f0 = c * kChunk, n = std::min(kChunk, nf - f0);
            for (int k = 0; k < n; ++k) ax[k] = f0 + k < nxf ? 0 : (f0 + k < nxyf ? 1 : 2);
            face_flux_batch(&UL[f0], &UR[f0], ax, n, &F[f0], sm);
            for (int k = 0; k < n; ++k)
                if (sm[k] > local) local = sm[k];
        }
#pragma omp critical(fv_riemann_max)
        if (local > smax_all) smax_all = local;
    }
    return smax_all;
}
// ---------------------------------------------------------------------------
// MUSCL
// ---------------------------------------------------------------------------

namespace {
// WENO-Z with two 2-point stencils (Borges et al. 2008): the value at the face on the right of c1, from the cells
// (c0, c1, c2). The right-hand state of the same face is weno3z(c3, c2, c1). The regularisation of the smoothness
// indicators is relative to the size of the data of the stencil, eps = eps_rel * mean(c^2) (+ a floor): a fixed
// absolute eps lowers the order at smooth extrema to about 2.5 whatever the units of the variable.
inline double weno3z(double c0, double c1, double c2, double eps_rel) {
    const double eps = eps_rel * (c0 * c0 + c1 * c1 + c2 * c2) * (1.0 / 3.0) + 1e-300;
    const double v0 = -0.5 * c0 + 1.5 * c1, v1 = 0.5 * c1 + 0.5 * c2;
    const double b0 = (c1 - c0) * (c1 - c0), b1 = (c2 - c1) * (c2 - c1);
    const double tau = std::abs(b0 - b1);
    const double a0 = (1.0 / 3.0) * (1.0 + tau / (b0 + eps)), a1 = (2.0 / 3.0) * (1.0 + tau / (b1 + eps));
    return (a0 * v0 + a1 * v1) / (a0 + a1);
}
// WENO5-Z (Borges et al. 2008): the value at the face on the right of c2, from the cells (c0 .. c4) = (i-2 .. i+2); the
// right-hand state of the same face is weno5z(c5, c4, c3, c2, c1) of the six cells c0 .. c5. Same relative regularisation
// as weno3z.
inline double weno5z(double c0, double c1, double c2, double c3, double c4, double eps_rel) {
    const double eps = eps_rel * (c0 * c0 + c1 * c1 + c2 * c2 + c3 * c3 + c4 * c4) * 0.2 + 1e-300;
    const double p0 = (2.0 * c0 - 7.0 * c1 + 11.0 * c2) / 6.0;
    const double p1 = (-c1 + 5.0 * c2 + 2.0 * c3) / 6.0;
    const double p2 = (2.0 * c2 + 5.0 * c3 - c4) / 6.0;
    const double s0 = c0 - 2.0 * c1 + c2, t0 = c0 - 4.0 * c1 + 3.0 * c2;
    const double s1 = c1 - 2.0 * c2 + c3, t1 = c1 - c3;
    const double s2 = c2 - 2.0 * c3 + c4, t2 = 3.0 * c2 - 4.0 * c3 + c4;
    const double b0 = (13.0 / 12.0) * s0 * s0 + 0.25 * t0 * t0;
    const double b1 = (13.0 / 12.0) * s1 * s1 + 0.25 * t1 * t1;
    const double b2 = (13.0 / 12.0) * s2 * s2 + 0.25 * t2 * t2;
    const double tau = std::abs(b0 - b2);
    const double a0 = 0.1 * (1.0 + tau / (b0 + eps)), a1 = 0.6 * (1.0 + tau / (b1 + eps)), a2 = 0.3 * (1.0 + tau / (b2 + eps));
    return (a0 * p0 + a1 * p1 + a2 * p2) / (a0 + a1 + a2);
}
}   // namespace

double MusclReconstruction::weno3_value(double c0, double c1, double c2) const { return weno3z(c0, c1, c2, weno_eps_); }

double MusclReconstruction::weno5_value(double c0, double c1, double c2, double c3, double c4) const {
    return weno5z(c0, c1, c2, c3, c4, weno_eps_);
}

void MusclReconstruction::weno_face_states(bool five, const State* W, const int* st, int n, State* UL, State* UR) const {
    const int ns = five ? 6 : 4;
    int done = 0;
#ifdef FV_SIMD
    if (detail::simd_available()) {
        done = n & ~3;
        if (done > 0) detail::weno_gather_simd(eos_, weno_eps_, five, W, st, done, UL, UR);
    }
#endif
    for (int k = done; k < n; ++k) {
        const int* s = st + static_cast<size_t>(ns) * k;
        State wl, wr;
        for (int q = 0; q < kNVar; ++q) {
            if (five) {
                wl[q] = weno5z(W[s[0]][q], W[s[1]][q], W[s[2]][q], W[s[3]][q], W[s[4]][q], weno_eps_);
                wr[q] = weno5z(W[s[5]][q], W[s[4]][q], W[s[3]][q], W[s[2]][q], W[s[1]][q], weno_eps_);
            } else {
                wl[q] = weno3z(W[s[0]][q], W[s[1]][q], W[s[2]][q], weno_eps_);
                wr[q] = weno3z(W[s[3]][q], W[s[2]][q], W[s[1]][q], weno_eps_);
            }
        }
        UL[k] = eos_.prim_to_cons(wl);
        UR[k] = eos_.prim_to_cons(wr);
    }
}

void MusclReconstruction::reconstruct_weno(Field& UL, Field& UR, const Field& U, int nx, int ny, int nz, const Halos& h,
                                           bool five) const {
    const int pad = five ? 3 : 2;   // cells read on each side of the two of a face (the stencil is 2 pad cells wide)
    const int padz = nz > 1 ? pad : 0;
    const int PX = nx + 2 * pad, PY = ny + 2 * pad, PZ = nz + 2 * padz;
    Mesh m;
    m.nx = nx;
    m.ny = ny;
    m.nz = nz;
    // a side with a halo (periodic wrap or ghost layers of a physical boundary) reads it, the others replicate the edge cell
    if ((h.l || h.r || h.b || h.t || h.f || h.bk) && h.depth < pad)
        throw std::logic_error(five ? "WENO5-Z needs halos three cells deep" : "WENO3-Z needs halos two cells deep");
    const int dh = h.depth;
    const auto clampi = [](int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); };
    // conservative state of cell (i, j, k), one axis possibly outside the domain
    const auto cell_state = [&](int i, int j, int k) -> const State& {
        if (i < 0 && h.l) return (*h.l)[(static_cast<size_t>(k) * ny + j) * dh + dh + i];
        if (i >= nx && h.r) return (*h.r)[(static_cast<size_t>(k) * ny + j) * dh + i - nx];
        if (j < 0 && h.b) return (*h.b)[(static_cast<size_t>(dh + j) * nz + k) * nx + i];
        if (j >= ny && h.t) return (*h.t)[(static_cast<size_t>(j - ny) * nz + k) * nx + i];
        if (k < 0 && h.f) return (*h.f)[(static_cast<size_t>(dh + k) * ny + j) * nx + i];
        if (k >= nz && h.bk) return (*h.bk)[(static_cast<size_t>(k - nz) * ny + j) * nx + i];
        return U[m.cell(clampi(i, nx), clampi(j, ny), clampi(k, nz))];
    };
    const size_t padded_cells = static_cast<size_t>(PZ) * PY * PX;
    const auto P = [&](int ii, int jj, int kk) -> size_t { return (static_cast<size_t>(kk) * PY + jj) * PX + ii; };
    // the primitive state of the padded cells, one plane per variable (the vector kernel reads four neighbouring faces along a line)
#ifdef FV_SIMD
    const bool vec = detail::simd_available();
#else
    const bool vec = false;
#endif
    thread_local std::vector<double> planes_scratch[kNVar];
    std::vector<double>(&planes_)[kNVar] = planes_scratch;   // the master's buffers: the worker threads write through this name
    for (int q = 0; q < kNVar; ++q) planes_[q].resize(padded_cells);
    double* const wplane[kNVar] = {planes_[0].data(), planes_[1].data(), planes_[2].data(), planes_[3].data(), planes_[4].data()};
#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
    for (int kk = 0; kk < PZ; ++kk) {
        const int k = kk - padz;
        for (int jj = 0; jj < PY; ++jj) {
            const int j = jj - pad;
            int iv = 0;   // the cells of the row of the domain that the vector kernel converts
#ifdef FV_SIMD
            if (vec && j >= 0 && j < ny && k >= 0 && k < nz) {
                iv = nx & ~3;
                double* o[kNVar];
                for (int q = 0; q < kNVar; ++q) o[q] = wplane[q] + P(pad, jj, kk);
                detail::cons_to_prim_planes_simd(eos_, &U[m.cell(0, j, k)], iv, o);
            }
#endif
            for (int ii = 0; ii < PX; ++ii) {
                const int i = ii - pad;
                if (i >= 0 && i < iv) continue;
                // a cell outside the domain along two axes (a corner) is not on any stencil: edge replication
                const int out = (i < 0 || i >= nx) + (j < 0 || j >= ny) + (k < 0 || k >= nz);
                const State w = eos_.cons_to_prim(out > 1 ? U[m.cell(clampi(i, nx), clampi(j, ny), clampi(k, nz))] : cell_state(i, j, k));
                for (int q = 0; q < kNVar; ++q) wplane[q][P(ii, jj, kk)] = w[q];
            }
        }
    }
    // immersed body: the solid cells on the same padded layout (a cell outside the domain is solid if the cell it replicates or the
    // cell a periodic wrap puts there is)
    thread_local std::vector<char> S_pad;
    const std::vector<char>* solid = five ? h.narrow : nullptr;   // WENO3-Z has no narrower stencil to fall back on
    if (solid) {
        if (solid->size() != static_cast<size_t>(nx) * ny * nz) throw std::logic_error("WENO5-Z: the solid mask is not that of the mesh");
        S_pad.resize(padded_cells);
        const auto wrapi = [](int i, int n) { return ((i % n) + n) % n; };
        for (int kk = 0; kk < PZ; ++kk)
            for (int jj = 0; jj < PY; ++jj)
                for (int ii = 0; ii < PX; ++ii) {
                    const int i = ii - pad, j = jj - pad, k = kk - padz;
                    S_pad[P(ii, jj, kk)] = ((*solid)[m.cell(clampi(i, nx), clampi(j, ny), clampi(k, nz))] != 0 ||
                                            (*solid)[m.cell(wrapi(i, nx), wrapi(j, ny), wrapi(k, nz))] != 0)
                                               ? 1 : 0;
                }
    }
    const std::vector<char>& Sp = S_pad;
    const EOS& eos = eos_;
    const double we = weno_eps_;
    const auto roundtrip = [&](State& u) { u = eos.prim_to_cons(eos.cons_to_prim(u)); };
    // Faces normal to `axis`: face t lies between cells t-1 and t; its six cells are padded positions t .. t+5.
    const auto sweep = [&](int axis, bool has_lo, bool has_hi) {
        const int n_a = axis == 0 ? nx : (axis == 1 ? ny : nz);
        const size_t stride = axis == 0 ? 1 : (axis == 1 ? static_cast<size_t>(PX) : static_cast<size_t>(PX) * PY);
        const int ni = axis == 0 ? nx + 1 : nx, nj = axis == 1 ? ny + 1 : ny, nk = axis == 2 ? nz + 1 : nz;
        const double* const pl[kNVar] = {wplane[0], wplane[1], wplane[2], wplane[3], wplane[4]};
#pragma omp parallel if (nx * ny * nz >= kOmpMinWork)
        {
#pragma omp for schedule(static)
        for (int kj = 0; kj < nk * nj; ++kj) {
            const int j = kj % nj, k = kj / nj;
            // faces vs .. vs+nv-1 of the line (a multiple of four, all with a full stencil) take the vector kernel, which writes their
            // conservative states; a face with a solid cell among its six is overwritten below, the others are left as they are
            int vs = 0, nv = 0;
#ifdef FV_SIMD
            if (vec) {
                if (axis == 0) {
                    vs = has_lo ? 0 : 1;
                    nv = ((has_hi ? nx : nx - 1) - vs + 1) & ~3;
                } else {
                    const int t = axis == 1 ? j : k;
                    nv = ((t > 0 && t < n_a) || (t == 0 && has_lo) || (t == n_a && has_hi)) ? (ni & ~3) : 0;
                }
                if (nv > 0) {
                    // the six cells of face i of the line are at b0 + i + s * stride, whatever the axis
                    const size_t b0 = axis == 0 ? P(vs, j + pad, k + padz) : (axis == 1 ? P(pad, j, k + padz) : P(pad, j + pad, k));
                    const int f0 = axis == 0 ? m.xface(vs, j, k) : (axis == 1 ? m.yface(0, j, k) : m.zface(0, j, k));
                    if (five)
                        detail::weno5_line_simd(eos, we, pl, b0, stride, nv, &UL[f0], &UR[f0]);
                    else
                        detail::weno3_line_simd(eos, we, pl, b0, stride, nv, &UL[f0], &UR[f0]);
                }
            }
#endif
            for (int i = 0; i < ni; ++i) {
                const int t = axis == 0 ? i : (axis == 1 ? j : k);
                const int f = axis == 0 ? m.xface(i, j, k) : (axis == 1 ? m.yface(i, j, k) : m.zface(i, j, k));
                if (!((t > 0 && t < n_a) || (t == 0 && has_lo) || (t == n_a && has_hi))) {
                    roundtrip(UL[f]);
                    roundtrip(UR[f]);
                    continue;
                }
                const size_t base = axis == 0 ? P(t, j + pad, k + padz)
                                              : (axis == 1 ? P(i + pad, t, k + padz) : P(i + pad, j + pad, t));
                State wl, wr;
                bool four = false;   // a solid cell among the six: the four-cell stencil (as the adaptive engine does)
                if (solid)
                    for (int s = 0; s < 6; ++s) four = four || Sp[base + s * stride] != 0;
                if (!four && i >= vs && i < vs + nv) continue;
                for (int q = 0; q < kNVar; ++q) {
                    const double* c = pl[q] + base;
                    if (!five) {
                        wl[q] = weno3z(c[0], c[stride], c[2 * stride], we);
                        wr[q] = weno3z(c[3 * stride], c[2 * stride], c[stride], we);
                        continue;
                    }
                    const double c0 = c[0], c1 = c[stride], c2 = c[2 * stride], c3 = c[3 * stride], c4 = c[4 * stride], c5 = c[5 * stride];
                    if (four) {
                        wl[q] = weno3z(c1, c2, c3, we);
                        wr[q] = weno3z(c4, c3, c2, we);
                        continue;
                    }
                    wl[q] = weno5z(c0, c1, c2, c3, c4, we);
                    wr[q] = weno5z(c5, c4, c3, c2, c1, we);
                }
                UL[f] = eos.prim_to_cons(wl);
                UR[f] = eos.prim_to_cons(wr);
            }
        }
        }
    };
    sweep(0, h.l, h.r);
    sweep(1, h.b, h.t);
    if (nz > 1) sweep(2, h.f, h.bk);
}

void MusclReconstruction::reconstruct(Field& UL, Field& UR, const Field& U, int nx, int ny, const Halos& h) const {
    if (scheme_ != Reconstruction::Muscl) return reconstruct_weno(UL, UR, U, nx, ny, 1, h, scheme_ == Reconstruction::Weno5);
    constexpr int pad = kHaloDepth;
    const int PW = nx + 2 * pad;   // padded row width
    thread_local Field W_pad;      // scratch buffer, per thread
    W_pad.resize(static_cast<size_t>(ny + 2 * pad) * PW);
    Field& Wp = W_pad;

    // Primitive state padded by edge replication
#pragma omp parallel for schedule(static) if (nx * ny >= kOmpMinWork)
    for (int jj = 0; jj < ny + 2 * pad; ++jj) {
        const int j = std::min(std::max(jj - pad, 0), ny - 1);
        for (int ii = 0; ii < PW; ++ii) {
            const int i = std::min(std::max(ii - pad, 0), nx - 1);
            Wp[jj * PW + ii] = eos_.cons_to_prim(U[j * nx + i]);
        }
    }
    // MPI halos replace the edge padding
    if (h.b)
        for (int d = 0; d < pad; ++d)
            for (int i = 0; i < nx; ++i) Wp[d * PW + pad + i] = eos_.cons_to_prim((*h.b)[d * nx + i]);
    if (h.t)
        for (int d = 0; d < pad; ++d)
            for (int i = 0; i < nx; ++i) Wp[(ny + pad + d) * PW + pad + i] = eos_.cons_to_prim((*h.t)[d * nx + i]);
    if (h.l)
        for (int j = 0; j < ny; ++j)
            for (int d = 0; d < pad; ++d) Wp[(j + pad) * PW + d] = eos_.cons_to_prim((*h.l)[j * pad + d]);
    if (h.r)
        for (int j = 0; j < ny; ++j)
            for (int d = 0; d < pad; ++d) Wp[(j + pad) * PW + nx + pad + d] = eos_.cons_to_prim((*h.r)[j * pad + d]);

    const Limiter lim = lim_;
    const EOS& eos = eos_;
    auto roundtrip = [&](State& u) { u = eos.prim_to_cons(eos.cons_to_prim(u)); };

    // X-direction: face i of row j sits between cells i-1 and i (padded cols i+1, i+2)
#pragma omp parallel for schedule(static) if (nx * ny >= kOmpMinWork)
    for (int j = 0; j < ny; ++j) {
        const State* row = &Wp[(j + pad) * PW];
        for (int i = 0; i <= nx; ++i) {
            const int f = j * (nx + 1) + i;
            const bool upgrade = (i > 0 && i < nx) || (i == 0 && h.l) || (i == nx && h.r);
            if (!upgrade) {
                roundtrip(UL[f]);
                roundtrip(UR[f]);
                continue;
            }
            State wl, wr;
            for (int q = 0; q < kNVar; ++q) {
                const double c0 = row[i][q], c1 = row[i + 1][q], c2 = row[i + 2][q], c3 = row[i + 3][q];
                const double b = c2 - c1;
                wl[q] = c1 + 0.5 * apply_limiter(lim, c1 - c0, b);
                wr[q] = c2 - 0.5 * apply_limiter(lim, c3 - c2, b);
            }
            UL[f] = eos.prim_to_cons(wl);
            UR[f] = eos.prim_to_cons(wr);
        }
    }

    // Y-direction: face j of column i sits between cells j-1 and j (padded rows j+1, j+2)
    const int nxf = (nx + 1) * ny;
#pragma omp parallel for schedule(static) if (nx * ny >= kOmpMinWork)
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int f = nxf + j * nx + i;
            const bool upgrade = (j > 0 && j < ny) || (j == 0 && h.b) || (j == ny && h.t);
            if (!upgrade) {
                roundtrip(UL[f]);
                roundtrip(UR[f]);
                continue;
            }
            const int col = i + pad;
            State wl, wr;
            for (int q = 0; q < kNVar; ++q) {
                const double r0 = Wp[j * PW + col][q], r1 = Wp[(j + 1) * PW + col][q];
                const double r2 = Wp[(j + 2) * PW + col][q], r3 = Wp[(j + 3) * PW + col][q];
                const double b = r2 - r1;
                wl[q] = r1 + 0.5 * apply_limiter(lim, r1 - r0, b);
                wr[q] = r2 - 0.5 * apply_limiter(lim, r3 - r2, b);
            }
            UL[f] = eos.prim_to_cons(wl);
            UR[f] = eos.prim_to_cons(wr);
        }
    }
}

// 3-D MUSCL: the same limited reconstruction along x, y and z on a primitive
// state padded by 2 cells in every direction.  Halos (MPI or periodic) replace
// the padding on their side; layouts as documented in Halos.
void MusclReconstruction::reconstruct_3d(Field& UL, Field& UR, const Field& U, const Mesh& m, const Halos& h) const {
    if (scheme_ != Reconstruction::Muscl) return reconstruct_weno(UL, UR, U, m.nx, m.ny, m.nz, h, scheme_ == Reconstruction::Weno5);
    constexpr int pad = kHaloDepth;
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const int PX = nx + 2 * pad, PY = ny + 2 * pad, PZ = nz + 2 * pad;
    thread_local Field W_pad;
    W_pad.resize(static_cast<size_t>(PZ) * PY * PX);
    Field& Wp = W_pad;
    auto P = [&](int ii, int jj, int kk) -> size_t { return (static_cast<size_t>(kk) * PY + jj) * PX + ii; };

#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
    for (int kk = 0; kk < PZ; ++kk) {
        const int k = std::min(std::max(kk - pad, 0), nz - 1);
        for (int jj = 0; jj < PY; ++jj) {
            const int j = std::min(std::max(jj - pad, 0), ny - 1);
            for (int ii = 0; ii < PX; ++ii)
                Wp[P(ii, jj, kk)] = eos_.cons_to_prim(U[m.cell(std::min(std::max(ii - pad, 0), nx - 1), j, k)]);
        }
    }
    for (int k = 0; k < nz; ++k)
        for (int d = 0; d < pad; ++d) {
            if (h.b)
                for (int i = 0; i < nx; ++i)
                    Wp[P(i + pad, d, k + pad)] = eos_.cons_to_prim((*h.b)[(d * nz + k) * nx + i]);
            if (h.t)
                for (int i = 0; i < nx; ++i)
                    Wp[P(i + pad, ny + pad + d, k + pad)] = eos_.cons_to_prim((*h.t)[(d * nz + k) * nx + i]);
            if (h.l)
                for (int j = 0; j < ny; ++j)
                    Wp[P(d, j + pad, k + pad)] = eos_.cons_to_prim((*h.l)[(k * ny + j) * pad + d]);
            if (h.r)
                for (int j = 0; j < ny; ++j)
                    Wp[P(nx + pad + d, j + pad, k + pad)] = eos_.cons_to_prim((*h.r)[(k * ny + j) * pad + d]);
        }
    for (int d = 0; d < pad; ++d)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                if (h.f) Wp[P(i + pad, j + pad, d)] = eos_.cons_to_prim((*h.f)[(d * ny + j) * nx + i]);
                if (h.bk) Wp[P(i + pad, j + pad, nz + pad + d)] = eos_.cons_to_prim((*h.bk)[(d * ny + j) * nx + i]);
            }

    const Limiter lim = lim_;
    const EOS& eos = eos_;
    auto roundtrip = [&](State& u) { u = eos.prim_to_cons(eos.cons_to_prim(u)); };
    // Faces normal to `axis`: face t along the axis lies between cells t-1 and t,
    // i.e. padded positions t+1 and t+2 (stencil t .. t+3).
    auto sweep = [&](int axis, const Field* lo, const Field* hi) {
        const int n_a = axis == 0 ? nx : (axis == 1 ? ny : nz);
        const size_t stride = axis == 0 ? 1 : (axis == 1 ? static_cast<size_t>(PX) : static_cast<size_t>(PX) * PY);
        const int ni = axis == 0 ? nx + 1 : nx, nj = axis == 1 ? ny + 1 : ny, nk = axis == 2 ? nz + 1 : nz;
#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
        for (int kj = 0; kj < nk * nj; ++kj) {
            const int j = kj % nj, k = kj / nj;
            for (int i = 0; i < ni; ++i) {
                const int t = axis == 0 ? i : (axis == 1 ? j : k);
                const int f = axis == 0 ? m.xface(i, j, k) : (axis == 1 ? m.yface(i, j, k) : m.zface(i, j, k));
                const bool upgrade = (t > 0 && t < n_a) || (t == 0 && lo) || (t == n_a && hi);
                if (!upgrade) {
                    roundtrip(UL[f]);
                    roundtrip(UR[f]);
                    continue;
                }
                const size_t base =
                    axis == 0 ? P(t, j + pad, k + pad) : (axis == 1 ? P(i + pad, t, k + pad) : P(i + pad, j + pad, t));
                State wl, wr;
                for (int q = 0; q < kNVar; ++q) {
                    const double c0 = Wp[base][q], c1 = Wp[base + stride][q];
                    const double c2 = Wp[base + 2 * stride][q], c3 = Wp[base + 3 * stride][q];
                    const double b = c2 - c1;
                    wl[q] = c1 + 0.5 * apply_limiter(lim, c1 - c0, b);
                    wr[q] = c2 - 0.5 * apply_limiter(lim, c3 - c2, b);
                }
                UL[f] = eos.prim_to_cons(wl);
                UR[f] = eos.prim_to_cons(wr);
            }
        }
    };
    sweep(0, h.l, h.r);
    sweep(1, h.b, h.t);
    sweep(2, h.f, h.bk);
}

// ---------------------------------------------------------------------------
// Fused reconstruction + Riemann solver on the uniform mesh (2-D when nz = 1, else 3-D)
// ---------------------------------------------------------------------------

namespace {

// What the line routine reads: the padded primitive state, the first-order states of the sides, the halos present.
struct FusedGrid {
    const RiemannSolver* riemann;
    const EOS* eos;
    Limiter lim;
    const double* Wp;      // padded primitive state, one plane per variable (MUSCL); null for the first-order scheme
    size_t plane;          // size of a plane of Wp
    const Field* U;        // cell states (first order)
    const Field* UL;
    const Field* UR;
    const Mesh* m;
    int nx, ny, nz, PX, PY, pad, zp;
    bool halo_lo[3], halo_hi[3];   // a halo on the low / high side of the axis
    int weno;                // 0 MUSCL (or first order), 3 WENO3-Z, 5 WENO5-Z (the states come from weno_line_states)
    double weno_eps;         // relative regularisation of the WENO smoothness indicators
    const char* solid;       // WENO5-Z with an immersed body: the solid cells on the padded layout (null: none)
    size_t P(int ii, int jj, int kk) const { return (static_cast<size_t>(kk) * PY + jj) * PX + ii; }
};

constexpr int kFusedChunk = 256;
struct FusedScratch {
    int ax[kFusedChunk];
    double sm[kFusedChunk];
    State ul[kFusedChunk], ur[kFusedChunk];   // first order: the states on both sides of the faces of a chunk
};

// The conservative WENO3-Z / WENO5-Z states of cnt consecutive faces of a line (the cells of face k at base + k + s * stride of the padded
// planes): the vector kernels for runs of four, the scalar functions for the rest and for the faces with a solid cell among their six (the
// four-cell stencil), exactly what MusclReconstruction::reconstruct_weno does.
void weno_line_states(const FusedGrid& g, size_t base, size_t stride, int cnt, State* UL, State* UR) {
    const bool five = g.weno == 5;
    int done = 0;
#ifdef FV_SIMD
    if (detail::simd_available()) {
        const double* pl[kNVar];
        for (int q = 0; q < kNVar; ++q) pl[q] = g.Wp + q * g.plane;
        done = cnt & ~3;
        if (done > 0) {
            if (five)
                detail::weno5_line_simd(*g.eos, g.weno_eps, pl, base, stride, done, UL, UR);
            else
                detail::weno3_line_simd(*g.eos, g.weno_eps, pl, base, stride, done, UL, UR);
        }
    }
#endif
    for (int k = 0; k < cnt; ++k) {
        const size_t b = base + static_cast<size_t>(k);
        bool four = false;
        if (g.solid)
            for (int s = 0; s < 6; ++s) four = four || g.solid[b + s * stride] != 0;
        if (!four && k < done) continue;
        State wl, wr;
        for (int q = 0; q < kNVar; ++q) {
            const double* c = g.Wp + q * g.plane + b;
            if (!five) {
                wl[q] = weno3z(c[0], c[stride], c[2 * stride], g.weno_eps);
                wr[q] = weno3z(c[3 * stride], c[2 * stride], c[stride], g.weno_eps);
                continue;
            }
            const double c0 = c[0], c1 = c[stride], c2 = c[2 * stride], c3 = c[3 * stride], c4 = c[4 * stride], c5 = c[5 * stride];
            if (four) {
                wl[q] = weno3z(c1, c2, c3, g.weno_eps);
                wr[q] = weno3z(c4, c3, c2, g.weno_eps);
                continue;
            }
            wl[q] = weno5z(c0, c1, c2, c3, c4, g.weno_eps);
            wr[q] = weno5z(c5, c4, c3, c2, c1, g.weno_eps);
        }
        UL[k] = g.eos->prim_to_cons(wl);
        UR[k] = g.eos->prim_to_cons(wr);
    }
}

// The fluxes of the faces of one line normal to `axis`, written to F_line[i]; `local` receives the largest wave speed.
// A line is: for axis 0 the x faces i = 0..nx of row (j, k); for axis 1 the y faces i = 0..nx-1 at face index j along y of plane
// k; for axis 2 the z faces i = 0..nx-1 of row j at face index k along z. Faces with a full four-cell stencil are reconstructed
// and solved together; the faces of a side without a halo take the first-order states through the conservative-primitive round
// trip. Face t along the axis lies between the cells t-1 and t, padded positions t+1 and t+2 (stencil t .. t+3).
void fused_line(const FusedGrid& g, int axis, int j, int k, State* F_line, double& local, FusedScratch& s) {
    const int nx = g.nx, ny = g.ny, nz = g.nz, pad = g.pad, zp = g.zp;
    const int n_a = axis == 0 ? nx : (axis == 1 ? ny : nz);
    const size_t stride = axis == 0 ? 1 : (axis == 1 ? static_cast<size_t>(g.PX) : static_cast<size_t>(g.PX) * g.PY);
    const int ni = axis == 0 ? nx + 1 : nx;
    const int t_line = axis == 0 ? -1 : (axis == 1 ? j : k);
    int first = 0, last = ni - 1;   // faces of the line that have a full stencil
    if (g.Wp == nullptr) {
        // first order: every face between two cells; the faces of the sides come from UL / UR
        if (axis == 0) {
            first = 1;
            last = nx - 1;
        } else if (!(t_line > 0 && t_line < n_a)) {
            first = 1;
            last = 0;
        }
    } else if (axis == 0) {
        first = g.halo_lo[0] ? 0 : 1;
        last = g.halo_hi[0] ? nx : nx - 1;
    } else if (!((t_line > 0 && t_line < n_a) || (t_line == 0 && g.halo_lo[axis]) || (t_line == n_a && g.halo_hi[axis]))) {
        first = 1;
        last = 0;   // the whole line is a side
    }
    for (int i0 = first; g.Wp == nullptr && i0 <= last; i0 += kFusedChunk) {
        const int cnt = std::min(kFusedChunk, last - i0 + 1);
        const Field& Uc = *g.U;
        for (int c = 0; c < cnt; ++c) {
            const int i = i0 + c;
            const int cr = axis == 0 ? (k * ny + j) * nx + i : (axis == 1 ? (k * ny + t_line) * nx + i : (t_line * ny + j) * nx + i);
            const int cl = axis == 0 ? cr - 1 : (axis == 1 ? cr - nx : cr - nx * ny);
            s.ul[c] = Uc[cl];
            s.ur[c] = Uc[cr];
            s.ax[c] = axis;
        }
        g.riemann->face_flux_batch(s.ul, s.ur, s.ax, cnt, F_line + i0, s.sm);
        for (int c = 0; c < cnt; ++c)
            if (s.sm[c] > local) local = s.sm[c];
    }
    for (int i0 = first; g.Wp != nullptr && i0 <= last; i0 += kFusedChunk) {
        const int cnt = std::min(kFusedChunk, last - i0 + 1);
        // face i of the line has its stencil at base + i + p * stride: the faces of a chunk are consecutive in memory
        const size_t base = axis == 0 ? g.P(i0, j + pad, k + zp)
                                      : (axis == 1 ? g.P(i0 + pad, t_line, k + zp) : g.P(i0 + pad, j + pad, t_line));
        if (g.weno) {
            weno_line_states(g, base, stride, cnt, s.ul, s.ur);
            for (int c = 0; c < cnt; ++c) s.ax[c] = axis;
            g.riemann->face_flux_batch(s.ul, s.ur, s.ax, cnt, F_line + i0, s.sm);
        } else {
            g.riemann->muscl_line_flux_batch(g.lim, g.Wp, g.plane, base, stride, axis, cnt, F_line + i0, s.sm, nz == 1);
        }
        for (int c = 0; c < cnt; ++c)
            if (s.sm[c] > local) local = s.sm[c];
    }
    // the faces of the sides: first-order ghost states through the round trip, then the Riemann solver
    const int f0 = axis == 0 ? g.m->xface(0, j, k) : (axis == 1 ? g.m->yface(0, j, k) : g.m->zface(0, j, k));
    for (int i = 0; i < ni; ++i) {
        if (i >= first && i <= last) {
            i = last;   // skip the reconstructed run
            continue;
        }
        State ul = (*g.UL)[f0 + i], ur = (*g.UR)[f0 + i];
        if (g.Wp != nullptr) {
            ul = g.eos->prim_to_cons(g.eos->cons_to_prim(ul));
            ur = g.eos->prim_to_cons(g.eos->cons_to_prim(ur));
        }
        double sp = 0.0;
        F_line[i] = g.riemann->face_flux(ul, ur, axis, sp);
        if (sp > local) local = sp;
    }
}

// The primitive state padded by `pad` cells (edge replication; halos, `pad` deep, replace it where present). The layouts of Halos reduce to
// the 2-D ones when nz = 1. The cells of the rows of the domain are converted by the vector kernel where the CPU has it.
void pad_primitives(const EOS& eos, const Field& U, const Mesh& m, const Halos& h, std::vector<double>& Wp, int& PX, int& PY, int& zp,
                    int pad, bool prim = true) {
    const auto conv = [&](const State& u) { return prim ? eos.cons_to_prim(u) : u; };
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const bool three_d = nz > 1;
    zp = three_d ? pad : 0;   // padding along z (none in 2-D)
    PX = nx + 2 * pad;
    PY = ny + 2 * pad;
    const int PZ = nz + 2 * zp;
    const size_t plane = static_cast<size_t>(PZ) * PY * PX;
    Wp.resize(5 * plane);
    const auto put = [&](size_t idx, const State& w) {
        for (int q = 0; q < kNVar; ++q) Wp[q * plane + idx] = w[q];
    };
    const auto P = [&](int ii, int jj, int kk) -> size_t { return (static_cast<size_t>(kk) * PY + jj) * PX + ii; };
#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
    for (int kj = 0; kj < PZ * PY; ++kj) {   // rows of the padded array (in 2-D, PZ = 1: the rows themselves)
        const int kk = kj / PY, jj = kj % PY;
        const int k = std::min(std::max(kk - zp, 0), nz - 1);
        const int j = std::min(std::max(jj - pad, 0), ny - 1);
        int iv = 0;   // the cells of the row that the vector kernel converts
#ifdef FV_SIMD
        if (prim && detail::simd_available() && jj >= pad && jj < pad + ny && kk >= zp && kk < zp + nz) {
            iv = nx & ~3;
            double* o[kNVar];
            for (int q = 0; q < kNVar; ++q) o[q] = Wp.data() + q * plane + P(pad, jj, kk);
            detail::cons_to_prim_planes_simd(eos, &U[m.cell(0, j, k)], iv, o);
        }
#endif
        for (int ii = 0; ii < PX; ++ii) {
            if (ii >= pad && ii < pad + iv) continue;
            put(P(ii, jj, kk), conv(U[m.cell(std::min(std::max(ii - pad, 0), nx - 1), j, k)]));
        }
    }
    for (int k = 0; k < nz; ++k)
        for (int d = 0; d < pad; ++d) {
            if (h.b)
                for (int i = 0; i < nx; ++i) put(P(i + pad, d, k + zp), conv((*h.b)[(d * nz + k) * nx + i]));
            if (h.t)
                for (int i = 0; i < nx; ++i)
                    put(P(i + pad, ny + pad + d, k + zp), conv((*h.t)[(d * nz + k) * nx + i]));
            if (h.l)
                for (int j = 0; j < ny; ++j) put(P(d, j + pad, k + zp), conv((*h.l)[(k * ny + j) * pad + d]));
            if (h.r)
                for (int j = 0; j < ny; ++j)
                    put(P(nx + pad + d, j + pad, k + zp), conv((*h.r)[(k * ny + j) * pad + d]));
        }
    if (three_d)
        for (int d = 0; d < pad; ++d)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    if (h.f) put(P(i + pad, j + pad, d), conv((*h.f)[(d * ny + j) * nx + i]));
                    if (h.bk) put(P(i + pad, j + pad, nz + pad + d), conv((*h.bk)[(d * ny + j) * nx + i]));
                }
}

// The grid description of a reconstruction `scheme` (the padded primitives are `pad` cells deep: 3 for WENO5-Z, else 2). With WENO5-Z and
// an immersed body the solid cells go on the padded layout (a cell outside the domain is solid if the cell it replicates or the cell a
// periodic wrap puts there is), as reconstruct_weno does.
FusedGrid make_grid(const RiemannSolver& riemann, const EOS& eos, Limiter lim, Reconstruction scheme, double weno_eps,
                    const std::vector<double>& Wp, std::vector<char>& S_pad, const Field& UL, const Field& UR, const Mesh& m,
                    const Halos& h, int PX, int PY, int zp, int pad) {
    if ((h.l || h.r || h.b || h.t || h.f || h.bk) && h.depth < pad)
        throw std::logic_error(pad == 3 ? "WENO5-Z needs halos three cells deep" : "the reconstruction needs halos two cells deep");
    FusedGrid g{};
    g.weno = scheme ==Reconstruction::Weno5 ? 5 : (scheme == Reconstruction::Weno3 ? 3 : 0);
    g.weno_eps = weno_eps;
    g.solid = nullptr;
    if (g.weno == 5 && h.narrow) {
        const int nx = m.nx, ny = m.ny, nz = m.nz;
        if (h.narrow->size() != static_cast<size_t>(nx) * ny * nz) throw std::logic_error("WENO5-Z: the solid mask is not that of the mesh");
        const int PZ = nz + 2 * zp;
        S_pad.resize(static_cast<size_t>(PZ) * PY * PX);
        const auto clampi = [](int i, int n) { return i < 0 ? 0 : (i >= n ? n - 1 : i); };
        const auto wrapi = [](int i, int n) { return ((i % n) + n) % n; };
        for (int kk = 0; kk < PZ; ++kk)
            for (int jj = 0; jj < PY; ++jj)
                for (int ii = 0; ii < PX; ++ii) {
                    const int i = ii - pad, j = jj - pad, k = kk - zp;
                    S_pad[(static_cast<size_t>(kk) * PY + jj) * PX + ii] =
                        ((*h.narrow)[m.cell(clampi(i, nx), clampi(j, ny), clampi(k, nz))] != 0 ||
                         (*h.narrow)[m.cell(wrapi(i, nx), wrapi(j, ny), wrapi(k, nz))] != 0)
                            ? 1 : 0;
                }
        g.solid = S_pad.data();
    }
    g.riemann = &riemann;
    g.eos = &eos;
    g.lim = lim;
    g.Wp = Wp.data();
    g.plane = Wp.size() / 5;
    g.U = nullptr;
    g.UL = &UL;
    g.UR = &UR;
    g.m = &m;
    g.nx = m.nx;
    g.ny = m.ny;
    g.nz = m.nz;
    g.PX = PX;
    g.PY = PY;
    g.pad = pad;
    g.zp = zp;
    g.halo_lo[0] = h.l != nullptr;
    g.halo_hi[0] = h.r != nullptr;
    g.halo_lo[1] = h.b != nullptr;
    g.halo_hi[1] = h.t != nullptr;
    g.halo_lo[2] = h.f != nullptr;
    g.halo_hi[2] = h.bk != nullptr;
    return g;
}

}   // namespace

double MusclReconstruction::muscl_fluxes(const RiemannSolver& riemann, const Field& UL, const Field& UR, const Field& U,
                                         const Mesh& m, const Halos& h, Field& F) const {
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    thread_local std::vector<double> W_pad;
    thread_local std::vector<char> S_pad;
    int PX, PY, zp;
    const int pad = scheme_ == Reconstruction::Weno5 ? 3 : kHaloDepth;
    pad_primitives(eos_, U, m, h, W_pad, PX, PY, zp, pad);
    F.resize(m.num_faces());
    const FusedGrid g = make_grid(riemann, eos_, lim_, scheme_, weno_eps_, W_pad, S_pad, UL, UR, m, h, PX, PY, zp, pad);
    double smax_all = 0.0;
#pragma omp parallel if (nx * ny * nz >= kOmpMinWork)
    {
        double local = 0.0;
        FusedScratch scratch;
#pragma omp for schedule(static) nowait
        for (int kj = 0; kj < ny * nz; ++kj) {   // x lines
            const int j = kj % ny, k = kj / ny;
            fused_line(g, 0, j, k, &F[m.xface(0, j, k)], local, scratch);
        }
#pragma omp for schedule(static) nowait
        for (int kj = 0; kj < (ny + 1) * nz; ++kj) {   // y lines
            const int j = kj % (ny + 1), k = kj / (ny + 1);
            fused_line(g, 1, j, k, &F[m.yface(0, j, k)], local, scratch);
        }
        if (nz > 1) {
#pragma omp for schedule(static) nowait
            for (int kj = 0; kj < ny * (nz + 1); ++kj) {   // z lines
                const int j = kj % ny, k = kj / ny;
                fused_line(g, 2, j, k, &F[m.zface(0, j, k)], local, scratch);
            }
        }
#pragma omp critical(fv_muscl_fluxes_max)
        if (local > smax_all) smax_all = local;
    }
    return smax_all;
}

namespace {

// The row-pipelined residual of a 2-D mesh for a grid description (MUSCL or first order); see MusclReconstruction::muscl_residual_2d.
double fused_residual_2d(const FusedGrid& g, const Mesh& m, Field& R) {
    const int nx = m.nx, ny = m.ny;
    const double inv_area = 1.0 / m.cell_area(), dxl = m.dx, dyl = m.dy;
    double smax_all = 0.0;
#pragma omp parallel if (nx * ny >= kOmpMinWork)
    {
        double local = 0.0;
        FusedScratch scratch;
        int nthreads = 1, tid = 0;
#ifdef _OPENMP
        nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
#endif
        // a block of consecutive rows per thread: the x faces of a row and the y faces above it are solved once and combined at
        // once, the y faces above a row are the ones below the next (only the first row of a block solves its lower ones)
        const int j0 = static_cast<int>(static_cast<long long>(ny) * tid / nthreads);
        const int j1 = static_cast<int>(static_cast<long long>(ny) * (tid + 1) / nthreads);
        if (j0 < j1) {
            std::vector<State> fx(nx + 1), below(nx), above(nx);
            fused_line(g, 1, j0, 0, below.data(), local, scratch);
            for (int j = j0; j < j1; ++j) {
                fused_line(g, 0, j, 0, fx.data(), local, scratch);
                fused_line(g, 1, j + 1, 0, above.data(), local, scratch);
                for (int i = 0; i < nx; ++i) {
                    const State &fl = fx[i], &fr = fx[i + 1], &fb = below[i], &ft = above[i];
                    State& r = R[j * nx + i];
                    for (int q = 0; q < kNVar; ++q)
                        r[q] = ((0.0 - (fr[q] * dyl - fl[q] * dyl)) - (ft[q] * dxl - fb[q] * dxl)) * inv_area;
                }
                below.swap(above);
            }
        }
#pragma omp critical(fv_muscl_residual_max)
        if (local > smax_all) smax_all = local;
    }
    return smax_all;
}

}   // namespace

double MusclReconstruction::muscl_residual_2d(const RiemannSolver& riemann, const Field& UL, const Field& UR, const Field& U,
                                              const Mesh& m, const Halos& h, Field& R) const {
    thread_local std::vector<double> W_pad;
    thread_local std::vector<char> S_pad;
    int PX, PY, zp;
    const int pad = scheme_ == Reconstruction::Weno5 ? 3 : kHaloDepth;
    pad_primitives(eos_, U, m, h, W_pad, PX, PY, zp, pad);
    R.resize(U.size());
    return fused_residual_2d(make_grid(riemann, eos_, lim_, scheme_, weno_eps_, W_pad, S_pad, UL, UR, m, h, PX, PY, zp, pad), m, R);
}

namespace {

// The plane-pipelined residual of a 3-D mesh: a thread takes a block of consecutive z planes; for each plane it solves the x faces of
// every row, the y faces above and below the rows and the z faces below and above the plane (the upper ones are the lower ones of the
// next plane), and balances the cells at once. The planes at the seam of two blocks are solved by both threads (the same bits).
double fused_residual_3d(const FusedGrid& g, const Mesh& m, Field& R) {
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const double Ax = m.face_area(0), Ay = m.face_area(1), Az = m.face_area(2), inv_vol = 1.0 / m.cell_volume();
    double smax_all = 0.0;
#pragma omp parallel if (nx * ny * nz >= kOmpMinWork)
    {
        double local = 0.0;
        FusedScratch scratch;
        int nthreads = 1, tid = 0;
#ifdef _OPENMP
        nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
#endif
        const int k0 = static_cast<int>(static_cast<long long>(nz) * tid / nthreads);
        const int k1 = static_cast<int>(static_cast<long long>(nz) * (tid + 1) / nthreads);
        if (k0 < k1) {
            std::vector<State> fx(nx + 1), below(nx), above(nx), zlow(static_cast<size_t>(nx) * ny), zhigh(zlow.size());
            for (int j = 0; j < ny; ++j) fused_line(g, 2, j, k0, &zlow[static_cast<size_t>(j) * nx], local, scratch);
            for (int k = k0; k < k1; ++k) {
                for (int j = 0; j < ny; ++j) fused_line(g, 2, j, k + 1, &zhigh[static_cast<size_t>(j) * nx], local, scratch);
                fused_line(g, 1, 0, k, below.data(), local, scratch);
                for (int j = 0; j < ny; ++j) {
                    fused_line(g, 0, j, k, fx.data(), local, scratch);
                    fused_line(g, 1, j + 1, k, above.data(), local, scratch);
                    for (int i = 0; i < nx; ++i) {
                        const State &fl = fx[i], &fr = fx[i + 1], &fb = below[i], &ft = above[i];
                        const State &ff = zlow[static_cast<size_t>(j) * nx + i], &fk = zhigh[static_cast<size_t>(j) * nx + i];
                        State& r = R[m.cell(i, j, k)];
                        for (int q = 0; q < kNVar; ++q)
                            r[q] = (((0.0 - (fr[q] * Ax - fl[q] * Ax)) - (ft[q] * Ay - fb[q] * Ay)) - (fk[q] * Az - ff[q] * Az)) * inv_vol;
                    }
                    below.swap(above);
                }
                zlow.swap(zhigh);
            }
        }
#pragma omp critical(fv_muscl_residual3d_max)
        if (local > smax_all) smax_all = local;
    }
    return smax_all;
}

}   // namespace

double MusclReconstruction::muscl_residual_3d(const RiemannSolver& riemann, const Field& UL, const Field& UR, const Field& U,
                                              const Mesh& m, const Halos& h, Field& R) const {
    thread_local std::vector<double> W_pad;
    thread_local std::vector<char> S_pad;
    int PX, PY, zp;
    const int pad = scheme_ == Reconstruction::Weno5 ? 3 : kHaloDepth;
    pad_primitives(eos_, U, m, h, W_pad, PX, PY, zp, pad);
    R.resize(U.size());
    return fused_residual_3d(make_grid(riemann, eos_, lim_, scheme_, weno_eps_, W_pad, S_pad, UL, UR, m, h, PX, PY, zp, pad), m, R);
}

double RiemannSolver::first_order_residual_2d(const Field& UL, const Field& UR, const Field& U, const Mesh& m, Field& R) const {
    R.resize(U.size());
    FusedGrid g{};
    g.riemann = this;
    g.eos = &eos_;
    g.U = &U;
    g.UL = &UL;
    g.UR = &UR;
    g.m = &m;
    g.nx = m.nx;
    g.ny = m.ny;
    g.nz = m.nz;
    g.weno = 0;
    g.solid = nullptr;
    return fused_residual_2d(g, m, R);
}

namespace {
std::vector<double>& conserved_scratch() {
    thread_local std::vector<double> w;
    return w;
}

// The WENO3-Z / WENO5-Z reconstruction of the conserved cell averages, one line of faces at a time (shared by reconstruct_conserved, which fills
// UL / UR, and the fused fourth-order residual, which keeps a few lines). The padded primitive copy is made once, by the thread that builds this.
struct ConservedLines {
    bool five;
    int pad, PX, PY, zp, nx, ny, nz;
    double we, g1, gp;
    const double* pl[kNVar];
    bool vec;

    ConservedLines(const EOS& eos, Reconstruction scheme, double weno_eps, const Field& U, const Mesh& m, const Halos& h)
        : five(scheme == Reconstruction::Weno5), pad(five ? 3 : kHaloDepth), nx(m.nx), ny(m.ny), nz(m.nz), we(weno_eps) {
        if ((h.l || h.r || h.b || h.t || h.f || h.bk) && h.depth < pad)
            throw std::logic_error(five ? "WENO5-Z needs halos three cells deep" : "WENO3-Z needs halos two cells deep");
        std::vector<double>& Wp = conserved_scratch();
        pad_primitives(eos, U, m, h, Wp, PX, PY, zp, pad, false);
        const size_t plane = Wp.size() / 5;
        const double* const W = Wp.data();
        for (int q = 0; q < kNVar; ++q) pl[q] = W + static_cast<size_t>(q) * plane;
#ifdef FV_SIMD
        vec = detail::simd_available();
#else
        vec = false;   // (read only by the vector branch below)
#endif
        // conserved reconstruction can give a negative pressure (or density) where the primitive one cannot: such a face takes the first-order states,
        // the two cells next to it (positive by induction)
        g1 = eos.gamma - 1.0;
        gp = eos.type == EOS::Type::IdealGas ? 0.0 : eos.gamma * eos.p_inf;
    }

    bool positive(const State& u) const {
        return u[0] > 0.0 && g1 * (u[3] * u[0] - 0.5 * (u[1] * u[1] + u[2] * u[2] + u[4] * u[4])) > gp * u[0];
    }

    // The faces of the line (axis, j, k) with a full stencil go to ul / ur (the first face of the line); returns false, and writes nothing, when the
    // line has none (a physical side without a ghost layer).
    bool line(int axis, int j, int k, bool has_lo, bool has_hi, State* ul, State* ur) const {
        const int n_a = axis == 0 ? nx : (axis == 1 ? ny : nz);
        const size_t stride = axis == 0 ? 1 : (axis == 1 ? static_cast<size_t>(PX) : static_cast<size_t>(PX) * PY);
        const int t_line = axis == 1 ? j : k;
        int first, last;   // the faces of the line with a full stencil
        if (axis == 0) {
            first = has_lo ? 0 : 1;
            last = has_hi ? nx : nx - 1;
        } else if ((t_line > 0 && t_line < n_a) || (t_line == 0 && has_lo) || (t_line == n_a && has_hi)) {
            first = 0;
            last = nx - 1;
        } else {
            return false;
        }
        const auto base_of = [&](int i) {
            return (static_cast<size_t>(axis == 2 ? k : k + zp) * PY + (axis == 1 ? j : j + pad)) * PX + (axis == 0 ? i : i + pad);
        };
        int done = 0;
        [[maybe_unused]] const int cnt = last - first + 1;
#ifdef FV_SIMD
        if (vec) {
            done = cnt & ~3;
            if (done > 0) {
                if (five)
                    detail::weno5_line_cons_simd(we, pl, base_of(first), stride, done, &ul[first], &ur[first]);
                else
                    detail::weno3_line_cons_simd(we, pl, base_of(first), stride, done, &ul[first], &ur[first]);
            }
        }
#endif
        for (int i = first + done; i <= last; ++i) {
            const size_t base = base_of(i);
            State wl, wr;
            for (int q = 0; q < kNVar; ++q) {
                const double* c = pl[q] + base;
                if (five) {
                    wl[q] = weno5z(c[0], c[stride], c[2 * stride], c[3 * stride], c[4 * stride], we);
                    wr[q] = weno5z(c[5 * stride], c[4 * stride], c[3 * stride], c[2 * stride], c[stride], we);
                } else {
                    wl[q] = weno3z(c[0], c[stride], c[2 * stride], we);
                    wr[q] = weno3z(c[3 * stride], c[2 * stride], c[stride], we);
                }
            }
            ul[i] = wl;
            ur[i] = wr;
        }
        // the faces of the line whose states are not admissible: the cells on both sides (padded positions 2 and 3 of the six, 1 and 2 of
        // the four)
        const int left = five ? 2 : 1, ncell = five ? 6 : 4;
        const auto prs = [&](const State& u) { return g1 * (u[3] - 0.5 * (u[1] * u[1] + u[2] * u[2] + u[4] * u[4]) / u[0]) - gp; };
        for (int i = first; i <= last; ++i) {
            const size_t base = base_of(i);
            // admissible: positive, and the density and the pressure inside the range of the stencil cells (a wild overshoot at a face)
            bool ok = positive(ul[i]) && positive(ur[i]);
            if (ok) {
                double rlo = 1e300, rhi = -1e300, plo = 1e300, phi = -1e300;
                for (int c = 0; c < ncell; ++c) {
                    State u;
                    for (int q = 0; q < kNVar; ++q) u[q] = pl[q][base + c * stride];
                    const double pc = prs(u);
                    rlo = std::min(rlo, u[0]);
                    rhi = std::max(rhi, u[0]);
                    plo = std::min(plo, pc);
                    phi = std::max(phi, pc);
                }
                // a smooth extremum puts the face value a little beyond the cell averages: the range is widened by its own width
                const double rw = rhi - rlo, pw = phi - plo;
                rlo -= rw;
                rhi += rw;
                plo -= pw;
                phi += pw;
                for (const State* st : {&ul[i], &ur[i]}) {
                    const double pf = prs(*st);
                    ok = ok && (*st)[0] >= rlo && (*st)[0] <= rhi && pf >= plo && pf <= phi;
                }
            }
            if (ok) continue;
            for (int q = 0; q < kNVar; ++q) {
                ul[i][q] = pl[q][base + left * stride];
                ur[i][q] = pl[q][base + (left + 1) * stride];
            }
        }
        return true;
    }
};
}   // namespace

void MusclReconstruction::reconstruct_conserved(Field& UL, Field& UR, const Field& U, const Mesh& m, const Halos& h) const {
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const ConservedLines lines(eos_, scheme_, weno_eps_, U, m, h);
    UL.resize(m.num_faces());
    UR.resize(m.num_faces());
    const auto sweep = [&](int axis, bool has_lo, bool has_hi) {
        const int nj = axis == 1 ? ny + 1 : ny, nk = axis == 2 ? nz + 1 : nz;
        const bool par = nx * ny * nz >= kOmpMinWork;
#pragma omp parallel for schedule(static) if (par)
        for (int kj = 0; kj < nk * nj; ++kj) {
            const int j = kj % nj, k = kj / nj;
            const int f0 = axis == 0 ? m.xface(0, j, k) : (axis == 1 ? m.yface(0, j, k) : m.zface(0, j, k));
            lines.line(axis, j, k, has_lo, has_hi, &UL[f0], &UR[f0]);
        }
    };
    sweep(0, h.l != nullptr, h.r != nullptr);
    sweep(1, h.b != nullptr, h.t != nullptr);
    if (nz > 1) sweep(2, h.f != nullptr, h.bk != nullptr);
}

namespace {
// The rows a thread keeps in the fused fourth-order residual (reused between calls).
struct FourthScratch {
    std::vector<State> xUL, xUR, xAL, xAR, xFp, xF;           // x faces: 5 rows of reconstructed states, 3 rows of point values / fluxes, one flux row
    std::vector<unsigned char> xgood, xpt;                      // 5 rows of flags, 3 rows of flags
    std::vector<State> yUL, yUR, yAL, yAR, yFp, yF;            // y faces: one row at a time, two rows of fluxes
    std::vector<unsigned char> ygood, ypt;
    std::vector<int> ax;
    std::vector<double> sm;
};
}   // namespace

double MusclReconstruction::fourth_order_residual_2d(const RiemannSolver& riemann, const Field& UL, const Field& UR, const Field& U, const Mesh& m,
                                                     const Halos& h, const bool periodic[2], Field& R) const {
    const int nx = m.nx, ny = m.ny;
    const ConservedLines lines(eos_, scheme_, weno_eps_, U, m, h);
    R.resize(U.size());
    const bool lo[2] = {h.l != nullptr, h.b != nullptr}, hi[2] = {h.r != nullptr, h.t != nullptr};
    const bool per_x = periodic[0], per_y = periodic[1];
    const double sb = riemann.shock_blend();
    const double g1 = eos_.gamma - 1.0, gpi = eos_.type == EOS::Type::IdealGas ? 0.0 : eos_.gamma * eos_.p_inf;
    const auto m2 = [](const State& u) { return u[1] * u[1] + u[2] * u[2] + u[4] * u[4]; };
    // the same tests as fourth_order_fluxes
    const auto positive = [&](const State& u) { return u[0] > 0.0 && g1 * (u[3] * u[0] - 0.5 * m2(u)) > gpi * u[0]; };
    const auto pressure = [&](const State& u) { return g1 * (u[3] - 0.5 * m2(u) / u[0]) - gpi; };
    // good: positive states, the shock sensor quiet (the position tests are made by the caller)
    const auto quiet = [&](const State& l, const State& r) {
        if (!positive(l) || !positive(r)) return false;
        if (sb > 0.0) {
            const double pL = pressure(l), pR = pressure(r);
            if (std::abs(pR - pL) > sb * std::min(pL, pR)) return false;
        }
        return true;
    };
    const double inv_area = 1.0 / m.cell_area(), dxl = m.dx, dyl = m.dy;
    const int nxf = nx + 1;
    // a row index of the x faces (c[1] in fourth_order_fluxes) as the row that holds its data, -1 when it is outside and not periodic
    const auto phys = [&](int r) {
        if (r >= 0 && r < ny) return r;
        if (!per_y) return -1;
        return ((r % ny) + ny) % ny;
    };
    const auto slot5 = [](int r) { return ((r % 5) + 5) % 5; };
    const auto slot3 = [](int r) { return ((r % 3) + 3) % 3; };
    double smax_all = 0.0;
    // one block of consecutive rows per thread (at least 16 rows: every block repeats the four rows around its seams)
    const bool par = nx * ny >= kOmpMinWork;
#pragma omp parallel if (par)
    {
        double local = 0.0;
        thread_local FourthScratch sc;
        int nthreads = 1, tid = 0;
#ifdef _OPENMP
        nthreads = omp_get_num_threads();
        tid = omp_get_thread_num();
#endif
        const int nblocks = std::max(1, std::min(nthreads, ny / 16));
        if (tid < nblocks) {
            const int J0 = static_cast<int>(static_cast<long long>(ny) * tid / nblocks);
            const int J1 = static_cast<int>(static_cast<long long>(ny) * (tid + 1) / nblocks);
            const size_t W = static_cast<size_t>(nxf);
            sc.xUL.resize(5 * W);
            sc.xUR.resize(5 * W);
            sc.xgood.resize(5 * W);
            sc.xAL.resize(3 * W);
            sc.xAR.resize(3 * W);
            sc.xFp.resize(3 * W);
            sc.xpt.resize(3 * W);
            sc.xF.resize(W);
            sc.yUL.resize(nx);
            sc.yUR.resize(nx);
            sc.yAL.resize(nx);
            sc.yAR.resize(nx);
            sc.yFp.resize(nx);
            sc.yF.resize(2 * static_cast<size_t>(nx));
            sc.ygood.resize(nx);
            sc.ypt.resize(nx);
            sc.ax.assign(W, 0);
            sc.sm.resize(W);
            std::vector<int> ones(static_cast<size_t>(nx), 1);

            // the states of the faces of row r (unwrapped index) and their flags
            const auto recon_x = [&](int r) {
                const int pr = phys(r);
                if (pr < 0) return;
                State* ul = &sc.xUL[slot5(r) * W];
                State* ur = &sc.xUR[slot5(r) * W];
                unsigned char* gd = &sc.xgood[slot5(r) * W];
                lines.line(0, pr, 0, lo[0], hi[0], ul, ur);
                const int first = lo[0] ? 0 : 1, last = hi[0] ? nx : nx - 1;
                for (int i = 0; i < first; ++i) {
                    ul[i] = UL[m.xface(i, pr, 0)];
                    ur[i] = UR[m.xface(i, pr, 0)];
                }
                for (int i = last + 1; i <= nx; ++i) {
                    ul[i] = UL[m.xface(i, pr, 0)];
                    ur[i] = UR[m.xface(i, pr, 0)];
                }
                for (int i = 0; i <= nx; ++i) {
                    gd[i] = 0;
                    if (!((i > 0 && i < nx) || (i == 0 && lo[0]) || (i == nx && hi[0]))) continue;
                    gd[i] = quiet(ul[i], ur[i]) ? 1 : 0;
                }
            };
            // the point values of row r, and the flux of the point values (or of the averages where there are none)
            const auto point_x = [&](int r) {
                const int s5 = slot5(r), s3 = slot3(r);
                const State *ul = &sc.xUL[s5 * W], *ur = &sc.xUR[s5 * W];
                const unsigned char* gd = &sc.xgood[s5 * W];
                State *al = &sc.xAL[s3 * W], *ar = &sc.xAR[s3 * W], *fp = &sc.xFp[s3 * W];
                unsigned char* pt = &sc.xpt[s3 * W];
                const bool nb = phys(r - 1) >= 0 && phys(r + 1) >= 0;
                const State *ulm = nb ? &sc.xUL[slot5(r - 1) * W] : nullptr, *ulp = nb ? &sc.xUL[slot5(r + 1) * W] : nullptr;
                const State *urm = nb ? &sc.xUR[slot5(r - 1) * W] : nullptr, *urp = nb ? &sc.xUR[slot5(r + 1) * W] : nullptr;
                const unsigned char *gm = nb ? &sc.xgood[slot5(r - 1) * W] : nullptr, *gq = nb ? &sc.xgood[slot5(r + 1) * W] : nullptr;
                for (int i = 0; i <= nx; ++i) {
                    al[i] = ul[i];
                    ar[i] = ur[i];
                    pt[i] = 0;
                    if (!nb || !gd[i] || !gm[i] || !gq[i]) continue;
                    State pl = ul[i], pr_ = ur[i];
                    for (int q = 0; q < kNVar; ++q) {
                        double dl = 0.0, dr = 0.0;
                        dl += ulp[i][q] - 2.0 * ul[i][q] + ulm[i][q];
                        dr += urp[i][q] - 2.0 * ur[i][q] + urm[i][q];
                        pl[q] -= dl / 24.0;
                        pr_[q] -= dr / 24.0;
                    }
                    if (!positive(pl) || !positive(pr_)) continue;
                    al[i] = pl;
                    ar[i] = pr_;
                    pt[i] = 1;
                }
                riemann.face_flux_batch(al, ar, sc.ax.data(), nxf, fp, sc.sm.data());
                for (int i = 0; i <= nx; ++i)
                    if (sc.sm[i] > local) local = sc.sm[i];
            };
            // the flux of the x faces of row j into sc.xF
            const auto flux_x = [&](int j) {
                const int s5 = slot5(j), s3 = slot3(j);
                const State *ul = &sc.xUL[s5 * W], *ur = &sc.xUR[s5 * W];
                const State* fp = &sc.xFp[s3 * W];
                const unsigned char* pt = &sc.xpt[s3 * W];
                const bool nb = phys(j - 1) >= 0 && phys(j + 1) >= 0;
                const State *fpm = nb ? &sc.xFp[slot3(j - 1) * W] : nullptr, *fpp = nb ? &sc.xFp[slot3(j + 1) * W] : nullptr;
                const unsigned char *ptm = nb ? &sc.xpt[slot3(j - 1) * W] : nullptr, *ptp = nb ? &sc.xpt[slot3(j + 1) * W] : nullptr;
                for (int i = 0; i <= nx; ++i) {
                    if (pt[i] && nb && ptm[i] && ptp[i]) {
                        for (int q = 0; q < kNVar; ++q) {
                            double d = 0.0;
                            d += fpp[i][q] - 2.0 * fp[i][q] + fpm[i][q];
                            sc.xF[i][q] = fp[i][q] + d / 24.0;
                        }
                    } else if (pt[i]) {   // a point value without the full correction: back to the flux of the averaged states
                        double sm = 0.0;
                        sc.xF[i] = riemann.face_flux(ul[i], ur[i], 0, sm);
                    } else {
                        sc.xF[i] = fp[i];
                    }
                }
            };
            // the fluxes of the y faces of row ry (all of it: the transverse neighbours are in the row) into slot ry mod 2 of sc.yF
            const auto row_y = [&](int ry) {
                State *ul = sc.yUL.data(), *ur = sc.yUR.data(), *al = sc.yAL.data(), *ar = sc.yAR.data(), *fp = sc.yFp.data();
                State* fo = &sc.yF[(ry & 1) * static_cast<size_t>(nx)];
                unsigned char *gd = sc.ygood.data(), *pt = sc.ypt.data();
                if (!lines.line(1, ry, 0, lo[1], hi[1], ul, ur))
                    for (int i = 0; i < nx; ++i) {
                        ul[i] = UL[m.yface(i, ry, 0)];
                        ur[i] = UR[m.yface(i, ry, 0)];
                    }
                const bool row_in = (ry > 0 && ry < ny) || (ry == 0 && lo[1]) || (ry == ny && hi[1]);
                for (int i = 0; i < nx; ++i) gd[i] = row_in && quiet(ul[i], ur[i]) ? 1 : 0;
                const auto left = [&](int i) { return i > 0 ? i - 1 : (per_x ? nx - 1 : -1); };
                const auto right = [&](int i) { return i + 1 < nx ? i + 1 : (per_x ? 0 : -1); };
                for (int i = 0; i < nx; ++i) {
                    al[i] = ul[i];
                    ar[i] = ur[i];
                    pt[i] = 0;
                    if (!gd[i]) continue;
                    const int a = left(i), b = right(i);
                    if (a < 0 || b < 0 || !gd[a] || !gd[b]) continue;
                    State pl = ul[i], pr_ = ur[i];
                    for (int q = 0; q < kNVar; ++q) {
                        double dl = 0.0, dr = 0.0;
                        dl += ul[b][q] - 2.0 * ul[i][q] + ul[a][q];
                        dr += ur[b][q] - 2.0 * ur[i][q] + ur[a][q];
                        pl[q] -= dl / 24.0;
                        pr_[q] -= dr / 24.0;
                    }
                    if (!positive(pl) || !positive(pr_)) continue;
                    al[i] = pl;
                    ar[i] = pr_;
                    pt[i] = 1;
                }
                riemann.face_flux_batch(al, ar, ones.data(), nx, fp, sc.sm.data());
                for (int i = 0; i < nx; ++i)
                    if (sc.sm[i] > local) local = sc.sm[i];
                for (int i = 0; i < nx; ++i) {
                    const int a = left(i), b = right(i);
                    if (pt[i] && a >= 0 && b >= 0 && pt[a] && pt[b]) {
                        for (int q = 0; q < kNVar; ++q) {
                            double d = 0.0;
                            d += fp[b][q] - 2.0 * fp[i][q] + fp[a][q];
                            fo[i][q] = fp[i][q] + d / 24.0;
                        }
                    } else if (pt[i]) {
                        double sm = 0.0;
                        fo[i] = riemann.face_flux(ul[i], ur[i], 1, sm);
                    } else {
                        fo[i] = fp[i];
                    }
                }
            };

            int xr = J0 - 3, xp = J0 - 2, yr = J0 - 1;   // the last row of each stage done
            for (int j = J0; j < J1; ++j) {
                while (xr < j + 2) recon_x(++xr);
                while (xp < j + 1) {
                    ++xp;
                    if (phys(xp) >= 0) point_x(xp);
                }
                while (yr < j + 1) row_y(++yr);
                flux_x(j);
                const State* fb = &sc.yF[(j & 1) * static_cast<size_t>(nx)];
                const State* ft = &sc.yF[((j + 1) & 1) * static_cast<size_t>(nx)];
                for (int i = 0; i < nx; ++i) {
                    const State &fl = sc.xF[i], &fr = sc.xF[i + 1];
                    State& r = R[j * nx + i];
                    for (int q = 0; q < kNVar; ++q)
                        r[q] = ((0.0 - (fr[q] * dyl - fl[q] * dyl)) - (ft[i][q] * dxl - fb[i][q] * dxl)) * inv_area;
                }
            }
        }
#pragma omp critical(fv_fourth_residual_max)
        if (local > smax_all) smax_all = local;
    }
    return smax_all;
}

namespace {
// What the fused 3-D fourth-order residual shares between its tiles (read only).
struct Fourth3Shared {
    const RiemannSolver& riemann;
    const ConservedLines& lines;
    const Mesh& m;
    const Field &UL, &UR;   // the faces of the sides, from the boundary condition
    int n[3];
    bool lo[3], hi[3], per[3];
    double sb, g1, gpi;

    bool positive(const State& u) const {
        return u[0] > 0.0 && g1 * (u[3] * u[0] - 0.5 * (u[1] * u[1] + u[2] * u[2] + u[4] * u[4])) > gpi * u[0];
    }
    double pressure(const State& u) const { return g1 * (u[3] - 0.5 * (u[1] * u[1] + u[2] * u[2] + u[4] * u[4]) / u[0]) - gpi; }
    bool quiet(const State& l, const State& r) const {
        if (!positive(l) || !positive(r)) return false;
        if (sb > 0.0) {
            const double pL = pressure(l), pR = pressure(r);
            if (std::abs(pR - pL) > sb * std::min(pL, pR)) return false;
        }
        return true;
    }
    // the row (or plane) index that holds the data of an index along `axis`, -1 when it is outside and the axis is not periodic
    int phys(int axis, int r) const {
        if (r >= 0 && r < n[axis]) return r;
        if (!per[axis]) return -1;
        return ((r % n[axis]) + n[axis]) % n[axis];
    }
};

// The storage of one face family in a tile: rows r (global, unwrapped index) of faces i, planes kp in a ring.
struct Fourth3Fam {
    int a = 0, L = 0, S5 = 1, S3 = 1, rb = 0, R = 0, H = 0;
    std::vector<State> UL, UR, Fp, F;
    std::vector<unsigned char> good, pt;
    std::vector<int> ax;
    size_t i5(int kp, int r, int i) const { return (static_cast<size_t>(((kp % S5) + S5) % S5) * R + (r - rb)) * L + i; }
    size_t i3(int kp, int r, int i) const { return (static_cast<size_t>(((kp % S3) + S3) % S3) * R + (r - rb)) * L + i; }
};

struct Fourth3Scratch {
    Fourth3Fam f[3];
    std::vector<State> al, ar;
    std::vector<double> sm;
};

// One tile: the cell rows [jlo, jhi) of the cell planes [K0, K1), streamed plane by plane.
struct Fourth3Tile {
    const Fourth3Shared& S;
    Fourth3Scratch& sc;
    double& smax;
    int jlo, jhi, K0, K1;

    void rows(int a, int stage, int& r0, int& r1) const {   // stage 0: states and flags, 1: point values and Fp, 2: flux
        if (a == 1) {
            r0 = jlo;
            r1 = jhi + 1;
        } else if (stage == 0) {
            r0 = jlo - 2;
            r1 = jhi + 2;
        } else if (stage == 1) {
            r0 = jlo - 1;
            r1 = jhi + 1;
        } else {
            r0 = jlo;
            r1 = jhi;
        }
    }

    // the states of the faces of plane kp and their flags
    void recon(int a, int kp) {
        Fourth3Fam& B = sc.f[a];
        const int pk = a == 2 ? kp : S.phys(2, kp);
        if (pk < 0) return;
        int r0, r1;
        rows(a, 0, r0, r1);
        const int nx = S.n[0];
        for (int r = r0; r < r1; ++r) {
            const int pr = a == 1 ? r : S.phys(1, r);
            if (pr < 0) continue;
            State* ul = &B.UL[B.i5(kp, r, 0)];
            State* ur = &B.UR[B.i5(kp, r, 0)];
            unsigned char* gd = &B.good[B.i5(kp, r, 0)];
            const bool wrote = S.lines.line(a, pr, pk, S.lo[a], S.hi[a], ul, ur);
            const auto gface = [&](int i) { return a == 0 ? S.m.xface(i, pr, pk) : (a == 1 ? S.m.yface(i, pr, pk) : S.m.zface(i, pr, pk)); };
            if (a == 0) {
                const int first = S.lo[0] ? 0 : 1, last = S.hi[0] ? nx : nx - 1;
                for (int i = 0; i < first; ++i) {
                    ul[i] = S.UL[gface(i)];
                    ur[i] = S.UR[gface(i)];
                }
                for (int i = last + 1; i <= nx; ++i) {
                    ul[i] = S.UL[gface(i)];
                    ur[i] = S.UR[gface(i)];
                }
            } else if (!wrote) {
                for (int i = 0; i < nx; ++i) {
                    ul[i] = S.UL[gface(i)];
                    ur[i] = S.UR[gface(i)];
                }
            }
            for (int i = 0; i < B.L; ++i) {
                gd[i] = 0;
                const int ca = a == 0 ? i : (a == 1 ? pr : pk);
                if (!((ca > 0 && ca < S.n[a]) || (ca == 0 && S.lo[a]) || (ca == S.n[a] && S.hi[a]))) continue;
                gd[i] = S.quiet(ul[i], ur[i]) ? 1 : 0;
            }
        }
    }

    void axes(int a, int& t0, int& t1) const {
        t0 = a == 0 ? 1 : 0;
        t1 = a == 2 ? 1 : 2;
    }

    // the point values of plane kp and the flux of the point values (the averaged states where there are none)
    void point(int a, int kp) {
        Fourth3Fam& B = sc.f[a];
        const int pk = a == 2 ? kp : S.phys(2, kp);
        if (pk < 0) return;
        int r0, r1;
        rows(a, 1, r0, r1);
        int tax[2];
        axes(a, tax[0], tax[1]);
        const int nx = S.n[0];
        const bool per0 = S.per[0];
        for (int r = r0; r < r1; ++r) {
            const int pr = a == 1 ? r : S.phys(1, r);
            if (pr < 0) continue;
            const State* ul = &B.UL[B.i5(kp, r, 0)];
            const State* ur = &B.UR[B.i5(kp, r, 0)];
            const unsigned char* gd = &B.good[B.i5(kp, r, 0)];
            unsigned char* pt = &B.pt[B.i3(kp, r, 0)];
            State *al = sc.al.data(), *ar = sc.ar.data();
            bool inrow[2], rowok = true;
            const State *nul[2][2], *nur[2][2];
            const unsigned char* ng[2][2];
            for (int u = 0; u < 2; ++u) {
                inrow[u] = tax[u] == 0;
                for (int sg = 0; sg < 2; ++sg) {
                    const int d = sg == 0 ? -1 : 1;
                    nul[u][sg] = ul;
                    nur[u][sg] = ur;
                    ng[u][sg] = gd;
                    if (inrow[u]) continue;
                    const int rr = tax[u] == 1 ? r + d : r, kk = tax[u] == 2 ? kp + d : kp;
                    if (S.phys(tax[u] == 1 ? 1 : 2, tax[u] == 1 ? rr : kk) < 0) {
                        rowok = false;
                        continue;
                    }
                    nul[u][sg] = &B.UL[B.i5(kk, rr, 0)];
                    nur[u][sg] = &B.UR[B.i5(kk, rr, 0)];
                    ng[u][sg] = &B.good[B.i5(kk, rr, 0)];
                }
            }
            for (int i = 0; i < B.L; ++i) {
                al[i] = ul[i];
                ar[i] = ur[i];
                pt[i] = 0;
                if (!gd[i] || !rowok) continue;
                int ia[2], ib[2];
                bool ok = true;
                for (int u = 0; u < 2 && ok; ++u) {
                    if (inrow[u]) {
                        ia[u] = i > 0 ? i - 1 : (per0 ? nx - 1 : -1);
                        ib[u] = i + 1 < nx ? i + 1 : (per0 ? 0 : -1);
                        ok = ia[u] >= 0 && ib[u] >= 0 && gd[ia[u]] && gd[ib[u]];
                    } else {
                        ia[u] = ib[u] = i;
                        ok = ng[u][0][i] && ng[u][1][i];
                    }
                }
                if (!ok) continue;
                State pl = ul[i], pr_ = ur[i];
                for (int q = 0; q < kNVar; ++q) {
                    double dl = 0.0, dr = 0.0;
                    for (int u = 0; u < 2; ++u) {
                        dl += nul[u][1][ib[u]][q] - 2.0 * ul[i][q] + nul[u][0][ia[u]][q];
                        dr += nur[u][1][ib[u]][q] - 2.0 * ur[i][q] + nur[u][0][ia[u]][q];
                    }
                    pl[q] -= dl / 24.0;
                    pr_[q] -= dr / 24.0;
                }
                if (!S.positive(pl) || !S.positive(pr_)) continue;
                al[i] = pl;
                ar[i] = pr_;
                pt[i] = 1;
            }
            S.riemann.face_flux_batch(al, ar, B.ax.data(), B.L, &B.Fp[B.i3(kp, r, 0)], sc.sm.data());
            for (int i = 0; i < B.L; ++i)
                if (sc.sm[i] > smax) smax = sc.sm[i];
        }
    }

    // the flux of the faces of plane kp (rows of the balance) into B.F
    void flux(int a, int kp) {
        Fourth3Fam& B = sc.f[a];
        int r0, r1;
        rows(a, 2, r0, r1);
        int tax[2];
        axes(a, tax[0], tax[1]);
        const int nx = S.n[0];
        const bool per0 = S.per[0];
        for (int r = r0; r < r1; ++r) {
            const int pr = a == 1 ? r : S.phys(1, r);
            if (pr < 0) continue;
            State* fo = &B.F[(a == 2 ? static_cast<size_t>(kp & 1) * B.H * B.L : 0) + static_cast<size_t>(r - jlo) * B.L];
            const State* ul = &B.UL[B.i5(kp, r, 0)];
            const State* ur = &B.UR[B.i5(kp, r, 0)];
            const unsigned char* pt = &B.pt[B.i3(kp, r, 0)];
            const State* fp = &B.Fp[B.i3(kp, r, 0)];
            bool inrow[2], rowok = true;
            const State* nfp[2][2];
            const unsigned char* npt[2][2];
            for (int u = 0; u < 2; ++u) {
                inrow[u] = tax[u] == 0;
                for (int sg = 0; sg < 2; ++sg) {
                    const int d = sg == 0 ? -1 : 1;
                    nfp[u][sg] = fp;
                    npt[u][sg] = pt;
                    if (inrow[u]) continue;
                    const int rr = tax[u] == 1 ? r + d : r, kk = tax[u] == 2 ? kp + d : kp;
                    if (S.phys(tax[u] == 1 ? 1 : 2, tax[u] == 1 ? rr : kk) < 0) {
                        rowok = false;
                        continue;
                    }
                    nfp[u][sg] = &B.Fp[B.i3(kk, rr, 0)];
                    npt[u][sg] = &B.pt[B.i3(kk, rr, 0)];
                }
            }
            for (int i = 0; i < B.L; ++i) {
                if (!pt[i]) {
                    fo[i] = fp[i];
                    continue;
                }
                int ia[2], ib[2];
                bool ok = rowok;
                for (int u = 0; u < 2 && ok; ++u) {
                    if (inrow[u]) {
                        ia[u] = i > 0 ? i - 1 : (per0 ? nx - 1 : -1);
                        ib[u] = i + 1 < nx ? i + 1 : (per0 ? 0 : -1);
                        ok = ia[u] >= 0 && ib[u] >= 0 && pt[ia[u]] && pt[ib[u]];
                    } else {
                        ia[u] = ib[u] = i;
                        ok = npt[u][0][i] && npt[u][1][i];
                    }
                }
                if (ok) {
                    for (int q = 0; q < kNVar; ++q) {
                        double d = 0.0;
                        for (int u = 0; u < 2; ++u) d += nfp[u][1][ib[u]][q] - 2.0 * fp[i][q] + nfp[u][0][ia[u]][q];
                        fo[i][q] = fp[i][q] + d / 24.0;
                    }
                } else {   // a point value without the full correction: back to the flux of the averaged states
                    double sm = 0.0;
                    fo[i] = S.riemann.face_flux(ul[i], ur[i], a, sm);
                }
            }
        }
    }
};
}   // namespace

double MusclReconstruction::fourth_order_residual_3d(const RiemannSolver& riemann, const Field& UL, const Field& UR, const Field& U, const Mesh& m,
                                                     const Halos& h, const bool periodic[3], Field& R) const {
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const ConservedLines lines(eos_, scheme_, weno_eps_, U, m, h);
    R.resize(U.size());
    Fourth3Shared S{riemann, lines, m, UL, UR, {nx, ny, nz},
                    {h.l != nullptr, h.b != nullptr, h.f != nullptr}, {h.r != nullptr, h.t != nullptr, h.bk != nullptr},
                    {periodic[0], periodic[1], periodic[2]}, riemann.shock_blend(), eos_.gamma - 1.0,
                    eos_.type == EOS::Type::IdealGas ? 0.0 : eos_.gamma * eos_.p_inf};
    const double Ax = m.face_area(0), Ay = m.face_area(1), Az = m.face_area(2), inv_vol = 1.0 / m.cell_volume();
    // tiles: strips of about 24 rows times blocks of planes (every tile repeats the rows and planes around its seams)
    const int nstrips = std::max(1, ny / 24);
    int nthreads_max = 1;
#ifdef _OPENMP
    nthreads_max = omp_get_max_threads();
#endif
    const bool par = nx * ny * nz >= kOmpMinWork;
    const int zblocks = std::max(1, std::min((3 * nthreads_max + nstrips - 1) / nstrips, nz / 16));
    const int ntiles = nstrips * zblocks;
    double smax_all = 0.0;
#pragma omp parallel if (par)
    {
        double local = 0.0;
        thread_local Fourth3Scratch sc;
#pragma omp for schedule(dynamic, 1) nowait
        for (int tile = 0; tile < ntiles; ++tile) {
            const int s = tile % nstrips, b = tile / nstrips;
            const int jlo = static_cast<int>(static_cast<long long>(ny) * s / nstrips), jhi = static_cast<int>(static_cast<long long>(ny) * (s + 1) / nstrips);
            const int K0 = static_cast<int>(static_cast<long long>(nz) * b / zblocks), K1 = static_cast<int>(static_cast<long long>(nz) * (b + 1) / zblocks);
            const int Hs = jhi - jlo;
            const int Ls[3] = {nx + 1, nx, nx};
            const int Rs[3] = {Hs + 4, Hs + 1, Hs + 4};
            const int rbs[3] = {jlo - 2, jlo, jlo - 2};
            const int S5[3] = {5, 5, 1}, S3[3] = {3, 3, 1};
            int maxL = 0;
            for (int a = 0; a < 3; ++a) {
                Fourth3Fam& B = sc.f[a];
                B.a = a;
                B.L = Ls[a];
                B.R = Rs[a];
                B.rb = rbs[a];
                B.S5 = S5[a];
                B.S3 = S3[a];
                B.H = Hs + (a == 1 ? 1 : 0);
                const size_t n5 = static_cast<size_t>(B.S5) * B.R * B.L, n3 = static_cast<size_t>(B.S3) * B.R * B.L;
                B.UL.resize(n5);
                B.UR.resize(n5);
                B.good.resize(n5);
                B.Fp.resize(n3);
                B.pt.resize(n3);
                B.F.resize(static_cast<size_t>(a == 2 ? 2 : 1) * B.H * B.L);
                B.ax.assign(static_cast<size_t>(B.L), a);
                maxL = std::max(maxL, B.L);
            }
            sc.al.resize(maxL);
            sc.ar.resize(maxL);
            sc.sm.resize(maxL);
            Fourth3Tile T{S, sc, local, jlo, jhi, K0, K1};
            int xr = K0 - 3, xp = K0 - 2;   // the last plane of the state / point value stages done (x and y faces)
            T.recon(2, K0);
            T.point(2, K0);
            T.flux(2, K0);
            for (int k = K0; k < K1; ++k) {
                while (xr < k + 2) {
                    ++xr;
                    T.recon(0, xr);
                    T.recon(1, xr);
                }
                while (xp < k + 1) {
                    ++xp;
                    T.point(0, xp);
                    T.point(1, xp);
                }
                T.flux(0, k);
                T.flux(1, k);
                T.recon(2, k + 1);
                T.point(2, k + 1);
                T.flux(2, k + 1);
                const Fourth3Fam &Fx = sc.f[0], &Fy = sc.f[1], &Fz = sc.f[2];
                const size_t zk = static_cast<size_t>(k & 1) * Fz.H * Fz.L, zk1 = static_cast<size_t>((k + 1) & 1) * Fz.H * Fz.L;
                for (int j = jlo; j < jhi; ++j)
                    for (int i = 0; i < nx; ++i) {
                        const State& fl = Fx.F[static_cast<size_t>(j - jlo) * Fx.L + i];
                        const State& fr = Fx.F[static_cast<size_t>(j - jlo) * Fx.L + i + 1];
                        const State& fb = Fy.F[static_cast<size_t>(j - jlo) * Fy.L + i];
                        const State& ft = Fy.F[static_cast<size_t>(j - jlo + 1) * Fy.L + i];
                        const State& ff = Fz.F[zk + static_cast<size_t>(j - jlo) * Fz.L + i];
                        const State& fk = Fz.F[zk1 + static_cast<size_t>(j - jlo) * Fz.L + i];
                        State& r = R[m.cell(i, j, k)];
                        for (int q = 0; q < kNVar; ++q)
                            r[q] = (((0.0 - (fr[q] * Ax - fl[q] * Ax)) - (ft[q] * Ay - fb[q] * Ay)) - (fk[q] * Az - ff[q] * Az)) * inv_vol;
                    }
            }
        }
#pragma omp critical(fv_fourth_residual_max)
        if (local > smax_all) smax_all = local;
    }
    return smax_all;
}

double fourth_order_fluxes(const RiemannSolver& riemann, const EOS& eos, const Field& UL, const Field& UR, const Mesh& m, const Halos& h,
                           const bool periodic[3], Field& F) {
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const int dims[3] = {nx, ny, nz};
    const int nax = nz > 1 ? 3 : 2;
    const bool lo[3] = {h.l != nullptr, h.b != nullptr, h.f != nullptr}, hi[3] = {h.r != nullptr, h.t != nullptr, h.bk != nullptr};
    const int nf = m.num_faces();
    const double sb = riemann.shock_blend();
    // p = g1 * rho e - gp
    const double g1 = eos.gamma - 1.0, gp = eos.type == EOS::Type::IdealGas ? 0.0 : eos.gamma * eos.p_inf;
    const auto m2 = [](const State& u) { return u[1] * u[1] + u[2] * u[2] + u[4] * u[4]; };
    // positive density and pressure without a division (rho > 0: p > 0 <=> g1 (E rho - m^2 / 2) > gp rho)
    const auto positive = [&](const State& u) { return u[0] > 0.0 && g1 * (u[3] * u[0] - 0.5 * m2(u)) > gp * u[0]; };
    const auto pressure = [&](const State& u) { return g1 * (u[3] - 0.5 * m2(u) / u[0]) - gp; };
    thread_local Field a_l, a_r, f_p;
    thread_local std::vector<unsigned char> st_good, st_pt, st_full;
    Field &AL = a_l, &AR = a_r, &Fp = f_p;
    std::vector<unsigned char> &good = st_good, &pt = st_pt, &full = st_full;
    // The work arrays keep their size between calls and every face of every family is written below, inside the parallel loops (copy of the
    // averaged states, flags cleared): a serial assign here was a fifth of the time of a 16-thread run and placed the pages on one thread.
    if (AL.size() != static_cast<size_t>(nf)) AL.resize(static_cast<size_t>(nf));
    if (AR.size() != static_cast<size_t>(nf)) AR.resize(static_cast<size_t>(nf));
    good.resize(static_cast<size_t>(nf));
    pt.resize(static_cast<size_t>(nf));
    full.resize(static_cast<size_t>(nf));
    // Every face of every family, one line at a time: fn(a, f0, e0, c) is called per line (f0 the first face, c[1], c[2] its lattice position)
    // and the line loop runs over c0 = 0 .. e0 - 1. Neighbours along the transverse axis t are f -+ stride, wrapped when periodic.
    for (int a = 0; a < nax; ++a) {
        const int e[3] = {nx + (a == 0), ny + (a == 1), nz + (a == 2)};
        const int fbase = a == 0 ? m.xface(0, 0, 0) : (a == 1 ? m.yface(0, 0, 0) : m.zface(0, 0, 0));
        const int nlines = e[1] * e[2];
        const auto lattice = [&](int line, int i, int* c) { c[0] = i; c[1] = line % e[1]; c[2] = line / e[1]; };
        // good: a full stencil, positive states, the shock sensor quiet
#pragma omp parallel for schedule(static) if (nf >= kOmpMinWork)
        for (int line = 0; line < nlines; ++line) {
            int c[3];
            for (int i = 0; i < e[0]; ++i) {
                lattice(line, i, c);
                const int f = fbase + line * e[0] + i;
                good[f] = 0;
                const int ca = c[a], n_a = dims[a];
                if (!((ca > 0 && ca < n_a) || (ca == 0 && lo[a]) || (ca == n_a && hi[a]))) continue;
                if (!positive(UL[f]) || !positive(UR[f])) continue;
                if (sb > 0.0) {
                    const double pL = pressure(UL[f]), pR = pressure(UR[f]);
                    if (std::abs(pR - pL) > sb * std::min(pL, pR)) continue;
                }
                good[f] = 1;
            }
        }
    }
    for (int a = 0; a < nax; ++a) {
        const int e[3] = {nx + (a == 0), ny + (a == 1), nz + (a == 2)};
        const long long st[3] = {1, e[0], static_cast<long long>(e[0]) * e[1]};
        const int fbase = a == 0 ? m.xface(0, 0, 0) : (a == 1 ? m.yface(0, 0, 0) : m.zface(0, 0, 0));
        const int nlines = e[1] * e[2];
        const auto nbr = [&](int f, const int* c, int t, int s) -> int {
            const int ct = c[t] + s;
            if (ct >= 0 && ct < dims[t]) return f + s * static_cast<int>(st[t]);
            if (!periodic[t]) return -1;
            return f - s * (dims[t] - 1) * static_cast<int>(st[t]);
        };
        // point values where the faces at +-1 along every transverse axis are good too
#pragma omp parallel for schedule(static) if (nf >= kOmpMinWork)
        for (int line = 0; line < nlines; ++line) {
            int c[3];
            c[1] = line % e[1];
            c[2] = line / e[1];
            for (int i = 0; i < e[0]; ++i) {
                const int f = fbase + line * e[0] + i;
                AL[f] = UL[f];
                AR[f] = UR[f];
                pt[f] = 0;
                if (!good[f]) continue;
                c[0] = i;
                int nb[2][2];
                int cnt = 0;
                bool ok = true;
                for (int t = 0; t < nax && ok; ++t) {
                    if (t == a) continue;
                    nb[cnt][0] = nbr(f, c, t, -1);
                    nb[cnt][1] = nbr(f, c, t, 1);
                    ok = nb[cnt][0] >= 0 && nb[cnt][1] >= 0 && good[nb[cnt][0]] && good[nb[cnt][1]];
                    ++cnt;
                }
                if (!ok) continue;
                State pl = UL[f], pr = UR[f];
                for (int q = 0; q < kNVar; ++q) {
                    double dl = 0.0, dr = 0.0;
                    for (int u = 0; u < cnt; ++u) {
                        dl += UL[nb[u][1]][q] - 2.0 * UL[f][q] + UL[nb[u][0]][q];
                        dr += UR[nb[u][1]][q] - 2.0 * UR[f][q] + UR[nb[u][0]][q];
                    }
                    pl[q] -= dl / 24.0;
                    pr[q] -= dr / 24.0;
                }
                if (!positive(pl) || !positive(pr)) continue;
                AL[f] = pl;
                AR[f] = pr;
                pt[f] = 1;
            }
        }
    }
    for (int a = 0; a < nax; ++a) {
        const int e[3] = {nx + (a == 0), ny + (a == 1), nz + (a == 2)};
        const long long st[3] = {1, e[0], static_cast<long long>(e[0]) * e[1]};
        const int fbase = a == 0 ? m.xface(0, 0, 0) : (a == 1 ? m.yface(0, 0, 0) : m.zface(0, 0, 0));
        const int nlines = e[1] * e[2];
        const auto nbr = [&](int f, const int* c, int t, int s) -> int {
            const int ct = c[t] + s;
            if (ct >= 0 && ct < dims[t]) return f + s * static_cast<int>(st[t]);
            if (!periodic[t]) return -1;
            return f - s * (dims[t] - 1) * static_cast<int>(st[t]);
        };
#pragma omp parallel for schedule(static) if (nf >= kOmpMinWork)
        for (int line = 0; line < nlines; ++line) {
            int c[3];
            c[1] = line % e[1];
            c[2] = line / e[1];
            for (int i = 0; i < e[0]; ++i) {
                const int f = fbase + line * e[0] + i;
                full[f] = 0;
                if (!pt[f]) continue;
                c[0] = i;
                bool ok = true;
                for (int t = 0; t < nax && ok; ++t) {
                    if (t == a) continue;
                    const int g0 = nbr(f, c, t, -1), g1n = nbr(f, c, t, 1);
                    ok = g0 >= 0 && g1n >= 0 && pt[g0] && pt[g1n];
                }
                full[f] = ok ? 1 : 0;
            }
        }
    }
    const double smax = riemann.compute_flux(AL, AR, m, Fp);   // the flux of the point values where there are any, else of the averages
    F.resize(static_cast<size_t>(nf));
    for (int a = 0; a < nax; ++a) {
        const int e[3] = {nx + (a == 0), ny + (a == 1), nz + (a == 2)};
        const long long st[3] = {1, e[0], static_cast<long long>(e[0]) * e[1]};
        const int fbase = a == 0 ? m.xface(0, 0, 0) : (a == 1 ? m.yface(0, 0, 0) : m.zface(0, 0, 0));
        const int nlines = e[1] * e[2];
        const auto nbr = [&](int f, const int* c, int t, int s) -> int {
            const int ct = c[t] + s;
            if (ct >= 0 && ct < dims[t]) return f + s * static_cast<int>(st[t]);
            if (!periodic[t]) return -1;
            return f - s * (dims[t] - 1) * static_cast<int>(st[t]);
        };
#pragma omp parallel for schedule(static) if (nf >= kOmpMinWork)
        for (int line = 0; line < nlines; ++line) {
            int c[3];
            c[1] = line % e[1];
            c[2] = line / e[1];
            for (int i = 0; i < e[0]; ++i) {
                const int f = fbase + line * e[0] + i;
                c[0] = i;
                if (full[f]) {
                    int nb[2][2];
                    int cnt = 0;
                    for (int t = 0; t < nax; ++t) {
                        if (t == a) continue;
                        nb[cnt][0] = nbr(f, c, t, -1);
                        nb[cnt][1] = nbr(f, c, t, 1);
                        ++cnt;
                    }
                    for (int q = 0; q < kNVar; ++q) {
                        double d = 0.0;
                        for (int u = 0; u < cnt; ++u) d += Fp[nb[u][1]][q] - 2.0 * Fp[f][q] + Fp[nb[u][0]][q];
                        F[f][q] = Fp[f][q] + d / 24.0;
                    }
                } else if (pt[f]) {   // a point value without the full correction: back to the flux of the averaged states
                    double sm = 0.0;
                    F[f] = riemann.face_flux(UL[f], UR[f], a, sm);
                } else {
                    F[f] = Fp[f];
                }
            }
        }
    }
    return smax;
}

}   // namespace fv
