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
                    int pad) {
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
        if (detail::simd_available() && jj >= pad && jj < pad + ny && kk >= zp && kk < zp + nz) {
            iv = nx & ~3;
            double* o[kNVar];
            for (int q = 0; q < kNVar; ++q) o[q] = Wp.data() + q * plane + P(pad, jj, kk);
            detail::cons_to_prim_planes_simd(eos, &U[m.cell(0, j, k)], iv, o);
        }
#endif
        for (int ii = 0; ii < PX; ++ii) {
            if (ii >= pad && ii < pad + iv) continue;
            put(P(ii, jj, kk), eos.cons_to_prim(U[m.cell(std::min(std::max(ii - pad, 0), nx - 1), j, k)]));
        }
    }
    for (int k = 0; k < nz; ++k)
        for (int d = 0; d < pad; ++d) {
            if (h.b)
                for (int i = 0; i < nx; ++i) put(P(i + pad, d, k + zp), eos.cons_to_prim((*h.b)[(d * nz + k) * nx + i]));
            if (h.t)
                for (int i = 0; i < nx; ++i)
                    put(P(i + pad, ny + pad + d, k + zp), eos.cons_to_prim((*h.t)[(d * nz + k) * nx + i]));
            if (h.l)
                for (int j = 0; j < ny; ++j) put(P(d, j + pad, k + zp), eos.cons_to_prim((*h.l)[(k * ny + j) * pad + d]));
            if (h.r)
                for (int j = 0; j < ny; ++j)
                    put(P(nx + pad + d, j + pad, k + zp), eos.cons_to_prim((*h.r)[(k * ny + j) * pad + d]));
        }
    if (three_d)
        for (int d = 0; d < pad; ++d)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    if (h.f) put(P(i + pad, j + pad, d), eos.cons_to_prim((*h.f)[(d * ny + j) * nx + i]));
                    if (h.bk) put(P(i + pad, j + pad, nz + pad + d), eos.cons_to_prim((*h.bk)[(d * ny + j) * nx + i]));
                }
}

// The grid description of a reconstruction `scheme` (the padded primitives are `pad` cells deep: 3 for WENO5-Z, else 2). With WENO5-Z and
// an immersed body the solid cells go on the padded layout (a cell outside the domain is solid if the cell it replicates or the cell a
// periodic wrap puts there is), as reconstruct_weno does.
FusedGrid make_grid(const RiemannSolver& riemann, const EOS& eos, Limiter lim, Reconstruction scheme, double weno_eps,
                    const std::vector<double>& Wp, std::vector<char>& S_pad, const Field& UL, const Field& UR, const Mesh& m,
                    const Halos& h, int PX, int PY, int zp, int pad) {
    FusedGrid g{};
    g.weno = scheme == Reconstruction::Weno5 ? 5 : (scheme == Reconstruction::Weno3 ? 3 : 0);
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
}   // namespace fv
