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

#ifdef FV_SIMD_AVX2
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace fv {

#ifdef FV_SIMD_AVX2
namespace detail {

// Here, not in SchemesAvx2.cpp: that file is compiled for AVX2, and the check must run on CPUs without it.
bool avx2_available() {
    static const bool ok = []() {
        if (switch_on("FENICE_NO_SIMD")) return false;
#if defined(_MSC_VER)
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
        return RiemannType::PreconditionedHLLC;
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
// HLLC (Toro et al., 1994) with Einfeldt or Weiss-Smith wave-speed bounds
// ---------------------------------------------------------------------------

State RiemannSolver::hllc(const State& UL, const State& UR, int axis, double& smax) const {
    double inv_L, inv_R;
    const State WL = eos_.cons_to_prim(UL, &inv_L);
    const State WR = eos_.cons_to_prim(UR, &inv_R);
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

    const double c_L = eos_.sound_speed_inv(p_L, inv_L);
    const double c_R = eos_.sound_speed_inv(p_R, inv_R);

    double S_L, S_R;
    if (type_ == RiemannType::PreconditionedHLLC) {
        // Weiss-Smith (1995) preconditioned eigenvalues
        auto ws = [&](double Vn, double c, double& lm, double& lp) {
            const double Vr = std::min(std::max(Vn * Vn, (ma_cut_ * c) * (ma_cut_ * c)), c * c);
            const double beta = Vr / (c * c + kDenomEps);
            const double a = (1.0 - beta) * 0.5 * Vn;
            const double disc = std::sqrt(a * a + Vr);
            const double mu = (1.0 + beta) * 0.5 * Vn;
            lm = mu - disc;
            lp = mu + disc;
        };
        double lmL, lpL, lmR, lpR;
        ws(Vn_L, c_L, lmL, lpL);
        ws(Vn_R, c_R, lmR, lpR);
        S_L = std::min(lmL, lmR);
        S_R = std::max(lpL, lpR);
    } else {
        // Einfeldt bounds
        S_L = std::min(Vn_L - c_L, Vn_R - c_R);
        S_R = std::max(Vn_L + c_L, Vn_R + c_R);
    }

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

    // Thornber (2008) low-Mach velocity-jump scaling, then standard HLLC
    State WL = eos_.cons_to_prim(UL);
    State WR = eos_.cons_to_prim(UR);
    const double c_L = eos_.sound_speed(WL[3], WL[0]);
    const double c_R = eos_.sound_speed(WR[3], WR[0]);
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
    return hllc(eos_.prim_to_cons(WL), eos_.prim_to_cons(WR), axis, smax);
}

void RiemannSolver::face_flux_batch(const State* UL, const State* UR, const int* axis, int n, State* F, double* smax) const {
    int done = 0;
#ifdef FV_SIMD_AVX2
    if (detail::avx2_available()) {
        done = n & ~3;
        if (done > 0) detail::riemann_batch_avx2(eos_, type_, ma_cut_, shock_blend_, UL, UR, axis, done, F, smax);
    }
#endif
    for (int k = done; k < n; ++k) F[k] = face_flux(UL[k], UR[k], axis[k], smax[k]);
}

void RiemannSolver::muscl_face_flux_batch(Limiter lim, const State* W, const int* st, const int* axis, int n, State* F,
                                          double* smax) const {
    int done = 0;
#ifdef FV_SIMD_AVX2
    if (detail::avx2_available()) {
        done = n & ~3;
        if (done > 0) detail::muscl_riemann_batch_avx2(eos_, type_, ma_cut_, shock_blend_, lim, W, st, axis, done, F, smax);
    }
#endif
    for (int k = done; k < n; ++k) {
        const State &w0 = W[st[4 * k]], &w1 = W[st[4 * k + 1]], &w2 = W[st[4 * k + 2]], &w3 = W[st[4 * k + 3]];
        State wl, wr;
        for (int q = 0; q < kNVar; ++q) {
            const double c0 = w0[q], c1 = w1[q], c2 = w2[q], c3 = w3[q];
            const double bq = c2 - c1;
            wl[q] = c1 + 0.5 * apply_limiter(lim, c1 - c0, bq);
            wr[q] = c2 - 0.5 * apply_limiter(lim, c3 - c2, bq);
        }
        F[k] = face_flux(eos_.prim_to_cons(wl), eos_.prim_to_cons(wr), axis[k], smax[k]);
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

void MusclReconstruction::reconstruct_weno5(Field& UL, Field& UR, const Field& U, int nx, int ny, int nz, const Halos& h) const {
    constexpr int pad = 3;
    const int padz = nz > 1 ? pad : 0;
    const int PX = nx + 2 * pad, PY = ny + 2 * pad, PZ = nz + 2 * padz;
    Mesh m;
    m.nx = nx;
    m.ny = ny;
    m.nz = nz;
    // a side with a halo (periodic wrap or ghost layers of a physical boundary) reads it, the others replicate the edge cell
    if ((h.l || h.r || h.b || h.t || h.f || h.bk) && h.depth < pad)
        throw std::logic_error("WENO5-Z needs halos three cells deep");
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
    thread_local Field W_pad;
    W_pad.resize(static_cast<size_t>(PZ) * PY * PX);
    Field& Wp = W_pad;
    const auto P = [&](int ii, int jj, int kk) -> size_t { return (static_cast<size_t>(kk) * PY + jj) * PX + ii; };
#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
    for (int kk = 0; kk < PZ; ++kk) {
        const int k = kk - padz;
        for (int jj = 0; jj < PY; ++jj) {
            const int j = jj - pad;
            for (int ii = 0; ii < PX; ++ii) {
                const int i = ii - pad;
                // a cell outside the domain along two axes (a corner) is not on any stencil: edge replication
                const int out = (i < 0 || i >= nx) + (j < 0 || j >= ny) + (k < 0 || k >= nz);
                Wp[P(ii, jj, kk)] = eos_.cons_to_prim(out > 1 ? U[m.cell(clampi(i, nx), clampi(j, ny), clampi(k, nz))] : cell_state(i, j, k));
            }
        }
    }
    const EOS& eos = eos_;
    const double we = weno_eps_;
    const auto roundtrip = [&](State& u) { u = eos.prim_to_cons(eos.cons_to_prim(u)); };
    // Faces normal to `axis`: face t lies between cells t-1 and t; its six cells are padded positions t .. t+5.
    const auto sweep = [&](int axis, bool has_lo, bool has_hi) {
        const int n_a = axis == 0 ? nx : (axis == 1 ? ny : nz);
        const size_t stride = axis == 0 ? 1 : (axis == 1 ? static_cast<size_t>(PX) : static_cast<size_t>(PX) * PY);
        const int ni = axis == 0 ? nx + 1 : nx, nj = axis == 1 ? ny + 1 : ny, nk = axis == 2 ? nz + 1 : nz;
#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
        for (int kj = 0; kj < nk * nj; ++kj) {
            const int j = kj % nj, k = kj / nj;
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
                for (int q = 0; q < kNVar; ++q) {
                    const double c0 = Wp[base][q], c1 = Wp[base + stride][q], c2 = Wp[base + 2 * stride][q];
                    const double c3 = Wp[base + 3 * stride][q], c4 = Wp[base + 4 * stride][q], c5 = Wp[base + 5 * stride][q];
                    wl[q] = weno5z(c0, c1, c2, c3, c4, we);
                    wr[q] = weno5z(c5, c4, c3, c2, c1, we);
                }
                UL[f] = eos.prim_to_cons(wl);
                UR[f] = eos.prim_to_cons(wr);
            }
        }
    };
    sweep(0, h.l, h.r);
    sweep(1, h.b, h.t);
    if (nz > 1) sweep(2, h.f, h.bk);
}

void MusclReconstruction::reconstruct(Field& UL, Field& UR, const Field& U, int nx, int ny, const Halos& h) const {
    if (scheme_ == Reconstruction::Weno5) return reconstruct_weno5(UL, UR, U, nx, ny, 1, h);
    const bool weno3 = scheme_ == Reconstruction::Weno3;
    const double we = weno_eps_;
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
                if (weno3) {
                    wl[q] = weno3z(c0, c1, c2, we);
                    wr[q] = weno3z(c3, c2, c1, we);
                    continue;
                }
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
                if (weno3) {
                    wl[q] = weno3z(r0, r1, r2, we);
                    wr[q] = weno3z(r3, r2, r1, we);
                    continue;
                }
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
    if (scheme_ == Reconstruction::Weno5) return reconstruct_weno5(UL, UR, U, m.nx, m.ny, m.nz, h);
    const bool weno3 = scheme_ == Reconstruction::Weno3;
    const double we = weno_eps_;
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
                    if (weno3) {
                        wl[q] = weno3z(c0, c1, c2, we);
                        wr[q] = weno3z(c3, c2, c1, we);
                        continue;
                    }
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
    const State* Wp;       // padded primitive state (MUSCL); null for the first-order scheme
    const Field* U;        // cell states (first order)
    const Field* UL;
    const Field* UR;
    const Mesh* m;
    int nx, ny, nz, PX, PY, pad, zp;
    bool halo_lo[3], halo_hi[3];   // a halo on the low / high side of the axis
    size_t P(int ii, int jj, int kk) const { return (static_cast<size_t>(kk) * PY + jj) * PX + ii; }
};

constexpr int kFusedChunk = 256;
struct FusedScratch {
    int st[4 * kFusedChunk], ax[kFusedChunk];
    double sm[kFusedChunk];
    State ul[kFusedChunk], ur[kFusedChunk];   // first order: the states on both sides of the faces of a chunk
};

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
        for (int c = 0; c < cnt; ++c) {
            const int i = i0 + c;
            const size_t base = axis == 0 ? g.P(i, j + pad, k + zp)
                                          : (axis == 1 ? g.P(i + pad, t_line, k + zp) : g.P(i + pad, j + pad, t_line));
            s.st[4 * c] = static_cast<int>(base);
            s.st[4 * c + 1] = static_cast<int>(base + stride);
            s.st[4 * c + 2] = static_cast<int>(base + 2 * stride);
            s.st[4 * c + 3] = static_cast<int>(base + 3 * stride);
            s.ax[c] = axis;
        }
        g.riemann->muscl_face_flux_batch(g.lim, g.Wp, s.st, s.ax, cnt, F_line + i0, s.sm);
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

// The primitive state padded by kHaloDepth cells (edge replication; halos replace it where present). The layouts of Halos reduce to
// the 2-D ones when nz = 1.
void pad_primitives(const EOS& eos, const Field& U, const Mesh& m, const Halos& h, Field& Wp, int& PX, int& PY, int& zp) {
    constexpr int pad = kHaloDepth;
    const int nx = m.nx, ny = m.ny, nz = m.nz;
    const bool three_d = nz > 1;
    zp = three_d ? pad : 0;   // padding along z (none in 2-D)
    PX = nx + 2 * pad;
    PY = ny + 2 * pad;
    const int PZ = nz + 2 * zp;
    Wp.resize(static_cast<size_t>(PZ) * PY * PX);
    const auto P = [&](int ii, int jj, int kk) -> size_t { return (static_cast<size_t>(kk) * PY + jj) * PX + ii; };
#pragma omp parallel for schedule(static) if (nx * ny * nz >= kOmpMinWork)
    for (int kj = 0; kj < PZ * PY; ++kj) {   // rows of the padded array (in 2-D, PZ = 1: the rows themselves)
        const int kk = kj / PY, jj = kj % PY;
        const int k = std::min(std::max(kk - zp, 0), nz - 1);
        const int j = std::min(std::max(jj - pad, 0), ny - 1);
        for (int ii = 0; ii < PX; ++ii)
            Wp[P(ii, jj, kk)] = eos.cons_to_prim(U[m.cell(std::min(std::max(ii - pad, 0), nx - 1), j, k)]);
    }
    for (int k = 0; k < nz; ++k)
        for (int d = 0; d < pad; ++d) {
            if (h.b)
                for (int i = 0; i < nx; ++i) Wp[P(i + pad, d, k + zp)] = eos.cons_to_prim((*h.b)[(d * nz + k) * nx + i]);
            if (h.t)
                for (int i = 0; i < nx; ++i)
                    Wp[P(i + pad, ny + pad + d, k + zp)] = eos.cons_to_prim((*h.t)[(d * nz + k) * nx + i]);
            if (h.l)
                for (int j = 0; j < ny; ++j) Wp[P(d, j + pad, k + zp)] = eos.cons_to_prim((*h.l)[(k * ny + j) * pad + d]);
            if (h.r)
                for (int j = 0; j < ny; ++j)
                    Wp[P(nx + pad + d, j + pad, k + zp)] = eos.cons_to_prim((*h.r)[(k * ny + j) * pad + d]);
        }
    if (three_d)
        for (int d = 0; d < pad; ++d)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    if (h.f) Wp[P(i + pad, j + pad, d)] = eos.cons_to_prim((*h.f)[(d * ny + j) * nx + i]);
                    if (h.bk) Wp[P(i + pad, j + pad, nz + pad + d)] = eos.cons_to_prim((*h.bk)[(d * ny + j) * nx + i]);
                }
}

FusedGrid make_grid(const RiemannSolver& riemann, const EOS& eos, Limiter lim, const Field& Wp, const Field& UL, const Field& UR,
                    const Mesh& m, const Halos& h, int PX, int PY, int zp) {
    FusedGrid g{};
    g.riemann = &riemann;
    g.eos = &eos;
    g.lim = lim;
    g.Wp = Wp.data();
    g.U = nullptr;
    g.UL = &UL;
    g.UR = &UR;
    g.m = &m;
    g.nx = m.nx;
    g.ny = m.ny;
    g.nz = m.nz;
    g.PX = PX;
    g.PY = PY;
    g.pad = kHaloDepth;
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
    thread_local Field W_pad;
    int PX, PY, zp;
    pad_primitives(eos_, U, m, h, W_pad, PX, PY, zp);
    F.resize(m.num_faces());
    const FusedGrid g = make_grid(riemann, eos_, lim_, W_pad, UL, UR, m, h, PX, PY, zp);
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
    thread_local Field W_pad;
    int PX, PY, zp;
    pad_primitives(eos_, U, m, h, W_pad, PX, PY, zp);
    R.resize(U.size());
    return fused_residual_2d(make_grid(riemann, eos_, lim_, W_pad, UL, UR, m, h, PX, PY, zp), m, R);
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
    return fused_residual_2d(g, m, R);
}
}   // namespace fv
