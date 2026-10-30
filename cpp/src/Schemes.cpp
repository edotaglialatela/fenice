#include "fv/Switches.hpp"
#include "fv/Schemes.hpp"

#include "SchemesSimd.hpp"

#include <cctype>
#include <cstdlib>
#include <stdexcept>

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

// ---------------------------------------------------------------------------
// HLLC (Toro et al., 1994) with Einfeldt or Weiss-Smith wave-speed bounds
// ---------------------------------------------------------------------------

State RiemannSolver::hllc(const State& UL, const State& UR, int axis, double& smax) const {
    const State WL = eos_.cons_to_prim(UL);
    const State WR = eos_.cons_to_prim(UR);
    const double rho_L = WL[0], p_L = WL[3];
    const double rho_R = WR[0], p_R = WR[3];
    const double vel_L[3] = {WL[1], WL[2], WL[4]};
    const double vel_R[3] = {WR[1], WR[2], WR[4]};
    const double E_L = UL[3] / rho_L, E_R = UR[3] / rho_R;

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

    const double c_L = eos_.sound_speed(p_L, rho_L);
    const double c_R = eos_.sound_speed(p_R, rho_R);

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
        if (done > 0) detail::riemann_batch_avx2(eos_, type_, ma_cut_, UL, UR, axis, done, F, smax);
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
        if (done > 0) detail::muscl_riemann_batch_avx2(eos_, type_, ma_cut_, lim, W, st, axis, done, F, smax);
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

void MusclReconstruction::reconstruct(Field& UL, Field& UR, const Field& U, int nx, int ny, const Halos& h) const {
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
}   // namespace fv
