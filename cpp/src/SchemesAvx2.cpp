// AVX2 batch of the Riemann solvers: four faces per call of the kernel.
//
// Bit-identity with the scalar solver (RiemannSolver::face_flux in Schemes.cpp) is the contract.  Every lane runs
// the same IEEE operations in the same order as the scalar code: only add / sub / mul / div / sqrt and bit logic are used
// (no fused multiply-add, no vector min / max, whose results differ from std::min / std::max on equal values and NaN: those are
// written as a compare and a blend), and every branch of the scalar code is computed for all lanes and selected by a mask.
// This file must be compiled without FMA contraction (the project builds with /fp:precise, resp. -ffp-contract=off, and the
// GCC / Clang build of this file enables AVX2 only, not FMA).
#include "SchemesSimd.hpp"

#ifdef FV_SIMD_AVX2

#include <immintrin.h>

#include <cstdlib>

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace fv {
namespace detail {

namespace {

using V = __m256d;

constexpr double kDenomEps = 1e-14;   // as in Schemes.cpp
constexpr double kMachEps = 1e-12;

inline V set1(double x) { return _mm256_set1_pd(x); }
inline V add(V a, V b) { return _mm256_add_pd(a, b); }
inline V sub(V a, V b) { return _mm256_sub_pd(a, b); }
inline V mul(V a, V b) { return _mm256_mul_pd(a, b); }
inline V dv(V a, V b) { return _mm256_div_pd(a, b); }
inline V vsqrt(V a) { return _mm256_sqrt_pd(a); }
inline V lt(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
inline V ge(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_GE_OQ); }
inline V band(V a, V b) { return _mm256_and_pd(a, b); }
inline V bor(V a, V b) { return _mm256_or_pd(a, b); }
// mask ? b : a, lane by lane
inline V blend(V a, V b, V mask) { return _mm256_blendv_pd(a, b, mask); }
// std::max(a, b) = (a < b) ? b : a   and   std::min(a, b) = (b < a) ? b : a
inline V vmax(V a, V b) { return blend(a, b, lt(a, b)); }
inline V vmin(V a, V b) { return blend(a, b, lt(b, a)); }
inline V vabs(V a) { return _mm256_andnot_pd(set1(-0.0), a); }
inline V vcopysign(V x, V y) {
    const V sign = set1(-0.0);
    return bor(_mm256_andnot_pd(sign, x), band(sign, y));
}

struct Vec5 {
    V q[5];
};

// Equation of state with its constants broadcast
struct VEos {
    bool ideal;
    V gamma, gm1, gpinf, pinf, eps, half;
    explicit VEos(const EOS& e)
        : ideal(e.type == EOS::Type::IdealGas), gamma(set1(e.gamma)), gm1(set1(e.gamma - 1.0)), gpinf(set1(e.gamma * e.p_inf)),
          pinf(set1(e.p_inf)), eps(set1(kEosEps)), half(set1(0.5)) {}

    // EOS::cons_to_prim, [rho, u, v, p, w]
    Vec5 cons_to_prim(const Vec5& U) const {
        Vec5 W;
        const V rho = vmax(U.q[0], eps);
        const V u = dv(U.q[1], rho), v = dv(U.q[2], rho), w = dv(U.q[4], rho);
        const V ke = mul(mul(half, rho), add(add(mul(u, u), mul(v, v)), mul(w, w)));
        V p;
        if (ideal)
            p = mul(gm1, sub(U.q[3], ke));
        else
            p = sub(mul(gm1, sub(U.q[3], ke)), gpinf);
        W.q[0] = rho;
        W.q[1] = u;
        W.q[2] = v;
        W.q[3] = vmax(p, eps);
        W.q[4] = w;
        return W;
    }
    // EOS::prim_to_cons
    Vec5 prim_to_cons(const Vec5& W) const {
        const V rho = W.q[0], u = W.q[1], v = W.q[2], p = W.q[3], w = W.q[4];
        const V ke = mul(mul(half, rho), add(add(mul(u, u), mul(v, v)), mul(w, w)));
        const V e = ideal ? dv(p, gm1) : dv(add(p, gpinf), gm1);
        Vec5 U;
        U.q[0] = rho;
        U.q[1] = mul(rho, u);
        U.q[2] = mul(rho, v);
        U.q[3] = add(e, ke);
        U.q[4] = mul(rho, w);
        return U;
    }
    // EOS::sound_speed
    V sound_speed(V p, V rho) const {
        if (ideal) return vsqrt(dv(mul(gamma, vmax(p, eps)), vmax(rho, eps)));
        return vsqrt(dv(mul(gamma, add(p, pinf)), vmax(rho, eps)));
    }
};

inline V safe_denom(V x) { return vcopysign(vmax(vabs(x), set1(kDenomEps)), x); }

struct AxisMasks {
    V m1, m2;   // lanes whose normal axis is 1 (y) / 2 (z); axis 0 is neither
    V select(V a0, V a1, V a2) const { return blend(blend(a0, a1, m1), a2, m2); }
};

// RiemannSolver::hllc for four faces
void hllc4(const VEos& E, RiemannType type, double ma_cut, const AxisMasks& ax, const Vec5& UL, const Vec5& UR, Vec5& F, V& smax) {
    const Vec5 WL = E.cons_to_prim(UL), WR = E.cons_to_prim(UR);
    const V rho_L = WL.q[0], p_L = WL.q[3], rho_R = WR.q[0], p_R = WR.q[3];
    const V velL[3] = {WL.q[1], WL.q[2], WL.q[4]}, velR[3] = {WR.q[1], WR.q[2], WR.q[4]};
    const V E_L = dv(UL.q[3], rho_L), E_R = dv(UR.q[3], rho_R);
    const V Vn_L = ax.select(velL[0], velL[1], velL[2]), Vn_R = ax.select(velR[0], velR[1], velR[2]);
    // lanes whose normal axis is a (axis 0 is neither 1 nor 2)
    const V all_ones = _mm256_cmp_pd(set1(0.0), set1(0.0), _CMP_EQ_OQ);
    const V is_axis[3] = {_mm256_andnot_pd(bor(ax.m1, ax.m2), all_ones), ax.m1, ax.m2};
    static const int kSlot[3] = {1, 2, 4};

    Vec5 FL, FR;
    FL.q[0] = mul(rho_L, Vn_L);
    FR.q[0] = mul(rho_R, Vn_R);
    for (int a = 0; a < 3; ++a) {
        const V fl = mul(mul(rho_L, velL[a]), Vn_L), fr = mul(mul(rho_R, velR[a]), Vn_R);
        FL.q[kSlot[a]] = blend(fl, add(fl, p_L), is_axis[a]);
        FR.q[kSlot[a]] = blend(fr, add(fr, p_R), is_axis[a]);
    }
    FL.q[3] = mul(Vn_L, add(mul(rho_L, E_L), p_L));
    FR.q[3] = mul(Vn_R, add(mul(rho_R, E_R), p_R));

    const V c_L = E.sound_speed(p_L, rho_L), c_R = E.sound_speed(p_R, rho_R);
    V S_L, S_R;
    if (type == RiemannType::PreconditionedHLLC) {
        const V cut = set1(ma_cut);
        const V one = set1(1.0), half = set1(0.5), deps = set1(kDenomEps);
        const auto ws = [&](V Vn, V c, V& lm, V& lp) {
            const V cc = mul(c, c);
            const V Vr = vmin(vmax(mul(Vn, Vn), mul(mul(cut, c), mul(cut, c))), cc);
            const V beta = dv(Vr, add(cc, deps));
            const V a = mul(mul(sub(one, beta), half), Vn);
            const V disc = vsqrt(add(mul(a, a), Vr));
            const V mu = mul(mul(add(one, beta), half), Vn);
            lm = sub(mu, disc);
            lp = add(mu, disc);
        };
        V lmL, lpL, lmR, lpR;
        ws(Vn_L, c_L, lmL, lpL);
        ws(Vn_R, c_R, lmR, lpR);
        S_L = vmin(lmL, lmR);
        S_R = vmax(lpL, lpR);
    } else {
        S_L = vmin(sub(Vn_L, c_L), sub(Vn_R, c_R));
        S_R = vmax(add(Vn_L, c_L), add(Vn_R, c_R));
    }

    // num = rho_R*Vn_R*(S_R - Vn_R) - rho_L*Vn_L*(S_L - Vn_L) + p_L - p_R   (left to right)
    const V numer = sub(add(sub(mul(mul(rho_R, Vn_R), sub(S_R, Vn_R)), mul(mul(rho_L, Vn_L), sub(S_L, Vn_L))), p_L), p_R);
    const V denom = sub(mul(rho_R, sub(S_R, Vn_R)), mul(rho_L, sub(S_L, Vn_L)));
    const V S_M = dv(numer, safe_denom(denom));

    const V den_star_L = safe_denom(sub(S_L, S_M)), den_star_R = safe_denom(sub(S_R, S_M));
    const V den_pL = safe_denom(sub(S_L, Vn_L)), den_pR = safe_denom(sub(S_R, Vn_R));
    const V fL = dv(mul(rho_L, sub(S_L, Vn_L)), den_star_L), fR = dv(mul(rho_R, sub(S_R, Vn_R)), den_star_R);
    Vec5 UsL, UsR;
    UsL.q[0] = fL;
    UsR.q[0] = fR;
    for (int a = 0; a < 3; ++a) {
        UsL.q[kSlot[a]] = mul(fL, blend(velL[a], S_M, is_axis[a]));
        UsR.q[kSlot[a]] = mul(fR, blend(velR[a], S_M, is_axis[a]));
    }
    UsL.q[3] = mul(fL, add(E_L, mul(sub(S_M, Vn_L), add(S_M, dv(p_L, mul(rho_L, den_pL))))));
    UsR.q[3] = mul(fR, add(E_R, mul(sub(S_M, Vn_R), add(S_M, dv(p_R, mul(rho_R, den_pR))))));

    const V zero = set1(0.0);
    const V c1 = ge(S_L, zero);
    const V c2 = band(lt(S_L, zero), ge(S_M, zero));
    const V c3 = band(lt(S_M, zero), ge(S_R, zero));
    const V c4 = lt(S_R, zero);
    for (int q = 0; q < 5; ++q) {
        V f = zero;
        f = blend(f, FL.q[q], c1);
        f = blend(f, add(FL.q[q], mul(S_L, sub(UsL.q[q], UL.q[q]))), c2);
        f = blend(f, add(FR.q[q], mul(S_R, sub(UsR.q[q], UR.q[q]))), c3);
        f = blend(f, FR.q[q], c4);
        F.q[q] = f;
    }
    smax = vmax(vabs(S_L), vabs(S_R));
}

Vec5 load4(const State* base) {
    Vec5 U;
    for (int q = 0; q < 5; ++q) U.q[q] = _mm256_set_pd(base[3][q], base[2][q], base[1][q], base[0][q]);
    return U;
}

void store4(State* base, const Vec5& F) {
    alignas(32) double t[4];
    for (int q = 0; q < 5; ++q) {
        _mm256_store_pd(t, F.q[q]);
        for (int i = 0; i < 4; ++i) base[i][q] = t[i];
    }
}

}   // namespace

bool avx2_available() {
    static const bool ok = []() {
        if (std::getenv("FENICE_NO_SIMD")) return false;
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

namespace {

// Masks of the lanes whose axis is 1 and 2
AxisMasks axis_masks(const int* axis) {
    const __m128i a4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(axis));
    const __m256i a64 = _mm256_cvtepi32_epi64(a4);
    AxisMasks ax;
    ax.m1 = _mm256_castsi256_pd(_mm256_cmpeq_epi64(a64, _mm256_set1_epi64x(1)));
    ax.m2 = _mm256_castsi256_pd(_mm256_cmpeq_epi64(a64, _mm256_set1_epi64x(2)));
    return ax;
}

// RiemannSolver::face_flux for four faces given their left / right conservative states
void flux4(const VEos& E, RiemannType type, double ma_cut, const AxisMasks& ax, Vec5 L, Vec5 R, Vec5& Fk, V& sm) {
    if (type == RiemannType::LowMachHLLC) {
        // Thornber low-Mach scaling of the velocity jump, then standard HLLC
        Vec5 WL = E.cons_to_prim(L), WR = E.cons_to_prim(R);
        const V c_L = E.sound_speed(WL.q[3], WL.q[0]), c_R = E.sound_speed(WR.q[3], WR.q[0]);
        const V meps = set1(kMachEps);
        const V M_L = dv(vsqrt(add(add(mul(WL.q[1], WL.q[1]), mul(WL.q[2], WL.q[2])), mul(WL.q[4], WL.q[4]))), add(c_L, meps));
        const V M_R = dv(vsqrt(add(add(mul(WR.q[1], WR.q[1]), mul(WR.q[2], WR.q[2])), mul(WR.q[4], WR.q[4]))), add(c_R, meps));
        const V z = vmin(vmax(vmax(M_L, M_R), set1(1e-4)), set1(1.0));
        const V half = set1(0.5);
        for (int m : {1, 2, 4}) {
            const V avg = mul(half, add(WL.q[m], WR.q[m]));
            const V hf = mul(half, sub(WL.q[m], WR.q[m]));
            WL.q[m] = add(avg, mul(z, hf));
            WR.q[m] = sub(avg, mul(z, hf));
        }
        L = E.prim_to_cons(WL);
        R = E.prim_to_cons(WR);
    }
    hllc4(E, type, ma_cut, ax, L, R, Fk, sm);
}

void store_results(State* F, double* smax, int k, const Vec5& Fk, V sm) {
    store4(F + k, Fk);
    alignas(32) double t[4];
    _mm256_store_pd(t, sm);
    for (int i = 0; i < 4; ++i) smax[k + i] = t[i];
}

// sgn(x) of Schemes.hpp: 1 for x > 0, -1 for x < 0, 0 otherwise
inline V vsgn(V x) {
    const V zero = set1(0.0);
    return blend(blend(zero, set1(-1.0), lt(x, zero)), set1(1.0), lt(zero, x));
}

// apply_limiter of Schemes.hpp
inline V vlimiter(Limiter l, V a, V b) {
    const V zero = set1(0.0), two = set1(2.0), half = set1(0.5);
    switch (l) {
    case Limiter::Minmod: return mul(mul(half, add(vsgn(a), vsgn(b))), vmin(vabs(a), vabs(b)));
    case Limiter::VanLeer: {
        const V ab = mul(a, b);
        return blend(zero, dv(mul(two, ab), add(a, b)), lt(zero, ab));
    }
    case Limiter::MC: {
        const V ab = mul(a, b);
        const V v = mul(vmin(vmin(mul(two, vabs(a)), mul(half, vabs(add(a, b)))), mul(two, vabs(b))), vsgn(a));
        return blend(zero, v, lt(zero, ab));
    }
    default: {   // Koren; fmin of positive finite numbers is the smaller of the two
        const V ab = mul(a, b);
        const V v = mul(vmin(vmin(mul(two, vabs(a)), dv(vabs(add(mul(two, a), b)), set1(3.0))), mul(two, vabs(b))), vsgn(a));
        return blend(zero, v, lt(zero, ab));
    }
    }
}

// Component q of the four stencil states of four faces: lanes = faces
inline V gather_q(const State* W, const int* st, int k, int point, int q) {
    return _mm256_set_pd(W[st[4 * (k + 3) + point]][q], W[st[4 * (k + 2) + point]][q], W[st[4 * (k + 1) + point]][q],
                         W[st[4 * k + point]][q]);
}

}   // namespace

void riemann_batch_avx2(const EOS& eos, RiemannType type, double ma_cut, const State* UL, const State* UR, const int* axis, int n,
                        State* F, double* smax) {
    const VEos E(eos);
    for (int k = 0; k < n; k += 4) {
        Vec5 Fk;
        V sm;
        flux4(E, type, ma_cut, axis_masks(axis + k), load4(UL + k), load4(UR + k), Fk, sm);
        store_results(F, smax, k, Fk, sm);
    }
    _mm256_zeroupper();
}

void muscl_riemann_batch_avx2(const EOS& eos, RiemannType type, double ma_cut, Limiter lim, const State* W, const int* st,
                              const int* axis, int n, State* F, double* smax) {
    const VEos E(eos);
    const V half = set1(0.5);
    for (int k = 0; k < n; k += 4) {
        // MUSCL: wl = c1 + 0.5 phi(c1 - c0, c2 - c1), wr = c2 - 0.5 phi(c3 - c2, c2 - c1) for each primitive variable
        Vec5 wl, wr;
        for (int q = 0; q < kNVar; ++q) {
            const V c0 = gather_q(W, st, k, 0, q), c1 = gather_q(W, st, k, 1, q), c2 = gather_q(W, st, k, 2, q),
                    c3 = gather_q(W, st, k, 3, q);
            const V bq = sub(c2, c1);
            wl.q[q] = add(c1, mul(half, vlimiter(lim, sub(c1, c0), bq)));
            wr.q[q] = sub(c2, mul(half, vlimiter(lim, sub(c3, c2), bq)));
        }
        Vec5 Fk;
        V sm;
        flux4(E, type, ma_cut, axis_masks(axis + k), E.prim_to_cons(wl), E.prim_to_cons(wr), Fk, sm);
        store_results(F, smax, k, Fk, sm);
    }
    _mm256_zeroupper();
}

}   // namespace detail
}   // namespace fv

#endif   // FV_SIMD_AVX2
