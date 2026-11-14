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
inline V le(V a, V b) { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); }
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

    // EOS::cons_to_prim, [rho, u, v, p, w]; `inv_rho` receives 1 / rho
    Vec5 cons_to_prim(const Vec5& U, V* inv_rho = nullptr) const {
        Vec5 W;
        const V rho = vmax(U.q[0], eps);
        const V inv = dv(set1(1.0), rho);
        if (inv_rho) *inv_rho = inv;
        const V u = mul(U.q[1], inv), v = mul(U.q[2], inv), w = mul(U.q[4], inv);
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
    // EOS::sound_speed_inv
    V sound_speed_inv(V p, V inv_rho) const {
        if (ideal) return vsqrt(mul(mul(gamma, vmax(p, eps)), inv_rho));
        return vsqrt(mul(mul(gamma, add(p, pinf)), inv_rho));
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

// The two sides of four faces as the HLLC core reads them: primitive and conservative states (primitive floored as by cons_to_prim),
// 1 / rho and the sound speed
struct Face4 {
    Vec5 WL, WR, UL, UR;
    V inv_L, inv_R, c_L, c_R;
};

// RiemannSolver::hllc_core for four faces
void hllc4(const VEos& E, double sb, const AxisMasks& ax, const Face4& f, Vec5& F, V& smax) {
    const Vec5 &UL = f.UL, &UR = f.UR, &WL = f.WL, &WR = f.WR;
    const V inv_L = f.inv_L, inv_R = f.inv_R;
    const V rho_L = WL.q[0], p_L = WL.q[3], rho_R = WR.q[0], p_R = WR.q[3];
    const V velL[3] = {WL.q[1], WL.q[2], WL.q[4]}, velR[3] = {WR.q[1], WR.q[2], WR.q[4]};
    const V E_L = mul(UL.q[3], inv_L), E_R = mul(UR.q[3], inv_R);
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

    const V c_L = f.c_L, c_R = f.c_R;
    const V S_L = vmin(sub(Vn_L, c_L), sub(Vn_R, c_R));
    const V S_R = vmax(add(Vn_L, c_L), add(Vn_R, c_R));

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
    // Shock-face dissipation, as in RiemannSolver::hllc (same operations, same order)
    if (sb > 0.0) {
        const V vsb = set1(sb), one = set1(1.0);
        const V s = dv(vabs(sub(p_R, p_L)), vmin(p_L, p_R));
        const V on = lt(vsb, s);
        const V phi = vmin(dv(sub(s, vsb), vsb), one);
        const V inv = dv(one, sub(S_R, S_L));
        const V hl = ge(S_L, zero), hr = le(S_R, zero);
        const V SLR = mul(S_L, S_R);
        for (int q = 0; q < 5; ++q) {
            const V hll = mul(add(sub(mul(S_R, FL.q[q]), mul(S_L, FR.q[q])), mul(SLR, sub(UR.q[q], UL.q[q]))), inv);
            const V Fh = blend(blend(hll, FR.q[q], hr), FL.q[q], hl);
            const V Fn = add(mul(sub(one, phi), F.q[q]), mul(phi, Fh));
            F.q[q] = blend(F.q[q], Fn, on);
        }
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

// Entry of the Riemann solver from conservative states f.UL, f.UR (RiemannSolver::hllc / face_flux): primitive states, 1 / rho, sound speed
void entry_cons4(const VEos& E, Face4& f) {
    f.WL = E.cons_to_prim(f.UL, &f.inv_L);
    f.WR = E.cons_to_prim(f.UR, &f.inv_R);
    f.c_L = E.sound_speed_inv(f.WL.q[3], f.inv_L);
    f.c_R = E.sound_speed_inv(f.WR.q[3], f.inv_R);
}

// RiemannSolver::lowmach_flux before the HLLC core: Thornber scaling of the velocity jump of the primitive states, conservative states
// from the scaled ones (density, pressure, 1 / rho and the sound speed are not touched)
void lowmach4(const VEos& E, Face4& f) {
    Vec5 &WL = f.WL, &WR = f.WR;
    const V meps = set1(kMachEps);
    const V M_L = dv(vsqrt(add(add(mul(WL.q[1], WL.q[1]), mul(WL.q[2], WL.q[2])), mul(WL.q[4], WL.q[4]))), add(f.c_L, meps));
    const V M_R = dv(vsqrt(add(add(mul(WR.q[1], WR.q[1]), mul(WR.q[2], WR.q[2])), mul(WR.q[4], WR.q[4]))), add(f.c_R, meps));
    const V z = vmin(vmax(vmax(M_L, M_R), set1(1e-4)), set1(1.0));
    const V half = set1(0.5);
    for (int m : {1, 2, 4}) {
        const V avg = mul(half, add(WL.q[m], WR.q[m]));
        const V hf = mul(half, sub(WL.q[m], WR.q[m]));
        WL.q[m] = add(avg, mul(z, hf));
        WR.q[m] = sub(avg, mul(z, hf));
    }
    f.UL = E.prim_to_cons(WL);
    f.UR = E.prim_to_cons(WR);
}

// The faces are taken in blocks of kBlock batches of four and each step of a face's flux (reconstruction, low-Mach states, HLLC) is
// a loop over the block: the steps of one batch form a single chain of some 600 operations, longer than the window in which a
// processor reorders instructions, so batches run back to back did not overlap; in the short loops they do. Same operations per
// face, same bits.
constexpr int kBlock = 8;

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

void riemann_batch_avx2(const EOS& eos, RiemannType type, double sb, const State* UL, const State* UR, const int* axis, int n,
                        State* F, double* smax) {
    const VEos E(eos);
    Face4 fb[kBlock];
    for (int k0 = 0; k0 < n; k0 += 4 * kBlock) {
        const int nb = (n - k0) / 4 < kBlock ? (n - k0) / 4 : kBlock;
        for (int b = 0; b < nb; ++b) {
            fb[b].UL = load4(UL + k0 + 4 * b);
            fb[b].UR = load4(UR + k0 + 4 * b);
            entry_cons4(E, fb[b]);
            if (type == RiemannType::LowMachHLLC) lowmach4(E, fb[b]);
        }
        for (int b = 0; b < nb; ++b) {
            const int k = k0 + 4 * b;
            Vec5 Fk;
            V sm;
            hllc4(E, sb, axis_masks(axis + k), fb[b], Fk, sm);
            store_results(F, smax, k, Fk, sm);
        }
    }
    _mm256_zeroupper();
}

void muscl_riemann_batch_avx2(const EOS& eos, RiemannType type, double sb, Limiter lim, const State* W, const int* st,
                              const int* axis, int n, State* F, double* smax, bool planar) {
    const VEos E(eos);
    const V half = set1(0.5);
    Face4 fb[kBlock];
    for (int k0 = 0; k0 < n; k0 += 4 * kBlock) {
        const int nb = (n - k0) / 4 < kBlock ? (n - k0) / 4 : kBlock;
        for (int b = 0; b < nb; ++b) {
            // MUSCL: wl = c1 + 0.5 phi(c1 - c0, c2 - c1), wr = c2 - 0.5 phi(c3 - c2, c2 - c1) for each primitive variable
            const int k = k0 + 4 * b;
            Vec5 wl, wr;
            if (planar) wl.q[4] = wr.q[4] = set1(0.0);   // rho w = 0 in 2-D: the limited reconstruction of +0 is +0
            for (int q = 0; q < (planar ? 4 : kNVar); ++q) {
                const V c0 = gather_q(W, st, k, 0, q), c1 = gather_q(W, st, k, 1, q), c2 = gather_q(W, st, k, 2, q),
                        c3 = gather_q(W, st, k, 3, q);
                const V bq = sub(c2, c1);
                wl.q[q] = add(c1, mul(half, vlimiter(lim, sub(c1, c0), bq)));
                wr.q[q] = sub(c2, mul(half, vlimiter(lim, sub(c3, c2), bq)));
            }
            // the face states go to the solver as conservative variables, as in the two-pass residual
            fb[b].UL = E.prim_to_cons(wl);
            fb[b].UR = E.prim_to_cons(wr);
            entry_cons4(E, fb[b]);
            if (type == RiemannType::LowMachHLLC) lowmach4(E, fb[b]);
        }
        for (int b = 0; b < nb; ++b) {
            const int k = k0 + 4 * b;
            Vec5 Fk;
            V sm;
            hllc4(E, sb, axis_masks(axis + k), fb[b], Fk, sm);
            store_results(F, smax, k, Fk, sm);
        }
    }
    _mm256_zeroupper();
}

// ---------------------------------------------------------------------------------------------------------------------
// Viscous cell kernels of the AMR engine (scalar twins in CellFlux.cpp: viscous_cell_quantities, viscous_cell_speed).  Same rules:
// the operation order of the scalar code, operands of commutative operations included.  The leaves are consecutive: their states
// are loaded like a batch of faces; neighbour states and the sizes of the levels are read with scalar loads (no gather
// instructions, which are slow on many CPUs).

namespace {

// Size of the levels of the leaves c .. c+3 from a table per level
inline V level_size(const double* h, const int* level, int c) {
    return _mm256_set_pd(h[level[c + 3]], h[level[c + 2]], h[level[c + 1]], h[level[c]]);
}

}   // namespace

void viscous_cells_avx2(const EOS& eos, const State* U, const int* nbr, const int* level, const double* dxl, const double* dyl,
                        int c0, int n, double* const* out) {
    const VEos E(eos);
    const V two = set1(2.0);
    for (int k = 0; k < n; k += 4) {
        const int c = c0 + k;
        const Vec5 W = E.cons_to_prim(load4(U + c));
        const int* code = nbr + static_cast<size_t>(4) * c;   // code[4 i + s]: side s of leaf c + i
        V nq[4][3];                                           // components 0..2 of the left, right, bottom and top neighbours
        for (int s = 0; s < 4; ++s) {
            const State &a = U[code[s]], &b = U[code[4 + s]], &d = U[code[8 + s]], &e = U[code[12 + s]];
            for (int q = 0; q < 3; ++q) nq[s][q] = _mm256_set_pd(e[q], d[q], b[q], a[q]);
        }
        const V tdx = mul(two, level_size(dxl, level, c)), tdy = mul(two, level_size(dyl, level, c));
        V gx[3], gy[3];
        for (int q = 0; q < 3; ++q) {
            gx[q] = dv(sub(nq[1][q], nq[0][q]), tdx);
            gy[q] = dv(sub(nq[3][q], nq[2][q]), tdy);
        }
        const V rho = W.q[0], u = W.q[1], v = W.q[2];
        const V r[7] = {u,
                        v,
                        dv(W.q[3], rho),
                        dv(sub(gx[1], mul(u, gx[0])), rho),
                        dv(sub(gy[1], mul(u, gy[0])), rho),
                        dv(sub(gx[2], mul(v, gx[0])), rho),
                        dv(sub(gy[2], mul(v, gy[0])), rho)};
        for (int j = 0; j < 7; ++j) _mm256_storeu_pd(out[j] + c, r[j]);
    }
    _mm256_zeroupper();
}

double viscous_speed_avx2(const EOS& eos, double mu, double k_cond, const State* U, const int* level, const double* dxl,
                          const double* dyl, int c0, int n) {
    const VEos E(eos);
    const V one = set1(1.0), two = set1(2.0), vmu = set1(mu), vk = set1(k_cond), gm1 = set1(eos.gamma - 1.0), c43 = set1(4.0 / 3.0),
            tiny = set1(1e-12);
    V local = set1(0.0);
    for (int k = 0; k < n; k += 4) {
        const int c = c0 + k;
        const V hx = level_size(dxl, level, c), hy = level_size(dyl, level, c);
        const V L = vmin(hx, hy), inv_h2 = add(dv(one, mul(hx, hx)), dv(one, mul(hy, hy)));
        const Vec5 W = E.cons_to_prim(load4(U + c));
        const V rho = W.q[0];
        const V nu_eff = add(vmax(mul(c43, dv(vmu, rho)), dv(mul(vk, gm1), rho)), tiny);
        const V s = add(add(vmax(vabs(W.q[1]), vabs(W.q[2])), E.sound_speed(W.q[3], rho)), mul(mul(mul(two, L), inv_h2), nu_eff));
        local = vmax(local, s);
    }
    alignas(32) double t[4];
    _mm256_store_pd(t, local);
    _mm256_zeroupper();
    double r = 0.0;   // every speed is > 0 or NaN (ignored, as by std::max(local, s)): the maximum does not depend on the order
    for (int i = 0; i < 4; ++i) r = (r < t[i]) ? t[i] : r;
    return r;
}

double viscous_row_cells_avx2(const EOS& eos, double mu, double k_cond, double L, double inv_h2, double dx, double dy, const State* cur,
                              const State* north, const State* south, int i0, int n, double* const* out) {
    const VEos E(eos);
    const V two = set1(2.0), tdx = set1(2.0 * dx), tdy = set1(2.0 * dy);
    const V vmu = set1(mu), vk = set1(k_cond), gm1 = set1(eos.gamma - 1.0), c43 = set1(4.0 / 3.0), tiny = set1(1e-12);
    const V two_l = mul(two, set1(L)), vinv_h2 = set1(inv_h2);
    V local = set1(0.0);
    for (int k = 0; k < n; k += 4) {
        const int c = i0 + k;
        const Vec5 W = E.cons_to_prim(load4(cur + c));
        const Vec5 pe = load4(cur + c + 1), pw = load4(cur + c - 1), pn = load4(north + c), ps = load4(south + c);
        V gx[3], gy[3];   // the central differences of rho, rho u, rho v
        for (int q = 0; q < 3; ++q) {
            gx[q] = dv(sub(pe.q[q], pw.q[q]), tdx);
            gy[q] = dv(sub(pn.q[q], ps.q[q]), tdy);
        }
        const V rho = W.q[0], u = W.q[1], v = W.q[2];
        const V r[7] = {u,
                        v,
                        dv(W.q[3], rho),
                        dv(sub(gx[1], mul(u, gx[0])), rho),
                        dv(sub(gy[1], mul(u, gy[0])), rho),
                        dv(sub(gx[2], mul(v, gx[0])), rho),
                        dv(sub(gy[2], mul(v, gy[0])), rho)};
        for (int j = 0; j < 7; ++j) _mm256_storeu_pd(out[j] + c, r[j]);
        // wave-speed bound of the cell: viscous_nu_eff and viscous_speed_bound (ViscousBound.hpp) in the same order
        const V nu_eff = add(vmax(mul(c43, dv(vmu, rho)), dv(mul(vk, gm1), rho)), tiny);
        const V s = add(add(vmax(vabs(u), vabs(v)), E.sound_speed(W.q[3], rho)), mul(mul(two_l, vinv_h2), nu_eff));
        local = vmax(local, s);
    }
    alignas(32) double t[4];
    _mm256_store_pd(t, local);
    _mm256_zeroupper();
    double r = 0.0;   // every speed is > 0 or NaN (ignored, as by std::max(local, s)): the maximum does not depend on the order
    for (int i = 0; i < 4; ++i) r = (r < t[i]) ? t[i] : r;
    return r;
}

// Face fluxes of the uniform 2-D viscous row sweep (scalar twins: the lambdas xflux / yflux of NavierStokes::viscous_residual_rows,
// interior faces only: no ghost value, no cross-derivative override).  Components 1..3 of the flux, the others are zero.
void viscous_row_xfaces_avx2(double mu, double k_cond, double dx, double dy, const double* u, const double* v, const double* T,
                             const double* dudy, const double* dvdy, int i0, int n, double* f1, double* f2, double* f3) {
    const V half = set1(0.5), c43 = set1(4.0 / 3.0), c23 = set1(2.0 / 3.0), vmu = set1(mu), mk = set1(-k_cond), vdx = set1(dx),
            vdy = set1(dy);
    for (int k = 0; k < n; k += 4) {
        const int i = i0 + k;   // face i lies between the cells i-1 and i
        const V uL = _mm256_loadu_pd(u + i - 1), uR = _mm256_loadu_pd(u + i);
        const V vL = _mm256_loadu_pd(v + i - 1), vR = _mm256_loadu_pd(v + i);
        const V TL = _mm256_loadu_pd(T + i - 1), TR = _mm256_loadu_pd(T + i);
        const V u_f = mul(half, add(uL, uR)), v_f = mul(half, add(vL, vR));
        const V du_dx = dv(sub(uR, uL), vdx), dv_dx = dv(sub(vR, vL), vdx), dT_dx = dv(sub(TR, TL), vdx);
        const V du_dy = mul(half, add(_mm256_loadu_pd(dudy + i - 1), _mm256_loadu_pd(dudy + i)));
        const V dv_dy = mul(half, add(_mm256_loadu_pd(dvdy + i - 1), _mm256_loadu_pd(dvdy + i)));
        const V tau_xx = mul(vmu, sub(mul(c43, du_dx), mul(c23, dv_dy)));
        const V tau_xy = mul(vmu, add(du_dy, dv_dx));
        const V q_x = mul(mk, dT_dx);
        _mm256_storeu_pd(f1 + i, mul(tau_xx, vdy));
        _mm256_storeu_pd(f2 + i, mul(tau_xy, vdy));
        _mm256_storeu_pd(f3 + i, mul(sub(add(mul(u_f, tau_xx), mul(v_f, tau_xy)), q_x), vdy));
    }
    _mm256_zeroupper();
}

// B and T: the cell quantities of the rows below and above the faces (u, v, T, du/dx, dv/dx); faces i0 .. i0+n-1 of the row.
void viscous_row_yfaces_avx2(double mu, double k_cond, double dx, double dy, const double* const* B, const double* const* Tp, int i0,
                             int n, double* g1, double* g2, double* g3) {
    const V half = set1(0.5), c43 = set1(4.0 / 3.0), c23 = set1(2.0 / 3.0), vmu = set1(mu), mk = set1(-k_cond), vdx = set1(dx),
            vdy = set1(dy);
    for (int k = 0; k < n; k += 4) {
        const int i = i0 + k;
        const V uB = _mm256_loadu_pd(B[0] + i), uT = _mm256_loadu_pd(Tp[0] + i);
        const V vB = _mm256_loadu_pd(B[1] + i), vT = _mm256_loadu_pd(Tp[1] + i);
        const V TB = _mm256_loadu_pd(B[2] + i), TT = _mm256_loadu_pd(Tp[2] + i);
        const V u_f = mul(half, add(uB, uT)), v_f = mul(half, add(vB, vT));
        const V dv_dy = dv(sub(vT, vB), vdy), du_dy = dv(sub(uT, uB), vdy), dT_dy = dv(sub(TT, TB), vdy);
        const V du_dx = mul(half, add(_mm256_loadu_pd(B[3] + i), _mm256_loadu_pd(Tp[3] + i)));
        const V dv_dx = mul(half, add(_mm256_loadu_pd(B[4] + i), _mm256_loadu_pd(Tp[4] + i)));
        const V tau_yy = mul(vmu, sub(mul(c43, dv_dy), mul(c23, du_dx)));
        const V tau_yx = mul(vmu, add(du_dy, dv_dx));
        const V q_y = mul(mk, dT_dy);
        _mm256_storeu_pd(g1 + i, mul(tau_yx, vdx));
        _mm256_storeu_pd(g2 + i, mul(tau_yy, vdx));
        _mm256_storeu_pd(g3 + i, mul(sub(add(mul(u_f, tau_yx), mul(v_f, tau_yy)), q_y), vdx));
    }
    _mm256_zeroupper();
}

}   // namespace detail
}   // namespace fv

#endif   // FV_SIMD_AVX2
