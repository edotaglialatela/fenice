// Vector kernels: the batch of the Riemann solvers (four faces per call) and the viscous kernels, written on the operations of
// SimdOps.hpp (AVX2 on x86-64, NEON on ARM64).
//
// Bit-identity with the scalar solver (RiemannSolver::face_flux in Schemes.cpp) is the contract.  Every lane runs
// the same IEEE operations in the same order as the scalar code: only add / sub / mul / div / sqrt and bit logic are used
// (no fused multiply-add, no vector min / max, whose results differ from std::min / std::max on equal values and NaN: those are
// written as a compare and a blend), and every branch of the scalar code is computed for all lanes and selected by a mask.
// This file must be compiled without FMA contraction (the project builds with /fp:precise, resp. -ffp-contract=off, and the
// GCC / Clang build of this file on x86-64 enables AVX2 only, not FMA).
#include "SchemesSimd.hpp"

#ifdef FV_SIMD

#include "SimdOps.hpp"


namespace fv {
namespace detail {

using namespace simd;   // V and its operations

namespace {

constexpr double kDenomEps = 1e-14;   // as in Schemes.cpp
constexpr double kMachEps = 1e-12;

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
void hllc4(const VEos& /*E*/, double sb, const AxisMasks& ax, const Face4& f, Vec5& F, V& smax) {
    const Vec5 &UL = f.UL, &UR = f.UR, &WL = f.WL, &WR = f.WR;
    const V inv_L = f.inv_L, inv_R = f.inv_R;
    const V rho_L = WL.q[0], p_L = WL.q[3], rho_R = WR.q[0], p_R = WR.q[3];
    const V velL[3] = {WL.q[1], WL.q[2], WL.q[4]}, velR[3] = {WR.q[1], WR.q[2], WR.q[4]};
    const V E_L = mul(UL.q[3], inv_L), E_R = mul(UR.q[3], inv_R);
    const V Vn_L = ax.select(velL[0], velL[1], velL[2]), Vn_R = ax.select(velR[0], velR[1], velR[2]);
    // lanes whose normal axis is a (axis 0 is neither 1 nor 2)
    const V is_axis[3] = {andnot(bor(ax.m1, ax.m2), mask_all(true)), ax.m1, ax.m2};
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
    for (int q = 0; q < 5; ++q) U.q[q] = set4(base[0][q], base[1][q], base[2][q], base[3][q]);
    return U;
}

void store4(State* base, const Vec5& F) {
    double t[4];
    for (int q = 0; q < 5; ++q) {
        store(t, F.q[q]);
        for (int i = 0; i < 4; ++i) base[i][q] = t[i];
    }
}

}   // namespace

namespace {

// Masks of the lanes whose axis is 1 and 2
AxisMasks axis_masks(const int* axis) {
    AxisMasks ax;
    ax.m1 = mask_int_eq(axis, 1);
    ax.m2 = mask_int_eq(axis, 2);
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
    double t[4];
    store(t, sm);
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
    return set4(W[st[4 * k + point]][q], W[st[4 * (k + 1) + point]][q], W[st[4 * (k + 2) + point]][q],
                W[st[4 * (k + 3) + point]][q]);
}

}   // namespace

void riemann_batch_simd(const EOS& eos, RiemannType type, double sb, const State* UL, const State* UR, const int* axis, int n,
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
    end_batch();
}

namespace {

// The MUSCL face batch for any way of reading the stencil: load(k, point, q) = the four faces k .. k+3 of the batch, point 0..3 of
// their stencils, primitive variable q; axes(k) = AxisMasks of the faces k .. k+3.
template <class Load, class Axes>
void muscl_riemann_body(const EOS& eos, RiemannType type, double sb, Limiter lim, int n, State* F, double* smax, bool planar,
                        const Load& load, const Axes& axes) {
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
                const V c0 = load(k, 0, q), c1 = load(k, 1, q), c2 = load(k, 2, q), c3 = load(k, 3, q);
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
            hllc4(E, sb, axes(k), fb[b], Fk, sm);
            store_results(F, smax, k, Fk, sm);
        }
    }
    end_batch();
}

}   // namespace

void muscl_riemann_batch_simd(const EOS& eos, RiemannType type, double sb, Limiter lim, const State* W, const int* st,
                              const int* axis, int n, State* F, double* smax, bool planar) {
    muscl_riemann_body(
        eos, type, sb, lim, n, F, smax, planar, [&](int k, int point, int q) { return gather_q(W, st, k, point, q); },
        [&](int k) { return axis_masks(axis + k); });
}

void muscl_riemann_line_simd(const EOS& eos, RiemannType type, double sb, Limiter lim, const double* W, size_t plane, size_t base,
                             size_t stride, int axis, int n, State* F, double* smax, bool planar) {
    AxisMasks ax;
    ax.m1 = mask_all(axis == 1);
    ax.m2 = mask_all(axis == 2);
    muscl_riemann_body(
        eos, type, sb, lim, n, F, smax, planar,
        [&](int k, int point, int q) { return load(W + q * plane + base + k + point * stride); },
        [&](int) { return ax; });
}

// ---------------------------------------------------------------------------------------------------------------------
// WENO5-Z reconstruction of the faces of one line (scalar twin: weno5z in Schemes.cpp, same operations in the same order).

namespace {

inline V weno5z_v(V c0, V c1, V c2, V c3, V c4, V eps_rel) {
    const V m1 = set1(-1.0);
    const V eps = add(mul(mul(eps_rel, add(add(add(add(mul(c0, c0), mul(c1, c1)), mul(c2, c2)), mul(c3, c3)), mul(c4, c4))), set1(0.2)),
                      set1(1e-300));
    const V six = set1(6.0);
    const V p0 = dv(add(sub(mul(set1(2.0), c0), mul(set1(7.0), c1)), mul(set1(11.0), c2)), six);
    const V p1 = dv(add(add(mul(c1, m1), mul(set1(5.0), c2)), mul(set1(2.0), c3)), six);
    const V p2 = dv(sub(add(mul(set1(2.0), c2), mul(set1(5.0), c3)), c4), six);
    const V s0 = add(sub(c0, mul(set1(2.0), c1)), c2), t0 = add(sub(c0, mul(set1(4.0), c1)), mul(set1(3.0), c2));
    const V s1 = add(sub(c1, mul(set1(2.0), c2)), c3), t1 = sub(c1, c3);
    const V s2 = add(sub(c2, mul(set1(2.0), c3)), c4), t2 = add(sub(mul(set1(3.0), c2), mul(set1(4.0), c3)), c4);
    const V k13 = set1(13.0 / 12.0), q = set1(0.25);
    const V b0 = add(mul(mul(k13, s0), s0), mul(mul(q, t0), t0));
    const V b1 = add(mul(mul(k13, s1), s1), mul(mul(q, t1), t1));
    const V b2 = add(mul(mul(k13, s2), s2), mul(mul(q, t2), t2));
    const V tau = vabs(sub(b0, b2));
    const V one = set1(1.0);
    const V a0 = mul(set1(0.1), add(one, dv(tau, add(b0, eps))));
    const V a1 = mul(set1(0.6), add(one, dv(tau, add(b1, eps))));
    const V a2 = mul(set1(0.3), add(one, dv(tau, add(b2, eps))));
    return dv(add(add(mul(a0, p0), mul(a1, p1)), mul(a2, p2)), add(add(a0, a1), a2));
}

}   // namespace

namespace {

inline V weno3z_v(V c0, V c1, V c2, V eps_rel) {
    const V eps = add(mul(mul(eps_rel, add(add(mul(c0, c0), mul(c1, c1)), mul(c2, c2))), set1(1.0 / 3.0)), set1(1e-300));
    const V v0 = add(mul(set1(-0.5), c0), mul(set1(1.5), c1)), v1 = add(mul(set1(0.5), c1), mul(set1(0.5), c2));
    const V d0 = sub(c1, c0), d1 = sub(c2, c1);
    const V b0 = mul(d0, d0), b1 = mul(d1, d1);
    const V tau = vabs(sub(b0, b1));
    const V one = set1(1.0);
    const V a0 = mul(set1(1.0 / 3.0), add(one, dv(tau, add(b0, eps))));
    const V a1 = mul(set1(2.0 / 3.0), add(one, dv(tau, add(b1, eps))));
    return dv(add(mul(a0, v0), mul(a1, v1)), add(a0, a1));
}

}   // namespace

void weno3_line_simd(const EOS& eos, double eps_rel, const double* const* planes, size_t base, size_t stride, int n, State* UL,
                     State* UR) {
    const V we = set1(eps_rel);
    const VEos E(eos);
    for (int k = 0; k < n; k += 4) {
        Vec5 wl, wr;
        for (int q = 0; q < 5; ++q) {
            const double* c = planes[q] + base + k;
            const V c0 = load(c), c1 = load(c + stride), c2 = load(c + 2 * stride), c3 = load(c + 3 * stride);
            wl.q[q] = weno3z_v(c0, c1, c2, we);
            wr.q[q] = weno3z_v(c3, c2, c1, we);
        }
        store4(UL + k, E.prim_to_cons(wl));
        store4(UR + k, E.prim_to_cons(wr));
    }
    end_batch();
}

void cons_to_prim_planes_simd(const EOS& eos, const State* U, int n, double* const* out) {
    const VEos E(eos);
    for (int k = 0; k < n; k += 4) {
        const Vec5 W = E.cons_to_prim(load4(U + k));
        for (int q = 0; q < 5; ++q) store(out[q] + k, W.q[q]);
    }
    end_batch();
}

void weno5_line_simd(const EOS& eos, double eps_rel, const double* const* planes, size_t base, size_t stride, int n, State* UL,
                     State* UR) {
    const V we = set1(eps_rel);
    const VEos E(eos);
    for (int k = 0; k < n; k += 4) {
        Vec5 wl, wr;
        for (int q = 0; q < 5; ++q) {
            const double* c = planes[q] + base + k;
            const V c0 = load(c), c1 = load(c + stride), c2 = load(c + 2 * stride), c3 = load(c + 3 * stride),
                    c4 = load(c + 4 * stride), c5 = load(c + 5 * stride);
            wl.q[q] = weno5z_v(c0, c1, c2, c3, c4, we);
            wr.q[q] = weno5z_v(c5, c4, c3, c2, c1, we);
        }
        store4(UL + k, E.prim_to_cons(wl));
        store4(UR + k, E.prim_to_cons(wr));
    }
    end_batch();
}

// ---------------------------------------------------------------------------------------------------------------------
// Viscous cell kernels of the AMR engine (scalar twins in CellFlux.cpp: viscous_cell_quantities, viscous_cell_speed).  Same rules:
// the operation order of the scalar code, operands of commutative operations included.  The leaves are consecutive: their states
// are loaded like a batch of faces; neighbour states and the sizes of the levels are read with scalar loads (no gather
// instructions, which are slow on many CPUs).

namespace {

// Size of the levels of the leaves c .. c+3 from a table per level
inline V level_size(const double* h, const int* level, int c) {
    return set4(h[level[c]], h[level[c + 1]], h[level[c + 2]], h[level[c + 3]]);
}

}   // namespace

void viscous_cells_simd(const EOS& eos, const State* U, const int* nbr, const int* level, const double* dxl, const double* dyl,
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
            for (int q = 0; q < 3; ++q) nq[s][q] = set4(a[q], b[q], d[q], e[q]);
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
        for (int j = 0; j < 7; ++j) store(out[j] + c, r[j]);
    }
    end_batch();
}

double viscous_speed_simd(const EOS& eos, double mu, double k_cond, const State* U, const int* level, const double* dxl,
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
    double t[4];
    store(t, local);
    end_batch();
    double r = 0.0;   // every speed is > 0 or NaN (ignored, as by std::max(local, s)): the maximum does not depend on the order
    for (int i = 0; i < 4; ++i) r = (r < t[i]) ? t[i] : r;
    return r;
}

double viscous_row_cells_simd(const EOS& eos, double mu, double k_cond, double L, double inv_h2, double dx, double dy, const State* cur,
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
        for (int j = 0; j < 7; ++j) store(out[j] + c, r[j]);
        // wave-speed bound of the cell: viscous_nu_eff and viscous_speed_bound (ViscousBound.hpp) in the same order
        const V nu_eff = add(vmax(mul(c43, dv(vmu, rho)), dv(mul(vk, gm1), rho)), tiny);
        const V s = add(add(vmax(vabs(u), vabs(v)), E.sound_speed(W.q[3], rho)), mul(mul(two_l, vinv_h2), nu_eff));
        local = vmax(local, s);
    }
    double t[4];
    store(t, local);
    end_batch();
    double r = 0.0;   // every speed is > 0 or NaN (ignored, as by std::max(local, s)): the maximum does not depend on the order
    for (int i = 0; i < 4; ++i) r = (r < t[i]) ? t[i] : r;
    return r;
}

// Face fluxes of the uniform 2-D viscous row sweep (scalar twins: the lambdas xflux / yflux of NavierStokes::viscous_residual_rows,
// interior faces only: no ghost value, no cross-derivative override).  Components 1..3 of the flux, the others are zero.
void viscous_row_xfaces_simd(double mu, double k_cond, double dx, double dy, const double* u, const double* v, const double* T,
                             const double* dudy, const double* dvdy, int i0, int n, double* f1, double* f2, double* f3) {
    const V half = set1(0.5), c43 = set1(4.0 / 3.0), c23 = set1(2.0 / 3.0), vmu = set1(mu), mk = set1(-k_cond), vdx = set1(dx),
            vdy = set1(dy);
    for (int k = 0; k < n; k += 4) {
        const int i = i0 + k;   // face i lies between the cells i-1 and i
        const V uL = load(u + i - 1), uR = load(u + i);
        const V vL = load(v + i - 1), vR = load(v + i);
        const V TL = load(T + i - 1), TR = load(T + i);
        const V u_f = mul(half, add(uL, uR)), v_f = mul(half, add(vL, vR));
        const V du_dx = dv(sub(uR, uL), vdx), dv_dx = dv(sub(vR, vL), vdx), dT_dx = dv(sub(TR, TL), vdx);
        const V du_dy = mul(half, add(load(dudy + i - 1), load(dudy + i)));
        const V dv_dy = mul(half, add(load(dvdy + i - 1), load(dvdy + i)));
        const V tau_xx = mul(vmu, sub(mul(c43, du_dx), mul(c23, dv_dy)));
        const V tau_xy = mul(vmu, add(du_dy, dv_dx));
        const V q_x = mul(mk, dT_dx);
        store(f1 + i, mul(tau_xx, vdy));
        store(f2 + i, mul(tau_xy, vdy));
        store(f3 + i, mul(sub(add(mul(u_f, tau_xx), mul(v_f, tau_xy)), q_x), vdy));
    }
    end_batch();
}

// B and T: the cell quantities of the rows below and above the faces (u, v, T, du/dx, dv/dx); faces i0 .. i0+n-1 of the row.
void viscous_row_yfaces_simd(double mu, double k_cond, double dx, double dy, const double* const* B, const double* const* Tp, int i0,
                             int n, double* g1, double* g2, double* g3) {
    const V half = set1(0.5), c43 = set1(4.0 / 3.0), c23 = set1(2.0 / 3.0), vmu = set1(mu), mk = set1(-k_cond), vdx = set1(dx),
            vdy = set1(dy);
    for (int k = 0; k < n; k += 4) {
        const int i = i0 + k;
        const V uB = load(B[0] + i), uT = load(Tp[0] + i);
        const V vB = load(B[1] + i), vT = load(Tp[1] + i);
        const V TB = load(B[2] + i), TT = load(Tp[2] + i);
        const V u_f = mul(half, add(uB, uT)), v_f = mul(half, add(vB, vT));
        const V dv_dy = dv(sub(vT, vB), vdy), du_dy = dv(sub(uT, uB), vdy), dT_dy = dv(sub(TT, TB), vdy);
        const V du_dx = mul(half, add(load(B[3] + i), load(Tp[3] + i)));
        const V dv_dx = mul(half, add(load(B[4] + i), load(Tp[4] + i)));
        const V tau_yy = mul(vmu, sub(mul(c43, dv_dy), mul(c23, du_dx)));
        const V tau_yx = mul(vmu, add(du_dy, dv_dx));
        const V q_y = mul(mk, dT_dy);
        store(g1 + i, mul(tau_yx, vdx));
        store(g2 + i, mul(tau_yy, vdx));
        store(g3 + i, mul(sub(add(mul(u_f, tau_yx), mul(v_f, tau_yy)), q_y), vdx));
    }
    end_batch();
}

void viscous_faces3d_simd(int d, double mu, double k_cond, double h, double area, const double* const* vel, const double* T,
                          const double* const* gc, long delta, int n, State* out) {
    const V half = set1(0.5), c43 = set1(4.0 / 3.0), c23 = set1(2.0 / 3.0), vmu = set1(mu), mk = set1(-k_cond), vh = set1(h),
            varea = set1(area), zero = set1(0.0);
    for (int k = 0; k < n; k += 4) {
        V Gn[3], vf[3];
        for (int a = 0; a < 3; ++a) {
            const V vL = load(vel[a] + k), vR = load(vel[a] + k + delta);
            Gn[a] = dv(sub(vR, vL), vh);
            vf[a] = mul(half, add(vL, vR));
        }
        const V dT = dv(sub(load(T + k + delta), load(T + k)), vh);
        V g[4];   // the cross terms, cross(): 0.5 * (left + right)
        for (int q = 0; q < 4; ++q) g[q] = mul(half, add(load(gc[q] + k), load(gc[q] + k + delta)));
        V tau[3];
        if (d == 0) {
            const V others = add(add(zero, g[1]), g[2]);
            tau[0] = mul(vmu, sub(mul(c43, Gn[0]), mul(c23, others)));
            tau[1] = mul(vmu, add(Gn[1], g[0]));
            tau[2] = mul(vmu, add(Gn[2], g[3]));
        } else if (d == 1) {
            const V others = add(add(zero, g[0]), g[2]);
            tau[0] = mul(vmu, add(Gn[0], g[1]));
            tau[1] = mul(vmu, sub(mul(c43, Gn[1]), mul(c23, others)));
            tau[2] = mul(vmu, add(Gn[2], g[3]));
        } else {
            const V others = add(add(zero, g[0]), g[1]);
            tau[0] = mul(vmu, add(Gn[0], g[2]));
            tau[1] = mul(vmu, add(Gn[1], g[3]));
            tau[2] = mul(vmu, sub(mul(c43, Gn[2]), mul(c23, others)));
        }
        const V q_d = mul(mk, dT);
        const V f[5] = {zero, mul(tau[0], varea), mul(tau[1], varea),
                        mul(sub(add(add(mul(vf[0], tau[0]), mul(vf[1], tau[1])), mul(vf[2], tau[2])), q_d), varea), mul(tau[2], varea)};
        double t[4];
        for (int q = 0; q < 5; ++q) {
            store(t, f[q]);
            for (int i = 0; i < 4; ++i) out[k + i][q] = t[i];
        }
    }
    end_batch();
}

void viscous_cells3d_simd(double dx, double dy, double dz, const State* W, const State* xm, const State* xp, const State* ym,
                          const State* yp, const State* zm, const State* zp, int n, double* const* out, double* const* g) {
    const V td[3] = {set1(2.0 * dx), set1(2.0 * dy), set1(2.0 * dz)};
    const State* lo[3] = {xm, ym, zm};
    const State* hi[3] = {xp, yp, zp};
    static const int kSlot3[3] = {1, 2, 4};
    for (int k = 0; k < n; k += 4) {
        const Vec5 w = load4(W + k);
        const V rho = w.q[0];
        const V vel[3] = {w.q[1], w.q[2], w.q[4]};
        store(out[0] + k, vel[0]);
        store(out[1] + k, vel[1]);
        store(out[2] + k, vel[2]);
        store(out[3] + k, dv(w.q[3], rho));
        for (int b = 0; b < 3; ++b) {
            const auto slot = [&](int q) {
                const State* l = lo[b] + k;
                const State* h = hi[b] + k;
                return dv(sub(set4(h[0][q], h[1][q], h[2][q], h[3][q]), set4(l[0][q], l[1][q], l[2][q], l[3][q])), td[b]);
            };
            const V g0 = slot(0);
            for (int a = 0; a < 3; ++a) store(g[3 * a + b] + k, dv(sub(slot(kSlot3[a]), mul(vel[a], g0)), rho));
        }
    }
    end_batch();
}

void axpy_stream_simd(double* out, const double* a, double s, const double* b, size_t n) {
    const V vs = set1(s);
    for (size_t i = 0; i < n; i += 4) store_stream(out + i, add(load(a + i), mul(vs, load(b + i))));
    stream_fence();
    end_batch();
}

}   // namespace detail
}   // namespace fv

#endif   // FV_SIMD
