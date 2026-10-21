"""Exact solutions used by the validation scripts."""
import numpy as np


def riemann(x, t, left=(1.0, 0.0, 1.0), right=(0.125, 0.0, 0.1), gamma=1.4, x0=0.5):
    """Exact solution of the 1-D Riemann problem (Toro 2009, ch. 4) at the points x and time t.

    left / right: (rho, u, p) on each side of the initial discontinuity at x0 (default: Sod).
    Returns arrays rho, u, p."""
    rL, uL, pL = left
    rR, uR, pR = right
    g = gamma
    cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)

    def f(p, rk, pk, ck):   # pressure function and derivative of one side
        if p > pk:   # shock
            A, B = 2.0 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
            return (p - pk) * np.sqrt(A / (p + B)), np.sqrt(A / (p + B)) * (1 - (p - pk) / (2 * (p + B)))
        pr = p / pk   # rarefaction
        return 2 * ck / (g - 1) * (pr ** ((g - 1) / (2 * g)) - 1), pr ** (-(g + 1) / (2 * g)) / (rk * ck)

    ps = 0.5 * (pL + pR)
    for _ in range(100):   # Newton for the star pressure
        fl, dl = f(ps, rL, pL, cL)
        fr, dr = f(ps, rR, pR, cR)
        dp = -(fl + fr + uR - uL) / (dl + dr)
        ps = max(ps + dp, 1e-12)
        if abs(dp) < 1e-12 * ps:
            break
    fl, _ = f(ps, rL, pL, cL)
    fr, _ = f(ps, rR, pR, cR)
    us = 0.5 * (uL + uR + fr - fl)

    def star_rho(rk, pk):
        if ps > pk:
            r = ps / pk
            return rk * (r + (g - 1) / (g + 1)) / (r * (g - 1) / (g + 1) + 1)
        return rk * (ps / pk) ** (1 / g)

    rsL, rsR = star_rho(rL, pL), star_rho(rR, pR)
    out = np.empty((3, np.size(x)))
    for i, xi in enumerate(np.atleast_1d(x)):
        s = (xi - x0) / t
        if s < us:   # left of the contact
            if ps > pL:
                sl = uL - cL * np.sqrt((g + 1) / (2 * g) * ps / pL + (g - 1) / (2 * g))
                out[:, i] = (rL, uL, pL) if s < sl else (rsL, us, ps)
            else:
                shl, stl = uL - cL, us - cL * (ps / pL) ** ((g - 1) / (2 * g))
                if s < shl:
                    out[:, i] = (rL, uL, pL)
                elif s > stl:
                    out[:, i] = (rsL, us, ps)
                else:   # inside the left fan
                    u = 2 / (g + 1) * (cL + (g - 1) / 2 * uL + s)
                    c = 2 / (g + 1) * (cL + (g - 1) / 2 * (uL - s))
                    out[:, i] = (rL * (c / cL) ** (2 / (g - 1)), u, pL * (c / cL) ** (2 * g / (g - 1)))
        else:        # right of the contact
            if ps > pR:
                sr = uR + cR * np.sqrt((g + 1) / (2 * g) * ps / pR + (g - 1) / (2 * g))
                out[:, i] = (rR, uR, pR) if s > sr else (rsR, us, ps)
            else:
                shr, str_ = uR + cR, us + cR * (ps / pR) ** ((g - 1) / (2 * g))
                if s > shr:
                    out[:, i] = (rR, uR, pR)
                elif s < str_:
                    out[:, i] = (rsR, us, ps)
                else:   # inside the right fan
                    u = 2 / (g + 1) * (-cR + (g - 1) / 2 * uR + s)
                    c = 2 / (g + 1) * (cR - (g - 1) / 2 * (uR - s))
                    out[:, i] = (rR * (c / cR) ** (2 / (g - 1)), u, pR * (c / cR) ** (2 * g / (g - 1)))
    return out[0], out[1], out[2]
