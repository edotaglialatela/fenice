"""Error norms, observed orders of convergence and the Grid Convergence Index, shared by the
validation studies. The GCI follows Roache, *Fundamentals of Verification and Validation* (2009),
chapter 5: the generalised Richardson extrapolation (5.4.1), the error estimator (5.4.2-3), the
fine-grid GCI (5.6.1), the observed order of a grid triplet (5.10.6.3-5), the asymptotic-range
check (5.10.5.2) and the factors of safety of section 5.9.2.

    >>> g = gci_triplet(f=(1.0 + 0.5 * 0.1 ** 2, 1.0 + 0.5 * 0.2 ** 2, 1.0 + 0.5 * 0.4 ** 2), h=(0.1, 0.2, 0.4), p_formal=2)
    >>> round(g.p_obs, 6), round(g.f_extrapolated, 9)
    (2.0, 1.0)

Run ``python -m fenice.convergence`` for the self-test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------- errors and orders
def l1_error(a, b, weights=None, relative=False):
    """sum(w |a - b|), relative to sum(w |b|) if asked; weights default to a plain mean."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.full(a.shape, 1.0 / a.size) if weights is None else np.asarray(weights, float)
    e = np.sum(w * np.abs(a - b))
    return e / np.sum(w * np.abs(b)) if relative else e


def l2_error(a, b, weights=None, relative=False):
    """sqrt(sum(w (a - b)^2)), relative to sqrt(sum(w b^2)) if asked."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.full(a.shape, 1.0 / a.size) if weights is None else np.asarray(weights, float)
    e = np.sqrt(np.sum(w * (a - b) ** 2))
    return e / np.sqrt(np.sum(w * b ** 2)) if relative else e


def observed_orders(h, err):
    """Order between consecutive refinements, log(e_i / e_{i+1}) / log(h_i / h_{i+1}), for any ratio.
    `h` and `err` run from the coarsest to the finest (decreasing h)."""
    h, err = np.asarray(h, float), np.asarray(err, float)
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


def fit_slope(h, err):
    """Least-squares slope of log(err) against log(h): the order from all the grids at once."""
    return float(np.polyfit(np.log(np.asarray(h, float)), np.log(np.asarray(err, float)), 1)[0])


# ----------------------------------------------------------------------------- Richardson and GCI
def richardson(f1, f2, r, p):
    """Extrapolated value from the fine (f1) and coarse (f2) solutions, refinement ratio r = h2/h1 (5.4.1)."""
    return f1 + (f1 - f2) / (r ** p - 1.0)


def apparent_order(f1, f2, f3, r12, r23, relaxation=0.5, tol=1e-10, max_iter=500):
    """Observed order p of a grid triplet (1 finest, 3 coarsest) from eq. (5.10.6.3),
    eps23 / (r23^p - 1) = r12^p eps12 / (r12^p - 1), eps12 = f2 - f1, eps23 = f3 - f2. For a constant
    ratio it is the closed form p = ln(eps23 / eps12) / ln r (5.10.6.4); otherwise the damped fixed-point
    iteration (5.10.6.5). Returns None when the convergence is not monotone (eps12, eps23 of opposite
    sign) or the solutions are identical: no order can be read from such a triplet."""
    e12, e23 = f2 - f1, f3 - f2
    if e12 == 0.0 or e23 == 0.0 or e12 * e23 < 0.0:
        return None
    if abs(r12 - r23) < 1e-12:
        return float(np.log(e23 / e12) / np.log(r12))
    p = abs(np.log(e23 / e12) / np.log(r12))
    for _ in range(max_iter):
        beta = ((r12 ** p - 1.0) * e23) / ((r23 ** p - 1.0) * e12)
        if beta <= 0.0:
            return None
        new = relaxation * p + (1.0 - relaxation) * np.log(beta) / np.log(r12)
        if abs(new - p) < tol:
            return float(new)
        p = new
    return None


def gci(f1, f2, r, p, fs, relative=True, scale=None):
    """Fine-grid Grid Convergence Index (5.6.1), Fs |eps| / (r^p - 1), eps = (f2 - f1) / f1 (5.4.3a) or
    f2 - f1 (5.4.3b, `relative=False`, then an absolute error band). `scale` replaces f1 as the
    normaliser (the triplet uses the finest solution for both of its indices)."""
    eps = (f2 - f1) / (f1 if scale is None else scale) if relative else f2 - f1
    return fs * abs(eps) / (r ** p - 1.0)


@dataclass
class GCI:
    """Result of a grid triplet. `gci_fine` is a fraction of f1 (or absolute, see `relative`)."""
    f: tuple
    r12: float
    r23: float
    p_formal: float
    p_obs: float | None          # None: not monotone, no order
    p_used: float | None
    fs: float
    gci_fine: float | None
    gci_coarse_pair: float | None
    f_extrapolated: float | None
    asymptotic_ratio: float | None   # GCI23 / (r12^p GCI12), 1 in the asymptotic range (5.10.5.2)
    monotone: bool
    notes: list = field(default_factory=list)

    @property
    def reliable(self):
        """Monotone, order close to the formal one and asymptotic ratio close to 1."""
        return self.monotone and self.fs < 3.0 and self.asymptotic_ratio is not None and abs(self.asymptotic_ratio - 1.0) < 0.25


def gci_triplet(f, h, p_formal, relative=True, band=0.25, fs_three_grids=1.25, fs_two_grids=3.0):
    """GCI of three solutions f = (f_fine, f_medium, f_coarse) on grids of sizes h = (h1, h2, h3).

    Roache's recommendations (5.9.2): Fs = 1.25 with three grids whose observed order is reasonable,
    Fs = 3 otherwise. 'Reasonable' is |p_obs - p_formal| <= band * p_formal; the order used in the
    formula is never larger than the formal one (it is imprudent to use p_obs > p). A non-monotone
    triplet (oscillatory convergence) gets no GCI: the numbers are flagged, not reported."""
    f1, f2, f3 = (float(x) for x in f)
    h1, h2, h3 = (float(x) for x in h)
    r12, r23 = h2 / h1, h3 / h2
    notes = []
    p_obs = apparent_order(f1, f2, f3, r12, r23)
    if p_obs is None:
        notes.append("not monotone: no observed order, no GCI (more or finer grids are needed)")
        return GCI((f1, f2, f3), r12, r23, p_formal, None, None, fs_two_grids, None, None, None, None, False, notes)
    if p_obs <= 0.0:
        notes.append(f"apparent order {p_obs:.2f} <= 0: the solutions do not converge on these grids, no GCI")
        return GCI((f1, f2, f3), r12, r23, p_formal, p_obs, None, fs_two_grids, None, None, None, None, False, notes)
    reasonable = abs(p_obs - p_formal) <= band * p_formal
    fs = fs_three_grids if reasonable else fs_two_grids
    if not reasonable:
        notes.append(f"observed order {p_obs:.2f} far from the formal {p_formal:g}: Fs = {fs_two_grids:g}")
    p_used = min(p_obs, p_formal)
    if p_obs > p_formal:
        notes.append(f"observed order {p_obs:.2f} > formal {p_formal:g}: the formal order is used")
    g12 = gci(f1, f2, r12, p_used, fs, relative)
    g23 = gci(f2, f3, r23, p_used, fs, relative, scale=f1)   # same normaliser: the ratio below is then exact
    ratio = g23 / (r12 ** p_used * g12) if g12 > 0.0 else None
    return GCI((f1, f2, f3), r12, r23, p_formal, p_obs, p_used, fs, g12, g23, richardson(f1, f2, r12, p_used), ratio,
               True, notes)


def gci_pair(f, h, p_formal, relative=True, fs=3.0):
    """Two grids only: the order cannot be checked, so Fs = 3 and the formal order are used (5.9.2)."""
    f1, f2 = float(f[0]), float(f[1])
    r = h[1] / h[0]
    return gci(f1, f2, r, p_formal, fs, relative)


def selftest():
    """Roache's own examples and a synthetic series."""
    # 5.8: steady Burgers equation, f1 = -529.41 on 2000 cells, f2 = -544.48 on 1600 (r = 1.25), p = 2
    g = gci(-529.41, -544.48, 1.25, 2.0, 3.0)
    assert abs(g - 0.1520) < 5e-4, g                      # GCI = 15.20 % with Fs = 3
    assert abs(gci(-529.41, -544.48, 1.25, 2.0, 1.25) - 0.0634) < 5e-4   # 6.34 % with Fs = 1.25
    # 5.5: Richardson's pi from the inscribed 4- and 6-gons: 3.1372583
    assert abs(richardson(3.0, 2 * np.sqrt(2), 1.5, 2.0) - 3.1372583) < 1e-6
    # exact h^2 series: observed order 2, extrapolation recovers the limit, constant and varying ratios
    for hs in ((0.1, 0.2, 0.4), (0.1, 0.15, 0.3)):
        f = [1.0 + 0.5 * h ** 2 for h in hs]
        t = gci_triplet(f, hs, 2)
        assert abs(t.p_obs - 2.0) < 1e-8 and abs(t.f_extrapolated - 1.0) < 1e-10 and t.fs == 1.25, (hs, t)
        assert abs(t.asymptotic_ratio - 1.0) < 1e-8
    # a first-order series reported against a formal order 2 is flagged, not trusted
    t = gci_triplet([1.0 + 0.5 * h for h in (0.1, 0.2, 0.4)], (0.1, 0.2, 0.4), 2)
    assert abs(t.p_obs - 1.0) < 1e-8 and t.fs == 3.0 and t.notes
    # oscillatory convergence: no GCI
    t = gci_triplet((1.0, 1.1, 1.05), (0.1, 0.2, 0.4), 2)
    assert t.p_obs is None and t.gci_fine is None and not t.monotone
    # differences that shrink too slowly (negative apparent order): not converging, no GCI
    t = gci_triplet((0.188, 0.206, 0.231), (2 * np.pi / 96, 2 * np.pi / 64, 2 * np.pi / 32), 2)
    assert t.p_obs is not None and t.p_obs < 0 and t.gci_fine is None and not t.monotone
    assert abs(observed_orders([0.4, 0.2, 0.1], [1.6, 0.4, 0.1])[0] - 2.0) < 1e-12
    assert abs(fit_slope([0.4, 0.2, 0.1], [1.6, 0.4, 0.1]) - 2.0) < 1e-12
    print("fenice.convergence self-test: OK")


if __name__ == "__main__":
    selftest()
