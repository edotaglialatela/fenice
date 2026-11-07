"""Curved immersed wall against an exact solution: the supersonic vortex around a cylinder, ghost wall with and without the curvature correction.

    python cases/validation/verification/ibm_vortex.py            # results/validation/ibm_vortex/  (40, 80, 160 cells per radius)

The steady irrotational vortex u_theta = A / r around a cylinder of radius R = 1 is an exact solution of the Euler equations
(isentropic, Mach 2.25 at the wall, c = 1 there): p grows outwards with dp/dr = rho u_theta^2 / r. The run starts from the exact
field with the full cylinder at the centre of a 4 x 4 box (slip wall, transmissive sides) and is read at t = 0.4, before the
sides can matter: the error of the cells 2.5 to 5 cells from the wall is the error the wall makes. The ghost wall with its zero
normal pressure gradient (`ibm_curvature = false`) puts a systematic overpressure on the whole ring that falls only as the first
power of h; with `ibm_curvature = true` (dp/dn = rho kappa u_t^2 and constant entropy along the normal in the ghost state) it is
an order of magnitude smaller and falls faster.

Bands (set after an exploratory run, so a regression guard, not a prediction):
  1. with the correction the mean relative pressure error at the finest mesh is at most 2e-3, and its observed order between the
     two finest meshes is at least 1.2;
  2. the error with the correction is at most a fifth of the one without it at the finest mesh.
  3. start-up: the Mach 2 cylinder (the case of the repository, 25 cells per diameter, started from the uniform state, t = 4) forms a
     detached bow shock with the correction on, its stand-off distance within 0.8 to 1.1 of the plain ghost wall's (the correction is
     weighted by how well the flow follows the wall; unweighted, the shock stays pinned to the wall).
"""
import math

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_vortex"
GAMMA, MACH, R, C, L, T_END = 1.4, 2.25, 1.0, 2.0, 4.0, 0.4
BAND_ERR, BAND_ORDER, BAND_RATIO = 2e-3, 1.2, 0.2
BAND_STANDOFF = (0.8, 1.1)

CASE = """[case]
name = "Supersonic vortex"
description = "irrotational vortex around a cylinder, exact steady solution"
[mesh]
Lx = {L}
Ly = {L}
nx = {n}
ny = {n}
[physics]
use_viscous = false
use_bodyforce = false
[boundaries]
bc_left = "transmissive"
bc_right = "transmissive"
bc_bottom = "transmissive"
bc_top = "transmissive"
[numerics]
riemann_solver = "HLLC"
reconstruction = "muscl"
muscl_limiter = "Van Leer"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.4
[run]
t_end = {t_end}
save_interval = 1000000000
[parallel]
use_mpi = false
[ibm]
use_ibm = true
ibm_shape = "Cylinder"
ibm_radius = {R}
ibm_cx = {C}
ibm_cy = {C}
[initial]
state = {{ rho = "pow({c2},2.5)", u = "-{M}*{R}*(y-{C})/{r2}", v = "{M}*{R}*(x-{C})/{r2}", p = "pow({c2},3.5)/1.4" }}
"""


def exact(x, y):
    """rho, u, v, p of the vortex (x, y measured from the cylinder centre; r clipped at 0.7 R inside the body)."""
    r2 = np.maximum(x * x + y * y, (0.7 * R) ** 2)
    c2 = 1.0 + 0.5 * (GAMMA - 1.0) * MACH ** 2 * (1.0 - R * R / r2)
    rho = c2 ** (1.0 / (GAMMA - 1.0))
    return rho, -MACH * R * y / r2, MACH * R * x / r2, rho * c2 / GAMMA


def vortex_case(n):
    dx, dy, m = f"(x-{C})", f"(y-{C})", (0.7 * R) ** 2
    raw = f"({dx}*{dx}+{dy}*{dy})"
    r2 = f"(0.5*({raw}+{m}+abs({raw}-{m})))"   # max(r^2, m): the continuation inside the body stays finite
    c2 = f"(1+0.2*{MACH ** 2}*(1-{R * R}/{r2}))"
    text = CASE.format(L=L, n=n, t_end=T_END, R=R, C=C, M=MACH, r2=r2, c2=c2)
    return study.write_case(STUDY, f"vortex_{n}", text)


def pressure_error(run_dir, cells_per_radius):
    n = int(L * cells_per_radius)
    h = 1.0 / cells_per_radius
    x = (np.arange(n) + 0.5) * h - C
    X, Y = np.meshgrid(x, x)
    X, Y = X.ravel(), Y.ravel()
    r = np.hypot(X, Y)
    _, _, _, p = fenice.primitive(fenice.read_dump(run_dir / "final.npz")["U"])
    pe = exact(X, Y)[3]
    sel = (r - R > 2.5 * h) & (r - R < 5.0 * h)
    return float(np.mean(p[sel] / pe[sel] - 1.0))


def standoff(run_dir, nx, ny, h):
    """Bow-shock stand-off distance over the diameter of the Mach 2 cylinder (centre row, density half way to the post-shock value), None if no shock."""
    rho = fenice.read_dump(run_dir / "final.npz")["U"][:, 0].reshape(ny, nx)
    row = rho[ny // 2 - 1: ny // 2 + 1].mean(axis=0)
    x = (np.arange(nx) + 0.5) * h
    mid = 0.5 * (1.0 + 2.667)
    found = (row > mid) & (x < 1.75 - 0.5 * h)
    i = int(np.argmax(found))
    if not found.any() or i == 0:
        return None
    return (1.75 - (x[i - 1] + (mid - row[i - 1]) / (row[i] - row[i - 1]) * h)) / 0.5


def main():
    ap = study.parser(__doc__.splitlines()[0])
    args = ap.parse_args()
    sizes = (40, 80, 160)
    out = {"sizes": list(sizes), "t": T_END}
    for tag, sets in (("plain", ["ibm_curvature=false"]), ("curv", ["ibm_curvature=true"])):
        errs = []
        for k in sizes:
            case = vortex_case(int(L * k))
            d = study.run_cached(STUDY, f"{tag}_{k}", case, "ibm_method=ghost", *sets, mpi=args.mpi, force=args.force)
            errs.append(pressure_error(d, k))
        out[tag] = errs
        print(f"{tag:5s} mean relative pressure error at {sizes} cells per radius: " + "  ".join(f"{e:+.3e}" for e in errs))
    order = math.log2(abs(out["curv"][-2]) / abs(out["curv"][-1]))
    ratio = abs(out["curv"][-1]) / abs(out["plain"][-1])
    out["order"], out["ratio"] = order, ratio
    out["pass_error"] = bool(abs(out["curv"][-1]) <= BAND_ERR)
    out["pass_order"] = bool(order >= BAND_ORDER)
    out["pass_ratio"] = bool(ratio <= BAND_RATIO)
    nx, ny, h = 400, 600, 0.02   # the Mach 2 cylinder, 25 cells per diameter
    so = {}
    for tag, sets in (("plain", []), ("curv", ["ibm_curvature=true"])):
        d = study.run_cached(STUDY, f"mach2_{tag}", "IBM Supersonic Flow over Cylinder", "ibm_method=ghost", *sets, f"nx={nx}", f"ny={ny}",
                             "t_end=4", "save_interval=1000000000", mpi=args.mpi, force=args.force)
        so[tag] = standoff(d, nx, ny, h)
    out["standoff"] = so
    ok = so["plain"] is not None and so["curv"] is not None
    out["pass_standoff"] = bool(ok and BAND_STANDOFF[0] <= so["curv"] / so["plain"] <= BAND_STANDOFF[1])
    print(f"Mach 2 cylinder, t = 4: stand-off / D plain {so['plain']}, with the correction {so['curv']} "
          f"(ratio band {BAND_STANDOFF}, {'pass' if out['pass_standoff'] else 'FAIL'})")
    out["passed"] = out["pass_error"] and out["pass_order"] and out["pass_ratio"] and out["pass_standoff"]
    print(f"with the correction: finest error {abs(out['curv'][-1]):.2e} (band {BAND_ERR:.0e}, {'pass' if out['pass_error'] else 'FAIL'}); "
          f"order {order:.2f} (band {BAND_ORDER}, {'pass' if out['pass_order'] else 'FAIL'}); "
          f"against without {ratio:.3f} (band {BAND_RATIO}, {'pass' if out['pass_ratio'] else 'FAIL'})")
    fenice.write_summary(STUDY, out)
    print("ALL BANDS PASSED" if out["passed"] else "SOME BANDS FAILED")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
