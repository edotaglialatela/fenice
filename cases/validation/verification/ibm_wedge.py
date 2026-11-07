"""Immersed wall against an exact solution: Mach 2 flow along a 10-degree half-angle wedge (oblique shock), mask against ghost.

    python cases/validation/verification/ibm_wedge.py            # results/validation/ibm_wedge/  (50, 100, 200 cells per unit length)
    python cases/validation/verification/ibm_wedge.py --full     # adds 400

The wedge is a polygon body (`ibm_shape = Airfoil` with a three-point coordinate file) whose faces are inclined 10 degrees to the
grid, so every wall cell is cut. The inviscid flow past it has the exact attached oblique shock (theta-beta-M relation): the state
between the shock and the face is uniform, p2/p1 = 1.7066 and rho2/rho1 = 1.4584 at Mach 2, beta = 39.31 degrees. The error is read
in the cells 2.5 to 5 cells from the upper face between x = 1.0 and 1.5 (the nose is at x = 0.75, the shoulder at 1.75, so the
fan from the shoulder has not reached them). Slip wall (`use_viscous = false`), HLLC, MUSCL (Minmod), SSP-RK2, CFL 0.4.

The ibm_curvature=true variant is run too: on a flat wall it must not do worse (its error at most twice the plain ghost one).

Bands (set after an exploratory run at 100 / 200 / 400 cells, so they are a regression guard for the ghost wall, not a prediction):
  1. ghost: the RMS pressure error at the finest mesh is at most 5e-4 (relative to the exact p2) and the observed order between
     the two finest meshes is at least 2;
  2. the ghost error at the finest mesh is at most a tenth of the mask error.
The mask is reported, not banded: its staircase leaves a wall layer whose density deficit does not shrink with the mesh.
"""
import math
from pathlib import Path

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_wedge"
GAMMA, MACH, THETA = 1.4, 2.0, math.radians(10.0)
P1 = 1.0 / GAMMA   # sound speed 1, so u = 2 is Mach 2
BAND_ERR, BAND_ORDER, BAND_RATIO = 5e-4, 2.0, 0.1

CASE = """[case]
name = "Wedge Mach 2"
description = "Mach 2 flow along a 10 degree wedge, exact oblique shock"
[mesh]
Lx = 4.0
Ly = 2.0
nx = 200
ny = 100
[physics]
use_viscous = false
use_bodyforce = false
[boundaries]
bc_left = "vel_inlet"
bc_right = "transmissive"
bc_bottom = "transmissive"
bc_top = "transmissive"
u_in = 2.0
rho_in = 1.0
p_out = {p}
[numerics]
riemann_solver = "HLLC"
use_muscl = true
muscl_limiter = "Minmod"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.4
[run]
t_end = 3.0
save_interval = 1000000000
[parallel]
use_mpi = false
[ibm]
use_ibm = true
ibm_shape = "Airfoil"
ibm_file = "{dat}"
ibm_chord = 1.0
ibm_aoa = 0.0
ibm_cx = 1.0
ibm_cy = 1.0
[initial]
state = {{ rho = 1.0, u = 2.0, v = 0.0, p = {p} }}
"""


def exact():
    """beta (weak branch), p2/p1, rho2/rho1 of the attached oblique shock."""
    g, M = GAMMA, MACH

    def f(b):
        return math.atan(2 / math.tan(b) * (M * M * math.sin(b) ** 2 - 1) / (M * M * (g + math.cos(2 * b)) + 2)) - THETA

    lo, hi = math.asin(1 / M) + 1e-9, math.pi / 2 - 1e-9
    bs = np.linspace(lo, hi, 20001)
    bm = bs[int(np.argmax([f(b) for b in bs]))]   # the weak branch lies between the Mach angle and the maximum-deflection angle
    a, b = lo, bm
    for _ in range(200):
        c = 0.5 * (a + b)
        a, b = (a, c) if f(a) * f(c) <= 0 else (c, b)
    beta = 0.5 * (a + b)
    mn2 = (M * math.sin(beta)) ** 2
    return beta, 1 + 2 * g / (g + 1) * (mn2 - 1), (g + 1) * mn2 / ((g - 1) * mn2 + 2)


def wedge_case():
    d = study.study_dir(STUDY) / "cases" / "wedge"
    d.mkdir(parents=True, exist_ok=True)
    t = math.tan(THETA)
    dat = d / "wedge.dat"
    dat.write_text(f"wedge\n1.0 {t:.12f}\n0.0 0.0\n1.0 {-t:.12f}\n")   # Selig order: upper tail, nose, lower tail
    return study.write_case(STUDY, "wedge", CASE.format(p=P1, dat=str(dat).replace("\\", "/")))


def errors(run_dir, n, pr, rr):
    """Relative pressure and density error of the cells 2.5-5 cells above the upper face, x in (1.0, 1.5)."""
    nx, ny, h = 4 * n, 2 * n, 1.0 / n
    rho, _, _, p = fenice.primitive(fenice.read_dump(run_dir / "final.npz")["U"])
    xs, ys = (np.arange(nx) + 0.5) * h, (np.arange(ny) + 0.5) * h
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    dn = ((Y - 1.0) - (X - 0.75) * math.tan(THETA)) * math.cos(THETA)   # normal distance to the upper face (nose at 0.75, 1)
    sel = (X > 1.0) & (X < 1.5) & (dn > 2.5 * h) & (dn < 5.0 * h)
    ep, er = p[sel] / P1 / pr - 1.0, rho[sel] / rr - 1.0
    return {"p_rms": float(np.sqrt(np.mean(ep ** 2))), "p_mean": float(ep.mean()), "rho_mean": float(er.mean())}


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="add the 400 cells per unit length mesh")
    args = ap.parse_args()
    beta, pr, rr = exact()
    print(f"exact oblique shock: beta = {math.degrees(beta):.3f} deg, p2/p1 = {pr:.5f}, rho2/rho1 = {rr:.5f}")
    case = wedge_case()
    sizes = (50, 100, 200, 400) if args.full else (50, 100, 200)
    out = {"beta_deg": math.degrees(beta), "p_ratio": pr, "rho_ratio": rr, "sizes": list(sizes)}
    for method in ("mask", "ghost", "ghost+curv"):
        rows = []
        for n in sizes:
            sets = ["ibm_method=" + method.split("+")[0], *(["ibm_curvature=true"] if "+" in method else [])]
            d = study.run_cached(STUDY, f"{method}_{n}", case, *sets, f"nx={4 * n}", f"ny={2 * n}",
                                 mpi=args.mpi, force=args.force)
            rows.append(errors(d, n, pr, rr))
        out[method] = rows
        for n, r in zip(sizes, rows):
            print(f"{method:5s} {n:4d}/unit  p error rms {r['p_rms']:.3e}  mean {r['p_mean']:+.3e}   rho error mean {r['rho_mean']:+.3e}")
    g = [r["p_rms"] for r in out["ghost"]]
    order = math.log2(g[-2] / g[-1])
    ratio = g[-1] / out["mask"][-1]["p_rms"]
    out["ghost_order"], out["ghost_to_mask"] = order, ratio
    out["pass_error"], out["pass_order"], out["pass_ratio"] = bool(g[-1] <= BAND_ERR), bool(order >= BAND_ORDER), bool(ratio <= BAND_RATIO)
    out["passed"] = out["pass_error"] and out["pass_order"] and out["pass_ratio"]
    # the curvature correction has nothing to correct on a flat wall (the polygon's curvature estimate is zero away from the corners)
    out["curv_to_ghost"] = out["ghost+curv"][-1]["p_rms"] / out["ghost"][-1]["p_rms"]
    out["pass_curv"] = bool(out["curv_to_ghost"] <= 2.0)
    out["passed"] = out["passed"] and out["pass_curv"]
    print(f"ghost+curv / ghost at the finest mesh: {out['curv_to_ghost']:.2f} (band 2, {'pass' if out['pass_curv'] else 'FAIL'})")
    print(f"ghost: finest rms {g[-1]:.2e} (band {BAND_ERR:.0e}, {'pass' if out['pass_error'] else 'FAIL'}); "
          f"order {order:.2f} (band {BAND_ORDER}, {'pass' if out['pass_order'] else 'FAIL'}); "
          f"ghost/mask {ratio:.2e} (band {BAND_RATIO}, {'pass' if out['pass_ratio'] else 'FAIL'})")
    fenice.write_summary(STUDY, out)
    print("ALL BANDS PASSED" if out["passed"] else "SOME BANDS FAILED")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
