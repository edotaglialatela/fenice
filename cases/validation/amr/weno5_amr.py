"""WENO5-Z on the adaptive mesh against WENO3-Z on the same mesh: vortex, shock tubes, cavity, cylinder, cost.

    python cases/validation/amr/weno5_amr.py            # results/validation/weno5_amr/
    python cases/validation/amr/weno5_amr.py --only vortex sod

On an adaptive mesh `reconstruction = weno5` uses its six cells where they are fluid leaves of the face's level and the four-cell
WENO3-Z stencil at and next to the level jumps and the immersed body. The study runs both schemes (SSP-RK3) with the same settings and
the indicator of the case, so the meshes follow the same features. Bands, fixed before the first run:

  vortex     isentropic vortex, one period, base 40 + one level: L2 density error of WENO5-Z at most that of WENO3-Z (ratio <= 1.0);
             the ratio to the uniform WENO5-Z run on 80 cells is reported.
  sod        Sod tube, base 100 + one level, exact solution: L1 density error at most 1.05 times that of WENO3-Z; over- and undershoot
             of the density not more than 1e-3 of the jump above those of WENO3-Z.
  shu_osher  shock / entropy wave, base 200 + one level, against the uniform WENO5-Z run on 3200 cells: same L1 band.
  cavity     lid-driven cavity, base 32 + one level, boundary_order = 2: RMS error of the centreline velocities against the
             tabulated reference, both schemes (reported).
  cylinder   Re = 40, base 100 x 50 + two levels (20 finest cells per diameter), ghost wall and mask: drag coefficient against the
             uniform 400 x 200 runs of the same schemes (reported).
  cost       wall time per step, WENO5-Z over WENO3-Z, on one thread, 16 threads and 4 ranks (reported).

A result outside a band is printed as OUTSIDE and the script exits 1. Table: results/validation/weno5_amr/weno5_amr.md.
"""
import sys
import time
from pathlib import Path

import numpy as np

import fenice
from fenice import study
from fenice.exact import riemann

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "verification"))
sys.path.insert(0, str(HERE.parent / "benchmarks"))
import cavity as C   # the tabulated centreline velocities
import isentropic_vortex as V   # exact density

STUDY = "weno5_amr"
RK3 = "time_scheme=SSP Runge-Kutta 3"
SCHEMES = {"WENO3-Z": "reconstruction=weno3", "WENO5-Z": "reconstruction=weno5"}
QUIET = "save_interval=1000000000"
# drag of the uniform 400 x 200 runs (t = 25, SSP-RK3): wall treatment -> scheme -> Cd
UNIFORM_CD = {"ghost": {"WENO3-Z": 1.90473, "WENO5-Z": 1.89881}, "mask": {"WENO3-Z": 1.84870, "WENO5-Z": 1.82219}}
CYLINDER = """[case]
name = "Cylinder Re40 Adaptive"
description = "IBM cylinder at Re = 40, adaptive mesh"
gui = false
requires_viscous = true

[mesh]
Lx = 4.0
Ly = 2.0
nx = 100
ny = 50

[physics]
use_viscous = true
mu = 0.005
use_bodyforce = false

[boundaries]
bc_left = "vel_inlet"
bc_right = "transmissive"
bc_bottom = "wall"
bc_top = "wall"
u_in = 1.0
rho_in = 1.0
p_out = 9.8

[numerics]
riemann_solver = "HLLC (Low-Mach Thornber)"
reconstruction = "weno3"
time_scheme = "SSP Runge-Kutta 3"
cfl = 0.5

[run]
t_end = 25.0
save_interval = 1000000000

[parallel]
use_mpi = false

[amr]
use_amr = true
amr_max_level = 2
amr_refine_every = 5
amr_refine_ibm = true

[ibm]
use_ibm = true
ibm_shape = "Cylinder"
ibm_radius = 0.1
ibm_cx = 1.0
ibm_cy = 1.0

[initial]
state = { rho = 1.0, u = 1.0, v = 0.0, p = 9.8 }
"""


def tag(*parts):
    return "_".join(parts).replace(" ", "_").replace("-", "").lower()


def leaves(run_dir):
    """Centres, areas and primitives of the leaves of an adaptive run (or the cells of a uniform one given nx)."""
    z = fenice.read_dump(run_dir / "final.npz")
    m = z["leaf_meta"]
    x, y, w = 0.5 * (m[:, 0] + m[:, 1]), 0.5 * (m[:, 2] + m[:, 3]), (m[:, 1] - m[:, 0]) * (m[:, 3] - m[:, 2])
    return x, y, w, fenice.primitive(z["U"])


def vortex(args, out):
    n, err = 40, {}
    for name, rec in SCHEMES.items():
        d = study.run_cached(STUDY, tag("vortex", name), "Isentropic Vortex", f"nx={n}", f"ny={n}", f"t_end={V.T_END}", QUIET,
                             "use_amr=true", "amr_max_level=1", rec, RK3, mpi=args.mpi, force=args.force)
        x, y, w, prim = leaves(d)
        err[name] = (np.sqrt(np.sum(w * (prim[0] - V.exact_rho(x, y, V.T_END)) ** 2) / np.sum(w)), len(x))
    d = study.run_cached(STUDY, "vortex_uniform80_weno5", "Isentropic Vortex", "nx=80", "ny=80", f"t_end={V.T_END}", QUIET,
                         SCHEMES["WENO5-Z"], RK3, mpi=0, force=args.force)
    rho = fenice.read_dump(d / "final.npz")["U"][:, 0].reshape(80, 80)
    xc = (np.arange(80) + 0.5) * V.L / 80
    X, Y = np.meshgrid(xc, xc)
    uni = np.sqrt(np.mean((rho - V.exact_rho(X, Y, V.T_END)) ** 2))
    ratio = err["WENO5-Z"][0] / err["WENO3-Z"][0]
    ok = ratio <= 1.0
    out.append(f"| vortex | L2 density error | {err['WENO3-Z'][0]:.3e} ({err['WENO3-Z'][1]} leaves) | {err['WENO5-Z'][0]:.3e} "
               f"({err['WENO5-Z'][1]} leaves) | {ratio:.3f} | {'ok' if ok else 'OUTSIDE'} (<= 1.0); uniform 80 WENO5-Z {uni:.3e}, "
               f"ratio {err['WENO5-Z'][0] / uni:.2f} |")
    return ok


def tube(args, out, name, case, base, t_end, exact):
    """Sod / Shu-Osher on base + one level; exact(x) is the reference density."""
    res = {}
    for label, rec in SCHEMES.items():
        d = study.run_cached(STUDY, tag(name, label), case, f"nx={base}", "ny=2", f"Ly={2.0 * exact.L / base}", f"t_end={t_end}", QUIET,
                             "use_amr=true", "amr_max_level=1", rec, RK3, mpi=args.mpi, force=args.force)
        x, _, w, prim = leaves(d)
        res[label] = (np.sum(w * np.abs(prim[0] - exact(x))) / np.sum(w), prim[0].max(), prim[0].min(), len(x))
    ratio = res["WENO5-Z"][0] / res["WENO3-Z"][0]
    ok = ratio <= 1.05
    note = f"{'ok' if ok else 'OUTSIDE'} (<= 1.05)"
    if exact.bounds:
        lo, hi = exact.bounds
        over = [max(res[s][1] - hi, 0.0) + max(lo - res[s][2], 0.0) for s in ("WENO3-Z", "WENO5-Z")]
        more = (over[1] - over[0]) / (hi - lo)
        ok_over = more <= 1e-3
        ok = ok and ok_over
        note += f"; over/undershoot / jump {over[0] / (hi - lo):.2e} -> {over[1] / (hi - lo):.2e}: {'ok' if ok_over else 'OUTSIDE'} (+1e-3)"
    out.append(f"| {name} | L1 density error | {res['WENO3-Z'][0]:.3e} ({res['WENO3-Z'][3]} leaves) | {res['WENO5-Z'][0]:.3e} "
               f"({res['WENO5-Z'][3]} leaves) | {ratio:.3f} | {note} |")
    return ok


def sod(args, out):
    def exact(x):
        return riemann(x, 0.2)[0]
    exact.L, exact.bounds = 1.0, (0.125, 1.0)
    return tube(args, out, "sod", "Sod Shock Tube", 100, 0.2, exact)


def shu_osher(args, out):
    n = 3200
    d = study.run_cached(STUDY, "shu_osher_uniform3200_weno5", "Shu-Osher Test", f"nx={n}", "ny=2", f"Ly={20.0 / n}", QUIET,
                         SCHEMES["WENO5-Z"], RK3, mpi=0, force=args.force)
    ref = fenice.read_dump(d / "final.npz")["U"][:n, 0]
    xr = (np.arange(n) + 0.5) * 10.0 / n

    def exact(x):
        return np.interp(x, xr, ref)
    exact.L, exact.bounds = 10.0, None
    return tube(args, out, "shu_osher", "Shu-Osher Test", 200, 1.8, exact)


def cavity(args, out):
    from scipy.interpolate import griddata
    res = {}
    for label, rec in SCHEMES.items():
        d = study.run_cached(STUDY, tag("cavity", label), C.CASE, "nx=32", "ny=32", "t_end=40", QUIET, "use_amr=true", "amr_max_level=1",
                             "boundary_order=2", rec, RK3, mpi=args.mpi, force=args.force)
        x, y, _, prim = leaves(d)
        inner = slice(1, -1)
        vk = ~C.V_SUSPECT
        vk[0] = vk[-1] = False
        pts = np.column_stack([x, y])
        uc = griddata(pts, prim[1], (np.full(len(C.GHIA_Y[inner]), 0.5), C.GHIA_Y[inner]), method="linear")
        vc = griddata(pts, prim[2], (C.GHIA_X[vk], np.full(int(vk.sum()), 0.5)), method="linear")
        res[label] = (np.sqrt(np.mean((uc - C.GHIA_U[inner]) ** 2)), np.sqrt(np.mean((vc - C.GHIA_V[vk]) ** 2)), len(x))
    a, b = res["WENO3-Z"], res["WENO5-Z"]
    out.append(f"| cavity | RMS centreline u / v | {a[0]:.2e} / {a[1]:.2e} ({a[2]} leaves) | {b[0]:.2e} / {b[1]:.2e} ({b[2]} leaves) | "
               f"{b[0] / a[0]:.3f} / {b[1] / a[1]:.3f} | reported |")
    return True


def cylinder(args, out):
    case = study.write_case(STUDY, "cylinder_re40_adaptive", CYLINDER)
    for wall in ("ghost", "mask"):
        cd = {}
        for label, rec in SCHEMES.items():
            d = study.run_cached(STUDY, tag("cylinder", wall, label), str(case), f"ibm_method={wall}", "monitor_interval=50", rec, mpi=args.mpi, force=args.force)
            mon = fenice.read_monitor(study.case_output(d))
            cd[label] = (mon["Cd"][-1], len(fenice.read_dump(d / "final.npz")["U"]))
        u = UNIFORM_CD[wall]
        out.append(f"| cylinder, {wall} | Cd (uniform 400 x 200: {u['WENO3-Z']:.4f}, {u['WENO5-Z']:.4f}) | {cd['WENO3-Z'][0]:.4f} "
                   f"({cd['WENO3-Z'][0] / u['WENO3-Z'] - 1:+.2%}, {cd['WENO3-Z'][1]} leaves) | {cd['WENO5-Z'][0]:.4f} "
                   f"({cd['WENO5-Z'][0] / u['WENO5-Z'] - 1:+.2%}, {cd['WENO5-Z'][1]} leaves) | {cd['WENO5-Z'][0] / cd['WENO3-Z'][0]:.4f} | "
                   f"reported; uniform ratio {u['WENO5-Z'] / u['WENO3-Z']:.4f} |")
    return True


def cost(args, out):
    steps = 300
    sets = ["Lx=1.0", "Ly=1.0", "nx=128", "ny=128", "use_amr=true", "amr_max_level=2", "amr_init_levels=1", "amr_refine_every=5",
            f"max_steps={steps}", "t_end=1e9", QUIET, RK3]
    for label, kw in (("1 thread", dict(threads=1)), ("16 threads", dict(threads=16)), ("4 ranks", dict(mpi=4))):
        t = {}
        for name, rec in SCHEMES.items():
            best = np.inf
            for _ in range(2):
                t0 = time.perf_counter()
                fenice.run("Sod Shock Tube", *sets, rec, out_root=study.study_dir(STUDY) / "cost", **kw)
                best = min(best, time.perf_counter() - t0)
            t[name] = best / steps
        out.append(f"| cost, {label} | wall time per step (whole run / {steps}) | {t['WENO3-Z'] * 1e3:.2f} ms | {t['WENO5-Z'] * 1e3:.2f} ms | "
                   f"{t['WENO5-Z'] / t['WENO3-Z']:.2f} | reported |")
    return True


PARTS = {"vortex": vortex, "sod": sod, "shu_osher": shu_osher, "cavity": cavity, "cylinder": cylinder, "cost": cost}


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=list(PARTS), default=list(PARTS), help="parts to run")
    args = ap.parse_args()
    rows = ["| part | quantity | WENO3-Z | WENO5-Z | WENO5-Z / WENO3-Z | band |", "|---|---|---|---|---|---|"]
    ok = True
    for name in args.only:
        n0 = len(rows)
        ok = PARTS[name](args, rows) and ok
        print("\n".join(rows[n0:]), flush=True)
    path = study.study_dir(STUDY) / ("weno5_amr.md" if len(args.only) == len(PARTS) else "weno5_amr_partial.md")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {path}")
    print("result: " + ("all bands met" if ok else "OUTSIDE a band (see above)"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
