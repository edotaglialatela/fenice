"""3-D sphere with the sharp ghost wall against the mask: drag at two resolutions.

    python cases/validation/features/ibm_sphere.py            # results/validation/ibm_sphere/  (10 and 20 cells per diameter)

A sphere of diameter 1 in a 6 x 3 x 3 channel (walls all round, 8.7 % blockage), Re = 20, Mach 0.3 (steady), run to t = 30.
No exact 3-D reference: the evidence is self-convergence. The staircase sphere (`mask`) has a resolution-dependent effective size
and its drag moves a lot between two meshes; the ghost wall (trilinear image points over eight cells) should move much less.
The drag of an unbounded sphere at Re = 20 is about 2.6 (Schiller-Naumann); the channel raises it, so this is a sanity range only.

Bands (chosen before the run, loosely; the result is marginal, 0.58 against 0.6, so this is weak evidence, not a convergence proof):
  1. both methods give a Cd in [2.0, 4.0] at the finer mesh (sanity);
  2. the change of Cd between the two meshes is at most 0.6 times the mask's change for the ghost wall.
"""
import numpy as np

import fenice
from fenice import study

STUDY = "ibm_sphere"
RE, MACH, T_END = 20.0, 0.3, 30.0
LX, LY, LZ, D = 6.0, 3.0, 3.0, 1.0
BAND_RANGE, BAND_RATIO = (2.0, 4.0), 0.6

CASE = """[case]
name = "Sphere Re20"
description = "sphere in a channel, Re = 20, Mach 0.3"
[mesh]
Lx = {Lx}
Ly = {Ly}
Lz = {Lz}
nx = {nx}
ny = {ny}
nz = {nz}
[physics]
use_viscous = true
mu = {mu}
use_bodyforce = false
[boundaries]
bc_left = "vel_inlet"
bc_right = "transmissive"
bc_bottom = "wall"
bc_top = "wall"
bc_front = "wall"
bc_back = "wall"
u_in = 1.0
rho_in = 1.0
p_out = {p}
[numerics]
riemann_solver = "HLLC (Low-Mach Thornber)"
use_muscl = true
muscl_limiter = "Van Leer"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.5
[run]
t_end = {t_end}
save_interval = 1000000000
monitor_interval = 50
[parallel]
use_mpi = false
[ibm]
use_ibm = true
ibm_shape = "Sphere"
ibm_radius = {R}
ibm_cx = 2.0
ibm_cy = 1.5
ibm_cz = 1.5
[initial]
state = {{ rho = 1.0, u = 1.0, v = 0.0, p = {p}, w = 0.0 }}
"""


def sphere_case(n):
    c = 1.0 / MACH   # u_in = 1 is Mach 0.3: sound speed 1/0.3
    return study.write_case(STUDY, f"sphere_{n}", CASE.format(
        Lx=LX, Ly=LY, Lz=LZ, nx=int(LX * n), ny=int(LY * n), nz=int(LZ * n), mu=1.0 * D / RE, p=c * c / 1.4,
        t_end=T_END, R=0.5 * D))


def final_cd(run_dir):
    d = fenice.read_monitor(study.case_output(run_dir))
    cd = np.asarray(d["Cd"])
    return float(cd[-1]), float(abs(cd[-1] - cd[-6]) / abs(cd[-1]))   # last value, and its change over the last five rows


def main():
    ap = study.parser(__doc__.splitlines()[0])
    args = ap.parse_args()
    sizes = (10, 20)
    out = {"sizes": list(sizes)}
    for method in ("mask", "ghost"):
        rows = []
        for n in sizes:
            d = study.run_cached(STUDY, f"{method}_{n}", sphere_case(n), f"ibm_method={method}", mpi=args.mpi, force=args.force)
            rows.append(final_cd(d))
        out[method] = rows
        print(f"{method:5s} Cd at {sizes} cells per diameter: " + "   ".join(f"{cd:.4f} (settling {s:.1e})" for cd, s in rows))
    dm = abs(out["mask"][1][0] - out["mask"][0][0])
    dg = abs(out["ghost"][1][0] - out["ghost"][0][0])
    out["change_mask"], out["change_ghost"], out["ratio"] = dm, dg, dg / dm if dm > 0 else float("inf")
    out["pass_range"] = bool(all(BAND_RANGE[0] <= out[m][1][0] <= BAND_RANGE[1] for m in ("mask", "ghost")))
    out["pass_ratio"] = bool(out["ratio"] <= BAND_RATIO)
    out["passed"] = out["pass_range"] and out["pass_ratio"]
    print(f"change of Cd between the meshes: mask {dm:.4f}, ghost {dg:.4f} (ratio {out['ratio']:.2f}, band {BAND_RATIO}, "
          f"{'pass' if out['pass_ratio'] else 'FAIL'}); range band {BAND_RANGE}: {'pass' if out['pass_range'] else 'FAIL'}")
    fenice.write_summary(STUDY, out)
    print("ALL BANDS PASSED" if out["passed"] else "SOME BANDS FAILED")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
