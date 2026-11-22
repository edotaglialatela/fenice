"""3-D sphere with the ghost-cell wall: drag at two resolutions.

    python cases/validation/features/ibm_sphere.py            # results/validation/ibm_sphere/  (10 and 20 cells per diameter)

A sphere of diameter 1 in a 6 x 3 x 3 channel (walls all round, 8.7 % blockage), Re = 20, Mach 0.3 (steady), run to t = 30.
No exact 3-D reference: the evidence is self-convergence, how little the drag of the wall (trilinear image points over eight
cells) moves between two meshes. The drag of an unbounded sphere at Re = 20 is about 2.6 (Schiller-Naumann); the channel raises it,
so this is a sanity range only.

Bands:
  1. Cd in [2.0, 4.0] at the finer mesh (sanity, chosen before the run);
  2. the change of Cd between the two meshes is at most 3 % of the finer value (set after the run, which gave 1.5 %: a regression
     guard, not a prediction; two meshes are not a convergence proof).
"""
from pathlib import Path

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_sphere"
RE, MACH, T_END = 20.0, 0.3, 30.0
LX, LY, LZ, D = 6.0, 3.0, 3.0, 1.0
BAND_RANGE, BAND_CHANGE = (2.0, 4.0), 0.03

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
bc_xmin = "vel_inlet"
bc_xmax = "transmissive"
bc_ymin = "wall"
bc_ymax = "wall"
bc_zmin = "wall"
bc_zmax = "wall"
u_in = 1.0
rho_in = 1.0
p_out = {p}
[numerics]
riemann_solver = "HLLC (Low-Mach Thornber)"
reconstruction = "muscl"
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


def monitor_dir(run_dir):
    """The folder of the run that holds the monitor file."""
    return next(Path(run_dir).rglob("*_monitor.csv")).parent


def final_cd(run_dir):
    d = fenice.read_monitor(monitor_dir(run_dir))
    cd = np.asarray(d["Cd"])
    return float(cd[-1]), float(abs(cd[-1] - cd[-6]) / abs(cd[-1]))   # last value, and its change over the last five rows


def figure(args, sizes, out):
    """ibm_sphere.png: drag histories and drag against the resolution (the flow field is in amr/octree_sphere.py, at 80 cells per diameter)."""
    plt = study.use_style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    col = "#27ae60"
    for n, ls in zip(sizes, (":", "-")):
        d = fenice.read_monitor(monitor_dir(study.study_dir(STUDY) / f"ghost_{n}"))
        a1.plot(d["t"], d["Cd"], ls=ls, color=col, label=f"{n} cells / D")
    a1.set_ylim(2.5, 4.5)
    a1.set_xlabel("$t$")
    a1.set_ylabel("$C_d$")
    a1.set_title("Drag history")
    a1.legend(fontsize=8)
    a2.plot(1.0 / np.array(sizes), [r[0] for r in out["ghost"]], "o-", color=col, ms=7)
    a2.set_xlabel("cell size / $D$")
    a2.set_ylabel("$C_d$ at $t = 30$")
    a2.set_xlim(0, 0.12)
    a2.set_title("Drag against resolution")
    plt.tight_layout()
    study.save(fig, STUDY, "ibm_sphere.png", args.docs)


def main():
    ap = study.parser(__doc__.splitlines()[0])
    args = ap.parse_args()
    sizes = (10, 20)
    out = {"sizes": list(sizes)}
    rows = []
    for n in sizes:   # (the run folders keep their tag ghost_<n>: other studies read them)
        d = study.run_cached(STUDY, f"ghost_{n}", sphere_case(n), mpi=args.mpi, force=args.force)
        rows.append(final_cd(d))
    out["ghost"] = rows
    print(f"Cd at {sizes} cells per diameter: " + "   ".join(f"{cd:.4f} (settling {s:.1e})" for cd, s in rows))
    dg = abs(rows[1][0] - rows[0][0])
    out["change_ghost"], out["relative_change"] = dg, dg / abs(rows[1][0])
    out["pass_range"] = bool(BAND_RANGE[0] <= rows[1][0] <= BAND_RANGE[1])
    out["pass_change"] = bool(out["relative_change"] <= BAND_CHANGE)
    out["passed"] = out["pass_range"] and out["pass_change"]
    print(f"change of Cd between the meshes: {dg:.4f} ({100 * out['relative_change']:.2f} %, band {100 * BAND_CHANGE:g} %, "
          f"{'pass' if out['pass_change'] else 'FAIL'}); range band {BAND_RANGE}: {'pass' if out['pass_range'] else 'FAIL'}")
    fenice.write_summary(STUDY, out)
    figure(args, sizes, out)
    print("ALL BANDS PASSED" if out["passed"] else "SOME BANDS FAILED")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
