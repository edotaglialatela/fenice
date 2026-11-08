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


def final_cd(run_dir):
    d = fenice.read_monitor(study.case_output(run_dir))
    cd = np.asarray(d["Cd"])
    return float(cd[-1]), float(abs(cd[-1] - cd[-6]) / abs(cd[-1]))   # last value, and its change over the last five rows


def figure(args, sizes, out):
    """ibm_sphere.png: drag histories, drag against the resolution, and the speed on the mid-plane z = Lz/2 of the finest ghost run."""
    plt = study.use_style()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    col = {"mask": "#c0392b", "ghost": "#27ae60"}
    for method in ("mask", "ghost"):
        for n, ls in zip(sizes, (":", "-")):
            d = fenice.read_monitor(study.case_output(study.study_dir(STUDY) / f"{method}_{n}"))
            a1.plot(d["t"], d["Cd"], ls=ls, color=col[method], label=f"{method}, {n} cells / D")
    a1.set_ylim(2.5, 4.5)
    a1.set_xlabel("$t$")
    a1.set_ylabel("$C_d$")
    a1.set_title("Drag history")
    a1.legend(fontsize=8)
    for method in ("mask", "ghost"):
        a2.plot(1.0 / np.array(sizes), [r[0] for r in out[method]], "o-", color=col[method], ms=7, label=method)
    a2.set_xlabel("cell size / $D$")
    a2.set_ylabel("$C_d$ at $t = 30$")
    a2.set_xlim(0, 0.12)
    a2.set_title("Drag against resolution")
    a2.legend()
    run = study.case_output(study.study_dir(STUDY) / f"ghost_{sizes[-1]}")
    f = fenice.read_vtr(fenice.read_pvd(run)[-1][1])
    k = f["nz"] // 2
    v = f["Velocity"]
    speed = 0.5 * (np.linalg.norm(v[k - 1], axis=-1) + np.linalg.norm(v[k], axis=-1)) / 1.0   # cells k-1 and k straddle z = Lz/2
    m = a3.pcolormesh(f["x"], f["y"], speed, cmap="viridis", shading="flat", vmin=0.0, vmax=float(speed.max()))
    th = np.linspace(0, 2 * np.pi, 200)
    a3.plot(2.0 + 0.5 * D * np.cos(th), 1.5 + 0.5 * D * np.sin(th), "w-", lw=1)
    a3.set_aspect("equal")
    a3.set_xlabel("$x$")
    a3.set_ylabel("$y$")
    a3.set_title(f"$|\\mathbf{{u}}| / U$ on $z = L_z/2$ (ghost wall, {sizes[-1]} cells / D)")
    a3.grid(False)
    fig.colorbar(m, ax=a3, fraction=0.03, pad=0.02)
    plt.tight_layout()
    study.save(fig, STUDY, "ibm_sphere.png", args.docs)


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
    figure(args, sizes, out)
    print("ALL BANDS PASSED" if out["passed"] else "SOME BANDS FAILED")
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
