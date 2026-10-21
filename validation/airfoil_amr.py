"""AMR with an immersed body: the airfoil on 192 x 96 + two levels against uniform meshes.

    python validation/airfoil_amr.py             # results/validation/airfoil_amr/
    python validation/airfoil_amr.py --docs      # also docs/img/airfoil_amr.png

Runs "IBM Airfoil" on the 192 x 96 base mesh and on 768 x 384 (the resolution AMR keeps at
the body), and "IBM Airfoil AMR" (192 x 96, 16 x 16 patches, two levels, body at the finest
level), all on OpenMP threads to t = 6. Prints the cells, the wall time and the rms
difference of Cd(t) and Cl(t) from the 768 x 384 run (the loads the solver writes to the
monitor). Figure: airfoil_amr.png (vorticity with the patches, load histories, cost).
"""
import time

import numpy as np

import fenice
from fenice import study

STUDY, CASE, CASE_AMR = "airfoil_amr", "IBM Airfoil", "IBM Airfoil AMR"
T_END, BASE, PATCH, LEVELS = 6.0, (192, 96), 16, 2
COMMON = (f"t_end={T_END}", "monitor_interval=10", "save_interval=1000000000", "use_mpi=false")


def timed(args, tag, case, *sets):
    """Run (or reuse) a cached run; the wall time of the solver is stored next to it."""
    d = study.study_dir(STUDY) / tag
    wall = d / "wall.txt"
    if args.force or not (d / "final.npz").exists() or not wall.exists():
        t0 = time.perf_counter()
        study.run_cached(STUDY, tag, case, *sets, *COMMON, force=True)
        wall.write_text(f"{time.perf_counter() - t0:.2f}")
    return d, float(wall.read_text())


def loads(run):
    d = fenice.read_monitor(study.case_output(run))
    return d["t"], d["Cd"], d["Cl"]


def fine_velocity(run):
    """Velocity on the finest grid (each leaf cell repeated to the finest cells) and the leaves."""
    d = fenice.read_dump(run / "final.npz")
    U, meta = d["U"], d["leaf_meta"]
    nxf, nyf = BASE[0] * 2 ** LEVELS, BASE[1] * 2 ** LEVELS
    u = np.zeros((nyf, nxf))
    v = np.zeros((nyf, nxf))
    h = 12.0 / nxf
    for k, (x0, x1, y0, y1, lv) in enumerate(meta):
        f = 2 ** (LEVELS - int(lv))
        c = U[k * PATCH * PATCH:(k + 1) * PATCH * PATCH]
        uu = np.repeat(np.repeat((c[:, 1] / c[:, 0]).reshape(PATCH, PATCH), f, 0), f, 1)
        vv = np.repeat(np.repeat((c[:, 2] / c[:, 0]).reshape(PATCH, PATCH), f, 0), f, 1)
        i0, j0 = int(round(x0 / h)), int(round(y0 / h))
        u[j0:j0 + PATCH * f, i0:i0 + PATCH * f] = uu
        v[j0:j0 + PATCH * f, i0:i0 + PATCH * f] = vv
    return u, v, h, meta


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    coarse, w_c = timed(args, "uniform_coarse", CASE, f"nx={BASE[0]}", f"ny={BASE[1]}")
    fine, w_f = timed(args, "uniform_fine", CASE, f"nx={BASE[0] * 2 ** LEVELS}", f"ny={BASE[1] * 2 ** LEVELS}")
    amr, w_a = timed(args, "amr", CASE_AMR)
    cells_c, cells_f = BASE[0] * BASE[1], BASE[0] * BASE[1] * 4 ** LEVELS
    u, v, h, meta = fine_velocity(amr)
    cells_a = len(meta) * PATCH * PATCH
    runs = [("Uniform 192$\\times$96", coarse, "#95a5a6", cells_c, w_c),
            ("Uniform 768$\\times$384", fine, "#2980b9", cells_f, w_f),
            ("AMR, 192$\\times$96 + 2 levels", amr, "#e74c3c", cells_a, w_a)]

    tf, cdf, clf = loads(fine)
    for label, run, _, cells, wall in runs:
        t, cd, cl = loads(run)
        m = t >= 1.0
        e_d = np.sqrt(np.mean((np.interp(tf[tf >= 1.0], t, cd) - cdf[tf >= 1.0]) ** 2))
        e_l = np.sqrt(np.mean((np.interp(tf[tf >= 1.0], t, cl) - clf[tf >= 1.0]) ** 2))
        print(f"{label.replace('$', '').replace(chr(92) + 'times', 'x'):32s} {cells:7d} cells {wall:7.1f} s   "
              f"rms vs 768x384, t >= 1: Cd {e_d:.4f}  Cl {e_l:.4f}")

    fig = plt.figure(figsize=(11, 8.6))
    a0 = fig.add_axes([0.06, 0.5, 0.9, 0.45])
    w = np.gradient(v, h, axis=1) - np.gradient(u, h, axis=0)
    a0.imshow(w, origin="lower", extent=(0, 12, 0, 6), cmap="RdBu_r", vmin=-8, vmax=8, interpolation="bilinear")
    colors = {0: "#bdc3c7", 1: "#f39c12", 2: "#c0392b"}
    for x0, x1, y0, y1, lv in meta:
        a0.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.7, ec=colors[int(lv)]))
    a0.set_xlim(1.0, 9.0)
    a0.set_ylim(1.5, 4.5)
    a0.set_xlabel("$x$")
    a0.set_ylabel("$y$")
    a0.grid(False)
    a0.set_title(rf"NACA 0012, Re = 1000, $t = {T_END:g}$: vorticity and the 16$\times$16 patches "
                 rf"(level 0 grey, 1 orange, 2 red; {cells_a} cells)", fontsize=12)
    a1 = fig.add_axes([0.06, 0.07, 0.27, 0.32])
    a2 = fig.add_axes([0.40, 0.07, 0.27, 0.32])
    for label, run, c, _, _ in runs:
        t, cd, cl = loads(run)
        a1.plot(t, cd, color=c, lw=1.5, label=label)
        a2.plot(t, cl, color=c, lw=1.5)
    a1.set_xlabel("$t$")
    a1.set_ylabel("$C_d$")
    a1.legend(loc="upper right", fontsize=8)
    a2.set_xlabel("$t$")
    a2.set_ylabel("$C_l$")
    a3 = fig.add_axes([0.76, 0.07, 0.2, 0.32])
    xs = np.arange(3)
    a3.bar(xs, [r[4] for r in runs], color=[r[2] for r in runs], width=0.6)
    a3.set_yscale("log")
    a3.set_ylabel("wall time [s]")
    a3.set_xticks(xs)
    a3.set_xticklabels([f"{r[3]:,} cells".replace(",", " ") for r in runs], fontsize=8)
    a3.grid(False, axis="x")
    for x, r in zip(xs, runs):
        a3.text(x, r[4] * 1.15, f"{r[4]:.0f} s", ha="center", fontsize=9)
    a3.set_ylim(top=max(r[4] for r in runs) * 3)
    study.save(fig, STUDY, "airfoil_amr.png", args.docs)


if __name__ == "__main__":
    main()
