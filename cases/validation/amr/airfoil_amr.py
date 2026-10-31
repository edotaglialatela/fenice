"""AMR with an immersed body: the airfoil on 192 x 96 + two levels against uniform meshes.

    python cases/validation/amr/airfoil_amr.py             # results/validation/airfoil_amr/
    python cases/validation/amr/airfoil_amr.py --docs      # also docs/img/airfoil_amr.png
    python cases/validation/amr/airfoil_amr.py --table     # also the accuracy-per-cost table (design doc, section 20)

Runs "IBM Airfoil" on the 192 x 96 base mesh and on 768 x 384 (the resolution AMR keeps at
the body), and "IBM Airfoil AMR" (192 x 96, two levels, body at the finest level), all on
OpenMP threads to t = 6. Prints the cells, the wall time and the rms difference of Cd(t) and
Cl(t) from the 768 x 384 run (the loads the solver writes to the monitor), also for a uniform
384 x 192 run. Figure: airfoil_amr.png (vorticity with the refinement levels, load histories, and
the work-precision plot: wall time against load error, since runs are compared at equal accuracy,
not at equal mesh).
"""
import time

import numpy as np

import fenice
from fenice import study

STUDY, CASE, CASE_AMR = "airfoil_amr", "IBM Airfoil", "IBM Airfoil AMR"
T_END, BASE, LEVELS = 6.0, (192, 96), 2
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
        i0, j0 = int(round(x0 / h)), int(round(y0 / h))
        u[j0:j0 + f, i0:i0 + f] = U[k, 1] / U[k, 0]
        v[j0:j0 + f, i0:i0 + f] = U[k, 2] / U[k, 0]
    return u, v, h, meta


# Accuracy per cost (docs/design/amr-cell-based.md, section 20): uniform 384 x 192 and the cell engine at stricter indicator settings
TABLE = [("uniform 384x192", CASE, ("nx=384", "ny=192")),
         ("cell engine, default (criterion 0.1)", CASE_AMR, ()),
         ("cell engine, criterion 0.01, diffusion 2", CASE_AMR, ("amr_criterion=0.01", "amr_diffusion=2")),
         ("cell engine, criterion 0.01, diffusion 4", CASE_AMR, ("amr_criterion=0.01", "amr_diffusion=4")),
         ("cell engine, criterion 0.005, diffusion 4", CASE_AMR, ("amr_criterion=0.005", "amr_diffusion=4")),
         ("cell engine, criterion 0.002, diffusion 4", CASE_AMR, ("amr_criterion=0.002", "amr_diffusion=4"))]


def load_errors(run, fine):
    """rms difference of Cd(t), Cl(t) from the fine uniform run, for t >= 1."""
    tf, cdf, clf = loads(fine)
    t, cd, cl = loads(run)
    m = tf >= 1.0
    return (np.sqrt(np.mean((np.interp(tf[m], t, cd) - cdf[m]) ** 2)),
            np.sqrt(np.mean((np.interp(tf[m], t, cl) - clf[m]) ** 2)))


def table(args, fine, w_f):
    """Cells at the end, wall time and load errors of the TABLE runs (and the two uniform references)."""
    print(f"\n{'run':44s} {'cells':>8s} {'wall [s]':>9s} {'Cd':>8s} {'Cl':>8s}")
    print(f"{'uniform 768x384 (reference)':44s} {BASE[0] * BASE[1] * 4 ** LEVELS:8d} {w_f:9.1f}")
    for k, (label, case, sets) in enumerate(TABLE):
        tag = "uniform_mid" if case == CASE else ("amr" if not sets else f"table{k}")
        run, wall = timed(args, tag, case, *sets)
        cells = 384 * 192 if case == CASE else len(fenice.read_dump(run / "final.npz")["U"])
        e_d, e_l = load_errors(run, fine)
        print(f"{label:44s} {cells:8d} {wall:9.1f} {e_d:8.4f} {e_l:8.4f}")


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--table", action="store_true", help="also the accuracy-per-cost table (section 20 of the design doc)")
    args = ap.parse_args()
    plt = study.use_style()
    coarse, w_c = timed(args, "uniform_coarse", CASE, f"nx={BASE[0]}", f"ny={BASE[1]}")
    fine, w_f = timed(args, "uniform_fine", CASE, f"nx={BASE[0] * 2 ** LEVELS}", f"ny={BASE[1] * 2 ** LEVELS}")
    mid, w_m = timed(args, "uniform_mid", CASE, "nx=384", "ny=192")
    amr, w_a = timed(args, "amr", CASE_AMR)
    cells_c, cells_f = BASE[0] * BASE[1], BASE[0] * BASE[1] * 4 ** LEVELS
    u, v, h, meta = fine_velocity(amr)
    cells_a = len(meta)
    runs = [("Uniform 192$\\times$96", coarse, "#95a5a6", cells_c, w_c),
            ("Uniform 768$\\times$384", fine, "#2980b9", cells_f, w_f),
            ("AMR, 192$\\times$96 + 2 levels", amr, "#e74c3c", cells_a, w_a)]

    for label, run, _, cells, wall in runs + [("Uniform 384$\\times$192", mid, "", 384 * 192, w_m)]:
        e_d, e_l = load_errors(run, fine)
        print(f"{label.replace('$', '').replace(chr(92) + 'times', 'x'):32s} {cells:7d} cells {wall:7.1f} s   "
              f"rms vs 768x384, t >= 1: Cd {e_d:.4f}  Cl {e_l:.4f}")

    fig = plt.figure(figsize=(11, 8.6))
    a0 = fig.add_axes([0.06, 0.5, 0.9, 0.45])
    w = np.gradient(v, h, axis=1) - np.gradient(u, h, axis=0)
    a0.imshow(w, origin="lower", extent=(0, 12, 0, 6), cmap="RdBu_r", vmin=-8, vmax=8, interpolation="bilinear")
    from matplotlib.collections import PatchCollection
    colors = {1: "#f39c12", 2: "#c0392b"}
    for lv in (1, 2):   # the refined leaves; level 0 is everything else
        boxes = [plt.Rectangle((x0, y0), x1 - x0, y1 - y0) for x0, x1, y0, y1, l in meta
                 if int(l) == lv and 1.0 <= x1 and x0 <= 9.0 and 1.5 <= y1 and y0 <= 4.5]
        a0.add_collection(PatchCollection(boxes, facecolor="none", edgecolor=colors[lv], linewidths=0.5))
    a0.set_xlim(1.0, 9.0)
    a0.set_ylim(1.5, 4.5)
    a0.set_xlabel("$x$")
    a0.set_ylabel("$y$")
    a0.grid(False)
    a0.set_title(rf"NACA 0012, Re = 1000, $t = {T_END:g}$: vorticity and the refined leaves "
                 rf"(level 1 orange, 2 red; {cells_a} leaves)", fontsize=12)
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
    # work-precision: wall time against the load error (the right comparison is at equal accuracy)
    a3 = fig.add_axes([0.78, 0.07, 0.19, 0.32])
    for label, run, c, cells, wall in runs + [("Uniform 384$\\times$192", mid, "#27ae60", 384 * 192, w_m)]:
        if run is fine:
            continue
        e_d, e_l = load_errors(run, fine)
        a3.plot(wall, e_l, "o", color=c, ms=9, mec="white", label=label)
        a3.plot(wall, e_d, "s", color=c, ms=7, mec="white")
    a3.axvline(w_f, color="#2980b9", ls=":", lw=1.2)
    a3.text(w_f, 1.0, r" 768$\times$384 ref.", color="#2980b9", fontsize=7, va="top", ha="right", transform=a3.get_xaxis_transform())
    a3.legend(loc="upper right", fontsize=6, frameon=False, bbox_to_anchor=(1, 0.9))
    a3.set_xscale("log")
    a3.set_yscale("log")
    a3.set_xlabel("wall time [s]")
    a3.set_ylabel(r"rms error vs 768$\times$384 (circle $C_l$, square $C_d$)", fontsize=8)
    study.save(fig, STUDY, "airfoil_amr.png", args.docs)
    if args.table:
        table(args, fine, w_f)


if __name__ == "__main__":
    main()
