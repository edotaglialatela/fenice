"""Sod shock tube with AMR against uniform meshes at the two finest AMR levels.

    python cases/validation/amr/sod_amr.py            # results/validation/sod_amr/
    python cases/validation/amr/sod_amr.py --docs     # also docs/img/sod_amr.png

Runs the "Sod Shock Tube" case (1st-order Godunov, t = 0.2) with AMR (base mesh 10 x 2, 4
levels, pre-refined at t = 0, remeshed every 2 steps) and on uniform meshes at the two finest AMR
levels (80 x 16 and 160 x 32). Compares the density on the tube axis with the exact solution and prints the L1
error and the cell count of each run. AMR runs are serial (threads only).
Figure: sod_amr.png.
"""
import numpy as np

import fenice
from fenice import study
from fenice.exact import riemann

STUDY, CASE = "sod_amr", "Sod Shock Tube"
PX, PY, LEVELS = 10, 2, 4   # base mesh and levels
COMMON = ("save_interval=1000000000",)


def uniform(run_dir, nx, ny):
    d = fenice.read_dump(run_dir / "final.npz")
    rho = d["U"][:, 0].reshape(ny, nx).mean(axis=0)
    return (np.arange(nx) + 0.5) / nx, rho, np.full(nx, 1.0 / nx), nx * ny, float(d["t"])


def amr(run_dir):
    """Leaf columns sorted by x: centre, density averaged over y, width, plus the leaf levels."""
    d = fenice.read_dump(run_dir / "final.npz")
    U, meta = d["U"], d["leaf_meta"]
    x0, x1, y0, y1, level = meta.T
    xs, w, hgt = 0.5 * (x0 + x1), x1 - x0, y1 - y0
    # several leaves share an x column (stacked in y): average them, weighted by their height
    key = np.round(np.stack([xs, w]), 12)
    cols, inv = np.unique(key, axis=1, return_inverse=True)
    inv = inv.ravel()
    avg = lambda a: np.bincount(inv, a * hgt) / np.bincount(inv, hgt)
    return cols[0], avg(U[:, 0]), cols[1], len(U), float(d["t"]), avg(level), meta


def l1(x, rho, w, t):
    return np.sum(np.abs(rho - riemann(x, t)[0]) * w)


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    level = lambda lv, tag: uniform(study.run_cached(STUDY, tag, CASE, f"nx={PX * 2 ** lv}", f"ny={PY * 2 ** lv}",
                                                     *COMMON, mpi=args.mpi, force=args.force),
                                    PX * 2 ** lv, PY * 2 ** lv)
    coarse, fine = level(LEVELS - 1, "uniform_coarse"), level(LEVELS, "uniform_fine")
    a = amr(study.run_cached(STUDY, "amr", CASE, "use_amr=true", f"nx={PX}", f"ny={PY}",
                             f"amr_max_level={LEVELS}", f"amr_init_levels={LEVELS}", "amr_refine_every=2", *COMMON, force=args.force))
    t = fine[4]
    size = lambda lv: f"{PX * 2 ** lv}$\\times${PY * 2 ** lv}"
    runs = [(f"Uniform {size(LEVELS - 1)}", coarse, "#95a5a6", "s"),
            (f"Uniform {size(LEVELS)}", fine, "#2980b9", "o"),
            (f"AMR, {LEVELS} levels", a, "#e74c3c", "^")]
    errs = [l1(r[0], r[1], r[2], t) for _, r, _, _ in runs]
    for (label, r, _, _), e in zip(runs, errs):
        name = label.replace("$\\times$", "x")
        print(f"{name:24s} {r[3]:6d} cells, L1 density error {e:.3e}")

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=(2.1, 1), height_ratios=(4, 1), hspace=0.08, wspace=0.28)
    ax, axl, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    fig.suptitle(f"Sod shock tube with AMR ($t = {t:g}$, 1st-order Godunov)", fontsize=15, fontweight="bold")
    xe = np.linspace(0, 1, 2000)
    ax.plot(xe, riemann(xe, t)[0], "k-", lw=1.4, label="Exact (Toro)", zorder=5)
    for label, r, c, m in runs:
        ax.plot(r[0], r[1], ls="none", marker=m, ms=3.5, mfc="none", mec=c, mew=0.9,
                label=label)
    ax.set_ylabel(r"Density $\rho$")
    ax.set_xlim(0, 1)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="lower left")
    axl.step(a[0], a[5], where="mid", color="#e74c3c", lw=1.6)
    axl.set_yticks(range(LEVELS + 1))
    axl.set_ylim(-0.4, LEVELS + 0.4)
    axl.set_xlim(0, 1)
    axl.set_xlabel("$x$")
    axl.set_ylabel("AMR level")

    pos = np.arange(len(runs))
    bars = axb.bar(pos, [r[3] for _, r, _, _ in runs], 0.55, color=[c for *_, c, _ in runs], alpha=0.75,
                   edgecolor="black")
    for b, (_, r, _, _) in zip(bars, runs):
        axb.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r[3]}", ha="center", va="bottom", fontweight="bold")
    axb.set_xticks(pos, [f"Uniform\nlevel {LEVELS - 1}", f"Uniform\nlevel {LEVELS}", "AMR"])
    axb.set_ylabel("Cells at $t$ = 0.2")
    axb.grid(False)
    ae = axb.twinx()
    ae.plot(pos, errs, "D--", color="#8e44ad", ms=9, mec="white", lw=1.8)
    for p, e in zip(pos, errs):
        ae.annotate(f"{e:.1e}", (p, e), textcoords="offset points", xytext=(8, 6), color="#8e44ad", fontsize=10)
    ae.set_ylabel(r"$L_1$ density error", color="#8e44ad")
    ae.set_ylim(0, 1.25 * max(errs))
    ae.tick_params(axis="y", labelcolor="#8e44ad")
    axb.set_title("Cells and error")
    study.save(fig, STUDY, "sod_amr.png", args.docs)


if __name__ == "__main__":
    main()
