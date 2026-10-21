"""Sod shock tube against the exact Riemann solution: 1st order, 1st vs 2nd order, limiters.

    python cases/validation/sod.py            # results/validation/sod/
    python cases/validation/sod.py --docs     # also docs/img/sod_*.png

Runs the "Sod Shock Tube" case (200 x 20 cells, t = 0.2) with 1st-order Godunov, with MUSCL
+ SSP-RK2 for each limiter, and compares the profiles on the centre row with the exact
solution (Toro 2009). Prints the L1 density error of every run.
Figures: sod_exact.png, sod_orders.png, sod_limiters.png.
"""
import numpy as np

import fenice
from fenice import study
from fenice.exact import riemann

STUDY, CASE = "sod", "Sod Shock Tube"
LIMITERS = ["Minmod", "Van Leer", "MC", "Koren"]
MUSCL = ("use_muscl=true", "time_scheme=SSP Runge-Kutta 2")


def profile(run_dir, nx=200, ny=20):
    d = fenice.read_dump(run_dir / "final.npz")
    rho, u, _, p = fenice.primitive(d["U"])
    row = lambda a: a.reshape(ny, nx)[ny // 2]
    return (np.arange(nx) + 0.5) / nx, row(rho), row(u), row(p), float(d["t"])


def four_panels(plt, title, runs, t):
    """rho, u, p, T = p / rho against the exact solution; runs: [(label, style, (x, rho, u, p))]."""
    xe = np.linspace(0, 1, 2000)
    re, ue, pe = riemann(xe, t)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for ax, name, ex, k in ((axs[0, 0], r"Density $\rho$", re, 1), (axs[0, 1], "Velocity $u$", ue, 2),
                            (axs[1, 0], "Pressure $p$", pe, 3), (axs[1, 1], r"Temperature $T = p/\rho$", pe / re, 4)):
        ax.plot(xe, ex, "k-", lw=1.4, label="Exact (Toro)", zorder=5)
        for label, style, prof in runs:
            y = prof[3] / prof[1] if k == 4 else prof[k]
            ax.plot(prof[0], y, label=label, **style)
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_xlabel("$x$")
    axs[0, 0].legend(loc="lower left")
    plt.tight_layout()
    return fig


def l1(prof, t):
    re, _, _ = riemann(prof[0], t)
    return np.mean(np.abs(prof[1] - re))


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    run = lambda tag, *sets: profile(study.run_cached(STUDY, tag, CASE, "save_interval=1000000000", *sets,
                                                      mpi=args.mpi, force=args.force))
    first = run("godunov")
    second = run("muscl_vanleer", *MUSCL, "muscl_limiter=Van Leer")
    lims = {lim: run("muscl_" + lim.lower().replace(" ", ""), *MUSCL, f"muscl_limiter={lim}") for lim in LIMITERS}
    t = first[4]
    print(f"L1 density error at t = {t:g}: Godunov {l1(first, t):.2e}; " +
          ", ".join(f"MUSCL {k} {l1(v, t):.2e}" for k, v in lims.items()))

    dots = lambda c, m="o": dict(ls="none", marker=m, ms=3.5, mfc="none", mec=c, mew=0.9)
    fig = four_panels(plt, f"Sod shock tube, 200 cells, 1st-order Godunov ($t = {t:g}$)",
                      [("Godunov (1st order)", dots("#1f4e9c"), first[:4])], t)
    study.save(fig, STUDY, "sod_exact.png", args.docs)
    fig = four_panels(plt, f"Sod shock tube: 1st vs 2nd order ($t = {t:g}$)",
                      [("Godunov (1st order)", dots("#c0392b"), first[:4]),
                       ("MUSCL Van Leer + SSP-RK2", dots("#2980b9", "^"), second[:4])], t)
    study.save(fig, STUDY, "sod_orders.png", args.docs)
    colors = {"Minmod": "#e67e22", "Van Leer": "#27ae60", "MC": "#8e44ad", "Koren": "#16a085"}
    markers = {"Minmod": "o", "Van Leer": "s", "MC": "^", "Koren": "D"}
    fig = four_panels(plt, f"Sod shock tube: MUSCL limiters ($t = {t:g}$)",
                      [(f"{k} (L1 {l1(v, t):.1e})", dots(colors[k], markers[k]), v[:4]) for k, v in lims.items()], t)
    study.save(fig, STUDY, "sod_limiters.png", args.docs)


if __name__ == "__main__":
    main()
