"""Shu-Osher shock / entropy-wave interaction: limiters and grid convergence against a fine reference.

    python validation/shu_osher.py            # results/validation/shu_osher/
    python validation/shu_osher.py --docs     # also docs/img/shu_osher.png

Runs the "Shu-Osher Test" case (MUSCL + SSP-RK2, t = 1.8) with the MC and Minmod limiters on
200, 400, 800 and 1600 cells, plus a 6400-cell MC reference (there is no closed-form
solution). The L1 density error is measured against the reference averaged onto each grid.
Figure: shu_osher.png.
"""
import numpy as np

import fenice
from fenice import study

STUDY, CASE = "shu_osher", "Shu-Osher Test"
GRIDS, N_REF, L = [200, 400, 800, 1600], 6400, 10.0
LIMITERS = {"MC": ("#c0392b", "o"), "Minmod": ("#2980b9", "s")}


def density(run_dir, n):
    d = fenice.read_dump(run_dir / "final.npz")
    return d["U"][:, 0].reshape(2, n).mean(axis=0), float(d["t"])


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    run = lambda tag, n, lim: density(study.run_cached(STUDY, tag, CASE, f"nx={n}", f"muscl_limiter={lim}",
                                                       "save_interval=1000000000", mpi=args.mpi, force=args.force), n)
    ref, t = run(f"reference_n{N_REF}", N_REF, "MC")
    xr = (np.arange(N_REF) + 0.5) * L / N_REF
    rho, err = {}, {}
    for lim in LIMITERS:
        for n in GRIDS:
            rho[lim, n] = run(f"{lim.lower()}_n{n}", n, lim)[0]
            err[lim, n] = np.mean(np.abs(rho[lim, n] - ref.reshape(n, -1).mean(axis=1))) * L
        e = np.array([err[lim, n] for n in GRIDS])
        print(f"{lim:7s} L1 error " + ", ".join(f"{n}: {v:.3e}" for n, v in zip(GRIDS, e)) +
              "; orders " + " ".join(f"{o:.2f}" for o in np.log2(e[:-1] / e[1:])))

    n_show = 400
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": (1.8, 1)})
    fig.suptitle(f"Shu-Osher shock / entropy-wave interaction ($t = {t:g}$, MUSCL + SSP-RK2)", fontsize=15,
                 fontweight="bold")
    xc = (np.arange(n_show) + 0.5) * L / n_show
    a1.plot(xr, ref, "k-", lw=1.1, label=f"Reference (MC, {N_REF} cells)", zorder=5)
    for lim, (c, m) in LIMITERS.items():
        a1.plot(xc, rho[lim, n_show], ls="none", marker=m, ms=3.2, mfc="none", mec=c, mew=0.8,
                label=f"{lim}, {n_show} cells")
    a1.set_xlim(0, L)
    a1.set_xlabel("$x$")
    a1.set_ylabel(r"Density $\rho$")
    a1.set_title("Density profile")
    a1.legend(loc="center right")
    ins = a1.inset_axes([0.05, 0.08, 0.38, 0.44])   # zoom on the post-shock entropy waves
    ins.plot(xr, ref, "k-", lw=1.0)
    for lim, (c, m) in LIMITERS.items():
        ins.plot(xc, rho[lim, n_show], ls="none", marker=m, ms=2.6, mfc="none", mec=c, mew=0.7)
    ins.set_xlim(4.6, 7.4)
    ins.set_ylim(2.9, 4.8)
    ins.tick_params(labelsize=8)
    a1.indicate_inset_zoom(ins, edgecolor="#7f8c8d")

    g = np.array(GRIDS, dtype=float)
    for lim, (c, m) in LIMITERS.items():
        e = np.array([err[lim, n] for n in GRIDS])
        p = np.polyfit(np.log(g), np.log(e), 1)[0]
        a2.loglog(g, e, m + "-", color=c, lw=2, ms=7, mec="white", label=f"{lim} (slope {p:.2f})")
    e0 = err["MC", GRIDS[0]]
    a2.loglog(g, e0 * (g[0] / g), ":", color="gray", label="1st order")
    a2.loglog(g, e0 * (g[0] / g) ** 2, "-.", color="gray", label="2nd order")
    a2.set_xticks(g, [str(n) for n in GRIDS])
    a2.minorticks_off()
    a2.set_xlabel("Cells, $n_x$")
    a2.set_ylabel(r"$L_1$ density error vs reference")
    a2.set_title("Grid convergence")
    a2.grid(True, which="both", ls="--", alpha=0.5)
    a2.legend()
    plt.tight_layout()
    study.save(fig, STUDY, "shu_osher.png", args.docs)


if __name__ == "__main__":
    main()
