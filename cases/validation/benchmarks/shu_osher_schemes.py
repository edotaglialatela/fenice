"""Shu-Osher shock / entropy-wave interaction: the reconstruction (MUSCL limiters, WENO3-Z, WENO5-Z) at four resolutions.

    python cases/validation/benchmarks/shu_osher_schemes.py            # results/validation/shu_osher_schemes/
    python cases/validation/benchmarks/shu_osher_schemes.py --docs     # also docs/img/shu_osher_schemes*.png

Runs the "Shu-Osher Test" case (t = 1.8) with MUSCL Minmod and MC (SSP-RK2), WENO3-Z and WENO5-Z (SSP-RK3) on 200, 400, 800 and
1600 cells. There is no closed-form solution, so two references are used, both on 6400 cells: MC (the reference of shu_osher.py) and
WENO5-Z (CFL 0.3). The L1 density error of each run is measured against the reference averaged onto its grid; the table prints both,
so the dependence of the ranking on the reference is visible (a MUSCL-MC reference favours MUSCL-MC a little, a WENO5-Z one WENO5-Z).
Figures: shu_osher_schemes.png (post-shock waves at 400 cells, error against the number of cells) and
shu_osher_schemes_profile.png (whole profile at 400 cells).
"""
import numpy as np

import fenice
from fenice import study

STUDY, CASE = "shu_osher_schemes", "Shu-Osher Test"
GRIDS, N_REF, L = [200, 400, 800, 1600], 6400, 10.0
RK2, RK3 = "time_scheme=SSP Runge-Kutta 2", "time_scheme=SSP Runge-Kutta 3"
SCHEMES = {   # label: (tag, sets, colour, marker)
    "MUSCL Minmod": ("Minmod", ("use_muscl=true", "muscl_limiter=Minmod", RK2), "#2980b9", "s"),
    "MUSCL MC": ("MC", ("use_muscl=true", "muscl_limiter=MC", RK2), "#c0392b", "o"),
    "WENO3-Z": ("WENO3Z", ("use_muscl=true", "reconstruction=weno3", RK3), "#27ae60", "^"),
    "WENO5-Z": ("WENO5Z", ("use_muscl=true", "reconstruction=weno5", RK3), "#8e44ad", "D"),
}


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()

    def rho(tag, n, sets):
        d = study.run_cached(STUDY, tag, CASE, f"nx={n}", "save_interval=1000000000", *sets, mpi=args.mpi, force=args.force)
        z = fenice.read_dump(d / "final.npz")
        return z["U"][:, 0].reshape(2, n).mean(axis=0)

    ref = {"MC": rho("ref_mc_n6400", N_REF, ("use_muscl=true", "muscl_limiter=MC", RK2)),
           "WENO5-Z": rho("ref_weno5_n6400", N_REF, ("use_muscl=true", "reconstruction=weno5", RK3, "cfl=0.3"))}
    xr = (np.arange(N_REF) + 0.5) * L / N_REF
    prof, err = {}, {}
    rows = ["| scheme | reference | " + " | ".join(str(n) for n in GRIDS) + " | orders |", "|---|---|" + "---|" * (len(GRIDS) + 1)]
    for label, (tag, sets, _, _) in SCHEMES.items():
        for n in GRIDS:
            prof[label, n] = rho(f"{tag}_n{n}", n, sets)
        for rname, r in ref.items():
            e = np.array([np.mean(np.abs(prof[label, n] - r.reshape(n, -1).mean(axis=1))) * L for n in GRIDS])
            err[label, rname] = e
            rows.append(f"| {label} | {rname} | " + " | ".join(f"{v:.3e}" for v in e) + " | " +
                        " ".join(f"{o:.2f}" for o in np.log2(e[:-1] / e[1:])) + " |")
            print(rows[-1], flush=True)
    path = study.study_dir(STUDY) / "shu_osher_schemes.md"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")

    rname = "WENO5-Z"   # the reference of the figures: the most accurate scheme, converged on 6400 cells
    xs = lambda n: (np.arange(n) + 0.5) * L / n
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle(r"Shu-Osher problem at $t = 1.8$: reconstruction and resolution", fontsize=15, fontweight="bold")
    a1.plot(xr, ref[rname], "k-", lw=1.1, label=f"Reference ({rname}, {N_REF} cells)", zorder=5)
    for label, (_, _, c, m) in SCHEMES.items():
        a1.plot(xs(400), prof[label, 400], ls="none", marker=m, ms=4.2, mfc="none", mec=c, mew=0.9, label=f"{label}, 400 cells")
    a1.set_xlim(4.6, 7.4)
    a1.set_ylim(2.9, 4.8)
    a1.set_xlabel("$x$")
    a1.set_ylabel(r"Density $\rho$")
    a1.set_title("Post-shock entropy waves")
    a1.legend(loc="lower left", fontsize=9)
    for label, (_, _, c, m) in SCHEMES.items():
        a2.loglog(GRIDS, err[label, rname], marker=m, color=c, lw=1.6, ms=6, label=label)
    a2.loglog(GRIDS, err["MUSCL MC", rname][0] * (GRIDS[0] / np.array(GRIDS, float)), "k--", lw=1, label="first order")
    a2.set_xlabel("Number of cells")
    a2.set_ylabel(r"$L_1$ density error against the reference")
    a2.set_title(f"Convergence (reference: {rname}, {N_REF} cells)")
    a2.set_xticks(GRIDS)
    a2.set_xticklabels([str(n) for n in GRIDS])
    a2.grid(True, which="both", ls="--", alpha=0.5)
    a2.legend(fontsize=9)
    plt.tight_layout()
    study.save(fig, STUDY, "shu_osher_schemes.png", args.docs)

    fig, a1 = plt.subplots(figsize=(11, 5.4))
    a1.plot(xr, ref[rname], "k-", lw=1.1, label=f"Reference ({rname}, {N_REF} cells)", zorder=5)
    for label, (_, _, c, m) in SCHEMES.items():
        a1.plot(xs(400), prof[label, 400], ls="none", marker=m, ms=3.2, mfc="none", mec=c, mew=0.7, label=f"{label}, 400 cells")
    a1.set_xlim(0, L)
    a1.set_xlabel("$x$")
    a1.set_ylabel(r"Density $\rho$")
    a1.set_title(r"Shu-Osher problem, $t = 1.8$: the four schemes on 400 cells", fontsize=14, fontweight="bold")
    a1.set_ylim(0.6, 5.0)
    a1.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    study.save(fig, STUDY, "shu_osher_schemes_profile.png", args.docs)
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
