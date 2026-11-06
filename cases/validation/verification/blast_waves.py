"""A2: Woodward-Colella interacting blast waves: conservation, positivity, self-convergence.

    python cases/validation/verification/blast_waves.py            # results/validation/blast_waves/
    python cases/validation/verification/blast_waves.py --docs     # also docs/img/blast_waves.png

The case "Woodward-Colella Blast Waves" (walls on all sides, t = 0.038) runs on 400, 800 and 1600 cells and on a 3200-cell
reference of the same code (there is no exact solution). Bands, fixed before the first run:
  1. mass and total energy equal their initial values to a relative 1e-11 (walls on every side);
  2. density and pressure positive and finite everywhere;
  3. self-convergence: the L1 density difference to the reference (the reference averaged onto the coarse cells) decreases
     with every refinement, with an observed rate of at least 0.4 between successive levels (a solution with shocks and
     contacts converges at about first order in L1);
The position of the density maximum is printed but is not a band. It was one (within 0.02 of the reference's, not growing by more
than a cell of the finer grid), and it failed for a reason that is not the scheme: the maximum is a one-or-two-cell spike and two
spikes have nearly the same height, so the argmax jumps between them under any perturbation. Repeating the whole study with the
CFL changed by 1e-12 (rounding-level changes of the time step; six runs at shock_blend 0.2) gave a distance of 0.003 or 0.12 at 400
cells, 0.006 to 0.009 at 800 and 0.0008 to 0.0073 at 1600. The same perturbations move the L1 error at 1600 cells between 0.020 and
0.034 (two values, three runs each) and leave the observed rates at 0.63 or more (band 3), so the L1 bands hold at the
default and the peak position is no basis for a pass or a fail.
"""
import numpy as np

import fenice
from fenice import study

STUDY, CASE = "blast_waves", "Woodward-Colella Blast Waves"
LEVELS = (400, 800, 1600)
REFERENCE = 3200
NY, LY = 4, 0.01
CONSERVATION_BAND = 1e-11
RATE_BAND = 0.4


def profile(run_dir, nx):
    """Density, pressure of the centre row, the conserved totals (mass, energy) and time of a run."""
    d = fenice.read_dump(run_dir / "final.npz")
    U = d["U"]
    rho, _, _, p = fenice.primitive(U)
    dx, dy = 1.0 / nx, LY / NY
    totals = np.array([U[:, 0].sum(), U[:, 3].sum()]) * dx * dy
    row = lambda a: a.reshape(NY, nx)[NY // 2]
    return row(rho), row(p), totals, float(d["t"]), bool(np.isfinite(U).all())


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    runs = {}
    for n in LEVELS + (REFERENCE,):
        d = study.run_cached(STUDY, f"n{n}", CASE, f"nx={n}", f"ny={NY}", mpi=args.mpi, force=args.force)
        runs[n] = profile(d, n)
    ref = runs[REFERENCE][0]
    out = {"levels": {}}
    ok = True
    errs, peaks = [], []
    # initial totals: the case's piecewise-constant state (mass, energy), exact on any mesh for cell edges at 0.1, 0.9
    gamma = 1.4
    rho0 = 1.0
    E0 = (0.1 * 1000.0 + 0.8 * 0.01 + 0.1 * 100.0) / (gamma - 1)
    init = np.array([rho0, E0]) * LY
    for n in LEVELS:
        rho, p, tot, t, finite = runs[n]
        coarse_ref = ref.reshape(n, REFERENCE // n).mean(axis=1)
        e = float(np.mean(np.abs(rho - coarse_ref)))
        errs.append(e)
        cons = float(np.max(np.abs(tot - init) / init))
        peak_x = float(abs((np.argmax(rho) + 0.5) / n - (np.argmax(ref) + 0.5) / REFERENCE))
        res = {"l1_rho": e, "conservation": cons, "rho_min": float(rho.min()), "p_min": float(p.min()),
               "finite": finite, "peak_error_x": peak_x}
        res["pass_conservation"] = cons < CONSERVATION_BAND
        res["pass_positive"] = finite and res["rho_min"] > 0 and res["p_min"] > 0
        peaks.append(peak_x)
        ok &= res["pass_conservation"] and res["pass_positive"]
        out["levels"][n] = res
        print(f"n={n:5d}  L1={e:.4e}  conservation={cons:.1e}  rho_min={res['rho_min']:.3g}  p_min={res['p_min']:.3g}  "
              f"peak off by {peak_x:.4f} in x")
    rates = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
    out["rates"] = rates
    out["pass_rates"] = bool(all(e1 > e2 for e1, e2 in zip(errs, errs[1:])) and all(r >= RATE_BAND for r in rates))
    ok &= out["pass_rates"]
    print(f"observed rates {', '.join(f'{r:.2f}' for r in rates)}  (band >= {RATE_BAND})")
    out["passed"] = bool(ok)
    fenice.write_summary(STUDY, out)

    plt = study.use_style()
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    xr = (np.arange(REFERENCE) + 0.5) / REFERENCE
    ax[0].plot(xr, ref, "k-", lw=1.2, label=f"reference {REFERENCE}")
    ax[1].semilogy(xr, runs[REFERENCE][1], "k-", lw=1.2)
    for n, st in zip(LEVELS, ("-.", "--", "-")):
        x = (np.arange(n) + 0.5) / n
        ax[0].plot(x, runs[n][0], st, lw=1, label=f"{n} cells")
        ax[1].semilogy(x, runs[n][1], st, lw=1)
    ax[0].set_ylabel(r"density $\rho$"); ax[1].set_ylabel(r"pressure $p$"); ax[1].set_xlabel("$x$")
    ax[0].legend()
    ax[0].set_title(f"Woodward-Colella blast waves, t = {runs[REFERENCE][3]:.3f}")
    plt.tight_layout()
    study.save(fig, STUDY, "blast_waves.png", docs=args.docs)
    print("ALL BANDS PASSED" if ok else "SOME BANDS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
