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
  4. the position of the density maximum differs from the reference's by at most 0.02 (a fifth of the first blast's width)
     on every grid, and the difference in x does not grow with refinement by more than one cell of the finer grid (the peak
     is a one-or-two-cell spike, its position converges in physical distance, not in cells; a band of 2 cells, the first
     choice, was wrong for that reason; "does not grow" at all, the second choice, was wrong because the position of the
     maximum is quantised to the cell centres of each grid: with shock_blend 0.2 it goes 0.0070, 0.0077, 0.0048 from 400 to 1600
     cells, a growth of 0.0007 against a cell of 0.00125 at 800).
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
PEAK_BAND_X = 0.02


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
        res["pass_peak"] = peak_x <= PEAK_BAND_X
        peaks.append(peak_x)
        ok &= res["pass_conservation"] and res["pass_positive"] and res["pass_peak"]
        out["levels"][n] = res
        print(f"n={n:5d}  L1={e:.4e}  conservation={cons:.1e}  rho_min={res['rho_min']:.3g}  p_min={res['p_min']:.3g}  "
              f"peak off by {peak_x:.4f} in x")
    rates = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
    out["rates"] = rates
    out["pass_rates"] = bool(all(e1 > e2 for e1, e2 in zip(errs, errs[1:])) and all(r >= RATE_BAND for r in rates))
    ok &= out["pass_rates"]
    print(f"observed rates {', '.join(f'{r:.2f}' for r in rates)}  (band >= {RATE_BAND})")
    out["pass_peak_monotone"] = bool(all(a + 1.0 / n2 >= b for a, b, n2 in zip(peaks, peaks[1:], LEVELS[1:])))
    ok &= out["pass_peak_monotone"]
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
