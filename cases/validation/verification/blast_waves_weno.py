"""Woodward-Colella blast waves with WENO3-Z, compared with the 800-cell MUSCL run.

    python cases/validation/verification/blast_waves_weno.py            # results/validation/blast_waves_weno/
    python cases/validation/verification/blast_waves_weno.py --docs     # also docs/img/blast_waves_weno.png

The case of blast_waves.py (MUSCL-MC, SSP-RK2, CFL 0.4, HLLC with the default shock-face dissipation) is run with
WENO3-Z and SSP-RK3 (everything else unchanged) on 400, 800, 1600 cells and on a 3200-cell reference of the same scheme, and with
MUSCL on 800 and 3200 cells (the reference of the MUSCL study). Reported:
  1. conservation and positivity of every WENO3-Z run (bands as in blast_waves.py: 1e-11, rho and p > 0);
  2. self-convergence of WENO3-Z: L1 density difference to its own reference averaged onto the coarse cells, decreasing at every
     refinement with an observed rate of at least 0.4 (the band of the MUSCL study);
  3. the comparison with MUSCL: L1 density difference between each WENO3-Z run and the 800-cell MUSCL run (both averaged onto the
     coarser of the two meshes), next to the difference between the two 3200-cell references, which is what the two schemes
     disagree on when the mesh is no longer the limit. Not a band: the schemes are different, so no value is a target;
  4. sensitivity: the 1600-cell WENO3-Z run and its reference repeated with the CFL changed by 1e-12 (relative), as in
     blast_waves.py, to see whether the error at the finest level also takes two values. Reported, not a band.
"""
import numpy as np

import fenice
from fenice import study

STUDY, CASE = "blast_waves_weno", "Woodward-Colella Blast Waves"
LEVELS = (400, 800, 1600)
REFERENCE = 3200
NY, LY = 4, 0.01
CONSERVATION_BAND = 1e-11
RATE_BAND = 0.4
WENO = ["reconstruction=weno3", "time_scheme=SSP Runge-Kutta 3"]
PERTURB = (0.4 * (1 + 1e-12), 0.4 * (1 - 1e-12))


def run(tag, n, *extra, args):
    d = study.run_cached(STUDY, tag, CASE, f"nx={n}", f"ny={NY}", *extra, mpi=args.mpi, force=args.force)
    z = fenice.read_dump(d / "final.npz")
    U = z["U"]
    rho, _, _, p = fenice.primitive(U)
    dx, dy = 1.0 / n, LY / NY
    tot = np.array([U[:, 0].sum(), U[:, 3].sum()]) * dx * dy
    row = lambda a: a.reshape(NY, n)[NY // 2]
    return row(rho), row(p), tot, bool(np.isfinite(U).all())


def avg(a, m):
    """Average a cell row onto m cells (m divides len(a))."""
    return a.reshape(m, len(a) // m).mean(axis=1)


def l1(a, b):
    """Mean |a - b| after averaging both onto the coarser of the two meshes."""
    m = min(len(a), len(b))
    return float(np.mean(np.abs(avg(a, m) - avg(b, m))))


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    weno = {n: run(f"weno3_n{n}", n, *WENO, args=args) for n in LEVELS + (REFERENCE,)}
    muscl = {n: run(f"muscl_n{n}", n, args=args) for n in (800, REFERENCE)}
    gamma = 1.4
    E0 = (0.1 * 1000.0 + 0.8 * 0.01 + 0.1 * 100.0) / (gamma - 1)
    init = np.array([1.0, E0]) * LY
    out = {"levels": {}}
    ok = True
    ref = weno[REFERENCE][0]
    errs = []
    for n in LEVELS:
        rho, p, tot, finite = weno[n]
        e = l1(rho, ref)
        errs.append(e)
        cons = float(np.max(np.abs(tot - init) / init))
        res = {"l1_to_own_reference": e, "conservation": cons, "rho_min": float(rho.min()), "p_min": float(p.min()),
               "finite": finite, "l1_to_muscl_800": l1(rho, muscl[800][0])}
        res["pass_conservation"] = cons < CONSERVATION_BAND
        res["pass_positive"] = finite and res["rho_min"] > 0 and res["p_min"] > 0
        ok &= res["pass_conservation"] and res["pass_positive"]
        out["levels"][n] = res
        print(f"WENO3-Z n={n:5d}  L1 to own ref={e:.4e}  to MUSCL-800={res['l1_to_muscl_800']:.4e}  conservation={cons:.1e}  "
              f"rho_min={res['rho_min']:.3g}  p_min={res['p_min']:.3g}")
    rates = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
    out["rates"] = rates
    out["pass_rates"] = bool(all(a > b for a, b in zip(errs, errs[1:])) and all(r >= RATE_BAND for r in rates))
    ok &= out["pass_rates"]
    print(f"observed rates {', '.join(f'{r:.2f}' for r in rates)}  (band >= {RATE_BAND})")
    # MUSCL against its own reference at 800 (the study of blast_waves.py), for the same measure
    out["muscl_800_to_own_reference"] = l1(muscl[800][0], muscl[REFERENCE][0])
    out["references_difference"] = l1(weno[REFERENCE][0], muscl[REFERENCE][0])
    print(f"MUSCL-800 to MUSCL-3200: {out['muscl_800_to_own_reference']:.4e};  WENO3-Z-3200 to MUSCL-3200: "
          f"{out['references_difference']:.4e}")
    # rounding-level sensitivity of the WENO3-Z error at 1600
    sens = []
    for i, cfl in enumerate(PERTURB):
        r1600 = run(f"weno3_n1600_cfl{i}", 1600, *WENO, f"cfl={cfl!r}", args=args)[0]
        rref = run(f"weno3_n3200_cfl{i}", REFERENCE, *WENO, f"cfl={cfl!r}", args=args)[0]
        sens.append(l1(r1600, rref))
    out["sensitivity_l1_1600"] = [errs[-1]] + sens
    print("L1 at 1600 for CFL 0.4 and the two perturbed values: " + ", ".join(f"{v:.4e}" for v in out["sensitivity_l1_1600"]))
    out["passed"] = bool(ok)
    fenice.write_summary(STUDY, out)

    plt = study.use_style()
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    xr = (np.arange(REFERENCE) + 0.5) / REFERENCE
    for col, (lo, hi) in enumerate(((0.0, 1.0), (0.55, 0.85))):
        a = ax[0, col]
        a.plot(xr, weno[REFERENCE][0], "k-", lw=1.0, label="WENO3-Z 3200")
        a.plot(xr, muscl[REFERENCE][0], "-", color="tab:gray", lw=1.0, label="MUSCL-MC 3200")
        for n, st in ((400, "-."), (800, "--"), (1600, "-")):
            a.plot((np.arange(n) + 0.5) / n, weno[n][0], st, lw=1, label=f"WENO3-Z {n}")
        a.plot((np.arange(800) + 0.5) / 800, muscl[800][0], "r:", lw=1.4, label="MUSCL-MC 800")
        a.set_xlim(lo, hi)
        a.set_ylabel(r"density $\rho$")
        b = ax[1, col]
        b.plot(xr, weno[REFERENCE][0] - muscl[REFERENCE][0], "-", color="tab:gray", lw=0.8, label="WENO3-Z - MUSCL, 3200")
        for n, st in ((800, "--"), (1600, "-")):
            m = min(n, 800)
            xm = (np.arange(m) + 0.5) / m
            b.plot(xm, avg(weno[n][0], m) - avg(muscl[800][0], m), st, lw=1, label=f"WENO3-Z {n} - MUSCL 800")
        b.set_xlim(lo, hi)
        b.set_xlabel("$x$")
        b.set_ylabel(r"$\Delta\rho$")
    ax[0, 0].legend(fontsize=7)
    ax[1, 0].legend(fontsize=7)
    ax[0, 1].set_title("zoom, 0.55 < x < 0.85")
    ax[0, 0].set_title("Woodward-Colella, t = 0.038")
    plt.tight_layout()
    study.save(fig, STUDY, "blast_waves_weno.png", docs=args.docs)
    print("ALL BANDS PASSED" if ok else "SOME BANDS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
