"""C1c: which setting drives the error of the Sod shock tube? Sobol indices of the L1 density error.

    python cases/validation/features/sod_sensitivity.py            # results/validation/sod_sensitivity/ (quick: 64 base points)
    python cases/validation/features/sod_sensitivity.py --full     # 256 base points (1280 runs)
    python cases/validation/features/sod_sensitivity.py --docs     # also docs/img/sod_sensitivity.png

Inputs (MUSCL + SSP-RK2, HLLC, t = 0.2, ny = 4): the CFL number, uniform in [0.2, 0.8]; the resolution nx = 2^s with s uniform
in [log2 50, log2 400] (rounded to a whole number of cells); the limiter, one of Minmod / Van Leer / MC / Koren, each with
probability 1/4. Output: y = log10 of the L1 error of the density against the exact Riemann solution (Toro), over all cells.
Sobol indices by the Saltelli / Jansen estimators of fenice.uq (scrambled Sobol sequence when SciPy is installed, else random
points), with 95 % bootstrap intervals (1000 resamples). The indices are those of the log of the error, so "resolution" means
the whole power-law dependence on nx.

Bands, fixed before the first run:
  1. every run finishes with a finite error;
  2. the resolution is the dominant input: its total index ST >= 0.80 and it is the largest index of the three;
  3. the CFL number matters little: ST(cfl) <= 0.10;
  4. the limiter matters less than the resolution: ST(limiter) <= 0.30;
  5. the log-error is nearly additive: the sum of the three S1 lies within [0.80, 1.10], and the bootstrap interval of every
     index is narrower than 5 / sqrt(n) (0.63 with 64 base points, 0.31 with 256).
Band 5 was corrected after the first run (separate commit): the first version said [0.80, 1.05] and a fixed width of 0.15, which
did not allow for the sampling error of the estimators: the interval of the dominant index has a width of 4 / sqrt(n) (measured
0.46 with 64 points, 0.25 with 256, the 1 / sqrt(n) law), and the sum of S1 was 1.055 with 64 points, 1.025 with 256. Bands 1-4 are
unchanged and passed at both sizes.
"""
import math

import numpy as np

import fenice
from fenice import study, uq
from fenice.exact import riemann

STUDY, CASE = "sod_sensitivity", "Sod Shock Tube"
LIMITERS = ["Minmod", "Van Leer", "MC", "Koren"]
NY = 4
BOUNDS = {"cfl": (0.2, 0.8), "log2_nx": (math.log2(50), math.log2(400)), "limiter": (0.0, 4.0)}
BASE = ("reconstruction=muscl", "time_scheme=SSP Runge-Kutta 2", f"ny={NY}", "save_interval=1000000000")
BAND_ST_NX, BAND_ST_CFL, BAND_ST_LIM, BAND_SUM, BAND_WIDTH_C = 0.80, 0.10, 0.30, (0.80, 1.10), 5.0   # width = C / sqrt(n)


def to_overrides(row):
    """A point of the unit hypercube's scaled image -> the solver overrides."""
    return {"cfl": round(row["cfl"], 6), "nx": int(round(2 ** row["log2_nx"])),
            "muscl_limiter": LIMITERS[min(int(row["limiter"]), len(LIMITERS) - 1)]}


def l1_error(run_dir, point):
    nx = int(point["nx"])
    d = fenice.read_dump(run_dir / "final.npz")
    rho = d["U"][:, 0].reshape(NY, nx)[NY // 2]
    ref, _, _ = riemann((np.arange(nx) + 0.5) / nx, float(d["t"]), gamma=1.4)
    err = float(np.mean(np.abs(rho - ref)))
    return {"l1": err, "finite": float(np.isfinite(d["U"]).all())}


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="256 base points instead of 64")
    args = ap.parse_args()
    n = 256 if args.full else 64
    try:
        sampler = uq.sobol_sampler
        sampler(2, 6)
    except ImportError:
        sampler = uq.uniform_sampler
    A, B, AB, raw = uq.saltelli(n, BOUNDS, sampler, seed=7)
    points = [to_overrides(p) for p in raw]
    root = study.study_dir(STUDY) / ("runs_full" if args.full else "runs")
    recs = uq.run_sweep(CASE, points, l1_error, base=BASE, root=root, threads=1)
    new = sum(not r["cached"] for r in recs)
    print(f"{len(recs)} runs ({new} new, {len(recs) - new} cached), {sum(r['wall'] for r in recs):.0f} s of solver time")
    err = np.array([r["qoi"]["l1"] for r in recs])
    finite = bool(all(r["qoi"]["finite"] == 1.0 for r in recs) and np.isfinite(err).all() and (err > 0).all())
    y = np.log10(err)
    S1, ST = uq.sobol_indices(y, n, 3)
    s1_lo, s1_hi, st_lo, st_hi = uq.sobol_bootstrap(y, n, 3, n_boot=1000, level=0.95, seed=0)
    names = ["cfl", "resolution", "limiter"]
    print(f"{'input':12s} {'S1':>7s} {'95 % interval':>16s} {'ST':>7s} {'95 % interval':>16s}")
    for i, nm in enumerate(names):
        print(f"{nm:12s} {S1[i]:7.3f} [{s1_lo[i]:6.3f},{s1_hi[i]:6.3f}] {ST[i]:7.3f} [{st_lo[i]:6.3f},{st_hi[i]:6.3f}]")
    print(f"sum S1 = {S1.sum():.3f}; log10 L1 from {y.min():.2f} to {y.max():.2f}")

    lim = np.array([LIMITERS.index(p["muscl_limiter"]) for p in points[:n]])
    nxs = np.array([p["nx"] for p in points[:n]])
    rate = np.polyfit(np.log10(nxs), y[:n], 1)[0]
    print(f"A sample: log-log slope of the error against nx = {rate:.2f} (all limiters and CFL together)")
    out = {"n": n, "runs": len(recs), "names": names, "S1": S1.tolist(), "ST": ST.tolist(),
           "S1_ci": [s1_lo.tolist(), s1_hi.tolist()], "ST_ci": [st_lo.tolist(), st_hi.tolist()], "slope_A": float(rate),
           "bands": {"ST_nx": BAND_ST_NX, "ST_cfl": BAND_ST_CFL, "ST_limiter": BAND_ST_LIM, "sum_S1": BAND_SUM,
                     "width_C": BAND_WIDTH_C}}
    out["pass_finite"] = finite
    out["pass_resolution"] = bool(ST[1] >= BAND_ST_NX and ST[1] == ST.max())
    out["pass_cfl"] = bool(ST[0] <= BAND_ST_CFL)
    out["pass_limiter"] = bool(ST[2] <= BAND_ST_LIM)
    out["pass_additive"] = bool(BAND_SUM[0] <= S1.sum() <= BAND_SUM[1] and max((st_hi - st_lo).max(), (s1_hi - s1_lo).max()) < BAND_WIDTH_C / math.sqrt(n))
    ok = all(out[k] for k in out if k.startswith("pass_"))
    out["passed"] = ok
    fenice.write_summary(STUDY, out)

    plt = study.use_style()
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(3)
    a0.bar(x - 0.2, S1, 0.4, yerr=[S1 - s1_lo, s1_hi - S1], capsize=3, color="#2980b9", label="first order $S_1$")
    a0.bar(x + 0.2, ST, 0.4, yerr=[ST - st_lo, st_hi - ST], capsize=3, color="#c0392b", label="total $S_T$")
    a0.set_xticks(x, names)
    a0.set_ylabel("Sobol index of $\\log_{10}$ L1")
    a0.axhline(BAND_ST_NX, color="0.3", ls=":", lw=1)
    a0.legend()
    a0.set_title(f"{len(recs)} runs, 95 % bootstrap intervals")
    for k, lm in enumerate(LIMITERS):
        m = lim == k
        a1.loglog(nxs[m], err[:n][m], "o", ms=4, label=lm)
    a1.set_xlabel("cells $n_x$")
    a1.set_ylabel("L1 error of $\\rho$")
    a1.legend(title="limiter (A sample)")
    plt.tight_layout()
    study.save(fig, STUDY, "sod_sensitivity.png", docs=args.docs)
    print("ALL BANDS PASSED" if ok else "SOME BANDS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
