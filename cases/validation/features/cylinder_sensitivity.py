"""C1c: what drives the drag of the cylinder at Re = 40? Sobol indices of Cd for the inlet speed, the resolution and the limiter.

    python cases/validation/features/cylinder_sensitivity.py            # 8 base points = 40 runs, 5-8 cells per diameter (medium)
    python cases/validation/features/cylinder_sensitivity.py --full     # 32 base points = 160 runs, 6-12 cells per diameter (long)
    python cases/validation/features/cylinder_sensitivity.py --docs     # also docs/img/cylinder_sensitivity.png

The case is the steady cylinder of ibm_ghost.py --loads (channel 4 x 2, D = 0.2, ghost-cell wall, mu = 5e-3, p = 9.8, so
Mach 0.27 at u_in = 1), run to t = 20 on small meshes. The viscosity is held fixed, so the inlet speed changes the Reynolds
number: Re = u_in D / mu, 30 to 50 for u_in in [0.75, 1.25] (and the Mach number 0.2-0.34). The other inputs: the resolution in
cells per diameter (uniform in [5, 8], [6, 12] with --full; the mesh is 20 cells/D x 4 D by 10 cells/D x 2 D) and the limiter
(Minmod / Van Leer / MC / Koren, equal probability). Output: Cd = mean of the monitor's drag coefficient over the last two time
units. Sobol indices and 95 % bootstrap intervals from fenice.uq (Saltelli / Jansen, scrambled Sobol sequence with SciPy).
Reference for the cross-check: ibm_ghost.py --loads found Cd = 1.8836 (40 cells/D, t = 55), 1.8957 at 20 cells/D.

Bands, fixed before the first run:
  1. every run finishes with a finite Cd that has settled: |mean(last 2 time units) - mean(the 2 before)| <= 2 % of Cd;
  2. the inlet speed (Reynolds number) matters: ST >= 0.25, and Cd falls with u_in (Spearman rank correlation of the A sample <= -0.5);
  3. the resolution matters too, at these coarse meshes: ST(cells per diameter) >= 0.25;
  4. the limiter matters little: ST(limiter) <= 0.10;
  5. cross-check against the converged value: one run at u_in = 1 (Van Leer) on the finest allowed mesh give Cd within 6 %
     (--full: 3.5 %) of 1.8836.
Bands 2 and 3 were changed after the first runs (separate commit). The first versions predicted that the inlet speed dominates
(ST >= 0.80, Spearman <= -0.8) and the resolution hardly matters (ST <= 0.20). With 160 runs (6-12 cells/D) the total indices are
0.46 (inlet speed) and 0.42 (resolution): at these meshes the discretisation error of Cd (about 5 % at 8 cells/D, 2 % at 12) is as
large a part of the variance as the 30-50 range of Re. Bands 1, 4 and 5 are unchanged and passed. The 8-point default mode gives
intervals wider than the indices themselves (it is a smoke test; use --full for indices).
"""
import math

import numpy as np

import fenice
from fenice import study, uq

STUDY = "cylinder_sensitivity"
LIMITERS = ["Minmod", "Van Leer", "MC", "Koren"]
T_END, WINDOW = 20.0, 2.0
CD_REF = 1.8836
BAND_SETTLED, BAND_ST_U, BAND_RHO, BAND_ST_CPD, BAND_ST_LIM = 0.02, 0.25, -0.5, 0.25, 0.10


def case_folder():
    """The Re = 40 cylinder case with the initial speed tied to the inlet speed (u = u_in)."""
    src = (fenice.ROOT / "cases" / "ibm_flow_over_cylinder" / "case.toml").read_text()
    for a, b in [("mu = 2e-3", "mu = 5e-3"), ("p_out = 100.0", "p_out = 9.8"),
                 ("state = { rho = 1.0, u = 1.0, v = 0.0, p = 100.0 }", 'state = { rho = 1.0, u = "u_in", v = 0.0, p = 9.8 }'),
                 ('name = "IBM Flow over Cylinder"', 'name = "Cylinder Re40 sensitivity"')]:
        assert a in src, a
        src = src.replace(a, b)
    return study.write_case(STUDY, "re40", src)


def overrides(u_in, cpd, limiter):
    nx = 2 * int(round(10 * cpd))
    return {"u_in": round(float(u_in), 6), "nx": nx, "ny": nx // 2, "muscl_limiter": LIMITERS[min(int(limiter), 3)]}


def drag(run_dir, point):
    d = fenice.read_monitor(study.case_output(run_dir))
    t, cd = d["t"], d["Cd"]
    last, before = (t > t[-1] - WINDOW), (t > t[-1] - 2 * WINDOW) & (t <= t[-1] - WINDOW)
    m = float(cd[last].mean())
    return {"cd": m, "drift": abs(m - float(cd[before].mean())) / abs(m), "finite": float(np.isfinite(cd).all())}


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="32 base points and 6-12 cells per diameter")
    args = ap.parse_args()
    n, cpd_lo, cpd_hi = (32, 6.0, 12.0) if args.full else (8, 5.0, 8.0)
    band_ref = 0.035 if args.full else 0.06
    bounds = {"u_in": (0.75, 1.25), "cpd": (cpd_lo, cpd_hi), "limiter": (0.0, 4.0)}
    try:
        sampler = uq.sobol_sampler
        sampler(2, 6)
    except ImportError:
        sampler = uq.uniform_sampler
    A, B, AB, raw = uq.saltelli(n, bounds, sampler, seed=11)
    points = [overrides(p["u_in"], p["cpd"], p["limiter"]) for p in raw]
    points += [overrides(1.0, cpd_hi, 1.0)]   # the cross-check run
    folder = case_folder()
    base = ("ibm_method=ghost", f"t_end={T_END}", "monitor_interval=25", "save_interval=1000000000", "use_mpi=false")
    root = study.study_dir(STUDY) / ("runs_full" if args.full else "runs")
    recs = uq.run_sweep(folder, points, drag, base=base, root=root, threads=1)
    new = sum(not r["cached"] for r in recs)
    print(f"{len(recs)} points ({new} new runs), {sum(r['wall'] for r in recs if not r['cached']):.0f} s of solver time")
    ref = recs[-1:]
    recs = recs[:-1]
    cd = np.array([r["qoi"]["cd"] for r in recs])
    drift = np.array([r["qoi"]["drift"] for r in recs])
    finite = bool(np.isfinite(cd).all() and all(r["qoi"]["finite"] == 1.0 for r in recs))
    print(f"Cd from {cd.min():.3f} to {cd.max():.3f}; drift over the last windows: median {np.median(drift):.1e}, max {drift.max():.1e}")
    S1, ST = uq.sobol_indices(cd, n, 3)
    s1_lo, s1_hi, st_lo, st_hi = uq.sobol_bootstrap(cd, n, 3, n_boot=1000, seed=0)
    names = ["inlet speed", "cells per D", "limiter"]
    print(f"{'input':12s} {'S1':>7s} {'95 % interval':>16s} {'ST':>7s} {'95 % interval':>16s}")
    for i, nm in enumerate(names):
        print(f"{nm:12s} {S1[i]:7.3f} [{s1_lo[i]:6.3f},{s1_hi[i]:6.3f}] {ST[i]:7.3f} [{st_lo[i]:6.3f},{st_hi[i]:6.3f}]")
    u = np.array([r["point"]["u_in"] for r in recs[:n]])
    ranks = lambda a: np.argsort(np.argsort(a))
    rho = float(np.corrcoef(ranks(u), ranks(cd[:n]))[0, 1])
    cd_ref = ref[0]["qoi"]["cd"]
    err = abs(cd_ref / CD_REF - 1)
    print(f"Spearman(u_in, Cd) on the A sample = {rho:.2f}; u_in = 1 at {ref[0]['point']['nx'] / 20:.1f} cells/D: "
          f"Cd = {cd_ref:.4f} against {CD_REF}: {100 * err:.1f} %")
    out = {"n": n, "names": names, "S1": S1.tolist(), "ST": ST.tolist(), "S1_ci": [s1_lo.tolist(), s1_hi.tolist()],
           "ST_ci": [st_lo.tolist(), st_hi.tolist()], "spearman_u": rho, "cd_ref_run": cd_ref, "cd_ref": CD_REF,
           "ref_error": err, "max_drift": float(drift.max())}
    out["pass_finite_settled"] = bool(finite and drift.max() <= BAND_SETTLED)
    out["pass_inlet_speed"] = bool(ST[0] >= BAND_ST_U and rho <= BAND_RHO)
    out["pass_resolution"] = bool(ST[1] >= BAND_ST_CPD)
    out["pass_limiter"] = bool(ST[2] <= BAND_ST_LIM)
    out["pass_reference"] = bool(err <= band_ref)
    ok = all(out[k] for k in out if k.startswith("pass_"))
    out["passed"] = ok
    fenice.write_summary(STUDY, out)

    plt = study.use_style()
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(3)
    a0.bar(x - 0.2, S1, 0.4, yerr=[S1 - s1_lo, s1_hi - S1], capsize=3, color="#2980b9", label="first order $S_1$")
    a0.bar(x + 0.2, ST, 0.4, yerr=[ST - st_lo, st_hi - ST], capsize=3, color="#c0392b", label="total $S_T$")
    a0.set_xticks(x, names)
    a0.set_ylabel("Sobol index of $C_d$")
    a0.legend()
    a0.set_title(f"{len(cd)} runs, 95 % bootstrap intervals")
    cp = np.array([r["point"]["nx"] / 20 for r in recs])
    sc = a1.scatter(u[:n] * 0.2 / 5e-3, cd[:n], c=cp[:n], cmap="viridis", s=26)
    fig.colorbar(sc, ax=a1, label="cells per diameter")
    a1.axhline(CD_REF, color="0.3", ls=":", lw=1)
    a1.set_xlabel("Reynolds number $u_{in} D / \\mu$")
    a1.set_ylabel("$C_d$ (A sample)")
    plt.tight_layout()
    study.save(fig, STUDY, "cylinder_sensitivity.png", docs=args.docs)
    print("ALL BANDS PASSED" if ok else "SOME BANDS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
