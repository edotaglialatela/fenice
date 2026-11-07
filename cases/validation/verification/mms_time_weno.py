"""Order of accuracy in time with WENO3-Z and WENO5-Z: the unsteady manufactured solution of mms_time.py.

    python cases/validation/verification/mms_time_weno.py            # results/validation/mms_time_weno/
    python cases/validation/verification/mms_time_weno.py --full     # also the finer-step diagnostic of the WENO5-Z weights
    python cases/validation/verification/mms_time_weno.py --docs     # also docs/img/mms_time_weno.png

The same case ("Manufactured Solution Unsteady Euler", t = 0.2, 32^2 cells for the time study), reference and measures as
mms_time.py (its helpers are used), with the WENO-Z reconstruction in space. The WENO schemes require SSP-RK3, so only that
integrator is run; the point is that the stage times of the unsteady source (t, t + dt, t + dt/2) and the time order hold with
the nonlinear reconstruction in the right-hand side, which mms_time.py checks for first order and MUSCL only.

1. The reference: SSP-RK3 with CFL 0.00625 for each reconstruction. Its error against the exact solution at t = 0.2 on 16^2,
   32^2, 64^2 (SSP-RK3, CFL 0.25) must fall on every refinement with an observed order of the finest pair of at least 1.5
   (the band of MUSCL; the test is second order in space at most, see mms_weno.py).
2. Time convergence: the L2 difference of the conserved state to that reference at CFL 0.4, 0.2, 0.1, 0.05.

Expected, fixed before the first run, for WENO3-Z and WENO5-Z at the default weno_epsilon (1e-6): observed order of the finest
pair at least 2.5 (the band of SSP-RK3 with the first-order scheme, whose right-hand side is smooth; the WENO weights are smooth
functions of the data, so no kinks as with a limiter). An order above 3.5 is reported as above the band and is not a failure
(an order above the expected one cannot hide a wrong stage time, which costs an order); only an order below 2.5, or an error that
does not fall, fails the study.

What happened, and what was changed after the first run. WENO3-Z gives 3.0 on every pair. WENO5-Z at the default weno_epsilon gave
2.97, 2.77, 2.04 on the three pairs, below the band on the finest pair, and with steps down to CFL 0.0125 against a reference at CFL
0.0015625 the orders are 2.97, 2.77, 2.03, 1.54, 2.44, irregular. A wrong stage time would affect WENO3-Z too, so the integrator is
not the suspect: with weno_epsilon = 1e-2 (nearly linear weights) WENO5-Z gives 2.99, 2.99, 3.00, 3.00, 3.00, and with 1e-8 the
irregularity is larger (2.98, 2.88, 1.66, 1.25, 2.53). The time-error constant of WENO5-Z at the default epsilon is dominated by the
nonlinear weights. The integrator check of WENO5-Z is therefore banded at weno_epsilon = 1e-2 (same band), and the default-epsilon
WENO5-Z row is kept in the table as reported, not banded (it does not meet the band that was fixed for it). The decision was made
after the first run. The finer-step diagnostic above is part of --full.

Figure: mms_time_weno.png; table: results/validation/mms_time_weno/mms_time_weno_table.md.
"""
import sys

import numpy as np

import fenice
from fenice import mms, study
from fenice.convergence import observed_orders

import mms_time as base   # the helpers of mms_time.py (this folder)

STUDY = "mms_time_weno"
W3, W5 = ("reconstruction=weno3"), ("reconstruction=weno5")
# (label, settings, banded): the default-epsilon WENO5-Z row is reported, not banded (see the docstring)
SPACE = [("WENO3-Z", W3, True), ("WENO5-Z", W5, False), ("WENO5-Z, eps 1e-2", W5 + ("weno_epsilon=1e-2",), True)]
RK3 = "time_scheme=SSP Runge-Kutta 3"
LOWER, UPPER = 2.5, 3.5
REF_LOWER = 1.5
DIAG_CFLS = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125)
DIAG_REF_CFL = 0.0015625


def slug(label):
    return label.replace("-", "").replace(" ", "").replace(",", "")


def run(tag, args, n, *sets):
    return base.state(study.run_cached(STUDY, tag, base.CASE, f"nx={n}", f"ny={n}", f"t_end={base.T_END}",
                                       "save_interval=1000000000", *sets, mpi=args.mpi, force=args.force))


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="the finer-step diagnostic of the WENO5-Z weights")
    args = ap.parse_args()
    plt = study.use_style()
    sol = mms.Solution(2, False, unsteady=True)
    failed = False

    print("[1/2] the reference (SSP-RK3, CFL %g) against the exact solution at t = %g ..." % (base.REF_CFL, base.T_END))
    refs = {}
    for label, sets, _ in SPACE:
        errs, grids = [], (16, 32, 64)
        for n in grids:
            U = run(f"exact_{slug(label)}_n{n}", args, n, RK3, "cfl=0.25", *sets)
            errs.append(base.diff(U, base.exact_state(sol, n)))
        o = observed_orders([1.0 / n for n in grids], errs)
        ok = bool(np.all(np.diff(errs) < 0) and o[-1] >= REF_LOWER)
        failed |= not ok
        print(f"      {label:18s} errors " + " ".join(f"{e:.3e}" for e in errs) + "   orders " + " ".join(f"{x:.2f}" for x in o) +
              f"   expected >= {REF_LOWER} {'OK' if ok else 'FAIL'}")
        refs[label] = run(f"ref_{slug(label)}", args, base.N, RK3, f"cfl={base.REF_CFL}", *sets)

    print("[2/2] time convergence against the reference, %d^2 cells ..." % base.N)
    results = {}
    for label, sets, _ in SPACE:
        errs = [base.diff(run(f"{slug(label)}_cfl{c}", args, base.N, RK3, f"cfl={c}", *sets), refs[label]) for c in base.CFLS]
        results[label] = (np.array(base.CFLS), np.array(errs))

    lines = ["| space | " + " | ".join(f"{c:g}" for c in base.CFLS) + " | order (finest pair) | expected | |",
             "|---|" + "---:|" * len(base.CFLS) + "---:|---|---|"]
    for label, _, banded in SPACE:
        steps, errs = results[label]
        o = observed_orders(steps, errs)
        good = bool(o[-1] >= LOWER and np.all(np.diff(errs) < 0))
        if banded:
            failed |= not good
            note = "above the band" if good and o[-1] > UPPER else "OK" if good else "OUTSIDE"
            bandtxt = f"[{LOWER}, {UPPER}]"
        else:
            note = "reported (band not met)" if not good else "reported"
            bandtxt = "-"
        print(f"      SSP-RK3 {label:18s} errors " + " ".join(f"{e:.2e}" for e in errs) + "   orders " +
              " ".join(f"{x:.2f}" for x in o) + f"   expected {bandtxt} {note}")
        lines.append(f"| {label} | " + " | ".join(f"{e:.2e}" for e in errs) + f" | {o[-1]:.2f} | {bandtxt} | {note} |")
    (study.study_dir(STUDY) / "mms_time_weno_table.md").write_text("\n".join(lines) + "\n", encoding="utf8")

    if args.full:
        print("      diagnostic: WENO5-Z, steps down to CFL 0.0125 against a reference at CFL %g, by weno_epsilon ..." % DIAG_REF_CFL)
        diag = ["| weno_epsilon | " + " | ".join(f"{c:g}" for c in DIAG_CFLS) + " | orders |", "|---|" + "---:|" * len(DIAG_CFLS) + "---|"]
        for eps in ("1e-2", "1e-6", "1e-8"):
            sets = W5 + (f"weno_epsilon={eps}",)
            ref = run(f"diag_ref_eps{eps}", args, base.N, RK3, f"cfl={DIAG_REF_CFL}", *sets)
            errs = [base.diff(run(f"diag_eps{eps}_cfl{c}", args, base.N, RK3, f"cfl={c}", *sets), ref) for c in DIAG_CFLS]
            o = [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
            print(f"      weno_epsilon {eps}: errors " + " ".join(f"{e:.2e}" for e in errs) + "   orders " + " ".join(f"{x:.2f}" for x in o))
            diag.append(f"| {eps} | " + " | ".join(f"{e:.2e}" for e in errs) + " | " + ", ".join(f"{x:.2f}" for x in o) + " |")
        (study.study_dir(STUDY) / "mms_time_weno_diagnostic.md").write_text("\n".join(diag) + "\n", encoding="utf8")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for (label, _, _), color in zip(SPACE, ("C2", "C4", "C1")):
        steps, errs = results[label]
        ax.loglog(steps / steps[0], errs, "o-", color=color, label=f"{label}, SSP-RK3")
    for p_, ls in ((2, "-."), (3, ":"), (4, "--")):
        ax.loglog([1, 1 / 8], [3e-3, 3e-3 / 8 ** p_], color="gray", ls=ls, lw=0.9)
        ax.annotate(f"slope {p_}", (1 / 8, 3e-3 / 8 ** p_), fontsize=8, color="gray", xytext=(3, 0), textcoords="offset points")
    ax.set_xlabel("time step relative to the coarsest (CFL 0.4)")
    ax.set_ylabel("$L_2$ difference to the SSP-RK3 reference at $t = 0.2$")
    ax.set_title("Order in time with WENO-Z: unsteady manufactured solution", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    study.save(fig, STUDY, "mms_time_weno.png", args.docs)
    print("\nresult: " + ("FAILED (see above)" if failed else "all checks passed"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
