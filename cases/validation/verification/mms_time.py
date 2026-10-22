"""Order of accuracy in time: an unsteady manufactured solution (Roache 2009, ch. 3 and sec. 8.1).

    python cases/validation/verification/mms_time.py            # results/validation/mms_time/
    python cases/validation/verification/mms_time.py --docs     # also docs/img/mms_time.png

The periodic Euler solution "Manufactured Solution Unsteady Euler" has travelling waves (the time enters their
phases) and a source term S = dU_m/dt + div(F(U_m)) that depends on t; the solver evaluates it at the time of every
stage. The study verifies the time integration inside the whole solver (spatial scheme, limiter, boundary, source):

1. The reference. RK4 with a very small step (CFL 0.0125) is the semi-discrete solution to within its own time error.
   Its error against the exact solution at t = 0.2 must fall on the grids 16^2, 32^2, 64^2: this is what catches a
   wrong dU_m/dt term in the generator or a wrong sign, which a self-convergence in time cannot see (the scheme
   would converge, to the wrong solution).
2. Time convergence. On 32^2 cells the difference to that reference at t = 0.2 (L2 of the conserved state) for Explicit
   Euler, SSP-RK2 and RK4 at CFL 0.4, 0.2, 0.1, 0.05, each with the first-order Godunov scheme in space and with MUSCL
   (Van Leer). A wrong stage time costs one order (the source is then evaluated at the wrong time: RK2 and RK4 would
   fall at 1), and the first-order-in-space runs have a smooth right-hand side, so the limiter's kinks cannot hide it.
3. BDF2 dual time. dt_phys = 0.04, 0.02, 0.01, 0.005 with the inner iteration converged (inner_tol_rel 1e-12). The
   defaults (1e-3, at most 50 iterations) are run too and reported: the iteration error then limits the accuracy.

Expected, fixed before the first run (orders of the finest pair):

* reference against the exact solution, finest pair: first-order space >= 0.7, MUSCL >= 1.5, error falling on every
  refinement;
* Explicit Euler in time: [0.9, 1.1] (both spatial schemes);
* SSP-RK2: [1.8, 2.2] (both);
* RK4: [3.5, 4.5] with first-order space; [2.0, 4.5] with MUSCL (the limiter is not smooth at extrema, which can
  reduce the order of a high-order scheme);
* BDF2 with a converged inner iteration: [1.8, 2.2] (both); the default inner tolerance: no band, reported.

Figure: mms_time.png; table: results/validation/mms_time/mms_time_table.md.
"""
import sys

import numpy as np

import fenice
from fenice import mms, study
from fenice.convergence import observed_orders

STUDY = "mms_time"
CASE = "Manufactured Solution Unsteady Euler"
T_END = 0.2
N = 32
REF_CFL = 0.0125
CFLS = (0.4, 0.2, 0.1, 0.05)
DTS = (0.04, 0.02, 0.01, 0.005)
SPACE = {"1st order": ("use_muscl=false",), "MUSCL": ("use_muscl=true", "muscl_limiter=Van Leer")}
RK = {"Explicit Euler": "Explicit Euler", "SSP-RK2": "SSP Runge-Kutta 2", "RK4": "Runge-Kutta 4"}
BANDS = {("Explicit Euler", "1st order"): (0.9, 1.1), ("Explicit Euler", "MUSCL"): (0.9, 1.1),
         ("SSP-RK2", "1st order"): (1.8, 2.2), ("SSP-RK2", "MUSCL"): (1.8, 2.2),
         ("RK4", "1st order"): (3.5, 4.5), ("RK4", "MUSCL"): (2.0, 4.5),
         ("BDF2", "1st order"): (1.8, 2.2), ("BDF2", "MUSCL"): (1.8, 2.2)}
REF_BANDS = {"1st order": 0.7, "MUSCL": 1.5}


def state(run_dir):
    return fenice.read_dump(run_dir / "final.npz")["U"]


def run(tag, args, n, *sets):
    return state(study.run_cached(STUDY, tag, CASE, f"nx={n}", f"ny={n}", f"t_end={T_END}", "save_interval=1000000000",
                                  *sets, mpi=args.mpi, force=args.force))


def exact_state(sol, n):
    xs = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xs, xs)
    rho, u, v, _, p = sol.exact()(X, Y, np.zeros_like(X), T_END)
    E = p / (sol.gamma - 1) + 0.5 * rho * (u ** 2 + v ** 2)
    return np.stack([a.ravel() for a in (rho, rho * u, rho * v, E)], axis=1)


def diff(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    sol = mms.Solution(2, False, unsteady=True)
    failed = False

    print("[1/3] the reference (RK4, CFL %g) against the exact solution at t = %g ..." % (REF_CFL, T_END))
    refs = {}
    for space, sets in SPACE.items():
        errs, grids = [], (16, 32, 64)
        for n in grids:
            U = run(f"exact_{space.replace(' ', '')}_n{n}", args, n, "time_scheme=Runge-Kutta 4", "cfl=0.25", *sets)
            errs.append(diff(U, exact_state(sol, n)))
        o = observed_orders([1.0 / n for n in grids], errs)
        ok = bool(np.all(np.diff(errs) < 0) and o[-1] >= REF_BANDS[space])
        failed |= not ok
        print(f"      {space:9s} errors " + " ".join(f"{e:.3e}" for e in errs) + "   orders " +
              " ".join(f"{x:.2f}" for x in o) + f"   expected >= {REF_BANDS[space]} {'OK' if ok else 'FAIL'}")
        refs[space] = run(f"ref_{space.replace(' ', '')}", args, N, "time_scheme=Runge-Kutta 4", f"cfl={REF_CFL}", *sets)

    print("[2/3] time convergence against the reference, %d^2 cells ..." % N)
    results, rows = {}, []
    for name, scheme in RK.items():
        for space, sets in SPACE.items():
            errs = [diff(run(f"{name.replace(' ', '').lower()}_{space.replace(' ', '')}_cfl{c}", args, N,
                             f"time_scheme={scheme}", f"cfl={c}", *sets), refs[space]) for c in CFLS]
            results[name, space] = (np.array(CFLS), np.array(errs))
    print("[3/3] BDF2 dual time, inner iteration converged and at the defaults ...")
    for space, sets in SPACE.items():
        for label, extra in (("BDF2", ("inner_tol_rel=1e-12", "max_inner_iter=20000")),
                             ("BDF2 (defaults)", ())):
            errs = [diff(run(f"{label.split()[0].lower()}{'_def' if extra == () else ''}_{space.replace(' ', '')}_dt{dt}",
                             args, N, "time_scheme=BDF2 Dual Time", f"dt_phys={dt}", *extra, *sets), refs[space])
                    for dt in DTS]
            results[label, space] = (np.array(DTS), np.array(errs))

    lines = ["| time scheme | space | " + " | ".join(f"{c:g}" for c in CFLS) + " | order (finest pair) | expected | |",
             "|---|---|" + "---:|" * len(CFLS) + "---:|---|---|"]
    for (name, space), (steps, errs) in results.items():
        o = observed_orders(steps, errs)
        band = BANDS.get((name, space))
        if band is None:
            note, bandtxt = "reported", "-"
        else:
            good = band[0] <= o[-1] <= band[1]
            failed |= not good
            note, bandtxt = ("OK" if good else "OUTSIDE"), f"[{band[0]}, {band[1]}]"
        print(f"      {name:16s} {space:9s} errors " + " ".join(f"{e:.2e}" for e in errs) + "   orders " +
              " ".join(f"{x:.2f}" for x in o) + f"   expected {bandtxt} {note}")
        lines.append(f"| {name} | {space} | " + " | ".join(f"{e:.2e}" for e in errs) + f" | {o[-1]:.2f} | {bandtxt} | {note} |")
    (study.study_dir(STUDY) / "mms_time_table.md").write_text("\n".join(lines) + "\n", encoding="utf8")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {"Explicit Euler": "C3", "SSP-RK2": "C0", "RK4": "C2", "BDF2": "C1", "BDF2 (defaults)": "C4"}
    for ax, space in zip(axs, SPACE):
        for (name, sp_), (steps, errs) in results.items():
            if sp_ != space:
                continue
            # BDF2 on the dt_phys axis, the explicit schemes on the CFL axis: both are proportional to the step
            x = steps / steps[0]
            o = observed_orders(steps, errs)[-1]
            ax.loglog(x, errs, "o-" if "defaults" not in name else "s--", color=colors[name], label=f"{name}: {o:.2f}")
        for p_, ls in ((1, ":"), (2, "-."), (4, "--")):
            ax.loglog([1, 1 / 8], [3e-3, 3e-3 / 8 ** p_], color="gray", ls=ls, lw=0.9)
            ax.annotate(f"slope {p_}", (1 / 8, 3e-3 / 8 ** p_), fontsize=8, color="gray", xytext=(3, 0), textcoords="offset points")
        ax.invert_xaxis()
        ax.set_title(f"space: {space}")
        ax.set_xlabel("time step relative to the coarsest (CFL 0.4 / $\\Delta t_{phys}$ 0.04)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, title="observed order, finest pair")
    axs[0].set_ylabel("$L_2$ difference to the RK4 reference at $t = 0.2$")
    fig.suptitle("Order of accuracy in time: unsteady manufactured solution", fontsize=12)
    fig.tight_layout()
    study.save(fig, STUDY, "mms_time.png", args.docs)
    print("\nresult: " + ("FAILED (see above)" if failed else "all checks passed"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
