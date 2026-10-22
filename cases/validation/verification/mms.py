"""Code verification by manufactured solutions: Euler and Navier-Stokes, 2-D and 3-D (Roache 2009, ch. 3).

    python cases/validation/verification/mms.py             # quick: up to 64^2 and 48^3, a few minutes
    python cases/validation/verification/mms.py --full      # up to 128^2 and 96^3 (about 20 minutes), as in the docs
    python cases/validation/verification/mms.py --full --docs      # also docs/img/mms.png

A smooth, non-separable, periodic solution is manufactured for each case (cases/manufactured_solution*/, written by
`python -m fenice.mms --write`): the source terms that make it an exact solution of the equations the solver
discretises are evaluated at the cell centres, the run starts from the exact state and the error against it is
measured at a fixed time on a sequence of grids refined by 2 in every direction, with the CFL fixed. Four steps:

0. the committed case files are what the generator writes;
1. transcription: on a uniform state one Explicit Euler step must move every cell by dt S(x, y, z), with S evaluated
   independently by sympy: this isolates the parsing of the expressions and their slots in the state;
2. residual: from the exact state, one Euler step gives (U1 - U0)/dt = R_h(U_m) + S, the local truncation error,
   which must fall with refinement (a floor would mean the generator's equations differ from the solver's);
3. order: the error of rho, u, v, (w), p at t = 0.5 in the L1, L2 and Linf norms and the observed orders, for the
   second-order scheme (MUSCL, Van Leer, SSP-RK2) and the first-order one (Godunov, SSP-RK2).

Expected orders, fixed before the first run: 1.0 +- 0.1 for the first-order scheme in every norm; for MUSCL
between 1.5 and 2.5 in L1 and L2 and between 1.0 and 2.5 in Linf (a TVD limiter clips the reconstruction at smooth
extrema, which costs order, most of all in Linf; an order above 2.5 would be suspicious). The orders are those of the
finest pair of grids. The quick mode stops at 64^2 and 48^3, where the errors are still pre-asymptotic (the first
order from below, the 3-D MUSCL Linf a little above 2.5): it allows 0.1 below the lower bounds and enforces no
upper bound, to catch gross regressions; the bands as fixed, and the diagnostic of the documented deviation, need
--full. Figure: mms.png; table: results/validation/mms/mms_table.md.
"""
import re
import sys

import numpy as np

import fenice
from fenice import mms, study
from fenice.convergence import observed_orders

STUDY = "mms"
T_END = 0.5
GRIDS_FULL = {2: (16, 32, 64, 128), 3: (12, 24, 48, 96)}
GRIDS_QUICK = {2: (16, 32, 64), 3: (12, 24, 48)}
GRIDS = GRIDS_QUICK
SCHEMES = {"MUSCL": "true", "1st order": "false"}
NORMS = ("L1", "L2", "Linf")
BANDS = {"1st order": {"L1": (0.9, 1.1), "L2": (0.9, 1.1), "Linf": (0.9, 1.1)},
         "MUSCL": {"L1": (1.5, 2.5), "L2": (1.5, 2.5), "Linf": (1.0, 2.5)}}
VARS = ("rho", "u", "v", "w", "p")
# Deviations from the bands above that were observed and explained (the bands stay as they were fixed): the
# first-order scheme on the Euler case approaches order 1 from below, because an h^2 term of opposite sign competes
# with the leading h term (Roache 2009, sec. 8.1.1); the truncation error of step 2 is first order, 0.95 to 0.99.
KNOWN = {("Manufactured Solution Euler", "1st order"): "pre-asymptotic, see the diagnostic below"}


def centres(n, dim):
    xs = (np.arange(n) + 0.5) / n
    if dim == 2:
        X, Y = np.meshgrid(xs, xs)
        return X, Y, np.zeros_like(X)
    Z, Y, X = np.meshgrid(xs, xs, xs, indexing="ij")
    return X, Y, Z


def primitive(U, gamma, dim):
    rho = U[:, 0]
    u, v = U[:, 1] / rho, U[:, 2] / rho
    w = U[:, 4] / rho if dim == 3 else np.zeros_like(rho)
    p = (gamma - 1) * (U[:, 3] - 0.5 * rho * (u ** 2 + v ** 2 + w ** 2))
    return rho, u, v, w, p


def exact_conservative(sol, n):
    X, Y, Z = centres(n, sol.dim)
    rho, u, v, w, p = sol.exact()(X, Y, Z)
    E = p / (sol.gamma - 1) + 0.5 * rho * (u ** 2 + v ** 2 + w ** 2)
    cols = [rho, rho * u, rho * v, E] + ([rho * w] if sol.dim == 3 else [])
    return np.stack([c.ravel() for c in cols], axis=1)


def grid_sets(sol, n):
    return [f"nx={n}", f"ny={n}"] + ([f"nz={n}"] if sol.dim == 3 else [])


def check_generator():
    cases = fenice.ROOT / "cases"
    ok = True
    for sol in mms.default_solutions():
        path = cases / sol.folder() / "case.toml"
        same = path.exists() and path.read_bytes().replace(b"
", b"
") == sol.case_toml().encode("utf8")   # CRLF checkout
        ok = ok and same
        print(f"      {sol.name():28s} case file {'matches the generator' if same else 'DIFFERS: run python -m fenice.mms --write'}")
    return ok


def transcription(sol, args):
    """One Euler step on a uniform state against sympy's S at the cell centres."""
    d = study.study_dir(STUDY) / f"transcription_{sol.folder()}"
    d.mkdir(parents=True, exist_ok=True)
    uniform = "state = { rho = 1.0, u = 0.0, v = 0.0, p = 1.0" + (", w = 0.0" if sol.dim == 3 else "") + " }"
    text = re.sub(r"^state = \{.*\}$", uniform, sol.case_toml(), flags=re.M)
    (d / "case.toml").write_bytes(text.encode("utf8"))
    n = 16 if sol.dim == 2 else 8
    fenice.run(str(d), *grid_sets(sol, n), "time_scheme=Explicit Euler", "max_steps=1", "save_interval=1000000000",
               out_root=d / "out", dump=d / "final.npz")
    dump = fenice.read_dump(d / "final.npz")
    U, dt = dump["U"], float(dump["t"])
    X, Y, Z = centres(n, sol.dim)
    S = sol.source_functions()
    names = ["source_rho", "source_mx", "source_my", "source_E"] + (["source_mz"] if sol.dim == 3 else [])
    U0 = np.array([1.0, 0.0, 0.0, 1.0 / (sol.gamma - 1)] + ([0.0] if sol.dim == 3 else []))
    err = scale = 0.0
    for q, name in enumerate(names):
        want = np.broadcast_to(S[name](X, Y, Z), X.shape).ravel()
        err = max(err, np.abs((U[:, q] - U0[q]) / dt - want).max())
        scale = max(scale, np.abs(want).max())
    return err, scale


def run_error(sol, n, scheme, args):
    d = study.run_cached(STUDY, f"{sol.folder()}_{scheme.replace(' ', '')}_n{n}", sol.name(), *grid_sets(sol, n),
                         f"t_end={T_END}", "save_interval=1000000000", f"use_muscl={SCHEMES[scheme]}",
                         mpi=args.mpi, force=args.force)
    U = fenice.read_dump(d / "final.npz")["U"]
    got = primitive(U, sol.gamma, sol.dim)
    X, Y, Z = centres(n, sol.dim)
    want = sol.exact()(X, Y, Z)
    out = {}
    for k, (g, w) in enumerate(zip(got, want)):
        if VARS[k] == "w" and sol.dim == 2:
            continue
        e = np.abs(g - np.asarray(w).ravel())
        out[VARS[k]] = {"L1": e.mean(), "L2": np.sqrt(np.mean(e ** 2)), "Linf": e.max()}
    return out


def residual(sol, n, scheme, args):
    d = study.run_cached(STUDY, f"resid_{sol.folder()}_{scheme.replace(' ', '')}_n{n}", sol.name(), *grid_sets(sol, n),
                         "time_scheme=Explicit Euler", "max_steps=1", "save_interval=1000000000",
                         f"use_muscl={SCHEMES[scheme]}", mpi=args.mpi, force=args.force)
    dump = fenice.read_dump(d / "final.npz")
    r = (dump["U"] - exact_conservative(sol, n)) / float(dump["t"])
    return float(np.sqrt(np.mean(r ** 2)))


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="the fine grids too (128^2, 96^3) and the Euler diagnostic")
    args = ap.parse_args()
    global GRIDS
    GRIDS = GRIDS_FULL if args.full else GRIDS_QUICK
    plt = study.use_style()
    sols = mms.default_solutions()
    failed = False

    print("[0/3] generator and case files ...")
    failed |= not check_generator()

    print("[1/3] transcription of the source terms (uniform state, one Euler step) ...")
    for sol in sols:
        err, scale = transcription(sol, args)
        ok = err < 1e-12 * scale
        failed |= not ok
        print(f"      {sol.name():28s} max |(U1 - U0)/dt - S| = {err:.2e} (max |S| = {scale:.1f}) {'OK' if ok else 'FAIL'}")

    print("[2/3] truncation error: rms of (U1 - U0)/dt from the exact state ...")
    resid = {}
    for sol in sols:
        for scheme in SCHEMES:
            r = [residual(sol, n, scheme, args) for n in GRIDS[sol.dim]]
            resid[sol.name(), scheme] = r
            o = observed_orders([1.0 / n for n in GRIDS[sol.dim]], r)
            ok = r[-1] < 0.5 * r[0] and np.all(np.diff(r) < 0)
            failed |= not ok
            print(f"      {sol.name():28s} {scheme:9s} " + " ".join(f"{v:.3e}" for v in r) + "   orders " +
                  " ".join(f"{x:.2f}" for x in o) + f"  {'OK' if ok else 'FAIL: no convergence'}")

    print(f"[3/3] error at t = {T_END} against the manufactured solution ...")
    rows, curves = [], {}
    for sol in sols:
        grids = GRIDS[sol.dim]
        h = [1.0 / n for n in grids]
        for scheme in SCHEMES:
            errs = [run_error(sol, n, scheme, args) for n in grids]
            names = [v for v in VARS if v in errs[0]]
            for norm in NORMS:
                e = {v: np.array([er[v][norm] for er in errs]) for v in names}
                pair = {v: observed_orders(h, e[v])[-1] for v in names}   # finest pair
                fit = {v: float(np.polyfit(np.log(h), np.log(e[v]), 1)[0]) for v in names}
                worst = min(pair, key=pair.get)
                lo, hi = BANDS[scheme][norm]
                # quick mode, to catch gross regressions: 0.1 slack below the lower bound (the coarse grids approach
                # the order from below), no upper bound
                ok = all(lo - (0.0 if args.full else 0.1) <= pair[v] <= (hi if args.full else np.inf) for v in names)
                known = (sol.name(), scheme) in KNOWN
                failed |= not ok and not known
                curves[sol.name(), scheme, norm] = (h, np.max([e[v] for v in names], axis=0), pair[worst])
                rows.append((sol.name(), scheme, norm, pair, fit, ok or known and "documented"))
                print(f"      {sol.name():28s} {scheme:9s} {norm:4s} orders " +
                      " ".join(f"{v} {pair[v]:.2f}" for v in names) + f"   finest-grid error (max over variables) "
                      f"{curves[sol.name(), scheme, norm][1][-1]:.2e}   expected [{lo}, {hi}] "
                      f"{'OK' if ok else 'OUTSIDE (documented)' if known else 'OUTSIDE'}")

    if args.full:   # diagnostic of the documented deviation: e = a h + b h^2 on the grids of the study and one finer
        sol = next(s for s in sols if s.name() == "Manufactured Solution Euler")
        grids = list(GRIDS[2]) + [2 * GRIDS[2][-1]]
        errs = [run_error(sol, n, "1st order", args) for n in grids]
        h = np.array([1.0 / n for n in grids])
        print(f"      diagnostic, {sol.name()}, 1st order, L1 error on {grids}:")
        for v in ("rho", "u", "v", "p"):
            e = np.array([er[v]["L1"] for er in errs])
            A = np.stack([h, h ** 2], axis=1)
            a, b = np.linalg.lstsq(A, e, rcond=None)[0]
            misfit = np.max(np.abs(A @ [a, b] / e - 1))
            print(f"        {v:3s} pair orders " + " ".join(f"{o:.2f}" for o in observed_orders(h, e)) +
                  f";  e = {a:.3f} h {b:+.2f} h^2 (largest misfit {misfit:.0%})")

    # table for the documentation
    lines = ["| case | scheme | norm | " + " | ".join(VARS) + " | expected | |", "|---|---|---|" + "---:|" * len(VARS) + "---|---|"]
    for name, scheme, norm, pair, fit, ok in rows:
        lo, hi = BANDS[scheme][norm]
        lines.append(f"| {name} | {scheme} | {norm} | " + " | ".join(f"{pair[v]:.2f}" if v in pair else "-" for v in VARS) +
                     f" | {lo} to {hi} | {'ok' if ok is True else 'outside (explained)' if ok == 'documented' else 'outside'} |")
    (study.study_dir(STUDY) / "mms_table.md").write_text("\n".join(lines) + "\n", encoding="utf8")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = {"Manufactured Solution": "C0", "Manufactured Solution Euler": "C1", "Manufactured Solution 3D": "C2"}
    marks = {"MUSCL": "o-", "1st order": "s--"}
    for sol in sols:
        h = [1.0 / n for n in GRIDS[sol.dim]]
        for scheme in SCHEMES:
            ax[0].loglog(h, resid[sol.name(), scheme], marks[scheme], color=colors[sol.name()], label=f"{sol.name()}, {scheme}")
            for k, norm in enumerate(("L2", "Linf")):
                hh, e, p = curves[sol.name(), scheme, norm]
                ax[k + 1].loglog(hh, e, marks[scheme], color=colors[sol.name()], label=f"{sol.name()}, {scheme}: {p:.2f}")
    for a, title, ylab in ((ax[0], "Truncation error from the exact state", "rms of $(U_1 - U_0)/\\Delta t$"),
                           (ax[1], f"$L_2$ error at $t = {T_END}$ (worst variable)", "$L_2$ error"),
                           (ax[2], f"$L_\\infty$ error at $t = {T_END}$ (worst variable)", "$L_\\infty$ error")):
        lim = a.get_xlim()
        for p, ls in ((1, ":"), (2, "-.")):
            y0 = a.get_ylim()[1] * 0.6
            a.loglog(lim, [y0 * (x / lim[1]) ** p for x in lim], "gray", ls=ls, lw=1, label=f"slope {p}")
        a.set_title(title, fontsize=11)
        a.set_xlabel("$h$")
        a.set_ylabel(ylab)
        a.legend(fontsize=7)
    fig.suptitle("Manufactured solutions, periodic: Euler and Navier-Stokes, 2-D and 3-D (legend: observed order, finest pair)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    study.save(fig, STUDY, "mms.png", args.docs)
    print("\nresult: " + ("FAILED (see above)" if failed else "all checks passed"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
