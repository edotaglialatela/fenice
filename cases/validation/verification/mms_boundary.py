"""Order of the manufactured solutions with non-periodic sides (walls, outflow), per reconstruction and boundary closure.

    python cases/validation/verification/mms_boundary.py                       # first-order boundary faces (the default)
    python cases/validation/verification/mms_boundary.py --boundary-order 2    # the second-order closure
    python cases/validation/verification/mms_boundary.py --schemes muscl,weno3 --euler-only

The 2-D solutions of `fenice.mms` with non-periodic sides: moving and slip walls, no-slip wall with outflow, and a box with
no-slip and slip walls on all four sides (the corners; --kinds box), each
as Navier-Stokes and as Euler (the Euler ones isolate the convective closure from the viscous one). Their variables are
even or odd about the boundaries exactly as the solver's ghost cells assume, so the ghost state IS the extension of the
solution across the boundary and the boundary closure itself is verified. They say nothing about the accuracy of a
transmissive or outlet condition on a real flow, whose normal gradient does not vanish.

Both the initial state and the source terms of the manufactured solutions are evaluated at the cell centres, which is
a second-order approximation of the cell averages: no order above 2 can show here, whatever the scheme. The table
reports the observed order of every pair of grids (16 -> 32 -> 64 -> 128) of the worst variable, in L2 and Linf, and the
density error on the finest grid. The reference table of the first-order boundary faces is mms_boundary_baseline.md.

Bands, fixed before the first run with boundary_order = 2 (the finest pair of grids, worst variable): L2 order at least
1.7 and Linf order at least 1.3 for every scheme on the Euler solutions; the same on the Navier-Stokes ones if the
viscous wall closure does not limit them. Reported as 'ok' / 'below'; the exit code is 0 either way (a study, not a test).
"""
import numpy as np

import fenice
from fenice import mms, study
from fenice.convergence import observed_orders

import mms as M   # the periodic-solution study: centres, primitive, grid_sets, T_END, VARS

STUDY = "mms_boundary"
GRIDS = (16, 32, 64, 128)
SCHEMES = {
    "muscl": ("MUSCL VL, SSP-RK2", ["reconstruction=muscl", "muscl_limiter=Van Leer", "time_scheme=SSP Runge-Kutta 2"]),
    "weno3": ("WENO3-Z, SSP-RK3", ["reconstruction=weno3", "time_scheme=SSP Runge-Kutta 3"]),
    "weno5": ("WENO5-Z, SSP-RK3", ["reconstruction=weno5", "time_scheme=SSP Runge-Kutta 3"]),
}
BAND_L2, BAND_LINF = 1.7, 1.3


KINDS = ("walls", "outflow", "box")


def solutions(euler_only, viscous_only, kinds=KINDS):
    out = []
    for kind in kinds:
        for viscous in (False, True):
            if (euler_only and viscous) or (viscous_only and not viscous):
                continue
            out.append(mms.Solution(dim=2, viscous=viscous, kind=kind))
    return out


def errors(sol, n, key, sets, args):
    case = study.write_case(STUDY, sol.folder(), sol.case_toml())
    tag = f"{sol.folder()}_{key}_{args.tag}_n{n}"
    d = study.run_cached(STUDY, tag, case, *M.grid_sets(sol, n), f"t_end={M.T_END}", "save_interval=1000000000", *sets,
                         mpi=args.mpi, force=args.force)
    U = fenice.read_dump(d / "final.npz")["U"]
    got = M.primitive(U, sol.gamma, sol.dim)
    X, Y, Z = M.centres(n, sol.dim)
    want = sol.exact()(X, Y, Z)
    out = {}
    for k, (g, w) in enumerate(zip(got, want)):
        if M.VARS[k] == "w":
            continue
        e = np.abs(g - np.asarray(w).ravel())
        out[M.VARS[k]] = {"L2": float(np.sqrt(np.mean(e ** 2))), "Linf": float(e.max())}
    return out


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--boundary-order", type=int, default=1, choices=(1, 2))
    ap.add_argument("--schemes", default="muscl,weno3,weno5")
    ap.add_argument("--euler-only", action="store_true")
    ap.add_argument("--viscous-only", action="store_true")
    ap.add_argument("--kinds", default=",".join(KINDS), help="solutions to run; with a subset the table rows of the others are kept")
    args = ap.parse_args()
    args.tag = f"b{args.boundary_order}"
    h = [1.0 / n for n in GRIDS]
    print(f"boundary_order = {args.boundary_order}; observed order per pair of grids {GRIDS}, worst variable\n")
    rows = ["| solution | scheme | norm | worst variable | orders per grid pair | finest rho error | band |", "|---|---|---|---|---|---|---|"]
    kinds = tuple(args.kinds.split(","))
    for sol in solutions(args.euler_only, args.viscous_only, kinds):
        for key in args.schemes.split(","):
            label, sets = SCHEMES[key]
            sets = list(sets) + [f"boundary_order={args.boundary_order}"]
            errs = [errors(sol, n, key, sets, args) for n in GRIDS]
            for norm, band in (("L2", BAND_L2), ("Linf", BAND_LINF)):
                names = list(errs[0])
                ords = {v: observed_orders(h, np.array([e[v][norm] for e in errs])) for v in names}
                worst = min(names, key=lambda v: ords[v][-1])
                ok = ords[worst][-1] >= band
                rows.append(f"| {sol.name()} | {label} | {norm} | {worst} | " + ", ".join(f"{x:.2f}" for x in ords[worst]) +
                            f" | {errs[-1]['rho'][norm]:.2e} | {'ok' if ok else 'below'} ({band}) |")
                print(rows[-1], flush=True)
    path = study.study_dir(STUDY) / f"mms_boundary_order{args.boundary_order}.md"
    if path.exists() and kinds != KINDS:   # a subset: replace the rows of the solutions run, keep the rest of the file
        ran = {r.split(" | ")[0] for r in rows[2:]}
        old = path.read_text(encoding="utf-8").splitlines()
        rows = [l for l in old if not (l.startswith("| ") and l.split(" | ")[0] in ran)] + rows[2:]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
