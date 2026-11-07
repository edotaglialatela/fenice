"""Order of the manufactured solutions with non-periodic sides on an adaptive mesh with a static refined region.

    python cases/validation/verification/mms_amr_boundary.py                       # both boundary closures
    python cases/validation/verification/mms_amr_boundary.py --boundary-order 2 --schemes muscl,weno3

The solutions and the boundary closures of `mms_boundary.py`, run with `use_amr`, one level of refinement, and a refined
region that does not change in time (`amr_force_box`; the indicator thresholds are set so high that nothing else is refined or
joined): a strip across the domain whose two seams meet the walls (the walls solution: a strip in x; the outflow one: in y),
and for the box solution a quarter of the domain that contains one corner. So the level jump touches the boundary faces and
the first interior faces next to the wall, the leaves with a finer or coarser neighbour along the wall, and (box) a corner.

The error is the area-weighted L2 and the maximum over the leaves of the worst variable against the exact solution at the
leaf centres, as a function of the base grid (16 -> 32 -> 64; the refined part always has twice that resolution).

Bands, fixed before the first run, for boundary_order = 2 (the finest pair of grids, worst variable): L2 order at least 1.5 and
Linf order at least 1.0 on the Euler solutions (the uniform bands, 1.7 and 1.3, less an allowance of 0.2 and 0.3 for the seam,
which is second order and has a first-order interpolated jump closure at the corner of a coarse and a fine cell); the same on
the Navier-Stokes ones if the viscous wall closure does not limit them. A result below them is reported as below, never moved.
The study is not a test: the exit code is 0 either way.
"""
import numpy as np

import fenice
from fenice import mms, study
from fenice.convergence import observed_orders

import mms as M   # the periodic-solution study: centres, primitive, VARS, T_END

STUDY = "mms_amr_boundary"
GRIDS = (16, 32, 64)
SCHEMES = {
    "muscl": ("MUSCL VL, SSP-RK2", ["reconstruction=muscl", "muscl_limiter=Van Leer", "time_scheme=SSP Runge-Kutta 2"]),
    "weno3": ("WENO3-Z, SSP-RK3", ["reconstruction=weno3", "time_scheme=SSP Runge-Kutta 3"]),
}
BAND_L2, BAND_LINF = 1.5, 1.0
BOX = {"walls": "0.25 0.75 0 1", "outflow": "0 1 0.25 0.75", "box": "0 0.5 0 0.5"}
FROZEN = ["use_amr=true", "amr_max_level=1", "amr_refine_threshold=1e9", "amr_derefine_threshold=1e9", "amr_refine_every=1000000"]


def errors(sol, n, key, sets, args):
    case = study.write_case(STUDY, sol.folder(), sol.case_toml())
    tag = f"{sol.folder()}_{key}_{args.tag}_n{n}"
    d = study.run_cached(STUDY, tag, case, f"nx={n}", f"ny={n}", f"t_end={M.T_END}", "save_interval=1000000000",
                         *FROZEN, f"amr_force_box={BOX[sol.kind]}", *sets, mpi=args.mpi, force=args.force)
    z = fenice.read_dump(d / "final.npz")
    U, meta = z["U"], z["leaf_meta"]   # leaf_meta columns: x0, x1, y0, y1, level
    x = 0.5 * (meta[:, 0] + meta[:, 1])
    y = 0.5 * (meta[:, 2] + meta[:, 3])
    area = (meta[:, 1] - meta[:, 0]) * (meta[:, 3] - meta[:, 2])
    got = M.primitive(U, sol.gamma, sol.dim)
    want = sol.exact()(x, y, np.zeros_like(x))
    out = {}
    for k, (g, w) in enumerate(zip(got, want)):
        if M.VARS[k] == "w":
            continue
        e = np.abs(g - np.asarray(w))
        out[M.VARS[k]] = {"L2": float(np.sqrt(np.sum(area * e ** 2) / np.sum(area))), "Linf": float(e.max())}
    return out, len(U)


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--boundary-order", type=int, default=0, choices=(0, 1, 2), help="0: both")
    ap.add_argument("--schemes", default="muscl,weno3")
    ap.add_argument("--euler-only", action="store_true")
    args = ap.parse_args()
    h = [1.0 / n for n in GRIDS]
    rows = ["| solution | scheme | boundary order | norm | worst variable | orders per grid pair | finest rho error | leaves | band |",
            "|---|---|---|---|---|---|---|---|---|"]
    for bo in ((1, 2) if args.boundary_order == 0 else (args.boundary_order,)):
        args.tag = f"b{bo}"
        for kind in ("walls", "outflow", "box"):
            for viscous in (False, True):
                if args.euler_only and viscous:
                    continue
                sol = mms.Solution(dim=2, viscous=viscous, kind=kind)
                for key in args.schemes.split(","):
                    label, sets = SCHEMES[key]
                    sets = list(sets) + ([f"boundary_order={bo}"] if bo == 2 else [])
                    res = [errors(sol, n, key, sets, args) for n in GRIDS]
                    errs, leaves = [r[0] for r in res], res[-1][1]
                    for norm, band in (("L2", BAND_L2), ("Linf", BAND_LINF)):
                        names = list(errs[0])
                        ords = {v: observed_orders(h, np.array([e[v][norm] for e in errs])) for v in names}
                        worst = min(names, key=lambda v: ords[v][-1])
                        ok = ords[worst][-1] >= band
                        rows.append(f"| {sol.name()} | {label} | {bo} | {norm} | {worst} | " +
                                    ", ".join(f"{x:.2f}" for x in ords[worst]) +
                                    f" | {errs[-1]['rho'][norm]:.2e} | {leaves} | {'ok' if ok else 'below'} ({band}) |")
                        print(rows[-1], flush=True)
    path = study.study_dir(STUDY) / "mms_amr_boundary.md"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
