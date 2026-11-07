"""Sod shock tube, every scheme against the exact Riemann solution, on 100 / 200 / 400 cells, with a published table.

    python cases/validation/verification/sod_schemes.py            # results/validation/sod_schemes/
    python cases/validation/verification/sod_schemes.py --docs     # also docs/img/sod_schemes.png

Standard Sod problem (left rho = 1, u = 0, p = 1; right 0.125, 0, 0.1; gamma = 1.4; diaphragm at 0.5; t = 0.2), solved on N x 4
square cells (N = 100, 200, 400) with HLLC (the case's default Riemann solver) and the shock-face dissipation at its default:
first order, MUSCL with each limiter (SSP-RK2), WENO3-Z and WENO5-Z (SSP-RK3), and the adaptive mesh (base N / 2 plus one level, the
indicator of the case, so the finest cells are those of N) with MUSCL Van Leer and WENO3-Z.

Reported per run: the L1 error of density, velocity and pressure against the exact solution at the cell centres, absolute (mean) and
relative (sum |q - q_exact| / sum |q_exact|, the normalisation of the published table below), and the observed order between
successive grids. The relative error is the one the reference quotes; which variable it uses is not stated there, so all three are
shown.

Published reference: Table 3 of "An Asymptotic-Preserving all-speed scheme for the Euler and Navier-Stokes equations"
(arXiv:1108.2876), Sod shock tube, second-order scheme, t = 0.2, dt = dx^2: relative L1 error
1.3e-2, 6.8e-3, 3.4e-3 for 100, 200, 400 cells (an order of 1, which the paper attributes to the discontinuities). That scheme, its
time step and the variable are not ours, so the comparison is of magnitude and order, not a reproduction.
"""
import numpy as np

import fenice
from fenice import study
from fenice.exact import riemann

STUDY, CASE = "sod_schemes", "Sod Shock Tube"
NS = (100, 200, 400)
PUBLISHED = {100: 1.3e-2, 200: 6.8e-3, 400: 3.4e-3}
T_END = 0.2
RK2, RK3 = "time_scheme=SSP Runge-Kutta 2", "time_scheme=SSP Runge-Kutta 3"
SCHEMES = [
    ("1st order", ["reconstruction=none", RK2], False),
    ("MUSCL Minmod", ["reconstruction=muscl", "muscl_limiter=Minmod", RK2], False),
    ("MUSCL Van Leer", ["reconstruction=muscl", "muscl_limiter=Van Leer", RK2], False),
    ("MUSCL MC", ["reconstruction=muscl", "muscl_limiter=MC", RK2], False),
    ("MUSCL Koren", ["reconstruction=muscl", "muscl_limiter=Koren", RK2], False),
    ("WENO3-Z", ["reconstruction=weno3", RK3], False),
    ("WENO5-Z", ["reconstruction=weno5", RK3], False),
    ("AMR MUSCL Van Leer", ["reconstruction=muscl", "muscl_limiter=Van Leer", RK2], True),
    ("AMR WENO3-Z", ["reconstruction=weno3", RK3], True),
]


def run(label, sets, amr, n, args):
    base = n // 2 if amr else n
    grid = [f"nx={base}", f"ny={4 if not amr else 2}", f"Lx=1.0", f"Ly={(4 if not amr else 2) / base}"]
    amr_sets = ["use_amr=true", "amr_max_level=1"] if amr else []
    tag = f"{label.replace(' ', '_').replace('-', '').lower()}_n{n}"
    d = study.run_cached(STUDY, tag, CASE, *grid, f"t_end={T_END}", "save_interval=1000000000", *amr_sets, *sets,
                         mpi=args.mpi, force=args.force)
    z = fenice.read_dump(d / "final.npz")
    rho, u, _, p = fenice.primitive(z["U"])
    if "leaf_meta" in z:
        m = z["leaf_meta"]
        x, y, w = 0.5 * (m[:, 0] + m[:, 1]), 0.5 * (m[:, 2] + m[:, 3]), (m[:, 1] - m[:, 0]) * (m[:, 3] - m[:, 2])
    else:
        x = np.tile(np.arange(base) + 0.5, len(rho) // base) / base
        w = np.ones_like(x)
    re, ue, pe = riemann(x, T_END)
    out = {}
    for name, got, ex in (("rho", rho, re), ("u", u, ue), ("p", p, pe)):
        e = np.abs(got - ex)
        out[name] = (np.sum(w * e) / np.sum(w), np.sum(w * e) / np.sum(w * np.abs(ex)))
    return out, len(rho)


def main():
    ap = study.parser(__doc__.splitlines()[0])
    args = ap.parse_args()
    rows = ["| scheme | variable | relative L1 error, N = 100 / 200 / 400 | orders | absolute L1, N = 400 | cells (N = 400) |", "|---|---|---|---|---|---|"]
    table = {}
    for label, sets, amr in SCHEMES:
        res = [run(label, sets, amr, n, args) for n in NS]
        table[label] = res
        for var in ("rho", "u", "p"):
            rel = [r[0][var][1] for r in res]
            ords = [np.log2(rel[i] / rel[i + 1]) for i in range(2)]
            rows.append(f"| {label} | {var} | " + " / ".join(f"{v:.2e}" for v in rel) + " | " + ", ".join(f"{o:.2f}" for o in ords) +
                        f" | {res[-1][0][var][0]:.2e} | {res[-1][1]} |")
            print(rows[-1], flush=True)
    pub = [PUBLISHED[n] for n in NS]
    rows.append("| published (arXiv:1108.2876, 2nd order) | not stated | " + " / ".join(f"{v:.1e}" for v in pub) + " | " +
                ", ".join(f"{np.log2(pub[i] / pub[i + 1]):.2f}" for i in range(2)) + " | | |")
    print(rows[-1])
    path = study.study_dir(STUDY) / "sod_schemes.md"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
