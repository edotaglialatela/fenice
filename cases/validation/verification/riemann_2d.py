"""A1: two-dimensional Riemann problems (four of the Lax-Liu configurations): robustness, symmetry, convergence, AMR.

    python cases/validation/verification/riemann_2d.py            # quick: 100 x 100 and 200 x 200, AMR on configuration 3
    python cases/validation/verification/riemann_2d.py --full     # + a 400 x 400 reference and the convergence bands (medium)
    python cases/validation/verification/riemann_2d.py --docs     # also docs/img/riemann_2d.png

The unit square is cut at (0.5, 0.5) into four constant states; the initial data are those of configurations 3, 4, 6 and 12 as
they are commonly tabulated (quadrants NE, NW, SW, SE; rho, u, v, p), transmissive on every side, gamma = 1.4, MUSCL (MC) +
SSP-RK2, HLLC. There is no exact solution and the repository holds no published contour data: the data below were entered
by hand and are NOT checked against the papers, so these checks make no claim about published shock positions. What they do
establish, with bands fixed before the first run:
  1. every run is finite with rho > 0 and p > 0;
  2. configurations 3 and 4 are symmetric under x <-> y (u <-> v): |rho - rho^T| and |u - v^T| below 1e-9 of the
     largest density / speed, to roundoff, on every grid;
  3. (--full) self-convergence: the mean absolute density difference to the 400 x 400 run (averaged onto the coarse grid)
     decreases with refinement, at an observed rate of at least 0.3 from 100 to 200;
  4. AMR (base 100 x 100, one level, so the finest cells are those of the 200 x 200 run), configuration 3: its density,
     projected onto the 200 x 200 grid, is closer to the uniform 200 x 200 run than the uniform 100 x 100 run is,
     by a factor of at least 0.75 (error <= 0.75 x that of the coarse run).
"""
import numpy as np

import fenice
from fenice import study

STUDY = "riemann_2d"
# configuration -> (t_end, NE, NW, SW, SE), each (rho, u, v, p)
CONFIGS = {
    3: (0.3, (1.5, 0.0, 0.0, 1.5), (0.5323, 1.206, 0.0, 0.3), (0.138, 1.206, 1.206, 0.029), (0.5323, 0.0, 1.206, 0.3)),
    4: (0.25, (1.1, 0.0, 0.0, 1.1), (0.5065, 0.8939, 0.0, 0.35), (1.1, 0.8939, 0.8939, 1.1), (0.5065, 0.0, 0.8939, 0.35)),
    6: (0.3, (1.0, 0.75, -0.5, 1.0), (2.0, 0.75, 0.5, 1.0), (1.0, -0.75, 0.5, 1.0), (3.0, -0.75, -0.5, 1.0)),
    12: (0.25, (1.0, 0.0, 0.0, 1.0), (0.5313, 0.0, 0.7276, 0.4), (0.8, 0.0, 0.0, 1.0), (1.0, 0.7276, 0.0, 0.4)),
}
SYMMETRIC = (3, 4)   # 12 is not: its NW and SE densities differ (0.5313, 1.0)
BAND_SYMMETRY, BAND_RATE, BAND_AMR = 1e-9, 0.3, 0.75
MUSCL = ("use_muscl=true", "muscl_limiter=MC", "time_scheme=SSP Runge-Kutta 2")


def config_case(k):
    t_end, ne, nw, sw, se = CONFIGS[k]
    st = lambda q: f"{{ rho = {q[0]!r}, u = {q[1]!r}, v = {q[2]!r}, p = {q[3]!r} }}"
    def quad(lo, hi, q):
        return f'[[initial.region]]\nshape = "box"\nmin = [{lo[0]!r}, {lo[1]!r}]\nmax = [{hi[0]!r}, {hi[1]!r}]\nstate = {st(q)}\n'
    return f"""[case]
name = "Riemann 2D config {k}"

[mesh]
Lx = 1.0
Ly = 1.0
nx = 100
ny = 100

[boundaries]
bc_left = "transmissive"
bc_right = "transmissive"
bc_bottom = "transmissive"
bc_top = "transmissive"

[numerics]
riemann_solver = "HLLC"
cfl = 0.4

[run]
t_end = {t_end!r}
save_interval = 100000000

[initial]
state = {st(ne)}

{quad((0.0, 0.5), (0.5, 1.0), nw)}
{quad((0.0, 0.0), (0.5, 0.5), sw)}
{quad((0.5, 0.0), (1.0, 0.5), se)}"""


def fields(run_dir, n):
    d = fenice.read_dump(run_dir / "final.npz")
    U = d["U"]
    rho, u, v, p = fenice.primitive(U)
    sq = lambda a: a.reshape(n, n)
    return sq(rho), sq(u), sq(v), sq(p), bool(np.isfinite(U).all())


def project_amr(run_dir, n):
    """The density of an AMR dump on the n x n uniform grid (each leaf fills the cells it covers)."""
    d = fenice.read_dump(run_dir / "final.npz")
    U, meta = d["U"], d["leaf_meta"]
    out = np.zeros((n, n))
    for rho, (x0, x1, y0, y1, _) in zip(U[:, 0], meta):
        i0, i1, j0, j1 = (int(round(v * n)) for v in (x0, x1, y0, y1))
        out[j0:j1, i0:i1] = rho
    return out


def coarsen(a, f):
    n = a.shape[0] // f
    return a.reshape(n, f, n, f).mean(axis=(1, 3))


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--full", action="store_true", help="add the 400 x 400 reference and the convergence bands")
    args = ap.parse_args()
    grids = (100, 200) + ((400,) if args.full else ())
    out, ok, fld = {"configs": {}}, True, {}
    for k in CONFIGS:
        folder = study.write_case(STUDY, f"config{k}", config_case(k))
        res = {"grids": {}}
        for n in grids:
            d = study.run_cached(STUDY, f"c{k}_n{n}", folder, f"nx={n}", f"ny={n}", *MUSCL, mpi=args.mpi, force=args.force)
            rho, u, v, p, finite = fields(d, n)
            fld[k, n] = rho
            g = {"finite": finite, "rho_min": float(rho.min()), "rho_max": float(rho.max()), "p_min": float(p.min())}
            g["positive"] = bool(finite and g["rho_min"] > 0 and g["p_min"] > 0)
            if k in SYMMETRIC:
                g["symmetry_rho"] = float(np.max(np.abs(rho - rho.T)) / rho.max())
                g["symmetry_uv"] = float(np.max(np.abs(u - v.T)) / max(np.abs(u).max(), np.abs(v).max()))
                g["pass_symmetry"] = bool(max(g["symmetry_rho"], g["symmetry_uv"]) <= BAND_SYMMETRY)
            g["pass"] = bool(g["positive"] and g.get("pass_symmetry", True))
            ok &= g["pass"]
            res["grids"][n] = g
            print(f"config {k:2d} {n:3d}x{n:<3d} rho in [{g['rho_min']:.3f}, {g['rho_max']:.3f}]  p_min {g['p_min']:.3g}  "
                  + (f"symmetry {max(g['symmetry_rho'], g['symmetry_uv']):.1e}  " if k in SYMMETRIC else "")
                  + ("pass" if g["pass"] else "FAIL"))
        if args.full:
            ref = fld[k, 400]
            e = {n: float(np.mean(np.abs(fld[k, n] - coarsen(ref, 400 // n)))) for n in (100, 200)}
            rate = float(np.log2(e[100] / e[200]))
            res["l1_to_reference"], res["rate"] = e, rate
            res["pass_rate"] = bool(e[200] < e[100] and rate >= BAND_RATE)
            ok &= res["pass_rate"]
            print(f"config {k:2d} L1 vs 400: 100 -> {e[100]:.4e}, 200 -> {e[200]:.4e}, rate {rate:.2f} "
                  f"({'pass' if res['pass_rate'] else 'FAIL'}, band >= {BAND_RATE})")
        out["configs"][k] = res

    folder = study.write_case(STUDY, "config3", config_case(3))
    d = study.run_cached(STUDY, "c3_amr", folder, "nx=100", "ny=100", *MUSCL, "use_amr=true", "amr_max_level=1",
                         force=args.force)
    amr = project_amr(d, 200)
    e_amr = float(np.mean(np.abs(amr - fld[3, 200])))
    e_coarse = float(np.mean(np.abs(coarsen_up(fld[3, 100]) - fld[3, 200])))
    out["amr"] = {"l1_amr_vs_uniform200": e_amr, "l1_uniform100_vs_uniform200": e_coarse,
                  "pass": bool(e_amr <= BAND_AMR * e_coarse)}
    ok &= out["amr"]["pass"]
    print(f"AMR config 3: |AMR - U200| = {e_amr:.4e}, |U100 - U200| = {e_coarse:.4e} -> "
          f"{'pass' if out['amr']['pass'] else 'FAIL'} (band {BAND_AMR})")
    out["passed"] = bool(ok)
    fenice.write_summary(STUDY, out)

    plt = study.use_style()
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, k in zip(axs, CONFIGS):
        n = grids[-1]
        ax.contour(np.linspace(0, 1, n), np.linspace(0, 1, n), fld[k, n], levels=30, colors="k", linewidths=0.5)
        ax.set_title(f"configuration {k}, {n}$^2$"); ax.set_aspect("equal"); ax.grid(False)
    plt.tight_layout()
    study.save(fig, STUDY, "riemann_2d.png", docs=args.docs)
    print("ALL BANDS PASSED" if ok else "SOME BANDS FAILED")
    return 0 if ok else 1


def coarsen_up(a):
    """A 100 x 100 field on the 200 x 200 grid (piecewise constant)."""
    return np.kron(a, np.ones((2, 2)))


if __name__ == "__main__":
    raise SystemExit(main())
