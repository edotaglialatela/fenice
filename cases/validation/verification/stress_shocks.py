"""B0: strong-shock stress cases. Which of them does the solver survive today?

    python cases/validation/verification/stress_shocks.py            # record the outcome of every case
    python cases/validation/verification/stress_shocks.py --strict   # exit 1 when a case fails its criterion

Cases (1-D tubes on 400 x 4 cells, walls top and bottom, transmissive ends; the 2-D ones are named): Sod with a 1000:1
pressure ratio, the 123 problem (two rarefactions, near vacuum), a Mach 10 and a Mach 20 shock tube, Mach 10 flow past the
immersed cylinder, and the odd-even decoupling test (a Mach 6 planar shock whose centre row starts one cell ahead).
Each runs with 1st-order Godunov and with MUSCL (MC limiter) + SSP-RK2, HLLC.

Pass criterion, fixed before the first run (it is a robustness criterion, not an accuracy one):
  1. the solver finishes (exit code 0);
  2. every cell is finite with rho > 0 and p > 0;
  3. 1-D tubes: mass, momentum and energy change by exactly the time integral of the far-field boundary fluxes
     (relative to the larger of |initial total|, |flux integral|, 1e-300): below 1e-10. The waves must not reach the ends
     (checked on the end cells, which must still hold the initial states).
Reported, not judged: the density L1 error against the exact Riemann solution (1-D) and, for the odd-even test, the spread
of the shock position over the rows in cells (a clean planar shock stays within 1 cell; more is the decoupling).
The outcome of every case is written to results/validation/stress_shocks/summary.json; a failing case is a finding for
track B of docs/proposals/benchmarks-robustness-uq.md, not an error of this script (unless --strict).
"""
import subprocess

import numpy as np

import fenice
from fenice import study
from fenice.exact import riemann

STUDY = "stress_shocks"
GAMMA = 1.4
CONSERVATION_BAND = 1e-10
SCHEMES = {"godunov": ["use_muscl=false"],
           "muscl": ["use_muscl=true", "muscl_limiter=MC", "time_scheme=SSP Runge-Kutta 2"]}
NX, NY = 400, 4


def shock_states(mach, rho1=1.0, p1=1.0):
    """Post-shock (rho, u, p) of a shock of Mach number `mach` moving into gas at rest (rho1, p1), and its speed."""
    g, c1 = GAMMA, np.sqrt(GAMMA * p1 / rho1)
    rho2 = rho1 * (g + 1) * mach ** 2 / ((g - 1) * mach ** 2 + 2)
    p2 = p1 * (2 * g * mach ** 2 - (g - 1)) / (g + 1)
    return (rho2, mach * c1 * (1 - rho1 / rho2), p2), mach * c1


def tube(name, left, right, t_end):
    """A 1-D Riemann problem on [0, 1] (diaphragm at 0.5) as a case file text."""
    s = lambda q: f"{{ rho = {q[0]!r}, u = {q[1]!r}, v = 0.0, p = {q[2]!r} }}"
    return f"""[case]
name = "stress {name}"

[mesh]
Lx = 1.0
Ly = 0.1
nx = {NX}
ny = {NY}

[boundaries]
bc_left = "transmissive"
bc_right = "transmissive"
bc_bottom = "wall"
bc_top = "wall"

[numerics]
riemann_solver = "HLLC"
cfl = 0.4

[run]
t_end = {t_end!r}
save_interval = 1000000000

[initial]
state = {s(right)}

[[initial.region]]
shape = "halfspace"
point = [0.5, 0.0]
normal = [-1.0, 0.0]
state = {s(left)}
"""


def tubes():
    """name -> (case text, left, right, t_end)."""
    out = {}
    L, R, t = (1.0, 0.0, 1000.0), (0.125, 0.0, 1.0), 0.01
    out["sod_1000"] = (tube("sod_1000", L, R, t), L, R, t)
    L, R, t = (1.0, -2.0, 0.4), (1.0, 2.0, 0.4), 0.1
    out["r123"] = (tube("r123", L, R, t), L, R, t)
    for mach in (10, 20):
        post, speed = shock_states(mach)
        R, t = (1.0, 0.0, 1.0), 0.3 / speed   # the shock moves 0.3, from 0.5 to 0.8
        out[f"mach{mach}_tube"] = (tube(f"mach{mach}", post, R, t), post, R, t)
    return out


def flux(q):
    """x-fluxes of (mass, momentum, energy) of the primitive state (rho, u, p)."""
    rho, u, p = q
    E = p / (GAMMA - 1) + 0.5 * rho * u * u
    return np.array([rho * u, rho * u * u + p, (E + p) * u])


def conserved(q):
    rho, u, p = q
    return np.array([rho, rho * u, p / (GAMMA - 1) + 0.5 * rho * u * u])


def totals(U, dx, dy):
    return np.array([U[:, 0].sum(), U[:, 1].sum(), U[:, 3].sum()]) * dx * dy


def judge_tube(run_dir, left, right, t_end):
    """Criteria 2 and 3 and the reported L1 error for a finished 1-D tube."""
    d = fenice.read_dump(run_dir / "final.npz")
    U = d["U"]
    rho, u, v, p = fenice.primitive(U)
    res = {"rho_min": float(rho.min()), "p_min": float(p.min()), "finite": bool(np.isfinite(U).all())}
    dx, dy = 1.0 / NX, 0.1 / NY
    row = lambda a: a.reshape(NY, NX)[NY // 2]
    res["ends_untouched"] = bool(abs(row(rho)[0] - left[0]) < 1e-9 * left[0] and abs(row(rho)[-1] - right[0]) < 1e-9 * right[0])
    init = np.zeros(3)
    xc = (np.arange(NX) + 0.5) * dx
    for xi in xc:
        init += conserved(left if xi < 0.5 else right) * dx * dy * NY
    expected = init + float(d["t"]) * (flux(left) - flux(right)) * 0.1
    got = totals(U, dx, dy)
    scale = np.maximum(np.maximum(np.abs(init), np.abs(expected - init)), 1e-300)
    res["conservation_error"] = float(np.max(np.abs(got - expected) / scale))
    try:
        re, _, _ = riemann(xc, float(d["t"]), left=left, right=right, gamma=GAMMA)
        res["l1_rho"] = float(np.mean(np.abs(row(rho) - re)))
    except Exception as e:   # the exact solver does not cover every state
        res["l1_rho"] = None
    return res


def run_case(tag, case, sets, args):
    """Run (cached); returns (run_dir, None) or (None, reason) when the solver failed."""
    try:
        return study.run_cached(STUDY, tag, case, *sets, mpi=args.mpi, force=args.force), None
    except subprocess.CalledProcessError as e:
        tail = " | ".join((e.stdout or "").strip().splitlines()[-2:] + (e.stderr or "").strip().splitlines()[-2:])
        return None, f"exit {e.returncode}: {tail}"[:300]


def passed(res):
    if not res.get("finished"):
        return False
    ok = res["finite"] and res["rho_min"] > 0 and res["p_min"] > 0
    if "conservation_error" in res:
        ok = ok and res["ends_untouched"] and res["conservation_error"] < CONSERVATION_BAND
    return bool(ok)


def cylinder_case(mach=10.0):
    """The Mach 2 immersed-cylinder case with the inflow and the initial state at `mach` (sound speed 1)."""
    text = (fenice.ROOT / "cases" / "ibm_supersonic_flow_over_cylinder" / "case.toml").read_text()
    text = text.replace('name = "IBM Supersonic Flow over Cylinder"', f'name = "stress cylinder mach {mach:g}"')
    text = text.replace("u_in = 2.0", f"u_in = {mach!r}").replace("u = 2.0,", f"u = {mach!r},")
    text = text.replace("nx = 800", "nx = 100").replace("ny = 1200", "ny = 150")
    text = text.replace("t_end = 5.0", "t_end = 1.0").replace("use_mpi = true", "use_mpi = false")
    return text


def oddeven_case(nx=400, ny=20, mach=6.0, x_shock=1.0):
    """Mach 6 planar shock moving in +x; the centre row starts one cell ahead (Quirk's perturbation)."""
    post, _ = shock_states(mach)
    L = 20.0
    dx = L / nx
    s = lambda q: f"{{ rho = {q[0]!r}, u = {q[1]!r}, v = 0.0, p = {q[2]!r} }}"
    dy = 1.0 / ny
    yc = (ny // 2 + 0.5) * dy
    return f"""[case]
name = "stress odd-even"

[mesh]
Lx = {L!r}
Ly = 1.0
nx = {nx}
ny = {ny}

[boundaries]
bc_left = "transmissive"
bc_right = "transmissive"
bc_bottom = "wall"
bc_top = "wall"

[numerics]
riemann_solver = "HLLC"
cfl = 0.4

[run]
t_end = 2.3
save_interval = 1000000000

[initial]
state = {{ rho = 1.0, u = 0.0, v = 0.0, p = 1.0 }}

[[initial.region]]
shape = "halfspace"
point = [{x_shock!r}, 0.0]
normal = [-1.0, 0.0]
state = {s(post)}

[[initial.region]]
shape = "box"
min = [0.0, {yc - 0.5 * dy!r}]
max = [{x_shock + dx!r}, {yc + 0.5 * dy!r}]
state = {s(post)}
"""


def shock_spread(run_dir, nx=400, ny=20, L=20.0):
    """Spread over the rows of the shock position (x of the half-way density), in cells."""
    d = fenice.read_dump(run_dir / "final.npz")
    rho = d["U"][:, 0].reshape(ny, nx)
    post, _ = shock_states(6.0)
    mid = 0.5 * (1.0 + post[0])
    pos = []
    for r in rho:
        i = int(np.argmax(r < mid))    # first cell ahead of the front (post-shock gas is on the left)
        pos.append(i)
    return float(max(pos) - min(pos)), float(rho.min()), float((d["U"][:, 3]).min())


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--strict", action="store_true", help="exit with 1 when a case fails its criterion")
    args = ap.parse_args()
    results = {}
    for name, (text, left, right, t_end) in tubes().items():
        case = study.write_case(STUDY, name, text)
        for scheme, sets in SCHEMES.items():
            run_dir, why = run_case(f"{name}_{scheme}", case, sets, args)
            res = {"finished": run_dir is not None}
            if run_dir is None:
                res["reason"] = why
            else:
                res.update(judge_tube(run_dir, left, right, t_end))
            res["passed"] = passed(res)
            results[f"{name}/{scheme}"] = res

    case = study.write_case(STUDY, "cylinder_mach10", cylinder_case(10.0))
    for scheme, sets in SCHEMES.items():
        run_dir, why = run_case(f"cylinder_mach10_{scheme}", case, sets + ["nx=100", "ny=150"], args)
        res = {"finished": run_dir is not None}
        if run_dir is None:
            res["reason"] = why
        else:
            U = fenice.read_dump(run_dir / "final.npz")["U"]
            rho, u, v, p = fenice.primitive(U)
            res.update(finite=bool(np.isfinite(U).all()), rho_min=float(rho.min()), p_min=float(p.min()))
        res["passed"] = passed(res)
        results[f"cylinder_mach10/{scheme}"] = res

    case = study.write_case(STUDY, "odd_even", oddeven_case())
    for scheme, sets in SCHEMES.items():
        run_dir, why = run_case(f"odd_even_{scheme}", case, sets, args)
        res = {"finished": run_dir is not None}
        if run_dir is None:
            res["reason"] = why
        else:
            spread, rho_min, E_min = shock_spread(run_dir)
            U = fenice.read_dump(run_dir / "final.npz")["U"]
            rho, u, v, p = fenice.primitive(U)
            res.update(finite=bool(np.isfinite(U).all()), rho_min=float(rho.min()), p_min=float(p.min()),
                       shock_spread_cells=spread)
        res["passed"] = passed(res)
        results[f"odd_even/{scheme}"] = res

    print(f"\n{'case/scheme':28s} {'result':6s} details")
    for key, r in results.items():
        detail = r.get("reason") or ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in r.items() if k not in ("passed", "finished"))
        print(f"{key:28s} {'pass' if r['passed'] else 'FAIL':6s} {detail}")
    failing = [k for k, r in results.items() if not r["passed"]]
    path = fenice.write_summary(STUDY, {"cases": results, "failing": failing, "passed": not failing})
    print(f"\n{len(failing)} of {len(results)} fail today: {', '.join(failing) or 'none'}   ({path})")
    return 1 if (args.strict and failing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
