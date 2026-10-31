"""Gate of the shock-face dissipation (`shock_blend`): does it cure odd-even decoupling without costing accuracy or robustness?

    python cases/validation/verification/shock_blend.py            # results/validation/shock_blend/
    python cases/validation/verification/shock_blend.py --blend 0.2

Compares `shock_blend = 0` (off) with the value under test (default 0.2, chosen from a sweep of the odd-even test of
stress_shocks.py over 0.2 ... 5 before these bands were written; 0.2 was the only value of the sweep that brought the
first-order spread under 1 cell). Bands, fixed before the first run with the value on:
  1. odd-even test (Mach 6 shock, centre row one cell ahead): the shock position spread over the rows is at most 1 cell with
     1st-order Godunov and with MUSCL (off: 9.6 and 0.2 cells);
  2. Sod, 200 cells, to t = 0.2: the L1 density error against the exact solution with the cure on is at most 1.10 times the
     error with it off, for 1st order and for MUSCL + SSP-RK2 (MC);
  3. every strong-shock tube of stress_shocks.py (pressure ratio 1000, 123 with and without vacuum, Mach 10 / 20 / 100) still
     passes its robustness criterion with the cure on.
The smooth cases (vortex, Taylor-Green, manufactured solutions) must be bit-identical with it on: the regression and the
unit tests with the default switched on say so; this script does not repeat that.
"""
import numpy as np

import fenice
from fenice import study
from fenice.exact import riemann

import stress_shocks as ss

STUDY = "shock_blend"
BAND_SPREAD, BAND_L1 = 1.0, 1.10


def spread(run_dir, nx=400, ny=20):
    rho = fenice.read_dump(run_dir / "final.npz")["U"][:, 0].reshape(ny, nx)
    post, _ = ss.shock_states(6.0)
    mid = 0.5 * (1.0 + post[0])
    pos = []
    for r in rho:
        i = int(np.argmax(r < mid))
        a, b = r[i - 1], r[i]
        pos.append(i - 1 + (a - mid) / (a - b))
    return float(max(pos) - min(pos))


def sod_l1(run_dir, nx=200, ny=20):
    d = fenice.read_dump(run_dir / "final.npz")
    rho = d["U"][:, 0].reshape(ny, nx)[ny // 2]
    re, _, _ = riemann((np.arange(nx) + 0.5) / nx, float(d["t"]))
    return float(np.mean(np.abs(rho - re)))


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--blend", type=float, default=0.2, help="shock_blend under test")
    args = ap.parse_args()
    out, ok = {"blend": args.blend}, True
    odd = study.write_case(ss.STUDY, "odd_even", ss.oddeven_case())
    sod_case = "Sod Shock Tube"
    for scheme, sets in ss.SCHEMES.items():
        res = {}
        for b in (0.0, args.blend):
            tag = f"b{b:g}_{scheme}"
            sp = spread(study.run_cached(STUDY, f"odd_even_{tag}", odd, *sets, f"shock_blend={b}", mpi=args.mpi, force=args.force))
            l1 = sod_l1(study.run_cached(STUDY, f"sod_{tag}", sod_case, *sets, "nx=200", "ny=20", "save_interval=1000000000",
                                         f"shock_blend={b}", mpi=args.mpi, force=args.force))
            res[b] = (sp, l1)
        (sp0, l10), (sp1, l11) = res[0.0], res[args.blend]
        r = {"spread_off": sp0, "spread_on": sp1, "l1_off": l10, "l1_on": l11, "l1_ratio": l11 / l10,
             "pass_spread": bool(sp1 <= BAND_SPREAD), "pass_l1": bool(l11 <= BAND_L1 * l10)}
        out[scheme] = r
        ok &= r["pass_spread"] and r["pass_l1"]
        print(f"{scheme:8s} odd-even spread {sp0:.2f} -> {sp1:.2f} cells ({'pass' if r['pass_spread'] else 'FAIL'})   "
              f"Sod L1 {l10:.4e} -> {l11:.4e} (x{l11 / l10:.3f}, {'pass' if r['pass_l1'] else 'FAIL'})")

    tubes = {}
    for name, (text, left, right, t_end) in ss.tubes().items():
        case = study.write_case(ss.STUDY, name, text)
        for scheme, sets in ss.SCHEMES.items():
            d = study.run_cached(STUDY, f"{name}_{scheme}_blend", case, *sets, f"shock_blend={args.blend}", mpi=args.mpi,
                                 force=args.force)
            res = {"finished": True, **ss.judge_tube(d, left, right, t_end)}
            res["passed"] = ss.passed(res)
            tubes[f"{name}/{scheme}"] = res
            ok &= res["passed"]
    failing = [k for k, v in tubes.items() if not v["passed"]]
    out["tubes"], out["tubes_failing"] = tubes, failing
    print(f"strong-shock tubes with the cure on: {len(tubes) - len(failing)} of {len(tubes)} pass"
          + (f"; failing: {', '.join(failing)}" if failing else ""))
    out["passed"] = bool(ok)
    fenice.write_summary(STUDY, out)
    print("ALL BANDS PASSED" if ok else "SOME BANDS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
