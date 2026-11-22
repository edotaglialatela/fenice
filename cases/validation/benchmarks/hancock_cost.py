"""Cost of the MUSCL-Hancock time scheme against MUSCL with SSP-RK2: wall time per step of the isentropic vortex (2-D) and of the Taylor-Green box (3-D).

    python cases/validation/benchmarks/hancock_cost.py                       # results/validation/hancock_cost/
    python cases/validation/benchmarks/hancock_cost.py --sizes 1024 --cubes 128 --threads 1 16 48

Each configuration runs `--steps` (2-D, default 40) or `--steps3d` (3-D, default 5) steps at the same CFL and keeps the fastest of `--repeats` runs (default 3);
the time is the time loop of the solver report, so setup is excluded. The table gives ms per step of SSP-RK2, of the fused MUSCL-Hancock step and of the
two-pass one (FENICE_NO_FUSED = 1), and the ratios rk2 / hancock (above 1: Hancock is faster). Pin the threads for the timings of a paper
(`start /affinity` on Windows, `taskset` on Linux): unpinned runs vary by tens of percent.
"""
import argparse
import os

import fenice
from fenice import study

STUDY = "hancock_cost"
SCHEMES = {"rk2": "time_scheme=SSP Runge-Kutta 2", "hancock": "time_scheme=MUSCL-Hancock"}


def best_ms(case, sets, out, threads, steps, repeats, fused):
    """Fastest of `repeats` runs, ms per step."""
    old = os.environ.pop("FENICE_NO_FUSED", None)
    if not fused:
        os.environ["FENICE_NO_FUSED"] = "1"
    try:
        wall = min(fenice.parse_summary(fenice.run(case, *sets, "reconstruction=muscl", "cfl=0.4", f"max_steps={steps}", "save_interval=1000000000",
                                                   out_root=out, threads=threads).stdout)["wall"] for _ in range(repeats))
    finally:
        os.environ.pop("FENICE_NO_FUSED", None)
        if old is not None:
            os.environ["FENICE_NO_FUSED"] = old
    return 1000.0 * wall / steps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024], help="cells per side of the 2-D meshes")
    ap.add_argument("--cubes", type=int, nargs="*", default=[96, 128], help="cells per side of the 3-D meshes")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 16], help="OpenMP threads")
    ap.add_argument("--steps", type=int, default=40, help="steps of a 2-D run")
    ap.add_argument("--steps3d", type=int, default=5, help="steps of a 3-D run")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    d = study.study_dir(STUDY)
    rows = ["| mesh | threads | SSP-RK2 [ms/step] | Hancock [ms/step] | Hancock two-pass [ms/step] | RK2 / Hancock | RK2 / two-pass |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    meshes = [("isentropic_vortex", f"{n}^2", [f"nx={n}", f"ny={n}"], args.steps) for n in args.sizes]
    meshes += [("taylor_green_vortex_3d", f"{n}^3", [f"nx={n}", f"ny={n}", f"nz={n}"], args.steps3d) for n in args.cubes]
    for case, name, size, steps in meshes:
        for t in args.threads:
            def go(tag, scheme, fused=True):
                return best_ms(case, [*size, SCHEMES[scheme]], d / f"{name}_{t}_{tag}", t, steps, args.repeats, fused)
            rk2, han, two = go("rk2", "rk2"), go("hancock", "hancock"), go("twopass", "hancock", fused=False)
            rows.append(f"| {name} | {t} | {rk2:.1f} | {han:.1f} | {two:.1f} | {rk2 / han:.2f} | {rk2 / two:.2f} |")
            print(rows[-1], flush=True)
    (d / "hancock_cost.md").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {d / 'hancock_cost.md'}")


if __name__ == "__main__":
    main()
