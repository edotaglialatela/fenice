"""Cost of `flux_quadrature = fourth` against `midpoint`: wall time per step of the isentropic vortex, WENO3-Z and WENO5-Z, SSP-RK3.

    python cases/validation/benchmarks/flux_quadrature_cost.py                      # results/validation/flux_quadrature_cost/
    python cases/validation/benchmarks/flux_quadrature_cost.py --sizes 1024 --threads 1 16 48

Each configuration runs `--steps` steps (default 40) at a fixed small CFL and keeps the fastest of `--repeats` runs (default 3);
the time is the time loop of the solver report, so setup is excluded. The table gives ms per step and the ratio fourth / midpoint.
"""
import argparse

import fenice
from fenice import study

STUDY, CASE = "flux_quadrature_cost", "isentropic_vortex"
SCHEMES = {"WENO3-Z": "reconstruction=weno3", "WENO5-Z": "reconstruction=weno5"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024], help="cells per side")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 16], help="OpenMP threads")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    d = study.study_dir(STUDY)
    rows = ["| scheme | cells | threads | midpoint [ms/step] | fourth [ms/step] | ratio |", "|---|---:|---:|---:|---:|---:|"]
    for label, rec in SCHEMES.items():
        for n in args.sizes:
            for t in args.threads:
                ms = {}
                for mode in ("midpoint", "fourth"):
                    best = min(fenice.parse_summary(fenice.run(
                        CASE, f"nx={n}", f"ny={n}", rec, "time_scheme=SSP Runge-Kutta 3", f"flux_quadrature={mode}", "cfl=0.2",
                        f"max_steps={args.steps}", "save_interval=1000000000", out_root=d / f"{label}_{n}_{t}_{mode}",
                        threads=t).stdout)["wall"] for _ in range(args.repeats))
                    ms[mode] = 1000.0 * best / args.steps
                rows.append(f"| {label} | {n}^2 | {t} | {ms['midpoint']:.1f} | {ms['fourth']:.1f} | {ms['fourth'] / ms['midpoint']:.2f} |")
                print(rows[-1], flush=True)
    (d / "flux_quadrature_cost.md").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {d / 'flux_quadrature_cost.md'}")


if __name__ == "__main__":
    main()
