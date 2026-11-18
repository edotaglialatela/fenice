"""Cost of `flux_quadrature = fourth` against `midpoint`: wall time per step of the isentropic vortex (2-D) and of the Taylor-Green box (3-D), WENO3-Z and WENO5-Z, SSP-RK3.

    python cases/validation/benchmarks/flux_quadrature_cost.py                      # results/validation/flux_quadrature_cost/
    python cases/validation/benchmarks/flux_quadrature_cost.py --sizes 1024 --cubes 128 --threads 1 16 48

Each configuration runs `--steps` (2-D, default 40) or `--steps3d` (3-D, default 5) steps at a fixed small CFL and keeps the fastest of `--repeats` runs (default 3);
the time is the time loop of the solver report, so setup is excluded. The table gives ms per step and the ratio fourth / midpoint.
"""
import argparse

import fenice
from fenice import study

STUDY = "flux_quadrature_cost"
SCHEMES = {"WENO3-Z": "reconstruction=weno3", "WENO5-Z": "reconstruction=weno5"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024], help="cells per side of the 2-D meshes")
    ap.add_argument("--cubes", type=int, nargs="*", default=[128], help="cells per side of the 3-D meshes (none: --cubes alone)")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 16], help="OpenMP threads")
    ap.add_argument("--steps", type=int, default=40, help="steps of a 2-D run")
    ap.add_argument("--steps3d", type=int, default=5, help="steps of a 3-D run")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    d = study.study_dir(STUDY)
    rows = ["| scheme | cells | threads | midpoint [ms/step] | fourth [ms/step] | ratio |", "|---|---:|---:|---:|---:|---:|"]
    meshes = [("isentropic_vortex", f"{n}^2", [f"nx={n}", f"ny={n}"], args.steps) for n in args.sizes]
    meshes += [("taylor_green_vortex_3d", f"{n}^3", [f"nx={n}", f"ny={n}", f"nz={n}"], args.steps3d) for n in args.cubes]
    for label, rec in SCHEMES.items():
        for case, name, size, steps in meshes:
            for t in args.threads:
                ms = {}
                for mode in ("midpoint", "fourth"):
                    best = min(fenice.parse_summary(fenice.run(
                        case, *size, rec, "time_scheme=SSP Runge-Kutta 3", f"flux_quadrature={mode}", "cfl=0.2",
                        f"max_steps={steps}", "save_interval=1000000000", out_root=d / f"{label}_{name}_{t}_{mode}",
                        threads=t).stdout)["wall"] for _ in range(args.repeats))
                    ms[mode] = 1000.0 * best / steps
                rows.append(f"| {label} | {name} | {t} | {ms['midpoint']:.1f} | {ms['fourth']:.1f} | {ms['fourth'] / ms['midpoint']:.2f} |")
                print(rows[-1], flush=True)
    (d / "flux_quadrature_cost.md").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwritten: {d / 'flux_quadrature_cost.md'}")


if __name__ == "__main__":
    main()
