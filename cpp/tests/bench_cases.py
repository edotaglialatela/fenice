#!/usr/bin/env python3
"""Wall time of the time loop of four uniform cases on 1 thread, 16 threads and 4 MPI ranks, and the saving of the AVX2 kernels
on the AMR airfoil (docs/source/guide/performance.md: Benchmark). Minimum of N runs, nothing written but the summary.

    python cpp/tests/bench_cases.py              # N = 5
    python cpp/tests/bench_cases.py --runs 3 --no-airfoil

Run it on an idle machine: the times are the point. The thread counts are those of the 16-thread workstation; change THREADS
for another machine. FENICE_EXE selects the solver, as everywhere.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import fenice  # noqa: E402

THREADS = 16
CASES = {
    "IBM cylinder 400x200, 20 steps": ("IBM Flow over Cylinder", ["nx=400", "ny=200", "max_steps=20"]),
    "Supersonic IBM 400x400, 20 steps": ("IBM Supersonic Flow over Cylinder", ["nx=400", "ny=400", "max_steps=20"]),
    "Shu-Osher 4000x2, 200 steps": ("Shu-Osher Test", ["nx=4000", "ny=2", "max_steps=200"]),
    "Lid cavity SSP-RK3 128x128, 20 steps": ("Lid Driven Cavity", ["nx=128", "ny=128", "time_scheme=SSP Runge-Kutta 3", "max_steps=20"]),
}
AIRFOIL = ["nx=720", "ny=360", "max_steps=50"]


def best(case, sets, runs, root, **kw):
    walls = []
    for _ in range(runs):
        r = fenice.run(case, *sets, "save_interval=1000000000", "monitor_interval=0", out_root=root, **kw)
        walls.append(fenice.parse_summary(r.stdout)["wall"])
    return min(walls)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--no-airfoil", action="store_true")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        print(f"| case | 1 thread | {THREADS} threads | MPI 4x1 | {THREADS} threads vs 1 |\n|---|---:|---:|---:|---:|")
        for name, (case, sets) in CASES.items():
            t1 = best(case, sets, a.runs, root, threads=1)
            tn = best(case, sets, a.runs, root, threads=THREADS)
            tm = best(case, sets + ["use_mpi=true"], a.runs, root, mpi=4)
            print(f"| {name} | {t1:.3f} s | {tn:.3f} s | {tm:.3f} s | {t1 / tn:.1f}x |", flush=True)
        if not a.no_airfoil:
            on = best("IBM Airfoil AMR", AIRFOIL, a.runs, root, threads=1)
            old = os.environ.get("FENICE_NO_SIMD")
            os.environ["FENICE_NO_SIMD"] = "1"
            off = best("IBM Airfoil AMR", AIRFOIL, a.runs, root, threads=1)
            if old is None:
                del os.environ["FENICE_NO_SIMD"]
            else:
                os.environ["FENICE_NO_SIMD"] = old
            print(f"\nAMR airfoil 720x360, 50 steps, 1 thread: {on:.2f} s with the vector kernels, {off:.2f} s without "
                  f"(FENICE_NO_SIMD=1): they save {off - on:.2f} s")


if __name__ == "__main__":
    main()
