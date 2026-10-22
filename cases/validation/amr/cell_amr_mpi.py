"""Cell-based AMR under MPI: bitwise equality with the serial run, and the scaling table.

Runs ``cell_amr_demo`` serially and under ``mpiexec -n R`` (R = 1..7, ``--poison``: a rank's non-held leaves are NaN, so a
missing halo cell cannot hide) on four cases that adapt during the run (dynamic Sod, the NACA 0012 airfoil with the immersed
body, the non-reflecting outlet and viscosity, a periodic manufactured solution with unsteady sources) and compares the final
states byte for byte. Not part of the CI (the cell engine is not behind the CLI yet): run it by hand.

    python cases/validation/amr/cell_amr_mpi.py [--cases sod,airfoil,mms,mms_unsteady] [--ranks 1,2,3,4,5,6,7]
    python cases/validation/amr/cell_amr_mpi.py --scaling          # the amr_mpi.py study (720 x 360, 50 steps) with the cell engine
"""
import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXE = ROOT / "build" / "release" / ("cell_amr_demo.exe" if sys.platform == "win32" else "cell_amr_demo")
OUT = ROOT / "results" / "cell_amr" / "mpi"

CASES = {
    "sod": ("Sod Shock Tube", ["nx=50", "ny=4", "Ly=0.08", "--levels", "2", "--every", "4", "--tend", "0.2"]),
    "airfoil": ("IBM Airfoil AMR", ["--vars", "rho,u", "--levels", "2", "--every", "5", "--tend", "0.6", "--monitor", "0"]),
    "mms": ("Manufactured Solution", ["--levels", "2", "--every", "4", "--tend", "0.05", "--criterion", "0.01"]),
    "mms_unsteady": ("Manufactured Solution Unsteady Euler",
                     ["--levels", "2", "--every", "4", "--tend", "0.05", "--criterion", "0.01"]),
}


def run(case, ranks, extra=(), tag=None):
    name, args = CASES[case]
    out = OUT / case / (tag or f"np{ranks}")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(EXE), name, *args, *extra, "--out", str(out)]
    if ranks > 0:
        cmd = ["mpiexec", "-n", str(ranks), *cmd]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stdout[-2000:], res.stderr[-2000:])
        raise SystemExit(f"{' '.join(cmd)} failed")
    digest = hashlib.sha256((out / "final.bin").read_bytes()).hexdigest()
    return digest, res.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--ranks", default="1,2,3,4,5,6,7")
    ap.add_argument("--scaling", action="store_true")
    a = ap.parse_args()
    if a.scaling:
        return scaling()
    ok = True
    for case in a.cases.split(","):
        ref, out = run(case, 0, tag="serial")
        leaves = re.search(r"final leaves (\d+)", out).group(1)
        line = f"{case:13s} serial {ref[:12]} ({leaves} leaves)"
        for r in [int(x) for x in a.ranks.split(",")]:
            d, _ = run(case, r, ["--poison"])
            same = d == ref
            ok = ok and same
            line += f" | np{r} {'==' if same else 'DIFFERS'}"
        print(line)
    print("ALL MATCH (bitwise)" if ok else "MISMATCH")
    return 0 if ok else 1


def scaling():
    """The study of amr_mpi.py (airfoil, 720 x 360 base, 2 levels, 50 steps, remesh every 5) with the cell engine: serial on one
    thread, OpenMP on 2-16 threads, MPI on 2-16 ranks (one thread each); wall time of the time loop, the part of it that is
    remesh, the largest number of leaves a rank holds, and whether the final state is the serial one."""
    import os
    name, _ = CASES["airfoil"]
    base = ["nx=720", "ny=360", "--vars", "rho,u", "--levels", "2", "--every", "5", "--steps", "50", "--monitor", "0"]
    print(f"{'run':>8} {'wall [s]':>9} {'speed-up':>9} {'remesh [s]':>11} {'valid':>9} {'work max/mean':>14}  state vs serial")
    print("         (valid: leaves with a valid state on the busiest rank; every rank stores the whole mesh and state)")
    ref_digest, serial_wall = None, None
    for label, ranks, threads in [("serial", 0, 1)] + [(f"omp {n}", 0, n) for n in (2, 4, 8, 16)] + [(f"mpi {n}", n, 1) for n in (2, 4, 8, 16)]:
        out = OUT / "scaling" / label.replace(" ", "")
        out.mkdir(parents=True, exist_ok=True)
        cmd = [str(EXE), name, *base, "--out", str(out)]
        if ranks > 0:
            cmd = ["mpiexec", "-n", str(ranks), *cmd]
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env).stdout
        m = re.search(r"wall ([\d.]+) s \(time stepping ([\d.]+) s, remesh .*? ([\d.]+) s = ([\d.]+) %\)", res)
        wall, remesh = float(m.group(1)), float(m.group(3))
        held = [int(x) for x in re.findall(r"held (\d+)", res)]
        bal = re.search(r"work max / mean = ([\d.]+)", res)
        digest = hashlib.sha256((out / "final.bin").read_bytes()).hexdigest()
        ref_digest = ref_digest or digest
        serial_wall = serial_wall or wall
        print(f"{label:>8} {wall:9.2f} {serial_wall / wall:9.2f} {remesh:11.2f} {max(held) if held else '-':>9} "
              f"{bal.group(1) if bal else '-':>14}  {'identical' if digest == ref_digest else 'DIFFERS'}")
        parts = re.search(r"remesh parts \(rank 0\): gather ([\d.]+) s, indicator ([\d.]+) s, adapt ([\d.]+) s, engine build ([\d.]+) s, "
                          r"speed bound ([\d.]+) s, partition ([\d.]+) s", res)
        if parts:
            names = ("gather", "indicator", "adapt", "build", "speed bound", "partition")
            print("         remesh: " + ", ".join(f"{n} {float(v):.2f}" for n, v in zip(names, parts.groups())))


if __name__ == "__main__":
    sys.exit(main())
