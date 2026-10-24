"""Golden dumps: the final states of adaptive runs, to check that a change does not move a single bit.

    python cases/validation/amr/golden.py dump OLD_DIR --exe path/to/old/fenice      # before the change (or the main binary)
    python cases/validation/amr/golden.py dump NEW_DIR                               # after it ($FENICE_EXE / build/release)
    python cases/validation/amr/golden.py compare OLD_DIR NEW_DIR                    # every state bit for bit, exit code 0 if equal
    options of dump: --quick (1 / 3 / 7 threads, the large airfoil on 3 only), --only NAME ..., --env KEY=VALUE ...

Ten adaptive runs (the four cases of amr_mpi.py --check, MMS 40 steps, cavity 200 steps, Sod remeshed every 2 steps, the two AMR
ghost-cell cases of ibm_ghost.py --check, the 720 x 360 airfoil for 50 steps) on 1 / 2 / 3 / 4 / 7 threads (the large airfoil on
1 / 3 / 4 / 7), each with the monitor every 5 steps (the monitor's loads take a residual between steps: it must not leak into the
state). The compare step checks every file against the same run in OLD_DIR and against NEW_DIR's 1-thread run (results do not
depend on the thread count). Use --env FENICE_FULL_REBUILD=1 / FENICE_NO_SIMD=1 for the A/B variants. Takes ~10 min (--quick ~4).
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import fenice

CASES = {   # name: (case, overrides)
    "sod": ("Sod Shock Tube", ["nx=50", "ny=4", "Ly=0.08", "use_amr=true", "amr_max_level=2", "amr_refine_every=4"]),
    "airfoil_short": ("IBM Airfoil AMR", ["t_end=0.6"]),
    "mms": ("Manufactured Solution", ["use_amr=true", "amr_max_level=2", "amr_refine_every=4", "max_steps=16"]),
    "mms_unsteady": ("Manufactured Solution Unsteady Euler", ["use_amr=true", "amr_max_level=2", "amr_refine_every=4", "max_steps=8"]),
    "mms40": ("Manufactured Solution", ["use_amr=true", "amr_max_level=2", "amr_refine_every=4", "max_steps=40"]),
    "cavity": ("Lid Driven Cavity", ["use_amr=true", "amr_max_level=2", "amr_refine_every=4", "max_steps=200"]),
    "sod_every2": ("Sod Shock Tube", ["Lx=2.0", "nx=128", "ny=16", "Ly=0.25", "use_amr=true", "amr_max_level=3",
                                      "amr_refine_every=2", "t_end=0.2"]),
    "ghost_airfoil": ("IBM Airfoil AMR", ["ibm_method=ghost", "max_steps=12"]),
    "ghost_cylinder": ("IBM Flow over Cylinder", ["ibm_method=ghost", "use_amr=true", "amr_max_level=2", "amr_refine_every=4",
                                                  "nx=100", "ny=50", "max_steps=16"]),
    "airfoil_720": ("IBM Airfoil AMR", ["nx=720", "ny=360", "max_steps=50"]),
}
THREADS, BIG_THREADS = [1, 2, 3, 4, 7], [1, 3, 4, 7]
QUICK, BIG_QUICK = [1, 3, 7], [3]


def digest(path):
    z = np.load(path)
    return hashlib.sha1(b"".join(np.ascontiguousarray(z[k]).tobytes() for k in sorted(z.files))).hexdigest()


def dump(args):
    out = Path(args.dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    exe = args.exe or fenice.exe()
    env = dict(os.environ, **dict(e.split("=", 1) for e in args.env))
    for name, (case, sets) in CASES.items():
        if args.only and name not in args.only:
            continue
        big = name == "airfoil_720"
        for t in (BIG_QUICK if big else QUICK) if args.quick else (BIG_THREADS if big else THREADS):
            d = out / f"{name}_t{t}"
            cmd = [str(exe), "--case", case, "--no-logo", "--threads", str(t), "--set", f"output_root={d}",
                   "--set", f"dump_state={d}.npz", "--set", "save_interval=1000000000", "--set", "use_mpi=false",
                   "--set", "monitor_interval=5"]
            for s in sets:
                cmd += ["--set", s]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if r.returncode:
                sys.exit(f"{name} on {t} threads failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            print(f"{name:15s} {t} threads", flush=True)


def compare(args):
    old, new = Path(args.old), Path(args.new)
    ok = True
    for f in sorted(new.glob("*.npz")):
        ref = old / f.name
        same = ref.exists() and digest(f) == digest(ref)
        one = new / (f.name.rsplit("_t", 1)[0] + "_t1.npz")
        threads = not one.exists() or digest(f) == digest(one)
        ok = ok and same and threads
        print(f"{f.stem:22s} {'==' if same else 'DIFFERS' if ref.exists() else 'missing in OLD'}"
              f"{'' if threads else '  (differs from the 1-thread run)'}")
    print("ALL BITWISE" if ok else "MISMATCH")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("dir")
    d.add_argument("--exe", help="solver executable (default: $FENICE_EXE or build/release)")
    d.add_argument("--quick", action="store_true")
    d.add_argument("--only", nargs="*", default=[])
    d.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE")
    c = sub.add_parser("compare")
    c.add_argument("old")
    c.add_argument("new")
    args = ap.parse_args()
    raise SystemExit(dump(args) if args.cmd == "dump" else compare(args))


if __name__ == "__main__":
    main()
