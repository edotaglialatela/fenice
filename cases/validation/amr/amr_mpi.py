"""AMR on threads and MPI ranks: same state as the serial run, time of the time loop and of the remesh.

    python cases/validation/amr/amr_mpi.py             # results/validation/amr_mpi/
    python cases/validation/amr/amr_mpi.py --docs      # also docs/img/amr_mpi.png
    python cases/validation/amr/amr_mpi.py --check     # bitwise equality with serial on 1 to 7 ranks, four cases
    python cases/validation/amr/amr_mpi.py --mesh 1440x720 --threads 1 2 4 8 16 24 48 --ranks 2 4 8 16 24 48
                                                       # another base mesh / thread and rank lists (results/validation/amr_mpi_1440x720/)

Runs "IBM Airfoil AMR" on 720 x 360 (two levels, the airfoil kept at the finest level and the wake
refined, so the leaves change during the run) for 50 steps: serial on one thread, OpenMP on 2 to 16
threads and MPI on 2 to 16 ranks (one thread per rank). Checks that every final state is bit for bit
the serial one, and prints the wall time of the time loop and the part of it that is remesh (the
gather of the state, the indicator, the adaptation and the rebuild, which every rank repeats).
Every rank stores the whole mesh. Run it on an idle machine: the times are the point. Figure: amr_mpi.png.

--check runs four cases that adapt during the run (dynamic Sod, the airfoil with the immersed body and the
non-reflecting outlet, a periodic viscous manufactured solution, and one with a time-dependent source)
serially and on 1 to 7 MPI ranks and compares the final states bit for bit; CI runs it on every platform.
"""
import json
import re

import numpy as np

import fenice
from fenice import study

STUDY, CASE = "amr_mpi", "IBM Airfoil AMR"
STEPS, MESH = 50, (720, 360)
BASE = (f"nx={MESH[0]}", f"ny={MESH[1]}", f"max_steps={STEPS}")
THREADS, RANKS = [1, 2, 4, 8, 16], [2, 4, 8, 16]


def measure(args, tag, mpi=0, threads=None):
    """Timing run, cached as JSON. Returns {"wall": time loop [s], "remesh": its remesh part [s], "leaves": at the end}."""
    d = study.study_dir(STUDY) / tag
    info = d / "info.json"
    if args.force or not info.exists():
        d.mkdir(parents=True, exist_ok=True)
        best = None
        for k in range(args.repeat):   # the minimum over the repeats (the state is the same every time)
            print(f"[RUN] {STUDY}/{tag}" + (f" ({k + 1} of {args.repeat})" if args.repeat > 1 else ""), flush=True)
            r = fenice.run(CASE, *BASE, "save_interval=1000000000", out_root=d, mpi=mpi, threads=threads, dump=d / "final.npz")
            wall = float(re.search(r"time loop ([0-9.]+) s", r.stdout).group(1))
            m = re.search(r"\[AMR\] (\d+) leaves at the end .*?remesh ([0-9.]+) s", r.stdout)
            parts = re.search(r"remesh parts: (.*)", r.stdout)
            run = {"wall": wall, "remesh": float(m.group(2)), "leaves": int(m.group(1)), "parts": parts.group(1) if parts else ""}
            best = run if best is None or run["wall"] < best["wall"] else best
        info.write_text(json.dumps(best))
    return d, json.loads(info.read_text())


CHECK_CASES = {   # name: (case, overrides)
    "sod": ("Sod Shock Tube", ["nx=50", "ny=4", "Ly=0.08", "amr_max_level=2", "amr_refine_every=4"]),
    "airfoil": ("IBM Airfoil AMR", ["t_end=0.6"]),
    # default indicator: all three levels stay in use and the leaves change during the run (with amr_criterion=0.01 the whole
    # domain was refined at the start and the 2 steps of t_end=0.05 never remeshed)
    "mms": ("Manufactured Solution", ["use_amr=true", "amr_max_level=2", "amr_refine_every=4", "max_steps=16"]),
    "mms_unsteady": ("Manufactured Solution Unsteady Euler",
                     ["use_amr=true", "amr_max_level=2", "amr_refine_every=4", "max_steps=8"]),
}


def check(ranks):
    """Serial against 1..N ranks on the four cases: the dumped states must be identical bit for bit."""
    ok = True
    for name, (case, sets) in CHECK_CASES.items():
        sets = ["use_amr=true", *sets]
        out = study.study_dir(STUDY) / "check" / name
        out.mkdir(parents=True, exist_ok=True)
        fenice.run(case, *sets, "save_interval=1000000000", out_root=out / "serial", dump=out / "serial.npz")
        ref = fenice.read_dump(out / "serial.npz")
        line = f"{name:13s} serial ({len(ref['U'])} leaves)"
        for n in range(1, ranks + 1):
            fenice.run(case, *sets, "save_interval=1000000000", out_root=out / f"np{n}", mpi=n, dump=out / f"np{n}.npz")
            got = fenice.read_dump(out / f"np{n}.npz")
            same = np.array_equal(got["U"], ref["U"]) and np.array_equal(got["leaf_meta"], ref["leaf_meta"])
            ok = ok and same
            line += f" | np{n} {'==' if same else 'DIFFERS'}"
        print(line)
    print("ALL MATCH (bitwise)" if ok else "MISMATCH")
    return 0 if ok else 1


def main():
    global STUDY, STEPS, BASE, THREADS, RANKS
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="bitwise equality with serial on 1..7 ranks")
    ap.add_argument("--mesh", default=f"{MESH[0]}x{MESH[1]}", help="base mesh NXxNY (default %(default)s)")
    ap.add_argument("--steps", type=int, default=STEPS, help="time steps (default %(default)s)")
    ap.add_argument("--threads", type=int, nargs="+", default=THREADS, help="OpenMP thread counts (1 first)")
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS, help="MPI rank counts (none: --ranks 0)")
    ap.add_argument("--repeat", type=int, default=1, help="runs per point, the fastest is kept (default %(default)s)")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check(7))
    nx, ny = (int(v) for v in args.mesh.split("x"))
    default = (nx, ny) == MESH and args.steps == STEPS
    if not default:   # results of another setup in their own folder and figure, never in the docs
        STUDY = f"amr_mpi_{nx}x{ny}" + ("" if args.steps == 50 else f"_{args.steps}steps")
        args.docs = False
    STEPS, BASE = args.steps, (f"nx={nx}", f"ny={ny}", f"max_steps={args.steps}")
    THREADS, RANKS = sorted(set([1] + args.threads)), sorted(r for r in set(args.ranks) if r > 1)
    plt = study.use_style()
    d0, serial = measure(args, "serial", threads=1)
    ref = fenice.read_dump(d0 / "final.npz")
    omp, mpi = {}, {}
    for n in THREADS[1:]:
        omp[n] = measure(args, f"omp{n}", threads=n)
    for n in RANKS:
        mpi[n] = measure(args, f"mpi{n}", mpi=n)
    omp[1] = (d0, serial)

    print(f"\n{'run':>10} {'wall [s]':>9} {'speed-up':>9} {'remesh [s]':>11}  state vs serial   ({serial['leaves']} leaves)")
    rows = [("serial", serial, d0)] + [(f"omp {n}", omp[n][1], omp[n][0]) for n in THREADS[1:]] \
        + [(f"mpi {n}", mpi[n][1], mpi[n][0]) for n in RANKS]
    for name, info, d in rows:
        got = fenice.read_dump(d / "final.npz")
        same = np.array_equal(got["U"], ref["U"]) and np.array_equal(got["leaf_meta"], ref["leaf_meta"])
        print(f"{name:>10} {info['wall']:9.2f} {serial['wall'] / info['wall']:9.2f} {info['remesh']:11.2f}  "
              f"{'identical' if same else 'DIFFERENT'}")
        if not same:
            raise SystemExit(f"{name}: the final state differs from the serial run")
    (study.study_dir(STUDY) / "summary.json").write_text(json.dumps(
        {"mesh": BASE[:2], "steps": STEPS, "leaves": serial["leaves"],
         "repeat": args.repeat, "runs": {name: {k: info.get(k) for k in ("wall", "remesh", "parts")} for name, info, _ in rows}},
        indent=1))

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    n_all = np.array(sorted(set(THREADS) | set(RANKS)))
    ax[0].loglog(n_all, serial["wall"] / n_all, "k:", label="ideal")
    ax[0].loglog(THREADS, [omp[n][1]["wall"] for n in THREADS], "o-", label="OpenMP threads")
    ax[0].loglog(RANKS, [mpi[n][1]["wall"] for n in RANKS], "s-", label="MPI ranks")
    ax[0].set(xlabel="threads or ranks", ylabel="wall time of 50 steps [s]", title="Time")
    ax[0].set_xticks(n_all, [str(n) for n in n_all])
    ax[0].legend()
    labels = ["serial"] + [f"omp {n}" for n in THREADS[1:]] + [f"mpi {n}" for n in RANKS]
    infos = [serial] + [omp[n][1] for n in THREADS[1:]] + [mpi[n][1] for n in RANKS]
    x = np.arange(len(labels))
    stepping = np.array([i["wall"] - i["remesh"] for i in infos])
    remesh = np.array([i["remesh"] for i in infos])
    ax[1].bar(x, stepping, label="time stepping", color="C0")
    ax[1].bar(x, remesh, bottom=stepping, label="remesh", color="C3")
    ax[1].set_xticks(x, labels, rotation=60, ha="right", fontsize=8)
    ax[1].set(ylabel="wall time [s]", title="Where the time goes")
    ax[1].legend()
    fig.suptitle(f"AMR airfoil, {BASE[0][3:]}×{BASE[1][3:]} base mesh, 2 levels, {STEPS} steps ({serial['leaves']} leaves at the end)",
                 y=1.02)
    study.save(fig, STUDY, f"{STUDY}.png", args.docs)


if __name__ == "__main__":
    main()
