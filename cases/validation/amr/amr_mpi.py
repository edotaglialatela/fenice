"""AMR on threads and MPI ranks: same state as the serial run, time of the time loop and of the remesh.

    python cases/validation/amr/amr_mpi.py             # results/validation/amr_mpi/
    python cases/validation/amr/amr_mpi.py --docs      # also docs/img/amr_mpi.png
    python cases/validation/amr/amr_mpi.py --check     # bitwise equality with serial on 1 to 7 ranks, four cases

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
STEPS = 50
BASE = ("nx=720", "ny=360", f"max_steps={STEPS}")
THREADS, RANKS = [1, 2, 4, 8, 16], [2, 4, 8, 16]


def measure(args, tag, mpi=0, threads=None):
    """Timing run, cached as JSON. Returns {"wall": time loop [s], "remesh": its remesh part [s], "leaves": at the end}."""
    d = study.study_dir(STUDY) / tag
    info = d / "info.json"
    if args.force or not info.exists():
        d.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {STUDY}/{tag}", flush=True)
        r = fenice.run(CASE, *BASE, "save_interval=1000000000", out_root=d, mpi=mpi, threads=threads, dump=d / "final.npz")
        wall = float(re.search(r"time loop ([0-9.]+) s", r.stdout).group(1))
        m = re.search(r"\[AMR\] (\d+) leaves at the end .*?remesh ([0-9.]+) s", r.stdout)
        info.write_text(json.dumps({"wall": wall, "remesh": float(m.group(2)), "leaves": int(m.group(1))}))
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
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="bitwise equality with serial on 1..7 ranks")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(check(7))
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

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    n_all = np.array([1, 2, 4, 8, 16])
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
    fig.suptitle(f"AMR airfoil, 720×360 base mesh, 2 levels, {STEPS} steps ({serial['leaves']} leaves at the end)", y=1.02)
    study.save(fig, STUDY, "amr_mpi.png", args.docs)


if __name__ == "__main__":
    main()
