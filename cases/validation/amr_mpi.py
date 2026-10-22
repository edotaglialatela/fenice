"""AMR on MPI ranks: same state as the serial run, time and memory per rank.

    python cases/validation/amr_mpi.py             # results/validation/amr_mpi/
    python cases/validation/amr_mpi.py --docs      # also docs/img/amr_mpi.png

Runs "IBM Airfoil AMR" on 720 x 360 with 24 x 24 patches for 50 steps (the airfoil is kept at the
finest level and the wake is refined, so the patches change during the run): serial on one thread,
OpenMP on 2 to 16 threads and MPI on 2 to 16 ranks (one thread per rank). Checks that every
final state is bit for bit the serial one, and prints the wall time of the time loop and the
largest number of cells a rank stores (owned patches plus the halo copies of their neighbours).
Run it on an idle machine: the times are the point. Figure: amr_mpi.png.
"""
import json
import re

import numpy as np

import fenice
from fenice import study

STUDY, CASE = "amr_mpi", "IBM Airfoil AMR"
PATCH, STEPS = 24, 50
BASE = ("nx=720", "ny=360", f"amr_patch_nx={PATCH}", f"amr_patch_ny={PATCH}", f"max_steps={STEPS}")
THREADS, RANKS = [1, 2, 4, 8, 16], [2, 4, 8, 16]


def measure(args, tag, mpi=0, threads=None):
    """Timing run (no output) and memory run (one status line at the last step), cached as JSON.
    Returns {"wall": s, "cells_max": largest cells held by a rank, "cells_all": cells of all patches}."""
    d = study.study_dir(STUDY) / tag
    info = d / "info.json"
    if args.force or not info.exists():
        d.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {STUDY}/{tag}", flush=True)
        r = fenice.run(CASE, *BASE, "save_interval=1000000000", out_root=d, mpi=mpi, threads=threads, dump=d / "final.npz")
        wall = float(re.search(r"time loop ([0-9.]+) s", r.stdout).group(1))
        r = fenice.run(CASE, *BASE, f"save_interval={STEPS}", out_root=d / "mem", mpi=mpi, threads=threads)
        patches = int(re.findall(r"patches = (\d+)", r.stdout)[-1])
        held = re.findall(r"cells/rank max (\d+)", r.stdout)   # printed on several ranks only
        cells_all = patches * PATCH * PATCH
        info.write_text(json.dumps({"wall": wall, "cells_all": cells_all,
                                    "cells_max": int(held[-1]) if held else cells_all}))
    return d, json.loads(info.read_text())


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    d0, serial = measure(args, "serial", threads=1)
    ref = fenice.read_dump(d0 / "final.npz")
    omp, mpi = {}, {}
    for n in THREADS[1:]:
        omp[n] = measure(args, f"omp{n}", threads=n)
    for n in RANKS:
        mpi[n] = measure(args, f"mpi{n}", mpi=n)
    omp[1] = (d0, serial)

    print(f"\n{'run':>10} {'wall [s]':>9} {'speed-up':>9} {'max cells/rank':>15} {'of all':>7}  state vs serial")
    rows = [("serial", serial, d0)] + [(f"omp {n}", omp[n][1], omp[n][0]) for n in THREADS[1:]] \
        + [(f"mpi {n}", mpi[n][1], mpi[n][0]) for n in RANKS]
    for name, info, d in rows:
        got = fenice.read_dump(d / "final.npz")
        same = np.array_equal(got["U"], ref["U"]) and np.array_equal(got["leaf_meta"], ref["leaf_meta"])
        print(f"{name:>10} {info['wall']:9.2f} {serial['wall'] / info['wall']:9.2f} {info['cells_max']:15d} "
              f"{info['cells_max'] / info['cells_all']:7.0%}  {'identical' if same else 'DIFFERENT'}")
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
    frac = [mpi[n][1]["cells_max"] / mpi[n][1]["cells_all"] for n in RANKS]
    ax[1].loglog(n_all, 1.0 / n_all, "k:", label="ideal (1/N)")
    ax[1].loglog([1] + RANKS, [1.0] + frac, "s-", color="C1", label="MPI: busiest rank")
    ax[1].set(xlabel="ranks", ylabel="cells stored by the busiest rank / all cells", title="Memory per rank")
    ax[1].set_xticks(n_all, [str(n) for n in n_all])
    ax[1].legend()
    fig.suptitle(f"AMR airfoil, 720×360 base mesh, {PATCH}×{PATCH} patches, {STEPS} steps "
                 f"({serial['cells_all']} cells in AMR patches)", y=1.02)
    study.save(fig, STUDY, "amr_mpi.png", args.docs)


if __name__ == "__main__":
    main()
