"""Memory and time of the replicated AMR on MPI ranks: the baseline of docs/design/amr-distributed.md (phase 0).

    python cases/validation/amr/amr_memory.py                    # results/validation/amr_memory/
    python cases/validation/amr/amr_memory.py --ranks 1 2 4 8    # a subset of the rank counts (1 = serial, one thread)
    python cases/validation/amr/amr_memory.py --only airfoil_720 --steps 20
    python cases/validation/amr/amr_memory.py --reach            # also the halo reach of the meshes (serial, FENICE_HALO_REACH=1)
    python cases/validation/amr/amr_memory.py --docs             # also docs/img/amr_memory.png

Runs "IBM Airfoil AMR" on 720 x 360 and on 1440 x 720 (two levels, the airfoil kept at the finest level and the wake refined)
serially on one thread and on 2 to 16 MPI ranks (one thread per rank), and reads what the solver prints at the end of an AMR
run: the resident memory of every rank (after setup, after the initial adaptation, at the end, and the peak: min / max over
the ranks), the bytes of its tables (mesh, engine, the arrays sized by the uniform mesh of a level), the wall time of the time
loop and its remesh parts. Today every rank stores the whole mesh, so the memory per rank does not shrink with the ranks;
that is what the distributed storage has to change. Run it on an idle machine; the runs are cached (--force re-runs them).
Prints a table and writes summary.json (and the figure amr_memory.png).
"""
import json
import os
import re

import fenice
from fenice import study

STUDY, CASE = "amr_memory", "IBM Airfoil AMR"
CASES = {"airfoil_720": ("nx=720", "ny=360"), "airfoil_1440": ("nx=1440", "ny=720")}
RANKS = [1, 2, 4, 8, 16]

NUM = r"([0-9.]+)"
PATTERNS = {
    "wall": re.compile(rf"time loop {NUM} s"),
    "leaves": re.compile(r"\[AMR\] (\d+) leaves at the end"),
    "remesh": re.compile(rf"remesh {NUM} s of the"),
    "parts": re.compile(rf"remesh parts: gather {NUM} s, indicator {NUM} s, adapt {NUM} s, engine build {NUM} s, "
                        rf"wave-speed bound {NUM} s, partition {NUM} s"),
    "memory": re.compile(rf"resident memory per rank \[MB\].*?after setup {NUM} / {NUM}, after the initial adaptation "
                         rf"{NUM} / {NUM}, at the end {NUM} / {NUM}, peak {NUM} / {NUM}"),
    "tables": re.compile(rf"tables of rank 0 \[MB\]: mesh {NUM} .*?engine {NUM} .*?engine uniform-mesh {NUM} .*?body band "
                         rf"{NUM}, indicator {NUM}"),
}
PARTS = ["gather", "indicator", "adapt", "build", "speed", "partition"]
MEMORY = ["setup", "init", "end", "peak"]
TABLES = ["mesh", "engine", "uniform", "band", "xi"]


def parse(out):
    """The numbers of the end-of-run report of an AMR run."""
    got = {k: p.search(out) for k, p in PATTERNS.items()}
    missing = [k for k, m in got.items() if m is None]
    if missing:
        raise RuntimeError(f"not in the solver output: {', '.join(missing)}")
    v = {k: [float(x) for x in m.groups()] for k, m in got.items()}
    mem = v["memory"]
    return {"wall": v["wall"][0], "leaves": int(v["leaves"][0]), "remesh": v["remesh"][0],
            "parts": dict(zip(PARTS, v["parts"])),
            "memory_min": dict(zip(MEMORY, mem[0::2])), "memory_max": dict(zip(MEMORY, mem[1::2])),
            "tables": dict(zip(TABLES, v["tables"]))}


def measure(args, name, ranks):
    """One run (cached as info.json): `ranks` == 1 is the serial run on one thread."""
    tag = f"{name}_s{args.steps}_{'serial' if ranks == 1 else f'mpi{ranks}'}"
    d = study.study_dir(STUDY) / tag
    info = d / "info.json"
    if args.force or not info.exists():
        d.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {STUDY}/{tag}", flush=True)
        r = fenice.run(CASE, *CASES[name], f"max_steps={args.steps}", "save_interval=1000000000", out_root=d,
                       mpi=ranks if ranks > 1 else 0, threads=1 if ranks == 1 else None)
        info.write_text(json.dumps(parse(r.stdout), indent=1))
    return json.loads(info.read_text())


def reach(name, steps):
    """The halo reach the solver prints with FENICE_HALO_REACH=1 (serial run; the reach does not depend on the partition)."""
    d = study.study_dir(STUDY) / f"{name}_reach"
    os.environ["FENICE_HALO_REACH"] = "1"
    try:
        r = fenice.run(CASE, *CASES[name], f"max_steps={steps}", "save_interval=1000000000", out_root=d)
    finally:
        del os.environ["FENICE_HALO_REACH"]
    return [line for line in r.stdout.splitlines() if "halo reach" in line]


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS, help="rank counts (1: serial, one thread)")
    ap.add_argument("--only", nargs="+", choices=list(CASES), help="only these meshes")
    ap.add_argument("--steps", type=int, default=50, help="time steps per run (default 50)")
    ap.add_argument("--reach", action="store_true", help="also print the halo reach of each mesh")
    args = ap.parse_args()
    names = args.only or list(CASES)
    results = {name: {n: measure(args, name, n) for n in args.ranks} for name in names}

    for name in names:
        res = results[name]
        first = res[args.ranks[0]]
        t = first["tables"]
        print(f"\n{name}: {first['leaves']} leaves at the end, {args.steps} steps; tables of a rank [MB]: mesh {t['mesh']:.0f}, "
              f"engine {t['engine']:.0f}, uniform-mesh arrays {t['uniform']:.0f}, body band {t['band']:.0f}")
        print(f"{'ranks':>6} {'wall':>7} {'remesh':>7} {'gather':>7} {'indic.':>7} {'adapt':>7} {'build':>7} {'part.':>7}"
              f" | {'RSS per rank [MB]: setup':>24} {'init':>13} {'end':>13} {'peak':>13}")
        for n in args.ranks:
            r = res[n]
            p, lo, hi = r["parts"], r["memory_min"], r["memory_max"]
            mem = " ".join(f"{lo[k]:6.0f}-{hi[k]:<6.0f}" for k in MEMORY)
            print(f"{n:>6} {r['wall']:7.2f} {r['remesh']:7.2f} {p['gather']:7.2f} {p['indicator']:7.2f} {p['adapt']:7.2f} "
                  f"{p['build']:7.2f} {p['partition']:7.2f} |          {mem}")
        if args.reach:
            for line in reach(name, min(args.steps, 10)):
                print(line)
    (study.study_dir(STUDY) / "summary.json").write_text(json.dumps(results, indent=1))

    plt = study.use_style()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    for k, name in enumerate(names):
        res = results[name]
        ax[0].plot(args.ranks, [res[n]["memory_max"]["peak"] for n in args.ranks], "o-", color=f"C{k}", label=f"{name} peak")
        ax[0].plot(args.ranks, [res[n]["memory_max"]["end"] for n in args.ranks], "s--", color=f"C{k}", label=f"{name} end")
        ax[1].plot(args.ranks, [res[n]["wall"] - res[n]["remesh"] for n in args.ranks], "o-", color=f"C{k}",
                   label=f"{name} time stepping")
        ax[1].plot(args.ranks, [res[n]["remesh"] for n in args.ranks], "s--", color=f"C{k}", label=f"{name} remesh")
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xticks(args.ranks, [str(n) for n in args.ranks])
        a.set_xlabel("MPI ranks (1: serial)")
        a.legend()
    ax[0].set(ylabel="resident memory per rank [MB] (max over ranks)", title="Memory")
    ax[1].set(ylabel=f"wall time of {args.steps} steps [s]", title="Time")
    fig.suptitle("Replicated AMR (every rank stores the whole mesh): the baseline of the distributed storage", y=1.02)
    study.save(fig, STUDY, "amr_memory.png", args.docs)


if __name__ == "__main__":
    main()
