"""What the adaptive mesh buys: time and memory against a uniform mesh of the same finest resolution, on 1 to 16 threads.

    python cases/validation/amr/amr_advantage.py            # results/validation/amr_advantage/
    python cases/validation/amr/amr_advantage.py --docs     # also docs/img/amr_advantage.png

The 2-D IBM airfoil (Re 1000, 720 x 360 base mesh, two levels: 2880 x 1440 at the finest level, the airfoil and the wake refined)
is run by the adaptive solver for 50 coarse steps and by the uniform solver on the 2880 x 1440 mesh for the same simulated time
(the uniform step is the finest-level step, a quarter of the coarse one: 200 steps). Both on 1, 2, 4, 8 and 16 OpenMP threads,
the same machine, one run each. Printed: the wall time of the time loop, the simulated time reached (they must agree), the cells
against the leaves and the resident memory.

The figure plots the wall time against the threads for both: the adaptive curve lies below the uniform one at every thread
count; the gap narrows slowly (the adaptive step parallelises less well than the uniform one) but stays large.
The 3-D octree sphere of octree_q8.py (time loop 0.09-0.13 of the uniform solver, read from its summary.json) is drawn beside it.
"""
import json

import numpy as np

import fenice
from fenice import study

STUDY, CASE = "amr_advantage", "IBM Airfoil AMR"
THREADS = [1, 2, 4, 8, 16]
STEPS_AMR, STEPS_UNI = 50, 200
AMR_SETS = ("nx=720", "ny=360", f"max_steps={STEPS_AMR}")
UNI_SETS = ("use_amr=false", "nx=2880", "ny=1440", f"max_steps={STEPS_UNI}")


def measure(args, tag, sets, threads):
    d = study.study_dir(STUDY) / tag
    info = d / "info.json"
    if args.force or not info.exists():
        d.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {STUDY}/{tag}", flush=True)
        r = fenice.run(CASE, *sets, "save_interval=1000000000", out_root=d, threads=threads, dump=d / "final.npz")
        got = fenice.parse_summary(r.stdout)
        dump = np.load(d / "final.npz")
        rec = {"wall": got["wall"], "steps": got["steps"], "t": float(dump["t"]), "leaves": got.get("leaves"),
               "remesh": got.get("remesh"), "memory": got.get("memory_max")}
        info.write_text(json.dumps(rec))
    return json.loads(info.read_text())


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    amr = {n: measure(args, f"amr_t{n}", AMR_SETS, n) for n in THREADS}
    uni = {n: measure(args, f"uni_t{n}", UNI_SETS, n) for n in THREADS}
    cells = 2880 * 1440
    leaves = amr[1]["leaves"]
    print(f"cells {cells}, leaves {leaves} ({100 * leaves / cells:.1f} %); simulated time AMR {amr[1]['t']:.4f}, uniform {uni[1]['t']:.4f}")
    for n in THREADS:
        a, u = amr[n], uni[n]
        print(f"{n:2d} threads: AMR {a['wall']:7.1f} s (remesh {a['remesh']:.1f} s)   uniform {u['wall']:7.1f} s   "
              f"ratio {a['wall'] / u['wall']:.3f} (uniform / AMR = {u['wall'] / a['wall']:.1f}x)")
    for tag, r in (("AMR", amr[1]), ("uniform", uni[1])):
        if r.get("memory"):
            print(f"{tag} resident memory (peak, 1 thread): {r['memory']}")

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].loglog(THREADS, [uni[n]["wall"] for n in THREADS], "o-", color="#7f8c8d", lw=2, label=f"uniform {2880}x{1440} ({cells / 1e6:.1f} M cells)")
    ax[0].loglog(THREADS, [amr[n]["wall"] for n in THREADS], "s-", color="#2a78d6", lw=2,
                 label=f"adaptive, same finest mesh ({leaves / 1e3:.0f} k leaves)")
    ax[0].loglog(THREADS, [amr[n]["wall"] - amr[n]["remesh"] for n in THREADS], "s--", color="#2a78d6", lw=1.2, alpha=0.7,
                 label="adaptive, time stepping only")
    ax[0].set_xticks(THREADS)
    ax[0].set_xticklabels(THREADS)
    ax[0].set_xlabel("OpenMP threads")
    ax[0].set_ylabel("wall time, same simulated time [s]")
    ax[0].set_title("Airfoil, Re 1000: time to the same finest-grid solution")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, which="both", ls="--", alpha=0.4)
    r16 = uni[16]["wall"] / amr[16]["wall"]
    ax[0].text(11.5, (uni[16]["wall"] * amr[16]["wall"]) ** 0.5, f"{r16:.0f}x faster\non 16 threads", ha="center", va="center",
               fontsize=10, color="#2a78d6")

    names = ["airfoil 2-D\n(16 threads)", "sphere 3-D, Re 20\n(16 threads)"]
    sph = json.loads((fenice.RESULTS / "validation" / "amr_octree_q8" / "summary.json").read_text())["runs"]
    sph_leaves = np.mean([sph[f"re20_oct_{m}_b5M2"]["leaves"] for m in ("mask", "ghost")])
    sph_time = np.mean([sph[f"re20_oct_{m}_b5M2"]["wall"] / sph[f"re20_uni_{m}_n20"]["wall"] for m in ("mask", "ghost")])
    leaf_share = [leaves / cells, sph_leaves / 432e3]
    time_share = [amr[16]["wall"] / uni[16]["wall"], sph_time]   # sphere: octree_q8.py, time loop octree / uniform, mask and ghost wall
    x = np.arange(2)
    ax[1].bar(x - 0.18, leaf_share, 0.34, color="#eb6834", label="leaves / uniform cells")
    ax[1].bar(x + 0.18, time_share, 0.34, color="#2a78d6", label="time / uniform time")
    for xi, v in zip(x - 0.18, leaf_share):
        ax[1].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    for xi, v in zip(x + 0.18, time_share):
        ax[1].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names, fontsize=9)
    ax[1].set_ylim(0, 0.5)
    ax[1].set_title("Adaptive / uniform, same finest mesh")
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    study.save(fig, STUDY, "amr_advantage.png", args.docs)


if __name__ == "__main__":
    main()
