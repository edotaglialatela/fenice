"""Phase Q8a of the 3-D octree AMR: what the 3-D octree costs and what it gives against the uniform 3-D solver (no engine change).

    python cases/validation/amr/octree_q8.py calibrate    # level-0 octree against the uniform solver (the monitor's Cd), ghost wall active
    python cases/validation/amr/octree_q8.py run          # accuracy runs to t = 30, cached: sphere Re 20 and 100, uniform and octree
    python cases/validation/amr/octree_q8.py timing       # steady-state cost: threads, MPI ranks, remesh share, memory (resumes the runs)
    python cases/validation/amr/octree_q8.py report       # tables, bands, summary.json and a figure

Everything goes to results/validation/amr_octree_q8/. Run it on an idle machine, one stage at a time: the times are the point. Not in CI.

The sphere is the performance target fixed in Q0 (section 11): diameter 1 in a 6 x 3 x 3 channel, Mach 0.3, SSP-RK2, MUSCL, HLLC low-Mach,
Re = 20 and 100 (both steady), run to t = 30 (the runs of features/ibm_sphere.py). The octree has the body band (2 finest cells for the
mask, 6 for the ghost wall) and the default indicator (density + velocity, criterion 0.1), remeshed every 4 steps.

Claims and bands, fixed before the first run:
  1. Same finest resolution: the octree with finest n cells per diameter gives the Cd of the uniform n-cells-per-diameter mesh within 2 %
     (mask and ghost wall, both Reynolds numbers). This needs no external reference.
  2. Cost at the same finest resolution, 16 threads, steady state: wall time per unit of physical time of the octree at most 0.5 of the
     uniform one (the gate of section 7); the Q0 estimate was 0.16-0.26 times the per-leaf cost ratio of the face engine.
  3. The ghost wall moves Cd between two finest resolutions less than the mask does (ratio <= 0.6, the band of ibm_sphere.py), on the octree
     (finest 20 -> 40, Re 20; needs `run --include-40`, skipped otherwise).
  4. Level-0 octree against the uniform solver (calibration): the Cd of the monitor within 0.5 % after 200 steps. (The two IBM resets are not
     bitwise the same, CLAUDE.md; the loads use the same normalisation.)
Reported without a band: scaling over threads and ranks, remesh share, memory per leaf and per rank, the observed order of Cd on the octree.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import fenice
from fenice import study

STUDY = "amr_octree_q8"
OUT = study.study_dir(STUDY)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from octree_q0 import sphere_case   # noqa: E402  (the case file of Q0: sphere in a channel, Re as a parameter)

T_END = 30.0
THREADS = 16
AMR_EVERY = 4
BAND_SAME, BAND_WALL, BAND_RATIO, BAND_LEVEL0 = 0.02, 0.5, 0.6, 0.005


def amr_sets(M, method):
    return ["use_amr=true", f"amr_max_level={M}", "amr_refine_ibm=true", f"amr_refine_every={AMR_EVERY}", f"ibm_method={method}"]


def configs(include40=False):
    """Accuracy runs in priority order: (tag, re, kind, finest n, base n, levels, method)."""
    rows = []

    def add(re, kind, fin, base, M, method):
        tag = f"re{re}_uni_{method}_n{fin}" if kind == "uni" else f"re{re}_oct_{method}_b{base}M{M}"
        rows.append(dict(tag=tag, re=re, kind=kind, fin=fin, base=base, M=M, method=method))

    for re in (20, 100):
        for method in ("mask", "ghost"):
            add(re, "uni", 10, 10, 0, method)
            add(re, "uni", 20, 20, 0, method)
            add(re, "oct", 10, 5, 1, method)
            add(re, "oct", 20, 10, 1, method)
            add(re, "oct", 20, 5, 2, method)
    if include40:   # the finest octree: 40 cells per diameter, the mesh the uniform solver cannot afford here (1-2 h each, optional)
        for method in ("mask", "ghost"):
            add(20, "oct", 40, 10, 2, method)
    return rows


def final_cd(d):
    p = next(d.rglob("*_monitor.csv"))
    m = np.genfromtxt(p, delimiter=",", names=True)
    _, keep = np.unique(m["t"], return_index=True)
    cd = np.asarray(m["Cd"])[np.sort(keep)]
    return float(cd[-1]), float(abs(cd[-1] - cd[-6]) / abs(cd[-1]))


def run_one(c, force=False):
    d = OUT / c["tag"]
    info = d / "info.json"
    if info.exists() and not force:
        return json.loads(info.read_text())
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    octree = c["kind"] == "oct"
    case = sphere_case(c["re"], c["base"], t_end=T_END)
    sets = amr_sets(c["M"], c["method"]) if octree else [f"ibm_method={c['method']}"]
    print(f"[RUN] {c['tag']}", flush=True)
    t0 = time.time()
    r = fenice.run(case, *sets, "monitor_interval=50", out_root=d, threads=THREADS, dump=d / "final.npz")
    total = time.time() - t0
    marker = "Ghost-cell wall active" if octree else "Ghost-cell wall ("   # (the uniform solver prints the wall, the octree whether it is active)
    if c["method"] == "ghost" and marker not in r.stdout:
        raise RuntimeError(f"{c['tag']}: the ghost-cell wall is not active (the mask forcing was used): this column would be a mask one")
    got = fenice.parse_summary(r.stdout, require=("amr",) if octree else ())
    cd, settling = final_cd(d)
    out = dict(c, cd=cd, settling=settling, wall=got["wall"], steps=got["steps"], total=total, t=float(np.load(d / "final.npz")["t"]))
    for k in ("leaves", "remesh", "levels", "memory_max", "tables"):
        if k in got:
            out[k] = got[k]
    info.write_text(json.dumps(out))
    print(f"      Cd {cd:.4f} (settling {settling:.1e}), time loop {got['wall']:.0f} s, {got['steps']} steps"
          + (f", {got['leaves']} leaves" if octree else ""), flush=True)
    return out


def cmd_calibrate(args):
    """Level-0 octree (nothing refined) against the uniform solver on the same mesh: the monitor Cd over 200 steps."""
    case = sphere_case(20, 10, t_end=T_END)
    res = {}
    for tag, extra in (("uni", []), ("oct0", ["use_amr=true", "amr_max_level=1", "amr_refine_ibm=false", "amr_criterion=1e9",
                                              f"amr_refine_every={AMR_EVERY}"])):
        d = OUT / "calibrate" / tag
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        fenice.run(case, "ibm_method=mask", "max_steps=200", "monitor_interval=10", *extra, out_root=d, threads=THREADS)
        res[tag] = np.genfromtxt(next(d.rglob("*_monitor.csv")), delimiter=",", names=True)
    a, b = res["uni"], res["oct0"]
    rel = float(np.max(np.abs(a["Cd"] - b["Cd"]) / np.abs(a["Cd"])))
    ok = rel <= BAND_LEVEL0
    print(f"level-0 octree vs uniform (Re 20, 10 cells/D, mask, 200 steps): max relative difference of Cd {rel:.2e} "
          f"(band {BAND_LEVEL0:g}) {'pass' if ok else 'FAIL'}; Cd {a['Cd'][-1]:.5f} / {b['Cd'][-1]:.5f}")
    (OUT / "calibrate.json").write_text(json.dumps({"max_rel_cd": rel, "band": BAND_LEVEL0, "passed": bool(ok)}))
    return 0 if ok else 1


def cmd_run(args):
    wanted = [c for c in configs(args.include_40) if not args.only or args.only in c["tag"]]
    for c in wanted:
        run_one(c, force=args.force)


# ---- timing ---------------------------------------------------------------------------------------------------------------------
def timing_run(c, workdir, steps, threads=None, mpi=0):
    """Resume the steady run `c` for `steps` steps in a copy of its folder: (time loop wall [s] per unit of physical time, summary)."""
    src = OUT / c["tag"]
    if workdir.exists():
        shutil.rmtree(workdir)
    shutil.copytree(src, workdir, ignore=shutil.ignore_patterns("*.vtu", "*.vtr", "*.pvtu", "info.json"))
    octree = c["kind"] == "oct"
    case = sphere_case(c["re"], c["base"], t_end=T_END)
    sets = amr_sets(c["M"], c["method"]) if octree else [f"ibm_method={c['method']}"]
    t_start = float(np.load(src / "final.npz")["t"])
    dump = workdir / "after.npz"
    r = fenice.run(case, *sets, "resume=true", f"max_steps={steps}", "t_end=1e9", "monitor_interval=0", "save_interval=1000000000",
                   out_root=workdir, threads=threads, mpi=mpi, dump=dump)
    got = fenice.parse_summary(r.stdout, require=("amr",) if octree else ())
    dt = float(np.load(dump)["t"]) - t_start
    return got["wall"] / dt, got, dt


def cmd_timing(args):
    steps, repeats = args.steps, 3
    work = OUT / "timing_work"
    out = {}
    cases = [c for c in configs() if c["re"] == 20 and c["method"] == "mask" and c["fin"] == 20]
    for c in cases:
        row = {"thread": {}, "rank": {}}
        for t in (1, 4, 8, 16):
            best = min(timing_run(c, work, steps, threads=t)[0] for _ in range(repeats))
            row["thread"][t] = best
            print(f"{c['tag']:22s} {t:2d} threads: {best:8.1f} s per unit time", flush=True)
        if c["kind"] == "oct":
            for n in (1, 2, 4, 8):
                best, parts, mem = None, None, None
                for _ in range(repeats):
                    w, got, _ = timing_run(c, work, steps, mpi=n)
                    if best is None or w < best:
                        best, parts, mem = w, got.get("parts"), got.get("memory_max")
                row["rank"][n] = dict(rate=best, parts=parts, memory_max=mem)
                print(f"{c['tag']:22s} {n:2d} ranks:   {best:8.1f} s per unit time, remesh parts {parts}", flush=True)
        out[c["tag"]] = row
    shutil.rmtree(work, ignore_errors=True)
    (OUT / "timing.json").write_text(json.dumps(out, indent=1))


# ---- report ---------------------------------------------------------------------------------------------------------------------
def cmd_report(args):
    runs = {c["tag"]: run_one(c) for c in configs(True) if (OUT / c["tag"] / "info.json").exists()}
    summary, ok = {"runs": runs}, True
    print("\nAccuracy runs (16 threads, t = 30):")
    print(f"{'run':22s} {'Cd':>8s} {'settle':>8s} {'leaves':>9s} {'steps':>7s} {'wall [s]':>9s} {'s / unit t':>10s}")
    for tag, r in runs.items():
        print(f"{tag:22s} {r['cd']:8.4f} {r['settling']:8.1e} {r.get('leaves', ''):>9} {r['steps']:7d} "
              f"{r['wall']:9.0f} {r['wall'] / r['t']:10.1f}")
    print("\nSame finest resolution (octree against uniform), bands: dCd <= 2 %, wall ratio <= 0.5 (whole-run time loops)")
    pairs = []
    for re in (20, 100):
        for method in ("mask", "ghost"):
            for n in (10, 20):
                u = runs.get(f"re{re}_uni_{method}_n{n}")
                for tag, r in runs.items():
                    if u and r["kind"] == "oct" and r["re"] == re and r["method"] == method and r["fin"] == n:
                        d = abs(r["cd"] - u["cd"]) / abs(u["cd"])
                        w = r["wall"] / u["wall"]
                        good = d <= BAND_SAME and (w <= BAND_WALL or n == 10)   # the cost band is for the 20-cell meshes
                        ok = ok and good
                        pairs.append(dict(oct=tag, uni=u["tag"], dcd=d, wall_ratio=w, steps_ratio=r["steps"] / u["steps"]))
                        print(f"  {tag:22s} vs {u['tag']:22s} dCd {100 * d:5.2f} %  wall {w:5.2f}  "
                              f"{'pass' if good else 'FAIL'}")
    summary["pairs"] = pairs
    print("\nSelf-convergence of Cd on the octree (finest 20 -> 40, Re 20): the ghost wall against the mask")
    ch = {}
    for method in ("mask", "ghost"):
        a, b = runs.get(f"re20_oct_{method}_b5M2"), runs.get(f"re20_oct_{method}_b10M2")
        if a and b:
            ch[method] = abs(b["cd"] - a["cd"])
            print(f"  {method:5s}: Cd {a['cd']:.4f} -> {b['cd']:.4f}  (change {ch[method]:.4f})")
    if len(ch) == 2:
        ratio = ch["ghost"] / ch["mask"]
        summary["wall_ratio_band3"] = ratio
        good = ratio <= BAND_RATIO
        ok = ok and good
        print(f"  ratio ghost / mask {ratio:.2f} (band {BAND_RATIO}) {'pass' if good else 'FAIL'}")
    tj = OUT / "timing.json"
    if tj.exists():
        tm = json.loads(tj.read_text())
        summary["timing"] = tm
        uni = tm.get("re20_uni_mask_n20")
        print("\nSteady-state cost, Re 20 mask, finest 20 (200 steps resumed from the t = 30 state, min of 3): wall seconds per unit of physical time")
        print(f"{'':22s} " + " ".join(f"{t:>8s}" for t in ("1 thr", "4 thr", "8 thr", "16 thr")) + "   speed-up 1->16, ratio to uniform at 1 / 16 threads")
        for tag, row in tm.items():
            th = row["thread"]
            ratio = (f"{th['1'] / uni['thread']['1']:.2f} / {th['16'] / uni['thread']['16']:.2f}" if uni else "")
            print(f"{tag:22s} " + " ".join(f"{th[k]:8.1f}" for k in ("1", "4", "8", "16")) + f"   {th['1'] / th['16']:.1f}x   {ratio}")
        print("\nMPI ranks (one thread each; the 1-rank row is the OpenMP run on 16 threads): wall per unit time, speed-up over the 1-thread run, remesh share")
        for tag, row in tm.items():
            if not row["rank"]:
                continue
            r = runs[tag]
            dt200 = 200.0 * r["t"] / r["steps"]   # the mean step of the accuracy run stands for the step of the steady state
            one = row["thread"]["1"]
            for n, e in row["rank"].items():
                share = sum(e["parts"].values()) / (e["rate"] * dt200) if e["parts"] and n != "1" else float("nan")
                mem = e["memory_max"]
                print(f"  {tag:22s} {n:>2s} ranks {e['rate']:7.1f} s  {one / e['rate']:4.1f}x  remesh {100 * share:4.0f} %  "
                      f"peak per rank {mem['peak']:.0f} MB, end {mem['end']:.0f} MB")
    summary["passed"] = bool(ok)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\nALL BANDS PASSED" if ok else "\nSOME BANDS FAILED")
    return 0 if ok else 1


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("calibrate", "run", "timing", "report"))
    ap.add_argument("--only", default="", help="run: substring filter on the run tags")
    ap.add_argument("--include-40", action="store_true", help="run: also the octree with 40 cells per diameter (1-2 h each)")
    ap.add_argument("--steps", type=int, default=200, help="timing: steps per resumed run")
    args = ap.parse_args()
    return {"calibrate": cmd_calibrate, "run": cmd_run, "timing": cmd_timing, "report": cmd_report}[args.stage](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
