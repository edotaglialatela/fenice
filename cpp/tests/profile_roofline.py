#!/usr/bin/env python3
"""Where does the uniform residual stand against the memory-bandwidth ceiling?

Runs fenice on a periodic Taylor-Green box (no output, fixed step count) over mesh size x threads x
scheme and prints ns per cell-stage and speed-up over one thread. Its bandwidth column is an implied
rate under the 160 B/cell-stage model, not a measurement of solver traffic.

    python cpp/tests/profile_roofline.py                  # quick: 3 sizes, 1/8/16 threads
    python cpp/tests/profile_roofline.py --sections       # also FENICE_PROFILE sections of the biggest run
    python cpp/tests/profile_roofline.py --sizes 256 1024 4096 --threads 1 4 8 16 --steps 30

Reading the table: ns/cell-stage that stops falling when threads go up (or rises with the mesh size
leaving L2/L3) = bandwidth bound; falling ~1/threads = compute bound. Nothing here changes the solver.
"""
import argparse, csv, datetime, os, platform, re, statistics, subprocess, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = Path(os.environ.get("FENICE_EXE", ROOT / "build" / "release" / "fenice.exe"))
if not EXE.exists():
    EXE = ROOT / "build" / "release" / "fenice"

# (label, --set options, stages per step)
SCHEMES = {
    "euler 1st":      (["use_muscl=false", "use_viscous=false", "time_scheme=Explicit Euler"], 1),
    "muscl rk2":      (["use_muscl=true",  "use_viscous=false", "time_scheme=SSP Runge-Kutta 2"], 2),
    "muscl rk2 visc": (["use_muscl=true",  "use_viscous=true",  "time_scheme=SSP Runge-Kutta 2"], 2),
}
BYTES_PER_STATE = 5 * 8   # State is five doubles in 2-D too


def stream_bandwidth(threads, mb=200, secs=2.0):
    """Aggregate copy bandwidth (read + write) in GB/s: `threads` threads copy 200 MB arrays at the same time
    (numpy releases the GIL). Processes that are not started together overstate it; one core already gets most of it."""
    import threading
    import numpy as np
    arrs = [(np.ones(mb * 2**20 // 8), np.empty(mb * 2**20 // 8)) for _ in range(threads)]
    for x, y in arrs:
        np.copyto(y, x)
    cnt = [0] * threads; bar = threading.Barrier(threads + 1); stop = [False]

    def work(i):
        x, y = arrs[i]; bar.wait()
        while not stop[0]:
            np.copyto(y, x); cnt[i] += 1
    ts = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    bar.wait(); t0 = time.perf_counter(); time.sleep(secs); stop[0] = True; el = time.perf_counter() - t0
    for t in ts:
        t.join()
    return sum(cnt) * 2 * mb * 2**20 / el / 1e9


def run(n, threads, opts, steps, extra_env=None, case="Taylor-Green Vortex", extra_sets=(), case_boundaries=False,
        switches=None):
    out = Path(tempfile.mkdtemp(prefix="fenice_prof_"))
    cmd = [str(EXE), "--case", case, "--threads", str(threads),
           "--set", f"nx={n}", "--set", f"ny={n}", "--set", f"max_steps={steps}",
           "--set", "save_interval=1000000", "--set", f"output_root={out}", "--set", "t_end=1e9"]
    if not case_boundaries:
        for side in ("left", "right", "bottom", "top"):
            cmd += ["--set", f"bc_{side}=periodic"]
    for o in (*opts, *extra_sets):
        cmd += ["--set", o]
    env = dict(os.environ, **(extra_env or {}))
    for switch in (switches or []):
        name, sep, value = switch.partition("=")
        env[name] = value if sep else "1"
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    m = re.search(r"time loop ([\d.]+) s, (\d+) steps", r.stdout)
    if not m:
        sys.exit(f"run failed:\n{' '.join(cmd)}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return float(m.group(1)), int(m.group(2)), r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--steps-per-size", action="append", default=[], metavar="N=COUNT",
                    help="override steps at a grid size, repeatable; useful to bound large-grid runs")
    ap.add_argument("--schemes", nargs="+", default=list(SCHEMES))
    ap.add_argument("--sections", action="store_true", help="save FENICE_PROFILE output of the largest run")
    ap.add_argument("--repeats", type=int, default=3, help="independent runs per size/thread/scheme (default: 3)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "validation" / "performance",
                    help="directory for measurement CSV and profile output")
    ap.add_argument("--case", default="Taylor-Green Vortex", help="case to profile (default: Taylor-Green Vortex)")
    ap.add_argument("--extra-set", action="append", default=[], metavar="KEY=VALUE", help="additional solver setting; repeatable")
    ap.add_argument("--case-boundaries", action="store_true", help="keep boundary conditions from the selected case")
    ap.add_argument("--switch", action="append", default=[], metavar="NAME[=VALUE]", help="environment switch; repeatable, e.g. FENICE_NO_FUSED=1")
    a = ap.parse_args()
    if a.repeats < 1:
        ap.error("--repeats must be at least 1")
    steps_by_size = {}
    for item in a.steps_per_size:
        try:
            size, count = (int(v) for v in item.split("=", 1))
        except (ValueError, TypeError):
            ap.error(f"invalid --steps-per-size {item!r}; expected N=COUNT")
        if count < 1:
            ap.error("--steps-per-size counts must be positive")
        steps_by_size[size] = count

    maxt = max(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"solver: {EXE}\nplatform: {platform.platform()}\nCPU: {platform.processor() or 'unknown'}; logical CPUs: {os.cpu_count()}\n"
          f"copy bandwidth reference: measuring with 1 and {maxt} concurrent copy threads ...", flush=True)
    bw1, bwn = stream_bandwidth(1), stream_bandwidth(maxt)
    print(f"copy bandwidth reference: {bw1:.1f} GB/s on 1 thread, {bwn:.1f} GB/s on {maxt}; not a hard DRAM ceiling\n")

    csv_path = a.out / "roofline.csv"
    rows = []
    print(f"{'scheme':16}{'mesh':>7}{'MB(U)':>8}{'thr':>5}{'min ns/cs':>13}{'median':>10}{'speedup':>9}{'implied GB/s*':>15}")
    for name in a.schemes:
        opts, stages = SCHEMES[name]
        for n in a.sizes:
            step_count = steps_by_size.get(n, a.steps)
            base = None
            for t in a.threads:
                samples = []
                for rep in range(a.repeats):
                    wall, steps, _ = run(n, t, opts, step_count, case=a.case, extra_sets=a.extra_set,
                                         case_boundaries=a.case_boundaries, switches=a.switch)
                    samples.append(wall / (steps * stages * n * n) * 1e9)
                    print(f"  {name}, {n}^2, {t} threads: run {rep + 1}/{a.repeats}: {samples[-1]:.2f} ns/cell-stage", flush=True)
                ns = min(samples)
                median = statistics.median(samples)
                if t == a.threads[0]:
                    base = ns
                # 160 B per cell-stage is an estimate; this is not measured DRAM traffic.
                implied = 160 / ns
                print(f"{name:16}{n:>5}^2{n * n * BYTES_PER_STATE / 2**20:>8.1f}{t:>5}{ns:>13.2f}{median:>10.2f}{base / ns:>8.2f}x{implied:>15.1f}", flush=True)
                rows.append({"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                             "case": a.case, "extra_sets": ";".join(a.extra_set), "switches": ";".join(a.switch),
                             "scheme": name, "nx": n, "ny": n, "threads": t, "steps": step_count,
                             "repeats": a.repeats, "sample_ns_per_cell_stage": ";".join(f"{v:.6f}" for v in samples),
                             "min_ns_per_cell_stage": f"{ns:.6f}", "median_ns_per_cell_stage": f"{median:.6f}",
                             "speedup_vs_first_thread_count": f"{base / ns:.6f}",
                             "implied_GB_s_at_160B": f"{implied:.6f}", "copy_GB_s_1_thread": f"{bw1:.6f}",
                             "copy_GB_s_max_threads": f"{bwn:.6f}", "assumed_bytes_per_cell_stage": 160,
                             "binary": str(EXE), "platform": platform.platform(), "cpu": platform.processor()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nCSV saved: {csv_path}")
    if a.sections:
        n = max(a.sizes)
        profile_text = []
        profile_rows = []
        assumed_bytes = {"flux": 120, "padded copy": 120, "viscous conversion": 120,
                         "viscous sweep": 160, "update axpy": 160, "update SSP-RK2": 160,
                         "update SSP-RK3": 160, "update RK4": 160}
        for name in a.schemes:
            opts, _ = SCHEMES[name]
            for t in a.threads:
                for rep in range(a.repeats):
                    _, _, out = run(n, t, opts, steps_by_size.get(n, a.steps), {"FENICE_PROFILE": "1"}, a.case, a.extra_set,
                                    a.case_boundaries, a.switch)
                    profile = "\n".join(l for l in out.splitlines() if "[PROFILE]" in l)
                    profile_text.append(f"FENICE_PROFILE, {name}, {n}^2, {t} threads, profile run {rep + 1}/{a.repeats}\n{profile}")
                    for line in profile.splitlines():
                        m = re.search(r"\[PROFILE\]\s+(.*?)\s+(\d+) calls\s+([0-9.]+)\s*/\s*([0-9.]+) s", line)
                        if not m:
                            continue
                        section, calls, lo, hi = m.group(1).strip(), int(m.group(2)), float(m.group(3)), float(m.group(4))
                        if calls == 0:
                            continue
                        sec_call = hi / calls
                        ns_cell = sec_call / (n * n) * 1e9
                        byte_model = assumed_bytes.get(section)
                        profile_rows.append({"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                             "case": a.case, "extra_sets": ";".join(a.extra_set), "switches": ";".join(a.switch),
                                             "scheme": name,
                                             "nx": n, "ny": n, "threads": t, "repeat": rep + 1,
                                             "section": section, "calls": calls, "profile_seconds_min_rank": f"{lo:.9f}",
                                             "profile_seconds_max_rank": f"{hi:.9f}", "seconds_per_call_max_rank": f"{sec_call:.12f}",
                                             "ns_per_cell_per_call": f"{ns_cell:.6f}",
                                             "assumed_bytes_per_cell": "" if byte_model is None else byte_model,
                                             "implied_GB_s_from_assumed_bytes": "" if byte_model is None else f"{byte_model / ns_cell:.6f}",
                                             "interpretation": "timing measured; bytes modeled, not measured" if byte_model is not None else "timing measured"})
        profile_path = a.out / "profile_sections.txt"
        profile_path.write_text("\n\n".join(profile_text) + "\n", encoding="utf-8")
        pass_csv = a.out / "profile_passes.csv"
        if profile_rows:
            with pass_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=profile_rows[0].keys())
                writer.writeheader()
                writer.writerows(profile_rows)
        print(f"\nProfiles saved: {profile_path}\nPass timing/model CSV saved: {pass_csv}")


if __name__ == "__main__":
    main()
