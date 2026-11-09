"""Fault injection: which checks catch which defects?

Known defects are put into the solver's source one at a time, the solver is rebuilt, and every check of the repository is
run on it. Two kinds of check are compared: the ones that compare against a TOLERANCE (the reference states at 1e-8, the
manufactured-solution convergence bands) and the ones that compare BITS between code paths (the bitwise unit tests,
serial against 1-7 MPI ranks and restarts on the adaptive mesh, the ghost-cell wall on 1-7 ranks).

    python cases/validation/verification/fault_injection.py            # all faults (about 100 s each), results/validation/fault_injection/
    python cases/validation/verification/fault_injection.py --only F03 F05
    python cases/validation/verification/fault_injection.py --list     # the faults and their predictions
    python cases/validation/verification/fault_injection.py --dry-run  # check that every patch applies, no build

THIS SCRIPT EDITS THE SOURCES (and restores them with git) and rebuilds build/release: run it in a disposable checkout (a git
worktree), never in one with work in progress. It refuses to start if cpp/ or CMakeLists.txt have uncommitted changes.

The prediction of every fault (which checks catch it) was written into this file before the first faulty build was run;
the report prints each prediction next to what was observed, and the disagreements are listed, not hidden.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "validation" / "fault_injection"
WIN = os.name == "nt"
EXT = ".exe" if WIN else ""
BUILD = ROOT / "build" / "release"
PY = sys.executable

# Detectors: (key, label, group, command, study folders whose cached runs must be dropped first)
DETECTORS = [
    ("reg", "regression (serial)", "tolerance", [PY, "cpp/tests/regression.py"], []),
    ("reg4", "regression (+4 ranks)", "tolerance", [PY, "cpp/tests/regression.py", "--mpi", "4"], []),
    ("mms", "MMS convergence bands", "tolerance", [PY, "cases/validation/verification/mms.py"], ["mms"]),
    ("unit", "unit tests", "bitwise", [str(BUILD / ("fenice_tests" + EXT))], []),
    ("amr", "AMR 1-7 ranks + restart", "bitwise", [PY, "cases/validation/amr/amr_mpi.py", "--check"], ["amr_mpi"]),
    ("ibm", "ghost-cell wall 1-7 ranks", "bitwise", [PY, "cases/validation/features/ibm_ghost.py", "--check"], ["ibm_ghost"]),
]
KEYS = [d[0] for d in DETECTORS]


def pred(*caught):
    """Prediction written before the run: the detectors (by key) expected to catch the fault."""
    return {k: (k in caught) for k in KEYS}


# id, title, size, file, old text, new text, predicted catches (fixed before the first faulty run)
FAULTS = [
    dict(id="F01", title="HLLC star-flux rearranged (scalar code only)", size="1e-16 relative",
         file="cpp/src/Schemes.cpp",
         old="for (int q = 0; q < kNVar; ++q) F[q] = FL[q] + S_L * (UsL[q] - UL[q]);",
         new="for (int q = 0; q < kNVar; ++q) F[q] = FL[q] + (S_L * UsL[q] - S_L * UL[q]);",
         predicted=pred("unit")),
    dict(id="F02", title="Van Leer limiter scaled by 1 + 1e-10 (scalar code only)", size="1e-10 relative",
         file="cpp/include/fv/Schemes.hpp",
         old="return (ab > 0) ? 2.0 * ab / (a + b) : 0.0;",
         new="return (ab > 0) ? 2.0 * ab / (a + b) * (1.0 + 1e-10) : 0.0;",
         predicted=pred("unit")),
    dict(id="F03", title="viscous cross-derivative on y-faces one-sided (uniform path)", size="O(h) in a viscous term",
         file="cpp/src/Physics.cpp",
         old="(gt ? cg->hi_a[i] : dudx[yT]));",
         new="(gt ? cg->hi_a[i] : dudx[yB]));",
         predicted=pred("reg", "reg4", "mms", "unit")),
    dict(id="F04", title="MPI interface: wrong index into the cross-derivative halo", size="interface cells only",
         file="cpp/src/Physics.cpp",
         old="(gb ? cg->lo_a[i] : dudx[yB])",
         new="(gb ? cg->lo_a[0] : dudx[yB])",
         predicted=pred("reg4")),
    dict(id="F05", title="AMR+MPI: coarse leaf's neighbours dropped from the halo list", size="only across rank interfaces",
         file="cpp/src/CellFlux.cpp",
         old="if (face.kind == CellFaceKind::CoarseFine) {   // the coarse leaf's own neighbours (second-order coarse side)",
         new="if (false && face.kind == CellFaceKind::CoarseFine) {   // the coarse leaf's own neighbours (second-order coarse side)",
         predicted=pred("unit", "amr"),
         equivalent="the branch is redundant: the neighbour marks of the two face cells (nbr_) already mark the same leaves, "
                    "the interpolated virtual cell of a coarser neighbour marks its coarse leaf (checked by an assertion that "
                    "the branch never added a cell, over the unit tests, the 4-rank regression, and the 1-7 rank AMR and "
                    "ghost-wall comparisons)"),
    dict(id="F06", title="restart: stored wave-speed bound scaled by 1 + 1e-10", size="1e-10 in the first time step",
         file="cpp/src/SimulationRestart.cpp", count=2,
         old='cell_speed_ = a.count("S") ? a.at("S").scalar() : 0.0;',
         new='cell_speed_ = a.count("S") ? a.at("S").scalar() * (1.0 + 1e-10) : 0.0;',
         predicted=pred("amr")),
    dict(id="F07", title="SSP-RK2: second stage evaluated at the old time", size="O(dt) in time-dependent sources",
         file="cpp/src/TimeScheme.cpp",
         old="    axpy(tmp_, U, dt, R1_);\n    stage_time = t + dt;\n    const double s2 = res(tmp_, R2_);\n    ssp_combine(U, tmp_, dt, R2_);\n    return {dt, std::max(s1, s2)};",
         new="    axpy(tmp_, U, dt, R1_);\n    stage_time = t;\n    const double s2 = res(tmp_, R2_);\n    ssp_combine(U, tmp_, dt, R2_);\n    return {dt, std::max(s1, s2)};",
         predicted=pred("reg", "reg4", "unit")),
    dict(id="F08", title="compiler flag /fp:precise -> /fp:fast (re-association, contraction)", size="1e-16 relative",
         file="cpp/CMakeLists.txt",
         old="/utf-8 /fp:precise /permissive-)", new="/utf-8 /fp:fast /permissive-)",
         predicted=pred("unit"), msvc_only=True),
    dict(id="F09", title="Prandtl number 0.1 % high in every path", size="1e-3 in the heat flux",
         file="cpp/include/fv/Physics.hpp",
         old="eos_(eos), mu_(mu), Pr_(Pr) {}", new="eos_(eos), mu_(mu), Pr_(Pr * 1.001) {}",
         predicted=pred("reg", "reg4")),
    dict(id="F10", title="flux-register weight 2e-7 too large (conservation)", size="2e-7 of a face flux",
         file="cpp/src/CellAdvance.cpp",
         old="accumulate_registers(mesh_, l, 0.5 * weights[stage]);",
         new="accumulate_registers(mesh_, l, 0.5000001 * weights[stage]);",
         predicted=pred("unit")),
    dict(id="F11", title="no-slip ghost velocity has the wrong sign (gross, control)", size="O(1)",
         file="cpp/src/Boundaries.cpp",
         old="    case BCType::NoSlip:\n        Wg[1] = -Wi[1];\n        Wg[2] = -Wi[2];\n        Wg[4] = -Wi[4];\n        break;\n    case BCType::MovingWall:",
         new="    case BCType::NoSlip:\n        Wg[1] = Wi[1];\n        Wg[2] = Wi[2];\n        Wg[4] = Wi[4];\n        break;\n    case BCType::MovingWall:",
         predicted=pred("reg", "reg4", "mms", "unit")),
    dict(id="F12", title="AMR+MPI halo list: neighbours of the two face cells not marked", size="viscous reads across rank interfaces",
         file="cpp/src/CellFlux.cpp",
         old="    cell(face.left);\n    cell(face.right);\n    if (face.kind == CellFaceKind::CoarseFine) {",
         new="    if (false) { cell(face.left); cell(face.right); }\n    if (face.kind == CellFaceKind::CoarseFine) {",
         predicted=pred("unit", "amr")),
]


def run(cmd, timeout=900, env=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, p.stdout + p.stderr, time.time() - t0
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or "") if isinstance(e.stdout, str) else "", time.time() - t0


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def build():
    cmd = ["powershell", "-NoProfile", "-File", "cpp/build.ps1"] if WIN else ["cmake", "--build", "build/release"]
    code, out, _ = run(cmd, timeout=1800)
    return code == 0, out


def apply(fault):
    f = ROOT / fault["file"]
    raw = f.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    n = text.count(fault["old"])
    want = fault.get("count", 1)   # F06: the resume reads the bound in two places (whole mesh and a rank's share)
    if n != want:
        raise SystemExit(f"{fault['id']}: the text to replace occurs {n} times in {fault['file']} (must be {want})")
    text = text.replace(fault["old"], fault["new"])
    f.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def restore(fault):
    git("checkout", "--", fault["file"])


def detect(env, log):
    """Run every detector; return {key: dict(caught, code, seconds, failed)}."""
    res = {}
    for key, label, group, cmd, caches in DETECTORS:
        for c in caches:   # cached runs are keyed by study and tag, not by the executable: drop them
            shutil.rmtree(ROOT / "results" / "validation" / c, ignore_errors=True)
        code, out, sec = run(cmd, env=env)
        failed = re.findall(r"\[\s*FAIL\w*\s*\]\s*(\S+)", out) if key == "unit" else []
        res[key] = dict(caught=code != 0, code=code, seconds=round(sec, 1), failed=failed[:6])
        log.write(f"--- {label}: exit {code}\n{out[-2000:]}\n")
        log.flush()
        print(f"      {label:28s} {'CAUGHT' if code != 0 else 'passes'}  ({sec:.0f} s)", flush=True)
    return res


def table(results):
    lines = ["| fault | size | " + " | ".join(d[1] for d in DETECTORS) + " | caught by |", "|---|---|" + "---|" * (len(DETECTORS) + 1)]
    for f in FAULTS:
        r = results.get(f["id"])
        if not r or "error" in r:
            lines.append(f"| {f['id']} {f['title']} | {f['size']} | " + " | ".join("n/a" for _ in DETECTORS) + " | build failed |")
            continue
        if f.get("equivalent"):
            lines.append(f"| {f['id']} {f['title']} | {f['size']} | " + " | ".join(
                "passes" if not r["detectors"][k]["caught"] else "**caught**" for k, *_ in DETECTORS) + " | equivalent mutant |")
            continue
        cells, groups = [], set()
        for key, _, group, _, _ in DETECTORS:
            c = r["detectors"][key]["caught"]
            exp = f["predicted"][key]
            cells.append(("**caught**" if c else "passes") + ("" if c == exp else " (predicted otherwise)"))
            if c:
                groups.add(group)
        lines.append(f"| {f['id']} {f['title']} | {f['size']} | " + " | ".join(cells) + " | " + (" + ".join(sorted(groups)) or "NOBODY") + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", help="fault ids")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="check that every patch applies; no build")
    ap.add_argument("--skip-baseline", action="store_true")
    a = ap.parse_args()
    faults = [f for f in FAULTS if not a.only or f["id"] in a.only]
    if a.list:
        for f in FAULTS:
            print(f"{f['id']}  {f['title']}{'  (EQUIVALENT)' if f.get('equivalent') else ''}  [{f['size']}]  predicted to be caught by: "
                  f"{', '.join(k for k in KEYS if f['predicted'][k]) or 'nobody'}")
        return 0
    dirty = git("status", "--porcelain", "--", "cpp", "CMakeLists.txt").stdout.strip()
    if dirty:
        print("cpp/ or CMakeLists.txt have uncommitted changes; commit or discard them first:\n" + dirty)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        for f in faults:
            apply(f)
            restore(f)
            print(f"{f['id']}: patch applies")
        return 0
    env = dict(os.environ, FENICE_EXE=str(BUILD / ("fenice" + EXT)))
    log = open(OUT / "log.txt", "w", encoding="utf-8")
    results = {}
    if not a.skip_baseline:
        print("baseline (no fault)", flush=True)
        ok, out = build()
        if not ok:
            print(out[-3000:])
            return 2
        base = detect(env, log)
        if any(v["caught"] for v in base.values()):
            print("the unmodified solver fails a check: no study possible")
            return 2
    try:
        for f in faults:
            if f.get("msvc_only") and not WIN:
                print(f"{f['id']}: skipped (MSVC flag)")
                continue
            print(f"{f['id']}  {f['title']}", flush=True)
            apply(f)
            ok, out = build()
            if not ok:
                results[f["id"]] = dict(error="build failed")
                restore(f)
                log.write(f"--- {f['id']} build failed\n{out[-3000:]}\n")
                continue
            results[f["id"]] = dict(detectors=detect(env, log))
            restore(f)
    finally:
        for f in faults:
            restore(f)
        build()   # leave the baseline binary in build/release
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    md = table(results)
    (OUT / "table.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    wrong = [(f["id"], k) for f in faults if f["id"] in results and "detectors" in results[f["id"]]
             for k in KEYS if results[f["id"]]["detectors"][k]["caught"] != f["predicted"][k]]
    print(f"\npredictions that did not hold: {len(wrong)} of {len(faults) * len(KEYS)}: {wrong}")
    nobody = [f["id"] for f in faults if f["id"] in results and "detectors" in results[f["id"]] and not f.get("equivalent")
              and not any(v["caught"] for v in results[f["id"]]["detectors"].values())]
    print("faults caught by no check:", nobody or "none")
    return 1 if nobody else 0


if __name__ == "__main__":
    sys.exit(main())
