"""Mutation study: a defect set that is not chosen by us.

The fault-injection study (`fault_injection.py`) has twelve defects that we picked, which says little about the defects nobody
thought of. This script draws single-line mutants of the numerical sources AT RANDOM (fixed seed), builds each one, and runs the same
six checks. A mutant is one change of one line: an arithmetic operator swapped (+ <-> -, * <-> /), a comparison moved across its
boundary (< <-> <=, > <-> >=) or a logical operator swapped (&& <-> ||). Mutants that do not compile are drawn again.

    python cases/validation/verification/mutation_study.py                    # 60 mutants (about 2-3 hours), results/validation/mutation_study/
    python cases/validation/verification/mutation_study.py --n 20 --seed 7
    python cases/validation/verification/mutation_study.py --list             # the drawn mutants, no build
    python cases/validation/verification/mutation_study.py --report           # table and statistics from results.json
    python cases/validation/verification/mutation_study.py --resume           # continue an interrupted run (results.json is kept per mutant)

THIS SCRIPT EDITS THE SOURCES and rebuilds build/release (see fault_injection.py): run it in a git worktree.

Written before the first run (hypotheses, not predictions per mutant, which cannot be made for random lines):
1. the checks that compare bits between code paths (unit tests, AMR on 1-7 ranks, ghost wall on 1-7 ranks) kill more mutants than the
   ones that compare against a tolerance (regression references, convergence bands);
2. the convergence bands kill the fewest;
3. the survivors are mostly equivalent mutants (a change that cannot alter any result: an unused branch, a symmetric expression,
   a comparison at a value that is never hit) or lines the suite does not reach (a gap); each survivor is classified by hand after the
   run, with the reason, and the two kinds are reported separately.
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fault_injection as FI   # noqa: E402  (detectors, build, git helpers)

OUT = FI.ROOT / "results" / "validation" / "mutation_study"
FILES = ["cpp/src/Schemes.cpp", "cpp/include/fv/Schemes.hpp", "cpp/src/Physics.cpp", "cpp/src/Boundaries.cpp", "cpp/src/CellFlux.cpp", "cpp/src/CellFluxResidual.cpp",
         "cpp/src/TimeScheme.cpp", "cpp/src/MPIDecomposition.cpp", "cpp/src/CellParallel.cpp", "cpp/src/CellAdvance.cpp",
         "cpp/src/CellRemesh.cpp", "cpp/src/Core.cpp"]
SKIP = re.compile(r'^\s*(#|//|template|static_assert|assert|throw|return\s*;)|"|std::cout|std::cerr|printf|getenv|FENICE_|\bfor\s*\(|\bwhile\s*\(')
OPS = [(" + ", " - "), (" - ", " + "), (" * ", " / "), (" / ", " * "), (" < ", " <= "), (" <= ", " < "), (" > ", " >= "),
       (" >= ", " > "), (" && ", " || "), (" || ", " && ")]


def read(path):
    raw = (FI.ROOT / path).read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n"), "\r\n" in raw


def sites(path):
    """Every (line number, operator, occurrence) of a mutable operator in the code part of a line."""
    text, _ = read(path)
    found = []
    for no, line in enumerate(text.split("\n"), 1):
        code = line.split("//")[0]
        if SKIP.search(code) or not code.strip():
            continue
        for a, b in OPS:
            for m in re.finditer(re.escape(a), code):
                # spaced binary operators only; skip pointer / reference declarations and template brackets
                before = code[:m.start()].rstrip()
                if not before or before[-1] in "(,=<>&|*":
                    continue
                if a.strip() in ("<", ">") and re.search(r"[A-Za-z_]<|>\s*[A-Za-z_:]+\s*[({]", code):
                    continue
                if a == " * " and re.search(r"\b(const|double|int|auto|Field|State|bool|size_t|char)\s*\*", code):
                    continue
                found.append((no, a, b, m.start()))
    return found


def mutate_line(line, a, b, pos):
    return line[:pos] + b + line[pos + len(a):]


def draw(n, seed):
    pool = [(f, s) for f in FILES for s in sites(f)]
    random.Random(seed).shuffle(pool)
    return pool


def apply_mutant(m):
    text, crlf = read(m["file"])
    lines = text.split("\n")
    old = lines[m["line"] - 1]
    assert old == m["old"], (m["id"], "the source changed")
    lines[m["line"] - 1] = m["new"]
    out = "\n".join(lines)
    (FI.ROOT / m["file"]).write_bytes((out.replace("\n", "\r\n") if crlf else out).encode("utf-8"))


def restore(m):
    FI.git("checkout", "--", m["file"])


def kill_strays():
    """A check that times out kills its python child but not the fenice it started, which then locks the executable (every later
    build fails at the link step). Kill the solver processes that run from this checkout's build folder."""
    if os.name != "nt":
        return
    import subprocess
    root = str(FI.BUILD).replace("'", "''")
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith('" + root + "') } | "
          "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True)


def stats(results):
    done = {k: v for k, v in results.items() if "detectors" in v}
    keys = FI.KEYS
    lines = []
    n = len(done)
    killed = {k: sum(v["detectors"][k]["caught"] for v in done.values()) for k in keys}
    any_k = sum(any(d["caught"] for d in v["detectors"].values()) for v in done.values())
    tol = sum(any(v["detectors"][k]["caught"] for k in ("reg", "reg4", "mms")) for v in done.values())
    bit = sum(any(v["detectors"][k]["caught"] for k in ("unit", "amr", "ibm")) for v in done.values())
    only_bit = sum((not any(v["detectors"][k]["caught"] for k in ("reg", "reg4", "mms"))) and any(v["detectors"][k]["caught"] for k in ("unit", "amr", "ibm")) for v in done.values())
    only_tol = sum(any(v["detectors"][k]["caught"] for k in ("reg", "reg4", "mms")) and not any(v["detectors"][k]["caught"] for k in ("unit", "amr", "ibm")) for v in done.values())
    lines.append(f"mutants built and checked: {n}")
    lines.append("killed by each check: " + ", ".join(f"{k} {killed[k]}" for k in keys))
    lines.append(f"killed by any check: {any_k} ({100 * any_k / max(n, 1):.0f} %); by a tolerance check: {tol}; by a bitwise check: {bit}; "
                 f"only by a tolerance check: {only_tol}; only by a bitwise check: {only_bit}")
    lines.append("survivors: " + ", ".join(k for k, v in done.items() if not any(d["caught"] for d in v["detectors"].values())))
    return "\n".join(lines)


def table(results):
    rows = ["| mutant | file:line | change | " + " | ".join(d[1] for d in FI.DETECTORS) + " |", "|---|---|---|" + "---|" * len(FI.DETECTORS)]
    for k, v in results.items():
        if "detectors" not in v:
            continue
        ch = f"`{v['a'].strip()}` -> `{v['b'].strip()}`"
        rows.append(f"| {k} | {v['file'].split('/')[-1]}:{v['line']} | {ch} | " + " | ".join(
            "**caught**" if v["detectors"][d[0]]["caught"] else "passes" for d in FI.DETECTORS) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261106)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    resf = OUT / "results.json"
    results = json.loads(resf.read_text(encoding="utf-8")) if (a.resume or a.report) and resf.exists() else {}
    if a.report:
        print(stats(results))
        (OUT / "table.md").write_text(table(results) + "\n", encoding="utf-8")
        return 0
    pool = draw(a.n, a.seed)
    if a.list:
        for i, (f, (no, x, y, pos)) in enumerate(pool[:a.n * 2]):
            print(f"M{i + 1:03d} {f}:{no} '{x.strip()}' -> '{y.strip()}'   {read(f)[0].split(chr(10))[no - 1].strip()[:90]}")
        print(f"{len(pool)} mutation sites in {len(FILES)} files")
        return 0
    if FI.git("status", "--porcelain", "--", "cpp", "CMakeLists.txt").stdout.strip():
        print("cpp/ has uncommitted changes: use a clean worktree")
        return 2
    env = dict(os.environ, FENICE_EXE=str(FI.BUILD / ("fenice" + FI.EXT)))
    log = open(OUT / "log.txt", "a" if a.resume else "w", encoding="utf-8")
    ok, out = FI.build()
    if not ok:
        print(out[-3000:])
        return 2
    if not a.resume:
        base = FI.detect(env, log)
        if any(v["caught"] for v in base.values()):
            print("the unmodified solver fails a check: no study possible")
            return 2
    taken, compiled, rejected = 0, len([1 for v in results.values() if "detectors" in v]), 0
    try:
        for i, (f, (no, x, y, pos)) in enumerate(pool):
            if compiled >= a.n:
                break
            mid = f"M{i + 1:03d}"
            if mid in results:
                continue
            text, _ = read(f)
            old = text.split("\n")[no - 1]
            m = dict(id=mid, file=f, line=no, old=old, new=mutate_line(old, x, y, pos), a=x, b=y)
            print(f"{mid} {f}:{no}  {x.strip()} -> {y.strip()}   {old.strip()[:80]}", flush=True)
            apply_mutant(m)
            ok, out = FI.build()
            if not ok:
                restore(m)
                results[mid] = dict(error="does not compile", file=f, line=no, a=x, b=y)
                log.write(f"--- {mid} does not compile\n")
                resf.write_text(json.dumps(results, indent=1), encoding="utf-8")
                continue
            res = FI.detect(env, log)
            kill_strays()
            restore(m)
            results[mid] = dict(detectors=res, file=f, line=no, a=x, b=y, old=old.strip(), new=m["new"].strip())
            compiled += 1
            resf.write_text(json.dumps(results, indent=1), encoding="utf-8")
    finally:
        for f in FILES:
            FI.git("checkout", "--", f)
        FI.build()
    print(stats(results))
    (OUT / "table.md").write_text(table(results) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
