"""Every check a change to the solver must pass, in one command, judged by exit codes.

    python cases/validation/check_all.py                  # --quick: the steps of CI, in the order of ci.yml
    python cases/validation/check_all.py --full           # + the unit tests without SIMD, the other remesh / storage variants of the
                                                          #   AMR and ghost-cell checks
    python cases/validation/check_all.py --golden OLD     # + golden dumps of the current build compared with the dump in OLD
                                                          #   (--full: also with FENICE_NO_SIMD=1 and FENICE_FULL_REBUILD=1)
    options: --keep-going (do not stop at the first failure), --only NAME .. (substring of a step name), --list

OLD is made with `golden.py dump OLD --exe <old fenice>` before the change (see golden.py). Each step's output goes to
results/validation/check_all/<step>.log; a failing step prints the end of its log. Exit code 0 only if every step passed.
Run it with the solver built (.\\cpp\\build.ps1): it uses the fenice, fenice_tests next to it ($FENICE_EXE or build/release).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import fenice

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "results" / "validation" / "check_all"
PY = sys.executable
GOLDEN = REPO / "cases" / "validation" / "amr" / "golden.py"


def tests_exe():
    d = Path(fenice.exe()).parent
    for name in ("fenice_tests.exe", "fenice_tests"):
        if (d / name).exists():
            return str(d / name)
    sys.exit(f"fenice_tests not found next to {fenice.exe()}: build with .\\cpp\\build.ps1 first")


def steps(args):
    """(name, command, extra environment, group) in running order; group is 'quick' (the CI steps) or 'full' (the extra variants)."""
    t = tests_exe()
    s = [
        ("unit tests", [t], {}, "quick"),
        ("regression serial", [PY, "cpp/tests/regression.py"], {}, "quick"),
        ("regression 4 ranks", [PY, "cpp/tests/regression.py", "--mpi", "4"], {}, "quick"),
        ("amr_mpi", [PY, "cases/validation/amr/amr_mpi.py", "--check"], {}, "quick"),
        ("amr_mpi replicated topology", [PY, "cases/validation/amr/amr_mpi.py", "--check"], {"FENICE_REPLICATED_TOPOLOGY": "1"}, "quick"),
        ("ibm_ghost", [PY, "cases/validation/features/ibm_ghost.py", "--check"], {}, "quick"),
        ("mms quick", [PY, "cases/validation/verification/mms.py"], {}, "quick"),
        ("unit tests no simd", [t], {"FENICE_NO_SIMD": "1"}, "full"),
        ("amr_mpi replicated storage", [PY, "cases/validation/amr/amr_mpi.py", "--check"], {"FENICE_REPLICATED_REMESH": "1"}, "full"),
        ("ibm_ghost replicated topology", [PY, "cases/validation/features/ibm_ghost.py", "--check"], {"FENICE_REPLICATED_TOPOLOGY": "1"}, "full"),
    ]
    if args.golden:
        new = Path(args.golden_new).resolve()
        variants = [("", {})]
        if args.full:
            variants += [("no simd", {"FENICE_NO_SIMD": "1"}), ("full rebuild", {"FENICE_FULL_REBUILD": "1"})]
        for tag, env in variants:
            d = new / (tag.replace(" ", "_") or "default")
            dump = [PY, str(GOLDEN), "dump", str(d), "--quick"] + [x for k, v in env.items() for x in ("--env", f"{k}={v}")]
            s.append((f"golden dump {tag or 'default'}", dump, {}, "golden"))
            s.append((f"golden compare {tag or 'default'}", [PY, str(GOLDEN), "compare", args.golden, str(d)], {}, "golden"))
    return s


def run(name, cmd, env, keep):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / (name.replace(" ", "_") + ".log")
    t0 = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        r = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, **env), stdout=f, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    ok = r.returncode == 0
    print(f"[{' OK ' if ok else 'FAIL'}] {name:34s} {dt:7.1f} s" + ("" if ok else f"   exit {r.returncode}, log {log}"), flush=True)
    if not ok:
        print("".join(log.read_text(encoding="utf-8", errors="replace").splitlines(True)[-15:]), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="the steps of CI (default)")
    ap.add_argument("--full", action="store_true", help="the CI steps and the extra variants")
    ap.add_argument("--golden", metavar="OLD", help="directory of golden dumps made before the change")
    ap.add_argument("--golden-new", default=str(REPO / "results" / "validation" / "check_all" / "golden"),
                    help="where the new dumps go (keep the path short on Windows)")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--only", nargs="+", metavar="NAME")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    chosen = [x for x in steps(args) if x[3] in ("quick", "golden") or args.full]
    if args.only:
        chosen = [x for x in chosen if any(o in x[0] for o in args.only)]
        if not chosen:
            sys.exit("no step matches --only")
    if args.list:
        for name, cmd, env, group in chosen:
            print(f"{name:34s} {group:7s} {' '.join(f'{k}={v}' for k, v in env.items())} {' '.join(Path(c).name if os.sep in c else c for c in cmd)}")
        return 0
    failed = []
    for name, cmd, env, _ in chosen:
        if not run(name, cmd, env, args.keep_going):
            failed.append(name)
            if not args.keep_going:
                break
    print("ALL CHECKS PASSED" if not failed else f"FAILED: {', '.join(failed)}" + ("" if args.keep_going else "  (stopped; --keep-going runs the rest)"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
