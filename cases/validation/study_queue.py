"""Run the pending validation studies in cost order, within a wall-time budget; safe to start at night and to kill at any time.

    python cases/validation/study_queue.py                       # the quick studies that are not current
    python cases/validation/study_queue.py --tier medium --budget 120   # up to medium, 120 minutes in total
    python cases/validation/study_queue.py --status              # print results/validation/queue.json and exit
    python cases/validation/study_queue.py --dry-run             # what would run, in which order

A study is "current" when it finished (exit code 0) with the same fingerprint: the hash of its script, of every case file
under cases/ and of the solver executable. Current studies are skipped. A study that reaches the budget stops at its next
checkpoint (exit code 75, see fenice.study) and is recorded "partial": the next call resumes it from the checkpoint. A
failing study is recorded "failed" and the queue goes on. The solver runs at below-normal priority on --threads threads.
results/validation/queue.json holds, per study: state (done / partial / failed / pending), minutes used, fingerprint.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import fenice
from run_all import PARTIAL, STUDIES, TIERS

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
QUEUE = REPO / "results" / "validation" / "queue.json"


def fingerprint(script):
    """Hash of the study script, all case files and the solver executable."""
    h = hashlib.sha1()
    for p in [HERE / script] + sorted((REPO / "cases").glob("*/case.toml")):
        h.update(p.name.encode())
        h.update(Path(p).read_bytes())
    h.update(Path(fenice.exe()).read_bytes())
    return h.hexdigest()[:16]


def load():
    try:
        return json.loads(QUEUE.read_text())
    except (OSError, ValueError):
        return {}


def save(state):
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(QUEUE)


def order(names):
    """Cost order: tier, then estimated core-hours (unknown last), then name."""
    def key(n):
        _, _, tier, est = STUDIES[n]
        return (TIERS.index(tier), float("inf") if est is None else est, n)
    return sorted(names, key=key)


def pending(state, tier, fp=fingerprint):
    """The studies up to `tier` that are not current, in cost order."""
    top = TIERS.index(tier)
    out = []
    for n in order(STUDIES):
        script = STUDIES[n][1][0]
        if TIERS.index(STUDIES[n][2]) > top:
            continue
        rec = state.get(n, {})
        if rec.get("state") == "done" and rec.get("fingerprint") == fp(script):
            continue
        out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", choices=TIERS, default="quick", help="studies up to this tier (default: quick)")
    ap.add_argument("--budget", type=float, default=None, metavar="MIN", help="total wall minutes; none: no limit")
    ap.add_argument("--threads", type=int, default=2, help="OpenMP threads of the solver (default 2)")
    ap.add_argument("--status", action="store_true", help="print the queue file and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the studies that would run and exit")
    args = ap.parse_args()
    state = load()
    if args.status:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0
    todo = pending(state, args.tier)
    for n in todo:
        print(f"  {'next' if args.dry_run else 'queued'}: {n:26s} {STUDIES[n][2]}")
    if args.dry_run or not todo:
        return 0
    t_start = time.time()
    failed = False
    for n in todo:
        left = None if args.budget is None else args.budget - (time.time() - t_start) / 60.0
        if left is not None and left <= 0:
            print("budget spent; the rest stays pending")
            break
        group, (script, *rest), _, _ = STUDIES[n]
        env = dict(os.environ, FENICE_STUDY_THREADS=str(args.threads))
        env.pop("FENICE_STUDY_BUDGET", None)
        if left is not None:
            env["FENICE_STUDY_BUDGET"] = f"{left:.3f}"
        print(f"=== {group}/{n}", flush=True)
        t0 = time.time()
        code = subprocess.run([sys.executable, str(HERE / script), *rest], env=env).returncode
        minutes = (time.time() - t0) / 60.0
        rec = state.get(n, {})
        rec["minutes"] = rec.get("minutes", 0.0) + minutes if rec.get("state") == "partial" else minutes
        rec["state"] = "done" if code == 0 else "partial" if code == PARTIAL else "failed"
        rec["fingerprint"] = fingerprint(script)
        state[n] = rec
        save(state)
        failed |= rec["state"] == "failed"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
