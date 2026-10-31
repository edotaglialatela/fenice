"""Check of the study machinery: tiers, cost estimate, budgeted resumable runs, queue ordering, summary file.

    python cases/validation/check_budget.py

1. a simulated slow solver is stopped by the budget after a chunk and finished by the next call (steps add up);
2. a real run (Sod, 400 cells) stopped after one chunk and resumed ends with the same final state, bit for bit,
   as an uninterrupted run;
3. the queue lists studies in cost order and skips a current one;
4. estimate() / write_summary() round trip.
Exit code 0 when everything holds.
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import fenice
from fenice import budget

sys.path.insert(0, str(Path(__file__).resolve().parent))
import study_queue

failures = []


def check(ok, what):
    print(f"[{' OK ' if ok else 'FAIL'}] {what}")
    if not ok:
        failures.append(what)


def fake_runner(total_steps, delay):
    """A solver that needs `total_steps` steps in all and takes `delay` seconds per call; progress lives in a file."""
    progress = {"done": 0}

    def runner(case, *sets, out_root=None, **kw):
        want = int(next(s.split("=")[1] for s in sets if s.startswith("max_steps=")))
        time.sleep(delay)
        n = min(want, total_steps - progress["done"])
        progress["done"] += n
        return SimpleNamespace(stdout=f"=== DONE (time loop 0.1 s, {n} steps)\n")
    return runner, progress


def test_fake():
    runner, progress = fake_runner(total_steps=50, delay=0.05)
    with tempfile.TemporaryDirectory() as tmp:
        r1 = budget.run_budgeted("x", out_root=tmp, budget_minutes=0.0, chunk_steps=20, runner=runner)
        check(r1["state"] == "partial" and r1["steps"] == 20, "fake solver: budget 0 stops after one chunk (partial, 20 steps)")
        r2 = budget.run_budgeted("x", out_root=tmp, budget_minutes=None, chunk_steps=20, runner=runner)
        check(r2["state"] == "done" and progress["done"] == 50 and r1["steps"] + r2["steps"] == 50,
              "fake solver: the next call finishes the run (20 + 30 steps)")


def test_real_resume():
    sets = ("nx=100", "ny=4", "t_end=0.03", "save_interval=10")
    tmp = Path(tempfile.mkdtemp(prefix="fenice_budget_"))
    try:
        a, b = tmp / "straight", tmp / "chunked"
        a.mkdir(); b.mkdir()
        ra = budget.run_budgeted("Sod Shock Tube", *sets, out_root=a, chunk_steps=10_000, dump=a / "final.npz")
        rb = budget.run_budgeted("Sod Shock Tube", *sets, out_root=b, budget_minutes=0.0, chunk_steps=7,
                                 dump=b / "final.npz")
        check(rb["state"] == "partial" and not (b / "final.npz").exists(), "real run: stopped after one chunk, no final.npz yet")
        rc = budget.run_budgeted("Sod Shock Tube", *sets, out_root=b, chunk_steps=7, dump=b / "final.npz")
        check(rc["state"] == "done" and rb["steps"] + rc["steps"] == ra["steps"],
              f"real run: resumed run takes the remaining steps ({rb['steps']} + {rc['steps']} = {ra['steps']})")
        U1 = fenice.read_dump(a / "final.npz")
        U2 = fenice.read_dump(b / "final.npz")
        U1 = U1["U"] if isinstance(U1, dict) else U1
        U2 = U2["U"] if isinstance(U2, dict) else U2
        check(np.array_equal(np.asarray(U1), np.asarray(U2)), "real run: chunked == uninterrupted, bit for bit")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_queue():
    order = study_queue.order(list(study_queue.STUDIES))
    tiers = [study_queue.STUDIES[n][2] for n in order]
    check(tiers == sorted(tiers, key=budget.tier_rank), "queue: studies come in tier order")
    state = {"sod": {"state": "done", "fingerprint": "same"}, "mms": {"state": "partial", "fingerprint": "same"}}
    todo = study_queue.pending(state, "quick", fp=lambda script: "same")
    check("sod" not in todo and "mms" in todo, "queue: a current study is skipped, a partial one is not")
    todo = study_queue.pending(state, "quick", fp=lambda script: "changed")
    check("sod" in todo, "queue: a changed fingerprint makes the study pending again")
    check("mms_full" not in study_queue.pending({}, "quick"), "queue: quick selection leaves long studies out")


def test_misc():
    e = budget.estimate(cells=400, steps=200)
    check(e["tier"] == "quick" and abs(e["core_hours"] - 400 * 200 / 1e6 / 3600) < 1e-15, "estimate: small run is quick")
    check(budget.estimate(96 ** 3, 4000, stages=2)["tier"] in ("medium", "long"), "estimate: 96^3 is not quick")
    with tempfile.TemporaryDirectory() as tmp:
        p = fenice.write_summary("t", {"passed": True, "x": np.float64(1.5)}, folder=tmp)
        d = json.loads(Path(p).read_text())
        check(d["study"] == "t" and d["passed"] is True and d["x"] == 1.5, "summary.json round trip")


if __name__ == "__main__":
    test_fake()
    test_real_resume()
    test_queue()
    test_misc()
    print("ALL PASSED" if not failures else f"FAILED: {failures}")
    sys.exit(1 if failures else 0)
