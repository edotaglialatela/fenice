"""Cost tiers, cost estimates and time-budgeted, resumable runs of the studies.

A study declares one tier: `quick` (under 2 min, one core), `medium` (under 30 min, 4 cores) or `long` (hours or MPI).
`run_budgeted` runs a case in chunks of `chunk_steps` steps with `resume`, and stops cleanly (at a checkpoint) once the
wall-time budget is spent; the next call continues from the checkpoint, bit for bit. A half-finished study costs disk only.
"""
import re
import time
from pathlib import Path

from .solver import run, study_threads

TIERS = ("quick", "medium", "long")

# Assumed cost of one cell update (one explicit residual evaluation, 2-D Euler, one core), NOT measured: a deliberately rough constant;
# a study that needs a better one passes its own to `estimate`.
DEFAULT_CELLS_PER_S = 1.0e6


def tier_rank(tier):
    """Position of a tier name in the order quick < medium < long (ValueError for an unknown name)."""
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {TIERS}")
    return TIERS.index(tier)


def estimate(cells, steps, cells_per_s=DEFAULT_CELLS_PER_S, stages=1):
    """Estimated cost of a run: {"cell_updates", "core_hours", "tier"}.

    cells = nx*ny*nz (an AMR run: the expected leaf count); steps = the step count (from the CFL step and t_end);
    stages = residual evaluations per step (Euler 1, SSP-RK2 2, RK4 4). The tier is the cheapest one that the estimate
    fits (quick: under 2 core-minutes, medium: under 2 core-hours = 30 min on 4 cores, else long)."""
    updates = float(cells) * float(steps) * stages
    hours = updates / cells_per_s / 3600.0
    tier = "quick" if hours <= 2.0 / 60.0 else "medium" if hours <= 2.0 else "long"
    return {"cell_updates": updates, "core_hours": hours, "tier": tier}


def steps_for(t_end, dx, smax, cfl):
    """Step count of a run to t_end with the convective step dt = cfl dx / smax (smax: largest wave speed)."""
    return int(t_end * smax / (cfl * dx)) + 1


_LOOP = re.compile(r"time loop ([0-9.]+) s, (\d+) steps")


def run_budgeted(case, *sets, out_root, budget_minutes=None, chunk_steps=200, dump=None, threads=None, mpi=0,
                 state_file=None, runner=run):
    """Run `case` to its end time in chunks, within a wall-time budget.

    Each chunk is `chunk_steps` steps (rounded up to a multiple of a `save_interval=` override, so that the snapshots stay on their schedule) (`max_steps`, counted from the resume); the first starts fresh unless a checkpoint
    exists in out_root (then it resumes), the others resume. The loop ends when a chunk does fewer steps than asked (the
    case reached t_end: the final state goes to `dump`) or when the budget is spent after a chunk: then it returns with
    state "partial". `runner` replaces `fenice.run` (tests use it to simulate a slow solver).

    Returns {"state": "done" | "partial", "steps": steps run by THIS call, "minutes": wall minutes used}.
    state_file (optional): the step count so far is written there after every chunk."""
    t0 = time.time()
    # a run writes a snapshot where it stops: chunks that end on the snapshot schedule keep the series uniform
    for item in sets:
        if str(item).startswith("save_interval="):
            si = int(float(str(item).split("=")[1]))
            if 0 < si < 10 ** 8 and chunk_steps % si:
                chunk_steps = (chunk_steps // si + 1) * si
    if not (mpi and mpi > 1):
        threads = threads or study_threads()
    out_root = Path(out_root)
    resume = any(out_root.glob("*/*_restart.npz"))
    partial = None if dump is None else Path(dump).with_name(Path(dump).stem + "_partial.npz")
    total = 0
    while True:
        sets_now = list(sets) + [f"max_steps={chunk_steps}", f"resume={'true' if resume else 'false'}"]
        r = runner(case, *sets_now, out_root=out_root, threads=threads, mpi=mpi, polite=True,
                   dump=partial)
        m = _LOOP.search(r.stdout)
        if m is None:
            raise RuntimeError("no 'time loop ... steps' line in the solver output (did the run finish?)")
        n = int(m.group(2))
        total += n
        resume = True
        if state_file is not None:
            Path(state_file).write_text(f"{total}\n")
        if n < chunk_steps:
            if partial is not None:
                partial.replace(dump)   # final.npz exists only for a finished run
            return {"state": "done", "steps": total, "minutes": (time.time() - t0) / 60.0}
        if budget_minutes is not None and (time.time() - t0) / 60.0 >= budget_minutes:
            return {"state": "partial", "steps": total, "minutes": (time.time() - t0) / 60.0}
