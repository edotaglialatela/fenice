"""Helpers shared by the validation scripts: one folder per study under
results/validation/<study>/, cached runs (re-run only with --force), one figure style, and
figures saved there and, with --docs, copied into docs/img/."""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

from .budget import run_budgeted
from .solver import RESULTS, ROOT, run, study_threads

PARTIAL = 75   # exit code of a study stopped by its budget (the next call continues it); EX_TEMPFAIL
_deadline = None

DOCS_IMG = ROOT / "docs" / "img"


class _StudyParser(argparse.ArgumentParser):
    """Carries --budget and --threads into the environment variables the runner reads, once the arguments are parsed."""

    def parse_args(self, args=None, namespace=None):
        ns = super().parse_args(args, namespace)
        if ns.budget is not None:
            os.environ["FENICE_STUDY_BUDGET"] = str(ns.budget)
        if ns.threads is not None:
            os.environ["FENICE_STUDY_THREADS"] = str(ns.threads)
        return ns


def parser(description):
    """argparse parser with the options every study has: --docs, --force, --mpi, --budget, --threads."""
    ap = _StudyParser(description=description)
    ap.add_argument("--docs", action="store_true", help="also copy the figures into docs/img/")
    ap.add_argument("--force", action="store_true", help="re-run the simulations even if their results exist")
    ap.add_argument("--mpi", type=int, default=0, metavar="N", help="run the solver on N MPI ranks (default: OpenMP)")
    ap.add_argument("--budget", type=float, default=None, metavar="MINUTES",
                    help="wall-time budget of the simulations: stop at the next checkpoint (exit 75), the next call "
                         "continues ($FENICE_STUDY_BUDGET)")
    ap.add_argument("--threads", type=int, default=None, metavar="N",
                    help="OpenMP threads of a study run (default 2, $FENICE_STUDY_THREADS)")
    return ap


def study_dir(study):
    """results/validation/<study>/ (created)."""
    d = RESULTS / "validation" / study
    d.mkdir(parents=True, exist_ok=True)
    return d


def budget_left():
    """Minutes left of the study's wall-time budget ($FENICE_STUDY_BUDGET, counted from the first run), or None."""
    global _deadline
    env = os.environ.get("FENICE_STUDY_BUDGET")
    if not env:
        return None
    if _deadline is None:
        _deadline = time.time() + float(env) * 60.0
    return max(0.0, (_deadline - time.time()) / 60.0)


def run_cached(study, tag, case, *sets, mpi=0, threads=None, force=False, done="final.npz", chunk_steps=200):
    """Run `case` with the `--set` overrides into results/validation/<study>/<tag>/, unless
    that run already finished and force is False: a file matching the glob `done` (default:
    the final state final.npz) exists in the run folder.
    The solver runs at below-normal priority on `threads` OpenMP threads ($FENICE_STUDY_THREADS, default 2).
    With a budget ($FENICE_STUDY_BUDGET, minutes) the run goes in chunks of `chunk_steps` steps and, when the budget is
    spent, the process exits with code PARTIAL: the next call resumes from the checkpoint.
    Returns the run folder: final.npz plus the case's output folder (snapshots, monitor)."""
    d = study_dir(study) / tag
    dump = d / "final.npz"
    if force or not any(d.glob(done)):
        if force and d.exists():   # a forced run starts over, not from an old checkpoint
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {study}/{tag}: {case} {' '.join(sets)}", flush=True)
        if not (mpi and mpi > 1):   # an MPI rank keeps its usual single thread
            threads = threads or study_threads()
        left = budget_left()
        resumable = any(d.glob("*/*_restart.npz")) and not force   # a partial run, stopped by an earlier budget
        if left is None and not resumable:
            run(case, *sets, out_root=d, mpi=mpi, dump=dump, threads=threads, polite=True)
        else:
            res = run_budgeted(case, *sets, out_root=d, budget_minutes=left, chunk_steps=chunk_steps, dump=dump,
                               threads=threads, mpi=mpi)
            if res["state"] == "partial":
                print(f"[PARTIAL] {study}/{tag}: budget spent after {res['steps']} steps; call again to continue",
                      flush=True)
                sys.exit(PARTIAL)
    return d


def case_output(run_dir):
    """The case's output folder inside a run folder (the one holding the .pvd)."""
    return next(p.parent for p in Path(run_dir).glob("*/*.pvd"))


def use_style():
    """The figure style of the documentation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "axes.labelsize": 12,
                         "axes.titlesize": 13, "legend.fontsize": 9, "figure.dpi": 100,
                         "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.5})
    return plt


def save(fig, study, name, docs=False, dpi=170):
    """Save `fig` as results/validation/<study>/<name>; with docs, also docs/img/<name>."""
    out = study_dir(study) / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[PLOT] {out}")
    if docs:
        shutil.copyfile(out, DOCS_IMG / name)
        print(f"[DOCS] docs/img/{name}")
    return out


def copy_to_docs(path, name=None):
    """Copy an existing file (e.g. an animation) into docs/img/."""
    name = name or Path(path).name
    shutil.copyfile(path, DOCS_IMG / name)
    print(f"[DOCS] docs/img/{name}")


def write_case(study, name, text):
    """Write a case file generated by a study to results/validation/<study>/cases/<name>/case.toml and return the
    folder (run it with run_cached(..., case=folder)). Variants of one initial condition need no folder under cases/."""
    d = study_dir(study) / "cases" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "case.toml").write_text(text)
    return d
