"""Helpers shared by the validation scripts: one folder per study under
results/validation/<study>/, cached runs (re-run only with --force), one figure style, and
figures saved there and, with --docs, copied into docs/img/."""
import argparse
import shutil
from pathlib import Path

from .solver import RESULTS, ROOT, run

DOCS_IMG = ROOT / "docs" / "img"


def parser(description):
    """argparse parser with the options every study has: --docs, --force, --mpi."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--docs", action="store_true", help="also copy the figures into docs/img/")
    ap.add_argument("--force", action="store_true", help="re-run the simulations even if their results exist")
    ap.add_argument("--mpi", type=int, default=0, metavar="N", help="run the solver on N MPI ranks (default: OpenMP)")
    return ap


def study_dir(study):
    """results/validation/<study>/ (created)."""
    d = RESULTS / "validation" / study
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_cached(study, tag, case, *sets, mpi=0, threads=None, force=False, done="final.npz"):
    """Run `case` with the `--set` overrides into results/validation/<study>/<tag>/, unless
    that run already finished and force is False: a file matching the glob `done` (default:
    the final state final.npz) exists in the run folder.
    Returns the run folder: final.npz plus the case's output folder (snapshots, monitor)."""
    d = study_dir(study) / tag
    dump = d / "final.npz"
    if force or not any(d.glob(done)):
        d.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {study}/{tag}: {case} {' '.join(sets)}", flush=True)
        run(case, *sets, out_root=d, mpi=mpi, dump=dump, threads=threads)
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
