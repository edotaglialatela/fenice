"""Bring the results of fenice_workstation.sh into the repository: import, replot the documentation figures, collect the numbers.

    python cases/validation/workstation_postprocess.py SRC            # SRC: the workstation's results/validation folder, its src/ or
                                                                      #      results/ parent, or a .tar / .tar.gz / .tgz / .zip of any of them
    python cases/validation/workstation_postprocess.py SRC --dry-run  # only say what would be imported and replotted
    python cases/validation/workstation_postprocess.py SRC --only mms_weno blast_waves_weno
    python cases/validation/workstation_postprocess.py --no-import    # results already in results/validation: only replot and report

What it does, in order:

1. Import. Every study folder of SRC replaces the folder of the same name in results/validation/ (the study cache is keyed by case
   name, not by settings, so merging an old laptop run with a workstation one would mix them; use --merge to merge anyway). The cases,
   dumps and monitors come along, so the studies below replot without a single simulation.
2. Replot. Each study script of the list STUDIES below whose results are present is run with --docs under a time limit (--timeout,
   default 20 min: a study that would have to simulate is stopped, not left to run for hours). The figures go to results/ and docs/img/;
   the output of each script (its tables and band checks) is kept in results/validation/workstation_report/<study>.txt.
3. Report. results/validation/workstation_report/report.md gathers the Trixi.jl comparison (table, figure copied to docs/img/
   compare_trixi.png), variants.md, recent.md, the status of every study (band met / missed / not replotted), the images of docs/img
   and of the README that changed, and the docs page each table most likely belongs to. The tables are NOT written into the pages:
   the pages are prose around the numbers, and a missed band or a changed order is something to read before it is pasted.

Nothing is committed or pushed. Review `git status` and `git diff --stat docs README.md`, then commit.
"""
import argparse
import datetime
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import fenice
from fenice import study

VAL = fenice.RESULTS / "validation"
REPORT = VAL / "workstation_report"
ROOT = study.ROOT
HERE = Path(__file__).resolve().parent

# name: (script relative to this folder, arguments, result folders that must be there, docs page the numbers belong to)
STUDIES = {
    "blast_waves_weno": ("verification/blast_waves_weno.py", [], ["blast_waves_weno"], "cases/blast-waves.md"),
    "mms_weno": ("verification/mms_weno.py", ["--full"], ["mms_weno"], "cases/manufactured-solution.md"),
    "mms_time_weno": ("verification/mms_time_weno.py", ["--full"], ["mms_time_weno"], "cases/manufactured-solution.md"),
    "mms_amr_weno": ("verification/mms_amr_weno.py", ["--full"], ["mms_amr_weno"], "cases/manufactured-solution.md"),
    "mms_time": ("verification/mms_time.py", [], ["mms_time"], "cases/manufactured-solution.md"),
    "cavity": ("benchmarks/cavity.py", [], ["cavity"], "cases/cavity.md"),
    "gci": ("benchmarks/gci.py", [], ["gci"], "guide/testing.md (V&V section)"),
    "vv20": ("benchmarks/vv20.py", [], ["vv20"], "guide/testing.md (V&V section)"),
    "ibm_loads_cylinder": ("benchmarks/ibm_loads.py", ["cylinder"], ["cylinder_loads"], "cases/cylinder.md"),
    "ibm_loads_airfoil": ("benchmarks/ibm_loads.py", ["airfoil"], ["airfoil_loads"], "cases/airfoil.md"),
    "cylinder_low_blockage": ("benchmarks/cylinder_low_blockage.py", [], ["cylinder_low_blockage"], "cases/cylinder-low-blockage.md"),
    "shu_osher_schemes": ("benchmarks/shu_osher_schemes.py", [], ["shu_osher_schemes"], "cases/shu-osher.md"),
    "taylor_green_3d": ("benchmarks/taylor_green_3d.py", [], ["taylor_green_3d"], "cases/taylor-green-3d.md and README.md"),
    "spectra_application": ("benchmarks/spectra_application.py", [], ["spectra_application"], "cases/taylor-green-3d.md"),
}
# tables written by the workstation's own helpers (not scripts of the repository: nothing to replot, only to read)
TABLES = {
    "compare_trixi/table.md": "no page yet: a comparison section (Trixi.jl DG degree 3 against MUSCL / WENO3-Z / WENO5-Z)",
    "variants/variants.md": "cases/sod.md, cases/shu-osher.md, cases/cavity.md (MUSCL against WENO, boundary_order 1 / 2)",
    "recent/recent.md": "cases/manufactured-solution.md (MUSCL at the outflow), cases/cylinder.md (probe loads), "
                        "cases/cylinder-low-blockage.md (80 cells per diameter)",
}


def resolve_source(src, tmp):
    """The folder that holds the study folders: SRC itself, or SRC/results/validation, SRC/src/results/validation, ..."""
    p = Path(src).expanduser()
    if p.is_file():
        out = Path(tmp) / "unpacked"
        if p.suffix == ".zip":
            zipfile.ZipFile(p).extractall(out)
        else:
            with tarfile.open(p) as t:
                t.extractall(out)
        p = out
    for cand in (p, p / "validation", p / "results" / "validation", p / "src" / "results" / "validation"):
        if cand.is_dir() and any((cand / n).is_dir() for n in ("compare_trixi", "variants", "recent", "mms_weno", "blast_waves_weno",
                                                               "cavity", "mms_time")):
            return cand
    for h in sorted(p.rglob("validation")):
        if h.is_dir() and any(h.iterdir()):
            return h
    raise SystemExit(f"no study folders (results/validation) found under {src}")


def import_results(src, merge, dry):
    done = []
    for d in sorted(x for x in src.iterdir() if x.is_dir()):
        if d.name == "workstation_report":
            continue
        dst = VAL / d.name
        action = "merge into" if (merge and dst.exists()) else ("replace" if dst.exists() else "new")
        print(f"  [{action:10s}] {d.name}")
        done.append(d.name)
        if dry:
            continue
        if dst.exists() and not merge:
            shutil.rmtree(dst)
        shutil.copytree(d, dst, dirs_exist_ok=True)
    return done


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def replot(name, timeout_min, docs):
    script, args, _, _ = STUDIES[name]
    cmd = [sys.executable, str(HERE / script), *args, *(["--docs"] if docs else [])]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_min * 60, encoding="utf-8", errors="replace")
        status = {0: "ok", 1: "band missed or failed", study.PARTIAL: "stopped by a budget (not all runs present)"}.get(
            p.returncode, f"exit code {p.returncode}")
        out = p.stdout + ("\n--- stderr\n" + p.stderr if p.stderr.strip() else "")
    except subprocess.TimeoutExpired as e:
        status = f"STOPPED after {timeout_min} min: runs are missing from the imported results"
        out = (e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")) + "\n(timeout)"
    (REPORT / f"{name}.txt").write_text(out, encoding="utf-8")
    return status, out


def changed_images():
    """Files of docs/img changed since the last commit, and which of them the README shows."""
    porcelain = git("status", "--porcelain", "--", "docs/img", "README.md", "docs/source")
    imgs = sorted(line[3:].strip().replace("docs/img/", "") for line in porcelain.splitlines() if "docs/img/" in line)
    readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace")
    used = set(re.findall(r"docs/img/([\w.\-]+)", readme))
    return imgs, [i for i in imgs if i in used]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?", help="workstation results (folder or archive)")
    ap.add_argument("--no-import", action="store_true", help="results are already in results/validation")
    ap.add_argument("--merge", action="store_true", help="merge into existing study folders instead of replacing them")
    ap.add_argument("--only", nargs="+", choices=list(STUDIES), help="replot only these studies")
    ap.add_argument("--no-docs", action="store_true", help="replot into results/ only, leave docs/img untouched")
    ap.add_argument("--timeout", type=float, default=20.0, metavar="MIN", help="time limit of each replot (default 20)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.no_import and not a.src:
        ap.error("give the workstation results (SRC) or --no-import")
    REPORT.mkdir(parents=True, exist_ok=True)
    imported, status = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        if not a.no_import:
            src = resolve_source(a.src, tmp)
            print(f"[import] {src} -> {VAL}")
            imported = import_results(src, a.merge, a.dry_run)
        names = list(a.only or STUDIES)
        ready = [n for n in names if all((VAL / d).is_dir() and any((VAL / d).iterdir()) for d in STUDIES[n][2])]
        skipped = [n for n in names if n not in ready]
        print(f"[replot] {len(ready)} studies: {', '.join(ready) or '-'}")
        if skipped:
            print(f"[skip]   no results for: {', '.join(skipped)}")
        if a.dry_run:
            return 0
        for n in ready:
            print(f"[run] {n} ...", flush=True)
            status[n], _ = replot(n, a.timeout, not a.no_docs)
            print(f"      {status[n]}", flush=True)
    trixi_png = VAL / "compare_trixi" / "compare_trixi.png"
    if trixi_png.exists() and not a.no_docs:   # the figure goes to the docs even though no page uses it yet
        shutil.copyfile(trixi_png, study.DOCS_IMG / "compare_trixi.png")
    imgs, in_readme = changed_images()
    lines = [f"# Workstation results, processed {datetime.datetime.now():%Y-%m-%d %H:%M}", "",
             f"Repository {git('rev-parse', '--short', 'HEAD')} ({git('log', '-1', '--format=%s')[:90]}). "
             f"Source: {a.src or 'results/validation (no import)'}. Imported: {', '.join(imported) or '-'}.", "",
             "## Studies replotted", "", "| study | result | docs page for the numbers |", "|---|---|---|"]
    lines += [f"| {n} | {status.get(n, 'not replotted: no results in results/validation')} | {STUDIES[n][3]} |" for n in names]
    lines += ["", "Each study's own output (tables, band checks) is in `results/validation/workstation_report/<study>.txt`.", ""]
    lines += ["## Changed images", "", f"docs/img: {', '.join(imgs) or 'none'}", "",
              f"shown by the README (check them): {', '.join(in_readme) or 'none'}", ""]
    lines += ["## Tables of the workstation helpers", ""]
    for rel, page in TABLES.items():
        f = VAL / rel
        lines += [f"### {rel}  (belongs in: {page})", "", f.read_text(encoding="utf-8") if f.exists() else "_not in the results_", ""]
    (REPORT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {REPORT / 'report.md'}")
    print("changed docs/img:", ", ".join(imgs) or "none", "| shown by the README:", ", ".join(in_readme) or "none")
    print("next: read the report, edit the pages (numbers, the sentences about orders and bands), git diff --stat, commit.")
    return 1 if any(s != "ok" for s in status.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
