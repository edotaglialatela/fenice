"""Run every validation study, in order of cost.

    python validation/run_all.py                  # quick studies only (minutes)
    python validation/run_all.py --all            # also the long MPI runs (cylinder, airfoil, 3-D TG)
    python validation/run_all.py --all --docs     # and refresh the figures in docs/img/
    python validation/run_all.py sod cavity       # only these studies

Each study keeps its runs in results/validation/<study>/ and re-uses them, so a second call
only redraws the figures; pass --force to re-run the simulations. Prints a summary and exits
with 1 if a study failed.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# name: (script arguments, long?)
STUDIES = {
    "sod": (["sod.py"], False),
    "sod_amr": (["sod_amr.py"], False),
    "shu_osher": (["shu_osher.py"], False),
    "isentropic_vortex": (["isentropic_vortex.py"], False),
    "taylor_green": (["taylor_green.py"], False),
    "poiseuille": (["poiseuille.py"], False),
    "cavity": (["cavity.py"], False),
    "taylor_green_3d": (["taylor_green_3d.py", "--grids", "32", "64", "96", "--vorticity-n", "96", "--mpi", "16",
                         "--q-animation"], True),   # the settings of the docs figures
    "cylinder_loads": (["ibm_loads.py", "cylinder"], True),
    "airfoil_loads": (["ibm_loads.py", "airfoil"], True),
    "airfoil_amr": (["airfoil_amr.py"], True),
    "cylinder_mesh_transition": (["cylinder_mesh_transition.py"], True),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("studies", nargs="*", choices=[[]] + list(STUDIES), metavar="STUDY",
                    help=f"studies to run (default: all quick ones): {', '.join(STUDIES)}")
    ap.add_argument("--all", action="store_true", help="include the long MPI studies")
    ap.add_argument("--docs", action="store_true", help="also copy the figures into docs/img/")
    ap.add_argument("--force", action="store_true", help="re-run the simulations")
    args = ap.parse_args()
    names = args.studies or [n for n, (_, long) in STUDIES.items() if args.all or not long]
    extra = (["--docs"] if args.docs else []) + (["--force"] if args.force else [])
    results = []
    for name in names:
        script, *rest = STUDIES[name][0]
        print(f"=== {name}", flush=True)
        t0 = time.time()
        ok = subprocess.run([sys.executable, str(HERE / script), *rest, *extra]).returncode == 0
        results.append((name, ok, time.time() - t0))
    print("\n=== summary")
    for name, ok, dt in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:26s} {dt:7.1f} s")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
