"""Run the validation studies, grouped by what they establish and ordered by cost.

    python cases/validation/run_all.py                    # the quick studies of every group (minutes)
    python cases/validation/run_all.py --all              # also the long ones (MPI runs, hours in total)
    python cases/validation/run_all.py --all --docs       # and refresh the figures in docs/img/
    python cases/validation/run_all.py verification       # one group
    python cases/validation/run_all.py sod cavity         # these studies only
    python cases/validation/run_all.py --list             # groups, studies and which ones are long

Groups (one folder each):
    verification  does the code solve the equations right? exact solutions, orders of accuracy
    benchmarks    reference solutions or data without a closed form, and grid convergence of them
    amr           the adaptive mesh: accuracy and cost against uniform meshes, MPI ranks
    features      one-off demonstrations of solver features

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
# group -> study -> (script relative to this folder and its arguments, long?)
GROUPS = {
    "verification": {
        "sod": (["verification/sod.py"], False),
        "shock_order": (["verification/shock_order.py"], False),   # order of accuracy at a shock (Sod, 100 to 1600 cells)
        "isentropic_vortex": (["verification/isentropic_vortex.py"], False),
        "taylor_green": (["verification/taylor_green.py"], False),
        "taylor_green_animation": (["verification/taylor_green_animation.py"], False),
        "poiseuille": (["verification/poiseuille.py"], False),
        "mms_time": (["verification/mms_time.py"], False),   # time order, unsteady manufactured solution (2 minutes)
        "mms": (["verification/mms.py"], False),   # manufactured solutions, 2-D and 3-D, up to 64^2 / 48^3
        "mms_full": (["verification/mms.py", "--full"], True),   # up to 128^2 / 96^3 (about 20 minutes), as in the docs
    },
    "benchmarks": {
        "shu_osher": (["benchmarks/shu_osher.py"], False),
        "cavity": (["benchmarks/cavity.py"], False),
        "taylor_green_3d": (["benchmarks/taylor_green_3d.py", "--grids", "32", "64", "96", "--vorticity-n", "96",
                             "--mpi", "16", "--q-animation"], True),   # the settings of the docs figures
        "vv20": (["benchmarks/vv20.py"], True),   # validation uncertainty (needs the runs of gci.py)
        "gci": (["benchmarks/gci.py"], True),   # grid convergence index of the cavity, airfoil and 3-D TG results
        "cylinder_loads": (["benchmarks/ibm_loads.py", "cylinder"], True),
        "airfoil_loads": (["benchmarks/ibm_loads.py", "airfoil"], True),
    },
    "amr": {
        "sod_amr": (["amr/sod_amr.py"], False),
        "sod_amr_animation": (["amr/sod_amr_animation.py"], False),
        "airfoil_amr": (["amr/airfoil_amr.py"], True),
        "amr_mpi": (["amr/amr_mpi.py"], True),   # timings: needs an idle machine
        "amr_memory": (["amr/amr_memory.py"], True),   # memory per MPI rank, remesh parts (the distributed-storage baseline)
    },
    "features": {
        "cylinder_mesh_transition": (["features/cylinder_mesh_transition.py"], True),
    },
}
STUDIES = {name: (group, *spec) for group, studies in GROUPS.items() for name, spec in studies.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", nargs="*", choices=[[]] + list(GROUPS) + list(STUDIES), metavar="STUDY_OR_GROUP",
                    help=f"studies or groups to run (default: all quick ones): {', '.join(GROUPS)}, "
                         f"{', '.join(STUDIES)}")
    ap.add_argument("--all", action="store_true", help="include the long studies")
    ap.add_argument("--list", action="store_true", help="list the groups and studies, and exit")
    ap.add_argument("--docs", action="store_true", help="also copy the figures into docs/img/")
    ap.add_argument("--force", action="store_true", help="re-run the simulations")
    args = ap.parse_args()
    if args.list:
        for group, studies in GROUPS.items():
            print(group)
            for name, (script, long) in studies.items():
                print(f"  {name:26s} {'long' if long else 'quick'}  {script[0]}")
        return
    names = []
    for w in args.what:
        names += list(GROUPS[w]) if w in GROUPS else [w]
    if not names:
        names = [n for n, (_, _, long) in STUDIES.items() if args.all or not long]
    extra = (["--docs"] if args.docs else []) + (["--force"] if args.force else [])
    results = []
    for name in names:
        group, (script, *rest), _ = STUDIES[name]
        print(f"=== {group}/{name}", flush=True)
        t0 = time.time()
        ok = subprocess.run([sys.executable, str(HERE / script), *rest, *extra]).returncode == 0
        results.append((f"{group}/{name}", ok, time.time() - t0))
    print("\n=== summary")
    for name, ok, dt in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:36s} {dt:7.1f} s")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
