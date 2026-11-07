"""Run the validation studies, grouped by what they establish and ordered by cost.

    python cases/validation/run_all.py                    # the quick studies of every group (minutes)
    python cases/validation/run_all.py --tier medium      # the quick and medium ones (up to a tier)
    python cases/validation/run_all.py --all              # also the long ones (MPI runs, hours in total)
    python cases/validation/run_all.py --budget 60        # stop a study at its next checkpoint after 60 minutes
    python cases/validation/run_all.py --all --docs       # and refresh the figures in docs/img/
    python cases/validation/run_all.py verification       # one group
    python cases/validation/run_all.py sod cavity         # these studies only
    python cases/validation/run_all.py --list             # groups, studies, tier and estimated cost

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
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIERS = ("quick", "medium", "long")
PARTIAL = 75   # exit code of a study stopped by its budget (fenice.study.PARTIAL)
# group -> study -> (script relative to this folder and its arguments, tier, estimated core-hours or None)
# tier: "quick" (under 2 min, one core), "medium" (under 30 min, 4 cores), "long" (hours or MPI)
GROUPS = {
    "verification": {
        "riemann_2d": (["verification/riemann_2d.py"], "quick", 0.01),   # A1: 2-D Riemann problems, 100/200 and AMR
        "riemann_2d_full": (["verification/riemann_2d.py", "--full"], "medium", 0.1),   # A1: 2-D Riemann problems with the 400x400 reference
        "kelvin_helmholtz": (["verification/kelvin_helmholtz.py"], "quick", 0.01),   # A5: Kelvin-Helmholtz linear growth rate
        "implosions": (["verification/implosions.py"], "medium", 0.11),   # A7: Noh (exact) and Sedov (scaling), 2-D
        "blast_waves": (["verification/blast_waves.py"], "quick", 0.02),   # A2: Woodward-Colella blast waves, self-convergence
        "blast_waves_weno": (["verification/blast_waves_weno.py"], "medium", 0.07),   # A2: blast waves with WENO3-Z against the MUSCL run
        "sod": (["verification/sod.py"], "quick", None),
        "shock_order": (["verification/shock_order.py"], "quick", None),   # order of accuracy at a shock (Sod, 100 to 1600 cells)
        "isentropic_vortex": (["verification/isentropic_vortex.py"], "quick", None),
        "taylor_green": (["verification/taylor_green.py"], "quick", None),
        "taylor_green_animation": (["verification/taylor_green_animation.py"], "quick", None),
        "poiseuille": (["verification/poiseuille.py"], "quick", None),
        "stress_shocks": (["verification/stress_shocks.py"], "quick", 0.02),   # B0: strong shocks, vacuum, Mach 10-100, odd-even
        "shock_blend": (["verification/shock_blend.py"], "quick", 0.01),   # B2: odd-even cure, Sod accuracy, strong-shock tubes
        "mms_time": (["verification/mms_time.py"], "quick", None),   # time order, unsteady manufactured solution (2 minutes)
        "mms": (["verification/mms.py"], "quick", None),   # manufactured solutions, 2-D and 3-D, up to 64^2 / 48^3
        "mms_full": (["verification/mms.py", "--full"], "long", None),   # up to 128^2 / 96^3 (about 20 minutes), as in the docs
        "mms_weno": (["verification/mms_weno.py"], "quick", None),   # periodic manufactured solutions with WENO3-Z / WENO5-Z, up to 64^2 / 48^3
        "mms_time_weno": (["verification/mms_time_weno.py"], "quick", None),   # time order with WENO3-Z / WENO5-Z (a few minutes)
        "mms_amr_weno": (["verification/mms_amr_weno.py"], "quick", None),   # interior level jumps, WENO3-Z / MUSCL, 16 to 64
    },
    "benchmarks": {
        "shu_osher": (["benchmarks/shu_osher.py"], "quick", None),
        "shu_osher_schemes": (["benchmarks/shu_osher_schemes.py"], "quick", None),
        "cavity": (["benchmarks/cavity.py"], "medium", 2.1),   # measured: 32 min on 4 threads (the 128^2 runs to t = 40 dominate)
        "taylor_green_3d": (["benchmarks/taylor_green_3d.py", "--grids", "32", "64", "96", "--vorticity-n", "96",
                             "--mpi", "16", "--q-animation"], "long", None),   # the settings of the docs figures
        "vv20": (["benchmarks/vv20.py"], "long", None),   # validation uncertainty (needs the runs of gci.py)
        "gci": (["benchmarks/gci.py"], "long", None),   # grid convergence index of the cavity, airfoil and 3-D TG results
        "cylinder_loads": (["benchmarks/ibm_loads.py", "cylinder"], "long", None),
        "cylinder_spectrum": (["benchmarks/cylinder_spectrum.py"], "long", None),   # spectra of the cylinder_loads run
        "cylinder_low_blockage": (["benchmarks/cylinder_low_blockage.py"], "long", None),   # Re 100, 1.7 % blockage, AMR l3 / l4 to t = 60 (l4 on 8 MPI ranks); the runs of cylinder_modal
        "spectra_application": (["benchmarks/spectra_application.py"], "long", None),   # D5: spectra of the Taylor-Green 3-D runs at peak dissipation, serial vs MPI, wake line (needs the runs of taylor_green_3d and cylinder_low_blockage)
        "cylinder_modal": (["benchmarks/cylinder_modal.py"], "long", None),   # POD / DMD of the cylinder_loads snapshots, St from the lift monitors (needs the runs of cylinder_loads and cylinder_low_blockage)
        "airfoil_loads": (["benchmarks/ibm_loads.py", "airfoil"], "long", None),
    },
    "amr": {
        "sod_amr": (["amr/sod_amr.py"], "quick", None),
        "sod_amr_animation": (["amr/sod_amr_animation.py"], "quick", None),
        "airfoil_amr": (["amr/airfoil_amr.py"], "long", None),
        "amr_mpi": (["amr/amr_mpi.py"], "long", None),   # timings: needs an idle machine
        "amr_memory": (["amr/amr_memory.py"], "long", None),   # memory per MPI rank, remesh parts (the distributed-storage baseline)
    },
    "features": {
        "cylinder_sensitivity": (["features/cylinder_sensitivity.py"], "medium", 0.5),   # C1c: 8 base points = 41 runs (smoke test; the indices need --full)
        "cylinder_sensitivity_full": (["features/cylinder_sensitivity.py", "--full"], "long", 5.0),   # 32 base points = 161 runs, measured 18000 s of solver time (about 40 min on 8 workers)
        "sod_sensitivity": (["features/sod_sensitivity.py"], "quick", 0.01),   # C1c: Sobol indices of the Sod L1 error (CFL, resolution, limiter), 320 runs
        "cylinder_mesh_transition": (["features/cylinder_mesh_transition.py"], "long", None),
    },
}
STUDIES = {name: (group, *spec) for group, studies in GROUPS.items() for name, spec in studies.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", nargs="*", choices=[[]] + list(GROUPS) + list(STUDIES), metavar="STUDY_OR_GROUP",
                    help=f"studies or groups to run (default: all quick ones): {', '.join(GROUPS)}, "
                         f"{', '.join(STUDIES)}")
    ap.add_argument("--all", action="store_true", help="include the long studies (same as --tier long)")
    ap.add_argument("--tier", choices=TIERS, default="quick", help="run the studies up to this tier (default: quick)")
    ap.add_argument("--budget", type=float, default=None, metavar="MIN",
                    help="wall-time budget per study [minutes]: it stops at the next checkpoint, the next call resumes")
    ap.add_argument("--threads", type=int, default=None, metavar="N", help="OpenMP threads of the solver (default 2)")
    ap.add_argument("--list", action="store_true", help="list the groups and studies, and exit")
    ap.add_argument("--docs", action="store_true", help="also copy the figures into docs/img/")
    ap.add_argument("--force", action="store_true", help="re-run the simulations")
    args = ap.parse_args()
    if args.list:
        for group, studies in GROUPS.items():
            print(group)
            for name, (script, tier, est) in studies.items():
                cost = "      ?" if est is None else f"{est:6.2f}h"
                print(f"  {name:26s} {tier:6s} {cost}  {script[0]}")
        return
    names = []
    for w in args.what:
        names += list(GROUPS[w]) if w in GROUPS else [w]
    if not names:
        top = TIERS.index("long" if args.all else args.tier)
        names = [n for n, (_, _, tier, _) in STUDIES.items() if TIERS.index(tier) <= top]
    extra = (["--docs"] if args.docs else []) + (["--force"] if args.force else [])
    if args.budget is not None:
        os.environ["FENICE_STUDY_BUDGET"] = str(args.budget)
    if args.threads is not None:
        os.environ["FENICE_STUDY_THREADS"] = str(args.threads)
    results = []
    for name in names:
        group, (script, *rest), _, _ = STUDIES[name]
        print(f"=== {group}/{name}", flush=True)
        t0 = time.time()
        code = subprocess.run([sys.executable, str(HERE / script), *rest, *extra]).returncode
        results.append((f"{group}/{name}", "OK  " if code == 0 else "PART" if code == PARTIAL else "FAIL",
                        time.time() - t0))
    print("\n=== summary")
    for name, state, dt in results:
        print(f"  {state}  {name:36s} {dt:7.1f} s")
    sys.exit(1 if any(state == "FAIL" for _, state, _ in results) else 0)


if __name__ == "__main__":
    main()
