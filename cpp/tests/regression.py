"""Regression check of the C++ solver against stored reference states.

The references in ``cpp/tests/reference/`` are final states for short
configurations covering every Riemann solver, limiter and time scheme, stiffened
gas, gravity, periodic / inlet / outlet BCs, viscous flow, IBM, hot start and AMR
with mixed levels.  The first 16 are states of the original Python solver (the
reference implementation, retired on 2026-10-20), which the C++ solver reproduced
bit for bit (BDF2 / mass-flow cases to ~1e-14).  Later cases (Taylor-Green,
isentropic vortex) were created from the C++ solver after validation against
their exact solutions (see the fenice_tests checks of the same names).

    python cpp/tests/regression.py             # serial C++ vs references
    python cpp/tests/regression.py --mpi 4     # also C++ MPI (4 ranks) vs references
    python cpp/tests/regression.py --only amr
    python cpp/tests/regression.py --create-missing   # write references for NEW cases only
    python cpp/tests/regression.py --update CASE...   # overwrite these references (deliberate, validated change)

Only numpy is needed.  Exit code 0 when everything matches.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

import fenice

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "reference")
# Solver executable: build/release/fenice ($FENICE_EXE overrides; see fenice.exe)
try:
    EXE = fenice.exe()
except FileNotFoundError as e:
    sys.exit(str(e))

# name: (test case, overrides on top of the case defaults)
CASES = {
    "sod_1st_euler":       ("Sod Shock Tube", dict(nx=100, ny=4, t_end=0.05)),
    "sod_muscl_vl_rk2":    ("Sod Shock Tube", dict(nx=100, ny=4, t_end=0.05, use_muscl=True, time_scheme="SSP Runge-Kutta 2")),
    "sod_muscl_mc_rk4_lm": ("Sod Shock Tube", dict(nx=80, ny=4, t_end=0.05, use_muscl=True, muscl_limiter="MC",
                                                   time_scheme="Runge-Kutta 4", riemann_solver="HLLC (Low-Mach Thornber)")),
    "sod_koren_precond":   ("Sod Shock Tube", dict(nx=80, ny=4, t_end=0.05, use_muscl=True, muscl_limiter="Koren",
                                                   riemann_solver="HLLC (Preconditioned)")),
    "sod_stiffened_grav":  ("Sod Shock Tube", dict(nx=60, ny=6, t_end=0.02, eos_type="Stiffened Gas", gamma=4.4,
                                                   p_inf=0.5, use_bodyforce=True, gx=0.3, gy=-1.0)),
    "shu_osher":           ("Shu-Osher Test", dict(nx=200, t_end=0.2)),
    "cavity_rk4_visc":     ("Lid Driven Cavity", dict(nx=20, ny=20, t_end=0.2, use_muscl=True)),
    "cavity_periodic_x":   ("Lid Driven Cavity", dict(nx=16, ny=12, t_end=0.1, bc_left="Periodic", bc_right="Periodic")),
    "poiseuille_bdf2":     ("Poiseuille Flow", dict(nx=30, ny=10, t_end=2.0, time_scheme="BDF2 Dual Time")),
    "poiseuille_euler":    ("Poiseuille Flow", dict(nx=30, ny=10, t_end=1.0)),
    "cylinder_ibm_visc":   ("IBM Flow over Cylinder", dict(nx=80, ny=40, t_end=0.05, use_mpi=False)),
    "supersonic_ibm":      ("IBM Supersonic Flow over Cylinder", dict(nx=60, ny=60, t_end=0.1, use_mpi=False)),
    "hot_start_mapping":   ("Sod Shock Tube", dict(nx=100, ny=8, t_end=0.03, use_hot_start=True,
                                                   hot_start_file="{REF}/hot_source.npz", use_muscl=True)),
    "amr_sod_1st":         ("Sod Shock Tube", dict(Lx=2.0, nx=128, ny=16, Ly=0.25, t_end=0.2, use_amr=True, amr_patch_nx=16,
                                                   amr_patch_ny=4, amr_max_level=3, amr_refine_every=2,
                                                   amr_init_levels=1, amr_buffer_layers=0,
                                                   time_scheme="SSP Runge-Kutta 2")),
    "amr_sod_muscl_init":  ("Sod Shock Tube", dict(nx=64, ny=16, Ly=0.25, t_end=0.15, use_amr=True, amr_patch_nx=16,
                                                   amr_patch_ny=4, amr_max_level=2, amr_refine_every=3, use_muscl=True,
                                                   amr_init_levels=2, amr_buffer_layers=0)),
    "amr_square_rk4":      ("Sod Shock Tube", dict(Lx=1.0, Ly=1.0, nx=32, ny=32, t_end=0.4, use_amr=True,
                                                   amr_patch_nx=8, amr_patch_ny=8, amr_max_level=3, amr_refine_every=2,
                                                   amr_init_levels=1, time_scheme="Runge-Kutta 4",
                                                   amr_refine_threshold=0.2, amr_derefine_threshold=0.05)),
    # Two root patches only: under MPI most ranks own no leaf at the coarse levels (empty ranks, migration)
    "amr_sod_few_roots":   ("Sod Shock Tube", dict(Lx=1.0, nx=32, ny=4, Ly=0.125, t_end=0.15, use_amr=True,
                                                   amr_patch_nx=16, amr_patch_ny=4, amr_max_level=3, amr_refine_every=2,
                                                   amr_buffer_layers=1)),
    # AMR seeded from a uniform hot-start field: owned leaves interpolate it, the halos are pushed
    "amr_hot_start":       ("Sod Shock Tube", dict(nx=100, ny=8, t_end=0.03, use_hot_start=True,
                                                   hot_start_file="{REF}/hot_source.npz", use_muscl=True, use_amr=True,
                                                   amr_patch_nx=10, amr_patch_ny=4, amr_max_level=2, amr_refine_every=3,
                                                   amr_buffer_layers=1)),
    # Added 2026-10-20 (C++ references, validated against the exact solutions)
    "taylor_green_visc":   ("Taylor-Green Vortex", dict(nx=32, ny=32, t_end=0.5, mu=0.05)),
    "isentropic_vortex":   ("Isentropic Vortex", dict(nx=40, ny=40, t_end=1.0)),
    "vortex_thin_xdecomp": ("Isentropic Vortex", dict(nx=64, ny=6, t_end=0.5)),
    # 3-D (added 2026-10-20; C++ references validated by the fenice_tests 3-D checks)
    "tgv3d_16":            ("Taylor-Green Vortex 3D", dict(nx=16, ny=16, nz=16, t_end=0.5, mu=0.01)),
    "tgv3d_xdecomp":       ("Taylor-Green Vortex 3D", dict(nx=16, ny=6, nz=8, t_end=0.3)),
    "cavity3d_noslip_z":   ("Lid Driven Cavity", dict(nx=10, ny=10, nz=6, Lz=1.0, t_end=0.1, bc_front="no_slip",
                                                  bc_back="no_slip", use_muscl=True)),
    "sod3d_walls_z":       ("Sod Shock Tube", dict(nx=40, ny=4, nz=4, Lz=0.1, t_end=0.05, bc_front="wall",
                                               bc_back="wall", use_muscl=True, time_scheme="SSP Runge-Kutta 2")),
    "sphere3d_ibm_visc":   ("IBM Flow over Cylinder", dict(nx=32, ny=16, nz=8, Lz=1.0, t_end=0.03, use_mpi=False,
                                                       ibm_shape="Sphere", ibm_cz=0.5, ibm_radius=0.25,
                                                       bc_front="wall", bc_back="wall")),
    # Non-reflecting outlet (C++ references; NRBC validated by fenice_tests nonreflecting_outlet and the
    # plane-wave / cylindrical-wave / vortex-exit reflection measurements on the Boundaries page)
    "vortex_nrbc":         ("Isentropic Vortex", dict(nx=40, ny=48, t_end=1.0, use_muscl=True, bc_left="transmissive",
                                                  bc_right="nonreflecting")),
    "vortex_nrbc_xdecomp": ("Isentropic Vortex", dict(nx=64, ny=6, t_end=0.5, bc_left="nonreflecting",
                                                  bc_right="nonreflecting", nrbc_sigma=1.0)),
    "tgv3d_nrbc":          ("Taylor-Green Vortex 3D", dict(nx=16, ny=16, nz=8, t_end=0.3, bc_left="transmissive",
                                                       bc_right="nonreflecting", bc_front="nonreflecting",
                                                       bc_back="nonreflecting")),
    # IBM airfoil (NACA 0012 at 10 deg, mu from reynolds; geometry checked by fenice_tests airfoil_geometry)
    "naca0012_ibm":        ("IBM Airfoil", dict(nx=192, ny=96, t_end=0.1, use_mpi=False)),
    # AMR airfoil: tiled 16 x 16 root, body kept at level 2, remeshing, IBM + viscous + NRBC (validated by
    # the tiled-vs-uniform unit test; loads checked against a uniform run of the body's resolution)
    "amr_naca0012":        ("IBM Airfoil", dict(nx=192, ny=96, t_end=0.4, use_amr=True, amr_patch_nx=16, amr_patch_ny=16,
                                                amr_max_level=2, amr_refine_every=5, use_mpi=False)),
}
# Cases also run under MPI (AMR: replicated tree, leaves distributed per level in Morton order); they cover Y- and X-decomposition,
# viscous / IBM / MUSCL / periodic halos and the mass-flow inlet reduction.
MPI_CASES = ["sod_muscl_vl_rk2", "shu_osher", "cavity_rk4_visc", "cavity_periodic_x", "poiseuille_bdf2",
             "cylinder_ibm_visc", "supersonic_ibm", "taylor_green_visc", "isentropic_vortex", "vortex_thin_xdecomp",
             "tgv3d_16", "tgv3d_xdecomp", "cavity3d_noslip_z", "sod3d_walls_z", "sphere3d_ibm_visc",
             "vortex_nrbc", "vortex_nrbc_xdecomp", "tgv3d_nrbc", "naca0012_ibm",
             "amr_sod_1st", "amr_sod_muscl_init", "amr_square_rk4", "amr_naca0012", "amr_sod_few_roots", "amr_hot_start"]


def run_cpp(name, tmp, n_mpi=0):
    tc, ov = CASES[name]
    dump = os.path.join(tmp, f"{name}_{n_mpi}.npz")
    cmd = [EXE, "--case", tc, "--no-logo",
           "--set", "save_interval=1000000000", "--set", f"output_root={os.path.join(tmp, 'out')}",
           "--set", f"dump_state={dump}"]
    for k, v in ov.items():
        cmd += ["--set", f"{k}={str(v).replace('{REF}', REF)}"]
    if n_mpi:
        cmd = ["mpiexec", "-n", str(n_mpi)] + cmd + ["--set", "use_mpi=true", "--set", f"n_cpus={n_mpi}", "--mpi-child"]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{name}: fenice failed ({res.returncode})\n{res.stdout[-1500:]}\n{res.stderr[-1500:]}")
    d = np.load(dump)
    return d["U"], (d["leaf_meta"] if "leaf_meta" in d else None), int(d["step"]), time.perf_counter() - t0


def compare(name, U, meta, steps, tol):
    ref = np.load(os.path.join(REF, f"{name}.npz"))
    ok = steps == int(ref["step"]) and U.shape == ref["U"].shape
    if "leaf_meta" in ref:
        ok = ok and meta is not None and meta.shape == ref["leaf_meta"].shape and np.allclose(meta, ref["leaf_meta"])
    diff = float("inf")
    if U.shape == ref["U"].shape:
        scale = np.maximum(np.max(np.abs(ref["U"]), axis=0), 1e-30)
        diff = float(np.max(np.abs(U - ref["U"]) / scale))
    return ok and diff < tol, diff, int(ref["step"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="substring filter on case names")
    ap.add_argument("--mpi", type=int, default=0, help="also run the MPI cases on N ranks")
    ap.add_argument("--tol", type=float, default=1e-8, help="max relative difference (default 1e-8)")
    ap.add_argument("--create-missing", action="store_true",
                    help="write references for cases that have none (never overwrites existing ones)")
    ap.add_argument("--update", nargs="+", metavar="CASE", choices=list(CASES),
                    help="overwrite the references of these cases with the current results, printing how far they "
                         "moved; only after a deliberate change of the numerics that was validated physically")
    args = ap.parse_args()
    if not os.path.isfile(EXE):
        sys.exit(f"Solver not built: {EXE}\nRun .\\cpp\\build.ps1 first.")

    tmp = tempfile.mkdtemp(prefix="fv_regression_")
    if args.create_missing:
        for name in CASES:
            path = os.path.join(REF, f"{name}.npz")
            if os.path.exists(path):
                continue
            U, meta, steps, _ = run_cpp(name, tmp)
            np.savez(path, U=U, step=steps, **({} if meta is None else {"leaf_meta": meta}))
            print(f"created reference {name} ({steps} steps, {U.shape[0]} cells)")
    for name in args.update or []:
        U, meta, steps, _ = run_cpp(name, tmp)
        _, diff, old_steps = compare(name, U, meta, steps, args.tol)
        np.savez(os.path.join(REF, f"{name}.npz"), U=U, step=steps, **({} if meta is None else {"leaf_meta": meta}))
        print(f"updated reference {name}: {old_steps} -> {steps} steps, max rel change {diff:.2e}")
    runs = [(n, 0) for n in CASES if args.only in n]
    if args.mpi:
        runs += [(n, args.mpi) for n in MPI_CASES if args.only in n]
    failures = 0
    print(f"{'case':22s} {'ranks':>5s} {'steps ref/c++':>14s} {'max rel diff':>13s} {'wall [s]':>9s}  result")
    for name, n in runs:
        U, meta, steps, wall = run_cpp(name, tmp, n)
        ok, diff, ref_steps = compare(name, U, meta, steps, args.tol)
        failures += 0 if ok else 1
        print(f"{name:22s} {n or 1:5d} {ref_steps:7d}/{steps:<6d} {diff:13.3e} {wall:9.2f}  {'OK' if ok else 'MISMATCH'}")
    print(f"\n{'ALL MATCH' if not failures else f'{failures} MISMATCH(ES)'}  (tolerance {args.tol:g})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
