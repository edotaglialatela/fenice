"""Regression check of the C++ solver against stored reference states.

The references in ``cpp/tests/reference/`` are final states for short
configurations covering every Riemann solver, limiter and time scheme, stiffened
gas, gravity, periodic / inlet / outlet BCs, viscous flow, IBM, hot start and AMR
with mixed levels.  The first 16 are states of the original Python solver (the
reference implementation, retired on 2026-10-20), which the C++ solver reproduced
bit for bit (mass-flow cases to ~1e-14).  Later cases (Taylor-Green,
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
    "sod_muscl_mc_rk3_lm": ("Sod Shock Tube", dict(nx=80, ny=4, t_end=0.05, use_muscl=True, muscl_limiter="MC",
                                                   time_scheme="SSP Runge-Kutta 3", riemann_solver="HLLC (Low-Mach Thornber)")),
    "sod_koren_precond":   ("Sod Shock Tube", dict(nx=80, ny=4, t_end=0.05, use_muscl=True, muscl_limiter="Koren",
                                                   riemann_solver="HLLC (Preconditioned)")),
    "sod_stiffened_grav":  ("Sod Shock Tube", dict(nx=60, ny=6, t_end=0.02, eos_type="Stiffened Gas", gamma=4.4,
                                                   p_inf=0.5, use_bodyforce=True, gx=0.3, gy=-1.0)),
    "shu_osher":           ("Shu-Osher Test", dict(nx=200, t_end=0.2)),
    "cavity_rk3_visc":     ("Lid Driven Cavity", dict(nx=20, ny=20, t_end=0.2, use_muscl=True)),
    "cavity_periodic_x":   ("Lid Driven Cavity", dict(nx=16, ny=12, t_end=0.1, bc_left="Periodic", bc_right="Periodic")),
    "poiseuille_euler":    ("Poiseuille Flow", dict(nx=30, ny=10, t_end=1.0)),
    "cylinder_ibm_visc":   ("IBM Flow over Cylinder", dict(nx=80, ny=40, t_end=0.05, use_mpi=False)),
    # source terms with the IBM (a streamwise and a cross-stream momentum source act on the fluid; the body stays at rest)
    "cylinder_ibm_source":  ("IBM Flow over Cylinder", dict(nx=80, ny=40, t_end=0.05, use_mpi=False,
                                                           source_mx="0.5*sin(2*pi*x/Lx)", source_my="0.3")),
    "amr_cylinder_source":  ("IBM Flow over Cylinder", dict(nx=80, ny=40, t_end=0.05, use_mpi=False, use_amr=True,
                                                           amr_max_level=1, amr_refine_every=5,
                                                           source_mx="0.5*sin(2*pi*x/Lx)", source_my="0.3")),
    "supersonic_ibm":      ("IBM Supersonic Flow over Cylinder", dict(nx=60, ny=60, Lx=4.0, Ly=4.0, ibm_cx=1.0, ibm_cy=2.0, t_end=0.1, use_mpi=False)),   # the geometry of the reference (the case is 8 x 8 now)
    "hot_start_mapping":   ("Sod Shock Tube", dict(nx=100, ny=8, t_end=0.03, use_hot_start=True,
                                                   hot_start_file="{REF}/hot_source.npz", use_muscl=True)),
    "amr_sod_1st":         ("Sod Shock Tube", dict(Lx=2.0, nx=128, ny=16, Ly=0.25, t_end=0.2, use_amr=True,
                                                   amr_max_level=3, amr_refine_every=2, amr_init_levels=1,
                                                   time_scheme="SSP Runge-Kutta 2")),
    "amr_sod_muscl_init":  ("Sod Shock Tube", dict(nx=64, ny=16, Ly=0.25, t_end=0.15, use_amr=True,
                                                   amr_max_level=2, amr_refine_every=3, use_muscl=True,
                                                   amr_init_levels=2)),
    "amr_square_rk2":      ("Sod Shock Tube", dict(Lx=1.0, Ly=1.0, nx=32, ny=32, t_end=0.4, use_amr=True,
                                                   amr_max_level=3, amr_refine_every=2, amr_init_levels=1,
                                                   time_scheme="SSP Runge-Kutta 2")),
    "amr_square_rk3":      ("Sod Shock Tube", dict(Lx=1.0, Ly=1.0, nx=32, ny=32, t_end=0.4, use_amr=True,
                                                   amr_max_level=3, amr_refine_every=2, amr_init_levels=1,
                                                   use_muscl=True, time_scheme="SSP Runge-Kutta 3")),
    "amr_weno3_rk3":       ("Sod Shock Tube", dict(Lx=1.0, Ly=1.0, nx=32, ny=32, t_end=0.4, use_amr=True,
                                                   amr_max_level=3, amr_refine_every=2, amr_init_levels=1,
                                                   use_muscl=True, reconstruction="weno3", time_scheme="SSP Runge-Kutta 3")),
    # second-order boundary faces on the adaptive mesh: walls on all four sides, the reflected shock refines against them
    "amr_bo2_walls":       ("Sod Shock Tube", dict(Lx=1.0, Ly=1.0, nx=32, ny=32, t_end=0.4, use_amr=True, bc_left="wall", bc_right="wall",
                                                   bc_bottom="wall", bc_top="wall", amr_max_level=3, amr_refine_every=2,
                                                   amr_init_levels=1, use_muscl=True, boundary_order=2,
                                                   time_scheme="SSP Runge-Kutta 3")),
    "amr_bo2_weno3":       ("Sod Shock Tube", dict(Lx=1.0, Ly=1.0, nx=32, ny=32, t_end=0.4, use_amr=True, bc_left="wall", bc_right="wall",
                                                   bc_bottom="wall", bc_top="wall", amr_max_level=3, amr_refine_every=2,
                                                   amr_init_levels=1, use_muscl=True, reconstruction="weno3", boundary_order=2,
                                                   time_scheme="SSP Runge-Kutta 3")),
    # viscous walls and a moving lid on the adaptive mesh with second-order boundary faces: the lid corner refines
    "amr_cavity_b2":       ("Lid Driven Cavity", dict(nx=16, ny=16, t_end=0.3, use_amr=True, amr_max_level=2, amr_refine_every=3,
                                                   amr_init_levels=1, use_muscl=True, boundary_order=2,
                                                   time_scheme="SSP Runge-Kutta 3")),
    # A tiny base mesh: under MPI most ranks own few or no leaves at the coarse levels (empty ranks, small halos)
    "amr_sod_few_roots":   ("Sod Shock Tube", dict(Lx=1.0, nx=32, ny=4, Ly=0.125, t_end=0.15, use_amr=True,
                                                   amr_max_level=3, amr_refine_every=2)),
    # AMR seeded from a uniform hot-start field: every leaf interpolates it at its centre
    "amr_hot_start":       ("Sod Shock Tube", dict(nx=100, ny=8, t_end=0.03, use_hot_start=True,
                                                   hot_start_file="{REF}/hot_source.npz", use_muscl=True, use_amr=True,
                                                   amr_max_level=2, amr_refine_every=3)),
    # The source of a hot start is the checkpoint of an AMR run (32 x 4 base mesh, 2 levels, 14 steps of the Sod tube): an AMR run
    # with fewer levels samples it at the leaf centres; a uniform run maps its finest-level field onto its own mesh
    "amr_hot_start_from_amr": ("Sod Shock Tube", dict(Lx=1.0, nx=32, ny=4, Ly=0.125, t_end=0.03, use_hot_start=True,
                                                     hot_start_file="{REF}/hot_source_amr.npz", use_muscl=True, use_amr=True,
                                                     amr_max_level=1, amr_refine_every=3)),
    "hot_start_from_amr":   ("Sod Shock Tube", dict(Lx=1.0, nx=64, ny=8, Ly=0.125, t_end=0.03, use_hot_start=True,
                                                   hot_start_file="{REF}/hot_source_amr.npz", use_muscl=True)),
    # Manufactured solutions with source terms (generated cases, see fenice.mms); covered by the order study
    # verification/mms.py. The thin variants make the MPI strips run along x.
    "mms_ns_2d":           ("Manufactured Solution", dict(nx=24, ny=24, t_end=0.1)),
    "mms_euler_xdecomp":   ("Manufactured Solution Euler", dict(nx=32, ny=6, t_end=0.1)),
    "mms_ns_3d":           ("Manufactured Solution 3D", dict(nx=12, ny=12, nz=12, t_end=0.05)),
    "mms_walls":           ("Manufactured Solution Walls", dict(nx=24, ny=24, t_end=0.1)),
    "mms_unsteady_rk3":    ("Manufactured Solution Unsteady Euler", dict(nx=16, ny=16, t_end=0.1, time_scheme="SSP Runge-Kutta 3")),
    "mms_unsteady_rk2":    ("Manufactured Solution Unsteady Euler", dict(nx=16, ny=16, t_end=0.1, time_scheme="SSP Runge-Kutta 2")),
    "mms_outflow_xdecomp": ("Manufactured Solution Outflow", dict(nx=32, ny=6, t_end=0.1)),
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
    # AMR airfoil: body kept at level 2, remeshing, IBM + viscous + NRBC (the level-0 equality with the uniform solver is
    # a unit test; loads checked against a uniform run of the body's resolution)
    "amr_naca0012":        ("IBM Airfoil", dict(nx=192, ny=96, t_end=0.4, use_amr=True,
                                                amr_max_level=2, amr_refine_every=5, use_mpi=False)),
    # Discontinuous compressible benchmarks (studies of cases/validation/verification), small meshes and short times;
    # "riemann2d:K" is configuration K of verification/riemann_2d.py written to a temporary case file
    "blast_waves_100":     ("Woodward-Colella Blast Waves", dict(nx=100, ny=4, t_end=0.02)),
    "kelvin_helmholtz_32": ("Kelvin-Helmholtz", dict(nx=32, ny=36, t_end=4.0)),
    "noh_40":              ("Noh Implosion", dict(nx=40, ny=40, t_end=0.2)),
    "riemann2d_c3_32":     ("riemann2d:3", dict(nx=32, ny=32, use_muscl=True, muscl_limiter="MC",
                                               time_scheme="SSP Runge-Kutta 2")),
    "riemann2d_c6_32":     ("riemann2d:6", dict(nx=32, ny=32, use_muscl=True, muscl_limiter="MC",
                                               time_scheme="SSP Runge-Kutta 2")),
    # WENO-Z reconstruction (order checked by the entropy-wave study; these pin the implementation): WENO3 in 2-D and 3-D
    # (also on MPI ranks: the stencil is as deep as MUSCL's) and WENO5 (serial), with the SSP-RK3 integrator
    "vortex_weno3_rk3":    ("Isentropic Vortex", dict(nx=48, ny=48, t_end=1.0, use_muscl=True, reconstruction="weno3",
                                                  time_scheme="SSP Runge-Kutta 3", cfl=0.4)),
    "vortex_weno5_rk3":    ("Isentropic Vortex", dict(nx=48, ny=48, t_end=1.0, use_muscl=True, reconstruction="weno5",
                                                  time_scheme="SSP Runge-Kutta 3", cfl=0.4, use_mpi=False)),
    "tgv3d_weno3_rk3":     ("Taylor-Green Vortex 3D", dict(nx=16, ny=16, nz=16, t_end=1.0, use_muscl=True,
                                                       reconstruction="weno3", time_scheme="SSP Runge-Kutta 3", cfl=0.4)),
    "tgv3d_weno5_rk3":     ("Taylor-Green Vortex 3D", dict(nx=16, ny=16, nz=16, t_end=0.5, use_muscl=True,
                                                       reconstruction="weno5", time_scheme="SSP Runge-Kutta 3", cfl=0.4)),
    # boundary_order = 2: ghost layers on the physical sides, the faces on the boundary reconstructed (the x strip decomposition
    # makes the left side physical on rank 0 and an interface on the others; the cavity has a moving and no-slip walls)
    "mms_walls_b2":        ("Manufactured Solution Walls", dict(nx=24, ny=24, t_end=0.1, use_muscl=True, boundary_order=2)),
    "mms_outflow_xdecomp_b2": ("Manufactured Solution Outflow", dict(nx=32, ny=6, t_end=0.1, use_muscl=True, boundary_order=2)),
    "sod3d_walls_z_b2":    ("Sod Shock Tube", dict(nx=40, ny=4, nz=4, Lz=0.1, t_end=0.05, bc_front="wall", bc_back="wall",
                                                use_muscl=True, time_scheme="SSP Runge-Kutta 2", boundary_order=2)),
    "cavity_weno3_b2":     ("Lid Driven Cavity", dict(nx=20, ny=20, t_end=0.2, use_muscl=True, reconstruction="weno3",
                                                  time_scheme="SSP Runge-Kutta 3", boundary_order=2)),
    "cavity3d_noslip_z_b2": ("Lid Driven Cavity", dict(nx=10, ny=10, nz=6, Lz=1.0, t_end=0.1, bc_front="no_slip",
                                                  bc_back="no_slip", use_muscl=True, boundary_order=2)),
    "cavity_weno5_b2":    ("Lid Driven Cavity", dict(nx=20, ny=20, t_end=0.2, use_muscl=True, reconstruction="weno5",
                                                 time_scheme="SSP Runge-Kutta 3", boundary_order=2, use_mpi=False)),
    "cavity3d_weno5_b2":  ("Lid Driven Cavity", dict(nx=12, ny=12, nz=8, Lz=1.0, t_end=0.1, bc_front="no_slip", bc_back="no_slip",
                                                 use_muscl=True, reconstruction="weno5", time_scheme="SSP Runge-Kutta 3",
                                                 boundary_order=2, use_mpi=False)),
    "cavity_xdecomp_b2":   ("Lid Driven Cavity", dict(nx=40, ny=6, t_end=0.1, use_muscl=True, boundary_order=2)),
    "cavity3d_xdecomp_b2": ("Lid Driven Cavity", dict(nx=24, ny=6, nz=6, Lz=1.0, t_end=0.1, bc_front="no_slip",
                                                 bc_back="no_slip", use_muscl=True, boundary_order=2)),
}
# Cases also run under MPI (AMR: replicated mesh, leaves distributed per level in Morton order); they cover Y- and X-decomposition,
# viscous / IBM / MUSCL / periodic halos and the mass-flow inlet reduction.
MPI_CASES = ["sod_muscl_vl_rk2", "shu_osher", "cavity_rk3_visc", "cavity_periodic_x", "poiseuille_euler",
             "cylinder_ibm_visc", "cylinder_ibm_source", "amr_cylinder_source", "supersonic_ibm", "taylor_green_visc", "isentropic_vortex", "vortex_thin_xdecomp",
             "tgv3d_16", "tgv3d_xdecomp", "cavity3d_noslip_z", "sod3d_walls_z", "sphere3d_ibm_visc",
             "vortex_nrbc", "vortex_nrbc_xdecomp", "tgv3d_nrbc", "naca0012_ibm",
             "amr_sod_1st", "amr_sod_muscl_init", "amr_square_rk2", "amr_square_rk3", "amr_weno3_rk3", "amr_bo2_walls", "amr_bo2_weno3", "amr_cavity_b2", "amr_naca0012", "amr_sod_few_roots", "amr_hot_start", "amr_hot_start_from_amr", "hot_start_from_amr", "mms_ns_2d", "mms_euler_xdecomp", "mms_ns_3d", "mms_walls", "mms_outflow_xdecomp", "mms_unsteady_rk3", "mms_unsteady_rk2",
             "blast_waves_100", "kelvin_helmholtz_32", "noh_40", "riemann2d_c3_32", "riemann2d_c6_32",
             "vortex_weno3_rk3", "tgv3d_weno3_rk3", "mms_walls_b2", "mms_outflow_xdecomp_b2", "sod3d_walls_z_b2", "cavity_weno3_b2", "cavity3d_noslip_z_b2", "cavity_xdecomp_b2", "cavity3d_xdecomp_b2"]


def riemann_case_file(k, tmp):
    """Write configuration k of the 2-D Riemann study to a case file under tmp and return its folder."""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "cases", "validation", "verification"))
    import riemann_2d
    folder = os.path.join(tmp, f"riemann2d_{k}")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "case.toml"), "w", newline="\n") as f:
        f.write(riemann_2d.config_case(k))
    return folder


def run_cpp(name, tmp, n_mpi=0):
    tc, ov = CASES[name]
    dump = os.path.join(tmp, f"{name}_{n_mpi}.npz")
    if tc.startswith("riemann2d:"):
        tc = riemann_case_file(int(tc.split(":")[1]), tmp)
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
