"""Free-stream preservation of a moving immersed body: a body carried by a uniform flow must not disturb it.

    python cases/validation/features/ibm_free_stream.py            # a few seconds to minutes
    python cases/validation/features/ibm_free_stream.py --mpi 4    # the same runs on 4 MPI ranks (that they work there)

A cylinder (2-D) or a sphere (3-D) moves with prescribed velocity equal to the velocity of a uniform stream in a box with
transmissive sides. Relative to the gas the body is at rest, so the exact solution is the uniform state: the ghost cells
(u_ghost = u_wall - (d_ghost / d_image)(u_image - u_wall) for the no-slip wall, the reflected normal part for the slip
wall) and the deeper solid cells (body velocity at the reference state) have to reproduce it. The body crosses several
cells, so the wall is rebuilt.

What it detects: a wrong wall or solid velocity or state on a moving body (a sign, a velocity not refreshed after a move,
a reference state that is not applied). What it cannot detect: a fresh cell that is not filled or is filled from the wrong
cells, and a wrong halo index, because every solid and fluid cell already holds the stream state; those are covered by the
dynamic cases of ibm_ghost.py --check (bit for bit against serial) and by the added-mass studies.
Runs: slip and no-slip wall, uniform mesh and adaptive mesh (two levels, the body band at the finest one), 2-D and 3-D.

Measured: the largest deviation of rho, u, v, (w), p from the uniform state over ALL cells (leaves) at the end.

Bands, fixed before the first run: at most 1e-10 in every run (roundoff). Negative control: the same uniform 2-D run
with the body at rest in the moving stream must deviate by more than 1e-2, so that the measure can see a wall at all.
A uniform state would also survive a body that never moved, so every moving run must have carried the wall at least 0.8
of the distance u t along x (the pose of the geometry in the checkpoint, which lags the motion law by less than a cell).
"""
import numpy as np

import fenice
from fenice import study

STUDY = "ibm_free_stream"
TOL, CONTROL, MOVED = 1e-10, 1e-2, 0.8
STATE = {"rho": 1.0, "u": 0.5, "v": 0.2, "w": 0.1, "p": 1.0}


def case_text(dim):
    sides = ["xmin", "xmax", "ymin", "ymax"] + (["zmin", "zmax"] if dim == 3 else [])
    keys = ("rho", "u", "v", "p") + (("w",) if dim == 3 else ())
    return "\n".join([
        "[case]", f'name = "Free Stream {dim}D"', 'description = "a body carried by a uniform stream"', "gui = false",
        *(["requires_3d = true"] if dim == 3 else []),
        "[mesh]", "Lx = 2.0", "Ly = 2.0", *(["Lz = 2.0", "nz = 32"] if dim == 3 else []),
        f"nx = {32 if dim == 3 else 80}", f"ny = {32 if dim == 3 else 80}",
        "[physics]", "use_viscous = false", "mu = 1e-2",
        "[boundaries]", *[f'bc_{s} = "transmissive"' for s in sides], f"rho_in = {STATE['rho']}", f"p_out = {STATE['p']}",
        "[numerics]", 'reconstruction = "muscl"', 'time_scheme = "SSP Runge-Kutta 2"', "cfl = 0.4",
        "[run]", "t_end = 0.6", "save_interval = 100000",
        "[parallel]", "use_mpi = false",
        "[ibm]", "use_ibm = true", f'ibm_shape = "{"Sphere" if dim == 3 else "Cylinder"}"', "ibm_radius = 0.25",
        "ibm_cx = 0.7", "ibm_cy = 0.85", *(["ibm_cz = 0.9"] if dim == 3 else []),
        "[initial]", "state = { " + ", ".join(f"{k} = {STATE[k]}" for k in keys) + " }", ""])


def deviation(U, dim):
    got = fenice.primitive(U)
    names = ("rho", "u", "v", "p") + (("w",) if dim == 3 else ())
    return {k: float(np.max(np.abs(g - STATE[k]))) for k, g in zip(names, got)}


def run(tag, dim, sets, args, moving=True):
    motion = [f"ibm_vx={STATE['u']}", f"ibm_vy={STATE['v']}"] + ([f"ibm_vz={STATE['w']}"] if dim == 3 else [])
    if not moving:
        motion = []
    case = study.write_case(STUDY, f"free_stream_{dim}d", case_text(dim))
    if args.mpi > 1:
        sets = [*sets, "use_mpi=true", f"n_cpus={args.mpi}"]
        tag = f"{tag}_np{args.mpi}"
    d = study.run_cached(STUDY, tag, case, *motion, *sets, mpi=args.mpi, force=args.force)
    final = fenice.read_dump(d / "final.npz")
    travelled = 0.0
    if moving:
        pose = fenice.read_dump(next(d.glob("**/*_restart.npz")))["body_pose"]
        travelled = float(pose[0]) / (STATE["u"] * float(final["t"]))   # 1 = the whole distance u t
    return deviation(final["U"], dim), travelled


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    amr = ["use_amr=true", "amr_max_level=2", "amr_refine_every=4", "nx=40", "ny=40"]
    amr3 = ["use_amr=true", "amr_max_level=1", "amr_refine_every=4", "nx=16", "ny=16", "nz=16"]
    runs = [("2d_slip", 2, []), ("2d_noslip", 2, ["use_viscous=true"]),
            ("2d_amr_slip", 2, amr), ("2d_amr_noslip", 2, [*amr, "use_viscous=true"]),
            ("3d_slip", 3, []), ("3d_noslip", 3, ["use_viscous=true"]),
            ("3d_amr_slip", 3, amr3), ("3d_amr_noslip", 3, [*amr3, "use_viscous=true"])]
    rows = ["| run | largest deviation | variable | distance travelled / (u t) | band | |", "|---|---:|---|---:|---|---|"]
    ok = True
    for tag, dim, sets in runs:
        dev, travelled = run(tag, dim, sets, args)
        worst = max(dev, key=dev.get)
        good = dev[worst] <= TOL and travelled >= MOVED
        ok = ok and good
        rows.append(f"| {tag} | {dev[worst]:.2e} | {worst} | {travelled:.3f} | <= {TOL:g}, >= {MOVED} | {'ok' if good else 'OUT OF BAND'} |")
        print(rows[-1], flush=True)
    dev, _ = run("2d_control_body_at_rest", 2, [], args, moving=False)
    worst = max(dev, key=dev.get)
    good = dev[worst] > CONTROL
    ok = ok and good
    rows.append(f"| negative control, body at rest | {dev[worst]:.2e} | {worst} | - | > {CONTROL:g} | {'ok' if good else 'OUT OF BAND'} |")
    print(rows[-1], flush=True)
    name = "ibm_free_stream" + (f"_np{args.mpi}" if args.mpi > 1 else "") + ".md"
    (study.study_dir(STUDY) / name).write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    print("ALL IN BAND" if ok else "OUT OF BAND")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
