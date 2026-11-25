"""A sphere that spins (prescribed rotation, 3-D): the torque of a sphere spun up in a viscous gas at rest, and the three axes.

    python cases/validation/features/ibm_spin_sphere.py            # 10 cells per diameter (a minute or two)
    python cases/validation/features/ibm_spin_sphere.py --full     # ... and 20 cells per diameter (about ten minutes more)

A sphere (R = 0.1) at the centre of a closed box of 1^3 (slip walls, isothermal start rho = p = 1, mu = 0.01, so nu = 0.01) starts to spin about z at the rate
w = 0.1 (the surface speed w R = 0.01 is Mach 0.008, the Reynolds number w R^2 / nu = 0.1), the wall of the ghost-cell method carrying the velocity w x r.
For a small Reynolds number the torque of the fluid on a sphere spun up impulsively is exact (Stokes; the Laplace image of the torque is
(8 pi mu R^3 w / 3) (x^2 + 3 x + 3) / (s (x + 1)) with x = R sqrt(s / nu)), which inverts to

    T(t) = (8 pi mu R^3 w / 3) [ a / sqrt(pi t) + 3 - exp(t / a^2) erfc(sqrt(t) / a) ],        a = R / sqrt(nu),

and tends to 8 pi mu R^3 w from above (a / sqrt(pi t) is the thin shear layer of the start). The monitor's moment is the one of the solid cells about the centre. The
reference holds until the shear layer reaches the walls (sqrt(nu t) = 0.4, t = 16); the run goes to t = 8.

Bands, set after the first run (10 cells per diameter: the ratio simulation / exact is 0.984 at every time from t = 0.5 to 8, spread 0.1 %): the ratio within 3 % of 1
for t = 1 .. 8, and its spread over that interval below 1 %. A constant ratio is a radius: the wall of the method sits within half a per cent of R (the torque goes with R^3). At 20 cells per diameter (`--full`, to t = 3: 0.15 s a step) the ratio is
0.991 at every time, the error halved: first order in the wall position, as for the other loads of the ghost-cell wall.

The spin about x and y (`ibm_omega_x`, `ibm_omega_y`; `ibm_omega` is the z component, as in 2-D) is not visible in a monitor that reports the z-moment, so it is checked by symmetry:
on a cube, a sphere spun about x (or y) in a gas at rest is the sphere spun about z with the axes relabelled cyclically (x, y, z) -> (z, x, y) ((y, z, x)); the final states of 40 steps
agree to roundoff (2e-15 measured, band 1e-12). `features/ibm_ghost.py --check` holds the rest: a sphere that translates and spins about all three axes, serial against 2-7 ranks
bit for bit and resumed across the end of the spin (`ibm_omega_until`), on a uniform mesh and an octree.
"""
import argparse
import math

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_spin_sphere"
R = 0.1
MU = 0.01
RHO = 1.0
W = 0.1
UREF = W * R   # only the normalisation of the monitor's coefficients
CASE = f"""[case]
name = "Spinning Sphere"
description = "a sphere spun up in a viscous gas at rest"
gui = false
[mesh]
Lx = 1.0
Ly = 1.0
Lz = 1.0
nx = 50
ny = 50
nz = 50
[physics]
use_viscous = true
mu = {MU}
use_bodyforce = false
[boundaries]
bc_xmin = "wall"
bc_xmax = "wall"
bc_ymin = "wall"
bc_ymax = "wall"
bc_zmin = "wall"
bc_zmax = "wall"
u_in = {UREF}
rho_in = {RHO}
p_out = 1.0
[numerics]
reconstruction = "muscl"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.4
[run]
t_end = 8.0
save_interval = 100000
monitor_interval = 5
[parallel]
use_mpi = true
[ibm]
use_ibm = true
ibm_shape = "Sphere"
ibm_radius = {R}
ibm_cx = 0.5
ibm_cy = 0.5
ibm_cz = 0.5
ibm_omega = {W}
[initial]
state = {{ rho = {RHO}, u = 0.0, v = 0.0, p = 1.0 }}
"""
CUBE = """[case]
name = "Spin Permutation"
description = "a sphere spun in a gas at rest"
gui = false
[mesh]
Lx = 1.0
Ly = 1.0
Lz = 1.0
nx = 24
ny = 24
nz = 24
[physics]
use_viscous = true
mu = 0.01
use_bodyforce = false
[boundaries]
bc_xmin = "wall"
bc_xmax = "wall"
bc_ymin = "wall"
bc_ymax = "wall"
bc_zmin = "wall"
bc_zmax = "wall"
u_in = 0.01
rho_in = 1.0
p_out = 1.0
[numerics]
reconstruction = "muscl"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.4
[run]
t_end = 100.0
max_steps = 40
save_interval = 100000
[parallel]
use_mpi = false
[ibm]
use_ibm = true
ibm_shape = "Sphere"
ibm_radius = 0.25
ibm_cx = 0.5
ibm_cy = 0.5
ibm_cz = 0.5
[initial]
state = { rho = 1.0, u = 0.0, v = 0.0, p = 1.0 }
"""


def exact_torque(t):
    a = R / math.sqrt(MU / RHO)
    x = math.sqrt(t) / a
    return (8.0 * math.pi * MU * R ** 3 * W / 3.0) * (a / math.sqrt(math.pi * t) + 3.0 - math.exp(x * x) * math.erfc(x))


def torque_run(name, *sets):
    out = study.study_dir(STUDY) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.toml").write_text(CASE)
    fenice.run(str(out), *sets, out_root=out / "run")
    d = fenice.read_monitor(study.case_output(out / "run"))
    t = d["t"]
    qa_l = 0.5 * RHO * UREF ** 2 * math.pi * R ** 2 * (2.0 * R)   # Cm = -Mz / (q A L)
    return t, d["Cm"] * qa_l


def torque_check(label, t, torque, band, spread_band, t_end=8.0):
    w = (t >= 1.0) & (t <= t_end)
    ratio = torque[w] / np.array([exact_torque(x) for x in t[w]])
    spread = float(ratio.max() - ratio.min())
    good = bool(np.all(np.abs(ratio - 1.0) < band)) and spread < spread_band
    print(f"{label}: torque / exact over t = 1 .. {t_end:g}: {ratio.min():.3f} .. {ratio.max():.3f} (spread {100 * spread:.2f} %)  {'ok' if good else 'OUT OF BAND'}", flush=True)
    return good


def relabelled(U, n, axis):
    """The final state of the spin about z, seen from axes (x, y, z) = (Z, X, Y) (spin about x) or (Y, Z, X) (spin about y); slots rho, rho u, rho v, rho E, rho w."""
    old = U.reshape(n, n, n, 5)   # [k, j, i]
    new = np.empty_like(old)
    for k in range(n):
        for j in range(n):
            for i in range(n):
                if axis == "x":
                    o = old[i, k, j]
                    new[k, j, i] = [o[0], o[4], o[1], o[3], o[2]]
                else:
                    o = old[j, i, k]
                    new[k, j, i] = [o[0], o[2], o[4], o[3], o[1]]
    return new


def axes_check():
    out = study.study_dir(STUDY) / "axes"
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.toml").write_text(CUBE)
    state = {}
    for name, key in (("z", "ibm_omega"), ("x", "ibm_omega_x"), ("y", "ibm_omega_y")):
        fenice.run(str(out), f"{key}=0.4", out_root=out / name, dump=out / f"{name}.npz")
        state[name] = fenice.read_dump(out / f"{name}.npz")["U"]
    ok = True
    for axis in ("x", "y"):
        diff = float(np.abs(state[axis].reshape(24, 24, 24, 5) - relabelled(state["z"], 24, axis)).max())
        good = diff < 1e-12
        ok = ok and good
        print(f"spin about {axis} against spin about z, axes relabelled: largest difference {diff:.1e} (the state is of order {np.abs(state['z']).max():.1f})  {'ok' if good else 'OUT OF BAND'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also 20 cells per diameter")
    args = ap.parse_args()
    t, torque = torque_run("n50")
    ok = torque_check("10 cells per diameter", t, torque, 0.03, 0.01)
    ok = axes_check() and ok
    if args.full:
        t, torque = torque_run("n100", "nx=100", "ny=100", "nz=100", "t_end=3.0", "monitor_interval=50")   # 0.15 s a step, 10 000 steps
        ok = torque_check("20 cells per diameter", t, torque, 0.03, 0.01, 3.0) and ok
    print("ALL IN BAND" if ok else "OUT OF BAND")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
