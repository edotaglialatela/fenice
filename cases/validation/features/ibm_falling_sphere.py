"""A sphere falling to its terminal speed in a stratified viscous gas (3-D octree): the body driven by the flow against a drag law.

    python cases/validation/features/ibm_falling_sphere.py            # the long fall at 10 cells per diameter (a few minutes) and the prescribed-speed check
    python cases/validation/features/ibm_falling_sphere.py --full     # ... and the mesh check at 20 cells per diameter (about ten minutes more)

A rigid sphere (R = 0.1, density ratio m / (rho V) = 12) is released from rest in an isothermal gas in hydrostatic balance (p / rho = 1, rho = exp(-g y), g = 0.0186,
mu = 8.9e-4, gamma = 1.4) in a box of 2 x 8 x 2 with slip walls, on an octree with two levels (the body band at the finest level, 10 cells per diameter, 25 x 100 x 25
base cells). The fall reaches Re = 32 (Mach 0.13) after about 40 s, a fall of 6 units of length (30 diameters) along which the gas density changes by 12 %.

Reference: the equation of motion with the Schiller-Naumann drag, a quasi-steady drag at the local density, and the added mass 0.5 rho V,

    (m + 0.5 rho V) a = -(m - rho V) g + 0.5 rho |v| v A Cd(Re),   Cd = 24 / Re (1 + 0.15 Re^0.687),   Re = rho |v| D / mu.

It has no history force and no compressibility, and the correlation itself scatters by a few per cent, so it is a yardstick, not an exact answer.

What the study measures, with bands set AFTER the sizing runs (the first runs were at 10, 20 and 40 cells per diameter and a box twice as wide, see below):

* the terminal speed (mean of the speed over t = 34 .. 44) within 10 % of the reference's (measured -6.6 %: 0.1516 against 0.1626);
* the whole fall: the speed within 15 % of the reference's from t = 6 on (measured -7 .. -13 %, the early excess drag of an accelerating body);
* the coupling: a sphere prescribed at the measured terminal speed in the same gas carries the weight, mean load over t = 20 .. 30 within 3 % of the mean load
  of the free fall over t = 34 .. 50 (measured 0.6 %). The monitored `Fy` is the surface load, which reads 4 % below the weight at terminal speed (the load that drives
  the body is the solid-cell balance, 1-2.5 % above the surface one for a ghost wall); both runs have the same offset, which is what the check compares.

Measured while sizing (not gated): the fall at 10 / 20 / 40 cells per diameter is within 0.7 % in distance at t = 20 (speed over the reference 0.918 / 0.924 / 0.917);
a box 4 x 8 x 4 gives the same fall (0.920); the terminal drag implied by the free fall is about 10 % above Schiller-Naumann (6 % above the correlation
Cd = 24 / Re (1 + 0.1935 Re^0.6305)) and its cause was not isolated (a fixed sphere in a uniform stream would tell the wall from the rest). The gas at rest without the body
keeps 0.4 % of the terminal speed as a spurious vertical momentum after 300 steps (the hydrostatic balance of the scheme).
"""
import argparse
import math

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_falling_sphere"
R = 0.1
D = 2 * R
V = 4.0 / 3.0 * math.pi * R ** 3
A = math.pi * R ** 2
G = 0.0186
MU = 8.9e-4
RATIO = 12.0
Y0 = 7.3
LX, LY, LZ = 2.0, 8.0, 2.0
RHO_C = math.exp(-G * Y0)
M = RATIO * RHO_C * V
CASE = f"""[case]
name = "Falling Sphere"
description = "a sphere falling in a stratified viscous gas"
gui = false
[mesh]
Lx = {LX}
Ly = {LY}
Lz = {LZ}
nx = 25
ny = 100
nz = 25
[thermo]
gamma = 1.4
[physics]
use_viscous = true
mu = {MU}
use_bodyforce = true
gx = 0.0
gy = -{G}
gz = 0.0
[boundaries]
bc_xmin = "wall"
bc_xmax = "wall"
bc_ymin = "wall"
bc_ymax = "wall"
bc_zmin = "wall"
bc_zmax = "wall"
rho_in = 1.0
p_out = 1.0
[numerics]
reconstruction = "muscl"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.4
[run]
t_end = 50.0
save_interval = 100000
monitor_interval = 10
[amr]
use_amr = true
amr_max_level = 2
amr_refine_every = 8
[ibm]
use_ibm = true
ibm_shape = "Sphere"
ibm_radius = {R}
ibm_cx = {LX / 2}
ibm_cy = {Y0}
ibm_cz = {LZ / 2}
ibm_mass = {M!r}
[initial]
state = {{ rho = "exp(-{G} * y)", u = 0.0, v = 0.0, p = "exp(-{G} * y)" }}
"""


def drag_coefficient(re):
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)


def reference(t_end, dt=1e-3):
    """The equation of motion above, integrated with explicit Euler at a small step: (t, displacement, velocity) every 0.1."""
    y, u, t = Y0, 0.0, 0.0
    ts, ys, us = [0.0], [0.0], [0.0]
    n = int(round(t_end / dt))
    for k in range(1, n + 1):
        rho = math.exp(-G * y)
        re = rho * abs(u) * D / MU
        drag = -0.5 * rho * u * abs(u) * A * drag_coefficient(re) if re > 1e-12 else 0.0
        u += dt * (-(M - rho * V) * G + drag) / (M + 0.5 * rho * V)
        y += u * dt
        if k % 100 == 0:
            ts.append(k * dt)
            ys.append(y - Y0)
            us.append(u)
    return np.array(ts), np.array(ys), np.array(us)


def run(name, *sets):
    out = study.study_dir(STUDY) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.toml").write_text(CASE)
    fenice.run(str(out), *sets, out_root=out / "run")
    return fenice.read_monitor(study.case_output(out / "run"))


def window_mean(t, f, a, b):
    w = (t >= a) & (t <= b)
    return float(np.trapezoid(f[w], t[w]) / (t[w][-1] - t[w][0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also the fall at 20 cells per diameter (to t = 20)")
    args = ap.parse_args()
    ok = True

    free = run("free_fall")
    t, y = free["t"], free["body_y"]
    tr, yr, ur = reference(50.0)
    v = np.gradient(y, t)
    v_term = window_mean(t, free["body_v"], 34.0, 44.0)
    w = (tr >= 34.0) & (tr <= 44.0)
    ref_term = float(ur[w].mean())
    ratio = v_term / ref_term
    good = abs(ratio - 1.0) < 0.10
    ok = ok and good
    print(f"terminal speed {v_term:+.4f}, reference {ref_term:+.4f} ({100 * (ratio - 1):+.1f} %)  {'ok' if good else 'OUT OF BAND'}", flush=True)

    late = (t >= 6.0)
    r = v[late] / np.interp(t[late], tr, ur)
    good = bool(np.all(np.abs(r - 1.0) < 0.15))
    ok = ok and good
    print(f"speed against the reference from t = 6: {100 * (r.min() - 1):+.1f} .. {100 * (r.max() - 1):+.1f} %  {'ok' if good else 'OUT OF BAND'}", flush=True)

    held = run("prescribed", "ibm_mass=0.0", f"ibm_vy={v_term!r}", "t_end=30")
    f_free = window_mean(t, free["Fy"], 34.0, 50.0)
    f_held = window_mean(held["t"], held["Fy"], 20.0, 30.0)
    weight = M * G
    good = abs(f_free / f_held - 1.0) < 0.03
    ok = ok and good
    print(f"load at the terminal speed: free fall {f_free / weight:.3f}, prescribed {f_held / weight:.3f} of the weight ({100 * (f_free / f_held - 1):+.1f} %)  "
          f"{'ok' if good else 'OUT OF BAND'}", flush=True)

    if args.full:
        fine = run("free_fall_20", "amr_max_level=3", "t_end=20")
        d10 = float(np.interp(20.0, t, y))
        d20 = float(np.interp(20.0, fine["t"], fine["body_y"]))
        good = abs(d20 / d10 - 1.0) < 0.02
        ok = ok and good
        print(f"fall at t = 20: 10 cells per D {d10:+.3f}, 20 cells per D {d20:+.3f} ({100 * (d20 / d10 - 1):+.1f} %)  {'ok' if good else 'OUT OF BAND'}", flush=True)

    print("ALL IN BAND" if ok else "OUT OF BAND")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
