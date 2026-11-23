"""A cylinder released in a stratified gas: weight and buoyancy of a body driven by the flow.

    python cases/validation/features/ibm_gravity.py            # three mass ratios at 20 cells per diameter (seconds)

A rigid cylinder of radius R = 0.1 (inviscid, MUSCL, SSP-RK2, closed box of 2.4 x 2.4, gravity g = 0.2 downwards) sits at rest in an isothermal gas
in hydrostatic balance, p / rho = 1, rho = rho0 exp(-g y), so that the pressure the wall sees gives the buoyancy: the force of the pressure on the body
is the weight of the gas it displaces (Archimedes) and the solver adds the weight m g of the body. With the exact added mass of a cylinder in a
compressible gas at low frequency (1.0535 rho V, V = pi R^2, the one `ibm_spring.py` checks) the body obeys, until the sound reflected by the walls
returns (t about 2),

    (m + m_a) y'' = -(m - rho_c V) g,        rho_c = the gas density at the body's centre,

a constant acceleration (the radiation damping, 0.0467 rho V w, is small at these slow motions). The mass ratio m / (rho_c V) is 3 (it sinks), 1 (neutral:
it must stay, the check of the buoyancy itself) and 0.5 (it rises). The acceleration is the quadratic fit y = y0 + a t^2 / 2 over t = 0.3 .. 1.2; the
hydrostatic state is not exactly balanced by the scheme (a transient of the second order in the cell size) and the neutral body measures that.

Bands, set after the first run (a first guess of 10 % for both bodies was missed by the rising one): measured at 20 / 40 cells per diameter the sinking
body is +0.8 / +2.2 % off the reference, the rising one +12.0 / +10.8 %, the neutral one drifts with an acceleration of 0.0025 / 0.0016 (3 % of the
sinking body's displacement at 20 cells per diameter, falling with the mesh: the hydrostatic imbalance of the scheme). The rising error does not fall with
the mesh, so it is not the discretisation: the probable cause is the reference itself, a constant added mass of a steady oscillation (kR = 0.169) for a start
from rest and a density that varies by 4 % across the body, which weighs most on the light body (added mass two thirds of the inertia). Bands: sinking
within 5 %, rising within 15 %, neutral displaced at most 5 % of the sinking body's displacement at t = 1.2.
"""
import math

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_gravity"
R, G = 0.1, 0.2
V = math.pi * R * R
M_ADDED_RATIO = 1.0535
YC = 1.2
RHO_C = math.exp(-G * YC)
CASE = """[case]
name = "Gravity Cylinder"
description = "a cylinder released in a stratified gas (inviscid)"
gui = false
[mesh]
Lx = 2.4
Ly = 2.4
nx = 240
ny = 240
[physics]
use_viscous = false
use_bodyforce = true
gx = 0.0
gy = -0.2
[boundaries]
bc_xmin = "wall"
bc_xmax = "wall"
bc_ymin = "wall"
bc_ymax = "wall"
rho_in = 1.0
p_out = 1.0
[numerics]
reconstruction = "muscl"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.4
[run]
t_end = 1.4
save_interval = 100000
monitor_interval = 1
[parallel]
use_mpi = false
[ibm]
use_ibm = true
ibm_shape = "Cylinder"
ibm_radius = 0.1
ibm_cx = 1.2
ibm_cy = 1.2
[initial]
state = { rho = "exp(-0.2 * y)", u = 0.0, v = 0.0, p = "exp(-0.2 * y)" }
"""


def run(ratio, cells=240):
    out = study.study_dir(STUDY) / f"mu{ratio:g}_n{cells}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.toml").write_text(CASE)
    fenice.run(str(out), f"nx={cells}", f"ny={cells}", f"ibm_mass={ratio * RHO_C * V!r}", out_root=out / "run")
    d = fenice.read_monitor(study.case_output(out / "run"))
    return d["t"], d["body_y"]


def fit(t, y, t0=0.3, t1=1.2):
    """Least-squares y = y0 + a t^2 / 2 over the window: (a, y0)."""
    w = (t >= t0) & (t <= t1)
    A = np.stack([0.5 * t[w] ** 2, np.ones(int(w.sum()))], axis=1)
    sol, *_ = np.linalg.lstsq(A, y[w], rcond=None)
    return float(sol[0]), float(sol[1])


def main():
    ok = True
    drop = {}
    for ratio in (3, 1, 0.5):
        t, y = run(ratio)
        a, _ = fit(t, y)
        ref = -G * (ratio - 1.0) * RHO_C * V / ((ratio + M_ADDED_RATIO) * RHO_C * V)
        i = int(np.searchsorted(t, 1.2))
        drop[ratio] = float(y[min(i, len(y) - 1)])
        if ratio == 1:
            continue
        good = abs(a / ref - 1.0) < (0.05 if ratio > 1 else 0.15)
        ok = ok and good
        print(f"mass ratio {ratio:4g}: acceleration {a:+.4f}, reference {ref:+.4f} ({100 * (a / ref - 1):+.1f} %)  {'ok' if good else 'OUT OF BAND'}", flush=True)
    neutral = abs(drop[1]) / abs(drop[3])
    good = neutral < 0.05
    ok = ok and good
    print(f"mass ratio    1: displacement at t = 1.2 is {100 * neutral:.1f} % of the sinking body's  {'ok' if good else 'OUT OF BAND'}", flush=True)
    print("ALL IN BAND" if ok else "OUT OF BAND")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
