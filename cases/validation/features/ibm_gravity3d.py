"""A sphere released in a stratified gas (3-D): weight and buoyancy of a body driven by the flow.

    python cases/validation/features/ibm_gravity3d.py            # three mass ratios at 19 cells per diameter (96^3, about a minute each)

The 3-D twin of `ibm_gravity.py`: a rigid sphere (R = 0.15, inviscid, MUSCL, SSP-RK2, closed box of 1.2^3, g = 0.5 downwards) at rest in an isothermal gas in
hydrostatic balance (p / rho = 1, rho = exp(-g y)). Until the sound reflected by the walls returns, the sphere obeys

    (m + 0.5 rho_c V) y'' = -(m - rho_c V) g,        V = 4 pi R^3 / 3,  rho_c = the gas density at the centre,

the incompressible added mass of a sphere (the compressible correction is of order (kR)^2 and small at these slow motions). The mass ratio m / (rho_c V) is
3 (sinks), 1 (neutral: must stay) and 0.5 (rises). Acceleration from the quadratic fit y = y0 + a t^2 / 2 over t = 0.25 .. 0.8.

Bands fixed BEFORE the first run (at 64^3): sinking within 5 %, rising within 10 %, neutral displaced at most 5 % of the sinking body's displacement.
Result at 96^3: sinking +1.1 %, rising +7.5 %, neutral 0.4 % of the sinking displacement. The ghost pressure follows the hydrostatic gradient of the gas
(`use_bodyforce`): before it, the wall gave a start-up pulse of the vertical load (a held body saw a load above the buoyancy that decayed within a few sound
crossing times), a kick upwards for every released body, and the rising body measured +12.7 % at 96^3 (+16.6 % at 64^3, +15.4 % with the CFL number cut to 0.1),
the sinking one -0.3 % and the neutral one drifted by 1.8 %. What is left of the rising offset is mostly the reference: a constant added mass for a start from rest.
The exact response of a sphere to a force step in a compressible gas, `m_a(s) = rho V (x + 1) / (x^2 + 2 x + 2)` with `x = s R / c` in
`Y(s) = F0 / (s^3 (m + m_a(s)))` inverted numerically, is +1.2 % (sinking) and +5..8 % (rising, t = 0.2 .. 0.8) from the runs.
"""
import math

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_gravity3d"
R, G = 0.15, 0.5
V = 4.0 / 3.0 * math.pi * R ** 3
M_ADDED_RATIO = 0.5
YC = 0.6
RHO_C = math.exp(-G * YC)
CASE = """[case]
name = "Gravity Sphere"
description = "a sphere released in a stratified gas (inviscid)"
gui = false
[mesh]
Lx = 1.2
Ly = 1.2
Lz = 1.2
nx = 64
ny = 64
nz = 64
[physics]
use_viscous = false
use_bodyforce = true
gx = 0.0
gy = -0.5
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
t_end = 0.85
save_interval = 100000
monitor_interval = 1
[parallel]
use_mpi = false
[ibm]
use_ibm = true
ibm_shape = "Sphere"
ibm_radius = 0.15
ibm_cx = 0.6
ibm_cy = 0.6
ibm_cz = 0.6
[initial]
state = { rho = "exp(-0.5 * y)", u = 0.0, v = 0.0, p = "exp(-0.5 * y)" }
"""


def run(ratio, cells=96):
    out = study.study_dir(STUDY) / f"mu{ratio:g}_n{cells}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.toml").write_text(CASE)
    fenice.run(str(out), f"nx={cells}", f"ny={cells}", f"nz={cells}", f"ibm_mass={ratio * RHO_C * V!r}", out_root=out / "run")
    d = fenice.read_monitor(study.case_output(out / "run"))
    return d["t"], d["body_y"]


def fit(t, y, t0=0.25, t1=0.8):
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
        i = int(np.searchsorted(t, 0.8))
        drop[ratio] = float(y[min(i, len(y) - 1)])
        if ratio == 1:
            continue
        good = abs(a / ref - 1.0) < (0.05 if ratio > 1 else 0.10)
        ok = ok and good
        print(f"mass ratio {ratio:4g}: acceleration {a:+.4f}, reference {ref:+.4f} ({100 * (a / ref - 1):+.1f} %)  {'ok' if good else 'OUT OF BAND'}", flush=True)
    neutral = abs(drop[1]) / abs(drop[3])
    good = neutral < 0.05
    ok = ok and good
    print(f"mass ratio    1: displacement at t = 0.8 is {100 * neutral:.1f} % of the sinking body's  {'ok' if good else 'OUT OF BAND'}", flush=True)
    print("ALL IN BAND" if ok else "OUT OF BAND")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
