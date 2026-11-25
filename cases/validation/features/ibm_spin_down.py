"""A body whose rotation is driven by the flow: a cylinder (2-D) and a sphere (3-D) released with a spin in a viscous gas at rest, against the exact linear transient.

    python cases/validation/features/ibm_spin_down.py          # the cylinder at three body densities, 20 cells per diameter; the sphere at two, 10 cells per diameter

A cylinder (R = 0.1, per unit depth) in a closed box of 2 x 2 (isothermal start rho = p = 1, mu = 0.01, so nu = 0.01) is released with the angular velocity w0 = 0.1
(surface speed 0.01: Mach 0.008, Reynolds number w0 R^2 / nu = 0.1, so the problem is linear). Its rotation is free (`ibm_inertia`, the body pinned at its centre): the torque of
the fluid decelerates it. The flow round a cylinder turning at the angular velocity w(t) in a fluid at rest is, in the Laplace variable s (q = sqrt(s / nu), x = q R),
u_theta = R w K1(q r) / K1(x), so the torque of the fluid on the body per unit depth is 2 pi mu R^2 w (2 + x K0(x) / K1(x)), and the equation of motion I dw/dt = -T gives

    w(s) = I w0 / (I s + 2 pi mu R^2 (2 + x K0(x) / K1(x))),

inverted numerically (fixed Talbot contour, M = 32; K0, K1 of complex argument from SciPy). There is no added inertia (the term in x is a history torque, as for the sphere): the
torque of the fluid grows like sqrt(s) at high frequency. The reference holds until the shear layer reaches the walls of the box (sqrt(nu t) of the order of the half width, 1).

A sphere (R = 0.1, mass density rho_b, I = 8 pi rho_b R^5 / 15) in a closed box of 1^3, free to spin about its centre (`ibm_inertia`, `ibm_omega` / `ibm_omega_x` the initial
spin) follows the same equation with the torque of the sphere that the spin-up study derives, Z(s) w (s) with Z(s) = (8 pi mu R^3 / 3) (x^2 + 3 x + 3) / (x + 1), x = R sqrt(s / nu):

    w(s) = I w0 / (I s + Z(s)).

The run about z checks the moment about z, the one about x the moments that only 3-D has.

The body is a disk of density rho_b (I = pi rho_b R^4 / 2): rho_b / rho = 10, 1 and 0.1. Measured (20 cells per diameter, h / R = 0.1): the angular velocity follows the exact
one to 0.3 % of w0 at all three (0.0032, 0.0023, 0.0029); bands 1 % of w0, set after the first run.

The loose coupling (the torque of the first stage of a step advances the body for the whole step) has a limit in the density ratio. A rotating body has no added inertia, but the
history torque of one step is that of a shear layer of thickness sqrt(nu dt), so the ratio of that torque to the body's inertia is of order (rho / rho_b) (h / R), and the update
loses its accuracy when rho_b / rho falls to about h / R and diverges at about half of that (scratch sweep of the 2 x 2 box, t_end 0.3, error in w / w0 at the largest time step of
the run, CFL 0.4 unless stated):
    h / R = 0.2 (n = 100):  rho_b / rho 0.15 error 0.16, 0.10 error 0.22, 0.07 diverged
    h / R = 0.1 (n = 200):  0.10 error 0.003, 0.07 error 0.17, 0.05 error 0.22, 0.035 error 33 (unstable), 0.02 diverged
    h / R = 0.05 (n = 400): 0.05 error 0.17, 0.025 error 0.22 (still stable)
    h / R = 0.1, CFL 0.2:   0.035 error 0.20, 0.02 error 0.34 (the limit does not move with the time step)
A body lighter than that needs an implicit coupling (not built).
"""
import argparse
import math

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_spin_down"
R = 0.1
MU = 0.01
RHO = 1.0
W0 = 0.1


def case(inertia, n=200, t_end=3.0):
    return f"""[case]
name = "Spin Down"
description = "a cylinder released with a spin in a viscous gas at rest"
gui = false
[mesh]
Lx = 2.0
Ly = 2.0
nx = {n}
ny = {n}
[physics]
use_viscous = true
mu = {MU}
use_bodyforce = false
[boundaries]
bc_xmin = "wall"
bc_xmax = "wall"
bc_ymin = "wall"
bc_ymax = "wall"
u_in = 0.01
rho_in = {RHO}
p_out = 1.0
[numerics]
reconstruction = "muscl"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.4
[run]
t_end = {t_end}
save_interval = 100000
monitor_interval = 2
[parallel]
use_mpi = false
[ibm]
use_ibm = true
ibm_shape = "Cylinder"
ibm_radius = {R}
ibm_cx = 1.0
ibm_cy = 1.0
ibm_omega = {W0}
ibm_inertia = {inertia}
[initial]
state = {{ rho = {RHO}, u = 0.0, v = 0.0, p = 1.0 }}
"""


def exact_omega(t, inertia, m=32):
    """w(t) / w0 by the fixed Talbot inversion of w(s) = I / (I s + 2 pi mu R^2 (2 + x K0 / K1))."""
    from scipy.special import kv

    def image(s):
        x = R * np.sqrt(s / (MU / RHO))
        return inertia / (inertia * s + 2.0 * math.pi * MU * R ** 2 * (2.0 + x * kv(0, x) / kv(1, x)))

    r = 2.0 * m / (5.0 * t)
    total = 0.5 * image(r + 0j).real * math.exp(r * t)
    for k in range(1, m):
        th = k * math.pi / m
        sigma = r * th * (1.0 / math.tan(th) + 1j)
        tau = th + (th / math.tan(th) - 1.0) / math.tan(th)
        total += (np.exp(t * sigma) * image(sigma) * (1.0 + 1j * tau)).real
    return r / m * total


SPHERE = f"""[case]
name = "Spin Down Sphere"
description = "a sphere released with a spin in a viscous gas at rest"
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
u_in = 0.01
rho_in = {RHO}
p_out = 1.0
[numerics]
reconstruction = "muscl"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.4
[run]
t_end = 2.0
save_interval = 100000
monitor_interval = 2
[parallel]
use_mpi = true
[ibm]
use_ibm = true
ibm_shape = "Sphere"
ibm_radius = {R}
ibm_cx = 0.5
ibm_cy = 0.5
ibm_cz = 0.5
[initial]
state = {{ rho = {RHO}, u = 0.0, v = 0.0, p = 1.0 }}
"""


def exact_omega_sphere(t, inertia, m=32):
    """w(t) / w0 of the free sphere: Talbot inversion of I / (I s + Z(s))."""
    def image(s):
        x = R * np.sqrt(s / (MU / RHO))
        return inertia / (inertia * s + (8.0 * math.pi * MU * R ** 3 / 3.0) * (x * x + 3.0 * x + 3.0) / (x + 1.0))

    r = 2.0 * m / (5.0 * t)
    total = 0.5 * image(r + 0j).real * math.exp(r * t)
    for k in range(1, m):
        th = k * math.pi / m
        sigma = r * th * (1.0 / math.tan(th) + 1j)
        tau = th + (th / math.tan(th) - 1.0) / math.tan(th)
        total += (np.exp(t * sigma) * image(sigma) * (1.0 + 1j * tau)).real
    return r / m * total


def run_sphere(name, ratio, axis):
    inertia = 8.0 * math.pi * ratio * RHO * R ** 5 / 15.0
    key = {"x": "ibm_omega_x", "z": "ibm_omega"}[axis]
    out = study.study_dir(STUDY) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.toml").write_text(SPHERE)
    fenice.run(str(out), f"ibm_inertia={inertia}", f"{key}={W0}", out_root=out / "run")
    d = fenice.read_monitor(study.case_output(out / "run"))
    return d["t"], d[f"body_omega_{axis}"], inertia


def run(name, ratio, **kw):
    inertia = 0.5 * math.pi * ratio * RHO * R ** 4
    out = study.study_dir(STUDY) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.toml").write_text(case(inertia, **kw))
    fenice.run(str(out), out_root=out / "run")
    d = fenice.read_monitor(study.case_output(out / "run"))
    return d["t"], d["body_omega"], inertia


def check(label, t, om, inertia, band, t_lo=0.1, exact=None):
    sel = t >= t_lo
    ref = np.array([(exact or exact_omega)(x, inertia) for x in t[sel]])
    err = float(np.abs(om[sel] / W0 - ref).max())   # w0-normalised: the decay is O(1)
    good = err < band
    print(f"{label}: largest |w - w_exact| / w0 over t >= {t_lo:g}: {err:.4f} (w goes {om[0] / W0:.2f} -> {om[-1] / W0:.3f}, exact {ref[-1]:.3f})  {'ok' if good else 'OUT OF BAND'}", flush=True)
    return good


def main():
    argparse.ArgumentParser().parse_args()
    ok = True
    for name, ratio, band in (("heavy", 10.0, 0.01), ("neutral", 1.0, 0.01), ("light", 0.1, 0.01)):
        t, om, inertia = run(name, ratio)
        ok = check(f"rho_b / rho = {ratio:g}", t, om, inertia, band) and ok
    for name, ratio, axis, band in (("sphere_heavy_x", 10.0, "x", 0.03), ("sphere_neutral_z", 1.0, "z", 0.03)):
        t, om, inertia = run_sphere(name, ratio, axis)
        ok = check(f"sphere, rho_b / rho = {ratio:g}, spin about {axis}", t, om, inertia, band, exact=exact_omega_sphere) and ok
    print("ALL IN BAND" if ok else "OUT OF BAND")


if __name__ == "__main__":
    main()
