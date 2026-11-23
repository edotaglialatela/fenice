"""A cylinder on a spring in a gas at rest: the body driven by the flow against the linear fluid-structure ODE.

    python cases/validation/features/ibm_spring.py            # three mass ratios at 20 cells per diameter (seconds)
    python cases/validation/features/ibm_spring.py --full     # ... and at 40 cells per diameter, and the time-step check

A rigid cylinder of radius R = 0.1 (inviscid, MUSCL, SSP-RK2, non-reflecting sides of a 2.4 x 2.4 box) is given the velocity v0 = 0.04 (Mach 0.034)
at t = 0 and held by a spring of stiffness k. For small amplitude the fluid acts as an added mass and a radiation damping: with the exact linear
compressible answer for a cylinder oscillating at kR = 0.169 (w = 2, c = sqrt(1.4)), F = -(1.0535 rho pi R^2) x'' - (0.0467 rho pi R^2 w) x', the body obeys
(m + m_a) x'' + c_r x' + k x = 0, with k chosen so that sqrt(k / (m + m_a)) = 2. The mass ratio m / (rho pi R^2) is 10, 1 and 0.3: the last one is
where an explicit loose coupling is most exposed (the added mass is three times the body's). The start is impulsive: the momentum of the body is
shared with the added mass, v(0+) = v0 m / (m + m_a), so the reference amplitude is v(0+) / w. The fit runs to t = 7 (two periods): the sides do not
reflect, which the 0.6 s window of the prescribed-motion test (open boundaries) could not afford.

Bands, set after the first runs (20 cells per diameter: frequency 0.1 / 0.3 / 0.4 % off, amplitude 0.3 / 1.1 / 0.4 % off, rms error of x(t) 0.8 / 5.4 / 8.1 %
of the amplitude for mass ratios 10 / 1 / 0.3; at 40 cells per diameter frequency 0.0 / 0.1 / 0.1 %, amplitude 0.3 / 0.7 / 1.4 %, rms 0.6 / 2.0 / 4.1 %): the
fitted frequency within 1 %, the fitted amplitude within 3 %, the rms error of x(t) within 12 % of the amplitude at 20 cells per diameter and within 6 % at
40; no divergence at any mass ratio; the rms error at 40 cells per diameter at most 0.8 of the one at 20 (mass ratios 1 and 0.3); the whole time-step error
(fluid scheme and coupling together: the same run at CFL 0.4 and 0.1, 1.1 % of the amplitude when measured) at most 2 % of the amplitude, so the coupling's
own error is below that. On the adaptive mesh (60^2 base cells and two levels, the finest level the same 240^2 mesh), bands set after the first run
(frequency 0.1 / 0.6 / 1.1 % off, amplitude 0.2 / 0.1 / 3.5 % off, rms error 2.1 / 9.5 / 11.3 % of the amplitude at mass ratios 10 / 1 / 0.3, 1.4 / 4.4 / 4.2 % from the
uniform run): frequency within 1.5 %, amplitude within 4 %, rms error within 12 %, within 5 % of the uniform run. The cells the body leaves keep a stale ghost
state on the adaptive mesh (no fresh-cell fill yet), which is where the larger error at low mass ratio most likely comes from. The fitted damping is not a band: the reference radiation damping is the one of a steady oscillation at w = 2, and the start
excites a band of frequencies (fitted 0.040 / 0.067 against the reference 0.023 / 0.0345 at mass ratios 1 / 0.3 and 20 cells per diameter, 0.024 / 0.045 at 40).
"""
import argparse
import math

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_spring"
R = 0.1
RHO_PR2 = math.pi * R * R
M_ADDED, C_RAD, W = 1.0535 * RHO_PR2, 0.0467 * RHO_PR2 * 2.0, 2.0   # exact linear compressible added mass and radiation damping at kR = 0.169
V0 = 0.04
T_END = 7.0
CASE = """[case]
name = "Spring Cylinder"
description = "a cylinder on a spring in a gas at rest (inviscid)"
gui = false
[mesh]
Lx = 2.4
Ly = 2.4
nx = 240
ny = 240
[physics]
use_viscous = false
[boundaries]
bc_xmin = "non_reflecting"
bc_xmax = "non_reflecting"
bc_ymin = "non_reflecting"
bc_ymax = "non_reflecting"
rho_in = 1.0
p_out = 1.0
[numerics]
reconstruction = "muscl"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.4
[run]
t_end = 7.0
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
state = { rho = 1.0, u = 0.0, v = 0.0, p = 1.0 }
"""


def run(ratio, cells, cfl=0.4, *extra, tag=""):
    """Run one mass ratio; returns the monitor (t, body_x)."""
    m = ratio * RHO_PR2
    k = (m + M_ADDED) * W * W
    out = study.study_dir(STUDY) / f"mu{ratio:g}_n{cells}_cfl{cfl:g}{tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.toml").write_text(CASE)
    fenice.run(str(out), f"nx={cells}", f"ny={cells}", f"ibm_mass={m!r}", f"ibm_stiff_x={k!r}", f"ibm_vx={V0}", f"cfl={cfl}", *extra, out_root=out / "run")
    d = fenice.read_monitor(study.case_output(out / "run"))
    return d["t"], d["body_x"]


def reference(ratio, t):
    """x(t) of the linear ODE with the added mass and the radiation damping, and its frequency, decay rate and amplitude."""
    m = ratio * RHO_PR2
    meff = m + M_ADDED
    sigma = C_RAD / (2.0 * meff)
    wd = math.sqrt(W * W - sigma * sigma)
    amp = V0 * m / meff / wd
    return amp * np.exp(-sigma * t) * np.sin(wd * t), wd, sigma, amp


def fit(t, x, wd, sigma):
    """Least-squares x = a exp(-s t) sin(w t) around the reference frequency and decay rate: (w, s, a)."""
    best = (math.inf, 0.0, 0.0, 0.0)
    for w in np.linspace(0.9 * wd, 1.1 * wd, 201):
        for s in np.linspace(-0.5, 3.0, 71) * sigma:
            b = np.exp(-s * t) * np.sin(w * t)
            a = (b @ x) / (b @ b)
            r = float(np.sum((x - a * b) ** 2))
            if r < best[0]:
                best = (r, w, s, a)
    return best[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", action="store_true", help="also 40 cells per diameter and the time-step check")
    args = ap.parse_args()
    ok = True
    err = {}
    for cells in (240, 480) if args.full else (240,):
        for ratio in (10, 1, 0.3):
            t, x = run(ratio, cells)
            xe, wd, sigma, amp = reference(ratio, t)
            w, s, a = fit(t, x, wd, sigma)
            rms = float(np.sqrt(np.mean((x - xe) ** 2))) / amp
            err[ratio, cells] = rms
            good = abs(w / wd - 1) < 0.01 and abs(a / amp - 1) < 0.03 and rms < (0.12 if cells == 240 else 0.06) and t[-1] > T_END - 0.1
            ok = ok and good
            print(f"mass ratio {ratio:4g}, {cells // 12:2d} cells/D: w {100 * (w / wd - 1):+5.2f} %, amplitude {100 * (a / amp - 1):+5.2f} %, "
                  f"decay {s:.4f} (reference {sigma:.4f}), rms error {100 * rms:4.1f} % of the amplitude  {'ok' if good else 'OUT OF BAND'}", flush=True)
    # the same body on an adaptive mesh: 60^2 base cells and two levels (the finest level is the 240^2 mesh); the band follows the body, the loads come
    # from one residual of the whole mesh per step. Same bands as the uniform mesh at 20 cells per diameter, and close to the uniform run itself
    for ratio in (10, 1, 0.3):
        t, x = run(ratio, 60, 0.4, "use_amr=true", "amr_max_level=2", "amr_refine_every=4", tag="_amr")
        xe, wd, sigma, amp = reference(ratio, t)
        w, s, a = fit(t, x, wd, sigma)
        rms = float(np.sqrt(np.mean((x - xe) ** 2))) / amp
        tu, xu = run(ratio, 240)
        dev = float(np.sqrt(np.mean((x - np.interp(t, tu, xu)) ** 2))) / amp
        good = abs(w / wd - 1) < 0.015 and abs(a / amp - 1) < 0.04 and rms < 0.12 and dev < 0.05 and t[-1] > T_END - 0.1
        ok = ok and good
        print(f"AMR, mass ratio {ratio:4g}: w {100 * (w / wd - 1):+5.2f} %, amplitude {100 * (a / amp - 1):+5.2f} %, rms error {100 * rms:4.1f} % of the amplitude, "
              f"{100 * dev:.2f} % from the uniform run  {'ok' if good else 'OUT OF BAND'}", flush=True)
    if args.full:
        for ratio in (1, 0.3):
            fall = err[ratio, 480] / err[ratio, 240]
            ok = ok and fall < 0.8
            print(f"mass ratio {ratio:4g}: error 20 -> 40 cells per diameter x {fall:.2f}  {'ok' if fall < 0.8 else 'OUT OF BAND'}", flush=True)
        ta, xa = run(0.3, 240, 0.4)
        tb, xb = run(0.3, 240, 0.1)
        _, _, _, amp = reference(0.3, ta)
        diff = float(np.sqrt(np.mean((xa - np.interp(ta, tb, xb)) ** 2))) / amp
        ok = ok and diff < 0.02
        print(f"mass ratio 0.3, CFL 0.4 against 0.1: rms difference {100 * diff:.2f} % of the amplitude (time step of the fluid and the coupling together)  "
              f"{'ok' if diff < 0.02 else 'OUT OF BAND'}", flush=True)
        # time outputs clamp a step to the time left before them, a much shorter step than its neighbours: the trajectory must not feel it
        tc, xc = run(0.3, 240, 0.4, "write_interval_time=0.0517", "save_interval=1000000000", tag="_outputs")
        diff = float(np.sqrt(np.mean((xa - np.interp(ta, tc, xc)) ** 2))) / amp
        ok = ok and diff < 0.01
        print(f"mass ratio 0.3, with time outputs every 0.0517 (clamped steps): rms difference {100 * diff:.2f} % of the amplitude  "
              f"{'ok' if diff < 0.01 else 'OUT OF BAND'}", flush=True)
    print("ALL IN BAND" if ok else "OUT OF BAND")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
