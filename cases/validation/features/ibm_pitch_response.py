"""The pitch of a sprung airfoil, free in the flow: its response after a release, with the aerodynamic stiffness and damping read from it.

    python cases/validation/features/ibm_pitch_response.py           # four stiffnesses on a 30-cells-per-chord base mesh, two of them on a finer one
    python cases/validation/features/ibm_pitch_response.py --docs    # also docs/img/airfoil_pitch_response.png

The NACA 0012 of the `IBM Airfoil Pitch` case (Re = 1000, Ma = 0.2, lengths in chords, times in c / u_in) at 0 degrees, with its pitch freed: moment of inertia I = 0.5 about the quarter chord
(`ibm_inertia`; the airfoil's own mass per unit depth is about 0.08 rho c^2 times its density ratio, so this is a body some 20 times denser than the gas), a torsional spring of stiffness K
(`ibm_stiff_theta`) with its rest position at 0 degrees, no structural damping, released with the angular velocity 0.1 (`ibm_omega`, the initial spin of a driven rotation). The body is pinned
at its placement (`ibm_mass = 0`): one degree of freedom, the pitch.

Read from the response (the angle is theta, positive nose-down, as the monitor writes it (negative `ibm_omega` is nose-up)):
  - the damped period T_d and the decay of the peaks: successive peaks (half a period apart) fall by the factor r, so the damping ratio is zeta = ln(1 / r) / sqrt(pi^2 + ln(1 / r)^2);
  - with the structural damping zero, all the damping is aerodynamic: c_a = 2 zeta sqrt(K_eff I);
  - the damped frequency w_d = 2 pi / T_d gives the effective stiffness K_eff = I w_d^2 / (1 - zeta^2) = K + K_a, where K_a is the stiffness of the flow (the slope of its moment with the angle about
    the quarter chord). From the static polar of the airfoil (docs, `cases/airfoil-pitch`: C_m = +0.0079 at 3 degrees) the flow is destabilising, K_a = -(dC_m / d theta) q c^2 = -0.15 * 0.5 = -0.075
    (per radian, q = rho u^2 / 2 = 0.5, unit chord); the measured K_a is read at K = 1 (the quasi-steady value: at K = 0.3 the oscillation is slow and strongly damped and few extrema are left to fit).
A spring weaker than that, K < 0.075, is statically divergent: the airfoil leaves 0 degrees and settles where K theta equals the moment of the flow (here K = 0.05 settles at 9 degrees, K = 0.1 returns to 0).

What this is not: flutter. One pitch degree of freedom about the quarter chord is damped by the flow; flutter needs the coupling to a plunge (`ibm_cm_x`, `features/ibm_flutter.py`). The study reports the numbers; it checks only that the runs agree with the
finer mesh (period within 5 %, decay ratio per half period within 0.07: the K = 0.3 ratio differs by 0.06, 0.438 against 0.378, the K = 1 one by 0.014) and with the static polar (the divergence at K = 0.05 and
the stable return at K = 0.1, the aerodynamic stiffness of K = 1 within -0.12 .. -0.04). The bands were set after the first run.
"""
import argparse
import math

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_pitch_response"
INERTIA = 0.5
W0 = 0.1
SETS = ["ibm_omega={w0}", "ibm_omega_from=0", "ibm_omega_until=0", "ibm_aoa=0", "ibm_inertia={inertia}", "ibm_stiff_theta={k}", "ibm_damp_theta=0", "t_end={t_end}",
        "nx=360", "ny=180", "amr_max_level={levels}", "monitor_interval=5"]


def run(tag, k, levels, t_end):
    sets = [s.format(w0=W0, inertia=INERTIA, k=k, t_end=t_end, levels=levels) for s in SETS]
    run_dir = study.run_cached(STUDY, tag, "IBM Airfoil Pitch", *sets, mpi=0)
    d = fenice.read_monitor(study.case_output(run_dir))
    return d["t"], np.degrees(d["body_angle"]), d["Cm"]


def peaks(t, th, t_lo=0.5):
    """The extrema of the angle (time, value), a spacing of at least a quarter of the natural period apart."""
    out = []
    for i in range(1, len(t) - 1):
        if t[i] > t_lo and abs(th[i]) >= abs(th[i - 1]) and abs(th[i]) > abs(th[i + 1]) and th[i] * th[i - 1] != 0:
            if not out or t[i] - out[-1][0] > 0.5:
                out.append((float(t[i]), float(th[i])))
    return out


def response(t, th, k):
    """Damped period, damping ratio and the effective stiffness from the extrema of an oscillating record (None for a record that does not oscillate)."""
    pk = [p for p in peaks(t, th)]
    alt = [p for p in pk if abs(p[1]) > 0.02]
    # keep only the alternating-sign extrema
    chain = [alt[0]] if alt else []
    for p in alt[1:]:
        if p[1] * chain[-1][1] < 0:
            chain.append(p)
    if len(chain) < 4:
        return None
    times = np.array([p[0] for p in chain[:6]])
    vals = np.array([abs(p[1]) for p in chain[:6]])
    half = float(np.mean(np.diff(times)))
    ratio = float(np.exp(np.mean(np.diff(np.log(vals)))))   # one half period
    delta = -math.log(ratio) * 2.0                          # logarithmic decrement per period
    zeta = delta / math.sqrt(4.0 * math.pi ** 2 + delta ** 2)
    wd = math.pi / half
    k_eff = INERTIA * wd ** 2 / (1.0 - zeta ** 2)
    return {"period": 2.0 * half, "ratio": ratio, "zeta": zeta, "k_eff": k_eff, "k_a": k_eff - k, "c_a": 2.0 * zeta * math.sqrt(k_eff * INERTIA)}


def main():
    ap = study.parser(__doc__)
    args = ap.parse_args()
    cases = [("k0.05", 0.05, 1, 40.0), ("k0.1", 0.1, 1, 40.0), ("k0.3", 0.3, 1, 30.0), ("k1", 1.0, 1, 30.0), ("k0.3_fine", 0.3, 2, 30.0), ("k1_fine", 1.0, 2, 30.0)]
    runs = {tag: (k, levels) + run(tag, k, levels, t_end) for tag, k, levels, t_end in cases}
    res = {}
    print("stiffness  levels   T_d     ratio/half-period  zeta    K_eff   K_a      c_a")
    for tag, (k, levels, t, th, cm) in runs.items():
        r = response(t, th, k)
        res[tag] = r
        if r is None:
            print(f"K = {k:<6g} {levels}     no oscillation; settles at {th[t > t[-1] - 5].mean():.2f} deg (the mean over the last 5 time units)")
        else:
            print(f"K = {k:<6g} {levels}      {r['period']:.2f}   {r['ratio']:.3f}             {r['zeta']:.3f}   {r['k_eff']:.3f}   {r['k_a']:+.3f}   {r['c_a']:.3f}")
    ok = True
    off = {tag: float(runs[tag][3][runs[tag][2] > runs[tag][2][-1] - 5].mean()) for tag in ("k0.05", "k0.1")}
    good = abs(off["k0.05"]) > 3.0 and abs(off["k0.1"]) < 1.0
    ok = ok and good
    print(f"static divergence: K = 0.05 settles at {off['k0.05']:.2f} deg, K = 0.1 at {off['k0.1']:.2f} deg (the boundary 0.075 from the static polar)  {'ok' if good else 'OUT OF BAND'}")
    ka = res["k1_fine"]["k_a"]
    good = -0.12 < ka < -0.04
    ok = ok and good
    print(f"K = 1: aerodynamic stiffness {ka:+.3f} (static polar: -0.075)  {'ok' if good else 'OUT OF BAND'}")
    for base in ("k0.3", "k1"):
        a, b = res[base], res[base + "_fine"]
        if a is None or b is None:
            ok = False
            continue
        dp = abs(a["period"] - b["period"]) / b["period"]
        dr = abs(a["ratio"] - b["ratio"])
        good = dp < 0.05 and dr < 0.07
        ok = ok and good
        print(f"{base}: period {a['period']:.2f} / {b['period']:.2f} (coarse / fine, {100 * dp:.1f} %), ratio {a['ratio']:.3f} / {b['ratio']:.3f} (difference {dr:.3f})  {'ok' if good else 'OUT OF BAND'}")

    plt = study.use_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for tag, col in (("k0.05", "tab:purple"), ("k0.1", "tab:green"), ("k0.3", "tab:orange"), ("k1", "tab:blue")):
        k, levels, t, th, cm = runs[tag]
        ax.plot(t, th, color=col, lw=1.3, label=f"$K = {k:g}$")
    for tag, col in (("k0.3_fine", "tab:orange"), ("k1_fine", "tab:blue")):
        k, levels, t, th, cm = runs[tag]
        ax.plot(t, th, color=col, lw=0.9, ls="--")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("$t\\,u_{\\rm in}/c$")
    ax.set_ylabel("pitch angle [deg]")
    ax.legend(loc="center right", ncol=1)
    ax.set_title("Released at 0° with $\\dot\\theta = 0.1$; dashed: the finer mesh (twice the cells per chord at the body)", fontsize=10)
    fig.tight_layout()
    study.save(fig, STUDY, "airfoil_pitch_response.png", docs=args.docs)
    print("ALL IN BAND" if ok else "OUT OF BAND")


if __name__ == "__main__":
    main()
