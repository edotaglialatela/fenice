"""Pitch and plunge of a sprung airfoil in the flow, against the inviscid flutter boundary of the typical section.

    python cases/validation/features/ibm_flutter.py           # four reduced velocities on a 30-cells-per-chord base mesh, two of them on a finer one
    python cases/validation/features/ibm_flutter.py --docs    # also docs/img/airfoil_pitch_flutter.png

The NACA 0012 of the `IBM Airfoil Pitch` case (Re = 1000, Ma = 0.2, lengths in chords c, times in c / u_in) at 0 degrees with its pitch and plunge free: mass m = 4, moment of inertia about the elastic
axis I = 0.5, the elastic axis (pivot) at the quarter chord, the centre of mass 0.15 c behind it (`ibm_cm_x`), springs in plunge and pitch with the ratio of the uncoupled frequencies
w_h / w_theta = 0.5, a stiff spring in x, no structural damping, released with the angular velocity 0.05. The reduced velocity U* = u_in / (b w_theta), b = c / 2, is set by the springs:
K_theta = I w_theta^2, K_h = m w_h^2.

Reference (inviscid, incompressible, free air): Theodorsen's unsteady loads of the thin section, harmonic motion: the flutter reduced velocity is where an eigenvalue of the aeroelastic system
crosses the real axis, found by a scan in the reduced frequency k; this section flutters at U* = 1.70 (w / w_theta = 0.87, k = 0.51).

What the runs show is the growth or decay of the pitch amplitude (the factor by which successive peaks change, half a period apart, while the pitch is below 20 degrees) and the frequency. They are reported against
the inviscid boundary and not gated: the flow here is viscous (Re = 1000), compressible and between slip walls three chords away, which the reference is not. The growing runs are ended before the airfoil
nears a wall (at U* = 3 it reaches the top one near t = 65, and the run stops there).
"""
import math

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_flutter"
MASS, INERTIA, CM, RATIO, W0, B = 4.0, 0.5, 0.15, 0.5, 0.05, 0.5


def inviscid_boundary(a=-0.5, rho=1.0):
    """(U*, k, w / w_theta) at the first crossing of an eigenvalue through the real axis (Theodorsen, harmonic motion q = (plunge down, pitch nose up))."""
    from scipy.special import hankel2
    m, ine, s = MASS, INERTIA, MASS * CM

    def lam(k):
        h0, h1 = hankel2(0, k), hankel2(1, k)
        c = h1 / (h1 + 1j * h0)
        u = B / k   # U / w
        w34 = np.array([1j, u + B * (0.5 - a) * 1j])   # downwash at the three-quarter chord, per unit w
        lift = np.pi * rho * B ** 2 * np.array([-1.0, u * 1j + B * a]) + 2 * np.pi * rho * u * B * c * w34
        mom = (np.pi * rho * B ** 2 * np.array([-B * a, -u * B * (0.5 - a) * 1j + B ** 2 * (0.125 + a * a)])
               + 2 * np.pi * rho * u * B ** 2 * (a + 0.5) * c * w34)
        aero = np.array([-lift, mom])
        mass = np.array([[m, s], [s, ine]])
        stiff = np.diag([m * RATIO ** 2, ine])   # in units of w_theta^2
        return np.linalg.eigvals(np.linalg.solve(mass + aero, stiff))   # w^2 / w_theta^2

    prev = None
    found = []
    for k in np.linspace(3.0, 0.02, 3000):
        l = lam(k)
        if prev is not None:
            if abs(l[0] - prev[0]) + abs(l[1] - prev[1]) > abs(l[1] - prev[0]) + abs(l[0] - prev[1]):
                l = l[::-1]
            for j in range(2):
                if prev[j].imag * l[j].imag < 0 and l[j].real > 0:
                    found.append((float(np.sqrt(l[j].real) / k), float(k), float(np.sqrt(l[j].real))))
        prev = l
    return min(found)


def run(tag, ustar, levels, t_end):
    wth = 1.0 / (B * ustar)
    sets = ["ibm_omega_from=0", "ibm_omega_until=0", "ibm_aoa=0", f"ibm_omega={W0}", f"ibm_mass={MASS}", f"ibm_inertia={INERTIA}", f"ibm_cm_x={CM}", "ibm_cm_y=0",
            f"ibm_stiff_theta={INERTIA * wth ** 2}", f"ibm_stiff_y={MASS * (RATIO * wth) ** 2}", "ibm_stiff_x=100", f"t_end={t_end}", "nx=360", "ny=180",
            f"amr_max_level={levels}", "monitor_interval=5"]
    run_dir = study.run_cached(STUDY, tag, "IBM Airfoil Pitch", *sets)
    d = fenice.read_monitor(study.case_output(run_dir))
    return d["t"], d["body_y"], np.degrees(d["body_angle"])


def growth(t, th, t_lo=5.0, th_max=20.0, floor=0.25):
    """The growth of the pitch while the motion is small: the peaks of |theta| (degrees, above `floor`) from t_lo until the pitch first exceeds th_max, the factor per half period
    (the slope of the logarithm of the peaks, by least squares) and the angular frequency (pi over the mean spacing of the peaks)."""
    over = np.nonzero((np.abs(th) > th_max) & (t > t_lo))[0]
    t_hi = t[over[0]] if len(over) else t[-1]
    pk = []
    for i in range(1, len(t) - 1):
        if t_lo < t[i] <= t_hi and abs(th[i]) >= abs(th[i - 1]) and abs(th[i]) > abs(th[i + 1]) and abs(th[i]) > floor:
            if not pk or t[i] - pk[-1][0] > 1.0:
                pk.append((float(t[i]), float(abs(th[i]))))
    if len(pk) < 3:
        return pk, None, None, t_hi
    n = np.arange(len(pk))
    factor = float(np.exp(np.polyfit(n, np.log([p[1] for p in pk]), 1)[0]))
    omega = float(math.pi / np.mean(np.diff([p[0] for p in pk])))
    return pk, factor, omega, t_hi


def main():
    ap = study.parser(__doc__)
    args = ap.parse_args()
    ustar_f, k_f, w_f = inviscid_boundary()
    print(f"inviscid flutter (Theodorsen): U* = {ustar_f:.2f}, k = {k_f:.2f}, w / w_theta = {w_f:.2f}")
    # (the growing runs end before the airfoil nears the walls three chords away: at U* = 3 it reaches them near t = 65)
    cases = [("u1", 1.0, 1, 60.0), ("u1.5", 1.5, 1, 60.0), ("u2", 2.0, 1, 40.0), ("u3", 3.0, 1, 30.0), ("u1.5_fine", 1.5, 2, 60.0), ("u2_fine", 2.0, 2, 40.0)]
    runs = {tag: (u, levels) + run(tag, u, levels, t_end) for tag, u, levels, t_end in cases}
    print("U*    levels  peaks up to |pitch| = 20 deg   factor per half period   w / w_theta   max |plunge| / c   max |pitch| [deg]")
    for tag, (u, levels, t, y, th) in runs.items():
        pk, g, om, t_hi = growth(t, th)
        wth = 1.0 / (B * u)
        print(f"{u:<5g} {levels}       {len(pk):>2} (t <= {t_hi:5.1f})              {'-' if g is None else f'{g:.3f}':<24} {'-' if om is None else f'{om / wth:.2f}':<13} "
              f"{np.abs(y).max():<18.3f} {np.abs(th).max():.1f}")
    plt = study.use_style()
    coarse = [tag for tag in runs if not tag.endswith("_fine")]
    fig, ax = plt.subplots(len(coarse), 2, figsize=(10, 1.9 * len(coarse)), sharex="col")
    for row, tag in enumerate(coarse):
        u, levels, t, y, th = runs[tag]
        ax[row, 0].plot(t, y, color="tab:blue", lw=1.0)
        ax[row, 1].plot(t, th, color="tab:red", lw=1.0)
        if tag + "_fine" in runs:
            _, _, tf, yf, thf = runs[tag + "_fine"]
            ax[row, 0].plot(tf, yf, color="k", lw=0.8, ls="--")
            ax[row, 1].plot(tf, thf, color="k", lw=0.8, ls="--")
        ax[row, 0].set_ylabel(f"$U^* = {u:g}$\n$y/c$")
        ax[row, 1].set_ylabel("pitch [deg]")
    ax[-1, 0].set_xlabel("$t\\,u_{\\rm in}/c$")
    ax[-1, 1].set_xlabel("$t\\,u_{\\rm in}/c$")
    ax[0, 0].set_title(f"plunge (inviscid flutter at $U^* = {ustar_f:.2f}$)", fontsize=10)
    ax[0, 1].set_title("pitch", fontsize=10)
    fig.tight_layout()
    study.save(fig, STUDY, "airfoil_pitch_flutter.png", docs=args.docs)


if __name__ == "__main__":
    main()
