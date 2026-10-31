"""Spectrum of the vortex shedding behind the IBM cylinder (Re = 100): Strouhal number from the lift and drag histories.

    python cases/validation/benchmarks/cylinder_spectrum.py           # results/validation/cylinder_spectrum/
    python cases/validation/benchmarks/cylinder_spectrum.py --docs    # also docs/img/cylinder_spectrum.png

Uses the monitor of the cylinder_loads study (ibm_loads.py cylinder: 400 x 200, t = 0..80; the run is made if it is
missing). Lift oscillates at the shedding frequency f, drag at 2 f. The spectrum is a Hann-windowed FFT of the record
t >= 50, zero-padded for the peak location (the resolution itself is 1 / T of the record, a few shedding periods only);
the second panel is the drag spectrum, the third the Strouhal number from a sliding window of 12 time units, to see how
much it depends on the part of the record that is used. St = f D / u_inf with the diameter D = 0.2 and u_inf = 1.
"""
import numpy as np

import fenice
from fenice import study

D, T_START, WINDOW = 0.2, 50.0, 12.0
REF = (0.164, 0.166)   # Re = 100 (Williamson 1996; Park et al. 1998)


def spectrum(t, y, pad=32):
    """One-sided amplitude spectrum (Hann window, coherent gain corrected) on a uniform time grid."""
    tu = np.linspace(t[0], t[-1], 4 * len(t))
    c = np.interp(tu, t, y)
    c = c - c.mean()
    w = np.hanning(len(c))
    n = pad * len(c)
    amp = 2.0 * np.abs(np.fft.rfft(c * w, n)) / w.sum()
    return np.fft.rfftfreq(n, tu[1] - tu[0]) * D, amp, tu[-1] - tu[0]


def peak(st, amp, lo, hi):
    m = (st >= lo) & (st <= hi)
    i = np.argmax(np.where(m, amp, 0.0))
    return st[i], amp[i]


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    run = study.run_cached("cylinder_loads", "run", "IBM Flow over Cylinder", "t_end=80", "monitor_interval=20",
                           mpi=args.mpi or 16, force=args.force, done="*/*_monitor.csv")
    d = fenice.read_monitor(study.case_output(run))
    t, cd, cl = d["t"], d["Cd"], d["Cl"]
    late = t >= T_START
    st_l, a_l, span = spectrum(t[late], cl[late])
    st_d, a_d, _ = spectrum(t[late], cd[late])
    s1, A1 = peak(st_l, a_l, 0.05, 0.4)
    s2, A2 = peak(st_d, a_d, 0.2, 0.7)
    print(f"record t = {T_START:g}..{t[-1]:g} ({span:.1f} time units, {span * s1 / D:.1f} periods); "
          f"resolution 1/T gives dSt = {D / span:.4f}")
    print(f"lift:  St = {s1:.4f}, amplitude {A1:.3f}")
    print(f"drag:  peak at {s2:.4f} = {s2 / s1:.3f} x the lift frequency, amplitude {A2:.4f}")

    centres, sts = [], []
    for t0 in np.arange(20.0, t[-1] - WINDOW + 1e-9, 2.0):
        m = (t >= t0) & (t <= t0 + WINDOW)
        s, a, _ = spectrum(t[m], cl[m])
        centres.append(t0 + WINDOW / 2)
        sts.append(peak(s, a, 0.05, 0.4)[0])
    print("sliding window St: " + ", ".join(f"{c:g}: {s:.3f}" for c, s in zip(centres, sts)))

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle(f"Cylinder, $Re = 100$ ($400\\times200$): spectra of the loads, $t \\geq {T_START:g}$", fontsize=14,
                 fontweight="bold")
    for ax, st, a, s0, name in ((a1, st_l, a_l, s1, "$C_l$"), (a2, st_d, a_d, s2, "$C_d$")):
        ax.semilogy(st, a, color="#2980b9", lw=1.5)
        if ax is a1:
            ax.axvspan(*REF, color="#27ae60", alpha=0.35, lw=0, label="literature St")
        ax.axvline(s0, color="#c0392b", ls="--", lw=1.2, label=f"peak {s0:.3f}")
        ax.set_xlim(0, 0.6)
        ax.set_ylim(a.max() * 1e-4, a.max() * 3)
        ax.set_xlabel(r"$\mathrm{St}=fD/u_\infty$")
        ax.set_ylabel(f"amplitude of {name}")
        ax.legend(loc="upper right")
    a1.set_title("Lift")
    a2.set_title("Drag (twice the shedding frequency)")
    a3.plot(centres, sts, "o-", color="#8e44ad", ms=5)
    a3.axhspan(*REF, color="#27ae60", alpha=0.35, lw=0, label="literature")
    a3.axhline(s1, color="#c0392b", ls="--", lw=1.2, label=f"whole record, {s1:.3f}")
    a3.set_xlabel("centre of the window, $t$")
    a3.set_ylabel("St")
    a3.set_title(f"Sliding window of {WINDOW:g} time units")
    a3.legend()
    for ax in (a1, a2, a3):
        ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    study.save(fig, "cylinder_spectrum", "cylinder_spectrum.png", args.docs)


if __name__ == "__main__":
    main()
