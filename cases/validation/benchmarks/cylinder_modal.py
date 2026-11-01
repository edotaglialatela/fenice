"""Modal analysis of the cylinder wake at Re = 100: POD, DMD and the Strouhal number against the lift signal (no new run).

    python cases/validation/benchmarks/cylinder_modal.py           # results/validation/cylinder_modal/
    python cases/validation/benchmarks/cylinder_modal.py --docs    # also copy cylinder_modal.png to docs/img/

Reads runs that already exist and starts no simulation (if one is missing it says which study makes it):
  * the cylinder_loads run (400 x 200, t = 0..80, one snapshot per 1000 steps = 0.366 time units, about 3.3 per shedding
    period): POD and exact DMD of the transverse velocity v over the snapshots with t >= 50, and the lift / drag monitor;
  * the cylinder_low_blockage runs l3 and l4 (AMR, to t = 60): their snapshots are 4 time units apart, more than three
    periods, so no spatial DMD is possible; only their lift monitors are used (time-delay DMD of the scalar record).
Snapshots written every N steps of a run with an adaptive step are uniform in time only to about 1e-3; the jitter is
printed, its phase error per step is 2 pi f dt jitter.

Bands, written before the first look at these data (St = f D / u_inf, D = 0.2, u_inf = 1; the record t >= 50 is about 25
periods, the Fourier resolution D / T of the lift peak is 0.0067):
  1. the St of the leading DMD pair of v agrees with the St of the lift spectrum (Hann FFT peak of the same record, the
     estimate of cylinder_spectrum.py) within 0.004;
  2. that pair is a limit cycle: |growth rate| / (2 pi f) <= 0.02;
  3. POD of v: the first two modes carry at least 0.85 of the fluctuation energy and their energies differ by at most 10 %
     of their sum (a travelling structure appears as a pair of equal energies);
  4. the time coefficients of POD modes 1 and 2 are in quadrature at the DMD frequency: the phase difference is 90 deg
     +- 15 deg;
  5. the time-delay DMD (Hankel, rank 2) of the lift of the same record gives the same St within 0.004, and that of the
     drag gives twice the lift frequency within 2 %;
  6. the lift monitors of the low-blockage runs l3 and l4, t >= 25: the time-delay DMD St agrees within 0.004 with both
     the spectrum peak and the zero-crossing estimate of cylinder_low_blockage.py, and |growth| / (2 pi f) <= 0.02;
  7. the DMD mode (real and imaginary parts) lies in the plane of the two leading POD modes: at least 0.95 of its
     weighted norm.
The literature value (St = 0.164-0.166 at Re = 100) is printed, not a band: the 4 x 2 domain of the cylinder_loads run
blocks 10 % of the channel (the low-blockage runs 1.7 %), and confinement raises St. The modes of the run are written as .vtr (pod1, pod2, dmd_re, dmd_im) for the workbench Postprocessing node.
"""
import sys
from pathlib import Path

import numpy as np

import fenice
from fenice import modal, spectra, study

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cylinder_low_blockage as lb   # noqa: E402  (sibling study: its spectrum and zero-crossing St)
import cylinder_spectrum as cs       # noqa: E402  (sibling study: the lift-spectrum St)

STUDY = "cylinder_modal"
D, U = 0.2, 1.0
T_START = 50.0
REF = cs.REF
B_ST, B_GROWTH, B_ENERGY, B_PAIR, B_PHASE, B_PLANE = 0.004, 0.02, 0.85, 0.10, 15.0, 0.95
DT_SIG, ROWS = 0.05, 60   # the time-delay DMD of a record: resampled step, rows of the Hankel matrix (3 time units)


def need(path, study_name):
    if not Path(path).exists():
        print(f"missing {path}: make it with  python cases/validation/run_all.py {study_name}  (this study starts no run)")
        sys.exit(2)
    return Path(path)


def hankel_dmd(t, y, rank=2):
    """(St list, growth/(2 pi f) of the leading pair) of a scalar record by time-delay DMD."""
    tu, yu, _ = spectra.resample_uniform(t, y, DT_SIG)
    d = modal.dmd(modal.delay_embed(yu - yu.mean(), ROWS), DT_SIG, rank=rank)
    i = int(np.argmax(d["frequency"]))   # the member of the conjugate pair with positive frequency
    f = d["frequency"][i]
    return f * D / U, d["growth"][i] / (2 * np.pi * f)


def phase_at(t, a, f):
    """Phase [deg] of a signal at frequency f: least-squares fit of c cos(2 pi f t) + s sin(2 pi f t)."""
    A = np.column_stack([np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)])
    c, s = np.linalg.lstsq(A, a - a.mean(), rcond=None)[0]
    return np.degrees(np.arctan2(s, c))


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    out_dir = study.study_dir(STUDY)
    ok = {}

    # ---- the cylinder_loads run: snapshots and monitor
    out = study.case_output(need(study.study_dir("cylinder_loads") / "run", "cylinder_loads"))
    d = fenice.read_monitor(out)
    t, cl, cd = d["t"], d["Cl"], d["Cd"]
    late = t >= T_START
    st_l, a_l, span = cs.spectrum(t[late], cl[late])
    st_lift, _ = cs.peak(st_l, a_l, 0.05, 0.4)
    print(f"cylinder_loads, record t = {T_START:g}..{t[-1]:g} ({span:.1f} time units, {span * st_lift / D:.1f} periods): "
          f"lift spectrum St = {st_lift:.4f}; literature St = {REF[0]}-{REF[1]}")

    series = fenice.read_pvd(out)
    times = np.array([s[0] for s in series])
    i0 = int(np.argmax(times >= T_START))
    i1 = len(times)
    step = np.median(np.diff(times))
    if abs(times[-1] - times[-2] - step) > 1e-2 * step:   # the last snapshot is written at t_end, not on the cadence
        i1 -= 1
    snap = modal.load_snapshots(out, "Velocity", component=1, first=i0, last=i1, tol=5e-3)
    X, dt, w = snap["X"], snap["dt"], snap["weights"]
    print(f"snapshots: {X.shape[1]} of v (t = {snap['times'][0]:.2f}..{snap['times'][-1]:.2f}), dt = {dt:.4f} "
          f"({1 / (dt * st_lift / D):.2f} per period), step jitter {snap['jitter']:.1e} "
          f"(phase error per step {2 * np.pi * st_lift / D * dt * snap['jitter']:.1e} rad), {X.shape[0]} cells")

    # ---- POD
    pod = modal.pod(X, w, rank=8)
    E = pod["energies"]
    tot = np.sum(modal.pod(X, w)["energies"])
    frac = E / tot
    print("POD energy fractions: " + ", ".join(f"{f:.4f}" for f in frac[:6]))
    ok["pod energy"] = frac[0] + frac[1] >= B_ENERGY
    ok["pod pair"] = abs(E[0] - E[1]) / (E[0] + E[1]) <= B_PAIR
    print(f"  modes 1+2: {frac[0] + frac[1]:.4f} (band >= {B_ENERGY}); pair imbalance "
          f"{abs(E[0] - E[1]) / (E[0] + E[1]):.4f} (band <= {B_PAIR})")

    # ---- DMD of the mean-removed snapshots
    Xc = X - pod["mean"][:, None]
    dm = modal.dmd(Xc, dt, rank=2)
    k = int(np.argmax(dm["frequency"]))
    st_dmd = dm["frequency"][k] * D / U
    gr = dm["growth"][k] / (2 * np.pi * dm["frequency"][k])
    print(f"DMD (rank 2): St = {st_dmd:.4f} (lift spectrum {st_lift:.4f}, difference {st_dmd - st_lift:+.4f}), "
          f"|lambda| = {abs(dm['eigenvalues'][k]):.5f}, growth/(2 pi f) = {gr:+.4f}")
    ok["dmd st"] = abs(st_dmd - st_lift) <= B_ST
    ok["dmd limit cycle"] = abs(gr) <= B_GROWTH
    d8 = modal.dmd(Xc, dt, rank=8)
    pos = d8["frequency"] > 0
    print("DMD (rank 8) frequencies / lift frequency: " + ", ".join(
        f"{f * D / U / st_lift:.3f} ({g / (2 * np.pi * f):+.3f})" for f, g in zip(d8["frequency"][pos], d8["growth"][pos])))

    # ---- quadrature of the POD coefficients at the DMD frequency
    f_dmd = dm["frequency"][k]
    ph = [phase_at(snap["times"], pod["coefficients"][j], f_dmd) for j in (0, 1)]
    dphi = abs((ph[0] - ph[1] + 180.0) % 360.0 - 180.0)
    print(f"POD coefficients 1, 2 at f = {f_dmd:.4f}: phase difference {dphi:.1f} deg (band 90 +- {B_PHASE:g})")
    ok["pod quadrature"] = abs(dphi - 90.0) <= B_PHASE

    # ---- the DMD mode against the POD plane
    m = dm["modes"][:, k]
    plane = []
    for part in (np.real(m), np.imag(m)):
        wn = w / w.sum()
        proj = sum(np.sum(wn * part * pod["modes"][:, j]) * pod["modes"][:, j] for j in (0, 1))
        plane.append(np.sqrt(np.sum(wn * proj ** 2) / np.sum(wn * part ** 2)))
    print(f"DMD mode in the POD plane: real part {plane[0]:.4f}, imaginary part {plane[1]:.4f} (band >= {B_PLANE})")
    ok["dmd mode in pod plane"] = min(plane) >= B_PLANE

    # ---- time-delay DMD of the lift and drag records
    sth_cl, g_cl = hankel_dmd(t[late], cl[late])
    sth_cd, g_cd = hankel_dmd(t[late], cd[late])
    print(f"time-delay DMD of the lift: St = {sth_cl:.4f} (growth/(2 pi f) {g_cl:+.4f}); of the drag: frequency "
          f"{sth_cd / sth_cl:.4f} x the lift frequency")
    ok["hankel lift st"] = abs(sth_cl - st_lift) <= B_ST
    ok["hankel drag harmonic"] = abs(sth_cd / sth_cl - 2.0) <= 0.02

    # ---- the low-blockage lift monitors
    lbres = {}
    lb_dir = study.study_dir("cylinder_low_blockage")
    for name in ("l3", "l4"):
        dd = fenice.read_monitor(study.case_output(need(lb_dir / name, "cylinder_low_blockage")))
        tt, cc = dd["t"], dd["Cl"]
        m_ = tt >= lb.T_ST
        f_, a_ = lb.spectrum(tt[m_], cc[m_])
        sel = (f_ > 0.05) & (f_ < 0.4)
        st_fft = f_[sel][np.argmax(a_[sel])]
        st_zc, nper = lb.st_crossings(tt[m_], cc[m_])
        st_h, g_h = hankel_dmd(tt[m_], cc[m_])
        lbres[name] = (st_fft, st_zc, st_h, g_h)
        print(f"{name}: St = {st_h:.4f} (time-delay DMD, growth/(2 pi f) {g_h:+.4f}), {st_fft:.4f} (spectrum), "
              f"{st_zc:.4f} (zero crossings, {nper} periods)")
        ok[f"{name} agreement"] = abs(st_h - st_fft) <= B_ST and abs(st_h - st_zc) <= B_ST and abs(g_h) <= B_GROWTH

    # ---- modes for the workbench
    ny, nx = snap["shape"]
    xn, yn = snap["x"], snap["y"]
    for nm, field in (("pod1", pod["modes"][:, 0]), ("pod2", pod["modes"][:, 1]),
                      ("dmd_re", np.real(m)), ("dmd_im", np.imag(m))):
        modal.export_modes_vtr(out_dir / f"{nm}.vtr", field, xn, yn, nm)

    # ---- figure
    xc, yc = 0.5 * (xn[1:] + xn[:-1]), 0.5 * (yn[1:] + yn[:-1])
    fig = plt.figure(figsize=(13, 8.6))
    gs = fig.add_gridspec(3, 3, height_ratios=(1.15, 1.15, 1.0), hspace=0.55, wspace=0.28)
    ext = (xn[0], xn[-1], yn[0], yn[-1])
    for j, (title, field) in enumerate((("POD mode 1", pod["modes"][:, 0]), ("POD mode 2", pod["modes"][:, 1]),
                                        ("DMD mode, real part", np.real(m)))):
        ax = fig.add_subplot(gs[0, j])
        a = field.reshape(ny, nx)
        lim = np.abs(a).max()
        ax.imshow(a, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="equal")
        ax.set_title(f"{title} of $v$" + (f" ({frac[j]:.1%} of the energy)" if j < 2 else f" ($\\mathrm{{St}}={st_dmd:.4f}$)"),
                     fontsize=9.5)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(np.arange(1, len(frac) + 1), frac, "o-", color="#2980b9")
    ax.set_xlabel("POD mode")
    ax.set_ylabel("energy fraction")
    ax.set_title("POD spectrum of $v$ ($t \\geq 50$)", fontsize=10)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(pod["coefficients"][0], pod["coefficients"][1], ".", color="#8e44ad", ms=4)
    ax.set_xlabel("coefficient of mode 1")
    ax.set_ylabel("coefficient of mode 2")
    ax.set_title(f"Phase portrait: quadrature, {dphi:.0f}$^\\circ$", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax = fig.add_subplot(gs[1, 2])
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color="0.5", lw=1)
    lam = d8["eigenvalues"]
    ax.plot(lam.real, lam.imag, "o", color="#c0392b", ms=6, label="DMD, rank 8")
    ax.plot(dm["eigenvalues"].real, dm["eigenvalues"].imag, "x", color="k", ms=8, label="rank 2")
    ax.set_aspect("equal")
    ax.set_xlabel("Re $\\lambda$")
    ax.set_ylabel("Im $\\lambda$")
    ax.set_title("DMD eigenvalues, unit circle", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax = fig.add_subplot(gs[2, :])
    labels = ["lift spectrum\n(cylinder_loads)", "snapshot DMD", "lift time-delay DMD",
              "l3 time-delay DMD", "l3 spectrum", "l4 time-delay DMD", "l4 spectrum", "l4 zero crossings"]
    vals = [st_lift, st_dmd, sth_cl, lbres["l3"][2], lbres["l3"][0], lbres["l4"][2], lbres["l4"][0], lbres["l4"][1]]
    ax.axhspan(*REF, color="#27ae60", alpha=0.35, lw=0, label=f"literature {REF[0]}-{REF[1]}")
    ax.plot(range(len(vals)), vals, "o", color="#2980b9", ms=7)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("St")
    ax.set_title("Strouhal number at Re = 100: 10 % blockage (first three) and 1.7 % blockage (the AMR runs l3, l4)",
                 fontsize=10)
    ax.set_ylim(0.1635, 0.1755)
    ax.legend(loc="center right", fontsize=8)
    study.save(fig, STUDY, "cylinder_modal.png", args.docs)

    passed = all(ok.values())
    for k_, v in ok.items():
        print(f"  [{'OK' if v else 'FAIL'}] {k_}")
    fenice.write_summary(STUDY, {
        "passed": bool(passed), "bands": {k_: bool(v) for k_, v in ok.items()},
        "St": {"lift_spectrum": float(st_lift), "snapshot_dmd": float(st_dmd), "lift_time_delay_dmd": float(sth_cl),
               **{f"{n}_{k_}": float(v_) for n, r in lbres.items() for k_, v_ in zip(("spectrum", "zero_crossings",
                                                                                      "time_delay_dmd"), r[:3])}},
        "pod_energy_fractions": [float(f) for f in frac[:6]], "step_jitter": snap["jitter"]})
    print("ALL BANDS PASSED" if passed else "BANDS FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
