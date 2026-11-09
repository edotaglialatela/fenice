"""3-D Taylor-Green vortex (Re = 1600): MUSCL against WENO3-Z and WENO5-Z, spectra to the Nyquist wavenumber.

    python cases/validation/benchmarks/taylor_green_3d_schemes.py            # 32^3 and 64^3, three schemes
    python cases/validation/benchmarks/taylor_green_3d_schemes.py --docs     # also docs/img/

Runs the "Taylor-Green Vortex 3D" case to t = 20 into results/validation/taylor_green_3d/<scheme>_n<N>/ (cached):
`muscl` (the case default: Van Leer, SSP-RK2), `weno3` and `weno5` (SSP-RK3, which the WENO schemes need), with the monitor and
snapshots. Figures:

* taylor_green_3d_schemes.png: dissipation rate -dE_k/dt of the three schemes at 32^3 and 64^3 against the DNS peak
  (van Rees et al. 2011), the shell spectrum E(k) of the velocity at the peak of the dissipation (log-log, up to the Nyquist
  shell n/2), and the decay of E(k) of the WENO5 run at six times;
* taylor_green_3d_vorticity_weno5.png: |omega| on the box faces and on the plane x = 0 at four times, WENO5-Z.

The peak values, the time of the peak and the share of the dissipation added by the scheme go to the output
(share = (-dK/dt - 2 nu Z)/(-dK/dt) with the spectral enstrophy Z of the snapshot nearest to the peak).
"""
import sys
from pathlib import Path

import numpy as np

import fenice
from fenice import spectra, study

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taylor_green_3d import (CASE, DNS_PEAK, L, NU, SNAP_TIMES, cube_faces, pick_snapshot, read_vtr,  # noqa: E402
                             vorticity)

STUDY = "taylor_green_3d"
SCHEMES = {   # label: (sets, colour)
    "muscl": ("MUSCL (Van Leer, RK2)", (), "#2980b9"),
    "weno3": ("WENO3-Z (RK3)", ("reconstruction=weno3", "time_scheme=SSP Runge-Kutta 3"), "#e67e22"),
    "weno5": ("WENO5-Z (RK3)", ("reconstruction=weno5", "time_scheme=SSP Runge-Kutta 3"), "#8e44ad"),
}
DECAY_TIMES = (4.0, 8.0, 12.0, 16.0, 20.0)


def run(scheme, n):
    sets = SCHEMES[scheme][1]
    # a snapshot about every 0.4 time units: 32^3 about 100 steps, 64^3 about 200 (RK2) / 150 (RK3)
    save = 100 if n == 32 else 200
    d = study.run_cached(STUDY, f"{scheme}_n{n}", CASE, f"nx={n}", f"ny={n}", f"nz={n}", *sets,
                         f"monitor_interval={5 if n == 32 else 10}", f"save_interval={save}", done="*/*_monitor.csv")
    return study.case_output(d)


def spectrum_at(series, t):
    ts = np.array([s[0] for s in series])
    i = int(np.argmin(np.abs(ts - t)))
    rho, vel = read_vtr(series[i][1])
    u, v, w = (vel[..., c] for c in range(3))
    return float(ts[i]), spectra.energy_spectrum(u, v, w, lengths=(L, L, L))


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--grids", type=int, nargs="+", default=[32, 64], metavar="N")
    args = ap.parse_args()
    plt = study.use_style()
    grids = sorted(args.grids)
    nbig = grids[-1]

    runs = {}
    for sc in SCHEMES:
        for n in grids:
            case = run(sc, n)
            mon = fenice.read_monitor(case)
            t, ke = mon["t"], mon["kinetic_energy"]
            eps = -np.gradient(ke, t)
            runs[sc, n] = dict(case=case, t=t, ke=ke, eps=eps)

    rows = []
    peak = {}
    for (sc, n), r in runs.items():
        k = int(np.argmax(r["eps"]))
        series = fenice.read_pvd(r["case"])
        ts, s = spectrum_at(series, float(r["t"][k]))
        Z = s["enstrophy"]
        share = float((np.interp(ts, r["t"], r["eps"]) - 2 * NU * Z) / np.interp(ts, r["t"], r["eps"]))
        peak[sc, n] = dict(t_snap=ts, spec=(s["k"], s["E"]))
        rows.append(dict(scheme=sc, n=n, eps_peak=float(r["eps"][k]), t_peak=float(r["t"][k]),
                         ke_end=float(r["ke"][-1] / r["ke"][0]), numerical_share=share))
        print(f"{sc:6s} {n}^3: peak eps {r['eps'][k]:.4f} at t = {r['t'][k]:.2f}, E_k(20)/E_k(0) = {r['ke'][-1] / r['ke'][0]:.3f}, "
              f"numerical share {100 * share:.0f} %")
    fenice.write_summary("taylor_green_3d_schemes", {"rows": rows, "dns_peak": list(DNS_PEAK)})

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    ls = {32: "--", 64: "-"} if len(grids) == 2 else {n: "-" for n in grids}
    for sc, (label, _, c) in SCHEMES.items():
        for n in grids:
            r = runs[sc, n]
            axs[0].plot(r["t"], r["eps"], ls[n], color=c, lw=2 if n == nbig else 1.4, label=f"{label}, {n}$^3$")
    axs[0].plot(*DNS_PEAK, "*", color="#2c3e50", ms=15, mec="white", label="DNS peak", zorder=5)
    axs[0].set_xlim(0, 20)
    axs[0].set_ylim(0, 0.0205)
    axs[0].set_xlabel("$t$")
    axs[0].set_ylabel(r"$\varepsilon = -dE_k/dt$")
    axs[0].set_title("Dissipation rate")
    axs[0].legend(fontsize=7, ncol=2, loc="upper right")
    axs[0].grid(True, ls="--", alpha=0.5)

    ymin = np.inf
    for sc, (label, _, c) in SCHEMES.items():
        k, E = peak[sc, nbig]["spec"]
        m = (k >= 1) & (k <= nbig // 2) & (E > 0)
        axs[1].loglog(k[m], E[m], "o-", ms=3, color=c, label=label)
        ymin = min(ymin, E[m & (k >= 2)].min())
    kk = np.array([3.0, 20.0])
    axs[1].loglog(kk, 0.35 * kk ** (-5 / 3), "k--", lw=1, label="$k^{-5/3}$ (guide)")
    axs[1].axvline(nbig // 2, color="0.4", ls=":", lw=1)
    axs[1].text(nbig // 2 * 0.93, 0.25, "Nyquist", rotation=90, ha="right", va="top", fontsize=8, color="0.35")
    axs[1].set_ylim(ymin / 3, 0.3)
    axs[1].set_xlabel("shell $k$")
    axs[1].set_ylabel("$E(k)$")
    axs[1].set_title(f"Spectrum at the dissipation peak, {nbig}$^3$")
    axs[1].legend(fontsize=8)

    series = fenice.read_pvd(runs["weno5", nbig]["case"])
    cmap = plt.cm.viridis
    ymin = np.inf
    for i, td in enumerate(DECAY_TIMES):
        ts, s = spectrum_at(series, td)
        k, E = s["k"], s["E"]
        m = (k >= 1) & (k <= nbig // 2) & (E > 0)
        axs[2].loglog(k[m], E[m], "o-", ms=3, color=cmap(i / (len(DECAY_TIMES) - 1) * 0.9), label=f"$t$ = {ts:.1f}")
        ymin = min(ymin, E[m & (k >= 2)].min())
    axs[2].loglog(kk, 0.35 * kk ** (-5 / 3), "k--", lw=1, label="$k^{-5/3}$ (guide)")
    axs[2].axvline(nbig // 2, color="0.4", ls=":", lw=1)
    axs[2].set_ylim(ymin / 3, 0.3)
    axs[2].set_xlabel("shell $k$")
    axs[2].set_ylabel("$E(k)$")
    axs[2].set_title(f"Decay of the spectrum, WENO5-Z, {nbig}$^3$")
    axs[2].legend(fontsize=8)
    fig.suptitle(r"3-D Taylor-Green vortex ($Re = 1600$): the three reconstructions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    study.save(fig, STUDY, "taylor_green_3d_schemes.png", args.docs, dpi=200)

    # box images of the WENO5 run
    shots = {}
    for ts, path in fenice.read_pvd(runs["weno5", nbig]["case"]):
        pick_snapshot(shots, ts, vorticity(read_vtr(path)[1]))
    vmax = max(np.percentile(f, 99.5) for _, f in shots.values())
    fig = plt.figure(figsize=(16, 7.6))
    fig.suptitle(rf"3-D Taylor-Green vortex, WENO5-Z, {nbig}$^3$: vorticity magnitude $|\omega|$", fontsize=15, fontweight="bold")
    for c, target in enumerate(SNAP_TIMES):
        ts, f = shots[target]
        cube_faces(fig.add_subplot(2, 4, c + 1, projection="3d"), f, vmax, f"$t = {ts:.1f}$")
        ax = fig.add_subplot(2, 4, c + 5)
        ax.imshow(f[:, :, 0], origin="lower", extent=(0, L, 0, L), cmap="magma", vmin=0, vmax=vmax, interpolation="bilinear")
        ax.set_xlabel("$y$")
        ax.set_ylabel("$z$")
        ax.set_title(f"plane $x = 0$, $t = {ts:.1f}$", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, vmax))
    fig.subplots_adjust(left=0.04, right=0.88, top=0.9, bottom=0.07, hspace=0.12, wspace=0.18)
    fig.colorbar(sm, cax=fig.add_axes([0.9, 0.2, 0.012, 0.6]), label=r"$|\omega|$")
    study.save(fig, STUDY, "taylor_green_3d_vorticity_weno5.png", args.docs)


if __name__ == "__main__":
    main()
