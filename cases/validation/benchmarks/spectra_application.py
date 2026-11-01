"""D5: the spectral tools on solver output: 3-D Taylor-Green at the peak of the dissipation, serial against MPI, an AMR wake.

    python cases/validation/benchmarks/spectra_application.py           # results/validation/spectra_application/
    python cases/validation/benchmarks/spectra_application.py --docs    # also docs/img/spectra_application.png

Uses runs that exist (it starts none of the big ones, and says what is missing):
  * results/validation/taylor_green_3d/n32 and n96 (benchmarks/taylor_green_3d.py: monitor and snapshots to t = 20, Re = 1600);
  * results/validation/cylinder_low_blockage/l4 (benchmarks/cylinder_low_blockage.py: AMR snapshots, Re = 100);
and makes one small run of its own: the 16^3 Taylor-Green vortex for 40 steps, serially and on 4 MPI ranks.

1. Taylor-Green at the time of peak dissipation (the snapshot nearest to the maximum of -dK/dt of the monitor), 32^3 and 96^3:
   shell spectrum E(k) of the velocity, the solenoidal / dilatational split, the Taylor micro-scale and the integral scale,
   and the dissipation the scheme adds, eps_num = -dK/dt (monitor) - 2 nu Z (enstrophy of the snapshot's spectrum).
2. Serial and 4-rank snapshots of the same run give the same spectrum, bit for bit.
3. The wake of the Re = 100 cylinder (AMR, four levels): the spectrum of v on the line y = 6, 4 <= x <= 11, resampled to the finest
   level, Hann window: the shedding wavelength.

Bands, fixed before the first run:
  a. the density-weighted spectrum at the peak sums to the monitor's kinetic energy of that time to 1e-4 (both are 1/2 <rho |u|^2>;
     the snapshots are float32);
  b. the dilatational part of the velocity energy is below 2 % (Mach 0.1: of the order Ma^2);
  c. the spectral enstrophy is at least the central-difference one of the snapshot (the difference stencil under-represents the
     high wavenumbers) and at most 1.5 times it, on 96^3;
  d. the numerical dissipation is non-negative at the peak on both grids and its share of the total falls from 32^3 to 96^3;
  e. the Taylor micro-scale at the peak lies in [0.1, 0.5] and the integral scale in [0.5, 2.0] (sanity bounds: the box is 2 pi);
  f. serial and MPI spectra are identical (array_equal), and so are the velocity fields;
  g. the wavelength 2 pi / k_peak of the wake line lies in [0.8, 1.5] (the shedding frequency St U / D = 0.83 of the Re = 100 runs
     times a convection speed of 0.9 to 1.1 U gives 1.1 to 1.3), with the number of cells resampled and the coarsened fraction printed.
"""
import numpy as np

import fenice
from fenice import spectra, study

STUDY = "spectra_application"
NU = 1.0 / 1600.0
L = 2.0 * np.pi
TG_CASE = "Taylor-Green Vortex 3D"
WAKE_Y, WAKE_X = 6.0, (4.0, 11.0)


def need(path, hint):
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing {p}: run  {hint}")
    return p


def fields(path):
    f = fenice.read_vtr(path)
    v = f["Velocity"]   # [k, j, i, component] = [z, y, x, c]
    return f["Density"], (v[..., 0], v[..., 1], v[..., 2])


def spec_of(path, rho_weighted=False):
    rho, (u, v, w) = fields(path)
    return spectra.energy_spectrum(u, v, w, lengths=(L, L, L), rho=rho if rho_weighted else None)


def fd_enstrophy(vel):
    n = vel[0].shape[0]
    h = L / n
    d = lambda f, ax: (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2 * h)
    u, v, w = vel
    om = (d(w, 1) - d(v, 0), d(u, 0) - d(w, 2), d(v, 2) - d(u, 1))
    return 0.5 * np.mean(sum(o * o for o in om))


def tg_at_peak(n, plt_data):
    out = need(study.study_dir("taylor_green_3d") / f"n{n}", "python cases/validation/run_all.py taylor_green_3d --all")
    case = study.case_output(out)
    mon = fenice.read_monitor(case)
    t, K = mon["t"], mon["kinetic_energy"]
    eps = -np.gradient(K, t)
    tp = float(t[np.argmax(eps)])
    series = fenice.read_pvd(case)
    ts = np.array([s[0] for s in series])
    ip = int(np.argmin(np.abs(ts - tp)))
    step = max(1, len(series) // 24)
    idx = sorted(set(range(0, len(series), step)) | {ip})
    Zs, tz = [], []
    for i in idx:
        s = spec_of(series[i][1])
        Zs.append(s["enstrophy"])
        tz.append(series[i][0])
    tz, Zs = np.array(tz), np.array(Zs)
    Kz = np.interp(tz, t, K)
    epsz = np.interp(tz, t, eps)
    num = epsz - 2 * NU * Zs
    path = series[ip][1]
    rho, (u, v, w) = fields(path)
    s_plain = spectra.energy_spectrum(u, v, w, lengths=(L, L, L))
    s_rho = spectra.energy_spectrum(u, v, w, lengths=(L, L, L), rho=rho)
    sol, dil = spectra.helmholtz(u, v, w, lengths=(L, L, L))
    e_sol = 0.5 * sum(np.mean(c * c) for c in sol)
    e_dil = 0.5 * sum(np.mean(c * c) for c in dil)
    r = dict(n=n, t_peak=tp, t_snap=float(ts[ip]), eps_peak=float(eps[np.argmax(eps)]),
             rho_energy_error=abs(s_rho["E"].sum() / float(np.interp(ts[ip], t, K)) - 1),
             dil_fraction=float(e_dil / (e_sol + e_dil)), Z=s_plain["enstrophy"], Z_fd=float(fd_enstrophy((u, v, w))),
             taylor=spectra.taylor_microscale(s_plain), integral=spectra.integral_scale(s_plain),
             num_fraction=float(np.interp(ts[ip], tz, num) / np.interp(ts[ip], tz, epsz)),
             spectrum=(s_plain["k"], s_plain["E"]))
    r["Re_lambda"] = float(np.sqrt(2.0 * np.sum(s_plain["E"]) / 3.0) * r["taylor"] / NU)
    plt_data[n] = (tz, epsz, 2 * NU * Zs, num)
    return r


def serial_vs_mpi():
    sets = ("nx=16", "ny=16", "nz=16", "max_steps=40", "save_interval=40", "monitor_interval=40")
    out = {}
    for tag, mpi in (("serial", 0), ("mpi4", 4)):
        d = study.run_cached(STUDY, tag, TG_CASE, *sets, mpi=mpi)
        path = fenice.read_pvd(study.case_output(d))[-1][1]
        out[tag] = (fields(path), spec_of(path))
    (_, (u0, v0, w0)), s0 = out["serial"][0], out["serial"][1]
    (_, (u1, v1, w1)), s1 = out["mpi4"][0], out["mpi4"][1]
    return bool(np.array_equal(s0["E"], s1["E"])), bool(all(np.array_equal(a, b) for a, b in ((u0, u1), (v0, v1), (w0, w1))))


def wake_line():
    run = need(study.study_dir("cylinder_low_blockage") / "l4", "python cases/validation/run_all.py cylinder_low_blockage")
    snap = fenice.read_vtu(fenice.read_pvd(study.case_output(run))[-1][1])
    h = float((snap["x1"] - snap["x0"]).min())
    region = (WAKE_X[0], WAKE_X[1], WAKE_Y - 2 * h, WAKE_Y + 2 * h)
    v, xc, yc, info = spectra.resample_amr(snap, "Velocity", component=1, region=region, h=h)
    line = v[v.shape[0] // 2]
    res = spectra.spectrum_1d(line, h, "hann")
    kpk = float(res["k"][1:][np.argmax(res["E"][1:])])
    return dict(h=h, cells=info["cells"], coarsened=info["coarsened"], k_peak=kpk, wavelength=2 * np.pi / kpk, dk=res["dk"],
                leakage=res["leakage"], spectrum=(res["k"], res["E"]), line=(xc, line))


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    curves, tg = {}, {}
    for n in (32, 96):
        tg[n] = tg_at_peak(n, curves)
        r = tg[n]
        print(f"{n}^3: peak eps = {r['eps_peak']:.4f} at t = {r['t_peak']:.2f} (snapshot t = {r['t_snap']:.2f}); "
              f"Z = {r['Z']:.3f} (central differences {r['Z_fd']:.3f}); dilatational share {100 * r['dil_fraction']:.2f} %; "
              f"Taylor micro-scale {r['taylor']:.3f}, integral scale {r['integral']:.3f}, Re_lambda {r['Re_lambda']:.0f}; "
              f"numerical share of the dissipation {100 * r['num_fraction']:.0f} %; rho-weighted energy vs monitor {r['rho_energy_error']:.1e}")
    same_spec, same_fields = serial_vs_mpi()
    print(f"serial vs 4 ranks (16^3, 40 steps): spectra identical {same_spec}, fields identical {same_fields}")
    wk = wake_line()
    print(f"wake line y = {WAKE_Y}, {WAKE_X[0]} <= x <= {WAKE_X[1]}: {wk['cells']} cells resampled at h = {wk['h']:g} "
          f"({100 * wk['coarsened']:.0f} % coarser leaves, staircase), peak k = {wk['k_peak']:.2f} (dk = {wk['dk']:.2f}), "
          f"wavelength {wk['wavelength']:.3f}, leakage above 4 k_peak {wk['leakage']:.1e}")

    out = {"tg": {str(n): {k: v for k, v in r.items() if k != "spectrum"} for n, r in tg.items()},
           "serial_mpi_spectra_identical": same_spec, "serial_mpi_fields_identical": same_fields,
           "wake": {k: v for k, v in wk.items() if k not in ("spectrum", "line")}}
    r96, r32 = tg[96], tg[32]
    out["pass_energy"] = bool(all(r["rho_energy_error"] < 1e-4 for r in tg.values()))
    out["pass_dilatational"] = bool(all(r["dil_fraction"] < 0.02 for r in tg.values()))
    out["pass_enstrophy"] = bool(1.0 <= r96["Z"] / r96["Z_fd"] <= 1.5)
    out["pass_numerical_dissipation"] = bool(r96["num_fraction"] >= 0 and r32["num_fraction"] >= 0
                                             and r96["num_fraction"] < r32["num_fraction"])
    out["pass_scales"] = bool(all(0.1 <= r["taylor"] <= 0.5 and 0.5 <= r["integral"] <= 2.0 for r in tg.values()))
    out["pass_serial_mpi"] = bool(same_spec and same_fields)
    out["pass_wake"] = bool(0.8 <= wk["wavelength"] <= 1.5)
    ok = all(out[k] for k in out if k.startswith("pass_"))
    out["passed"] = ok
    fenice.write_summary(STUDY, out)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    for n, c in ((32, "#2980b9"), (96, "#27ae60")):
        k, E = tg[n]["spectrum"]
        m = (k >= 1) & (k <= n // 2)
        axs[0].loglog(k[m], E[m], "o-", ms=3, color=c, label=f"{n}$^3$, $t$ = {tg[n]['t_snap']:.1f}")
    kk = np.array([3.0, 20.0])
    axs[0].loglog(kk, 0.35 * kk ** (-5 / 3), "k--", lw=1, label="$k^{-5/3}$ (guide)")
    axs[0].set_xlabel("shell $k$")
    axs[0].set_ylabel("$E(k)$")
    axs[0].legend()
    axs[0].set_title("Taylor-Green at the peak of the dissipation")
    for n, c in ((32, "#2980b9"), (96, "#27ae60")):
        tz, epsz, visc, num = curves[n]
        axs[1].plot(tz, epsz, "-", color=c, label=f"{n}$^3$ total $-dK/dt$")
        axs[1].plot(tz, visc, "--", color=c, label=f"{n}$^3$ viscous $2\\nu Z$")
    axs[1].set_xlabel("$t$")
    axs[1].set_ylabel("$\\varepsilon$")
    axs[1].legend(fontsize=7)
    axs[1].set_title("what the scheme adds = the gap")
    k, E = wk["spectrum"]
    axs[2].loglog(k[1:], E[1:], color="#c0392b")
    axs[2].axvline(wk["k_peak"], color="0.3", ls=":")
    axs[2].set_xlabel("$k$")
    axs[2].set_ylabel("$E_v(k)$")
    axs[2].set_title(f"wake line, wavelength {wk['wavelength']:.2f}")
    plt.tight_layout()
    study.save(fig, STUDY, "spectra_application.png", docs=args.docs)
    print("ALL BANDS PASSED" if ok else "SOME BANDS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
