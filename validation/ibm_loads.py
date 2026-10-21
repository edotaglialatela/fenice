"""Loads on an IBM body: vorticity around it, Cl / Cd / Cm history and the Strouhal number.

    python validation/ibm_loads.py airfoil              # results/validation/airfoil_loads/
    python validation/ibm_loads.py cylinder --docs      # results/validation/cylinder_loads/, docs/img/

Runs the "IBM Airfoil" or "IBM Flow over Cylinder" case (16 MPI ranks unless --mpi says
otherwise; tens of minutes) into results/validation/<body>_loads/run/, unless that run exists.
Reads the loads the solver writes to the monitor (Fx, Fy, Cd, Cl, Cm) and the last snapshot,
prints the mean coefficients over t >= --t-avg and the Strouhal number of Cl, and compares
the cylinder with the Re = 100 literature values.
Figure: <body>_loads.png.
"""
import numpy as np

import fenice
from fenice import study

BODIES = {
    # case, reference length (Strouhal), averaging start, vorticity window and colour range,
    # extra --set, reference values
    "airfoil": dict(case="IBM Airfoil", L=1.0, t_avg=10.0, window=(1.5, 9.0, 1.5, 4.5), wmax=8.0, sets=(), ref=None),
    "cylinder": dict(case="IBM Flow over Cylinder", L=0.2, t_avg=2.0, window=(0.5, 4.0, 0.5, 1.5), wmax=20.0,
                     sets=("monitor_interval=20",),
                     ref="Re = 100 (Williamson 1996; Park et al. 1998): St = 0.164-0.166, Cd = 1.33-1.35"),
}


def strouhal(t, cl, L, u=1.0):
    """St = f L / u of the dominant frequency of Cl: Hann-windowed FFT, zero-padded 16x so the
    peak is located to a fraction of the 1 / window bin (a few shedding periods only)."""
    tu = np.linspace(t[0], t[-1], 4 * len(t))
    c = np.interp(tu, t, cl)
    c = (c - c.mean()) * np.hanning(len(c))
    n = 16 * len(c)
    spec = np.abs(np.fft.rfft(c, n))
    return np.fft.rfftfreq(n, tu[1] - tu[0])[np.argmax(spec[1:]) + 1] * L / u


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("body", choices=BODIES)
    ap.add_argument("--t-avg", type=float, help="average from this time on (default: per body)")
    args = ap.parse_args()
    b = BODIES[args.body]
    name = f"{args.body}_loads"
    plt = study.use_style()
    run = study.run_cached(name, "run", b["case"], *b["sets"], mpi=args.mpi or 16, force=args.force,
                           done="*/*_monitor.csv")
    out_dir = study.case_output(run)
    d = fenice.read_monitor(out_dir)
    if "Cl" not in d.dtype.names:
        raise SystemExit("The monitor has no loads: run with use_ibm = true and monitor_interval > 0")
    t, cd, cl, cm = d["t"], d["Cd"], d["Cl"], d["Cm"]
    t_avg = b["t_avg"] if args.t_avg is None else args.t_avg
    late = t >= t_avg
    st = strouhal(t[late], cl[late], b["L"])
    print(f"t >= {t_avg:g}: Cd = {cd[late].mean():.3f}, Cl = {cl[late].mean():.3f} "
          f"(from {cl[late].min():.3f} to {cl[late].max():.3f}), Cm = {cm[late].mean():.3f}; St = {st:.3f}")
    if b["ref"]:
        print(f"reference, {b['ref']}")

    ts, path = fenice.read_pvd(out_dir)[-1]
    f = fenice.read_vtr(path)
    x, y, vel = f["x"], f["y"], f["Velocity"]
    h = x[1] - x[0]
    w = np.gradient(vel[..., 1], h, axis=1) - np.gradient(vel[..., 0], h, axis=0)
    x0, x1, y0, y1 = b["window"]
    wmax = b["wmax"]
    fig = plt.figure(figsize=(11, 8.5))
    a0 = fig.add_axes([0.07, 0.47, 0.88, 0.48])
    a0.imshow(w, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="RdBu_r", vmin=-wmax, vmax=wmax,
              interpolation="bilinear")
    a0.set_xlim(x0, x1)
    a0.set_ylim(y0, y1)
    a0.set_xlabel("$x$")
    a0.set_ylabel("$y$")
    a0.grid(False)
    a0.set_title(rf"{b['case']}: vorticity $\omega_z$ at $t = {ts:.1f}$", fontsize=13)
    a1 = fig.add_axes([0.07, 0.07, 0.88, 0.3])
    a1.plot(t, cl, color="#2980b9", lw=1.6, label=rf"$C_l$ (mean {cl[late].mean():.3f})")
    a1.plot(t, cd, color="#c0392b", lw=1.6, label=rf"$C_d$ (mean {cd[late].mean():.3f})")
    a1.plot(t, cm, color="#7f8c8d", lw=1.2, label=rf"$C_m$ (mean {cm[late].mean():.3f})")
    a1.axvspan(t_avg, t[-1], color="#bdc3c7", alpha=0.25, lw=0, label=f"average, St = {st:.3f}")
    a1.set_xlim(t[0], t[-1])
    a1.set_ylim(min(-0.1, cl.min(), cm.min()) * 1.15, max(cl[t > 1].max(), cd.max()) * 1.25)
    a1.set_xlabel("Time, $t$")
    a1.set_ylabel("Coefficient")
    a1.legend(loc="upper right", ncol=4, frameon=True)
    study.save(fig, name, f"{name}.png", args.docs)


if __name__ == "__main__":
    main()
