"""Loads on an IBM body: vorticity around it and the Cl / Cd / Cm history from the monitor.

    python validation/airfoilLoads.py                         # results/ibm_airfoil
    python validation/airfoilLoads.py results/ibm_airfoil --t-avg 10

Reads <case>_monitor.csv (run the case with monitor_interval > 0; the IBM Airfoil case does)
and the last snapshot. Prints the mean coefficients over t >= --t-avg and the dominant
frequency of Cl; writes results/<case>_loads.png.
"""
import argparse
import glob
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from fenice import vtk_array

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case_dir", nargs="?", default=os.path.join(ROOT, "results", "ibm_airfoil"))
    ap.add_argument("--t-avg", type=float, default=10.0, help="average from this time on (default 10)")
    ap.add_argument("--window", type=float, nargs=4, default=[1.5, 9.0, 1.5, 4.5], metavar=("X0", "X1", "Y0", "Y1"),
                    help="vorticity window (default: around the airfoil of the IBM Airfoil case)")
    args = ap.parse_args()

    name = os.path.basename(os.path.normpath(args.case_dir))
    d = np.genfromtxt(os.path.join(args.case_dir, f"{name}_monitor.csv"), delimiter=",", names=True)
    if "Cl" not in d.dtype.names:
        raise SystemExit("The monitor has no loads: run with use_ibm = true and monitor_interval > 0")
    t, cd, cl, cm = d["t"], d["Cd"], d["Cl"], d["Cm"]
    late = t >= args.t_avg
    # Dominant Cl frequency (uniform resampling + FFT)
    tu = np.linspace(t[late][0], t[late][-1], 4 * late.sum())
    c = np.interp(tu, t[late], cl[late])
    spec = np.abs(np.fft.rfft(c - c.mean()))
    f = np.fft.rfftfreq(len(tu), tu[1] - tu[0])[np.argmax(spec[1:]) + 1]
    print(f"t >= {args.t_avg:g}: Cd = {cd[late].mean():.3f}, Cl = {cl[late].mean():.3f} "
          f"(from {cl[late].min():.3f} to {cl[late].max():.3f}), Cm = {cm[late].mean():.3f}; Cl frequency {f:.3f}")

    pvd = ET.parse(os.path.join(args.case_dir, f"{name}.pvd")).getroot()
    last = list(pvd.iter("DataSet"))[-1]
    root = ET.parse(os.path.join(args.case_dir, last.get("file"))).getroot()
    nx, _, ny = (int(v) for v in root.find("RectilinearGrid").get("WholeExtent").split()[1:4])
    x = vtk_array(root.find('.//DataArray[@Name="x"]'))
    y = vtk_array(root.find('.//DataArray[@Name="y"]'))
    vel = vtk_array(root.find('.//DataArray[@Name="Velocity"]')).reshape(ny, nx, 3)
    h = x[1] - x[0]
    w = np.gradient(vel[..., 1], h, axis=1) - np.gradient(vel[..., 0], h, axis=0)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "axes.labelsize": 12})
    fig = plt.figure(figsize=(11, 8.5))
    a0 = fig.add_axes([0.07, 0.47, 0.88, 0.48])
    x0, x1, y0, y1 = args.window
    a0.imshow(w, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="RdBu_r", vmin=-8, vmax=8,
              interpolation="bilinear")
    a0.set_xlim(x0, x1); a0.set_ylim(y0, y1)
    a0.set_xlabel("$x$"); a0.set_ylabel("$y$")
    a0.set_title(rf"Vorticity $\omega_z$ at $t = {float(last.get('timestep')):.1f}$", fontsize=13)
    a1 = fig.add_axes([0.07, 0.07, 0.88, 0.3])
    a1.plot(t, cl, color="#2980b9", lw=1.6, label=rf"$C_l$ (mean {cl[late].mean():.3f})")
    a1.plot(t, cd, color="#c0392b", lw=1.6, label=rf"$C_d$ (mean {cd[late].mean():.3f})")
    a1.plot(t, cm, color="#7f8c8d", lw=1.2, label=rf"$C_m$ (mean {cm[late].mean():.3f})")
    a1.axvspan(args.t_avg, t[-1], color="#bdc3c7", alpha=0.25, lw=0)
    a1.set_xlim(t[0], t[-1])
    a1.set_ylim(min(-0.1, cm.min()), max(cl[t > 1].max(), cd.max()) * 1.15)
    a1.set_xlabel("Time, $t$"); a1.set_ylabel("Coefficient")
    a1.grid(True, ls="--", alpha=0.5)
    a1.legend(loc="upper right", ncol=3, frameon=True)
    out = os.path.join(ROOT, "results", f"{name}_loads.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=170)
    print(f"[PLOT] {out}")


if __name__ == "__main__":
    main()
