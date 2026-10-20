"""Taylor-Green vortex: viscous decay and grid convergence against the exact solution.

    python validation/validationTaylorGreen.py

Runs the C++ solver (cpp/build/fenice) on the periodic Taylor-Green vortex:

1. 64x64 to t = 5 with snapshots, to compare the kinetic-energy decay KE(t)/KE(0)
   with the exact exp(-4 nu k^2 t), with and without viscous fluxes (the inviscid run
   isolates the scheme's numerical dissipation);
2. N = 16, 32, 64, 128 to t = 1, for the velocity L2 error against the exact field.

Writes results/taylor_green_validation.png.
"""
import glob
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "cpp", "build", "fenice.exe" if os.name == "nt" else "fenice")
MU, L = 0.02, 2.0 * np.pi
K = 2.0 * np.pi / L


def run(out_root, *sets):
    cmd = [EXE, "--case", "Taylor-Green Vortex", "--no-logo", "--set", f"output_root={out_root}"]
    for s in sets:
        cmd += ["--set", s]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def read_vtr(path):
    root = ET.parse(path).getroot()
    arr = lambda name: np.array(root.find(f'.//DataArray[@Name="{name}"]').text.split(), dtype=float)
    vel = arr("Velocity").reshape(-1, 3)
    return arr("Density"), vel[:, 0], vel[:, 1]


def ke_history(case_dir):
    pvd = ET.parse(glob.glob(os.path.join(case_dir, "*.pvd"))[0]).getroot()
    t, ke = [], []
    for ds in pvd.iter("DataSet"):
        rho, u, v = read_vtr(os.path.join(case_dir, ds.attrib["file"]))
        t.append(float(ds.attrib["timestep"]))
        ke.append(np.sum(0.5 * rho * (u ** 2 + v ** 2)))
    return np.array(t), np.array(ke) / ke[0]


def velocity_error(n, t_end, tmp):
    dump = os.path.join(tmp, f"tgv_{n}.npz")
    run(tmp, f"nx={n}", f"ny={n}", f"mu={MU}", f"t_end={t_end}", "save_interval=1000000000", f"dump_state={dump}")
    U = np.load(dump)["U"]
    xc = (np.arange(n) + 0.5) * L / n
    X, Y = np.meshgrid(xc, xc)
    F = np.exp(-2.0 * MU * K ** 2 * t_end)
    ue, ve = -np.cos(K * X) * np.sin(K * Y) * F, np.sin(K * X) * np.cos(K * Y) * F
    u, v = (U[:, 1] / U[:, 0]).reshape(n, n), (U[:, 2] / U[:, 0]).reshape(n, n)
    return np.sqrt(np.sum((u - ue) ** 2 + (v - ve) ** 2) / np.sum(ue ** 2 + ve ** 2))


def main():
    tmp = tempfile.mkdtemp(prefix="fv_tgv_")
    print("[1/2] Kinetic-energy decay, 64x64 ...")
    run(os.path.join(tmp, "visc"), "save_interval=20")
    run(os.path.join(tmp, "invisc"), "save_interval=20", "use_viscous=false")
    t_v, ke_v = ke_history(os.path.join(tmp, "visc", "taylor-green_vortex"))
    t_i, ke_i = ke_history(os.path.join(tmp, "invisc", "taylor-green_vortex"))
    nu_eff = -np.polyfit(t_v, np.log(ke_v), 1)[0] / (4 * K ** 2)
    nu_num = -np.polyfit(t_i, np.log(ke_i), 1)[0] / (4 * K ** 2)
    print(f"      nu = {MU}: fitted total {nu_eff:.5f}, numerical {nu_num:.5f}, physical {nu_eff - nu_num:.5f}")

    print("[2/2] Grid convergence, t = 1 ...")
    ns = np.array([16, 32, 64, 128])
    errs = np.array([velocity_error(n, 1.0, tmp) for n in ns])
    orders = np.log2(errs[:-1] / errs[1:])
    for n, e in zip(ns, errs):
        print(f"      N = {n:4d}: velocity L2 error {e:.3e}")
    print("      observed orders:", " ".join(f"{o:.2f}" for o in orders))

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "axes.labelsize": 12,
                         "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Taylor-Green vortex ($\nu = %.2f$, periodic $2\pi \times 2\pi$)" % MU, fontsize=15, fontweight="bold")
    tt = np.linspace(0, t_v[-1], 200)
    a1.plot(tt, np.exp(-4 * MU * K ** 2 * tt), color="#2c3e50", lw=2.5, label=r"Exact $e^{-4\nu k^2 t}$")
    a1.plot(t_v[::3], ke_v[::3], "o", color="#e74c3c", ms=6, mec="white", label=f"Navier-Stokes 64$^2$ (fit $\\nu$ = {nu_eff:.4f})")
    a1.plot(t_i[::3], ke_i[::3], "s", color="#2980b9", ms=5, mec="white", label=f"Euler 64$^2$ (numerical $\\nu$ = {nu_num:.4f})")
    a1.set_xlabel("Time, $t$ [s]")
    a1.set_ylabel(r"Kinetic energy, $E_k(t)/E_k(0)$")
    a1.set_title("Kinetic-energy decay")
    a1.grid(True, ls="--", alpha=0.6)
    a1.legend(frameon=True)

    a2.loglog(ns, errs, "D-", color="#8e44ad", lw=2, ms=7, label="Velocity $L_2$ error, $t = 1$")
    a2.loglog(ns, errs[0] * (ns[0] / ns) ** 2, "--", color="gray", label="2nd order")
    for n, e, o in zip(ns[1:], errs[1:], orders):
        a2.annotate(f"{o:.2f}", (n, e), textcoords="offset points", xytext=(6, 6), fontsize=10)
    a2.set_xticks(ns, [str(n) for n in ns])
    a2.minorticks_off()
    a2.set_xlabel("Cells per direction, $N$")
    a2.set_ylabel("Relative $L_2$ error")
    a2.set_title("Grid convergence (MUSCL Van Leer, SSP-RK2)")
    a2.grid(True, which="both", ls="--", alpha=0.5)
    a2.legend(frameon=True)
    plt.tight_layout()
    out = os.path.join(ROOT, "results", "taylor_green_validation.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[PLOT] {out}")


if __name__ == "__main__":
    main()
