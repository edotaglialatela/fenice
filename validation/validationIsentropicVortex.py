"""Isentropic vortex: advection accuracy and order of convergence.

    python validation/validationIsentropicVortex.py

Runs the C++ solver (cpp/build/fenice) on the periodic isentropic vortex for one
full period (t = 10, the vortex returns to its start) at N = 20, 40, 80, 160, and
compares the density with the exact solution.  Also compares the four TVD limiters at
N = 80.  Writes results/isentropic_vortex_validation.png.
"""
import os
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "cpp", "build", "fenice.exe" if os.name == "nt" else "fenice")
L, GAMMA, BETA, T_END = 10.0, 1.4, 5.0, 10.0


def exact_rho(x, y, t):
    dx = (x - (0.5 * L + t) + 0.5 * L) % L - 0.5 * L
    dy = (y - (0.5 * L + t) + 0.5 * L) % L - 0.5 * L
    T = 1.0 - (GAMMA - 1.0) * BETA ** 2 / (8.0 * GAMMA * np.pi ** 2) * np.exp(1.0 - dx ** 2 - dy ** 2)
    return T ** (1.0 / (GAMMA - 1.0))


def run(n, tmp, limiter="MC"):
    dump = os.path.join(tmp, f"vortex_{n}_{limiter.replace(' ', '')}.npz")
    subprocess.run([EXE, "--case", "Isentropic Vortex", "--no-logo", "--set", f"nx={n}", "--set", f"ny={n}",
                    "--set", f"t_end={T_END}", "--set", f"muscl_limiter={limiter}", "--set", "save_interval=1000000000",
                    "--set", f"output_root={tmp}", "--set", f"dump_state={dump}"], check=True, capture_output=True)
    rho = np.load(dump)["U"][:, 0].reshape(n, n)
    xc = (np.arange(n) + 0.5) * L / n
    X, Y = np.meshgrid(xc, xc)
    return xc, rho, exact_rho(X, Y, T_END)


def main():
    tmp = tempfile.mkdtemp(prefix="fv_vortex_")
    ns = np.array([20, 40, 80, 160])
    errs, profiles = [], {}
    for n in ns:
        xc, rho, ex = run(n, tmp)
        errs.append(np.mean(np.abs(rho - ex)))
        profiles[n] = (xc, rho[n // 2, :], ex[n // 2, :])
        print(f"N = {n:4d}: L1 density error {errs[-1]:.3e}")
    errs = np.array(errs)
    orders = np.log2(errs[:-1] / errs[1:])
    print("observed orders:", " ".join(f"{o:.2f}" for o in orders))
    lim_err = {}
    for lim in ("Minmod", "Van Leer", "Koren", "MC"):
        _, rho, ex = run(80, tmp, lim)
        lim_err[lim] = np.mean(np.abs(rho - ex))
        print(f"limiter {lim:8s} (N = 80): L1 error {lim_err[lim]:.3e}")

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "axes.labelsize": 12,
                         "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"Isentropic vortex after one period ($t = 10$, MUSCL MC, SSP-RK2)", fontsize=15, fontweight="bold")
    xf = np.linspace(0, L, 400)
    a1.plot(xf, exact_rho(xf, 0.5 * L * np.ones_like(xf) + 0.5 * L / 160, T_END), color="#2c3e50", lw=2.5, label="Exact")
    for n, color, marker in ((40, "#2980b9", "s"), (80, "#e67e22", "^"), (160, "#e74c3c", "o")):
        xc, num, _ = profiles[n]
        a1.plot(xc, num, marker, color=color, ms=4 if n == 160 else 6, mec="white", mew=0.5, label=f"$N = {n}$")
    a1.set_xlim(2, 8)
    a1.set_xlabel("$x$ (through the vortex centre)")
    a1.set_ylabel(r"Density, $\rho$")
    a1.set_title("Centreline density")
    a1.grid(True, ls="--", alpha=0.6)
    a1.legend(frameon=True)

    a2.loglog(ns, errs, "D-", color="#8e44ad", lw=2, ms=7, label="$L_1$ density error (MC)")
    a2.loglog(ns, errs[0] * (ns[0] / ns) ** 2, "--", color="gray", label="2nd order")
    for n, e, o in zip(ns[1:], errs[1:], orders):
        a2.annotate(f"{o:.2f}", (n, e), textcoords="offset points", xytext=(6, 6), fontsize=10)
    for lim, mk in zip(lim_err, ("v", "<", ">", "x")):
        if lim != "MC":
            a2.loglog([80], [lim_err[lim]], mk, ms=8, label=f"{lim} ($N = 80$)")
    a2.set_xticks(ns, [str(n) for n in ns])
    a2.minorticks_off()
    a2.set_xlabel("Cells per direction, $N$")
    a2.set_ylabel(r"$L_1$ error of $\rho$")
    a2.set_title("Grid convergence and limiter comparison")
    a2.grid(True, which="both", ls="--", alpha=0.5)
    a2.legend(frameon=True)
    plt.tight_layout()
    out = os.path.join(ROOT, "results", "isentropic_vortex_validation.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[PLOT] {out}")


if __name__ == "__main__":
    main()
