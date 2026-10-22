"""Figures of the cell-based AMR development runs (docs/design/amr-cell-based.md).

Runs ``cell_amr_demo`` for the Sod tube and the NACA 0012 airfoil study (192 x 96 base, 2 levels, t = 6) and draws
the leaves: refinement level and density for the airfoil, density against the exact solution and the level for Sod.
Output: results/cell_amr/*.png (and the leaves.csv of each run next to them).

    python cases/validation/amr/cell_amr_p5.py [--sod] [--airfoil] [--reuse]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
EXE = ROOT / "build" / "release" / ("cell_amr_demo.exe" if sys.platform == "win32" else "cell_amr_demo")
OUT = ROOT / "results" / "cell_amr"


def run(case, out, *args):
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(EXE), case, *args, "--out", str(out)]
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("\n".join(l for l in res.stdout.splitlines() if l.startswith("[cell-amr]")))


def read_leaves(path):
    return np.genfromtxt(path, delimiter=",", names=True)


def sod_exact(x, t, g=1.4):
    """Density of the Sod problem (1, 0, 1 | 0.125, 0, 0.1), diaphragm at 0.5."""
    rl, pl, rr, pr = 1.0, 1.0, 0.125, 0.1
    cl, cr = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)

    def f(p, rk, pk, ck):
        if p > pk:
            return (p - pk) * np.sqrt((2 / ((g + 1) * rk)) / (p + (g - 1) / (g + 1) * pk))
        return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)

    p = 0.3
    for _ in range(100):
        h = 1e-7 * p
        fv = f(p, rl, pl, cl) + f(p, rr, pr, cr)
        p = max(p - fv / ((f(p + h, rl, pl, cl) + f(p + h, rr, pr, cr) - fv) / h), 1e-6)
    us = 0.5 * (f(p, rr, pr, cr) - f(p, rl, pl, cl))
    s = (x - 0.5) / t
    rsl, rsr = rl * (p / pl) ** (1 / g), rr * ((p / pr + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * p / pr + 1))
    sr = cr * np.sqrt((g + 1) / (2 * g) * p / pr + (g - 1) / (2 * g))
    cs = cl * (p / pl) ** ((g - 1) / (2 * g))
    c_fan = 2 / (g + 1) * (cl - (g - 1) / 2 * s)
    return np.select([s < -cl, s < us - cs, s < us, s < sr], [rl, rl * (c_fan / cl) ** (2 / (g - 1)), rsl, rsr], rr)


def figure_sod(reuse):
    d = OUT / "sod"
    if not reuse:
        run("Sod Shock Tube", d, "nx=50", "ny=4", "Ly=0.08", "--levels", "2", "--every", "4")
    lv = read_leaves(d / "leaves.csv")
    row = np.abs(lv["y"] - 0.0425) < 0.5 * lv["dx"]   # the leaf row that contains y = 0.0425
    x, rho, lev, dx = lv["x"][row], lv["rho"][row], lv["level"][row], lv["dx"][row]
    xs = np.linspace(0, 1, 2000)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    a0.plot(xs, sod_exact(xs, 0.2), "k-", lw=1, label="exact")
    sc = a0.scatter(x, rho, c=lev, cmap=ListedColormap(["#4c78a8", "#f58518", "#e45756"]), s=14, vmin=-0.5, vmax=2.5,
                    label="leaf centres")
    a0.set_ylabel(r"$\rho$")
    a0.set_title("Sod tube, t = 0.2: base 50 x 4, 2 levels, face-variation indicator")
    a0.legend(loc="upper right")
    a1.step(np.sort(x), lev[np.argsort(x)], where="mid", color="#e45756")
    a1.set_ylabel("level")
    a1.set_yticks([0, 1, 2])
    a1.set_xlabel("x")
    fig.colorbar(sc, ax=[a0, a1], ticks=[0, 1, 2], label="level", pad=0.01)
    fig.savefig(OUT / "sod_cell_amr.png", dpi=150)
    print("wrote", OUT / "sod_cell_amr.png")


def figure_airfoil(reuse):
    d = OUT / "airfoil"
    if not reuse:
        run("IBM Airfoil AMR", d, "--vars", "rho,u", "--levels", "2", "--every", "5", "--tend", "6.0")
    lv = read_leaves(d / "leaves.csv")
    dxf = lv["dx"].min()
    Lx, Ly = 12.0, 6.0
    nxf, nyf = int(round(Lx / dxf)), int(round(Ly / dxf))
    level = np.zeros((nyf, nxf), dtype=np.int8)
    speed = np.full((nyf, nxf), np.nan)
    for x, y, h, l, r in zip(lv["x"], lv["y"], lv["dx"], lv["level"], np.where(lv["solid"] > 0, np.nan, np.hypot(lv["u"], lv["v"]))):
        i0, j0 = int(round((x - h / 2) / dxf)), int(round((y - h / 2) / dxf))
        n = int(round(h / dxf))
        level[j0:j0 + n, i0:i0 + n] = l
        speed[j0:j0 + n, i0:i0 + n] = r
    counts = [int((lv["level"] == k).sum()) for k in range(3)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    ext = [0, Lx, 0, Ly]
    im = axes[0].imshow(speed, origin="lower", extent=ext, cmap="viridis", vmin=0.0, vmax=1.4)
    axes[0].set_title(r"NACA 0012, Re = 1000, t = 6: speed $|u|$ on the adaptive mesh")
    fig.colorbar(im, ax=axes[0], label=r"$|u|$", fraction=0.025)
    im = axes[1].imshow(level, origin="lower", extent=ext, cmap=ListedColormap(["#4c78a8", "#f58518", "#e45756"]),
                        vmin=-0.5, vmax=2.5)
    axes[1].set_title(f"refinement level: {len(lv)} leaves (level 0: {counts[0]}, 1: {counts[1]}, 2: {counts[2]})")
    cb = fig.colorbar(im, ax=axes[1], ticks=[0, 1, 2], label="level", fraction=0.025)
    for a in axes:
        a.set_xlabel("x")
        a.set_ylabel("y")
        a.set_xlim(1.5, 7.5)
        a.set_ylim(1.5, 4.5)
    fig.tight_layout()
    fig.savefig(OUT / "airfoil_cell_amr.png", dpi=130)
    print("wrote", OUT / "airfoil_cell_amr.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sod", action="store_true")
    ap.add_argument("--airfoil", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="draw from the leaves.csv of the last run")
    a = ap.parse_args()
    if not (a.sod or a.airfoil):
        a.sod = a.airfoil = True
    OUT.mkdir(parents=True, exist_ok=True)
    if a.sod:
        figure_sod(a.reuse)
    if a.airfoil:
        figure_airfoil(a.reuse)
