"""Order of the flux quadrature on a smooth two-dimensional flow: the isentropic vortex, `flux_quadrature = midpoint` against `fourth`.

    python cases/validation/verification/flux_quadrature_order.py            # results/validation/flux_quadrature_order/
    python cases/validation/verification/flux_quadrature_order.py --check    # exit 1 when an order leaves its band

The flux balance with face-centre values is second order in several dimensions whatever the reconstruction. To see the order of the
reconstruction the data have to be cell averages: the initial state is the exact cell average of the vortex (5 x 5 Gauss points per cell),
written to a restart file, and the error is the L2 difference of the density from the exact cell average at t = 1 (one period of the
box is 10, the vortex moves one unit). WENO5-Z with SSP-RK3; the CFL number falls like h^(2/3) so that the time error (third order) stays
below the space error (fourth order) on the finest mesh. Meshes 16, 24, 32, 48, 64, 96 cells per side.

Bands, set AFTER the first run (the study was written to measure the new option, not to test a prediction): the observed order of the
finest pair of meshes is in [1.8, 2.3] for `midpoint` and in [3.8, 5.2] for `fourth`, and the `fourth` error is below the `midpoint` one on
every mesh from 32 cells on.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXE = ROOT / "build" / "release" / ("fenice.exe" if os.name == "nt" else "fenice")
OUT = ROOT / "results" / "validation" / "flux_quadrature_order"
GAM, L, BETA, TEND = 1.4, 10.0, 5.0, 1.0
NS = [16, 24, 32, 48, 64, 96]
GX, GW = np.polynomial.legendre.leggauss(5)


def exact(x, y, t):
    dx = x - (0.5 * L + t)
    dy = y - (0.5 * L + t)
    dx -= L * np.round(dx / L)
    dy -= L * np.round(dy / L)
    r2 = dx * dx + dy * dy
    g = BETA / (2 * np.pi) * np.exp(0.5 * (1 - r2))
    T = 1 - (GAM - 1) * BETA ** 2 / (8 * GAM * np.pi ** 2) * np.exp(1 - r2)
    rho = T ** (1 / (GAM - 1))
    u = 1 - g * dy
    v = 1 + g * dx
    p = rho * T
    return np.array([rho, rho * u, rho * v, p / (GAM - 1) + 0.5 * rho * (u * u + v * v)])


def cell_average(n, t):
    """[q, i, j] exact cell averages by 5 x 5 Gauss quadrature."""
    h = L / n
    xc = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xc, xc, indexing="ij")
    acc = np.zeros((4, n, n))
    for a, wa in zip(GX, GW):
        for b, wb in zip(GX, GW):
            acc += wa * wb * exact(X + 0.5 * h * a, Y + 0.5 * h * b, t)
    return acc / 4.0


def run(args):
    r = subprocess.run([str(EXE), "--no-logo"] + args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"fenice failed: {r.stdout[-600:]} {r.stderr[-600:]}")


def error(mode, n, work):
    h = L / n
    cfl = 0.4 * (h / (L / 16)) ** (2 / 3)
    d = os.path.join(work, f"{mode}_{n}")
    base = ["--case", "isentropic_vortex", "--set", f"nx={n}", "--set", f"ny={n}", "--set", "reconstruction=weno5",
            "--set", "time_scheme=SSP Runge-Kutta 3", "--set", f"flux_quadrature={mode}", "--set", f"cfl={cfl}",
            "--set", f"output_root={d}", "--threads", "4"]
    run(base + ["--set", "max_steps=1"])
    ck = glob.glob(os.path.join(d, "**", "*_restart.npz"), recursive=True)
    assert len(ck) == 1, ck
    u0 = cell_average(n, 0.0)
    state = np.stack([u0[q].T.reshape(-1) for q in range(4)], axis=1)   # row-major (j * nx + i)
    np.savez(ck[0], U=state, t=np.float64(0.0), step=np.int64(0), format_version=np.int64(1))
    dump = os.path.join(d, "final_dump.npz")
    run(base + ["--set", "resume=true", "--set", f"t_end={TEND}", "--set", f"dump_state={dump}", "--set", "save_interval=100000"])
    a = np.load(dump)
    ex = cell_average(n, float(a["t"]))
    exn = np.stack([ex[q].T.reshape(-1) for q in range(4)], axis=1)
    return cfl, float(np.sqrt(np.mean((a["U"][:, 0] - exn[:, 0]) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit with 1 when an order leaves its band")
    args = ap.parse_args()
    if not EXE.exists():
        sys.exit(f"{EXE} not found: build first (cpp/build.ps1)")
    OUT.mkdir(parents=True, exist_ok=True)
    err = {}
    with tempfile.TemporaryDirectory() as work:
        for mode in ("midpoint", "fourth"):
            err[mode] = []
            for n in NS:
                cfl, e = error(mode, n, work)
                err[mode].append(e)
                print(f"{mode:8s} n={n:3d} cfl={cfl:.3f} L2(rho)={e:.3e}", flush=True)
    order = {m: [np.log(err[m][i - 1] / err[m][i]) / np.log(NS[i] / NS[i - 1]) for i in range(1, len(NS))] for m in err}
    lines = ["| n | midpoint L2(rho) | order | fourth L2(rho) | order |", "|---:|---:|---:|---:|---:|"]
    for i, n in enumerate(NS):
        o = lambda m: "" if i == 0 else f"{order[m][i - 1]:.2f}"
        lines.append(f"| {n} | {err['midpoint'][i]:.3e} | {o('midpoint')} | {err['fourth'][i]:.3e} | {o('fourth')} |")
    (OUT / "flux_quadrature_order.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    ok = (1.8 <= order["midpoint"][-1] <= 2.3 and 3.8 <= order["fourth"][-1] <= 5.2
          and all(err["fourth"][i] < err["midpoint"][i] for i in range(2, len(NS))))
    print("bands:", "pass" if ok else "FAIL")
    return 1 if (args.check and not ok) else 0


if __name__ == "__main__":
    raise SystemExit(main())
