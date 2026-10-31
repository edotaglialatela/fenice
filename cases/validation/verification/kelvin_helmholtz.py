"""A5: Kelvin-Helmholtz instability of a smooth tanh shear layer: linear growth rate against the analytic value.

    python cases/validation/verification/kelvin_helmholtz.py            # results/validation/kelvin_helmholtz/
    python cases/validation/verification/kelvin_helmholtz.py --docs     # also docs/img/kelvin_helmholtz.png

Case "Kelvin-Helmholtz": u = U tanh(y/delta), U = 0.2, delta = 1, sound speed 1 (Mach 0.2 on each side), one wavelength of the
fastest mode, alpha delta = 0.4446. The incompressible tanh layer grows at sigma = 0.1897 U / delta (Michalke 1964; the layer
is u = tanh(y), velocity jump 2U). The growth of the transverse kinetic energy K(t) = sum rho v^2 is fitted as
K ~ exp(2 sigma t) on 30 <= t <= 80. The run uses 96 x 109 and 144 x 163 cells.
Bands, fixed before the first run:
  1. sigma within 20 % of 0.1897 U / delta on the finer grid (the compressibility at Mach 0.2 lowers it by a few percent,
     the scheme's dissipation by more on the coarser grid);
  2. the two grids' sigma differ by at most 10 % of the finer one;
  3. mass and total energy conserved to a relative 1e-11 (periodic in x, walls in y);
  4. finite, rho > 0, p > 0.
"""
import numpy as np

import fenice
from fenice import study

STUDY, CASE = "kelvin_helmholtz", "Kelvin-Helmholtz"
U, DELTA = 0.2, 1.0
SIGMA_THEORY = 0.1897 * U / DELTA
GRIDS = ((96, 109), (144, 163))
FIT = (30.0, 80.0)
BAND_SIGMA, BAND_GRIDS, BAND_CONS = 0.20, 0.10, 1e-11


def growth(run_dir, nx, ny):
    """sigma from the snapshots, and the series."""
    t, K = [], []
    for ti, path in fenice.read_pvd(study.case_output(run_dir)):
        f = fenice.read_vtr(path)
        v = f["Velocity"][..., 1]
        t.append(ti)
        K.append(float(np.sum(f["Density"] * v * v)))
    t, K = np.array(t), np.array(K)
    sel = (t >= FIT[0]) & (t <= FIT[1])
    sigma = 0.5 * np.polyfit(t[sel], np.log(K[sel]), 1)[0]
    return float(sigma), t, K


def initial_energy(nx, ny, lx=14.1319, ly=16.0):
    """Mean total energy density of the initial field (the case's expressions at the cell centres)."""
    x = (np.arange(nx) + 0.5) * lx / nx
    y = (np.arange(ny) + 0.5) * ly / ny
    X, Y = np.meshgrid(x, y)
    u = 0.2 * np.tanh(Y - 8)
    v = 0.0002 * np.sin(2 * np.pi * X / lx) * np.exp(-(Y - 8) ** 2)
    return float(np.mean(0.7142857142857143 / 0.4 + 0.5 * (u * u + v * v)))


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    out, series, ok = {"sigma_theory": SIGMA_THEORY, "grids": {}}, {}, True
    for nx, ny in GRIDS:
        # about 30 snapshots whatever the grid: the step count scales with nx
        d = study.run_cached(STUDY, f"{nx}x{ny}", CASE, f"nx={nx}", f"ny={ny}", f"save_interval={int(70 * nx / 96)}",
                             mpi=args.mpi, force=args.force)
        sigma, t, K = growth(d, nx, ny)
        series[nx] = (t, K)
        dump = fenice.read_dump(d / "final.npz")
        Ud = dump["U"]
        rho, _, _, p = fenice.primitive(Ud)
        e0 = initial_energy(nx, ny)
        rec = {"sigma": sigma, "mass_error": abs(Ud[:, 0].sum() / (nx * ny) - 1.0),
               "energy_error": abs(Ud[:, 3].sum() / (nx * ny) / e0 - 1.0), "relative_error": sigma / SIGMA_THEORY - 1,
               "finite": bool(np.isfinite(Ud).all()), "rho_min": float(rho.min()), "p_min": float(p.min()),
               }
        out["grids"][f"{nx}x{ny}"] = rec
        print(f"{nx}x{ny}: sigma = {sigma:.5f}  (theory {SIGMA_THEORY:.5f}, {100 * rec['relative_error']:+.1f} %)  "
              f"rho_min {rec['rho_min']:.3g}  p_min {rec['p_min']:.3g}")
    fine = out["grids"][f"{GRIDS[1][0]}x{GRIDS[1][1]}"]
    coarse = out["grids"][f"{GRIDS[0][0]}x{GRIDS[0][1]}"]
    out["pass_theory"] = bool(abs(fine["relative_error"]) <= BAND_SIGMA)
    out["pass_grids"] = bool(abs(coarse["sigma"] - fine["sigma"]) <= BAND_GRIDS * fine["sigma"])
    out["pass_conservation"] = bool(all(r["mass_error"] < BAND_CONS and r["energy_error"] < BAND_CONS
                                        for r in out["grids"].values()))
    out["pass_positive"] = all(r["finite"] and r["rho_min"] > 0 and r["p_min"] > 0 for r in out["grids"].values())
    ok = out["pass_theory"] and out["pass_grids"] and out["pass_positive"] and out["pass_conservation"]
    out["passed"] = bool(ok)
    fenice.write_summary(STUDY, out)

    plt = study.use_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for nx, (t, K) in series.items():
        ax.semilogy(t, K, "o-", ms=3, lw=1, label=f"{nx} cells")
    t = np.linspace(*FIT, 20)
    k0 = series[GRIDS[1][0]][1][np.argmin(abs(series[GRIDS[1][0]][0] - FIT[0]))]
    ax.semilogy(t, k0 * np.exp(2 * SIGMA_THEORY * (t - FIT[0])), "k--", label=r"theory $2\sigma = 0.3794\,U/\delta$")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\sum \rho v^2$"); ax.legend()
    plt.tight_layout()
    study.save(fig, STUDY, "kelvin_helmholtz.png", docs=args.docs)
    print("ALL BANDS PASSED" if ok else "SOME BANDS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
