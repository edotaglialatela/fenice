"""A7: Noh implosion (exact solution) and Sedov-Taylor blast (scaling laws), 2-D, one quarter of the plane.

    python cases/validation/verification/implosions.py            # results/validation/implosions/
    python cases/validation/verification/implosions.py --docs     # also docs/img/implosions.png

Noh (gamma = 5/3, 200 x 200 and 100 x 100): exact density 16 behind the shock at r = t/3 and 1 + t/r ahead of it.
Sedov (gamma = 1.4, energy 1 deposited in a disc of 3.5 cells, 200 and 400 cells per unit length, runs to t = 0.05 ... 0.2):
the exact shock radius is R = (E t^2 / (alpha rho))^(1/4) with a constant alpha that this study does not compute, so it checks
the scaling (exponent 1/2), the peak density, conservation and symmetry, not the prefactor.
Bands, fixed before the first run:
  Noh:   all finite, rho, p > 0; shock radius (half-way density between the plateau 16 and the local pre-shock value) within
         3 cells of t/3; mean density on 0.08 <= r <= 0.15 within 10 % of 16; relative L1 density error on 0.04 <= r <= 0.5
         below 8 %; the error decreases from 100 to 200 cells; |rho(i,j) - rho(j,i)| <= 1e-9 rho_max.
  Sedov: energy equal to its initial value to a relative 1e-11 (the waves stay inside the domain); finite, rho, p > 0;
         exponent of R(t) (radius of the largest radially averaged density, t = 0.05, 0.1, 0.15, 0.2) within 0.5 +- 0.03;
         peak radially averaged density between 3.5 and 6.2 (strong-shock limit 6); R(0.2) of 200 and 400 cells within
         2 cells of the coarser grid; symmetry as above.
"""
import numpy as np

import fenice
from fenice import study

STUDY = "implosions"
GAMMA_NOH, GAMMA_SEDOV = 5.0 / 3.0, 1.4
NOH_SHOCK_SPEED = (GAMMA_NOH - 1) / 2   # unit inflow speed


def noh_case(n):
    text = (fenice.ROOT / "cases" / "noh_implosion" / "case.toml").read_text()
    return text.replace("nx = 200", f"nx = {n}").replace("ny = 200", f"ny = {n}").replace(
        'name = "Noh Implosion"', f'name = "Noh {n}"')


def sedov_case(n, t_end, energy=1.0, radius_cells=3.5):
    """Sedov blast in the quarter [0, 1]^2: `energy` (the full-plane value) in the cells whose centre is within
    radius_cells of the origin; the background pressure is 1e-5, density 1."""
    dx = 1.0 / n
    cells = sum(1 for i in range(n) for j in range(n) if (i + 0.5) ** 2 + (j + 0.5) ** 2 <= radius_cells ** 2)
    p_blast = (GAMMA_SEDOV - 1) * (energy / 4) / (cells * dx * dx)
    return f"""# Sedov-Taylor blast in the quarter plane (symmetry walls on x = 0, y = 0): energy {energy} of the full plane in {cells} cells
[case]
name = "Sedov {n} t{t_end}"

[mesh]
Lx = 1.0
Ly = 1.0
nx = {n}
ny = {n}

[boundaries]
bc_left = "wall"
bc_bottom = "wall"
bc_right = "transmissive"
bc_top = "transmissive"

[numerics]
riemann_solver = "HLLC"
reconstruction = "muscl"
muscl_limiter = "MC"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.4

[run]
t_end = {t_end!r}
save_interval = 100000000

[initial]
state = {{ rho = 1.0, u = 0.0, v = 0.0, p = 1e-5 }}

[[initial.region]]
shape = "sphere"
center = [0.0, 0.0]
radius = {radius_cells * dx!r}
state = {{ p = {p_blast!r} }}
""", cells


def radial_profile(rho, n, nbins=None):
    """Mean of rho over the cells of each radial bin (width 1 cell), r = distance of the cell centre from the origin."""
    i, j = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
    r = np.hypot(i, j).ravel() / n
    b = (r * n).astype(int)
    nb = int(nbins or n)
    mask = b < nb
    sums = np.bincount(b[mask], weights=rho.ravel()[mask], minlength=nb)
    counts = np.bincount(b[mask], minlength=nb)
    return (np.arange(nb) + 0.5) / n, sums / np.maximum(counts, 1)


def symmetry(a, n):
    A = a.reshape(n, n)
    return float(np.max(np.abs(A - A.T)) / np.max(np.abs(A)))


def noh_exact(r, t):
    return np.where(r < NOH_SHOCK_SPEED * t, 16.0, 1.0 + t / r)


def analyse_noh(run_dir, n):
    d = fenice.read_dump(run_dir / "final.npz")
    U = d["U"]
    t = float(d["t"])
    rho, _, _, p = fenice.primitive(U)
    r, prof = radial_profile(rho, n)
    ex = noh_exact(r, t)
    sel = (r >= 0.04) & (r <= 0.5)
    l1 = float(np.sum(np.abs(prof - ex)[sel]) / np.sum(ex[sel]))
    plateau = (r >= 0.08) & (r <= 0.15)
    # shock: first radius (from outside) where the profile rises above half-way between the pre-shock value and 16
    rs_exact = NOH_SHOCK_SPEED * t
    pre = 1.0 + t / rs_exact
    half = 0.5 * (16.0 + pre)
    k = np.where(prof >= half)[0]
    k = k[r[k] > 0.05]
    rs = float(r[k.max()] + 0.5 / n) if k.size else float("nan")
    return {"t": t, "l1_rel": l1, "plateau_mean": float(prof[plateau].mean()), "shock_radius": rs,
            "shock_radius_exact": rs_exact, "shock_error_cells": float(abs(rs - rs_exact) * n),
            "finite": bool(np.isfinite(U).all()), "rho_min": float(rho.min()), "p_min": float(p.min()),
            "symmetry": symmetry(rho, n)}


def analyse_sedov(run_dir, n, e_init):
    d = fenice.read_dump(run_dir / "final.npz")
    U = d["U"]
    rho, _, _, p = fenice.primitive(U)
    r, prof = radial_profile(rho, n)
    k = int(np.argmax(prof))
    return {"t": float(d["t"]), "R": float(r[k]), "peak": float(prof[k]), "energy": float(U[:, 3].sum() / n ** 2),
            "energy_error": float(abs(U[:, 3].sum() / n ** 2 - e_init) / e_init),
            "finite": bool(np.isfinite(U).all()), "rho_min": float(rho.min()), "p_min": float(p.min()),
            "symmetry": symmetry(rho, n)}


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    out, ok = {}, True

    noh = {}
    for n in (100, 200):
        folder = study.write_case(STUDY, f"noh_{n}", noh_case(n))
        noh[n] = analyse_noh(study.run_cached(STUDY, f"noh_{n}", folder, mpi=args.mpi, force=args.force), n)
    out["noh"] = noh
    for n, a in noh.items():
        a["pass"] = bool(a["finite"] and a["rho_min"] > 0 and a["p_min"] > 0 and a["shock_error_cells"] <= 3.0
                         and abs(a["plateau_mean"] / 16.0 - 1) <= 0.10 and a["l1_rel"] < 0.08 and a["symmetry"] <= 1e-9)
        print(f"Noh {n}: shock r={a['shock_radius']:.4f} (exact {a['shock_radius_exact']:.4f}, {a['shock_error_cells']:.2f} cells)  "
              f"plateau {a['plateau_mean']:.2f}  L1 {a['l1_rel']:.3f}  symmetry {a['symmetry']:.1e}  rho_min {a['rho_min']:.2g}  "
              f"{'pass' if a['pass'] else 'FAIL'}")
        ok &= a["pass"]
    out["noh_l1_decreases"] = bool(noh[200]["l1_rel"] < noh[100]["l1_rel"])
    print(f"Noh error decreases with resolution: {out['noh_l1_decreases']}")
    ok &= out["noh_l1_decreases"]

    sedov = {}
    times = (0.05, 0.1, 0.15, 0.2)
    for n in (200, 400):
        sedov[n] = []
        for t in times:
            text, cells = sedov_case(n, t)
            folder = study.write_case(STUDY, f"sedov_{n}_t{t}", text)
            # initial energy density summed over the cells, per n^2: blast energy / 4 (area of the quarter is 1) + background
            e_init = 0.25 + (1 - cells / n ** 2) * 1e-5 / (GAMMA_SEDOV - 1)
            sedov[n].append(analyse_sedov(study.run_cached(STUDY, f"sedov_{n}_t{t}", folder, mpi=args.mpi,
                                                            force=args.force), n, e_init))
    out["sedov"] = sedov
    for n, runs in sedov.items():
        tt = np.array([a["t"] for a in runs])
        R = np.array([a["R"] for a in runs])
        exponent = float(np.polyfit(np.log(tt), np.log(R), 1)[0])
        peak = max(a["peak"] for a in runs)
        res = {"exponent": exponent, "peak_density": peak, "R_by_time": R.tolist(),
               "max_energy_error": max(a["energy_error"] for a in runs),
               "max_symmetry": max(a["symmetry"] for a in runs),
               "positive": all(a["finite"] and a["rho_min"] > 0 and a["p_min"] > 0 for a in runs)}
        res["pass"] = bool(abs(exponent - 0.5) <= 0.03 and 3.5 <= peak <= 6.2 and res["max_energy_error"] < 1e-11
                           and res["max_symmetry"] <= 1e-9 and res["positive"])
        out[f"sedov_{n}"] = res
        print(f"Sedov {n}: exponent {exponent:.3f}  peak density {peak:.2f}  energy error {res['max_energy_error']:.1e}  "
              f"symmetry {res['max_symmetry']:.1e}  R(t) = {np.round(R, 4).tolist()}  {'pass' if res['pass'] else 'FAIL'}")
        ok &= res["pass"]
    dR = abs(sedov[200][-1]["R"] - sedov[400][-1]["R"]) * 200
    out["sedov_R_difference_cells"] = float(dR)
    out["sedov_R_agree"] = bool(dR <= 2.0)
    print(f"Sedov R(0.2): 200 vs 400 cells differ by {dR:.2f} coarse cells (band 2)")
    ok &= out["sedov_R_agree"]
    out["passed"] = bool(ok)
    fenice.write_summary(STUDY, out)

    plt = study.use_style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for n, c in ((100, "C0"), (200, "C1")):
        d = fenice.read_dump(study.study_dir(STUDY) / f"noh_{n}" / "final.npz")
        r, prof = radial_profile(fenice.primitive(d["U"])[0], n)
        ax[0].plot(r, prof, c, lw=1, label=f"{n} x {n}")
    rr = np.linspace(0.005, 1, 800)
    ax[0].plot(rr, noh_exact(rr, 0.6), "k-", lw=1.2, label="exact")
    ax[0].set_xlabel("$r$"); ax[0].set_ylabel(r"density $\rho$"); ax[0].set_title("Noh, t = 0.6"); ax[0].legend()
    for n, c in ((200, "C0"), (400, "C1")):
        d = fenice.read_dump(study.study_dir(STUDY) / f"sedov_{n}_t0.2" / "final.npz")
        r, prof = radial_profile(fenice.primitive(d["U"])[0], n)
        ax[1].plot(r, prof, c, lw=1, label=f"{n} cells / unit")
    ax[1].set_xlabel("$r$"); ax[1].set_ylabel(r"radially averaged density"); ax[1].set_title("Sedov-Taylor, t = 0.2")
    ax[1].legend()
    plt.tight_layout()
    study.save(fig, STUDY, "implosions.png", docs=args.docs)
    print("ALL BANDS PASSED" if ok else "SOME BANDS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
