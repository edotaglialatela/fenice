"""Grid convergence of benchmark results: observed order and Grid Convergence Index (Roache, 2009, ch. 5).

    python cases/validation/benchmarks/gci.py                  # cavity, airfoil, 3-D Taylor-Green
    python cases/validation/benchmarks/gci.py cavity           # one of them
    python cases/validation/benchmarks/gci.py --docs           # also docs/img/gci.png

Quantities without an exact solution get an error band from a systematic grid refinement (factor 2, the
same ratio in every direction), not just a comparison with another result:

* lid-driven cavity, Re = 400: the stream function of the primary vortex and the extrema of the
  centreline velocities on 32^2, 64^2, 128^2 (and 256^2 with --cavity-grids), for both orders, against
  Ghia, Ghia & Shin (1982);
* IBM airfoil (NACA 0012, 10 degrees, Re = 1000): Cd and Cl averaged over 2 <= t <= 6 on 192 x 96,
  384 x 192 and 768 x 384, plus the AMR run of airfoil_amr.py as an indicative extra point (an effective
  grid size from its cell count: changing thresholds is not a systematic refinement);
* 3-D Taylor-Green vortex, Re = 1600: E_k(20)/E_k(0) and the peak dissipation on 32^3, 64^3, 96^3 (a
  variable ratio, solved iteratively), against the spectral DNS peak.

For every functional it prints the observed order p, the factor of safety Fs (1.25 with three grids whose
order is close to the formal one, 3 otherwise), the fine-grid GCI, the Richardson-extrapolated value and
the asymptotic-range check GCI23 / (r^p GCI12), which is 1 in the asymptotic range. Triplets that do not
converge monotonically get no GCI. The runs are those of the other studies (same folders and tags).
Figure: gci.png; the table goes to results/validation/gci/gci_table.md.
"""
import json
import sys

import numpy as np

import fenice
from fenice import study
from fenice.convergence import gci_triplet

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))   # sibling helpers of this folder
from cavity import (CASE as CAVITY, GHIA_U, GHIA_V, GHIA_VORTEX, GHIA_X, GHIA_Y, centrelines,  # noqa: E402
                    fields, stream_function)

STUDY = "gci"
AIRFOIL, AIRFOIL_AMR = "IBM Airfoil", "IBM Airfoil AMR"
T_AIR = (2.0, 6.0)
DNS_PEAK = 0.0127   # spectral DNS 512^3 at Re = 1600, t ~ 9
L_TG = 2.0 * np.pi


# ----------------------------------------------------------------------------- functionals
def parabolic_extremum(x, f, k):
    """Value and position of the extremum of the parabola through f[k-1], f[k], f[k+1]."""
    xs, fs = np.asarray(x[k - 1:k + 2], float), np.asarray(f[k - 1:k + 2], float)
    a, b, c = np.polyfit(xs - xs[1], fs, 2)
    if a == 0.0:
        return float(f[k]), float(x[k])
    return float(c - b * b / (4 * a)), float(x[k] - b / (2 * a))


def extremum_1d(x, f, kind):
    k = int(np.argmin(f) if kind == "min" else np.argmax(f))
    return parabolic_extremum(x, f, k)[0] if 0 < k < len(f) - 1 else float(f[k])


def extremum_2d(psi):
    """Minimum of a smooth cell-centred field: least-squares quadratic through the 3 x 3 cells around it."""
    j, i = np.unravel_index(np.argmin(psi), psi.shape)
    if not (0 < i < psi.shape[1] - 1 and 0 < j < psi.shape[0] - 1):
        return float(psi[j, i])
    dy, dx = np.meshgrid([-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], indexing="ij")
    f = psi[j - 1:j + 2, i - 1:i + 2].ravel()
    A = np.stack([np.ones(9), dx.ravel(), dy.ravel(), dx.ravel() ** 2, dy.ravel() ** 2, (dx * dy).ravel()], 1)
    c = np.linalg.lstsq(A, f, rcond=None)[0]
    M = np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])
    s = np.linalg.solve(M, -c[1:3])
    return float(c[0] + 0.5 * c[1:3] @ s)


def window_mean(t, f, t0, t1):
    """Time average over [t0, t1] by the trapezoidal rule on the (non-uniform) monitor samples."""
    m = (t > t0) & (t < t1)
    tt = np.concatenate([[t0], t[m], [t1]])
    ff = np.concatenate([[np.interp(t0, t, f)], f[m], [np.interp(t1, t, f)]])
    return float(np.sum(0.5 * (ff[1:] + ff[:-1]) * np.diff(tt)) / (t1 - t0))


# ----------------------------------------------------------------------------- studies
class Row:
    """One functional on a sequence of grids, coarsest first."""

    def __init__(self, study_name, name, h, values, p_formal, reference=None, ref_label="", note=""):
        self.study, self.name, self.h, self.values = study_name, name, list(h), list(values)
        self.p_formal, self.reference, self.ref_label, self.note = p_formal, reference, ref_label, note
        self.triplets = []
        for k in range(len(self.h) - 3, -1, -1):   # finest triplet first
            f = (self.values[k + 2], self.values[k + 1], self.values[k])
            h3 = (self.h[k + 2], self.h[k + 1], self.h[k])
            self.triplets.append(gci_triplet(f, h3, p_formal))

    @property
    def main(self):
        return self.triplets[0]


def cavity_rows(args):
    rows = []
    for order in (2, 1):
        data = {}
        for n in args.cavity_grids:
            d = study.run_cached("cavity", f"n{n}_o{order}", CAVITY, f"nx={n}", f"ny={n}", "t_end=40",
                                 "save_interval=1000000000", "monitor_interval=50",
                                 f"use_muscl={'true' if order == 2 else 'false'}", mpi=args.mpi, force=False)
            u, v = fields(d / "final.npz", n)
            y, uc, vc = centrelines(u, v, n)
            data[n] = {"psi": extremum_2d(stream_function(u, n)), "u_min": extremum_1d(y, uc, "min"),
                       "v_max": extremum_1d(y, vc, "max"), "v_min": extremum_1d(y, vc, "min")}
        h = [1.0 / n for n in args.cavity_grids]
        ghia = {"psi": GHIA_VORTEX[2], "u_min": float(GHIA_U.min()), "v_max": float(GHIA_V.max()),
                "v_min": float(GHIA_V.min())}
        labels = {"psi": r"$\psi_{\min}$", "u_min": r"$u_{\min}$ on $x = 0.5$", "v_max": r"$v_{\max}$ on $y = 0.5$",
                  "v_min": r"$v_{\min}$ on $y = 0.5$"}
        for key in ("psi", "u_min", "v_max", "v_min"):
            rows.append(Row(f"cavity, {'MUSCL' if order == 2 else '1st order'}", labels[key], h,
                            [data[n][key] for n in args.cavity_grids], order, ghia[key], "Ghia (1982)",
                            "tabulated extremum" if key != "psi" else ""))
    return rows


def airfoil_rows(args):
    grids = [(192, 96), (384, 192), (768, 384)]
    tags = ["uniform_coarse", "uniform_mid", "uniform_fine"]
    common = (f"t_end={T_AIR[1]}", "monitor_interval=10", "save_interval=1000000000", "use_mpi=false")
    loads = []
    for (nx, ny), tag in zip(grids, tags):
        d = study.run_cached("airfoil_amr", tag, AIRFOIL, f"nx={nx}", f"ny={ny}", *common, force=False)
        m = fenice.read_monitor(study.case_output(d))
        loads.append((window_mean(m["t"], m["Cd"], *T_AIR), window_mean(m["t"], m["Cl"], *T_AIR)))
    amr = study.run_cached("airfoil_amr", "amr", AIRFOIL_AMR, *common, force=False)
    ma = fenice.read_monitor(study.case_output(amr))
    amr_loads = (window_mean(ma["t"], ma["Cd"], *T_AIR), window_mean(ma["t"], ma["Cl"], *T_AIR))
    cells_amr = len(fenice.read_dump(amr / "final.npz")["leaf_meta"]) * 16 * 16
    h_amr = 12.0 / 192 * np.sqrt(192 * 96 / cells_amr)   # effective: the uniform mesh with as many cells
    rows = []
    for k, (key, label) in enumerate((("Cd", r"$\overline{C_d}$, $2 \leq t \leq 6$"), ("Cl", r"$\overline{C_l}$, $2 \leq t \leq 6$"))):
        r = Row("IBM airfoil, Re = 1000", label, [12.0 / nx for nx, _ in grids],
                [x[k] for x in loads], 2)
        r.amr = (h_amr, amr_loads[k])
        rows.append(r)
    return rows


def tg3d_rows(args):
    grids = [32, 64, 96]
    ke, peak = [], []
    for n in grids:
        d = study.study_dir("taylor_green_3d") / f"n{n}"
        m = fenice.read_monitor(study.case_output(d))
        t, e = m["t"], m["kinetic_energy"]
        ke.append(float(e[-1] / e[0]))
        peak.append(float(np.max(-np.gradient(e, t))))
    h = [L_TG / n for n in grids]
    return [Row("3-D Taylor-Green, Re = 1600", r"$E_k(20)/E_k(0)$", h, ke, 2),
            Row("3-D Taylor-Green, Re = 1600", r"peak $\varepsilon$", h, peak, 2, DNS_PEAK, "DNS 512$^3$")]


# ----------------------------------------------------------------------------- report
def fmt(x, spec=".4g"):
    return "-" if x is None else format(x, spec)


def report(rows):
    lines = ["| study | quantity | grids (coarse to fine) | $p_{obs}$ | $F_s$ | fine-grid GCI | extrapolated | asymptotic ratio | reference |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        for k, t in enumerate(r.triplets):
            sub = r.values[len(r.values) - 3 - k:len(r.values) - k]
            note = "; ".join(t.notes)
            ref = ""
            if k == 0 and r.reference is not None:
                fine = r.values[-1]
                ex = t.f_extrapolated if t.f_extrapolated is not None else fine
                gci_abs = (t.gci_fine or 0.0) * abs(fine)
                band = ("no GCI band" if t.gci_fine is None else
                        f"{'inside' if abs(fine - r.reference) <= gci_abs else 'outside'} the GCI band")
                ref = f"{r.ref_label} {r.reference:.4g}; fine - ref = {fine - r.reference:+.2e}, {band}"
            print(f"{r.study:30s} {r.name.replace('$', '')[:34]:34s} p_obs {fmt(t.p_obs, '.2f'):>5s}  Fs {t.fs:4.2f}  "
                  f"GCI {fmt(None if t.gci_fine is None else 100 * t.gci_fine, '.3g'):>7s} %  "
                  f"f_extrap {fmt(t.f_extrapolated):>10s}  asym {fmt(t.asymptotic_ratio, '.2f'):>5s}  "
                  f"{'' if k == 0 else '(coarser triplet)'} {note}")
            if ref:
                print(f"{'':30s}   {ref}")
            vals = ", ".join(f"{v:.4g}" for v in sub)
            lines.append(f"| {r.study}{'' if k == 0 else ' (coarser triplet)'} | {r.name} | {vals} | {fmt(t.p_obs, '.2f')} | "
                         f"{t.fs:g} | {'-' if t.gci_fine is None else fmt(100 * t.gci_fine, '.3g') + ' %'} | "
                         f"{fmt(t.f_extrapolated)} | {fmt(t.asymptotic_ratio, '.2f')} | {ref or note} |")
    return "\n".join(lines)


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("which", nargs="*", choices=[[]] + ["cavity", "airfoil", "taylor_green_3d"], default=[],
                    help="studies to include (default: all)")
    ap.add_argument("--cavity-grids", type=int, nargs="+", default=[32, 64, 128],
                    help="cells per side of the cavity grids (add 256 for a second triplet)")
    args = ap.parse_args()
    which = args.which or ["cavity", "airfoil", "taylor_green_3d"]
    plt = study.use_style()
    rows = []
    for name, fn in (("cavity", cavity_rows), ("airfoil", airfoil_rows), ("taylor_green_3d", tg3d_rows)):
        if name in which:
            rows += fn(args)
    table = report(rows)
    out = study.study_dir(STUDY)
    (out / "gci_table.md").write_text(table + "\n", encoding="utf8")
    (out / "gci.json").write_text(json.dumps([
        {"study": r.study, "name": r.name, "h": r.h, "values": r.values, "p_formal": r.p_formal,
         "triplets": [{"p_obs": t.p_obs, "fs": t.fs, "gci": t.gci_fine, "f_extrapolated": t.f_extrapolated,
                       "asymptotic_ratio": t.asymptotic_ratio, "notes": t.notes} for t in r.triplets]}
        for r in rows], indent=1), encoding="utf8")

    # figure: one panel per functional of the finest triplet, value against h with the GCI band
    groups = [r for r in rows if r.study.startswith(("cavity, MUSCL", "IBM", "3-D"))]
    n = len(groups)
    cols = 4
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4.0 * cols, 3.3 * ((n + cols - 1) // cols)))
    for ax, r in zip(np.atleast_1d(axes).ravel(), groups):
        t = r.main
        ax.plot(r.h, r.values, "o-", color="C0", label="solver")
        if t.f_extrapolated is not None:
            ax.axhline(t.f_extrapolated, color="k", ls=":", lw=1.2, label="Richardson")
            ax.errorbar([r.h[-1]], [r.values[-1]], yerr=[abs(t.gci_fine) * abs(r.values[-1])], color="C3", capsize=4,
                        lw=2, label=f"GCI {100 * abs(t.gci_fine):.2g} %")
        if r.reference is not None:
            ax.axhline(r.reference, color="C2", ls="--", lw=1.4, label=r.ref_label)
        if hasattr(r, "amr"):
            ax.plot([r.amr[0]], [r.amr[1]], "*", color="C1", ms=11, label="AMR (indicative)")
        ax.set_xscale("log")
        ax.set_title(f"{r.name}\n{r.study}", fontsize=8.5)
        ax.set_xlabel("grid size $h$")
        ax.legend(fontsize=7)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    study.save(fig, STUDY, "gci.png", args.docs)


if __name__ == "__main__":
    main()
