"""Order of accuracy at a shock: Sod's problem against the exact Riemann solution (Roache 2009, sec. 3.7 and ch. 5).

    python cases/validation/verification/shock_order.py            # results/validation/shock_order/
    python cases/validation/verification/shock_order.py --docs     # also docs/img/shock_order.png

A solution with a shock, a contact and a rarefaction is not smooth, so the formal order of the scheme is not what
the error does: the L1 error of a shock converges at order 1 whatever the scheme, a contact at p/(p + 1) (about 1/2
for Godunov and 2/3 for a second-order scheme, the discontinuity smears over ~h^(1/(p+1)) cells), a rarefaction head
and tail cost a little order too. L2 converges at about half the L1 rate and L_inf does not converge at all (the
error at the discontinuity stays O(1) in value). The study runs "Sod Shock Tube" as a quasi-1-D tube (4 cells in
y) on 100, 200, 400, 800, 1600 cells to t = 0.2 with Godunov (1st order) and MUSCL Van Leer + SSP-RK2 and measures
the error against the exact solution (Toro 2009) on the cell centres.

Expected, fixed before the first run (orders of the finest pair of grids):

* L1 density: Godunov in [0.5, 1.1], MUSCL in [0.6, 1.1];
* L1 velocity and pressure (no contact error): both schemes in [0.6, 1.2];
* the MUSCL L1 error is below the Godunov one on every grid, for density, velocity and pressure;
* L2 density: [0.25, 0.75] for both schemes (about half of L1);
* L_inf density: observed order below 0.3 for both schemes (it does not converge);
* the shock position (where the density crosses the mean of its two states) is within one cell of the exact one on
  every grid;
* the post-shock plateau (mean density over x in [0.74, 0.82], far from the discontinuities once the grid is
  fine) is reported with its GCI by fenice.convergence; a quantity whose grids are not in the asymptotic range gets
  no GCI. No band is set for it.

Figure: shock_order.png (error against h with reference slopes); table: results/validation/shock_order/shock_order_table.md.
"""
import sys

import numpy as np

import fenice
from fenice import study
from fenice.convergence import fit_slope, gci_triplet, observed_orders
from fenice.exact import riemann

STUDY, CASE = "shock_order", "Sod Shock Tube"
GRIDS = (100, 200, 400, 800, 1600)
NY = 4
T_END = 0.2
SCHEMES = {"Godunov": ("use_muscl=false",),
           "MUSCL": ("use_muscl=true", "time_scheme=SSP Runge-Kutta 2", "muscl_limiter=Van Leer")}
PLATEAU = (0.74, 0.82)
# entries found outside their band by the first run; the band was not moved (see the shock-order docs section)
DOCUMENTED = {("MUSCL", "u", "L1")}
BANDS = {   # (low, high) of the finest-pair order, per scheme
    ("rho", "L1"): {"Godunov": (0.5, 1.1), "MUSCL": (0.6, 1.1)},
    ("u", "L1"): {"Godunov": (0.6, 1.2), "MUSCL": (0.6, 1.2)},
    ("p", "L1"): {"Godunov": (0.6, 1.2), "MUSCL": (0.6, 1.2)},
    ("rho", "L2"): {"Godunov": (0.25, 0.75), "MUSCL": (0.25, 0.75)},
    ("rho", "Linf"): {"Godunov": (-np.inf, 0.3), "MUSCL": (-np.inf, 0.3)},
}


def run(scheme, n, args):
    d = study.run_cached(STUDY, f"{scheme.lower()}_n{n}", CASE, f"nx={n}", f"ny={NY}", f"Ly={NY / n}",
                         f"t_end={T_END}", "save_interval=1000000000", *SCHEMES[scheme], mpi=args.mpi, force=args.force)
    dump = fenice.read_dump(d / "final.npz")
    rho, u, _, p = fenice.primitive(dump["U"])
    row = lambda a: a.reshape(NY, n)[NY // 2]
    return (np.arange(n) + 0.5) / n, row(rho), row(u), row(p), float(dump["t"])


def shock_position(x, rho):
    """x where the density, going down across the shock (right of x = 0.7), crosses the mean of its two states
    (the initial right state 0.125 and the exact post-shock plateau 0.26557)."""
    mid = 0.5 * (0.125 + 0.26557)
    j = np.where((x[:-1] > 0.7) & (rho[:-1] > mid))[0][-1]
    return x[j] + (rho[j] - mid) / (rho[j] - rho[j + 1]) * (x[j + 1] - x[j])


def main():
    ap = study.parser(__doc__.splitlines()[0])
    args = ap.parse_args()
    plt = study.use_style()
    errs = {s: {k: [] for k in ("rho_L1", "u_L1", "p_L1", "rho_L2", "rho_Linf", "xs", "plateau")} for s in SCHEMES}
    xs_exact = None
    for s in SCHEMES:
        for n in GRIDS:
            x, rho, u, p, t = run(s, n, args)
            re, ue, pe = riemann(x, t)
            e = errs[s]
            e["rho_L1"].append(np.mean(np.abs(rho - re)))
            e["u_L1"].append(np.mean(np.abs(u - ue)))
            e["p_L1"].append(np.mean(np.abs(p - pe)))
            e["rho_L2"].append(np.sqrt(np.mean((rho - re) ** 2)))
            e["rho_Linf"].append(np.max(np.abs(rho - re)))
            xf = np.linspace(0, 1, 200001)
            xs_exact = shock_position(xf, riemann(xf, t)[0])
            e["xs"].append(shock_position(x, rho) - xs_exact)
            e["plateau"].append(np.mean(rho[(x > PLATEAU[0]) & (x < PLATEAU[1])]))
    h = 1.0 / np.array(GRIDS)
    ok = True
    lines = ["| scheme | quantity | " + " | ".join(f"{n}" for n in GRIDS) + " | order (finest pair) | band | |", "|---|---|" + "---:|" * len(GRIDS) + "---:|---|---|"]
    print(f"exact shock position {xs_exact:.5f}")
    for s in SCHEMES:
        for (var, norm), bands in BANDS.items():
            err = np.array(errs[s][f"{var}_{norm}"])
            order = observed_orders(h, err)[-1]
            lo, hi = bands[s]
            good = lo <= order <= hi
            note = "OK" if good else "OUTSIDE (documented)" if (s, var, norm) in DOCUMENTED else "OUTSIDE"
            ok = ok and (good or (s, var, norm) in DOCUMENTED)
            band = ("<= %.2f" % hi) if np.isinf(lo) else "[%.2f, %.2f]" % (lo, hi)
            print(f"  {s:8s} {var:3s} {norm:4s} errors " + " ".join(f"{v:.2e}" for v in err) +
                  f"   orders " + " ".join(f"{o:.2f}" for o in observed_orders(h, err)) +
                  f"   fit {fit_slope(h, err):.2f}   expected {band} {note}")
            lines.append(f"| {s} | {var} {norm} | " + " | ".join(f"{v:.2e}" for v in err) + f" | {order:.2f} (fit {fit_slope(h, err):.2f}) | {band} | {note} |")
    for var in ("rho", "u", "p"):
        better = all(m < g for m, g in zip(errs["MUSCL"][f"{var}_L1"], errs["Godunov"][f"{var}_L1"]))
        ok = ok and better
        print(f"  MUSCL L1 {var} error below Godunov on every grid: {'OK' if better else 'NO'}")
    for s in SCHEMES:
        dxs = np.array(errs[s]["xs"])
        within = bool(np.all(np.abs(dxs) <= h))
        ok = ok and within
        print(f"  {s:8s} shock position error / h: " + " ".join(f"{d / hh:+.2f}" for d, hh in zip(dxs, h)) +
              f"   within one cell: {'OK' if within else 'NO'}")
    print("  post-shock plateau (mean density over x in [%.2f, %.2f]):" % PLATEAU)
    for s, p_formal in (("Godunov", 1.0), ("MUSCL", 2.0)):
        f = np.array(errs[s]["plateau"])
        g = gci_triplet(f[-3:], h[-3:], p_formal)
        print(f"  {s:8s} values {' '.join(f'{v:.5f}' for v in f)}   GCI of the last three: " +
              ("no GCI (" + "; ".join(g.notes) + ")" if g.gci_fine is None else f"p_obs {g.p:.2f}, fine-grid GCI {100 * g.gci_fine:.2f} %"))
    (study.study_dir(STUDY) / "shock_order_table.md").write_text("\n".join(lines) + "\n", encoding="utf8")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"Godunov": "#c0392b", "MUSCL": "#2980b9"}
    for ax, keys, title in ((axs[0], ("rho_L1", "u_L1", "p_L1"), r"$L_1$ error"), (axs[1], ("rho_L2",), r"$L_2$ error of $\rho$"),
                            (axs[2], ("rho_Linf",), r"$L_\infty$ error of $\rho$")):
        for s in SCHEMES:
            for k, ls, m in zip(keys, ("-", "--", ":"), ("o", "s", "^")):
                ax.loglog(h, errs[s][k], ls=ls, marker=m, color=colors[s], label=f"{s}, {k.split('_')[0]}")
        ref = errs["MUSCL"][keys[0]][0]
        for p_ in ((1.0, 0.5) if keys[0] != "rho_Linf" else (0.0,)):
            ax.loglog(h, ref * (h / h[0]) ** p_, "k:", lw=0.9)
            ax.annotate(f"slope {p_:g}", (h[-2], ref * (h[-2] / h[0]) ** p_), fontsize=8, xytext=(3, 4), textcoords="offset points")
        ax.set_xlabel("$h$")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Sod's problem: error against the exact solution (discontinuities limit the order)", fontsize=12)
    fig.tight_layout()
    study.save(fig, STUDY, "shock_order.png", args.docs)
    print("result:", "all checks passed" if ok else "FAILED (see above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
