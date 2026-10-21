"""Lid-driven cavity at Re = 400 against Ghia, Ghia & Shin (1982), with grid convergence.

    python cases/validation/cavity.py                 # 32^2, 64^2, 128^2; 1st and 2nd order
    python cases/validation/cavity.py --grids 32 64 128 256 --orders 2 --mpi 16 --docs

Runs the "Lid Driven Cavity" case into results/validation/cavity/n<N>_o<order>/ (existing
runs are reused; --force re-runs them) to a steady state
(t_end = 40, checked from the kinetic energy of the monitor) for each grid and spatial order
(1: Godunov, 2: MUSCL), then compares

* u on the vertical centreline and v on the horizontal centreline with Ghia's 17-point
  tables (Re = 400), as the RMS error over the interior points;
* the primary vortex: minimum of the stream function and its position.

Figures: cavity_validation.png (profiles, error vs grid), cavity_streamlines.png.
"""
import numpy as np

import fenice
from fenice import study

STUDY, CASE = "cavity", "Lid Driven Cavity"

# Ghia, Ghia & Shin (1982), Re = 400: Table I (u on x = 0.5) and Table II (v on y = 0.5)
GHIA_Y = np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
                   0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000])
GHIA_U = np.array([1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477,
                   -0.17119, -0.32726, -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.00000])
GHIA_X = np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000,
                   0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000])
GHIA_V = np.array([0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598, 0.05186,
                   0.30174, 0.30203, 0.28124, 0.22965, 0.20920, 0.19713, 0.18360, 0.00000])
GHIA_VORTEX = (0.5547, 0.6055, -0.113909)   # primary vortex centre and stream function there
# Table II's v = -0.23827 at x = 0.9063 cannot lie on a smooth profile between its neighbours
# (-0.44993 at 0.8594, -0.22847 at 0.9453); every grid converges to about -0.39 there. It looks
# like a misprint: it is plotted but left out of the error norm.
V_SUSPECT = np.isclose(GHIA_X, 0.9063)


def run(n, order, args):
    d = study.run_cached(STUDY, f"n{n}_o{order}", CASE, f"nx={n}", f"ny={n}", "t_end=40",
                         "save_interval=1000000000", "monitor_interval=50",
                         f"use_muscl={'true' if order == 2 else 'false'}", mpi=args.mpi, force=args.force)
    return d / "final.npz"


def fields(dump, n):
    U = fenice.read_dump(dump)["U"]
    rho = U[:, 0].reshape(n, n)
    return U[:, 1].reshape(n, n) / rho, U[:, 2].reshape(n, n) / rho   # [j, i], row 0 at y = 0


def centrelines(u, v, n):
    """u(y) on x = 0.5 and v(x) on y = 0.5 at the cell centres, plus the wall values."""
    c = (np.arange(n) + 0.5) / n
    mid = lambda a, axis: a.take(n // 2, axis) if n % 2 else 0.5 * (a.take(n // 2 - 1, axis) + a.take(n // 2, axis))
    y = np.concatenate([[0.0], c, [1.0]])
    uc = np.concatenate([[0.0], mid(u, 1), [1.0]])     # lid: u = 1
    vc = np.concatenate([[0.0], mid(v, 0), [0.0]])
    return y, uc, vc


def stream_function(u, n):
    """psi(x, y) = integral of u dy from the bottom wall (cell-centre values, psi = 0 on the walls)."""
    h = 1.0 / n
    psi = np.cumsum(u, axis=0) * h - 0.5 * u * h
    return psi


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--grids", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    args = ap.parse_args()
    plt = study.use_style()

    inner = slice(1, -1)   # Ghia points without the wall values
    res = {}
    for order in args.orders:
        for n in args.grids:
            u, v = fields(run(n, order, args), n)
            y, uc, vc = centrelines(u, v, n)
            eu = np.sqrt(np.mean((np.interp(GHIA_Y[inner], y, uc) - GHIA_U[inner]) ** 2))
            vk = ~V_SUSPECT
            vk[0] = vk[-1] = False   # walls
            ev = np.sqrt(np.mean((np.interp(GHIA_X[vk], y, vc) - GHIA_V[vk]) ** 2))
            psi = stream_function(u, n)
            j, i = np.unravel_index(np.argmin(psi), psi.shape)
            vortex = ((i + 0.5) / n, (j + 0.5) / n, psi[j, i])
            res[(order, n)] = dict(y=y, uc=uc, vc=vc, eu=eu, ev=ev, vortex=vortex, u=u, v=v)
            print(f"      order {order}, {n:4d}^2: RMS error u {eu:.4f}, v {ev:.4f}; vortex at "
                  f"({vortex[0]:.4f}, {vortex[1]:.4f}), psi_min {vortex[2]:.5f}")
    print(f"      Ghia: vortex at ({GHIA_VORTEX[0]}, {GHIA_VORTEX[1]}), psi_min {GHIA_VORTEX[2]}")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5.6))
    fig.suptitle(r"Lid-driven cavity, $Re = 400$: centreline velocities against Ghia et al. (1982)", fontsize=15,
                 fontweight="bold")
    shades = {1: plt.cm.Oranges(np.linspace(0.45, 0.95, len(args.grids))),
              2: plt.cm.Blues(np.linspace(0.45, 0.95, len(args.grids)))}
    for order in args.orders:
        for k, n in enumerate(args.grids):
            r = res[(order, n)]
            ls = "-" if order == 2 else "--"
            lab = f"{n}$^2$, {'MUSCL' if order == 2 else '1st order'}"
            a1.plot(r["uc"], r["y"], ls, color=shades[order][k], lw=1.6, label=lab)
            a2.plot(r["y"], r["vc"], ls, color=shades[order][k], lw=1.6, label=lab)
    for a, xs, ys in ((a1, GHIA_U, GHIA_Y), (a2, GHIA_X[~V_SUSPECT], GHIA_V[~V_SUSPECT])):
        a.plot(xs, ys, "o", ms=6, mfc="white", mec="k", mew=1.4, label="Ghia et al. (1982)", zorder=5)
        a.grid(True, ls="--", alpha=0.5)
    a2.plot(GHIA_X[V_SUSPECT], GHIA_V[V_SUSPECT], "x", ms=8, mew=2, color="#c0392b", zorder=5,
            label="Ghia, x = 0.9063 (likely misprint;\nnot in the error)")
    a2.legend(loc="lower left")
    a1.set_xlabel("$u$ on $x = 0.5$"); a1.set_ylabel("$y$"); a1.set_title("Vertical centreline")
    a2.set_xlabel("$x$"); a2.set_ylabel("$v$ on $y = 0.5$"); a2.set_title("Horizontal centreline")
    a1.legend(loc="lower right")
    grids = np.array(args.grids, dtype=float)
    for order in args.orders:
        e = np.array([0.5 * (res[(order, n)]["eu"] + res[(order, n)]["ev"]) for n in args.grids])
        a3.loglog(grids, e, "o-" if order == 2 else "s--", color=shades[order][-1], lw=2,
                  label=f"{'MUSCL' if order == 2 else '1st order'} (slope {np.polyfit(np.log(grids), np.log(e), 1)[0]:.2f})")
    a3.loglog(grids, (grids / grids[0]) ** -1.0 * e[0], ":", color="gray", label="slope $-1$")
    a3.loglog(grids, (grids / grids[0]) ** -2.0 * e[0], "-.", color="gray", label="slope $-2$")
    a3.set_xlabel("Cells per side"); a3.set_ylabel("RMS error vs Ghia (u and v)"); a3.set_title("Grid convergence")
    a3.grid(True, which="both", ls="--", alpha=0.5)
    a3.legend()
    plt.tight_layout()
    study.save(fig, STUDY, "cavity_validation.png", args.docs)

    order, n = max(args.orders), max(args.grids)
    r = res[(order, n)]
    c = (np.arange(n) + 0.5) / n
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    speed = np.hypot(r["u"], r["v"])
    ax.imshow(speed, origin="lower", extent=(0, 1, 0, 1), cmap="magma", alpha=0.85)
    ax.streamplot(c, c, r["u"], r["v"], color="white", density=2.2, linewidth=0.7, arrowsize=0.7)
    ax.plot(*r["vortex"][:2], "o", ms=8, mfc="#2ecc71", mec="white", label="vortex centre (Fenice)")
    ax.plot(*GHIA_VORTEX[:2], "+", ms=14, mew=2.2, color="#3498db", label="Ghia et al. (1982)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title(rf"Streamlines and $|\mathbf{{u}}|$, {n}$^2$, {'MUSCL' if order == 2 else '1st order'}")
    ax.legend(loc="lower left", fontsize=9)
    study.save(fig, STUDY, "cavity_streamlines.png", args.docs)


if __name__ == "__main__":
    main()
