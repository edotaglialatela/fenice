"""Cylinder at Re = 100 in a box with 1.7 % blockage, on an AMR mesh: loads, Strouhal number, vorticity and mesh.

    python cases/validation/benchmarks/cylinder_low_blockage.py          # results/validation/cylinder_low_blockage/
    python cases/validation/benchmarks/cylinder_low_blockage.py --docs   # also docs/img/cylinder_low_blockage.png

Runs the "IBM Cylinder Low Blockage AMR" case (16 x 12 box, 200 x 150 base mesh, ghost-cell wall; speed indicator) twice to t = 60, unless
the runs exist: "l3" with three levels (20 cells per diameter at the body, OpenMP threads) and "l4" with four levels
(40 cells per diameter, MPI on 8 ranks by default).
Prints, for both, the mean drag, the lift amplitude and the Strouhal number over t >= 25: the dominant frequency of the
Hann-windowed lift spectrum, and the mean period between upward zero crossings of the lift.
Figure: cylinder_low_blockage.png (vorticity and AMR level of the l4 final state, lift histories, lift spectrum).
"""
import numpy as np
from scipy.ndimage import gaussian_filter

import fenice
from fenice import study

CASE = "IBM Cylinder Low Blockage AMR"
D, T_ST = 0.2, 25.0
REF = (0.164, 0.166)   # Re = 100 (Williamson 1996; Park et al. 1998); Cd = 1.33-1.35 there
WINDOW = (2.4, 11.0, 4.6, 7.4)   # x0, x1, y0, y1 of the pictures


def spectrum(t, y, pad=32):
    tu = np.linspace(t[0], t[-1], 4 * len(t))
    c = np.interp(tu, t, y)
    c = c - c.mean()
    w = np.hanning(len(c))
    amp = 2.0 * np.abs(np.fft.rfft(c * w, pad * len(c))) / w.sum()
    return np.fft.rfftfreq(pad * len(c), tu[1] - tu[0]) * D, amp


def st_crossings(t, y):
    """St from the mean period between upward zero crossings of the (mean-free) signal."""
    y = y - 0.5 * (y.max() + y.min())
    i = np.where((y[:-1] < 0) & (y[1:] >= 0))[0]
    tc = t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i])
    return D / np.mean(np.diff(tc)), len(tc) - 1


def raster(snap, window, h):
    """Leaf values of a .vtu snapshot on a uniform grid of cell size h over the window: level, u, v."""
    x0, x1, y0, y1 = window
    nx, ny = int(round((x1 - x0) / h)), int(round((y1 - y0) / h))
    lev, u, v = (np.zeros((ny, nx)) for _ in range(3))
    vel = snap["Velocity"]
    keep = np.where((snap["x1"] > x0) & (snap["x0"] < x1) & (snap["y1"] > y0) & (snap["y0"] < y1))[0]
    for k in keep:
        i0, i1 = max(int(round((snap["x0"][k] - x0) / h)), 0), min(int(round((snap["x1"][k] - x0) / h)), nx)
        j0, j1 = max(int(round((snap["y0"][k] - y0) / h)), 0), min(int(round((snap["y1"][k] - y0) / h)), ny)
        lev[j0:j1, i0:i1], u[j0:j1, i0:i1], v[j0:j1, i0:i1] = snap["level"][k], vel[k, 0], vel[k, 1]
    return lev, u, v


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    r3 = study.run_cached("cylinder_low_blockage", "l3", CASE, mpi=0, threads=16, force=args.force,
                          done="*/*_monitor.csv")
    out3 = study.case_output(r3)
    ck = next(out3.glob("*_restart.npz"))
    r4 = study.run_cached("cylinder_low_blockage", "l4", CASE, "amr_max_level=4", mpi=args.mpi or 8, force=args.force,
                          done="*/*_monitor.csv")
    out4 = study.case_output(r4)
    d3, d4 = fenice.read_monitor(out3), fenice.read_monitor(out4)
    res = {}
    for name, d, cells in (("l3", d3, 20), ("l4", d4, 40)):
        t, cl, cd = d["t"], d["Cl"], d["Cd"]
        late = t >= T_ST
        st_f, a_f = spectrum(t[late], cl[late])
        m = (st_f > 0.05) & (st_f < 0.4)
        st_spec = st_f[m][np.argmax(a_f[m])]
        st_zc, n = st_crossings(t[late], cl[late])
        res[name] = (st_f, a_f, st_spec)
        print(f"{name} ({cells} cells/D), t >= {T_ST:g}: Cd = {cd[late].mean():.4f}, lift amplitude = "
              f"{0.5 * (cl[late].max() - cl[late].min()):.4f}, St = {st_spec:.4f} (spectrum), {st_zc:.4f} "
              f"(zero crossings, {n} periods)")
    print(f"literature, Re = 100: St = {REF[0]}-{REF[1]}, Cd = 1.33-1.35")
    t3, t4 = d3["t"], d4["t"]
    st_f, a_f, st_spec = res["l4"]

    snap = fenice.read_vtu(fenice.read_pvd(out4)[-1][1])
    n_leaf, levels = len(snap["level"]), np.bincount(snap["level"])
    print(f"l4 final mesh: {n_leaf} leaves, per level {levels.tolist()}; uniform 40 cells/D equivalent "
          f"{int(16 / 0.005) * int(12 / 0.005)} cells")
    h = 0.01
    lev, u, v = raster(snap, WINDOW, h)
    # the leaves are piecewise constant: smooth the velocity over about one coarse-wake cell before differentiating
    u, v = gaussian_filter(u, 4.0), gaussian_filter(v, 4.0)
    w_z = np.gradient(v, h, axis=1) - np.gradient(u, h, axis=0)

    fig = plt.figure(figsize=(11.5, 9.6))
    gs = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.15), hspace=0.38)
    ext = WINDOW
    a0 = fig.add_subplot(gs[0, :])
    im = a0.imshow(w_z, origin="lower", extent=ext, cmap="RdBu_r", vmin=-4, vmax=4, aspect="equal")
    a0.set_title(f"Vorticity at $t = {t4[-1]:.0f}$ (four levels, 40 cells per diameter at the body)")
    fig.colorbar(im, cax=a0.inset_axes([1.012, 0, 0.014, 1]), label=r"$\omega$")
    a1 = fig.add_subplot(gs[1, :])
    im = a1.imshow(lev, origin="lower", extent=ext, cmap="viridis", aspect="equal")
    a1.set_title(f"AMR level ({n_leaf:,} leaves in the whole $16\\times12$ box)")
    fig.colorbar(im, cax=a1.inset_axes([1.012, 0, 0.014, 1]), label="level", ticks=range(int(lev.max()) + 1))
    for a in (a0, a1):
        a.set_ylabel("$y$")
    a1.set_xlabel("$x$")
    a2 = fig.add_subplot(gs[2, 0])
    a2.plot(t3[t3 >= 40], d3["Cl"][t3 >= 40], color="#2980b9", lw=1.2, label="3 levels (20 cells/D)")
    a2.plot(t4[t4 >= 40], d4["Cl"][t4 >= 40], color="#c0392b", lw=1.2, label="4 levels (40 cells/D)")
    a2.set_xlabel("$t$")
    a2.set_ylabel("$C_l$")
    a2.set_ylim(-0.45, 0.62)
    a2.legend(loc="upper center", ncol=2, fontsize=8)
    a3 = fig.add_subplot(gs[2, 1])
    a3.semilogy(res["l3"][0], res["l3"][1], color="#2980b9", lw=1.4, label="20 cells/D")
    a3.semilogy(st_f, a_f, color="#c0392b", lw=1.4, label="40 cells/D")
    a3.axvspan(*REF, color="#27ae60", alpha=0.35, lw=0, label="literature St")
    a3.axvline(st_spec, color="#c0392b", ls="--", lw=1.2, label=f"peak (40 cells/D) {st_spec:.3f}")
    a3.set_xlim(0, 0.6)
    a3.set_ylim(a_f.max() * 1e-4, a_f.max() * 3)
    a3.set_xlabel(r"$\mathrm{St}=fD/u_\infty$")
    a3.set_ylabel(r"amplitude of $C_l$ ($t\geq25$)")
    a3.legend(loc="upper right", fontsize=8)
    for a in (a2, a3):
        a.grid(True, ls="--", alpha=0.5)
    study.save(fig, "cylinder_low_blockage", "cylinder_low_blockage.png", args.docs)


if __name__ == "__main__":
    main()
