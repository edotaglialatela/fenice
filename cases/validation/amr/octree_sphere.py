"""The 3-D sphere on the octree at 20, 40 and 80 cells per diameter: the figure of the sphere page (docs/img/ibm_sphere_octree.png).

    python cases/validation/amr/octree_sphere.py                 # results/validation/amr_octree_sphere/
    python cases/validation/amr/octree_sphere.py --docs          # also docs/img/ibm_sphere_octree.png
    python cases/validation/amr/octree_sphere.py --runs DIR      # the runs of octree_q8.py (default: results/validation/amr_octree_q8/)

Reads the cached runs of `octree_q8.py run --include-40` and `octree_q8.py long` (Re 20, sphere of diameter 1 in a 6 x 3 x 3 channel, inflow speed 1):
the drag histories, the drag at t = 30 against the finest cell size, and the speed on the mid-plane z = Lz / 2 of the finest run, from its final
state (the leaves are painted on the finest grid), with the refinement levels of the zoom around the body. No solver run.
"""
import json
import sys
from pathlib import Path

import numpy as np

from fenice import study

STUDY = "amr_octree_sphere"
RUNS = (("re20_oct_ghost_b5M2", 20), ("re20_oct_ghost_b10M2", 40), ("re20_oct_ghost_b10M3_t60", 80))
LX, LY, LZ, D, CX, CY = 6.0, 3.0, 3.0, 1.0, 2.0, 1.5
COL = ("#c0392b", "#2980b9", "#27ae60")


def history(run):
    p = next(Path(run).rglob("*_monitor.csv"))
    m = np.genfromtxt(p, delimiter=",", names=True)
    _, keep = np.unique(m["t"], return_index=True)
    keep = np.sort(keep)
    return np.asarray(m["t"])[keep], np.asarray(m["Cd"])[keep]


def midplane(run, n):
    """The speed and the level of the leaves on z = Lz / 2, painted on the grid of n cells per diameter."""
    f = np.load(Path(run) / "final.npz")
    U, meta = f["U"], f["leaf_meta"]   # leaf_meta: x0, x1, y0, y1, z0, z1, level
    zc = 0.5 * LZ + 1e-9
    sel = (meta[:, 4] <= zc) & (zc < meta[:, 5])
    U, meta = U[sel], meta[sel]
    speed = np.linalg.norm(U[:, [1, 2, 4]], axis=1) / U[:, 0]
    nxg, nyg = int(round(LX * n)), int(round(LY * n))
    img = np.full((nyg, nxg), np.nan)
    lev = np.zeros((nyg, nxg))
    i0 = np.rint(meta[:, 0] * n).astype(int)
    i1 = np.rint(meta[:, 1] * n).astype(int)
    j0 = np.rint(meta[:, 2] * n).astype(int)
    j1 = np.rint(meta[:, 3] * n).astype(int)
    for a, b, c, d, s, l in zip(i0, i1, j0, j1, speed, meta[:, 6]):
        img[c:d, a:b] = s
        lev[c:d, a:b] = l
    return img, lev


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--runs", default=None, help="the folder of the runs of octree_q8.py")
    args = ap.parse_args()
    root = Path(args.runs) if args.runs else study.study_dir("amr_octree_q8")
    plt = study.use_style()
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.75, 1])
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[0, 1])
    a3 = fig.add_subplot(gs[1, 0])
    a4 = fig.add_subplot(gs[1, 1])
    cd30, info = [], {}
    for (tag, n), col in zip(RUNS, COL):
        run = root / tag
        t, cd = history(run)
        a1.plot(t, cd, color=col, label=f"{n} cells / D")
        cd30.append(float(np.interp(30.0, t, cd)))
        info[n] = json.loads((run / "info.json").read_text())
    a1.set_ylim(3.2, 4.2)
    a1.set_xlim(0, 60)
    a1.axvline(30.0, color="0.6", lw=0.8, ls=":")
    a1.set_xlabel("$t$")
    a1.set_ylabel("$C_d$")
    a1.set_title("Drag history (octree, Re 20)")
    a1.legend(loc="upper right")
    h = 1.0 / np.array([n for _, n in RUNS])
    a2.plot(h, cd30, "o-", color="0.25", ms=7)
    for x, y, (_, n) in zip(h, cd30, RUNS):
        a2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=9)
    a2.set_xlabel("finest cell size / $D$")
    a2.set_ylabel("$C_d$ at $t = 30$")
    a2.set_xlim(0, 0.06)
    a2.set_ylim(min(cd30) - 0.004, max(cd30) + 0.006)
    a2.set_title("Drag against resolution")
    run80 = root / RUNS[-1][0]
    n = RUNS[-1][1]
    img, lev = midplane(run80, n)
    vmax = float(np.nanmax(img))
    ext = (0, LX, 0, LY)
    m = a3.imshow(img, origin="lower", extent=ext, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal")
    th = np.linspace(0, 2 * np.pi, 200)
    a3.plot(CX + 0.5 * D * np.cos(th), CY + 0.5 * D * np.sin(th), "w-", lw=1)
    a3.set_xlabel("$x$")
    a3.set_ylabel("$y$")
    a3.set_title(f"$|\\mathbf{{u}}| / U$ on $z = L_z/2$, {n} cells / D, $t = {info[n]['t']:.0f}$")
    a3.grid(False)
    fig.colorbar(m, ax=a3, fraction=0.025, pad=0.02)
    x0, x1, y0, y1 = 1.0, 3.5, 0.5, 2.5
    sl = (slice(int(y0 * n), int(y1 * n)), slice(int(x0 * n), int(x1 * n)))
    a4.imshow(img[sl], origin="lower", extent=(x0, x1, y0, y1), cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal")
    xs = (np.arange(sl[1].start, sl[1].stop) + 0.5) / n
    ys = (np.arange(sl[0].start, sl[0].stop) + 0.5) / n
    lz = lev[sl].copy()
    xx, yy = np.meshgrid(xs, ys)
    lz[np.hypot(xx - CX, yy - CY) < 0.5 * D + 0.1] = lev.max()   # the levels of the fluid only: the body is at the finest level
    a4.contour(xs, ys, lz, levels=np.arange(0.5, lev.max()), colors="w", linewidths=0.5, alpha=0.7)
    a4.plot(CX + 0.5 * D * np.cos(th), CY + 0.5 * D * np.sin(th), "w-", lw=1)
    a4.set_xlabel("$x$")
    a4.set_title("Zoom, white lines: levels of the leaves")
    a4.grid(False)
    plt.tight_layout()
    study.save(fig, STUDY, "ibm_sphere_octree.png", args.docs)
    print("Cd at t = 30:", {n: round(c, 4) for (_, n), c in zip(RUNS, cd30)})


if __name__ == "__main__":
    sys.exit(main())
