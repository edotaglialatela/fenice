"""Explanatory schematics of the immersed-boundary pages (docs/img/ibm_*.png).

    python docs/schematics.py            # every figure
    python docs/schematics.py fresh      # only the figures whose name contains "fresh"

Needs numpy and matplotlib. Nothing is simulated: the cells are classified with the rules of the
solver (a cell is solid when its centre is in the body; a ghost cell is a solid cell with a fluid
cell in the 5 x 5 box around it; the image point lies at max(1.5 h, d + h) along the wall normal),
so each picture shows the construction the pages describe.

Figures:
  ibm_ghost_geometry.png   ghost cell, wall point, image point and the velocity line through the wall
  ibm_loads.png            the momentum residual of the solid cells: inner faces cancel, the surface faces remain
  ibm_kinematics.png       the rigid-body wall velocity of a pitching airfoil and the rest / turn / hold angle
  ibm_coupling.png         the fluid and the body exchange pose, wall velocity and load through the surface
  ibm_step.png             one coupled step n -> n+1: explicit, no sub-iterations
  ibm_fresh_cells.png      the cells a moving body uncovers and how one of them is filled
  amr_coarse_fine.png      a coarse-fine face: one face per fine leaf, the virtual cell, the flux register
  amr_subcycling.png       the order in which three levels advance in one coarse step
  mpi_strips.png           y-strips with their two-row halo, and x-strips for a flat mesh
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Circle

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")
DPI = 170

FLUID = "#ffffff"
SOLID = "#c9ced6"
DEEP = "#9aa3ae"
GHOST = "#f0a35e"
FRESH = "#5fb3e6"
WALL = "#1f3a5f"
ACCENT = "#c0392b"
GREEN = "#27864a"
GRID = "#b8bec7"

plt.rcParams.update({"font.size": 10, "mathtext.fontset": "cm", "axes.linewidth": 0.8})


def save(fig, name):
    path = os.path.join(IMG, name)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def arrow(ax, p, q, color="k", lw=1.4, style="-|>", ms=10, z=5, **kw):
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(q), arrowstyle=style, mutation_scale=ms, color=color, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0, **kw))


def dim(ax, p, q, text, off, color="k", fs=11):
    """A dimension line from p to q, its label shifted by off from the middle."""
    arrow(ax, p, q, color=color, lw=0.9, style="<|-|>", ms=7)
    m = (np.asarray(p) + np.asarray(q)) / 2 + np.asarray(off)
    ax.text(m[0], m[1], text, color=color, ha="center", va="center", fontsize=fs, zorder=7)


def naca00(t, n=200):
    """A symmetric 4-digit section of unit chord, leading edge at x = 0 (closed trailing edge)."""
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
    return np.concatenate([x[::-1], x[1:]]), np.concatenate([yt[::-1], -yt[1:]])


def cells(ax, nx, ny, x0, y0, h, colour, lw=0.6):
    for j in range(ny):
        for i in range(nx):
            ax.add_patch(Rectangle((x0 + i * h, y0 + j * h), h, h, fc=colour(i, j), ec=GRID, lw=lw, zorder=1))


def classify(inside, nx, ny):
    """solid mask and ghost mask (solid with a fluid cell in the 5 x 5 box) of a cell array."""
    s = np.array([[inside(i, j) for i in range(nx)] for j in range(ny)])
    g = np.zeros_like(s)
    for j in range(ny):
        for i in range(nx):
            if s[j, i]:
                box = s[max(j - 2, 0):j + 3, max(i - 2, 0):i + 3]
                g[j, i] = not box.all()
    return s, g


# ------------------------------------------------------------------ the wall page

def ghost_geometry():
    h = 1.0
    nx, ny = 9, 9
    cx, cy, R = 2.0, -6.5, 10.6   # a large circle: the wall crosses the window as a gentle arc
    inside = lambda i, j: (i + 0.5 - cx) ** 2 + (j + 0.5 - cy) ** 2 < R * R
    s, g = classify(inside, nx, ny)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.12})
    cells(ax, nx, ny, 0, 0, h, lambda i, j: (GHOST if g[j, i] else DEEP) if s[j, i] else FLUID)
    th = np.linspace(np.pi / 4, 3 * np.pi / 4, 400)
    ax.plot(cx + R * np.cos(th), cy + R * np.sin(th), color=WALL, lw=2.4, zorder=3)
    ax.text(0.3, 0.35, r"$\Omega_b$", fontsize=14, zorder=4)
    ax.text(8.25, ny - 0.65, r"$\Omega_f$", fontsize=14, zorder=4)
    ax.text(0.15, cy + np.sqrt(R * R - (0.4 - cx) ** 2) + 0.25, r"$\Gamma$", color=WALL, fontsize=14, zorder=4)

    gi, gj = 5, 2                                 # the ghost cell shown
    assert s[gj, gi] and g[gj, gi]
    xg = np.array([gi + 0.5, gj + 0.5])
    v = xg - [cx, cy]
    n = v / np.linalg.norm(v)
    xw = np.array([cx, cy]) + R * n
    dg = np.linalg.norm(xw - xg)
    di = max(1.5 * h, dg + h)
    xi = xw + di * n
    # the four fluid cells of the bilinear interpolation around the image point
    i0, j0 = int(np.floor(xi[0] - 0.5)), int(np.floor(xi[1] - 0.5))
    for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        assert not s[j0 + b, i0 + a]
        c = np.array([i0 + a + 0.5, j0 + b + 0.5])
        ax.plot([xi[0], c[0]], [xi[1], c[1]], color=GREEN, lw=0.8, ls=":", zorder=4)
        ax.plot(*c, "s", color=GREEN, ms=5, zorder=5)
    ax.plot([xg[0], xi[0]], [xg[1], xi[1]], color="k", lw=0.8, ls="--", zorder=4)
    ax.plot(*xg, "o", color=ACCENT, ms=7, zorder=6)
    ax.plot(*xw, "o", color=WALL, ms=6, zorder=6)
    ax.plot(*xi, "o", color=GREEN, ms=7, mfc="white", mew=2, zorder=6)
    t = np.array([n[1], -n[0]])                   # the tangent pointing right
    dim(ax, xg + 0.55 * t, xw + 0.55 * t, r"$d_g$", 0.3 * t)
    dim(ax, xw + 0.55 * t, xi + 0.55 * t, r"$d_i$", 0.3 * t)
    arrow(ax, xw - 1.3 * t, xw - 1.3 * t + 0.8 * n, color=WALL, lw=1.6)
    ax.text(*(xw - 1.3 * t + 0.85 * n + [-0.3, 0.0]), r"$\mathbf{n}$", color=WALL, fontsize=12, zorder=6)
    ax.text(*(xg + [-0.6, -0.12]), r"$\mathbf{x}_g$", color=ACCENT, fontsize=12, zorder=6)
    ax.text(*(xw + [-0.62, -0.05]), r"$\mathbf{x}_w$", color=WALL, fontsize=12, zorder=6)
    ax.text(*(xi + [0.12, 0.12]), r"$\mathbf{x}_i$", color=GREEN, fontsize=12, zorder=6)
    ax.set_xlim(-0.05, nx + 0.05)
    ax.set_ylim(-0.05, ny + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("ghost cell, wall point and image point", fontsize=11)
    handles = [Rectangle((0, 0), 1, 1, fc=c, ec=GRID) for c in (FLUID, GHOST, DEEP)]
    handles += [plt.Line2D([], [], marker="s", ls="", color=GREEN, ms=5)]
    ax.legend(handles, ["fluid", r"ghost ($\mathcal{G}$)", "deep solid (reset)", "image stencil"],
              loc="upper left", bbox_to_anchor=(0.0, -0.01), ncol=4, frameon=False, fontsize=9, handlelength=1.2)

    # right: the velocity along the normal line
    ui, uw = 1.0, 0.25
    s_ = dg / di
    ug = uw - s_ * (ui - uw)
    bx.axvline(0, color=WALL, lw=2.4)
    bx.axhline(0, color="0.6", lw=0.6)
    bx.axvspan(-dg - 0.3, 0, color=SOLID, alpha=0.5, lw=0)
    bx.plot([-dg, di], [ug, ui], color="k", lw=1.3, ls="--")
    bx.plot([-dg], [ug], "o", color=ACCENT, ms=8)
    bx.plot([di], [ui], "o", color=GREEN, ms=8, mfc="white", mew=2)
    bx.plot([0], [uw], "o", color=WALL, ms=7)
    bx.text(-dg - 0.2, ug - 0.12, r"$\mathbf{u}_g=\mathbf{u}_w-\varsigma\,(\mathbf{u}_i-\mathbf{u}_w)$", fontsize=11, color=ACCENT, va="top")
    bx.text(di - 0.1, ui + 0.12, r"$\mathbf{u}_i$ (interpolated)", fontsize=11, color=GREEN, ha="right")
    bx.text(0.12, uw - 0.06, r"$\mathbf{u}_w$ on $\Gamma$", fontsize=11, color=WALL, va="top")
    bx.text(-dg / 2 - 0.15, 1.25, "solid", ha="center", color="0.35")
    bx.text(di / 2, 1.25, "fluid", ha="center", color="0.35")
    bx.text(0.98, 0.04, r"$\varsigma=d_g/d_i,\quad d_i\geq\max(1.5h,\ d_g+h)$" + "\n" + r"$\rho_g=\rho_i,\quad p_g=p_i$",
            transform=bx.transAxes, ha="right", va="bottom", fontsize=10.5)
    bx.set_xlim(-dg - 0.3, di + 0.3)
    bx.set_ylim(-1.0, 1.45)
    bx.set_xticks([-dg, 0, di], [r"$-d_g$", "0", r"$d_i$"])
    bx.set_yticks([])
    bx.set_xlabel("distance along the normal")
    bx.set_title(r"no-slip: the ghost-image line takes $\mathbf{u}_w$ on the wall", fontsize=11)
    for sp in ("top", "right", "left"):
        bx.spines[sp].set_visible(False)
    save(fig, "ibm_ghost_geometry.png")


def loads():
    h = 1.0
    nx, ny = 12, 8
    cx, cy, R = 6.0, 4.0, 2.75
    inside = lambda i, j: (i + 0.5 - cx) ** 2 + (j + 0.5 - cy) ** 2 < R * R
    s, _ = classify(inside, nx, ny)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    cells(ax, nx, ny, 0, 0, h, lambda i, j: SOLID if s[j, i] else FLUID)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(cx + R * np.cos(th), cy + R * np.sin(th), color=WALL, lw=1.6, ls="--", zorder=3)
    # faces between two solid cells: equal and opposite fluxes; faces between solid and fluid: the load
    for j in range(ny):
        for i in range(nx):
            for di_, dj_ in ((1, 0), (0, 1)):
                a, b = (j, i), (j + dj_, i + di_)
                if b[0] >= ny or b[1] >= nx or not (s[a] or s[b]):
                    continue
                fx, fy = (i + 1, j + 0.5) if di_ else (i + 0.5, j + 1)
                d = np.array([di_, dj_]) * 0.22
                if s[a] and s[b]:
                    arrow(ax, (fx - d[0] * 1.6, fy - d[1] * 1.6), (fx - d[0] * 0.25, fy - d[1] * 0.25), color="0.45", lw=0.8, ms=6)
                    arrow(ax, (fx + d[0] * 1.6, fy + d[1] * 1.6), (fx + d[0] * 0.25, fy + d[1] * 0.25), color="0.45", lw=0.8, ms=6)
                else:
                    inward = d if s[b] else -d   # from the fluid cell into the solid one
                    p0 = np.array([fx, fy]) - 2.0 * inward
                    arrow(ax, p0, p0 + 2.6 * inward, color=ACCENT, lw=1.6, ms=9)
    ax.set_xlim(-0.05, nx + 0.05)
    ax.set_ylim(-0.05, ny + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(nx + 0.4, ny - 0.6, r"$\mathbf{F}_f\simeq\sum_{c\,\in\,\mathrm{solid}} V_c\,\mathbf{R}^{\mathrm{mom}}_c$", fontsize=14, va="top")
    ax.text(nx + 0.4, ny - 2.0, "red: momentum flux through a face\nbetween fluid and solid, kept", color=ACCENT, va="top", fontsize=10)
    ax.text(nx + 0.4, ny - 3.6, "grey: a face between two solid cells,\nequal and opposite in the two sums,\ncancels", color="0.35", va="top", fontsize=10)
    ax.text(nx + 0.4, ny - 5.6, "dashed: the true surface $\\Gamma$", color=WALL, va="top", fontsize=10)
    save(fig, "ibm_loads.png")


# ------------------------------------------------------------------ the moving-bodies page

def kinematics():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.45, 1], "wspace": 0.35})
    xs, ys = naca00(0.12)
    xp = np.array([0.25, 0.0])
    ang = np.deg2rad(8.0)                          # nose up
    rot = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])   # clockwise = nose up
    pts = (rot @ (np.vstack([xs, ys]) - xp[:, None])) + xp[:, None]
    ax.add_patch(Polygon(pts.T, closed=True, fc=SOLID, ec=WALL, lw=1.8, zorder=2))
    ax.plot(*xp, "o", color="k", ms=6, zorder=6)
    ax.text(xp[0] - 0.03, xp[1] - 0.12, r"$\mathbf{x}_p$", fontsize=12, ha="center", zorder=6)
    Ub = np.array([0.0, 0.10])
    Om = -0.55
    arrow(ax, xp, xp + 2.2 * Ub, color=GREEN, lw=2.0)
    ax.text(*(xp + 2.2 * Ub + [0.02, 0.03]), r"$\mathbf{U}_b$", color=GREEN, fontsize=12)
    tt = np.linspace(np.deg2rad(200), np.deg2rad(330), 50)
    r0 = 0.17
    ax.plot(xp[0] + r0 * np.cos(tt), xp[1] + r0 * np.sin(tt), color=ACCENT, lw=1.4)
    arrow(ax, (xp[0] + r0 * np.cos(tt[-3]), xp[1] + r0 * np.sin(tt[-3])), (xp[0] + r0 * np.cos(tt[-1]), xp[1] + r0 * np.sin(tt[-1])),
          color=ACCENT, lw=1.4, ms=9)
    ax.text(xp[0] + 0.14, xp[1] - 0.24, r"$\Omega<0$" + "\n(nose up)", color=ACCENT, fontsize=10, ha="left")
    for k in np.r_[14:190:22, 215:395:22]:
        x = pts[:, k]
        uw = Ub + Om * np.array([-(x[1] - xp[1]), x[0] - xp[0]])
        arrow(ax, x, x + 0.45 * uw, color=WALL, lw=1.0, ms=7)
    ax.text(0.97, 0.21, r"$\mathbf{u}_w=\mathbf{U}_b+\Omega\,\hat{\mathbf{z}}\times(\mathbf{x}-\mathbf{x}_p)$", fontsize=12, ha="right", color=WALL)
    ax.plot([xp[0], 1.08], [0, 0], color="0.6", lw=0.7, ls=":")
    ta = np.linspace(-ang, 0, 30)
    ax.plot(xp[0] + 0.78 * np.cos(ta), xp[1] + 0.78 * np.sin(ta), color="0.4", lw=0.9)
    ax.text(xp[0] + 0.81, -0.075, r"$-\theta$", fontsize=11, color="0.3")
    ax.set_xlim(-0.12, 1.15)
    ax.set_ylim(-0.32, 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("wall velocity of a rigid body (pivot at the quarter chord)", fontsize=11)

    t0, t1, Om_ = 1.0, 3.0, np.deg2rad(-6.0) / 2.0
    t = np.linspace(0, 4.5, 400)
    theta = Om_ * (np.minimum(np.maximum(t, t0), t1) - t0)
    bx.plot(t, -np.rad2deg(theta), color=WALL, lw=2)
    bx.axvspan(t0, t1, color=ACCENT, alpha=0.08, lw=0)
    for x in (t0, t1):
        bx.axvline(x, color="0.5", lw=0.7, ls="--")
    bx.set_xticks([0, t0, t1], ["0", r"$t_0$", r"$t_1$"])
    bx.text(t0 / 2, 0.4, "rest", ha="center", color="0.3")
    bx.text((t0 + t1) / 2 - 0.35, 4.6, r"turn, $\dot\theta=\Omega$", ha="center", color=ACCENT)
    bx.text((t1 + 4.5) / 2, 5.4, "hold", ha="center", color="0.3")
    bx.set_ylim(-0.6, 7.0)
    bx.set_yticks([0, 6], ["0", r"$|\Omega|(t_1-t_0)$"])
    bx.set_xlabel("$t$   ($t_0$ = ibm_omega_from, $t_1$ = ibm_omega_until)", fontsize=9.5)
    bx.set_ylabel(r"nose-up angle $-\theta$")
    bx.set_title(r"$\theta=\Omega\,(\min(\max(t,t_0),t_1)-t_0)$", fontsize=11)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    save(fig, "ibm_kinematics.png")


def spring_path(p, q, n=7, w=0.09):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]]) / L
    s = np.linspace(0.18, 0.82, 2 * n + 1)
    zz = [p, p + 0.18 * d]
    for k, sk in enumerate(s[1:-1]):
        zz.append(p + sk * d + (w if k % 2 == 0 else -w) * nrm)
    zz += [p + 0.82 * d, q]
    return np.array(zz)


def coupling():
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    # fluid block
    ax.add_patch(FancyBboxPatch((0.2, 0.55), 4.6, 3.5, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#eef5fb", ec=WALL, lw=1.4))
    ax.text(2.5, 3.75, r"fluid on $\Omega_f(t)=\Omega\setminus\Omega_b(t)$", ha="center", fontsize=12, weight="bold", color=WALL)
    ax.text(2.5, 3.0, r"$\partial_t\mathbf{U}+\nabla\cdot\mathbf{F}=\nabla\cdot\mathbf{F}_v+\rho\,[0,\mathbf{g},\mathbf{u}\cdot\mathbf{g}]^\top$", ha="center", fontsize=11.5)
    ax.text(2.5, 2.25, "wall condition on $\\Gamma(t)$:", ha="center", fontsize=10.5)
    ax.text(2.5, 1.75, r"no-slip $\mathbf{u}=\mathbf{U}_b$,  slip $(\mathbf{u}-\mathbf{U}_b)\cdot\mathbf{n}=0$", ha="center", fontsize=11)
    ax.text(2.5, 1.2, r"$\partial_n\rho=\partial_n p=0$", ha="center", fontsize=11)
    ax.text(2.5, 0.75, "Runge-Kutta step, ghost-cell wall", ha="center", fontsize=9.5, color="0.4")
    # body block
    ax.add_patch(FancyBboxPatch((6.8, 0.55), 4.6, 3.5, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#fbf3ec", ec=ACCENT, lw=1.4))
    ax.text(9.1, 3.75, "rigid body (translation)", ha="center", fontsize=12, weight="bold", color=ACCENT)
    ax.text(9.1, 3.15, r"$m\,\dot{\mathbf{U}}_b=\mathbf{F}_f-\mathsf{K}(\mathbf{x}_b-\mathbf{x}_b(0))-\mathsf{C}\,\mathbf{U}_b+m\,\mathbf{g}$", ha="center", fontsize=11)
    # a mass on a spring and a damper, with its weight and the fluid load
    mx, my = 9.1, 1.85
    ax.add_patch(Circle((mx, my), 0.36, fc=SOLID, ec=WALL, lw=1.4, zorder=3))
    ax.text(mx, my, "$m$", ha="center", va="center", fontsize=12, zorder=4)
    wall_x = 7.3
    ax.plot([wall_x, wall_x], [1.25, 2.55], color="k", lw=1.6)
    for yy in np.linspace(1.3, 2.5, 7):
        ax.plot([wall_x - 0.12, wall_x], [yy - 0.1, yy], color="k", lw=0.7)
    sp = spring_path((wall_x, my + 0.18), (mx - 0.36, my + 0.18), n=6, w=0.08)
    ax.plot(sp[:, 0], sp[:, 1], color="k", lw=1.1)
    ax.text((wall_x + mx) / 2 - 0.15, my + 0.42, r"$\mathsf{K}$", fontsize=11, ha="center")
    yd = my - 0.2
    ax.plot([wall_x, 7.85], [yd, yd], color="k", lw=1.1)
    ax.add_patch(Rectangle((7.85, yd - 0.1), 0.45, 0.2, fc="none", ec="k", lw=1.1))
    ax.plot([8.05, 8.05 + 0.42], [yd, yd], color="k", lw=1.1)
    ax.plot([8.05, 8.05], [yd - 0.08, yd + 0.08], color="k", lw=1.6)
    ax.plot([8.47, mx - 0.34], [yd, yd], color="k", lw=1.1)
    ax.text(8.1, yd - 0.33, r"$\mathsf{C}$", fontsize=11, ha="center")
    arrow(ax, (mx, my - 0.36), (mx, my - 1.0), color="k", lw=1.4)
    ax.text(mx + 0.1, my - 0.95, r"$m\mathbf{g}$", fontsize=11)
    arrow(ax, (mx + 0.36, my), (mx + 1.05, my + 0.45), color=ACCENT, lw=1.8)
    ax.text(mx + 1.08, my + 0.5, r"$\mathbf{F}_f$", color=ACCENT, fontsize=12)
    ax.text(10.6, 0.72, "buoyancy is part\nof $\\mathbf{F}_f$ (pressure),\nnot added", ha="center", fontsize=8.8, color="0.4")
    # exchanges through the surface
    arrow(ax, (6.75, 2.95), (4.85, 2.95), color=ACCENT, lw=2.2, ms=14)
    ax.text(5.8, 3.12, "pose $\\mathbf{x}_b$ (shape of $\\Omega_f$)", ha="center", fontsize=9.5, color=ACCENT)
    ax.text(5.8, 2.62, "velocity $\\mathbf{U}_b$ (wall)", ha="center", fontsize=9.5, color=ACCENT)
    arrow(ax, (4.85, 1.6), (6.75, 1.6), color=WALL, lw=2.2, ms=14)
    ax.text(5.8, 1.77, "load $\\mathbf{F}_f$", ha="center", fontsize=9.5, color=WALL)
    ax.text(5.8, 1.45, "pressure and viscous\nstress on $\\Gamma$", ha="center", va="top", fontsize=9, color=WALL)
    ax.text(5.8, 4.35, "the two systems meet only at the surface $\\Gamma(t)$", ha="center", fontsize=11, color="0.25")
    save(fig, "ibm_coupling.png")


def box(ax, x, y, w, hgt, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.2, zorder=2))
    ax.text(x + w / 2, y + hgt - 0.22, title, ha="center", va="top", fontsize=10.5, weight="bold", color=ec, zorder=3)
    ax.text(x + w / 2, y + hgt - 0.62, body, ha="center", va="top", fontsize=9.5, zorder=3, linespacing=1.45)


def step():
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.set_xlim(-0.3, 15.3)
    ax.set_ylim(-0.75, 4.6)
    ax.axis("off")
    W, H, y = 2.75, 2.45, 1.45
    xs = [0.0, 3.05, 6.1, 9.15, 12.2]
    items = [
        ("1  pose", "geometry rebuilt only if\nthe body moved $> h/4$;\nelse only $\\mathbf{u}_w$ refreshed\n$\\Delta t\\,|U_b|/h\\leq 1$", "#f4f6f9", WALL),
        ("2  fresh cells", "cells uncovered by the\nnew pose are filled\nfrom their image point", "#eaf5fc", "#2a7ab0"),
        ("3  fluid step", "Runge-Kutta stages\nwith the wall at\n$\\mathbf{x}_b^n$, $\\mathbf{U}_b^n$ (frozen)", "#eef5fb", WALL),
        ("4  loads", "$\\mathbf{F}_f^n=\\sum_{\\rm solid}V_c\\mathbf{R}_c^{\\rm mom}$\nfrom stage 1, summed in\none global cell order", "#fbf3ec", ACCENT),
        ("5  body update", "$\\mathbf{a}^n$ from $\\mathbf{F}_f^n$, spring,\ndamper, weight; AB2 with\n$\\mathbf{a}^{n-1}$ for $\\mathbf{U}_b^{n+1}$,\ntrapezoid for $\\mathbf{x}_b^{n+1}$", "#fbf3ec", ACCENT),
    ]
    for x, (t, b, fc, ec) in zip(xs, items):
        box(ax, x, y, W, H, t, b, fc, ec)
    for a, b_ in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + W + 0.02, y + H / 2), (b_ - 0.02, y + H / 2), color="0.3", lw=1.4, ms=11)
    # stages inside the fluid step and the read-out of stage 1
    sx = xs[2]
    for k, lab in enumerate(["stage 1", "stage 2", "(stage 3)"]):
        ax.add_patch(Rectangle((sx + 0.12 + k * 0.85, y + 0.12), 0.81, 0.38, fc="white", ec=WALL, lw=0.9, zorder=3))
        ax.text(sx + 0.525 + k * 0.85, y + 0.31, lab, ha="center", va="center", fontsize=7.2, zorder=4)
    arrow(ax, (sx + 0.52, y + 0.1), (xs[3] + 0.6, y + 0.05), color=ACCENT, lw=1.1, ms=9,
          connectionstyle="arc3,rad=0.35", ls="--")
    ax.text((sx + xs[3]) / 2 + 0.95, y - 0.8, "residual of stage 1 (AMR: one extra residual at $t^n$)", ha="center", fontsize=8.8, color=ACCENT)
    # time axis
    ax.annotate("", (15.1, 0.1), (-0.1, 0.1), arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1))
    ax.text(-0.1, -0.2, "$t^n$:  $\\mathbf{U}^n$, $\\mathbf{x}_b^n$, $\\mathbf{U}_b^n$, $\\mathbf{a}^{n-1}$", fontsize=10, va="top")
    ax.text(15.1, -0.2, "$t^{n+1}$:  $\\mathbf{U}^{n+1}$, $\\mathbf{x}_b^{n+1}$, $\\mathbf{U}_b^{n+1}$, $\\mathbf{a}^{n}$", fontsize=10, va="top", ha="right")
    ax.text(7.5, 4.35, "explicit coupling: one fluid solve per step, no sub-iterations; every rank advances the same body from the same bits",
            ha="center", fontsize=10.5, color="0.25")
    save(fig, "ibm_step.png")


def fresh_cells():
    h = 1.0
    nx, ny = 14, 10
    R = 3.3
    c_old = np.array([6.6, 5.0])
    c_new = c_old + [0.9, 0.0]                     # the body moves right (exaggerated: the geometry is rebuilt after h/4)
    ins = lambda c: (lambda i, j: (i + 0.5 - c[0]) ** 2 + (j + 0.5 - c[1]) ** 2 < R * R)
    s_old, _ = classify(ins(c_old), nx, ny)
    s_new, g_new = classify(ins(c_new), nx, ny)
    fresh = s_old & ~s_new
    covered = ~s_old & s_new

    def colour(i, j):
        if fresh[j, i]:
            return FRESH
        if s_new[j, i]:
            return GHOST if g_new[j, i] else DEEP
        return FLUID

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    cells(ax, nx, ny, 0, 0, h, colour)
    for j in range(ny):
        for i in range(nx):
            if covered[j, i]:
                ax.add_patch(Rectangle((i, j), h, h, fc="none", ec="0.45", lw=0, hatch="////", zorder=1.5))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(c_old[0] + R * np.cos(th), c_old[1] + R * np.sin(th), color="0.35", lw=1.4, ls="--", zorder=3)
    ax.plot(c_new[0] + R * np.cos(th), c_new[1] + R * np.sin(th), color=WALL, lw=2.2, zorder=3)
    arrow(ax, c_old, c_new + [0.55, 0], color=GREEN, lw=2.0)
    ax.text(*(c_old + [-0.15, 0.25]), r"$\mathbf{U}_b$", color=GREEN, fontsize=12)

    # one fresh cell and how it is filled
    fi, fj = 3, 4
    assert fresh[fj, fi]
    xc = np.array([fi + 0.5, fj + 0.5])
    v = xc - c_new
    n = v / np.linalg.norm(v)
    xw = c_new + R * n
    dc = np.linalg.norm(xc - xw)
    di = max(1.5 * h, dc + h)
    xi = xw + di * n
    ax.plot([xw[0], xi[0]], [xw[1], xi[1]], color="k", lw=0.9, ls="--", zorder=4)
    ax.plot(*xw, "o", color=WALL, ms=6, zorder=6)
    ax.plot(*xc, "o", color="#155a8a", ms=7, zorder=6)
    ax.plot(*xi, "o", color=GREEN, ms=7, mfc="white", mew=2, zorder=6)
    ax.text(*(xw + [0.12, -0.42]), r"$\mathbf{x}_w$", color=WALL, fontsize=12, zorder=6)
    ax.text(*(xc + [-0.1, -0.5]), r"$\mathbf{x}_c$", color="#155a8a", fontsize=12, zorder=6)
    ax.text(*(xi + [-0.45, 0.25]), r"$\mathbf{x}_i$", color=GREEN, fontsize=12, zorder=6)
    ax.set_xlim(-0.05, nx + 0.05)
    ax.set_ylim(-0.05, ny + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")

    tx = nx + 0.5
    ax.text(tx, ny - 0.3, "dashed: the pose the geometry was built at\nsolid: the new pose", va="top", fontsize=10)
    rows = [(FRESH, None, "fresh: solid before, fluid now; no valid state"),
            (GHOST, "////", "covered: fluid before, solid now"),
            (GHOST, None, "ghost cells of the new pose"),
            (DEEP, None, "deep solid cells (reset)")]
    for k, (c, hatch, lab) in enumerate(rows):
        yy = ny - 1.9 - 0.75 * k
        ax.add_patch(Rectangle((tx, yy - 0.25), 0.5, 0.5, fc=c, ec=GRID, hatch=hatch, clip_on=False))
        ax.text(tx + 0.7, yy, lab, va="center", fontsize=10)
    ax.text(tx, 2.95, "a fresh cell takes the state of its own image point:", fontsize=10, va="top")
    ax.text(tx, 2.35, r"$\rho_c=\rho_i,\quad p_c=p_i$", fontsize=12, va="top")
    ax.text(tx, 1.7, r"$\mathbf{u}_c=\mathbf{u}_w+\dfrac{d_c}{d_i}\,(\mathbf{u}_i-\mathbf{u}_w)$", fontsize=12, va="top")
    ax.text(tx, 0.65, r"$d_c=|\mathbf{x}_c-\mathbf{x}_w|,\ \ d_i\geq\max(1.5h,\ d_c+h)$" + "\n" + "the image reads settled fluid only", fontsize=10, va="top")
    save(fig, "ibm_fresh_cells.png")


# ------------------------------------------------------------------ the AMR and MPI pages

def amr_coarse_fine():
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    # a coarse leaf (level l) next to four fine leaves (level l+1); two of them touch its right side
    ax.add_patch(Rectangle((0, 0), 2, 2, fc="#eef5fb", ec=WALL, lw=1.6))
    ax.text(1, 1, "coarse leaf\nlevel $\\ell$", ha="center", va="center", fontsize=11, color=WALL)
    for (x, y) in ((2, 0), (2, 1), (3, 0), (3, 1)):
        ax.add_patch(Rectangle((x, y), 1, 1, fc="#fbf3ec", ec=ACCENT, lw=1.2))
    ax.text(2.7, 0.5, "$f_1$", ha="center", va="center", color=ACCENT, fontsize=11)
    ax.text(2.7, 1.5, "$f_2$", ha="center", va="center", color=ACCENT, fontsize=11)
    ax.text(3.5, 1.0, "level $\\ell+1$", ha="center", va="center", color=ACCENT, fontsize=10, rotation=90)
    for yy in (0.5, 1.5):    # one face per fine leaf: its flux is computed once, by the owner of the fine leaf
        arrow(ax, (1.55, yy), (2.45, yy), color=GREEN, lw=1.8, ms=11)
    ax.plot([2, 2], [0, 2], color="k", lw=2.2, zorder=4)
    ax.text(1.75, 0.62, "$F_1$", color=GREEN, fontsize=12, ha="center")
    ax.text(1.75, 1.62, "$F_2$", color=GREEN, fontsize=12, ha="center")
    ax.text(2.0, -0.3, "coarse-fine face = two faces, one per fine leaf", ha="center", fontsize=10)
    ax.text(-0.3, -0.95, r"fine side: $F_k$ accumulated over the sub-steps with the stage weights (the register)", fontsize=10.5)
    ax.text(-0.3, -1.4, r"coarse side: replaces its own flux by $\sum_k A_k F_k$, so both sides exchange the same amount", fontsize=10.5)
    ax.set_xlim(-0.4, 6.6)
    ax.set_ylim(-1.6, 2.4)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "amr_coarse_fine.png")


def amr_subcycling():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    cols = ["#4c78a8", "#e08e45", "#59a14f"]
    order = []                                  # (level, t0, length) in the order advance() performs them

    def advance(L, t0, dt, top=2):
        if L < top:
            advance(L + 1, t0, dt / 2, top)
            advance(L + 1, t0 + dt / 2, dt / 2, top)
        order.append((L, t0, dt))

    advance(0, 0.0, 1.0)
    for k, (L, t0, dt) in enumerate(order, 1):
        y = (2 - L) * 1.2
        ax.add_patch(Rectangle((t0 + 0.004, y), dt - 0.008, 0.9, fc=cols[L], ec="white"))
        ax.text(t0 + dt / 2, y + 0.45, str(k), ha="center", va="center", color="white", fontsize=12, weight="bold")
    for L in range(3):
        ax.text(-0.02, (2 - L) * 1.2 + 0.45, f"level {L}\n" + (r"$\Delta t_0$" if L == 0 else rf"$\Delta t_0/{2**L}$"), ha="right", va="center", fontsize=10)
    ax.annotate("", (1.0, -0.3), (0, -0.3), arrowprops=dict(arrowstyle="-|>", color="0.4"))
    ax.text(0.5, -0.65, "one step of the coarsest level; the numbers give the order in which the levels advance (finer first)", ha="center", fontsize=10)
    ax.text(0.5, 3.75, "the coarse leaves keep their value from the start of the step while the finer levels run;\nthe fluxes of the coarse-fine faces are replaced at the end", ha="center", fontsize=9.5, color="0.3")
    ax.set_xlim(-0.25, 1.02)
    ax.set_ylim(-0.9, 4.3)
    ax.axis("off")
    save(fig, "amr_subcycling.png")


def mpi_strips():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"wspace": 0.1})
    N, nx, ny = 4, 8, 12
    cols = ["#dbe7f3", "#f6e3d3", "#dcefdc", "#eee0f1"]
    per = ny // N
    for j in range(ny):
        for i in range(nx):
            ax.add_patch(Rectangle((i, j), 1, 1, fc=cols[j // per], ec=GRID, lw=0.5))
    for r in range(N):
        ax.text(nx + 0.25, r * per + per / 2, f"rank {r}", va="center", fontsize=10)
        if r:
            ax.plot([0, nx], [r * per, r * per], color="k", lw=1.6, zorder=4)
    r = 1                                       # the two-cell halo of rank 1: the outer rows of its neighbours
    for y0 in (r * per - 2, (r + 1) * per):
        ax.add_patch(Rectangle((0, y0), nx, 2, fc="none", ec=ACCENT, lw=1.6, hatch="///", zorder=3))
    arrow(ax, (nx / 2, r * per + 0.5), (nx / 2, r * per - 1.4), color=ACCENT, lw=1.6)
    arrow(ax, (nx / 2, (r + 1) * per - 0.5), (nx / 2, (r + 1) * per + 1.4), color=ACCENT, lw=1.6)
    ax.set_xlim(-0.1, nx + 1.8)
    ax.set_ylim(-0.1, ny + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("y-strips: rank 1 sends its two outer rows (arrows)\nand receives those of its neighbours (hatched)", fontsize=10.5)

    nx2, ny2 = 16, 2                            # a mesh too flat for y-strips: x-strips
    per2 = nx2 // 4
    for j in range(ny2):
        for i in range(nx2):
            bx.add_patch(Rectangle((i, j), 1, 1, fc=cols[i // per2], ec=GRID, lw=0.5))
    for r in range(1, 4):
        bx.plot([r * per2, r * per2], [0, ny2], color="k", lw=1.6)
    for r in range(4):
        bx.text(r * per2 + per2 / 2, -0.6, f"rank {r}", ha="center", fontsize=10)
    bx.text(nx2 / 2, 4.2, "x-strips when $n_y<2N$ (here $n_y=2$, $N=4$):\neach rank keeps at least two columns", ha="center", va="center", fontsize=10.5)
    bx.set_xlim(-0.2, nx2 + 0.2)
    bx.set_ylim(-1.2, 6)
    bx.set_aspect("equal")
    bx.axis("off")
    save(fig, "mpi_strips.png")


FIGURES = {"ibm_ghost_geometry": ghost_geometry, "ibm_loads": loads, "ibm_kinematics": kinematics,
           "ibm_coupling": coupling, "ibm_step": step, "ibm_fresh_cells": fresh_cells,
           "amr_coarse_fine": amr_coarse_fine, "amr_subcycling": amr_subcycling, "mpi_strips": mpi_strips}

if __name__ == "__main__":
    pick = sys.argv[1:]
    for name, f in FIGURES.items():
        if not pick or any(p in name for p in pick):
            f()
