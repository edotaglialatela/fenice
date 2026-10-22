"""3-D Taylor-Green vortex (Re = 1600, Ma = 0.1): transition to turbulence on a coarse grid.

    python cases/validation/benchmarks/taylor_green_3d.py            # 32^3 and 64^3 (the 64^3 run takes minutes)
    python cases/validation/benchmarks/taylor_green_3d.py --skip-64  # 32^3 only
    python cases/validation/benchmarks/taylor_green_3d.py --grids 32 64 96 --vorticity-n 96 --mpi 16 --q-animation --docs

Runs the "Taylor-Green Vortex 3D" case to t = 20 into results/validation/taylor_green_3d/n<N>/
(existing runs are reused; --force re-runs them):

1. every grid of --grids (default 32^3, the case default, and 64^3) with the volume-average
   monitor (``monitor_interval``); 32^3 and the --vorticity-n grid also write snapshots;
2. kinetic energy E_k(t) and its dissipation rate eps = -dE_k/dt (from the monitor), plus the
   resolved physical dissipation 2 nu zeta (enstrophy, from the snapshots), against the
   DNS peak eps ~ 0.0127 at t ~ 9 (van Rees et al. 2011; Wang et al. 2013);
3. vorticity magnitude |omega| on the box faces and on the periodic plane x = 0 at four times
   (from the --vorticity-n grid).

The grids are deliberately coarse: the case is a 3-D bug check (vortex stretching, the
dissipation peak, the energy cascade), not a converged DNS.

Figures: taylor_green_3d_validation.png and taylor_green_3d_vorticity.png; with --q-animation
(needs PyVista: python prerequisites.py --viz3d) also taylor_green_3d_q.gif,
the iso-surface Q = 3 of the Q-criterion coloured by omega_z, from t = 3.5 (Q first exceeds 3) to 20.
"""
import os

import numpy as np

import fenice
from fenice import study

STUDY, CASE = "taylor_green_3d", "Taylor-Green Vortex 3D"
NU = 1.0 / 1600.0
L = 2.0 * np.pi
DNS_PEAK = (9.0, 0.0127)      # (t, eps_max), spectral DNS 512^3 at Re = 1600
SNAP_TIMES = (0.0, 4.0, 8.0, 12.0)


def read_monitor(case_dir):
    d = fenice.read_monitor(case_dir)
    t, ke = d["t"], d["kinetic_energy"]
    return t, ke, -np.gradient(ke, t)


def read_vtr(path):
    f = fenice.read_vtr(path)   # fields indexed [k, j, i] = [z, y, x]
    return f["Density"], f["Velocity"]


def vorticity(vel):
    """Periodic 2nd-order central differences; arrays are indexed [k, j, i] = [z, y, x]."""
    n = vel.shape[0]
    h = L / n
    d = lambda f, ax: (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2 * h)
    u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
    return np.stack([d(w, 1) - d(v, 0), d(u, 0) - d(w, 2), d(v, 2) - d(u, 1)], axis=-1)


def snapshots(case_dir):
    return fenice.read_pvd(case_dir)


def velocity_gradient(vel):
    """g[i][j] = du_i/dx_j by periodic central differences; arrays are indexed [k, j, i] = [z, y, x]."""
    h = L / vel.shape[0]
    d = lambda f, ax: (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2 * h)
    return [[d(vel[..., i], 2 - j) for j in range(3)] for i in range(3)]


def q_animation(case_dir, n, docs, t_start=3.5, frames=60, size=520, q_level=3.0, wz_range=8.0):
    """Iso-surface Q = q_level of the Q-criterion Q = (|Omega|^2 - |S|^2)/2, coloured by omega_z, one frame
    per snapshot from t_start while the camera turns 60 degrees; writes taylor_green_3d_q.gif."""
    import pyvista as pv   # optional: python prerequisites.py --viz3d
    from PIL import Image
    snaps = [s for s in snapshots(case_dir) if s[0] >= t_start - 1e-9]
    pick = [snaps[int(k)] for k in np.rint(np.linspace(0, len(snaps) - 1, min(frames, len(snaps))))]
    h = L / n
    images = []
    for k, (ts, path) in enumerate(pick):
        g = velocity_gradient(read_vtr(path)[1])
        grid = pv.ImageData(dimensions=(n, n, n), spacing=(h, h, h), origin=(h / 2, h / 2, h / 2))
        grid.point_data["Q"] = (-0.5 * sum(g[i][j] * g[j][i] for i in range(3) for j in range(3))).ravel()
        grid.point_data["wz"] = (g[1][0] - g[0][1]).ravel()
        p = pv.Plotter(off_screen=True, window_size=(size, size))
        p.set_background("white")
        iso = grid.contour([q_level], scalars="Q")
        if iso.n_points:
            p.add_mesh(iso, scalars="wz", cmap="coolwarm", clim=(-wz_range, wz_range), smooth_shading=True,
                       specular=0.3, show_scalar_bar=False)
        p.add_mesh(pv.Box(bounds=(0, L, 0, L, 0, L)).outline(), color="#9aa0a6", line_width=1)
        p.camera_position = "iso"
        p.camera.Azimuth(60.0 * k / max(len(pick) - 1, 1))   # vtkCamera rotation about the view-up axis
        p.add_text(f"t = {ts:.1f}", position="lower_left", font_size=10, color="#51606f")
        images.append(Image.fromarray(p.screenshot(return_img=True)))
        p.close()
    # One palette for every frame (no flicker), from a strip of sample frames
    sample = Image.new("RGB", (size, 4 * size))
    for i, im in enumerate(images[:: max(1, len(images) // 4)][:4]):
        sample.paste(im, (0, i * size))
    palette = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gif = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    out = study.study_dir(STUDY) / "taylor_green_3d_q.gif"
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=[80] * (len(gif) - 1) + [1500], loop=0)
    print(f"[GIF] {out}  ({len(gif)} frames, {os.path.getsize(out) / 1e6:.1f} MB)")
    if docs:
        study.copy_to_docs(out)


def pick_snapshot(shots, ts, om):
    """Keep |omega| of the snapshot closest to each of SNAP_TIMES."""
    for target in SNAP_TIMES:
        if target not in shots or abs(ts - target) < abs(shots[target][0] - target):
            shots[target] = (ts, np.linalg.norm(om, axis=-1))


def cube_faces(ax, field, vmax, title):
    """Paint the three visible faces of the periodic box (x = 2pi, y = 0, z = 2pi) with |omega|."""
    n = field.shape[0]
    g = np.linspace(0, L, n)
    A, B = np.meshgrid(g, g, indexing="ij")
    import matplotlib.pyplot as plt
    cmap, norm = plt.cm.magma, plt.Normalize(0, vmax)
    face = lambda f: cmap(norm(f))
    # field[k, j, i]
    ax.plot_surface(np.full_like(A, L), A, B, facecolors=face(field[:, :, -1].T), shade=False, rstride=1, cstride=1)
    ax.plot_surface(A, np.zeros_like(A), B, facecolors=face(field[:, 0, :].T), shade=False, rstride=1, cstride=1)
    ax.plot_surface(A, B, np.full_like(A, L), facecolors=face(field[-1, :, :].T), shade=False, rstride=1, cstride=1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-58)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("$x$", labelpad=-12); ax.set_ylabel("$y$", labelpad=-12); ax.set_zlabel("$z$", labelpad=-12)
    ax.set_title(title, pad=0)


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--grids", type=int, nargs="+", default=[32, 64], metavar="N",
                    help="grids of the energy / dissipation curves (default: 32 64)")
    ap.add_argument("--skip-64", action="store_true", help="same as --grids 32")
    ap.add_argument("--vorticity-n", type=int, default=32, metavar="N",
                    help="grid of the vorticity figure (default 32); its run keeps snapshots and joins the curves")
    ap.add_argument("--q-animation", action="store_true",
                    help="also animate the Q-criterion iso-surface of the --vorticity-n run (needs PyVista)")
    args = ap.parse_args()
    plt = study.use_style()
    nv = args.vorticity_n
    grids = sorted(set([32] if args.skip_64 else args.grids) | {32, nv})

    # Every grid to t = 20 with the monitor; 32^3 and the vorticity grid also keep snapshots
    runs, case_dirs = {}, {}
    for n in grids:
        save = 100 if n == 32 else (50 if n == nv else 1000000000)
        d = study.run_cached(STUDY, f"n{n}", CASE, f"nx={n}", f"ny={n}", f"nz={n}",
                             f"monitor_interval={5 if n == 32 else 10}", f"save_interval={save}",
                             mpi=args.mpi, force=args.force, done="*/*_monitor.csv")
        case_dirs[n] = study.case_output(d)
        runs[n] = read_monitor(case_dirs[n])
        t, ke, eps = runs[n]
        k = int(np.argmax(eps))
        print(f"      {n}^3: E_k(20)/E_k(0) = {ke[-1] / ke[0]:.3f}, peak eps = {eps[k]:.4f} at t = {t[k]:.2f}")

    # Resolved (physical) dissipation 2 nu zeta from the snapshots (32^3 and the vorticity grid)
    resolved, shots = {}, {}
    for n in sorted({32, nv}):
        t_s, eps_phys = [], []
        for ts, path in snapshots(case_dirs[n]):
            rho, vel = read_vtr(path)
            om = vorticity(vel)
            t_s.append(ts)
            eps_phys.append(NU * np.mean(rho * np.sum(om ** 2, axis=-1)))
            if n == nv:
                pick_snapshot(shots, ts, om)
        resolved[n] = (np.array(t_s), np.array(eps_phys))

    colors = {32: "#2980b9", 64: "#e74c3c", 96: "#27ae60", 128: "#8e44ad"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"3-D Taylor-Green vortex ($Re = 1600$, $Ma = 0.1$, periodic $(2\pi)^3$)", fontsize=15,
                 fontweight="bold")
    for n, (t, ke, eps) in runs.items():
        a1.plot(t, ke / ke[0], color=colors.get(n, "#7f8c8d"), lw=2.2, label=f"{n}$^3$")
        a2.plot(t, eps, color=colors.get(n, "#7f8c8d"), lw=2.2, label=f"$-dE_k/dt$, {n}$^3$")
    for n, (t_s, eps_phys) in resolved.items():
        step = max(1, len(t_s) // 40)   # about 40 markers per curve
        a2.plot(t_s[::step], eps_phys[::step], "o", color=colors.get(n, "#7f8c8d"), ms=5, mec="white", alpha=0.85,
                label=rf"$2\nu\zeta$ (resolved), {n}$^3$")
    a2.plot(*DNS_PEAK, "*", color="#2c3e50", ms=16, mec="white", label="DNS peak (van Rees et al. 2011)")
    a1.set_xlabel("Time, $t$"); a1.set_ylabel(r"Kinetic energy, $E_k(t)/E_k(0)$")
    a1.set_title("Kinetic-energy decay")
    a2.set_xlabel("Time, $t$"); a2.set_ylabel(r"Dissipation rate, $\varepsilon$")
    a2.set_title(r"Dissipation: total $-dE_k/dt$ vs resolved $2\nu\zeta$")
    for a in (a1, a2):
        a.set_xlim(0, 20)
        a.grid(True, ls="--", alpha=0.6)
    a1.legend(frameon=True)
    a2.set_ylim(0, 0.0215)   # head-room for the legend above the DNS peak
    a2.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    study.save(fig, STUDY, "taylor_green_3d_validation.png", args.docs, dpi=200)

    vmax = max(np.percentile(f, 99.5) for _, f in shots.values())
    fig = plt.figure(figsize=(16, 7.6))
    fig.suptitle(rf"3-D Taylor-Green vortex, {nv}$^3$: vorticity magnitude $|\omega|$", fontsize=15,
                 fontweight="bold")
    for c, target in enumerate(SNAP_TIMES):
        ts, f = shots[target]
        cube_faces(fig.add_subplot(2, 4, c + 1, projection="3d"), f, vmax, f"$t = {ts:.1f}$")
        ax = fig.add_subplot(2, 4, c + 5)
        ax.imshow(f[:, :, 0], origin="lower", extent=(0, L, 0, L), cmap="magma", vmin=0, vmax=vmax,
                  interpolation="bilinear")
        ax.set_xlabel("$y$"); ax.set_ylabel("$z$")
        ax.set_title(f"plane $x = 0$, $t = {ts:.1f}$", fontsize=11)
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, vmax))
    fig.subplots_adjust(left=0.04, right=0.88, top=0.9, bottom=0.07, hspace=0.12, wspace=0.18)
    fig.colorbar(sm, cax=fig.add_axes([0.9, 0.2, 0.012, 0.6]), label=r"$|\omega|$")
    study.save(fig, STUDY, "taylor_green_3d_vorticity.png", args.docs)

    if args.q_animation:
        q_animation(case_dirs[nv], nv, args.docs)


if __name__ == "__main__":
    main()
