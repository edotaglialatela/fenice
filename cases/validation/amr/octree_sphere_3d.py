"""3-D render of the sphere on the octree at 80 cells per diameter (docs/img/ibm_sphere_octree_3d.png).

    python cases/validation/amr/octree_sphere_3d.py              # results/validation/amr_octree_sphere/
    python cases/validation/amr/octree_sphere_3d.py --docs       # also docs/img/ibm_sphere_octree_3d.png
    python cases/validation/amr/octree_sphere_3d.py --animation  # also ibm_sphere_octree_3d.gif / .mp4: the camera turns once round the body
    python cases/validation/amr/octree_sphere_3d.py --volume     # also ibm_sphere_octree_vorticity.gif / .mp4: volume rendering of |omega|
    python cases/validation/amr/octree_sphere_3d.py --runs DIR   # the runs of octree_q8.py (default: results/validation/amr_octree_q8/)

Reads the final state of the run `re20_oct_ghost_b10M3_t60` (`octree_q8.py long`, Re 20: the leaves of the octree, t = 60) and paints it on the
grid of the finest cells. Shows the sphere, the speed on the plane z = Lz / 2 and on two cross-sections of the wake (x = 3.6 and 5.6), and streamlines that start upstream of the body.
The animation shows the wake as nested translucent isosurfaces of the speed (0.3, 0.6 and 0.85 of the inflow speed) around the sphere while the camera makes one turn;
the state is the final one, so the flow does not change (a transient needs the snapshots of a run).
Needs pyvista (`python prerequisites.py --viz3d`); no solver run.
"""
import sys
from pathlib import Path

import numpy as np

from fenice import study

STUDY = "amr_octree_sphere"
RUN, N = "re20_oct_ghost_b10M3_t60", 80
LX, LY, LZ, D, CX, CY, CZ = 6.0, 3.0, 3.0, 1.0, 2.0, 1.5, 1.5


def paint(run):
    """The velocity of the final state on the grid of the finest cells: (nz, ny, nx, 3) float32 and the number of leaves."""
    f = np.load(Path(run) / "final.npz")
    U, meta = f["U"], f["leaf_meta"]   # leaf_meta: x0, x1, y0, y1, z0, z1, level
    nx, ny, nz = int(LX * N), int(LY * N), int(LZ * N)
    vel = np.zeros((nz, ny, nx, 3), dtype=np.float32)
    v = (U[:, [1, 2, 4]] / U[:, :1]).astype(np.float32)
    lo = np.rint(meta[:, [0, 2, 4]] * N).astype(int)
    size = np.rint((meta[:, 1] - meta[:, 0]) * N).astype(int)
    for s in np.unique(size):
        g = size == s
        i0, j0, k0, vv = lo[g, 0], lo[g, 1], lo[g, 2], v[g]
        for c in range(s):
            for b in range(s):
                for a in range(s):
                    vel[k0 + c, j0 + b, i0 + a] = vv
    return vel, len(U)


def save_animation(images, name, size, docs):
    """The frames as <name>.gif (one palette for all) and <name>.mp4 (when ffmpeg is there)."""
    from PIL import Image
    out = study.study_dir(STUDY) / f"{name}.gif"
    sample = Image.new("RGB", (size[0], 4 * size[1]))
    for i, im in enumerate(images[:: max(1, len(images) // 4)][:4]):
        sample.paste(im, (0, i * size[1]))
    palette = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gif = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=70, loop=0)
    print(f"[GIF] {out}  ({len(gif)} frames, {out.stat().st_size / 1e6:.1f} MB)")
    outs = [out]
    try:
        import imageio.v2 as iio
        mp4 = out.with_suffix(".mp4")
        with iio.get_writer(mp4, fps=15, codec="libx264", quality=8, macro_block_size=2) as w:
            for im in images:
                w.append_data(np.asarray(im))
        print(f"[MP4] {mp4}  ({mp4.stat().st_size / 1e6:.1f} MB)")
        outs.append(mp4)
    except Exception as e:   # no ffmpeg: the gif is enough
        print(f"[warn] no mp4: {e}")
    if docs:
        study.copy_to_docs(out)   # the documentation uses the gif; the mp4 stays in results/


def orbit(p, k, frames, radius=9.6, elevation=24.0):
    """Camera k of one turn round the body, z up; the first one looks at it from the side and upstream."""
    a = 2.0 * np.pi * k / frames - 0.6 * np.pi
    el = np.radians(elevation)
    focus = np.array([3.0, CY, CZ])
    p.camera.position = tuple(focus + radius * np.array([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)]))
    p.camera.focal_point = tuple(focus)
    p.camera.up = (0.0, 0.0, 1.0)


def animation(pts, size, frames, docs):
    """One turn of the camera round the body, nested isosurfaces of the speed; writes ibm_sphere_octree_3d.gif and .mp4."""
    import pyvista as pv
    from PIL import Image
    levels = ((0.3, "#2c3e91"), (0.6, "#1f9e89"), (0.85, "#b8de29"))
    # the far field is painted from coarse leaves (blocks of up to 8 finest cells): smooth the steps of the surfaces, which keeps their volume
    soft = pts.gaussian_smooth(std_dev=2.5, radius_factor=2.5, scalars="speed")   # a few finest cells: the steps of the coarse leaves
    isos = [(soft.contour([v], scalars="speed").smooth_taubin(n_iter=60, pass_band=0.05).compute_normals(), col, op)
            for (v, col), op in zip(levels, (0.35, 0.28, 0.18))]
    sphere = pv.Sphere(radius=0.5 * D, center=(CX, CY, CZ), theta_resolution=96, phi_resolution=96)
    box = pv.Box(bounds=(0, LX, 0, LY, 0, LZ)).outline()
    focus = np.array([3.0, CY, CZ])
    images = []
    for k in range(frames):
        a = 2.0 * np.pi * k / frames - 0.6 * np.pi   # starts looking at the body from the side and upstream
        p = pv.Plotter(off_screen=True, window_size=tuple(size))
        p.set_background("white")
        p.enable_depth_peeling(number_of_peels=8)
        for iso, col, op in isos:
            if iso.n_points:
                p.add_mesh(iso, color=col, opacity=op, smooth_shading=True, specular=0.2)
        p.add_mesh(sphere, color="#dfe3e8", smooth_shading=True, specular=0.4)
        p.add_mesh(box, color="#7f8c8d", line_width=1.5)
        r, el = 8.2, np.radians(24.0)
        p.camera.position = tuple(focus + r * np.array([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)]))
        p.camera.focal_point = tuple(focus)
        p.camera.up = (0.0, 0.0, 1.0)
        p.add_text("speed isosurfaces: 0.3, 0.6, 0.85 of the inflow speed (dark to light), Re 20", position=(0.04, 0.03), viewport=True,
                   font_size=11, color="black")
        images.append(Image.fromarray(p.screenshot(return_img=True)))
        p.close()
    save_animation(images, "ibm_sphere_octree_3d", size, docs)


def vorticity(vel, sigma=2.0, step=2):
    """|omega| (in U / D) of the painted velocity, smoothed over a few finest cells (the coarse leaves are blocks) and kept every `step` cells."""
    from scipy.ndimage import gaussian_filter
    h = 1.0 / N
    v = [gaussian_filter(vel[..., i], sigma) for i in range(3)]   # arrays (z, y, x)
    d = lambda f, ax: np.gradient(f, h, axis=ax)
    wx = d(v[2], 1) - d(v[1], 0)
    wy = d(v[0], 0) - d(v[2], 2)
    wz = d(v[1], 2) - d(v[0], 1)
    return np.sqrt(wx * wx + wy * wy + wz * wz)[::step, ::step, ::step].astype(np.float32)


def volume_animation(vel, size, frames, docs, step=2):
    """One turn of the camera round the body, volume rendering of |omega|: the opacity follows the gradient of |omega|, so the boundary layer and
    the edge of the wake show and the sphere is seen through them; writes ibm_sphere_octree_vorticity.gif and .mp4."""
    import pyvista as pv
    import vtk
    from PIL import Image
    w = vorticity(vel, step=step)
    h = step / N
    grid = pv.ImageData(dimensions=(w.shape[2], w.shape[1], w.shape[0]), spacing=(h, h, h), origin=(0.5 * h, 0.5 * h, 0.5 * h))
    # the vorticity of the wake (0.1 to 1) is small against the boundary layer (up to ~ 8): a logarithmic scale shows both, and the gradient of it
    # (large at the edge of the layer and of the wake) sets the opacity
    lo = 0.05
    lw = np.log10(np.maximum(w, lo)).astype(np.float32)
    grid.point_data["log_omega"] = lw.ravel()
    gm = 0.03 * float(np.percentile(np.linalg.norm(np.gradient(lw, h), axis=0), 99))
    top = float(np.percentile(w, 99.9))
    sphere = pv.Sphere(radius=0.5 * D, center=(CX, CY, CZ), theta_resolution=96, phi_resolution=96)
    box = pv.Box(bounds=(0, LX, 0, LY, 0, LZ)).outline()
    images = []
    for k in range(frames):
        p = pv.Plotter(off_screen=True, window_size=tuple(size))
        p.set_background("#10141c")
        actor = p.add_volume(grid, scalars="log_omega", cmap="inferno", clim=(np.log10(lo), np.log10(top)), show_scalar_bar=False,
                             opacity=[0, 0, 0.03, 0.1, 0.25, 0.45, 0.7, 0.9], opacity_unit_distance=0.35)
        gof = vtk.vtkPiecewiseFunction()
        for x, y in ((0.0, 0.0), (0.03 * gm, 0.0), (0.3 * gm, 0.55), (gm, 1.0)):
            gof.AddPoint(x, y)
        actor.prop.SetGradientOpacity(gof)
        actor.prop.SetDisableGradientOpacity(0)
        p.add_mesh(sphere, color="#c9d1da", smooth_shading=True, specular=0.4, opacity=1.0)
        p.add_mesh(box, color="#6b7685", line_width=1.5)
        orbit(p, k, frames)
        p.add_text("vorticity magnitude (log scale, 0.05 to %.0f U/D), Re 20" % top, position=(0.04, 0.03), viewport=True, font_size=11, color="#c9d1da")
        images.append(Image.fromarray(p.screenshot(return_img=True)))
        p.close()
    save_animation(images, "ibm_sphere_octree_vorticity", size, docs)


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--runs", default=None, help="the folder of the runs of octree_q8.py")
    ap.add_argument("--size", type=int, nargs=2, default=(2000, 1100), metavar=("W", "H"), help="image size in pixels")
    ap.add_argument("--animation", action="store_true", help="also the animation (one turn of the camera, isosurfaces of the speed)")
    ap.add_argument("--volume", action="store_true", help="also the volume rendering of the vorticity (one turn of the camera)")
    ap.add_argument("--frames", type=int, default=60, help="frames of the animation")
    ap.add_argument("--anim-size", type=int, nargs=2, default=(960, 540), metavar=("W", "H"), help="frame size of the animation")
    args = ap.parse_args()
    import pyvista as pv   # optional: python prerequisites.py --viz3d
    root = Path(args.runs) if args.runs else study.study_dir("amr_octree_q8")
    vel, leaves = paint(root / RUN)
    h = 1.0 / N
    grid = pv.ImageData(dimensions=(vel.shape[2] + 1, vel.shape[1] + 1, vel.shape[0] + 1), spacing=(h, h, h), origin=(0, 0, 0))
    grid.cell_data["velocity"] = vel.reshape(-1, 3)
    grid.cell_data["speed"] = np.linalg.norm(vel, axis=-1).reshape(-1)
    pts = grid.cell_data_to_point_data()
    vmax = float(grid.cell_data["speed"].max())
    p = pv.Plotter(off_screen=True, window_size=tuple(args.size))
    p.set_background("white")
    kw = dict(scalars="speed", cmap="viridis", clim=(0.0, 1.2), show_scalar_bar=False)
    bar = dict(title=" ", n_labels=7, vertical=False, position_x=0.34, position_y=0.07, width=0.32, height=0.045, label_font_size=20,
               color="black", fmt="%.1f")
    p.add_mesh(pts.slice(normal="z", origin=(CX, CY, CZ)), **dict(kw, show_scalar_bar=True, scalar_bar_args=bar))
    p.add_mesh(pts.slice(normal="x", origin=(3.6, CY, CZ)), **kw)
    p.add_mesh(pts.slice(normal="x", origin=(5.6, CY, CZ)), **kw)
    seeds = pv.Plane(center=(0.4, CY, CZ), direction=(1, 0, 0), i_size=1.9, j_size=1.9, i_resolution=4, j_resolution=4)
    lines = pts.streamlines_from_source(seeds, vectors="velocity", max_length=14.0, integration_direction="forward", initial_step_length=0.3)
    p.add_mesh(lines.tube(radius=0.01), color="#2c3e50", opacity=0.85)
    p.add_mesh(pv.Sphere(radius=0.5 * D, center=(CX, CY, CZ), theta_resolution=96, phi_resolution=96), color="#dfe3e8", smooth_shading=True,
               specular=0.4)
    p.add_mesh(pv.Box(bounds=(0, LX, 0, LY, 0, LZ)).outline(), color="#7f8c8d", line_width=1.5)
    p.add_text("speed / inflow speed, Re 20, 80 cells per diameter, t = 60 (cut at z = Lz/2)", position=(0.30, 0.025), viewport=True, font_size=13, color="black")
    p.camera_position = [(-2.2, -7.5, 5.0), (3.1, 1.5, 1.0), (0.0, 0.0, 1.0)]
    p.camera.zoom(1.0)
    img = p.screenshot(return_img=True)
    out = study.study_dir(STUDY) / "ibm_sphere_octree_3d.png"
    from PIL import Image
    Image.fromarray(img).save(out)
    print(f"[PLOT] {out}  ({leaves} leaves, speed up to {vmax:.2f})")
    if args.docs:
        study.copy_to_docs(out)
    if args.animation:
        animation(pts, args.anim_size, args.frames, args.docs)
    if args.volume:
        volume_animation(vel, args.anim_size, args.frames, args.docs)


if __name__ == "__main__":
    sys.exit(main())
