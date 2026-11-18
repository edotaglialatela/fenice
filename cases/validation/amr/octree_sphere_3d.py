"""3-D render of the sphere on the octree at 80 cells per diameter (docs/img/ibm_sphere_octree_3d.png).

    python cases/validation/amr/octree_sphere_3d.py              # results/validation/amr_octree_sphere/
    python cases/validation/amr/octree_sphere_3d.py --docs       # also docs/img/ibm_sphere_octree_3d.png
    python cases/validation/amr/octree_sphere_3d.py --volume     # also ibm_sphere_octree_vorticity.gif / .mp4: volume rendering of |omega|, the threshold sweeps down
    python cases/validation/amr/octree_sphere_3d.py --orbit 90   # the camera of the animation turns by 90 degrees over the animation (default 0: it does not move)
    python cases/validation/amr/octree_sphere_3d.py --runs DIR   # the runs of octree_q8.py (default: results/validation/amr_octree_q8/)

Reads the final state of the run `re20_oct_ghost_b10M3_t60` (`octree_q8.py long`, Re 20: the leaves of the octree, t = 60) and paints it on the
grid of the finest cells. The still shows the sphere, the speed on the plane z = Lz / 2 and on two cross-sections of the wake (x = 3.6 and 5.6), and streamlines that start upstream of the body.
The state is the final one, so the flow does not change in the animation (a transient needs the snapshots of a run): the camera is fixed and what moves is a threshold.
The lowest |omega| shown falls from the boundary layer to the edge of the wake.
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


def save_animation(images, name, size, docs, duration=70, hold=0):
    """The frames as <name>.gif (one palette for all) and <name>.mp4 (when ffmpeg is there)."""
    from PIL import Image
    out = study.study_dir(STUDY) / f"{name}.gif"
    sample = Image.new("RGB", (size[0], 8 * size[1]))
    for i, im in enumerate(images[:: max(1, len(images) // 8)][:8]):
        sample.paste(im, (0, i * size[1]))
    palette = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    gif = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=[duration] * (len(gif) - 1) + [duration + hold], loop=0)
    print(f"[GIF] {out}  ({len(gif)} frames, {out.stat().st_size / 1e6:.1f} MB)")
    outs = [out]
    try:
        import imageio.v2 as iio
        mp4 = out.with_suffix(".mp4")
        with iio.get_writer(mp4, fps=1000.0 / duration, codec="libx264", quality=8, macro_block_size=2) as w:
            for im in images:
                w.append_data(np.asarray(im))
        print(f"[MP4] {mp4}  ({mp4.stat().st_size / 1e6:.1f} MB)")
        outs.append(mp4)
    except Exception as e:   # no ffmpeg: the gif is enough
        print(f"[warn] no mp4: {e}")
    if docs:
        study.copy_to_docs(out)   # the documentation uses the gif; the mp4 stays in results/


def orbit(p, k, frames, degrees=0.0, radius=9.6, elevation=24.0):
    """Camera of frame k: from the side and upstream, z up; it turns by `degrees` over the whole animation (0: it does not move)."""
    a = np.radians(degrees) * k / max(frames - 1, 1) - 0.6 * np.pi
    el = np.radians(elevation)
    focus = np.array([3.0, CY, CZ])
    p.camera.position = tuple(focus + radius * np.array([np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)]))
    p.camera.focal_point = tuple(focus)
    p.camera.up = (0.0, 0.0, 1.0)


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


def volume_animation(vel, size, frames, docs, degrees=0.0, step=2):
    """Volume rendering of |omega| with a fixed camera: the opacity follows the gradient of |omega|, so the boundary layer and the edge of the wake show and
    the sphere is seen through them; the lowest value shown falls from the boundary layer to the wake over the frames. Writes ibm_sphere_octree_vorticity.gif and .mp4."""
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
    ramp = np.interp(np.linspace(0, 1, 64), np.linspace(0, 1, 8), [0, 0, 0.03, 0.1, 0.25, 0.45, 0.7, 0.9])   # opacity along the fixed colour scale
    images = []
    for k, thr in enumerate(np.geomspace(0.8 * top, lo, frames)):   # the lowest |omega| shown
        cut = int(np.searchsorted(np.linspace(np.log10(lo), np.log10(top), 64), np.log10(thr)))
        op = ramp.copy()
        op[:cut] = 0.0
        p = pv.Plotter(off_screen=True, window_size=tuple(size))
        p.set_background("#10141c")
        actor = p.add_volume(grid, scalars="log_omega", cmap="inferno", clim=(np.log10(lo), np.log10(top)), show_scalar_bar=False,
                             opacity=op, opacity_unit_distance=0.35)
        gof = vtk.vtkPiecewiseFunction()
        for x, y in ((0.0, 0.0), (0.03 * gm, 0.0), (0.3 * gm, 0.55), (gm, 1.0)):
            gof.AddPoint(x, y)
        actor.prop.SetGradientOpacity(gof)
        actor.prop.SetDisableGradientOpacity(0)
        p.add_mesh(sphere, color="#c9d1da", smooth_shading=True, specular=0.4, opacity=1.0)
        p.add_mesh(box, color="#6b7685", line_width=1.5)
        orbit(p, k, frames, degrees)
        p.add_text("vorticity above %.2f U/D (log scale), Re 20" % thr, position=(0.04, 0.03),
                   viewport=True, font_size=11, color="#c9d1da")
        images.append(Image.fromarray(p.screenshot(return_img=True)))
        p.close()
    save_animation(images, "ibm_sphere_octree_vorticity", size, docs, duration=80, hold=1500)


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--runs", default=None, help="the folder of the runs of octree_q8.py")
    ap.add_argument("--size", type=int, nargs=2, default=(2000, 1100), metavar=("W", "H"), help="image size in pixels")
    ap.add_argument("--volume", action="store_true", help="also the volume rendering of the vorticity (the threshold sweeps)")
    ap.add_argument("--frames", type=int, default=60, help="frames of the animation")
    ap.add_argument("--orbit", type=float, default=0.0, help="degrees the camera of the animation turns over the whole animation (default 0: fixed)")
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
    if args.volume:
        volume_animation(vel, args.anim_size, args.frames, args.docs, args.orbit)


if __name__ == "__main__":
    sys.exit(main())
