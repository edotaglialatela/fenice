"""3-D render of the sphere on the octree at 80 cells per diameter (docs/img/ibm_sphere_octree_3d.png).

    python cases/validation/amr/octree_sphere_3d.py              # results/validation/amr_octree_sphere/
    python cases/validation/amr/octree_sphere_3d.py --docs       # also docs/img/ibm_sphere_octree_3d.png
    python cases/validation/amr/octree_sphere_3d.py --runs DIR   # the runs of octree_q8.py (default: results/validation/amr_octree_q8/)

Reads the final state of the run `re20_oct_ghost_b10M3_t60` (`octree_q8.py long`, Re 20: the leaves of the octree, t = 60) and paints it on the
grid of the finest cells. Shows the sphere, the speed on the plane z = Lz / 2 and on two cross-sections of the wake (x = 3.6 and 5.6), and streamlines that start upstream of the body.
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


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--runs", default=None, help="the folder of the runs of octree_q8.py")
    ap.add_argument("--size", type=int, nargs=2, default=(2000, 1100), metavar=("W", "H"), help="image size in pixels")
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


if __name__ == "__main__":
    sys.exit(main())
