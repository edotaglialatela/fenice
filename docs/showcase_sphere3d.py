"""Render the 3-D sphere showcase (Re = 100, Mach 0.3, octree AMR, 20 cells per diameter) from its .vtu snapshots.

    fenice --case <sphere case> --set use_amr=true --set amr_max_level=2 ...   (see the command in docs/source/cases/sphere.md)
    python docs/showcase_sphere3d.py render [--frame N] [--dir results/showcase/sphere3d]

The sphere is a solid grey body; the flow is two translucent iso-surfaces of the vorticity magnitude (coloured by it) and a
translucent mid-plane of the streamwise velocity, seen from above and downstream at an angle. Needs pyvista and imageio.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
CX, CY, CZ, R = 2.0, 1.5, 1.5, 0.5
LX, LY, LZ = 6.0, 3.0, 3.0
H = 0.05            # sampling grid: the finest cell size
LEVELS = (1.5, 4.0)  # vorticity magnitude of the two iso-surfaces (U/D units)


def snapshots(d):
    files = sorted(Path(d).rglob("*_[0-9][0-9][0-9][0-9][0-9].vtu"), key=lambda p: int(re.search(r"_(\d{5})\.vtu$", p.name).group(1)))
    return files


def fields(vtu):
    m = pv.read(vtu)
    grid = pv.ImageData(dimensions=(int(LX / H) + 1, int(LY / H) + 1, int(LZ / H) + 1), spacing=(H, H, H), origin=(0, 0, 0))
    s = grid.sample(m)            # cell data of the leaves at the grid points
    vel = np.asarray(s.point_data["Velocity"])
    s.point_data["Velocity"] = vel
    g = s.compute_derivative(scalars="Velocity", vorticity=True, gradient=False)
    w = np.linalg.norm(np.asarray(g.point_data["vorticity"]), axis=1)
    p = np.asarray(s.points)
    r = np.linalg.norm(p - np.array([CX, CY, CZ]), axis=1)
    w *= np.clip((r - R) / 0.15, 0.0, 1.0) ** 2   # no boundary-layer shell on the body (it would hide it); also zero inside (ghost cells)
    s.point_data["omega"] = w
    s.point_data["u"] = vel[:, 0]
    return s


def render(vtu, out, t, size=(1280, 720)):
    s = fields(vtu)
    pl = pv.Plotter(off_screen=True, window_size=size)
    pl.set_background("white", top="#dfe6ee")
    # streamlines from a disc upstream of the body, tubes coloured by speed
    seeds = [(0.6, CY + rr * np.cos(t), CZ + rr * np.sin(t)) for rr, n in ((0.3, 6), (0.55, 10), (0.9, 14)) for t in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    sl = s.streamlines_from_source(pv.PolyData(np.array(seeds)), vectors="Velocity", max_length=12.0, integration_direction="forward")
    if sl.n_points:
        sl.point_data["speed"] = np.linalg.norm(np.asarray(sl.point_data["Velocity"]), axis=1)
        pl.add_mesh(sl.tube(radius=0.009), scalars="speed", cmap="Blues_r", clim=(0.0, 1.3), opacity=0.5, show_scalar_bar=False)
    for lev, op in zip(LEVELS, (0.16, 0.4)):
        iso = s.contour([lev], scalars="omega")
        if iso.n_points:
            pl.add_mesh(iso, scalars="omega", cmap="viridis", clim=(0.0, 6.0), opacity=op, smooth_shading=True, show_scalar_bar=False)
    pl.enable_depth_peeling(12)
    ball = pv.Sphere(radius=R, center=(CX, CY, CZ), theta_resolution=64, phi_resolution=64)
    pl.add_mesh(ball, color="#8a8f98", smooth_shading=True, specular=0.5)
    box = pv.Box(bounds=(0, LX, 0, LY, 0, LZ))
    pl.add_mesh(box, style="wireframe", color="#555555", line_width=1.0, opacity=0.5)
    pl.camera_position = [(0.2, -3.2, 5.8), (3.0, 1.5, 1.0), (0, 0, 1)]
    pl.camera.zoom(1.0)
    pl.add_text(f"sphere, Re = 100, octree AMR     t = {t:.2f}", font_size=10, color="#333333", position="upper_left")
    pl.screenshot(str(out))
    pl.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render"])
    ap.add_argument("--dir", default=str(ROOT / "results" / "showcase" / "sphere3d"))
    ap.add_argument("--frame", type=int, help="render this frame only (test)")
    a = ap.parse_args()
    files = snapshots(a.dir)
    if not files:
        sys.exit("no snapshots in " + a.dir)
    frames = Path(a.dir) / "frames"
    frames.mkdir(exist_ok=True)
    todo = [(a.frame, files[a.frame])] if a.frame is not None else list(enumerate(files))
    for i, f in todo:
        out = frames / (f.stem + ".png")
        render(f, out, i * 0.25)
        print(out.name, flush=True)
    if a.frame is None:
        import imageio.v2 as imageio
        imgs = [imageio.imread(frames / (f.stem + ".png")) for f in files]
        imageio.mimsave(Path(a.dir) / "sphere3d.mp4", imgs, fps=20, quality=8)
        imageio.mimsave(Path(a.dir) / "sphere3d.gif", imgs[::2], duration=0.1, loop=0)


if __name__ == "__main__":
    main()
