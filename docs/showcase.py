"""Showcase animations for the README and the case pages (docs/img/*.gif).

    python docs/showcase.py run        # simulations into results/showcase/ (about 1.5 h on 16 cores)
    python docs/showcase.py render     # GIFs from those results into docs/img/

Needs Python >= 3.11 (tomllib), numpy, scipy, matplotlib and Pillow.

Runs (fenice under MPI, one rank per core, binary VTK):

* IBM Flow over Cylinder (Re = 100): a 400 x 200 precursor to t = 10 (a copy of the case
  with a small transverse-velocity kick that starts the shedding) is hot-started on the
  unmodified case at 1200 x 600 (60 cells per diameter): t = 0..3 densely sampled for
  the GIFs, which loop exactly one shedding period (found from a wake probe), then
  resumed to t = 8.

The force histories of the cylinder page come from the validation studies
(validation/ibm_loads.py, validation/cylinder_mesh_transition.py).
* IBM Supersonic Flow over Cylinder (Mach 2): 1600 x 1600 from t = 0 to 5.

Every frame is the simulated field only (no axes, legend or background), box-averaged
from the mesh to the GIF width; the solid cylinder is drawn in a flat colour. Each GIF
uses a fixed 255-step colormap palette, so frames neither dither nor flicker.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import fenice
from fenice import vtk_array

OUT = os.path.join(ROOT, "results", "showcase")
IMG = os.path.join(ROOT, "docs", "img")

CYL = ("IBM Flow over Cylinder", "ibm_flow_over_cylinder")
SUP = ("IBM Supersonic Flow over Cylinder", "ibm_supersonic_flow_over_cylinder")


# ---------------------------------------------------------------------------
# Simulations
# ---------------------------------------------------------------------------

def run_case(case, out_root, *sets):
    """Run one case: MPI with one rank per core when mpiexec is available (about 20 %
    faster than OpenMP threads on these meshes), else OpenMP threads."""
    cmd = [fenice.exe(), "--case", case, "--no-logo", "--set", f"output_root={out_root}"]
    ranks = os.cpu_count() or 1
    if ranks > 1 and shutil.which("mpiexec"):
        cmd = ["mpiexec", "-n", str(ranks)] + cmd + ["--set", "use_mpi=true", "--set", f"n_cpus={ranks}",
                                                     "--mpi-child"]
    else:
        cmd += ["--set", "use_mpi=false"]
    for s in sets:
        cmd += ["--set", s]
    print("[RUN]", " ".join(cmd[1:]), flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def precursor_case():
    """Copy of the cylinder case with a transverse-velocity kick behind the cylinder: a
    symmetric start takes very long to shed, the kick decays into periodic shedding."""
    folder = os.path.join(OUT, "precursor_case")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(ROOT, "cases", CYL[1], "case.toml"), encoding="utf-8") as f:
        text = f.read()
    text = text.replace('name = "IBM Flow over Cylinder"', 'name = "Cylinder Precursor"', 1)
    text += ('\n[[initial.region]]   # symmetry-breaking kick, decays once shedding starts\n'
             'shape = "box"\nmin = [1.1, 1.0]\nmax = [1.5, 1.3]\nstate = { v = 0.3 }\n')
    with open(os.path.join(folder, "case.toml"), "w", encoding="utf-8") as f:
        f.write(text)
    return folder


def last_time(name, key):
    """Last snapshot time of a run in OUT, or -1 if it has not finished."""
    pvd = os.path.join(OUT, name, key, f"{key}.pvd")
    return snapshots(os.path.dirname(pvd))[-1][0] if os.path.exists(pvd) else -1.0


def run_all():
    # Coarse precursor, densely sampled
    pre = os.path.join(OUT, "precursor")
    ck = os.path.join(pre, "cylinder_precursor", "cylinder_precursor_restart.npz")
    if last_time("precursor", "cylinder_precursor") < 10:
        run_case(precursor_case(), pre, "t_end=10", "save_interval=64")
    # Fine cylinder: t = 0..3 densely sampled (the GIFs), then to t = 8 for the force history
    cyl = os.path.join(OUT, "cylinder")
    if last_time("cylinder", CYL[1]) < 3:
        run_case(CYL[0], cyl, "nx=1200", "ny=600", "t_end=3", "use_hot_start=true", f"hot_start_file={ck}",
               "save_interval=96")
    if last_time("cylinder", CYL[1]) < 8:
        run_case(CYL[0], cyl, "nx=1200", "ny=600", "t_end=8", "resume=true", "save_interval=192")
    if last_time("supersonic", SUP[1]) < 5:
        run_case(SUP[0], os.path.join(OUT, "supersonic"), "nx=1600", "ny=1600", "save_interval=250")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def snapshots(case_dir):
    pvd = ET.parse(glob.glob(os.path.join(case_dir, "*.pvd"))[0]).getroot()
    return [(float(d.get("timestep")), os.path.join(case_dir, d.get("file"))) for d in pvd.iter("DataSet")]


def read_2d(path):
    """Cell fields of a 2-D .vtr as (ny, nx) arrays, plus the cell size."""
    root = ET.parse(path).getroot()
    nx, ny = (int(v) for v in root.find("RectilinearGrid").get("WholeExtent").split()[1:4:2])
    arr = lambda name: vtk_array(root.find(f'.//DataArray[@Name="{name}"]'))
    x = arr("x")
    vel = arr("Velocity").reshape(ny, nx, 3)
    return {"rho": arr("Density").reshape(ny, nx), "u": vel[..., 0], "v": vel[..., 1], "h": x[1] - x[0]}


def solid_mask(case_folder, shape):
    """The IBM cylinder of the case (same test as the solver: |centre - c| <= r)."""
    with open(os.path.join(ROOT, "cases", case_folder, "case.toml"), "rb") as f:
        cf = tomllib.load(f)
    ny, nx = shape
    lx, ly, ibm = cf["mesh"]["Lx"], cf["mesh"]["Ly"], cf["ibm"]
    xc = (np.arange(nx) + 0.5) * lx / nx
    yc = (np.arange(ny) + 0.5) * ly / ny
    X, Y = np.meshgrid(xc, yc)
    return np.sqrt((X - ibm["ibm_cx"]) ** 2 + (Y - ibm["ibm_cy"]) ** 2) <= ibm["ibm_radius"]


def vorticity(f):
    return np.gradient(f["v"], f["h"], axis=1) - np.gradient(f["u"], f["h"], axis=0)


def speed(f):
    return np.hypot(f["u"], f["v"])


def schlieren(f, gref, k=12.0):
    """Numerical schlieren exp(-k |grad rho| / gref) (Quirk 1994): 1 in smooth flow, 0 at shocks."""
    gy, gx = np.gradient(f["rho"], f["h"])
    return np.exp(-k * np.hypot(gx, gy) / gref)


WIDTH = {"cylinder": 1000, "supersonic": 700}   # GIF widths [px]; box-averaged from the mesh


def frame(field, solid, vmin, vmax, width, fill=0.0):
    """Palette indices: 0..254 for the colormap, 255 for the solid.  The field (with
    `fill` inside the solid) and the solid mask are box-averaged to `width` pixels."""
    ny, nx = field.shape
    size = (width, round(width * ny / nx))
    f = np.where(solid, fill, field).astype(np.float32)
    f = np.asarray(Image.fromarray(f, "F").resize(size, Image.BOX))
    s = np.asarray(Image.fromarray(solid.astype(np.float32), "F").resize(size, Image.BOX)) >= 0.5
    idx = np.clip(np.rint((f - vmin) / (vmax - vmin) * 254), 0, 254).astype(np.uint8)
    idx[s] = 255
    return Image.fromarray(np.flipud(idx), "P")   # row 0 is y = 0: put it at the bottom


def write_gif(path, frames, cmap, solid_color, ms, hold_last_ms=None):
    pal = (np.array(plt.get_cmap(cmap)(np.linspace(0, 1, 255)))[:, :3] * 255).round().astype(np.uint8)
    pal = np.vstack([pal, np.array(to_rgb(solid_color)) * 255]).astype(np.uint8).ravel().tolist()
    for im in frames:
        im.putpalette(pal)
    durations = [ms] * len(frames)
    if hold_last_ms:
        durations[-1] = hold_last_ms
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
    print(f"[GIF] {os.path.relpath(path, ROOT)}  {frames[0].size[0]}x{frames[0].size[1]}, "
          f"{len(frames)} frames, {os.path.getsize(path) / 1e6:.1f} MB")


def one_period(snaps, probe):
    """Snapshots spanning the last full shedding period (upward zero crossings of the
    transverse velocity at `probe` = (j, i)), from t >= 1 after the hot start."""
    t = np.array([s[0] for s in snaps])
    vp = np.array([read_2d(p)["v"][probe] for _, p in snaps])
    vp -= vp[t >= 1.0].mean()
    up = [k for k in range(1, len(t)) if t[k] >= 1.0 and vp[k - 1] < 0 <= vp[k]]
    if len(up) < 2:
        raise RuntimeError("no full shedding period after t = 1: run longer")
    a, b = up[-2], up[-1]
    print(f"[CYL] shedding period {t[b] - t[a]:.3f} (t = {t[a]:.3f} .. {t[b]:.3f}, {b - a} snapshots)")
    return snaps[a:b]


def render_cylinder(n_frames):
    snaps = [s for s in snapshots(os.path.join(OUT, "cylinder", CYL[1])) if s[0] <= 3.0 + 1e-9]   # dense part
    first = read_2d(snaps[0][1])
    ny, nx = first["rho"].shape
    solid = solid_mask(CYL[1], (ny, nx))
    period = one_period(snaps, (ny // 2, int(nx * 0.6)))
    pick = [period[int(k)] for k in np.linspace(0, len(period), n_frames, endpoint=False)]
    fields = [read_2d(p) for _, p in pick]
    width = WIDTH["cylinder"]
    w = [vorticity(f) for f in fields]
    wmax = 0.5 * np.percentile(np.abs(w[0])[~solid], 99.9)   # saturate the boundary layer
    write_gif(os.path.join(IMG, "vonKarman_vortices.gif"),
              [frame(x, solid, -wmax, wmax, width) for x in w], "RdBu_r", "#3a3a3a", 50)
    s = [speed(f) for f in fields]
    smax = max(np.percentile(x[~solid], 99.9) for x in s)
    write_gif(os.path.join(IMG, "vonKarman_velocity.gif"),
              [frame(x, solid, 0, smax, width) for x in s], "magma", "#9aa0a6", 50)


def render_supersonic(n_frames):
    snaps = snapshots(os.path.join(OUT, "supersonic", SUP[1]))
    pick = [snaps[int(k)] for k in np.rint(np.linspace(0, len(snaps) - 1, n_frames))]
    fields = [read_2d(p) for _, p in pick]
    solid = solid_mask(SUP[1], fields[0]["rho"].shape)
    gy, gx = np.gradient(fields[-1]["rho"], fields[-1]["h"])
    gref = np.percentile(np.hypot(gx, gy)[~solid], 99.9)
    width = WIDTH["supersonic"]
    write_gif(os.path.join(IMG, "supersonic_cylinder_schlieren.gif"),
              [frame(schlieren(f, gref), solid, 0, 1, width, 1.0) for f in fields], "gray", "#56677a", 80, 1500)
    s = [speed(f) for f in fields]
    smax = max(np.percentile(x[~solid], 99.9) for x in s)
    write_gif(os.path.join(IMG, "supersonic_cylinder_velocity.gif"),
              [frame(x, solid, 0, smax, width) for x in s], "magma", "#9aa0a6", 80, 1500)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["run", "render"])
    ap.add_argument("--frames", type=int, default=48, help="frames per GIF (default 48)")
    args = ap.parse_args()
    if args.stage == "run":
        run_all()
    else:
        render_cylinder(args.frames)
        render_supersonic(args.frames)


if __name__ == "__main__":
    main()
