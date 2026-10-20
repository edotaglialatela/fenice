"""3-D Taylor-Green vortex (Re = 1600, Ma = 0.1): transition to turbulence on a coarse grid.

    python validation/validationTaylorGreen3D.py            # 32^3 and 64^3 (the 64^3 run takes minutes)
    python validation/validationTaylorGreen3D.py --skip-64  # 32^3 only

Runs the C++ solver (cpp/build/fvsolver) on the "Taylor-Green Vortex 3D" case to t = 20:

1. 32^3 (the case default) with snapshots every 100 steps and the volume-average monitor
   (``monitor_interval``), and 64^3 with the monitor only;
2. kinetic energy E_k(t) and its dissipation rate eps = -dE_k/dt (from the monitor), plus the
   resolved physical dissipation 2 nu zeta (enstrophy, from the 32^3 snapshots), against the
   DNS peak eps ~ 0.0127 at t ~ 9 (van Rees et al. 2011; Wang et al. 2013);
3. vorticity magnitude |omega| on the box faces and on the periodic plane x = 0 at four times.

The grids are deliberately coarse: the case is a 3-D bug check (vortex stretching, the
dissipation peak, the energy cascade), not a converged DNS.

Writes results/taylor_green_3d_validation.png and results/taylor_green_3d_vorticity.png.
"""
import argparse
import glob
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(ROOT, "cpp", "build", "fvsolver.exe" if os.name == "nt" else "fvsolver")
CASE, PREFIX = "Taylor-Green Vortex 3D", "taylor-green_vortex_3d"
NU = 1.0 / 1600.0
L = 2.0 * np.pi
DNS_PEAK = (9.0, 0.0127)      # (t, eps_max), spectral DNS 512^3 at Re = 1600
SNAP_TIMES = (0.0, 4.0, 8.0, 12.0)


def run(out_root, *sets):
    cmd = [EXE, "--case", CASE, "--no-logo", "--set", f"output_root={out_root}"]
    for s in sets:
        cmd += ["--set", s]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return os.path.join(out_root, PREFIX)


def read_monitor(case_dir):
    d = np.genfromtxt(os.path.join(case_dir, f"{PREFIX}_monitor.csv"), delimiter=",", names=True)
    t, ke = d["t"], d["kinetic_energy"]
    _, keep = np.unique(t, return_index=True)      # the final row can repeat the last t
    t, ke = t[keep], ke[keep]
    return t, ke, -np.gradient(ke, t)


def read_vtr(path):
    root = ET.parse(path).getroot()
    nx, _, ny, _, nz = (int(v) for v in root.find("RectilinearGrid").attrib["WholeExtent"].split()[1:])
    arr = lambda name: np.array(root.find(f'.//DataArray[@Name="{name}"]').text.split(), dtype=float)
    vel = arr("Velocity").reshape(nz, ny, nx, 3)     # cell index (k*ny + j)*nx + i
    return arr("Density").reshape(nz, ny, nx), vel


def vorticity(vel):
    """Periodic 2nd-order central differences; arrays are indexed [k, j, i] = [z, y, x]."""
    n = vel.shape[0]
    h = L / n
    d = lambda f, ax: (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2 * h)
    u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
    return np.stack([d(w, 1) - d(v, 0), d(u, 0) - d(w, 2), d(v, 2) - d(u, 1)], axis=-1)


def snapshots(case_dir):
    pvd = ET.parse(glob.glob(os.path.join(case_dir, "*.pvd"))[0]).getroot()
    return [(float(ds.attrib["timestep"]), os.path.join(case_dir, ds.attrib["file"])) for ds in pvd.iter("DataSet")]


def cube_faces(ax, field, vmax, title):
    """Paint the three visible faces of the periodic box (x = 2pi, y = 0, z = 2pi) with |omega|."""
    n = field.shape[0]
    g = np.linspace(0, L, n)
    A, B = np.meshgrid(g, g, indexing="ij")
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-64", action="store_true", help="only run the 32^3 grid")
    ap.add_argument("--reuse", metavar="DIR", help="reuse runs in DIR/n32 and DIR/n64 instead of running")
    args = ap.parse_args()

    tmp = args.reuse or tempfile.mkdtemp(prefix="tgv3d_")
    runs = {}
    for n in (32,) if args.skip_64 else (32, 64):
        case_dir = os.path.join(tmp, f"n{n}", PREFIX)
        if not args.reuse:
            print(f"[RUN] {n}^3 to t = 20 ...")
            sets = [f"nx={n}", f"ny={n}", f"nz={n}", f"monitor_interval={5 if n == 32 else 10}",
                    f"save_interval={100 if n == 32 else 1000000000}"]
            case_dir = run(os.path.join(tmp, f"n{n}"), *sets)
        runs[n] = read_monitor(case_dir)
        t, ke, eps = runs[n]
        k = int(np.argmax(eps))
        print(f"      {n}^3: E_k(20)/E_k(0) = {ke[-1] / ke[0]:.3f}, peak eps = {eps[k]:.4f} at t = {t[k]:.2f}")

    # Resolved (physical) dissipation 2 nu zeta from the 32^3 snapshots
    snaps = snapshots(os.path.join(tmp, "n32", PREFIX))
    t_s, eps_phys, shots = [], [], {}
    for ts, path in snaps:
        rho, vel = read_vtr(path)
        om = vorticity(vel)
        t_s.append(ts)
        eps_phys.append(NU * np.mean(rho * np.sum(om ** 2, axis=-1)))
        for target in SNAP_TIMES:
            if target not in shots or abs(ts - target) < abs(shots[target][0] - target):
                shots[target] = (ts, np.linalg.norm(om, axis=-1))
    t_s, eps_phys = np.array(t_s), np.array(eps_phys)

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "axes.labelsize": 12,
                         "axes.titlesize": 13, "legend.fontsize": 10, "figure.dpi": 150})
    colors = {32: "#2980b9", 64: "#e74c3c"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(r"3-D Taylor-Green vortex ($Re = 1600$, $Ma = 0.1$, periodic $(2\pi)^3$)", fontsize=15,
                 fontweight="bold")
    for n, (t, ke, eps) in runs.items():
        a1.plot(t, ke / ke[0], color=colors[n], lw=2.2, label=f"{n}$^3$")
        a2.plot(t, eps, color=colors[n], lw=2.2, label=f"$-dE_k/dt$, {n}$^3$")
    a2.plot(t_s, eps_phys, "o", color=colors[32], ms=5, mec="white", mfc=colors[32], alpha=0.85,
            label=r"$2\nu\zeta$ (resolved), 32$^3$")
    a2.plot(*DNS_PEAK, "*", color="#2c3e50", ms=16, mec="white", label="DNS peak (van Rees et al. 2011)")
    a1.set_xlabel("Time, $t$"); a1.set_ylabel(r"Kinetic energy, $E_k(t)/E_k(0)$")
    a1.set_title("Kinetic-energy decay")
    a2.set_xlabel("Time, $t$"); a2.set_ylabel(r"Dissipation rate, $\varepsilon$")
    a2.set_title(r"Dissipation: total $-dE_k/dt$ vs resolved $2\nu\zeta$")
    for a in (a1, a2):
        a.set_xlim(0, 20)
        a.grid(True, ls="--", alpha=0.6)
    a1.legend(frameon=True)
    a2.set_ylim(0, 0.0185)   # head-room for the legend above the DNS peak
    a2.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    out1 = os.path.join(ROOT, "results", "taylor_green_3d_validation.png")
    os.makedirs(os.path.dirname(out1), exist_ok=True)
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    print(f"[PLOT] {out1}")

    vmax = max(np.percentile(f, 99.5) for _, f in shots.values())
    fig = plt.figure(figsize=(16, 7.6))
    fig.suptitle(r"3-D Taylor-Green vortex, 32$^3$: vorticity magnitude $|\omega|$", fontsize=15, fontweight="bold")
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
    out2 = os.path.join(ROOT, "results", "taylor_green_3d_vorticity.png")
    plt.savefig(out2, dpi=170, bbox_inches="tight")
    print(f"[PLOT] {out2}")


if __name__ == "__main__":
    main()
