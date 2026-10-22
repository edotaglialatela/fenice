"""AMR following the waves of the Sod shock tube (animated).

    python cases/validation/amr/sod_amr_animation.py            # results/validation/sod_amr_animation/
    python cases/validation/amr/sod_amr_animation.py --docs     # also docs/img/sod_amr_animation.gif

Runs the Sod case on a 40 x 4 base mesh with 8 x 2 patches and three AMR levels (320 x 32 at the
finest level), remeshing every 2 steps, once per frame to t = k t_end / frames (the steps before the
last one are identical, so the frames are one run). Animates the density at the leaf cells coloured
by level against the exact solution, the patches below it (the refined patches travel with the
rarefaction fan, the contact and the shock, and are released behind them) and, at the bottom, the
mesh itself with its cell edges.
Figure: sod_amr_animation.gif (plus a still, sod_amr_animation.png).
"""
import numpy as np

import fenice
from fenice import study
from fenice.exact import riemann

STUDY, CASE = "sod_amr_animation", "Sod Shock Tube"
NX, NY, PX, PY, LEVELS = 40, 4, 8, 2, 3
T_END, FRAMES = 0.2, 60
SETS = (f"nx={NX}", f"ny={NY}", "Lx=1.0", "use_amr=true", f"amr_patch_nx={PX}", f"amr_patch_ny={PY}",
        f"amr_max_level={LEVELS}", f"amr_init_levels={LEVELS}", "amr_refine_every=2", "amr_buffer_layers=1",
        "save_interval=1000000000")
COLORS = ["#95a5a6", "#f39c12", "#e74c3c", "#8e44ad"]


def frame(args, k):
    """Leaf cells (centre, density averaged over the patch rows, level) and the leaves at frame k."""
    t = T_END * k / FRAMES
    run = study.run_cached(STUDY, f"f{k:03d}", CASE, *SETS, f"t_end={t:.10f}", force=args.force)
    d = fenice.read_dump(run / "final.npz")
    U, meta = d["U"], d["leaf_meta"]
    xs, rho, lev = [], [], []
    for n, (x0, x1, _, _, level) in enumerate(meta):
        c = U[n * PX * PY:(n + 1) * PX * PY, 0].reshape(PY, PX).mean(axis=0)
        xs.append(x0 + (np.arange(PX) + 0.5) * (x1 - x0) / PX)
        rho.append(c)
        lev.append(np.full(PX, int(level)))
    return t, np.concatenate(xs), np.concatenate(rho), np.concatenate(lev), meta


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    from matplotlib.animation import FuncAnimation, PillowWriter
    data = [frame(args, k) for k in range(1, FRAMES + 1)]
    xe = np.linspace(0, 1, 800)

    fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(9, 7.4), gridspec_kw=dict(height_ratios=[3, 1.1, 1.0]),
                                     sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, hspace=0.14)
    from matplotlib.collections import PolyCollection
    ly = 0.1   # the case's Ly
    a0.set_ylim(0.05, 1.08)
    a0.set_xlim(0, 1)
    a0.set_ylabel(r"density $\rho$")
    a1.set_ylim(-0.5, LEVELS + 0.5)
    a1.set_yticks(range(LEVELS + 1))
    a1.set_ylabel("level")
    a1.grid(False)
    a2.set_ylim(0, ly)
    a2.set_yticks([0, ly])
    a2.set_ylabel("$y$")
    a2.set_xlabel("$x$")
    a2.grid(False)
    a2.set_title("mesh (cell edges, coloured by level)", fontsize=10, pad=3)
    mesh_pc = []
    (exact,) = a0.plot([], [], color="k", lw=1.1, label="exact")
    dots = [a0.plot([], [], "o", ms=3.2, color=COLORS[lv], label=f"AMR cells, level {lv}")[0] for lv in range(LEVELS + 1)]
    a0.legend(loc="upper right", ncol=3, fontsize=8)
    title = a0.set_title("")
    bars = []

    def draw(i):
        t, x, rho, lev, meta = data[i]
        exact.set_data(xe, riemann(xe, t, x0=0.5)[0])
        for lv, d in enumerate(dots):
            m = lev == lv
            d.set_data(x[m], rho[m])
        while bars:
            bars.pop().remove()
        for x0, x1, _, _, level in meta[::1]:
            if (x0, x1) in {(b[0], b[1]) for b in meta[:0]}:
                continue
            bars.append(a1.add_patch(plt.Rectangle((x0, level - 0.35), x1 - x0, 0.7, fc=COLORS[int(level)], ec="white", lw=0.4)))
        while mesh_pc:
            mesh_pc.pop().remove()
        quads, colors = [], []
        for x0, x1, y0, y1, level in meta:
            dx, dy = (x1 - x0) / PX, (y1 - y0) / PY
            for j in range(PY):
                for i in range(PX):
                    xa, ya = x0 + i * dx, y0 + j * dy
                    quads.append([(xa, ya), (xa + dx, ya), (xa + dx, ya + dy), (xa, ya + dy)])
                    colors.append(COLORS[int(level)])
        pc = PolyCollection(quads, facecolors=colors, edgecolors="k", linewidths=0.25, alpha=0.55)
        mesh_pc.append(a2.add_collection(pc))
        n_cells = len(meta) * PX * PY
        title.set_text(rf"Sod shock tube, AMR: $t = {t:.3f}$, {len(meta) // 1} patches, {n_cells} cells "
                       rf"(uniform at the finest level: {NX * 2 ** LEVELS * NY * 2 ** LEVELS})")
        return dots + [exact, title]

    draw(FRAMES // 2)
    study.save(fig, STUDY, "sod_amr_animation.png")
    anim = FuncAnimation(fig, draw, frames=FRAMES, blit=False)
    out = study.study_dir(STUDY) / "sod_amr_animation.gif"
    anim.save(out, writer=PillowWriter(fps=15))
    print(f"[GIF] {out}")
    if args.docs:
        study.copy_to_docs(out)


if __name__ == "__main__":
    main()
