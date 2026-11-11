"""Where the time of an adaptive run goes: pie charts of the measured splits of the performance guide.

    python cases/validation/amr/time_split.py            # results/validation/amr_time_split/
    python cases/validation/amr/time_split.py --docs     # also docs/img/amr_time_split.png

The numbers are wall times measured with FENICE_PROFILE=1 on a Ryzen 7 7800X3D (8 cores, 16 threads), listed in
docs/source/guide/performance.md ("Where the AMR time goes"); nothing is run here. Cases: the 3-D octree sphere (Re 20,
base mesh 5 cells per box edge, 2 levels, 73 k leaves, immersed mask, remesh every 4 steps) and the 2-D IBM airfoil
(Re 1000, 192 x 96 base mesh, 2 levels, t = 20, 8 threads). Figure: amr_time_split.png.
"""
from fenice import study

STUDY = "amr_time_split"
# categorical slots 1-4 of the documentation palette, in fixed order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

# (title, [(part, seconds), ...]); the "rest" slices are differences of measured totals
PANELS = [
    ("Octree step, 16 threads\n100 steps, 1.94 s",
     [("convective faces", 0.68), ("viscous cells and faces", 0.55), ("primitive state, virtual cells", 0.33), ("rest of the step", 0.38)]),
    ("200 steps, 16 threads\n4.6 s",
     [("time step", 4.23), ("remesh (7 %)", 0.34)]),
    ("200 steps, 8 MPI ranks\n10.9 s",
     [("time step", 8.56), ("remesh (22 %)", 2.35)]),
    ("Remesh on 8 ranks\n2.35 s",
     [("fetch of the leaves", 1.33), ("indicator and flags", 0.89), ("rest of the remesh", 0.13)]),
    ("Airfoil remesh, 8 threads\n3.1 s of a 19.2 s loop",
     [("indicator", 1.15), ("engine build", 0.75), ("wave-speed bound", 0.68), ("mesh adaptation", 0.55)]),
]


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 9.4), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(False)
    for ax, (title, parts) in zip(axes.flat, PANELS):
        vals = [v for _, v in parts]
        tot = sum(vals)
        ax.pie(vals, colors=COLORS[:len(vals)], startangle=90, counterclock=False,
               wedgeprops={"edgecolor": SURFACE, "linewidth": 2},
               autopct=lambda p: f"{p:.0f} %" if p >= 8 else "", pctdistance=0.68, textprops={"color": "white", "fontsize": 11})
        ax.set_title(title, color=INK, fontsize=12)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=f"{n}: {v:.2f} s") for (n, v), c in zip(parts, COLORS)],
                  loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False, labelcolor=INK2, fontsize=10)
    axes.flat[-1].axis("off")
    fig.tight_layout(h_pad=3.5)
    study.save(fig, STUDY, "amr_time_split.png", docs=args.docs)


if __name__ == "__main__":
    main()
