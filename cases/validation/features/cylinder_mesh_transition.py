"""Cylinder loads across a hot start from a coarse to a fine mesh (animated).

    python cases/validation/features/cylinder_mesh_transition.py            # results/validation/cylinder_mesh_transition/
    python cases/validation/features/cylinder_mesh_transition.py --docs     # also docs/img/cylinder_mesh_transition.gif

Runs the "IBM Flow over Cylinder" case on the default 400 x 200 mesh to t = 6, then continues
on 800 x 400 for 3 more time units by hot start from the coarse checkpoint (16 MPI ranks unless
--mpi says otherwise; the fine stage takes the longest). Animates Cd and Cl from the monitor of
both stages: the shedding carries on across the switch, without a new start-up transient.
Figure: cylinder_mesh_transition.gif (plus a still, cylinder_mesh_transition.png).
"""
import numpy as np

import fenice
from fenice import study

STUDY, CASE = "cylinder_mesh_transition", "IBM Flow over Cylinder"
COARSE, FINE, T_COARSE, T_FINE = (400, 200), (800, 400), 6.0, 3.0
CD_REF = 1.34   # Re = 100 (Williamson 1996; Park et al. 1998)


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    from matplotlib.animation import FuncAnimation, PillowWriter
    mpi = args.mpi or 16
    coarse = study.run_cached(STUDY, "coarse", CASE, f"nx={COARSE[0]}", f"ny={COARSE[1]}", f"t_end={T_COARSE}",
                              "monitor_interval=20", mpi=mpi, force=args.force, done="*/*_monitor.csv")
    src = next(coarse.glob("*/*_restart.npz"))
    fine = study.run_cached(STUDY, "fine", CASE, f"nx={FINE[0]}", f"ny={FINE[1]}", f"t_end={T_FINE}",
                            "monitor_interval=20", "use_hot_start=true", f"hot_start_file={src.as_posix()}",
                            mpi=mpi, force=args.force, done="*/*_monitor.csv")
    mc, mf = fenice.read_monitor(study.case_output(coarse)), fenice.read_monitor(study.case_output(fine))
    tc, tf = mc["t"], mf["t"]
    if tf[0] < tc[-1] - 1e-9:   # the hot-started run counts time from 0
        tf = tf + tc[-1]
    late_c, late_f = tc >= tc[-1] - 2.0, tf >= tf[-1] - 2.0
    print(f"last 2 time units: Cd coarse {mc['Cd'][late_c].mean():.3f}, fine {mf['Cd'][late_f].mean():.3f}; "
          f"Cl amplitude coarse {np.ptp(mc['Cl'][late_c]) / 2:.3f}, fine {np.ptp(mf['Cl'][late_f]) / 2:.3f}")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    fig.suptitle(rf"Cylinder, $Re = 100$: hot start {COARSE[0]}$\times${COARSE[1]} $\to$ {FINE[0]}$\times${FINE[1]}",
                 fontsize=15, fontweight="bold")
    t_end = tf[-1]
    ys = {"Cd": np.concatenate([mc["Cd"], mf["Cd"]]), "Cl": np.concatenate([mc["Cl"], mf["Cl"]])}
    for ax, key, label in ((a1, "Cd", "$C_d$"), (a2, "Cl", "$C_l$")):
        ax.set_xlim(0, t_end)
        pad = 0.1 * np.ptp(ys[key][tc.size // 6:])
        ax.set_ylim(ys[key][tc.size // 6:].min() - pad, ys[key][tc.size // 6:].max() + pad)
        ax.axvline(tc[-1], color="k", ls="-.", lw=1.3, alpha=0.8)
        ax.set_ylabel(label)
    a1.axhline(CD_REF, color="gray", ls="--", lw=1.2, label=f"Literature $C_d$ = {CD_REF}")
    a2.set_xlabel("Time, $t$")
    a2.text(tc[-1], a2.get_ylim()[1], " hot start", va="top", fontsize=10)
    lines = {}
    for ax, key in ((a1, "Cd"), (a2, "Cl")):
        lines[key, "c"], = ax.plot([], [], color="#5dade2", lw=1.6, label=f"{COARSE[0]}$\\times${COARSE[1]}")
        lines[key, "f"], = ax.plot([], [], color="#c0392b", lw=2.0, label=f"{FINE[0]}$\\times${FINE[1]}")
        lines[key, "h"], = ax.plot([], [], "o", color="k", ms=5)
    a1.legend(loc="upper left", ncol=3)
    plt.tight_layout()

    frames = np.linspace(tc[1], t_end, 120)

    def draw(t):
        for key, m in (("Cd", None), ("Cl", None)):
            kc, kf = tc <= t, tf <= t
            lines[key, "c"].set_data(tc[kc], mc[key][kc])
            lines[key, "f"].set_data(tf[kf], mf[key][kf])
            tt, yy = (tf[kf], mf[key][kf]) if kf.any() else (tc[kc], mc[key][kc])
            lines[key, "h"].set_data(tt[-1:], yy[-1:])
        return list(lines.values())

    draw(t_end)
    study.save(fig, STUDY, "cylinder_mesh_transition.png")   # still for results only; the docs use the GIF
    out = study.study_dir(STUDY) / "cylinder_mesh_transition.gif"
    anim = FuncAnimation(fig, draw, frames=list(frames) + [t_end] * 15, blit=True)
    anim.save(out, writer=PillowWriter(fps=15), dpi=90)
    print(f"[GIF] {out}")
    if args.docs:
        study.copy_to_docs(out)


if __name__ == "__main__":
    main()
