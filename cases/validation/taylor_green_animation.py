"""Taylor-Green vortex decaying, vorticity drawn as a surface (animated).

    python cases/validation/taylor_green_animation.py            # results/validation/taylor_green_animation/
    python cases/validation/taylor_green_animation.py --docs     # also docs/img/taylor_green_animation.gif

Runs the "Taylor-Green Vortex" case (64 x 64, t = 10) once with snapshots, at a viscosity five
times the case's (nu = 0.1) so that the decay is clearly visible: the energy falls to
exp(-4) of its initial value, with CFL = 0.25: at this viscosity the default 0.5 lets a slowly
growing mode eat the energy after t ~ 2 (the time step is the smaller of the convective and
the viscous limits, not a combination of them), and at nu = 0.2 it is unstable. Each frame shows the vorticity w = dv/dx - du/dy of the solver as
a surface z = w(x, y), with the exact w = 2 k cos(kx) cos(ky) exp(-2 nu k^2 t) as a wireframe
over it, and the kinetic energy against exp(-4 nu k^2 t). Figure: taylor_green_animation.gif
(plus a still, taylor_green_animation.png).
"""
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

import fenice
from fenice import study

STUDY, CASE = "taylor_green_animation", "Taylor-Green Vortex"
MU, L, N, T_END = 0.1, 2.0 * np.pi, 64, 10.0
K = 2.0 * np.pi / L
WIRE = 4   # the exact solution is drawn on every 4th cell


def snapshots(run_dir):
    """[(t, x, y, vorticity, kinetic energy)] of every snapshot; centred differences, periodic."""
    out = []
    for t, path in fenice.read_pvd(study.case_output(run_dir)):
        f = fenice.read_vtr(path)
        x, y = 0.5 * (f["x"][1:] + f["x"][:-1]), 0.5 * (f["y"][1:] + f["y"][:-1])
        u, v = f["Velocity"][..., 0], f["Velocity"][..., 1]
        dx, dy = x[1] - x[0], y[1] - y[0]
        w = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dx) \
            - (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)
        out.append((t, x, y, w, np.sum(0.5 * f["Density"] * (u ** 2 + v ** 2))))
    return out


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    run = study.run_cached(STUDY, "run", CASE, f"nx={N}", f"ny={N}", f"mu={MU}", f"t_end={T_END}", "cfl=0.25", "save_interval=80",
                           mpi=args.mpi, force=args.force)
    data = snapshots(run)
    ke = np.array([d[4] for d in data]) / data[0][4]
    ts = np.array([d[0] for d in data])
    X, Y = np.meshgrid(data[0][1], data[0][2])
    wmax = 2.0 * K
    exact = lambda t: wmax * np.cos(K * X) * np.cos(K * Y) * np.exp(-2.0 * MU * K ** 2 * t)
    # the solver's vorticity at t = 0 must be the analytic one (centred differences: second order)
    print(f"t = 0: max |w_solver - w_exact| = {np.abs(data[0][3] - exact(0.0)).max():.2e} (w_max = {wmax:.2f})")
    print(f"t = {ts[-1]:.1f}: max |w_solver - w_exact| = {np.abs(data[-1][3] - exact(ts[-1])).max():.2e}, "
          f"E/E0 = {ke[-1]:.4f} (exact {np.exp(-4 * MU * K ** 2 * ts[-1]):.4f})")

    fig = plt.figure(figsize=(10.5, 4.6))
    a3 = fig.add_subplot(1, 2, 1, projection="3d")
    a2 = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.0, right=0.97, bottom=0.12, top=0.9, wspace=0.12)
    tt = np.linspace(0.0, ts[-1], 200)
    a2.semilogy(tt, np.exp(-4 * MU * K ** 2 * tt), "k", lw=1.6, label=r"exact $e^{-4\nu k^2 t}$")
    (curve,) = a2.semilogy([], [], color="C3", lw=2.2, label="solver")
    (dot,) = a2.semilogy([], [], "o", color="C3", ms=7, mec="white")
    a2.set(xlim=(0, ts[-1]), ylim=(0.01, 1.3), xlabel="$t$", ylabel=r"$E_k(t)/E_k(0)$", title="Kinetic energy")
    a2.legend(loc="upper right")

    def draw(i):
        t, x, y, w, _ = data[i]
        a3.clear()
        a3.plot_surface(X, Y, w, cmap="RdBu_r", vmin=-wmax, vmax=wmax, rstride=2, cstride=2, linewidth=0,
                        antialiased=True, alpha=0.9)
        a3.plot_wireframe(X[::WIRE, ::WIRE], Y[::WIRE, ::WIRE], exact(t)[::WIRE, ::WIRE], color="k", lw=0.8, alpha=0.9)
        a3.set(xlim=(0, L), ylim=(0, L), zlim=(-wmax, wmax), xlabel="$x$", ylabel="$y$")
        a3.set_zlabel(r"$\omega = \partial_x v - \partial_y u$", labelpad=2)
        a3.set_box_aspect((1, 1, 0.7))
        a3.view_init(elev=32, azim=-58)
        a3.set_title(rf"surface: solver, wireframe: exact   $t = {t:.2f}$", fontsize=11, pad=-4)
        curve.set_data(ts[: i + 1], ke[: i + 1])
        dot.set_data([t], [ke[i]])
        return []

    draw(0)
    study.save(fig, STUDY, "taylor_green_animation.png")
    anim = FuncAnimation(fig, draw, frames=len(data), blit=False)
    out = study.study_dir(STUDY) / "taylor_green_animation.gif"
    anim.save(out, writer=PillowWriter(fps=15))
    print(f"[GIF] {out}")
    if args.docs:
        study.copy_to_docs(out)


if __name__ == "__main__":
    main()
