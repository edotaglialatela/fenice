"""Hagen-Poiseuille channel flow: developed profile, grid convergence and serial vs MPI.

    python cases/validation/verification/poiseuille.py            # results/validation/poiseuille/
    python cases/validation/verification/poiseuille.py --docs     # also docs/img/poiseuille_*.png

Runs the "Poiseuille Flow" case (BDF2 dual time, t = 80) on ny = 10, 20, 40 (nx = 3 ny) and
compares u(y) at x = 0.95 Lx with the parabola u = 6 u_in y (Ly - y) / Ly^2. Runs the ny = 20
grid again on 4 MPI ranks (needs mpiexec) and checks it against the serial run, which must
agree to round-off (the all-reduced mass-flow inlet average sums in a different order). Figures: poiseuille_convergence.png, poiseuille_mpi.png.
"""
import numpy as np

import fenice
from fenice import study

STUDY, CASE = "poiseuille", "Poiseuille Flow"
GRIDS, U_IN, LX, LY = [10, 20, 40], 0.01, 3.0, 1.0
MPI_RANKS = 4


def profile(run_dir, ny):
    """u(y) on the column at x = 0.95 Lx, cell centres y, and the analytic profile there."""
    nx = 3 * ny
    U = fenice.read_dump(run_dir / "final.npz")["U"]
    u = (U[:, 1] / U[:, 0]).reshape(ny, nx)[:, int(0.95 * nx)]
    y = (np.arange(ny) + 0.5) * LY / ny
    return y, u, 6 * U_IN * y * (LY - y) / LY ** 2


def main():
    args = study.parser(__doc__.splitlines()[0]).parse_args()
    plt = study.use_style()
    run = lambda tag, ny, mpi: study.run_cached(STUDY, tag, CASE, f"nx={3 * ny}", f"ny={ny}", mpi=mpi,
                                                force=args.force)
    prof, err = {}, []
    for ny in GRIDS:
        prof[ny] = profile(run(f"ny{ny}", ny, args.mpi), ny)
        y, u, ue = prof[ny]
        err.append(np.sqrt(np.mean((u - ue) ** 2)) / ue.max())
        print(f"ny = {ny:3d}: relative L2 error {err[-1]:.3e}, flow rate {np.sum(u) * LY / ny / (U_IN * LY):.5f} u_in Ly")
    err = np.array(err)
    print("observed orders:", " ".join(f"{o:.2f}" for o in np.log2(err[:-1] / err[1:])))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(r"Hagen-Poiseuille flow: developed profile at $x = 0.95\,L_x$ and grid convergence", fontsize=15,
                 fontweight="bold")
    yf = np.linspace(0, LY, 300)
    a1.plot(6 * U_IN * yf * (LY - yf) / LY ** 2, yf, color="#2c3e50", lw=2.2, label="Exact")
    for ny, c, m in zip(GRIDS, ("#85c1e9", "#2e86c1", "#1b4f72"), ("s", "^", "o")):
        y, u, _ = prof[ny]
        a1.plot(u, y, m, color=c, ms=5, mec="white", mew=0.5, label=f"$n_y = {ny}$")
    a1.set_xlabel("$u$")
    a1.set_ylabel("$y$")
    a1.set_title("Velocity profile")
    a1.legend()
    h = LY / np.array(GRIDS, dtype=float)
    p = np.polyfit(np.log(h), np.log(err), 1)[0]
    a2.loglog(h, err, "D-", color="#8e44ad", lw=2, ms=7, mec="white", label=f"Relative $L_2$ error (slope {p:.2f})")
    a2.loglog(h, err[0] * (h / h[0]) ** 2, "--", color="gray", label="2nd order")
    a2.set_xlabel(r"Grid spacing $\Delta y$")
    a2.set_ylabel("Relative $L_2$ error of $u$")
    a2.set_title("Grid convergence")
    a2.grid(True, which="both", ls="--", alpha=0.5)
    a2.legend()
    plt.tight_layout()
    study.save(fig, STUDY, "poiseuille_convergence.png", args.docs)

    ny = 20
    serial = profile(run(f"ny{ny}", ny, args.mpi), ny)
    par = profile(run(f"ny{ny}_mpi{MPI_RANKS}", ny, MPI_RANKS), ny)
    U_s = fenice.read_dump(study.study_dir(STUDY) / f"ny{ny}" / "final.npz")["U"]
    U_p = fenice.read_dump(study.study_dir(STUDY) / f"ny{ny}_mpi{MPI_RANKS}" / "final.npz")["U"]
    diff = np.max(np.abs(U_s - U_p))
    print(f"serial vs {MPI_RANKS} MPI ranks, whole field: max |dU| = {diff:.3e}")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(f"Hagen-Poiseuille flow: serial vs {MPI_RANKS} MPI ranks ($n_y = {ny}$)", fontsize=15,
                 fontweight="bold")
    a1.plot(6 * U_IN * yf * (LY - yf) / LY ** 2, yf, color="#2c3e50", lw=2.2, label="Exact")
    a1.plot(serial[1], serial[0], "o", color="#2e86c1", ms=7, mfc="none", mew=1.2, label="Serial")
    a1.plot(par[1], par[0], "x", color="#e74c3c", ms=6, mew=1.4, label=f"MPI, {MPI_RANKS} ranks")
    for k in range(1, MPI_RANKS):   # y-strip interfaces
        a1.axhline(k * LY / MPI_RANKS, color="#e74c3c", lw=0.8, ls=":")
    a1.set_xlabel("$u$")
    a1.set_ylabel("$y$")
    a1.set_title("Velocity profile (dotted: rank interfaces)")
    a1.legend()
    a2.plot(np.abs(serial[1] - par[1]), serial[0], "D-", color="#8e44ad", ms=5)
    a2.set_xlabel(r"$|u_{\rm serial} - u_{\rm MPI}|$")
    a2.set_ylabel("$y$")
    a2.set_title(f"Difference (whole field: max $|\\Delta U|$ = {diff:.1e})")
    plt.tight_layout()
    study.save(fig, STUDY, "poiseuille_mpi.png", args.docs)


if __name__ == "__main__":
    main()
