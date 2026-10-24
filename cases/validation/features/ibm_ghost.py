"""Ghost-cell (sharp-interface) IBM wall: MPI against serial, bit for bit.

    python cases/validation/features/ibm_ghost.py --check     # serial against 2 to 7 MPI ranks, six cases
    python cases/validation/features/ibm_ghost.py --loads     # cylinder Re = 40: monitor loads against a control volume (hours)

Runs short ibm_method = ghost cases serially and on 2 to 7 MPI ranks and compares the final states bit for bit, and the loads
the monitor writes (Cd, Cl) to roundoff (the strips' load sums are added in another order): a viscous cylinder (no-slip), a
short channel whose strips are a few rows thick (the image points of a strip's ghost cells lie in other strips), the Mach 2
cylinder (slip wall), the NACA 0012 airfoil (a polygon body), and on AMR meshes the airfoil and an adapting cylinder (the
ghost table lives on the finest level and its image leaves are part of the MPI halo). Each rank fills its own ghost cells
from the cells their image points read, so the states must not depend on the rank count. CI runs it on every platform.

--loads: the steady cylinder at Re = 40 (Mach 0.27 to shorten the runs; channel 4 x 2, 10 % blockage) with the ghost-cell wall at
20 cells per diameter (to t = 40) and 40 (hot-started from it, 15 more time units). The monitor's loads (the momentum flux into the
solid cells, from a residual of the state) are compared with an independent control-volume momentum balance of the final field on
three boxes (face values and central-difference stresses from the cell averages), after checking that Cd has settled. Then the
unsteady case: the NACA 0012 airfoil at Re = 1000 (360 x 180) at step 1250 (t = 3.2, vortex shedding), against the same balance
with the momentum change of the fluid cells in the box (central difference of the states at steps 1249 and 1251).
"""
from pathlib import Path

import numpy as np

import fenice
from fenice import study

STUDY = "ibm_ghost"
CHECK_CASES = {   # name: (case, overrides)
    "cylinder": ("IBM Flow over Cylinder", ["nx=200", "ny=100", "max_steps=30"]),
    "thin_strips": ("IBM Flow over Cylinder", ["Lx=1.5", "Ly=0.5", "nx=150", "ny=50", "ibm_cx=0.5", "ibm_cy=0.25",
                                               "max_steps=30"]),
    "supersonic": ("IBM Supersonic Flow over Cylinder", ["nx=100", "ny=100", "max_steps=30"]),
    "airfoil": ("IBM Airfoil", ["nx=192", "ny=96", "max_steps=20"]),
    # AMR: the ghost table on the finest level, the image leaves in the MPI halo
    "amr_airfoil": ("IBM Airfoil AMR", ["max_steps=12"]),
    "amr_cylinder": ("IBM Flow over Cylinder", ["use_amr=true", "amr_max_level=2", "amr_refine_every=4", "nx=100", "ny=50",
                                                "max_steps=16"]),
}


def loads(run):
    d = fenice.read_monitor(study.case_output(run))
    return np.array([d["Cd"][-1], d["Cl"][-1]])


def check(ranks):
    """Serial against 2..N ranks: identical final states, loads equal to roundoff."""
    ok = True
    for name, (case, sets) in CHECK_CASES.items():
        sets = ["ibm_method=ghost", *sets, "save_interval=1000000000", "monitor_interval=10"]
        out = study.study_dir(STUDY) / "check" / name
        out.mkdir(parents=True, exist_ok=True)
        fenice.run(case, *sets, out_root=out / "serial", dump=out / "serial.npz")
        ref, ref_loads = fenice.read_dump(out / "serial.npz"), loads(out / "serial")
        line = f"{name:12s} serial (Cd {ref_loads[0]:.6f})"
        for n in range(2, ranks + 1):
            fenice.run(case, *sets, out_root=out / f"np{n}", mpi=n, dump=out / f"np{n}.npz")
            same = np.array_equal(fenice.read_dump(out / f"np{n}.npz")["U"], ref["U"])
            close = np.allclose(loads(out / f"np{n}"), ref_loads, rtol=1e-10, atol=1e-12)
            ok = ok and same and close
            line += f" | np{n} {'==' if same else 'DIFFERS'}{'' if close else ' (loads differ)'}"
        print(line, flush=True)
    print("ALL MATCH (bitwise)" if ok else "MISMATCH")
    return 0 if ok else 1


LOAD_RUNS = {   # cells per diameter: (mesh, extra settings, hot start from)
    20: (["nx=400", "ny=200", "t_end=40"], None),
    40: (["nx=800", "ny=400", "t_end=15"], 20),
}
RE40 = {"mu": 5e-3, "p": 9.8, "D": 0.2, "cx": 1.0, "cy": 1.0, "Lx": 4.0, "Ly": 2.0}


def re40_case():
    """The cylinder case at Re = 40, Mach 0.27 (a case folder in the study directory: the initial pressure is part of the case)."""
    src = (fenice.ROOT / "cases" / "ibm_flow_over_cylinder" / "case.toml").read_text()
    for a, b in [("mu = 2e-3", f"mu = {RE40['mu']}"), ("p_out = 100.0", f"p_out = {RE40['p']}"),
                 ("p = 100.0 }", f"p = {RE40['p']} }}"), ('name = "IBM Flow over Cylinder"', 'name = "Cylinder Re40"')]:
        assert a in src, a
        src = src.replace(a, b)
    d = study.study_dir(STUDY) / "re40_case"
    d.mkdir(exist_ok=True)
    (d / "case.toml").write_text(src)
    return d


def control_volume(U, nx, ny, half):
    """(Cd, Cl) of the cylinder from the steady momentum balance on the box of half-width `half` diameters around it."""
    c = RE40
    D = c["D"]
    box = (c["cx"] - half * D, c["cx"] + half * D, c["cy"] - half * D, c["cy"] + half * D)
    fx, fy = box_force(U, nx, ny, c["Lx"], c["Ly"], c["mu"], box)
    return fx / (0.5 * D), fy / (0.5 * D)


def box_force(U, nx, ny, Lx, Ly, mu, box, dmdt=None):
    """Force on what the box (x0, x1, y0, y1) holds: F = - sum over its faces of (rho u u.n + p n - tau.n) dS (face values = mean of
    the two cells, stresses by central differences) - d/dt of the momentum inside (dmdt: per cell, (n, 2), optional)."""
    dx, dy = Lx / nx, Ly / ny
    rho, u, v, p = (a.reshape(ny, nx) for a in fenice.primitive(U))
    ux, uy = np.gradient(u, dx, dy, axis=(1, 0))
    vx, vy = np.gradient(v, dx, dy, axis=(1, 0))
    div = ux + vy
    Pxx = rho * u * u + p - mu * (2 * ux - 2 / 3 * div)
    Pyy = rho * v * v + p - mu * (2 * vy - 2 / 3 * div)
    Pxy = rho * u * v - mu * (uy + vx)
    i0, i1, j0, j1 = round(box[0] / dx), round(box[1] / dx), round(box[2] / dy), round(box[3] / dy)
    xs = lambda P, i: (0.5 * (P[j0:j1, i - 1] + P[j0:j1, i])).sum() * dy   # x-face between cells i-1 and i
    ys = lambda P, j: (0.5 * (P[j - 1, i0:i1] + P[j, i0:i1])).sum() * dx
    fx = -(xs(Pxx, i1) - xs(Pxx, i0) + ys(Pxy, j1) - ys(Pxy, j0))
    fy = -(xs(Pxy, i1) - xs(Pxy, i0) + ys(Pyy, j1) - ys(Pyy, j0))
    if dmdt is not None:
        fx -= dmdt[:, 0].reshape(ny, nx)[j0:j1, i0:i1].sum() * dx * dy
        fy -= dmdt[:, 1].reshape(ny, nx)[j0:j1, i0:i1].sum() * dx * dy
    return fx, fy


def airfoil_unsteady(args):
    """The airfoil (Re = 1000, chord 1, quarter chord at (3, 3)) at step 1250: monitor loads against the unsteady box balance."""
    nx, ny, step = 360, 180, 1250
    sets = ["ibm_method=ghost", f"nx={nx}", f"ny={ny}", "save_interval=1000000000"]
    runs = {k: study.run_cached(STUDY, f"airfoil_{k}", "IBM Airfoil", *sets, f"max_steps={k}", f"monitor_interval={step}",
                                mpi=args.mpi, force=args.force) for k in (step - 1, step, step + 1)}
    # the solid cells (zero velocity after the mask method's initial forcing): their states are imposed, not part of the balance
    mask_run = study.run_cached(STUDY, "airfoil_mask", "IBM Airfoil", f"nx={nx}", f"ny={ny}", "max_steps=1",
                                "save_interval=1000000000", force=args.force)
    fluid = fenice.read_dump(Path(mask_run) / "final.npz")["U"][:, 1] != 0.0
    a, b, c = (fenice.read_dump(Path(runs[k]) / "final.npz") for k in (step - 1, step, step + 1))
    dmdt = (c["U"][:, 1:3] - a["U"][:, 1:3]) / (float(c["t"]) - float(a["t"])) * fluid[:, None]
    d = fenice.read_monitor(study.case_output(runs[step]))
    cv = [box_force(b["U"], nx, ny, 12.0, 6.0, 1e-3, box, dmdt) for box in
          ((2.3, 4.3, 2.4, 3.6), (2.0, 4.8, 2.2, 3.8), (1.7, 5.3, 2.0, 4.0))]
    print(f"airfoil Re = 1000, t = {float(b['t']):.2f}: monitor Cd {d['Cd'][-1]:.4f} Cl {d['Cl'][-1]:.4f}; control volume "
          + " / ".join(f"Cd {fx / 0.5:.4f} Cl {fy / 0.5:.4f}" for fx, fy in cv) + " (three boxes)")


def loads_study(args):
    case = re40_case()
    for cpd, (sets, src) in LOAD_RUNS.items():
        sets = ["ibm_method=ghost", *sets, "monitor_interval=200", "save_interval=1000000000"]
        if src:
            restart = next((study.study_dir(STUDY) / f"re40_{src}").glob("*/*_restart.npz"))
            sets += ["use_hot_start=true", f"hot_start_file={restart}"]
        run = study.run_cached(STUDY, f"re40_{cpd}", str(case), *sets, mpi=args.mpi, force=args.force)
        d = fenice.read_monitor(study.case_output(run))
        late = d["t"] >= d["t"][-1] - 5.0
        drift = (d["Cd"][late].max() - d["Cd"][late].min()) / d["Cd"][-1]
        nx = int(next(x for x in sets if x.startswith("nx=")).split("=")[1])
        U = fenice.read_dump(Path(run) / "final.npz")["U"]
        cv = [control_volume(U, nx, nx // 2, h) for h in (1.5, 2.5, 3.5)]
        print(f"{cpd} cells per diameter, t = {d['t'][-1]:.1f}: monitor Cd {d['Cd'][-1]:.4f} Cl {d['Cl'][-1]:+.1e} "
              f"(Cd varies by {drift:.1e} over the last 5 time units); control volume Cd "
              + " / ".join(f"{a:.4f}" for a, _ in cv) + " (boxes of 3, 5, 7 diameters)")
    airfoil_unsteady(args)
    return 0


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="MPI (2..7 ranks) against serial, bit for bit")
    ap.add_argument("--loads", action="store_true", help="cylinder Re = 40: monitor loads against a control volume")
    args = ap.parse_args()
    if not (args.check or args.loads):
        ap.error("choose --check or --loads")
    raise SystemExit(check(7) if args.check else loads_study(args))


if __name__ == "__main__":
    main()
