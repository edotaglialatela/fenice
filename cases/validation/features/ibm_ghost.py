"""Ghost-cell (sharp-interface) IBM wall: MPI against serial, bit for bit.

    python cases/validation/features/ibm_ghost.py --check     # serial against 2 to 7 MPI ranks, four cases

Runs short ibm_method = ghost cases serially and on 2 to 7 MPI ranks and compares the final states bit for bit, and the loads
the monitor writes (Cd, Cl) to roundoff (the strips' load sums are added in another order): a viscous cylinder (no-slip), a
short channel whose strips are a few rows thick (the image points of a strip's ghost cells lie in other strips), the Mach 2
cylinder (slip wall) and the NACA 0012 airfoil (a polygon body). Each rank fills its own ghost cells and fetches the cells of
other strips their image points read, so the states must not depend on the rank count. CI runs it on every platform.
"""
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


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="MPI (2..7 ranks) against serial, bit for bit")
    args = ap.parse_args()
    if not args.check:
        ap.error("only --check is available so far")
    raise SystemExit(check(7))


if __name__ == "__main__":
    main()
