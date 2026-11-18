"""positivity_fallback on MPI ranks: bitwise equal to serial.

    python cases/validation/features/positivity_mpi.py --check     # serial against 2 to 4 ranks, strips in x and in y

Two periodic jets collide head on (Mach ~ 90 against a pressure of 0.4: the stage leaves the admissible set and is repeated with
first-order faces around the cells that lost density or pressure). The jump sits at a few places around the interface of two
strips: the flags of the cells across the interface are exchanged like the states, and the "the set grew" decision is the OR over
the ranks, so the ranks repeat a stage together. Serial and ranks must give the same bits whatever the position of the jump.
"""
import sys

import numpy as np

import fenice
from fenice import study

STUDY = "positivity_mpi"
CASE = """[case]
name = "jets_{axis}"
[mesh]
Lx = {lx}
Ly = {ly}
nx = {nx}
ny = {ny}
[boundaries]
bc_left = "periodic"
bc_right = "periodic"
bc_bottom = "periodic"
bc_top = "periodic"
[numerics]
riemann_solver = "HLLC"
reconstruction = "weno3"
time_scheme = "SSP Runge-Kutta 3"
cfl = 0.8
positivity_fallback = true
[run]
t_end = 0.0006
save_interval = 1000000000
[initial]
state = {{ rho = 1.0, u = {u0}, v = {v0}, p = 0.4 }}
[[initial.region]]
shape = "halfspace"
point = [{px}, {py}]
normal = [{nx_}, {ny_}]
state = {{ rho = 1.0, u = {u1}, v = {v1}, p = 0.45 }}
"""


def case_file(axis, jump):
    """A 400 x 4 (axis x) or 4 x 400 (axis y) strip; the jump at cell `jump` of the long axis."""
    x = axis == "x"
    s = CASE.format(axis=axis, lx=1.0 if x else 0.01, ly=0.01 if x else 1.0, nx=400 if x else 4, ny=4 if x else 400,
                    u0=-80.0 if x else 0.0, v0=0.0 if x else -80.0, u1=72.0 if x else 0.0, v1=0.0 if x else 72.0,
                    px=jump / 400 if x else 0.0, py=0.0 if x else jump / 400, nx_=1.0 if x else 0.0, ny_=0.0 if x else 1.0)
    d = study.study_dir(STUDY) / f"case_{axis}{jump}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "case.toml").write_text(s, newline="\n") if sys.version_info >= (3, 10) else (d / "case.toml").write_text(s)
    return d


def check(ranks):
    ok = True
    for axis in ("x", "y"):
        for jump in (133, 198, 199, 200, 201, 202):   # the strips of 2 and 4 ranks meet at 200 and 100 / 300; 3 ranks at 133 / 267
            folder = case_file(axis, jump)
            out = study.study_dir(STUDY) / "check" / f"{axis}{jump}"
            out.mkdir(parents=True, exist_ok=True)
            fenice.run(str(folder), out_root=out / "serial", dump=out / "serial.npz")
            ref = fenice.read_dump(out / "serial.npz")["U"]
            line = f"{axis} jump {jump:3d} serial"
            for n in range(2, ranks + 1):
                fenice.run(str(folder), out_root=out / f"np{n}", mpi=n, dump=out / f"np{n}.npz")
                same = np.array_equal(fenice.read_dump(out / f"np{n}.npz")["U"], ref)
                ok = ok and same
                line += f" | np{n} {'==' if same else 'DIFFERS'}"
            print(line, flush=True)
    print("ALL MATCH (bitwise)" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check(4) if "--check" in sys.argv else print(__doc__) or 0)
