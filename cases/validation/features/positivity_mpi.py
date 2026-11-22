"""positivity_fallback on MPI ranks: bitwise equal to serial.

    python cases/validation/features/positivity_mpi.py --check     # serial against 2 to 4 ranks, strips in x and in y

Two periodic jets collide head on (Mach ~ 90 against a pressure of 0.4: the stage leaves the admissible set and is repeated with
first-order faces around the cells that lost density or pressure). The jump sits at a few places around the interface of two
strips: the flags of the cells across the interface are exchanged like the states, and the "the set grew" decision is the OR over
the ranks, so the ranks repeat a stage together. Serial and ranks must give the same bits whatever the position of the jump.
The same on the adaptive mesh (three levels, a forced finest region): there the flags travel along the stage lists of each level.
"""
import re
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
bc_xmin = "periodic"
bc_xmax = "periodic"
bc_ymin = "periodic"
bc_ymax = "periodic"
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
    with open(d / "case.toml", "w", newline="\n") as f:
        f.write(s)
    return d


# the adaptive mesh: three levels, the middle fifth of the strip forced to the finest level (the opening of the vacuum sits in it for
# the jump at the middle; with the jump at 0.3 / 0.7 of the length the opening is on coarser leaves and a seam of levels)
AMR = ["use_amr=true", "amr_max_level=2", "amr_force_box=0.4 0.6 0.0 0.01", "cfl=0.95"]


def check(ranks):
    ok = True
    runs = [("x", j, []) for j in (133, 198, 199, 200, 201, 202)]   # the strips of 2 and 4 ranks meet at 200 and 100 / 300; 3 ranks at 133 / 267
    runs += [("y", j, []) for j in (133, 198, 199, 200, 201, 202)]
    runs += [("x", j, AMR) for j in (120, 200, 280)]
    for axis, jump, sets in runs:
        folder = case_file(axis, jump)
        tag = f"{axis}{jump}{'_amr' if sets else ''}"
        out = study.study_dir(STUDY) / "check" / tag
        out.mkdir(parents=True, exist_ok=True)
        r = fenice.run(str(folder), *sets, out_root=out / "serial", dump=out / "serial.npz")
        fired = re.search(r"\[POSITIVITY\] (\d+) repair passes", r.stdout)
        if not fired or int(fired.group(1)) == 0:   # a run that never repairs proves nothing
            print(f"{tag}: the repair did not fire")
            ok = False
        ref = fenice.read_dump(out / "serial.npz")["U"]
        line = f"{axis} jump {jump:3d}{' AMR' if sets else '    '} serial"
        for n in range(2, ranks + 1):
            fenice.run(str(folder), *sets, out_root=out / f"np{n}", mpi=n, dump=out / f"np{n}.npz")
            same = np.array_equal(fenice.read_dump(out / f"np{n}.npz")["U"], ref)
            ok = ok and same
            line += f" | np{n} {'==' if same else 'DIFFERS'}"
        print(line, flush=True)
    print("ALL MATCH (bitwise)" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(check(4) if "--check" in sys.argv else print(__doc__) or 0)
