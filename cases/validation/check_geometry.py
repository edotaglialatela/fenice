"""check_geometry.py -- the immersed-body preview (fenice.geometry) against the solver.

For every case with an immersed body: the number of solid cells of the preview's mask on the case's mesh must equal the one the solver
prints at setup (``[IBM] ... -> N solid cells``), for the case as it is and at a coarser and a finer mesh. Exit code 0 if all agree.

    python cases/validation/check_geometry.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import fenice  # noqa: E402
from fenice import geometry  # noqa: E402

COUNT = re.compile(r"\[IBM\] .*-> (\d+) solid cells")


def solver_count(case: str, sets: list[str]) -> int | None:
    """Solid cells the solver reports for the case with `sets` applied (one step is enough: the count is printed at setup)."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [str(fenice.exe()), "--case", case, "--no-logo", "--set", "max_steps=1", "--set", f"output_root={tmp}"]
        for s in sets:
            cmd += ["--set", s]
        out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout
    m = COUNT.search(out)
    return int(m[1]) if m else None


def resolved(case: str) -> dict:
    out = subprocess.run([str(fenice.exe()), "--case", case, "--print-config"], capture_output=True, text=True, cwd=ROOT).stdout
    return dict(line.split(" = ", 1) for line in out.splitlines() if " = " in line)


def airfoil_file_checks() -> tuple[int, int]:
    """A Selig and a Lednicer file of a NACA 4412 section, placed with a chord and an angle, against the solver."""
    u = geometry.naca4("4412", 60)
    n = 60
    up, lo = u[: n + 1][::-1], u[n:]
    nl = chr(10)
    ok_count = 0
    with tempfile.TemporaryDirectory() as tmp:
        selig = Path(tmp) / "selig.dat"
        selig.write_text("section" + nl + nl.join(f"{x:.6f} {y:.6f}" for x, y in u) + nl)
        led = Path(tmp) / "lednicer.dat"
        led.write_text("section" + nl + f"{len(up)}. {len(lo)}." + nl + nl + nl.join(f"{x:.6f} {y:.6f}" for x, y in up) + nl + nl
                       + nl.join(f"{x:.6f} {y:.6f}" for x, y in lo) + nl)
        for path in (selig, led):
            q = {"use_ibm": "true", "ibm_shape": "Airfoil", "ibm_file": str(path), "ibm_cx": "3.0", "ibm_cy": "3.0", "ibm_chord": "1.3",
                 "ibm_aoa": "7.5"}
            mine = int(geometry.solid_mask(geometry.body_from_params(q, ROOT), 12.0, 6.0, 480, 240).sum())
            theirs = solver_count("IBM Airfoil", ["use_amr=false", "nx=480", "ny=240", "Lx=12", "Ly=6", "ibm_shape=Airfoil",
                                                  f"ibm_file={path}", "ibm_cx=3.0", "ibm_cy=3.0", "ibm_chord=1.3", "ibm_aoa=7.5"])
            ok = mine == theirs
            ok_count += ok
            print(f"[{'OK' if ok else 'FAIL'}] airfoil file {path.name:14s}              preview {mine:7d}  solver {theirs}")
    return ok_count, 2


def main() -> int:
    cases = subprocess.run([str(fenice.exe()), "--list-cases"], capture_output=True, text=True, cwd=ROOT).stdout.split(chr(10))
    bad, n = 0, 0
    for case in (c.strip() for c in cases if c.strip()):
        p = resolved(case)
        if p.get("use_ibm", "false").lower() != "true":
            continue
        for scale in (1.0, 0.5, 1.5):
            nx, ny = int(int(p["nx"]) * scale), int(int(p["ny"]) * scale)
            q = dict(p, nx=str(nx), ny=str(ny))
            body = geometry.body_from_params(q, ROOT)
            mine = int(geometry.solid_mask(body, float(p["Lx"]), float(p["Ly"]), nx, ny).sum())
            theirs = solver_count(case, [f"nx={nx}", f"ny={ny}"] + (["use_amr=false"] if p.get("use_amr", "false") == "true" else []))
            ok = theirs == mine
            n += 1
            bad += not ok
            print(f"[{'OK' if ok else 'FAIL'}] {case:34s} {nx:5d} x {ny:<5d} preview {mine:7d}  solver {theirs}")
    good, m = airfoil_file_checks()
    n, bad = n + m, bad + (m - good)
    print(f"{n - bad}/{n} agree")
    return 1 if bad or n == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
