"""Post-processing declared in the case file ([[probe]], [[sample]], [[force]], [[average]], time-dependent boundary values).

    python cases/validation/features/postprocess.py --check     # serial against 2 to 7 MPI ranks, resume, no side effect on the state

Short runs of case files that declare the outputs (written into the study folder: the shipped cases get the tables appended):

* the CSV files of every probe, line, force and average are identical byte for byte on 1 to 7 ranks (uniform mesh with the mask and
  the ghost-cell IBM, 3-D, AMR with and without a body), and the final state does too;
* declaring the outputs does not change the state: the final states with and without them are bit for bit equal (the time loop only
  reads the state; the force output evaluates the residual an extra time);
* a run stopped halfway and resumed writes the CSV rows of the uninterrupted run (plus the row of the stop step, no row twice), on the
  same and on another rank count, and ends in the same state;
* time-dependent boundary values (an expression of t, a table) give the same state on 1 to 7 ranks.

CI runs it on every platform.
"""
import shutil

import numpy as np

import fenice
from fenice import study

STUDY = "postprocess"

PROBES = """
[[probe]]
name = "p1"
x = {px}
y = {py}
interval = 3

[[sample]]
name = "line"
from = [{fx}, {fy}]
to = [{tx}, {ty}]
points = 7
interval = 4

[[average]]
name = "mean"
variables = ["rho", "u", "p"]
interval = 5
"""
FORCE = """
[[force]]
name = "body"
reference_length = 0.2
reference_density = 1.0
reference_velocity = 1.0
interval = 3
"""
PROBES_3D = """
[[probe]]
name = "p1"
x = 1.0
y = 2.0
z = 3.0
interval = 2

[[sample]]
name = "line"
from = [0.5, 0.5, 0.5]
to = [5.5, 5.0, 5.0]
points = 5
interval = 2

[[average]]
name = "mean"
variables = ["rho", "u", "v", "w", "p"]
interval = 2
"""


def case_folder(tag, source, tables="", replace=()):
    """results/validation/postprocess/cases/<tag>/case.toml: a shipped case with `tables` appended."""
    text = (fenice.ROOT / "cases" / source / "case.toml").read_text(encoding="utf8").replace("\r\n", "\n")
    for a, b in replace:
        assert a in text, a
        text = text.replace(a, b)
    d = study.study_dir(STUDY) / "cases" / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "case.toml").write_text(text + tables, encoding="utf8", newline="\n")
    return d


def tables(px, py, fx, fy, tx, ty, force=False):
    return PROBES.format(px=px, py=py, fx=fx, fy=fy, tx=tx, ty=ty) + (FORCE if force else "")


def csvs(run):
    """name -> bytes of every post-processing CSV of a run (the monitor sums are partition dependent to roundoff: not included)."""
    out = {}
    for p in sorted(study.case_output(run).glob("*.csv")):
        if not p.name.endswith("_monitor.csv"):
            out[p.name.split("_", 1)[1] if "_" in p.name else p.name] = p.read_bytes()
    return out


def run_dir(base, tag):
    d = base / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def same_files(a, b):
    return a.keys() == b.keys() and all(a[k] == b[k] for k in a)


def resumed_like(whole, resumed, stop):
    """The resumed run's files are the uninterrupted run's, apart from the row of the stop step (the first run writes the state it
    stops at) and the running averages (a resume starts a new window); no row appears twice."""
    if whole.keys() != resumed.keys():
        return False
    for k in whole:
        if k.endswith("_average.csv"):
            continue
        a, b = whole[k].decode().splitlines(), resumed[k].decode().splitlines()
        if len(set(b)) != len(b):
            return False
        keep = lambda rows: [r for r in rows if r.split(",")[0] != str(stop)]
        if keep(a) != keep(b):
            return False
    return True


def final(path):
    return fenice.read_dump(path)


def check_mpi(name, case, sets, ranks, base):
    """Serial against 2..ranks: CSV files and final state identical."""
    serial = run_dir(base, f"{name}_serial")
    fenice.run(case, *sets, "save_interval=1000000000", out_root=serial, dump=serial / "final.npz")
    ref, ref_csv = final(serial / "final.npz"), csvs(serial)
    assert ref_csv, f"{name}: no CSV written"
    ok, line = True, f"{name:14s} serial ({len(ref_csv)} files)"
    for n in range(2, ranks + 1):
        d = run_dir(base, f"{name}_np{n}")
        fenice.run(case, *sets, "save_interval=1000000000", out_root=d, mpi=n, dump=d / "final.npz")
        got = final(d / "final.npz")
        state = np.array_equal(got["U"], ref["U"])
        files = same_files(csvs(d), ref_csv)
        ok = ok and state and files
        line += f" | np{n} {'==' if state and files else ('state' if not state else 'csv') + ' DIFFERS'}"
    print(line, flush=True)
    return ok


def check_neutral(name, plain, declared, sets, base):
    """The final state does not depend on declaring the outputs."""
    a, b = run_dir(base, f"{name}_plain"), run_dir(base, f"{name}_declared")
    fenice.run(plain, *sets, "save_interval=1000000000", out_root=a, dump=a / "final.npz")
    fenice.run(declared, *sets, "save_interval=1000000000", out_root=b, dump=b / "final.npz")
    same = np.array_equal(final(a / "final.npz")["U"], final(b / "final.npz")["U"])
    print(f"{name:14s} outputs declared: state {'==' if same else 'DIFFERS from'} the plain run", flush=True)
    return same


RESUMES = [(0, 0), (2, 3)]   # ranks before and after (0: serial)


def check_resume(name, case, sets, steps, base):
    """Stopped at steps/2 and resumed: the same CSV files as the uninterrupted run."""
    whole = run_dir(base, f"{name}_whole")
    fenice.run(case, *sets, f"max_steps={steps}", "save_interval=1000000000", out_root=whole, dump=whole / "final.npz")
    ref_csv, ref = csvs(whole), final(whole / "final.npz")
    ok, line = True, f"{name:14s} resume"
    for a, b in RESUMES:
        d = run_dir(base, f"{name}_{a}_{b}")
        half = steps // 2
        fenice.run(case, *sets, f"max_steps={half}", f"save_interval={half}", out_root=d, mpi=a)
        fenice.run(case, *sets, f"max_steps={half}", "save_interval=1000000000", "resume=true", out_root=d, mpi=b,
                   dump=d / "final.npz")
        same = resumed_like(ref_csv, csvs(d), half) and np.array_equal(final(d / "final.npz")["U"], ref["U"])
        ok = ok and same
        line += f" | {a}->{b} {'==' if same else 'DIFFERS'}"
    print(line, flush=True)
    return ok


def check_hard_kill(name, case, sets, base):
    """A run killed after its last checkpoint left rows past it; resuming drops them: no step twice, the rows of the clean run."""
    d = run_dir(base, f"{name}_kill")
    fenice.run(case, *sets, "max_steps=12", "save_interval=12", out_root=d)
    restart = next(study.case_output(d).glob("*_restart.npz"))
    kept = restart.read_bytes()                      # the checkpoint of step 12
    fenice.run(case, *sets, "max_steps=12", "save_interval=12", "resume=true", out_root=d)   # rows up to step 24
    clean = csvs(d)
    restart.write_bytes(kept)                        # as if the run had been killed before saving the later checkpoint
    fenice.run(case, *sets, "max_steps=12", "save_interval=12", "resume=true", out_root=d)
    again = csvs(d)
    same = same_files(clean, again)
    print(f"{name:14s} killed run resumed: rows {'==' if same else 'DIFFER from'} the clean run", flush=True)
    return same


def check(ranks):
    base = study.study_dir(STUDY) / "check"
    base.mkdir(parents=True, exist_ok=True)
    ok = True
    ibm = ["max_steps=20", "nx=100", "ny=50"]
    cyl = tables(1.2, 1.0, 0.8, 1.0, 1.6, 1.0, force=True)
    mask = case_folder("cylinder_mask", "ibm_flow_over_cylinder", cyl)
    plain = case_folder("cylinder_plain", "ibm_flow_over_cylinder")
    ghost = ["ibm_method=ghost"]
    ok &= check_mpi("cylinder_mask", mask, ibm, ranks, base)
    ok &= check_mpi("cylinder_ghost", mask, ibm + ghost, ranks, base)
    ok &= check_neutral("cylinder_mask", plain, mask, ibm, base)
    ok &= check_neutral("cylinder_ghost", plain, mask, ibm + ghost, base)
    amr = ["use_amr=true", "amr_max_level=2", "amr_refine_every=4"]
    ok &= check_mpi("cylinder_amr", mask, ibm + amr, ranks, base)
    ok &= check_neutral("cylinder_amr", plain, mask, ibm + amr, base)
    sod_t = tables(0.3, 0.04, 0.0, 0.04, 1.0, 0.04)
    sod = case_folder("sod", "sod_shock_tube", sod_t)
    sod_plain = case_folder("sod_plain", "sod_shock_tube")
    sset = ["nx=50", "ny=4", "Ly=0.08", "max_steps=24"]
    ok &= check_mpi("sod", sod, sset, ranks, base)
    ok &= check_mpi("sod_amr", sod, sset + amr, ranks, base)
    ok &= check_neutral("sod_amr", sod_plain, sod, sset + amr, base)
    ok &= check_resume("sod", sod, sset, 24, base)
    ok &= check_resume("sod_amr", sod, sset + amr, 24, base)
    ok &= check_hard_kill("sod", sod, sset, base)
    ok &= check_resume("cylinder_force", mask, ibm, 20, base)
    tg = case_folder("tg3d", "taylor_green_vortex_3d", PROBES_3D)
    tg_plain = case_folder("tg3d_plain", "taylor_green_vortex_3d")
    tset = ["nx=16", "ny=16", "nz=4", "max_steps=6"]
    ok &= check_mpi("tg3d", tg, tset, ranks, base)
    ok &= check_neutral("tg3d", tg_plain, tg, tset, base)
    ok &= check_time_output("sod_time", sod_plain, sset, ranks, base)
    ok &= check_time_output("sod_amr_time", sod_plain, sset + amr, ranks, base)
    # time-dependent boundary values: the same state on any rank count
    ramp = case_folder("cavity_ramp", "lid_driven_cavity", replace=[("u_in = 1.0", 'u_in = "1.0 + 0.5*sin(2*pi*t)"')])
    ok &= check_mpi_state("cavity_ramp", ramp, ["nx=24", "ny=24", "max_steps=20"], ranks, base)
    print("ALL MATCH (bitwise)" if ok else "MISMATCH")
    return 0 if ok else 1


def check_time_output(name, case, sets, ranks, base):
    """write_interval_time: snapshots at exact multiples (and the end), the same on any rank count; dt_max caps the step."""
    sets = [x for x in sets if not x.startswith("max_steps=")]
    ok, line = True, f"{name:14s} snapshots"
    for n in (0, 2, ranks):
        d = run_dir(base, f"{name}_t{n}")
        fenice.run(case, *sets, "write_interval_time=0.01", "t_end=0.055", "save_interval=1000000000", out_root=d, mpi=n)
        times = np.array([t for t, _ in fenice.read_pvd(study.case_output(d))])
        want = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.055])
        good = len(times) == len(want) and np.allclose(times, want, rtol=0, atol=1e-13)
        ok = ok and good
        line += f" | {n or 'serial'} {'==' if good else 'DIFFERS ' + str(times)}"
    d = run_dir(base, f"{name}_dtmax")
    fenice.run(case, *sets, "dt_max=0.001", "t_end=0.0105", "save_interval=1000000000", "monitor_interval=1", out_root=d)
    mon = fenice.read_monitor(study.case_output(d))
    steps = np.diff(np.asarray(mon["t"]))
    good = np.all(steps <= 0.001 * (1 + 1e-12)) and abs(mon["t"][-1] - 0.0105) < 1e-13
    ok = ok and good
    line += f" | dt_max {'==' if good else 'DIFFERS'}"
    print(line, flush=True)
    return ok


def check_mpi_state(name, case, sets, ranks, base):
    """Serial against 2..ranks: the final state identical."""
    serial = run_dir(base, f"{name}_serial")
    fenice.run(case, *sets, "save_interval=1000000000", out_root=serial, dump=serial / "final.npz")
    ref = final(serial / "final.npz")
    ok, line = True, f"{name:14s} serial"
    for n in range(2, ranks + 1):
        d = run_dir(base, f"{name}_np{n}")
        fenice.run(case, *sets, "save_interval=1000000000", out_root=d, mpi=n, dump=d / "final.npz")
        same = np.array_equal(final(d / "final.npz")["U"], ref["U"])
        ok = ok and same
        line += f" | np{n} {'==' if same else 'DIFFERS'}"
    print(line, flush=True)
    return ok


def main():
    ap = study.parser(__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="MPI (2..7 ranks) against serial, resume, no side effect on the state")
    args = ap.parse_args()
    if not args.check:
        ap.error("choose --check")
    raise SystemExit(check(7))


if __name__ == "__main__":
    main()
