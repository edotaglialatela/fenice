"""Locate and run the fenice executable."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repository root (python/fenice/solver.py)
RESULTS = ROOT / "results"
_NAME = "fenice.exe" if os.name == "nt" else "fenice"


def exe():
    """Path of the solver: $FENICE_EXE if set, else build/release/fenice, else the most
    recently built build/*/fenice. Raises FileNotFoundError with the build command."""
    env = os.environ.get("FENICE_EXE")
    if env:
        if not Path(env).is_file():
            raise FileNotFoundError(f"FENICE_EXE points to a missing file: {env}")
        return str(env)
    release = ROOT / "build" / "release" / _NAME
    if release.is_file():
        return str(release)
    built = sorted((ROOT / "build").glob(f"*/{_NAME}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if built:
        return str(built[0])
    raise FileNotFoundError("fenice is not built: run .\\cpp\\build.ps1 (Windows) or "
                            "cmake --preset release && cmake --build --preset release (Linux / macOS)")


def run(case, *sets, out_root=None, mpi=0, dump=None, threads=None, check=True, capture=True):
    """Run `case` (name, folder or case.toml) with `--set` overrides ("nx=200", ...).

    out_root: output_root (results go to out_root/<case>/); mpi: number of MPI ranks
    (> 1: under mpiexec); dump: write the final state to this .npz; threads: OpenMP threads.
    Returns the CompletedProcess (stdout / stderr captured unless capture=False)."""
    cmd = [exe(), "--case", str(case), "--no-logo"]
    if out_root is not None:
        cmd += ["--set", f"output_root={out_root}"]
    if dump is not None:
        cmd += ["--set", f"dump_state={dump}"]
    if threads:
        cmd += ["--threads", str(threads)]
    for s in sets:
        cmd += ["--set", s]
    if mpi and mpi > 1:
        cmd = ["mpiexec", "-n", str(mpi)] + cmd + ["--set", "use_mpi=true", "--set", f"n_cpus={mpi}", "--mpi-child"]
    else:
        cmd += ["--set", "use_mpi=false"]
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)
