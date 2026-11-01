"""Head-to-head with Trixi.jl on two of its own examples: accuracy and cost per degree of freedom.

    python cases/validation/benchmarks/compare_trixi.py --setup          # once: Julia + Trixi.jl in an environment (Linux / WSL)
    python cases/validation/benchmarks/compare_trixi.py                  # everything, threads 1 and 44
    python cases/validation/benchmarks/compare_trixi.py --threads 44 --tgv-n 32 --tgv-levels 3     # a quick pass
    python cases/validation/benchmarks/compare_trixi.py --fenice-only    # only the Fenice side (any OS)

Two cases, each with the SAME problem in both codes (the Trixi.jl examples tree_2d_dgsem/elixir_euler_vortex.jl and
tree_3d_dgsem/elixir_navierstokes_taylor_green_vortex.jl, copied in trixi/trixi_case.jl):

1. 2-D isentropic vortex (Shu 1997 as set up in that example: box [-10, 10]^2, amplitude 5, rho = 1, p = 25, velocity
   (1, 1), periodic; one period t = 20, so the exact solution is the initial condition). Fenice: HLLC, MUSCL with MC,
   SSP-RK2, CFL 0.5 (second order, finite volume) on N^2 cells. Trixi.jl: DGSEM polynomial degree 3, local
   Lax-Friedrichs, CarpenterKennedy2N54, CFL 1.1 (high order) on (4 * 2^level)^2 degrees of freedom. The error is the
   root-mean-square density error over the domain at t = 20, against the number of DOFs per direction and against the wall time.
2. 3-D Taylor-Green vortex at Re = 1600, Ma = 0.1 to t = 20. Fenice: the shipped case (low-Mach HLLC, Van Leer, SSP-RK2)
   on N^3 cells. Trixi.jl: DGSEM degree 3, flux differencing with Ranocha's flux, HLLC surface flux, adaptive RDPK3SpFSAL49
   at tolerance 1e-8, on (4 * 2^level)^3 DOFs. Reported: the peak of the dissipation rate -dE_k/dt and its time, against
   the spectral DNS value 0.0127 at t ~ 9 (van Rees et al. 2011), and the wall time.

What it measures and what it does not: Fenice is second order, Trixi.jl is order 4 at degree 3, so at equal DOFs
the smooth vortex error is expected to favour Trixi.jl; the interesting numbers are the error per second of wall time
and the thread scaling. The wall time of Trixi.jl is the second solve of the same problem in one Julia session (the first
is a warm-up that compiles); the wall time of Fenice includes its start-up (milliseconds). The machine should be idle.
Results: results/validation/compare_trixi/ (runs/*.json cached, table.md, compare_trixi.png). --force re-runs.
The Julia side was written from the Trixi.jl examples and is run only on the workstation: if an API detail changed
between versions, the Julia error appears in the run's log (results/validation/compare_trixi/logs/).
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import fenice
from fenice import study

HERE = Path(__file__).resolve().parent
OUT = study.RESULTS / "validation" / "compare_trixi"
TRIXI_DIR = HERE / "trixi"
ENV_DIR = Path(os.environ.get("TRIXI_ENV", Path.home() / ".fenice_trixi_env"))
JULIA_VERSION = "1.10.5"
GAMMA, DNS_PEAK = 1.4, (9.0, 0.0127)
L_TGV = 2.0 * np.pi

VORTEX_TOML = """[case]
name = "Trixi Vortex"
description = "isentropic vortex of the Trixi.jl example elixir_euler_vortex.jl"
order = 1

[mesh]
Lx = 20.0
Ly = 20.0
nx = 64
ny = 64

[thermo]
gamma = 1.4

[physics]
use_viscous = false
use_bodyforce = false

[boundaries]
bc_left = "periodic"
bc_right = "periodic"
bc_bottom = "periodic"
bc_top = "periodic"

[numerics]
riemann_solver = "HLLC"
use_muscl = true
muscl_limiter = "MC"
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.5

[run]
t_end = 20.0
save_interval = 1000000000

[initial]
state = {{ rho = "{rho}", u = "{u}", v = "{v}", w = 0.0, p = "{p}" }}
"""


def vortex_exact(x, y):
    """Density of the example's vortex (centre at the origin of [-10, 10]^2; x, y in [0, 20] are shifted by 10)."""
    X, Y = x - 10.0, y - 10.0
    du = 5.0 / (2.0 * np.pi) * np.exp(0.5 * (1.0 - (X ** 2 + Y ** 2)))
    dtemp = -(GAMMA - 1.0) / (2.0 * GAMMA * 25.0) * du ** 2
    return (1.0 + dtemp) ** (1.0 / (GAMMA - 1.0))


def vortex_case_file():
    r2 = "((x-10)^2+(y-10)^2)"
    du = f"(5/(2*pi)*exp(0.5*(1-{r2})))"
    dt = f"(1-0.4/70*{du}^2)"
    case = OUT / "cases" / "trixi_vortex" / "case.toml"
    case.parent.mkdir(parents=True, exist_ok=True)
    case.write_text(VORTEX_TOML.format(rho=f"{dt}^(1/0.4)", p=f"25*{dt}^(1.4/0.4)", u=f"1-{du}*(y-10)",
                                       v=f"1+{du}*(x-10)"), encoding="utf-8")
    return case


def cached(name, force, fn):
    f = OUT / "runs" / f"{name}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    res = fn()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res, indent=1))
    return res


def fenice_vortex(n, threads):
    case = vortex_case_file()
    out = OUT / "fenice" / f"vortex_n{n}_t{threads}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    t0 = time.perf_counter()
    fenice.run(case, f"nx={n}", f"ny={n}", out_root=out, dump=out / "final.npz", threads=threads)
    wall = time.perf_counter() - t0
    rho = fenice.read_dump(out / "final.npz")["U"][:, 0].reshape(n, n)
    xc = (np.arange(n) + 0.5) * 20.0 / n
    X, Y = np.meshgrid(xc, xc)
    e = rho - vortex_exact(X, Y)
    return dict(code="fenice", case="vortex", n=n, threads=threads, wall=wall, l2_rho=float(np.sqrt(np.mean(e ** 2))),
                linf_rho=float(np.abs(e).max()))


def fenice_tgv(n, threads):
    out = OUT / "fenice" / f"tgv_n{n}_t{threads}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    t0 = time.perf_counter()
    fenice.run("Taylor-Green Vortex 3D", f"nx={n}", f"ny={n}", f"nz={n}", "monitor_interval=5",
               "save_interval=1000000000", out_root=out, dump=out / "final.npz", threads=threads)
    wall = time.perf_counter() - t0
    m = fenice.read_monitor(next(p.parent for p in out.glob("*/*_monitor.csv")))
    return tgv_summary("fenice", n, threads, wall, m["t"], m["kinetic_energy"])


def tgv_summary(code, n, threads, wall, t, ke):
    eps = -np.gradient(ke, t)
    k = int(np.argmax(eps))
    return dict(code=code, case="tgv", n=n, threads=threads, wall=wall, ke0=float(ke[0]), eps_peak=float(eps[k]),
                t_peak=float(t[k]), peak_error=float(eps[k] / DNS_PEAK[1] - 1.0))


def parse_analysis(path):
    """time and kinetic energy (integral) from Trixi's analysis.dat; the first line starting with '#' names the columns."""
    lines = Path(path).read_text().splitlines()
    head = next(l for l in lines if l.lstrip().startswith("#"))
    names = head.lstrip("# ").split()
    if "time" not in names or "energy_kinetic" not in names:
        raise RuntimeError(f"unexpected columns in {path}: {names}")
    rows = np.array([[float(v) for v in l.split()] for l in lines if l.strip() and not l.lstrip().startswith("#")])
    return rows[:, names.index("time")], rows[:, names.index("energy_kinetic")] / L_TGV ** 3


def trixi_run(kind, level, threads):
    out = OUT / "trixi" / f"{kind}_l{level}_t{threads}"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    julia = shutil.which("julia") or str(ENV_DIR / "julia" / "bin" / "julia")
    cmd = [julia, f"--project={ENV_DIR}", f"--threads={threads}", str(TRIXI_DIR / "trixi_case.jl"), kind, str(level), "3",
           str(out)]
    log = OUT / "logs"
    log.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    (log / f"{kind}_l{level}_t{threads}.log").write_text(p.stdout + "\n--- stderr\n" + p.stderr, encoding="utf-8")
    line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith("RESULT ")), None)
    if p.returncode != 0 or line is None:
        raise RuntimeError(f"Trixi.jl run failed ({' '.join(cmd[-4:])}); see {log}/{kind}_l{level}_t{threads}.log")
    r = json.loads(line[len("RESULT "):])
    if kind == "vortex":
        return dict(code="trixi", case="vortex", n=r["nx"], threads=threads, wall=r["wall"], l2_rho=r["l2_rho"],
                    linf_rho=r["linf_rho"], trixi=r["trixi"])
    t, ke = parse_analysis(r["analysis"])
    s = tgv_summary("trixi", r["nx"], threads, r["wall"], t, ke)
    s["trixi"] = r["trixi"]
    return s


def setup():
    """Julia and Trixi.jl in $TRIXI_ENV (default ~/.fenice_trixi_env). Needs network, Linux or WSL."""
    if platform.system() != "Linux":
        raise SystemExit("--setup is for Linux / WSL")
    ENV_DIR.mkdir(parents=True, exist_ok=True)
    if not shutil.which("julia") and not (ENV_DIR / "julia" / "bin" / "julia").exists():
        short = ".".join(JULIA_VERSION.split(".")[:2])
        url = f"https://julialang-s3.julialang.org/bin/linux/x64/{short}/julia-{JULIA_VERSION}-linux-x86_64.tar.gz"
        tar = ENV_DIR / "julia.tar.gz"
        subprocess.run(["curl", "-fL", "-o", str(tar), url], check=True)
        (ENV_DIR / "julia").mkdir(exist_ok=True)
        subprocess.run(["tar", "xzf", str(tar), "-C", str(ENV_DIR / "julia"), "--strip-components=1"], check=True)
        tar.unlink()
    julia = shutil.which("julia") or str(ENV_DIR / "julia" / "bin" / "julia")
    subprocess.run([julia, f"--project={ENV_DIR}", "-e",
                    'using Pkg; Pkg.add(["Trixi", "OrdinaryDiffEqLowStorageRK"]); Pkg.precompile(); Pkg.status()'],
                   check=True)


def table(res):
    rows = ["### 2-D isentropic vortex, one period (L2 density error)", "",
            "| code | DOFs/direction | threads | wall [s] | L2 error | Linf error |", "|---|---|---|---|---|---|"]
    for r in sorted((r for r in res if r["case"] == "vortex"), key=lambda r: (r["code"], r["threads"], r["n"])):
        rows.append(f"| {r['code']} | {r['n']} | {r['threads']} | {r['wall']:.2f} | {r['l2_rho']:.3e} | {r['linf_rho']:.3e} |")
    rows += ["", "### 3-D Taylor-Green, Re = 1600 (DNS peak 0.0127 at t ~ 9)", "",
             "| code | DOFs/direction | threads | wall [s] | E_k(0) | peak dissipation | t at peak | error of the peak |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in res if r["case"] == "tgv"), key=lambda r: (r["code"], r["threads"], r["n"])):
        rows.append(f"| {r['code']} | {r['n']}^3 | {r['threads']} | {r['wall']:.1f} | {r['ke0']:.4f} | {r['eps_peak']:.5f} "
                    f"| {r['t_peak']:.2f} | {100 * r['peak_error']:+.1f} % |")
    return "\n".join(rows)


def figure(res):
    plt = study.use_style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    style = {"fenice": ("C0", "o"), "trixi": ("C3", "s")}
    for code in ("fenice", "trixi"):
        for th in sorted({r["threads"] for r in res}):
            d = sorted((r for r in res if r["case"] == "vortex" and r["code"] == code and r["threads"] == th),
                       key=lambda r: r["n"])
            if not d:
                continue
            c, m = style[code]
            ls = "-" if th == max(r["threads"] for r in res) else ":"
            lab = f"{'Fenice (order 2)' if code == 'fenice' else 'Trixi.jl (DG, degree 3)'}, {th} thr."
            ax[0].loglog([r["n"] for r in d], [r["l2_rho"] for r in d], ls, color=c, marker=m, label=lab)
            ax[1].loglog([r["wall"] for r in d], [r["l2_rho"] for r in d], ls, color=c, marker=m, label=lab)
    ax[0].set(xlabel="degrees of freedom per direction", ylabel="L2 density error")
    ax[1].set(xlabel="wall time [s]", ylabel="L2 density error")
    ax[0].legend()
    ax[0].set_title("Isentropic vortex, one period")
    ax[1].set_title("Cost against error")
    fig.tight_layout()
    fig.savefig(OUT / "compare_trixi.png", dpi=170)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--setup", action="store_true", help="install Julia and Trixi.jl (Linux / WSL), then exit")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 44])
    ap.add_argument("--vortex-n", type=int, nargs="*", default=[32, 64, 128, 256])
    ap.add_argument("--vortex-levels", type=int, nargs="*", default=[3, 4, 5, 6], help="Trixi.jl levels (DOFs 4 * 2^level)")
    ap.add_argument("--tgv-n", type=int, nargs="*", default=[32, 64])
    ap.add_argument("--tgv-levels", type=int, nargs="*", default=[3, 4])
    ap.add_argument("--fenice-only", action="store_true")
    ap.add_argument("--trixi-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.setup:
        setup()
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    res, failed = [], []
    jobs = []
    for th in a.threads:
        if not a.trixi_only:
            jobs += [(f"fenice_vortex_n{n}_t{th}", lambda n=n, th=th: fenice_vortex(n, th)) for n in a.vortex_n]
            jobs += [(f"fenice_tgv_n{n}_t{th}", lambda n=n, th=th: fenice_tgv(n, th)) for n in a.tgv_n]
        if not a.fenice_only:
            jobs += [(f"trixi_vortex_l{lv}_t{th}", lambda lv=lv, th=th: trixi_run("vortex", lv, th)) for lv in a.vortex_levels]
            jobs += [(f"trixi_tgv_l{lv}_t{th}", lambda lv=lv, th=th: trixi_run("tgv", lv, th)) for lv in a.tgv_levels]
    for name, fn in jobs:
        try:
            print(f"[RUN] {name}", flush=True)
            res.append(cached(name, a.force, fn))
        except Exception as e:   # one failed run must not lose the others
            failed.append((name, str(e)))
            print(f"[FAILED] {name}: {e}", flush=True)
    if res:
        md = table(res)
        (OUT / "table.md").write_text(md + "\n", encoding="utf-8")
        print("\n" + md)
        if any(r["case"] == "vortex" for r in res):
            try:
                figure(res)
            except ImportError as e:   # the tables are already written
                print("no figure: " + str(e))
    for name, e in failed:
        print(f"FAILED {name}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
