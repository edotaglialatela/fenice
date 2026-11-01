"""Known-answer checks of the analysis layer: fenice.uq, fenice.modal, fenice.spectra.

    python cases/validation/check_analysis.py

Every check compares with an exact or analytic answer; the bands are fixed here, before the first run.
  uq        Ishigami function: first-order and total Sobol indices against the analytic values (|error| <= 0.05 with 20000
            random points, <= 0.03 with 8192 Sobol points when SciPy is installed); Latin hypercube strata; a real sweep of
            four Sod runs, repeated: the second call runs nothing (the cache) and returns the same numbers.
  modal     DMD of a travelling wave and of a damped oscillation (two spatial structures in quadrature): eigenvalues exact to 1e-10; POD of two weighted
            orthogonal modes: energies exact to 1e-12 relative, modes weight-orthonormal to 1e-12; a mode written as .vtr
            and read back (float32 text) to 1e-6.
  spectra   a sine wave: one PSD peak at its frequency (error < df / 20) with power sum(P) df = A^2 / 2 to 1e-3; Parseval
            for the PSD of one segment to roundoff; the Taylor-Green initial field has all its energy in one shell (|k| = sqrt(2) -> shell 1,
            E = 1/4 in 2-D; sqrt(3) -> shell 2, E = 1/8 in 3-D; the others < 1e-14) and sum E = mean energy to 1e-12 (2-D and 3-D); the solenoidal part of a gradient
            field and the dilatational part of a curl field vanish below 1e-12 of the field.
Exit code 0 when everything holds.
"""
import itertools
import sys
import tempfile
from pathlib import Path

import numpy as np

import fenice
from fenice import modal, spectra, uq

failures = []


def check(ok, what):
    print(f"[{' OK ' if ok else 'FAIL'}] {what}")
    if not ok:
        failures.append(what)


# ---------------------------------------------------------------- uq
A_, B_ = 7.0, 0.1
ISHIGAMI_S1 = np.array([0.3139, 0.4424, 0.0])
ISHIGAMI_ST = np.array([0.5576, 0.4424, 0.2437])


def ishigami(p):
    x1, x2, x3 = p["x1"], p["x2"], p["x3"]
    return np.sin(x1) + A_ * np.sin(x2) ** 2 + B_ * x3 ** 4 * np.sin(x1)


def test_uq():
    bounds = {n: (-np.pi, np.pi) for n in ("x1", "x2", "x3")}
    n = 20000
    A, B, AB, pts = uq.saltelli(n, bounds, uq.uniform_sampler, seed=3)
    S1, ST = uq.sobol_indices([ishigami(p) for p in pts], n, 3)
    e1, et = np.abs(S1 - ISHIGAMI_S1).max(), np.abs(ST - ISHIGAMI_ST).max()
    check(e1 <= 0.05 and et <= 0.05, f"Ishigami, random points: S1 error {e1:.3f}, ST error {et:.3f} (band 0.05)")
    try:
        n = 8192
        A, B, AB, pts = uq.saltelli(n, bounds, uq.sobol_sampler, seed=3)
        S1, ST = uq.sobol_indices([ishigami(p) for p in pts], n, 3)
        e1, et = np.abs(S1 - ISHIGAMI_S1).max(), np.abs(ST - ISHIGAMI_ST).max()
        check(e1 <= 0.03 and et <= 0.03, f"Ishigami, Sobol points: S1 error {e1:.3f}, ST error {et:.3f} (band 0.03)")
    except ImportError:
        print("[SKIP] Ishigami with Sobol points: SciPy not installed")
    u = uq.lhs_sampler(50, 3, seed=1)
    strata = all(sorted(np.floor(u[:, j] * 50).astype(int)) == list(range(50)) for j in range(3))
    check(strata, "Latin hypercube: one point per stratum in every coordinate")
    check(len(uq.grid(a=[1, 2, 3], b=[4, 5])) == 6 and uq.grid(a=[1], b=[2])[0] == {"a": 1, "b": 2}, "grid: Cartesian product")
    # a sweep through the solver, cached
    import math
    from fenice.exact import riemann
    calls = {"n": 0}

    def counting(case, *sets, **kw):
        calls["n"] += 1
        return fenice.run(case, *sets, **kw)

    def l1(run_dir, point):
        nx = int(point["nx"])
        d = fenice.read_dump(run_dir / "final.npz")
        rho = d["U"][:, 0].reshape(4, nx)[2]
        re, _, _ = riemann((np.arange(nx) + 0.5) / nx, float(d["t"]), gamma=1.4)
        return {"l1": float(np.mean(np.abs(rho - re)))}

    with tempfile.TemporaryDirectory() as tmp:
        pts = uq.grid(nx=[50, 100], cfl=[0.3, 0.5])
        base = ("Sod Shock Tube", )
        kw = dict(base=("ny=4", "t_end=0.1", "save_interval=1000000000"), root=tmp, workers=2, runner=counting)
        r1 = uq.run_sweep(*base, pts, l1, **kw)
        n1 = calls["n"]
        r2 = uq.run_sweep(*base, pts, l1, **kw)
        check(n1 == 4 and calls["n"] == 4, f"sweep: four runs, repeated sweep runs none ({n1} then {calls['n'] - n1} new)")
        check(all(b["cached"] for b in r2) and [a["qoi"] for a in r1] == [b["qoi"] for b in r2], "sweep: cached records equal")
        fine, coarse = [r["qoi"]["l1"] for r in r1 if r["point"]["nx"] == 100], [r["qoi"]["l1"] for r in r1 if r["point"]["nx"] == 50]
        check(all(f < c for f, c in zip(fine, coarse)) and all(math.isfinite(x) for x in fine + coarse),
              "sweep: the L1 error falls from 50 to 100 cells at every CFL")


# ---------------------------------------------------------------- modal
def test_modal():
    n, m, dt = 128, 60, 0.05
    x = np.arange(n) * 2 * np.pi / n
    t = np.arange(m) * dt
    k, om = 3.0, 4.0
    X = np.sin(k * x[:, None] - om * t[None, :])
    d = modal.dmd(X, dt, rank=2)
    err = np.abs(np.sort_complex(d["omega"]) - np.sort_complex(np.array([-1j * om, 1j * om]))).max()
    check(err < 1e-10 and np.abs(np.abs(d["eigenvalues"]) - 1).max() < 1e-10,
          f"DMD, travelling wave: omega error {err:.1e}, |lambda| = 1 to {np.abs(np.abs(d['eigenvalues']) - 1).max():.1e}")
    sigma, om2 = 0.3, 5.0
    # a damped oscillation needs two spatial structures in quadrature (one shape times cos(om t) has rank 1: DMD cannot see
    # its two exponents without delay embedding)
    X = np.exp(-sigma * t)[None, :] * (np.sin(2 * x)[:, None] * np.cos(om2 * t)[None, :]
                                       + np.sin(3 * x)[:, None] * np.sin(om2 * t)[None, :])
    d = modal.dmd(X, dt, rank=2)
    want = np.sort_complex(np.array([-sigma - 1j * om2, -sigma + 1j * om2]))
    err = np.abs(np.sort_complex(d["omega"]) - want).max()
    check(err < 1e-10, f"DMD, damped oscillation: omega error {err:.1e}; growth {d['growth'][0]:.6f} vs {-sigma}")
    # one scalar record: the delay embedding turns one shape times cos(om t) into the rank-2 problem DMD can solve
    ts = np.arange(400) * 0.04
    y = np.exp(-0.2 * ts) * np.cos(6.0 * ts) + 0.5 * np.exp(-0.05 * ts) * np.cos(15.0 * ts + 0.7)
    H = modal.delay_embed(y, 30)
    check(H.shape == (30, 371) and H[3, 5] == y[8] and H[29, 370] == y[399], "delay_embed: Hankel layout, H[i, j] = y[i + j]")
    d = modal.dmd(H, 0.04, rank=4)
    want = np.sort_complex(np.array([-0.2 - 6j, -0.2 + 6j, -0.05 - 15j, -0.05 + 15j]))
    err = np.abs(np.sort_complex(d["omega"]) - want).max()
    check(err < 1e-8, f"DMD of a delay-embedded scalar record: two damped oscillations, omega error {err:.1e}")
    # POD of two weighted-orthogonal modes with orthogonal temporal coefficients
    rng = np.random.default_rng(0)
    npts = 200
    w = rng.random(npts) + 0.5
    G = rng.standard_normal((npts, 2))
    Wn = w / w.sum()
    # weighted Gram-Schmidt
    q1 = G[:, 0] / np.sqrt(np.sum(Wn * G[:, 0] ** 2))
    g2 = G[:, 1] - np.sum(Wn * G[:, 1] * q1) * q1
    q2 = g2 / np.sqrt(np.sum(Wn * g2 ** 2))
    m2 = 80
    tt = np.arange(m2)
    psi1, psi2 = np.sqrt(2) * np.cos(2 * np.pi * 3 * tt / m2), np.sqrt(2) * np.sin(2 * np.pi * 5 * tt / m2)   # <psi^2> = 1, orthogonal
    s1, s2 = 3.0, 1.5
    X = 7.0 + s1 * np.outer(q1, psi1) + s2 * np.outer(q2, psi2)   # a constant (the mean) is removed by pod
    p = modal.pod(X, w, rank=2)
    err = np.abs(p["energies"] - np.array([s1 ** 2, s2 ** 2])).max() / s1 ** 2
    gram = (p["modes"] * Wn[:, None]).T @ p["modes"]
    check(err < 1e-12 and np.abs(gram - np.eye(2)).max() < 1e-12,
          f"POD, two modes: energy error {err:.1e}, orthonormality {np.abs(gram - np.eye(2)).max():.1e}")
    rec = p["mean"][:, None] + p["modes"] @ p["coefficients"]
    check(np.abs(rec - X).max() < 1e-10, f"POD: mean + modes x coefficients rebuilds the data ({np.abs(rec - X).max():.1e})")
    with tempfile.TemporaryDirectory() as tmp:
        ny, nx = 6, 8
        xn, yn = np.linspace(0, 2, nx + 1), np.linspace(0, 1, ny + 1)
        field = np.random.default_rng(1).standard_normal((ny, nx))
        path = modal.export_modes_vtr(Path(tmp) / "m.vtr", field, xn, yn, "mode")
        f = fenice.read_vtr(path)
        check(np.abs(f["mode"] - field).max() < 1e-6 * np.abs(field).max(), "mode export: .vtr written and read back")


# ---------------------------------------------------------------- spectra
def test_spectra():
    dt, n, f0, amp = 0.01, 4096, 7.0, 1.7
    t = np.arange(n) * dt
    y = amp * np.sin(2 * np.pi * f0 * t + 0.4) + 3.0   # a constant offset: removed by the detrend
    f, P, df = spectra.psd(y, dt, nperseg=1024, overlap=0.5)
    fp, pw = spectra.peaks(f, P, 1)[0]
    check(abs(fp - f0) < df / 20, f"PSD of a sine: peak {fp:.4f} for {f0} (df = {df:.4f})")
    check(abs(np.sum(P) * df / (amp ** 2 / 2) - 1) < 1e-3, f"PSD of a sine: power {np.sum(P) * df:.4f} vs A^2/2 = {amp ** 2 / 2:.4f}")
    seg = y[:1024] - np.polyval(np.polyfit(np.arange(1024.0), y[:1024], 1), np.arange(1024.0))
    w = np.hanning(1024)
    f1, P1, df1 = spectra.psd(y[:1024], dt)
    par = abs(np.sum(P1) * df1 / (np.sum((w * seg) ** 2) / np.sum(w * w)) - 1)
    check(par < 1e-12, f"PSD Parseval, one segment: {par:.1e}")
    tu, yu, ratio = spectra.resample_uniform(t[::2] + 0.001 * np.sin(t[::2]), y[::2])
    check(ratio < 1.01 and len(tu) > 2000, f"resample_uniform: step ratio {ratio:.3f}")
    # Taylor-Green initial field
    for N, dim in ((64, 2), (32, 3)):
        x = (np.arange(N) + 0.5) * 2 * np.pi / N
        if dim == 2:
            Y, X = np.meshgrid(x, x, indexing="ij")
            u, v, w3 = np.sin(X) * np.cos(Y), -np.cos(X) * np.sin(Y), None
        else:
            Z, Y, X = np.meshgrid(x, x, x, indexing="ij")
            u, v, w3 = np.sin(X) * np.cos(Y) * np.cos(Z), -np.cos(X) * np.sin(Y) * np.cos(Z), np.zeros((N, N, N))
        s = spectra.energy_spectrum(u, v, w3)
        E = s["E"]
        want, shell = (0.25, 1) if dim == 2 else (0.125, 2)   # |k| = sqrt(2) -> shell 1, sqrt(3) -> shell 2
        rest = np.abs(np.delete(E, shell)).max()
        check(abs(E[shell] - want) < 1e-12 and rest < 1e-14 and abs(E.sum() - s["mean_energy"]) < 1e-12 * s["mean_energy"],
              f"Taylor-Green {dim}-D: E({shell}) = {E[shell]:.6f} (exact {want}), other shells {rest:.1e}, "
              f"Parseval {abs(E.sum() - s['mean_energy']):.1e}")
    # Parseval for a random field, and the density-weighted option
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal((2, 48, 48))
    rho = 1.0 + 0.3 * rng.random((48, 48))
    s = spectra.energy_spectrum(a, b)
    sr = spectra.energy_spectrum(a, b, rho=rho)
    check(abs(s["E"].sum() / (0.5 * np.mean(a * a + b * b)) - 1) < 1e-12
          and abs(sr["E"].sum() / (0.5 * np.mean(rho * (a * a + b * b))) - 1) < 1e-12, "Parseval, random field, plain and density-weighted")
    # Helmholtz
    N = 64
    x = np.arange(N) * 2 * np.pi / N
    Y, X = np.meshgrid(x, x, indexing="ij")
    phi = np.sin(X) * np.cos(2 * Y) + 0.5 * np.cos(3 * X + Y)
    gx = np.cos(X) * np.cos(2 * Y) - 1.5 * np.sin(3 * X + Y)
    gy = -2 * np.sin(X) * np.sin(2 * Y) - 0.5 * np.sin(3 * X + Y)
    sol, dil = spectra.helmholtz(gx, gy)
    s0 = max(np.abs(sol[0]).max(), np.abs(sol[1]).max()) / max(np.abs(gx).max(), np.abs(gy).max())
    psi = np.sin(2 * X) * np.sin(Y) + 0.3 * np.cos(X - 2 * Y)
    cx = 2 * np.sin(2 * X) * np.cos(Y) * 0 + np.sin(2 * X) * np.cos(Y) + 0.3 * 2 * np.sin(X - 2 * Y)   # d psi / dy
    cy = -(2 * np.cos(2 * X) * np.sin(Y) - 0.3 * np.sin(X - 2 * Y))                                      # - d psi / dx
    sol2, dil2 = spectra.helmholtz(cx, cy)
    d0 = max(np.abs(dil2[0]).max(), np.abs(dil2[1]).max()) / max(np.abs(cx).max(), np.abs(cy).max())
    check(s0 < 1e-12 and d0 < 1e-12, f"Helmholtz: solenoidal part of a gradient {s0:.1e}, dilatational part of a curl {d0:.1e}")


if __name__ == "__main__":
    test_uq()
    test_modal()
    test_spectra()
    print("ALL PASSED" if not failures else f"FAILED: {failures}")
    sys.exit(1 if failures else 0)
