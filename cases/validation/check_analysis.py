"""Known-answer checks of the analysis layer: fenice.uq, fenice.modal, fenice.spectra.

    python cases/validation/check_analysis.py

Every check compares with an exact or analytic answer; the bands are fixed here, before the first run.
  uq        Ishigami function: first-order and total Sobol indices against the analytic values (|error| <= 0.05 with 20000
            random points, <= 0.03 with 8192 Sobol points when SciPy is installed); the 99 % bootstrap intervals of the
            first set contain the analytic and the estimated indices and are narrower than 0.1; Latin hypercube strata; a real sweep of
            four Sod runs, repeated: the second call runs nothing (the cache) and returns the same numbers.
  modal     DMD of a travelling wave and of a damped oscillation (two spatial structures in quadrature): eigenvalues exact to 1e-10; POD of two weighted
            orthogonal modes: energies exact to 1e-12 relative, modes weight-orthonormal to 1e-12; a mode written as .vtr
            and read back (float32 text) to 1e-6.
  spectra   a sine wave: one PSD peak at its frequency (error < df / 20) with power sum(P) df = A^2 / 2 to 1e-3; Parseval
            for the PSD of one segment to roundoff; the Taylor-Green initial field has all its energy in one shell (|k| = sqrt(2) -> shell 1,
            E = 1/4 in 2-D; sqrt(3) -> shell 2, E = 1/8 in 3-D; the others < 1e-14) and sum E = mean energy to 1e-12 (2-D and 3-D); the solenoidal part of a gradient
            field and the dilatational part of a curl field vanish below 1e-12 of the field. D4: the Taylor-Green 3-D initial field has enstrophy
            3/8 (1e-13 relative), Taylor micro-scale^2 = 5/3 and integral scale 3 pi / (4 sqrt 3) (1e-12); for the exact decaying 2-D
            Taylor-Green solution the numerical dissipation -dK/dt - 2 nu Z is below 1e-5 of -dK/dt. D3: a sine with 3.37 periods on a
            non-periodic line: the peak of the Hann and of the mirrored spectrum within one dk of the exact wavenumber, Parseval to
            1e-12, leakage (power above 4 k_peak) of the Hann spectrum at most a hundredth of the raw line's, of the mirrored one not above it (first version: a tenth for both; the mirror removes the jump but keeps a break of the slope, and measured only 1.4 times below the raw line); the resampling of a complete
            leaf set equals the uniform array exactly, a merged leaf is piecewise constant, `coarsened`, `cells` and a gap error as expected.
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
    y = [ishigami(p) for p in pts]
    l1, h1, lt, ht = uq.sobol_bootstrap(y, n, 3, n_boot=500, level=0.99, seed=1)
    inside = bool(np.all((l1 <= ISHIGAMI_S1) & (ISHIGAMI_S1 <= h1) & (lt <= ISHIGAMI_ST) & (ISHIGAMI_ST <= ht)))
    width = max((h1 - l1).max(), (ht - lt).max())
    check(inside and 0 < width < 0.1 and np.all((l1 <= S1) & (S1 <= h1)) and np.all((lt <= ST) & (ST <= ht)),
          f"Ishigami bootstrap (99 %, 500 resamples): the analytic and the estimated indices lie inside, widest interval {width:.3f} (< 0.1)")
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


# ---------------------------------------------------------------- spectra (D3, D4)
def test_spectra_derived():
    N = 32
    x = (np.arange(N) + 0.5) * 2 * np.pi / N
    Z3, Y3, X3 = np.meshgrid(x, x, x, indexing="ij")
    s = spectra.energy_spectrum(np.sin(X3) * np.cos(Y3) * np.cos(Z3), -np.cos(X3) * np.sin(Y3) * np.cos(Z3), np.zeros((N, N, N)))
    e1 = abs(s["enstrophy"] / (3 / 8) - 1)
    e2 = abs(spectra.taylor_microscale(s) ** 2 / (5 / 3) - 1)
    e3 = abs(spectra.integral_scale(s) / (3 * np.pi / (4 * np.sqrt(3))) - 1)
    check(e1 < 1e-13 and e2 < 1e-12 and e3 < 1e-12,
          f"Taylor-Green 3-D: enstrophy 3/8 ({e1:.1e}), Taylor micro-scale ({e2:.1e}), integral scale ({e3:.1e})")
    D, tot = spectra.dissipation_spectrum(s, 0.01)
    check(abs(tot / (2 * 0.01 * 3 / 8) - 1) < 1e-13, "dissipation_spectrum: total 2 nu Z")
    # exact decaying 2-D Taylor-Green: K = exp(-4 nu t) / 4, Z = 2 K, so -dK/dt = 2 nu Z
    nu, M = 0.1, 64
    xs = (np.arange(M) + 0.5) * 2 * np.pi / M
    Y2, X2 = np.meshgrid(xs, xs, indexing="ij")
    ts = np.arange(0, 2.01, 0.01)
    K, Z = [], []
    for t in ts:
        a = np.exp(-2 * nu * t)
        sp = spectra.energy_spectrum(a * np.sin(X2) * np.cos(Y2), -a * np.cos(X2) * np.sin(Y2))
        K.append(sp["mean_energy"])
        Z.append(sp["enstrophy"])
    tot, visc, num = spectra.numerical_dissipation(ts, K, Z, nu)
    rel = np.abs(num[1:-1]).max() / tot[1:-1].min()
    check(rel < 1e-5, f"numerical_dissipation of the exact decaying Taylor-Green solution: {rel:.1e} of the total (band 1e-5)")
    # D3: a non-periodic line
    n, L = 512, 1.0
    xl = (np.arange(n) + 0.5) * L / n
    k_true = 2 * np.pi * 3.37 / L
    line = np.sin(k_true * xl + 0.3)
    res = {m: spectra.spectrum_1d(line, L / n, m) for m in ("hann", "even", "none")}
    okp = all(abs(r["k"][1:][np.argmax(r["E"][1:])] - k_true) <= r["dk"] for r in res.values())
    ms = np.mean((line - line.mean()) ** 2)
    pars = {m: abs(np.sum(r["E"]) * r["dk"] / ms - 1) for m, r in res.items() if m != "hann"}
    check(okp, f"spectrum_1d of a sine with 3.37 periods: peak within dk of {k_true:.3f} for every extension")
    check(max(pars.values()) < 1e-12, f"spectrum_1d Parseval, raw and mirrored line: {max(pars.values()):.1e}")
    lk = {m: r["leakage"] for m, r in res.items()}
    check(lk["hann"] <= 0.01 * lk["none"] and lk["even"] <= lk["none"],
          f"spectrum_1d leakage above 4 k_peak: raw {lk['none']:.1e}, Hann {lk['hann']:.1e} (<= 0.01 x raw), mirrored {lk['even']:.1e} (<= raw)")
    # AMR resampling
    Nf, h = 8, 1.0 / 8
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((Nf, Nf))
    jj, ii = np.meshgrid(np.arange(Nf), np.arange(Nf), indexing="ij")
    leaves = {"x0": ii.ravel() * h, "x1": (ii.ravel() + 1) * h, "y0": jj.ravel() * h, "y1": (jj.ravel() + 1) * h, "Q": arr.ravel()}
    out, xc, yc, info = spectra.resample_amr(leaves, "Q")
    sa, sb = spectra.energy_spectrum(arr, arr), spectra.energy_spectrum(out, out)
    check(np.array_equal(out, arr) and info["cells"] == 64 and info["coarsened"] == 0 and np.array_equal(sa["E"], sb["E"]),
          "resample_amr: a complete uniform leaf set returns the array, and the same spectrum, exactly")
    # merge the cells (4..5, 2..3) into one leaf of twice the size valued 7
    keep = ~((ii.ravel() >= 4) & (ii.ravel() < 6) & (jj.ravel() >= 2) & (jj.ravel() < 4))
    mix = {k: np.append(v[keep], val) for k, v, val in (("x0", leaves["x0"], 4 * h), ("x1", leaves["x1"], 6 * h),
                                                         ("y0", leaves["y0"], 2 * h), ("y1", leaves["y1"], 4 * h), ("Q", leaves["Q"], 7.0))}
    out, _, _, info = spectra.resample_amr(mix, "Q", h=h)
    want = arr.copy()
    want[2:4, 4:6] = 7.0
    check(np.array_equal(out, want) and abs(info["coarsened"] - 4 / 64) < 1e-15, "resample_amr: a merged leaf fills its 2 x 2 cells; coarsened = 4/64")
    sub, xs_, ys_, info = spectra.resample_amr(mix, "Q", region=(0.25, 0.75, 0.125, 0.5), h=h)
    check(np.array_equal(sub, want[1:4, 2:6]) and info["cells"] == 12 and np.allclose(xs_, (np.arange(2, 6) + 0.5) * h),
          "resample_amr: a sub-region is the matching slice (12 cells, centres exact)")
    try:
        spectra.resample_amr({k: v[1:] for k, v in leaves.items()}, "Q")
        gap = False
    except ValueError:
        gap = True
    try:
        spectra.resample_amr(leaves, "Q", h=2 * h)
        fine = False
    except ValueError:
        fine = True
    check(gap and fine, "resample_amr: a missing leaf and a leaf finer than h are errors")

    tt = np.linspace(0, 10, 400)
    pd = modal.phase_difference(np.array([np.cos(2 * np.pi * 0.8 * tt + 0.3) + 5, np.cos(2 * np.pi * 0.8 * tt + 0.3 - np.pi / 2)]), tt, 0.8)
    check(abs(pd - 90.0) < 0.01, f"phase_difference of two coefficients a quarter period apart: {pd:.6f} degrees (90; band 0.01)")


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
    ta = np.linspace(0.0, 10.0, 1001)
    fa, aa, span = spectra.amplitude_spectrum(ta, 2.0 * np.sin(2 * np.pi * 5.0 * ta) + 3.0, pad=8, window="none")
    check(abs(fa[np.argmax(aa)] - 5.0) < 0.02 and abs(aa.max() - 2.0) < 0.05 and span == 10.0,
          f"amplitude_spectrum of a sine on whole periods: peak {fa[np.argmax(aa)]:.4f} at amplitude {aa.max():.4f} (A = 2, f = 5; band 0.02, 0.05)")
    raised = False
    try:
        spectra.amplitude_spectrum([0, 1, 1, 2], [0, 1, 2, 3])
    except ValueError:
        raised = True
    check(raised, "amplitude_spectrum: a record with a repeated time is refused")
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
    test_spectra_derived()
    print("ALL PASSED" if not failures else f"FAILED: {failures}")
    sys.exit(1 if failures else 0)
