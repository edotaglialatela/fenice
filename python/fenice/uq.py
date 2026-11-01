"""Parameter sweeps and sensitivity indices (NumPy only; SciPy is used for Sobol sequences when installed).

    from fenice import uq
    pts = uq.grid(nx=[100, 200], cfl=[0.3, 0.5])                  # a parameter grid ...
    pts = uq.latin_hypercube(64, {"u_in": (0.8, 1.2), "cfl": (0.3, 0.5)})   # ... or a sample set
    rows = uq.run_sweep("Sod Shock Tube", pts, qoi, root=results_dir)       # runs, cached, one record per point
    A, B, AB, points = uq.saltelli(2048, bounds); y = [f(p) for p in points]
    S1, ST = uq.sobol_indices(y, 2048, len(bounds))

A point is a dict {parameter: value}; a value becomes a `--set parameter=value` override. A quantity of interest is a function
qoi(run_dir, point) -> float or dict of floats, evaluated on the finished run (its folder holds final.npz and the case's output
folder). Samplers are functions sampler(n, d, seed) -> array (n, d) in [0, 1); pass your own to `saltelli` / `sample` to use an
external generator. Every run is stored under root/<hash>/ with a summary.json (point, qoi, wall time); a run whose hash (case
file text, overrides, solver executable) is unchanged is not repeated. Runs go through `fenice.run` at below-normal priority,
`workers` at a time with `threads` OpenMP threads each (default: half of the cores in all).
"""
import hashlib
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .solver import ROOT, exe, run


def grid(**axes):
    """Cartesian product of the axes: grid(nx=[100, 200], cfl=[0.3, 0.5]) -> 4 points (dicts)."""
    names = list(axes)
    return [dict(zip(names, vals)) for vals in itertools.product(*(axes[n] for n in names))]


def uniform_sampler(n, d, seed=0):
    """Independent uniform points in [0, 1)^d."""
    return np.random.default_rng(seed).random((n, d))


def lhs_sampler(n, d, seed=0):
    """Latin hypercube: one point in each of n equal strata of every coordinate."""
    rng = np.random.default_rng(seed)
    u = (np.arange(n)[:, None] + rng.random((n, d))) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


def sobol_sampler(n, d, seed=0):
    """Scrambled Sobol sequence (SciPy); n should be a power of two. ImportError without SciPy."""
    try:
        from scipy.stats import qmc
    except ImportError as e:
        raise ImportError("sobol_sampler needs SciPy (pip install scipy); lhs_sampler / uniform_sampler need nothing") from e
    return qmc.Sobol(d, scramble=True, seed=seed).random(n)


def scale(unit, bounds):
    """Map points of [0, 1)^d onto bounds {name: (lo, hi)}; returns the array (n, d) in the order of `bounds`."""
    lo = np.array([b[0] for b in bounds.values()], float)
    hi = np.array([b[1] for b in bounds.values()], float)
    return lo + unit * (hi - lo)


def as_points(values, bounds):
    names = list(bounds)
    return [dict(zip(names, map(float, row))) for row in values]


def sample(n, bounds, sampler=lhs_sampler, seed=0):
    """n points inside bounds {name: (lo, hi)} drawn with `sampler`, as a list of dicts."""
    return as_points(scale(sampler(n, len(bounds), seed), bounds), bounds)


def latin_hypercube(n, bounds, seed=0):
    """n Latin hypercube points inside bounds {name: (lo, hi)} (list of dicts)."""
    return sample(n, bounds, lhs_sampler, seed)


def saltelli(n, bounds, sampler=uniform_sampler, seed=0):
    """The sample matrices of the Saltelli estimator for d = len(bounds) parameters.

    One 2d-column unit sample is split into A and B (n rows each); AB_i is A with column i taken from B. Returns
    (A, B, AB, points): A, B arrays (n, d), AB array (d, n, d), and `points`, the n (d + 2) points to evaluate in the
    order A, B, AB_1, ..., AB_d (the order `sobol_indices` expects its outputs in)."""
    d = len(bounds)
    unit = sampler(n, 2 * d, seed)
    A, B = scale(unit[:, :d], bounds), scale(unit[:, d:], bounds)
    AB = np.repeat(A[None], d, axis=0)
    for i in range(d):
        AB[i][:, i] = B[:, i]
    rows = np.vstack([A, B] + [AB[i] for i in range(d)])
    return A, B, AB, as_points(rows, bounds)


def sobol_indices(y, n, d):
    """First-order and total Sobol indices from the outputs y of the points of `saltelli` (same order, length n (d + 2)).

    First order: Saltelli et al. (2010), mean(f(B) (f(AB_i) - f(A))) / Var; total: Jansen, mean((f(A) - f(AB_i))^2) / (2 Var).
    Returns (S1, ST), arrays of length d."""
    y = np.asarray(y, float)
    if y.shape != (n * (d + 2),):
        raise ValueError(f"expected {n * (d + 2)} outputs, got shape {y.shape}")
    yA, yB = y[:n], y[n:2 * n]
    var = np.var(np.concatenate([yA, yB]))
    S1 = np.empty(d)
    ST = np.empty(d)
    for i in range(d):
        yAB = y[(2 + i) * n:(3 + i) * n]
        S1[i] = np.mean(yB * (yAB - yA)) / var
        ST[i] = 0.5 * np.mean((yA - yAB) ** 2) / var
    return S1, ST


def sobol_bootstrap(y, n, d, n_boot=1000, level=0.95, seed=0):
    """Percentile confidence intervals of the Sobol indices: the n base rows (A, B and the d AB_i rows of one row index
    together) are resampled with replacement `n_boot` times and the indices recomputed on each resample.

    Returns (S1_lo, S1_hi, ST_lo, ST_hi), arrays of length d. The interval holds the sampling error of the estimator only: the
    error of a model that is itself a numerical solution (discretisation) is not in it."""
    y = np.asarray(y, float)
    if y.shape != (n * (d + 2),):
        raise ValueError(f"expected {n * (d + 2)} outputs, got shape {y.shape}")
    cols = y.reshape(d + 2, n)   # row r of every block belongs to the same base sample
    yA, yB, yAB = cols[0], cols[1], cols[2:]
    rng = np.random.default_rng(seed)
    s1 = np.empty((n_boot, d))
    st = np.empty((n_boot, d))
    for b in range(n_boot):
        k = rng.integers(0, n, n)
        a, bb, ab = yA[k], yB[k], yAB[:, k]
        var = np.var(np.concatenate([a, bb]))
        s1[b] = np.mean(bb * (ab - a), axis=1) / var
        st[b] = 0.5 * np.mean((a - ab) ** 2, axis=1) / var
    q = [100 * (1 - level) / 2, 100 * (1 + level) / 2]
    s1_lo, s1_hi = np.percentile(s1, q, axis=0)
    st_lo, st_hi = np.percentile(st, q, axis=0)
    return s1_lo, s1_hi, st_lo, st_hi


def _case_text(case):
    p = Path(str(case))
    if p.is_dir():
        p = p / "case.toml"
    if p.is_file():
        return p.read_text()
    f = ROOT / "cases"
    for c in f.glob("*/case.toml"):   # a case given by its name: find the file that declares it
        if f'"{case}"' in c.read_text().split("[mesh]")[0]:
            return c.read_text()
    return str(case)


_EXE_HASH = {}


def run_hash(case, sets):
    """Fingerprint of a run: case file text, the overrides and the solver executable."""
    path = exe()
    if path not in _EXE_HASH:
        _EXE_HASH[path] = hashlib.sha1(Path(path).read_bytes()).hexdigest()
    h = hashlib.sha1()
    for part in (_case_text(case), "\n".join(sets), _EXE_HASH[path]):
        h.update(part.encode())
    return h.hexdigest()[:16]


def run_sweep(case, points, qoi, *, base=(), root, workers=None, threads=1, runner=run):
    """Run `case` at every point (with the `base` overrides), evaluate `qoi` on each finished run.

    Returns one record per point, in order: {"point", "qoi" (dict), "hash", "wall" [s], "cached"}. `root` is the folder
    holding one sub-folder per run. `runner` replaces fenice.run (tests count calls with it)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) // 2 // max(1, threads))

    def one(point):
        sets = list(base) + [f"{k}={v!r}" if isinstance(v, float) else f"{k}={v}" for k, v in point.items()]
        key = run_hash(case, sets)
        d = root / key
        summary = d / "summary.json"
        if summary.exists():
            rec = json.loads(summary.read_text())
            rec["cached"] = True
            return rec
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        runner(case, *sets, out_root=d, dump=d / "final.npz", threads=threads, polite=True)
        q = qoi(d, point)
        q = dict(q) if isinstance(q, dict) else {"value": q}
        rec = {"point": point, "qoi": {k: float(v) for k, v in q.items()}, "hash": key, "wall": time.time() - t0}
        tmp = d / "summary.json.tmp"
        tmp.write_text(json.dumps(rec, indent=2) + "\n")
        tmp.replace(summary)
        rec["cached"] = False
        return rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, points))
