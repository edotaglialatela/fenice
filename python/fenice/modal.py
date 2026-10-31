"""Modal analysis of snapshot series: POD (snapshot method) and exact DMD, in plain NumPy.

    from fenice import modal
    snap = modal.load_snapshots(case_output_folder, "Velocity", component=1)    # .pvd series of .vtr (or .vtu) files
    pod = modal.pod(snap["X"], snap["weights"])                                 # energies, modes, coefficients
    dmd = modal.dmd(snap["X"], dt=snap["dt"], rank=2)                           # eigenvalues, frequencies, growth rates
    modal.export_modes_vtr("mode0.vtr", pod["modes"][:, 0], snap["x"], snap["y"], "pod0")

Conventions. X is (points, snapshots), one column per snapshot at a uniform time step dt. `weights` are the cell areas (or
volumes), so inner products are <a, b> = sum(w a b); POD energies are the eigenvalues of the weighted covariance of the
mean-removed data (their sum is the mean over time of the weighted squared fluctuation, per unit weight sum when the weights
are normalised, which `pod` does), modes are weight-orthonormal. DMD is the exact variant (Tu et al. 2014): the snapshot
pairs X[:, :-1] -> X[:, 1:] define the operator A, rank-truncated at `rank` or at the singular values above `tol` times the
largest; the continuous-time exponents are omega = log(lambda) / dt. Adaptive-mesh snapshots (.vtu) are resampled to the
finest level of the mesh, a uniform grid of (domain / smallest leaf)^2 cells: the cost is that grid times the snapshots.
"""
from pathlib import Path

import numpy as np

from .io import read_pvd, read_vtr, read_vtu


def _resample_leaves(f, name, component=None):
    """An adaptive-mesh snapshot on the uniform grid of its finest level: array (ny, nx), plus node coordinates."""
    x0, x1, y0, y1 = (np.asarray(f[k], float) for k in ("x0", "x1", "y0", "y1"))
    h = min((x1 - x0).min(), (y1 - y0).min())
    nx, ny = int(round(x1.max() / h)), int(round(y1.max() / h))
    a = np.asarray(f[name], float)
    if component is not None:
        a = a[..., component]
    out = np.zeros((ny, nx))
    for v, a0, a1, b0, b1 in zip(a, x0, x1, y0, y1):
        out[int(round(b0 / h)):int(round(b1 / h)), int(round(a0 / h)):int(round(a1 / h))] = v
    return out, np.linspace(0, nx * h, nx + 1), np.linspace(0, ny * h, ny + 1)


def load_snapshots(where, field="Density", component=None, every=1, first=0, last=None):
    """Read a .pvd series (a folder holding it, or the file) into a snapshot matrix.

    field: a cell field name ("Density", "Pressure", "Velocity", ...); component: its component for a vector field.
    every / first / last select snapshots. Returns {"X" (points, snapshots), "times", "dt" (the mean step; the series must
    be uniform to 1e-6 relative, else ValueError), "weights" (cell areas, flattened like a column), "x", "y" (node
    coordinates), "shape" (ny, nx)}."""
    series = read_pvd(where)[first:last:every]
    if len(series) < 2:
        raise ValueError("a modal analysis needs at least two snapshots")
    cols, x, y = [], None, None
    for _, path in series:
        if str(path).endswith((".vtu", ".pvtu")):
            a, x, y = _resample_leaves(read_vtu(path), field, component)
        else:
            f = read_vtr(path)
            a, x, y = f[field], f["x"], f["y"]
            if component is not None:
                a = a[..., component]
        cols.append(np.asarray(a, float).ravel())
    t = np.array([s[0] for s in series])
    steps = np.diff(t)
    if np.max(np.abs(steps - steps.mean())) > 1e-6 * abs(steps.mean()):
        raise ValueError("the snapshots are not uniformly spaced in time; resample them first")
    w = np.outer(np.diff(y), np.diff(x)).ravel()
    return {"X": np.array(cols).T, "times": t, "dt": float(steps.mean()), "weights": w, "x": x, "y": y,
            "shape": (len(y) - 1, len(x) - 1)}


def pod(X, weights=None, mean=True, rank=None):
    """Proper orthogonal decomposition by the snapshot method.

    X (points, snapshots); weights (points) or None for equal weights; mean=True removes the temporal mean first.
    Returns {"energies" (descending, the eigenvalues), "modes" (points, r; weighted-orthonormal: sum(w phi_i phi_j) =
    delta_ij), "coefficients" (r, snapshots; X - mean = modes @ coefficients), "mean"}; the weights are normalised to sum to 1
    so the energies sum to the time-averaged weighted fluctuation energy."""
    X = np.asarray(X, float)
    n, m = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mu = X.mean(axis=1) if mean else np.zeros(n)
    Xc = X - mu[:, None]
    Xw = np.sqrt(w)[:, None] * Xc
    C = Xw.T @ Xw / m   # m x m
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, V = np.maximum(lam[order], 0.0), V[:, order]
    r = int(rank) if rank else int(np.sum(lam > lam[0] * 1e-14))
    lam, V = lam[:r], V[:, :r]
    phi_w = Xw @ V / np.sqrt(m * lam)   # weighted-orthonormal
    modes = phi_w / np.sqrt(w)[:, None]
    coeffs = modes.T @ (w[:, None] * Xc)
    return {"energies": lam, "modes": modes, "coefficients": coeffs, "mean": mu}


def dmd(X, dt, rank=None, tol=1e-10):
    """Exact dynamic mode decomposition of the snapshot matrix X (points, snapshots) with time step dt.

    Returns {"eigenvalues" lambda (r,), "omega" = log(lambda)/dt, "frequency" = Im(omega)/(2 pi), "growth" = Re(omega),
    "modes" (points, r), "amplitudes" (r,) fitted to the first snapshot}, sorted by decreasing |amplitude|."""
    X = np.asarray(X, float)
    X1, X2 = X[:, :-1], X[:, 1:]
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    r = int(rank) if rank else int(np.sum(s > tol * s[0]))
    U, s, Vh = U[:, :r], s[:r], Vh[:r]
    Atilde = U.conj().T @ X2 @ Vh.conj().T / s
    lam, W = np.linalg.eig(Atilde)
    modes = X2 @ Vh.conj().T / s @ W
    amp = np.linalg.lstsq(modes, X[:, 0].astype(complex), rcond=None)[0]
    order = np.argsort(-np.abs(amp))
    lam, modes, amp = lam[order], modes[:, order], amp[order]
    omega = np.log(lam.astype(complex)) / dt
    return {"eigenvalues": lam, "omega": omega, "frequency": omega.imag / (2 * np.pi), "growth": omega.real,
            "modes": modes, "amplitudes": amp}


def export_modes_vtr(path, mode, x, y, name="mode"):
    """Write a 2-D mode (flattened (ny*nx) or (ny, nx), real) as an ASCII .vtr cell field `name` on the node
    coordinates x (nx+1), y (ny+1), for the workbench Postprocessing node or any VTK viewer."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x) - 1, len(y) - 1
    a = np.real(np.asarray(mode)).reshape(ny * nx)
    arr = lambda nm, v: (f'<DataArray type="Float32" Name="{nm}" format="ascii">'
                         + " ".join(f"{t:.9g}" for t in v) + "</DataArray>")
    text = f"""<?xml version="1.0"?>
<VTKFile type="RectilinearGrid" version="1.0" byte_order="LittleEndian">
<RectilinearGrid WholeExtent="0 {nx} 0 {ny} 0 0">
<Piece Extent="0 {nx} 0 {ny} 0 0">
<Coordinates>{arr("x", x)}{arr("y", y)}{arr("z", [0.0])}</Coordinates>
<CellData Scalars="{name}">{arr(name, a)}</CellData>
</Piece>
</RectilinearGrid>
</VTKFile>
"""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(p)
    return p
