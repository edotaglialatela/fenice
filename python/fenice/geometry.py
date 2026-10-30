"""The immersed body of a run, for previews: its outline and the solid cells it makes on a mesh.

The same rules as the solver (``Geometry.cpp``, ``ImmersedBoundary`` in ``Physics.cpp``): a cylinder / sphere is the cells whose centre is
within the radius, an airfoil section (``NACA dddd`` or a Selig / Lednicer file, scaled to the chord, rotated by the angle of attack about
the quarter chord, placed at ``ibm_cx, ibm_cy``) the cells whose centre is inside the polygon. The count of solid cells equals the one the
solver prints (``[IBM] ... -> N solid cells``); ``cases/validation/check_geometry.py`` checks that on the cases with a body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Body:
    kind: str                    # "circle" or "polygon"
    outline: np.ndarray          # (n, 2) points of the contour, closed implicitly
    centre: tuple[float, float]  # circle centre, or the quarter chord of an airfoil
    radius: float = 0.0          # circle
    chord: float = 0.0           # airfoil
    description: str = ""

    def solid(self, xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
        """Boolean mask of the points (xc[ny, nx], yc[ny, nx]) inside the body, as the solver decides it."""
        if self.kind == "circle":
            dx, dy = xc - self.centre[0], yc - self.centre[1]
            return np.sqrt(dx * dx + dy * dy) <= self.radius
        return polygon_contains(self.outline, xc, yc)


def polygon_contains(pts: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Even-odd test of every (x, y) against the polygon `pts`, with the operations of ``Polygon2D::contains``."""
    inside = np.zeros(np.shape(x), dtype=bool)
    xs, ys = pts[:, 0], pts[:, 1]
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    near = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)   # the solver skips the far cells the same way
    xn, yn = x[near], y[near]
    flip = np.zeros(xn.shape, dtype=bool)
    n = len(pts)
    for i in range(n):
        j = i - 1 if i > 0 else n - 1
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        if (yi > yn).any() or (yj > yn).any():
            cross = ((yi > yn) != (yj > yn)) & (xn < (xj - xi) * (yn - yi) / (yj - yi) + xi)
            flip ^= cross
    inside[near] = flip
    return inside


def naca4(digits: str, n: int = 200) -> np.ndarray:
    """Unit-chord NACA 4-digit section with a closed trailing edge, `n` cosine-spaced points per surface (``naca4``)."""
    if len(digits) != 4 or not digits.isdigit():
        raise ValueError(f"NACA 4-digit code must be 4 digits, got '{digits}'")
    m, p, t = int(digits[0]) / 100.0, int(digits[1]) / 10.0, int(digits[2:]) / 100.0
    if not t > 0.0:
        raise ValueError(f"NACA {digits}: zero thickness")
    if m > 0.0 and not p > 0.0:
        raise ValueError(f"NACA {digits}: camber needs a camber position")
    up, lo = np.zeros((n + 1, 2)), np.zeros((n + 1, 2))
    for i in range(n + 1):
        x = 0.5 * (1.0 - math.cos(math.pi * i / n))
        yt = 5.0 * t * (0.2969 * math.sqrt(x) - 0.1260 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1036 * x * x * x * x)
        yc = dyc = 0.0
        if m > 0.0:
            if x < p:
                yc = m / (p * p) * (2.0 * p * x - x * x)
                dyc = 2.0 * m / (p * p) * (p - x)
            else:
                yc = m / ((1 - p) * (1 - p)) * (1.0 - 2.0 * p + 2.0 * p * x - x * x)
                dyc = 2.0 * m / ((1 - p) * (1 - p)) * (p - x)
        th = math.atan(dyc)
        up[i] = (x - yt * math.sin(th), yc + yt * math.cos(th))
        lo[i] = (x + yt * math.sin(th), yc - yt * math.cos(th))
    return np.vstack([up[::-1], lo[1:n]])   # TE -> LE over the upper surface, LE -> TE over the lower (TE shared)


def naca_digits(shape: str) -> str:
    """'2412' for 'NACA 2412' (spaces and case ignored), '' if `shape` is not a NACA 4-digit name."""
    s = "".join(shape.split()).lower()
    return s[4:] if len(s) == 8 and s.startswith("naca") and s[4:].isdigit() else ""


def _pair(line: str):
    parts = line.split()
    try:
        return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        return None


def load_airfoil(path: str | Path) -> np.ndarray:
    """A Selig or Lednicer coordinate file, normalised to unit chord with the leading edge at the origin (``load_airfoil``)."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    k = 1 if lines and _pair(lines[0]) is None else 0
    lednicer, n_up = False, 0
    first = _pair(lines[k]) if k < len(lines) else None
    if first and first[0] > 1.0 and first[1] > 1.0:
        lednicer, n_up, n_lo = True, int(first[0]), int(first[1])
        k += 1
    data = []
    for ln in lines[k:]:
        pr = _pair(ln)
        if pr is None:
            raise ValueError(f"{path}: expected 'x y', got '{ln}'")
        data.append(pr)
    if lednicer:
        if len(data) != n_up + n_lo:
            raise ValueError(f"{path}: Lednicer header announces {n_up} + {n_lo} points, the file has {len(data)}")
        pts = data[:n_up][::-1] + [d for i, d in enumerate(data[n_up:], start=n_up) if i > n_up or d != data[0]]
    else:
        pts = data
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError(f"{path}: an airfoil needs at least 3 points")
    a = np.array(pts, dtype=float)
    le = a[int(np.argmin(a[:, 0]))]   # first point of smallest x, as std::min_element
    c = a[:, 0].max() - le[0]
    if not c > 0.0:
        raise ValueError(f"{path}: the coordinates have zero chord")
    return (a - le) / c


def place_airfoil(unit: np.ndarray, chord: float, aoa_deg: float, xq: float, yq: float) -> np.ndarray:
    """The unit section scaled to `chord`, nose up by `aoa_deg` about the quarter chord, which sits at (xq, yq) (``place_airfoil``)."""
    a = aoa_deg * math.pi / 180.0
    ca, sa = math.cos(a), math.sin(a)
    x, y = (unit[:, 0] - 0.25) * chord, unit[:, 1] * chord
    return np.column_stack([xq + x * ca + y * sa, yq - x * sa + y * ca])


def find_airfoil_file(name: str, root: str | Path) -> Path | None:
    """The airfoil file `name` as the solver looks for it (absolute, next to a case.toml, current folder), searching every case folder."""
    p = Path(name)
    if p.is_absolute():
        return p if p.exists() else None
    for case in sorted(Path(root).glob("cases/*/case.toml")):
        if (case.parent / p).exists():
            return case.parent / p
    return p if p.exists() else None


def body_from_params(p: dict, root: str | Path) -> Body | None:
    """The body of a run from its resolved parameters (strings or typed values, solver key names), None if there is no body."""
    on = p.get("use_ibm", False)
    if isinstance(on, str):
        on = on.strip().lower() in ("1", "true", "yes", "on")
    if not on:
        return None
    shape = str(p.get("ibm_shape", "Cylinder"))
    cx, cy = float(p.get("ibm_cx", 0.5)), float(p.get("ibm_cy", 0.5))
    digits = naca_digits(shape)
    if digits or shape.strip().lower() == "airfoil":
        if digits:
            unit, label = naca4(digits), f"NACA {digits}"
        else:
            path = find_airfoil_file(str(p.get("ibm_file", "")), root)
            if path is None:
                raise FileNotFoundError(f"airfoil file '{p.get('ibm_file', '')}' not found in the case folders or the current folder")
            unit, label = load_airfoil(path), path.name
        chord, aoa = float(p.get("ibm_chord", 1.0)), float(p.get("ibm_aoa", 0.0))
        return Body("polygon", place_airfoil(unit, chord, aoa, cx, cy), (cx, cy), chord=chord,
                    description=f"{label}, chord {chord:g}, {aoa:g} deg, quarter chord at ({cx:g}, {cy:g})")
    if shape.strip().lower() not in ("cylinder", "sphere"):
        raise ValueError(f"unknown immersed-body shape '{shape}'")
    r = float(p.get("ibm_radius", 0.1))
    t = np.linspace(0.0, 2.0 * math.pi, 181)[:-1]
    return Body("circle", np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)]), (cx, cy), radius=r,
                description=f"{shape.lower()} of radius {r:g} at ({cx:g}, {cy:g})")


def solid_window(body: Body, Lx: float, Ly: float, nx: int, ny: int):
    """Solid cells of `body` on the mesh, only in the window that holds the body: (mask, (i0, i1, j0, j1)) with mask[j - j0, i - i0];
    cells outside the window are never solid. The mesh has its origin at (0, 0)."""
    dx, dy = Lx / nx, Ly / ny
    xs, ys = body.outline[:, 0], body.outline[:, 1]
    i0, i1 = max(int(math.floor(xs.min() / dx)) - 1, 0), min(int(math.ceil(xs.max() / dx)) + 1, nx)
    j0, j1 = max(int(math.floor(ys.min() / dy)) - 1, 0), min(int(math.ceil(ys.max() / dy)) + 1, ny)
    if i1 <= i0 or j1 <= j0:
        return np.zeros((0, 0), dtype=bool), (i0, i0, j0, j0)
    xc = (np.arange(i0, i1) + 0.5) * dx + 0.0
    yc = (np.arange(j0, j1) + 0.5) * dy + 0.0
    gx, gy = np.meshgrid(xc, yc)
    return body.solid(gx, gy), (i0, i1, j0, j1)


def solid_mask(body: Body, Lx: float, Ly: float, nx: int, ny: int) -> np.ndarray:
    """Solid cells [ny, nx] of `body` on the mesh (origin at (0, 0)): the cells whose centre is inside, as the solver's mask."""
    window, (i0, i1, j0, j1) = solid_window(body, Lx, Ly, nx, ny)
    full = np.zeros((ny, nx), dtype=bool)
    full[j0:j1, i0:i1] = window
    return full
