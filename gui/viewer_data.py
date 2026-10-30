"""Loading and sampling of fenice output for the viewer (no Qt: it can be tested without a display).

``Series`` lists the snapshots of a results folder (a ``.vtr`` of a uniform mesh, a ``.vtu`` / ``.pvtu`` of an adaptive one), found
from the files themselves, because the ``.pvd`` index is only written when a run ends. ``Frame`` is one snapshot: its scalar fields,
and for an adaptive mesh a rasterisation of the leaves onto the pixels of the view. All reading goes through ``fenice.io``.
"""
from __future__ import annotations

import io
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import fenice

_SNAPSHOT = re.compile(r"^(?P<prefix>.+)_(?P<step>\d+)\.(?P<ext>vtr|vtu|pvtu)$")
_FIELD_ORDER = ("Density", "Pressure", "Temperature_nondim", "Mach", "|Velocity|", "Velocity_x", "Velocity_y", "Velocity_z",
                "AMR_Level")
_KEY_SHIFT = 32   # leaf (i, j) at a level is packed as i << 32 | j


@dataclass(frozen=True)
class Snapshot:
    step: int
    path: Path
    time: float | None   # from the .pvd if there is one, else from the monitor at that step, else unknown
    stamp: tuple = ()    # (modification time, size) of the file: a snapshot a rerun overwrote is another snapshot


def _newest_prefix_files(folder: Path) -> list[tuple[int, Path, tuple]]:
    """(step, file, stamp) of the snapshots of the case written last in `folder`; a .pvtu wins over a .vtu of the same step.
    The snapshots a previous run left beyond the point the current one has reached (older than a lower step) are not listed."""
    by_prefix: dict[str, dict[int, tuple[Path, tuple]]] = {}
    newest: dict[str, int] = {}
    try:
        entries = list(folder.iterdir())
    except OSError:
        return []
    for p in entries:
        m = _SNAPSHOT.match(p.name)
        if not m:
            continue
        try:
            st = p.stat()
        except OSError:
            continue   # renamed or removed since the listing
        pre, step = m["prefix"], int(m["step"])
        slot = by_prefix.setdefault(pre, {})
        if step not in slot or (m["ext"] == "pvtu" and slot[step][0].suffix != ".pvtu"):
            slot[step] = (p, (st.st_mtime_ns, st.st_size))
        newest[pre] = max(newest.get(pre, 0), st.st_mtime_ns)
    if not by_prefix:
        return []
    best = max(newest, key=newest.get)
    out, latest = [], 0
    for step, (p, stamp) in sorted(by_prefix[best].items()):
        if stamp[0] >= latest:   # else a lower step is newer: this one belongs to an earlier run
            out.append((step, p, stamp))
        latest = max(latest, stamp[0])
    return out


class Series:
    """The snapshots of one results folder, refreshed from disk while a run is going."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.snapshots: list[Snapshot] = []
        self._monitor = None
        self._cache: OrderedDict[tuple, Frame] = OrderedDict()
        self._pvd_stamp: tuple | None = None
        self._pvd_times: dict[str, float] = {}
        self.refresh()

    def refresh(self) -> bool:
        """Rescan the folder; True if the list of snapshots changed."""
        files = _newest_prefix_files(self.folder)
        pvd = self._pvd()
        self._monitor = self._read_monitor()
        snaps = [Snapshot(step, p, self._time_of(step, p.name, pvd), stamp) for step, p, stamp in files]
        changed = [(s.step, s.path, s.stamp, s.time) for s in snaps] != [(s.step, s.path, s.stamp, s.time) for s in self.snapshots]
        self.snapshots = snaps
        return changed

    def _read_monitor(self):
        """The monitor CSV; while a run is writing it the last row may be half there, and is then left out."""
        try:
            return fenice.read_monitor(self.folder)
        except StopIteration:
            return None   # no monitor file (yet)
        except (OSError, ValueError, IndexError):
            pass
        try:
            path = next(self.folder.glob("*_monitor.csv"))
            lines = path.read_text().splitlines()[:-1]
            d = np.genfromtxt(io.StringIO(chr(10).join(lines)), delimiter=",", names=True)
            return d if d.ndim == 1 and d.size > 1 else None
        except (StopIteration, OSError, ValueError, IndexError):
            return None

    def _pvd(self) -> dict[str, float]:
        """The times of the .pvd index, read again only when the file changed (a file another process has open cannot be
        replaced on Windows, so it is not opened every second)."""
        try:
            path = next(self.folder.glob("*.pvd"))
            st = path.stat()
        except (StopIteration, OSError):
            self._pvd_stamp, self._pvd_times = None, {}
            return self._pvd_times
        if (st.st_mtime_ns, st.st_size) != self._pvd_stamp:
            try:
                self._pvd_times = {Path(f).name: t for t, f in fenice.read_pvd(path)}
                self._pvd_stamp = (st.st_mtime_ns, st.st_size)
            except (OSError, ValueError):
                pass
        return self._pvd_times

    def _time_of(self, step: int, name: str, pvd: dict) -> float | None:
        if name in pvd:
            return pvd[name]
        m = self._monitor
        if m is not None and m.size > 1 and "step" in m.dtype.names and m["step"][0] <= step <= m["step"][-1]:
            return float(np.interp(step, m["step"], m["t"]))
        return None

    def monitor(self):
        """The monitor CSV as a structured array, or None."""
        return self._monitor

    def frame(self, index: int) -> "Frame":
        snap = self.snapshots[index]
        key = (snap.path, snap.stamp)
        if key not in self._cache:
            self._cache[key] = Frame(snap.path)
            while len(self._cache) > 3:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]


class Frame:
    """One snapshot. `kind` is "grid" (uniform mesh; 2-D or 3-D) or "amr" (leaf list)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.kind = "grid" if self.path.suffix == ".vtr" else "amr"
        raw = fenice.read_vtr(self.path) if self.kind == "grid" else fenice.read_vtu(self.path)
        self.fields: dict[str, np.ndarray] = {}
        for k, v in raw.items():
            if k in ("Velocity",):
                self.fields["Velocity_x"], self.fields["Velocity_y"], self.fields["Velocity_z"] = (v[..., c] for c in range(3))
                self.fields["|Velocity|"] = np.sqrt(np.sum(v * v, axis=-1))
            elif k == "level":
                self.fields["AMR_Level"] = v.astype(float)
            elif k in ("Density", "Pressure", "Temperature_nondim", "Mach"):
                self.fields[k] = v
        if self.kind == "grid":
            self.x, self.y, self.z = raw["x"], raw["y"], raw["z"]
            self.nx, self.ny, self.nz = raw["nx"], raw["ny"], raw["nz"]
        else:
            self.nz = 1
        if not self.is_3d:
            self.fields.pop("Velocity_z", None)
        if self.kind == "amr":
            self.x0, self.x1, self.y0, self.y1 = raw["x0"], raw["x1"], raw["y0"], raw["y1"]
            self.level = raw["level"]
            self._levels: dict[int, tuple[np.ndarray, np.ndarray, float, float]] | None = None

    # ---- what there is to show
    def names(self) -> list[str]:
        return [n for n in _FIELD_ORDER if n in self.fields]

    @property
    def is_3d(self) -> bool:
        return self.kind == "grid" and self.nz > 1

    def extent(self) -> tuple[float, float, float, float]:
        """xmin, xmax, ymin, ymax of the domain."""
        if self.kind == "grid":
            return float(self.x[0]), float(self.x[-1]), float(self.y[0]), float(self.y[-1])
        return float(self.x0.min()), float(self.x1.max()), float(self.y0.min()), float(self.y1.max())

    def slice_count(self, axis: str) -> int:
        return {"z": self.nz, "y": self.ny, "x": self.nx}[axis] if self.kind == "grid" else 1

    def n_leaves(self) -> int:
        return int(self.level.size) if self.kind == "amr" else self.nx * self.ny * self.nz

    # ---- uniform mesh
    def _slice(self, name: str, axis: str, index: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        a = self.fields[name]
        if self.nz == 1:
            return a, self.extent()
        index = int(np.clip(index, 0, self.slice_count(axis) - 1))
        if axis == "z":
            return a[index], (self.x[0], self.x[-1], self.y[0], self.y[-1])
        if axis == "y":
            return a[:, index, :], (self.x[0], self.x[-1], self.z[0], self.z[-1])
        return a[:, :, index], (self.y[0], self.y[-1], self.z[0], self.z[-1])

    # ---- adaptive mesh
    def _tables(self):
        """Per level: the sorted packed (i, j) keys of its leaves, the leaf index in that order, and the cell sizes."""
        if self._levels is None:
            X0, Y0 = self.x0.min(), self.y0.min()
            out = {}
            for l in np.unique(self.level):
                sel = np.flatnonzero(self.level == l)
                dx = float(np.min(self.x1[sel] - self.x0[sel]))
                dy = float(np.min(self.y1[sel] - self.y0[sel]))
                i = np.rint((self.x0[sel] - X0) / dx).astype(np.int64)
                j = np.rint((self.y0[sel] - Y0) / dy).astype(np.int64)
                key = (i << _KEY_SHIFT) | j
                order = np.argsort(key)
                out[int(l)] = (key[order], sel[order], dx, dy)
            self._levels = out
        return self._levels

    def _lookup(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Index of the leaf containing each point (xs[n], ys[n]), -1 outside the mesh."""
        X0, Y0 = self.x0.min(), self.y0.min()
        found = np.full(xs.shape, -1, dtype=np.int64)
        todo = np.arange(xs.size)
        for l, (keys, leaf, dx, dy) in sorted(self._tables().items(), key=lambda kv: -kv[1][0].size):
            if todo.size == 0:
                break
            i = np.floor((xs[todo] - X0) / dx).astype(np.int64)
            j = np.floor((ys[todo] - Y0) / dy).astype(np.int64)
            key = (i << _KEY_SHIFT) | j
            pos = np.minimum(np.searchsorted(keys, key), keys.size - 1)
            hit = (keys[pos] == key) & (i >= 0) & (j >= 0)
            found[todo[hit]] = leaf[pos[hit]]
            todo = todo[~hit]
        return found

    def raster(self, name: str, bounds: tuple[float, float, float, float], size: tuple[int, int]):
        """The field of an adaptive mesh on a `size` = (width, height) pixel grid over `bounds` (xmin, xmax, ymin, ymax):
        an image [height, width], row 0 at ymin, NaN where no leaf is."""
        w, h = size
        xmin, xmax, ymin, ymax = bounds
        xc = xmin + (np.arange(w) + 0.5) * (xmax - xmin) / w
        yc = ymin + (np.arange(h) + 0.5) * (ymax - ymin) / h
        gx, gy = np.meshgrid(xc, yc)
        leaf = self._lookup(gx.ravel(), gy.ravel())
        img = np.full(leaf.shape, np.nan)
        ok = leaf >= 0
        img[ok] = self.fields[name][leaf[ok]]
        return img.reshape(h, w)

    def leaves_in(self, bounds, limit: int = 5000):
        """The leaf rectangles (x0, x1, y0, y1) that intersect `bounds`, or None if there are more than `limit`."""
        xmin, xmax, ymin, ymax = bounds
        sel = np.flatnonzero((self.x1 >= xmin) & (self.x0 <= xmax) & (self.y1 >= ymin) & (self.y0 <= ymax))
        if sel.size > limit:
            return None
        return self.x0[sel], self.x1[sel], self.y0[sel], self.y1[sel]

    # ---- one point
    def value_at(self, name: str, x: float, y: float, axis: str = "z", index: int = 0) -> float | None:
        """The field at the point (x, y) of the picture (of the slice for a 3-D mesh), None outside the domain."""
        if self.kind == "amr":
            leaf = self._lookup(np.array([x]), np.array([y]))[0]
            return None if leaf < 0 else float(self.fields[name][leaf])
        a, (x0, x1, y0, y1) = self._slice(name, axis, index)
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return None
        j = min(int((y - y0) / (y1 - y0) * a.shape[0]), a.shape[0] - 1)
        i = min(int((x - x0) / (x1 - x0) * a.shape[1]), a.shape[1] - 1)
        return float(a[j, i])

    def picture(self, name: str, axis: str, index: int, bounds=None, size=None):
        """(image, extent) of a field: the slice itself on a uniform mesh, the leaves rasterised on `size` pixels over `bounds`
        (the whole domain if None) on an adaptive one."""
        if self.kind == "grid":
            a, ext = self._slice(name, axis, index)
            return a, tuple(float(e) for e in ext)
        b = bounds if bounds is not None else self.extent()
        return self.raster(name, b, size or (1200, 600)), b
