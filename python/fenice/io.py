"""Readers for fenice output: VTK snapshots (.vtr, ASCII or binary), the .pvd time series,
the monitor CSV and the final-state / checkpoint .npz."""
import base64
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_DTYPES = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8",
           "UInt8": "<u1", "UInt32": "<u4", "UInt64": "<u8"}


def vtk_array(node, header="<u4"):
    """Values of a VTK ``<DataArray>`` element as a flat float64 array.

    ``format="ascii"`` is plain text; ``format="binary"`` (fenice's default, ``vtk_format``)
    is inline base64 of a UInt32 byte count followed by little-endian values, one stream.
    ``header``: dtype of that byte count (``"<u8"`` for files with ``header_type="UInt64"``).
    """
    if node.get("format", "ascii") == "binary":
        raw = base64.b64decode("".join(node.text.split()))
        hsize = np.dtype(header).itemsize
        nbytes = int(np.frombuffer(raw[:hsize], dtype=header)[0])
        data = np.frombuffer(raw[hsize:hsize + nbytes], dtype=_DTYPES[node.get("type", "Float32")])
        return data.astype(float)
    return np.array(node.text.split(), dtype=float)


def read_pvd(where):
    """Snapshots of a .pvd time series as [(time, path)], in order. `where`: the .pvd or the
    case output folder holding it."""
    p = Path(where)
    if p.is_dir():
        p = next(p.glob("*.pvd"))
    root = ET.parse(p).getroot()
    return [(float(d.get("timestep")), str(p.parent / d.get("file"))) for d in root.iter("DataSet")]


def read_vtr(path):
    """A fenice .vtr snapshot as a dict: nx, ny, nz, the node coordinates x, y, z, and every
    cell field by name (Density, Pressure, Temperature_nondim, Mach, Velocity, ...), shaped
    [ny, nx] in 2-D or [nz, ny, nx] in 3-D (Velocity gets a last axis of 3)."""
    root = ET.parse(path).getroot()
    nx, _, ny, _, nz = (int(v) for v in root.find("RectilinearGrid").get("WholeExtent").split()[1:])
    shape = (ny, nx) if nz == 0 else (nz, ny, nx)
    out = {"nx": nx, "ny": ny, "nz": max(nz, 1)}
    for c in ("x", "y", "z"):
        out[c] = vtk_array(root.find(f'.//Coordinates/DataArray[@Name="{c}"]'))
    for node in root.find(".//CellData"):
        ncomp = int(node.get("NumberOfComponents", "1"))
        a = vtk_array(node)
        out[node.get("Name")] = a.reshape(shape + ((ncomp,) if ncomp > 1 else ()))
    return out


def read_monitor(where):
    """The monitor CSV (step, t, mass, kinetic_energy, total_energy[, Fx, Fy, Cd, Cl, Cm]) as a
    numpy structured array, one row per distinct time. `where`: the CSV or its case folder."""
    p = Path(where)
    if p.is_dir():
        p = next(p.glob("*_monitor.csv"))
    d = np.genfromtxt(p, delimiter=",", names=True)
    _, keep = np.unique(d["t"], return_index=True)   # the last row may repeat the final time
    return d[np.sort(keep)]


def read_dump(path):
    """A dump_state / checkpoint .npz as a dict of arrays (U, t, step, ...)."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
