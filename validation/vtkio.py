"""Read fenice's VTK XML data arrays, ASCII or binary.

fenice writes ``<DataArray format="ascii">`` (plain text) or ``format="binary"``
(``vtk_format = binary``, the default): inline base64 of a UInt32 byte count
followed by little-endian Float32 values, encoded as one stream. Both are valid
XML, so files are still parsed with ElementTree, and only the array payload
needs this helper::

    root = ET.parse(path).getroot()
    rho = vtk_array(root.find('.//DataArray[@Name="Density"]'))
"""
import base64

import numpy as np

_DTYPES = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8",
           "UInt8": "<u1", "UInt32": "<u4", "UInt64": "<u8"}


def vtk_array(node, header="<u4"):
    """Values of a ``<DataArray>`` element as a flat float64 array.

    ``header``: dtype of the binary byte count (``"<u8"`` for files written with
    ``header_type="UInt64"``; fenice uses the VTK default UInt32).
    """
    if node.get("format", "ascii") == "binary":
        raw = base64.b64decode("".join(node.text.split()))
        hsize = np.dtype(header).itemsize
        nbytes = int(np.frombuffer(raw[:hsize], dtype=header)[0])
        data = np.frombuffer(raw[hsize:hsize + nbytes], dtype=_DTYPES[node.get("type", "Float32")])
        return data.astype(float)
    return np.array(node.text.split(), dtype=float)
