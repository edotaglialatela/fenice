"""Python helpers for the Fenice solver, shared by the GUI, the validation scripts, the docs
tools and the regression check (``python prerequisites.py`` installs it in editable mode).

    import fenice
    fenice.run("Sod Shock Tube", "nx=200", out_root=fenice.RESULTS / "validation" / "sod")
    for t, path in fenice.read_pvd(fenice.RESULTS / "validation" / "sod" / "sod_shock_tube"):
        f = fenice.read_vtr(path)          # f["Density"], f["Velocity"], f["x"], ...
    fenice.read_vtu(path)                  # an adaptive-mesh snapshot: f["x0"], f["level"], f["Density"], ...
"""
from .io import primitive, read_dump, read_monitor, read_pvd, read_vtr, read_vtu, vtk_array
from .solver import RESULTS, ROOT, exe, run

__all__ = ["ROOT", "RESULTS", "exe", "run", "vtk_array", "read_pvd", "read_vtr", "read_vtu", "read_monitor", "read_dump",
           "primitive"]
