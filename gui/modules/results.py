"""Postprocessing module: the result viewer on the folder of the run of the solver it is linked to, followed while the run goes on."""
from __future__ import annotations

from gui.modules import ModuleSpec, Port


def _make(project):
    from gui.viewer import ViewerPanel

    panel = ViewerPanel()
    project.results_changed.connect(panel.open_path)
    if project.results_dir is not None:   # linked after the run (or in the middle of it): open what the solver has reported
        panel.open_path(str(project.results_dir))
    return panel


MODULE = ModuleSpec("results", "Postprocessing", _make, order=40, requires=("PyQt6", "matplotlib", "numpy"),
                    inputs=(Port("results", "results"),),
                    description="Fields, slices, adaptive meshes and monitor curves of the run; follows a run in progress.")
