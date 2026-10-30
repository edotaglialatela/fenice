"""Results module: the result viewer on the folder of the run, followed while the run goes on."""
from __future__ import annotations

from gui.modules import ModuleSpec


def _make(project):
    from gui.viewer import ViewerPanel

    panel = ViewerPanel()
    project.results_changed.connect(panel.open_path)
    if project.results_dir is not None:
        panel.open_path(str(project.results_dir))
    return panel


MODULE = ModuleSpec("results", "Results", _make, order=30, requires=("PyQt6", "matplotlib", "numpy"),
                    description="Fields, slices, adaptive meshes and monitor curves of the run; follows a run in progress.")
