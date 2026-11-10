"""Analysis tools for quantitative inspection of solver results."""
from __future__ import annotations

from gui.modules import ModuleSpec, Port


def _make(project):
    from gui.analysis import AnalysisPanel

    panel = AnalysisPanel()
    project.started.connect(panel.wait_for_results)
    project.results_changed.connect(panel.open_path)
    if project.results_dir is not None:
        panel.open_path(str(project.results_dir))
    return panel


MODULE = ModuleSpec(
    "analysis", "Analysis Tools", _make, order=50, requires=("PyQt6", "matplotlib", "numpy"),
    inputs=(Port("results", "results"),),
    description="Field statistics and profiles, monitor spectra, POD/DMD modes and Sobol sensitivity intervals.",
)
