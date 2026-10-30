"""Setup module: the setup dialog (case, mesh, physics, boundaries, numerics, immersed body, AMR) inside the workbench."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QDialogButtonBox

from gui.modules import ModuleSpec


def _make(project):
    from gui.setup_dialog import SetupDialog

    class EmbeddedSetup(SetupDialog):
        """The dialog as a page: Run Simulation hands the validated parameters to the project instead of closing."""

        def __init__(self):
            super().__init__()
            self.setWindowFlags(Qt.WindowType.Widget)
            self.setMinimumSize(0, 0)
            self.buttons.button(QDialogButtonBox.StandardButton.Cancel).hide()
            ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
            ok.setAutoDefault(False)   # Enter in a field must not start a run
            ok.setDefault(False)

        def validate_and_accept(self):
            if self._collect_params():
                project.set_params(self.sim_params)
                project.start_run()

        def reject(self):
            pass   # Escape does not close a page

    page = EmbeddedSetup()
    project.provider = page.peek_params
    project.editor = page
    return page


MODULE = ModuleSpec("setup", "Setup", _make, order=10, requires=("PyQt6",),
                    description="Case, mesh, physics, boundaries, numerics, immersed body and AMR; Run Simulation starts the run.")
