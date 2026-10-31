"""Solver module: the setup dialog (case, physics, numerics, AMR, output, MPI) as the editor of a solver node.

The geometry (domain, mesh, immersed body, the boundary condition of each side) is not edited here: it belongs to the geometry node linked
to the solver, which writes into the same fields of this page (hidden), so that the page stays the only copy of every parameter. The
node's face carries Run / Stop, the Low-Mach switch and the monitor interval.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
                             QWidget)

from gui.modules import ModuleSpec, Port
from gui.project import IDLE, RUNNING

LOW_MACH, PLAIN = "HLLC (Low-Mach Thornber)", "HLLC"


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
            self._hide_geometry()
            self.combo_testcase.currentTextChanged.connect(lambda _: project.setup_changed.emit())   # after the defaults are applied

        def _hide_geometry(self) -> None:
            """The domain, the body and the sides are the geometry node's: hidden here, but still the fields that hold the values."""
            self.group_domain.hide()
            self.tabs.setTabVisible(self.tabs.indexOf(self.ibm_tab), False)
            self._side_widgets = [self.bc_left, self.bc_right, self.bc_bottom, self.bc_top, self.bc_front, self.bc_back]
            for w in self._side_widgets:
                w.hide()
                label = self.form_bc.labelForField(w)
                if label is not None:
                    label.hide()
                    self._side_widgets.append(label)

        def _update_dim_inputs(self, *args):
            super()._update_dim_inputs(*args)
            for w in getattr(self, "_side_widgets", []):
                w.setVisible(False)

        def validate_and_accept(self):
            if self._collect_params():
                try:
                    project.start_run(self.sim_params)
                except RuntimeError as exc:
                    QMessageBox.information(self, "Fenice", str(exc))

        def reject(self):
            pass   # Escape does not close a page

    page = EmbeddedSetup()
    project.provider = page.peek_params
    project.editor = page
    return page


def _face(project, page):
    """The solver node's face: Run / Stop, the Low-Mach switch (the Riemann solver of the page) and the monitor interval."""
    face = QWidget()
    lay = QVBoxLayout(face)
    lay.setContentsMargins(6, 2, 6, 4)
    lay.setSpacing(3)
    row = QHBoxLayout()
    run, stop = QPushButton("▶ Run"), QPushButton("■ Stop")
    run.setToolTip("Run the solver with the setup and the linked geometry")
    stop.setToolTip("Stop cleanly: the solver saves its checkpoint and index, then exits")
    row.addWidget(run)
    row.addWidget(stop)
    lay.addLayout(row)
    low = QCheckBox("Low-Mach correction")
    lay.addWidget(low)
    rec = QHBoxLayout()
    rec.addWidget(QLabel("Record every"))
    spin = QSpinBox()
    spin.setRange(0, 1_000_000)
    spin.setValue(project.monitor_every)
    spin.setSpecialValueText("case")
    spin.setToolTip("Write the monitor file (residuals, averages, loads) every N steps in the next run; 'case' keeps the case's own "
                    "monitor_interval. Each record costs one extra residual evaluation.")
    rec.addWidget(spin)
    rec.addWidget(QLabel("steps"))
    lay.addLayout(rec)

    combo = page.combo_riemann
    syncing = [False]

    def from_combo(*_):
        text = combo.currentText()
        syncing[0] = True
        low.setChecked(text == LOW_MACH)
        low.setEnabled(text in (PLAIN, LOW_MACH) and project.state == IDLE)
        low.setToolTip("Scales the numerical dissipation of the HLLC solver at low Mach number (Thornber)." if text in (PLAIN, LOW_MACH)
                       else f"The Riemann solver is '{text}', set in the Setup editor: the switch leaves it alone.")
        syncing[0] = False

    def to_combo(on):
        if syncing[0]:
            return
        if combo.currentText() not in (PLAIN, LOW_MACH):   # another Riemann solver: not for the switch to overwrite
            from_combo()
            return
        i = combo.findText(LOW_MACH if on else PLAIN)
        if i >= 0:
            combo.setCurrentIndex(i)

    def state(s):
        run.setEnabled(s == IDLE)
        stop.setEnabled(s == RUNNING)
        spin.setEnabled(s == IDLE)
        from_combo()

    combo.currentTextChanged.connect(from_combo)
    low.toggled.connect(to_combo)
    spin.valueChanged.connect(lambda v: setattr(project, "monitor_every", v))
    run.clicked.connect(page.validate_and_accept)
    stop.clicked.connect(project.stop)
    project.state_changed.connect(state)
    state(project.state)
    return face


MODULE = ModuleSpec("solver", "Solver", _make, order=20, requires=("PyQt6",), owns_project=True, face=_face,
                    inputs=(Port("geometry", "geometry", label="geometry"),),
                    outputs=(Port("run", "run", multi=True, label="run"), Port("results", "results", multi=True, label="results")),
                    description="Case, physics, numerics, AMR, output and MPI; Run, Stop and the Low-Mach switch. Needs a linked geometry.")
