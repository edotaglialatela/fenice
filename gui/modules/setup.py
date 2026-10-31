"""Solver module: the setup dialog (case, physics, numerics, AMR, output, MPI) as the editor of a solver node.

The geometry (domain, mesh, immersed body, the boundary condition of each side) is not edited here: it belongs to the geometry node linked
to the solver, which writes into the same fields of this page (hidden), so that the page stays the only copy of every parameter. The
node's face carries Run / Stop, the Low-Mach switch and the monitor interval.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout,
                             QWidget)

from gui.modules import ModuleSpec, Port
from gui.project import IDLE, RUNNING

LOW_MACH, PLAIN = "HLLC (Low-Mach Thornber)", "HLLC"


class Switch(QSlider):
    """A two-position slider (inactive / active): a click anywhere, or the space bar, moves it to the other end."""

    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(0, 1)
        self.setFixedSize(40, 22)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("QSlider::groove:horizontal { height: 10px; border-radius: 5px; background: #b8bfcc; }"
                           "QSlider::sub-page:horizontal { border-radius: 5px; background: #9b2d3a; }"
                           "QSlider::handle:horizontal { width: 18px; margin: -5px 0; border-radius: 9px; background: #ffffff;"
                           " border: 1px solid #7a8190; }"
                           "QSlider:disabled::sub-page:horizontal { background: #c9ced8; }")
        self.valueChanged.connect(lambda v: self.toggled.emit(bool(v)))

    def isChecked(self) -> bool:
        return self.value() == 1

    def setChecked(self, on: bool) -> None:
        self.setValue(1 if on else 0)

    def mousePressEvent(self, event) -> None:
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self.isChecked())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.setChecked(not self.isChecked())
        else:
            super().keyPressEvent(event)


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
            self.tabs.setTabVisible(self.tabs.indexOf(self.amr_tab), False)   # the refinement is the geometry node's too
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

        def _geometry_problem(self) -> str | None:
            """The first geometry field (held here, edited in the geometry node) that is not a number, as a message."""
            v = self.geometry_values()
            for key, kind in (("Lx", float), ("Ly", float), ("Lz", float), ("ibm_radius", float), ("ibm_cx", float), ("ibm_cy", float),
                              ("ibm_cz", float), ("nx", int), ("ny", int), ("nz", int),
                              ("amr_max_level", int), ("amr_init_levels", int), ("amr_diffusion", int), ("amr_refine_every", int),
                              ("amr_criterion", float), ("amr_refine_threshold", float), ("amr_derefine_threshold", float)):
                try:
                    kind(v[key])
                except ValueError:
                    return f"The geometry has an invalid value: {key} = '{v[key]}'. Correct it in the Geometry node."
            return None

        def validate_and_accept(self):
            problem = self._geometry_problem()
            if problem:
                QMessageBox.information(self, "Fenice", problem)
                return
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
    low = Switch()
    low_text = QLabel("inactive")
    lowrow = QHBoxLayout()
    lowrow.addWidget(QLabel("Low-Mach"))
    lowrow.addStretch(1)
    lowrow.addWidget(low_text)
    lowrow.addWidget(low)
    lay.addLayout(lowrow)
    rec = QHBoxLayout()
    rec.addWidget(QLabel("Record every"))
    spin = QSpinBox()
    spin.setRange(0, 1_000_000)
    spin.setValue(project.monitor_every)
    spin.setSpecialValueText("case")
    spin.setFixedWidth(78)
    spin.setToolTip("Write the monitor file (residuals, averages, loads) every N steps in the next run; 'case' keeps the case's own "
                    "monitor_interval. Each record costs one extra residual evaluation.")
    rec.addWidget(spin)
    rec.addWidget(QLabel("steps"))
    rec.addStretch(1)
    lay.addLayout(rec)

    combo = page.combo_riemann
    syncing = [False]

    def from_combo(*_):
        text = combo.currentText()
        syncing[0] = True
        low.setChecked(text == LOW_MACH)
        low_text.setText("active" if text == LOW_MACH else "inactive")
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
    low.toggled.connect(lambda on: low_text.setText("active" if on else "inactive"))
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
