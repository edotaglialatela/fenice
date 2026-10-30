"""The Fenice workbench: one window in which the steps of a study are modules around a shared project.

    python gui/workbench.py

A navigation list on the left (Setup, Run, Results, and whatever else sits in ``gui/modules``), the module on show in the middle,
the state of the run below. Run Simulation in the Setup module starts the solver as a child process of the project, the Run module
shows its output, and the Results module follows the folder it writes. ``gui/run.py`` (the dialog, then the solver in the terminal)
is unchanged.
"""
from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,  # noqa: E402
                             QStackedWidget, QVBoxLayout, QWidget)

from gui.modules import ModuleSpec, discover  # noqa: E402
from gui.project import IDLE, Project  # noqa: E402
from gui.solver import SolverNotFound, available_cases  # noqa: E402
from gui.theme import apply_theme  # noqa: E402


class Workbench(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fenice")
        self.resize(1360, 860)
        self.project = Project(self)
        self.nav = QListWidget()
        self.nav.setFixedWidth(150)
        self.stack = QStackedWidget()
        self.panels: dict[str, QWidget] = {}
        self._rows: dict[str, int] = {}
        self.state_label = QLabel("idle")
        self.state_label.setObjectName("note")

        body = QWidget()
        lay = QHBoxLayout(body)
        lay.addWidget(self.nav)
        lay.addWidget(self.stack, 1)
        outer = QWidget()
        col = QVBoxLayout(outer)
        col.addWidget(body, 1)
        col.addWidget(self.state_label)
        self.setCentralWidget(outer)

        for spec in discover():
            self._add_module(spec)
        self.nav.currentRowChanged.connect(self._select)
        first = next((r for r in range(self.nav.count()) if self.nav.item(r).flags() & Qt.ItemFlag.ItemIsEnabled), -1)
        self.nav.setCurrentRow(first)

        self.project.state_changed.connect(self._state)
        self.project.progress.connect(lambda step, t, dt, smax: self.state_label.setText(f"running   step {step}   t = {t:.6g}"))
        self.project.finished.connect(self._finished)
        self.project.results_changed.connect(lambda folder: self.statusBar().showMessage(f"results: {folder}"))
        self.project.state_changed.connect(lambda s: s == "running" and self.show_module("run"))

    # ---- modules
    def _add_module(self, spec: ModuleSpec) -> None:
        item = QListWidgetItem(spec.title)
        item.setToolTip(spec.description)
        missing = spec.missing() if spec.factory is not None else ["(see the tooltip)"]
        panel = None
        if not missing:
            try:
                panel = spec.factory(self.project)
            except Exception as exc:   # noqa: BLE001 - one module failing must not take the others down
                missing = [f"{type(exc).__name__}: {exc}"]
        if panel is None:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            item.setToolTip(f"{spec.description}\nUnavailable: {', '.join(missing)}")
        else:
            self.panels[spec.name] = panel
            self._rows[spec.name] = self.nav.count()
            self.stack.addWidget(panel)
            if hasattr(panel, "title_changed"):
                panel.title_changed.connect(lambda title: self.statusBar().showMessage(title))
        self.nav.addItem(item)

    def _select(self, row: int) -> None:
        for name, r in self._rows.items():
            if r == row:
                self.stack.setCurrentWidget(self.panels[name])

    def show_module(self, name: str) -> None:
        if name in self._rows:
            self.nav.setCurrentRow(self._rows[name])

    # ---- the run
    def _state(self, state: str) -> None:
        self.state_label.setText(state)

    def _finished(self, code: int, stopped: bool) -> None:
        self.state_label.setText(("stopped" if stopped else "finished") + f" (exit code {code})")

    def closeEvent(self, event) -> None:
        if self.project.state != IDLE:
            box = QMessageBox(self)
            box.setWindowTitle("Fenice")
            box.setText("A run is going.")
            stop = box.addButton("Stop it (save the checkpoint)", QMessageBox.ButtonRole.AcceptRole)
            kill = box.addButton("Kill it", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("Keep the window", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is stop:
                self.project.stop()
                event.ignore()   # the window closes when the solver has exited
                self.project.finished.connect(lambda *_: self.close())
                return
            if box.clickedButton() is kill:
                self.project.kill()
            else:
                event.ignore()
                return
        for panel in self.panels.values():
            if hasattr(panel, "shutdown"):
                panel.shutdown()
        self.project.shutdown()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv[:1])
    apply_theme(app)
    try:
        available_cases()
    except SolverNotFound as exc:
        QMessageBox.critical(None, "Fenice", str(exc))
        return 1
    win = Workbench()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
