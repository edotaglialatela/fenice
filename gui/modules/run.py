"""Run module: the solver's output as it arrives, the step and time, and Stop / Kill."""
from __future__ import annotations

from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from gui.modules import ModuleSpec
from gui.project import IDLE, RUNNING, STOPPING

_MAX_LINES = 20000


class RunPanel(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.btn_run = QPushButton("Run again")
        self.btn_run.setToolTip("Run the solver again with the last parameters")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setToolTip("Stop as Ctrl+C does: the solver saves its checkpoint and index, then exits")
        self.btn_kill = QPushButton("Kill")
        self.btn_kill.setToolTip("End the solver and everything it started at once, without a checkpoint")
        self.status = QLabel("idle")
        self.status.setObjectName("note")
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(_MAX_LINES)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.text.setFont(mono)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        row = QHBoxLayout()
        for w in (self.btn_run, self.btn_stop, self.btn_kill):
            row.addWidget(w)
        row.addWidget(self.status, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(row)
        lay.addWidget(self.text, 1)
        self.btn_run.clicked.connect(self._run_again)
        self.btn_stop.clicked.connect(project.stop)
        self.btn_kill.clicked.connect(project.kill)
        project.log.connect(self._append)
        project.progress.connect(self._progress)
        project.state_changed.connect(self._state)
        project.finished.connect(self._finished)
        self._state(IDLE)

    def _run_again(self) -> None:
        self.text.clear()
        try:
            self.project.start_run()
        except RuntimeError as exc:
            self.status.setText(str(exc))

    def _append(self, line: str) -> None:
        bar = self.text.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 4
        self.text.appendPlainText(line)
        if follow:
            self.text.moveCursor(QTextCursor.MoveOperation.End)

    def _progress(self, step: int, t: float, dt: float, smax: float) -> None:
        self.status.setText(f"step {step}   t = {t:.6g}   dt = {dt:.3g}   S_max = {smax:.4g}")

    def _state(self, state: str) -> None:
        self.btn_run.setEnabled(state == IDLE and bool(self.project.params))
        self.btn_stop.setEnabled(state == RUNNING)
        self.btn_kill.setEnabled(state != IDLE)
        if state == STOPPING:
            self.status.setText("stopping: the solver is saving its checkpoint")

    def _finished(self, code: int, stopped: bool) -> None:
        how = "stopped" if stopped else "finished"
        note = "   (no final checkpoint: mpiexec aborts an MPI job on Ctrl+C; the last periodic checkpoint is kept)" if stopped and code else ""
        self.status.setText(f"{how} (exit code {code}){note}")
        self.btn_run.setEnabled(bool(self.project.params))


MODULE = ModuleSpec("run", "Run", RunPanel, order=20, requires=("PyQt6",),
                    description="The solver's output as it arrives, the step and time, Stop and Kill.")
