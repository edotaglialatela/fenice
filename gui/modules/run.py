"""Run module: statistics of the run above, the solver's console below, Stop and Kill.

The curves come from what the solver already writes: the progress lines (every `save_interval` steps: time step, wave speed, number of
leaves of an AMR run) and the monitor CSV (every `monitor_interval` steps, when the case sets one). They are not residuals.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget

from gui import ansi
from gui.modules import ModuleSpec
from gui.project import IDLE, RUNNING, STOPPING

_MAX_LINES = 20000
_CONSOLE_BG, _CONSOLE_FG, _CONSOLE_DIM = "#1b1d23", "#d8dae0", "#8a8f9c"   # the banner colours are made for a dark terminal


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


class RunPanel(QWidget):
    def __init__(self, project, parent=None):
        super().__init__(parent)
        self.project = project
        self.btn_run = QPushButton("Run again")
        self.btn_run.setToolTip("Run the solver with what the Setup page holds now")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setToolTip("Stop cleanly: the solver saves its checkpoint and index, then exits (serial or MPI)")
        self.btn_kill = QPushButton("Kill")
        self.btn_kill.setToolTip("End the solver and everything it started at once, without a checkpoint")
        self.status = QLabel("idle")
        self.status.setObjectName("note")
        self.cb_quantity = QComboBox()
        self.cb_quantity.setToolTip("The quantity of the lower plot: a column of the monitor file, or the number of leaves of an AMR run")
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(_MAX_LINES)
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.text.setFont(mono)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setStyleSheet(f"QPlainTextEdit {{ background: {_CONSOLE_BG}; color: {_CONSOLE_FG}; border: 1px solid #3a3d46; }}")

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from gui.theme import COLORS
        self._colors = COLORS
        self.fig = Figure(figsize=(8, 4))
        self.fig.set_facecolor(COLORS["bg_window"])
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax_dt, self.ax_s, self.ax_q = self.fig.subplots(3, 1, sharex=True)
        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.1, hspace=0.12)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.canvas)
        split.addWidget(self.text)
        split.setChildrenCollapsible(False)
        self.split = split

        row = QHBoxLayout()
        for w in (self.btn_run, self.btn_stop, self.btn_kill):
            row.addWidget(w)
        row.addWidget(QLabel("Lower plot"))
        row.addWidget(self.cb_quantity)
        row.addWidget(self.status, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(row)
        lay.addWidget(split, 1)

        self._reset_data()
        self._dirty = False
        self._timer = QTimer(self)   # the plots are redrawn about once a second, not on every line
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self.btn_run.clicked.connect(self._run_again)
        self.btn_stop.clicked.connect(project.stop)
        self.btn_kill.clicked.connect(project.kill)
        self.cb_quantity.currentIndexChanged.connect(lambda _: self._redraw())
        project.started.connect(self._started)
        project.log.connect(self._append)
        project.progress.connect(self._progress)
        project.progress_extra.connect(self._progress_extra)
        project.state_changed.connect(self._state)
        project.finished.connect(self._finished)
        self._state(IDLE)

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_halved", False):   # the first time: plots and console share the height equally
            self._halved = True
            h = self.split.height()
            self.split.setSizes([h // 2, h - h // 2])

    # ---- the run
    def _reset_data(self) -> None:
        self.t_hist: list[float] = []
        self.dt_hist: list[float] = []
        self.s_hist: list[float] = []
        self.leaf_t: list[float] = []
        self.leaf_hist: list[int] = []
        self.monitor = None
        self.t_end = 0.0
        self.t_now = 0.0
        self.next_mark = 10
        self.t0 = time.monotonic()
        self.wall = ""

    def _run_again(self) -> None:
        params = self.project.current_params()
        if params is None:
            self.status.setText("the setup has an input that is not valid: correct it in Setup first")
            return
        try:
            self.project.start_run(params)
        except RuntimeError as exc:
            self.status.setText(str(exc))

    def _started(self) -> None:
        """A fresh run, however it was started: the console and the plots begin empty."""
        self.text.clear()
        self._reset_data()
        try:
            self.t_end = float(self.project.params.get("t_end", 0) or 0)
        except (TypeError, ValueError):
            self.t_end = 0.0
        if self.t_end > 0:
            self._note(f"total physical time: {self.t_end:g} s")
        else:
            self._note("no end time set: the run ends at its step limit")
        self.cb_quantity.blockSignals(True)
        self.cb_quantity.clear()
        self.cb_quantity.blockSignals(False)
        self._redraw()
        self._timer.start()

    def _note(self, text: str) -> None:
        self._append(f"[GUI] {text}")

    # ---- the console
    def _append(self, line: str) -> None:
        bar = self.text.verticalScrollBar()
        follow = bar.value() >= bar.maximum() - 4
        cur = self.text.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        if not self.text.document().isEmpty():
            cur.insertBlock()
        for chunk, style in ansi.spans(line.replace("\r", "")):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(style.get("color", _CONSOLE_DIM if style.get("dim") else _CONSOLE_FG)))
            if style.get("bold"):
                fmt.setFontWeight(QFont.Weight.Bold)
            cur.insertText(chunk, fmt)
        if follow:
            self.text.moveCursor(QTextCursor.MoveOperation.End)

    # ---- progress
    def _progress(self, step: int, t: float, dt: float, smax: float) -> None:
        self.t_now = max(self.t_now, t)
        self.t_hist.append(t)
        self.dt_hist.append(dt)
        self.s_hist.append(smax)
        self._dirty = True
        self._marks()
        self._status(step)

    def _progress_extra(self, leaves: int, wall: str) -> None:
        self.wall = wall
        if leaves and self.t_hist:
            self.leaf_t.append(self.t_hist[-1])
            self.leaf_hist.append(leaves)

    def _marks(self) -> None:
        """A note in the console each time the run passes a multiple of 10 % of the physical time."""
        if self.t_end <= 0:
            return
        pct = 100.0 * self.t_now / self.t_end
        reached = min(int(pct // 10) * 10, 100)
        if reached >= self.next_mark:
            self._note(f"{reached} % of the physical time (t = {self.t_now:.5g} of {self.t_end:g} s)")
            self.next_mark = reached + 10

    def _status(self, step: int | None = None) -> None:
        bits = []
        if step is not None:
            bits.append(f"step {step}")
        bits.append(f"t = {self.t_now:.6g}" + (f" / {self.t_end:g} s" if self.t_end > 0 else ""))
        if self.t_end > 0 and self.t_now > 0:
            f = min(self.t_now / self.t_end, 1.0)
            bits.append(f"{100 * f:.0f} %")
            elapsed = time.monotonic() - self.t0
            if 0 < f < 1:
                bits.append(f"about {_clock(elapsed * (1 - f) / f)} left")
        if self.t_hist:
            bits.append(f"dt = {self.dt_hist[-1]:.3g}   S_max = {self.s_hist[-1]:.4g}")
        if self.leaf_hist:
            bits.append(f"{self.leaf_hist[-1]:,} leaves")
        if self.wall:
            bits.append(f"wall {self.wall}")
        self.status.setText("   ".join(bits))

    # ---- the plots
    def _read_monitor(self) -> None:
        d = self.project.results_dir
        if d is None:
            return
        try:
            from gui.viewer_data import Series
            ser = getattr(self, "_series", None)
            if ser is None or Path(ser.folder) != Path(d):
                self._series = ser = Series(Path(d))
            ser._monitor = ser._read_monitor()
            m = ser.monitor()
        except Exception:   # noqa: BLE001 - a half-written file: the next tick reads it again
            return
        if m is not None and m.size > 1 and "t" in m.dtype.names:
            self.monitor = m
            self.t_now = max(self.t_now, float(m["t"][-1]))
            self._marks()

    def _tick(self) -> None:
        if self.project.running:
            self._read_monitor()
            self._fill_choices()
            self._status()
            self._redraw()
        elif self._dirty:
            self._redraw()
            self._timer.stop()
        else:
            self._timer.stop()

    def _fill_choices(self) -> None:
        names = []
        if self.monitor is not None:
            names += [n for n in self.monitor.dtype.names if n not in ("step", "t")]
        if self.leaf_hist:
            names.append("leaves")
        have = [self.cb_quantity.itemText(i) for i in range(self.cb_quantity.count())]
        if names != have:
            keep = self.cb_quantity.currentText()
            self.cb_quantity.blockSignals(True)
            self.cb_quantity.clear()
            self.cb_quantity.addItems(names)
            if keep in names:
                self.cb_quantity.setCurrentText(keep)
            self.cb_quantity.blockSignals(False)

    def _redraw(self) -> None:
        c = self._colors
        for ax in (self.ax_dt, self.ax_s, self.ax_q):
            ax.clear()
            ax.set_facecolor(c["bg_surface"])
            ax.grid(True, color=c["border_strong"], lw=0.4)
            ax.tick_params(labelsize=8)
        if self.t_hist:
            self.ax_dt.plot(self.t_hist, self.dt_hist, color=c["primary"], lw=1.2, marker="." if len(self.t_hist) < 60 else None)
            self.ax_dt.set_yscale("log")
            self.ax_s.plot(self.t_hist, self.s_hist, color=c["primary"], lw=1.2, marker="." if len(self.t_hist) < 60 else None)
        self.ax_dt.set_ylabel("dt [s]", fontsize=8)
        self.ax_s.set_ylabel("S_max [m/s]", fontsize=8)
        q = self.cb_quantity.currentText()
        if q == "leaves" and self.leaf_hist:
            self.ax_q.plot(self.leaf_t, self.leaf_hist, color=c["primary"], lw=1.2, marker=".")
        elif q and self.monitor is not None and q in self.monitor.dtype.names:
            y = np.asarray(self.monitor[q], dtype=float)
            if q.startswith("res_"):   # residual norms: relative to the first row that is not zero, on a log axis
                ref = next((v for v in y if v > 0), 0.0)
                if ref > 0:
                    y = np.where(y > 0, y / ref, np.nan)
                    self.ax_q.set_yscale("log")
                    q = q + " / first"
            self.ax_q.plot(self.monitor["t"], y, color=c["primary"], lw=1.2)
        self.ax_q.set_ylabel(q or "monitor", fontsize=8)
        self.ax_q.set_xlabel("t [s]", fontsize=8)
        if self.t_end > 0:
            self.ax_q.set_xlim(0, self.t_end)
        if not self.t_hist and self.monitor is None:
            self.ax_s.text(0.5, 0.5, "the plots fill as the run writes its progress lines (every save_interval steps)\nand the monitor file",
                           transform=self.ax_s.transAxes, ha="center", va="center", fontsize=9, color=c["ink_soft"])
        self._dirty = False
        self.canvas.draw_idle()

    # ---- states
    def _state(self, state: str) -> None:
        self.btn_run.setEnabled(state == IDLE and (bool(self.project.params) or self.project.provider is not None))
        self.btn_stop.setEnabled(state == RUNNING)
        self.btn_kill.setEnabled(state != IDLE)
        if state == STOPPING:
            self.status.setText("stopping: the solver is saving its checkpoint")

    def _finished(self, code: int, stopped: bool) -> None:
        how = "stopped" if stopped else "finished"
        note = "   (the solver did not exit cleanly: the last periodic checkpoint is the one kept)" if stopped and code else ""
        self._read_monitor()
        self._fill_choices()
        self._redraw()
        self.status.setText(f"{how} (exit code {code}){note}")
        self.btn_run.setEnabled(True)
        self._timer.stop()


MODULE = ModuleSpec("run", "Run", RunPanel, order=20, requires=("PyQt6", "matplotlib"),
                    description="Plots of the run (time step, wave speed, monitor quantities), the solver's console, Stop and Kill.")
