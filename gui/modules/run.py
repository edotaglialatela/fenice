"""Monitor module: statistics of the run of the solver it is linked to above, the solver's console below, Stop and Kill.

The curves come from what the solver writes: the progress lines (every `save_interval` steps: time step, wave speed, number of leaves of
an AMR run) and the monitor CSV (every N steps, N set on the solver node: residual norms, averages, loads). The run's history lives in the
project, so a monitor linked in the middle of a run shows it from its start.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSplitter, QVBoxLayout, QWidget

from gui import ansi
from gui.modules import ModuleSpec, Port
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
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setToolTip("Stop cleanly: the solver saves its checkpoint and index, then exits (serial or MPI)")
        self.btn_kill = QPushButton("Kill")
        self.btn_kill.setToolTip("End the solver and everything it started at once, without a checkpoint")
        self.status = QLabel("idle")
        self.status.setObjectName("note")
        self.cb_quantity = QComboBox()
        self.cb_quantity.setToolTip("The quantity of the lowest plot: a column of the monitor file, or the number of leaves of an AMR run")
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
        self.ax_dt, self.ax_s, self.ax_r, self.ax_q = self.fig.subplots(4, 1, sharex=True)
        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.09, hspace=0.14)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.canvas)
        split.addWidget(self.text)
        split.setChildrenCollapsible(False)
        self.split = split

        row = QHBoxLayout()
        for w in (self.btn_stop, self.btn_kill):
            row.addWidget(w)
        row.addWidget(QLabel("Lowest plot"))
        row.addWidget(self.cb_quantity)
        row.addWidget(self.status, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(row)
        lay.addWidget(split, 1)

        self.monitor = None
        self._series = None
        self._shown = 0   # lines of project.log_history already in the console
        self._timer = QTimer(self)   # the plots are redrawn about once a second, not on every line
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self.btn_stop.clicked.connect(project.stop)
        self.btn_kill.clicked.connect(project.kill)
        self.cb_quantity.currentIndexChanged.connect(lambda _: self._redraw())
        project.started.connect(self._started)
        project.log.connect(self._on_line)
        project.progress.connect(self._on_progress)
        project.state_changed.connect(self._state)
        project.finished.connect(self._finished)
        self._state(project.state)
        self._replay()

    def showEvent(self, event):
        super().showEvent(event)
        if not getattr(self, "_halved", False):   # the first time: plots and console share the height equally
            self._halved = True
            h = self.split.height()
            self.split.setSizes([h // 2, h - h // 2])

    # ---- from the history of the project: also what a monitor linked in the middle of a run missed
    def _replay(self) -> None:
        """Fill the console and the plots with what the run has shown so far."""
        self.text.clear()
        self._shown = 0
        for line in list(self.project.log_history):
            self._append(line)
            self._shown += 1
        self._read_monitor()
        self._fill_choices()
        self._status()
        self._redraw()
        if self.project.running:
            self._timer.start()

    def _started(self) -> None:
        """A fresh run: the console and the plots begin empty."""
        self.text.clear()
        self._shown = 0
        self.monitor = None
        self._series = None
        self.cb_quantity.blockSignals(True)
        self.cb_quantity.clear()
        self.cb_quantity.blockSignals(False)
        self._redraw()
        self._timer.start()

    # ---- the console
    def _on_line(self, line: str) -> None:
        self._shown += 1
        self._append(line)

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
    def _on_progress(self, *_) -> None:
        self._status()

    def _status(self) -> None:
        pr = self.project
        bits = []
        if pr.progress_history:
            step, _, dt, smax, leaves, wall = pr.progress_history[-1]
            bits.append(f"step {step}")
        bits.append(f"t = {pr.t_now:.6g}" + (f" / {pr.t_end:g} s" if pr.t_end > 0 else ""))
        if pr.t_end > 0 and pr.t_now > 0:
            f = min(pr.t_now / pr.t_end, 1.0)
            bits.append(f"{100 * f:.0f} %")
            elapsed = time.monotonic() - pr.t_started
            if 0 < f < 1 and pr.running:
                bits.append(f"about {_clock(elapsed * (1 - f) / f)} left")
        if pr.progress_history:
            bits.append(f"dt = {dt:.3g}   S_max = {smax:.4g}")
            if leaves:
                bits.append(f"{leaves:,} leaves")
            if wall:
                bits.append(f"wall {wall}")
        self.status.setText("   ".join(bits) if pr.progress_history or pr.t_now else self.project.state)

    # ---- the plots
    def _read_monitor(self) -> None:
        d = self.project.results_dir
        if d is None:
            return
        try:
            from gui.viewer_data import Series
            ser = self._series
            if ser is None or Path(ser.folder) != Path(d):
                self._series = ser = Series(Path(d))
            ser._monitor = ser._read_monitor()
            m = ser.monitor()
        except Exception:   # noqa: BLE001 - a half-written file: the next tick reads it again
            return
        if m is not None and m.size > 1 and "t" in m.dtype.names:
            self.monitor = m
            self.project.advance_time(float(m["t"][-1]))

    def _tick(self) -> None:
        if self.project.running:
            self._read_monitor()
            self._fill_choices()
            self._status()
            self._redraw()
        else:
            self._timer.stop()

    def _fill_choices(self) -> None:
        names = []
        if self.monitor is not None:
            names += [n for n in self.monitor.dtype.names if n not in ("step", "t")]
        if any(row[4] for row in self.project.progress_history):
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
        c, pr = self._colors, self.project
        hist = pr.progress_history
        for ax in (self.ax_dt, self.ax_s, self.ax_r, self.ax_q):
            ax.clear()
            ax.set_facecolor(c["bg_surface"])
            ax.grid(True, color=c["border_strong"], lw=0.4)
            ax.tick_params(labelsize=8)
        if hist:
            t = [r[1] for r in hist]
            mark = "." if len(t) < 60 else None
            self.ax_dt.plot(t, [r[2] for r in hist], color=c["primary"], lw=1.2, marker=mark)
            self.ax_dt.set_yscale("log")
            self.ax_s.plot(t, [r[3] for r in hist], color=c["primary"], lw=1.2, marker=mark)
        self._draw_residuals()
        self.ax_dt.set_ylabel("dt [s]", fontsize=8)
        self.ax_s.set_ylabel("S_max [m/s]", fontsize=8)
        q = self.cb_quantity.currentText()
        if q == "leaves":
            self.ax_q.plot([r[1] for r in hist if r[4]], [r[4] for r in hist if r[4]], color=c["primary"], lw=1.2, marker=".")
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
        if pr.t_end > 0:
            self.ax_q.set_xlim(0, pr.t_end)
        if not hist:
            self.ax_s.text(0.5, 0.5, "dt and S_max come from the progress lines (every save_interval steps)",
                           transform=self.ax_s.transAxes, ha="center", va="center", fontsize=8, color=c["ink_soft"])
        self.canvas.draw_idle()

    def _draw_residuals(self) -> None:
        """The three residual norms of the monitor file, each relative to its first non-zero value, on one log axis."""
        ax, c, m = self.ax_r, self._colors, self.monitor
        drawn = False
        if m is not None and "res_rho" in m.dtype.names:
            for name, colour in (("res_rho", c["primary"]), ("res_mom", c["ink"]), ("res_E", c["ink_soft"])):
                y = np.asarray(m[name], dtype=float)
                ref = next((v for v in y if v > 0), 0.0)
                if ref > 0:
                    ax.plot(m["t"], np.where(y > 0, y / ref, np.nan), color=colour, lw=1.2, label=name.replace("res_", ""))
                    drawn = True
        if drawn:
            ax.set_yscale("log")
            ax.legend(loc="upper right", fontsize=7, ncol=3, frameon=False)
        else:
            ax.text(0.5, 0.5, "residuals appear once the monitor file has rows (set 'Record every' on the solver)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=c["ink_soft"])
        ax.set_ylabel("residual / first", fontsize=8)

    # ---- states
    def _state(self, state: str) -> None:
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
        self._timer.stop()

    def shutdown(self) -> None:
        self._timer.stop()


MODULE = ModuleSpec("monitor", "Monitor", RunPanel, order=30, requires=("PyQt6", "matplotlib", "numpy"),
                    inputs=(Port("run", "run"),),
                    description="Live plots of the run (time step, wave speed, residuals, monitor quantities) and the solver's console.")
