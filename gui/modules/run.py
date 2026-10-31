"""Monitor module: statistics of the run of the solver it is linked to above, the solver's console below, Stop and Kill.

The curves come from what the solver writes: the progress lines (every `save_interval` steps: time step, wave speed, number of leaves of
an AMR run) and the monitor CSV (every N steps, N set on the solver node: residual norms, averages, loads). The run's history lives in the
project, so a monitor linked in the middle of a run shows it from its start.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget

from gui import ansi
from gui.modules import ModuleSpec, Port
from gui.project import IDLE, RUNNING, STOPPING

import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "cm"   # the math of the labels in the Computer Modern of TeX, without needing a TeX installation

_MAX_LINES = 20000
_CONSOLE_BG, _CONSOLE_FG, _CONSOLE_DIM = "#1b1d23", "#d8dae0", "#8a8f9c"   # the banner colours are made for a dark terminal


def _tick_format(v, _pos=None) -> str:
    """A number for an axis or a corner: three significant figures, and a power of ten as mathtext when it is far from 1."""
    if not np.isfinite(v):
        return ""
    if v == 0:
        return "0"
    text = f"{v:.3g}"
    if "e" in text:
        mant, exp = text.split("e")
        text = (f"{mant}\\times" if float(mant) != 1 else "") + f"10^{{{int(exp)}}}"
        return f"${text}$"
    return text


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
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)   # a narrow column: long lines wrap, no sideways scrolling
        self.text.setStyleSheet(f"QPlainTextEdit {{ background: {_CONSOLE_BG}; color: {_CONSOLE_FG}; border: 1px solid #3a3d46; }}")

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from gui.theme import COLORS
        self._colors = COLORS
        self.fig = Figure(figsize=(8, 5))
        self.fig.set_facecolor(COLORS["bg_window"])
        self.canvas = FigureCanvasQTAgg(self.fig)
        # the residuals are what a run is judged by: the largest panel; the time step and the wave speed the smallest
        self.ax_dt, self.ax_s, self.ax_r, self.ax_q = self.fig.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [1, 1, 2.4, 1.4]})
        self.fig.subplots_adjust(left=0.065, right=0.985, top=0.985, bottom=0.085, hspace=0.07)
        self.console = self.text   # shown by the workbench in its right-hand column, the full height of the window

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.status.setStyleSheet("font-size: 10.5pt;")
        row = QHBoxLayout()
        for w in (self.btn_stop, self.btn_kill):
            row.addWidget(w)
        row.addWidget(QLabel("Lowest plot"))
        row.addWidget(self.cb_quantity)
        row.addWidget(self.status, 1)
        lay = QVBoxLayout(self)
        lay.addLayout(row)
        lay.addWidget(self.progress)
        lay.addWidget(self.canvas, 1)

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
        line = line.replace("\r", "")
        spans = ansi.spans(line) if "\x1b" in line else ansi.colorize(line)   # the solver's own colours, or ours for its plain lines
        for chunk, style in spans:
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
        self.progress.setValue(int(1000 * min(pr.t_now / pr.t_end, 1.0)) if pr.t_end > 0 else 0)

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

    def _prepare(self, ax, label: str, last=None, bottom: bool = False, nbins: int = 3) -> None:
        """A clean panel: the name inside the top-left corner, the latest value inside the top-right one, few ticks, no frame on two sides."""
        from matplotlib.ticker import MaxNLocator, NullFormatter
        c = self._colors
        ax.clear()
        ax.set_facecolor(c["bg_surface"])
        ax.grid(True, color=c["border_strong"], lw=0.35, alpha=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=7, length=2.5, pad=2, labelbottom=bottom)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=2))
        ax.yaxis.set_major_formatter(_tick_format)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.text(0.008, 0.94, label, transform=ax.transAxes, ha="left", va="top", fontsize=9, color=c["ink"], zorder=5)
        if last is not None and np.isfinite(last):
            ax.text(0.992, 0.94, _tick_format(last, None), transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                    color=c["primary"], fontweight="bold", zorder=5)

    @staticmethod
    def _fit(ax, y) -> None:
        """The vertical range of what is drawn, with room for the labels in the corner above; a nearly constant quantity is not magnified."""
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return
        lo, hi = float(y.min()), float(y.max())
        mid = 0.5 * (lo + hi)
        span = max(hi - lo, 0.02 * abs(mid), 1e-300)
        ax.set_ylim(mid - 0.5 * span - 0.12 * span, mid + 0.5 * span + 0.30 * span)

    def _redraw(self) -> None:
        c, pr = self._colors, self.project
        hist = pr.progress_history
        t = np.array([r[1] for r in hist], dtype=float)
        dt = np.array([r[2] for r in hist], dtype=float)
        t_dt = t   # the times of the dt curve: those of the progress lines, or of the monitor file when it is there
        m = self.monitor
        if m is not None and len(m) > 2 and "step" in m.dtype.names:   # the mean time step between two rows of the monitor file: live at its pace
            steps, times = np.asarray(m["step"], dtype=float), np.asarray(m["t"], dtype=float)
            ds = np.diff(steps)
            ok = ds > 0
            if ok.any():
                t_dt, dt = times[1:][ok], (np.diff(times)[ok] / ds[ok])
        smax = np.array([r[3] for r in hist], dtype=float)
        mark = "." if len(t) < 60 else None
        mark_dt = "." if len(t_dt) < 60 else None
        self._prepare(self.ax_dt, r"$\Delta t$  [s]", dt[-1] if len(dt) else None, nbins=2)
        self._prepare(self.ax_s, r"$S_{\max}$  [m/s]", smax[-1] if len(smax) else None, nbins=2)
        if len(t_dt):
            self.ax_dt.plot(t_dt, dt, color=c["primary"], lw=1.2, marker=mark_dt, ms=3)
            self._fit(self.ax_dt, dt)
        else:
            self._note(self.ax_dt, "the time step comes from the monitor file or the progress lines")
        if len(t):
            self.ax_s.plot(t, smax, color=c["primary"], lw=1.2, marker=mark, ms=3)
            self._fit(self.ax_s, smax)
        else:
            self._note(self.ax_s, "the wave speed comes from the progress lines (every save_interval steps)")
        self._draw_residuals()
        q = self.cb_quantity.currentText()
        label, y, tq, log = q or "monitor", None, None, False
        if q == "leaves":
            tq = np.array([r[1] for r in hist if r[4]], dtype=float)
            y = np.array([r[4] for r in hist if r[4]], dtype=float)
            label = "leaves"
        elif q and self.monitor is not None and q in self.monitor.dtype.names:
            tq, y = np.asarray(self.monitor["t"], dtype=float), np.asarray(self.monitor[q], dtype=float)
        self._prepare(self.ax_q, label, y[-1] if y is not None and len(y) else None, bottom=True)
        if y is not None and len(y):
            self.ax_q.plot(tq, y, color=c["primary"], lw=1.2)
            self._fit(self.ax_q, y)
        self.ax_q.set_xlabel(r"$t$  [s]", fontsize=8, labelpad=1)
        if pr.t_end > 0:
            self.ax_q.set_xlim(0, pr.t_end)
        self.canvas.draw_idle()

    def _note(self, ax, text: str) -> None:
        ax.text(0.5, 0.45, text, transform=ax.transAxes, ha="center", va="center", fontsize=8, color=self._colors["ink_soft"])

    def _draw_residuals(self) -> None:
        """The three residual norms of the monitor file, each relative to its first non-zero value, on one log axis (decades only)."""
        from matplotlib.ticker import LogLocator, NullFormatter
        c, m = self._colors, self.monitor
        curves = []
        if m is not None and "res_rho" in m.dtype.names:
            for name, colour, label in (("res_rho", c["primary"], r"$\rho$"), ("res_mom", c["ink"], r"$\rho\mathbf{u}$"),
                                        ("res_E", c["ink_soft"], r"$\rho E$")):
                y = np.asarray(m[name], dtype=float)
                ref = next((v for v in y if v > 0), 0.0)
                if ref > 0:
                    curves.append((np.where(y > 0, y / ref, np.nan), colour, label))
        last = curves[0][0][np.isfinite(curves[0][0])][-1] if curves and np.isfinite(curves[0][0]).any() else None
        self._prepare(self.ax_r, r"$R\,/\,R_0$", last)
        ax = self.ax_r
        if not curves:
            self._note(ax, "residuals appear once the monitor file has rows (set 'Record every' on the solver)")
            return
        for y, colour, label in curves:
            ax.plot(m["t"], y, color=colour, lw=1.3, label=label)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(), numticks=5))
        ax.yaxis.set_minor_formatter(NullFormatter())
        finite = np.concatenate([y[np.isfinite(y)] for y, _, _ in curves])
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            ax.set_ylim(10 ** np.floor(np.log10(lo) - 0.05), 10 ** (np.ceil(np.log10(hi)) + 0.45))   # whole decades, room for the corner text
        leg = ax.legend(loc="upper center", fontsize=10, ncol=3, frameon=True, handlelength=1.6, columnspacing=1.4, borderaxespad=0.25,
                        borderpad=0.3, handletextpad=0.5)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.65)   # a translucent white behind the text only, so that the curves below stay visible
        leg.get_frame().set_edgecolor("none")

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
