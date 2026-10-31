"""The project of a solver node: the parameters of a run, its results folder, its state and its history.

A module bound to the project reads it and tells it what it changed; modules do not talk to each other. The project owns the solver
process (`gui.runner.SolverRun`), so a run goes on whichever editor is on show, and it keeps what the run printed (the log lines, the
progress lines, the physical time reached), so that a monitor linked in the middle of a run can show it from its start. In memory only:
nothing is saved or loaded yet.
"""
from __future__ import annotations

import os
import re
import tempfile
import time
from collections import deque
from pathlib import Path

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from gui.ansi import strip as strip_ansi
from gui.runner import SolverRun
from gui.solver import ROOT_DIR, solver_exe, write_config

_OUTPUT_DIR = re.compile(r"\[EXPORTER\] Output directory: '(?P<path>[^']+)'")
_PROGRESS = re.compile(r"^Step\s+(?P<step>\d+)\s*\|\s*t = (?P<t>\S+) s\s*\|\s*dt = (?P<dt>\S+) s\s*\|\s*S_max = (?P<smax>\S+) m/s(?P<rest>.*)$")
_LEAVES = re.compile(r"leaves = (\d+)")
_WALL = re.compile(r"wall = (\S+)")

IDLE, RUNNING, STOPPING = "idle", "running", "stopping"
_MAX_LOG = 20000


class Project(QObject):
    params_changed = pyqtSignal(dict)
    setup_changed = pyqtSignal()              # the setup page loaded another case: what depends on its fields is to be read again
    state_changed = pyqtSignal(str)           # idle / running / stopping
    results_changed = pyqtSignal(str)         # the results folder of the run, as soon as the solver reports it
    log = pyqtSignal(str)                     # lines of the solver output (and the notes of the workbench, '[GUI] ...')
    progress = pyqtSignal(int, float, float, float)   # step, t, dt, S_max of the last progress line
    progress_extra = pyqtSignal(int, str)             # leaves (0 if not AMR), wall-clock text of the same line
    started = pyqtSignal()                    # a run begins: before its first log line
    finished = pyqtSignal(int, bool)          # return code, stopped by the user

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params: dict = {}
        self.results_dir: Path | None = None
        self.state = IDLE
        self.editor = None     # the setup page, which takes mesh, boundary and body edits from the geometry node: apply_geometry(dict), ...
        self.monitor_every = 10  # >0: the monitor file (residuals, loads, averages) is written every this many steps, whatever the case says
        self.provider = None   # the setup page registers a function returning the live parameters (None if invalid)
        self.run_gate = None   # a function of the parameters to run returning the reason why a run cannot start (a message), or None
        # what the run has shown so far
        self.log_history: deque[str] = deque(maxlen=_MAX_LOG)
        self.progress_history: list[tuple] = []   # (step, t, dt, S_max, leaves, wall text)
        self.t_end = 0.0
        self.t_now = 0.0
        self.t_started = time.monotonic()
        self._next_mark = 10
        self._run: SolverRun | None = None
        self._stop_asked = False
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)

    # ---- parameters
    def current_params(self) -> dict | None:
        """The parameters as they are being edited (from the provider, None if invalid), else those of the last run."""
        if self.provider is not None:
            return self.provider()
        return dict(self.params) if self.params else None

    def set_params(self, params: dict) -> None:
        self.params = dict(params)
        self.params_changed.emit(self.params)

    # ---- the run
    @property
    def running(self) -> bool:
        return self.state != IDLE

    def _emit_log(self, line: str) -> None:
        self.log_history.append(line)
        self.log.emit(line)

    def note(self, text: str) -> None:
        """A line of the workbench in the console, '[GUI] ...'."""
        self._emit_log(f"[GUI] {text}")

    def advance_time(self, t: float) -> None:
        """The run has reached physical time `t`: a note each time it passes a multiple of 10 % of the end time."""
        self.t_now = max(self.t_now, t)
        if self.t_end <= 0:
            return
        reached = min(int(100.0 * self.t_now / self.t_end // 10) * 10, 100)
        if reached >= self._next_mark:
            self.note(f"{reached} % of the physical time (t = {self.t_now:.5g} of {self.t_end:g} s)")
            self._next_mark = reached + 10

    def start_run(self, params: dict | None = None) -> None:
        """Run the solver with `params` (the last ones if None). Raises RuntimeError if it cannot start, SolverNotFound if it is not built."""
        if self.running:
            raise RuntimeError("a run is already going")
        if self.run_gate is not None:
            reason = self.run_gate(params if params is not None else self.params)
            if reason:
                raise RuntimeError(reason)
        if params is not None:
            self.set_params(params)
        if not self.params:
            raise RuntimeError("no parameters to run")
        fd, cfg = tempfile.mkstemp(prefix="fenice_", suffix=".cfg")
        os.close(fd)
        run_params = dict(self.params)
        if self.monitor_every > 0:
            run_params["monitor_interval"] = self.monitor_every
        write_config(run_params, cfg)
        self.results_dir = None
        self._stop_asked = False
        self.log_history.clear()
        self.progress_history = []
        self.t_now, self._next_mark, self.t_started = 0.0, 10, time.monotonic()
        try:
            self.t_end = float(self.params.get("t_end", 0) or 0)
        except (TypeError, ValueError):
            self.t_end = 0.0
        self.started.emit()
        self.note(f"total physical time: {self.t_end:g} s" if self.t_end > 0 else "no end time set: the run ends at its step limit")
        if self.monitor_every > 0:
            self.note(f"monitor file (residuals, averages) every {self.monitor_every} steps")
        self._run = SolverRun([solver_exe(), "--config", cfg], cwd=ROOT_DIR, cleanup=(cfg,))
        self._run.start()
        self.note(f"{self._run.argv[0]} --config {cfg}")
        self._set_state(RUNNING)
        self._timer.start()

    def stop(self) -> None:
        """Stop cleanly: the solver saves its checkpoint and index, then exits (asked with a STOP file in its output folder)."""
        if self._run is not None and self.state == RUNNING:
            self._stop_asked = True
            self._set_state(STOPPING)
            if self.results_dir is not None and self.results_dir.is_dir():
                # a STOP file in the output folder is the solver's clean stop, also under MPI (where Ctrl+C aborts the job)
                (self.results_dir / "STOP").write_text("")
            else:
                self._run.interrupt()

    def kill(self) -> None:
        """Stop the solver and every process it started, without a checkpoint."""
        if self._run is not None:
            self._stop_asked = True
            self._run.kill()

    def _set_state(self, state: str) -> None:
        self.state = state
        self.state_changed.emit(state)

    def _poll(self) -> None:
        run = self._run
        for line in run.drain():
            self._line(line)
        if not run.running:
            for line in run.drain():
                self._line(line)
            self._timer.stop()
            code = run.returncode
            self._run = None
            self._set_state(IDLE)
            self.finished.emit(-1 if code is None else code, self._stop_asked)

    def _line(self, line: str) -> None:
        self._emit_log(line)
        plain = strip_ansi(line)
        m = _OUTPUT_DIR.search(plain)
        if m:
            self.results_dir = Path(m["path"])
            self.results_changed.emit(str(self.results_dir))
        m = _PROGRESS.match(plain)
        if m:
            lv, wl = _LEAVES.search(m["rest"]), _WALL.search(m["rest"])
            row = (int(m["step"]), float(m["t"]), float(m["dt"]), float(m["smax"]), int(lv[1]) if lv else 0, wl[1] if wl else "")
            self.progress_history.append(row)
            self.progress.emit(*row[:4])
            self.progress_extra.emit(row[4], row[5])
            self.advance_time(row[1])

    def shutdown(self) -> None:
        """Called when the application closes: no solver is left running."""
        self._timer.stop()
        if self._run is not None:
            self._run.kill()
