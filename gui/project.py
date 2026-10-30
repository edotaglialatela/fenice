"""The project of the workbench: the parameters of a run, its results folder and its state, shared by the modules.

A module reads the project and tells it what it changed; modules do not talk to each other. The project owns the solver process
(`gui.runner.SolverRun`), so a run goes on whichever module is on show. In memory only: nothing is saved or loaded yet.
"""
from __future__ import annotations

import os
import re
import tempfile
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


class Project(QObject):
    params_changed = pyqtSignal(dict)
    state_changed = pyqtSignal(str)           # idle / running / stopping
    results_changed = pyqtSignal(str)         # the results folder of the run, as soon as the solver reports it
    log = pyqtSignal(str)                     # lines of the solver output
    progress = pyqtSignal(int, float, float, float)   # step, t, dt, S_max of the last progress line
    progress_extra = pyqtSignal(int, str)             # leaves (0 if not AMR), wall-clock text of the same line
    started = pyqtSignal()                    # a run begins: before its first log line
    finished = pyqtSignal(int, bool)          # return code, stopped by the user

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params: dict = {}
        self.results_dir: Path | None = None
        self.state = IDLE
        self.editor = None     # the module that can take mesh and boundary edits from another page (Setup): apply_geometry(dict), bc_choices()
        self.monitor_every = 0   # >0: the monitor file (residuals, loads, averages) is written every this many steps, whatever the case says
        self.provider = None   # a module that holds the live parameters (Setup) registers a function returning them, or None
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

    def start_run(self, params: dict | None = None) -> None:
        """Run the solver with `params` (the last ones if None). Raises SolverNotFound if it is not built."""
        if self.running:
            raise RuntimeError("a run is already going")
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
        self.started.emit()
        self._run = SolverRun([solver_exe(), "--config", cfg], cwd=ROOT_DIR, cleanup=(cfg,))
        self._run.start()
        self.log.emit(f"[GUI] {self._run.argv[0]} --config {cfg}")
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
            self.log.emit(line)
            line = strip_ansi(line)
            m = _OUTPUT_DIR.search(line)
            if m:
                self.results_dir = Path(m["path"])
                self.results_changed.emit(str(self.results_dir))
            m = _PROGRESS.match(line)
            if m:
                self.progress.emit(int(m["step"]), float(m["t"]), float(m["dt"]), float(m["smax"]))
                lv, wl = _LEAVES.search(m["rest"]), _WALL.search(m["rest"])
                self.progress_extra.emit(int(lv[1]) if lv else 0, wl[1] if wl else "")
        if not run.running:
            for line in run.drain():
                self.log.emit(line)
            self._timer.stop()
            code = run.returncode
            self._run = None
            self._set_state(IDLE)
            self.finished.emit(-1 if code is None else code, self._stop_asked)

    def shutdown(self) -> None:
        """Called when the application closes: no solver is left running."""
        self._timer.stop()
        if self._run is not None:
            self._run.kill()
