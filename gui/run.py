"""Fenice entry point: setup dialog, then the native C++ solver.

    python gui/run.py

Shows :class:`gui.setup_dialog.SetupDialog`, writes the validated parameters to a
temporary config file and runs ``fenice --config <file>`` in this terminal.
The solver re-launches itself under ``mpiexec`` when MPI is enabled; Ctrl+C
reaches it and it saves a checkpoint and the ``.pvd`` index before stopping.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from gui.setup_dialog import SetupDialog  # noqa: E402
from gui.solver import SolverNotFound, available_cases, solver_exe, write_config  # noqa: E402
from gui.theme import apply_theme  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    apply_theme(app)
    try:
        available_cases()
    except SolverNotFound as exc:
        QMessageBox.critical(None, "Fenice", str(exc))
        return 1

    dialog = SetupDialog()
    if not dialog.exec():
        return 0
    params = dialog.sim_params
    app.quit()

    fd, cfg = tempfile.mkstemp(prefix="fenice_", suffix=".cfg")
    os.close(fd)
    write_config(params, cfg)
    print(f"[GUI] Parameters written to {cfg}")
    try:
        proc = subprocess.Popen([solver_exe(), "--config", cfg], cwd=ROOT_DIR)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            return proc.wait()   # the solver got the same Ctrl+C and is saving its checkpoint
    finally:
        try:
            os.remove(cfg)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
