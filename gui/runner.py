"""The solver as a child process of the GUI (no Qt): its output line by line, a stop that does what Ctrl+C does in a terminal (the solver
saves its checkpoint and the .pvd index, then exits), and a hard stop of the whole process tree for when that does not work.

The solver runs in a console of its own (hidden on Windows, its own session elsewhere) so that the interrupt reaches it, and the
processes it starts (mpiexec and the ranks of an MPI run), without reaching the GUI. On Windows the interrupt is sent by a small
helper process (`python gui/runner.py --interrupt PID`): attaching to another console would detach the GUI from its own.
"""
from __future__ import annotations

import codecs
import os
import queue
import signal
import subprocess
import sys
import threading

IS_WINDOWS = os.name == "nt"


def _job_for(proc: "subprocess.Popen"):
    """Windows: put the (suspended) process in a job object that kills everything in it when the job handle closes, so that the solver
    and the processes it starts (mpiexec, the ranks) do not outlive the GUI if it dies. Returns the job handle, or None."""
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class Counters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

    class Extended(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("Io", Counters), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k = ctypes.windll.kernel32
    k.CreateJobObjectW.restype = ctypes.c_void_p
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    k.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    job = k.CreateJobObjectW(None, None)
    if not job:
        return None
    info = Extended()
    info.Basic.LimitFlags = 0x2000   # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) and k.AssignProcessToJobObject(job, int(proc._handle))
    if not ok:
        k.CloseHandle(ctypes.c_void_p(job))
        return None
    return job


def _close_job(job) -> None:
    import ctypes
    ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(job))   # kills whatever is still in the job


class SolverRun:
    """One run of `argv`. Output lines are collected by a reader thread; call `drain()` from the GUI thread to get them."""

    def __init__(self, argv: list[str], cwd: str | None = None, env: dict | None = None, cleanup: tuple[str, ...] = ()):
        self.argv = list(argv)
        self.cwd = cwd
        self.env = dict(os.environ if env is None else env, FORCE_COLOR="1", COLORTERM="truecolor")   # the banner in colour; the GUI strips codes to parse
        self.env.pop("NO_COLOR", None)
        self.cleanup = cleanup   # files removed once the run is over (the config is read again by every MPI process)
        self.proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._job = None

    def start(self) -> None:
        kw: dict = {}
        if IS_WINDOWS:
            # A process started from a parent that ignores Ctrl+C inherits that and never sees the interrupt: clear it first
            import ctypes
            ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            kw = {"creationflags": subprocess.CREATE_NEW_CONSOLE | 0x4, "startupinfo": si}   # 0x4: suspended until it is in the job
        else:
            kw = {"start_new_session": True}
        self.proc = subprocess.Popen(self.argv, cwd=self.cwd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kw)
        if IS_WINDOWS:
            import ctypes
            try:
                self._job = _job_for(self.proc)
            finally:   # whatever happened, the solver must run
                ctypes.windll.ntdll.NtResumeProcess(ctypes.c_void_p(int(self.proc._handle)))
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        while True:
            chunk = self.proc.stdout.read1(65536)
            if not chunk:
                break
            pending += decoder.decode(chunk)
            *done, pending = pending.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            for line in done:
                self._lines.put(line)
        if pending:
            self._lines.put(pending)
        self.proc.wait()
        if self._job:
            _close_job(self._job)   # the solver is over: anything it left behind goes with the job
            self._job = None
        for path in self.cleanup:
            try:
                os.remove(path)
            except OSError:
                pass
        self._lines.put(None)   # the end of the output

    def drain(self) -> list[str]:
        """The lines that arrived since the last call."""
        out = []
        while True:
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                return out
            if line is not None:
                out.append(line)

    @property
    def running(self) -> bool:
        return self.proc is not None and (self.proc.poll() is None or (self._reader is not None and self._reader.is_alive()))

    @property
    def returncode(self) -> int | None:
        return None if self.proc is None else self.proc.poll()

    def interrupt(self) -> None:
        """Ask the solver to stop as Ctrl+C does: it saves a checkpoint and the .pvd, then exits."""
        if self.proc is None or self.proc.poll() is not None:
            return
        if IS_WINDOWS:
            subprocess.Popen([sys.executable, os.path.abspath(__file__), "--interrupt", str(self.proc.pid)],
                             creationflags=subprocess.DETACHED_PROCESS, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        else:
            os.killpg(self.proc.pid, signal.SIGINT)

    def kill(self) -> None:
        """Stop the solver and everything it started, without a checkpoint."""
        if self.proc is None:
            return
        if self.proc.poll() is not None and not self._children_may_live():
            return
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _children_may_live(self) -> bool:
        return self._reader is not None and self._reader.is_alive()   # the pipe is still open: some process holds it


def _send_ctrl_c(pid: int) -> None:
    """Windows: send CTRL_C to the console of process `pid` (and so to everything attached to it)."""
    import ctypes
    k = ctypes.windll.kernel32
    k.FreeConsole()
    if not k.AttachConsole(pid):
        raise SystemExit(f"cannot attach to the console of process {pid}")
    k.SetConsoleCtrlHandler(None, True)   # this helper ignores the event it sends
    k.GenerateConsoleCtrlEvent(0, 0)
    k.FreeConsole()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--interrupt" and IS_WINDOWS:
        _send_ctrl_c(int(sys.argv[2]))
