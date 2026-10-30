"""Fenice result viewer: the fields of a run, snapshot by snapshot, and its monitor curves.

    python gui/viewer.py [results/<case> | a snapshot file]
    python gui/viewer.py results/<case> --screenshot out.png [--field Mach] [--step 100] [--size 1100x700]

Reads the .vtr / .vtu / .pvtu snapshots of a results folder (found from the files, so a run in progress can be followed: the folder is
rescanned every second) and the monitor CSV. A uniform mesh is drawn as it is (a 3-D one as a slice); an adaptive mesh is drawn by
sampling its leaves on the pixels of the view, so zooming re-samples it at the finest cells in sight.
"""
from __future__ import annotations

import argparse
import copy
import functools
import os
import sys
import traceback

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PyQt6.QtCore import Qt, QTimer, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow,  # noqa: E402
                             QPushButton, QSlider, QTabWidget, QVBoxLayout, QWidget)

from gui.theme import COLORS, apply_theme  # noqa: E402
from gui.viewer_data import Frame, Series  # noqa: E402

_COLORMAPS = ("viridis", "turbo", "coolwarm", "magma", "gray", "Spectral_r")
_PIXEL_CAP = 1_500_000   # largest raster of an adaptive mesh, in pixels
_LEAF_LIMIT = 4000       # draw the leaf outlines only when this few are in sight
_AXIS_LABELS = {"z": ("x", "y"), "y": ("x", "z"), "x": ("y", "z")}


def _style(fig: Figure) -> None:
    fig.set_facecolor(COLORS["bg_window"])


def _guarded(method):
    """A slot called by a timer or a signal must not raise: PyQt6 aborts the application on an exception in a slot. A snapshot that
    cannot be read yet (a file being replaced, a piece not on disk) is retried on the next tick; the last picture stays."""
    @functools.wraps(method)
    def run(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:   # noqa: BLE001 - the whole point is to keep the viewer alive
            message = f"{type(exc).__name__}: {exc}"
            if message != getattr(self, "_last_error", None):   # a retry every second would repeat the same traceback
                traceback.print_exc()
            self._last_error = message
            self._failed(message)
    return run


class FieldView(QWidget):
    """The picture of one field of one snapshot, with pan / zoom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(8, 5))
        _style(self.fig)
        grid = self.fig.add_gridspec(1, 2, width_ratios=[40, 1], wspace=0.04, left=0.07, right=0.95, top=0.95, bottom=0.1)
        self.ax = self.fig.add_subplot(grid[0])
        self.cax = self.fig.add_subplot(grid[1])
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.readout = QLabel(" ")
        self.readout.setObjectName("note")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.readout)
        self.frame: Frame | None = None
        self.name, self.cmap, self.axis, self.index = "Density", "viridis", "z", 0
        self.lock, self.show_leaves = False, False
        self._im = None
        self._sig = None
        self._clim: tuple[float, float] | None = None
        self._leaves = None
        self._busy = False
        self._timer = QTimer(self)   # re-sample an adaptive mesh once a zoom or a resize has settled
        self._timer.setSingleShot(True)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._resample)
        self.ax.callbacks.connect("xlim_changed", self._view_changed)
        self.ax.callbacks.connect("ylim_changed", self._view_changed)
        self.canvas.mpl_connect("resize_event", lambda e: self._timer.start())
        self.canvas.mpl_connect("motion_notify_event", self._hover)

    # ---- drawing
    def show_frame(self, frame: Frame, name: str, cmap: str, axis: str, index: int, lock: bool, leaves: bool) -> None:
        self.frame, self.cmap, self.axis, self.index, self.lock, self.show_leaves = frame, cmap, axis, index, lock, leaves
        if name != self.name:
            self._clim = None   # a new field has its own range
        self.name = name
        self._render()

    def _signature(self):
        f = self.frame
        return (f.kind, f.extent(), self.axis if f.is_3d else "z", self.name, self.cmap)

    def _pixels(self) -> tuple[int, int]:
        bb = self.ax.bbox
        w, h = max(int(bb.width), 64), max(int(bb.height), 64)
        scale = min(1.0, (_PIXEL_CAP / (w * h)) ** 0.5)
        return max(int(w * scale), 16), max(int(h * scale), 16)

    def _render(self) -> None:
        f = self.frame
        if f is None:
            return
        if self.name not in f.fields:
            self.name = f.names()[0]
        sig = self._signature()
        fresh = self._im is None or sig != self._sig
        bounds = None if fresh or f.kind == "grid" else (*self.ax.get_xlim(), *self.ax.get_ylim())
        img, ext = f.picture(self.name, self.axis, self.index, bounds, self._pixels())
        data = f.fields[self.name]
        if not (self.lock and self._clim):
            lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
            self._clim = (lo, hi if hi > lo else lo + 1.0)
        self._busy = True
        try:
            if fresh:
                self._rebuild(img, ext, sig)
            else:
                self._im.set_data(img)
                self._im.set_extent(ext if f.kind == "amr" else ext)
            self._im.set_clim(*self._clim)
            self._leaf_outlines(bounds or ext)
        finally:
            self._busy = False
        self.canvas.draw_idle()

    def _rebuild(self, img, ext, sig) -> None:
        f = self.frame
        self.ax.clear()
        self.cax.clear()
        cmap = copy.copy(colormaps[self.cmap])
        cmap.set_bad(COLORS["bg_panel"])
        self._im = self.ax.imshow(img, origin="lower", extent=ext, cmap=cmap, interpolation="nearest", aspect="equal")
        self.ax.set_autoscale_on(False)
        full = f.extent() if f.kind == "amr" else ext
        self.ax.set_xlim(full[0], full[1])
        self.ax.set_ylim(full[2], full[3])
        lx, ly = _AXIS_LABELS[self.axis if f.is_3d else "z"]
        self.ax.set_xlabel(lx)
        self.ax.set_ylabel(ly)
        self.fig.colorbar(self._im, cax=self.cax).set_label(self.name)
        self._sig = sig
        self._leaves = None
        self.toolbar.update()   # "home" is the whole domain
        if f.kind == "amr":
            self._timer.start()   # the first raster used the whole domain at the canvas size; do it for the settled box

    def _leaf_outlines(self, bounds) -> None:
        if self._leaves is not None:
            self._leaves.remove()
            self._leaves = None
        f = self.frame
        if not self.show_leaves or f.kind != "amr":
            return
        lv = f.leaves_in(bounds, _LEAF_LIMIT)
        if lv is None:
            return
        x0, x1, y0, y1 = lv
        quads = np.stack([np.column_stack([x0, y0]), np.column_stack([x1, y0]), np.column_stack([x1, y1]),
                          np.column_stack([x0, y1])], axis=1)
        self._leaves = self.ax.add_collection(PolyCollection(quads, facecolors="none", edgecolors="black", linewidths=0.4))

    def _view_changed(self, ax) -> None:
        if not self._busy and self.frame is not None and self.frame.kind == "amr":
            self._timer.start()

    def _failed(self, message: str) -> None:
        self.readout.setText(message)

    @_guarded
    def _resample(self) -> None:
        if self.frame is not None and self.frame.kind == "amr":
            self._render()

    # ---- pointer readout
    @_guarded
    def _hover(self, event) -> None:
        f = self.frame
        if f is None or event.inaxes is not self.ax or event.xdata is None:
            return
        v = f.value_at(self.name, event.xdata, event.ydata, self.axis, self.index)
        if v is None:
            self.readout.setText(" ")
            return
        lx, ly = _AXIS_LABELS[self.axis if f.is_3d else "z"]
        text = f"{lx} = {event.xdata:.5g}   {ly} = {event.ydata:.5g}   {self.name} = {v:.6g}"
        if f.kind == "amr":
            leaf = f._lookup(np.array([event.xdata]), np.array([event.ydata]))[0]
            text += f"   level {int(f.level[leaf])}"
        self.readout.setText(text)


class MonitorView(QWidget):
    """A column of the monitor CSV against time, with the time of the snapshot on show."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(8, 4))
        _style(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.column = QComboBox()
        self.column.currentTextChanged.connect(lambda _: self.draw())
        row = QHBoxLayout()
        row.addWidget(QLabel("Column"))
        row.addWidget(self.column)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(row)
        lay.addWidget(self.canvas, 1)
        self.data = None
        self.now: float | None = None

    def set_data(self, data, now: float | None) -> None:
        self.data, self.now = data, now
        names = [n for n in (data.dtype.names if data is not None else ()) if n not in ("step", "t")]
        if names != [self.column.itemText(i) for i in range(self.column.count())]:
            keep = self.column.currentText()
            self.column.blockSignals(True)
            self.column.clear()
            self.column.addItems(names)
            if keep in names:
                self.column.setCurrentText(keep)
            self.column.blockSignals(False)
        self.draw()

    def draw(self) -> None:
        self.ax.clear()
        d, c = self.data, self.column.currentText()
        if d is None or not c:
            self.ax.text(0.5, 0.5, "no monitor file in this folder", ha="center", va="center", transform=self.ax.transAxes)
        else:
            self.ax.plot(d["t"], d[c], color=COLORS["primary"], lw=1.4)
            if self.now is not None:
                self.ax.axvline(self.now, color=COLORS["ink_muted"], lw=1.0, ls="--")
            self.ax.set_xlabel("t")
            self.ax.set_ylabel(c)
            self.ax.grid(alpha=0.3)
        self.canvas.draw_idle()


class ViewerPanel(QWidget):
    """The viewer as a widget, to sit in its own window or in a workbench (docs/proposals/gui-workbench.md): everything it needs is
    its folder, and it tells the host its title through `title_changed`."""

    title_changed = pyqtSignal(str)

    def __init__(self, folder: str | None = None, step: int | None = None, parent=None):
        super().__init__(parent)
        self.series: Series | None = None
        self._retry = False
        self.field = FieldView()
        self.monitor = MonitorView()

        self.btn_open = QPushButton("Open folder...")
        self.btn_open.clicked.connect(self._open_dialog)
        self.cb_field = QComboBox()
        self.cb_cmap = QComboBox()
        self.cb_cmap.addItems(_COLORMAPS)
        self.chk_lock = QCheckBox("Lock range")
        self.chk_leaves = QCheckBox("Leaf outlines")
        self.cb_axis = QComboBox()
        self.cb_axis.addItems(["z", "y", "x"])
        self.sl_slice = QSlider(Qt.Orientation.Horizontal)
        self.lbl_slice = QLabel("")
        self.btn_play = QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.sl_step = QSlider(Qt.Orientation.Horizontal)
        self.lbl_step = QLabel("no snapshots")
        self.lbl_step.setMinimumWidth(260)
        self.chk_follow = QCheckBox("Follow latest")
        self.chk_follow.setChecked(True)

        top = QHBoxLayout()
        for w in (self.btn_open, QLabel("Field"), self.cb_field, QLabel("Colours"), self.cb_cmap, self.chk_lock, self.chk_leaves):
            top.addWidget(w)
        top.addStretch(1)
        self.slice_row = QWidget()
        sr = QHBoxLayout(self.slice_row)
        sr.setContentsMargins(0, 0, 0, 0)
        for w in (QLabel("Slice normal to"), self.cb_axis, self.sl_slice, self.lbl_slice):
            sr.addWidget(w)
        bottom = QHBoxLayout()
        for w, stretch in ((self.btn_play, 0), (self.sl_step, 1), (self.lbl_step, 0), (self.chk_follow, 0)):
            bottom.addWidget(w, stretch)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.field, "Field")
        self.tabs.addTab(self.monitor, "Monitor")
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.slice_row)
        lay.addWidget(self.tabs, 1)
        lay.addLayout(bottom)

        for sig in (self.cb_field.currentTextChanged, self.cb_cmap.currentTextChanged, self.cb_axis.currentTextChanged):
            sig.connect(lambda _: self._show())
        for chk in (self.chk_lock, self.chk_leaves):
            chk.toggled.connect(lambda _: self._show())
        self.sl_slice.valueChanged.connect(lambda _: self._show())
        self.sl_step.valueChanged.connect(lambda _: self._show())
        self.btn_play.toggled.connect(self._play_toggled)
        self._play = QTimer(self)
        self._play.setInterval(250)
        self._play.timeout.connect(self._advance)
        self._live = QTimer(self)
        self._live.setInterval(1000)
        self._live.timeout.connect(self._refresh)
        self._live.start()

        if folder:
            self.open_path(folder, step)

    # ---- opening
    def _open_dialog(self) -> None:
        start = os.path.join(ROOT_DIR, "results")
        d = QFileDialog.getExistingDirectory(self, "Results folder", start if os.path.isdir(start) else ROOT_DIR)
        if d:
            self.open_path(d)

    def open_path(self, path: str, step: int | None = None) -> None:
        p = os.path.abspath(path)
        folder = p if os.path.isdir(p) else os.path.dirname(p)
        self.series = Series(folder)
        self.title_changed.emit(f"Fenice viewer - {os.path.basename(folder)}")
        n = len(self.series.snapshots)
        self.sl_step.blockSignals(True)
        self.sl_step.setRange(0, max(n - 1, 0))
        pick = n - 1
        if os.path.isfile(p):
            pick = next((i for i, s in enumerate(self.series.snapshots) if str(s.path) == p), pick)
        if step is not None:
            pick = min(range(n), key=lambda i: abs(self.series.snapshots[i].step - step)) if n else 0
        self.sl_step.setValue(max(pick, 0))
        self.sl_step.blockSignals(False)
        self.field._im = None
        self._show()

    def _failed(self, message: str) -> None:
        self._retry = True   # the next tick of the live timer shows the snapshot again
        self.lbl_step.setText(f"cannot read the snapshot yet ({message})")

    # ---- state -> picture
    @_guarded
    def _show(self) -> None:
        self._retry = False
        s = self.series
        if s is None or not s.snapshots:
            self.lbl_step.setText("no snapshots in this folder" if s is not None else "no snapshots")
            self.monitor.set_data(s.monitor() if s else None, None)
            return
        i = min(self.sl_step.value(), len(s.snapshots) - 1)
        snap = s.snapshots[i]
        frame = s.frame(i)
        names = frame.names()
        if names != [self.cb_field.itemText(k) for k in range(self.cb_field.count())]:
            keep = self.cb_field.currentText()
            self.cb_field.blockSignals(True)
            self.cb_field.clear()
            self.cb_field.addItems(names)
            self.cb_field.setCurrentText(keep if keep in names else names[0])
            self.cb_field.blockSignals(False)
        self.slice_row.setVisible(frame.is_3d)
        axis = self.cb_axis.currentText() if frame.is_3d else "z"
        count = frame.slice_count(axis)
        if self.sl_slice.maximum() != count - 1:
            self.sl_slice.blockSignals(True)
            self.sl_slice.setRange(0, count - 1)
            self.sl_slice.setValue(count // 2)
            self.sl_slice.blockSignals(False)
        index = self.sl_slice.value()
        self.lbl_slice.setText(f"{axis} cell {index + 1} / {count}")
        self.field.show_frame(frame, self.cb_field.currentText(), self.cb_cmap.currentText(), axis, index,
                              self.chk_lock.isChecked(), self.chk_leaves.isChecked())
        when = f"t = {snap.time:.6g}" if snap.time is not None else "t unknown"
        extra = f"{frame.n_leaves():,} leaves" if frame.kind == "amr" else f"{frame.nx} x {frame.ny}" + (f" x {frame.nz}" if frame.nz > 1 else "")
        self.lbl_step.setText(f"snapshot {i + 1}/{len(s.snapshots)}   step {snap.step}   {when}   {extra}")
        self.monitor.set_data(s.monitor(), snap.time)

    # ---- playback and live refresh
    def _play_toggled(self, on: bool) -> None:
        self.btn_play.setText("Pause" if on else "Play")
        if on:
            self.chk_follow.setChecked(False)
            self._play.start()
        else:
            self._play.stop()

    @_guarded
    def _advance(self) -> None:
        v = self.sl_step.value()
        if v >= self.sl_step.maximum():
            self.btn_play.setChecked(False)
        else:
            self.sl_step.setValue(v + 1)

    @_guarded
    def _refresh(self) -> None:
        if self.series is None:
            return
        if self.series.refresh():
            last = len(self.series.snapshots) - 1
            self.sl_step.blockSignals(True)
            self.sl_step.setRange(0, max(last, 0))
            if self.chk_follow.isChecked():
                self.sl_step.setValue(max(last, 0))
            self.sl_step.blockSignals(False)
            self._show()
        elif self._retry or self.tabs.currentWidget() is self.monitor:
            self._show()

    def shutdown(self) -> None:
        """Stop the timers (the host calls this before it drops the panel)."""
        self._play.stop()
        self._live.stop()


class ViewerWindow(QMainWindow):
    """A window around one ViewerPanel (`python gui/viewer.py`)."""

    def __init__(self, folder: str | None = None, step: int | None = None):
        super().__init__()
        self.panel = ViewerPanel(folder, step)
        self.panel.title_changed.connect(self.setWindowTitle)
        self.setCentralWidget(self.panel)
        self.resize(1200, 780)

    def closeEvent(self, event) -> None:
        self.panel.shutdown()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fenice result viewer")
    ap.add_argument("path", nargs="?", help="a results folder or one of its snapshot files")
    ap.add_argument("--screenshot", metavar="PNG", help="save the window to PNG and exit")
    ap.add_argument("--field", help="field to show first")
    ap.add_argument("--step", type=int, help="snapshot (nearest step) to show first; default the last")
    ap.add_argument("--size", default="1200x780", help="window size WxH (default 1200x780)")
    ap.add_argument("--tab", choices=("field", "monitor"), default="field")
    ap.add_argument("--zoom", metavar="X0,X1,Y0,Y1", help="view window of the field picture")
    ap.add_argument("--leaves", action="store_true", help="draw the leaf outlines")
    args = ap.parse_args(argv)
    app = QApplication(sys.argv[:1])
    apply_theme(app)
    win = ViewerWindow(args.path, args.step)
    pan = win.panel
    w, h = (int(v) for v in args.size.lower().split("x"))
    win.resize(w, h)
    if args.field and args.field in [pan.cb_field.itemText(i) for i in range(pan.cb_field.count())]:
        pan.cb_field.setCurrentText(args.field)
    if args.leaves:
        pan.chk_leaves.setChecked(True)
    if args.tab == "monitor":
        pan.tabs.setCurrentWidget(pan.monitor)
    win.show()
    if args.zoom:
        x0, x1, y0, y1 = (float(v) for v in args.zoom.split(","))
        pan.field.ax.set_xlim(x0, x1)
        pan.field.ax.set_ylim(y0, y1)
    if args.path is None:
        QTimer.singleShot(0, pan._open_dialog)
    if args.screenshot:
        for _ in range(8):   # let the layout settle and the adaptive mesh re-sample at the final size
            app.processEvents()
            QTimer.singleShot(150, lambda: None)
            pan.field.canvas.draw()
        app.processEvents()
        win.grab().save(args.screenshot)
        return 0
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
