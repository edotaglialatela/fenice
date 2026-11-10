"""Quantitative inspection of fields in one Fenice results folder."""
from __future__ import annotations

import os
import traceback

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout,
                             QLabel, QPushButton, QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from gui.theme import COLORS
from gui.viewer_data import Series

_AXIS_LABELS = {"z": ("x", "y"), "y": ("x", "z"), "x": ("y", "z")}


class FieldAnalysisPanel(QWidget):
    """Field statistics and line profiles for the results linked to a solver."""

    title_changed = pyqtSignal(str)
    folder_changed = pyqtSignal(str)

    def __init__(self, folder: str | None = None, parent=None):
        super().__init__(parent)
        self.series: Series | None = None
        self.frame = None
        self._profile: tuple[np.ndarray, np.ndarray, np.ndarray, str, str] | None = None
        self._spatial: tuple[np.ndarray, np.ndarray, str, float, float] | None = None
        self._plot_kind = "profile"
        self._has_profile = False
        self._updating = False

        heading = QLabel("Field analysis")
        heading.setObjectName("sectionTitle")
        intro = QLabel("Measure a field on the selected snapshot, then sample it along a line.")
        intro.setObjectName("note")
        intro.setWordWrap(True)

        self.btn_open = QPushButton("Open results folder...")
        self.btn_open.setObjectName("secondaryButton")
        self.btn_open.clicked.connect(self._open_dialog)
        self.cb_snapshot = QComboBox()
        self.cb_snapshot.setMinimumWidth(220)
        self.chk_latest = QCheckBox("Follow latest")
        self.chk_latest.setChecked(True)
        self.cb_snapshot.currentIndexChanged.connect(self._snapshot_selected)
        self.chk_latest.toggled.connect(self._follow_changed)

        source_row = QHBoxLayout()
        source_row.addWidget(self.btn_open)
        source_row.addWidget(QLabel("Snapshot"))
        source_row.addWidget(self.cb_snapshot, 1)
        source_row.addWidget(self.chk_latest)

        self.cb_field = QComboBox()
        self.cb_field.setMinimumWidth(170)
        self.cb_field.currentTextChanged.connect(lambda _text: self._selection_changed())
        self.cb_axis = QComboBox()
        self.cb_axis.addItems(["z", "y", "x"])
        self.cb_axis.currentTextChanged.connect(lambda _text: self._selection_changed())
        self.sl_slice = QSlider(Qt.Orientation.Horizontal)
        self.lbl_slice = QLabel("")
        self.lbl_slice.setMinimumWidth(74)
        self.sl_slice.valueChanged.connect(lambda _value: self._selection_changed())
        self.axis_label = QLabel("Slice normal to")
        self.slice_label = QLabel("Slice")
        plane_row = QHBoxLayout()
        plane_row.addWidget(QLabel("Field"))
        plane_row.addWidget(self.cb_field, 1)
        plane_row.addSpacing(8)
        plane_row.addWidget(self.axis_label)
        plane_row.addWidget(self.cb_axis)
        plane_row.addWidget(self.slice_label)
        plane_row.addWidget(self.sl_slice, 1)
        plane_row.addWidget(self.lbl_slice)

        self.lbl_stats_note = QLabel("Across cells in the displayed field or slice.")
        self.lbl_stats_note.setObjectName("note")
        self.lbl_stats_note.setWordWrap(True)
        self.stat_values: dict[str, QLabel] = {}
        stats = QGridLayout()
        stats.setHorizontalSpacing(16)
        stats.setVerticalSpacing(7)
        stat_rows = (("min", "Minimum"), ("max", "Maximum"), ("mean", "Mean"), ("std", "Std. deviation"),
                     ("rms", "RMS"), ("median", "Median"), ("p05", "5th percentile"), ("p95", "95th percentile"))
        for row, (key, label) in enumerate(stat_rows):
            name = QLabel(label)
            value = QLabel("—")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setMinimumWidth(100)
            self.stat_values[key] = value
            stats.addWidget(name, row, 0)
            stats.addWidget(value, row, 1)

        self.coordinate_inputs: dict[str, QDoubleSpinBox] = {}
        self.endpoint_labels: dict[str, QLabel] = {}
        coordinates = QFormLayout()
        coordinates.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        for key, title in (("x0", "Start x"), ("y0", "Start y"), ("x1", "End x"), ("y1", "End y")):
            self.endpoint_labels[key] = QLabel(title)
            spin = QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(-1e9, 1e9)
            spin.setSingleStep(0.1)
            spin.setKeyboardTracking(False)
            spin.setMinimumWidth(116)
            spin.valueChanged.connect(self._profile_inputs_changed)
            self.coordinate_inputs[key] = spin
            coordinates.addRow(self.endpoint_labels[key], spin)

        self.samples = QSpinBox()
        self.samples.setRange(10, 100000)
        self.samples.setValue(200)
        self.samples.setSuffix(" points")
        self.samples.setToolTip("Number of values sampled along the line")
        self.samples.valueChanged.connect(self._profile_inputs_changed)
        coordinates.addRow("Samples", self.samples)
        self.btn_wake_preset = QPushButton("Apply wake-line example")
        self.btn_wake_preset.setObjectName("secondaryButton")
        self.btn_wake_preset.setToolTip("Optional example for the low-blockage cylinder result: Velocity_y along its wake centreline, sampled at finest-leaf spacing.")
        self.btn_wake_preset.setEnabled(False)
        self.btn_wake_preset.clicked.connect(self._plot_documented_wake)
        coordinates.addRow(self.btn_wake_preset)

        self.btn_profile = QPushButton("Plot line profile")
        self.btn_profile.setObjectName("primaryButton")
        self.btn_profile.clicked.connect(self._plot_profile)
        self.cb_spatial_window = QComboBox()
        self.cb_spatial_window.addItem("Hann window", "hann")
        self.cb_spatial_window.addItem("Even extension", "even")
        self.cb_spatial_window.addItem("No window", "none")
        self.cb_spatial_window.setToolTip("Window or mirror the finite line before its spatial FFT")
        self.btn_spatial = QPushButton("Spatial spectrum")
        self.btn_spatial.setObjectName("secondaryButton")
        self.btn_spatial.clicked.connect(self._plot_spatial_spectrum)
        self.btn_export = QPushButton("Export CSV...")
        self.btn_export.setObjectName("secondaryButton")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_profile)
        actions = QHBoxLayout()
        actions.addWidget(self.btn_profile, 1)
        actions.addWidget(self.btn_export)
        spatial_actions = QHBoxLayout()
        spatial_actions.addWidget(self.cb_spatial_window, 1)
        spatial_actions.addWidget(self.btn_spatial)

        self.lbl_status = QLabel("Connect this module to a Solver with results, or open a results folder.")
        self.lbl_status.setObjectName("note")
        self.lbl_status.setWordWrap(True)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 14, 16, 14)
        left_layout.setSpacing(10)
        stats_title = QLabel("Field statistics")
        stats_title.setObjectName("subsectionTitle")
        line_title = QLabel("Line profile")
        line_title.setObjectName("subsectionTitle")
        left_layout.addWidget(stats_title)
        left_layout.addWidget(self.lbl_stats_note)
        left_layout.addLayout(stats)
        left_layout.addWidget(self._rule())
        left_layout.addWidget(line_title)
        left_layout.addWidget(QLabel("Sample the selected field between two coordinates."))
        left_layout.addLayout(coordinates)
        left_layout.addLayout(actions)
        left_layout.addLayout(spatial_actions)
        left_layout.addWidget(self.lbl_status)
        left_layout.addStretch(1)
        left.setMinimumWidth(280)
        left.setMaximumWidth(390)

        self.figure = Figure(figsize=(7, 4))
        self.figure.set_facecolor(COLORS["bg_window"])
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.plot_hint = QLabel("Choose a results folder, snapshot and field to begin.")
        self.plot_hint.setObjectName("note")
        self.plot_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_stack = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_stack)
        self.plot_layout.setContentsMargins(12, 12, 12, 12)
        self.plot_layout.addWidget(self.plot_hint)
        self.plot_layout.addWidget(self.canvas)
        self.canvas.hide()
        self.lbl_profile_info = QLabel("")
        self.lbl_profile_info.setObjectName("note")
        self.lbl_profile_info.setMinimumHeight(24)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        plot_title = QLabel("Profile plot")
        plot_title.setObjectName("subsectionTitle")
        right_layout.addWidget(plot_title)
        right_layout.addWidget(self.plot_stack, 1)
        right_layout.addWidget(self.lbl_profile_info)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([330, 760])

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(10)
        outer.addWidget(heading)
        outer.addWidget(intro)
        outer.addLayout(source_row)
        outer.addLayout(plane_row)
        outer.addWidget(self.splitter, 1)

        self._live = QTimer(self)
        self._live.setInterval(1000)
        self._live.timeout.connect(self._refresh)
        self._live.start()
        self._set_plot_style()
        self._clear_selection()
        if folder:
            self.open_path(folder)

    @staticmethod
    def _rule() -> QFrame:
        rule = QFrame()
        rule.setObjectName("hsep")
        rule.setFixedHeight(1)
        return rule

    def _set_plot_style(self) -> None:
        self.ax.set_facecolor(COLORS["bg_surface"])
        for spine in self.ax.spines.values():
            spine.set_color(COLORS["border_strong"])
        self.ax.tick_params(colors=COLORS["ink_soft"])
        self.ax.xaxis.label.set_color(COLORS["ink"])
        self.ax.yaxis.label.set_color(COLORS["ink"])
        self.ax.grid(color=COLORS["grid"], linewidth=0.8)

    def _open_dialog(self) -> None:
        start = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        folder = QFileDialog.getExistingDirectory(self, "Results folder", start if os.path.isdir(start) else os.getcwd())
        if folder:
            self.open_path(folder)

    def open_path(self, path: str) -> None:
        try:
            self.series = Series(path)
            self._profile = None
            self._has_profile = False
            self.btn_export.setEnabled(False)
            self.chk_latest.blockSignals(True)
            self.chk_latest.setChecked(True)
            self.chk_latest.blockSignals(False)
            self.title_changed.emit(f"Field analysis - {os.path.basename(os.path.abspath(path))}")
            self.folder_changed.emit(str(self.series.folder))
            self._populate_snapshots(prefer_latest=True)
            if not self.series.snapshots:
                self.lbl_status.setText("No snapshots found in this folder yet. A running solver will appear here as it writes results.")
        except Exception as exc:  # noqa: BLE001 - a malformed folder must not close the workbench
            self._show_error(exc)

    def _current_step(self) -> int | None:
        if self.series is None or not self.series.snapshots or self.cb_snapshot.currentIndex() < 0:
            return None
        i = min(self.cb_snapshot.currentIndex(), len(self.series.snapshots) - 1)
        return self.series.snapshots[i].step

    def _populate_snapshots(self, preferred: int | None = None, prefer_latest: bool = False) -> None:
        series = self.series
        snapshots = series.snapshots if series is not None else []
        labels = [f"Step {snap.step}" + (f"   t = {snap.time:.6g}" if snap.time is not None else "") for snap in snapshots]
        self._updating = True
        self.cb_snapshot.blockSignals(True)
        self.cb_snapshot.clear()
        self.cb_snapshot.addItems(labels)
        if snapshots:
            if prefer_latest or preferred is None:
                index = len(snapshots) - 1
            else:
                index = min(range(len(snapshots)), key=lambda k: abs(snapshots[k].step - preferred))
            self.cb_snapshot.setCurrentIndex(index)
        self.cb_snapshot.blockSignals(False)
        self._updating = False
        self._selection_changed(reset_line=preferred is None)

    def _snapshot_selected(self, _index: int) -> None:
        if not self._updating:
            if self.chk_latest.isChecked():
                self.chk_latest.blockSignals(True)
                self.chk_latest.setChecked(False)
                self.chk_latest.blockSignals(False)
            self._selection_changed()

    def _follow_changed(self, on: bool) -> None:
        if on and self.series is not None and self.series.snapshots:
            self._populate_snapshots(prefer_latest=True)

    def _selection_changed(self, reset_line: bool = False) -> None:
        if self._updating or self.series is None or not self.series.snapshots or self.cb_snapshot.currentIndex() < 0:
            self.frame = None
            self._clear_selection()
            return
        try:
            index = min(self.cb_snapshot.currentIndex(), len(self.series.snapshots) - 1)
            self.frame = self.series.frame(index)
            names = self.frame.names()
            self._sync_wake_preset()
            keep = self.cb_field.currentText()
            self.cb_field.blockSignals(True)
            self.cb_field.clear()
            self.cb_field.addItems(names)
            preferred = "Velocity_y" if "Velocity_y" in names else (names[0] if names else "")
            self.cb_field.setCurrentText(keep if keep in names else preferred)
            self.cb_field.blockSignals(False)
            self.cb_field.setEnabled(bool(names))
            is_3d = self.frame.is_3d
            self.axis_label.setVisible(is_3d)
            self.cb_axis.setVisible(is_3d)
            self.slice_label.setVisible(is_3d)
            self.sl_slice.setVisible(is_3d)
            self.lbl_slice.setVisible(is_3d)
            self.cb_axis.setEnabled(is_3d)
            axis = self.cb_axis.currentText() if is_3d else "z"
            count = self.frame.slice_count(axis)
            self.sl_slice.blockSignals(True)
            self.sl_slice.setRange(0, count - 1)
            if reset_line or self.sl_slice.value() >= count:
                self.sl_slice.setValue(count // 2)
            self.sl_slice.blockSignals(False)
            self.lbl_slice.setText(f"{axis} cell {self.sl_slice.value() + 1}/{count}")
            self._sync_coordinate_bounds(reset_line)
            self._show_statistics()
            for spin in self.coordinate_inputs.values():
                spin.setEnabled(bool(names))
            self.samples.setEnabled(bool(names))
            self.btn_profile.setEnabled(bool(names))
            self.btn_spatial.setEnabled(bool(names))
            snap = self.series.snapshots[index]
            self.lbl_status.setText(f"Step {snap.step} selected. Statistics use the cells in the displayed field or slice.")
            if self._has_profile:
                if self._plot_kind == "spectrum":
                    self._profile = None
                    self._has_profile = False
                    self._plot_spatial_spectrum()
                else:
                    self._plot_profile()
            else:
                self._show_plot_hint("Set the line endpoints and choose Plot line profile.")
        except Exception as exc:  # noqa: BLE001
            self._show_error(exc)

    def _wake_line_spacing(self) -> float | None:
        if self.frame is None or self.frame.kind != "amr" or "Velocity_y" not in self.frame.names():
            return None
        widths = np.minimum(self.frame.x1 - self.frame.x0, self.frame.y1 - self.frame.y0)
        h = float(np.min(widths)) if widths.size else 0.0
        return h if np.isfinite(h) and h > 0 else None

    def _sync_wake_preset(self) -> None:
        h = self._wake_line_spacing()
        enabled = False
        if h is not None:
            xmin, xmax, ymin, ymax = self.frame.extent()
            count = int(round(7.0 / h))
            x_start, x_end, y = 4.0 + 0.5 * h, 11.0 - 0.5 * h, 6.0 + 0.5 * h
            enabled = xmin <= x_start and xmax >= x_end and ymin <= y <= ymax and count <= self.samples.maximum()
        self.btn_wake_preset.setEnabled(enabled)
        if enabled:
            self.btn_wake_preset.setToolTip(f"Set Velocity_y along x = 4–11 near y = 6, resample at finest leaf spacing ({count:,} points), and apply the Hann spectrum.")

    def _plot_documented_wake(self) -> None:
        h = self._wake_line_spacing()
        if h is None or not self.btn_wake_preset.isEnabled():
            self.lbl_status.setText("Open the low-blockage cylinder AMR results to use this wake preset.")
            return
        self.cb_field.setCurrentText("Velocity_y")
        self.coordinate_inputs["x0"].setValue(4.0 + 0.5 * h)
        self.coordinate_inputs["x1"].setValue(11.0 - 0.5 * h)
        self.coordinate_inputs["y0"].setValue(6.0 + 0.5 * h)
        self.coordinate_inputs["y1"].setValue(6.0 + 0.5 * h)
        self.samples.setValue(int(round(7.0 / h)))
        index = self.cb_spatial_window.findData("hann")
        if index >= 0:
            self.cb_spatial_window.setCurrentIndex(index)
        self._plot_spatial_spectrum()

    def _sync_coordinate_bounds(self, reset_line: bool) -> None:
        if self.frame is None or not self.cb_field.currentText():
            return
        axis = self.cb_axis.currentText() if self.frame.is_3d else "z"
        _, bounds = self.frame.section(self.cb_field.currentText(), axis, self.sl_slice.value())
        x0, x1, y0, y1 = (float(v) for v in bounds)
        for key, (low, high) in {"x0": (x0, x1), "x1": (x0, x1), "y0": (y0, y1), "y1": (y0, y1)}.items():
            spin = self.coordinate_inputs[key]
            spin.blockSignals(True)
            spin.setRange(low, high)
            spin.setSingleStep(max(abs(high - low) / 100.0, 1e-6))
            spin.blockSignals(False)
        if reset_line:
            self.coordinate_inputs["x0"].setValue(x0 + 0.2 * (x1 - x0))
            self.coordinate_inputs["y0"].setValue(y0 + 0.5 * (y1 - y0))
            self.coordinate_inputs["x1"].setValue(x0 + 0.8 * (x1 - x0))
            self.coordinate_inputs["y1"].setValue(y0 + 0.5 * (y1 - y0))
        x_name, y_name = _AXIS_LABELS[axis]
        self.endpoint_labels["x0"].setText(f"Start {x_name}")
        self.endpoint_labels["y0"].setText(f"Start {y_name}")
        self.endpoint_labels["x1"].setText(f"End {x_name}")
        self.endpoint_labels["y1"].setText(f"End {y_name}")

    def _show_statistics(self) -> None:
        name = self.cb_field.currentText()
        if self.frame is None or not name:
            for value in self.stat_values.values():
                value.setText("—")
            return
        axis = self.cb_axis.currentText() if self.frame.is_3d else "z"
        values, _ = self.frame.section(name, axis, self.sl_slice.value())
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        if not finite.size:
            for value in self.stat_values.values():
                value.setText("—")
            return
        for key, number in (("min", np.min(finite)), ("max", np.max(finite)), ("mean", np.mean(finite)),
                            ("std", np.std(finite)), ("rms", np.sqrt(np.mean(finite * finite))),
                            ("median", np.median(finite)), ("p05", np.percentile(finite, 5)),
                            ("p95", np.percentile(finite, 95))):
            self.stat_values[key].setText(f"{number:.6g}")

    def _plot_profile(self) -> None:
        if self.frame is None or not self.cb_field.currentText():
            return
        try:
            start_x = self.coordinate_inputs["x0"].value()
            start_y = self.coordinate_inputs["y0"].value()
            end_x = self.coordinate_inputs["x1"].value()
            end_y = self.coordinate_inputs["y1"].value()
            if start_x == end_x and start_y == end_y:
                self.lbl_status.setText("Choose different start and end points for the profile.")
                return
            t = np.linspace(0.0, 1.0, self.samples.value())
            xs = start_x + t * (end_x - start_x)
            ys = start_y + t * (end_y - start_y)
            axis = self.cb_axis.currentText() if self.frame.is_3d else "z"
            name = self.cb_field.currentText()
            vals = np.fromiter((self.frame.value_at(name, x, y, axis, self.sl_slice.value()) for x, y in zip(xs, ys)),
                               dtype=float, count=t.size)
            distance = t * float(np.hypot(end_x - start_x, end_y - start_y))
            x_name, y_name = _AXIS_LABELS[axis]
            self._profile = distance, vals, np.column_stack((xs, ys)), x_name, y_name
            self._spatial = None
            self._plot_kind = "profile"
            self._has_profile = True
            self.ax.clear()
            self._set_plot_style()
            self.ax.plot(distance, vals, color=COLORS["route_results"], linewidth=1.8)
            self.ax.set_xlabel("Distance along line")
            self.ax.set_ylabel(name)
            finite = np.isfinite(vals)
            valid = int(finite.sum())
            self.canvas.draw_idle()
            self.canvas.show()
            self.plot_hint.hide()
            self.btn_export.setEnabled(valid > 0)
            self.lbl_profile_info.setText(
                f"{valid} of {len(vals)} samples inside the field   ·   "
                f"{x_name}: {start_x:.5g} to {end_x:.5g}   ·   {y_name}: {start_y:.5g} to {end_y:.5g}"
            )
            if valid:
                self.lbl_status.setText(f"Profile calculated for {name} at step {self._current_step()}.")
            else:
                self.lbl_status.setText("The selected line falls outside the field; move its endpoints inside the domain.")
        except Exception as exc:  # noqa: BLE001 - bad or partial result files should not close the workbench
            self._show_error(exc)

    def _profile_inputs_changed(self, _value) -> None:
        if self._has_profile:
            self._show_plot_hint("Profile inputs changed. Plot the line again to update the chart and CSV.")

    def _plot_spatial_spectrum(self) -> None:
        """Plot a wavenumber spectrum along the selected line through a field."""
        if self._profile is None or not self._has_profile:
            self._plot_profile()
        if self._profile is None or self.series is None:
            return
        try:
            from fenice import spectra

            distance, values, _points, _x_name, _y_name = self._profile
            if not np.isfinite(values).all():
                raise ValueError("The line crosses cells outside the domain. Move both endpoints into the field.")
            if values.size < 8:
                raise ValueError("A spatial spectrum needs at least 8 line samples.")
            result = spectra.spectrum_1d(values, float(distance[1] - distance[0]), self.cb_spatial_window.currentData())
            k, energy = np.asarray(result["k"]), np.asarray(result["E"])
            peak_i = int(np.argmax(energy[1:]) + 1)
            k_peak = float(k[peak_i])
            wavelength = 2 * np.pi / k_peak if k_peak > 0 else float("inf")
            coarse_fraction = None
            if self.frame is not None and self.frame.kind == "amr":
                leaf = self.frame._lookup(self._profile[2][:, 0], self._profile[2][:, 1])
                if np.all(leaf >= 0):
                    leaf_size = np.minimum(self.frame.x1[leaf] - self.frame.x0[leaf],
                                           self.frame.y1[leaf] - self.frame.y0[leaf])
                    finest = float(np.min(np.minimum(self.frame.x1 - self.frame.x0, self.frame.y1 - self.frame.y0)))
                    coarse_fraction = float(np.mean(leaf_size > finest * (1 + 1e-4)))
            self._spatial = k, energy, self.cb_field.currentText(), k_peak, float(result["leakage"])
            self._plot_kind = "spectrum"
            self.ax.clear()
            self._set_plot_style()
            self.ax.semilogy(k[1:], np.maximum(energy[1:], np.finfo(float).tiny), color=COLORS["route_results"], linewidth=1.8)
            self.ax.axvline(k_peak, color=COLORS["primary"], linestyle="--", linewidth=1.1, label=f"peak k = {k_peak:.4g}")
            self.ax.legend(loc="best")
            self.ax.set_xlabel("Wavenumber k [rad / length]")
            self.ax.set_ylabel("One-sided spectral density")
            self.canvas.draw_idle()
            self.canvas.show()
            self.plot_hint.hide()
            coarse_note = f"   ·   coarse AMR samples = {coarse_fraction:.0%}" if coarse_fraction is not None else ""
            self.lbl_profile_info.setText(f"Peak k = {k_peak:.5g}   ·   wavelength = {wavelength:.5g}   ·   high-k leakage = {result['leakage']:.3g}{coarse_note}")
            self.btn_export.setEnabled(True)
            if coarse_fraction is None:
                self.lbl_status.setText("Spatial spectrum calculated.")
            else:
                self.lbl_status.setText(f"Spatial spectrum calculated; {coarse_fraction:.0%} of the line crosses leaves coarser than the finest mesh level.")
        except Exception as exc:  # noqa: BLE001 - unsupported snapshots must not close the workbench
            self.lbl_status.setText(f"{type(exc).__name__}: {exc}")

    def _show_plot_hint(self, text: str) -> None:
        self._profile = None
        self._spatial = None
        self._has_profile = False
        self._plot_kind = "profile"
        self.btn_export.setEnabled(False)
        self.canvas.hide()
        self.plot_hint.setText(text)
        self.plot_hint.show()
        self.lbl_profile_info.clear()
        self.ax.clear()
        self._set_plot_style()
        self.canvas.draw_idle()

    def _clear_selection(self) -> None:
        self.btn_wake_preset.setEnabled(False)
        self.cb_field.blockSignals(True)
        self.cb_field.clear()
        self.cb_field.blockSignals(False)
        self.cb_field.setEnabled(False)
        self.btn_profile.setEnabled(False)
        self.btn_spatial.setEnabled(False)
        for spin in self.coordinate_inputs.values():
            spin.setEnabled(False)
        self.samples.setEnabled(False)
        self.axis_label.setVisible(False)
        self.cb_axis.setVisible(False)
        self.slice_label.setVisible(False)
        self.sl_slice.setVisible(False)
        self.lbl_slice.setVisible(False)
        for value in self.stat_values.values():
            value.setText("—")
        self._show_plot_hint("No snapshot is available yet. Start a run or open a results folder.")

    def _refresh(self) -> None:
        if self.series is None:
            return
        try:
            if self.series.refresh():
                step = self._current_step()
                self._populate_snapshots(preferred=step, prefer_latest=self.chk_latest.isChecked())
        except Exception as exc:  # noqa: BLE001
            self._show_error(exc)

    def wait_for_results(self) -> None:
        """Clear the previous run while the solver creates its next results folder."""
        self.series = None
        self.frame = None
        self._profile = None
        self._has_profile = False
        self.btn_export.setEnabled(False)
        self.chk_latest.blockSignals(True)
        self.chk_latest.setChecked(True)
        self.chk_latest.blockSignals(False)
        self.cb_snapshot.blockSignals(True)
        self.cb_snapshot.clear()
        self.cb_snapshot.blockSignals(False)
        self._clear_selection()
        self.lbl_status.setText("Waiting for the solver to create a results folder.")

    def _show_error(self, exc: Exception) -> None:
        message = f"{type(exc).__name__}: {exc}"
        self.lbl_status.setText(message)
        self._show_plot_hint("This snapshot could not be read. Fenice will retry as results are updated.")
        if message != getattr(self, "_last_error", None):
            traceback.print_exc()
        self._last_error = message

    def _export_profile(self) -> None:
        if self._profile is None:
            return
        folder = str(self.series.folder) if self.series is not None else os.getcwd()
        spectrum = self._plot_kind == "spectrum" and self._spatial is not None
        default_name = "spatial-spectrum.csv" if spectrum else "line-profile.csv"
        title = "Export spatial spectrum" if spectrum else "Export line profile"
        path, _ = QFileDialog.getSaveFileName(self, title, os.path.join(folder, default_name), "CSV files (*.csv)")
        if not path:
            return
        try:
            field = self.cb_field.currentText() or "value"
            if self._plot_kind == "spectrum" and self._spatial is not None:
                k, energy, name, _peak, _leakage = self._spatial
                np.savetxt(path, np.column_stack((k, energy)), delimiter=",", fmt="%.12g",
                           header=f"wavenumber,spectral_density_{name}", comments="")
            else:
                distance, values, points, x_name, y_name = self._profile
                np.savetxt(path, np.column_stack((distance, points, values)), delimiter=",", fmt="%.12g",
                           header=f"distance,{x_name},{y_name},{field}", comments="")
            self.lbl_status.setText(f"{'Spectrum' if spectrum else 'Profile'} saved to {path}")
        except OSError as exc:
            self._show_error(exc)

    def save_state(self) -> dict:
        return {
            "step": self._current_step(), "follow_latest": self.chk_latest.isChecked(), "field": self.cb_field.currentText(),
            "axis": self.cb_axis.currentText(), "slice": self.sl_slice.value(),
            "line": {key: spin.value() for key, spin in self.coordinate_inputs.items()},
            "samples": self.samples.value(), "plot_kind": self._plot_kind, "has_plot": self._has_profile,
            "spatial_window": self.cb_spatial_window.currentData(),
        }

    def restore_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        self.chk_latest.setChecked(bool(state.get("follow_latest", False)))
        self.cb_axis.setCurrentText(str(state.get("axis", "z")))
        if self.series is not None:
            self._populate_snapshots(preferred=state.get("step"), prefer_latest=bool(state.get("follow_latest", False)))
        field = str(state.get("field", ""))
        if self.cb_field.findText(field) >= 0:
            self.cb_field.setCurrentText(field)
        self._selection_changed()
        self.sl_slice.setValue(int(state.get("slice", self.sl_slice.value())))
        for key, spin in self.coordinate_inputs.items():
            if key in state.get("line", {}):
                spin.setValue(float(state["line"][key]))
        self.samples.setValue(int(state.get("samples", self.samples.value())))
        index = self.cb_spatial_window.findData(state.get("spatial_window", "hann"))
        if index >= 0:
            self.cb_spatial_window.setCurrentIndex(index)
        if state.get("has_plot"):
            self._plot_spatial_spectrum() if state.get("plot_kind") == "spectrum" else self._plot_profile()

    def shutdown(self) -> None:
        self._live.stop()


class AnalysisPanel(QWidget):
    """Separate field, load, modal and sensitivity tools for one results folder."""

    title_changed = pyqtSignal(str)

    def __init__(self, folder: str | None = None, parent=None):
        super().__init__(parent)
        from gui.analysis_tools import LoadAnalysisTab, ModalAnalysisTab, SensitivityAnalysisTab

        self.series: Series | None = None
        heading = QLabel("Analysis Tools")
        heading.setObjectName("sectionTitle")
        intro = QLabel("Analyze fields, monitor signals, flow modes and parameter studies.")
        intro.setObjectName("note")
        intro.setWordWrap(True)

        self.btn_open = QPushButton("Open results folder...")
        self.btn_open.setObjectName("secondaryButton")
        self.btn_open.clicked.connect(self._open_dialog)
        self.lbl_folder = QLabel("No results folder")
        self.lbl_folder.setObjectName("note")
        self.lbl_folder.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        source = QHBoxLayout()
        source.addWidget(self.btn_open)
        source.addWidget(self.lbl_folder, 1)

        self.tabs = QTabWidget()
        self.field = FieldAnalysisPanel()
        self.field.btn_open.hide()
        self.loads = LoadAnalysisTab()
        self.modes = ModalAnalysisTab()
        self.sensitivity = SensitivityAnalysisTab()
        for page, label in ((self.field, "Fields / profiles"), (self.loads, "Signals / spectra"),
                            (self.modes, "POD / DMD"), (self.sensitivity, "Sensitivity")):
            self.tabs.addTab(page, label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(heading)
        layout.addWidget(intro)
        layout.addLayout(source)
        layout.addWidget(self.tabs, 1)
        self.field.title_changed.connect(self.title_changed.emit)
        self.field.folder_changed.connect(self._folder_changed)
        if folder:
            self.open_path(folder)

    def _open_dialog(self) -> None:
        start = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        folder = QFileDialog.getExistingDirectory(self, "Results folder", start if os.path.isdir(start) else os.getcwd())
        if folder:
            self.open_path(folder)

    def _folder_changed(self, folder: str) -> None:
        self.lbl_folder.setText(folder)
        self.lbl_folder.setToolTip(folder)
        self.loads.set_folder(folder)
        self.modes.set_folder(folder)
        self.series = self.field.series

    def open_path(self, path: str) -> None:
        self.field.open_path(path)

    def wait_for_results(self) -> None:
        self.field.wait_for_results()
        self.series = None
        self.lbl_folder.setText("Waiting for a results folder")
        self.loads.set_folder(None)
        self.modes.set_folder(None)

    def save_state(self) -> dict:
        return {
            "tab": self.tabs.currentIndex(), "folder": str(self.series.folder) if self.series is not None else None,
            "field": self.field.save_state(), "loads": self.loads.save_state(), "modes": self.modes.save_state(),
            "sensitivity": self.sensitivity.save_state(),
        }

    def restore_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if state.get("folder"):
            self.open_path(str(state["folder"]))
        self.field.restore_state(state.get("field", {}))
        self.loads.restore_state(state.get("loads", {}))
        self.modes.restore_state(state.get("modes", {}))
        self.sensitivity.restore_state(state.get("sensitivity", {}))
        self.tabs.setCurrentIndex(max(0, min(int(state.get("tab", 0)), self.tabs.count() - 1)))

    def shutdown(self) -> None:
        self.field.shutdown()
        self.loads.shutdown()
        self.modes.shutdown()
