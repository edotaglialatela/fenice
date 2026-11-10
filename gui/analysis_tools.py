"""Statistical analysis pages used by the workbench Analysis Tools module."""
from __future__ import annotations

import csv
import json
import traceback
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QPlainTextEdit, QSpinBox, QVBoxLayout, QWidget)

from gui.theme import COLORS
from gui.viewer_data import Series


def _spin(value: float, *, low: float = -1e9, high: float = 1e9, decimals: int = 4, step: float = 0.1) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setKeyboardTracking(False)
    box.setValue(value)
    return box


def _style_axes(axes) -> None:
    for ax in np.ravel(axes):
        ax.set_facecolor(COLORS["bg_surface"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["border_strong"])
        ax.tick_params(colors=COLORS["ink_soft"])
        ax.xaxis.label.set_color(COLORS["ink"])
        ax.yaxis.label.set_color(COLORS["ink"])
        ax.title.set_color(COLORS["ink"])
        ax.grid(color=COLORS["grid"], linewidth=0.8)


def _monitor_data(series: Series | None):
    if series is None:
        return None
    series.refresh()
    return series.monitor()


def _signal_names(data) -> list[str]:
    if data is None or not getattr(data.dtype, "names", None):
        return []
    return [name for name in data.dtype.names if name not in ("t", "step") and np.issubdtype(data.dtype[name], np.number)]


def _set_combo(combo: QComboBox, values: list[str], preferred: str = "") -> None:
    keep = preferred or combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(values)
    if keep in values:
        combo.setCurrentText(keep)
    elif values:
        combo.setCurrentIndex(0)
    combo.blockSignals(False)


class LoadAnalysisTab(QWidget):
    """General monitor statistics, spectra, frequency drift and cycle estimates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.folder: Path | None = None
        self.series: Series | None = None
        self._last_analysis: dict | None = None
        self._last_rows = 0

        note = QLabel("Inspect any numeric monitor signal. Choose a frequency band, optional L/U scaling, and time-window checks.")
        note.setObjectName("note")
        note.setWordWrap(True)
        self.cb_signal = QComboBox()
        self.cb_compare = QComboBox()
        self.cb_signal.setMinimumWidth(120)
        self.cb_compare.setMinimumWidth(110)
        self.t_start = _spin(0.0, low=-1e9, high=1e9, decimals=3)
        self.cb_scale = QComboBox()
        self.cb_scale.addItems(("Frequency", "fL/U"))
        self.length_scale = _spin(1.0, low=1e-12, high=1e9, decimals=6)
        self.reference_speed = _spin(1.0, low=1e-12, high=1e9, decimals=6)
        self.frequency_low = _spin(0.0, low=0.0, high=1e9, decimals=5)
        self.frequency_high = _spin(1.0, low=0.0, high=1e9, decimals=5)
        self.window_length = _spin(10.0, low=1e-9, high=1e9, decimals=3)
        self.window_step = _spin(2.0, low=1e-9, high=1e9, decimals=3)
        self.cb_window = QComboBox()
        self.cb_window.addItems(("Hann", "Hamming", "None"))
        self.zero_padding = QSpinBox()
        self.zero_padding.setRange(1, 128)
        self.zero_padding.setValue(32)
        self.chk_harmonic = QCheckBox("Check harmonic")
        self.harmonic_order = QSpinBox()
        self.harmonic_order.setRange(2, 10)
        self.harmonic_order.setValue(3)
        self.harmonic_order.setEnabled(False)
        self.chk_harmonic.toggled.connect(self.harmonic_order.setEnabled)
        self.chk_crossings = QCheckBox("Estimate from crossings")
        self.chk_reference = QCheckBox("Reference band")
        self.chk_reference.toggled.connect(self._reference_toggled)
        self.reference_low = _spin(0.0, low=0.0, high=1e9, decimals=5)
        self.reference_high = _spin(1.0, low=0.0, high=1e9, decimals=5)
        self.reference_low.setEnabled(False)
        self.reference_high.setEnabled(False)
        for widget in (self.cb_signal, self.cb_compare):
            widget.setMaximumWidth(145)
        for widget in (self.t_start, self.length_scale, self.reference_speed, self.frequency_low, self.frequency_high,
                       self.window_length, self.window_step, self.reference_low, self.reference_high):
            widget.setMaximumWidth(112)
        self.cb_scale.setMaximumWidth(125)
        self.cb_window.setMaximumWidth(125)
        self.zero_padding.setMaximumWidth(78)
        self.harmonic_order.setMaximumWidth(65)

        controls = QGridLayout()
        controls.setHorizontalSpacing(9)
        controls.setVerticalSpacing(6)
        row0 = (("Signal", self.cb_signal), ("From t", self.t_start), ("Frequency axis", self.cb_scale),
                ("Length L", self.length_scale), ("Speed U", self.reference_speed))
        row1 = (("Peak band", self._range_widget(self.frequency_low, self.frequency_high)), ("Window", self.window_length),
                ("Slide by", self.window_step), ("Spectrum window", self.cb_window), ("Zero padding", self.zero_padding))
        reference_row = QWidget()
        reference_layout = QHBoxLayout(reference_row)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        reference_layout.setSpacing(4)
        reference_layout.addWidget(self.chk_reference)
        reference_layout.addWidget(self.reference_low)
        reference_layout.addWidget(QLabel("to"))
        reference_layout.addWidget(self.reference_high)
        reference_row.setMaximumWidth(390)
        harmonic_row = self._paired_widget(self.chk_harmonic, self.harmonic_order)
        harmonic_row.setMaximumWidth(240)
        row2 = (("Compare signal", self.cb_compare), ("", harmonic_row),
                ("", self.chk_crossings), ("", reference_row))
        for row, values in enumerate((row0, row1, row2)):
            for col, (label, widget) in enumerate(values):
                if label:
                    controls.addWidget(QLabel(label), row, col * 2)
                controls.addWidget(widget, row, col * 2 + 1)

        self.btn_analyze = QPushButton("Analyze signal")
        self.btn_analyze.setObjectName("primaryButton")
        self.btn_analyze.clicked.connect(self.analyze)
        self.btn_export = QPushButton("Export CSV...")
        self.btn_export.setObjectName("secondaryButton")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_csv)
        actions = QHBoxLayout()
        actions.addWidget(self.btn_analyze)
        actions.addWidget(self.btn_export)
        actions.addStretch(1)

        self.figure = Figure(figsize=(8, 6))
        self.figure.set_facecolor(COLORS["bg_window"])
        self.axes = self.figure.subplots(3, 1)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setMaximumHeight(148)
        self.report.setPlaceholderText("Signal statistics, peak frequency, window drift and optional comparisons appear here.")
        self.status = QLabel("Open a results folder with a monitor CSV to begin.")
        self.status.setObjectName("note")
        self.status.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(note)
        layout.addLayout(controls)
        layout.addLayout(actions)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.report)
        layout.addWidget(self.status)
        self.cb_signal.currentTextChanged.connect(self._signal_changed)
        self.cb_scale.currentTextChanged.connect(self._scale_changed)
        self._live = QTimer(self)
        self._live.setInterval(1500)
        self._live.timeout.connect(self._refresh)
        self._live.start()
        self._scale_changed()
        _style_axes(self.axes)

    @staticmethod
    def _range_widget(low: QDoubleSpinBox, high: QDoubleSpinBox) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(low)
        lay.addWidget(QLabel("to"))
        lay.addWidget(high)
        return row

    @staticmethod
    def _paired_widget(first: QWidget, second: QWidget) -> QWidget:
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(first)
        lay.addWidget(second)
        return row

    def _reference_toggled(self, on: bool) -> None:
        self.reference_low.setEnabled(on)
        self.reference_high.setEnabled(on)

    def _scale_changed(self) -> None:
        normalized = self.cb_scale.currentText() == "fL/U"
        self.length_scale.setEnabled(normalized)
        self.reference_speed.setEnabled(normalized)

    def _signal_changed(self, signal: str) -> None:
        data = _monitor_data(self.series)
        names = _signal_names(data)
        values = ["None"] + [name for name in names if name != signal]
        _set_combo(self.cb_compare, values)

    def set_folder(self, folder: str | None) -> None:
        self.folder = Path(folder) if folder else None
        self.series = Series(self.folder) if self.folder else None
        self._last_analysis = None
        self.btn_export.setEnabled(False)
        self.report.clear()
        self._clear_result_view()
        self._refresh(force=True)

    def _clear_result_view(self) -> None:
        for axis in self.axes:
            axis.clear()
            axis.set_visible(False)
        self.axes[0].set_visible(True)
        self.axes[0].set_axis_off()
        self.axes[0].text(0.5, 0.5, "Run a signal analysis to view plots.", ha="center", va="center",
                          color=COLORS["ink_soft"], transform=self.axes[0].transAxes)
        self.canvas.draw_idle()

    def _refresh(self, force: bool = False) -> None:
        data = _monitor_data(self.series)
        names = _signal_names(data)
        old = self.cb_signal.currentText()
        if names != [self.cb_signal.itemText(i) for i in range(self.cb_signal.count())]:
            _set_combo(self.cb_signal, names, old)
            self._signal_changed(self.cb_signal.currentText())
        rows = len(data) if data is not None else 0
        self.btn_analyze.setEnabled(rows >= 4 and bool(names))
        if self.folder is None:
            self.status.setText("Open a results folder with a monitor CSV to begin.")
        elif rows < 4:
            self.status.setText("No usable monitor rows yet. A running solver will appear here as it writes its monitor CSV.")
        elif not force and self._last_analysis is not None and rows != self._last_rows:
            self.status.setText(f"{rows - self._last_rows} new monitor rows are available. Analyze again to include them.")
        elif self._last_analysis is None:
            self.status.setText(f"{rows} monitor rows ready in {self.folder.name}.")

    @staticmethod
    def _spectrum(t, values, pad: int = 32, window_name: str = "Hann"):
        t, values = np.asarray(t, float), np.asarray(values, float)
        if len(t) < 4 or np.any(np.diff(t) <= 0):
            raise ValueError("Need at least four rows with strictly increasing time values.")
        uniform_t = np.linspace(t[0], t[-1], 4 * len(t))
        uniform_y = np.interp(uniform_t, t, values)
        uniform_y -= uniform_y.mean()
        window = {"Hann": np.hanning, "Hamming": np.hamming}.get(window_name)
        weights = window(len(uniform_y)) if window else np.ones(len(uniform_y))
        nfft = pad * len(uniform_y)
        amplitude = 2.0 * np.abs(np.fft.rfft(uniform_y * weights, nfft)) / max(float(weights.sum()), np.finfo(float).tiny)
        frequency = np.fft.rfftfreq(nfft, uniform_t[1] - uniform_t[0])
        return frequency, amplitude, float(uniform_t[-1] - uniform_t[0])

    @staticmethod
    def _peak(frequency, amplitude, lo: float, hi: float) -> tuple[float, float]:
        if hi <= lo:
            raise ValueError("The peak band upper value must be greater than its lower value.")
        mask = (frequency > 0) & (frequency >= lo) & (frequency <= hi)
        if not np.any(mask):
            raise ValueError(f"No non-zero frequency bins fall in the selected band {lo:g} to {hi:g}.")
        indexes = np.flatnonzero(mask)
        index = indexes[int(np.argmax(amplitude[mask]))]
        return float(frequency[index]), float(amplitude[index])

    @staticmethod
    def _crossing_frequency(t, values) -> tuple[float, int]:
        t, values = np.asarray(t, float), np.asarray(values, float)
        centred = values - 0.5 * (float(values.max()) + float(values.min()))
        indexes = np.flatnonzero((centred[:-1] < 0) & (centred[1:] >= 0))
        if len(indexes) < 2:
            return float("nan"), max(0, len(indexes) - 1)
        times = t[indexes] - centred[indexes] * (t[indexes + 1] - t[indexes]) / (centred[indexes + 1] - centred[indexes])
        period = float(np.mean(np.diff(times)))
        return 1.0 / period, len(times) - 1

    def analyze(self) -> None:
        data = _monitor_data(self.series)
        names = _signal_names(data)
        signal = self.cb_signal.currentText()
        if data is None or signal not in names:
            self.status.setText("This folder has no usable numeric monitor signal. Open a results folder with a monitor CSV.")
            return
        try:
            all_t = np.asarray(data["t"], float)
            all_values = np.asarray(data[signal], float)
            valid = np.isfinite(all_t) & np.isfinite(all_values) & (all_t >= self.t_start.value())
            t, values = all_t[valid], all_values[valid]
            if len(t) < 4:
                raise ValueError("The selected time range has fewer than four usable monitor rows.")
            scale = self.length_scale.value() / self.reference_speed.value() if self.cb_scale.currentText() == "fL/U" else 1.0
            low, high = sorted((self.frequency_low.value(), self.frequency_high.value()))
            raw_frequency, amplitude, span = self._spectrum(t, values, self.zero_padding.value(), self.cb_window.currentText())
            frequency = raw_frequency * scale
            peak, peak_amplitude = self._peak(frequency, amplitude, low, high)
            reference_low, reference_high = sorted((self.reference_low.value(), self.reference_high.value()))

            compared = self.cb_compare.currentText()
            compare_data = None
            if compared not in ("", "None") and compared in names:
                compare_values = np.asarray(data[compared], float)
                compare_mask = np.isfinite(all_t) & np.isfinite(compare_values) & (all_t >= self.t_start.value())
                if int(compare_mask.sum()) >= 4:
                    compare_t = all_t[compare_mask]
                    compare_y = compare_values[compare_mask]
                    compare_f_raw, compare_amp, _ = self._spectrum(compare_t, compare_y, self.zero_padding.value(), self.cb_window.currentText())
                    compare_f = compare_f_raw * scale
                    compare_peak, compare_peak_amplitude = self._peak(compare_f, compare_amp, low, high)
                    compare_data = (compared, compare_f, compare_amp, compare_peak, compare_peak_amplitude)

            harmonic = None
            if self.chk_harmonic.isChecked():
                order = self.harmonic_order.value()
                harmonic_center = peak * order
                try:
                    harmonic = self._peak(frequency, amplitude, 0.82 * harmonic_center, 1.18 * harmonic_center)
                except ValueError:
                    harmonic = None

            window_frequency, window_time = [], []
            if t[-1] - t[0] >= self.window_length.value():
                start = float(t[0])
                end = float(t[-1] - self.window_length.value())
                starts = np.arange(start, end + self.window_step.value() * 1e-9, self.window_step.value())
                for t0 in starts:
                    keep = (t >= t0) & (t <= t0 + self.window_length.value())
                    if int(keep.sum()) < 4:
                        continue
                    fw, aw, _ = self._spectrum(t[keep], values[keep], self.zero_padding.value(), self.cb_window.currentText())
                    peak_w, _ = self._peak(fw * scale, aw, low, high)
                    window_frequency.append(peak_w)
                    window_time.append(t0 + 0.5 * self.window_length.value())

            crossing_frequency = float("nan")
            crossing_count = 0
            if self.chk_crossings.isChecked():
                crossing_frequency, crossing_count = self._crossing_frequency(t, values)
                crossing_frequency *= scale

            mean = float(np.mean(values))
            std = float(np.std(values))
            rms = float(np.sqrt(np.mean(values * values)))
            amplitude_range = float(np.max(values) - np.min(values))
            self._last_analysis = {
                "signal": signal, "time": t, "values": values, "frequency": frequency, "spectrum": amplitude,
                "peak_frequency": peak, "peak_amplitude": peak_amplitude, "window_time": np.asarray(window_time),
                "window_frequency": np.asarray(window_frequency), "compare": compare_data,
            }
            self._last_rows = len(data)
            for axis in self.axes:
                axis.set_visible(True)
                axis.set_axis_on()
                axis.clear()
            self.axes[0].clear()
            self.axes[0].plot(t, values, color=COLORS["route_results"], linewidth=1.35)
            self.axes[0].set_ylabel(signal)
            self.axes[0].set_title(f"{signal} monitor record, t >= {self.t_start.value():g}")

            self.axes[1].clear()
            self.axes[1].semilogy(frequency[1:], np.maximum(amplitude[1:], np.finfo(float).tiny),
                                  color=COLORS["route_results"], linewidth=1.35, label=signal)
            self.axes[1].axvline(peak, color=COLORS["primary"], linestyle="--", linewidth=1.0,
                                 label=f"{signal} peak {peak:.5g}")
            if compare_data:
                compare_name, compare_frequency, compare_amplitude, compare_peak, _ = compare_data
                self.axes[1].semilogy(compare_frequency[1:], np.maximum(compare_amplitude[1:], np.finfo(float).tiny),
                                      color=COLORS["route_run"], linewidth=1.15, label=compare_name)
                self.axes[1].axvline(compare_peak, color=COLORS["route_run"], linestyle=":", linewidth=1.0,
                                     label=f"{compare_name} peak {compare_peak:.5g}")
            if harmonic is not None:
                order = self.harmonic_order.value()
                self.axes[1].axvline(harmonic[0], color=COLORS["action"], linestyle=":", linewidth=1.1,
                                     label=f"{order}x component {harmonic[0]:.5g}")
            if self.chk_reference.isChecked():
                self.axes[1].axvspan(reference_low, reference_high, color=COLORS["route_geometry"], alpha=0.18,
                                     label="reference band")
            self.axes[1].set_ylabel("Amplitude")
            self.axes[1].set_xlabel("Frequency, fL/U" if scale != 1.0 else "Frequency, inverse time")
            self.axes[1].set_title(f"{self.cb_window.currentText()} spectrum, zero padded {self.zero_padding.value()}x")
            visible_high = max(high, reference_high if self.chk_reference.isChecked() else 0.0, peak * 1.12)
            self.axes[1].set_xlim(0.0, min(float(frequency[-1]), max(visible_high, 0.05)))
            self.axes[1].legend(loc="best", fontsize=8)

            self.axes[2].clear()
            if window_frequency:
                self.axes[2].plot(window_time, window_frequency, "o-", color=COLORS["route_run"], markersize=3.5)
                if self.chk_reference.isChecked():
                    self.axes[2].axhspan(reference_low, reference_high, color=COLORS["route_geometry"], alpha=0.18)
                self.axes[2].axhline(peak, color=COLORS["primary"], linestyle="--", linewidth=1.0,
                                     label=f"whole record {peak:.5g}")
                self.axes[2].legend(loc="best", fontsize=8)
            else:
                self.axes[2].text(0.5, 0.5, "Record is shorter than one selected window", ha="center", va="center",
                                  transform=self.axes[2].transAxes, color=COLORS["ink_soft"])
            self.axes[2].set_xlabel("Window centre, t")
            self.axes[2].set_ylabel("Peak frequency")
            self.axes[2].set_title(f"Sliding windows: {self.window_length.value():g} time units")
            _style_axes(self.axes)
            self.figure.tight_layout()
            self.canvas.draw_idle()

            raw_peak = peak / scale
            lines = [f"{signal}: mean {mean:.6g}; standard deviation {std:.6g}; RMS {rms:.6g}; peak-to-peak {amplitude_range:.6g}.",
                     f"Peak frequency = {peak:.6g} (amplitude {peak_amplitude:.5g}); record resolution = {scale / span:.5g}; "
                     f"about {raw_peak * span:.2f} cycles in the selected record."]
            if compare_data:
                compare_name, _, _, compare_peak, compare_peak_amplitude = compare_data
                lines.append(f"{compare_name} peak = {compare_peak:.6g} (amplitude {compare_peak_amplitude:.5g}); "
                             f"frequency ratio {compare_peak / peak:.5g} relative to {signal}.")
            if self.chk_harmonic.isChecked():
                lines.append(f"{self.harmonic_order.value()}x component: frequency = {harmonic[0]:.6g}, amplitude = {harmonic[1]:.5g}."
                             if harmonic else f"No {self.harmonic_order.value()}x component was resolved in the selected band.")
            if self.chk_crossings.isChecked():
                lines.append(f"Upward-crossing frequency = {crossing_frequency:.6g} from {crossing_count} cycles."
                             if np.isfinite(crossing_frequency) else "Upward crossings: not enough complete cycles in this time range.")
            if window_frequency:
                lines.append(f"Sliding-window peak frequency = {min(window_frequency):.6g} to {max(window_frequency):.6g} "
                             f"over {len(window_frequency)} windows.")
            lines.append("Zero padding refines peak interpolation; the true frequency resolution is set by record duration.")
            self.report.setPlainText("\n".join(lines))
            self.status.setText(f"Analyzed {len(t)} rows from {t[0]:g} to {t[-1]:g} time units.")
            self.btn_export.setEnabled(True)
        except Exception as exc:  # noqa: BLE001 - invalid monitor data stays inside this module
            self.status.setText(f"{type(exc).__name__}: {exc}")
            self.report.clear()
            traceback.print_exc()

    def export_csv(self) -> None:
        if self._last_analysis is None:
            return
        start = str(self.folder / "signal-analysis.csv") if self.folder else "signal-analysis.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export signal analysis", start, "CSV files (*.csv)")
        if not path:
            return
        result = self._last_analysis
        rows = max(len(result["frequency"]), len(result["window_frequency"]))
        matrix = np.full((rows, 4), np.nan)
        matrix[:len(result["frequency"]), 0] = result["frequency"]
        matrix[:len(result["spectrum"]), 1] = result["spectrum"]
        matrix[:len(result["window_time"]), 2] = result["window_time"]
        matrix[:len(result["window_frequency"]), 3] = result["window_frequency"]
        np.savetxt(path, matrix, delimiter=",", fmt="%.12g",
                   header="frequency,amplitude,window_centre_t,window_peak_frequency", comments="")
        self.status.setText(f"Analysis data saved to {path}")

    def save_state(self) -> dict:
        return {"signal": self.cb_signal.currentText(), "compare": self.cb_compare.currentText(), "t_start": self.t_start.value(),
                "scale": self.cb_scale.currentText(), "length_scale": self.length_scale.value(), "reference_speed": self.reference_speed.value(),
                "frequency_low": self.frequency_low.value(), "frequency_high": self.frequency_high.value(),
                "window_length": self.window_length.value(), "window_step": self.window_step.value(),
                "window": self.cb_window.currentText(), "zero_padding": self.zero_padding.value(),
                "harmonic": self.chk_harmonic.isChecked(), "harmonic_order": self.harmonic_order.value(),
                "crossings": self.chk_crossings.isChecked(), "reference_low": self.reference_low.value(),
                "reference_high": self.reference_high.value(), "show_reference": self.chk_reference.isChecked(),
                "analyzed": self._last_analysis is not None}

    def restore_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if "diameter" in state and "length_scale" not in state:
            state = dict(state, length_scale=state["diameter"], scale="fL/U",
                         frequency_low=state.get("st_low", self.frequency_low.value()),
                         frequency_high=state.get("st_high", self.frequency_high.value()))
        if "velocity" in state and "reference_speed" not in state:
            state = dict(state, reference_speed=state["velocity"])
        for key, combo in (("signal", self.cb_signal), ("compare", self.cb_compare), ("scale", self.cb_scale), ("window", self.cb_window)):
            if combo.findText(str(state.get(key, ""))) >= 0:
                combo.setCurrentText(str(state[key]))
        for key in ("t_start", "length_scale", "reference_speed", "frequency_low", "frequency_high", "window_length", "window_step",
                    "reference_low", "reference_high"):
            if key in state:
                getattr(self, key).setValue(float(state[key]))
        for key in ("zero_padding", "harmonic_order"):
            if key in state:
                getattr(self, key).setValue(int(state[key]))
        self.chk_harmonic.setChecked(bool(state.get("harmonic", False)))
        self.chk_crossings.setChecked(bool(state.get("crossings", False)))
        self.chk_reference.setChecked(bool(state.get("show_reference", False)))
        if state.get("analyzed"):
            self.analyze()

    def shutdown(self) -> None:
        self._live.stop()


class ModalAnalysisTab(QWidget):
    """Snapshot POD and DMD plus time-delay DMD for numeric monitor signals."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.folder: Path | None = None
        self.series: Series | None = None
        self._result = None
        self._load_result = None
        self._view_kind = "snapshot"

        note = QLabel("POD and exact DMD use uniformly timed snapshots. Run time-delay DMD on monitor data alone when snapshots are sparse.")
        note.setObjectName("note")
        note.setWordWrap(True)
        self.cb_field = QComboBox()
        self.cb_component = QComboBox()
        self.cb_component.addItems(["x", "y", "z"])
        self.cb_component.setCurrentText("y")
        self.cb_field.currentTextChanged.connect(lambda field: self.cb_component.setEnabled(field == "Velocity"))
        self.cb_component.setEnabled(False)
        self.first_time = _spin(0.0, low=-1e9, high=1e9, decimals=3)
        self.every = QSpinBox(); self.every.setRange(1, 100000); self.every.setValue(1)
        self.tolerance = _spin(0.005, low=1e-8, high=1.0, decimals=6, step=0.001)
        self.rank = QSpinBox(); self.rank.setRange(2, 32); self.rank.setValue(8)
        self.cb_signal = QComboBox()
        self.cb_compare = QComboBox()
        self.length_scale = _spin(1.0, low=1e-12, high=1e9, decimals=6)
        self.reference_speed = _spin(1.0, low=1e-12, high=1e9, decimals=6)
        self.delay_dt = _spin(0.1, low=1e-8, high=1e9, decimals=5)
        self.delay_rows = QSpinBox(); self.delay_rows.setRange(2, 10000); self.delay_rows.setValue(60)
        controls = QGridLayout()
        rows = (("Snapshot field", self.cb_field), ("Vector component", self.cb_component), ("First snapshot t", self.first_time),
                ("Take every", self.every), ("Step tolerance", self.tolerance), ("POD rank", self.rank), ("Monitor signal", self.cb_signal),
                ("Compare signal", self.cb_compare), ("Length L", self.length_scale), ("Speed U", self.reference_speed),
                ("Load step", self.delay_dt), ("Delay rows", self.delay_rows))
        for group, title in enumerate(("Snapshot sequence", "Monitor signals")):
            heading = QLabel(title)
            heading.setObjectName("subsectionTitle")
            row = group * 2
            controls.addWidget(heading, row, 0, 1, 12)
            for i, (label, widget) in enumerate(rows[group * 6:(group + 1) * 6]):
                col = i * 2
                controls.addWidget(QLabel(label), row + 1, col)
                controls.addWidget(widget, row + 1, col + 1)
        self.btn_run = QPushButton("Run modal analysis")
        self.btn_run.setObjectName("primaryButton")
        self.btn_run.clicked.connect(self.analyze)
        self.btn_load_only = QPushButton("Run time-delay DMD only")
        self.btn_load_only.setObjectName("secondaryButton")
        self.btn_load_only.setToolTip("Analyze monitor data without loading snapshots, including when snapshot output is sparse.")
        self.btn_load_only.clicked.connect(self.analyze_load_dmd)
        self.btn_load_only.setEnabled(False)
        analysis_actions = QHBoxLayout()
        analysis_actions.addWidget(self.btn_run, 1)
        analysis_actions.addWidget(self.btn_load_only, 1)
        self.status = QLabel("Open a results folder with snapshots and monitor data.")
        self.status.setObjectName("note")
        self.status.setWordWrap(True)
        self.report = QPlainTextEdit(); self.report.setReadOnly(True); self.report.setMaximumHeight(160)
        self.report.setPlaceholderText("POD energy, modal phase, snapshot DMD and monitor time-delay DMD will be summarized here.")
        self.figure = Figure(figsize=(9, 5.5))
        self.figure.set_facecolor(COLORS["bg_window"])
        self.axes = self.figure.subplots(2, 3)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(note)
        layout.addLayout(controls)
        layout.addLayout(analysis_actions)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.report)
        layout.addWidget(self.status)
        self.cb_signal.currentTextChanged.connect(self._signal_changed)
        _style_axes(self.axes)

    def set_folder(self, folder: str | None) -> None:
        self.folder = Path(folder) if folder else None
        self.series = Series(self.folder) if self.folder else None
        self._result = None
        self._load_result = None
        self._view_kind = "snapshot"
        self.report.clear()
        self._clear_result_view()
        self._populate()

    def _clear_result_view(self) -> None:
        self.figure.clear()
        self.axes = self.figure.subplots(1, 1, squeeze=False)
        axis = self.axes[0, 0]
        axis.set_axis_off()
        axis.text(0.5, 0.5, "Run a modal analysis to view results.", ha="center", va="center",
                  color=COLORS["ink_soft"], transform=axis.transAxes)
        self.canvas.draw_idle()

    def _populate(self) -> None:
        if self.series is None:
            _set_combo(self.cb_field, [])
            _set_combo(self.cb_signal, [])
            self._signal_changed("")
            self.btn_run.setEnabled(False)
            self.btn_load_only.setEnabled(False)
            self.status.setText("Open a results folder with snapshots or monitor data.")
            return
        try:
            data = _monitor_data(self.series)
            signals = _signal_names(data)
            _set_combo(self.cb_signal, signals)
            self._signal_changed(self.cb_signal.currentText())
            self.btn_load_only.setEnabled(bool(signals))
            if not self.series.snapshots:
                _set_combo(self.cb_field, [])
                self.btn_run.setEnabled(False)
                self.cb_component.setEnabled(False)
                self.status.setText("No snapshots found. Time-delay DMD can still use the monitor data.")
                return
            frame = self.series.frame(0)
            fields = [field for field in frame.names() if not field.startswith("Velocity_")]
            if "Velocity_y" in frame.names():
                fields.append("Velocity")
            _set_combo(self.cb_field, fields, "Velocity")
            self.cb_component.setEnabled(self.cb_field.currentText() == "Velocity")
            self.btn_run.setEnabled(bool(fields))
            self.status.setText(f"{len(self.series.snapshots)} snapshots available in {self.folder.name}.")
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f"Cannot inspect this results folder: {exc}")

    def _signal_changed(self, signal: str) -> None:
        data = _monitor_data(self.series)
        names = _signal_names(data)
        _set_combo(self.cb_compare, ["None"] + [name for name in names if name != signal])

    def _delay_signal(self, data, signal: str) -> dict:
        from fenice import modal, spectra

        if data is None or signal not in _signal_names(data):
            raise ValueError(f"Monitor signal {signal!r} is unavailable.")
        t = np.asarray(data["t"], float)
        values = np.asarray(data[signal], float)
        keep = np.isfinite(t) & np.isfinite(values) & (t >= self.first_time.value())
        t, values = t[keep], values[keep]
        if len(t) <= self.delay_rows.value() + 2:
            raise ValueError(f"Record is too short for {self.delay_rows.value()} delay rows.")
        uniform_t, uniform_y, step_ratio = spectra.resample_uniform(t, values, self.delay_dt.value())
        delay = modal.delay_embed(uniform_y - uniform_y.mean(), self.delay_rows.value())
        result = modal.dmd(delay, self.delay_dt.value(), rank=2)
        positive = np.flatnonzero(result["frequency"] > 0)
        if not len(positive):
            raise ValueError("No positive-frequency time-delay DMD mode was found.")
        index = int(positive[np.argmax(result["frequency"][positive])])
        frequency = float(result["frequency"][index])
        return {"signal": signal, "time": t, "values": values, "uniform_time": uniform_t,
                "step_ratio": float(step_ratio), "dmd": result, "index": index, "frequency": frequency,
                "scaled_frequency": frequency * self.length_scale.value() / self.reference_speed.value(),
                "growth_ratio": float(result["growth"][index] / (2 * np.pi * frequency))}

    def _delay_results(self, data) -> tuple[dict[str, dict], list[str]]:
        names = _signal_names(data)
        selected = self.cb_signal.currentText()
        if selected not in names:
            return {}, ["Choose a numeric monitor signal before running time-delay DMD."]
        signals = [selected]
        compared = self.cb_compare.currentText()
        if compared not in ("", "None") and compared in names and compared != selected:
            signals.append(compared)
        results, errors = {}, []
        for signal in signals:
            try:
                results[signal] = self._delay_signal(data, signal)
            except (ValueError, IndexError, np.linalg.LinAlgError) as exc:
                errors.append(f"Time-delay DMD {signal}: {exc}")
        return results, errors

    @staticmethod
    def _delay_report(results: dict[str, dict], errors: list[str]) -> str:
        lines = [f"Time-delay DMD {signal}: fL/U = {result['scaled_frequency']:.5f}; |λ| = {abs(result['dmd']['eigenvalues'][result['index']]):.6g}; "
                 f"normalized growth = {result['growth_ratio']:+.4g}; largest input step / uniform step = {result['step_ratio']:.4g}."
                 for signal, result in results.items()]
        if len(results) == 2:
            first, second = results.values()
            lines.append(f"{second['signal']} / {first['signal']} frequency ratio = "
                         f"{second['frequency'] / first['frequency']:.4f}.")
        lines.extend(errors)
        return "\n".join(lines) if lines else "No monitor signal selected."

    def analyze_load_dmd(self) -> None:
        if self.series is None:
            self.status.setText("Open a results folder with monitor data before running time-delay DMD.")
            return
        try:
            data = _monitor_data(self.series)
            results, errors = self._delay_results(data)
            if not results:
                raise ValueError(" ".join(errors) or "No usable monitor data was found.")
            selected = self.cb_signal.currentText()
            plotted = results.get(selected, next(iter(results.values())))
            self._load_result = plotted
            self._view_kind = "load"
            self.figure.clear()
            self.axes = self.figure.subplots(1, 2)
            time_ax, eigen_ax = self.axes
            time_ax.plot(plotted["time"], plotted["values"], color=COLORS["route_results"], linewidth=1.25)
            time_ax.set_title(f"{plotted['signal']} monitor record")
            time_ax.set_xlabel("Time, t")
            time_ax.set_ylabel(plotted["signal"])
            theta = np.linspace(0.0, 2 * np.pi, 256)
            eigen_ax.plot(np.cos(theta), np.sin(theta), color=COLORS["border_strong"], linewidth=1.0, label="unit circle")
            palette = (COLORS["route_results"], COLORS["route_run"], COLORS["primary"])
            all_eigenvalues = []
            for index, (signal, result) in enumerate(results.items()):
                eigenvalues = result["dmd"]["eigenvalues"]
                all_eigenvalues.extend(eigenvalues)
                eigen_ax.plot(eigenvalues.real, eigenvalues.imag, "o", color=palette[index % len(palette)],
                              markersize=5, label=f"{signal} modes")
                leading = eigenvalues[result["index"]]
                eigen_ax.plot([leading.real], [leading.imag], "x", color=COLORS["action"], markersize=8, mew=1.8)
            limit = max(1.05, 1.1 * max((abs(value) for value in all_eigenvalues), default=1.0))
            eigen_ax.set_xlim(-limit, limit)
            eigen_ax.set_ylim(-limit, limit)
            eigen_ax.set_aspect("equal", adjustable="box")
            eigen_ax.set_title("Time-delay DMD eigenvalues")
            eigen_ax.set_xlabel("Real part")
            eigen_ax.set_ylabel("Imaginary part")
            eigen_ax.legend(loc="best")
            _style_axes(self.axes)
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self.report.setPlainText(self._delay_report(results, errors))
            self.status.setText(f"Time-delay DMD complete for {len(results)} monitor signal{'s' if len(results) != 1 else ''}.")
        except Exception as exc:  # noqa: BLE001 - sparse or malformed monitor data stays in the analysis view
            self.status.setText(f"{type(exc).__name__}: {exc}")
            self.report.setPlainText("Check the monitor signal, start time, load step and delay rows, then run again.")

    def analyze(self) -> None:
        if self.folder is None or self.series is None:
            self.status.setText("Open a results folder before running modal analysis.")
            return
        try:
            from fenice import modal

            self.series.refresh()
            snapshots = self.series.snapshots
            if len(snapshots) < 3:
                raise ValueError("Snapshot DMD needs at least three snapshots. Use time-delay DMD for a sparse load record.")
            field_name = self.cb_field.currentText()
            component = None
            raw_field = field_name
            if field_name == "Velocity":
                raw_field = "Velocity"
                component = {"x": 0, "y": 1, "z": 2}[self.cb_component.currentText()]
            times = [snap.time for snap in snapshots]
            if any(time is None for time in times):
                raise ValueError("The snapshot index has no times. A PVD time index is required for POD/DMD.")
            first = next((i for i, time in enumerate(times) if time >= self.first_time.value()), len(times))
            last = len(times)
            if last - first >= 3:
                intervals = np.diff(np.asarray(times, dtype=float))
                cadence = float(np.median(intervals[:-1])) if intervals.size > 1 else float(intervals[-1])
                if cadence > 0 and abs(float(intervals[-1]) - cadence) > 1e-2 * cadence:
                    last -= 1  # a final snapshot at t_end is often off the regular output cadence
            if last - first < 3:
                raise ValueError(f"Fewer than three snapshots occur at or after t = {self.first_time.value():g}.")
            snap = modal.load_snapshots(self.folder, raw_field, component=component, every=self.every.value(), first=first, last=last,
                                        tol=self.tolerance.value())
            X, dt, weights = snap["X"], snap["dt"], snap["weights"]
            if X.shape[1] < 3:
                raise ValueError("Selected snapshot interval leaves fewer than three snapshots.")
            rank = min(self.rank.value(), X.shape[1] - 1)
            pod = modal.pod(X, weights, rank=rank)
            fractions = pod["energies"] / max(float(np.sum(modal.pod(X, weights)["energies"])), np.finfo(float).tiny)
            centred = X - pod["mean"][:, None]
            dmd = modal.dmd(centred, dt, rank=min(2, X.shape[1] - 1))
            positive = np.flatnonzero(dmd["frequency"] > 0)
            if not len(positive):
                raise ValueError("Snapshot DMD did not find a positive-frequency eigenvalue pair.")
            index = int(positive[np.argmax(dmd["frequency"][positive])])
            frequency = float(dmd["frequency"][index])
            scaled_frequency = frequency * self.length_scale.value() / self.reference_speed.value()
            growth_ratio = float(dmd["growth"][index] / (2 * np.pi * frequency))
            pair_energy = float(fractions[0] + fractions[1])
            pair_imbalance = float(abs(pod["energies"][0] - pod["energies"][1]) /
                                   max(float(pod["energies"][0] + pod["energies"][1]), np.finfo(float).tiny))
            phase_difference = float("nan")
            if pod["coefficients"].shape[0] >= 2:
                phase = []
                for coeff in pod["coefficients"][:2]:
                    design = np.column_stack((np.cos(2 * np.pi * frequency * snap["times"]),
                                              np.sin(2 * np.pi * frequency * snap["times"])))
                    c, s = np.linalg.lstsq(design, coeff - coeff.mean(), rcond=None)[0]
                    phase.append(float(np.degrees(np.arctan2(s, c))))
                phase_difference = abs((phase[0] - phase[1] + 180) % 360 - 180)

            mode_plane = []
            norm_w = weights / weights.sum()
            for part in (np.real(dmd["modes"][:, index]), np.imag(dmd["modes"][:, index])):
                projected = np.zeros_like(part)
                for mode in pod["modes"][:, :min(2, pod["modes"].shape[1])].T:
                    projected += np.sum(norm_w * part * mode) * mode
                denominator = float(np.sum(norm_w * part * part))
                mode_plane.append(float(np.sqrt(np.sum(norm_w * projected * projected) / denominator)) if denominator else 0.0)

            signal = self.cb_signal.currentText()
            delay_results, delay_errors = self._delay_results(_monitor_data(self.series))
            load_report = self._delay_report(delay_results, delay_errors)
            self._load_result = delay_results.get(signal)

            ny, nx = snap["shape"]
            if self.axes.size != 6:
                self.figure.clear()
                self.axes = self.figure.subplots(2, 3)
            extent = (float(snap["x"][0]), float(snap["x"][-1]), float(snap["y"][0]), float(snap["y"][-1]))
            mode_fields = ((pod["modes"][:, 0], "POD mode 1"), (pod["modes"][:, 1], "POD mode 2"),
                           (np.real(dmd["modes"][:, index]), "Exact DMD · real"),
                           (np.imag(dmd["modes"][:, index]), "Exact DMD · imaginary"))
            for ax, (mode, title) in zip(self.axes.flat[:4], mode_fields):
                field_image = np.asarray(mode).reshape(ny, nx)
                scale = float(np.max(np.abs(field_image)))
                if scale > 0:
                    field_image = field_image / scale
                ax.clear()
                ax.imshow(field_image, origin="lower", extent=extent, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
                ax.set_title(title + " (normalized)")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.grid(False)
            energy_ax, phase_ax = self.axes.flat[4], self.axes.flat[5]
            energy_ax.clear()
            energy_ax.semilogy(np.arange(1, len(fractions) + 1), np.maximum(fractions, np.finfo(float).tiny), "o-",
                               color=COLORS["route_run"], markersize=3.5)
            energy_ax.set_xlabel("POD mode")
            energy_ax.set_ylabel("Fluctuation energy fraction")
            energy_ax.set_title("POD energy")
            phase_ax.clear()
            if pod["coefficients"].shape[0] >= 2:
                phase_ax.plot(pod["coefficients"][0], pod["coefficients"][1], ".", color=COLORS["route_results"], markersize=3)
            phase_ax.set_xlabel("Mode 1 coefficient")
            phase_ax.set_ylabel("Mode 2 coefficient")
            phase_ax.set_title(f"POD phase portrait ({phase_difference:.1f}° apart)")
            phase_ax.set_aspect("equal", adjustable="datalim")
            _style_axes(self.axes)
            for ax in self.axes.flat[:4]:
                ax.grid(False)
            self.figure.tight_layout()
            self.canvas.draw_idle()

            lines = [f"{field_name}: {X.shape[1]} snapshots, dt = {dt:.6g}, step jitter = {snap['jitter']:.3g}, {X.shape[0]:,} cells.",
                     f"POD modes 1–2: {pair_energy:.3f} of fluctuation energy; relative pair imbalance {pair_imbalance:.3f}.",
                     "POD energy fractions: " + ", ".join(f"{value:.3f}" for value in fractions[:8]),
                     f"Leading exact DMD pair: fL/U = {scaled_frequency:.5f}; |λ| = {abs(dmd['eigenvalues'][index]):.6g}; "
                     f"growth / (2πf) = {growth_ratio:+.4g}.",
                     f"POD modes 1 and 2 phase difference = {phase_difference:.2f}°; DMD mode share in POD 1–2 plane "
                     f"(real, imaginary) = {mode_plane[0]:.3f}, {mode_plane[1]:.3f}.", load_report]
            self.report.setPlainText("\n".join(lines))
            self._result = {"snap": snap, "pod": pod, "fractions": fractions, "dmd": dmd, "index": index,
                            "phase_difference": phase_difference, "frequency": frequency, "scaled_frequency": scaled_frequency,
                            "pair_energy": pair_energy,
                            "pair_imbalance": pair_imbalance, "load_report": load_report}
            self._view_kind = "snapshot"
            self.status.setText("Modal analysis complete. Exact DMD uses the uniformly spaced snapshot sequence; inspect reported jitter before interpreting phase.")
        except Exception as exc:  # noqa: BLE001 - invalid spacing or field selection must not close the workbench
            self.status.setText(f"{type(exc).__name__}: {exc}")
            self.report.clear()
            traceback.print_exc()

    def save_state(self) -> dict:
        return {"field": self.cb_field.currentText(), "component": self.cb_component.currentText(), "first_time": self.first_time.value(),
                "every": self.every.value(), "tolerance": self.tolerance.value(), "rank": self.rank.value(),
                "signal": self.cb_signal.currentText(), "compare": self.cb_compare.currentText(),
                "length_scale": self.length_scale.value(), "reference_speed": self.reference_speed.value(),
                "delay_dt": self.delay_dt.value(), "delay_rows": self.delay_rows.value(), "analyzed": self._result is not None,
                "load_analyzed": self._load_result is not None, "view": self._view_kind}

    def restore_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        for key, combo in (("field", self.cb_field), ("component", self.cb_component), ("signal", self.cb_signal), ("compare", self.cb_compare)):
            if combo.findText(str(state.get(key, ""))) >= 0:
                combo.setCurrentText(str(state[key]))
        if "diameter" in state and "length_scale" not in state:
            state = dict(state, length_scale=state["diameter"])
        for key in ("first_time", "tolerance", "length_scale", "reference_speed", "delay_dt"):
            if key in state:
                getattr(self, key).setValue(float(state[key]))
        if "velocity" in state and "reference_speed" not in state:
            self.reference_speed.setValue(float(state["velocity"]))
        for key in ("every", "rank", "delay_rows"):
            if key in state:
                getattr(self, key).setValue(int(state[key]))
        if state.get("view") == "load" and state.get("load_analyzed"):
            self.analyze_load_dmd()
        elif state.get("analyzed"):
            self.analyze()
        elif state.get("load_analyzed"):
            self.analyze_load_dmd()

    def shutdown(self) -> None:
        pass


class SensitivityAnalysisTab(QWidget):
    """Sobol indices and response plots for general sampled parameter studies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._summary: dict | None = None
        self._sample: dict | None = None
        note = QLabel("Compare input effects with a sampled response. Load a summary, a response table, or Saltelli outputs.")
        note.setObjectName("note")
        note.setWordWrap(True)
        self.btn_summary = QPushButton("Load summary JSON...")
        self.btn_summary.setObjectName("secondaryButton")
        self.btn_summary.clicked.connect(self.load_summary_dialog)
        self.btn_sample = QPushButton("Load sample CSV...")
        self.btn_sample.setObjectName("secondaryButton")
        self.btn_sample.setToolTip("Load a numeric table of sampled inputs and one or more responses.")
        self.btn_sample.clicked.connect(self.load_sample_dialog)
        self.btn_csv = QPushButton("Calculate indices from CSV...")
        self.btn_csv.setObjectName("secondaryButton")
        self.btn_csv.setToolTip("Calculate Sobol indices from output values ordered A, B, AB1 … ABd.")
        self.btn_csv.clicked.connect(self.load_csv_dialog)
        self.n = QSpinBox(); self.n.setRange(2, 1000000); self.n.setValue(32)
        self.parameters = QLineEdit("input 1, input 2")
        self.parameters.setMinimumWidth(280)
        self.controls = QHBoxLayout()
        self.controls.addWidget(self.btn_summary)
        self.controls.addWidget(self.btn_sample)
        self.controls.addWidget(self.btn_csv)
        self.controls.addSpacing(12)
        self.controls.addWidget(QLabel("Base samples n"))
        self.controls.addWidget(self.n)
        self.controls.addWidget(QLabel("Parameters"))
        self.controls.addWidget(self.parameters, 1)
        self.cb_sample_x = QComboBox()
        self.cb_sample_y = QComboBox()
        self.cb_sample_color = QComboBox()
        self.chk_reference = QCheckBox("Reference")
        self.reference_value = _spin(0.0, low=-1e12, high=1e12, decimals=6)
        self.reference_value.setEnabled(False)
        self.chk_reference.toggled.connect(self.reference_value.setEnabled)
        self.chk_reference.toggled.connect(self._draw)
        self.reference_value.valueChanged.connect(self._draw)
        self.cb_sample_x.currentTextChanged.connect(self._draw)
        self.cb_sample_y.currentTextChanged.connect(self._draw)
        self.cb_sample_color.currentTextChanged.connect(self._draw)
        self.sample_controls = QHBoxLayout()
        for label, widget in (("X", self.cb_sample_x), ("Response Y", self.cb_sample_y), ("Color by", self.cb_sample_color)):
            self.sample_controls.addWidget(QLabel(label))
            self.sample_controls.addWidget(widget)
        self.sample_controls.addWidget(self.chk_reference)
        self.sample_controls.addWidget(self.reference_value)
        self.sample_controls.addStretch(1)
        self.figure = Figure(figsize=(10, 5))
        self.figure.set_facecolor(COLORS["bg_window"])
        self.axes = self.figure.subplots(1, 2, gridspec_kw={"width_ratios": (1.1, 1.0)})
        self.ax, self.sample_ax = self.axes
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(260)
        self.report = QPlainTextEdit(); self.report.setReadOnly(True); self.report.setMaximumHeight(160)
        self.report.setPlaceholderText("Sobol intervals and study diagnostics will appear here.")
        self.status = QLabel("Load a study summary and sample table, or calculate indices from Saltelli outputs.")
        self.status.setObjectName("note")
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(note)
        layout.addLayout(self.controls)
        layout.addLayout(self.sample_controls)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.report)
        layout.addWidget(self.status)
        self._draw()

    def load_summary_dialog(self) -> None:
        start = str(Path("results") / "validation")
        path, _ = QFileDialog.getOpenFileName(self, "Sobol study summary", start, "JSON files (*.json)")
        if path:
            try:
                with open(path, encoding="utf-8") as handle:
                    self.show_summary(json.load(handle), Path(path))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.status.setText(f"Cannot read the sensitivity summary: {exc}")

    def load_csv_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Saltelli output values", str(Path("results") / "validation"), "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.reader(handle))
            rows = [row for row in rows if row and any(cell.strip() for cell in row)]
            if len(rows) < 2:
                raise ValueError("The CSV needs a header and one numeric output column.")
            header = [cell.strip() for cell in rows[0]]
            values = np.asarray([[float(cell) for cell in row] for row in rows[1:]], dtype=float)
            if values.ndim == 1:
                values = values[:, None]
            if len(header) != values.shape[1]:
                raise ValueError("CSV header and data column counts do not match.")
            if values.shape[1] != 1:
                raise ValueError("The CSV must have one numeric output column, ordered A, B, AB1, ... .")
            names = [part.strip() for part in self.parameters.text().split(",") if part.strip()]
            if not names:
                raise ValueError("Enter at least one parameter name.")
            if values.shape[0] != self.n.value() * (len(names) + 2):
                raise ValueError(f"Expected n(d+2) = {self.n.value() * (len(names) + 2)} rows; found {values.shape[0]}.")
            from fenice import uq
            y = values[:, 0]
            s1, st = uq.sobol_indices(y, self.n.value(), len(names))
            s1_lo, s1_hi, st_lo, st_hi = uq.sobol_bootstrap(y, self.n.value(), len(names), n_boot=1000, seed=0)
            result = {"names": names, "S1": s1.tolist(), "ST": st.tolist(),
                      "S1_ci": [s1_lo.tolist(), s1_hi.tolist()], "ST_ci": [st_lo.tolist(), st_hi.tolist()]}
            self.show_summary(result, Path(path))
        except Exception as exc:  # noqa: BLE001 - malformed files stay in the analysis view
            self.status.setText(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

    @staticmethod
    def _normalize_sample(sample: dict) -> dict[str, list[float]]:
        if not isinstance(sample, dict):
            raise ValueError("Sample data must be an object of named numeric columns.")
        columns = {}
        for raw_name, raw_values in sample.items():
            name = str(raw_name).strip()
            if not name:
                continue
            try:
                values = np.asarray(raw_values, dtype=float)
            except (TypeError, ValueError):
                continue
            if values.ndim == 1 and values.size:
                columns[name] = values
        if len(columns) < 2:
            raise ValueError("The sample needs at least two numeric columns: an input and a response.")
        if len({values.size for values in columns.values()}) != 1:
            raise ValueError("Sample columns must contain the same number of rows.")
        if not all(np.isfinite(values).all() for values in columns.values()):
            raise ValueError("Sample columns must contain only finite numbers.")
        return {key: values.tolist() for key, values in columns.items()}

    @staticmethod
    def _read_sample_csv(path: Path) -> dict[str, list[float]]:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("The sample CSV needs a header row.")
            rows = list(reader)
        if not rows:
            raise ValueError("The sample CSV has no data rows.")
        columns = {}
        for field in reader.fieldnames:
            name = str(field).strip()
            if not name:
                continue
            try:
                columns[name] = [float(row[field]) for row in rows]
            except (KeyError, TypeError, ValueError):
                continue
        return SensitivityAnalysisTab._normalize_sample(columns)

    def _configure_sample_fields(self) -> None:
        if self._sample is None:
            _set_combo(self.cb_sample_x, [])
            _set_combo(self.cb_sample_y, [])
            _set_combo(self.cb_sample_color, ["None"])
            self.chk_reference.setChecked(False)
            self.reference_value.setEnabled(False)
            for widget in (self.cb_sample_x, self.cb_sample_y, self.cb_sample_color, self.chk_reference):
                widget.setEnabled(False)
            return
        for widget in (self.cb_sample_x, self.cb_sample_y, self.cb_sample_color, self.chk_reference):
            widget.setEnabled(True)
        self.reference_value.setEnabled(self.chk_reference.isChecked())
        names = list(self._sample)
        output_names = {"response", "output", "result", "y", "cd", "cl", "drag", "lift"}
        y_default = next((name for name in names if name.casefold().replace(" ", "_") in output_names), names[-1])
        x_default = next((name for name in names if name != y_default), names[0])
        color_default = next((name for name in names if name not in (x_default, y_default)), "None")
        old_x, old_y, old_color = (self.cb_sample_x.currentText(), self.cb_sample_y.currentText(), self.cb_sample_color.currentText())
        _set_combo(self.cb_sample_x, names, old_x if old_x in names else x_default)
        _set_combo(self.cb_sample_y, names, old_y if old_y in names else y_default)
        color_names = ["None"] + [name for name in names if name != self.cb_sample_y.currentText()]
        _set_combo(self.cb_sample_color, color_names, old_color if old_color in color_names else color_default)
        reference = self._summary_reference()
        if reference is not None:
            self.reference_value.setValue(reference)
            response_name = str(self._summary.get("response_name", "")).casefold() if self._summary else ""
            legacy_response = self._summary is not None and "cd_ref" in self._summary and self.cb_sample_y.currentText().casefold() == "cd"
            if (response_name and self.cb_sample_y.currentText().casefold() == response_name) or legacy_response:
                self.chk_reference.setChecked(True)

    def _summary_reference(self) -> float | None:
        if not self._summary:
            return None
        value = self._summary.get("reference_value", self._summary.get("cd_ref"))
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if np.isfinite(number) else None

    def load_sample_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Sample response table", str(Path("results") / "validation"), "CSV files (*.csv)")
        if not path:
            return
        try:
            self._sample = self._read_sample_csv(Path(path))
            self._configure_sample_fields()
            self._draw()
            rows = len(next(iter(self._sample.values())))
            self.status.setText(f"Loaded {rows} sample rows from {Path(path).name}.")
        except (OSError, ValueError, TypeError) as exc:
            self.status.setText(f"Cannot read the sample CSV: {exc}")

    def _draw(self, *_args) -> None:
        self.figure.clear()
        self.axes = self.figure.subplots(1, 2, gridspec_kw={"width_ratios": (1.1, 1.0)})
        self.ax, self.sample_ax = self.axes
        if self._summary is not None:
            summary = self._summary
            names = [str(name) for name in summary["names"]]
            s1, st = np.asarray(summary["S1"], float), np.asarray(summary["ST"], float)
            s1_ci = np.asarray(summary["S1_ci"], float)
            st_ci = np.asarray(summary["ST_ci"], float)
            x = np.arange(len(names))
            err1 = np.maximum(np.vstack((s1 - s1_ci[0], s1_ci[1] - s1)), 0)
            errt = np.maximum(np.vstack((st - st_ci[0], st_ci[1] - st)), 0)
            self.ax.bar(x - 0.19, s1, 0.38, yerr=err1, capsize=3, color=COLORS["route_results"], label="First order S₁")
            self.ax.bar(x + 0.19, st, 0.38, yerr=errt, capsize=3, color=COLORS["primary"], label="Total Sₜ")
            self.ax.set_xticks(x, names)
            self.ax.set_ylabel("Sobol index")
            self.ax.set_title("Parameter effects · 95% bootstrap intervals")
            self.ax.legend(loc="best")
        else:
            self.ax.text(0.5, 0.5, "Load a sensitivity summary\nto see Sobol indices", ha="center", va="center",
                         transform=self.ax.transAxes, color=COLORS["ink_soft"])
            self.ax.set_title("Parameter effects")

        self.sample_ax.set_title("Response in the sample")
        x_name, y_name, color_name = (self.cb_sample_x.currentText(), self.cb_sample_y.currentText(),
                                      self.cb_sample_color.currentText())
        if self._sample is not None and x_name in self._sample and y_name in self._sample:
            sample = self._sample
            scatter_args = {"s": 34, "edgecolors": COLORS["bg_surface"], "linewidths": 0.5}
            if color_name in sample:
                scatter = self.sample_ax.scatter(sample[x_name], sample[y_name], c=sample[color_name], cmap="viridis", **scatter_args)
                self.figure.colorbar(scatter, ax=self.sample_ax, label=color_name)
            else:
                self.sample_ax.scatter(sample[x_name], sample[y_name], color=COLORS["route_results"], **scatter_args)
            if self.chk_reference.isChecked():
                reference = self.reference_value.value()
                self.sample_ax.axhline(reference, color=COLORS["ink_soft"], linestyle=":", linewidth=1.2,
                                       label=f"Reference {y_name} = {reference:.5g}")
                self.sample_ax.legend(loc="best")
            self.sample_ax.set_title(f"{y_name} against {x_name}")
            self.sample_ax.set_xlabel(x_name)
            self.sample_ax.set_ylabel(y_name)
        else:
            self.sample_ax.text(0.5, 0.5, "Load a numeric sample table\nto plot a response against an input",
                                ha="center", va="center", transform=self.sample_ax.transAxes, color=COLORS["ink_soft"])
            self.sample_ax.set_xlabel("Input")
            self.sample_ax.set_ylabel("Response")
        _style_axes(self.axes)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def show_summary(self, summary: dict, source: Path | None = None) -> None:
        if not isinstance(summary, dict):
            raise ValueError("The sensitivity summary must be a JSON object.")
        names = [str(name) for name in summary["names"]]
        s1, st = np.asarray(summary["S1"], float), np.asarray(summary["ST"], float)
        if not names or s1.shape != (len(names),) or st.shape != (len(names),):
            raise ValueError("Summary names, S1 and ST must contain the same number of parameters.")
        s1_ci = np.asarray(summary["S1_ci"], float)
        st_ci = np.asarray(summary["ST_ci"], float)
        if s1_ci.shape != (2, len(names)) or st_ci.shape != (2, len(names)):
            raise ValueError("Summary confidence intervals must be [lower, upper] arrays matching the parameter list.")
        if "n" in summary:
            self.n.setValue(int(summary["n"]))
        self.parameters.setText(", ".join(names))
        self._summary = dict(summary)
        lines = [f"{name}: S1 = {s1[i]:.4g} [{s1_ci[0, i]:.4g}, {s1_ci[1, i]:.4g}], "
                 f"ST = {st[i]:.4g} [{st_ci[0, i]:.4g}, {st_ci[1, i]:.4g}]" for i, name in enumerate(names)]
        diagnostics = summary.get("diagnostics", [])
        if isinstance(diagnostics, list):
            lines.extend(str(value) for value in diagnostics)
        if "spearman_u" in summary:
            lines.append(f"Study rank-correlation diagnostic = {float(summary['spearman_u']):+.4g}.")
        reference = self._summary_reference()
        observed = summary.get("reference_observed", summary.get("cd_ref_run"))
        if reference is not None and observed is not None:
            observed = float(observed)
            error = float(summary.get("reference_error", summary.get("ref_error", abs(observed / reference - 1))))
            lines.append(f"Reference check: observed {observed:.5g} vs expected {reference:.5g}; error {100 * error:.2f}%.")
        if "max_drift" in summary:
            lines.append(f"Maximum late-window response drift: {100 * float(summary['max_drift']):.3g}%.")
        checks = [key for key in summary if key.startswith("pass_")]
        if checks:
            passed = sum(bool(summary[key]) for key in checks)
            lines.append(f"Study checks: {passed} of {len(checks)} passed.")
        lines.append("Intervals describe estimator sampling error; they do not include solver discretization error.")
        self.report.setPlainText("\n".join(lines))
        if isinstance(summary.get("a_sample"), dict):
            self._sample = self._normalize_sample(summary["a_sample"])
        elif source and source.suffix.casefold() == ".json" and (source.parent / "a_sample.csv").is_file():
            self._sample = self._read_sample_csv(source.parent / "a_sample.csv")
        else:
            self._sample = None
        if reference is not None:
            self.reference_value.setValue(reference)
        self._configure_sample_fields()
        self._draw()
        self.status.setText(f"Loaded {source.name}." if source else "Sensitivity indices calculated from the Saltelli output order.")

    def save_state(self) -> dict:
        return {"summary": self._summary, "a_sample": self._sample, "n": self.n.value(), "parameters": self.parameters.text(),
                "sample_x": self.cb_sample_x.currentText(), "sample_y": self.cb_sample_y.currentText(),
                "sample_color": self.cb_sample_color.currentText(), "show_reference": self.chk_reference.isChecked(),
                "reference_value": self.reference_value.value()}

    def restore_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        self.n.setValue(int(state.get("n", self.n.value())))
        self.parameters.setText(str(state.get("parameters", self.parameters.text())))
        if isinstance(state.get("summary"), dict):
            try:
                self.show_summary(state["summary"])
            except (ValueError, KeyError, TypeError) as exc:
                self.status.setText(f"Saved sensitivity result is invalid: {exc}")
        if isinstance(state.get("a_sample"), dict):
            try:
                self._sample = self._normalize_sample(state["a_sample"])
                self._configure_sample_fields()
            except (ValueError, KeyError, TypeError) as exc:
                self.status.setText(f"Saved sample data is invalid: {exc}")
        for key, combo in (("sample_x", self.cb_sample_x), ("sample_y", self.cb_sample_y), ("sample_color", self.cb_sample_color)):
            if combo.findText(str(state.get(key, ""))) >= 0:
                combo.setCurrentText(str(state[key]))
        if "reference_value" in state:
            self.reference_value.setValue(float(state["reference_value"]))
        self.chk_reference.setChecked(bool(state.get("show_reference", self.chk_reference.isChecked())))
        self._draw()
