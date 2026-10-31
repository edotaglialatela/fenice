"""Geometry module: the domain, the immersed body, the Cartesian mesh, the side conditions and the solid cells they make.

The node owns its geometry (a parameter set of its own, from a case's defaults at first); linked to a solver it writes them into the
solver's setup fields, which the run reads."""
from __future__ import annotations

from gui.modules import ModuleSpec, Port

_MAX_CELLS = 40_000_000   # the largest level whose solid cells are computed


def _make(project=None):
    import logging

    import numpy as np
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget

    import fenice
    from fenice import geometry
    from gui.setup_dialog import SetupDialog
    from gui.solver import amr_gui_defaults, available_cases, case_defaults
    from gui.theme import COLORS

    logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)   # "ignoring fixed limits" with the equal aspect

    def flag(p: dict, key: str, default: str = "false") -> bool:
        return str(p.get(key, default)).strip().lower() in ("1", "true", "yes", "on")

    OPPOSITE = {'left': 'right', 'right': 'left', 'bottom': 'top', 'top': 'bottom', 'front': 'back', 'back': 'front'}

    class GeometryPanel(QWidget):
        def __init__(self):
            super().__init__()
            self.fig = Figure(figsize=(8, 5))
            self.fig.set_facecolor(COLORS["bg_window"])
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            self.btn_refresh = QPushButton("Refresh from Setup")
            self.btn_zoom = QPushButton("Zoom to body")
            self.cb_level = QComboBox()
            self.chk_grid = QCheckBox("Grid")
            self.chk_grid.setChecked(True)
            self.chk_solid = QCheckBox("Solid cells")
            self.chk_solid.setChecked(True)
            self.info = QLabel("")
            self.info.setWordWrap(True)
            self.info.setObjectName("note")
            self.warn = QLabel("")
            self.warn.setStyleSheet("color: #b00020;")
            self.warn.setWordWrap(True)
            self.edit_nx, self.edit_ny = QLineEdit(), QLineEdit()
            for e in (self.edit_nx, self.edit_ny):
                e.setMaximumWidth(70)
            self.btn_apply = QPushButton("Apply mesh")
            self.btn_apply.setToolTip("Set nx and ny of the base mesh in Setup")
            self.btn_half = QPushButton("\u00bd\u00d7")
            self.btn_half.setToolTip("Halve nx and ny: a quarter of the cells in 2-D, an eighth in 3-D")
            self.btn_double = QPushButton("2\u00d7")
            self.btn_double.setToolTip("Double nx and ny: four times the cells in 2-D, eight times in 3-D (the usual step of a grid study)")
            self.fields: dict[str, QLineEdit] = {}   # solver key -> edit: the domain and the body
            for key in ("Lx", "Ly", "Lz", "nz", "ibm_radius", "ibm_cx", "ibm_cy", "ibm_cz"):
                self.fields[key] = e = QLineEdit()
                e.setMaximumWidth(70)
                e.editingFinished.connect(lambda k=key: self._push({k: self.fields[k].text()}))
            self.amr_fields: dict[str, QLineEdit] = {}   # solver key -> edit: the adaptive mesh
            for key in ("amr_max_level", "amr_init_levels", "amr_criterion", "amr_refine_threshold", "amr_derefine_threshold",
                        "amr_diffusion", "amr_refine_every"):
                self.amr_fields[key] = e = QLineEdit()
                e.setMaximumWidth(54)
                e.editingFinished.connect(lambda k=key: self._push({k: self.amr_fields[k].text()}))
            self.chk_amr = QCheckBox("AMR")
            self.chk_amr.setToolTip("Adaptive mesh refinement (2-D only): leaves are refined where a chosen variable jumps across a face")
            self.chk_amr.toggled.connect(lambda on: self._push({"use_amr": on}))
            self.amr_ind: dict[str, QCheckBox] = {}
            for key, text in (("density", "\u03c1"), ("velocity", "|u|"), ("pressure", "p")):
                self.amr_ind[key] = b = QCheckBox(text)
                b.setToolTip(f"Refine where the {key} varies across a face by more than the criterion")
                b.toggled.connect(lambda _on: self._push({"amr_indicator": self._indicator_text()}))
            amr_tips = {"amr_max_level": "Finest refinement level", "amr_init_levels": "Uniform levels at the start, before the indicator acts",
                        "amr_criterion": "Relative variation across a face that counts as 1 in the indicator",
                        "amr_refine_threshold": "Refine a leaf where the indicator reaches this value",
                        "amr_derefine_threshold": "Join four siblings when their parent's indicator is below this value (not above the refine threshold)",
                        "amr_diffusion": "Smoothing iterations of the refinement flags",
                        "amr_refine_every": "Remesh every N steps"}
            for key, tip in amr_tips.items():
                self.amr_fields[key].setToolTip(tip)
            amr_row = QHBoxLayout()
            amr_row.addWidget(self.chk_amr)
            self.domain_labels_amr: dict[str, QLabel] = {}
            for key, text in (("amr_max_level", "levels"), ("amr_init_levels", "initial")):
                self.domain_labels_amr[key] = QLabel(text)
                amr_row.addWidget(self.domain_labels_amr[key])
                amr_row.addWidget(self.amr_fields[key])
            self.domain_labels_amr["indicator"] = QLabel("indicator")
            amr_row.addWidget(self.domain_labels_amr["indicator"])
            for b in self.amr_ind.values():
                amr_row.addWidget(b)
            for key, text in (("amr_criterion", "criterion"), ("amr_refine_threshold", "refine \u2265"), ("amr_derefine_threshold", "join <"),
                              ("amr_diffusion", "smooth"), ("amr_refine_every", "every")):
                self.domain_labels_amr[key] = QLabel(text)
                amr_row.addWidget(self.domain_labels_amr[key])
                amr_row.addWidget(self.amr_fields[key])
            amr_row.addWidget(QLabel("steps"))
            amr_row.addStretch(1)
            self.chk_body = QCheckBox("Immersed body")
            self.cb_shape = QComboBox()
            self.cb_shape.setEditable(True)   # any "NACA dddd"; Airfoil reads the case's ibm_file
            self.cb_shape.addItems(["Cylinder", "Sphere", "NACA 0012", "Airfoil"])
            self.cb_shape.setMaximumWidth(130)
            self.chk_body.toggled.connect(lambda on: self._push({"use_ibm": on}))
            self.cb_shape.activated.connect(lambda _: self._push({"ibm_shape": self.cb_shape.currentText()}))
            self.cb_shape.lineEdit().editingFinished.connect(lambda: self._push({"ibm_shape": self.cb_shape.currentText()}))
            self.domain_labels: dict[str, QLabel] = {}
            domain_row = QHBoxLayout()
            for key, text in (("Lx", "Domain Lx"), ("Ly", "Ly"), ("Lz", "Lz"), ("nz", "nz")):
                self.domain_labels[key] = QLabel(text)
                domain_row.addWidget(self.domain_labels[key])
                domain_row.addWidget(self.fields[key])
            domain_row.addStretch(1)
            body_row = QHBoxLayout()
            body_row.addWidget(self.chk_body)
            body_row.addWidget(self.cb_shape)
            for key, text in (("ibm_radius", "radius"), ("ibm_cx", "cx"), ("ibm_cy", "cy"), ("ibm_cz", "cz")):
                self.domain_labels[key] = QLabel(text)
                body_row.addWidget(self.domain_labels[key])
                body_row.addWidget(self.fields[key])
            body_row.addStretch(1)
            self.bc_combos: dict[str, QComboBox] = {}
            bc_row = QHBoxLayout()
            for side in ("left", "right", "bottom", "top", "front", "back"):
                cb = QComboBox()
                self.bc_combos[side] = cb
                lbl = QLabel(side.capitalize())
                cb._label = lbl
                bc_row.addWidget(lbl)
                bc_row.addWidget(cb)
            bc_row.addStretch(1)
            mesh_row = QHBoxLayout()
            for w in (QLabel("Base mesh nx"), self.edit_nx, QLabel("ny"), self.edit_ny, self.btn_apply, self.btn_half, self.btn_double):
                mesh_row.addWidget(w)
            mesh_row.addStretch(1)
            row = QHBoxLayout()
            for w in (self.btn_refresh, QLabel("Mesh level"), self.cb_level, self.chk_grid, self.chk_solid, self.btn_zoom):
                row.addWidget(w)
            row.addStretch(1)
            lay = QVBoxLayout(self)
            lay.addLayout(row)
            lay.addLayout(domain_row)
            lay.addLayout(mesh_row)
            lay.addLayout(body_row)
            lay.addLayout(amr_row)
            lay.addLayout(bc_row)
            lay.addWidget(self.warn)
            lay.addWidget(self.toolbar)
            lay.addWidget(self.canvas, 1)
            lay.addWidget(self.info)
            self.p: dict | None = None
            self.body = None
            self._grid = None
            self._bc_texts: list = []
            self._busy = False
            self._last_case = None
            self._timer = QTimer(self)   # redraw the grid lines once a zoom or a resize has settled
            self._timer.setSingleShot(True)
            self._timer.setInterval(90)
            self._timer.timeout.connect(self._draw_grid)
            self.btn_refresh.clicked.connect(self.refresh)
            self.project = None
            self._touched = False   # the user has edited this geometry: it wins over the solver's when they are linked
            self.btn_zoom.clicked.connect(self._zoom_to_body)
            self.btn_apply.clicked.connect(lambda: self._set_mesh(self.edit_nx.text(), self.edit_ny.text()))
            self.btn_half.clicked.connect(lambda: self._scale_mesh(0.5))
            self.btn_double.clicked.connect(lambda: self._scale_mesh(2.0))
            self.edit_nx.returnPressed.connect(self.btn_apply.click)
            self.edit_ny.returnPressed.connect(self.btn_apply.click)
            for side, cb in self.bc_combos.items():
                cb.activated.connect(lambda _, s=side: self._set_bc(s))
            self.cb_level.currentIndexChanged.connect(lambda _: self._draw())
            self.chk_grid.toggled.connect(lambda _: self._draw())
            self.chk_solid.toggled.connect(lambda _: self._draw())
            self._grid_view = None
            self.canvas.mpl_connect("draw_event", self._view_changed)   # the limits can also change while drawing (equal aspect)
            self.canvas.mpl_connect("resize_event", lambda e: self._timer.start())
            self._drag = None
            self.canvas.mpl_connect("scroll_event", self._on_scroll)
            self.canvas.mpl_connect("button_press_event", self._on_press)
            self.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self.canvas.mpl_connect("button_release_event", self._on_release)

        def _on_setup_changed(self) -> None:
            self.refresh(reset_view=True)

        def attach(self, new) -> None:
            """Link to the project of a solver (None: unlinked). An edited geometry is written into the solver's fields, an untouched one
            adopts the solver's."""
            if self.project is not None:
                try:
                    self.project.setup_changed.disconnect(self._on_setup_changed)
                except (TypeError, RuntimeError):
                    pass
            self.project = new
            if new is not None:
                new.setup_changed.connect(self._on_setup_changed)   # another case loaded on the solver
                if self._touched and self.p is not None and new.editor is not None:
                    try:
                        new.editor.apply_geometry({k: v for k, v in self.p.items() if k in _GEOMETRY_KEYS})
                    except ValueError:
                        pass
            self.refresh(reset_view=True)

        def _indicator_text(self) -> str:
            return "+".join(k for k, b in self.amr_ind.items() if b.isChecked())

        def _choices(self) -> dict:
            ed = self.project.editor if self.project is not None else None
            return ed.bc_choices() if ed is not None else dict(SetupDialog._BC_LABELS)

        def _load_defaults(self) -> None:
            try:
                cases = available_cases()
                name = "Sod Shock Tube" if "Sod Shock Tube" in cases else cases[0]
                p = {**case_defaults(name), **amr_gui_defaults(name)}
                p["test_case"] = name
            except Exception:   # noqa: BLE001 - the solver is not built: an empty domain is shown
                p = {"Lx": "1.0", "Ly": "1.0", "nx": "100", "ny": "100", "nz": "1", "Lz": "1.0"}
            self.p = p

        def showEvent(self, event):
            super().showEvent(event)
            self.refresh()   # the page shows what Setup holds when it comes up

        # ---- the parameters
        def refresh(self, reset_view: bool = True) -> None:
            ed = self.project.editor if self.project is not None else None
            if ed is None:   # not linked: the geometry of its own
                if self.p is None:
                    self._load_defaults()
                p = self.p
            else:
                p = self._from_solver(ed)
                if p is None:
                    return
                self.p = p
            self._after_load(p, reset_view)

        def _from_solver(self, ed):
            project = self.project
            case = ed.combo_testcase.currentText()
            try:
                p = dict(case_defaults(case))   # what the case sets besides the geometry fields: chord, angle, file, AMR
            except Exception:   # noqa: BLE001 - the solver is not built or the case is unknown: the live values alone will do
                p = {}
            cur = project.current_params()
            if cur is not None:
                p.update({k: str(v) for k, v in cur.items()})
            p.update({k: str(v) for k, v in ed.geometry_values().items()})   # the geometry fields as they stand, valid or not
            p["test_case"] = case
            try:
                float(p["Lx"]), float(p["Ly"]), int(p["nx"]), int(p["ny"]), int(p["nz"])
            except (KeyError, ValueError):
                self.info.setText("A domain or mesh field is not a valid number: correct it above.")
                return None
            return p

        def _after_load(self, p: dict, reset_view: bool) -> None:
            try:
                self.body = geometry.body_from_params(p, fenice.ROOT)
            except (ValueError, OSError) as exc:
                self.body = None
                self.info.setText(f"The body cannot be drawn: {exc}")
            levels = int(p.get("amr_max_level", "0")) if flag(p, "use_amr") else 0
            keep = self.cb_level.currentIndex()
            if (p.get("test_case"), levels) != self._last_case:   # a new case or level count: show the finest mesh, where the body is resolved most
                keep, self._last_case = levels, (p.get("test_case"), levels)
            nx, ny = int(p["nx"]), int(p["ny"])
            self.cb_level.blockSignals(True)
            self.cb_level.clear()
            self.cb_level.addItems([f"{l}: {nx << l} x {ny << l}" for l in range(levels + 1)])
            self.cb_level.setCurrentIndex(min(max(keep, 0), levels))
            self.cb_level.blockSignals(False)
            self._fill_controls(p)
            self._draw(reset_view=reset_view)

        # ---- editing: the changes go to the Setup page, which keeps the only copy
        def _fill_controls(self, p: dict) -> None:
            choices = self._choices()
            by_name = {v: k for k, v in choices.items()}
            nz = int(p.get("nz", "1"))
            enabled = True
            for e, key in ((self.edit_nx, "nx"), (self.edit_ny, "ny")):
                e.setText(str(int(p[key])))
            for key, e in self.fields.items():
                if not e.hasFocus():
                    e.setText(str(p.get(key, "")))
            for key in ("Lz", "ibm_cz"):
                self.fields[key].setVisible(nz > 1 and (key == "Lz" or flag(p, "use_ibm")))
                self.domain_labels[key].setVisible(self.fields[key].isVisible())
            amr = flag(p, "use_amr")
            self.chk_amr.blockSignals(True)
            self.chk_amr.setChecked(amr)
            self.chk_amr.blockSignals(False)
            self.chk_amr.setEnabled(enabled)
            for key, e in self.amr_fields.items():
                if not e.hasFocus():
                    e.setText(str(p.get(key, "")))
                e.setEnabled(amr)
            tokens = [t.strip() for t in str(p.get("amr_indicator", "")).split("+")]
            for key, b in self.amr_ind.items():
                b.blockSignals(True)
                b.setChecked(key in tokens)
                b.blockSignals(False)
                b.setEnabled(amr)
            self.chk_body.blockSignals(True)
            self.chk_body.setChecked(flag(p, "use_ibm"))
            self.chk_body.blockSignals(False)
            self.cb_shape.blockSignals(True)
            self.cb_shape.setCurrentText(p.get("ibm_shape", ""))
            self.cb_shape.blockSignals(False)
            for key in ("ibm_radius", "ibm_cx", "ibm_cy"):
                self.fields[key].setEnabled(flag(p, "use_ibm") and enabled)
            self.cb_shape.setEnabled(flag(p, "use_ibm") and enabled)
            for w in (self.fields["Lx"], self.fields["Ly"], self.fields["Lz"], self.fields["nz"], self.chk_body):
                w.setEnabled(enabled)
            for w in (self.edit_nx, self.edit_ny, self.btn_apply, self.btn_half, self.btn_double):
                w.setEnabled(enabled)
            for side, cb in self.bc_combos.items():
                shown = side in ("left", "right", "bottom", "top") or nz > 1
                cb.setVisible(shown)
                cb._label.setVisible(shown)
                cb.blockSignals(True)
                cb.clear()
                cb.addItems(list(choices))
                cb.setEnabled(enabled)
                i = cb.findText(by_name.get(p.get(f"bc_{side}", ""), ""))
                cb.setCurrentIndex(max(i, 0) if choices else -1)
                cb.blockSignals(False)
            pairs = [("left", "right"), ("bottom", "top")] + ([("front", "back")] if nz > 1 else [])
            bad = [f"{a} / {b}" for a, b in pairs if (str(p.get(f"bc_{a}")) == "periodic") != (str(p.get(f"bc_{b}")) == "periodic")]
            msgs = ["Periodic must be set on both sides of a pair (" + ", ".join(bad) + "): the solver will refuse the run until it is."] if bad else []
            if flag(p, "use_amr"):
                if nz > 1:
                    msgs.append("AMR is 2-D only (nz must be 1).")
                if not any(t in self.amr_ind for t in tokens):
                    msgs.append("Tick at least one AMR indicator variable.")
                try:
                    if float(p["amr_derefine_threshold"]) > float(p["amr_refine_threshold"]):
                        msgs.append("The join threshold must not exceed the refine threshold.")
                except (KeyError, ValueError):
                    pass
            self.warn.setText("  ".join(msgs))

        def _push(self, values: dict) -> None:
            self._touched = True
            ed = self.project.editor if self.project is not None else None
            try:
                if ed is not None:
                    ed.apply_geometry(values)
                else:
                    new = {k: ("true" if v is True else "false" if v is False else str(v)) for k, v in values.items()}
                    for key in ("Lx", "Ly", "Lz", "ibm_radius", "ibm_cx", "ibm_cy", "ibm_cz"):
                        if key in new:
                            float(new[key])
                    for key in ("nx", "ny", "nz", "amr_max_level", "amr_init_levels", "amr_diffusion", "amr_refine_every"):
                        if key in new:
                            new[key] = str(int(new[key]))
                    for key in ("amr_criterion", "amr_refine_threshold", "amr_derefine_threshold"):
                        if key in new:
                            float(new[key])
                    self.p.update(new)
            except ValueError:
                self.warn.setText("A value is not a valid number.")
                return
            self.refresh(reset_view=False)

        def _set_mesh(self, nx_text: str, ny_text: str) -> None:
            try:
                nx, ny = int(nx_text), int(ny_text)
            except ValueError:
                self.warn.setText("nx and ny must be whole numbers.")
                return
            if nx < 1 or ny < 1:
                self.warn.setText("nx and ny must be at least 1.")
                return
            self._push({"nx": nx, "ny": ny})
            self._warn_mpi(nx, ny)

        def _scale_mesh(self, factor: float) -> None:
            if self.p is None:
                return
            nx, ny = int(self.p["nx"]), int(self.p["ny"])
            self._set_mesh(str(max(1, round(nx * factor))), str(max(1, round(ny * factor))))

        def _warn_mpi(self, nx: int, ny: int) -> None:
            cur = self.project.current_params() if self.project is not None else None
            if cur and str(cur.get("use_mpi")).lower() in ("true", "1") and int(cur.get("n_cpus", 1)) > max(nx, ny):
                self.warn.setText("The number of MPI processes exceeds max(nx, ny): Setup will refuse the run.")

        def _set_bc(self, side: str) -> None:
            name = self._choices()[self.bc_combos[side].currentText()]
            values = {f"bc_{side}": name}
            if name == "periodic":   # periodic is a pair: the opposite side follows
                values["bc_" + OPPOSITE[side]] = name
            self._push(values)

        # ---- drawing
        def _draw(self, reset_view: bool = False) -> None:
            p = self.p
            if p is None:
                return
            ax = self.ax
            Lx, Ly, nx, ny = float(p["Lx"]), float(p["Ly"]), int(p["nx"]), int(p["ny"])
            level = max(self.cb_level.currentIndex(), 0)
            view = None if reset_view else (ax.get_xlim(), ax.get_ylim())
            solid = "-"
            self._busy = True
            try:
                ax.clear()
                self._grid = None
                self._bc_texts = []
                ax.set_aspect("equal", adjustable="datalim")   # a zoomed region fills the plot, with square cells
                ax.fill([0, Lx, Lx, 0], [0, 0, Ly, Ly], facecolor=COLORS["bg_surface"], edgecolor=COLORS["ink"], lw=1.2, zorder=0)
                if self.body is not None:
                    if (nx << level) * (ny << level) <= _MAX_CELLS:
                        window, (i0, i1, j0, j1) = geometry.solid_window(self.body, Lx, Ly, nx << level, ny << level)
                        solid = f"{int(window.sum()):,}"
                        if self.chk_solid.isChecked() and window.size:
                            dx, dy = Lx / (nx << level), Ly / (ny << level)
                            ax.imshow(np.ma.masked_where(~window, window), origin="lower", extent=(i0 * dx, i1 * dx, j0 * dy, j1 * dy),
                                      cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest", zorder=1, aspect="equal")
                    else:
                        solid = "too many cells to count at this level"
                    o = np.vstack([self.body.outline, self.body.outline[:1]])
                    ax.plot(o[:, 0], o[:, 1], color=COLORS["primary"], lw=1.6, zorder=3)
                if view is None:
                    pad = 0.02 * max(Lx, Ly)
                    ax.set_xlim(-pad, Lx + pad)
                    ax.set_ylim(-pad, Ly + pad)
                else:
                    ax.set_xlim(*view[0])
                    ax.set_ylim(*view[1])
                ax.set_autoscale_on(False)
            finally:
                self._busy = False
            self._describe(level, solid)
            self._draw_grid()
            self.toolbar.update()
            self.toolbar.push_current()   # the toolbar's Home returns to this view

        def _draw_grid(self) -> None:
            if self._busy or self.p is None:
                return
            p, ax = self.p, self.ax
            self._grid_view = self._view_key()
            if self._grid is not None:
                self._grid.remove()
                self._grid = None
            self._draw_bc()
            if self.chk_grid.isChecked():
                Lx, Ly = float(p["Lx"]), float(p["Ly"])
                level = max(self.cb_level.currentIndex(), 0)
                dx, dy = Lx / (int(p["nx"]) << level), Ly / (int(p["ny"]) << level)
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                xa, xb, ya, yb = max(x0, 0.0), min(x1, Lx), max(y0, 0.0), min(y1, Ly)
                pixels_per_unit = ax.bbox.width / max(x1 - x0, 1e-300)
                if xb > xa and yb > ya and dx * pixels_per_unit >= 4.0:   # only when a cell is a few pixels wide
                    xs = np.arange(int(np.ceil(xa / dx)), int(np.floor(xb / dx)) + 1) * dx
                    ys = np.arange(int(np.ceil(ya / dy)), int(np.floor(yb / dy)) + 1) * dy
                    segs = [((x, ya), (x, yb)) for x in xs] + [((xa, y), (xb, y)) for y in ys]
                    self._grid = ax.add_collection(LineCollection(segs, colors=COLORS["border_strong"], linewidths=0.4, zorder=2))
            self.canvas.draw_idle()

        def _draw_bc(self) -> None:
            """The boundary condition of each side, written beside the domain edge while that edge is in view."""
            for t in self._bc_texts:
                t.remove()
            self._bc_texts = []
            p, ax = self.p, self.ax
            Lx, Ly = float(p["Lx"]), float(p["Ly"])
            (xa, xb), (ya, yb) = ax.get_xlim(), ax.get_ylim()
            for side, (x, y, rot, ha, va) in {"left": (0.0, Ly / 2, 90, "right", "center"), "right": (Lx, Ly / 2, 90, "left", "center"),
                                              "bottom": (Lx / 2, 0.0, 0, "center", "top"), "top": (Lx / 2, Ly, 0, "center", "bottom")}.items():
                if xa <= x <= xb and ya <= y <= yb:
                    self._bc_texts.append(ax.text(x, y, p.get(f"bc_{side}", ""), rotation=rot, ha=ha, va=va, fontsize=9,
                                                  color=COLORS["ink_soft"], zorder=4))

        # ---- the view: wheel zooms about the pointer, a drag pans (when the toolbar has no tool on), a double click shows the domain
        def _on_scroll(self, event) -> None:
            if event.inaxes is not self.ax or event.xdata is None:
                return
            k = 1 / 1.3 if event.step > 0 else 1.3
            (x0, x1), (y0, y1) = self.ax.get_xlim(), self.ax.get_ylim()
            self.ax.set_xlim(event.xdata - (event.xdata - x0) * k, event.xdata + (x1 - event.xdata) * k)
            self.ax.set_ylim(event.ydata - (event.ydata - y0) * k, event.ydata + (y1 - event.ydata) * k)
            self.canvas.draw_idle()

        def _on_press(self, event) -> None:
            if event.inaxes is not self.ax or self.toolbar.mode:
                return
            if event.dblclick:
                self._show_domain()
            elif event.button == 1:
                self._drag = (event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim())

        def _on_motion(self, event) -> None:
            if self._drag is None or event.x is None:
                return
            x, y, (x0, x1), (y0, y1) = self._drag
            bb = self.ax.bbox
            dx = (event.x - x) * (x1 - x0) / bb.width
            dy = (event.y - y) * (y1 - y0) / bb.height
            self.ax.set_xlim(x0 - dx, x1 - dx)
            self.ax.set_ylim(y0 - dy, y1 - dy)
            self.canvas.draw_idle()

        def _on_release(self, event) -> None:
            self._drag = None

        def _show_domain(self) -> None:
            if self.p is None:
                return
            Lx, Ly = float(self.p["Lx"]), float(self.p["Ly"])
            pad = 0.02 * max(Lx, Ly)
            self.ax.set_xlim(-pad, Lx + pad)
            self.ax.set_ylim(-pad, Ly + pad)
            self.canvas.draw_idle()

        def _view_changed(self, event=None) -> None:
            if not self._busy and self._grid_view != self._view_key():
                self._timer.start()

        def _view_key(self):
            return (self.ax.get_xlim(), self.ax.get_ylim(), self.ax.bbox.bounds)

        def _zoom_to_body(self) -> None:
            if self.body is None:
                return
            xs, ys = self.body.outline[:, 0], self.body.outline[:, 1]
            m = 0.25 * max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-9)
            self.ax.set_xlim(xs.min() - m, xs.max() + m)
            self.ax.set_ylim(ys.min() - m, ys.max() + m)
            self.canvas.draw_idle()

        def _describe(self, level: int, solid: str) -> None:
            p = self.p
            Lx, Ly, nx, ny = float(p["Lx"]), float(p["Ly"]), int(p["nx"]), int(p["ny"])
            nz = int(p.get("nz", "1"))
            dx, dy = Lx / nx, Ly / ny
            lines = [f"Domain {Lx:g} x {Ly:g}" + (f" x {float(p['Lz']):g}" if nz > 1 else "") + f", base mesh {nx} x {ny}"
                     + (f" x {nz}" if nz > 1 else "") + f", cells {dx:.5g} x {dy:.5g}."]
            if nz > 1:
                lines.append("3-D: the picture is the x-y plane (an airfoil or cylinder is extruded along z; a sphere is cut through its centre).")
            if self.body is None:
                lines.append("The body cannot be drawn." if flag(p, "use_ibm") else "No immersed body.")
            else:
                fine = dx / (1 << level)
                size = 2.0 * self.body.radius if self.body.kind == "circle" else self.body.chord
                lines.append(f"Body: {self.body.description}.")
                lines.append(f"At mesh level {level} (cells {fine:.5g}): {size / fine:.1f} cells across the "
                             f"{'diameter' if self.body.kind == 'circle' else 'chord'}, {solid} solid cells (cell centre inside the body).")
            if flag(p, "use_amr"):
                L = int(p.get("amr_max_level", "0"))
                tail = (f"; with amr_refine_ibm the leaves that touch the body and the cells the solid reset uses stay at level {L}."
                        if flag(p, "amr_refine_ibm", "true") and self.body is not None else ".")
                lines.append(f"AMR: {L} levels above the base mesh, finest cell {dx / (1 << L):.5g}{tail}")
            self.info.setText("   ".join(lines))

    return GeometryPanel()


_GEOMETRY_KEYS = ("Lx", "Ly", "Lz", "nx", "ny", "nz", "use_ibm", "ibm_shape", "ibm_radius", "ibm_cx", "ibm_cy", "ibm_cz",
                  "bc_left", "bc_right", "bc_bottom", "bc_top", "bc_front", "bc_back", "use_amr", "amr_indicator", "amr_max_level",
                  "amr_init_levels", "amr_criterion", "amr_refine_threshold", "amr_derefine_threshold", "amr_diffusion", "amr_refine_every")

MODULE = ModuleSpec("geometry", "Geometry", _make, order=10, standalone=True, requires=("PyQt6", "matplotlib", "numpy"),
                    outputs=(Port("geometry", "geometry", label="geometry"),),
                    description="The domain, the immersed body, the Cartesian mesh and the solid cells it makes, as set up.")
