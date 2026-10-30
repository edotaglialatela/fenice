"""Geometry module: the domain, the immersed body, the Cartesian mesh and the solid cells it makes, as set up (before any run)."""
from __future__ import annotations

from gui.modules import ModuleSpec

_MAX_CELLS = 40_000_000   # the largest level whose solid cells are computed


def _make(project):
    import numpy as np
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

    import fenice
    from fenice import geometry
    from gui.solver import case_defaults
    from gui.theme import COLORS

    def flag(p: dict, key: str, default: str = "false") -> bool:
        return str(p.get(key, default)).strip().lower() in ("1", "true", "yes", "on")

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
            row = QHBoxLayout()
            for w in (self.btn_refresh, QLabel("Mesh level"), self.cb_level, self.chk_grid, self.chk_solid, self.btn_zoom):
                row.addWidget(w)
            row.addStretch(1)
            lay = QVBoxLayout(self)
            lay.addLayout(row)
            lay.addWidget(self.toolbar)
            lay.addWidget(self.canvas, 1)
            lay.addWidget(self.info)
            self.p: dict | None = None
            self.body = None
            self._grid = None
            self._bc_texts: list = []
            self._busy = False
            self._timer = QTimer(self)   # redraw the grid lines once a zoom or a resize has settled
            self._timer.setSingleShot(True)
            self._timer.setInterval(90)
            self._timer.timeout.connect(self._draw_grid)
            self.btn_refresh.clicked.connect(self.refresh)
            self.btn_zoom.clicked.connect(self._zoom_to_body)
            self.cb_level.currentIndexChanged.connect(lambda _: self._draw())
            self.chk_grid.toggled.connect(lambda _: self._draw())
            self.chk_solid.toggled.connect(lambda _: self._draw())
            self.ax.callbacks.connect("xlim_changed", self._view_changed)
            self.ax.callbacks.connect("ylim_changed", self._view_changed)
            self.canvas.mpl_connect("resize_event", lambda e: self._timer.start())

        def showEvent(self, event):
            super().showEvent(event)
            self.refresh()   # the page shows what Setup holds when it comes up

        # ---- the parameters
        def refresh(self) -> None:
            cur = project.current_params()
            if cur is None:
                self.info.setText("The setup has an input that is not valid yet: correct it in Setup, then refresh.")
                return
            try:
                p = dict(case_defaults(cur["test_case"])) if "test_case" in cur else {}
            except Exception:   # noqa: BLE001 - the solver is not built or the case is unknown: the live values alone will do
                p = {}
            p.update({k: str(v) for k, v in cur.items()})
            self.p = p
            try:
                self.body = geometry.body_from_params(p, fenice.ROOT)
            except (ValueError, OSError) as exc:
                self.body = None
                self.info.setText(f"The body cannot be drawn: {exc}")
            levels = int(p.get("amr_max_level", "0")) if flag(p, "use_amr") else 0
            keep = self.cb_level.currentIndex()
            nx, ny = int(p["nx"]), int(p["ny"])
            self.cb_level.blockSignals(True)
            self.cb_level.clear()
            self.cb_level.addItems([f"{l}: {nx << l} x {ny << l}" for l in range(levels + 1)])
            self.cb_level.setCurrentIndex(min(max(keep, 0), levels))
            self.cb_level.blockSignals(False)
            self._draw(reset_view=True)

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
                ax.set_aspect("equal")
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

        def _draw_grid(self) -> None:
            if self._busy or self.p is None:
                return
            p, ax = self.p, self.ax
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

        def _view_changed(self, ax) -> None:
            if not self._busy:
                self._timer.start()

        def _zoom_to_body(self) -> None:
            if self.body is None:
                return
            xs, ys = self.body.outline[:, 0], self.body.outline[:, 1]
            m = 0.25 * max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-9)
            self.ax.set_xlim(xs.min() - m, xs.max() + m)
            self.ax.set_ylim(ys.min() - m, ys.max() + m)

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


MODULE = ModuleSpec("geometry", "Geometry", _make, order=15, requires=("PyQt6", "matplotlib", "numpy"),
                    description="The domain, the immersed body, the Cartesian mesh and the solid cells it makes, as set up.")
