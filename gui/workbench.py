"""The Fenice workbench: an empty space on which modules are dropped and linked.

    python gui/workbench.py

The palette on the left lists the modules (Geometry, Solver, Monitor, Postprocessing, ...); drag one onto the canvas, or double-click it.
Each module is a node with typed ports: drag from an output port to an input port of the same type to link them. One Geometry node
can feed several Solvers; each Solver can have one Geometry input. A Monitor linked to a solver shows the run live, a Postprocessing node linked to it shows the
results (also after the run). Double-click a node to open its editor. Monitor tabs have separate Plots and Console views. A node that
is not linked to a solver has nothing to show; it takes the project of the solver it is linked to. ``gui/run.py`` (the dialog, then the
solver in the terminal) is unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PyQt6.QtCore import QByteArray, QPointF, QRectF, QSize, Qt, QTimer, QUrl  # noqa: E402
from PyQt6.QtGui import QDesktopServices, QIcon, QTransform  # noqa: E402
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
                             QPushButton, QSizePolicy, QSplitter, QStackedLayout, QStackedWidget, QTabWidget, QVBoxLayout,
                             QWidget)  # noqa: E402

from gui.canvas import NODE_W, GraphScene, GraphView, LinkItem, NodeItem, Palette  # noqa: E402
from gui.graph import Graph, Link  # noqa: E402
from gui.modules import ModuleSpec, discover  # noqa: E402
from gui.project import IDLE, Project  # noqa: E402
from gui.solver import SolverNotFound, available_cases  # noqa: E402
from gui.theme import apply_theme  # noqa: E402
from gui.workbench_state import new_state, read_state, write_state  # noqa: E402


def _placeholder(text: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    label = QLabel(text)
    label.setObjectName("note")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    lay.addWidget(label)
    return w


class _SubWindow(QWidget):
    """The window of a node's editor: closing it only hides it; the editor can be swapped when the node is linked to another solver."""

    def __init__(self, title: str, parent: QWidget) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(title)
        self.resize(1040, 760)
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(4, 4, 4, 4)

    def set_editor(self, editor: QWidget) -> None:
        self._lay.addWidget(editor)

    def take_editor(self, editor: QWidget) -> None:
        self._lay.removeWidget(editor)


class Workbench(QMainWindow):
    def __init__(self, default_graph: bool = False):
        super().__init__()
        self.setWindowTitle("Fenice")
        screen = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() is not None else None
        self.resize(min(1600, screen.width() - 40) if screen else 1600, min(980, screen.height() - 40) if screen else 980)
        self.setMinimumSize(1120, 720)
        icon_path = os.path.join(ROOT_DIR, "gui", "logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.graph = Graph()
        self.specs = {s.name: s for s in discover()}
        self.items: dict[int, NodeItem] = {}
        self.links: dict[Link, object] = {}        # model link -> its item
        self.projects: dict[int, Project] = {}     # solver node id -> its project
        self.editors: dict[int, QWidget] = {}      # node id -> editor (the real one, or a placeholder while unlinked)
        self._bound: dict[int, int | tuple[int, ...] | None] = {}    # node id -> solver node(s) its editor is bound to
        self._placeholders: set[int] = set()
        self.monitor_pages: dict[int, QTabWidget] = {}

        self.palette = Palette()
        self.palette.setObjectName("modulePalette")
        self.palette.setFixedWidth(232)
        self.palette.setDragEnabled(True)
        self.palette.setSpacing(3)
        self.palette.setWordWrap(True)
        for spec in sorted(self.specs.values(), key=lambda s: (s.order, s.name)):
            summary = spec.description.split(".", 1)[0].strip()
            item = QListWidgetItem(f"{spec.title}\n{summary}")
            item.setSizeHint(QSize(0, 58))
            item.setData(Qt.ItemDataRole.UserRole, spec.name)
            missing = spec.missing() if spec.factory is not None else ["see details"]
            item.setToolTip(spec.description + (f"\nUnavailable: {', '.join(missing)}" if missing else ""))
            if missing:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.palette.addItem(item)
        self.palette.itemDoubleClicked.connect(self._palette_add)

        self.scene = GraphScene(self)
        self.scene.can_link = self._can_link
        self.scene.on_link = self._link
        self.scene.on_drop = lambda name, pos: self.add_node(name, pos)
        self.scene.on_delete = self._delete
        self.scene.on_open = lambda item: self.open_editor(item.node_id)
        self.view = GraphView(self.scene)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._context_menu)
        self.windows: dict[int, QWidget] = {}   # node id -> its editor window (every editor but the monitors')
        self.tabs = QTabWidget()   # one page per linked monitor
        self.tabs.setDocumentMode(True)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)   # an editor's own minimum height must not squeeze the canvas
        self.empty = QLabel("Run activity appears here when a Monitor is linked to a Solver.")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setObjectName("note")
        self.bottom = QStackedWidget()
        self.bottom.addWidget(self.empty)
        self.bottom.addWidget(self.tabs)
        self.bottom.hide()

        palette_panel = QWidget()
        palette_panel.setObjectName("moduleSidebar")
        palette_panel.setFixedWidth(self.palette.width() + 32)
        palette_layout = QVBoxLayout(palette_panel)
        palette_layout.setContentsMargins(16, 20, 16, 16)
        palette_layout.setSpacing(10)
        brand = QLabel("Fenice")
        brand.setObjectName("sidebarTitle")
        brand_note = QLabel("Simulation workbench")
        brand_note.setObjectName("sidebarNote")
        palette_title = QLabel("Modules")
        palette_title.setObjectName("sidebarSection")
        palette_hint = QLabel("Select a module and add it, or drag it onto the canvas.")
        palette_hint.setObjectName("sidebarNote")
        palette_hint.setWordWrap(True)
        palette_layout.addWidget(brand)
        palette_layout.addWidget(brand_note)
        palette_layout.addSpacing(14)
        palette_layout.addWidget(palette_title)
        palette_layout.addWidget(palette_hint)
        palette_layout.addWidget(self.palette, 1)
        self.add_button = QPushButton("Add selected module")
        self.add_button.setObjectName("sidebarPrimary")
        self.add_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.add_button.setEnabled(False)
        self.add_button.clicked.connect(lambda: self._palette_add(self.palette.currentItem()))
        self.palette.currentItemChanged.connect(lambda current, _previous: self.add_button.setEnabled(current is not None and bool(current.flags() & Qt.ItemFlag.ItemIsEnabled)))
        self.palette.setCurrentRow(0)
        palette_layout.addWidget(self.add_button)
        self.state_label = QLabel("Ready · no active runs")
        self.state_label.setObjectName("sidebarNote")
        self.state_label.setWordWrap(True)
        palette_layout.addWidget(self.state_label)
        self.github_btn = QPushButton("GitHub")
        self.github_btn.setObjectName("sidebarButton")
        self.github_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.github_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/edotaglialatela/fenice")))
        palette_layout.addWidget(self.github_btn)
        self.docs_btn = QPushButton("Documentation")
        self.docs_btn.setObjectName("sidebarButton")
        self.docs_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.docs_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://edotaglialatela.github.io/fenice/index.html#quick-start")))
        palette_layout.addWidget(self.docs_btn)

        canvas_panel = QWidget()
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(18, 16, 18, 16)
        canvas_layout.setSpacing(12)
        canvas_header = QWidget()
        header_layout = QHBoxLayout(canvas_header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(10)
        header_text = QVBoxLayout()
        header_text.setContentsMargins(0, 0, 0, 0)
        header_text.setSpacing(3)
        workflow_title = QLabel("Workflow")
        workflow_title.setObjectName("canvasTitle")
        workflow_hint = QLabel("Build a graph from geometry, solver runs, monitoring, and results.")
        workflow_hint.setObjectName("canvasSubtitle")
        workflow_hint.setWordWrap(True)
        header_text.addWidget(workflow_title)
        header_text.addWidget(workflow_hint)
        header_layout.addLayout(header_text, 1)
        self.edit_button = QPushButton("Edit selected")
        self.edit_button.setObjectName("secondaryButton")
        self.edit_button.setEnabled(False)
        self.edit_button.setToolTip("Open the selected module's editor")
        self.edit_button.clicked.connect(self._edit_selected)
        header_layout.addWidget(self.edit_button)
        self.save_state_button = QPushButton("Save state...")
        self.save_state_button.setObjectName("secondaryButton")
        self.save_state_button.setToolTip("Save the workflow, setup and analysis views to a state file")
        self.save_state_button.clicked.connect(self._save_state_dialog)
        header_layout.addWidget(self.save_state_button)
        self.restore_state_button = QPushButton("Restore state...")
        self.restore_state_button.setObjectName("secondaryButton")
        self.restore_state_button.setToolTip("Replace this workbench with a saved workflow state")
        self.restore_state_button.clicked.connect(self._restore_state_dialog)
        header_layout.addWidget(self.restore_state_button)
        fit_button = QPushButton("Fit view")
        fit_button.setObjectName("secondaryButton")
        fit_button.setToolTip("Show every module in the workflow")
        fit_button.clicked.connect(self.fit_graph)
        header_layout.addWidget(fit_button)
        canvas_layout.addWidget(canvas_header)

        self.canvas_stage = QWidget()
        stage_layout = QStackedLayout(self.canvas_stage)
        stage_layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self.stage_layout = stage_layout
        stage_layout.addWidget(self.view)
        self.empty_overlay = QWidget()
        self.empty_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.empty_overlay.setObjectName("emptyCanvasState")
        empty_layout = QVBoxLayout(self.empty_overlay)
        empty_layout.setContentsMargins(40, 40, 40, 40)
        empty_layout.addStretch(1)
        empty_title = QLabel("Start with a Geometry module")
        empty_title.setObjectName("emptyStateTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_body = QLabel("Add Geometry and one or more Solvers, then connect matching ports.\nOne Geometry can feed several Solvers.")
        empty_body.setObjectName("emptyStateBody")
        empty_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_body.setWordWrap(True)
        empty_route = QLabel("Geometry  →  Solver  →  Monitor / Postprocessing / Analysis Tools")
        empty_route.setObjectName("subsectionTitle")
        empty_route.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_title)
        empty_layout.addSpacing(8)
        empty_layout.addWidget(empty_body)
        empty_layout.addSpacing(12)
        empty_layout.addWidget(empty_route)
        empty_layout.addStretch(1)
        stage_layout.addWidget(self.empty_overlay)
        canvas_layout.addWidget(self.canvas_stage, 1)

        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.setChildrenCollapsible(True)
        self.split.addWidget(canvas_panel)
        self.split.addWidget(self.bottom)
        self.split.setStretchFactor(0, 1)
        self.split.setStretchFactor(1, 0)
        self.split.setSizes([760, 0])

        outer = QWidget()
        outer.setObjectName("mainShell")
        workspace_layout = QHBoxLayout(outer)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)
        workspace_layout.addWidget(palette_panel)
        workspace_layout.addWidget(self.split, 1)
        self.setCentralWidget(outer)
        self.scene.selectionChanged.connect(self._selection_changed)
        self._sync_empty_state()

        self._timer = QTimer(self)   # the status line of every node
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start()
        if default_graph:
            geo = self.add_node("geometry", QPointF(-330, -60))
            sol = self.add_node("solver", QPointF(-40, -90))
            self.connect_nodes(geo.node_id, "geometry", sol.node_id, "geometry")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_laid_out", False):
            self._laid_out = True
            QTimer.singleShot(0, self._layout_once)   # the sizes are known once the window has been laid out

    def _layout_once(self) -> None:
        h = self.split.height()
        if self.tabs.count():
            bottom = min(max(int(h * 0.34), 220), h // 2)
            self.split.setSizes([h - bottom, bottom])
        else:
            self.split.setSizes([h, 0])
        self.fit_graph()

    def _sync_empty_state(self) -> None:
        is_empty = not self.items
        self.empty_overlay.setVisible(is_empty)
        self.stage_layout.setCurrentWidget(self.empty_overlay if is_empty else self.view)

    def _selection_changed(self) -> None:
        selected = [item.node_id for item in self.scene.selectedItems() if isinstance(item, NodeItem)]
        self.edit_button.setEnabled(bool(selected))
        if not selected:
            for link_item in self.links.values():
                link_item.set_focus(None)
            return
        focused_links = set()
        for node_id in selected:
            # Follow dependencies toward their sources without crossing from a
            # shared source into its sibling branches.
            visited = set()
            pending = [node_id]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                for link in self.graph.links:
                    if link.dst == current:
                        focused_links.add(link)
                        pending.append(link.src)

            # Also show the work and results that depend on this node.
            visited = set()
            pending = [node_id]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                for link in self.graph.links:
                    if link.src == current:
                        focused_links.add(link)
                        pending.append(link.dst)
        for link, link_item in self.links.items():
            link_item.set_focus(link in focused_links)

    def _edit_selected(self) -> None:
        selected = next((item for item in self.scene.selectedItems() if isinstance(item, NodeItem)), None)
        if selected is not None:
            self.open_editor(selected.node_id)

    def fit_graph(self) -> None:
        """Show every node, at most at normal size."""
        if not self.items:
            return
        self.view.resetTransform()
        rect = self.scene.itemsBoundingRect().adjusted(-30, -30, 30, 30)
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        if self.view.transform().m11() > 1.0:
            self.view.resetTransform()
            self.view.centerOn(rect.center())

    # ---- workbench state files
    def _capture_state(self) -> dict:
        node_records = []
        monitor_views = {}
        subwindows = {}
        for node_id, item in self.items.items():
            node = self.graph.nodes[node_id]
            editor = self.editors.get(node_id)
            record = {"id": node_id, "module": node.spec.name, "title": node.title,
                      "position": [float(item.pos().x()), float(item.pos().y())], "editor": {}}
            if node.spec.owns_project:
                project = self.projects[node_id]
                params = project.current_params()
                if params is None:
                    params = dict(project.params)
                record["project"] = {
                    "params": params or {}, "monitor_every": project.monitor_every,
                    "results_dir": str(project.results_dir) if project.results_dir is not None else None,
                    "log_history": list(project.log_history), "progress_history": project.progress_history,
                    "t_now": project.t_now, "t_end": project.t_end,
                }
                if editor is not None:
                    record["editor"] = {"setup_tab": editor.tabs.currentIndex(),
                                         "config_tab": editor.config_sections.currentIndex(),
                                         "setup_controls": editor.save_ui_state()}
            elif node.spec.name == "geometry" and editor is not None and hasattr(editor, "save_state"):
                record["editor"] = editor.save_state()
            elif node.spec.name in ("results", "analysis") and editor is not None and hasattr(editor, "save_state"):
                record["editor"] = editor.save_state()
            elif node.spec.name == "monitor" and editor is not None:
                if hasattr(editor, "cb_quantity"):
                    record["editor"] = {"quantity": editor.cb_quantity.currentText()}
                page = self.monitor_pages.get(node_id)
                if page is not None:
                    monitor_views[str(node_id)] = page.currentIndex()

            if node.spec.name != "monitor" and node_id in self.windows:
                win = self.windows[node_id]
                subwindows[str(node_id)] = {"geometry": bytes(win.saveGeometry().toBase64()).decode("ascii"),
                                            "visible": win.isVisible()}
            node_records.append(record)

        transform = self.view.transform()
        center = self.view.mapToScene(self.view.viewport().rect().center())
        matrix = [transform.m11(), transform.m12(), transform.m13(), transform.m21(), transform.m22(), transform.m23(),
                  transform.m31(), transform.m32(), transform.m33()]
        workspace = {
            "geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"), "split_sizes": self.split.sizes(),
            "view_transform": matrix, "view_center": [center.x(), center.y()],
            "selected_nodes": [item.node_id for item in self.scene.selectedItems() if isinstance(item, NodeItem)],
            "monitor_tab": self.tabs.currentIndex(), "monitor_views": monitor_views, "subwindows": subwindows,
            "palette_module": self.palette.currentItem().data(Qt.ItemDataRole.UserRole) if self.palette.currentItem() else None,
        }
        links = [{"src": link.src, "src_port": link.src_port, "dst": link.dst, "dst_port": link.dst_port} for link in self.graph.links]
        return new_state(node_records, links, workspace)

    def _save_state_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save workbench state", "fenice-workbench.fenice.json",
                                              "Fenice workbench state (*.fenice.json);;JSON files (*.json)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".fenice.json"
        try:
            saved = write_state(path, self._capture_state())
            self.statusBar().showMessage(f"Workbench state saved to {saved}", 7000)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Could not save workbench state", f"{type(exc).__name__}: {exc}")

    def _restore_state_dialog(self) -> None:
        if any(project.running for project in self.projects.values()):
            QMessageBox.information(self, "Run in progress", "Stop all solver runs before restoring a workbench state.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Restore workbench state", "",
                                              "Fenice workbench state (*.fenice.json *.json);;JSON files (*.json)")
        if not path:
            return
        try:
            state = read_state(path, self.specs)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            QMessageBox.critical(self, "Could not restore workbench state", f"{type(exc).__name__}: {exc}")
            return
        if self.graph.nodes:
            answer = QMessageBox.question(self, "Replace current workbench?",
                                          "Restoring this file will replace the current workflow and its editor state.",
                                          QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
                                          QMessageBox.StandardButton.Cancel)
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self._apply_state(state)
            self.statusBar().showMessage(f"Workbench state restored from {path}", 7000)
        except Exception as exc:  # noqa: BLE001 - report state-file or editor errors in the shell
            QMessageBox.critical(self, "Could not restore workbench state", f"{type(exc).__name__}: {exc}")

    def _apply_state(self, state: dict) -> None:
        self._delete(list(self.items.values()), [])
        id_map: dict[int, int] = {}
        records = {record["id"]: record for record in state["nodes"]}
        for record in state["nodes"]:
            item = self.add_node(record["module"], QPointF(*record["position"]))
            id_map[record["id"]] = item.node_id
            node = self.graph.nodes[item.node_id]
            node.title = record["title"]
            item.title = record["title"]
            item.update()
            if item.node_id in self.windows:
                self.windows[item.node_id].setWindowTitle(record["title"])
            if item.node_id in self.monitor_pages:
                self.tabs.setTabText(self.tabs.indexOf(self.monitor_pages[item.node_id]), record["title"])

            if node.spec.owns_project:
                project = self.projects[item.node_id]
                saved_project = record.get("project", {})
                params = saved_project.get("params", {})
                editor = self.editors[item.node_id]
                if isinstance(params, dict) and params:
                    editor.restore_params(params)
                    project.set_params(params)
                setup_state = record.get("editor", {})
                editor.restore_ui_state(setup_state.get("setup_controls", {}))
                project.monitor_every = int(saved_project.get("monitor_every", project.monitor_every))
                results_dir = saved_project.get("results_dir")
                project.results_dir = Path(results_dir) if results_dir else None
                project.log_history.clear()
                project.log_history.extend(str(line) for line in saved_project.get("log_history", []))
                project.progress_history = [tuple(row) for row in saved_project.get("progress_history", [])]
                project.t_now = float(saved_project.get("t_now", 0.0))
                project.t_end = float(saved_project.get("t_end", 0.0))
                editor.tabs.setCurrentIndex(max(0, min(int(setup_state.get("setup_tab", 0)), editor.tabs.count() - 1)))
                editor.config_sections.setCurrentIndex(max(0, min(int(setup_state.get("config_tab", 0)),
                                                                    editor.config_sections.count() - 1)))
            elif node.spec.name == "geometry":
                editor = self.editors[item.node_id]
                if hasattr(editor, "restore_state"):
                    editor.restore_state(record.get("editor", {}))

        for link in state["links"]:
            self.connect_nodes(id_map[link["src"]], link["src_port"], id_map[link["dst"]], link["dst_port"])

        for old_id, record in records.items():
            node_id = id_map[old_id]
            node = self.graph.nodes[node_id]
            editor = self.editors.get(node_id)
            saved_editor = record.get("editor", {})
            if node.spec.name == "geometry" and editor is not None and hasattr(editor, "restore_state"):
                editor.restore_state(saved_editor)  # linking can adopt a solver's initial view; apply the saved view last
            elif node.spec.name in ("results", "analysis") and editor is not None and hasattr(editor, "restore_state"):
                editor.restore_state(saved_editor)
            elif node.spec.name == "monitor" and editor is not None:
                quantity = str(saved_editor.get("quantity", ""))
                if hasattr(editor, "cb_quantity") and editor.cb_quantity.findText(quantity) >= 0:
                    editor.cb_quantity.setCurrentText(quantity)
                page = self.monitor_pages.get(node_id)
                if page is not None:
                    page.setCurrentIndex(max(0, min(int(state["workspace"].get("monitor_views", {}).get(str(old_id), 0)),
                                                    page.count() - 1)))

        self._sync_empty_state()
        self._sync_bottom()
        self._refresh_status()
        self._selection_changed()
        QTimer.singleShot(0, lambda: self._restore_workspace(state["workspace"], id_map))

    def _restore_workspace(self, workspace: dict, id_map: dict[int, int]) -> None:
        geometry = workspace.get("geometry")
        if geometry:
            self.restoreGeometry(QByteArray.fromBase64(str(geometry).encode("ascii")))
        sizes = workspace.get("split_sizes")
        if isinstance(sizes, list) and len(sizes) == 2:
            self.split.setSizes([max(0, int(size)) for size in sizes])
        matrix = workspace.get("view_transform")
        if isinstance(matrix, list) and len(matrix) == 9:
            self.view.setTransform(QTransform(*map(float, matrix)))
        center = workspace.get("view_center")
        if isinstance(center, list) and len(center) == 2:
            self.view.centerOn(float(center[0]), float(center[1]))
        monitor_index = int(workspace.get("monitor_tab", 0))
        if self.tabs.count():
            self.tabs.setCurrentIndex(max(0, min(monitor_index, self.tabs.count() - 1)))
        selected = {id_map[old] for old in workspace.get("selected_nodes", []) if old in id_map}
        for item in self.items.values():
            item.setSelected(item.node_id in selected)
        palette_module = workspace.get("palette_module")
        if palette_module:
            for index in range(self.palette.count()):
                entry = self.palette.item(index)
                if entry.data(Qt.ItemDataRole.UserRole) == palette_module:
                    self.palette.setCurrentItem(entry)
                    break
        for old_id, view_state in workspace.get("subwindows", {}).items():
            if not str(old_id).isdigit() or int(old_id) not in id_map:
                continue
            win = self.windows.get(id_map[int(old_id)])
            if win is None:
                continue
            encoded = view_state.get("geometry")
            if encoded:
                win.restoreGeometry(QByteArray.fromBase64(str(encoded).encode("ascii")))
            if view_state.get("visible"):
                win.show()
                win.raise_()

    # ---- nodes
    def add_node(self, name: str, pos: QPointF | None = None) -> NodeItem:
        spec = self.specs[name]
        node = self.graph.add(spec)
        item = NodeItem(node.id, spec, node.title)
        item.setPos(pos if pos is not None else QPointF(0, 0))
        self.scene.addItem(item)
        self.items[node.id] = item
        self._sync_empty_state()
        if spec.owns_project:
            project = Project(self)
            project.run_gate = lambda params, nid=node.id: self._gate(nid, params)
            self.projects[node.id] = project
            editor = spec.factory(project)
            self.editors[node.id] = editor
            self._bound[node.id] = node.id
            self._place(node.id)
            if spec.face is not None:
                item.set_face(spec.face(project, editor))
            project.started.connect(lambda nid=node.id: self._started(nid))
            project.finished.connect(lambda *_: self._refresh_status())
        elif spec.standalone:
            self.editors[node.id] = spec.factory(None)
            self._bound[node.id] = -1
            self._place(node.id)
            self._bind(node.id)
        else:
            self._bind(node.id)
        self._refresh_status()
        return item

    def _palette_add(self, entry: QListWidgetItem) -> None:
        if entry is None or not (entry.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        center = self.view.mapToScene(self.view.viewport().rect().center())
        origin = center - QPointF(NODE_W / 2, 120)
        slots = [(0, 0), (1, 0), (0, 1), (-1, 0), (1, 1), (-1, 1), (0, -1), (1, -1), (-1, -1)]
        for ring in range(2, 10):
            slots.extend((x, -ring) for x in range(-ring, ring + 1))
            slots.extend((ring, y) for y in range(-ring + 1, ring + 1))
            slots.extend((x, ring) for x in range(ring - 1, -ring - 1, -1))
            slots.extend((-ring, y) for y in range(ring - 1, -ring, -1))
        occupied = [item.sceneBoundingRect().adjusted(-24, -20, 24, 20) for item in self.items.values()]
        at = None
        for col, row in slots:
            candidate = origin + QPointF(col * (NODE_W + 72), row * 280)
            rect = QRectF(candidate.x(), candidate.y(), NODE_W, 240)
            if not any(rect.intersects(other) for other in occupied):
                at = candidate
                break
        if at is None:
            right = max((rect.right() for rect in occupied), default=origin.x() - 72)
            at = QPointF(right + 72, origin.y())
        self.add_node(entry.data(Qt.ItemDataRole.UserRole), at)

    def _context_menu(self, pos) -> None:
        menu = QMenu(self)
        at = self.view.mapToScene(pos)
        for spec in sorted(self.specs.values(), key=lambda s: (s.order, s.name)):
            act = menu.addAction(f"Add {spec.title}")
            act.setEnabled(not spec.missing() and spec.factory is not None)
            act.triggered.connect(lambda _=False, n=spec.name: self.add_node(n, at))
        item = self.scene.itemAt(at, self.view.transform())
        if isinstance(item, (NodeItem, LinkItem)):
            menu.addSeparator()
            menu.addAction("Delete").triggered.connect(
                lambda: self._delete([item] if isinstance(item, NodeItem) else [], [item] if isinstance(item, LinkItem) else []))
        menu.exec(self.view.viewport().mapToGlobal(pos))

    # ---- links
    def _ids(self, out, inp):
        return out.node.node_id, out.port.name, inp.node.node_id, inp.port.name

    def _can_link(self, out, inp) -> str | None:
        return self.graph.check(*self._ids(out, inp))

    def _link(self, out, inp) -> None:
        self.connect_nodes(*self._ids(out, inp))

    def connect_nodes(self, src: int, src_port: str, dst: int, dst_port: str) -> None:
        link = self.graph.connect(src, src_port, dst, dst_port)
        self.links[link] = self.scene.add_link(self.items[src].out_ports[src_port], self.items[dst].in_ports[dst_port])
        self._rebind()
        self._selection_changed()

    def _remove_link(self, link: Link) -> str | None:
        """None if removed, else why not."""
        project = self.projects.get(link.dst)
        if project is not None and link.dst_port == "geometry" and project.running:
            return "The solver is running: stop it before unlinking its geometry."
        self.graph.disconnect(link)
        self.scene.remove_link(self.links.pop(link))
        self._rebind()
        return None

    def _delete(self, nodes, link_items) -> None:
        refused = []
        for li in link_items:
            link = next((l for l, it in self.links.items() if it is li), None)
            if link is not None:
                reason = self._remove_link(link)
                if reason:
                    refused.append(reason)
        for item in nodes:
            if item.node_id in self.projects and self.projects[item.node_id].running:
                refused.append(f"{item.title}: the solver is running; stop it first.")
                continue
            if any(l.src == item.node_id and l.dst in self.projects and self.projects[l.dst].running and l.dst_port == "geometry"
                   for l in self.graph.links):
                refused.append(f"{item.title}: the solver it feeds is running; stop it first.")
                continue
            for link in [l for l in self.graph.links if item.node_id in (l.src, l.dst)]:
                self._remove_link(link)
            self._drop_editor(item.node_id)
            win = self.windows.pop(item.node_id, None)
            if win is not None:
                win.close()
                win.deleteLater()
            if item.node_id in self.projects:
                self.projects.pop(item.node_id).shutdown()
            self.graph.remove(item.node_id)
            self.items.pop(item.node_id)
            self.scene.removeItem(item)
        if refused:
            self.statusBar().showMessage(" ".join(refused), 8000)
        self._sync_empty_state()
        self._selection_changed()

    # ---- editors: bound to the project of the solver the node is linked to
    def _rebind(self) -> None:
        for nid, node in self.graph.nodes.items():
            if not node.spec.owns_project:
                self._bind(nid)
        self._refresh_status()

    def _bind(self, nid: int) -> None:
        node = self.graph.nodes[nid]
        if node.spec.standalone:
            solver_ids = tuple(n.id for n in self.graph.downstream(nid) if n.spec.owns_project)
            if self._bound.get(nid) == solver_ids and nid in self.editors:
                return
            projects = [self.projects[i] for i in solver_ids]
            self.editors[nid].attach(projects)
            self._bound[nid] = solver_ids
            return
        solver = self.graph.solver_of(nid)
        want = solver.id if solver is not None else None
        if nid in self._bound and self._bound[nid] == want and nid in self.editors:
            return
        self._drop_editor(nid)
        if want is None:
            editor = _placeholder(f"{node.title} is not linked to a solver.\nDrag from its port to a Solver node to link it.")
            self._placeholders.add(nid)
        else:
            try:
                editor = node.spec.factory(self.projects[want])
                self._placeholders.discard(nid)
            except Exception as exc:   # noqa: BLE001 - one module failing must not take the others down
                editor = _placeholder(f"{node.title} cannot be built: {type(exc).__name__}: {exc}")
                self._placeholders.add(nid)
        self.editors[nid] = editor
        self._bound[nid] = want
        self._place(nid)

    def _place(self, nid: int) -> None:
        """Put the editor of a node where it belongs: the monitors in the bottom half, every other editor in a subwindow of its node."""
        editor, node = self.editors[nid], self.graph.nodes[nid]
        if node.spec.name == "monitor":
            if nid not in self._placeholders:
                page = QTabWidget()
                page.setDocumentMode(True)
                page.addTab(editor, "Plots")
                page.addTab(editor.console, "Console")
                page.setTabToolTip(0, "Live plots and run controls")
                page.setTabToolTip(1, "Solver output for this monitor")
                self.monitor_pages[nid] = page
                self.tabs.addTab(page, node.title)
                self.tabs.setCurrentWidget(page)
            self._sync_bottom()
            return
        win = self.windows.get(nid)
        if win is None:
            win = self.windows[nid] = _SubWindow(node.title, self)
        win.set_editor(editor)

    def _sync_bottom(self) -> None:
        was_hidden = self.bottom.isHidden()
        has_monitors = bool(self.tabs.count())
        self.bottom.setCurrentWidget(self.tabs if self.tabs.count() else self.empty)
        self.bottom.setVisible(has_monitors)
        if has_monitors and was_hidden and self.isVisible() and self.split.height() > 0:
            h = self.split.height()
            bottom = min(max(int(h * 0.34), 220), h // 2)
            self.split.setSizes([h - bottom, bottom])

    def _drop_editor(self, nid: int) -> None:
        editor = self.editors.pop(nid, None)
        self._bound.pop(nid, None)
        if editor is None:
            return
        page = self.monitor_pages.pop(nid, None)
        i = self.tabs.indexOf(page) if page is not None else -1
        if i >= 0:
            self.tabs.removeTab(i)
        if page is not None:
            page.deleteLater()
        win = self.windows.get(nid)
        if win is not None:
            win.take_editor(editor)
        if hasattr(editor, "shutdown"):
            editor.shutdown()
        editor.setParent(None)
        editor.deleteLater()
        self._sync_bottom()

    def open_editor(self, nid: int) -> None:
        """A monitor: show its tab in the bottom half. Any other node: show its subwindow."""
        editor = self.editors[nid]
        node = self.graph.nodes[nid]
        if node.spec.name == "monitor":
            page = self.monitor_pages.get(nid)
            if page is not None and self.tabs.indexOf(page) >= 0:
                page.setCurrentIndex(0)
                self.tabs.setCurrentWidget(page)
            else:
                self.statusBar().showMessage(f"{node.title} is not linked to a solver: link its run port to watch a run.", 6000)
            return
        win = self.windows[nid]
        win.show()
        win.raise_()
        win.activateWindow()

    def editor_of(self, nid: int):
        return self.editors.get(nid)

    # ---- the run
    def _gate(self, solver_id: int, params: dict | None = None) -> str | None:
        if not any(l.dst == solver_id and l.dst_port == "geometry" for l in self.graph.links):
            return "Link a Geometry node to the solver first: the geometry is not part of the solver's setup."
        mine = params if params is not None else (self.projects[solver_id].current_params() or {})
        key = (mine.get("test_case"), str(mine.get("use_amr")), str(mine.get("output_root", "")))
        for other_id, other in self.projects.items():
            if other_id != solver_id and other.running and (other.params.get("test_case"), str(other.params.get("use_amr")),
                                                              str(other.params.get("output_root", ""))) == key:
                return (f"{self.graph.nodes[other_id].title} is running the same case into the same results folder: "
                        "stop it, or run another case.")
        return None

    def _started(self, solver_id: int) -> None:
        for node in self.graph.downstream(solver_id):
            if node.spec.name == "monitor":
                self.open_editor(node.id)

    def _refresh_status(self) -> None:
        for nid, item in self.items.items():
            node = self.graph.nodes[nid]
            name = node.spec.name
            project = self.projects.get(self.graph.solver_of(nid).id) if self.graph.solver_of(nid) else None
            if name == "solver":
                if project.state != IDLE:
                    pct = f"  {min(100 * project.t_now / project.t_end, 100):.0f} %" if project.t_end > 0 and project.t_now > 0 else ""
                    item.set_status(project.state + pct)
                else:
                    ok = self._gate(nid) is None
                    item.set_status("idle" if ok else "idle – link a geometry")
            elif project is None:
                item.set_status("not linked")
            elif name == "geometry":
                ed = project.editor
                v = ed.geometry_values() if ed is not None else {}
                feeds = sum(1 for link in self.graph.links if link.src == nid and link.src_port == "geometry")
                suffix = f"feeds {feeds} Solver{'s' if feeds != 1 else ''}" if feeds else "not linked"
                item.set_status(f"{v.get('nx', '?')} × {v.get('ny', '?')} cells · {suffix}")
            elif name == "monitor":
                item.set_status(f"t = {project.t_now:.4g}" if project.t_now else "waiting for a run")
            elif name == "results":
                item.set_status("following the run" if project.running else ("results ready" if project.results_dir else "no results yet"))
            elif name == "analysis":
                panel = self.editors.get(nid)
                series = getattr(panel, "series", None)
                snapshots = getattr(series, "snapshots", ())
                if snapshots:
                    item.set_status(f"{len(snapshots)} snapshots")
                else:
                    item.set_status("waiting for snapshots" if project.results_dir else "waiting for results")
        running = [self.graph.nodes[i].title for i, p in self.projects.items() if p.state != IDLE]
        self.state_label.setText("Active: " + ", ".join(running) if running else "Ready · no active runs")

    def closeEvent(self, event) -> None:
        busy = [p for p in self.projects.values() if p.state != IDLE]
        if busy:
            box = QMessageBox(self)
            box.setWindowTitle("Fenice")
            box.setText(f"{len(busy)} run(s) going.")
            stop = box.addButton("Stop them (save the checkpoints)", QMessageBox.ButtonRole.AcceptRole)
            kill = box.addButton("Kill them", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("Keep the window", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            if box.clickedButton() is stop:
                for p in busy:
                    p.stop()
                    p.finished.connect(lambda *_: self.close() if all(q.state == IDLE for q in self.projects.values()) else None)
                event.ignore()   # the window closes when the solvers have exited
                return
            if box.clickedButton() is kill:
                for p in busy:
                    p.kill()
            else:
                event.ignore()
                return
        self._timer.stop()
        for editor in list(self.editors.values()):
            if hasattr(editor, "shutdown"):
                editor.shutdown()
        for p in self.projects.values():
            p.shutdown()
        for win in self.windows.values():
            win.close()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv[:1])
    apply_theme(app)
    try:
        available_cases()
    except SolverNotFound as exc:
        QMessageBox.critical(None, "Fenice", str(exc))
        return 1
    win = Workbench()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
