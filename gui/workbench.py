"""The Fenice workbench: an empty space on which modules are dropped and linked.

    python gui/workbench.py

The palette on the left lists the modules (Geometry, Solver, Monitor, Postprocessing, ...); drag one onto the canvas, or double-click it.
Each module is a node with typed ports: drag from an output port to an input port of the same type to link them. The default is a
Geometry node linked to a Solver node; a Monitor linked to the solver shows the run live, a Postprocessing node linked to it shows the
results (also after the run). Double-click a node to open its editor in a tab under the canvas. A node that is not linked to a solver
has nothing to show; it takes the project of the solver it is linked to. ``gui/run.py`` (the dialog, then the solver in the terminal) is
unchanged.
"""
from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PyQt6.QtCore import QPointF, Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QListWidgetItem, QMainWindow, QMenu, QMessageBox,  # noqa: E402
                             QSizePolicy, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from gui.canvas import GraphScene, GraphView, LinkItem, NodeItem, Palette  # noqa: E402
from gui.graph import Graph, Link  # noqa: E402
from gui.modules import ModuleSpec, discover  # noqa: E402
from gui.project import IDLE, Project  # noqa: E402
from gui.solver import SolverNotFound, available_cases  # noqa: E402
from gui.theme import apply_theme  # noqa: E402


def _placeholder(text: str) -> QWidget:
    w = QWidget()
    lay = QVBoxLayout(w)
    label = QLabel(text)
    label.setObjectName("note")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    lay.addWidget(label)
    return w


class Workbench(QMainWindow):
    def __init__(self, default_graph: bool = True):
        super().__init__()
        self.setWindowTitle("Fenice")
        self.resize(1360, 900)
        self.graph = Graph()
        self.specs = {s.name: s for s in discover()}
        self.items: dict[int, NodeItem] = {}
        self.links: dict[Link, object] = {}        # model link -> its item
        self.projects: dict[int, Project] = {}     # solver node id -> its project
        self.editors: dict[int, QWidget] = {}      # node id -> editor (the real one, or a placeholder while unlinked)
        self._bound: dict[int, int | None] = {}    # node id -> the solver node its editor was made for
        self._placeholders: set[int] = set()

        self.palette = Palette()
        self.palette.setFixedWidth(170)
        self.palette.setDragEnabled(True)
        for spec in sorted(self.specs.values(), key=lambda s: (s.order, s.name)):
            item = QListWidgetItem(spec.title)
            item.setData(Qt.ItemDataRole.UserRole, spec.name)
            missing = spec.missing() if spec.factory is not None else ["(see the tooltip)"]
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
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)   # an editor's own minimum height must not squeeze the canvas
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(lambda i: self.tabs.removeTab(i))
        self.split = QSplitter(Qt.Orientation.Vertical)
        self.split.addWidget(self.view)
        self.split.addWidget(self.tabs)
        self.split.setStretchFactor(1, 1)
        self.split.setSizes([300, 560])

        body = QWidget()
        lay = QHBoxLayout(body)
        lay.addWidget(self.palette)
        lay.addWidget(self.split, 1)
        self.state_label = QLabel("")
        self.state_label.setObjectName("note")
        outer = QWidget()
        col = QVBoxLayout(outer)
        col.addWidget(body, 1)
        col.addWidget(self.state_label)
        self.setCentralWidget(outer)

        self._timer = QTimer(self)   # the status line of every node
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start()
        if default_graph:
            geo = self.add_node("geometry", QPointF(-330, -60))
            sol = self.add_node("solver", QPointF(-40, -90))
            self.connect_nodes(geo.node_id, "geometry", sol.node_id, "geometry")
            self.open_editor(geo.node_id)
            self.open_editor(sol.node_id)
            self.tabs.setCurrentIndex(0)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not getattr(self, "_laid_out", False):
            self._laid_out = True
            QTimer.singleShot(0, self._layout_once)   # the sizes are known once the window has been laid out

    def _layout_once(self) -> None:
        h = self.split.height()
        self.split.setSizes([min(340, h // 2), h - min(340, h // 2)])
        self.fit_graph()

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

    # ---- nodes
    def add_node(self, name: str, pos: QPointF | None = None) -> NodeItem:
        spec = self.specs[name]
        node = self.graph.add(spec)
        item = NodeItem(node.id, spec, node.title)
        item.setPos(pos if pos is not None else QPointF(0, 0))
        self.scene.addItem(item)
        self.items[node.id] = item
        if spec.owns_project:
            project = Project(self)
            project.run_gate = lambda params, nid=node.id: self._gate(nid, params)
            self.projects[node.id] = project
            editor = spec.factory(project)
            self.editors[node.id] = editor
            self._bound[node.id] = node.id
            if spec.face is not None:
                item.set_face(spec.face(project, editor))
            project.started.connect(lambda nid=node.id: self._started(nid))
            project.finished.connect(lambda *_: self._refresh_status())
        elif spec.standalone:
            self.editors[node.id] = spec.factory(None)
            self._bound[node.id] = -1
            self._bind(node.id)
        else:
            self._bind(node.id)
        self._refresh_status()
        return item

    def _palette_add(self, entry: QListWidgetItem) -> None:
        if not (entry.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        at = self.view.mapToScene(self.view.viewport().rect().center())
        while any(abs(i.scenePos().x() - at.x()) < 40 and abs(i.scenePos().y() - at.y()) < 40 for i in self.items.values()):
            at += QPointF(40, 40)   # not on top of a node already there
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
            if item.node_id in self.projects:
                self.projects.pop(item.node_id).shutdown()
            self.graph.remove(item.node_id)
            self.items.pop(item.node_id)
            self.scene.removeItem(item)
        if refused:
            self.statusBar().showMessage(" ".join(refused), 8000)

    # ---- editors: bound to the project of the solver the node is linked to
    def _rebind(self) -> None:
        for nid, node in self.graph.nodes.items():
            if not node.spec.owns_project:
                self._bind(nid)
        self._refresh_status()

    def _bind(self, nid: int) -> None:
        solver = self.graph.solver_of(nid)
        want = solver.id if solver is not None else None
        if nid in self._bound and self._bound[nid] == want and nid in self.editors:
            return
        node, was_open = self.graph.nodes[nid], self._tab_of(nid) >= 0
        if node.spec.standalone:   # its editor stays; it is told which solver it feeds
            self.editors[nid].attach(self.projects.get(want) if want is not None else None)
            self._bound[nid] = want
            return
        self._drop_editor(nid, keep_tab=was_open)
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
        if was_open:
            self.tabs.addTab(editor, node.title)
            self.tabs.setCurrentWidget(editor)

    def _drop_editor(self, nid: int, keep_tab: bool = False) -> None:
        editor = self.editors.pop(nid, None)
        self._bound.pop(nid, None)
        if editor is None:
            return
        i = self.tabs.indexOf(editor)
        if i >= 0:
            self.tabs.removeTab(i)
        if hasattr(editor, "shutdown"):
            editor.shutdown()
        editor.setParent(None)
        editor.deleteLater()

    def _tab_of(self, nid: int) -> int:
        editor = self.editors.get(nid)
        return self.tabs.indexOf(editor) if editor is not None else -1

    def open_editor(self, nid: int) -> None:
        editor = self.editors[nid]
        if self.tabs.indexOf(editor) < 0:
            self.tabs.addTab(editor, self.graph.nodes[nid].title)
        self.tabs.setCurrentWidget(editor)

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
                body = f", {v.get('ibm_shape', '')}" if v.get("use_ibm") == "true" else ""
                item.set_status(f"{v.get('nx', '?')} x {v.get('ny', '?')}{body}")
            elif name == "monitor":
                item.set_status(f"t = {project.t_now:.4g}" if project.t_now else "waiting for a run")
            elif name == "results":
                item.set_status("following the run" if project.running else ("results ready" if project.results_dir else "no results yet"))
        running = [self.graph.nodes[i].title for i, p in self.projects.items() if p.state != IDLE]
        self.state_label.setText("running: " + ", ".join(running) if running else "")

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
