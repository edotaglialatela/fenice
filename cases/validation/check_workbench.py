"""Checks of the workbench canvas, driven with the mouse (QTest, offscreen): dragging a node, linking ports, refused links, the Delete key,
the editors bound to a solver, the Low-Mach switch and the geometry edits reaching the solver's fields. No solver is run.

    python cases/validation/check_workbench.py        # exit 0 if every check passes; needs PyQt6, matplotlib and a built solver
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui.theme import apply_theme  # noqa: E402
from gui.modules.setup import Switch  # noqa: E402
from gui.workbench import Workbench  # noqa: E402

app = QApplication([])
apply_theme(app)
w = Workbench()
w.resize(1360, 900)
w.show()
failures = []


def pump(s):
    t = time.time()
    while time.time() - t < s:
        app.processEvents()
        time.sleep(0.02)


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


vp=w.view.viewport()
def vpos(scene_pt): return w.view.mapFromScene(scene_pt)
def drag(a, b, steps=6):
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a); pump(0.05)
    for i in range(1,steps+1):
        p=QPoint(a.x()+(b.x()-a.x())*i//steps, a.y()+(b.y()-a.y())*i//steps)
        QTest.mouseMove(vp, p); pump(0.02)
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b); pump(0.15)
geo, sol = w.items[1], w.items[2]
# 1 drag a node by its header
before=geo.scenePos(); before_v=vpos(geo.scenePos()); h=vpos(geo.mapToScene(QPointF(60,10)))
drag(h, h+QPoint(0,60)); ok(abs(geo.scenePos().y()-before.y())>20 or abs(vpos(geo.scenePos()).y()-before_v.y())>20, f'node dragged ({before.y():.0f} -> {geo.scenePos().y():.0f})')
# link line follows
ok(w.items[1].out_ports['geometry'].links[0].path().boundingRect().height()>0, 'link follows')
# 2 add a monitor from the palette (double click) and a post node
pal={w.palette.item(i).text():w.palette.item(i) for i in range(w.palette.count())}
w._palette_add(pal['Monitor']); w._palette_add(pal['Postprocessing']); pump(0.3)
mon=w.items[3]; post=w.items[4]
mon.setPos(sol.scenePos()+QPointF(330,-60)); post.setPos(sol.scenePos()+QPointF(330,120)); w.fit_graph(); pump(0.3)
# incompatible: geometry output -> monitor run input
a=vpos(geo.out_ports['geometry'].scene_pos()); b=vpos(mon.in_ports['run'].scene_pos())
n=len(w.graph.links); drag(a,b); ok(len(w.graph.links)==n, 'incompatible link (geometry -> monitor) refused')
# run -> monitor, results -> post
drag(vpos(sol.out_ports['run'].scene_pos()), vpos(mon.in_ports['run'].scene_pos())); ok(len(w.graph.links)==n+1, 'solver run -> monitor linked by mouse')
drag(vpos(sol.out_ports['results'].scene_pos()), vpos(post.in_ports['results'].scene_pos())); ok(len(w.graph.links)==n+2, 'solver results -> postprocessing linked')
# a second link into the same input is refused
w._palette_add(pal['Solver']); pump(0.3); sol2=w.items[5]; sol2.setPos(sol.scenePos()+QPointF(0,260)); w.fit_graph(); pump(0.3)
drag(vpos(sol2.out_ports['run'].scene_pos()), vpos(mon.in_ports['run'].scene_pos())); ok(len(w.graph.links)==n+2, 'second link into a taken input refused')
# solver 2 without geometry cannot run
try: w.projects[5].start_run({'x':1}); ok(False,'run without geometry refused')
except RuntimeError as e: ok('Geometry' in str(e), 'run without a geometry refused: '+str(e)[:50])
w.fit_graph(); pump(0.3); w.grab().save('C:/fgold/shots/wb1.png')
# delete a link with the Delete key
w.scene.clearSelection(); li=sol.out_ports['results'].links[0]; li.setSelected(True); w.scene.setFocus(); QTest.keyClick(vp, Qt.Key.Key_Delete); pump(0.2)
ok(len(w.graph.links)==n+1 and not sol.out_ports['results'].links, 'link deleted with the Delete key')
ok(w.editors[4].findChild(object) is not None and 4 in w._placeholders, 'postprocessing back to a placeholder when unlinked')
drag(vpos(sol.out_ports['results'].scene_pos()), vpos(post.in_ports['results'].scene_pos())); ok(4 not in w._placeholders, 'relinked: viewer rebuilt')
# delete node solver 2 and the post node via selection
sol2.setSelected(True); QTest.keyClick(vp, Qt.Key.Key_Delete); pump(0.2); ok(5 not in w.items and 5 not in w.projects, 'solver node deleted')
print('links', [(l.src,l.src_port,l.dst,l.dst_port) for l in w.graph.links])

# ---- the Low-Mach switch and the geometry edits (the solver page is node 2, a geometry node 1)
page, geo = w.editors[2], w.editors[1]
face = w.items[2].face.widget()
low = face.findChild(Switch)
for case, want in (("Sod Shock Tube", False), ("IBM Airfoil AMR", True)):
    page.combo_testcase.setCurrentText(case)
    pump(0.3)
    ok(low.isChecked() == want and low.isEnabled(), f"{case}: Low-Mach switch follows the case ({page.combo_riemann.currentText()})")
low.setChecked(False)
ok(page.combo_riemann.currentText() == "HLLC", "the switch sets the Riemann solver")
page.combo_riemann.setCurrentText("HLLC (Preconditioned)")
low.setChecked(True)
ok(not low.isEnabled() and page.combo_riemann.currentText() == "HLLC (Preconditioned)", "another Riemann solver is left alone")
ok(w.tabs.count() == 1 and w.tabs.widget(0) is w.editors[3], "the bottom half holds the monitors only")
mon_panel = w.editors[3]
ok(mon_panel.text.mapTo(mon_panel, QPoint(0, 0)).x() > mon_panel.canvas.mapTo(mon_panel, QPoint(0, 0)).x(), "the console is a column on the right of the plots")
ok(not w.windows[1].isVisible(), "an editor is not open until its node is opened")
w.open_editor(1)
ok(w.windows[1].isVisible() and w.windows[1].isWindow() and w.editors[1].window() is w.windows[1], "a module opens in a subwindow of its own")
pump(0.5)
page.combo_riemann.setCurrentText("HLLC")
pump(0.1)
QTest.mouseClick(low, Qt.MouseButton.LeftButton)
ok(low.isChecked() and page.combo_riemann.currentText() == "HLLC (Low-Mach Thornber)", "a click on the Low-Mach slider moves it to active")
QTest.mouseClick(low, Qt.MouseButton.LeftButton)
ok(not low.isChecked() and page.combo_riemann.currentText() == "HLLC", "and back to inactive")
geo.fields["Lx"].setText("2.5")
geo.fields["Lx"].editingFinished.emit()
ok(page.input_Lx.text() == "2.5", "a domain edit reaches the solver's field")
page.combo_testcase.setCurrentText("IBM Flow over Cylinder")
pump(0.5)
ok(geo.chk_body.isChecked() and geo.fields["Lx"].text() != "2.5", "a new case on the solver reloads the geometry node")
ok(not page.group_domain.isVisibleTo(page) and not page.bc_left.isVisibleTo(page), "the solver page shows no domain or side fields")

# ---- keys typed in a node's own widget are that widget's: Backspace in the face's spin box must not delete the selected node
from PyQt6.QtWidgets import QSpinBox  # noqa: E402

w.scene.clearSelection()
w.items[1].setSelected(True)
spin = w.items[2].face.widget().findChild(QSpinBox)
w.items[2].face.setFocus()
spin.setFocus()
pump(0.1)
n_nodes, before = len(w.graph.nodes), spin.value()
QTest.keyClick(spin, Qt.Key.Key_Backspace)
pump(0.1)
ok(len(w.graph.nodes) == n_nodes and 1 in w.items, "Backspace in the face's box does not delete the selected node")

# ---- a drop from the palette (the drag and drop events, as QDrag.exec cannot run offscreen)
from PyQt6.QtCore import QMimeData  # noqa: E402
from PyQt6.QtGui import QDragEnterEvent, QDropEvent  # noqa: E402
from gui.canvas import MIME  # noqa: E402

mime = QMimeData()
mime.setData(MIME, b"monitor")
at = QPoint(300, 200)
n_nodes = len(w.graph.nodes)
enter = QDragEnterEvent(at, Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
QApplication.sendEvent(w.view.viewport(), enter)
drop = QDropEvent(QPointF(at), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
QApplication.sendEvent(w.view.viewport(), drop)
pump(0.2)
new = [i for nid, i in w.items.items() if w.graph.nodes[nid].spec.name == "monitor" and nid not in (3,)]
near = bool(new) and (w.view.mapFromScene(new[-1].scenePos()) - at).manhattanLength() < 5
ok(len(w.graph.nodes) == n_nodes + 1 and near, "a module dropped from the palette appears where it was dropped")

# ---- two solvers on the same case into the same folder: the second run is refused
g2, s2 = w.add_node("geometry"), w.add_node("solver")
w.connect_nodes(g2.node_id, "geometry", s2.node_id, "geometry")
w.editors[s2.node_id].combo_testcase.setCurrentText("Sod Shock Tube")
page.combo_testcase.setCurrentText("Sod Shock Tube")
pump(0.3)
a = page.peek_params()
a.update({"nx": "20000", "t_end": "0.2", "output_root": str(Path(__file__).resolve().parents[2] / "results" / "check_workbench")})
w.projects[2].start_run(a)
pump(0.3)
try:
    w.projects[s2.node_id].start_run(dict(a))
    ok(False, "the same case into the same folder is refused")
except RuntimeError as exc:
    ok("same results folder" in str(exc), "the same case into the same folder is refused")
w.projects[2].kill()
t0 = time.time()
while w.projects[2].running and time.time() - t0 < 30:
    pump(0.2)
# ---- the geometry node owns its geometry: edited unlinked, it is written into the solver it is then linked to
g3, s3 = w.add_node("geometry"), w.add_node("solver")
geo3 = w.editors[g3.node_id]
geo3.fields["Lx"].setText("7.5")
geo3.fields["Lx"].editingFinished.emit()
geo3._set_mesh("123", "45")
pump(0.2)
ok(geo3.p["Lx"] == "7.5" and geo3.p["nx"] == "123" and geo3.project is None, "an unlinked geometry node can be edited")
w.connect_nodes(g3.node_id, "geometry", s3.node_id, "geometry")
pump(0.3)
page3 = w.editors[s3.node_id]
ok(page3.input_Lx.text() == "7.5" and page3.input_nx.text() == "123", "an edited geometry is written into the solver it is linked to")
ok(page3._geometry_problem() is None, "valid geometry: no problem reported")
page3.input_Lx.setText("abc")
ok("Lx" in (page3._geometry_problem() or ""), "an invalid geometry value is named, with where to correct it")
page3.input_Lx.setText("7.5")
link = next(l for l in w.graph.links if l.src == g3.node_id)
w._remove_link(link)
pump(0.2)
ok(w.editors[g3.node_id] is geo3 and geo3.project is None and geo3.p["nx"] == "123", "unlinking keeps the geometry")
w.close()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED")
sys.exit(1 if failures else 0)
