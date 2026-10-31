"""The canvas of the workbench: nodes to drag about, ports to drag a link from, links that follow the nodes.

Drawing and mouse handling only; what a link means, and what a node holds, is the business of ``gui.graph`` and ``gui.workbench``.
The scene asks the workbench through callbacks (``can_link``, ``on_link``, ``on_drop``, ``on_delete``, ``on_open``) so that it can be
driven the same way by a mouse and by a test.
"""
from __future__ import annotations

from PyQt6.QtCore import QMimeData, QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QDrag, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QGraphicsItem, QGraphicsPathItem, QGraphicsProxyWidget, QGraphicsScene, QGraphicsView, QListWidget,
                             QStyleOptionGraphicsItem)

from gui.theme import COLORS

MIME = "application/x-fenice-module"
NODE_W, HEADER, STATUS, ROW, PORT_R = 220, 26, 22, 22, 6.5
_TYPE_COLOR = {"geometry": "#2E7D52", "run": "#9A2D3A", "results": "#2B6CB0"}   # by port type, also used for the links


def type_color(kind: str) -> QColor:
    return QColor(_TYPE_COLOR.get(kind, COLORS["ink_soft"]))


class PortItem(QGraphicsItem):
    """A circle on the edge of a node; press on it and drag to another node's port to link them."""

    def __init__(self, node: "NodeItem", port, output: bool):
        super().__init__(node)
        self.node, self.port, self.output = node, port, output
        self.links: list["LinkItem"] = []
        self.hot = False   # a compatible target while a link is being dragged
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._hover = False

    def boundingRect(self) -> QRectF:
        return QRectF(-PORT_R - 3, -PORT_R - 3, 2 * PORT_R + 6, 2 * PORT_R + 6)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        colour = type_color(self.port.type)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(COLORS["ink"]) if (self._hover or self.hot) else colour.darker(130), 2 if self.hot else 1.2))
        painter.setBrush(QBrush(colour if (self.links or self.hot) else QColor(COLORS["bg_surface"])))
        painter.drawEllipse(QPointF(0, 0), PORT_R, PORT_R)

    def scene_pos(self) -> QPointF:
        return self.mapToScene(QPointF(0, 0))

    def hoverEnterEvent(self, event) -> None:
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, event) -> None:
        self._hover = False
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.scene().begin_link(self, event.scenePos())
            event.accept()
        else:
            super().mousePressEvent(event)


class LinkItem(QGraphicsPathItem):
    def __init__(self, a: PortItem, b: PortItem | None = None):
        super().__init__()
        self.a, self.b = a, b          # a is the first port pressed; b the port it ends on (None while dragging)
        self.end = a.scene_pos()
        self.setZValue(-1)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, b is not None)
        self.update_path()

    def ports(self):
        """(output port, input port)."""
        return (self.a, self.b) if self.a.output else (self.b, self.a)

    def update_path(self) -> None:
        p0 = self.a.scene_pos()
        p1 = self.b.scene_pos() if self.b is not None else self.end
        src, dst = (p0, p1) if self.a.output else (p1, p0)   # the curve leaves an output to the right and enters an input from the left
        dx = max(60.0, abs(dst.x() - src.x()) * 0.5)
        path = QPainterPath(src)
        path.cubicTo(QPointF(src.x() + dx, src.y()), QPointF(dst.x() - dx, dst.y()), dst)
        self.setPath(path)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget=None) -> None:
        colour = type_color(self.a.port.type)
        pen = QPen(colour, 3.0 if self.isSelected() else 2.0)
        if self.b is None:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(self.path())

    def shape(self) -> QPainterPath:   # a wider target than the line itself
        from PyQt6.QtGui import QPainterPathStroker
        stroker = QPainterPathStroker()
        stroker.setWidth(12)
        return stroker.createStroke(self.path())


class NodeItem(QGraphicsItem):
    def __init__(self, node_id: int, spec, title: str):
        super().__init__()
        self.node_id, self.spec, self.title = node_id, spec, title
        self.status = ""
        self.face: QGraphicsProxyWidget | None = None
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
                      | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.in_ports = {p.name: PortItem(self, p, False) for p in spec.inputs}
        self.out_ports = {p.name: PortItem(self, p, True) for p in spec.outputs}
        self._port_rows = max(len(spec.inputs), len(spec.outputs), 1)
        self._face_h = 0.0
        self._layout()

    def set_face(self, widget) -> None:
        self.face = QGraphicsProxyWidget(self)
        widget.setStyleSheet("QWidget { background: transparent; } QPushButton, QSpinBox { background: #FFFFFF; }")
        self.face.setWidget(widget)
        self._face_h = float(widget.sizeHint().height())
        self.face.resize(NODE_W - 12, self._face_h)
        self._layout()

    def set_status(self, text: str) -> None:
        if text != self.status:
            self.status = text
            self.update()

    def height(self) -> float:
        return HEADER + STATUS + self._port_rows * ROW + self._face_h + 8

    def _layout(self) -> None:
        self.prepareGeometryChange()
        y0 = HEADER + STATUS
        for i, p in enumerate(self.in_ports.values()):
            p.setPos(0, y0 + ROW * (i + 0.5))
        for i, p in enumerate(self.out_ports.values()):
            p.setPos(NODE_W, y0 + ROW * (i + 0.5))
        if self.face is not None:
            self.face.setPos(6, y0 + self._port_rows * ROW)

    def boundingRect(self) -> QRectF:
        return QRectF(-2, -2, NODE_W + 4, self.height() + 4)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, NODE_W, self.height())
        selected = self.isSelected()
        painter.setPen(QPen(QColor(COLORS["primary"]) if selected else QColor(COLORS["border_strong"]), 2 if selected else 1))
        painter.setBrush(QColor(COLORS["bg_surface"]))
        painter.drawRoundedRect(rect, 7, 7)
        painter.save()
        painter.setClipRect(QRectF(0, 0, NODE_W, HEADER))   # the top of the rounded rectangle, square below
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["primary"]))
        painter.drawRoundedRect(QRectF(0, 0, NODE_W, HEADER + 10), 7, 7)
        painter.restore()
        bold = QFont(painter.font())
        bold.setBold(True)
        painter.setFont(bold)
        painter.setPen(QColor(COLORS["on_primary"]))
        painter.drawText(QRectF(10, 0, NODE_W - 20, HEADER), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.title)
        small = QFont(painter.font())
        small.setBold(False)
        small.setPointSizeF(max(small.pointSizeF() - 1, 7))
        painter.setFont(small)
        painter.setPen(QColor(COLORS["ink_muted"]))
        painter.drawText(QRectF(10, HEADER, NODE_W - 20, STATUS), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                         painter.fontMetrics().elidedText(self.status, Qt.TextElideMode.ElideRight, NODE_W - 20))
        painter.setPen(QColor(COLORS["ink_soft"]))
        y0 = HEADER + STATUS
        for i, p in enumerate(self.in_ports.values()):
            painter.drawText(QRectF(12, y0 + ROW * i, NODE_W / 2 - 14, ROW), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             p.port.label or p.port.name)
        for i, p in enumerate(self.out_ports.values()):
            painter.drawText(QRectF(NODE_W / 2, y0 + ROW * i, NODE_W / 2 - 12, ROW),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, p.port.label or p.port.name)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for port in list(self.in_ports.values()) + list(self.out_ports.values()):
                for link in port.links:
                    link.update_path()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event) -> None:
        self.scene().on_open(self)
        event.accept()


class GraphScene(QGraphicsScene):
    """Callbacks, set by the workbench: can_link(out_port, in_port) -> reason or None; on_link(out_port, in_port); on_drop(name, pos);
    on_delete(nodes, links); on_open(node)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.can_link = lambda out, inp: None
        self.on_link = lambda out, inp: None
        self.on_drop = lambda name, pos: None
        self.on_delete = lambda nodes, links: None
        self.on_open = lambda node: None
        self._drag: LinkItem | None = None
        self.setSceneRect(-2000, -2000, 4000, 4000)

    # ---- links
    def ports(self):
        return [i for i in self.items() if isinstance(i, PortItem)]

    def begin_link(self, port: PortItem, pos: QPointF) -> None:
        self._drag = LinkItem(port)
        self._drag.end = pos
        self.addItem(self._drag)
        self._drag.update_path()
        for other in self.ports():   # light up what the link can end on
            other.hot = other is not port and other.output != port.output and self._reason(port, other) is None
            other.update()

    def _reason(self, a: PortItem, b: PortItem) -> str | None:
        out, inp = (a, b) if a.output else (b, a)
        return self.can_link(out, inp)

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None:
            self._drag.end = event.scenePos()
            self._drag.update_path()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is not None:
            drag, self._drag = self._drag, None
            target = next((i for i in self.items(event.scenePos()) if isinstance(i, PortItem) and i is not drag.a), None)
            self.removeItem(drag)
            for other in self.ports():
                other.hot = False
                other.update()
            if target is not None and target.output != drag.a.output and self._reason(drag.a, target) is None:
                out, inp = (drag.a, target) if drag.a.output else (target, drag.a)
                self.on_link(out, inp)
            grabber = self.mouseGrabberItem()
            if grabber is not None:   # the port that took the press keeps the mouse until it is told the button is up
                grabber.ungrabMouse()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def add_link(self, out: PortItem, inp: PortItem) -> LinkItem:
        link = LinkItem(out, inp)
        out.links.append(link)
        inp.links.append(link)
        self.addItem(link)
        out.update()
        inp.update()
        return link

    def remove_link(self, link: LinkItem) -> None:
        for port in (link.a, link.b):
            if link in port.links:
                port.links.remove(link)
            port.update()
        self.removeItem(link)

    # ---- keys and drops
    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and not isinstance(self.focusItem(), QGraphicsProxyWidget):
            sel = self.selectedItems()   # (a key typed in a node's own widget is that widget's)
            self.on_delete([i for i in sel if isinstance(i, NodeItem)], [i for i in sel if isinstance(i, LinkItem)])
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME):
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        if event.mimeData().hasFormat(MIME):
            self.on_drop(bytes(event.mimeData().data(MIME)).decode(), event.scenePos())
            event.acceptProposedAction()


class GraphView(QGraphicsView):
    def __init__(self, scene: GraphScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(COLORS["bg_window"]))
        self.setAcceptDrops(True)
        self._pan = None

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        super().drawBackground(painter, rect)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))   # a dotted grid, so that the space reads as a workbench
        step = 30
        x0, y0 = int(rect.left() // step) * step, int(rect.top() // step) * step
        pts = [QPointF(x, y) for x in range(x0, int(rect.right()) + step, step) for y in range(y0, int(rect.bottom()) + step, step)]
        painter.drawPoints(pts)

    def wheelEvent(self, event) -> None:
        k = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(k, k)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:   # the middle button pans
            self._pan = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan is not None:
            d = event.position() - self._pan
            self._pan = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton and self._pan is not None:
            self._pan = None
            self.unsetCursor()
            return
        super().mouseReleaseEvent(event)


class Palette(QListWidget):
    """The modules that can be added; drag one onto the canvas, or double-click it."""

    def startDrag(self, actions) -> None:
        item = self.currentItem()
        if item is None or not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        mime = QMimeData()
        mime.setData(MIME, str(item.data(Qt.ItemDataRole.UserRole)).encode())
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)
