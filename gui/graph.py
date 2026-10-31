"""The graph of the workbench, without any drawing: nodes made from module records, typed ports and the links between them.

A link joins an output port to an input port of the same type; an input takes one link, an output with ``multi`` can feed several.
The solver is the node that owns a project; every other node is bound to the solver found by following its input links upstream
(``solver_of``), so linking a geometry, a monitor or a post-processing node to a solver is what gives it something to show.
"""
from __future__ import annotations

from dataclasses import dataclass

from gui.modules import ModuleSpec, Port


@dataclass
class Node:
    id: int
    spec: ModuleSpec
    title: str


@dataclass(frozen=True)
class Link:
    src: int
    src_port: str
    dst: int
    dst_port: str


class Graph:
    def __init__(self):
        self.nodes: dict[int, Node] = {}
        self.links: list[Link] = []
        self._next = 1
        self._counts: dict[str, int] = {}

    # ---- nodes
    def add(self, spec: ModuleSpec) -> Node:
        n = self._counts[spec.name] = self._counts.get(spec.name, 0) + 1
        node = Node(self._next, spec, f"{spec.title} {n}")
        self._next += 1
        self.nodes[node.id] = node
        return node

    def remove(self, node_id: int) -> list[Link]:
        """Remove a node and the links of it; returns the links removed."""
        gone = [l for l in self.links if node_id in (l.src, l.dst)]
        self.links = [l for l in self.links if l not in gone]
        del self.nodes[node_id]
        return gone

    # ---- links
    def port(self, node_id: int, name: str, output: bool) -> Port | None:
        spec = self.nodes[node_id].spec
        return next((p for p in (spec.outputs if output else spec.inputs) if p.name == name), None)

    def check(self, src: int, src_port: str, dst: int, dst_port: str) -> str | None:
        """None if the link can be made, else the reason it cannot."""
        if src == dst:
            return "a node cannot be linked to itself"
        a, b = self.port(src, src_port, True), self.port(dst, dst_port, False)
        if a is None or b is None:
            return "no such port"
        if a.type != b.type:
            return f"a {a.type} output cannot feed a {b.type} input"
        link = Link(src, src_port, dst, dst_port)
        if link in self.links:
            return "already linked"
        if any(l.dst == dst and l.dst_port == dst_port for l in self.links):
            return "that input is already linked"
        if not a.multi and any(l.src == src and l.src_port == src_port for l in self.links):
            return "that output already feeds a node"
        return None

    def connect(self, src: int, src_port: str, dst: int, dst_port: str) -> Link:
        reason = self.check(src, src_port, dst, dst_port)
        if reason:
            raise ValueError(reason)
        link = Link(src, src_port, dst, dst_port)
        self.links.append(link)
        return link

    def disconnect(self, link: Link) -> None:
        self.links.remove(link)

    # ---- who is bound to what
    def upstream(self, node_id: int) -> list[Node]:
        return [self.nodes[l.src] for l in self.links if l.dst == node_id]

    def downstream(self, node_id: int) -> list[Node]:
        return [self.nodes[l.dst] for l in self.links if l.src == node_id]

    def solver_of(self, node_id: int) -> Node | None:
        """The solver node a node is bound to: itself if it owns a project, else the one it is linked to, upstream (an input link
        from a solver: monitor, post-processing) or downstream (the geometry feeds a solver)."""
        node = self.nodes[node_id]
        if node.spec.owns_project:
            return node
        for other in self.upstream(node_id) + self.downstream(node_id):
            if other.spec.owns_project:
                return other
        return None
