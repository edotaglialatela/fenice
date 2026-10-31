"""The modules of the workbench: every file here defines one ``MODULE`` record that the workbench finds and lists in its palette.

A module is a node of the graph. It has typed ports (what it takes from other nodes, what it gives them), and an editor widget made by
``factory(project)``. A module with ``owns_project`` (the solver) makes its own ``gui.project.Project``; the others are bound to the
project of the solver they are linked to, and their editor is built when the link is made and dropped when it is removed. The editor may
emit a ``title_changed`` signal; if it has a ``shutdown()`` method it is called when the editor is dropped. A module whose Python packages
are missing is listed as unavailable, with the reason, so an install without the optional pieces still runs.
"""
from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Port:
    name: str
    type: str            # a link joins an output and an input of the same type
    multi: bool = False  # an output that can feed several inputs (an input takes one link)
    label: str = ""


@dataclass(frozen=True)
class ModuleSpec:
    name: str                       # short key
    title: str                      # label in the palette and on the node
    factory: Callable               # factory(project) -> QWidget (the editor)
    order: int = 100                # position in the palette
    requires: tuple[str, ...] = field(default_factory=tuple)   # Python packages the module imports
    description: str = ""
    inputs: tuple[Port, ...] = field(default_factory=tuple)
    outputs: tuple[Port, ...] = field(default_factory=tuple)
    owns_project: bool = False      # the solver: makes its own Project; the other nodes use the one of the solver they are linked to
    standalone: bool = False        # the editor is made once with the node and holds its own state (factory(None)); attach(project) links it
    face: Callable | None = None    # face(project, editor) -> QWidget shown on the node itself (the solver's Run button and switches)

    def missing(self) -> list[str]:
        return [pkg for pkg in self.requires if importlib.util.find_spec(pkg) is None]


def discover() -> list[ModuleSpec]:
    """The MODULE record of every module file of this package, in `order`."""
    found = []
    for info in pkgutil.iter_modules(__path__):
        try:
            mod = importlib.import_module(f"{__name__}.{info.name}")
        except Exception as exc:   # noqa: BLE001 - a broken module file must not take the shell down
            found.append(ModuleSpec(info.name, info.name, None, 900, description=f"cannot be loaded: {exc}"))
            continue
        spec = getattr(mod, "MODULE", None)
        if isinstance(spec, ModuleSpec):
            found.append(spec)
    return sorted(found, key=lambda s: (s.order, s.name))
