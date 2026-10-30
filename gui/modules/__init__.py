"""The modules of the workbench: every file here defines one ``MODULE`` record that the shell finds and lists.

A module is a widget made by ``factory(project)``. It reads the shared ``gui.project.Project`` and may emit a ``title_changed`` signal; if
it has a ``shutdown()`` method the shell calls it when the application closes. A module whose Python packages are missing is listed as
unavailable, with the reason, so an install without the optional pieces still runs.
"""
from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class ModuleSpec:
    name: str                       # short key
    title: str                      # label in the navigation list
    factory: Callable               # factory(project) -> QWidget
    order: int = 100                # position in the list
    requires: tuple[str, ...] = field(default_factory=tuple)   # Python packages the module imports
    description: str = ""

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
