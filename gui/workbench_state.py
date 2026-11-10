"""JSON state files for the Fenice workbench graph and its editor state."""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from gui.graph import Graph

SCHEMA_VERSION = 1


def json_value(value):
    """Convert common Qt, NumPy and path values to strict JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("State contains a non-finite number.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "item"):
        return json_value(value.item())
    raise TypeError(f"State value {type(value).__name__} cannot be written to JSON.")


def write_state(path: str | Path, state: dict) -> Path:
    """Write one complete state file atomically beside its destination."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(json_value(state), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path


def new_state(nodes: list[dict], links: list[dict], workspace: dict) -> dict:
    return {"schema_version": SCHEMA_VERSION, "saved_at": datetime.now(timezone.utc).isoformat(),
            "nodes": nodes, "links": links, "workspace": workspace}


def read_state(path: str | Path, specs: dict) -> dict:
    """Read and validate the entire graph before the current workbench is replaced."""
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    if (not isinstance(state, dict) or isinstance(state.get("schema_version"), bool)
            or state.get("schema_version") != SCHEMA_VERSION):
        raise ValueError(f"Unsupported or missing state schema. This Fenice build reads schema {SCHEMA_VERSION}.")
    nodes = state.get("nodes")
    links = state.get("links")
    workspace = state.get("workspace", {})
    if not isinstance(nodes, list) or not isinstance(links, list) or not isinstance(workspace, dict):
        raise ValueError("State file needs node, link and workspace records.")
    if len(nodes) > 1000 or len(links) > 5000:
        raise ValueError("State file has too many graph records.")

    graph = Graph()
    saved_to_live: dict[int, int] = {}
    seen = set()
    for record in nodes:
        if not isinstance(record, dict):
            raise ValueError("Every node record must be an object.")
        old_id = record.get("id")
        name = record.get("module")
        title = record.get("title")
        position = record.get("position")
        editor = record.get("editor", {})
        project = record.get("project", {})
        if not isinstance(old_id, int) or isinstance(old_id, bool) or old_id < 1 or old_id in seen:
            raise ValueError("Node IDs must be unique positive integers.")
        if not isinstance(name, str) or name not in specs or specs[name].factory is None or specs[name].missing():
            raise ValueError(f"Module '{name}' is unavailable in this Fenice installation.")
        if not isinstance(title, str) or not isinstance(position, list) or len(position) != 2:
            raise ValueError(f"Node {old_id} has an invalid title or position.")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in position):
            raise ValueError(f"Node {old_id} has a non-finite canvas position.")
        if not isinstance(editor, dict) or not isinstance(project, dict):
            raise ValueError(f"Node {old_id} editor and project state must be objects.")
        if "params" in project and not isinstance(project["params"], dict):
            raise ValueError(f"Node {old_id} has invalid solver parameters.")
        for key in ("t_now", "t_end"):
            value = project.get(key, 0.0)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Node {old_id} has an invalid {key} value.")
        results_dir = project.get("results_dir")
        if results_dir is not None and not isinstance(results_dir, str):
            raise ValueError(f"Node {old_id} has an invalid results path.")
        if "monitor_every" in project and (not isinstance(project["monitor_every"], int)
                                             or isinstance(project["monitor_every"], bool)):
            raise ValueError(f"Node {old_id} has an invalid monitor interval.")
        if "log_history" in project and (not isinstance(project["log_history"], list)
                                           or not all(isinstance(line, str) for line in project["log_history"])):
            raise ValueError(f"Node {old_id} has invalid solver log history.")
        if "progress_history" in project and not isinstance(project["progress_history"], list):
            raise ValueError(f"Node {old_id} has invalid solver progress history.")
        if "progress_history" in project and any(not isinstance(row, list) for row in project["progress_history"]):
            raise ValueError(f"Node {old_id} has malformed solver progress rows.")
        node = graph.add(specs[name])
        node.title = title
        saved_to_live[old_id] = node.id
        seen.add(old_id)

    for record in links:
        if not isinstance(record, dict):
            raise ValueError("Every link record must be an object.")
        src, dst = record.get("src"), record.get("dst")
        if (not isinstance(src, int) or isinstance(src, bool) or not isinstance(dst, int) or isinstance(dst, bool)
                or not isinstance(record.get("src_port"), str) or not isinstance(record.get("dst_port"), str)):
            raise ValueError("Links need integer node IDs and string port names.")
        if src not in saved_to_live or dst not in saved_to_live:
            raise ValueError("A link points to a node that is not in the state file.")
        src_port, dst_port = record.get("src_port"), record.get("dst_port")
        reason = graph.check(saved_to_live[src], src_port, saved_to_live[dst], dst_port)
        if reason:
            raise ValueError(f"Invalid link {src}:{src_port} → {dst}:{dst_port}: {reason}.")
        graph.connect(saved_to_live[src], src_port, saved_to_live[dst], dst_port)

    if "geometry" in workspace and not isinstance(workspace["geometry"], str):
        raise ValueError("Workspace geometry must be a base64 string.")
    split_sizes = workspace.get("split_sizes")
    if split_sizes is not None and (not isinstance(split_sizes, list) or len(split_sizes) != 2
                                    or any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in split_sizes)):
        raise ValueError("Workspace monitor splitter sizes are invalid.")
    for key, size in (("view_transform", 9), ("view_center", 2)):
        values = workspace.get(key)
        if values is not None and (not isinstance(values, list) or len(values) != size
                                   or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
                                          for value in values)):
            raise ValueError(f"Workspace {key.replace('_', ' ')} is invalid.")
    selected = workspace.get("selected_nodes", [])
    if not isinstance(selected, list) or any(not isinstance(node_id, int) or isinstance(node_id, bool) or node_id not in seen
                                             for node_id in selected):
        raise ValueError("Workspace selection points to an unknown node.")
    monitor_tab = workspace.get("monitor_tab", 0)
    if not isinstance(monitor_tab, int) or isinstance(monitor_tab, bool) or monitor_tab < -1:
        raise ValueError("Workspace monitor tab index is invalid.")
    for key in ("monitor_views", "subwindows"):
        entries = workspace.get(key, {})
        if not isinstance(entries, dict):
            raise ValueError(f"Workspace {key.replace('_', ' ')} must be an object.")
        for saved_id, view in entries.items():
            expected = int if key == "monitor_views" else dict
            if not str(saved_id).isdigit() or int(saved_id) not in seen or not isinstance(view, expected):
                raise ValueError(f"Workspace {key.replace('_', ' ')} contains an invalid entry.")
            if key == "monitor_views" and (not isinstance(view, int) or isinstance(view, bool) or view < 0):
                raise ValueError("Workspace monitor page index is invalid.")
            if key == "subwindows" and (not isinstance(view.get("geometry", ""), str)
                                         or not isinstance(view.get("visible", False), bool)):
                raise ValueError("Workspace editor window state is invalid.")
    palette_module = workspace.get("palette_module")
    if palette_module is not None and not isinstance(palette_module, str):
        raise ValueError("Workspace palette selection is invalid.")
    return state
