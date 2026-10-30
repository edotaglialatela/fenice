"""The numbers of the end-of-run report of a solver run, read from its standard output."""
import re

NUM = r"([0-9.]+)"
_WALL = re.compile(rf"time loop {NUM} s, (\d+) steps")
_AMR = re.compile(rf"\[AMR\] (\d+) leaves at the end \(([^)]*)\); remesh {NUM} s of the {NUM} s time loop")
_PARTS = re.compile(rf"\[AMR\] remesh parts: gather {NUM} s, indicator {NUM} s, adapt {NUM} s, engine build {NUM} s, "
                    rf"wave-speed bound {NUM} s, partition {NUM} s")
_MEMORY = re.compile(rf"resident memory per rank \[MB\].*?after setup {NUM} / {NUM}, after the initial adaptation "
                     rf"{NUM} / {NUM}, at the end {NUM} / {NUM}, peak {NUM} / {NUM}")
_TABLES = re.compile(rf"tables of rank 0 \[MB\]: mesh {NUM} .*?engine {NUM} .*?engine uniform-mesh {NUM} .*?body band "
                     rf"{NUM}, indicator {NUM}")
PARTS = ["gather", "indicator", "adapt", "build", "speed", "partition"]
MEMORY = ["setup", "init", "end", "peak"]
TABLES = ["mesh", "engine", "uniform", "band", "xi"]
SECTIONS = ("amr", "parts", "memory", "tables")


def parse_summary(out, require=()):
    """Read the end-of-run report of a run from its output `out`.

    Always present: "wall" (the time loop [s]) and "steps". For an AMR run also "leaves", "remesh" [s] (and "levels", the
    leaves per level), then, when the solver printed them, "parts" (the remesh by part [s]), "memory_min" / "memory_max"
    (resident memory per rank [MB]: setup, init, end, peak) and "tables" (the tables of rank 0 [MB]).
    `require` names the sections that must be there, from "amr", "parts", "memory", "tables": a missing one raises
    RuntimeError (a message that changed in the solver must not turn into a silent None); so does a missing time loop line.
    """
    unknown = set(require) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown section(s) {sorted(unknown)}; choose from {SECTIONS}")
    m = _WALL.search(out)
    if m is None:
        raise RuntimeError("no 'time loop ... s, ... steps' line in the solver output (did the run finish?)")
    res = {"wall": float(m.group(1)), "steps": int(m.group(2))}
    found = {}
    m = _AMR.search(out)
    if m is not None:
        found["amr"] = True
        res["leaves"] = int(m.group(1))
        res["levels"] = [int(x) for x in re.findall(r"level \d+: (\d+)", m.group(2))]
        res["remesh"] = float(m.group(3))
    m = _PARTS.search(out)
    if m is not None:
        found["parts"] = True
        res["parts"] = dict(zip(PARTS, (float(x) for x in m.groups())))
    m = _MEMORY.search(out)
    if m is not None:
        found["memory"] = True
        v = [float(x) for x in m.groups()]
        res["memory_min"], res["memory_max"] = dict(zip(MEMORY, v[0::2])), dict(zip(MEMORY, v[1::2]))
    m = _TABLES.search(out)
    if m is not None:
        found["tables"] = True
        res["tables"] = dict(zip(TABLES, (float(x) for x in m.groups())))
    missing = [s for s in require if s not in found]
    if missing:
        raise RuntimeError(f"not in the solver output: {', '.join(missing)}")
    return res
