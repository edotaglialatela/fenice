"""The table of environment switches, the code and the documentation agree.

    python cases/validation/check_switches.py

Checks that (1) no FENICE_* variable is read with getenv outside cpp/src/Switches.cpp, (2) every switch the solver reads through
switch_on / switch_value is in the table printed by `fenice --print-switches`, and every table entry for the solver is read
somewhere (no dead entries), (3) every Python-side entry is read by the Python package, (4) the documentation page of the
switches takes its table from the solver. Exit code 0 when everything agrees.
"""
import re
import subprocess
import sys
from pathlib import Path

import fenice

REPO = Path(__file__).resolve().parents[2]
sources = [p for d in ("cpp/src", "cpp/apps", "cpp/include") for p in (REPO / d).rglob("*") if p.suffix in (".cpp", ".hpp")]

out = subprocess.run([fenice.exe(), "--print-switches"], capture_output=True, text=True, check=True).stdout
table = {m[0]: m[2] for m in re.findall(r"^  (FENICE_\w+)  \((\w+), read by the (\w+)\)$", out, re.M)}
problems = []
if not table:
    problems.append("fenice --print-switches printed no switch")

used = set()
for p in sources:
    text = p.read_text(encoding="utf-8", errors="replace")
    rel = p.relative_to(REPO).as_posix()
    if rel != "cpp/src/Switches.cpp":
        for m in re.finditer(r'getenv\(\s*"(FENICE_\w+)"', text):
            problems.append(f"{rel}: getenv(\"{m.group(1)}\"): read it through switch_on / switch_value")
        for m in re.finditer(r'switch_(?:on|value)\(\s*"(FENICE_\w+)"', text):
            used.add(m.group(1))
            if m.group(1) not in table:
                problems.append(f"{rel}: {m.group(1)} is not in the table")
for name, reader in table.items():
    if reader == "solver" and name not in used:
        problems.append(f"{name} is in the table but no code reads it")
py = "".join(p.read_text(encoding="utf-8", errors="replace") for p in (REPO / "python").rglob("*.py"))
for name, reader in table.items():
    if reader == "python" and name not in py:
        problems.append(f"{name} is in the table for Python but python/ does not read it")

page = (REPO / "docs" / "source" / "guide" / "switches.md").read_text(encoding="utf-8")
if "{{switches}}" not in page:
    problems.append("docs/source/guide/switches.md does not take its table from the solver ({{switches}})")

for p in problems:
    print("PROBLEM:", p)
print(f"{len(table)} switches, {len(used)} read by the solver" + ("" if not problems else f", {len(problems)} problem(s)"))
sys.exit(1 if problems else 0)
