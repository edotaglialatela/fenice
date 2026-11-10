# Fenice desktop GUI

Mode: operate

FORM: seed:0131b1b0b62abd15

Build path: code

## Scope

Redesign the PyQt6 workbench, solver setup, geometry editor, monitor, and results viewer as one consistent desktop product. Preserve the solver as the source of truth for cases and defaults, existing run and validation behavior, and the in-memory graph model.

## User and task

CFD researchers and the solver developer build a simulation workflow, configure a solver from its case defaults, run it, monitor progress, and inspect VTK results. The workbench is the primary surface. On launch, show an empty workbench with the module palette so the next action is clear.

## Product mechanism

The graph links a shared Geometry node to one or more Solver nodes, then links each run to Monitor and Postprocessing modules. Selecting a node should make its real dependencies easy to follow without hiding other branches.

## Chosen visual direction

Research Campus Wayfinding: borrow the high legibility, fixed-size humanist sans, committed marine and porcelain palette, clear semantic route colors, and a single route-trace interaction from technical wayfinding systems in research institutes. Keep native Qt navigation and controls; do not turn the interface into a literal sign, map, timetable, or transit diagram.

The first view has a persistent Modules rail, a spacious central graph canvas, a clear workflow header, and an initially quiet run/monitor area. Add Geometry, one or more Solvers, then Monitor and Postprocessing modules. Selecting a node traces its connected dependency path with labeled geometry, run, and results colors; neighboring branches remain visible.

Carry the same section cues into solver setup and the same type colors into geometry previews, monitor plots, and result legends. Use color with labels and status text so state never depends on color alone.

## Risk

Route color can imply a fixed order or become decoration. Preserve the graph's actual topology, label every port, and highlight only the selected route.
