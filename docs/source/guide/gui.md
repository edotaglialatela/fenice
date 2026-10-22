---
title: GUI
nav: GUI
section: Guide
order: 40
---

```powershell
python gui\run.py
```

The PyQt6 setup dialog collects every parameter of a run, validates it, writes a
temporary config file and runs `fenice --config <file>` in the same terminal. The
solver output streams there. Ctrl+C reaches the solver, which writes its checkpoint and
`.pvd` before exiting.

**Requirements:** Python 3.9+, `PyQt6` and `numpy` (`python prerequisites.py`), plus a
built solver (`.\cpp\build.ps1`). Set `FENICE_EXE` to use an executable other than
`build\release\fenice.exe`.

## Tabs

* **Configuration.** Test case, mesh, thermodynamics (ideal / stiffened gas), viscous
  fluxes, body forces, boundary conditions, numerical schemes (MUSCL and limiter,
  Riemann solver, time integration, BDF2 settings), output, and MPI rank count.
  Setting **Nz > 1** makes the run 3-D and shows the z inputs: Lz, the front/back BCs and,
  when enabled, the body force gz. Nz = 1 is a 2-D run.
* **Immersed Boundaries.** Obstacle shape, radius and centre (CZ appears for 3-D).
* **Solution Mapping.** Hot start from a checkpoint `.npz`. The source grid size is read
  from the file; legacy files without metadata ask for Nx / Ny.
* **AMR.** Refinement levels, indicator variables, the relative variation that flags a cell, the
  refine and join thresholds, smoothing iterations, remesh interval and initial levels. AMR can run on MPI ranks
  (the leaves are distributed over them; on one node OpenMP threads are usually faster), and it requires Nz = 1.

## Saving a case

**Save as case…** (next to *Run Simulation*) saves the configured run as a new test case.
It asks for a name and writes `cases/<name>/case.toml` with the dialog's parameters and
the initial conditions of the selected case. The new case appears in the test-case list
right away. To change its initial conditions, edit the file; see
[Writing a test case]({{root}}guide/writing-cases.html).

## Single source of truth

The dialog has no physics or defaults of its own. The test-case list comes from
`fenice --list-cases`, and picking a case loads `fenice --case NAME --print-config`.
A new case file under `cases/` therefore appears in the GUI with its defaults
automatically.

| Module | Role |
|---|---|
| `gui/run.py` | Entry point: dialog, config file, solver launch |
| `gui/setup_dialog.py` | `SetupDialog` (four tabs, validation, `sim_params`) |
| `gui/solver.py` | Bridge to the executable (`available_cases`, `case_defaults`, `write_config`, `save_case`) |
| `gui/theme.py` | Qt stylesheet and palette shared with the documentation theme |

The dialog's `sim_params` keys are exactly the solver's
[parameters]({{root}}guide/parameters.html), so a config written by the GUI can be
re-run from the command line.
