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

**Requirements:** Python 3.9+, `PyQt6`, `numpy` and, for the result viewer, `matplotlib` (`python prerequisites.py`), plus a
built solver (`.\cpp\build.ps1`). Set `FENICE_EXE` to use an executable other than
`build\release\fenice.exe`.

## Workbench

```powershell
python gui\workbench.py
```

One window for the steps of a study. A list on the left holds the modules, the module on show is in the middle, and the state of the run
is below; the modules share one project (the parameters of the run, its results folder and its state).

* **Setup.** The dialog described below. *Run Simulation* hands the validated parameters to the project and starts the solver as a
  child process (`fenice --config <file>`); the window switches to Run.
* **Geometry.** The domain, the immersed body, the Cartesian mesh and the solid cells it makes, as the Setup page holds them now (it
  refreshes when the page comes up). A cylinder, a sphere (cut through its centre in 3-D), a `NACA dddd` section or an airfoil file,
  placed and rotated as the solver does; the cells whose centre is inside the body are drawn solid, the grid appears when a cell is a few
  pixels wide, and the *Mesh level* box shows the same body on the finer levels of an AMR run (cells across the chord or diameter, solid
  cells, the finest cell). The boundary condition of each side is written beside it. The shapes come from `fenice.geometry`, which follows
  the solver's rules; `cases/validation/check_geometry.py` checks that the number of solid cells equals the solver's on every case with a
  body, on three meshes each.
* **Run.** The solver's output as it arrives, the last step, time, time step and wave speed, and the buttons *Stop*, *Kill* and *Run
  again*. *Stop* does what Ctrl+C does in a terminal: the solver saves its checkpoint and the `.pvd` index, then exits. Under MPI,
  `mpiexec` aborts the job on Ctrl+C (also in a terminal), so the last periodic checkpoint is the one kept. *Kill* ends the solver and
  every process it started at once. Closing the window with a run going asks which of the two to do; if the workbench itself is ended (on
  Windows) the solver and its MPI processes end with it.
* **Results.** The result viewer, below, opened on the folder the solver reports and following it while the run goes on.

Modules are files in `gui/modules/`, each defining a `MODULE` record (name, title, a factory that takes the project, the Python packages
it needs); a module whose packages are missing is listed as unavailable. The project (`gui/project.py`) lives in memory: nothing is
saved or loaded yet. `gui/run.py`, described next, is unchanged.

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

## Result viewer

```powershell
python guiiewer.py results\<case>          # a results folder, or one of its snapshot files
python guiiewer.py results\<case> --screenshot view.png --field Mach --step 100 --zoom 2.5,4.2,2.6,3.5 --leaves
```

The viewer shows the fields of a run snapshot by snapshot, and its monitor curves. It reads the `.vtr` snapshots of a uniform mesh and
the `.vtu` / `.pvtu` snapshots of an adaptive one (also from an MPI run) through `fenice.io`, with no physics of its own.

* **Field tab.** Density, pressure, temperature, Mach number, the speed and the velocity components, and for an adaptive mesh the
  refinement level; the colour map, an optional locked colour range, pan and zoom (matplotlib's toolbar), and the value under the
  pointer (with the leaf level). A 3-D snapshot is shown as a slice normal to z, y or x, picked with a slider. An adaptive mesh is
  sampled on the pixels of the view, so zooming shows the finest leaves in sight; *Leaf outlines* draws them when fewer than 4000
  are in the view.
* **Monitor tab.** Any column of the monitor CSV against time, with the time of the snapshot on show marked.
* **Snapshots.** A slider and *Play*. The folder is rescanned every second, so a run in progress can be followed (*Follow latest*): the
  snapshots are found from the files, since the `.pvd` index is only written when a run ends. The time of a snapshot comes from the
  `.pvd` if there is one, else from the monitor CSV at that step.

The viewer is a widget (`gui.viewer.ViewerPanel`), so it can sit in its own window, as here, or in a larger front end
(`docs/proposals/gui-workbench.md`).

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
| `gui/workbench.py`, `gui/project.py`, `gui/runner.py`, `gui/modules/` | The workbench window, the shared project, the solver as a child process, the modules |
| `python/fenice/geometry.py` | The immersed body for previews (outline, solid cells), shared with the validation scripts |
| `gui/viewer.py`, `gui/viewer_data.py` | The result viewer (`ViewerPanel`; the data side has no Qt) |
| `gui/solver.py` | Bridge to the executable (`available_cases`, `case_defaults`, `write_config`, `save_case`) |
| `gui/theme.py` | Qt stylesheet and palette shared with the documentation theme |

The dialog's `sim_params` keys are exactly the solver's
[parameters]({{root}}guide/parameters.html), so a config written by the GUI can be
re-run from the command line.
