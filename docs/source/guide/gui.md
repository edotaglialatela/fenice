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

An empty space on which modules are dropped and linked. The palette on the left lists the modules; drag one onto the canvas (or
double-click it, or use the right-click menu). Each module is a node with typed ports: drag from a port to a port of the same type to
link them (the ports that fit light up; a link that does not fit is refused). *Delete* removes the selected nodes or links, the wheel
zooms the canvas and the middle button pans it. Double-click a node to open its editor: every module opens in a window of its own (closing it only hides it), except the
Monitors, which fill the bottom half of the workbench (one tab each, the plots on the left and the console as a column on the right).
The default is a **Geometry** node linked to a **Solver** node.

* **Geometry** (output: geometry). The domain, the Cartesian mesh, the immersed body and the boundary condition of each side, with the
  solid cells they make, as a picture. A cylinder, a sphere (cut through its centre in 3-D), a `NACA dddd` section or an airfoil file,
  placed and rotated as the solver does; the cells whose centre is inside the body are drawn solid, the grid appears when a cell is a few
  pixels wide (the wheel zooms about the pointer, a drag pans, a double click shows the whole domain), and the *Mesh level* box shows
  the same body on the finer levels of an AMR run (cells across the chord or diameter, solid cells, the finest cell). The domain
  (`Lx`, `Ly`, `nz`, `Lz`), the base mesh (`nx`, `ny`, with *Apply mesh*, *2×* and *½×*: each doubles or halves `nx` and `ny`, four times
  or a quarter of the cells in 2-D), the body (shape, radius, centre) and the side conditions are edited here; they are written into the
  fields of the linked solver, which are the only copy. The node holds its own geometry (it starts from the defaults of a case) and can be
  edited before it is linked, and keeps it when unlinked; linking an edited geometry writes it into the solver, linking an untouched one
  adopts the solver's. Choosing *Periodic* on a side sets the opposite side too. Loading another case on the solver reloads the geometry.
  A geometry value that is not a number is named by the solver's Run, with the advice to correct it in the Geometry node. The shapes come from `fenice.geometry`, which follows the solver's rules;
  `cases/validation/check_geometry.py` checks that the number of solid cells equals the solver's on every case with a body.
* **Solver** (input: geometry; outputs: run, results). The setup dialog described below, without the geometry: case, thermodynamics,
  physics, numerical scheme, output, MPI, hot start and AMR. It does not run without a linked geometry. The node itself carries
  *Run* and *Stop*, the **Low-Mach** slider (inactive / active; a click moves it) (the Riemann solver of the dialog: `HLLC` or `HLLC (Low-Mach Thornber)`; it is
  disabled when another Riemann solver is chosen in the dialog, and leaves that choice alone) and **Record every** *N* steps (the
  `monitor_interval` of the next run; *case* keeps the case's own value; each record costs one extra residual evaluation). *Stop*
  creates a `STOP` file in the output folder: the solver saves its checkpoint and the `.pvd` index, then exits, serial or under MPI. A
  solver is its own project, so several solvers can sit on the canvas and run at the same time; a running solver cannot be deleted or
  have its geometry unlinked.
* **Monitor** (input: run). Four panels share the time axis, the residuals being the largest: the time step $\Delta t$ and the wave
  speed $S_{\max}$ (small), the residual norms of the density, momentum and energy equations relative to their first value, $R/R_0$
  (`res_rho`, `res_mom`, `res_E`, log axis with whole decades only), and one more quantity of your choice (a column of the monitor file, or
  the number of leaves of an AMR run). The name of each panel is written inside its top-left corner and its latest value inside the
  top-right one, a panel has at most three ticks, and the vertical range follows the data (a nearly constant quantity is not
  magnified). The labels are typeset with mathtext (Computer Modern), which needs no TeX installation. The curves are filled from the
  progress lines (every `save_interval` steps) and the monitor file. Above the plots, a bar and a line give the percentage of the
  physical time, the step, $t$, $\Delta t$, $S_{\max}$ and an estimate of the time left. The solver's console is a column on the
  right of the workbench, the full height of the window, showing the console of the monitor on show: the solver's own colours, and
  colours for its plain lines (step lines with $t$, $\Delta t$, $S_{\max}$ and the leaves in their own colours, section tags, errors in
  red, warnings in amber, checkpoints and the end of the run in green); only rank 0 writes under MPI. The console states the total
  physical time and notes each time the run passes a multiple of 10 % of it. A monitor linked in the middle of a run shows it from its
  start (the project keeps the history), and every run starts with an empty console and empty plots. *Stop* and *Kill* are here too;
  *Kill* ends the solver and every process it started at once.
* **Postprocessing** (input: results). The result viewer, below, opened on the folder the solver reports and following it while the
  run goes on; linked after a run, it opens that run's results.

Closing the window with runs going asks whether to stop them (checkpoints saved) or kill them; if the workbench itself is ended (on
Windows) the solvers and their MPI processes end with it. Modules are files in `gui/modules/`, each defining a `MODULE` record (name,
title, ports, whether it owns a project, an editor factory, the Python packages it needs); a module whose packages are missing is
listed as unavailable. The graph and the projects live in memory: nothing is saved or loaded yet. `gui/run.py`, described next, is
unchanged.

## Tabs

* **Configuration.** Test case, mesh, thermodynamics (ideal / stiffened gas), viscous
  fluxes, body forces, boundary conditions, numerical schemes (MUSCL and limiter,
  Riemann solver, time integration, BDF2 settings), output, and MPI rank count.
  Setting **Nz > 1** makes the run 3-D and shows the z inputs: Lz, the front/back BCs and,
  when enabled, the body force gz. Nz = 1 is a 2-D run.
* **Immersed Boundaries.** Obstacle shape, radius and centre (CZ appears for 3-D).
* **Solution Mapping.** Hot start from a checkpoint `.npz`. The source grid size is read
  from the file; legacy files without metadata ask for Nx / Ny.
* **AMR.** Refinement levels, indicator variables (check boxes: density, velocity, pressure), the relative variation that flags a cell, the
  refine and join thresholds, smoothing iterations, remesh interval and initial levels. AMR can run on MPI ranks
  (the leaves are distributed over them; on one node OpenMP threads are usually faster), and it requires Nz = 1.

## Result viewer

```powershell
python gui\viewer.py results\<case>          # a results folder, or one of its snapshot files
python gui\viewer.py results\<case> --screenshot view.png --field Mach --step 100 --zoom 2.5,4.2,2.6,3.5 --leaves
```

The viewer shows the fields of a run snapshot by snapshot, and its monitor curves. It reads the `.vtr` snapshots of a uniform mesh and
the `.vtu` / `.pvtu` snapshots of an adaptive one (also from an MPI run) through `fenice.io`, with no physics of its own.

* **Field tab.** Density, pressure, temperature, Mach number, the speed and the velocity components, and for an adaptive mesh the
  refinement level and, for a run on MPI ranks, `MPI_Rank` (the rank that owns each leaf; a uniform-mesh MPI run writes one file and has
  none); the colour map, an optional locked colour range, pan and zoom (matplotlib's toolbar), and the value under the
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
| `gui/workbench.py`, `gui/canvas.py`, `gui/graph.py`, `gui/project.py`, `gui/runner.py`, `gui/modules/` | The workbench window, the shared project, the solver as a child process, the modules |
| `python/fenice/geometry.py` | The immersed body for previews (outline, solid cells), shared with the validation scripts |
| `gui/viewer.py`, `gui/viewer_data.py` | The result viewer (`ViewerPanel`; the data side has no Qt) |
| `gui/solver.py` | Bridge to the executable (`available_cases`, `case_defaults`, `write_config`, `save_case`) |
| `gui/theme.py` | Qt stylesheet and palette shared with the documentation theme |

The dialog's `sim_params` keys are exactly the solver's
[parameters]({{root}}guide/parameters.html), so a config written by the GUI can be
re-run from the command line.
