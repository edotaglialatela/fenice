---
title: GUI
nav: GUI
section: Guide
order: 40
---

Fenice's main GUI is the workbench: it connects geometry, solver setup, runs, monitoring and results in one graph. A separate setup
dialog remains available for running a case directly.

**Requirements:** Python 3.9+, `PyQt6`, `numpy` and, for the result viewer and analysis tools, `matplotlib` (`python prerequisites.py`), plus a
built solver (`.\cpp\build.ps1`). Set `FENICE_EXE` to use an executable other than
`build\release\fenice.exe`.

## Workbench

```powershell
python gui\workbench.py
```

The workbench opens to an empty canvas with the module palette ready. Select a module and use **Add selected module**, double-click a
palette item, or drag one onto the canvas. The add and documentation buttons span the full palette width. Each module is a node with
typed ports: drag from an output to a compatible input to link them. A Solver's results output can feed both the Postprocessing and
Analysis Tools modules. Selecting a node highlights its connected workflow; use **Edit
selected** to open its editor and **Fit view** to bring every node into view. The wheel zooms the canvas, the middle button pans it,
and *Delete* removes selected nodes or links. Most editors open in their own window (closing it only hides it). The monitor panel
appears when a Monitor is linked, with separate **Plots** and **Console** tabs; its height is set when it first appears and is preserved
as modules are added. One Geometry node can feed multiple Solvers. Its
geometry is shared: edits and case-default reloads from any linked Solver update all of them.

**Save state...** writes the workflow to a `.fenice.json` file: graph nodes and links, their canvas positions, solver setup, geometry,
result-folder selections, analysis settings, open editor windows and monitor layout. **Restore state...** rebuilds that view; solver
processes return idle and are not resumed from a state file. Restoring replaces the current workflow after confirmation and is
unavailable while a solver is running.

* **Geometry** (output: geometry). Its editor keeps the domain, base mesh, immersed body, adaptive mesh and boundary conditions in a
  scrollable control column beside a larger mesh preview. The initial view and **Show domain** fit the exact simulation domain, without
  displaying space outside the solver mesh. It visualizes the Cartesian mesh, immersed body, side conditions
  and solid cells. A cylinder, a sphere (cut through its centre in 3-D), a `NACA dddd` section or an airfoil file,
  placed and rotated as the solver does; the cells whose centre is inside the body are drawn solid, the grid appears when a cell is a few
  pixels wide (the wheel zooms about the pointer, a drag pans, a double click shows the whole domain), and the *Mesh level* box shows
  the same body on the finer levels of an AMR run (cells across the chord or diameter, solid cells, the finest cell). The domain
  (`Lx`, `Ly`, `nz`, `Lz`), the adaptive mesh (AMR on or off, the finest and the initial levels, density/velocity/pressure indicators, criterion, refine and join thresholds, smoothing and remesh interval), the base mesh (`nx`, `ny`, with *Apply mesh*, *2×* and *½×*: each doubles or halves `nx` and `ny`, four times
  or a quarter of the cells in 2-D), the body (shape, radius, centre) and the side conditions are edited here; they are written into the
  fields of its linked solvers, which each run reads. The node holds its shared geometry (initially from a case's defaults) and can be
  edited before it is linked; on its first link it adopts the solver's geometry if untouched, then sends its geometry to any additional
  linked solver. Edits are sent to all linked solvers, and changing the case on any one of them reloads the shared geometry. The AMR
  fields start with all three indicator variables ticked, the refine and join thresholds at 0.11 and a remesh every step; a case that
  sets one of them itself (the airfoil case sets the interval and the indicator) keeps its own value. Choosing *Periodic* on a side sets the opposite side too. Loading another case on the solver reloads the geometry.
  A geometry value that is not a number is named by the solver's Run, with the advice to correct it in the Geometry node. The shapes come from `fenice.geometry`, which follows the solver's rules;
  `cases/validation/check_geometry.py` checks that the number of solid cells equals the solver's on every case with a body.
* **Solver** (input: geometry; outputs: run, results). The setup dialog described below, without the geometry: case, thermodynamics,
  physics, numerical scheme, output, MPI and hot start (the adaptive mesh is the geometry's). It does not run without a linked geometry. The node itself carries
  *Run* and *Stop*, the **Low-Mach** slider (inactive / active; a click moves it) (the Riemann solver of the dialog: `HLLC` or `HLLC (Low-Mach Thornber)`; it is
  disabled when another Riemann solver is chosen in the dialog, and leaves that choice alone) and **Record every** *N* steps (the
  `monitor_interval` of the next run; *case* keeps the case's own value; each record costs one extra residual evaluation). *Stop*
  creates a `STOP` file in the output folder: the solver saves its checkpoint and the `.pvd` index, then exits, serial or under MPI. A
  solver is its own project, so several solvers can sit on the canvas and run at the same time; a running solver cannot be deleted or
  have its geometry unlinked.
* **Monitor** (input: run). Four panels share the time axis, the residuals being the largest: the time step $\Delta t$ and the wave
  speed $S_{\max}$ (small), the residual norms of the density, momentum and energy equations relative to their first value, $R/R_0$
  (`res_rho`, `res_mom`, `res_E`, log axis with whole decades only, the legend on a translucent white background behind the text only), and one more quantity of your choice (a column of the monitor file, or
  the number of leaves of an AMR run). When a Monitor is present, the lower panel takes about a third of the work area, capped at half. The name of each panel is written inside its top-left corner and its latest value inside the
  top-right one, a panel has at most three ticks (two for the small ones), and the vertical range follows the data (a nearly constant quantity is not
  magnified). The labels are typeset with mathtext (Computer Modern), which needs no TeX installation. The curves are filled from the
  progress lines (every `save_interval` steps) and the monitor file: $\Delta t$ is the mean time step between two rows of the monitor
  file, so it is live at that pace, and $S_{\max}$ comes from the progress lines. Above the plots, a bar and a line give the percentage of the
  physical time, the step, $t$, $\Delta t$, $S_{\max}$ and an estimate of the time left. The **Console** tab shows the solver's output with its original colours plus colour coding for plain lines (step lines with $t$, $\Delta t$, $S_{\max}$ and the leaves in their own colours, section tags, errors in
  red, warnings in amber, checkpoints and the end of the run in green); only rank 0 writes under MPI. The console states the total
  physical time and notes each time the run passes a multiple of 10 % of it. A monitor linked in the middle of a run shows it from its
  start (the project keeps the history), and every run starts with an empty console and empty plots. *Stop* and *Kill* are here too;
  *Kill* ends the solver and every process it started at once.
* **Postprocessing** (input: results). The snapshot viewer, below, opens the folder the solver reports and follows it while the run goes
  on; linked after a run, it opens that run's results. It clears the previous snapshot when a new run begins and waits for the new output
  folder, so results from an earlier run are not left on screen.
* **Analysis Tools** (input: results). Inspect field statistics and sampled profiles, analyze any numeric monitor signal, compare
  snapshot modes, and explore parameter sensitivity. The cylinder tools are one example; the controls accept other fields, signals,
  and sample tables.

The monitor is light on the machine. The monitor file is read incrementally (only the bytes added since the last tick are parsed, so a run of
hundreds of thousands of rows costs about the same per tick as a short one), a curve of more than about 1500 points is drawn thinned to the
smallest and largest value of each bucket (no spike is lost), a panel on a tab that is not on show is redrawn only when it comes back, the
redraw interval grows with its cost (at least one second; the plots take at most about 7 % of a core), and the console writes a burst of
lines with one repaint. The plots are drawn by the CPU (matplotlib).

Closing the window with runs going asks whether to stop them (checkpoints saved) or kill them; if the workbench itself is ended (on
Windows) the solvers and their MPI processes end with it. Modules are files in `gui/modules/`, each defining a `MODULE` record (name,
title, ports, whether it owns a project, an editor factory, the Python packages it needs); a module whose packages are missing is
listed as unavailable. The graph and projects live in memory while the workbench is open; **Save state...** and **Restore state...**
save or rebuild the graph and its editor state. `gui/run.py`, described next, is unchanged.

## Run setup

The run setup uses a left navigation rail for **Configuration**, **Immersed body**, **Hot start** and **Adaptive mesh**. Configuration is
split into **Case and physics**, **Domain and boundaries**, and **Numerics and output**, so related settings stay together while longer
sections remain scrollable.

Run the setup dialog on its own with:

```powershell
python gui\run.py
```

It validates the selected parameters, writes a temporary config file and runs `fenice --config <file>` in the same terminal. Solver
output streams there. Ctrl+C reaches the solver, which writes its checkpoint and `.pvd` before exiting.

* **Configuration.** Test case, mesh, thermodynamics (ideal / stiffened gas), viscous
  fluxes, body forces, boundary conditions, numerical schemes (spatial reconstruction: 1st order, MUSCL with a limiter, WENO3-Z or WENO5-Z, and the order of the
  faces on the physical boundaries; Riemann solver,
  time integration), output, and MPI rank count. Choosing a WENO reconstruction sets the time integration
  to SSP Runge-Kutta 3 and colours, with a tooltip, the integrators that do not suit it; a combination the solver refuses
  (WENO with SSP Runge-Kutta 2, WENO5 with AMR or MPI) is explained under the list and
  blocked when the run is started. Explicit Euler with WENO is allowed with a warning.
  Setting **Nz > 1** makes the run 3-D and shows the z inputs: Lz, the front/back BCs and,
  when enabled, the body force gz. Nz = 1 is a 2-D run.
* **Immersed body.** Obstacle shape, radius and centre (CZ appears for 3-D).
* **Hot start.** Solution mapping from a checkpoint `.npz`. The source grid size is read
  from the file; legacy files without metadata ask for Nx / Ny.
* **Adaptive mesh.** Refinement levels, indicator variables (check boxes: density, velocity, pressure), the relative variation that flags a cell, the
  refine and join thresholds, smoothing iterations, remesh interval and initial levels. AMR can run on MPI ranks
  (the leaves are distributed over them; on one node OpenMP threads are usually faster), and it requires Nz = 1.

## Postprocessing viewer

```powershell
python gui\viewer.py results\<case>          # a results folder, or one of its snapshot files
python gui\viewer.py results\<case> --screenshot view.png --field Mach --step 100 --zoom 2.5,4.2,2.6,3.5 --leaves
```

The Postprocessing module shows the fields of a run snapshot by snapshot, and its monitor curves. Its toolbar groups folder, field and colour-map
selection separately from display options; 3-D slice controls appear only for 3-D results. The playback row keeps the snapshot slider,
Play and **Follow latest** together. It reads the `.vtr` snapshots of a uniform mesh and
the `.vtu` / `.pvtu` snapshots of an adaptive one (also from an MPI run) through `fenice.io`, with no physics of its own.

* **Field tab.** Density, pressure, temperature, Mach number, the speed and the velocity components, and for an adaptive mesh the
  refinement level and, for a run on MPI ranks, `MPI_Rank` (the rank that owns each leaf; a uniform-mesh MPI run writes one file and has
  none); the colour map, an optional locked colour range, pan and zoom (matplotlib's toolbar), and the value under the
  pointer (with the leaf level). A 3-D snapshot on a uniform mesh is shown as a slice normal to z, y or x, picked with a slider; a 3-D adaptive snapshot (the sphere) is shown on its mid-plane z = Lz / 2. An adaptive mesh is
  sampled on the pixels of the view, so zooming shows the finest leaves in sight; *Leaf outlines* draws them when fewer than 4000
  are in the view.
* **Monitor tab.** Any column of the monitor CSV against time, with the time of the snapshot on show marked.
* **Snapshots.** A slider and *Play*. The folder is rescanned every second, so a run in progress can be followed (*Follow latest*): the
  snapshots are found from the files, since the `.pvd` index is only written when a run ends. The time of a snapshot comes from the
  `.pvd` if there is one, else from the monitor CSV at that step.

The viewer is a widget (`gui.viewer.ViewerPanel`), so it can sit in its own window, as here, or in a larger front end, the workbench
(`python gui/workbench.py`).

## Analysis Tools

The workbench's separate **Analysis Tools** module reads the same results folder as Postprocessing and groups quantitative work into
four general-purpose pages. The cylinder recipes in the case guide are examples of how to set them up.

* **Fields / profiles.** See minimum, maximum, mean, standard deviation, RMS, median and the 5th/95th percentiles across the displayed
  cells or slice. Sample a line between editable coordinates and export it as CSV. A spatial spectrum can use a Hann window, even
  extension or no window; it reports peak wavelength and high-wavenumber leakage. On AMR data, the view reports how much of the line
  crosses leaves coarser than the finest level. The optional wake-line example applies coordinates from the low-blockage cylinder.
* **Signals / spectra.** Analyze any numeric monitor column. Set a start time with **From t**; analysis includes all later monitor
  rows. Choose a peak band, use raw frequency or nondimensional
  frequency `fL/U`, and select Hann, Hamming or no window with adjustable zero padding. Compare a second signal, inspect a selected
  harmonic, estimate cycles from upward zero crossings, and track peak drift with sliding windows. A reference band is optional and
  editable. Export the spectrum and sliding-window peaks to CSV.
* **POD / DMD.** Run weighted POD and exact DMD on a selected, uniformly timed snapshot series. The report includes modal energy,
  snapshot-step jitter, phase difference between the leading POD coefficients, DMD eigenvalue magnitude and growth, and the leading
  mode's projection onto the first two POD modes. Time-delay DMD works on any numeric monitor signal and can compare a second one,
  including when snapshots are sparse. Controls expose the first snapshot time, snapshot interval, step tolerance, delay embedding and
  optional `fL/U` scaling.
* **Sensitivity.** Load a Sobol `summary.json` to chart first-order and total indices with their 95% bootstrap intervals, or calculate
  them from a one-column CSV in Saltelli order `A, B, AB1, ...`. For that CSV workflow, set **Base samples n** to the sampler's base
  count and enter **Parameters** in the same order as the sampled inputs. Load a numeric sample table to choose an input for X, a response for Y,
  and an optional third input for point color; a reference value can be drawn on the response plot. Sensitivity scripts can place their
  A-sample columns in the summary or save them beside it as `a_sample.csv`. Intervals estimate sampling error in the indices; they do not
  include solver discretization error.

## Saving a case

**Save as case…** (next to *Run Simulation*) saves the configured run as a new test case.
It asks for a name and writes `cases/<name>/case.toml` with the dialog's parameters and
the initial conditions of the selected case. The new case appears in the test-case list
right away. To change its initial conditions, edit the file; see
[Writing a test case]({{root}}guide/writing-cases.html).

## Single source of truth

The dialog has no physics or defaults of its own. The test-case list comes from
`fenice --list-gui-cases` (the cases that only check the code, such as the manufactured solutions, are run from the command line), and picking a case loads `fenice --case NAME --print-config`.
A new case file under `cases/` therefore appears in the GUI with its defaults
automatically.

| Module | Role |
|---|---|
| `gui/run.py` | Entry point: dialog, config file, solver launch |
| `gui/setup_dialog.py` | `SetupDialog` (four tabs, validation, `sim_params`) |
| `gui/workbench.py`, `gui/canvas.py`, `gui/graph.py`, `gui/project.py`, `gui/runner.py`, `gui/modules/` | The workbench window, the shared project, the solver as a child process, the modules |
| `python/fenice/geometry.py` | The immersed body for previews (outline, solid cells), shared with the validation scripts |
| `gui/viewer.py`, `gui/viewer_data.py` | Postprocessing viewer (`ViewerPanel`) and its data reader (`Series`, `Frame`) |
| `gui/analysis.py`, `gui/analysis_tools.py`, `gui/modules/analysis.py` | Field, load, modal and sensitivity analysis tools |
| `gui/workbench_state.py` | Validated, atomic `.fenice.json` workbench state files |
| `gui/solver.py` | Bridge to the executable (`available_cases`, `case_defaults`, `write_config`, `save_case`) |
| `gui/theme.py` | Qt stylesheet and palette shared with the documentation theme |

The dialog's `sim_params` keys are exactly the solver's
[parameters]({{root}}guide/parameters.html), so a config written by the GUI can be
re-run from the command line.
