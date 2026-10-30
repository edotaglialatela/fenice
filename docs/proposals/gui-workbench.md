# A modular front end: proposal

Status: proposal. The first module (the result viewer, `gui/viewer.py`) exists and is written as a widget so that it can be hosted;
the shell and the rest are not started. The scope of each module, and the order, are the user's.

## Aim

One application in which the steps of a study are modules that plug into a common project: set the case up, look at the geometry
and the grid, run, watch the run, look at the results, and later set up the thermochemistry. Today there are two separate things: a
setup dialog (`gui/run.py`) that launches the solver in the terminal, and the viewer. The front end stays what the GUI already is, a
thin client: no physics, the solver is the single source of truth (`fenice --list-cases`, `--print-config`, `--save-case`), the files
it reads and writes are the solver's own.

## Shape

* **Project.** A project is a case (a `case.toml`, or the parameters of the dialog) and its results folder. It holds the parameter
  dictionary (keys exactly the solver's), the path of the results, and the state of the run (idle, running, finished). Modules read it,
  and tell it what they changed; they do not talk to each other directly.
* **Module.** A widget with a small contract: a name and an icon; `set_project(project)` (called when the project or its parameters
  change); the signals `title_changed` and `parameters_changed(dict)`; `shutdown()` (stop timers and processes). The viewer already
  has this shape (`ViewerPanel`, `title_changed`, `shutdown`).
* **Shell.** A window with a project tree or tab strip on the left, the active module in the middle, and a status line and a log below.
  It owns the project and the solver process, and opens a module on a tab or in its own window.
* **Registration.** A module is a file in `gui/modules/` that defines one `MODULE` record (name, title, factory, the Python packages it
  needs). The shell imports whatever it finds; a module whose packages are missing is listed as unavailable with the reason, so that an
  install without the optional pieces still runs. Modules from outside the repository can be added later through the same record.

## Modules

| Module | What it does | State |
| --- | --- | --- |
| Setup | the existing dialog (case, mesh, physics, boundaries, numerics, IBM, AMR) as a module | exists as a dialog |
| Geometry and grid | the immersed body drawn in the domain (cylinder, sphere, NACA section, airfoil file), the Cartesian grid, the mask at the chosen resolution ("cells per chord"), where AMR would refine; an airfoil generator | proposed |
| Boundaries | the boundary condition of each side drawn on the domain, with its values | proposed |
| Run | starts the solver, streams its log, shows the step, the time and the residual-like monitor values, stops it with the checkpoint | proposed |
| Results | the viewer: fields, slices, adaptive meshes, monitor curves, following a run | exists |
| Studies | the validation studies and the regression, with their pass or fail | proposed |
| Thermochemistry | species, mechanism and reference reactor, when the species work exists (`species-target-cases.md`) | later |
| Mesh (unstructured) | only if the unstructured solver is built (`unstructured-solver.md`) | later |

## Run: what has to be decided

Now the dialog quits and the solver runs in the terminal, where Ctrl+C reaches it and it saves a checkpoint and the `.pvd`. A Run
module would start it as a child process and stream the output. Stopping it cleanly from a window needs one of: sending the console
interrupt to the child (platform-dependent), or a stop file that the solver polls once per step and answers with the same
checkpoint and index. The second changes the solver (a small, numerics-free addition) and is the user's decision.

## Order

1. The result viewer (done), then the shell around it with the setup dialog as the first neighbour.
2. Run module, so that the three steps (set up, run, look) are in one window.
3. Geometry and grid, then boundaries.
4. Studies. Thermochemistry and the unstructured mesh only with their solver work.

## Questions

1. Is a project a case folder (`cases/<name>/`, files on disk) or a session object that is saved on demand?
2. A stop file in the solver for the Run module, or the console interrupt?
3. The order above, and which modules are in the first version of the shell.
4. Embedded rendering for 3-D (a GPU-backed viewer) later, or slices and the 2-D picture only?
