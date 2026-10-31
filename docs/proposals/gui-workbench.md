# A modular front end: proposal

Status: the workbench is a graph (`gui/workbench.py`, `gui/canvas.py`, `gui/graph.py`): an empty canvas on which Geometry, Solver, Monitor
and Postprocessing nodes are dropped and linked (see the GUI page). The thermochemistry node and the saving of the graph are proposals.
The scope of each remaining module, and the order, are the user's.

## Aim

One application in which the steps of a study are modules that plug into a common graph: set the geometry up, run, watch the run, look
at the results, and later set up the thermochemistry. The front end stays what the GUI already is, a thin client: no physics, the
solver is the single source of truth (`fenice --list-cases`, `--print-config`, `--save-case`), the files it reads and writes are the
solver's own.

## Shape

* **Node and ports.** A module is a node with typed ports (`geometry`, `run`, `results`). A link joins an output to an input of the same
  type; an input takes one link, an output may feed several. A new kind of data is a new port type.
* **The solver owns the project.** A solver node makes its own `Project` (the parameters of a run, its results folder, its state and its
  history: log lines, progress lines, physical time). Every other node is bound to the project of the solver it is linked to: its editor
  is built when the link is made and dropped when it is removed, so a monitor linked mid-run shows the run from its start, and a
  postprocessing node linked after a run opens its results.
* **The geometry is a node, not part of the solver.** The Geometry node holds its own parameter set (domain, mesh, immersed body, side
  conditions), which can be edited unlinked. Linked, it is written into the fields of the solver page (hidden there), which stay the only
  copy of every parameter the run reads, so the solver, the dialog and the run files are unchanged. A solver does not run without a
  linked geometry.
* **Registration.** A module is a file in `gui/modules/` that defines one `MODULE` record (name, title, ports, whether it owns a project,
  an editor factory, an optional face for the node, the Python packages it needs). The workbench imports whatever it finds; a module
  whose packages are missing is listed as unavailable with the reason, so that an install without the optional pieces still runs.
  Modules from outside the repository can be added later through the same record.
* **Editors.** Double-click a node: its editor opens in a tab under the canvas; several can stay open.

## Modules

| Module | What it does | State |
| --- | --- | --- |
| Geometry | the domain, the mesh, the immersed body (cylinder, sphere, NACA section, airfoil file), the side conditions, the grid and the solid cells at the chosen resolution, the finer AMR levels; all edited here (an airfoil generator, the values of the conditions: velocity, pressure, mass flow, are not done: they are on the solver) | exists |
| Solver | the setup dialog without the geometry, Run and Stop, the Low-Mach switch, the monitor interval; one project per solver node | exists |
| Monitor | live plots (time step, wave speed, residual norms, monitor columns) and the console, linked to a solver, also in the middle of a run | exists |
| Postprocessing | the viewer: fields, slices, adaptive meshes, monitor curves, following a run | exists |
| Studies | the validation studies and the regression, with their pass or fail | proposed |
| Thermochemistry | species, mechanism and reference reactor, when the species work exists (`species-target-cases.md`) | later |
| Mesh (unstructured) | only if the unstructured solver is built (`unstructured-solver.md`) | later |

## Run: stopping

The solver node starts the solver in a console of its own. Stop writes a file named `STOP` in the solver's output folder, which the solver polls once per step and answers with its checkpoint and the `.pvd`; this works under MPI, where `mpiexec` aborts the job on Ctrl+C. Without a known output folder Stop falls back to the console interrupt, sent from a small helper process so that the workbench keeps its own console. Kill ends the solver and its processes at once, and a job object ends them if the workbench dies.

## Order

1. The result viewer, the graph, the Geometry, Solver, Monitor and Postprocessing nodes (done).
2. Saving and loading the graph; the values of the boundary conditions on the Geometry node.
3. Studies. Thermochemistry and the unstructured mesh only with their solver work.

## Questions

1. Is a project a case folder (`cases/<name>/`, files on disk) or a session object that is saved on demand?
2. The order above.
3. Embedded rendering for 3-D (a GPU-backed viewer) later, or slices and the 2-D picture only?
