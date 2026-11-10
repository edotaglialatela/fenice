# Product

<!-- impeccable:product-schema 1 -->

## Users

CFD researchers and the solver's developer. They understand compressible Euler and
Navier-Stokes physics, Riemann solvers, MUSCL, immersed boundaries, AMR, and MPI.
They configure, launch, monitor, and inspect simulations from a desktop workstation,
often before a long compute job. The interface should help them specify a run
correctly and see what is happening without duplicating solver configuration by hand.

## Product Purpose

Fenice is a C++17 finite-volume solver for compressible Euler and Navier-Stokes
equations on structured Cartesian grids in 2-D and 3-D. Its command-line interface
and PyQt6 GUI let users configure and run cases, inspect live progress, and view
results. Success is a physically and numerically correct run that starts without a
configuration surprise and produces usable output for analysis.

## Positioning

Fenice's PyQt6 workbench is a thin client of the solver: it reads the solver's
available cases and defaults, so a new case appears without a separate GUI form.
Its graph connects geometry, solver setup and runs, live monitors, and result viewing
in one workflow. This links the GUI to the same case definitions and parameters used
by the executable.

## Operating Context

- Runs are configured and started on a desktop workstation; OpenMP uses local
  threads, and MPI can distribute runs across ranks.
- Users can work through the command line, the single-window setup dialog, or the
  modular workbench. The workbench supports Geometry, Solver, Monitor,
  Postprocessing, and Analysis Tools modules; one Geometry node can feed multiple
  Solvers, and one Solver can feed both results modules.
- Cases live under `cases/`, one `case.toml` per case. Solver output includes VTK
  time series and checkpoint files; users can inspect VTK output in ParaView and
  browse or analyze saved results in the GUI.

## Capabilities and Constraints

- Numerics include HLLC variants, first-order, MUSCL, WENO3-Z and WENO5-Z
  reconstruction, and explicit Euler or SSP Runge-Kutta time integration.
- Physics includes ideal and stiffened gases, viscous and heat fluxes, body forces,
  source terms, immersed boundaries, and boundary conditions for walls, inlets,
  outlets, and periodic domains.
- Adaptive meshes use a quadtree in 2-D and an octree in 3-D. OpenMP is available;
  MPI is optional at build time. Runs are intended to remain bitwise consistent
  across thread and rank counts.
- The solver is built with CMake. The GUI is a PyQt6 desktop application and needs a
  built solver executable. The repository documents Windows, Linux, and macOS
  development environments.
- The separate, general-purpose Analysis Tools module supports field profiles and
  spectra, monitor signal statistics and frequency analysis, snapshot POD / DMD,
  monitor time-delay DMD, Sobol indices, and sampled input-response plots.
- Workbench state can be saved to and restored from a `.fenice.json` file, including
  the graph, module settings, result selections, and workspace layout. Restore
  recreates solver projects idle; it does not resume an in-progress process.
- Geometry settings are shared across all Solvers connected to the same Geometry
  node. Changing a linked Solver's case reloads that shared geometry.

## Brand Commitments

The product name is **Fenice**. Its established voice is technical, precise, calm,
and unembellished, with exact units and terminology and no marketing tone. The GUI
logo is `gui/logo.png`.

## Evidence on Hand

- Built-in case definitions are in `cases/`.
- The repository includes 75 stored solver reference states in
  `cpp/tests/reference/` and physical verification and benchmark scripts in
  `cases/validation/`.
- User-facing documentation is in `docs/`; the workbench screenshot is
  `docs/img/gui_workbench.png`.
- No customer testimonials, press coverage, or external certifications are present
  in the repository. Do not invent them.

## Product Principles

1. Catch invalid values and incompatible configurations before a long solver run.
2. Keep solver case definitions authoritative; the GUI should reflect them rather
   than maintain duplicate case forms.
3. Make linked geometry and solver settings consistent with what each run will use.
4. Use exact physical terms, units, and solver terminology.
5. Preserve reproducibility across parallel configurations.

## Accessibility & Inclusion

The existing product requirement is WCAG 2.1 AA contrast for text, visible keyboard
focus, and clear state and validation cues that do not rely on color alone. The GUI
should remain legible at standard desktop DPI.
