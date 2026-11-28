---
title: Fenice
nav: Overview
section: Start here
order: 0
layout: home
tagline: A finite-volume solver for compressible flow. Euler and Navier-Stokes in 2-D and 3-D on Cartesian grids, with adaptive refinement and bodies immersed in the mesh.
---

**Fenice** (Italian for *phoenix*) simulates compressible gas flow: shock tubes, blast waves, flow past a cylinder, a wing or
a sphere, turbulence in a box. You describe a problem in a small text file, run one command, and open the result in
ParaView. The solver is written in C++17 and uses OpenMP and MPI; the result does not depend on how many threads or ranks
you use.

## Try it in three commands

```powershell
.\cpp\build.ps1 -Test                                   # build the solver and run its unit tests
build\release\fenice.exe --case "Sod Shock Tube"        # run a benchmark
python gui\run.py                                       # or set up and start a run in a window
```

Results go to `results/<case>/`: open the `<case>.pvd` file in ParaView. The
[Getting started]({{root}}guide/getting-started.html) page walks through the first run step by step.

## Results

{{showcase}}

Every benchmark, with the command that reproduces it, is in the [gallery]({{root}}cases/index.html).

## I want to...

| Goal | Go to |
|---|---|
| install Fenice and run a first case | [Getting started]({{root}}guide/getting-started.html) |
| change a setting of a run (mesh, scheme, boundaries, time) | [Parameters]({{root}}guide/parameters.html) |
| set up my own problem | [Writing a test case]({{root}}guide/writing-cases.html) |
| use the graphical front end | [GUI]({{root}}guide/gui.html) |
| make a run faster, or use more cores | [Performance]({{root}}guide/performance.html), [Switches]({{root}}guide/switches.html) |
| choose a numerical scheme and understand what it does | [How Fenice works]({{root}}modules/overview.html) and the Methods pages |
| see a worked example with results | the [gallery of test cases]({{root}}cases/index.html) |
| judge how far the results can be trusted | [Testing]({{root}}guide/testing.html), [Validation]({{root}}guide/validation.html) |
| read or extend the C++ code | [API reference]({{root}}api/index.html) |

## What it can do

* **Equations.** Euler and Navier-Stokes, with ideal-gas or stiffened-gas equations of state, body forces and user-defined
  source terms (`[source]` expressions of x, y, z and t).
* **Numerical schemes.** HLLC and low-Mach HLLC Riemann solvers; first order, MUSCL (four TVD limiters) or WENO3-Z / WENO5-Z
  reconstruction; explicit Euler, SSP-RK2 and SSP-RK3 in time. See [Schemes]({{root}}modules/Schemes.html) and
  [TimeScheme]({{root}}modules/TimeScheme.html).
* **Bodies in the flow.** A cylinder, a sphere or an airfoil (polygon) placed on the Cartesian grid with a sharp
  ghost-cell wall; lift, drag and surface loads are monitored. See [Physics]({{root}}modules/Physics.html).
* **Adaptive meshes.** Cell-based refinement (a quadtree in 2-D, an octree in 3-D) with subcycling in time, also on many MPI
  ranks. See [AMR]({{root}}modules/AMR.html).
* **Parallel and reproducible.** OpenMP threads and MPI ranks give the same bits as a serial run; the regression
  suite checks it. See [Testing]({{root}}guide/testing.html).
* **Output and restart.** VTK snapshots for ParaView, force and norm monitors, checkpoints that resume a run bit for
  bit, and hot start from a run on another mesh. See [IO]({{root}}modules/IO.html).
* **Tools.** A PyQt6 workbench with a post-processing viewer and analysis tools (spectra, modal decomposition, sensitivity), and
  a library of verification and validation studies. See [GUI]({{root}}guide/gui.html) and
  [Validation]({{root}}guide/validation.html).
