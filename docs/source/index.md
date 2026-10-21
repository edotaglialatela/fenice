---
title: Fenice
nav: Overview
section: Guide
order: 0
---

**Fenice** (Italian for *phoenix*) is a structured-Cartesian compressible finite-volume
solver for the **Euler and Navier-Stokes equations** in 2-D and 3-D, written in C++17 with OpenMP threading and MPI
domain decomposition. A run is 2-D when `nz = 1` (the default) and 3-D when `nz > 1`.

* **Spatial schemes:** HLLC, Low-Mach HLLC (Thornber) and preconditioned HLLC
  (Weiss-Smith) Riemann solvers, with 2nd-order MUSCL and four TVD limiters.
* **Physics:** ideal and stiffened-gas EOS, Navier-Stokes viscous fluxes, body
  forces, and an immersed boundary method for cylinders, spheres and airfoils (NACA
  4-digit or any coordinate file), with lift / drag / moment monitoring.
* **Time integration:** Explicit Euler, SSP-RK2, RK4, and BDF2 with dual time stepping.
* **Mesh:** uniform 2-D or 3-D Cartesian grids. Block-structured AMR (2-D only) adds
  subcycling, refluxing and 2:1 balance.
* **I/O:** VTK output for ParaView, plus `.npz` checkpoints with restart and
  hot-start mapping between resolutions.
* **GUI:** a PyQt6 setup dialog that configures and launches the solver.

## Quick start

```powershell
.\cpp\build.ps1 -Test                                   # build + unit tests
build\release\fenice.exe --case "Sod Shock Tube"          # run a benchmark
python gui\run.py                                       # or configure it in the GUI
```

Results go to `results/<case>/` as `.vtr` snapshots plus a `.pvd` time
series. Open the `.pvd` in ParaView. See
[Getting started]({{root}}guide/getting-started.html) for the build options,
the command line, MPI and output files, and [Parameters]({{root}}guide/parameters.html)
for every run setting.

## Governing equations

The solver advances the conservative state
$\mathbf{U} = [\rho,\ \rho u,\ \rho v,\ \rho E,\ \rho w]^\top$ of

$$\frac{\partial \mathbf{U}}{\partial t} + \nabla\cdot\mathbf{F}_c(\mathbf{U})
  = \nabla\cdot\mathbf{F}_v(\mathbf{U}, \nabla\mathbf{U}) + \mathbf{S}(\mathbf{U})$$

with the inviscid flux $\mathbf{F}_c$, the viscous flux $\mathbf{F}_v$ (Navier-Stokes only) and
volumetric sources $\mathbf{S}$ (body forces). The cell-averaged semi-discrete form

$$\frac{\mathrm{d}\mathbf{U}_i}{\mathrm{d}t} = \mathcal{R}_i(\mathbf{U})
  = -\frac{1}{V_i}\sum_{f\in\partial i} \mathbf{F}^*_f\,A_f
  + \mathcal{R}^{\rm visc}_i + \mathbf{S}_i$$

is integrated in time with an explicit or dual-time scheme. Here $\mathbf{F}^*_f$ is the
Riemann-solver flux through face $f$ of area $A_f$, and $V_i$ is the cell volume (in 2-D,
the face length and cell area). The z-momentum $\rho w$ is stored last so that a 2-D state
keeps its original layout; in 2-D it stays exactly zero.

## How a time step works

1. **Ghost cells** ([Boundaries]({{root}}modules/Boundaries.html)). Each boundary
   face receives a ghost state from its BC. Under MPI, faces on a rank interface
   take the neighbour's halo instead. AMR patch faces take the neighbouring
   leaf's state.
2. **Reconstruction** ([Schemes]({{root}}modules/Schemes.html)). Optional MUSCL
   reconstruction of the primitive variables lifts the face states to 2nd order.
3. **Riemann fluxes**, then the flux divergence per cell.
4. **Viscous residual and sources** ([Physics]({{root}}modules/Physics.html)).
5. **Time update** ([TimeScheme]({{root}}modules/TimeScheme.html)), with
   $\Delta t = \mathrm{CFL}\,h/S_{\max}$, with $h = \min(\Delta x,\Delta y)$ in 2-D and
   $h = 2/(1/\Delta x + 1/\Delta y + 1/\Delta z)$ in 3-D. The 3-D length keeps the sum of
   the three directional Courant numbers at the 2-D bound ($2\,\mathrm{CFL}$). The IBM
   solid reset follows each complete step.
6. **Output.** VTK snapshots and checkpoints every `save_interval` steps
   ([IO]({{root}}modules/IO.html)).

The orchestration lives in [Simulation]({{root}}modules/Simulation.html).

## Data conventions

* **State arrays** are `Field = std::vector<State>`, one `[rho, rho*u, rho*v, rho*E, rho*w]`
  per cell (`kNVar = 5`; the momentum slots are `kMom = {1, 2, 4}`).
* **Cells** are row-major: `cell = (k*ny + j)*nx + i` (z outermost, x innermost). In 2-D
  (`k = 0`) this is `j*nx + i`.
* **Faces**: the x-normal faces come first, `(k*ny + j)*(nx+1) + i`, then the y-normal faces,
  `num_xfaces + (k*(ny+1) + j)*nx + i`, then (3-D only) the z-normal faces,
  `num_xfaces + num_yfaces + (k*ny + j)*nx + i`.
* **Primitive variables** are `W = [rho, u, v, p, w]` (`EOS::cons_to_prim` / `EOS::prim_to_cons`).

## Repository layout

| Path | Content |
|---|---|
| `cpp/include/fv/`, `cpp/src/` | Solver library (one header per module page) |
| `cpp/apps/main.cpp` | `fenice` command-line driver |
| `cpp/tests/` | Unit tests (`fenice_tests`), regression references and `regression.py` |
| `cases/` | Test cases, one folder with a `case.toml` each ([Writing a test case]({{root}}guide/writing-cases.html)) |
| `cpp/third_party/` | toml++ and TinyExpr (case files) |
| `gui/` | PyQt6 setup dialog and launcher |
| `python/fenice/` | Python package shared by the GUI, validation and docs tools: finds and runs the solver, reads its output |
| `cases/validation/` | One script per validation study, into `results/validation/<study>/` ([Testing]({{root}}guide/testing.html)) |
| `docs/` | This documentation: `docs/source/` and `docs/img/`, built into `docs/site/` by `python docs/build.py` |
| `build/` | Builds: `build/release/`, `build/debug/` |
| `results/` | Simulation output (not in git) |
