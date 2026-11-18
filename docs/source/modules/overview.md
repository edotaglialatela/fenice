---
title: How Fenice works
nav: How Fenice works
section: Methods
order: 0
---

This page is the map of the solver: the equations it advances, what happens in one time step, and which Methods page
describes each stage. Read it first if you want to know what a parameter changes, or before reading one of the module pages.

## The equations

The solver advances the conservative state
$\mathbf{U} = [\rho,\ \rho u,\ \rho v,\ \rho E,\ \rho w]^\top$ of

$$\frac{\partial \mathbf{U}}{\partial t} + \nabla\cdot\mathbf{F}_c(\mathbf{U})
  = \nabla\cdot\mathbf{F}_v(\mathbf{U}, \nabla\mathbf{U}) + \mathbf{S}(\mathbf{U})$$

with the inviscid flux $\mathbf{F}_c$, the viscous flux $\mathbf{F}_v$ (Navier-Stokes only) and
volumetric sources $\mathbf{S}$ (body forces). With `use_viscous = false` the right-hand side has only the sources and the
equations are the Euler equations.

The mesh is divided into cells, and each cell holds the average of $\mathbf{U}$ over its volume. The cell-averaged
semi-discrete form is

$$\frac{\mathrm{d}\mathbf{U}_i}{\mathrm{d}t} = \mathcal{R}_i(\mathbf{U})
  = -\frac{1}{V_i}\sum_{f\in\partial i} \mathbf{F}^*_f\,A_f
  + \mathcal{R}^{\rm visc}_i + \mathbf{S}_i$$

and it is integrated in time with an explicit Runge-Kutta scheme. Here $\mathbf{F}^*_f$ is the
Riemann-solver flux through face $f$ of area $A_f$, and $V_i$ is the cell volume (in 2-D,
the face length and cell area). The finite-volume form is conservative: what leaves one cell enters its neighbour.

A run is 2-D when `nz = 1` (the default) and 3-D when `nz > 1`. The z-momentum $\rho w$ is stored last so that a 2-D state
keeps its original layout; in 2-D it stays exactly zero.

## One time step

1. **Ghost cells** ([Boundaries]({{root}}modules/Boundaries.html)). Each boundary
   face receives a ghost state from its boundary condition. Under MPI, faces on a rank interface
   take the neighbour's halo instead. AMR faces between levels use the virtual
   cells of the face list.
2. **Reconstruction** ([Schemes]({{root}}modules/Schemes.html)). Optional MUSCL or WENO
   reconstruction of the primitive variables lifts the face states to higher order.
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

## Where each choice lives

| You want to choose | Parameter | Page |
|---|---|---|
| the Riemann solver | `riemann_solver` | [Schemes]({{root}}modules/Schemes.html) |
| the reconstruction (first order, MUSCL, WENO) and the limiter | `reconstruction`, `muscl_limiter` | [Schemes]({{root}}modules/Schemes.html) |
| the time integrator and the time step | `time_scheme`, `cfl` | [TimeScheme]({{root}}modules/TimeScheme.html) |
| the boundary conditions | `bc_left`, `bc_right`, `bc_bottom`, `bc_top`, `bc_front`, `bc_back` | [Boundaries]({{root}}modules/Boundaries.html) |
| viscosity, conductivity, body forces, an immersed body | `use_viscous`, `use_bodyforce`, `use_ibm` | [Physics]({{root}}modules/Physics.html) |
| the equation of state | `eos_type`, `gamma` | [Core]({{root}}modules/Core.html) |
| adaptive refinement | `use_amr` | [AMR]({{root}}modules/AMR.html) |
| restart, hot start, snapshots | `resume`, `use_hot_start`, `save_interval` | [IO]({{root}}modules/IO.html), [SolutionMapping]({{root}}modules/SolutionMapping.html) |
| threads and ranks | `n_cpus`, `use_mpi` | [MPIDecomposition]({{root}}modules/MPIDecomposition.html), [Performance]({{root}}guide/performance.html) |

Every key, with its default, is on the [Parameters]({{root}}guide/parameters.html) page.

## Data conventions

* **State arrays** are `Field = std::vector<State>`, one `[rho, rho*u, rho*v, rho*E, rho*w]`
  per cell (`kNVar = 5`; the momentum slots are `kMom = {1, 2, 4}`).
* **Cells** are row-major: `cell = (k*ny + j)*nx + i` (z outermost, x innermost). In 2-D
  (`k = 0`) this is `j*nx + i`.
* **Faces**: the x-normal faces come first, `(k*ny + j)*(nx+1) + i`, then the y-normal faces,
  `num_xfaces + (k*(ny+1) + j)*nx + i`, then (3-D only) the z-normal faces,
  `num_xfaces + num_yfaces + (k*ny + j)*nx + i`.
* **Primitive variables** are `W = [rho, u, v, p, w]` (`EOS::cons_to_prim` / `EOS::prim_to_cons`).
