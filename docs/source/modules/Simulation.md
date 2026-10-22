---
title: Simulation - the run driver
nav: Simulation
section: Modules
order: 100
header: Simulation.hpp
---

`Simulation` owns every solver object of a run and drives it:

```text
setup(params)        mesh, EOS, BCs, Riemann solver, MUSCL, Navier-Stokes, body force,
                     IBM, time scheme, AMR leaf mesh and engine, exporter, test case
initialize_field()   test-case initializer or hot start -> IBM forcing -> case BCs
                     -> AMR initial adaptation -> checkpoint (when resuming)
solve()              serial time loop                run_mpi(comm)   MPI time loop (uniform)
solve_amr()          AMR time loop, serial / OpenMP / MPI
```

`fenice` (`cpp/apps/main.cpp`) resolves the parameters and calls these in order.
`initial_field()` runs the case initializer. On a 3-D mesh, a 2-D case is extruded
unchanged along z; the 3-D Taylor-Green case requires `nz > 1`. If MPI
is requested it first re-launches itself under `mpiexec`; every rank then calls `run_mpi`.

## Spatial residual

`assemble` builds the right-hand side of one (sub)domain, used for the serial domain and
for each MPI strip alike:

$$\mathcal R = \mathcal R_{\rm conv} + \mathcal R_{\rm visc} + \mathbf S$$

`assemble` handles 2-D meshes; `assemble_3d` is the same pipeline for `nz > 1`, with z-faces,
z halos and the 3-D viscous residual.

1. **IBM forcing.** Zero the velocity in solid cells (every stage).
2. **Face states.** Interior faces take the adjacent cells; boundary faces take the BC
   ghost. Under MPI, interface faces take the neighbour halo.
3. **MUSCL** reconstruction (optional).
4. **Riemann fluxes** $\mathbf F^*_f$, and the convective divergence
   $\mathcal R_{\rm conv,i} = -\frac1{V_i}\sum_f \mathbf F^*_f\,A_f$.
5. **Viscous residual** (optional) from the padded state and cell gradients. It returns the
   speed $\max_d|u_d| + c + h/\Delta t_{\rm visc}$ of the worst cell, where $\Delta t_{\rm visc}$ is the
   parabolic limit and $h$ = `dt_length` is the length the time loop uses ($\min(\Delta x,\Delta y)$
   in 2-D, $2/\sum_d \Delta x_d^{-1}$ in 3-D); $S_{\max}$ is the larger of this and the convective
   wave speed, so the convective and viscous stability limits are used together (see
   [TimeScheme]({{root}}modules/TimeScheme.html)).
6. **Body force** (optional).

The function returns $S_{\max}$; under MPI it is all-reduced.

The AMR residual is the face-list engine of the [AMR]({{root}}modules/AMR.html) page: the same physics
(convective, viscous, IBM, body force, sources) through the same expressions, evaluated per face.

## Time loops

* **Serial.** Step with the chosen scheme, apply the IBM solid reset, and write a
  snapshot and checkpoint every `save_interval` steps (and a monitor line every
  `monitor_interval` steps). A final snapshot, the checkpoint
  and the `.pvd` index are written at the end.
* **AMR.** The coarse $\Delta t$ comes from the previous step's $S_{\max}$ (a one-step
  lag, bootstrapped from all leaves at $t = 0$). The recursion of `CellAdvance` subcycles the
  hierarchy, and every `amr_refine_every` steps the mesh is remeshed (indicator, flags, adaptation,
  face-list rebuild). Under MPI every rank advances its owned leaves and exchanges halos.
* **MPI.** Rank 0 builds the initial field and scatters it. Every rank advances its strip
  with halo exchanges inside each residual. Snapshots and checkpoints are gathered to
  rank 0. A Ctrl+C on any rank is agreed by all-reduce, so no rank is left waiting in a
  collective.

The progress line reports step, time, $\Delta t$, $S_{\max}$ (and the BDF2 inner
iterations or the number of AMR leaves) and the wall time. The final line gives the
time-loop duration.

{{api}}
