---
title: From Python to C++
nav: Python → C++
section: Guide
order: 60
---

Until 2026-10-20 the solver was written in Python (NumPy, Numba JIT above 25 000 cells,
mpi4py). It was then ported to C++17 module by module and checked against the Python
implementation. The Python solver was retired once the port matched it. A backup of the
Python sources, tests and previous documentation is kept outside the repository at
`Desktop\Uni\python_solver_backup_2026-10-20.zip`.

## Equivalence

On 16 configurations exercising every scheme and feature, the serial C++ solver
reproduced the Python final states **bit for bit**. The two exceptions, at ~1e-14, are
BDF2 and the mass-flow inlet, whose sums are associated differently. AMR runs matched
leaf by leaf. Those states are now the [regression references]({{root}}guide/testing.html).
Checkpoints stay compatible both ways for hot start, and C++ resumes Python checkpoints.

## Deliberate changes

**MPI reproduces the serial solution.** The Python MPI path differed from serial at
rank interfaces, so its results depended on the rank count: about 1e-2 relative on a
short viscous Poiseuille run with 4 ranks. Fixed:

* Viscous cross-derivatives at an interface used a zero-gradient copy of the rank's own
  cell instead of the neighbour's value.
* Halos were exchanged before IBM forcing, and the IBM fringe test could not see
  neighbours across an interface.
* Mass-flow inlets averaged density per rank instead of over the whole inlet.
* The local $\Delta y = L_{y,\rm local}/n_{y,\rm local}$ could differ from the global
  value by one ulp.
* IBM masks were computed from local coordinates. In X-decomposition this also placed
  the cylinder at the wrong x.

**Decomposition direction.** Python chose row strips whenever $n_y \ge N$, allowing
1-row strips whose 2-deep halo exchange read uninitialised memory. The C++ version
requires at least two cells per strip.

**Periodic seams (fixed after the port).** A new translation-invariance test showed
that MUSCL left periodic boundary faces 1st order, and that the viscous cross-derivatives
there used a zero-gradient copy of the cell. Both now wrap around: shifted periodic
solutions agree bit for bit, and the Taylor-Green viscosity error dropped from 2.2 % to
0.4 %. No regression reference changed, because the only periodic reference case is
uniform in the periodic direction.

**AMR boundary conditions.** Patches on the domain edge use the case's forced BCs, as the
uniform-mesh path does, rather than the raw GUI values. This is identical for every
built-in case's defaults.

**3-D extension (2026-10-20).** The solver now runs 3-D meshes (`nz > 1`):

* The state gained a fifth component, $\rho w$, stored last so the 2-D layout is unchanged.
* Riemann solvers work per face axis.
* There are front/back BCs, 3-D MUSCL, gradients and viscous stresses, $g_z$, and a 3-D IBM
  (sphere or z-cylinder).
* MPI uses slabs; checkpoints, hot start and VTK output handle 3-D.

AMR stays 2-D. All 2-D results are unchanged: every regression reference still matches,
serial and MPI. Two 3-D-only choices: the CFL length (see
[TimeScheme]({{root}}modules/TimeScheme.html)) and the 3-D parabolic viscous limit
$1/(2\nu\sum_d\Delta x_d^{-2})$. Both keep the stability margins of the 2-D formulas. See the
[3-D Taylor-Green case]({{root}}cases/taylor-green-3d.html).

**Smaller changes.**

* Limiter names are case-insensitive.
* The checkpoint stores the `.pvd` index as plain arrays instead of a pickled list.
* The GUI reads test cases and defaults from the solver.
* `max_steps`, `dump_state`, `output_root`, `monitor_interval` and `--print-case-defaults` are new.

**Unchanged on purpose:**

* The AMR path is convective-only.
* Multi-stage schemes keep an $O(\Delta t^2)$ coarse-fine conservation term under AMR.
* The IBM solid reset uses the ideal-gas pressure formula whatever the EOS.

<figure>
<img src="{{root}}img/poiseuille_mpi_comparison.png" style="width:100%">
<figcaption>After the fixes: serial vs 4-rank MPI on the Poiseuille case, maximum difference 0.</figcaption>
</figure>
