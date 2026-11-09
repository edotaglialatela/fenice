---
title: AMR - adaptive mesh refinement on leaf cells
nav: AMR
section: Modules
order: 60
header: CellMesh.hpp
---

The adaptive mesh is a set of **leaf cells**, each a cell of the level-0 mesh split $\ell$
times by two in each direction (size $h_\ell = h_0/2^\ell$). There are no patches: the base
mesh is the `nx × ny` (`× nz`) grid of the case (any sizes), and a single cell is refined or
joined at a time. Leaves are stored in flat arrays sorted by level and Morton (Z-order) key
within the level, so cells that are close in space are close in memory and in the MPI
partition. The same engine runs in 2-D (a quadtree, `nz = 1`) and in 3-D (an octree,
`nz > 1`); the drawings below are 2-D.

```text
Level 0   ┌───────┬───────┬───────┐
          │       │ ┌─┬─┐ │       │
          │       │ ├─┼─┤ │       │      2:1 balance: leaves that touch
          ├───────┼─┴─┴─┴─┼───────┤      (also at a corner) differ by
          │       │       │       │      at most one level
          └───────┴───────┴───────┘
```

## One formulation for $d = 2$ and $d = 3$

The dimension $d$ is a run-time value of the engine, not a second implementation. Everything above and below is written for a leaf
with $2d$ sides and holds for both; what changes is only a count:

| | quadtree ($d = 2$) | octree ($d = 3$) |
|---|---|---|
| children of a refined leaf | 4 | 8 (joined again only as a complete set of 8 siblings) |
| Morton key | interleaved bits of $(i, j)$ | interleaved bits of $(i, j, k)$ |
| sides of a leaf | 4 | 6 (periodic in $z$ by default) |
| 2:1 balance | across sides and corners (8 neighbours) | across faces, edges and corners (26 neighbours) |
| fine faces on a coarse-fine face | 2 | 4 |
| mean over a split cell (virtual cell, indicator) | of 4 children | of 8 children |
| time-step length | $\min(\Delta x, \Delta y)$ | $2 / (1/\Delta x + 1/\Delta y + 1/\Delta z)$, as in the uniform 3-D solver |
| state per leaf | $\rho, \rho u, \rho v, \rho E$ | five variables, with $\rho w$ |

The face fluxes, the virtual cells, the jump closure, the subcycling with flux registers and the indicator are the same expressions
in every dimension; the level-0 octree reproduces the uniform 3-D solver bit for bit, as the level-0 quadtree does the 2-D one. A 3-D
adaptive run is bit-identical for any number of OpenMP threads and of MPI ranks, and the remesh, the partition and the checkpoint
work as described below.

The immersed body is classified **per leaf** in 3-D (there is no uniform mask over the finest level, which would be as large as the
uniform mesh): a leaf is solid if its centre is inside the body, and the body band is kept at the finest level wherever a finest
cell lies within the margin of the surface (`near_body_surface`). The mask forcing and the ghost-cell wall (`ibm_method = ghost`) work
on octree leaves, also on the shares of an MPI run (bitwise equal to serial); `auto` chooses the ghost wall wherever `amr_refine_ibm` is on.

Not yet available in 3-D: the body-load rows (`[[force]]`; a notice is printed) and WENO5 (as in 2-D AMR). Cost of the octree against the uniform solver on a sphere: 0.2-0.3 of the time loop at the same finest resolution, drag within 1 %; MPI is efficient at 2-3 ranks only (the remesh is the limit). Design, phases and
measurements: `docs/design/amr-octree.md`.

## Faces and the residual

The solver does not sweep cells; it works on a **face list** (`CellFluxEngine`). Every face
is one of

* an **interior** face between two leaves of the same level,
* a **coarse-fine** face between a leaf and the finer leaves on its other side (one face
  per fine leaf),
* a **boundary** face on the edge of the domain.

A face computes its flux once (HLLC in any of its variants, MUSCL reconstruction from the
same limiters as the uniform solver, the viscous flux, the immersed-body and boundary
treatment) and adds it, with opposite signs, to the two cells it separates. Conservation
therefore follows from the data layout, not from a correction. The MUSCL stencil of a face
takes the cell on each side and one more beyond it. Where a stencil point falls on a
different level it is a **virtual cell**: the mean of the finer leaves under it, or an
interpolation of the coarser leaf. The level-0 mesh reproduces the uniform solver bit for
bit (the same expressions in the same order), so refinement is switched on without a second
implementation of the physics. Periodic axes are supported at level 0 and at every level
(the stencil wraps), the other boundary conditions use the same ghost states as the uniform
solver.

### Jump closure

At a coarse-fine face the fine side is evaluated with the fine leaf's own (second order)
reconstruction against the interpolated coarse state, and the coarse side with a second-order
reconstruction from the virtual cells under it (`interpolate_jumps`, `coarse_side_second_order`,
both on). With the unreconstructed coarse state the scheme loses an order of accuracy
across every level jump (the manufactured-solution test would not reach the bound on the
adaptive mesh); this closure keeps it.

## Subcycling and flux replacement

A level $\ell$ advances with $\Delta t_\ell = \Delta t_0/2^\ell$, the finer level first:

```text
advance(L, dt):
    if a finer level exists: advance(L+1, dt/2); advance(L+1, dt/2)
    advance every leaf of level L by dt  (the time scheme's stages, other levels frozen)
```

Time is not interpolated: the coarse leaves are held at their value at the start of the
coarse step while the fine levels run. At a coarse-fine face the fine side accumulates its
flux over the sub-steps with the time scheme's stage weights; when the coarse level advances
it **replaces** the flux it would have used by that accumulated one (the registers), so the
coarse cell and the fine cells exchange exactly the same amount. Mass, momentum and energy are
conserved to roundoff with every time scheme, including the multi-stage ones.

## Error indicator

The indicator looks at the variation of the solution across faces, level by level (`CellIndicator`).
On the cells of a level (the leaves, plus the split ancestors of the deeper leaves, each holding the
mean of its children):

1. $\xi = 0$ everywhere.
2. For every face and every variable of `amr_indicator` (density, pressure, speed) with values
   $c_l$, $c_r$ on the two sides, $v = |c_r - c_l| / \max(\min(|c_l|, |c_r|), \text{floor})$.
   If $v \ge$ `amr_criterion`, both cells get $\xi = 1$.
3. `amr_diffusion` times: across every face where one cell has $\xi > 0.05$, add 0.1 to the cells on
   both sides. The refined band thus extends a few cells beyond the feature, so it holds a moving feature
   between two remeshes.
4. A leaf is **refined** when $\xi \ge$ `amr_refine_threshold` (below `amr_max_level`). Four sibling
   leaves are **joined** when the $\xi$ of their **parent** cell, evaluated at the parent's own level,
   is below `amr_derefine_threshold`. Judging the parent rather than the four children is what keeps a
   marginal gradient from flickering: a refined child sees half the variation per face, and would
   fall under the threshold the moment it is refined.

Refinement ripples: a leaf that must be refined next to a coarser one first refines that neighbour
(2:1 balance, including diagonal neighbours). A relative variation floor (0.01 for density and pressure,
0.1 for the speed) keeps the indicator quiet where the variable is near zero. Use `density+velocity` for a
wake, where the density barely varies, and add `pressure` for shocks in a flow with little density change.

* **Initial condition.** At $t = 0$ each leaf evaluates the case's initial condition at its own cell
  centre, so a refined leaf sees the exact discontinuity. `amr_init_levels` first refines uniformly; the
  initial adaptation then repeats indicator and remesh until nothing changes. A hot start is
  interpolated at the leaf centres.
* **Immersed bodies.** With `use_ibm` and `amr_refine_ibm` (default) every leaf whose footprint touches the
  body, or the cells around it that the solid reset treats (the fringe), is kept at the finest level from
  $t = 0$ and never joined. With `ibm_method = ghost` the band is widened to the ghost cells' interpolation stencils (about 6
  finest cells), so the sharp wall works on the finest level. The test is on the finest-level mask, so a thin trailing edge cannot slip
  between cell centres.
* **Remeshing** happens every `amr_refine_every` coarse steps. A remesh that changes no leaf costs only the
  indicator. After one that does, the face list, neighbour codes, stencils and virtual cells are updated, not rebuilt: a leaf
  is recomputed if it is new or a square of its size across one of its sides holds a new leaf, everything else is copied with
  the indices renumbered (virtual cells are numbered by key, so the numbering depends on the mesh only). The result is the
  same as a full rebuild, table for table (`FENICE_FULL_REBUILD=1` forces the full one).

## Physics on the adaptive mesh

Viscous terms, the body force, the immersed-boundary forcing, manufactured-solution sources
(steady and time dependent) and every boundary condition run on the
face list. The immersed boundary forcing acts in every residual, the solid reset once per complete step.
A closed viscous box conserves energy to roundoff, and the airfoil study below uses the full
Navier-Stokes + IBM path.

<figure style="margin:1em 0;">
<img src="{{root}}img/sod_amr.png" style="width:100%">
<figcaption>Sod shock tube, base 10×2 and 4 levels, remeshed every 2 steps, against uniform meshes at the
two finest levels (<code>cases/validation/amr/sod_amr.py</code>): the error of the 160×32 mesh with
45 % of its cells.</figcaption>
</figure>

## Output, checkpoints, restart

Each snapshot is one `<case>_<step>.vtu` (an unstructured grid with one quad per leaf, or one hexahedron per leaf of a 3-D octree;
cell fields Density, Pressure, Temperature_nondim, Mach, Velocity and `AMR_Level`). Under MPI each rank writes the `.vtu` of
its owned leaves and rank 0 a `<case>_<step>.pvtu` that lists them; the `.pvd` series points at the
`.vtu` or `.pvtu`. `python/fenice` reads both with `fenice.read_vtu`.

The checkpoint `<case>_restart.npz` holds the leaf list (level, `i`, `j`, and `k` in 3-D), the state of every leaf (five columns in
3-D, with $\rho w$), `t`, the step number and the lagged wave-speed bound, so a restarted run continues the same trajectory bit for
bit, also on a different number of ranks. A checkpoint resumes only on the base mesh it was written for. The final state
(`dump_state`) is the per-leaf `U` plus `leaf_meta` with columns `x0, x1, y0, y1, level` (`x0, x1, y0, y1, z0, z1, level` in 3-D).
A hot start from a uniform or an adaptive checkpoint (of the same dimension) interpolates the source at the leaf centres, bilinearly
in 2-D and trilinearly in 3-D. Probes, sample lines and time averages work on the leaves in 3-D too; the body loads (`[[force]]`)
are written for a 2-D adaptive run only.

## Parallel runs

OpenMP threads share the face loops of a level. With `use_mpi` each level is cut into contiguous blocks of equal leaf count along
its Morton order and a rank advances its owned leaves. A rank keeps only its owned leaves and the halo, the leaves its faces read
(stencil points, virtual cells, gradients): its share of the mesh, faces, stencils and state, with the same numbers a whole-mesh
build gives for them (a face that read a leaf outside the share would be an error, checked when the share is made). The remesh
does not gather the state: each rank takes the indicator of its owned leaves from the leaves around them (the ones it does not hold
are fetched from their owners), the refine / coarsen flags of all ranks are gathered (one number per leaf), every rank applies
them to the topology of the whole mesh (leaves and faces, no state; the ripple and the merges are the same on every rank), builds
the state of its new leaves from the old leaves they come from, and rebuilds its share
(`FENICE_REPLICATED_REMESH=1` keeps the whole mesh and state on every rank and remeshes it there instead). The owners exchange the halo after each stage, the final state of a level
after it advanced, and the registers of the coarse-fine faces. A same-level face is computed by both owners with the same inputs, a coarse-fine face
by the owner of the fine leaf. The result is identical to the serial run for any number of ranks (1 to 7 are checked in
`amr_mpi.py --check`, on cases with remeshing, an immersed body, viscosity, a non-reflecting outlet and a time-dependent source).

**Known limits.** The indicator, the flags and the state of the new leaves are local, each rank builds its share of the mesh and the engine tables directly from the leaves near its own (a wider reach is tried if a face reads what the share lacks), and the first mesh is adapted the same way, so no rank builds the engine of the whole mesh or adapts a mesh with faces (the base mesh itself is constructed once at setup, and every rank keeps the leaves and lookups of the whole topology). No rank holds the topology of the whole mesh: the ranks fetch the leaves around their own from the owners, settle the ripple and the merges between them and migrate the new leaves; same result bit for bit). The memory per rank at the end of the run falls to 121 / 76 / 50 / 39 MB on 2 / 4 / 8 / 16 ranks (peak 213 / 145 / 109 / 93 MB), but the wall time rises to 6.0 / 4.1 / 4.2 / 5.7 s on this mesh (the fetches; on the 1.06 M-leaf airfoil 1.5 x on 4 ranks and 2 x on 16, with 40-64 % less memory at the end). An earlier remesh in which every rank adapted the topology of the whole mesh (leaves and lookups only, no faces) was 1.5-2 x faster on these meshes and kept that topology on every rank; it was removed, and fetching only around the leaves on the border of a rank's region is the way to recover part of the time. The resident memory per rank on the airfoil is 257 MB serial and 171 / 121 / 95 / 83 MB at the end of the run on 2 / 4 / 8 / 16 ranks, the peak (the remesh and the first share) 348 MB serial and 227 / 158 / 128 / 116 MB. A checkpoint is gathered on rank 0 only, which holds the whole state for the file: with one every 10 steps the peak of rank 0 is 176 / 137 MB on 4 / 16 ranks, the others stay at 159 / 115 MB. On one node
OpenMP is the better choice; MPI gives the same result for any rank count and a moderate speed-up. Checkpoint and `dump_state` gather the
leaves on rank 0 (a transient copy of the whole state); a restart cuts each rank's share from the leaf list of the checkpoint (three integer arrays) and reads only the state rows of its own leaves and their halo (`NpzFile`), so no rank holds the whole state or mesh; a hot start reads the whole source field.
At the end of a run the solver prints the resident memory of the ranks (after setup, after the initial adaptation, at the end,
the peak: minimum / maximum over the ranks) and the size of the tables of a rank: about 1.5 kB per leaf on the airfoil
(`cases/validation/amr/amr_memory.py` measures it on 1 to 16 ranks; `docs/design/amr-distributed.md` is the plan to distribute
the remesh as well).

The table below compares AMR with itself on 1 to 16 cores; it says nothing about AMR against a uniform mesh (that comparison is
at equal accuracy, time to a given error, see the [airfoil]({{root}}cases/airfoil.html) page). It does show that AMR parallelises
worse than the uniform solver (about 2.9× on 16 threads, against 4 to 5× for the uniform cases): the remesh scales poorly, the
small levels fall below the OpenMP threshold of 4096 items per loop, and 50 steps are short (about 1 s of the 5.9 s serial run is
the remesh and set-up, a share that a long run amortises, so the table probably understates the efficiency of AMR). More cores
therefore shrink the AMR advantage slightly, by that efficiency gap, but the cell reduction is much larger and wins at 16 threads too.

| run (720×360, 2 levels, 50 steps, 268 362 leaves) | wall time | of which remesh |
|---|---:|---:|
| serial, 1 thread | 5.9 s | 1.0 s |
| OpenMP, 2 / 4 / 8 / 16 threads | 3.5 / 2.3 / 2.1 / 2.0 s | 0.75 / 0.65 / 0.65 / 0.74 s |
| MPI, 2 / 4 / 8 / 16 ranks, each rank its share of the mesh (default) | 4.1 / 2.6 / 2.3 / 2.6 s | 1.8 / 1.2 / 1.2 / 1.5 s |
| MPI, 2 / 4 / 8 / 16 ranks, `FENICE_REPLICATED_REMESH=1` (every rank the whole mesh) | 3.9 / 3.1 / 3.8 / 7.5 s | 1.1 / 1.3 / 2.0 / 4.4 s |

Ryzen 7 7800X3D (8 cores, 16 logical processors), MSVC build, one run after the incremental remesh update (scalar viscous loops); the 16-rank row is
oversubscribed. Before it: serial 6.1 s (remesh 1.3 s), OpenMP 3.6 / 2.5 / 2.4 / 2.3 s, MPI 4.3 / 3.5 / 4.5 / 7.7 s. With the
viscous vector kernels (MSVC with `/arch:AVX2`) the serial run takes 5.20 s (remesh 0.93 s, minimum of three). Serial, minimum of three:

| machine, compiler, serial | time loop / remesh before | after |
|---|---:|---:|
| Ryzen 7 7800X3D, MSVC (Windows) | 6.01 / 1.23 s | 5.20 / 0.93 s |
| Ryzen 7 7800X3D, gcc (WSL) | 5.95 / 1.22 s | 4.88 / 0.69 s |
| Threadripper PRO 5965WX (24 cores), gcc (WSL) | 6.01 / 1.24 s | 5.03 / 0.74 s |

On the Threadripper (gcc, without the viscous vector kernels, which save another ~0.5 s on one thread) OpenMP is fastest at
16 threads (5.50 -> 1.45 s, 3.8×; 1.63 / 1.74 / 1.80 s on 24 / 32 / 48 threads) and MPI at 8 to 16 ranks (2.96 to 3.08 s). There the
stepping alone keeps scaling up to 16 ranks (2.68 s on 2, 1.09 s on 16) while the replicated remesh grows from 0.98 s on 2 ranks to
1.99 s on 16 and 6.1 s on 48 (with `FENICE_REPLICATED_REMESH=1`). Every run gave the serial state bit for bit.

<figure style="margin:1em 0;">
<img src="{{root}}img/amr_mpi.png" style="width:100%">
<figcaption>AMR airfoil on a 720×360 base mesh with 2 levels, 50 steps, 16 logical processors
(<code>cases/validation/amr/amr_mpi.py</code>). Left: wall time on OpenMP threads and on MPI ranks.
Right: time stepping and remesh. Every final state is bit for bit the serial one.</figcaption>
</figure>

### 3-D octree against the uniform solver (sphere)

The sphere of the [sphere case]({{root}}cases/sphere.html) (diameter 1 in a 6 × 3 × 3 channel, Mach 0.3, run to $t = 30$, 16 threads),
octree and uniform solver at the same finest resolution (`cases/validation/amr/octree_q8.py`):

| finest resolution 20 cells / D | Re 20, mask | Re 20, ghost wall | Re 100, mask | Re 100, ghost wall |
|---|---:|---:|---:|---:|
| $C_d$ difference to the uniform solver | 0.19-0.27 % | 0.04 % | 0.25-0.84 % | 0.02-0.11 % |
| time loop, octree / uniform | 0.20-0.21 | 0.24-0.25 | 0.24-0.27 | 0.30-0.32 |

The octree at Re 20 ends with 72 k leaves (5 cells / D on the base mesh, 2 levels) against 432 k uniform cells. In steady state (200 steps resumed
from the settled run) the octree needs 0.21 of the uniform time on 16 threads and 0.27-0.33 on one thread; memory is 98 MB (139 MB at the
remesh peak). On MPI the remesh limits the speed-up: 2.8× on 8 ranks (1.4× on 2) with the remesh at 18-52 % of the time loop, against 4.3× on
16 OpenMP threads. The peak memory per rank stops falling at 80-90 MB (the remesh fetches), the share itself at 42-55 MB on 8 ranks.

{{widget:AMR}}

### References

* Löhner, R. (1987). An adaptive finite element scheme for transient problems in CFD.
  *Comput. Methods Appl. Mech. Eng.* 61(3), 323-338.
* Morton, G. M. (1966). A computer oriented geodetic data base and a new technique in file
  sequencing. Technical report, IBM Ltd., Ottawa.

{{api}}
