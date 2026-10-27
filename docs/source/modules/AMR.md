---
title: AMR - adaptive mesh refinement on leaf cells
nav: AMR
section: Modules
order: 60
header: CellMesh.hpp
---

The adaptive mesh is a set of **leaf cells**, each a square-ish cell of the level-0 mesh
split $\ell$ times by two in each direction (size $h_\ell = h_0/2^\ell$). There are no
patches: the base mesh is the `nx × ny` grid of the case (any sizes), and a single cell is
refined or joined at a time. Leaves are stored in flat arrays sorted by level and Morton
(Z-order) key within the level, so cells that are close in space are close in memory and
in the MPI partition.

```text
Level 0   ┌───────┬───────┬───────┐
          │       │ ┌─┬─┐ │       │
          │       │ ├─┼─┤ │       │      2:1 balance: leaves that touch
          ├───────┼─┴─┴─┴─┼───────┤      (also at a corner) differ by
          │       │       │       │      at most one level
          └───────┴───────┴───────┘
```

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
solver. The one boundary that cannot be used with AMR is the BDF2 dual time step.

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
(steady and time dependent) and every boundary condition except the BDF2 dual time step run on the
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

Each snapshot is one `<case>_<step>.vtu` (an unstructured grid with one quad per leaf, cell fields
Density, Pressure, Temperature_nondim, Mach, Velocity and `AMR_Level`). Under MPI each rank writes the `.vtu` of
its owned leaves and rank 0 a `<case>_<step>.pvtu` that lists them; the `.pvd` series points at the
`.vtu` or `.pvtu`. `python/fenice` reads both with `fenice.read_vtu`.

The checkpoint `<case>_restart.npz` holds the leaf list (level, `i`, `j`), the state of every leaf, `t`, the step
number and the lagged wave-speed bound, so a restarted run continues the same trajectory bit for bit, also on a different
number of ranks. The final state (`dump_state`) is the per-leaf `U` plus `leaf_meta` with columns `x0, x1, y0, y1, level`.
A hot start from a uniform or an adaptive checkpoint interpolates the source at the leaf centres.

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

**Known limits.** The indicator, the flags and the state of the new leaves are local, but every rank still adapts the topology of
the whole mesh (update of the leaves and of the face list), builds the engine tables of the whole mesh, in full (the incremental
update needs the tables of the previous build, which a rank no longer holds), and then keeps its share. That work is the same on
every rank, so it is a serial fraction across ranks (on the airfoil below 1.6 of the 2.9 s on 4 ranks, 4.1 of 5.3 s on 16), the MPI speed-up
flattens at a few ranks, and the peak memory per rank, reached during the remesh, does not fall with the number of ranks. What falls
is the memory between remeshes, except the topology of the whole mesh (87 MB on the airfoil): 256 MB serial, 229 / 184 / 160 / 151 MB per rank
on 2 / 4 / 8 / 16 ranks. On one node
OpenMP is the better choice; MPI gives the same result for any rank count and a moderate speed-up. Checkpoint and `dump_state` gather the
leaves on every rank (a transient copy of the whole state); a restart reads the whole checkpoint on every rank.
At the end of a run the solver prints the resident memory of the ranks (after setup, after the initial adaptation, at the end,
the peak: minimum / maximum over the ranks) and the size of the tables of a rank: about 1.5 kB per leaf on the airfoil
(`cases/validation/amr/amr_memory.py` measures it on 1 to 16 ranks; `docs/design/amr-distributed.md` is the plan to distribute
the remesh as well).

| run (720×360, 2 levels, 50 steps, 268 362 leaves) | wall time | of which remesh |
|---|---:|---:|
| serial, 1 thread | 5.9 s | 1.0 s |
| OpenMP, 2 / 4 / 8 / 16 threads | 3.5 / 2.3 / 2.1 / 2.0 s | 0.75 / 0.65 / 0.65 / 0.74 s |
| MPI, 2 / 4 / 8 / 16 ranks, each rank its share of the mesh (default) | 4.0 / 2.9 / 3.4 / 5.3 s | 1.7 / 1.6 / 2.3 / 4.1 s |
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

{{widget:AMR}}

### References

* Löhner, R. (1987). An adaptive finite element scheme for transient problems in CFD.
  *Comput. Methods Appl. Mech. Eng.* 61(3), 323-338.
* Morton, G. M. (1966). A computer oriented geodetic data base and a new technique in file
  sequencing. Technical report, IBM Ltd., Ottawa.

{{api}}
