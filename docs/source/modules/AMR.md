---
title: AMR - adaptive mesh refinement on leaf cells
nav: AMR
section: Modules
order: 60
header: CellMesh.hpp
---

The adaptive mesh is a set of **leaf cells** in $d$ dimensions. A leaf is a cell of the level-0 mesh (the `nx × ny × nz` grid of the
case, any sizes) split $\ell$ times by two along each of the $d$ axes, so its size is $h_\ell = h_0/2^\ell$. There are no patches: a
single cell is refined or joined at a time. Leaves are stored in flat arrays sorted by level and Morton (Z-order) key within the
level, so cells that are close in space are close in memory and in the MPI partition.

The dimension is a run-time value of the engine, not a second implementation: `nz > 1` gives an **octree** ($d = 3$), `nz = 1` a
**quadtree** ($d = 2$), the particular case. Everything on this page is written for a leaf with $2d$ sides and holds for both; the drawings
are 2-D because they are easier to draw.

```text
Level 0   ┌───────┬───────┬───────┐
          │       │ ┌─┬─┐ │       │
          │       │ ├─┼─┤ │       │      2:1 balance: leaves that touch
          ├───────┼─┴─┴─┴─┼───────┤      (also at a corner) differ by
          │       │       │       │      at most one level
          └───────┴───────┴───────┘
```

## What adaptivity buys

A feature that is thin in one direction (a shock, a boundary layer, a wake, the surface of a body) needs the fine mesh only
where it is: the fraction of a uniform fine mesh that an adaptive mesh needs falls with the number of levels, and falls faster in 3-D, where
a uniform mesh pays for a third direction too. A leaf carries overhead that a cell of a uniform mesh does not (face lists, virtual cells, flux registers, the remesh) and the adaptive step
parallelises less well, so the comparison that means anything is at **the same finest resolution**, which [Cost and scaling](#cost-and-scaling) measures:

| same finest mesh, 16 OpenMP threads | uniform cells | adaptive leaves | adaptive time loop / uniform |
|---|---:|---:|---:|
| 2-D airfoil, Re 1000, 2 levels | 4.1 M | 268 k (6.5 %) | 0.04 (**23× faster**) |
| 3-D sphere, Re 20, 2 levels | 432 k | 73 k (17 %) | 0.09-0.13 (**8-11× faster**) |

The same solution costs a fraction of the uniform time, on one thread and on 16, and the fraction shrinks as the levels deepen. The sections below
describe the engine; the last one measures it.

## Leaves, levels and balance

| | general $d$ | quadtree ($d = 2$) | octree ($d = 3$) |
|---|---|---|---|
| children of a refined leaf | $2^d$ (joined again only as a complete set of siblings) | 4 | 8 |
| Morton key | $d$ interleaved bit streams of $(i, j, k)$ | $(i, j)$ | $(i, j, k)$ |
| sides of a leaf | $2d$ | 4 | 6 (periodic in $z$ by default) |
| 2:1 balance | across sides, edges and corners: $3^d - 1$ neighbours | 8 | 26 |
| fine faces on a coarse-fine face | $2^{d-1}$ | 2 | 4 |
| mean over a split cell (virtual cell, indicator) | of $2^d$ children | 4 | 8 |
| time-step length | as in the uniform solver | $\min(\Delta x, \Delta y)$ | $2 / (1/\Delta x + 1/\Delta y + 1/\Delta z)$ |
| state per leaf | | $\rho, \rho u, \rho v, \rho E$ | five variables, with $\rho w$ |

The face fluxes, the virtual cells, the jump closure, the subcycling with flux registers and the indicator are the same expressions in every
dimension. The level-0 mesh reproduces the uniform solver bit for bit (the quadtree the 2-D one, the octree the 3-D one), and an adaptive
run is bit-identical for any number of OpenMP threads and of MPI ranks.

The immersed body is classified **per leaf** (there is no uniform mask over the finest level, which would be as large as the uniform
mesh): a leaf is solid if its centre is inside the body, and the body band is kept at the finest level wherever a finest cell lies within
the margin of the surface (`near_body_surface`). The mask forcing and the ghost-cell wall (`ibm_method = ghost`) work on leaves in 2-D and 3-D,
also on the shares of an MPI run (bitwise equal to serial); `auto` chooses the ghost wall wherever `amr_refine_ibm` is on.

Not available on the adaptive mesh in either dimension: WENO5 (WENO3-Z is). The open items are WENO5-Z, the remesh cost on MPI and the validation runs.

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
4. A leaf is **refined** when $\xi \ge$ `amr_refine_threshold` (below `amr_max_level`). The $2^d$ sibling
   leaves (four in 2-D, eight in 3-D) are **joined** when the $\xi$ of their **parent** cell, evaluated at the parent's own level,
   is below `amr_derefine_threshold`, where that $\xi$ is computed (steps 2 and 3) with the criterion
   lowered to 0.9 times `amr_criterion`. Judging the parent rather than the children keeps a marginal
   gradient from flickering: a refined child sees half the variation per face, and would fall under the
   threshold the moment it is refined. The lowered criterion is a dead band: a flow whose variation sits
   exactly at `amr_criterion` (a settled flow past a body does) would otherwise be joined and refined
   again every few remeshes.

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
in 2-D and trilinearly in 3-D. Probes, sample lines and time averages work on the leaves in 3-D too, and so do the body loads
(`[[force]]`): Fx, Fy and Mz (about z) are the momentum balance of the solid leaves, summed in leaf order, the quantity the monitor and the
uniform 3-D solver report, so a 3-D row equals the serial one for any number of ranks.

## Parallel runs

OpenMP threads share the face loops of a level. With `use_mpi` each level is cut into contiguous blocks of equal leaf count along
its Morton order and a rank advances its owned leaves. A rank keeps only its owned leaves and the halo, the leaves its faces read
(stencil points, virtual cells, gradients): its share of the mesh, faces, stencils and state, with the same numbers a whole-mesh
build gives for them (a face that read a leaf outside the share would be an error, checked when the share is made). The remesh
does not gather the state: each rank takes the indicator of its owned leaves from the leaves around them (the ones it does not hold
are fetched from their owners), the refine / coarsen flags of all ranks are gathered (one number per leaf), every rank applies
them to the topology of the whole mesh (leaves and faces, no state; the ripple and the merges are the same on every rank), builds
the state of its new leaves from the old leaves they come from, and builds its new share over the old one: only the margin
around its owned leaves is fetched and looked up, and the faces, neighbour codes and stencils of a leaf that had no leaf come
or go next to it are copied from the old share (`FENICE_FULL_REBUILD=1` computes them all; the tables are the same)
(`FENICE_REPLICATED_REMESH=1` keeps the whole mesh and state on every rank and remeshes it there instead). The owners exchange the halo after each stage, the final state of a level
after it advanced, and the registers of the coarse-fine faces. A same-level face is computed by both owners with the same inputs, a coarse-fine face
by the owner of the fine leaf. The result is identical to the serial run for any number of ranks (1 to 7 are checked in
`amr_mpi.py --check`, on cases with remeshing, an immersed body, viscosity, a non-reflecting outlet and a time-dependent source).

**Known limits.** The indicator, the flags and the state of the new leaves are local, each rank builds its share of the mesh and the engine tables directly from the leaves near its own (a wider reach is tried if a face reads what the share lacks), and the first mesh is adapted the same way, so no rank builds the engine of the whole mesh or adapts a mesh with faces (the base mesh itself is constructed once at setup, and every rank keeps the leaves and lookups of the whole topology). No rank holds the topology of the whole mesh: the ranks fetch the leaves around their own from the owners, settle the ripple and the merges between them and migrate the new leaves; same result bit for bit). The memory per rank at the end of the run falls from 279 MB serial to 140 / 92 / 64 / 51 MB on 2 / 4 / 8 / 16 ranks (peak 373 MB serial, 210 / 128 / 84 / 81 MB), and the wall time of the 720×360 airfoil is 5.0 / 2.6 / 2.1 / 2.4 s (table above); on the 1.06 M-leaf airfoil (1440×720) the memory per rank at the end falls from 1071 MB to 506 / 304 / 200 / 154 MB (peak 1447 MB serial, 774 / 448 / 297 / 297 MB) and the wall time goes from 20.9 s to 24.2 / 14.3 / 10.3 / 9.5 s (17.4 / 10.7 s on 2 / 4 ranks with the ranks pinned to physical cores: on one node the cores the ranks land on matter). The peak is the remesh and the first share. A checkpoint is gathered on rank 0 only, which holds the whole state for the file, so rank 0 has the higher peak when one is written. On one node
OpenMP is the better choice; MPI gives the same result for any rank count and a moderate speed-up. Checkpoint and `dump_state` gather the
leaves on rank 0 (a transient copy of the whole state); a restart cuts each rank's share from the leaf list of the checkpoint (three integer arrays) and reads only the state rows of its own leaves and their halo (`NpzFile`), so no rank holds the whole state or mesh; a hot start reads the whole source field.
At the end of a run the solver prints the resident memory of the ranks (after setup, after the initial adaptation, at the end,
the peak: minimum / maximum over the ranks) and the size of the tables of a rank: about 1.5 kB per leaf on the airfoil
(`cases/validation/amr/amr_memory.py` measures it on 1 to 16 ranks; the remesh is distributed as well, see below).

## Cost and scaling

Machine for all numbers: Ryzen 7 7800X3D (8 cores, 16 threads), MSVC build, `FENICE_PROFILE=1` where the time is split, one run each, on an idle
machine. The right question is what an adaptive run costs **against a uniform run that gives the same answer**, that is, on the same finest mesh;
the adaptive run against itself on more cores (further below) only says how well it parallelises.

### Against the uniform solver on the same finest mesh

<figure style="margin:1em 0;">
<img src="{{root}}img/amr_advantage.png" style="width:100%">
<figcaption>Left: the 2-D airfoil (Re 1000, 720×360 base mesh, two levels, so the finest mesh is 2880×1440; the airfoil and the wake refined)
on 1 to 16 OpenMP threads, adaptive and uniform for the same simulated time (<code>cases/validation/amr/amr_advantage.py</code>). Right: leaves
over uniform cells and time over uniform time for the 2-D airfoil and the 3-D octree sphere.</figcaption>
</figure>

| | 2-D airfoil, 2 levels | 3-D sphere, Re 20, 2 levels |
|---|---:|---:|
| uniform cells at the finest resolution | 4 147 200 | 432 000 |
| adaptive leaves at the end | 268 299 (6.5 %) | 73 012 (17 %) |
| time loop, adaptive / uniform, 1 thread | 5.0 s / 179 s: **36× faster** | 0.27 (steady state): **3.7× faster** |
| time loop, adaptive / uniform, 16 threads | 1.4 s / 34 s: **23× faster** | 0.09-0.13: **8-11× faster** |
| difference of the result | loads of the fine mesh (see the [airfoil]({{root}}cases/airfoil.html) page) | $C_d$ within 0.02-0.85 % (see the [sphere]({{root}}cases/sphere.html) page) |

Two things make the adaptive solver win by more than the cell count would suggest at first sight, and one limits the gain:

* A uniform mesh pays for the finest cell **everywhere**, and the time step of the finest level too: the uniform run takes four times more steps than
  one coarse step of the adaptive run (200 against 50 here), the adaptive one subcycles only the fine levels, which hold a small part of the cells.
* The 2-D airfoil needs 6.5 % of the cells, the 3-D sphere 17 %, at only two levels: a refined region of a given thickness is a smaller fraction of the
  domain the finer the target resolution is, so the saving grows with the number of levels.
* The adaptive run has overhead that a uniform mesh does not (the face list, the virtual cells, the flux registers, the remesh), and its step scales less well on threads (below). With more cores the ratio therefore creeps up (0.027 on one thread, 0.042 on 16 for the airfoil), it does not reverse.

A coarse uniform mesh is of course cheaper than a fine adaptive one (the 192×96 airfoil mesh takes 1.2 s); but it does not resolve the flow, and what
counts is the time to a given error: AMR reaches the loads of the 768×384 mesh in 6.5 s against 56.5 s (see the [airfoil]({{root}}cases/airfoil.html) page).

### Scaling on one node

The same airfoil run (720×360 base mesh, 2 levels, 50 steps, 268 209 leaves):

| run | wall time | speed-up | of which remesh |
|---|---:|---:|---:|
| serial, 1 thread | 4.94 s | 1.0 | 0.88 s |
| OpenMP, 2 / 4 / 8 / 16 threads | 2.71 / 1.67 / 1.42 / 1.36 s | 1.8 / 3.0 / 3.5 / 3.6 | 0.54 / 0.40 / 0.37 / 0.39 s |
| uniform 2880×1440, 1 / 16 threads (200 steps) | 179 / 33.7 s | 1.0 / 5.3 | |
| MPI, 2 / 4 / 8 / 16 ranks, each rank its share of the mesh | 4.99 / 2.64 / 2.05 / 2.41 s | 0.99 / 1.87 / 2.41 / 2.05 | 2.41 / 1.33 / 0.99 / 1.12 s |

Every final state is bit for bit the serial one (`amr_mpi.py`). On a node, **use OpenMP threads for adaptive runs**: 3.6× on 16 threads against 5.3× for the
uniform solver, whose time stays 25 times higher. The adaptive step parallelises less well than the uniform sweep because its levels are small (loops
under 4096 items run on one thread), the remesh does not scale, and 50 steps are short (the remesh and the setup are a fixed part of them, which a long run
amortises less than it looks: the remesh stays 18-29 % of the time loop). MPI gives the same result for any number of ranks but on one node it does not pay for this case: the
remesh of a rank's share costs 1.0-2.4 s against 0.4-0.9 s on threads, and the stepping alone scales (2.6 s on 2 ranks, 1.1 s on 8), so MPI is for runs that
do not fit in the memory of one node. The ranks of this table are placed by the system; pinned to physical cores the 2 / 4 / 8-rank runs take
4.25 / 2.49 / 1.74 s (remesh 2.00 / 1.23 / 0.83 s). Eight ranks on the eight cores of this machine run each about 1.35 times slower than alone and wait for
each other in the collective calls, which a run on several nodes does not pay. It scales better on a larger 3-D mesh, where the stepping is the larger share (the 3-D sphere below).

<figure style="margin:1em 0;">
<img src="{{root}}img/amr_mpi.png" style="width:100%">
<figcaption>AMR airfoil on a 720×360 base mesh with 2 levels, 50 steps, 16 logical processors
(<code>cases/validation/amr/amr_mpi.py</code>). Left: wall time on OpenMP threads and on MPI ranks.
Right: time stepping and remesh. Every final state is bit for bit the serial one.</figcaption>
</figure>

The 3-D sphere (octree, 73 k leaves, 200 steps from the steady state): 16 OpenMP threads 4.6 s (the remesh 7 % of it, 9× faster than on one thread), 8 MPI ranks
10.9 s (the remesh 22 %), 4 ranks 17.1 s. Memory is 87 MB at the end (128 MB at the peak); per rank on 8 ranks 34-39 MB (peak 63-68 MB).

### Where the time goes

<img src="{{root}}img/amr_time_split.png" style="width:100%">

Pie charts of the measured splits (`cases/validation/amr/time_split.py`; the "rest" slices are differences of measured totals). Reading them:

* **The step is faces.** In the 3-D sphere step (100 steps on 16 threads, 1.94 s) the convective faces are 35 %, the viscous cell quantities and faces 28 %, the primitive state and
  the virtual cells (means of the fine leaves under a coarse neighbour, interpolated values; evaluated for every level's call) 17 %, and the rest (stage copies, flux
  registers, the gather of the cell fluxes, the immersed-body forcing, halo exchanges) about a fifth.
* **The remesh is the price of adaptivity** and what limits the parallel efficiency: 7-11 % of a 200-step run on 16 threads, but a fifth to a half of it on 8 ranks:
  2.35 s of 10.9 s on the sphere, where no leaf changes and the fetch of the region the indicator reads is 1.3 s of it, and 1.0 s of 2.05 s on the airfoil above, where
  the leaves change at every remesh and each rank builds its new share over the old one.
* **On the 2-D airfoil** (192×96 base mesh, 8 threads, t = 20) the remesh is 3.1 s of a 19.2 s loop: the indicator 1.15 s, the engine build 0.75 s, the wave-speed
  bound 0.68 s, the adaptation of the mesh 0.55 s. A remesh that changes no leaf costs only the indicator, and the engine tables are updated, not rebuilt.

`FENICE_PROFILE=1` prints these sections (the `amr:` and `fetch:` lines); `cases/validation/amr/amr_memory.py` measures the memory per rank, about 1.5 kB per leaf on the airfoil.

{{widget:AMR}}

### References

* Löhner, R. (1987). An adaptive finite element scheme for transient problems in CFD.
  *Comput. Methods Appl. Mech. Eng.* 61(3), 323-338.
* Morton, G. M. (1966). A computer oriented geodetic data base and a new technique in file
  sequencing. Technical report, IBM Ltd., Ottawa.

{{api}}
