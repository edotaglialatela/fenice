---
title: AMR - adaptive mesh refinement on leaf cells
nav: AMR
section: Methods
order: 60
---

> **In short.** With `use_amr = true` the mesh refines itself where the flow changes quickly (shocks, shear layers, the
> neighbourhood of a body) and coarsens elsewhere, so a run costs a fraction of a uniform mesh with the same finest cells. The base
> mesh is `nx x ny (x nz)`; `amr_max_level` sets how many times a cell may be split, and the `amr_*` keys tune the indicator. Results are bit-identical on any number of MPI
> ranks. The [cost](#cost-and-scaling) section shows what to expect.

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
| 2-D airfoil, Re 1000, 2 levels | 4.1 M | 268 k (6.5 %) | 0.07 (**15× faster**) |
| 3-D sphere, Re 20, 2 levels | 432 k | 76 k (18 %) | 0.27-0.30 (**3.4-3.7× faster**) |

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
the margin of the surface (`near_body_surface`). The ghost-cell wall works on leaves in 2-D and 3-D,
also on the shares of an MPI run (bitwise equal to serial).

`reconstruction = weno5` runs on the adaptive mesh in 2-D and 3-D, also on MPI ranks: a face whose six cells along its normal are fluid leaves
of its level (replicated or mirrored at a physical side, wrapped on a periodic axis) takes the WENO5-Z stencil; every other face (a coarse-fine face,
a face within two cells of a level jump or of the immersed body) keeps the four-cell stencil with the WENO3-Z formula, so the seams are those of a WENO3-Z run.
A periodic axis needs at least three base cells. The level-0 mesh equals the uniform solver bit for bit. The open item is the remesh cost on MPI.

Against WENO3-Z on the same adaptive mesh (`cases/validation/amr/weno5_amr.py`, SSP-RK3, the indicator of each case):

| case | quantity | WENO3-Z | WENO5-Z | ratio |
|---|---|---:|---:|---:|
| isentropic vortex, one period, base 40 + 1 level | $L_2$ density error | 1.01e-2 | 8.12e-3 | 0.80 |
| Sod, base 100 + 1 level | $L_1$ density error | 3.60e-3 | 3.15e-3 | 0.88 |
| Shu-Osher, base 200 + 1 level | $L_1$ density error (reference: uniform WENO5-Z, 3200 cells) | 7.38e-2 | 2.99e-2 | 0.40 |
| cavity Re 400, base 32 + 1 level | RMS of centreline $u$ / $v$ against the tabulated values | 4.23e-3 / 4.40e-3 | 3.65e-3 / 3.53e-3 | 0.86 / 0.80 |
| cylinder Re 40, 20 finest cells per diameter | $C_d$ (uniform 400×200: 1.9047 / 1.9041) | 1.9424 | 1.8922 | 0.974 |
| cost per step, 1 thread / 16 threads / 4 ranks | wall time | 40.4 / 9.3 / 15.6 ms | 50.8 / 10.3 / 17.8 ms | 1.26 / 1.11 / 1.14 |

The bands fixed before the runs are met (vortex: not above WENO3-Z; the tubes: within 5 % of WENO3-Z and no new over- or undershoot above $10^{-3}$ of the
jump); the cavity, the cylinder and the cost are reported. The gain is largest where smooth structure is carried through the fine leaves (Shu-Osher) and
small where the seam or a shock sets the error. On the vortex the adaptive error stays about 30 times that of the uniform WENO5-Z run on the finest grid:
the seam, not the reconstruction, limits a smooth solution (see the [manufactured solutions]({{root}}cases/manufactured-solution.html)).

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
* **Immersed bodies.** With `use_ibm` every leaf whose footprint touches the
  body, or lies in the band of the ghost cells' interpolation stencils around it (about 6
  finest cells), is kept at the finest level from $t = 0$ and never joined, so the sharp wall works on the finest level.
  The test is on the finest-level cells, so a thin trailing edge cannot slip between cell centres.
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
its Morton order and a rank advances its owned leaves.

* A rank keeps only its owned leaves and the halo, the leaves its faces read
(stencil points, virtual cells, gradients): its share of the mesh, faces, stencils and state, with the same numbers a whole-mesh
build gives for them (a face that read a leaf outside the share would be an error, checked when the share is made).
* The remesh
does not gather the state: each rank takes the indicator of its owned leaves from the leaves around them (the ones it does not hold
are fetched from their owners), the refine / coarsen flags of all ranks are gathered (one number per leaf), every rank applies
them to the topology of the whole mesh (leaves and faces, no state; the ripple and the merges are the same on every rank), builds
the state of its new leaves from the old leaves they come from, and builds its new share over the old one: only the margin
around its owned leaves is fetched and looked up, and the faces, neighbour codes and stencils of a leaf that had no leaf come
or go next to it are copied from the old share (`FENICE_FULL_REBUILD=1` computes them all; the tables are the same)
(`FENICE_REPLICATED_REMESH=1` keeps the whole mesh and state on every rank and remeshes it there instead).
* The owners exchange the halo after each stage, the final state of a level
after it advanced, and the registers of the coarse-fine faces.
* A same-level face is computed by both owners with the same inputs, a coarse-fine face
by the owner of the fine leaf.
* The result is identical to the serial run for any number of ranks (1 to 7 are checked in
`amr_mpi.py --check`, on cases with remeshing, an immersed body, viscosity, a non-reflecting outlet and a time-dependent source).

**Known limits.** The indicator, the flags and the state of the new leaves are local, each rank builds its share of the mesh and the engine tables directly from the leaves near its own (a wider reach is tried if a face reads what the share lacks), and the first mesh is adapted the same way, so no rank builds the engine of the whole mesh or adapts a mesh with faces (the base mesh itself is constructed once at setup, and every rank keeps the leaves and lookups of the whole topology).

* No rank holds the topology of the whole mesh: the ranks fetch the leaves around their own from the owners, settle the ripple and the merges between them and migrate the new leaves; same result bit for bit).
* **Memory and time per rank** (the largest over the ranks, airfoil with 2 levels, 50 steps; the 1440×720 wall times are from the separate memory study):

  | Ranks | 720×360: end MB | peak MB | wall s | 1440×720 (1.06 M leaves): end MB | peak MB | wall s |
  |---|---|---|---|---|---|---|
  | 1 | 424 | 487 | 5.0 | 1315 | 1566 | 19.5 |
  | 2 | 246 | 300 | 3.9 | 897 | 1033 | 14.8 |
  | 4 | 178 | 187 | 2.3 | 548 | 626 | 10.2 |
  | 8 | 114 | 114 | 1.4 | 376 | 396 | 7.0 |
  | 16 | 83 | 83 | 0.88 | 268 | 273 | 4.3 |

  On 24 and 48 ranks the peak of the large mesh stays at 270 MB (the end falls to 237 and 212 MB).
* The peak is the remesh and the first share.
* A checkpoint is written by rank 0 as the leaves of each rank reach it, one rank after the other: it holds its own leaves and those of one other rank, never the whole state.
* The whole mesh is gathered on rank 0 only for the post-processing outputs (probes, samples, averages) and for `dump_state`.
* On one node
OpenMP is the better choice; MPI gives the same result for any rank count and a moderate speed-up.
* A checkpoint on several ranks is written rank after rank (an `NpzWriter` with reserved entries, zip64 for 4 GB and more), so no rank holds the whole state;
the final `dump_state` gathers the leaves on rank 0 (a transient copy of the whole state); a restart cuts each rank's share from the leaf list of the checkpoint (three integer arrays) and reads only the state rows of its own leaves and their halo (`NpzFile`), so no rank holds the whole state or mesh; a hot start reads the whole source field.
* At the end of a run the solver prints the resident memory of the ranks (after setup, after the initial adaptation, at the end,
the peak: minimum / maximum over the ranks) and the size of the tables of a rank: about 1.4 kB per leaf on the airfoil
(`cases/validation/amr/amr_memory.py` measures it on 1 to 48 ranks; the remesh is distributed as well, see below).

## Cost and scaling

Machine for all numbers: a workstation with a Threadripper PRO 5965WX (24 cores, 48 hardware threads), gcc build under Linux, `FENICE_PROFILE=1` where the time is split, on an idle
machine (the solver was also built and run on other x86-64 and ARM64 machines, see [Getting started]({{root}}guide/getting-started.html)). The right question is what an adaptive run costs **against a uniform run that gives the same answer**, that is, on the same finest mesh;
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
| adaptive leaves at the end | 268 299 (6.5 %) | 76 120 (18 %) |
| time loop, adaptive / uniform, 1 thread | 5.0 s / 104.6 s: **21× faster** | 0.34-0.42 (steady state): **2.4-2.9× faster** |
| time loop, adaptive / uniform, 16 threads | 0.9 s / 13.6 s: **15× faster** | 0.27-0.30 (steady state): **3.4-3.7× faster** |
| difference of the result | loads of the fine mesh (see the [airfoil]({{root}}cases/airfoil.html) page) | $C_d$ within 0.02-0.13 % (Re 20 and 100, see the [sphere]({{root}}cases/sphere.html) page) |

Two things make the adaptive solver win by more than the cell count would suggest at first sight, and one limits the gain:

* A uniform mesh pays for the finest cell **everywhere**, and the time step of the finest level too: the uniform run takes four times more steps than
  one coarse step of the adaptive run (200 against 50 here), the adaptive one subcycles only the fine levels, which hold a small part of the cells.
* The 2-D airfoil needs 6.5 % of the cells, the 3-D sphere 17 %, at only two levels: a refined region of a given thickness is a smaller fraction of the
  domain the finer the target resolution is, so the saving grows with the number of levels.
* The adaptive run has overhead that a uniform mesh does not (the face list, the virtual cells, the flux registers, the remesh), and its step scales less well on threads (below). With more cores the ratio therefore creeps up (0.048 on one thread, 0.067 on 16 for the airfoil), it does not reverse.

A coarse uniform mesh is of course cheaper than a fine adaptive one (the 192×96 airfoil mesh takes 2.9 s); but it does not resolve the flow, and what
counts is the time to a given error: AMR reaches the loads of the 768×384 mesh in 13.6 s against 200.4 s (see the [airfoil]({{root}}cases/airfoil.html) page).

### Scaling on one node

The same airfoil run (720×360 base mesh, 2 levels, 50 steps, 268 299 leaves):

| run | wall time | speed-up | of which remesh |
|---|---:|---:|---:|
| serial, 1 thread | 4.89 s | 1.0 | 0.84 s |
| OpenMP, 2 / 4 / 8 / 16 threads | 2.74 / 1.81 / 1.28 / 0.87 s | 1.8 / 2.7 / 3.8 / 5.6 | 0.48 / 0.32 / 0.25 / 0.22 s |
| uniform 2880×1440, 1 / 16 threads (200 steps) | 104.6 / 13.6 s | 1.0 / 7.7 | |
| MPI, 2 / 4 / 8 / 16 ranks, each rank its share of the mesh | 3.84 / 2.43 / 1.53 / 0.95 s | 1.3 / 2.0 / 3.2 / 5.2 | 1.59 / 1.00 / 0.62 / 0.47 s |

Every final state is bit for bit the serial one (`amr_mpi.py`). Up to 16 workers OpenMP threads are ahead of MPI ranks at every count, and the two meet at 16 (5.6× and 5.2×);
the uniform solver gains 7.7× on 16 threads and its time stays 15 times higher.

* The adaptive step parallelises less well than the uniform sweep because its levels are small (loops
under 4096 items run on one thread), the remesh scales less than the stepping, and 50 steps are short (the remesh and the setup are a fixed part of them, which a long run
amortises less than it looks: the remesh stays 17-25 % of the time loop on threads, and 41-49 % on ranks).
* The remesh of a rank's share costs 0.5-1.6 s against 0.2-0.5 s on threads,
while the stepping alone scales on ranks as well as on threads, so MPI is for runs that do not fit in the memory of one node, or that go beyond it.
* The larger airfoil (1440×720 base mesh, 1.06 M leaves, 50 steps) takes 19.2 s serial, 3.5 s on 16 threads (5.5×) and 4.3 s on 16 ranks (4.5×), and its best time on MPI is 3.7 s on 48 ranks (5.2×);
with WENO5-Z the same run takes 37.5 s serial and 4.9 s on 48 ranks (7.7×), and the remesh is a smaller share of it (34 % against 43 % for MUSCL on 48 ranks). The scaling over 1 to 48 workers of this and
of the other adaptive and uniform cases is on the [Performance]({{root}}guide/performance.html#scaling-on-a-24-core-workstation) page.

<figure style="margin:1em 0;">
<img src="{{root}}img/amr_mpi.png" style="width:100%">
<figcaption>AMR airfoil on a 720×360 base mesh with 2 levels, 50 steps, 1 to 16 workers on the 24-core workstation
(<code>cases/validation/amr/amr_mpi.py</code>). Left: speed-up over the serial adaptive run on OpenMP threads and on MPI ranks,
for the time stepping alone and with the remesh. Right: time of the uniform solver on the
same finest mesh over the time of the adaptive run, on the same number of threads. Every final state is bit for bit the serial one.</figcaption>
</figure>

The 3-D sphere (octree, 76 k leaves, Re 20, 200 steps from the steady state), time per unit of physical time:

| Workers | OpenMP threads (s) | MPI ranks (s) | Remesh share (ranks) | MB per rank at the end (peak) |
|---|---|---|---|---|
| 1 | 29.6 | | | |
| 2 | | 19.2 | 8 % | |
| 4 | 9.4 | 11.7 | 11 % | |
| 8 | 7.1 | 7.5 | 16 % | 63 (77) |
| 16 | 4.9 | | | 111 (142), one process of 16 threads |
| 24 | 4.9 | 5.0 | 30 % | 59 (76) |
| 48 | 4.8 | 5.1 | 38 % | 65 (76) |

Threads reach 6.0× on 16 and gain nothing more on 24 and 48.

<figure style="margin:1em 0;">
<img src="{{root}}img/amr_memory.png" style="width:100%">
<figcaption>Memory and time per rank of the airfoil on 1 to 16 MPI ranks (720×360 and 1440×720 base meshes, 2 levels, 50 steps; <code>cases/validation/amr/amr_memory.py</code>):
the resident memory of the largest rank, at the peak and at the end, and the wall time split into the time stepping and the remesh.</figcaption>
</figure>

### Where the time goes

<img src="{{root}}img/amr_time_split.png" style="width:100%">

Pie charts of the measured splits (`cases/validation/amr/time_split.py`; the "rest" slices are differences of measured totals). Reading them:

* **The step is faces.** In the 3-D sphere step (60 steps on one thread, 4.26 s) the face fluxes take 3.54 s (83 %): the convective faces 1.44 s, the primitive state and
  the virtual cells (means of the fine leaves under a coarse neighbour, interpolated values; evaluated for every level's call) 0.89 s, and the rest, the viscous cell quantities and faces, about 1.2 s. The stage copies, flux
  registers, the gather of the cell fluxes, the immersed-body forcing and the halo exchanges are the other 17 %.
* **The remesh is the price of adaptivity** and what limits the parallel efficiency: 17-25 % of the 50-step airfoil run on threads (17 % of the 60-step sphere run on 16 threads), but about 40-50 % of it on ranks:
  0.62 s of 1.53 s on 8 ranks for the airfoil above, where the leaves change at every remesh and each rank builds its new share over the old one, and 1.28 s of 1.87 s on 16 ranks
  for the sphere (60 steps), of which the rebuild is 0.96 s and the fetch of the leaves 0.46 s.
* **On the 2-D airfoil** (720×360 base mesh, 1 thread, 50 steps) the remesh is 0.86 s of 5.0 s: the wave-speed
  bound 0.31 s, the indicator 0.28 s, the adaptation of the mesh 0.24 s and the engine build 0.21 s. A remesh that changes no leaf costs only the indicator, and the engine tables are updated, not rebuilt.

`FENICE_PROFILE=1` prints these sections (the `amr:` and `fetch:` lines); `cases/validation/amr/amr_memory.py` measures the memory per rank, about 1.4 kB per leaf on the airfoil.

{{widget:AMR}}

### References

* Löhner, R. (1987). An adaptive finite element scheme for transient problems in CFD.
  *Comput. Methods Appl. Mech. Eng.* 61(3), 323-338.
* Morton, G. M. (1966). A computer oriented geodetic data base and a new technique in file
  sequencing. Technical report, IBM Ltd., Ottawa.

---

*Declarations: [`fv/CellMesh.hpp`]({{root}}api/AMR.html).*
