---
title: AMR - block-structured adaptive mesh refinement
nav: AMR
section: Modules
order: 60
header: AMR.hpp
---

A **block-structured quad-tree** AMR layer, following the algorithmic structure of
PARAMESH (MacNeice et al. 2000). Each *leaf patch* is a small uniform sub-mesh with
its own state. Refining a patch replaces it with four children of half the linear
size:

```text
Level 0  ┌───────────────┐
         │  root  (L0)   │
         └───────────────┘
                 │ refine_patch()
Level 1  ┌───────┬───────┐
         │ TL(L1)│ TR(L1)│
         ├───────┼───────┤
         │ BL(L1)│ BR(L1)│
         └───────┴───────┘
              │ refine_patch(BL)
Level 2  ┌───────┬───────┐
         │ TL(L1)│ TR(L1)│
         ├───┬───┼───────┤
         │TLL│TLR│ BR(L1)│   2:1 balance satisfied
         ├───┼───┤       │
         │BLL│BLR│       │
         └───┴───┴───────┘
```

All patches share the same **block size** (`amr_patch_nx × amr_patch_ny` cells). The
cell size halves per level, $h_\ell = h_0/2^\ell$. The block dimensions **must be
even**: 2 fine cells tile 1 coarse cell only if the half-domain boundary lands on a
coarse cell boundary, and `AMRTree` rejects odd sizes.

The level-0 patches **tile the mesh**: `nx / amr_patch_nx` by `ny / amr_patch_ny` roots
(both must divide exactly), so the base grid is exactly the `nx × ny` mesh and the
refinement granularity is one patch. A 192×96 mesh with 16×16 patches has 12×6 roots and
refines in blocks of 16×16 cells at every level. With one root (`nx = amr_patch_nx`) the
tree is the quad-tree drawn above. Neighbour, point and ghost lookups go through a slot
grid at the footprint of the finest patches, rebuilt after every refine or derefine.

## Prolongation and restriction

* **Refinement** copies each coarse cell into its 2×2 fine block (piecewise-constant
  injection). This is exactly mass-conservative and monotone: no over- or undershoots.
* **Derefinement** restricts the children by area-weighted averaging,
  $U_c = \tfrac14\sum U_f$, the exact inverse.
* **Initial condition.** At $t = 0$ every leaf evaluates the case's initial condition at its
  own cell centres, so refined leaves see exact discontinuities. A hot start is instead
  interpolated bilinearly onto the leaves.

## Ghost cells between patches

Every ghost cell outside a patch comes from one routine, a function of the ghost cell and
of the patch's level only:

* **Same-level neighbour.** The neighbour's cell.
* **Finer neighbour.** The mean of the 2×2 fine cells under the ghost cell.
* **Coarser neighbour.** Linear interpolation of the coarse state.
* **Domain edge.** The case's boundary condition; at the corner of an interface and a
  boundary, the boundary condition applied to the ghost cell next to it.

The same ghost cells feed the Riemann states on the interface faces, the two MUSCL halo
layers (so a seam between equal-level patches is reconstructed as in the uniform solver),
the padded state of the viscous flux, and the halos of the non-reflecting outlet. Both
patches sharing a face therefore compute the same flux, which keeps a viscous closed box
conservative. Solid cells of an immersed body are held at rest in the ghost cells too.

A level advances as **one concatenated state**: every Runge-Kutta stage reads the current
stage of its same-level neighbours (as the uniform solver does), while coarser and finer
leaves are read from a snapshot taken when the level starts. A tiling of equal-level
patches is therefore the uniform solver bit for bit, for every time scheme.

The periodic and mass-flow inlet boundaries act on a whole boundary and are rejected with
AMR.

## Subcycling and refluxing

Fine patches advance at $\Delta t_\ell = \Delta t_0/2^\ell$, each level at its own CFL
limit. `advance_amr_level` is recursive:

```text
advance_amr_level(L, dt):
    if a finer level exists:
        init_reflux_registers(L+1)
        2x { advance_amr_level(L+1, dt/2);  accumulate fine face fluxes * dt/2 }
    advance every leaf at level L by dt   (one state, stages synchronised; other levels frozen)
    if a finer level exists:
        apply_reflux(L);  restrict children -> parents
```

**Refluxing** (PARAMESH `amr_flux_conserve`) works between neighbouring leaves. Each
coarse cell next to a coarse-fine face is corrected by
$\pm(\sum F_{\rm fine} - F_{\rm coarse}\Delta t)/A$, so both sides transport the same
amount across the interface. The registers hold the total flux, so the viscous flux is
corrected too. With Explicit Euler this is exactly conservative; multi-stage schemes keep a
small $O(\Delta t^2)$ interface term, because the coarse-fine ghost cells are not
interpolated in time.

## Error indicator and marking

The Löhner (1987) second-derivative sensor is the ratio of second to first density
differences:

$$\eta = \max_{\rm cells}\frac{\sqrt{(\Delta^2_x\rho)^2 + (\Delta^2_y\rho)^2}}
  {|\Delta^+_x\rho| + |\Delta^-_x\rho| + |\Delta^+_y\rho| + |\Delta^-_y\rho| + \varepsilon\,\rho_{\max}}$$

It is $\approx 0$ on smooth gradients such as rarefaction fans, and $\approx 1$ at
shocks and contacts, which keeps refinement focused on discontinuities. Patch edges use
anti-symmetric extension, $\rho_g = 2\rho_{\rm edge} - \rho_{\rm in}$, so linear ramps
give $\eta = 0$.

* **Refine** a leaf when $\eta > \theta_r$ (default 0.3) and it is below `max_level`.
  `amr_buffer_layers` rings of face neighbours are added as well. This PARAMESH
  lookahead keeps a moving feature inside the refined band between remeshes.
* **Derefine** a parent when all four children are leaves with $\eta < \theta_d$
  (default 0.1), unless a child is in the buffered refine set (anti-thrash guard).
* **2:1 balance.** `enforce_2to1_balance` runs after every remesh, so no two
  neighbouring leaves differ by more than one level.
* **Immersed bodies.** With `use_ibm` and `amr_refine_ibm` (default), every leaf whose box,
  grown by one cell, overlaps the body's bounding box is refined to `max_level` from
  $t = 0$ and never coarsened. The test is geometric, so a thin trailing edge cannot slip
  between cell centres.

## Immersed boundary, viscous terms and body force

The leaf residual `patch_residual` adds, when enabled, the viscous flux, a body force and
the immersed-boundary forcing to the convective flux. Each leaf builds its solid mask,
with a two-cell ghost ring, from the body geometry. The velocity of solid cells is zeroed
in every residual, and the fringe and deep-solid reset runs once per complete step on a
snapshot of all leaves, so it does not depend on the leaf order. The total flux
(convective minus viscous) is what the reflux registers store. The monitor file gets the
area-weighted means and the loads Fx, Fy, Cd, Cl, Cm summed over the leaves.

## Output and performance

Each snapshot writes one `.vtr` per leaf into `<case>_<step>/`, grouped by a
`<case>_<step>.vtm` multi-block file. Only the `.vtm` is registered in the `.pvd`,
because ParaView's PVD reader takes one file per time step. The `AMR_Level` cell field
colours the refinement level.

The patches of each level advance concurrently on all OpenMP threads.

**MPI.** With `use_mpi`, every rank holds the whole tree and all patch data, so remeshing,
refluxing, the solid reset and the 2:1 balance need no communication and every rank takes
the same decisions. The patches of each level are shared out over the ranks in contiguous
blocks (leaf order), each rank advances its own, and the replicas are refreshed with an
`MPI_Allgatherv` after every stage (the same-level neighbours' state) and after every
level (state and face fluxes). A patch's arithmetic does not depend on its rank, so the
result is identical to the serial run for any number of ranks. The cost is memory (one
full copy per rank) and a gather per stage; rank 0 writes the output.

<figure style="margin:1em 0;">
<img src="{{root}}img/sod_amr.png" style="width:100%">
<figcaption>Sod shock tube with 10×2 patches and 4 levels against uniform meshes at the two
finest levels: the error of the finest uniform mesh with 53% of its cells
(<code>validation/sod_amr.py</code>).</figcaption>
</figure>

{{widget:AMR}}

### References

* MacNeice, P. et al. (2000). PARAMESH: A parallel adaptive mesh refinement community
  toolkit. *Comput. Phys. Commun.* 126(3), 330-354.
* Löhner, R. (1987). An adaptive finite element scheme for transient problems in CFD.
  *Comput. Methods Appl. Mech. Eng.* 61(3), 323-338.

{{api}}
