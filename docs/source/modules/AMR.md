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
coarse cell boundary, and `AMRTree` rejects odd sizes. For example, to match a uniform
200×20 grid use a 100×10 patch with `max_level = 1`, not 50×5 with `max_level = 2`.

## Prolongation and restriction

* **Refinement** copies each coarse cell into its 2×2 fine block (piecewise-constant
  injection). This is exactly mass-conservative and monotone: no over- or undershoots.
* **Derefinement** restricts the children by area-weighted averaging,
  $U_c = \tfrac14\sum U_f$, the exact inverse.
* **Seeding.** At $t = 0$ every leaf is seeded by bilinear interpolation of the
  full-domain initial field at the leaf's global cell centres.

## Ghost cells between patches

At every *internal* patch face the ghost layer is filled from the neighbouring leaf:

* **Coarser neighbour.** Linear prolongation of the coarse state, which avoids
  stair-stepping.
* **Same-level or finer neighbour.** Piecewise-constant injection, copying the
  neighbour cell that covers the ghost point.

Physical domain edges use the case's boundary conditions. While a level advances, every
leaf reads a **frozen snapshot** of its neighbours (`ghost_source`). This is required for
conservation: without it a later-updated patch would read its neighbour's *new* state,
and the two sides would disagree on the shared-face flux.

## Subcycling and refluxing

Fine patches advance at $\Delta t_\ell = \Delta t_0/2^\ell$, each level at its own CFL
limit. `advance_amr_level` is recursive:

```text
advance_amr_level(L, dt):
    if a finer level exists:
        init_reflux_registers(L+1)
        2x { advance_amr_level(L+1, dt/2);  accumulate fine face fluxes * dt/2 }
    advance every leaf at level L by dt   (patches in parallel, frozen neighbour snapshot)
    if a finer level exists:
        apply_reflux(L);  restrict children -> parents
```

**Refluxing** (PARAMESH `amr_flux_conserve`) works between neighbouring leaves. Each
coarse cell next to a coarse-fine face is corrected by
$\pm(\sum F_{\rm fine} - F_{\rm coarse}\Delta t)/A$, so both sides transport the same
amount across the interface. With Explicit Euler this is exactly conservative. Multi-stage
schemes keep a small $O(\Delta t^2)$ interface term, because each stage reads the frozen
$\mathbf U^n$ snapshot.

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

## Output and performance

Each snapshot writes one `.vtr` per leaf into `<case>_<step>/`, grouped by a
`<case>_<step>.vtm` multi-block file. Only the `.vtm` is registered in the `.pvd`,
because ParaView's PVD reader takes one file per time step. The `AMR_Level` cell field
colours the refinement level.

AMR runs are serial-only (MPI is ignored), but the patches of each level advance
concurrently on all OpenMP threads.

<figure style="margin:1em 0;">
<img src="{{root}}img/sod_amr_comparison.png" style="width:100%">
<figcaption>Sod shock tube: uniform mesh vs AMR. Density profile, cell count and L2 error.</figcaption>
</figure>

{{widget:AMR}}

### References

* MacNeice, P. et al. (2000). PARAMESH: A parallel adaptive mesh refinement community
  toolkit. *Comput. Phys. Commun.* 126(3), 330-354.
* Löhner, R. (1987). An adaptive finite element scheme for transient problems in CFD.
  *Comput. Methods Appl. Mech. Eng.* 61(3), 323-338.

{{api}}
