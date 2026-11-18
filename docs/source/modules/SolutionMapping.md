---
title: SolutionMapping - interpolation and restriction
nav: SolutionMapping
section: Methods
order: 70
---

> **In short.** Interpolation between meshes. Its main use is the **hot start**: `use_hot_start = true` with `hot_start_file` loads a
> checkpoint from a run on another mesh (coarser, finer, or 2-D into 3-D) and continues from it.

## Hot start (solution mapping)

`map_solution` maps a converged coarse field onto a finer (or coarser) mesh of the same
domain $[0,L_x]\times[0,L_y]$ by bilinear interpolation between cell centres. Outside the
outermost centres it extrapolates linearly. This is what `use_hot_start` does with a
checkpoint: a 400×200 cylinder wake can be developed cheaply and then continued on
800×400 without repeating the start-up transient. Uniform and linear fields are mapped
exactly; total mass is conserved to interpolation accuracy.

In 3-D, `map_solution_3d` does the same by trilinear interpolation. A 2-D checkpoint can
also seed a 3-D run: it is mapped in the x-y plane and extruded along z. A 3-D checkpoint
cannot seed a 2-D run (explicit error).

Checkpoints carry their grid metadata (`nx`, `ny`, `Lx`, `Ly`, plus `nz`, `Lz`). For legacy files without
it, the source size comes from `hot_start_src_nx` / `hot_start_src_ny`. If the source
file is the run's own checkpoint, it is first copied to `<case>_hotstart_source.npz`,
so the first save cannot overwrite it. The immersed body is set up again on the target mesh
after interpolation.

`GridInterp2D` implements the semantics of scipy's `RegularGridInterpolator`
(`method='linear'`, `bounds_error=False`, `fill_value=None`). The same interpolator fills
AMR ghost layers from coarser neighbours.

A worked example, with the loads across a 400×200 → 800×400 hot start, is on the [cylinder page]({{root}}cases/cylinder.html).

---

*Declarations: [`fv/SolutionMapping.hpp`]({{root}}api/SolutionMapping.html).*
