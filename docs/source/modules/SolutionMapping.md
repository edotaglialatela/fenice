---
title: SolutionMapping - interpolation and restriction
nav: SolutionMapping
section: Modules
order: 70
header: SolutionMapping.hpp
---

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
so the first save cannot overwrite it. The IBM mask is re-applied on the target mesh
after interpolation.

`GridInterp2D` implements the semantics of scipy's `RegularGridInterpolator`
(`method='linear'`, `bounds_error=False`, `fill_value=None`). The same interpolator fills
AMR ghost layers from coarser neighbours.

<figure style="margin:1em 0;">
<img src="{{root}}img/cylinder_mesh_transition.gif" style="width:100%">
<figcaption>Loads across a hot start: the 400×200 run of the loads study ($t \le 80$, developed street), then 800×400 mapped
from its checkpoint for three more time units. Shedding carries on without a new start-up transient. After a short
drag spike while the field adapts to the finer mask, C<sub>l</sub> has an amplitude of 0.21 (0.17 on the coarse mesh
over the last two time units, where it is still decaying slowly) and C<sub>d</sub> averages 1.43 (1.40 on the coarse mesh)
(<code>cases/validation/features/cylinder_mesh_transition.py</code>).</figcaption>
</figure>

{{api}}
