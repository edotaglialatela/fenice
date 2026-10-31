---
title: Supersonic flow over a cylinder - Mach 2 bow shock
nav: Supersonic cylinder (Mach 2)
section: Test cases
order: 60
---

At supersonic speed a **bow shock** forms ahead of the cylinder. The stand-off
distance $\Delta$ between the shock and the stagnation point is the main validation
quantity: at Mach 2, $\Delta/r \approx 0.4$ (Ambrosio & Wortman 1962). The case validates
the IBM fringe-cell Neumann condition in the supersonic regime.

The inlet Mach number is exactly 2:

$$p_{\rm in} = \frac{1}{\gamma} = \frac1{1.4}, \qquad \rho_{\rm in} = 1, \qquad
  c = \sqrt{\gamma p_{\rm in}/\rho_{\rm in}} = 1, \qquad u_{\rm in} = 2$$

The cylinder has radius $r = 0.25$ and is centred at $(2, 6)$ in an $8\times12$ domain
(800×1200 cells, 50 per diameter). The left boundary is a velocity inlet; the others are transmissive.

The box is large on purpose. A transmissive side is only non-reflecting for waves that
meet it head-on; an oblique shock is partly reflected, and in a $4\times4$ box the reflections of the bow
shock returned into the wake and crossed the slip lines. Even an $8\times8$ box (cylinder at $(2, 4)$) still showed a faint
reflected line, because the bow shock, which is steeper than the Mach angle near the body, meets the top and bottom sides at
$x \approx 4.8$. In the $8\times12$ box the shock leaves through the top and bottom sides near the right-hand corner, and
by $t = 5$ no reflected wave reaches the flow around the body or the wake. The arc across the wake
is the start-up wave of the impulsive start, which travels downstream, not a boundary effect.

* **MUSCL with Minmod** is preferred over MC here. The strong bow shock makes the more
  compressive limiters prone to carbuncle instabilities on Cartesian grids.
* **SSP-RK2** provides TVD-preserving time integration.
* **MPI:** 16 ranks by default (Y-direction strips).

```powershell
build\release\fenice.exe --case "IBM Supersonic Flow over Cylinder"
```

## Default parameters

{{defaults:IBM Supersonic Flow over Cylinder}}

## Visualisation

<div style="display:flex; gap:1em; align-items:flex-start; flex-wrap:wrap;">
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/supersonic_cylinder_schlieren.gif" style="width:100%">
<figcaption>Numerical schlieren ~ |∇ρ|: bow shock, slip lines, recompression shocks and the start-up wave in the wake.</figcaption>
</figure>
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/supersonic_cylinder_velocity.gif" style="width:100%">
<figcaption>Velocity magnitude: subsonic pocket and high-speed wake.</figcaption>
</figure>
</div>

From t = 0 to 5 on a 1600 × 2400 mesh (100 cells per diameter) over the whole $8\times12$ box; `python docs/showcase.py run`,
then `render`, regenerates both animations. They show only the part $1 \le x \le 8$, $3 \le y \le 9$ of the box (`SUP_VIEW` in `docs/showcase.py`), around the body, the bow shock and the wake.

## Reference

* Ambrosio, A. & Wortman, A. (1962). Stagnation-point shock-detachment distance for flow
  around spheres and cylinders in air. *J. Aero/Space Sci.* 29(7), 875.
