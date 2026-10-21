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

The cylinder has radius $r = 0.25$ and is centred at $(1, 2)$ in a $4\times4$ domain
(400×400). The left boundary is a velocity inlet; the others are transmissive.

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
<figcaption>Numerical schlieren ~ |∇ρ|: bow shock, slip lines, Mach reflection.</figcaption>
</figure>
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/supersonic_cylinder_velocity.gif" style="width:100%">
<figcaption>Velocity magnitude: subsonic pocket and high-speed wake.</figcaption>
</figure>
</div>

From t = 0 to 5 on a 1200 × 1200 mesh; `python docs/showcase.py run`, then `render`,
regenerates both animations.

## Reference

* Ambrosio, A. & Wortman, A. (1962). Stagnation-point shock-detachment distance for flow
  around spheres and cylinders in air. *J. Aero/Space Sci.* 29(7), 875.
