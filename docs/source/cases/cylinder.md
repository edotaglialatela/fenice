---
title: Flow over a cylinder - von Kármán vortex street
nav: Cylinder (IBM, Re = 100)
section: Test cases
order: 40
---

At moderate Reynolds number the wake of a circular cylinder becomes time-periodic,
shedding counter-rotating vortices alternately from the top and bottom of the body. This
**von Kármán vortex street** is a classical 2-D benchmark for viscous unsteady solvers.
Here it also validates the [immersed boundary]({{root}}modules/Physics.html#immersed-boundary-method).

The cylinder has radius $r = 0.1$ and is centred at $(1, 1)$ in a $4\times2$ domain. The
Reynolds number based on the diameter $D = 2r$ is

$$\mathrm{Re} = \frac{\rho\,u_{\rm in}\,D}{\mu} = 100$$

with the defaults ($\rho = 1$, $u_{\rm in} = 1$, $\mu = 2\times10^{-3}$), a regime of
stable periodic shedding. The background pressure $p_{\rm out} = 100$ gives
$c \approx 11.8$ and $\mathrm{Ma} \approx 0.085$ (quasi-incompressible). The left boundary
is a velocity inlet, the right one is transmissive, and the bottom and top are slip walls.
The Strouhal number of the shedding frequency $f$ is

$$\mathrm{St} = \frac{f\,D}{u_{\rm in}} \approx 0.165 \quad (\mathrm{Re} = 100)$$

```powershell
build\release\fenice.exe --case "IBM Flow over Cylinder"          # uses MPI (16 ranks) by default
python cases\validation\benchmarks\ibm_loads.py cylinder                # loads, Strouhal number
python cases\validation\features\cylinder_mesh_transition.py         # hot start 400×200 -> 800×400
```

## Default parameters

{{defaults:IBM Flow over Cylinder}}

## Visualisation

<div style="display:flex; gap:1em; align-items:flex-start; flex-wrap:wrap;">
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/cylinder_vortices.gif" style="width:100%">
<figcaption>Vorticity.</figcaption>
</figure>
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/cylinder_velocity.gif" style="width:100%">
<figcaption>Velocity magnitude.</figcaption>
</figure>
</div>

One shedding period on a 1200 × 600 mesh (60 cells per diameter), hot-started from a
developed 400 × 200 street. `python docs/showcase.py run`, then `render`, regenerates both
animations.

## Aerodynamic coefficients

Once the flow reaches periodic shedding, the lift and drag coefficients
The solver writes the [IBM loads]({{root}}modules/Physics.html#immersed-boundary-method)
(pressure and viscous stresses) to the monitor:

$$C_L = \frac{F_y}{\tfrac12\rho u_{\rm in}^2 D}, \qquad C_D = \frac{F_x}{\tfrac12\rho u_{\rm in}^2 D}$$

The Strouhal number is the dominant frequency of $C_L(t)$: the peak of its windowed,
zero-padded spectrum, which is accurate to a fraction of the 1/window bin even over a few
shedding periods.

<figure>
<img src="{{root}}img/cylinder_loads.png" style="width:100%">
<figcaption>Default 400×200 mesh to t = 6: vorticity and loads. Over t ≥ 2,
C<sub>d</sub> = 1.43, C<sub>l</sub> = ±0.37 and St = 0.178, against 1.33-1.35, ±0.33 and
0.164-0.166 in the literature for Re = 100.</figcaption>
</figure>
<figure>
<img src="{{root}}img/cylinder_loads_1200x600.gif" style="width:100%">
<figcaption>C_L and C_D over time on a 1200×600 mesh, from the pressure integrated on a
circle just outside the cylinder (no viscous stresses).</figcaption>
</figure>
<figure>
<img src="{{root}}img/cylinder_mesh_transition.gif" style="width:100%">
<figcaption>Loads across a hot start: 400×200 to t = 6, then 800×400 mapped from its
checkpoint. Shedding carries on without a new start-up transient; after a short drag
spike while the field adapts to the finer mask, C<sub>l</sub> keeps its amplitude (0.37)
and C<sub>d</sub> settles at 1.46 (1.43 on the coarse mesh)
(<code>cases/validation/features/cylinder_mesh_transition.py</code>).</figcaption>
</figure>
<figure>
<img src="{{root}}img/cylinder_no_shedding.gif" style="width:100%">
<figcaption>Stable case: without perturbations no vortex shedding develops.</figcaption>
</figure>
