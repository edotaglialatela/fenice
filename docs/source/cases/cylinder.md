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
python validation\aeroCoeffCylinder.py
```

## Default parameters

{{defaults:IBM Flow over Cylinder}}

## Visualisation

<div style="display:flex; gap:1em; align-items:flex-start; flex-wrap:wrap;">
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/vonKarman_vortices.gif" style="width:100%">
<figcaption>Vorticity.</figcaption>
</figure>
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/vonKarman_velocity.gif" style="width:100%">
<figcaption>Velocity magnitude.</figcaption>
</figure>
</div>

One shedding period on a 1200 × 600 mesh (60 cells per diameter), hot-started from a
developed 400 × 200 street. `python docs/showcase.py run`, then `render`, regenerates both
animations.

## Aerodynamic coefficients

Once the flow reaches periodic shedding, the lift and drag coefficients
$C_L(t)$ and $C_D(t)$ are obtained by integrating the pressure on a circle just outside the
cylinder ($r = D/2 + 0.015$; viscous stresses are not included):

$$C_L = \frac{F_y}{\tfrac12\rho u_{\rm in}^2 D}, \qquad C_D = \frac{F_x}{\tfrac12\rho u_{\rm in}^2 D}$$

The Strouhal number is the dominant frequency of $C_L(t)$, found by FFT. The results below
use the 1200×600 mesh, [hot-started]({{root}}modules/SolutionMapping.html) from a developed
400×200 street and run for 8 time units; the spectrum starts at t = 1, after the hot-start
transient.

<figure>
<img src="{{root}}img/strouhal_fft_fine.png" style="width:100%">
<figcaption>Energy content in the frequency domain: shedding frequency and Strouhal number.</figcaption>
</figure>
<figure>
<img src="{{root}}img/aero_coeff_fine.gif" style="width:100%">
<figcaption>C_L and C_D over time, 1200×600 mesh.</figcaption>
</figure>
<figure>
<img src="{{root}}img/aero_coeff_meshTransition.gif" style="width:100%">
<figcaption>Aerodynamic coefficients across the mesh mapping: 400×200 to t = 10 (a small
transverse-velocity kick behind the cylinder starts the shedding), then 1200×600.</figcaption>
</figure>
<figure>
<img src="{{root}}img/aero_coeff_stableCase.gif" style="width:100%">
<figcaption>Stable case: without perturbations no vortex shedding develops.</figcaption>
</figure>
