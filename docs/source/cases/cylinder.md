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
python cases\validation\features\cylinder_mesh_transition.py         # hot start 400×200 -> 800×400, from that run
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

The default run ends at $t = 12$, before the shedding has grown: a symmetric body in a symmetric
domain only sheds once the tiny numerical asymmetry has been amplified, which takes about 20 time units
(100 diameters at $u_{m in} = 1$). `ibm_loads.py cylinder` runs to $t = 80$ and measures from $t = 50$.

Once the flow reaches periodic shedding, the lift and drag coefficients
The solver writes the [IBM loads]({{root}}modules/Physics.html#immersed-boundary-method)
(pressure and viscous stresses) to the monitor:

$$C_L = \frac{F_y}{\tfrac12\rho u_{\rm in}^2 D}, \qquad C_D = \frac{F_x}{\tfrac12\rho u_{\rm in}^2 D}$$

The Strouhal number is the dominant frequency of $C_L(t)$: the peak of its windowed,
zero-padded spectrum, which is accurate to a fraction of the 1/window bin even over a few
shedding periods.

<figure>
<img src="{{root}}img/cylinder_loads.png" style="width:100%">
<figcaption>Default 400×200 mesh to $t = 80$: vorticity at the end and loads. The flow stays symmetric
until about $t = 20$ and starts to shed from numerical noise around $t = 25$; over $t \ge 50$,
C<sub>d</sub> = 1.41, C<sub>l</sub> = ±0.28 (still decaying from ±0.34 at $t \approx 35$, so
not yet a limit cycle) and St = 0.175, against 1.33-1.35, ±0.33 and 0.164-0.166 in the literature for
Re = 100 (unbounded domain; the 4×2 box with slip walls blocks 10 % of the channel, which raises both
C<sub>d</sub> and St). The run is <code>benchmarks/ibm_loads.py cylinder</code>.</figcaption>
</figure>
<figure>
<img src="{{root}}img/cylinder_spectrum.png" style="width:100%">
<figcaption>Spectral analysis of the shedding on the 400×200 run, $t \ge 50$ (30 time units, 26 periods; Hann window, zero
padded). The lift peaks at St = 0.1746 with amplitude 0.23; the drag peaks at 0.350, 2.006 times the lift frequency
(each vortex of either sign adds a drag pulse), and the lift has a third harmonic at 0.525. The resolution of the record is
$D/T = 0.0067$ in St. A window of 12 time units slid along the record gives St = 0.173-0.178: it drifts
down as the lift amplitude decays towards its limit cycle, so the record is not stationary and the value is
quoted to two digits, 0.17-0.18, 5-7 % above the literature band 0.164-0.166 (green), the direction expected from the 10 % blockage, which the [low-blockage run]({{root}}cases/cylinder-low-blockage.html) confirms (St = 0.166 in a box with 1.7 % blockage)
(<code>cases/validation/benchmarks/cylinder_spectrum.py</code>).</figcaption>
</figure>
<figure>
<img src="{{root}}img/cylinder_mesh_transition.gif" style="width:100%">
<figcaption>Loads across a hot start: the 400×200 run of the loads study ($t \le 80$, developed street), then 800×400 mapped
from its checkpoint for three more time units. Shedding carries on without a new start-up transient. After a short
drag spike while the field adapts to the finer mask, C<sub>l</sub> has an amplitude of 0.21 (0.17 on the coarse mesh
over the last two time units, where it is still decaying slowly) and C<sub>d</sub> averages 1.43 (1.40 on the coarse mesh)
(<code>cases/validation/features/cylinder_mesh_transition.py</code>).</figcaption>
</figure>
<figure>
<img src="{{root}}img/cylinder_loads_1200x600.gif" style="width:100%">
<figcaption>C_L and C_D over time on a 1200×600 mesh, from the pressure integrated on a
circle just outside the cylinder (no viscous stresses).</figcaption>
</figure>
<figure>
<img src="{{root}}img/cylinder_no_shedding.gif" style="width:100%">
<figcaption>Stable case: without perturbations no vortex shedding develops.</figcaption>
</figure>
