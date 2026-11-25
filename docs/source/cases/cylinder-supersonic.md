---
title: Supersonic flow over a cylinder
nav: Cylinder, supersonic (IBM, Mach 2)
section: Test cases
order: 41
---

> **Run it** (output in `results/`; [how to open it]({{root}}guide/getting-started.html#your-first-run)):
>
> * `build\release\fenice.exe --case "IBM Supersonic Flow over Cylinder"`

At Mach 2 the circular cylinder carries a **bow shock**. Two runs of the same case with the [immersed boundary]({{root}}ibm/wall.html): the fixed cylinder, whose stand-off distance is the validation quantity,
and the same cylinder moved up and down, which the shock follows. The subsonic cylinder (vortex shedding at Re = 100, with and without confinement) is on the [subsonic page]({{root}}cases/cylinder.html).

## Mach 2: the bow shock

At supersonic speed a **bow shock** forms ahead of the cylinder. The stand-off
distance $\Delta$ between the shock and the stagnation point is the main validation
quantity: at Mach 2, $\Delta/r \approx 0.4$ (Ambrosio & Wortman 1962). The case validates
the immersed wall in the supersonic regime.

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

### Default parameters, bow shock

{{defaults:IBM Supersonic Flow over Cylinder}}

### Visualisation

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

### Reference

* Ambrosio, A. & Wortman, A. (1962). Stagnation-point shock-detachment distance for flow
  around spheres and cylinders in air. *J. Aero/Space Sci.* 29(7), 875.

## Mach 2: a cylinder moving up and down

The Mach 2 case with a cylinder that heaves in $y$: the [moving immersed boundary]({{root}}ibm/moving-bodies.html) gives the wall the velocity of the body, and the bow shock follows it. The body is a heavy cylinder on a spring, released
from its placement with an initial velocity: a body [driven by the flow]({{root}}ibm/moving-bodies.html#the-coupled-system), $m\,\ddot y = F_y - k_y\,y$ (the load of the flow plus the spring), which for a heavy body is a sinusoidal heave

$$y(t) \approx A \sin \omega t, \qquad \omega = \sqrt{\frac{k_y}{m + m_a}} \approx \sqrt{\frac{k_y}{m}}, \qquad A = \frac{v_0}{\omega}.$$

| key | value | role |
|---|---:|---|
| `ibm_mass` | 60 | mass per unit depth, about 300 times the gas the cylinder displaces ($\rho\pi r^2 = 0.196$): the loads of the flow only perturb the heave |
| `ibm_stiff_y` | 60 | spring in $y$: $\omega \approx 1$, period $2\pi \approx 6.3$ |
| `ibm_vy` | 0.6 | initial velocity (the key is the initial velocity of a driven body): amplitude $A = 0.6$, 1.2 diameters, peak lateral Mach number 0.6 |
| `ibm_stiff_x` | 6000 | pins the cylinder in $x$: the drag moves it by less than $10^{-3}$ |

The aerodynamic damping of the heave is small for a body this heavy (the damping force does not depend on the mass, so its rate falls like $1/m$): the amplitude falls by about 3 % per half period.
The mesh is $200 \times 300$ with three levels: the finest cell is 0.005 (100 cells per diameter), the body is kept at the finest level and the shock is followed by the indicator.

```powershell
build\release\fenice.exe --case "IBM Supersonic Flow over Cylinder" --set use_mpi=false --set nx=200 --set ny=300 --set use_amr=true --set amr_max_level=3 `
  --set ibm_mass=60 --set ibm_stiff_y=60 --set ibm_stiff_x=6000 --set ibm_vy=0.6 --set t_end=20 --set save_interval=100 --set monitor_interval=5
```

The monitor records the position and velocity of the body (`body_y`, `body_v`; `body_y` is the displacement from the placement) with the loads. The path of the shock is read from the snapshots: for each height $y$, the front $x_s(y,t)$ is the first point from the inflow side
where the density passes 1.25, and the stand-off distance at the height of the body is the distance from $x_s$ to the front of the cylinder, $(2 + x_b - r) - x_s(y_b, t)$.

### Results

The heave has period 6.29 ($= 2\pi/\omega$ with $\omega = 1.00$), and its peaks are 0.59, 0.57, 0.55, 0.54, 0.52 and 0.50 in turn; the cylinder drifts in $x$ by less than $6\cdot10^{-4}$. The bow shock follows the body:
the **nose** of the front, its most upstream point, stays at $x \approx 1.40$, 0.35 ahead of the front of the cylinder, and its height moves with the cylinder with the same amplitude (0.56 against 0.54) and a **lag of 0.40 time units**
(6 % of the period, 23°). The stand-off at the height of the cylinder settles at 0.34 after $t \approx 5$ (between 0.335 and 0.345) and does not oscillate within the resolution of the measurement, although the lateral Mach number of the body reaches 0.6.
The front is sampled on a grid of 0.01 (twice the finest cell) every 0.05 time units; the lag is a least-squares fit over $t > 7$. The mesh ends the run with 171 483 leaves.

<figure>
<img src="{{root}}img/supersonic_cylinder_heave_path.png" style="width:70%">
<figcaption>Top: the height of the cylinder (black) and of the nose of the bow shock (red) above the placement; the nose follows with a lag of 0.40. Bottom: the stand-off distance at the height of the cylinder.</figcaption>
</figure>

<figure>
<img src="{{root}}img/supersonic_cylinder_heave.gif" style="width:45%">
<figcaption>Numerical schlieren ($\log(1 + |\nabla\rho|)$), $t = 0$ to 20, with the front of the bow shock traced in red.</figcaption>
</figure>
