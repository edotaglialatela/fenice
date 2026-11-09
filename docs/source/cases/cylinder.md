---
title: Flow over a cylinder
nav: Cylinder (IBM, AMR, Mach 2)
section: Test cases
order: 40
---

At moderate Reynolds number the wake of a circular cylinder becomes time-periodic,
shedding counter-rotating vortices alternately from the top and bottom of the body. This
**von Kármán vortex street** is a classical 2-D benchmark for viscous unsteady solvers, and at Mach 2 the same body
carries a bow shock. Three cases of the solver run on this body with the
[immersed boundary]({{root}}modules/Physics.html#immersed-boundary-method) and differ in what they test:

| case | flow | domain | mesh | what it checks |
|---|---|---|---|---|
| [`IBM Flow over Cylinder`](#re-100-in-a-42-box) | Re = 100, Ma 0.085 | $4\times2$, $r = 0.1$, blockage 10 % | uniform, 400×200 | shedding, loads, spectra, hot start across meshes |
| [`IBM Cylinder Low Blockage AMR`](#re-100-in-a-large-box-amr) | Re = 100, Ma 0.085 | $16\times12$, blockage 1.7 % | AMR, 20 or 40 cells per diameter | Strouhal number free of confinement, ghost-cell wall |
| [`IBM Supersonic Flow over Cylinder`](#mach-2-the-bow-shock) | Mach 2 | $8\times12$, $r = 0.25$ | uniform, 800×1200 | bow shock and its stand-off distance |

## Default parameters

### Confined, Re = 100

{{defaults:IBM Flow over Cylinder}}

### Large box, AMR

{{defaults:IBM Cylinder Low Blockage AMR}}

### Supersonic

{{defaults:IBM Supersonic Flow over Cylinder}}

## Re = 100 in a 4×2 box

The cylinder is centred at $(1, 1)$ in a $4\times2$ domain. The
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

### Visualisation

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

### Aerodynamic coefficients

The default run ends at $t = 12$, before the shedding has grown: a symmetric body in a symmetric
domain only sheds once the tiny numerical asymmetry has been amplified, which takes about 20 time units
(100 diameters at $u_{
m in} = 1$). `ibm_loads.py cylinder` runs to $t = 80$ and measures from $t = 50$.

Once the flow reaches periodic shedding, the solver writes the lift and drag coefficients from the [IBM loads]({{root}}modules/Physics.html#immersed-boundary-method)
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
quoted to two digits, 0.17-0.18, 5-7 % above the literature band 0.164-0.166 (green), the direction expected from the 10 % blockage, which the [low-blockage run](#re-100-in-a-large-box-amr) confirms (St = 0.166 in a box with 1.7 % blockage)
(<code>cases/validation/benchmarks/cylinder_spectrum.py</code>).</figcaption>
</figure>
<figure>
<img src="{{root}}img/cylinder_modal.png" style="width:100%">
<figcaption>Modal analysis of the same run, from the 82 snapshots of the transverse velocity $v$ at $t \ge 50$ (3.15 snapshots per
period; the steps are uniform to $7\times10^{-4}$, a phase error of $1.4\times10^{-3}$ rad per step). Two POD modes carry 96.3 % of the fluctuation
energy (52.1 % and 44.2 %, the third has 1.1 %); their time coefficients draw a circle, in quadrature to 89.8°, and the pair
is the travelling wave of the vortex street. The exact DMD of the same snapshots has a leading pair at St = 0.1748
($|\lambda| = 1.0001$), against 0.1746 from the lift spectrum; a time-delay DMD of the lift record gives 0.1747
and, for the drag, 2.005 times that frequency. The DMD mode lies in the plane of the two POD modes. On the 1.7 %-blockage AMR runs
([low blockage](#re-100-in-a-large-box-amr)) the lift record gives St = 0.1657 (20 cells per diameter) and 0.1660
(40), by DMD, spectrum and zero crossings alike to $10^{-4}$, inside the literature band (green); the snapshots of those runs are
four time units apart, three periods, so only their lift records are used. Bands fixed beforehand, no simulation run for it
(<code>cases/validation/benchmarks/cylinder_modal.py</code>).</figcaption>
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

### Sensitivity of the drag at Re = 40

<figure>
<img src="{{root}}img/cylinder_sensitivity.png" style="width:100%">
<figcaption>Sobol indices of the steady drag coefficient of the Re = 40 cylinder (ghost-cell wall, channel 4 × 2, $t = 20$) for
the inlet speed (viscosity fixed, so Re = 30 to 50), the resolution (6 to 12 cells per diameter) and the limiter, with 95 %
bootstrap intervals; 160 runs. Right: $C_d$ of the A sample against Re, coloured by resolution; the dotted line is the
converged value 1.8836 of the loads study.</figcaption>
</figure>

`cases/validation/features/cylinder_sensitivity.py --full` (about 40 minutes on 8 workers) gives total indices of $0.46$ for the
inlet speed (interval 0.26 to 0.74), $0.42$ for the resolution (0.26 to 0.66) and $0.05$ for the limiter (0.02 to 0.09). $C_d$ ranges from 1.73 to 2.15
and falls with the Reynolds number (rank correlation $-0.64$). At these meshes the discretisation error of $C_d$ is therefore as large
a part of the variance as the whole 30-50 range of Re; the limiter hardly matters. A run at Re = 40 on 12 cells per diameter gives 1.918, 1.8 % above the converged 1.8836.
The default mode (8 base points, 41 runs) is a smoke test: its intervals are wider than the indices.

## Re = 100 in a large box, AMR

The $4\times2$ box above blocks 10 % of the channel, which raises both the drag and the Strouhal number above the
unbounded values. This case removes the confinement: the same cylinder and flow at $(3, 6)$ in a $16\times12$ box,
$80\times60$ diameters, with a blockage $D/L_y$ of 1.7 %. Covering that box with 40 cells per diameter takes
$3200\times2400 = 7.7$ million cells, so the case uses the [cell-based AMR]({{root}}modules/AMR.html):

* a $200\times150$ base mesh and three refinement levels (20 cells per diameter at the body; `amr_max_level = 4` gives 40),
* the body kept at the finest level (`amr_refine_ibm`) with the **ghost-cell wall** (`ibm_method = ghost`), which needs about
  20 cells per diameter at $\mathrm{Re} = 40$ where the mask needs four times that per direction,
* the wake followed by the speed indicator (`amr_indicator = "velocity"`: the density hardly varies at this Mach number),
* a velocity inlet, a **transmissive** right side and slip walls above and below.
  The `nonreflecting` outlet was tried first and was unstable at this pressure ($p = 100$, sound speed 11.8):
  the pressure grew from 100 to over 300 within eight time units,
* a small transverse kick behind the cylinder (`[[initial.region]]`) that starts the shedding at once; a symmetric start takes
  about 25 time units.

### The mesh

The $200\times150$ base mesh (level 0) has cells of size $h_0 = 16/200 = 0.08$, 2.5 per diameter. Every level halves the cell
size, $h_\ell = 0.08/2^\ell$:

| level | cell size | cells per diameter | where it is used |
|---|---:|---:|---|
| 0 | 0.08 | 2.5 | far field, inlet side, above and below the wake (29 506 of 42 126 leaves at the end of the 4-level run) |
| 1 | 0.04 | 5 | the vortex street downstream of $x \approx 5$ |
| 2 | 0.02 | 10 | transition between the body and the wake |
| 3 | 0.01 | 20 | the body of the 3-level run (`amr_max_level = 3`); a transition level in the 4-level run |
| 4 | 0.005 | 40 | the body of the 4-level run (8 412 leaves, around the body and the first part of the wake) |

* The cells touching the body, and a band of 6 finest-level cells around it (what the ghost-cell wall reads), are on the
  finest level from $t = 0$ and are never joined (`amr_refine_ibm`).
* Elsewhere a leaf is refined when the relative variation of the **speed** across its faces reaches
  `amr_refine_threshold` (0.15); the indicator is smoothed `amr_diffusion = 2` times, which widens the band, and four siblings are joined
  when the indicator of their parent is below `amr_derefine_threshold` (0.1). The mesh is adapted every
  `amr_refine_every = 5` coarse steps, so the fine regions follow the vortices.
* Each level advances with its own time step $\Delta t_0/2^\ell$ (subcycling), finer levels first, and the fluxes through the faces
  between levels are replaced through flux registers, so the scheme stays conservative.
* In the picture below the AMR-level panel near the body shows this: the finest level around the body, a step down through the near wake,
  and level 1 for the rest of the street (the street farther downstream is not in the zoom; level 1 holds the vortices there, as the table says).

```powershell
build\release\fenice.exe --case "IBM Cylinder Low Blockage AMR"                    # 20 cells per diameter, t = 60
build\release\fenice.exe --case "IBM Cylinder Low Blockage AMR" --set amr_max_level=4   # 40 cells per diameter
python cases\validation\benchmarks\cylinder_low_blockage.py                          # both runs, loads, spectra, figure
```

### Results

Over $25 \le t \le 60$ (28 shedding periods), 3 levels (20 cells per diameter) and 4 levels (40 cells per diameter):

| | 20 cells/D | 40 cells/D | literature, Re = 100 |
|---|---:|---:|---:|
| $\overline{C_d}$ | 1.421 | 1.367 | 1.33-1.35 |
| lift amplitude | 0.429 | 0.362 | about 0.33 |
| St (spectrum of $C_l$) | 0.1657 | 0.1659 | 0.164-0.166 |
| St (zero crossings of $C_l$) | 0.1657 | 0.1659 | |
| leaves at the end | 36 186 | 42 126 | |

<figure>
<img src="{{root}}img/cylinder_low_blockage.png" style="width:100%">
<figcaption>Re = 100 in the $16\times12$ box, 40 cells per diameter at the body: vorticity at $t = 60$ (the velocity of the
leaves is smoothed over one coarse wake cell before the derivative; the body is drawn), a zoom on the body at the finest
cell size (smoothing 2.5 cells, so the boundary layer is not widened; vorticity saturated at $\pm 20$) with the AMR level
there, the lift of both resolutions and its spectrum around the peak (the literature band is drawn at its true width). The final mesh has 42 126 leaves in place of 7.7 million cells;
the run takes about 6 minutes on 16 OpenMP threads at 20 cells per diameter and about 13 minutes on 8 MPI ranks at 40
(Ryzen 7 7800X3D).</figcaption>
</figure>

The Strouhal number is $0.166$ at both resolutions, inside the literature band, and the frequency is the same to the
fourth digit, whereas the 10 %-blockage run gives $0.175$: the excess of the confined case is a blockage effect,
not a resolution one. The drag and the lift amplitude are more sensitive to the resolution: the mean drag falls from 1.42 to
1.37 and the lift amplitude from 0.43 to 0.36 when the cells are halved, towards the literature values (1.33-1.35 and
about 0.33); the two runs do not allow an extrapolation (two grids give no observed order), so no further digit is claimed.

### Spectra of the loads

The record $25 \le t \le 60$ (35 time units, 29 shedding periods; frequency resolution $D/T = 0.0057$ in St) is analysed with a
Hann window and zero padding.

<figure>
<img src="{{root}}img/cylinder_low_blockage_spectra.png" style="width:100%">
<figcaption>Top: lift spectrum and a detail of its peak on a linear scale, where the literature band $0.164$-$0.166$ is visible.
Bottom: drag spectrum and the Strouhal number from a sliding window of 12 time units, at 20 and 40 cells per diameter.</figcaption>
</figure>

The lift peaks at $\mathrm{St} = 0.1657$ (20 cells per diameter, amplitude 0.39) and $0.1659$ (40, amplitude 0.33) with a third
harmonic at $0.495$ and $0.497$. The drag peaks at $2.000$ and $2.001$ times the shedding frequency, and has no component at
the shedding frequency itself, as the symmetry of the street requires. A window of 12 time units gives $0.1656$-$0.1658$ and
$0.1658$-$0.1662$ along the record: the signal is stationary, unlike that of the confined case, where the lift amplitude was
still decaying.

In space, the spectrum of $v$ along the wake axis ($y = 6$, $4 \le x \le 11$) of the 40-cells-per-diameter final state, resampled to the
finest level (5 600 cells, Hann window), peaks at wavenumber $6.28$, a wavelength of $1.00$ (5 diameters; resolution in $k$: $0.90$, so the wavelength is known to about
15 %). With the shedding frequency $f = \mathrm{St}\,U/D = 0.83$ this is a convection speed of the vortices of about $0.83\,U$. The whole
line lies in leaves coarser than the finest level, so the resampled field is a staircase (`resample_amr` reports the fraction) and
its spectrum above the peak carries the steps (`benchmarks/spectra_application.py`).

The wake is carried on level 1 of the mesh (a vortex is about 7 of its cells across) and the vortices lose strength faster than at the
body resolution; the loads depend on the near body, where the finest level is kept.

**Hot start.** A hot start (`use_hot_start`) takes the checkpoint of a uniform mesh or of an AMR run, whatever its number of levels:
an adaptive checkpoint is read as its finest-level field (each leaf fills the cells it covers) and an AMR target samples it at its
leaf centres, a uniform target maps it onto its mesh. Resuming an AMR run on its own mesh (`resume = true`) works as before.

## Mach 2: the bow shock

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
