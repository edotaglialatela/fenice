---
title: Flow over a cylinder
nav: Cylinder (IBM, AMR, Mach 2)
section: Test cases
order: 40
---

> **Run it** (output in `results/`; [how to open it]({{root}}guide/getting-started.html#your-first-run)):
>
> * `build\release\fenice.exe --case "IBM Flow over Cylinder"`
> * `build\release\fenice.exe --case "IBM Cylinder Low Blockage AMR"`
> * `build\release\fenice.exe --case "IBM Supersonic Flow over Cylinder"`

At moderate Reynolds number the wake of a circular cylinder becomes time-periodic,
shedding counter-rotating vortices alternately from the top and bottom of the body. This
**von Kármán vortex street** is a classical 2-D benchmark for viscous unsteady solvers, and at Mach 2 the same body
carries a bow shock. Three cases of the solver run on this body with the
[immersed boundary]({{root}}ibm/wall.html) and differ in what they test:

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

Once the flow reaches periodic shedding, the solver writes the lift and drag coefficients from the [IBM loads]({{root}}ibm/wall.html)
(pressure and viscous stresses) to the monitor:

$$C_L = \frac{F_y}{\tfrac12\rho u_{\rm in}^2 D}, \qquad C_D = \frac{F_x}{\tfrac12\rho u_{\rm in}^2 D}$$

The Strouhal number is the dominant frequency of $C_L(t)$: the peak of its windowed,
zero-padded spectrum, which is accurate to a fraction of the 1/window bin even over a few
shedding periods.

<figure>
<img src="{{root}}img/cylinder_loads.png" style="width:100%">
<figcaption>Default 400×200 mesh to $t = 80$: vorticity at the end and loads. The flow stays symmetric
until about $t = 20$ and starts to shed from numerical noise around $t = 25$; over $t \ge 50$,
C<sub>d</sub> = 1.54, C<sub>l</sub> = ±0.46 at its peak and St = 0.178, against 1.33-1.35, ±0.33 and 0.164-0.166 in the literature for
Re = 100 (unbounded domain; the 4×2 box with slip walls blocks 10 % of the channel, which raises both
C<sub>d</sub> and St). The run is <code>benchmarks/ibm_loads.py cylinder</code>.</figcaption>
</figure>
<figure>
<img src="{{root}}img/cylinder_spectrum.png" style="width:100%">
<figcaption>Spectral analysis of the shedding on the 400×200 run, $t \ge 50$ (30 time units, 26 periods; Hann window, zero
padded). The lift peaks at St = 0.1775 with amplitude 0.42; the drag peaks at 0.355, 2.001 times the lift frequency
(each vortex of either sign adds a drag pulse). The resolution of the record is
$D/T = 0.0067$ in St. A window of 12 time units slid along the record gives St = 0.176 at its first position and 0.178 from then on:
the record is stationary, and the value is 7-8 % above the literature band 0.164-0.166 (green), the direction expected from the 10 % blockage, which the [low-blockage run](#re-100-in-a-large-box-amr) confirms (St = 0.166 in a box with 1.7 % blockage)
(<code>cases/validation/benchmarks/cylinder_spectrum.py</code>).</figcaption>
</figure>
<figure>
<img src="{{root}}img/cylinder_modal.png" style="width:100%">
<figcaption>Modal analysis of the same run, from the 89 snapshots of the transverse velocity $v$ at $t \ge 50$ (3.33 snapshots per
period; the steps are uniform to $7.6\times10^{-4}$, a phase error of $1.4\times10^{-3}$ rad per step). Two POD modes carry 96.5 % of the fluctuation
energy (51.5 % and 45.0 %, the third has 1.0 %); their time coefficients draw a circle, in quadrature to 89.3°, and the pair
is the travelling wave of the vortex street. The exact DMD of the same snapshots has a leading pair at St = 0.1776
($|\lambda| = 0.99975$), against 0.1775 from the lift spectrum; a time-delay DMD of the lift record gives 0.1776
and, for the drag, 2.0002 times that frequency. The DMD mode lies in the plane of the two POD modes. On the 1.7 %-blockage AMR runs
([low blockage](#re-100-in-a-large-box-amr)) the lift record gives St = 0.1658 (20 cells per diameter) and 0.1661
(40), by DMD, spectrum and zero crossings alike to $10^{-4}$, inside the literature band (green); the snapshots of those runs are
four time units apart, three periods, so only their lift records are used. Bands fixed beforehand, no simulation run for it
(<code>cases/validation/benchmarks/cylinder_modal.py</code>).</figcaption>
</figure>
<figure>
<img src="{{root}}img/cylinder_mesh_transition.gif" style="width:100%">
<figcaption>Loads across a hot start: the 400×200 run of the loads study ($t \le 80$, developed street), then 800×400 mapped
from its checkpoint for three more time units. Shedding carries on without a new start-up transient. After a short
drag spike while the field adapts to the finer mesh, C<sub>l</sub> has an amplitude of 0.42 (0.44 on the coarse mesh
over the last two time units) and C<sub>d</sub> averages 1.47 (1.53 on the coarse mesh)
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

### Analysis in the workbench

The workbench's Analysis Tools module is general-purpose; these settings reproduce selected analyses above:

| Page | Cylinder example |
|---|---|
| **Fields / profiles** | Open the 40-cells-per-diameter AMR result, select `Velocity_y`, and choose **Apply wake-line example**. It samples at finest-leaf spacing from $x=4+h/2$ to $11-h/2$ along $y=6+h/2$, where $h$ is the finest cell width, then applies a Hann window and plots the spatial spectrum. The half-cell inset keeps samples within leaf cells; the line is near, but not exactly on, the $y=6$ centreline used for the reference plot. The report includes peak wavelength, high-wavenumber leakage and the fraction sampled on coarser leaves. |
| **Signals / spectra** | Select `Cl`, set frequency axis to `fL/U`, $L=D=0.2$, $U=1$, start time 50 and peak band 0.05-0.4. Keep Hann and 32x zero padding, set the window to 12 time units sliding by 2, compare `Cd`, enable the 3x harmonic and upward crossings, and add the 0.164-0.166 reference band. The report gives load statistics, peak resolution, lift/drag frequency ratio, harmonic and crossing estimates, and window drift. |
| **POD / DMD** | On the confined run select `Velocity`, component `y`, first snapshot time 50, every snapshot, step tolerance 0.005, $L=0.2$ and $U=1$. The page reports the leading POD energy pair and phase, exact-DMD frequency and growth, and its projection onto the POD plane. For low-blockage AMR runs, whose snapshots are four time units apart, select `Cl`, compare `Cd`, set **Length L** to 0.2, **Speed U** to 1, **Load step** to 0.1 and **Delay rows** to 60, then use **Run time-delay DMD only**. |
| **Sensitivity** | Load `results/validation/cylinder_sensitivity/summary.json` and its `a_sample.csv`. Choose `Re` for X, `Cd` for response Y, `cells_per_D` for point color and 1.8836 as the reference. The left plot shows $S_1$ and $S_T$ with bootstrap intervals; the right shows the sampled drag response. |

## Re = 100 in a large box, AMR

The $4\times2$ box above blocks 10 % of the channel, which raises both the drag and the Strouhal number above the
unbounded values. This case removes the confinement: the same cylinder and flow at $(3, 6)$ in a $16\times12$ box,
$80\times60$ diameters, with a blockage $D/L_y$ of 1.7 %. Covering that box with 40 cells per diameter takes
$3200\times2400 = 7.7$ million cells, so the case uses the [cell-based AMR]({{root}}modules/AMR.html):

* a $200\times150$ base mesh and three refinement levels (20 cells per diameter at the body; `amr_max_level = 4` gives 40),
* the body kept at the finest level with the **ghost-cell wall**, which needs about
  20 cells per diameter at $\mathrm{Re} = 40$,
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
| 0 | 0.08 | 2.5 | far field, inlet side, above and below the wake (most of the 42 621 leaves at the end of the 4-level run) |
| 1 | 0.04 | 5 | the vortex street downstream of $x \approx 5$ |
| 2 | 0.02 | 10 | transition between the body and the wake |
| 3 | 0.01 | 20 | the body of the 3-level run (`amr_max_level = 3`); a transition level in the 4-level run |
| 4 | 0.005 | 40 | the body of the 4-level run (8 412 leaves, around the body and the first part of the wake) |

* The cells touching the body, and a band of 6 finest-level cells around it (what the ghost-cell wall reads), are on the
  finest level from $t = 0$ and are never joined.
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
build\release\fenice.exe --case "IBM Cylinder Low Blockage AMR" --set amr_max_level=5   # 80 cells per diameter
python cases\validation\benchmarks\cylinder_low_blockage.py                          # the first two runs, loads, spectra, figure
```

### Results

Over $t \ge 25$ (to $t = 60$), 3, 4 and 5 levels (20, 40 and 80 cells per diameter at the body):

| | 20 cells/D | 40 cells/D | 80 cells/D | literature, Re = 100 |
|---|---:|---:|---:|---:|
| $\overline{C_d}$ | 1.4204 | 1.3670 | 1.3601 | 1.33-1.35 |
| lift amplitude | 0.4290 | 0.3632 | 0.3508 | about 0.33 |
| St (spectrum of $C_l$) | 0.1657 | 0.1661 | 0.1659 | 0.164-0.166 |
| St (zero crossings of $C_l$) | 0.1658 | 0.1660 | 0.1659 | |
| leaves at the end | 36 693 | 42 621 | 55 941 | |

<figure>
<img src="{{root}}img/cylinder_low_blockage.png" style="width:100%">
<figcaption>Re = 100 in the $16\times12$ box, 40 cells per diameter at the body: vorticity at $t = 60$ (the velocity of the
leaves is smoothed over one coarse wake cell before the derivative; the body is drawn), a zoom on the body at the finest
cell size (smoothing 2.5 cells, so the boundary layer is not widened; vorticity saturated at $\pm 20$) with the AMR level
there, the lift of both resolutions and its spectrum around the peak (the literature band is drawn at its true width). The final mesh has 42 621 leaves in place of 7.7 million cells.</figcaption>
</figure>

The Strouhal number is $0.166$ at all three resolutions, inside the literature band, and the frequency agrees to $4 \cdot 10^{-4}$,
whereas the 10 %-blockage run gives $0.178$: the excess of the confined case is a blockage effect, not a resolution one. The drag and
the lift amplitude are more sensitive to the resolution and converge monotonically towards the literature values (1.33-1.35 and about 0.33):
the mean drag falls from 1.420 to 1.367 and 1.360 when the cells are halved twice, observed order 3.0 (above the formal 2, so the formal
order is used), $F_s = 3$, fine-grid GCI 0.50 % of the finest value, extrapolated 1.358, asymptotic ratio 1.94; the lift amplitude falls from 0.429 to
0.363 and 0.351, observed order 2.4, $F_s = 1.25$, GCI 1.5 %, extrapolated 0.347, asymptotic ratio 1.33. The 80-cell drag is 0.8 % above the
upper end of the literature range and the lift amplitude about 6 % above 0.33.

### Spectra of the loads

The record $25 \le t \le 60$ (35 time units, 29 shedding periods; frequency resolution $D/T = 0.0057$ in St) is analysed with a
Hann window and zero padding.

<figure>
<img src="{{root}}img/cylinder_low_blockage_spectra.png" style="width:100%">
<figcaption>Top: lift spectrum and a detail of its peak on a linear scale, where the literature band $0.164$-$0.166$ is visible.
Bottom: drag spectrum and the Strouhal number from a sliding window of 12 time units, at 20 and 40 cells per diameter.</figcaption>
</figure>

The lift peaks at $\mathrm{St} = 0.1657$ (20 cells per diameter, amplitude 0.39) and $0.1661$ (40, amplitude 0.33) with a third
harmonic near $0.5$. The drag peaks at twice the shedding frequency, and has no component at
the shedding frequency itself, as the symmetry of the street requires. A window of 12 time units gives $0.166$ to the third digit
along the record at both resolutions: the signal is stationary.

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
