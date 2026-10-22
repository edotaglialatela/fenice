---
title: Airfoil - IBM section from a coordinate file or the NACA formulas
nav: Airfoil (IBM, Re = 1000)
section: Test cases
order: 45
---

Low-Reynolds-number flow past an airfoil, with the section traced by the
[immersed boundary]({{root}}modules/Physics.html#immersed-boundary-method). The default is a
NACA 0012 at 10° incidence, $\mathrm{Re} = \rho\,u_{\rm in}\,c/\mu = 1000$ and Ma = 0.2, in
a $12c \times 6c$ box with the quarter chord at $(3c, 3c)$: a velocity inlet on the left, the
[non-reflecting outlet]({{root}}modules/Boundaries.html#non-reflecting-outlet) on the right,
slip walls three chords above and below.

```powershell
build\release\fenice.exe --case "IBM Airfoil"                       # uses MPI (16 ranks) by default
build\release\fenice.exe --case "IBM Airfoil" --set "ibm_shape=NACA 4412" --set ibm_aoa=5
python cases\validation\benchmarks\ibm_loads.py airfoil                        # loads figure + mean coefficients
```

## Loads

With IBM on, every `monitor_interval` steps the monitor records $F_x$, $F_y$ and the
coefficients $C_d$, $C_l$ and $C_m$ (about the quarter chord, nose-up positive), on the chord
and $q = \tfrac12\rho_{\rm in}u_{\rm in}^2$. They are computed inside the solver from the
momentum flux into the body, so they include pressure and friction; see
[Loads on the body]({{root}}modules/Physics.html#loads-on-the-body).

## Default parameters

{{defaults:IBM Airfoil}}

## Results

<figure>
<img src="{{root}}img/airfoil_loads.png" style="width:100%">
<figcaption>NACA 0012 at 10°, Re = 1000, 720 × 360 cells. Top: vorticity at t = 20, with the
shear layers rolling up into a vortex street. Bottom: lift, drag and moment coefficients;
over t ≥ 10 (shaded) C<sub>l</sub> = 0.38 (between 0.26 and 0.51), C<sub>d</sub> = 0.17 and
C<sub>m</sub> ≈ 0 about the quarter chord, as thin-airfoil theory predicts. The dominant lift
frequency is f c/U ≈ 1.21.</figcaption>
</figure>

## Adaptive mesh refinement

The case `IBM Airfoil AMR` is the same flow on a 192×96 base mesh with
[AMR]({{root}}modules/AMR.html): two refinement levels, the airfoil kept at the
finest level (the resolution of a 768×384 mesh, chord = 64 cells) and the wake followed by the
indicator: the default `amr_indicator = "density+velocity"` tests the speed as well as the
density, because at Ma = 0.2 the density hardly varies in the wake (the density alone would
leave it coarse).

```powershell
build\release\fenice.exe --case "IBM Airfoil AMR"        # OpenMP threads; writes Cl, Cd, Cm like the uniform case
python cases\validation\amr\airfoil_amr.py --docs                  # AMR against uniform 192x96 and 768x384
```

<figure style="margin:1em 0;">
<img src="{{root}}img/airfoil_amr.png" style="width:100%">
<figcaption>NACA 0012 at 10°, Re = 1000, up to $t = 6$ (<code>cases/validation/amr/airfoil_amr.py</code>).
Top: vorticity with the refined leaves of the AMR run, which follow the airfoil and the wake. Bottom: drag and lift of the AMR run and of
uniform 192×96 and 768×384 meshes, and the cost of the three runs on 16 OpenMP threads.</figcaption>
</figure>

| run | cells | wall time | rms error of $C_d$ | rms error of $C_l$ |
|---|---:|---:|---:|---:|
| uniform 192×96 | 18 432 | 2.5 s | 0.034 | 0.36 |
| uniform 768×384 | 294 912 | 244 s | (reference) | (reference) |
| AMR, 192×96 + 2 levels | 23 526 | 14 s | 0.011 | 0.054 |

The errors are rms differences of $C_d(t)$ and $C_l(t)$ from the 768×384 run over
$1 \le t \le 6$ (this is the impulsive start of the flow, not a statistically steady state).
AMR reproduces the loads of the fine mesh with 8 % of its cells and 6 % of its time. The
base mesh alone misses most of the lift.

## Using your own airfoil

1. Copy the case: `fenice --new-case "My Airfoil" --from "IBM Airfoil"`.
2. Put the coordinate file in its folder, `cases/my_airfoil/`, for example a `.dat` from
   the [UIUC airfoil database](https://m-selig.ae.illinois.edu/ads/coord_database.html).
   Both of its formats are read: Selig (one loop from the trailing edge over the upper
   surface and back along the lower) and Lednicer (point counts, then the upper and lower
   surfaces from the leading edge).
3. In `cases/my_airfoil/case.toml` set `ibm_shape = "Airfoil"` and `ibm_file = "my_airfoil.dat"`,
   and adjust `ibm_chord`, `ibm_aoa` (degrees, nose up) and the quarter-chord position
   `ibm_cx`, `ibm_cy`.

The coordinates are normalised to unit chord, so the file's scale does not matter.
`reynolds` sets the viscosity from the chord, so changing the chord or the airfoil keeps
the Reynolds number. NACA 4-digit sections need no file: `ibm_shape = "NACA 2412"`.

Fenice is laminar: there is no turbulence model. Reynolds numbers up to a few thousand are
physical; at airliner Reynolds numbers (10⁵ and above) a 2-D laminar run is an unresolved,
unmodelled flow, not a prediction. The staircase IBM also places the wall only to within half
a cell, so the thin trailing edge needs a fine grid: the default resolves the chord with 60
cells.
