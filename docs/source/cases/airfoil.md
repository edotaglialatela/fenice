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
cpp\build\fenice.exe --case "IBM Airfoil"                       # uses MPI (16 ranks) by default
cpp\build\fenice.exe --case "IBM Airfoil" --set "ibm_shape=NACA 4412" --set ibm_aoa=5
python validation\airfoilLoads.py                              # loads figure + mean coefficients
```

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
<img src="{{root}}img/airfoil_naca0012_loads.png" style="width:100%">
<figcaption>NACA 0012 at 10°, Re = 1000, 720 × 360 cells. Top: vorticity at t = 20, with the
shear layers rolling up into a vortex street. Bottom: lift, drag and moment coefficients;
over t ≥ 10 (shaded) C<sub>l</sub> = 0.38 (between 0.26 and 0.51), C<sub>d</sub> = 0.17 and
C<sub>m</sub> ≈ 0 about the quarter chord, as thin-airfoil theory predicts. The dominant lift
frequency is f c/U ≈ 1.2.</figcaption>
</figure>
