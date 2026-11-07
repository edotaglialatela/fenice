---
title: Sod shock tube
nav: Sod shock tube
section: Test cases
order: 10
---

The canonical validation case for Riemann solvers (Sod 1978). A membrane at
$x = L_x/2$ separates two regions of quiescent ideal gas:

$$\rho_L = 1.0,\quad p_L = 1.0,\quad \rho_R = 0.125,\quad p_R = 0.1,\quad u = 0$$

At $t = 0$ the membrane is removed. The exact solution contains three waves: a
left-running rarefaction, a contact discontinuity and a right-running shock. The mesh is
thin in $y$ ($L_y = 0.1$, $n_y = 20$), and the case forces transmissive left/right
boundaries and slip walls at the bottom and top, so the problem stays one-dimensional.

```powershell
build\release\fenice.exe --case "Sod Shock Tube"
build\release\fenice.exe --case "Sod Shock Tube" --set reconstruction=muscl --set time_scheme=ssprk2
python cases\validation\verification\sod.py           # exact solution, orders, limiters
python cases\validation\amr\sod_amr.py       # AMR vs uniform meshes
```

## Default parameters

{{defaults:Sod Shock Tube}}

## Validation results

<figure>
<img src="{{root}}img/sod_exact.png" style="width:100%">
<figcaption>1st-order Godunov (HLLC, 200 cells) against the exact solution at t = 0.2:
L1 density error 1.1e-2.</figcaption>
</figure>

<figure>
<img src="{{root}}img/sod_orders.png" style="width:100%">
<figcaption>1st-order Godunov vs MUSCL (Van Leer) + SSP-RK2: the L1 density error drops
from 1.1e-2 to 3.0e-3, mostly at the contact and the rarefaction corners.</figcaption>
</figure>

<figure>
<img src="{{root}}img/sod_limiters.png" style="width:100%">
<figcaption>The four MUSCL limiters (L1 density error in the legend): Minmod is the most
diffusive, MC the sharpest, with a small temperature overshoot at the contact.</figcaption>
</figure>

The same case validates [AMR]({{root}}modules/AMR.html): `cases/validation/amr/sod_amr.py` runs it
on a 10×2 base mesh with 4 levels (pre-refined at t = 0, remeshed every 2 steps) and on uniform meshes at the two
finest levels.

<figure>
<img src="{{root}}img/sod_amr.png" style="width:100%">
<figcaption>AMR vs uniform 80×16 and 160×32 meshes at t = 0.2. AMR keeps level 4 around
the contact and the shock and reaches the error of the 160×32 mesh (L1 1.5e-2 vs 1.4e-2) with
45% of its cells (a cell count; the run time of this tiny case is not measured, so no time saving is claimed). Right: the error against the number of cells; the two uniform meshes define the dashed line, and the AMR point lies below it.</figcaption>
</figure>

The refinement follows the waves. `cases/validation/amr/sod_amr_animation.py` runs a 40×4 base mesh with
three levels, remeshing every two steps: the finest cells sit on the contact and the
shock, the rarefaction fan is refined where it is steepest, and each cell is joined again once the wave has
moved on.

<figure>
<img src="{{root}}img/sod_amr_animation.gif" style="width:100%">
<figcaption>Density at the AMR cells (coloured by level) against the exact solution, the levels
below and the mesh with its cell edges at the bottom, for $0 < t \le 0.2$. At $t = 0.1$ the mesh has 2 248 cells instead of the
10 240 of the uniform finest mesh.</figcaption>
</figure>

## Which setting drives the error?

<figure>
<img src="{{root}}img/sod_sensitivity.png" style="width:100%">
<figcaption>Sobol indices of $\log_{10}$ of the L1 density error (MUSCL, SSP-RK2, $t = 0.2$) for the CFL number
(0.2 to 0.8), the resolution (50 to 400 cells) and the limiter (Minmod, Van Leer, MC, Koren), with 95 % bootstrap
intervals; 1 280 runs (256 base points). Right: the 64 points of the A sample.</figcaption>
</figure>

`cases/validation/features/sod_sensitivity.py` samples the three inputs together (Saltelli scheme on a Sobol sequence,
`fenice.uq`) and attributes the variance of the log-error: the resolution carries 0.94 (total index, interval 0.82 to 1.07), the
limiter 0.06 and the CFL number 0.005. Over the whole range the error falls as $n_x^{-0.9}$
(a fit through the A sample). Within this range the choice of limiter is worth about 6 % of the variance, the CFL number nothing. The default mode runs 64
base points (320 runs, about half a minute); the intervals scale as $1/\sqrt{n}$.

## Every scheme, and a published table

`cases/validation/verification/sod_schemes.py` solves the standard problem ($t = 0.2$, HLLC, shock-face dissipation at its default) on
100, 200 and 400 cells with first order, MUSCL with each limiter, WENO3-Z, WENO5-Z and, on the adaptive mesh (base $N/2$ plus one level, the
case's indicator, so the finest cells are those of $N$), MUSCL Van Leer and WENO3-Z. The error is the relative $L_1$ norm
$\sum|q - q_\mathrm{exact}| / \sum|q_\mathrm{exact}|$ at the cell centres against the exact solution; the full table (absolute errors and the
velocity too) is written by the script to `sod_schemes.md`.

<figure>
<img src="{{root}}img/sod_schemes.png" style="width:100%">
<figcaption>Left: relative $L_1$ error of the density at $t = 0.2$ on 100, 200 and 400 cells for the published second-order scheme
(arXiv:1108.2876, black) and for the three best of our reconstructions, one per family (WENO5-Z, MUSCL with the MC limiter, WENO3-Z), with
a line of order 1. Right: the density of the same three at 100 cells around the contact (x = 0.69) and the shock (x = 0.85) against the exact solution.
All three are below the published values at every resolution, with slopes of 0.85 to 1.06, like the published ones.</figcaption>
</figure>

The table of every scheme (relative and absolute $L_1$ of density, velocity and pressure, the orders, the cells of the adaptive runs) is
written to `results/validation/sod_schemes/sod_schemes.md` by the script.

* **Order.** Every second-order scheme sits at 0.7 to 1.1 in density and 0.9 to 1.0 in pressure and velocity (first order 0.65 to 0.8), the one the contact and the shock impose
  whatever the formal order; the published second-order scheme shows the same (0.93, 1.0). Reconstruction changes the error constant, not the order.
* **Magnitude.** The published values (1.3e-2, 6.8e-3, 3.4e-3) lie among ours: MUSCL Minmod gives 1.4e-2, 7.6e-3, 4.3e-3 in density and
  1.2e-2, 6.1e-3, 3.1e-3 in pressure, the sharper limiters and WENO3-Z 0.65 to 0.75 of that, and WENO5-Z about half. The variable of the published
  error is not stated, and its scheme, time step ($\Delta t = \Delta x^2$) and solver are not ours, so this is agreement of magnitude and order, not a reproduction.
* **Ranking.** WENO5-Z is the most accurate on every grid, then MC, Koren, WENO3-Z and Van Leer (within about 10 % of each other), Minmod, and the first-order scheme
  with 3.5 to 8.5 times the error of the second-order ones at 400 cells.
* **Adaptive mesh.** With 568 to 574 cells instead of 1 600 at the finest, the adaptive runs have 1.2 to 1.3 times the error of the uniform grid of the same finest
  resolution (MUSCL Van Leer, density 3.45e-3 against 2.94e-3), at 36 % of the cells.

## Reference

* Sod, G.A. (1978). A survey of several finite difference methods for systems of
  nonlinear hyperbolic conservation laws. *J. Comput. Phys.* 27(1), 1-31.
