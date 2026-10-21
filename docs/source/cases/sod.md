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
build\release\fenice.exe --case "Sod Shock Tube" --set use_muscl=true --set time_scheme=ssprk2
python cases\validation\sod.py           # exact solution, orders, limiters
python cases\validation\sod_amr.py       # AMR vs uniform meshes
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

The same case validates [AMR]({{root}}modules/AMR.html): `cases/validation/sod_amr.py` runs it
with 10×2 patches and 4 levels (pre-refined at t = 0) and on uniform meshes at the two finest
levels.

<figure>
<img src="{{root}}img/sod_amr.png" style="width:100%">
<figcaption>AMR vs uniform 80×16 and 160×32 meshes at t = 0.2. AMR keeps level 4 around
the contact and the shock and reaches the error of the 160×32 mesh (L1 1.5e-2 vs 1.4e-2) with
53% of its cells.</figcaption>
</figure>

The refinement follows the waves. `cases/validation/sod_amr_animation.py` runs a 40×4 base mesh with 8×2
patches and three levels, remeshing every two steps: the finest patches sit on the contact and the
shock, the rarefaction fan is refined where it is steepest, and each patch is released once the wave has
moved on.

<figure>
<img src="{{root}}img/sod_amr_animation.gif" style="width:100%">
<figcaption>Density at the AMR cells (coloured by level) against the exact solution, and the
patches below and the mesh with its cell edges at the bottom, for $0 < t \le 0.2$. At $t = 0.1$ the 178 patches hold 2 848 cells instead of the
10 240 of the uniform finest mesh.</figcaption>
</figure>

## Reference

* Sod, G.A. (1978). A survey of several finite difference methods for systems of
  nonlinear hyperbolic conservation laws. *J. Comput. Phys.* 27(1), 1-31.
