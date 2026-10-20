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
cpp\build\fvsolver.exe --case "Sod Shock Tube"
cpp\build\fvsolver.exe --case "Sod Shock Tube" --set use_muscl=true --set time_scheme=ssprk2
python validation\validationSod.py
```

## Default parameters

{{defaults:Sod Shock Tube}}

## Validation results

<figure>
<img src="{{root}}img/SOD_mpi_1stOrder.png" style="width:100%">
<figcaption>Density at t = 0.2 s: HLLC, 1st-order Godunov, MPI parallel run.</figcaption>
</figure>

<div style="display:flex; gap:1em; align-items:flex-start; flex-wrap:wrap; margin-top:1em;">
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/Sod_1st_vs_2nd_Order.png" style="width:100%">
<figcaption>1st-order vs 2nd-order MUSCL.</figcaption>
</figure>
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/Sod_Limiters_Comparison.png" style="width:100%">
<figcaption>Limiter comparison (Minmod / Van Leer / MC / Koren).</figcaption>
</figure>
</div>

The same case validates [AMR]({{root}}modules/AMR.html): `validation/amrComparisonSod.py`
compares a uniform mesh with an AMR run of equal finest resolution.

<figure>
<img src="{{root}}img/sod_amr_comparison.png" style="width:100%">
<figcaption>Uniform mesh vs AMR: density profile, cell count and L2 error.</figcaption>
</figure>

## Reference

* Sod, G.A. (1978). A survey of several finite difference methods for systems of
  nonlinear hyperbolic conservation laws. *J. Comput. Phys.* 27(1), 1-31.
