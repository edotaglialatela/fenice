---
title: Lid-driven cavity
nav: Lid-driven cavity
section: Test cases
order: 30
---

A square box ($L_x = L_y = 1$) whose top wall moves at $u_{\rm lid}$ drives a primary
recirculation vortex. The Reynolds number is

$$\mathrm{Re} = \frac{\rho\,u_{\rm lid}\,L}{\mu} = 400$$

with the defaults ($\rho = 1$, $u_{\rm lid} = 1$, $\mu = 2.5\times10^{-3}$), a regime where
the flow is steady with a single dominant vortex. The ambient pressure
$p_{\rm out} = 100$ suppresses compressibility: $c = \sqrt{\gamma p/\rho} \approx 11.8$ gives
$\mathrm{Ma} \approx 0.085$. Density variations are therefore $O(\mathrm{Ma}^2) \approx 0.7\%$,
and the results compare directly with the incompressible Ghia (1982) reference. The
walls are no-slip, and the lid is a `moving_wall` with velocity `u_in`.

```powershell
cpp\build\fvsolver.exe --case "Lid Driven Cavity"
python validation\validationDrivenCavity.py
```

## Default parameters

{{defaults:Lid Driven Cavity}}

## Validation results

<div style="display:flex; gap:1em; align-items:flex-start; flex-wrap:wrap;">
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/streamlines_cavity_1stOrder.png" style="width:100%">
<figcaption>Streamlines: 1st-order Godunov.</figcaption>
</figure>
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/streamlines_cavity_2ndOrder.png" style="width:100%">
<figcaption>Streamlines: 2nd-order MUSCL.</figcaption>
</figure>
</div>

<div style="display:flex; gap:1em; align-items:flex-start; flex-wrap:wrap; margin-top:1em;">
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/convergence_cavity_1stOrder.png" style="width:100%">
<figcaption>Grid convergence: 1st order.</figcaption>
</figure>
<figure style="margin:0; flex:1 1 45%;">
<img src="{{root}}img/convergence_cavity_2ndOrder.png" style="width:100%">
<figcaption>Grid convergence: 2nd order.</figcaption>
</figure>
</div>

## Reference

* Ghia, U., Ghia, K.N. & Shin, C.T. (1982). High-Re solutions for incompressible flow
  using the Navier-Stokes equations and a multigrid method. *J. Comput. Phys.* 48(3), 387-411.
