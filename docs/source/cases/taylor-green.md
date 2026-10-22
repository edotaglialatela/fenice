---
title: Taylor-Green vortex - viscous decay on a periodic domain
nav: Taylor-Green vortex
section: Test cases
order: 70
---

The 2-D Taylor-Green vortex is an exact solution of the incompressible Navier-Stokes
equations: a periodic array of counter-rotating vortices that keeps its shape while
viscosity damps it. On the periodic square $[0, L]^2$ with $k = 2\pi/L$:

$$u = -U_0\cos(kx)\sin(ky)\,F(t), \qquad v = U_0\sin(kx)\cos(ky)\,F(t), \qquad F(t) = e^{-2\nu k^2 t}$$

$$p = p_0 - \frac{\rho_0 U_0^2}{4}\bigl(\cos 2kx + \cos 2ky\bigr)F(t)^2$$

The nonlinear advection balances the pressure gradient exactly, so only viscosity acts,
and the kinetic energy decays as

$$\frac{E_k(t)}{E_k(0)} = e^{-4\nu k^2 t}$$

The case uses $U_0 = 1$, $\rho_0 = 1$ and $p_0 = 100$. This gives
$c \approx 11.8$ and $\mathrm{Ma} \approx 0.085$, so the compressible solver is close to
the incompressible limit. The boundaries are periodic on all four sides.

**What it validates.**

* **The viscous fluxes in isolation.** Running the same case with and without
  `use_viscous` separates the scheme's numerical dissipation from the physical
  viscosity. The difference of the two fitted decay rates must recover $\nu$.
* **Order of accuracy** of the full Navier-Stokes discretisation (MUSCL + central
  viscous fluxes).
* **Periodic boundaries in both directions.** A periodic domain has no special
  location, so the solution must be exactly translation-invariant.

```powershell
build\release\fenice.exe --case "Taylor-Green Vortex"
python cases\validation\verification\taylor_green.py
```

## Default parameters

{{defaults:Taylor-Green Vortex}}

## Validation results

<figure>
<img src="{{root}}img/taylor_green_validation.png" style="width:100%">
<figcaption>Left: kinetic-energy decay at 64², Navier-Stokes vs Euler. The fitted total rate is
ν = 0.0201 and the numerical part 0.0002, so the physical viscosity recovered is 0.0199 against
0.02 (0.3 %). Right: velocity L2 error at t = 1; the observed orders are 1.8, 2.9 and 2.4.</figcaption>
</figure>

<figure>
<img src="{{root}}img/taylor_green_animation.gif" style="width:100%">
<figcaption>The vortex decaying at $
u = 0.1$ (<code>cases/validation/verification/taylor_green_animation.py</code>).
Left: the vorticity $\omega = \partial_x v - \partial_y u$ of the solver as a surface
$z = \omega(x, y)$, with the exact $2k\cos kx \cos ky\, e^{-2
u k^2 t}$ as a wireframe over it.
Right: the kinetic energy against $e^{-4
u k^2 t}$. The surface flattens by a factor of
$e^{-2
u k^2 t}$ while keeping its shape, and at $t = 10$ the vorticity differs from the exact one by
$8 \cdot 10^{-4}$ (peak value 0.27).</figcaption>
</figure>

The explicit time step uses the convective and the viscous stability limits together
([TimeScheme]({{root}}modules/TimeScheme.html)), so the default CFL of 0.5 is stable and accurate
also at high viscosity: the unit test `taylor_green_viscous_time_step_budget` checks the velocity error
at $\nu = 0.1$ and $0.2$.

The automated checks in `fenice_tests` run smaller versions:

* `taylor_green_viscous_decay`: the recovered physical ν must be within 2 % at 32²
  (measured 0.4 %).
* `taylor_green_convergence`: order > 1.7 between 32² and 64² (measured 2.9).
* `periodic_translation_invariance`: shifting the initial field by whole cells must
  shift the solution bit for bit, with MUSCL and viscous terms on or off.

The last check holds because periodic boundaries are not seams: MUSCL reconstructs the
periodic boundary faces from wrap-around halos (2nd order like any interior face), and
the viscous cross-derivatives there use the opposite edge cell.

## Reference

* Taylor, G.I. & Green, A.E. (1937). Mechanism of the production of small eddies from
  large ones. *Proc. R. Soc. Lond. A* 158(895), 499-521.
