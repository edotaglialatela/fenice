---
title: TimeScheme - time integration
nav: TimeScheme
section: Modules
order: 50
header: TimeScheme.hpp
---

Every scheme advances the semi-discrete system
$\mathrm{d}\mathbf U/\mathrm{d}t = \mathcal R(\mathbf U)$ in place. The residual callback
returns $\mathcal R(\mathbf U)$ together with the largest wave speed $S_{\max}$
(acoustic + advective, plus the viscous equivalent speed). The adaptive step follows the
**CFL condition**:

$$\Delta t = \mathrm{CFL}\cdot\frac{h}{S_{\max}}, \qquad
  h = \min(\Delta x, \Delta y)\ \text{(2-D)}, \qquad h = \frac{2}{\Delta x^{-1} + \Delta y^{-1} + \Delta z^{-1}}\ \text{(3-D)}$$

The schemes update all directions at once (unsplit), so the directional Courant numbers
add up. With $h = \min(\Delta x,\Delta y)$ their sum is at most $2\,\mathrm{CFL}$ on a square
2-D cell. The 3-D length gives exactly the same bound, $\sum_d S_{\max}\Delta t/\Delta x_d \le 2\,\mathrm{CFL}$,
so a CFL that is stable in 2-D stays stable in 3-D. Using $\min(\Delta x,\Delta y,\Delta z)$
instead would allow a sum of $3\,\mathrm{CFL}$: the default 3-D Taylor-Green run (CFL 0.5)
blows up after about 30 steps that way.

It is clamped so that the run ends exactly at `t_end`. $S_{\max}$ is taken from the first
stage of each step. `advance_fixed_dt` instead advances by an imposed $\Delta t$; the
[AMR]({{root}}modules/AMR.html) subcycling uses it, because each level's step is dictated
by the hierarchy.

## Explicit Euler

$$\mathbf U^{n+1} = \mathbf U^n + \Delta t\,\mathcal R(\mathbf U^n)$$

First order, with minimal memory; useful for debugging. Stable for $\mathrm{CFL} \le 1$ in
1-D linear advection; use $\mathrm{CFL} \le 0.8$ for nonlinear problems.

## SSP-RK2

The optimal two-stage strong-stability-preserving Runge-Kutta method (Heun) in Shu-Osher form:

$$\mathbf U^{(1)} = \mathbf U^n + \Delta t\,\mathcal R(\mathbf U^n), \qquad
  \mathbf U^{n+1} = \tfrac12\mathbf U^n + \tfrac12\bigl(\mathbf U^{(1)} + \Delta t\,\mathcal R(\mathbf U^{(1)})\bigr)$$

The update is a convex combination of forward-Euler steps. Any TVD property of the
Euler step therefore carries over, which makes SSP-RK2 the natural partner for MUSCL.

## RK4

The classical four-stage scheme, 4th order in time:

$$k_1 = \mathcal R(\mathbf U^n),\quad k_2 = \mathcal R\bigl(\mathbf U^n + \tfrac{\Delta t}2 k_1\bigr),\quad
  k_3 = \mathcal R\bigl(\mathbf U^n + \tfrac{\Delta t}2 k_2\bigr),\quad k_4 = \mathcal R(\mathbf U^n + \Delta t\,k_3)$$

$$\mathbf U^{n+1} = \mathbf U^n + \tfrac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

The linear-advection limit is $\mathrm{CFL} \le 2\sqrt2$; use $\mathrm{CFL} \le 1.2$ for
compressible flow. RK4 costs four residual evaluations per step, against two for SSP-RK2.

## BDF2 with dual time stepping

Physical time uses the 2nd-order backward differentiation formula

$$\frac{3\mathbf U^{n+1} - 4\mathbf U^n + \mathbf U^{n-1}}{2\Delta t_{\rm phys}} = \mathcal R(\mathbf U^{n+1})$$

It is solved by marching the modified residual
$\mathcal R^*(\mathbf U) = \mathcal R(\mathbf U) - (3\mathbf U - 4\mathbf U^n + \mathbf U^{n-1})/(2\Delta t_{\rm phys})$
to zero with explicit-Euler pseudo-time iterations. The pseudo step is sized to the
spectral radius of $\mathcal R^* = \mathcal R - \alpha I$, with $\alpha = 3/(2\Delta t_{\rm phys})$:

$$\Delta\tau = \frac{\mathrm{CFL}\,h}{S_{\max} + \alpha\,h}$$

The inner loop stops after `max_inner_iter` iterations, or when
$\lVert\mathcal R^*\rVert_\infty < \texttt{inner\_tol\_rel}\cdot\lVert\mathcal R^*_0\rVert_\infty$.
This relative criterion is mesh-independent. The first step bootstraps with BDF1
($\mathbf U^{n-1} := \mathbf U^n$). The scheme suits low-Mach flows: setting
$\Delta t_{\rm phys}$ to the convective time scale, instead of the acoustic one, removes
the 100-1000× acoustic CFL penalty. Under MPI the norms are all-reduced, so every rank
leaves the inner loop at the same iteration.

{{widget:timeScheme}}

{{api}}
