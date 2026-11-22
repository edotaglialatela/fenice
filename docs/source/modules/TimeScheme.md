---
title: TimeScheme - time integration
nav: TimeScheme
section: Methods
order: 50
---

> **In short.** `time_scheme` picks the integrator: explicit Euler, SSP-RK2, SSP-RK3 or MUSCL-Hancock. The time step follows from `cfl`
> and the fastest wave in the domain, so you normally set only the CFL number (default 0.5). Use SSP-RK3 with the WENO schemes
> (they need it, SSP-RK2 is rejected); the CFL values that kept each WENO scheme stable on the strong-shock tests are in the [limits of the Schemes page]({{root}}modules/Schemes.html#limits).

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

**Viscous flows.** The convective and the viscous (diffusive) eigenvalues of the explicit operator
share one stability budget, and the viscous one is counted twice, so the step must satisfy
$\Delta t\,(2 S_{\rm adv}/h + 2/\Delta t_{\rm visc}) \le 2\,\mathrm{CFL}$, with the parabolic limit
$\Delta t_{\rm visc} = 1/(2\,D \sum_d \Delta x_d^{-2})$ ($\Delta x^2/4D$ on a square 2-D cell). The residual
callback folds it into the speed, cell by cell,

$$S = \max_d|u_d| + c + \frac{h}{\Delta t_{\rm visc}}, \qquad D = \max\!\Bigl(\frac43\frac{\mu}{\rho}, \frac{k}{\rho\, c_v}\Bigr),$$

where $D$ is the diffusivity of the fastest mode of the viscous stencil (a frozen-coefficient Fourier
analysis on a uniform mesh: the longitudinal stress mode and the energy mode; the decay rate is bounded by
$4D\sum_d\Delta x_d^{-2}$),

and takes the maximum over the cells, so it is the same on every MPI strip and AMR rank. A viscosity
that goes to zero leaves $S$ at the convective wave speed. Taking only the smaller of the two
limits is not enough: at CFL 0.5 the convective part alone uses the whole 2-D budget, and a viscous
flow with a cell Reynolds number of order one (Taylor-Green at $\nu = 0.1$) then loses energy too fast
or goes unstable. Adding the two eigenvalues once is not enough either, because both are already at their
marginal limit at CFL 0.5: a manufactured solution with $\mu = 0.02$ goes unstable at CFL 0.5 with the plain
sum and is stable up to CFL 0.7 with the viscous term counted twice, which is what the solver does (for pure
diffusion the step is then $\mathrm{CFL}\cdot\Delta t_{\rm visc}$).

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

## SSP-RK3

The three-stage, third-order strong-stability-preserving method of Shu and Osher:

$$\mathbf U^{(1)} = \mathbf U^n + \Delta t\,\mathcal R(\mathbf U^n), \qquad
  \mathbf U^{(2)} = \tfrac34\mathbf U^n + \tfrac14\bigl(\mathbf U^{(1)} + \Delta t\,\mathcal R(\mathbf U^{(1)})\bigr), \qquad
  \mathbf U^{n+1} = \tfrac13\mathbf U^n + \tfrac23\bigl(\mathbf U^{(2)} + \Delta t\,\mathcal R(\mathbf U^{(2)})\bigr)$$

The residual is evaluated at $t$, $t+\Delta t$ and $t+\Delta t/2$ (time-dependent sources see these stage times). It has the same
CFL limit as SSP-RK2 for the Euler step and is the integrator that goes with the WENO reconstructions
([Schemes]({{root}}modules/Schemes.html)): SSP-RK2 is second order in time and is refused with them. Three residual evaluations per step.
The adaptive-mesh path does not take it yet.

## MUSCL-Hancock

A one-stage, second-order scheme for `reconstruction = muscl`. Each cell first advances its own limited face states by half a step with its
own flux difference (the predictor),

$$\mathbf U^{\pm}_{f} \leftarrow \mathbf U^{\pm}_{f} - \frac{\Delta t}{2}\Bigl(\frac{\mathbf F(\mathbf U^{+}_{x}) - \mathbf F(\mathbf U^{-}_{x})}{\Delta x} + \frac{\mathbf F(\mathbf U^{+}_{y}) - \mathbf F(\mathbf U^{-}_{y})}{\Delta y} + \dots\Bigr) + \frac{\Delta t}{2}\,\mathbf S,$$

then one Riemann problem per face is solved with the predicted states and the cells are updated once, $\mathbf U^{n+1} = \mathbf U^n + \Delta t\,\mathcal R$. So a
step costs one flux evaluation and one Riemann solve per face, where SSP-RK2 needs two of each. The details:

* A predicted state that is not admissible (density or pressure below its limit) stays as the limiter left it. Cells of an immersed body are not
  predicted, and a side that faces a body keeps its limited state. Wall ghost states are made from the predicted state, which keeps a closed box
  conservative.
* Body forces and sources enter the predictor (sources at $t + \Delta t/2$) and the corrector; the viscous term only the corrector.
* The step uses the largest wave speed of the previous step, so a step needs no extra pass; the first step of a run (or of a hot start from another
  scheme) seeds it, and a checkpoint stores it (key `S`).
* `time_scheme = auto` (the default) means MUSCL-Hancock with `reconstruction = muscl` and explicit Euler otherwise. The scheme needs MUSCL (the WENO
  schemes keep SSP-RK3) and is refused with `positivity_fallback`.
* It runs on uniform meshes (2-D and 3-D, OpenMP, MPI strips along y or x) and with [AMR]({{root}}modules/AMR.html) (quadtree and octree, MPI ranks). It gives the same bits for any thread count, rank count and
  SIMD setting, and an AMR run at level 0 equals the uniform run. Fused and two-pass residuals agree bit for bit.
* A wall modelled by an immersed boundary carries a small remaining offset of order $\Delta t$ (cylinder at Re 40: 0.08 % in the drag at CFL 0.25, 0.14 % at CFL 0.5, against SSP-RK2).

On the desktop CPU it was tested on, a step took about 0.6-0.7 of the SSP-RK2 time on large 2-D meshes and 0.7-0.8 in 3-D (a 512x512 mesh: 0.66 on one thread, parity at 16 threads).

{{widget:timeScheme}}

---

*Declarations: [`fv/TimeScheme.hpp`]({{root}}api/TimeScheme.html).*
