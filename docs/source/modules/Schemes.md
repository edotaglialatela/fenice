---
title: Schemes - Riemann solvers, TVD limiters, MUSCL, WENO
nav: Schemes
section: Modules
order: 30
header: Schemes.hpp
---

## Riemann solvers

At each interface the solver receives the left and right conservative states
$\mathbf{U}_L$, $\mathbf{U}_R$ and returns the numerical flux $\mathbf{F}^*$
that approximates the exact solution of the Riemann problem. With the unit
normal $\hat{\mathbf n} = (n_x, n_y, n_z)$ and $V_n = \mathbf u\cdot\hat n$, the normal Euler flux is

$$\mathbf{F}\cdot\hat n = \bigl[\rho V_n,\ \rho u V_n + p n_x,\ \rho v V_n + p n_y,\ V_n(\rho E + p),\ \rho w V_n + p n_z\bigr]^\top$$

On a Cartesian mesh every face is normal to one axis, so the solvers take the axis
(`face_flux(UL, UR, axis, smax)`, axis 0/1/2 = x/y/z) and use the momentum slot
`kMom[axis]` as the normal one; the other two momenta are tangential and are simply advected.

`RiemannSolver` is stateless and evaluates every face in one OpenMP-parallel loop. It
also returns the largest wave speed $\max_f \max(|S_L|, |S_R|)$, which sets the CFL time step.

### HLLC (Toro, Spruce & Speares 1994)

HLLC estimates three waves: the left acoustic wave $S_L$, the contact $S_M$ and the
right acoustic wave $S_R$. It then builds intermediate "starred" states on each side
of the contact. The Einfeldt wave-speed bounds are

$$S_L = \min(V_{n,L} - c_L,\ V_{n,R} - c_R), \qquad S_R = \max(V_{n,L} + c_L,\ V_{n,R} + c_R)$$

and the contact speed (Toro 2009, eq. 10.37) is

$$S_M = \frac{p_L - p_R + \rho_R V_{n,R}(S_R - V_{n,R}) - \rho_L V_{n,L}(S_L - V_{n,L})}
            {\rho_R(S_R - V_{n,R}) - \rho_L(S_L - V_{n,L})}$$

The flux is selected by wave ordering:

- if $S_L \geq 0$: $\mathbf{F}^* = \mathbf{F}_L$
- if $S_L < 0 \leq S_M$: $\mathbf{F}^* = \mathbf{F}_L + S_L(\mathbf{U}^*_L - \mathbf{U}_L)$
- if $S_M < 0 \leq S_R$: $\mathbf{F}^* = \mathbf{F}_R + S_R(\mathbf{U}^*_R - \mathbf{U}_R)$
- if $S_R < 0$: $\mathbf{F}^* = \mathbf{F}_R$

The scheme resolves isolated shocks, contacts and rarefactions exactly, and
preserves positivity of density and pressure. Near-zero denominators are guarded
with a sign-preserving floor of $10^{-14}$.

{{widget:solveRiemann}}

### Low-Mach HLLC (Thornber et al. 2008)

Standard HLLC adds $O(1)$ dissipation to the velocity jump across each face. At low
Mach number this is far larger than the physical $O(\mathrm{Ma}^2)$ pressure
fluctuation, which causes odd-even decoupling and excessive diffusion. Thornber's fix
scales the velocity jump by the local Mach number:

$$z = \operatorname{clip}\bigl(\max(\mathrm{Ma}_L, \mathrm{Ma}_R),\ 10^{-4},\ 1\bigr), \qquad
  u_L^* = \bar u + z\,\delta u, \quad u_R^* = \bar u - z\,\delta u$$

with $\bar u = \tfrac12(u_L+u_R)$ and $\delta u = \tfrac12(u_L - u_R)$ (and the same for $v$ and $w$).
Standard HLLC is then applied to the modified states. Shock capturing is unaffected,
because $z = 1$ above Mach 1.

### Preconditioned HLLC (Weiss & Smith 1995)

At low Mach number the acoustic speeds $\pm c$ dwarf the flow speed, which makes the
system stiff. Preconditioning replaces $c$ with a reduced reference speed:

$$V_r = \operatorname{clip}\bigl(V_n^2,\ (\mathrm{Ma}_{\rm cut}\,c)^2,\ c^2\bigr), \qquad \beta = \frac{V_r}{c^2},
  \qquad \lambda_\pm = V_n\,\frac{1+\beta}{2} \pm \sqrt{\Bigl(\frac{1-\beta}{2}V_n\Bigr)^2 + V_r}$$

The $\lambda_\pm$ replace the Einfeldt bounds. The solver uses
$\mathrm{Ma}_{\rm cut} = 0.1$; $\mathrm{Ma}_{\rm cut} = 1$ recovers standard HLLC exactly.

## TVD slope limiters

A limiter $\phi(a, b)$ constrains the reconstruction slope so that no new extrema
appear (the **Total Variation Diminishing** property). Its arguments are the upwind
difference $a = W_i - W_{i-1}$ and the downwind difference $b = W_{i+1} - W_i$:

| Limiter | $\phi(a,b)$ | Character |
|---|---|---|
| Minmod | $\tfrac12(\operatorname{sgn}a + \operatorname{sgn}b)\min(\lvert a\rvert, \lvert b\rvert)$ | most diffusive, unconditionally TVD |
| Van Leer | $\dfrac{2ab}{a+b}$ if $ab > 0$, else 0 | harmonic mean, smooth and symmetric (default) |
| Koren | $\operatorname{minmod}\bigl(2a, \tfrac{2a+b}{3}, 2b\bigr)$ | 3rd order in smooth regions |
| MC | $\operatorname{minmod}\bigl(2a, \tfrac{a+b}{2}, 2b\bigr)$ | most compressive, sharpest |

All four return zero at extrema ($ab \le 0$) and stay inside the TVD region
$|\phi| \le 2\min(|a|, |b|)$.

{{widget:limitersTVD}}

## MUSCL reconstruction

MUSCL upgrades the 1st-order Godunov scheme to **2nd-order spatial accuracy**. The
primitive variables are reconstructed linearly, with limiting, on each side of every
face before the Riemann problem is solved. At face $i+\tfrac12$:

$$W_{L,i+1/2} = W_i + \tfrac12\phi(a, b), \qquad W_{R,i+1/2} = W_{i+1} - \tfrac12\phi(a', b)$$

with $a = W_i - W_{i-1}$, $b = W_{i+1} - W_i$ and $a' = W_{i+2} - W_{i+1}$. The stencil
needs two cells on each side of a face (`kHaloDepth = 2`), the same depth as the MPI
halo exchange. Faces on a rank interface are therefore also reconstructed to 2nd
order, and so are periodic boundary faces, through wrap-around halos. The other
physical-boundary faces get ghost layers by default (`boundary_order = 2`: the mirror of the boundary condition) and are reconstructed too; with
`boundary_order = 1`, or with a first-order scheme, they keep their 1st-order ghost states ([measured]({{root}}cases/manufactured-solution.html)). In 3-D,
`reconstruct_3d` applies the same 1-D reconstruction along x, y and z; each direction
reads its own pair of halo slabs (`Halos::l/r`, `b/t`, `f/bk`).

Without limiting ($\phi = b$) the scheme is 2nd order but oscillatory near shocks.
With limiting it is TVD and drops locally to 1st order at extrema.

{{widget:MUSCL}}

## WENO-Z reconstruction

`reconstruction = weno3` or `weno5` replaces the limited linear reconstruction by a
weighted combination of candidate stencils whose weights fall to zero across a discontinuity. The primitive
variables are reconstructed component by component, as MUSCL does, and the rest of the face update (Riemann solver, shock blend,
time integration) is unchanged. The WENO-Z weights (Borges et al. 2008) are

$$\alpha_k = d_k\Bigl(1 + \frac{\tau}{\beta_k + \varepsilon}\Bigr), \qquad \omega_k = \frac{\alpha_k}{\sum_m \alpha_m}, \qquad
  W_{L,i+1/2} = \sum_k \omega_k\,p_k$$

with $d_k$ the weights that make the combination the linear upwind-biased scheme, $\beta_k$ the smoothness indicators of the
stencils, $p_k$ their face values and $\tau$ a global indicator of the stencil.

| | stencil | candidates | $d_k$ | $\tau$ |
|---|---|---|---|---|
| `weno3` | 3 cells $(i-1\ldots i+1)$ | 2 linear, 2 cells each | $\tfrac13, \tfrac23$ | $\lvert\beta_0 - \beta_1\rvert$ |
| `weno5` | 5 cells $(i-2\ldots i+2)$ | 3 quadratic, 3 cells each | $\tfrac1{10}, \tfrac35, \tfrac3{10}$ | $\lvert\beta_0 - \beta_2\rvert$ |

The right state of the face is the mirror image of the left one. $\varepsilon$ is relative to the data of the stencil,
$\varepsilon = \texttt{weno\_epsilon}\cdot\overline{c^2}$: a fixed absolute $\varepsilon$ lowers the order at smooth extrema
to about 2.5 whatever the units of the variable. The reconstruction is dimension by dimension on the cell averages, so the
formal order of a multi-dimensional run is limited by the quadrature and by the nonlinear flux evaluation at the faces to second order, whatever the
one-dimensional order; the higher order shows as a smaller error constant.

**Measured** (1-D entropy wave with exact cell-average data, one period, SSP-RK3, observed order between successive meshes, $16\to128$ cells):
`weno3` with `weno_epsilon = 1e-4` converges at 3.0 on the coarsest pair and 3.4 to 4.3 on the finer ones; `weno5` at 5.0 (CFL 0.1; at CFL 0.5 the time error takes over
and the order falls to 3). The 2-D isentropic vortex at $256^2$: density error $6.4\cdot10^{-5}$ with MUSCL (MC),
$4.8\cdot10^{-5}$ with `weno3` and $2.1\cdot10^{-6}$ with `weno5`. The 3-D Taylor-Green vortex at $64^3$, Re 1600:
peak dissipation $-10.8\,\%$ (MUSCL), $-14.1\,\%$ (`weno3`) and $-0.2\,\%$ (`weno5`) from the reference value 0.0127; at $32^3$ the three
give $-13.3\,\%$, $-5.4\,\%$ and $-8.8\,\%$, so the `weno3` error is not monotone in the resolution and a single peak value is a trend, not a convergence
result. `weno5` takes about 3 to 5 times as long as MUSCL on one thread for the same run (it also uses a three-stage time scheme): 2.7 on this case,
4.8 on the vortex at $256^2$ (table in [Performance]({{root}}guide/performance.html)). `weno3` is not more accurate than MUSCL on the turbulent case.

**Limits.**

* The stencil is as deep as the MUSCL one for `weno3` (so it runs on MPI ranks, with results identical to the serial run bit
  for bit and independent of the thread count); `weno5` reads three cells on each side and is serial.
* Faces on a non-periodic physical boundary keep the first-order ghost states, as with MUSCL.
* There is no positivity limiter. Estimate (2026-11-05, 1-D tubes of `stress_shocks.py` on 400 x 4 cells, HLLC, the suite's CFL): no shock is strong enough to need it. A shock of Mach 30000 into gas at rest (pressure ratio about 1e9) and a 1e10 pressure-ratio Sod tube keep rho and p positive and finite with MUSCL, `weno3` and `weno5` (the smallest pressure is 0.88 of the ambient one, an undershoot of `weno5`). What breaks a WENO scheme is an expansion into near-vacuum, not a shock: for two receding rarefactions with speed +-u, vacuum forms at u = 3.7 (for p = 0.4, rho = 1); `weno3` and MUSCL survive u = 20 (5.3 times that), `weno5` survives u = 8 (2.1 times) and diverges at u = 12 and 20 even at CFL 0.3. The strong-shock suite (Sod with a 1000:1 pressure ratio, the 123 problem, the near-vacuum tube,
  Mach 10, 20 and 100 tubes, the Mach 10 cylinder, the odd-even test) is passed by `weno3` at the suite's CFL (0.4 for the tubes and the odd-even test, the case's own value for the cylinder) and by `weno5` at CFL 0.3.
  `weno5` with SSP-RK3 diverges on the near-vacuum tube at CFL 0.4 (it is stable up to 0.3), so
  use a lower CFL for problems that open a vacuum. A run whose state diverges stops with an error naming the step.
* `weno5` is not available with AMR, and neither WENO scheme with SSP-RK2 (rejected when the case is read).
* `weno3` runs on the adaptive mesh with the same four-cell stencil per face as MUSCL, so the halo, the virtual cells and the
  flux registers are unchanged. On a coarse-fine face both sides are reconstructed with WENO-Z from the data the limited
  formula uses: the fine side from its two fine cells and the interpolated coarse value, the coarse side from its own cell,
  its neighbour away from the face and the mean of the fine leaves. The interpolated value itself still uses the limited
  slopes of the coarse leaf, and those and the averaged values are second order. Measured on the Sod tube (base 50 cells, two levels, static refined band, SSP-RK3): the density L1 error is 2.972e-3, against 2.972e-3 on the uniform 200-cell grid and 1.05e-2 on the uniform 50-cell grid; with the band narrowed so that the waves cross the jumps, the error inside the window stays within 1.6 times the uniform-200 one. Smooth flow across the seams (isentropic vortex, one period, a 40 x 40 base mesh with one level in the strip 2.5 < x < 7.5, so the vortex crosses both seams): the L1 density error is 4.3e-3 with `weno3` (3.1e-3 with MUSCL MC), between the uniform 40 (5.9e-3 and 4.4e-3) and uniform 80 (9.6e-4 and 9.3e-4) grids, in the same ratio to MUSCL as on the uniform grids; no overshoot or loss of order specific to the seam shows, but the error is dominated by the coarse part, so this is a sanity check, not an order measurement. The transmissive ends of a WENO run leak about 1e-10 of the mass (the tails of the waves), also on a uniform grid.
  The level-0 mesh equals the uniform solver bit for bit, and the state is the same on any number of ranks.

### References

* Toro, E.F., Spruce, M. & Speares, W. (1994). Restoration of the contact surface in
  the HLL-Riemann solver. *Shock Waves* 4(1), 25-34.
* Toro, E.F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed. Springer.
* Thornber, B. et al. (2008). An improved reconstruction method for compressible flows
  with low Mach number features. *J. Comput. Phys.* 227(10), 4873-4894.
* Weiss, J.M. & Smith, W.A. (1995). Preconditioning applied to variable and constant
  density flows. *AIAA J.* 33(11), 2050-2057.
* Borges, R., Carmona, M., Costa, B. & Don, W.S. (2008). An improved weighted essentially non-oscillatory scheme for
  hyperbolic conservation laws. *J. Comput. Phys.* 227(6), 3191-3211.
* Shu, C.-W. & Osher, S. (1988). Efficient implementation of essentially non-oscillatory shock-capturing schemes.
  *J. Comput. Phys.* 77(2), 439-471.
* LeVeque, R.J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
* Koren, B. (1993). A robust upwind discretisation method for advection, diffusion and
  source terms. *Notes on Numerical Fluid Mechanics* 45.

{{api}}
