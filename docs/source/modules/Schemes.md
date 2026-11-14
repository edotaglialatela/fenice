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

* Toro, E.F., Spruce, M. & Speares, W. (1994). Restoration of the contact surface in
  the HLL-Riemann solver. *Shock Waves* 4(1), 25-34.
* Toro, E.F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed. Springer.
* Thornber, B. et al. (2008). An improved reconstruction method for compressible flows
  with low Mach number features. *J. Comput. Phys.* 227(10), 4873-4894.
* Borges, R., Carmona, M., Costa, B. & Don, W.S. (2008). An improved weighted essentially non-oscillatory scheme for
  hyperbolic conservation laws. *J. Comput. Phys.* 227(6), 3191-3211.
* Shu, C.-W. & Osher, S. (1988). Efficient implementation of essentially non-oscillatory shock-capturing schemes.
  *J. Comput. Phys.* 77(2), 439-471.
* LeVeque, R.J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press.
* Koren, B. (1993). A robust upwind discretisation method for advection, diffusion and
  source terms. *Notes on Numerical Fluid Mechanics* 45.

{{api}}
