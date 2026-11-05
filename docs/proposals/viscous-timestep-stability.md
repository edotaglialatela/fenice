# Stability study for the explicit viscous time step

## Result

The current viscous CFL blend is supported by useful case experiments, but its
diffusion coefficient is not the spectral bound of Fenice's compressible
viscous operator. A frozen-coefficient Fourier analysis of the implemented
uniform-grid stencil gives the following conservative viscous decay-rate
bound:

$$
\Lambda_{\mathrm{visc}} =
4\max\!\left(\frac{4}{3}\nu,\;\frac{\kappa}{\rho c_v}\right)
\sum_d \frac{1}{h_d^2},
\qquad \nu=\frac{\mu}{\rho}.
$$

For Fenice's ideal-gas transport coefficients,
\(\kappa=\mu\gamma/[\mathrm{Pr}(\gamma-1)]\), so
\(\kappa/(\rho c_v)=\gamma\nu/\mathrm{Pr}\). The current speed uses
\(\max(\nu,\kappa/(\rho c_p))=\max(\nu,\nu/\mathrm{Pr})\) instead. It
therefore leaves out the \(4/3\) longitudinal-stress factor and uses \(c_p\)
where the high-frequency energy mode is governed by \(c_v\).

The candidate combined explicit step is

$$
\Delta t = \min\!\left(\Delta t_{\mathrm{end}},\;
\frac{\mathrm{CFL}\,L}
{H+2L D_*\sum_d h_d^{-2}}\right),
\quad
D_* = \max\!\left(\frac{4}{3}\nu,\frac{\kappa}{\rho c_v}\right),
\quad H=\max_d(|u_d|+c),
$$

with the same \(L\) already used by the solver: \(\min(h_x,h_y)\) in 2-D
and \(2/(h_x^{-1}+h_y^{-1}+h_z^{-1})\) in 3-D. In the corresponding rate
form, \(\Lambda_{\mathrm{adv}}=2H/L\) and the step satisfies
\(\Delta t(\Lambda_{\mathrm{adv}}+\Lambda_{\mathrm{visc}})\le2\,\mathrm{CFL}\).
Keeping \(\mathrm{CFL}\le0.5\) makes this normalized combined budget at most
one. That is a conservative common setting for the explicit schemes; it does
not claim to be the exact stability boundary of every reconstructed flux and
boundary treatment in the full solver.

This criterion is implemented (2026-11-05): `nu_eff` in the six speed
expressions (`Physics.cpp` x3, `CellFlux.cpp` x2, the AVX2 twin) is now
\(D_*\). For air the energy term dominates (the \(4/3\) term only binds for
\(\mathrm{Pr}>3\gamma/4\)), so the practical change is \(c_p\to c_v\). The
analysis below is exact for the frozen, periodic, uniform-grid viscous
stencil, while the convective HLLC/reconstruction operator, boundaries, IBM,
and AMR need separate amplification or nonlinear stability checks. The
momentum bound via \(\operatorname{tr}C\) is slightly loose (the checkerboard
mode gives \(\nu(4\sum_d h_d^{-2}+\tfrac43\max_d h_d^{-2})\), about 1.14x lower
in 2-D), which is fine for a conservative bound.

## What the code advances

The explicit methods advance \(U_t=\mathcal R(U)\) with Forward Euler,
SSP-RK2, SSP-RK3, or classical RK4. On a uniform mesh, the residual computes
normal viscous derivatives with adjacent-cell face differences and tangential
derivatives by averaging cell-centred central gradients to the face. It adds
Newtonian stress and Fourier heat conduction. The viscous callback returns an
equivalent speed, and the time integrator uses \(\Delta t=\mathrm{CFL}\,L/S\).
(See `cpp/src/Physics.cpp`, `cpp/src/TimeScheme.cpp`,
and `cpp/src/Simulation.cpp`.)

For the current formula, define

$$
D_{\mathrm{old}}=\max\!\left(\nu,\frac{\kappa}{\rho c_p}\right)
=\max\!\left(\nu,\frac{\nu}{\mathrm{Pr}}\right).
$$

The callback's local speed is \(H+2L D_{\mathrm{old}}\sum_d h_d^{-2}\).
Thus it already combines convection and diffusion in one denominator; the
open question is whether the diffusive rate represented there matches the
implemented equations closely enough and how that budget relates to each time
integrator.

## Fourier analysis of the viscous stencil

Freeze \(\rho,\mu,\kappa\), linearize about a constant state at rest, and use a
periodic Fourier mode with \(\theta_d=k_d h_d\). For each axis, the normal
face-difference followed by a flux divergence has symbol

$$
-A_d,\qquad A_d=\frac{4\sin^2(\theta_d/2)}{h_d^2}.
$$

The averaged transverse derivative on a face contributes

$$
B_{de}=\frac{\sin(\theta_d)\sin(\theta_e)}{h_dh_e},
\qquad B_{de}^2\le A_dA_e.
$$

Let \(A=\sum_d A_d\), and let \(C\) have diagonal entries \(A_d\) and
off-diagonal entries \(B_{de}\). With \(q_d=\sin(\theta_d)/h_d\),

$$
C=\operatorname{diag}(A_d-q_d^2)+q q^T.
$$

Since \(A_d\ge q_d^2\), \(C\) is positive semidefinite and
\(\lambda_{\max}(C)\le\operatorname{tr}(C)=A\). The velocity symbol is

$$
\widehat{R}_{u}= -\nu\left(A I+\frac{1}{3}C\right)\widehat{u},
$$

so all its eigenvalues are real and non-positive and their largest decay
rate is at most \((4/3)\nu A\). Since
\(A\le4\sum_d h_d^{-2}\), the momentum part is bounded by
\((16/3)\nu\sum_d h_d^{-2}\).

The heat-flux term diffuses the linearized pressure/energy mode at rate
\((\kappa/\rho c_v)A\), bounded by
\(4(\kappa/\rho c_v)\sum_d h_d^{-2}\). Density has no direct viscous flux.
Taking the larger of these two bounds gives \(\Lambda_{\mathrm{visc}}\) above.

This analysis also explains why using only the scalar heat-equation limit is
incomplete: the stress tensor's longitudinal mode and the compressible energy
equation have different diffusion coefficients. For \(\gamma=1.4\) and
\(\mathrm{Pr}=0.72\), for example, \(D_{\mathrm{old}}=1.389\nu\), while
\(D_*=1.944\nu\), a factor of 1.4. With the current formula at
\(\mathrm{CFL}=0.5\), the pure-diffusion Fourier bound is
\(\Delta t\Lambda_{\mathrm{visc}}\le D_*/D_{\mathrm{old}}=1.4\). Ignoring
the additional acoustic restriction in \(H\), this is inside Forward Euler's
negative-real stability interval (length 2), but it is not the exact rate the
current coefficient advertises. The old formula's observed stability near
\(\mathrm{CFL}=0.7\) is consistent with this bound being close to the Forward
Euler endpoint; it does not establish a safe limit for other states or
operators.

The repository's existing empirical anchor is narrower: the
`taylor_green_viscous_time_step_budget` case in `cpp/tests/test_periodic.cpp`
uses periodic 64-by-64 SSP-RK2 runs at \(\mu=0.1\) and \(0.2\), and requires
the velocity error at \(t=2\) to stay below \(5\times10^{-3}\). The test
comment records that using only the smaller of the convective and viscous
steps failed at CFL 0.5. The module notes also report that a manufactured
solution at \(\mu=0.02\) needed the viscous term counted twice. These results
support a combined budget, while covering only selected 2-D cases and
integrators.

## Time-integrator dependence

For a scalar mode \(y'=\lambda y\), each explicit scheme multiplies the mode
by its stability polynomial \(R(z)\), where \(z=\Delta t\lambda\). Their
negative-real-axis stability endpoints are:

| Integrator | Stability polynomial | Maximum \(-z\) on the negative real axis |
|---|---|---:|
| Forward Euler | \(1+z\) | 2 |
| SSP-RK2 | \(1+z+z^2/2\) | 2 |
| SSP-RK3 | \(1+z+z^2/2+z^3/6\) | 2.512745 |
| Classical RK4 | \(1+z+z^2/2+z^3/6+z^4/24\) | 2.785294 |

For pure diffusion, the exact linear criterion is
\(\Delta t\,\lambda_{\max}\le r_-\), using the endpoint for the chosen
integrator. Using the conservative \(\Lambda_{\mathrm{visc}}\) bound gives
\(\Delta t\le r_-/\Lambda_{\mathrm{visc}}\). The larger SSP-RK3 and RK4
negative-axis intervals can permit larger pure-diffusion steps, but they do
not directly give a safe mixed convection/diffusion step: advective modes are
complex, and stability requires \(|R(\Delta t\lambda)|\le1\) for every
eigenvalue, not merely a bound on its magnitude or real part.

Published advection-diffusion analyses likewise derive different stable
limits for different spatial advection operators and time integrators, then
check the resulting estimates against numerical amplification results and
Navier-Stokes examples ([Saad and Karam, 2022](https://doi.org/10.1016/j.compfluid.2022.105564)).
That work is a useful methodological reference; its formulas do not directly
cover Fenice's HLLC flux, its reconstruction choices, or this viscous stencil.

## Validation against Fenice cases

The coefficient's practical effect was evaluated on the 64-by-64 Taylor-Green
case. At its initial state, \(p=100\), \(\rho=1\), \(u_{\max}=1\),
\(\gamma=1.4\), and \(\mathrm{Pr}=0.72\), so \(H=12.832\) and
\(L=2\pi/64\). The table compares the old and candidate speed terms at that
state. The adjusted CFL column gives the old-formula CFL that produces the
same first-step size as candidate \(\mathrm{CFL}=0.5\).

| \(\mu\) | Old viscous share of speed | Candidate / current \(\Delta t\) | Equivalent current CFL |
|---:|---:|---:|---:|
| 0.001 | 0.44% | 0.9982 | 0.499 |
| 0.02 | 8.10% | 0.9686 | 0.484 |
| 0.1 | 30.60% | 0.8909 | 0.445 |
| 0.2 | 46.86% | 0.8421 | 0.421 |

This quantifies the hyperbolic-dominated regime: at the default viscosity the
coefficient correction changes the step by about 0.2%; for the shipped
Taylor-Green viscosity it changes the initial step by about 3.1%. The effect
becomes material as the viscous speed approaches the acoustic speed.

I ran the existing `taylor_green_viscous_time_step_budget` check. Its periodic
SSP-RK2 runs at \(\mathrm{CFL}=0.5\) completed with relative velocity errors
\(9.51\times10^{-4}\) at \(\mu=0.1\) and \(5.21\times10^{-4}\) at
\(\mu=0.2\).

I also ran the full solver at \(64^2\), \(t=2\), SSP-RK2, with
\(\mu=0.02\) and \(0.2\). To approximate the candidate without changing
solver code, I repeated each run using the equivalent reduced current-formula
CFL from the table. This is an initial-state match; the candidate CFL ratio
varies slightly as the solution changes.

| \(\mu\) | Current CFL | Steps | Relative velocity \(L_2\) error | Reduced-CFL proxy | Steps | Relative velocity \(L_2\) error |
|---:|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.500 | 568 | \(2.1678\times10^{-3}\) | 0.484 | 587 | \(2.1675\times10^{-3}\) |
| 0.2 | 0.500 | 972 | \(5.2058\times10^{-4}\) | 0.421 | 1154 | \(5.2070\times10^{-4}\) |

Both settings remained finite and produced essentially identical velocity
errors. Matching the proposed correction costs about 3.3% more steps at
\(\mu=0.02\) and 18.7% more at \(\mu=0.2\), with no accuracy gain in this
smooth-flow comparison.

## Decision

\(D_*\) is adopted: it is the conservative Fourier bound of the implemented
stencil and does not depend on \(\gamma/\mathrm{Pr}\) staying near air values.
At the default \(\mathrm{CFL}=0.5\) and \(\gamma=1.4\), \(\mathrm{Pr}=0.72\)
the old estimate already kept \(\Delta t\Lambda_{\mathrm{visc}}\le1.4\), so the
change is a margin, not a stability fix: hyperbolic-dominated runs change by
about 0.2%, viscous-heavy ones pay the step-count increase of the table above
(no accuracy change in the smooth-flow comparison).

The 27 viscous regression references moved with the step size and were
regenerated (the inviscid ones are unchanged); unit tests, regression serial
and on 4 ranks, `amr_mpi`, `ibm_ghost` and the MMS quick study pass. The full
reconstructed operator and wall/IBM/AMR closures still need separate checks
before claiming a solver-wide bound. Not addressed: the speed uses the constant
`mu`, so a temperature-dependent viscosity law would need the local value.

## Scope and follow-up

The Fourier bound assumes constant coefficients, periodicity, and a uniform
Cartesian mesh. It does not by itself bound modes introduced by physical-wall
closures, immersed-boundary forcing, coarse/fine AMR interfaces, or nonlinear
coefficient variation. The current viscous flux is explicit, so these effects
must be included in any claim covering every supported configuration.

The validation here covers periodic 2-D Taylor-Green flow with SSP-RK2 and
supports keeping the current default. If a broader parameter range or a new
instability motivates adopting \(D_*\), first compare measured stability
thresholds with the frozen symbol on periodic viscous modes, then cover the
manufactured-solution cases, 2-D/3-D, all explicit integrators, and a boundary
case. Retain the current empirical CFL results as comparison points. AMR needs
a separate check of its face-list stencil and subcycled levels.
