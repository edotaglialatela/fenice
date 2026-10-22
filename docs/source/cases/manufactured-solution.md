---
title: Manufactured solutions - code verification
nav: Manufactured solutions
section: Test cases
order: 100
---

Exact solutions of the full compressible Navier-Stokes equations hardly exist, and the few that do
exercise only some terms. The **method of manufactured solutions** (MMS) turns the problem around: a
smooth solution $U_m(x, y, z)$ is chosen freely, pushed through the equations the solver
discretises, and the *residual* that remains becomes a source term $S$ that makes $U_m$ an exact
solution of the modified equations,

$$\frac{\partial U}{\partial t} + \nabla\cdot\bigl(F - F_v\bigr) = S, \qquad S = \nabla\cdot\bigl(F(U_m) - F_v(U_m)\bigr),$$

with $U = [\rho, \rho u, \rho v, \rho E, \rho w]$, the ideal-gas energy
$\rho E = p/(\gamma - 1) + \tfrac12\rho|\mathbf u|^2$, the Stokes stress
$\tau_{ij} = \mu(\partial_j u_i + \partial_i u_j) - \tfrac23\mu\,\delta_{ij}\nabla\cdot\mathbf u$ and the Fourier heat flux
$\mathbf q = -k\nabla T$, $T = p/\rho$, $k = \mu\gamma/(\mathrm{Pr}(\gamma - 1))$. The solution does not have to be
physical, only smooth and non-trivial. The code is then run on a sequence of grids and the error against
$U_m$ must fall at the order of the scheme. Roache (2009, chapter 3) calls this the gold standard of
code verification: it tests the discretisation of every term, the order of the scheme and the
encoding together, with no exact solution of the real problem needed.

## The source term in the solver

The source is part of the case file ([Parameters]({{root}}guide/parameters.html#source-terms-manufactured-solutions)):

```toml
[source]
source_rho = "..."     # S for each slot of the state; numbers or expressions of x, y, z, pi, e and the parameters
source_mx  = "..."
source_my  = "..."
source_E   = "..."
```

Each expression is evaluated once per cell, at the cell centre, on the mesh the residual is assembled on (so MPI
strips work), and added to the residual of every stage. It is steady: it has no time dependence, so the time
accuracy is checked separately (the time-scheme order tests of `fenice_tests`). Sources are not available with AMR or
the immersed boundary.

## The manufactured solution

The solution lives on the periodic unit square (cube) with $k = 2\pi$. It is **non-separable**: every variable has its
own wave numbers and phases, and the arguments of the sines and cosines mix the coordinates, so that no
cross-derivative term of the truncation error cancels. In two dimensions

$$\begin{aligned}
\rho &= 1 + 0.2\,\sin(kx + 0.4)\cos(2ky - 0.3),\\
u &= 0.3\,\sin\bigl(k(x + y) + 0.5\bigr) + 0.1\cos(2kx),\\
v &= -0.25\cos\bigl(k(x - y) + 0.2\bigr) + 0.1\sin(2ky + 0.9),\\
p &= 1 + 0.2\cos(kx - 2ky + 0.6) + 0.1\sin(ky + 0.3),
\end{aligned}$$

and in three dimensions $\rho$, $u$, $v$, $p$ get $z$-dependent factors and terms, and
$w = 0.2\sin(k(y + z) + 0.3) + 0.1\cos(kx - kz + 0.7)$ (the exact formulas are in `python/fenice/mms.py`). The
Mach number stays below 0.5, $\nabla\cdot\mathbf u \ne 0$ and $\nabla T \ne 0$ everywhere, and $\mu = 0.02$,
$\mathrm{Pr} = 0.72$, $\gamma = 1.4$: the viscous source is a sizeable fraction of the convective one, so a viscous
error cannot hide below the convective one.

## The cases

| Case (`--case`) | Equations | Default mesh |
|---|---|---|
| `Manufactured Solution` | 2-D Navier-Stokes | 32² |
| `Manufactured Solution Euler` | 2-D Euler (no viscous terms in the source) | 32² |
| `Manufactured Solution 3D` | 3-D Navier-Stokes | 16³ |
| `Manufactured Solution Walls` | 2-D Navier-Stokes, moving wall below, slip wall above, periodic in x | 32² |
| `Manufactured Solution Outflow` | 2-D Navier-Stokes, no-slip wall on the left, transmissive on the right, periodic in y | 32² |

```powershell
build\release\fenice.exe --case "Manufactured Solution" --set nx=64 --set ny=64
python cases\validation\verification\mms.py            # the verification study (quick mode)
python cases\validation\verification\mms.py --full     # all grids, the bands as fixed
python -m fenice.mms --write                           # regenerate the case files
python -m fenice.mms --check                           # are the committed case files what the generator writes?
```

The case files are **generated** (`fenice.mms`, which needs sympy): the source expressions are the sympy
derivatives of the solution, written with `pow()` and parenthesised so that the expression parser's precedence
rules cannot matter, and are up to 34 000 characters long. Do not edit them by hand: change the solution in
`mms.py` and regenerate. The study checks that the committed files match the generator.

## The study

`verification/mms.py` runs four steps on all five cases. The expected orders were fixed before the first run (for the cases with boundaries they are the periodic ones,
unchanged): first order is
$1.0 \pm 0.1$ in every norm; for MUSCL, between 1.5 and 2.5 in $L_1$ and $L_2$ and between 1.0 and 2.5 in $L_\infty$
(a TVD limiter clips the reconstruction at smooth extrema, which costs order, most of all in $L_\infty$; above
2.5 would be suspicious). The orders are those of the finest pair of grids.

The default run is a **quick mode** (grids up to 64² and 48³, a few minutes) meant to catch gross regressions:
it allows 0.1 below the lower bounds and enforces no upper bound, because the coarse grids are still pre-asymptotic.
`--full` adds 128² and 96³ (about 20 minutes), enforces the bands as fixed above and runs the diagnostic below; the
figure and the table of this page come from it.

0. **The case files are what the generator writes.**
1. **Transcription of the source.** On a uniform state, one Explicit Euler step must move every cell by
   $\Delta t\,S(x, y, z)$, with $S$ evaluated independently by sympy. This isolates the parsing of the
   expressions, their place in the state and the cell-centre positions from everything else. It holds to
   $5\cdot10^{-14}$ against sources of size 20 to 32.
2. **Truncation error.** From the exact state, one Euler step gives $(U_1 - U_0)/\Delta t = R_h(U_m) + S$, the local
   truncation error. It must fall with refinement: a floor would mean that the generator's equations differ from
   the solver's. It falls at 0.90 to 0.99 for the first-order scheme and at 1.4 to 1.55 for MUSCL (the clipped extrema,
   $h^{1/2}\cdot h$ in the rms).
3. **Order of accuracy.** The run starts from the exact state and the error of $\rho$, $u$, $v$, ($w$), $p$ is
   measured at $t = 0.5$ on grids refined by 2 in every direction (16² to 128², and 12³ to 96³; the 2-D cases with boundaries use the same grids) with the CFL fixed,
   in the $L_1$, $L_2$ and $L_\infty$ norms. This is a fixed-time error of a steady manufactured solution, not a
   steady-state error, and the time step shrinks with the mesh, so the order is the combined one of the scheme
   (MUSCL with Van Leer's limiter, SSP-RK2; or Godunov with SSP-RK2).

<figure>
<img src="{{root}}img/mms.png" style="width:100%">
<figcaption>Left: truncation error from the exact state. Middle and right: $L_2$ and $L_\infty$ error at
$t = 0.5$ of the worst variable; the legend gives the observed order of the finest pair of grids.</figcaption>
</figure>

Observed orders (finest pair), from `results/validation/mms/mms_table.md`:

| case | scheme | norm | rho | u | v | w | p | expected | |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Manufactured Solution | MUSCL | L1 | 2.01 | 2.11 | 2.07 | - | 2.10 | 1.5 to 2.5 | ok |
| Manufactured Solution | MUSCL | L2 | 1.96 | 2.13 | 2.09 | - | 2.02 | 1.5 to 2.5 | ok |
| Manufactured Solution | MUSCL | Linf | 1.59 | 2.20 | 1.91 | - | 1.65 | 1.0 to 2.5 | ok |
| Manufactured Solution | 1st order | L1 | 0.98 | 0.95 | 0.95 | - | 0.98 | 0.9 to 1.1 | ok |
| Manufactured Solution | 1st order | L2 | 0.98 | 0.94 | 0.95 | - | 0.98 | 0.9 to 1.1 | ok |
| Manufactured Solution | 1st order | Linf | 0.97 | 0.93 | 0.93 | - | 1.00 | 0.9 to 1.1 | ok |
| Manufactured Solution Euler | MUSCL | L1 | 2.30 | 2.18 | 2.14 | - | 2.20 | 1.5 to 2.5 | ok |
| Manufactured Solution Euler | MUSCL | L2 | 2.28 | 2.16 | 2.15 | - | 2.21 | 1.5 to 2.5 | ok |
| Manufactured Solution Euler | MUSCL | Linf | 1.93 | 1.81 | 1.82 | - | 2.04 | 1.0 to 2.5 | ok |
| Manufactured Solution Euler | 1st order | L1 | 0.91 | 0.91 | 0.91 | - | 0.73 | 0.9 to 1.1 | outside (explained) |
| Manufactured Solution Euler | 1st order | L2 | 0.90 | 0.89 | 0.89 | - | 0.71 | 0.9 to 1.1 | outside (explained) |
| Manufactured Solution Euler | 1st order | Linf | 0.83 | 0.73 | 0.77 | - | 0.46 | 0.9 to 1.1 | outside (explained) |
| Manufactured Solution 3D | MUSCL | L1 | 2.01 | 2.13 | 2.17 | 2.19 | 2.12 | 1.5 to 2.5 | ok |
| Manufactured Solution 3D | MUSCL | L2 | 1.96 | 2.15 | 2.17 | 2.22 | 2.04 | 1.5 to 2.5 | ok |
| Manufactured Solution 3D | MUSCL | Linf | 1.79 | 2.19 | 2.02 | 2.35 | 1.81 | 1.0 to 2.5 | ok |
| Manufactured Solution 3D | 1st order | L1 | 0.96 | 0.93 | 0.93 | 0.93 | 0.97 | 0.9 to 1.1 | ok |
| Manufactured Solution 3D | 1st order | L2 | 0.96 | 0.93 | 0.93 | 0.93 | 0.97 | 0.9 to 1.1 | ok |
| Manufactured Solution 3D | 1st order | Linf | 0.95 | 0.91 | 0.90 | 0.92 | 0.99 | 0.9 to 1.1 | ok |
| Manufactured Solution Walls | MUSCL | L1 | 1.09 | 1.38 | 1.16 | - | 1.56 | 1.5 to 2.5 | outside (explained) |
| Manufactured Solution Walls | MUSCL | L2 | 1.09 | 1.39 | 1.25 | - | 1.45 | 1.5 to 2.5 | outside (explained) |
| Manufactured Solution Walls | MUSCL | Linf | 0.83 | 1.22 | 1.30 | - | 0.99 | 1.0 to 2.5 | outside (explained) |
| Manufactured Solution Walls | 1st order | L1 | 0.95 | 0.97 | 0.95 | - | 0.96 | 0.9 to 1.1 | ok |
| Manufactured Solution Walls | 1st order | L2 | 0.96 | 0.96 | 0.95 | - | 0.97 | 0.9 to 1.1 | ok |
| Manufactured Solution Walls | 1st order | Linf | 1.00 | 0.84 | 0.95 | - | 0.99 | 0.9 to 1.1 | outside (explained) |
| Manufactured Solution Outflow | MUSCL | L1 | 1.95 | 1.14 | 1.31 | - | 1.63 | 1.5 to 2.5 | outside (explained) |
| Manufactured Solution Outflow | MUSCL | L2 | 1.68 | 1.01 | 1.15 | - | 1.49 | 1.5 to 2.5 | outside (explained) |
| Manufactured Solution Outflow | MUSCL | Linf | 0.94 | 0.73 | 1.14 | - | 0.94 | 1.0 to 2.5 | outside (explained) |
| Manufactured Solution Outflow | 1st order | L1 | 0.98 | 0.96 | 0.93 | - | 0.97 | 0.9 to 1.1 | ok |
| Manufactured Solution Outflow | 1st order | L2 | 0.98 | 0.96 | 0.94 | - | 0.98 | 0.9 to 1.1 | ok |
| Manufactured Solution Outflow | 1st order | Linf | 0.96 | 0.96 | 0.95 | - | 0.97 | 0.9 to 1.1 | ok |

* **Navier-Stokes, second order.** MUSCL gives 1.96 to 2.13 in $L_1$ and $L_2$ in 2-D and 1.96 to 2.22 in 3-D;
  $L_\infty$ is lower, as expected (1.6 to 2.35), and the first-order scheme gives 0.90 to 1.00. The viscous terms,
  the cross-derivatives, the heat flux, the periodic seams and the 3-D stencils are all exercised by the same
  solution.
* **Euler, second order.** 2.14 to 2.30 in $L_1$ and $L_2$ and 1.8 to 2.0 in $L_\infty$.
* **Euler, first order: outside the band, explained.** The orders approach 1 from below ($\rho$: 0.83, 0.90, 0.91, 0.92
  over the five grids up to 256²; $p$: 1.00, 0.79, 0.73, 0.81), independently of the CFL number and of the time
  scheme, so the cause is spatial. A fit $e = a h + b h^2$ describes the density error to 8 % ($1.46h - 4.88h^2$):
  the leading term is first order and an $h^2$ term of opposite sign competes with it, the behaviour Roache (2009,
  sec. 8.1.1) describes for first-order methods. The pressure error is less regular (29 % misfit), so its asymptotic
  order is not established on these grids. The truncation error of the same case falls at 0.95 to 0.99, so the scheme
  is first-order consistent. The band was not moved; the table marks the entries "outside (explained)".

## Solutions with boundaries

A periodic solution cannot test a boundary condition. The two cases with boundaries are built so that the **ghost
state the solver puts across the boundary is exactly the extension of the manufactured solution**: the variables
are even or odd about the boundary the way the boundary condition mirrors them, and `fenice.mms` checks the parities
numerically when it builds the solution (it refuses to write a case that violates them).

| Case | Side | Boundary condition | Condition on the solution at the boundary |
|---|---|---|---|
| Walls | bottom, $y = 0$ | `moving_wall`, $u_{in} = 0.1$ | $\rho$, $p$ even, $v$ odd, $u - 0.1$ odd |
| Walls | top, $y = 1$ | `wall` (slip) | $\rho$, $p$, $u$ even, $v$ odd |
| Outflow | left, $x = 0$ | `no_slip` | $\rho$, $p$ even, $u$, $v$ odd |
| Outflow | right, $x = 1$ | `transmissive` | $\rho$, $p$, $u$, $v$ even (zero normal gradient) |

The other axis is periodic, so there are no corners. The parities come from $\cos(\pi y)$ (even about both walls),
$\sin(\pi y)$ (odd about both) and $\sin(\pi y/2)$ (odd about 0, even about 1), mixed into the phases of the
periodic waves so that the solution stays non-separable. The momentum slot of a wall is the mirrored one, so the
no-slip, moving-wall and slip-wall ghosts, the viscous padded state and the adiabatic heat flux at the wall are all
compared with the exact solution. The cases are in the regression too (`mms_walls`, and `mms_outflow_xdecomp`
on x-strips), serial and on 2 to 4 MPI ranks.

**What the study found.** With the first-order scheme the orders are those of the periodic cases (0.9 to 1.0; one
$L_\infty$ entry at 0.84). With MUSCL they are **not**: $L_1$ and $L_2$ fall at 1.0 to 1.9 instead of 2, and $L_\infty$
at 0.7 to 1.3, and the maximum error sits in the cell next to the wall (the study prints where it is). The bands were
not moved. A floor of 0.7, set after the first run, keeps a worse loss from passing; the entries are marked
"outside (explained)". Three things in the solver explain it (the first two were seen in the code, the third by
an experiment):

* the face on a non-periodic boundary is not reconstructed: it takes the interior and the ghost state as they
  are (first order), and the first interior face has a zero slope on the boundary side because its stencil is
  filled by replicating the edge cell;
* an inviscid version of the walls case (not committed) loses order the same way, so this part is convective: $L_1$
  1.8, $L_\infty$ 0.5 to 1.3;
* the viscous padded state of a **moving wall** flips the momentum about the wall velocity but copies the energy, so
  when the wall moves the ghost temperature is not the interior one and the wall is not exactly adiabatic. With the
  wall velocity set to zero the $L_1$ orders of the walls case recover from 1.1 to 1.4 to about 1.6. A no-slip or slip
  wall ($|u|$ unchanged by the mirror) has no such inconsistency. The velocity $L_1$ orders of the outflow case
  (1.0 to 1.3) are still lower with viscosity than without (1.4 to 1.5), so the viscous wall closure has a further
  loss that was not traced.

A first-order boundary face is a common closure and global order 2 is still reached where the normal gradients
vanish, as they do here by construction; a flow with a non-zero gradient at the boundary would lose more. These are
properties of the solver, not of the test: changing them is a change of the numerics, so they are reported here.

What the two cases leave untested: no-slip on the bottom/top, slip and moving walls on the left/right (separate
branches of the boundary code), the inlets and the outlets (their ghost states set a value instead of mirroring one,
so they are first order by construction and no manufactured solution matches them), the non-reflecting outlet, and
corners where two non-periodic sides meet.

## What it does not verify

* **Most boundary conditions.** Only the wall, moving-wall, no-slip and transmissive ghosts are verified (see above).
  Roache (sec. 3.4) shows how one solution also serves Dirichlet, Neumann and Robin conditions with boundary
  data taken from it; the solver has no boundary condition that reads its value from an expression, so the inlets and
  outlets are checked by the analytic cases (Poiseuille, cavity) and the non-reflecting-outlet tests instead.
* **Conservation.** The method verifies the order, not that the scheme conserves exactly (Roache, sec. 3.4): the
  conservation tests of `fenice_tests` do.
* **Shocks and discontinuities.** The solution is smooth; the treatment of shocks by the limiters and the Riemann
  solvers is checked by Sod and Shu-Osher against the exact and reference solutions (Roache, sec. 3.7).
* **AMR, the immersed boundary, time accuracy.** Sources are not available with AMR or the IBM, and the source is
  steady.

## Adding a manufactured solution

```python
from fenice import mms
sol = mms.Solution(dim=3, viscous=True, mu=0.05)     # constants of the equations
sol.write_cases("my_cases")                         # my_cases/<name>/case.toml (not "cases": --check compares those)
rho, u, v, w, p = sol.exact()(X, Y, Z)                # numpy fields, for the error
S = sol.source_functions()["source_E"]                # the same source as a numpy function
```

To use another solution, change the expressions in `Solution.__post_init__` (smooth, periodic on the unit domain,
$\rho$ and $p$ away from zero, Mach number moderate, all variables varying in all coordinates) and rerun the study.

### References

* Roache, P. J. (2009). *Fundamentals of Verification and Validation*. Hermosa Publishers, chapter 3.
* Roache, P. J. (2002). Code verification by the method of manufactured solutions. *J. Fluids Eng.* 124(1), 4-10.
* Salari, K. & Knupp, P. (2000). Code verification by the method of manufactured solutions. Sandia National
  Laboratories report SAND2000-1444.
