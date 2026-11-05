# Three method choices left open: positivity for WENO at strong shocks, corners of two walls, second-order inlets and outlets

Status: proposal (2026-11-05). Nothing here is implemented; each item needs a decision before any code. The three are independent
and can be taken in any order. Every option that touches the solver is opt-in with the default unchanged, so the 59 regression
references and the bitwise MPI checks stay as they are until a default is changed on purpose.

| item | evidence today | recommendation |
|---|---|---|
| 1. Positivity for WENO | the strong-shock suite is passed by `weno3` and `weno5`; one failure: `weno5` diverges on the near-vacuum tube at CFL 0.4 | **B** (face fallback, opt-in), only if a case needs it; otherwise keep the CFL rule |
| 2. Corner of two walls | not measured; the cavity (four corners) and the Ghia comparison are the only indirect evidence | **measure first** (a four-wall manufactured solution), then decide |
| 3. Inlets and outlets with a normal gradient | the prescribed variable is second order; the free ones (rho, p at a velocity inlet; velocity at a pressure outlet) are copied, so first order | **B** (linear extrapolation of the free variables, opt-in) after a manufactured solution that has a gradient there |

## 1. Positivity of the reconstructed states with WENO

**Today.** The reconstruction works on the primitive variables, face by face. MUSCL with a TVD limiter keeps each face value between the
neighbouring cell values, so rho and p stay positive and no protection exists or is needed (decided with B2, see
`benchmarks-robustness-uq.md`). WENO3-Z and WENO5-Z are not bounded: a face value can leave the range of its neighbours. The suite
`stress_shocks.py` (Sod 1000:1, the 123 problem, a vacuum-opening variant, Mach 10 / 20 / 100 tubes, the Mach 10 cylinder, the odd-even
test) is passed by both at the suite's CFL; the smallest values reached are rho 1.8e-3 / 9.1e-4 and p 1.1e-3 / 7.2e-4 on the vacuum tube
(WENO3 / WENO5). The one failure is `weno5` with SSP-RK3 at CFL 0.4 on that tube (stable up to 0.3). The guide already says so.

**Options.**

* **A. Scaling of the face states toward the cell value** (the Zhang-Shu idea, adapted to face values): for each face state, find the
  largest theta in [0, 1] such that rho and p of `cell + theta (face - cell)` stay above a floor, and use it. Keeps the order wherever
  the states are positive already (theta = 1, bit-identical), costs a few operations per face, is smooth in the data. In a
  finite-volume scheme with only face values it protects the states, not the update (positivity of the update still needs a CFL
  bound), so it narrows the failure but does not remove it.
* **B. Face fallback** (a posteriori, local): where either state of a face has rho or p below the floor, replace both by the first-order
  states (the two cell values) for that face. Simple, one test per face, no tuning beyond the floor, bit-identical elsewhere. Loses
  accuracy only where the state would have been invalid. Does not by itself guarantee a positive update either.
* **C. Positivity-preserving flux limiter** (blend each face flux with a first-order flux so the cell update stays positive): the only
  option with a guarantee under a CFL bound; the most intrusive (needs the cell-wise sum of fluxes, which the fused kernels do not
  expose) and adds a pass over the cells. Not worth it for one failing CFL.
* **D. Nothing.** Keep "use CFL <= 0.3 for WENO5 on problems that open a vacuum" in the guide.

**Recommendation.** D for now; B as `positivity_floor = <value>` (0: off, the default) if a real case needs it. B has the smallest code
(the three WENO reconstruction routines and their AVX2 twins, or one pass after the reconstruction), the guarantee A and B give is the
same in practice for the face states, and it is easy to explain in the paper. Gate, fixed before the first run: with the option off all
references unchanged; with it on, the vacuum tube with `weno5` finishes at CFL 0.4 and 0.5 (rho, p above the floor to the end), the other
stress cases give the same outcome as without it, and the Sod and Shu-Osher errors change by less than 1 %.

**Decision needed:** whether a case you care about (a strong-shock application) justifies the option, or the guide line is enough.

## 2. The corner where two walls meet

**Today.** Faces are reconstructed row by row and column by column, so the inviscid fluxes never read a diagonal neighbour: the corner
ghost cell matters only to the viscous cross derivatives of the corner cell (and, in 3-D, the edge cells). With `boundary_order = 2`
the padded corner entries are rebuilt by `BoundaryManager::fix_padded_corners`: the corner takes the ghost of one wall applied to the
neighbouring ghost entry of the other (in 3-D, the lower axis first). That is a consistent reflection for two walls of the same kind;
for different kinds (a moving lid against a fixed wall, a velocity inlet above a wall) the order of the two reflections matters
and the result is a choice, not a derivation. Nothing measures the effect: the manufactured wall solutions are periodic in one direction.

**Options.**

* **A. Leave it.** The first-order closure also rounds the corner; the error is one cell wide.
* **B. Symmetric rule**: the corner ghost is the average of the two orders of reflection (x then y, y then x). Removes the arbitrariness
  for mixed corners; costs a few lines in `fix_padded_corners` (2-D and 3-D edges).
* **C. Corner with a singular solution** (a lid meeting a wall: the velocity is discontinuous at the corner): no ghost rule restores
  the order there, because the exact solution is not smooth; what helps is a local refinement, which AMR provides. Nothing to implement
  for the scheme.

**Recommendation.** Measure before choosing. Add a four-wall manufactured solution (`kind = "box"` in `fenice.mms`: rho, p even and u, v
odd about every side, so that the ghost is the extension; checked by `check_symmetry` like the others) and run `mms_boundary.py` with it for
`boundary_order` 1 and 2, MUSCL / WENO3 / WENO5, Euler and Navier-Stokes. Band fixed before the run: L2 order >= 1.7 on the finest pair,
as for the other wall solutions. If it is met, take A and delete the limitation; if not, B first. Cost: the generator (about 30 lines), one
case folder, one table; no solver change. This does not test mixed corners; a second solution with a moving lid on one side would
(u - U_lid odd about the lid), at the same cost.

**Decision needed:** only whether to run the measurement (recommended; it is cheap).

## 3. Inlets and outlets with a normal gradient

**Today.** The ghost of a velocity inlet is `2 u_in - u_cell` for the velocity (the face value is the prescribed one, second order) and
the cell value for rho and p (zero gradient). A pressure outlet fixes p at `p_out` through the ghost and copies the velocity when the
flow leaves (and sets it to zero when it enters). The characteristic outlet (`nonreflecting`) corrects only the incoming wave and keeps
first-order face states. So the free variables are first order wherever their true normal gradient is not zero, and with
`boundary_order = 2` they are still first order, because the second ghost layer is a mirror of the first. The manufactured outflow
solution has zero normal gradient by construction, which is why it reaches order 2 and says nothing about this case.

**Options.**

* **A. Leave it** and state it (the paper already does).
* **B. Linear extrapolation of the free variables** with `boundary_order = 2`: ghost layer k = `(k + 1) U_1 - k U_2` for the variables
  the condition does not prescribe (rho, p at a velocity inlet; the velocity components at a pressure outlet), with the prescribed one
  still set by its own rule. Second order for a smooth gradient, a few lines in `ghost_state` and `fill_ghost_slab`. Risks: overshoot at
  a shock or a vortex leaving the domain (extrapolation is unbounded), and a change of the character of the outlet (more reflection).
  A bound by the neighbouring cell values (clip to [U_1, U_2] range) removes the overshoot and returns to first order at extrema.
* **C. Characteristic boundary conditions to second order** (extend the existing characteristic outlet to the inlets and to the
  second ghost layer): the physically consistent route, the largest piece of work (inlet waves, a matching test for each Mach
  range), with a risk of new instabilities. Worth a separate proposal only if a case demands it.

**Recommendation.** B with the clip, opt-in as part of `boundary_order = 2`, after a manufactured solution that has the gradient there:
a `kind = "inout"` solution with a velocity inlet on the left and a pressure outlet on the right, whose boundary values come from the
exact solution (the boundary values already accept expressions of `t` and tables; the transverse coordinate needs checking, as for the
double-Mach case in `benchmarks-robustness-uq.md`, A4). Band fixed before the run: L2 order >= 1.7 with the rule, and the first-order
result recorded for contrast. Gate: all references unchanged with `boundary_order = 1`; the channel and cylinder cases with an
outlet change by less than their reference tolerance when the option is on (they are not expected to improve).

**Decision needed:** whether a case of yours has a real normal gradient at an inlet or outlet. If not, A (a documented limit) is a
fair choice; the paper's list can say so.

## Order of work if all three are taken

2 (measurement only, no solver change) -> 3 (generator first, its first-order result is the baseline; then the rule) -> 1 (only when a
case needs it). Each is one branch, about a day of work with the validation runs, and ends with its table in `docs/source/cases/`.
