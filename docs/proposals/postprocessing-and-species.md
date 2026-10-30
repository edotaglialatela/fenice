# Post-processing in the case file, and multi-species flow: proposals

Status: proposal. Part A needs no method choice and is implemented item by item on branch `postprocessing`; part B lists the
choices that are the user's and is not started.

## Where the solver stands

* The monitor (`monitor_interval`, `Simulation::monitor_row`) writes volume averages and the IBM loads to a CSV. Under MPI its sums
  go through `reduce_sum` in rank order, but the partial sums depend on the partition, so the monitor values agree with serial to
  roundoff only. The bitwise checks compare states, not the monitor.
* Boundary values (`u_in`, `rho_in`, ...) are plain numbers. Only the source terms know the time `t` (`TimeScheme::stage_time`).
* `State` has five slots everywhere (Riemann solvers, AVX2 kernels, AMR registers, MPI buffers, checkpoints).
* The equation of state is one struct (ideal or stiffened gas). The IBM solid reset uses the ideal-gas pressure formula whatever the
  EOS; any refactor of the EOS must keep that.

## A. Post-processing declared in the case file

Every item is off by default, changes no state and no existing reference, and is tested serial against 1-7 ranks (files identical).

1. **Probes and line samples.** `[[probe]]` (a point) and `[[sample]]` (a line of points) write the primitive variables of the cell
   that holds each point at a given interval. Nearest cell (no interpolation) so that the value is a state value, the same one for
   any rank count and, with AMR, the leaf that holds the point. Output: one CSV per probe / line under the output folder.
2. **Forces and averages.** `[[force]]` (IBM loads with a reference length and pressure/density, giving coefficients) and
   `[[average]]` (time average of chosen variables). Any sum is made in a fixed order (cells in global index order, or a fixed
   binary tree over blocks) so that the result does not depend on the partition. *Open choice for the user:* with AMR a time average is
   ill-defined across a remesh: reset it at a remesh, average on the level-0 grid only, or support uniform meshes only.
3. **Output at exact times and a maximum dt.** `write_interval_time` clips the step so that a snapshot lands on 0.01, 0.02, ...;
   `dt_max` caps the step next to the CFL. Both off by default (the references do not move); with AMR the clip is applied to the
   base step.
4. **Boundary values that vary in time.** `u_in`, `p_out`, ... accept the expression engine's expressions of `t`, evaluated at
   `TimeScheme::stage_time` as the sources are (ramps, sines); a table (CSV of t, value) with linear interpolation.
5. **Source and thermophysics interfaces.** One small interface for the right-hand-side contributions (body force, expression
   sources, later reactions) and the equation of state split into pressure law, internal energy and transport. Strictly bitwise
   neutral: regression, golden dumps, `FENICE_NO_SIMD=1`; no virtual call inside a face or cell loop; timed against `main` (min of 3)
   before it is committed.

## B. Choices that are the user's (not started)

* **Boundary conditions:** a total-pressure inlet, and an inlet/outlet condition that switches with the sign of the normal velocity.
* **Species.** Keep the five flow variables and carry the species as a separate per-cell array `rho Y_k`. The cost is in the paths
  that move a state: `assemble`, `assemble_3d`, the AMR face engine and registers, the MPI halos, checkpoints, the IBM reset, the
  viscous fluxes. With no species the five-variable path is untouched and the references do not move. Choices: reconstruct `Y_k` or
  `rho Y_k`; the species flux of HLLC (`rho* Y_k` from the upwind side of the contact is the usual one); the sound speed (frozen mixture
  gamma).
* **Material interfaces.** A conservative scheme with a mixture gamma that varies with `Y` produces pressure and velocity errors at a
  contact even when they start uniform. The test that decides: an interface advected at uniform p and u keeps them, with a band fixed
  beforehand. If it fails, the fix is a choice (a quasi-conservative energy update, a pressure-equilibrium correction, or accepting and
  documenting the error). The multi-species Sod problem has an exact solution only with a different gamma on each side of the contact,
  which `fenice.exact` would have to be extended for.
* **Chemistry.** Operator splitting (Strang: half a chemistry step, the flow step, half a chemistry step); a stiff implicit integrator per
  cell, independent across cells, so the result does not depend on the thread or rank count. With AMR the half steps sit per level and
  per stage of the recursion, and conservative flux replacement must stay intact: it needs its own design section. The explicit time
  step follows the speed of sound, so this suits high-speed reacting flow (detonations, shock-induced ignition, supersonic reacting flow
  behind a body), not slow flames.
* **Where the thermochemistry comes from.** Either link a thermochemistry library behind an optional build switch, or generate a
  mechanism file with a Python tool (like `fenice.mms` does for the manufactured solutions) holding the polynomial coefficients,
  molecular weights and reaction data, which the solver reads. The second keeps the C++ build and CI unchanged and keeps the references
  independent of the library's version; the solver then needs its own rate evaluation (Arrhenius, third bodies, falloff) and integrator.

## Suggested order

A1, A4, A3, A2, A5, then B in the order passive scalars, thermodynamics, interface test, chemistry. Verification of B: a transported
scalar that stays in [0, 1] and sums to one (under a stated CFL), the interface test above, a 0-D ignition against a reference reactor,
a 1-D steady detonation structure and the Chapman-Jouguet speed.
