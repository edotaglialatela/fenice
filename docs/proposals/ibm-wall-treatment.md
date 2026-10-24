# IBM wall treatment: problem statement and proposals

Status: proposal C (ghost cells) implemented as an option, `ibm_method = ghost` (branch `ibm-ghost-cell`); see "Results" at the end.

## Current method

The immersed boundary (`ImmersedBoundary`, `cpp/src/Physics.cpp`) is a binary,
cell-centred direct forcing:

* **Mask.** A cell is solid when its centre lies inside the body
  (`|centre - c| <= r`). The body seen by the flow is a staircase.
* **Forcing** (`apply_forcing`, inside every residual / RK stage). Solid cells get
  `u = v = w = 0`; density and pressure are kept.
* **Solid reset** (`apply_solid_reset`, once per complete step). Fringe cells (solid
  with a fluid neighbour) copy rho and p from that neighbour (zero-gradient); deep
  solid cells are frozen to (`rho_in`, `p_out`). The energy uses the ideal-gas
  relation whatever the EOS.

## Problems

1. **Spurious waves from the staircase (most visible).** Each corner of the stepped
   wall acts as a small forward-facing step. At Mach 2 this produces the speckled,
   feathery features hugging the cylinder at about ±45-70° in the numerical schlieren.
   Finer meshes shrink the steps but do not remove them.
2. **Grid-dependent effective body.** The wall position is only known within ~h/2,
   so the effective radius depends on how the grid cuts the circle. At the default
   20 cells per diameter, Cd and St carry an O(h) error of a few percent that
   changes non-monotonically with resolution. Forces integrated over the staircase
   (`validation/aeroCoeffCylinder.py`) are additionally noisy.
3. **First-order wall.** No-slip is imposed at solid cell centres, not at the true
   wall: wall shear is first order. Acceptable for Re = 100, not for wall-resolved flows.
4. **No slip condition for inviscid bodies.** Forcing zeroes the whole velocity, so
   even the inviscid Mach 2 case has a no-slip (stagnant) body instead of a slip wall.
5. **EOS.** The reset uses the ideal-gas energy formula for any EOS (the p_inf term of
   a stiffened gas is missing).

## Proposal A: smoothed indicator (tanh)

Replace the 0/1 mask with a smooth indicator of the signed wall distance d,

    chi = 0.5 * (1 - tanh(d / eps)),

and damp the velocity with it.

* Pros: removes the staircase corners (problem 1), smooths the force signal.
* Cons:
  * the body gets a porous skin of width ~eps; the effective size depends on the new
    parameter eps, so problem 2 is moved rather than removed;
  * `u <- (1 - chi) u` applied at every stage is a drag of strength ~chi/dt: results
    depend on the time step / CFL;
  * the solid reset (fringe / deep logic) is defined on the binary mask and needs a
    rule for partially solid cells.

## Proposal B: volume-fraction penalisation (recommended first step)

Same idea as A, without the free parameter and without the dt dependence:

* **Indicator = geometric volume fraction** `chi` of each cell inside the body,
  computed once at setup by sub-sampling each cell (e.g. 8x8, or 8x8x8 in 3-D). The
  transition is about one cell wide and set by the grid, not by a tuning constant.
* **Implicit Brinkman penalisation** instead of a partial reset:
  `u <- u / (1 + dt * chi / eta)` with a small permeability `eta`. The implicit
  update is stable for any stiffness and converges to the penalised solution
  independently of dt (Angot, Bruneau & Fabrie 1999).
* **Solid reset** on `chi >= 0.5` cells only; fringe = such cells with a `chi < 0.5`
  neighbour. Use the EOS for the energy (fixes problem 5).
* **Opt-in** (`ibm_smoothing = true`, default `false`) so the default path stays
  bit-identical and the three IBM regression references (`cylinder_ibm_visc`,
  `supersonic_ibm`, `sphere3d_ibm_visc`) do not change.
* MPI: `chi` is cut from the global field like the current mask; the reset halo
  logic is unchanged apart from the threshold.

Validation:

* cylinder Re = 100: St and Cd against the literature (St ~ 0.165, Cd ~ 1.35 for
  an unconfined cylinder; correct for the 10 % blockage) at 20, 40 and 60 cells per
  diameter, sharp vs smoothed;
* Mach 2 cylinder: before/after schlieren of the surface region; bow-shock stand-off
  distance against the Billig correlation;
* serial vs MPI bit-identity, extruded 2-D vs 3-D, as for every new stencil.

Effort: about a day, mostly validation.

## Proposal C: ghost-cell IBM (sharp interface, the proper fix)

Ghost-cell method (Tseng & Ferziger 2003; Mittal et al. 2008): for each solid cell
next to the fluid (ghost cell), find its image point by mirroring across the *true*
surface, interpolate the flow state at the image point from the surrounding fluid
cells, and set the ghost state so the boundary condition holds on the true wall:

* slip wall: reflect the normal velocity, keep the tangential one (problem 4);
* no-slip: reflect the whole velocity;
* rho, p: zero normal gradient (or a pressure-gradient correction for curvature).

* Pros: sharp, second-order wall; staircase noise gone (problems 1-4); a real
  choice of slip / no-slip; accurate surface pressure for force integration.
* Cons: image-point interpolation stencils (degenerate cases when the stencil touches
  other ghost cells or thin bodies), and those stencils cross MPI interfaces, so they
  need halo treatment like every other stencil (see "MPI == serial" in the developer
  notes). More code than B.

## Recommendation

Do B first, as an opt-in: it fixes the visible artefacts and stabilises the force
coefficients for little code. Move to C if wall quantities (skin friction, surface
pressure, heat flux) need to be quantitative.

## References

* Angot, P., Bruneau, C.-H. & Fabrie, P. (1999). A penalization method to take into
  account obstacles in incompressible viscous flows. *Numer. Math.* 81, 497-520.
* Tseng, Y.-H. & Ferziger, J.H. (2003). A ghost-cell immersed boundary method for flow
  in complex geometry. *J. Comput. Phys.* 192, 593-623.
* Mittal, R. et al. (2008). A versatile sharp interface immersed boundary method for
  incompressible flows with complex boundaries. *J. Comput. Phys.* 227, 4825-4852.
* Mittal, R. & Iaccarino, G. (2005). Immersed boundary methods. *Annu. Rev. Fluid
  Mech.* 37, 239-261.

## Results (proposal C, `ibm_method = ghost`)

Chosen over B because B stays first order at the wall and makes it porous. Implemented in 2-D for the uniform solver (serial,
OpenMP, MPI strips) and the AMR engine (finest-level band), opt-in; the default `mask` path is unchanged bit for bit.

* **Method.** Ghost cells = solid cells within two cells of a fluid cell. Image point at least 1.5 cells into the fluid along the
  normal from the closest wall point (further out, in half-cell steps up to 4 cells, where the four bilinear cells are not all
  fluid: concave corners, thin trailing edges; solid cells are dropped from the bilinear weights). Ghost state: density and
  pressure of the image point (zero normal gradient), velocity -(d_ghost / d_image) times the image velocity (no-slip) or only its
  normal component (slip, `use_viscous = false`). Deeper solid cells at rest, reset to the reference state with the EOS (fixes
  problem 5). Loads: the momentum flux into the solid cells, as with the mask (see "Loads" below); the surface integral from probes
  at d and 2 d along the normal (wall pressure extrapolated linearly, wall shear from the quadratic through zero at the wall) is
  kept for surface distributions.
* **MPI.** The ghost table is built on the whole mesh on every rank; each rank fills its own ghost cells and fetches the cells of
  other strips their image points read (an all-to-all of a few cells, any strip width). AMR: the band of `amr_refine_ibm` is widened
  to the stencil reach (6 finest cells for the cylinder and the airfoil), the image leaves are part of the halo.
  `cases/validation/features/ibm_ghost.py --check` (in CI): serial against 2-7 ranks bit for bit, uniform and AMR.
* **Cylinder, Re = 40** (steady; Mach 0.27 to shorten the runs; channel 4 x 2, 10 % blockage), Cd from the same control-volume
  momentum balance on both flows (three box sizes agree within 0.2 %):

  | cells per diameter | mask | ghost |
  |---|---:|---:|
  | 20 | 1.838 | 1.897 |
  | 40 | 1.864 | 1.894 |

  The ghost wall is grid-converged to 0.15 % at 20 cells per diameter; the mask converges at about first order towards the same
  value, needing roughly four times the resolution per direction. (Re-run for the loads below, the flow still drifts slowly at
  t = 40: Cd 1.903 / 1.900 / 1.896 at t = 20 / 30 / 40 at 20 cells per diameter, so the 0.15 % is a comparison at similar times, not
  of fully settled flows.)
* **Loads** (`ibm_ghost.py --loads`). The probe integral of the true surface was 1-3 % below the control-volume balance (2.7 % on
  the cylinder at 20 cells per diameter, 4 % in Cd and 3 % in Cl on the airfoil). The monitor now uses the momentum flux into the
  solid cells, the residual of the current state summed over them, as the mask does: the fluxes through faces between two solid
  cells cancel, those through the faces between fluid and ghost cells remain, and they close the momentum budget of the fluid
  cells exactly. Against an independent balance of the field on boxes around the body (face values and central-difference stresses):

  | flow | monitor (flux into the solid) | box balance (3 boxes) | probe integral |
  |---|---:|---:|---:|
  | cylinder Re = 40, 20 cells / D, t = 40 | Cd 1.8957 | 1.8945 / 1.8923 / 1.8890 | 1.8437 |
  | cylinder Re = 40, 40 cells / D (hot start, + 15) | Cd 1.8836 | 1.8831 / 1.8822 / 1.8807 | |
  | airfoil Re = 1000, 360 x 180, t = 3.2 (shedding) | Cd 0.1855, Cl 0.5048 | Cd 0.1851-0.1856, Cl 0.5051-0.5064 | Cd 0.1778, Cl 0.4880 |

  On the cylinder the box balance has no momentum-change term, so the larger boxes pick up the slow drift (it decreases with the
  box); on the airfoil it includes the change of the fluid cells' momentum (central difference over two steps) and the three boxes
  agree with the monitor to 0.3 % in Cd and 0.3 % in Cl. Cost: one residual per monitor row (with `monitor_interval = 20` and
  SSP-RK2 about 2.5 % of the run; nothing without a monitor); the probe integral was nearly free. The states are unchanged bit
  for bit; under MPI the strips' sums are added in another order (loads equal to roundoff, `ibm_ghost.py --check`).
* **Mach 2 cylinder**: the staircase waves at 45-75 degrees and the kinks of the surface pressure are gone with the ghost wall
  (slip). The bow-shock stand-off is grid-converged for both and differs between them: 0.332 / 0.330 (ghost, 50 / 100 cells per
  diameter) against 0.319 / 0.316 (mask), with 0.310 from the Billig correlation (an empirical fit). The difference comes from the
  wall condition (a slip wall against the mask's stagnant staircase); without an exact inviscid reference this case does not say
  which stand-off is closer.
