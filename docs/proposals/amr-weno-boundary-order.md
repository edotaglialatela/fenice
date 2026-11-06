# WENO and second-order boundary faces on the adaptive mesh

Status: proposal, nothing implemented. SSP-RK3 with AMR is done (`f16cbd6`); what AMR still rejects (`Params.cpp`) is `reconstruction = weno3 | weno5` and `boundary_order = 2`. The method choices below are yours; each has a recommendation.

Everything here comes from reading `CellFlux.cpp`, `CellFlux.hpp`, `Schemes.cpp`, `Boundaries.hpp` and `docs/design/amr-cell-based.md`; no number is measured yet.

## What the adaptive residual looks like today

* **Stencils.** Every interior face has a fixed four-cell stencil `(c0, c1, c2, c3)` along its normal (`stencil_[f]`, built once per mesh change). A cell beyond a neighbour that is coarser is read by injection, one that is finer by the mean of the leaves under it (a virtual cell, recursive). The MPI halo is not a fixed depth: it is exactly the set of leaves `face_dependencies` says the faces read, so it follows the stencil.
* **Coarse-fine faces.** The fine side is reconstructed against the coarse value interpolated to the centre of the fine cell that would sit there (`interpolate_jumps`, limited slopes of the coarse leaf), and the coarse side by the usual limited formula against the mean of the fine leaves (`coarse_side_second_order`). Both sides are second order, from the same data as an interior face.
* **Time.** Recursive subcycling by 2, finer level first, flux registers on the coarse-fine faces, no time interpolation (the stencils are not time-consistent at the jump, measured as harmless with MUSCL). SSP-RK3 uses the same registers with the stage weights.
* **Physical boundary faces.** One ghost state from the boundary condition applied to the boundary leaf, converted through the primitive round trip; no stencil, so the face is first order. The viscous cross derivatives read the neighbour codes, where a code below zero is the diffusive ghost at a physical edge.
* **Invariant to keep.** A level-0 mesh gives the uniform solver's residual bit for bit (`celladvance_level0_steps_match_uniform_steps_bitwise`), and the state does not depend on the rank count.

## Part A: WENO3-Z

WENO3-Z reads exactly the four cells of the existing stencil: the left state is `weno3z(c0, c1, c2)`, the right state `weno3z(c3, c2, c1)` (`Schemes.cpp`), where MUSCL uses `c1 + 0.5 phi(...)`. So for interior faces it replaces one formula by another with no new data, no new halo and no new virtual cells.

The decision is at the **coarse-fine faces**:

* **A1 (recommended): WENO on both sides, with the data MUSCL already uses.** Fine side `weno3z(c0, c1, c2)` with `c2` the interpolated coarse value; coarse side `weno3z(wb, wc, wf)` with `wf` the mean of the fine leaves. No extra data. Third order in smooth flow on the seam, but the interpolated and averaged values are second order, so expect second order there.
* **A2: the limited MUSCL formula at the seam only.** Safest at a jump (a smooth-looking limiter at the seam), second order locally, and WENO3-Z everywhere else.

The Riemann batch in the engine (`muscl_face_flux_batch`, with an AVX2 twin) is written for the limiter formula. The first version would run WENO faces through the scalar path (`face_flux_batch` after reconstructing the states, as the coarse-fine faces do now); a vector twin comes later, only if the profile asks for it.

## Part B: `boundary_order = 2`

The uniform solver builds ghost layers: layer `l` is the boundary condition's ghost state of the interior cell `l` cells from the boundary (`fill_ghost_slab`), and the boundary face is reconstructed through them. On the adaptive mesh the same rule needs, for a boundary leaf `c`:

* layer 0 = ghost state of `c` (what exists today), layer 1 = the same map applied to the interior neighbour `n1` of `c` along the normal (the leaf behind it, or its virtual or interpolated cell when `n1` is on another level), and for WENO5 layer 2 from `n2`;
* a boundary face then gets a stencil like an interior face, `(g1, g0, c, n1)`, with the ghost entries computed at each stage from the stage state (they are not cells, so they are not in `stencil_`; the face needs a small ghost-recipe table next to it);
* the face's dependencies grow by `n1` (and `n2`), so the halo follows automatically, and the result depends only on leaf states: rank independence is kept.

Open points in this part, each with a proposed answer:

* **A leaf whose interior neighbour is on another level.** Use the same virtual or interpolated cell as an interior face does. The boundary face is then second order in smooth flow and the level jump next to the wall costs the order locally.
* **A side with too few cells.** The uniform code returns "no slab" when the side has fewer than `depth` cells; the adaptive code keeps the first-order boundary face for those leaves.
* **Which boundary types.** Walls (slip, no-slip, moving), inlets, outlets and transmissive as in the uniform solver. Periodic is already an interior seam; non-reflecting keeps its treatment; the mass-flow target of the stage is needed as in the uniform path.
* **Viscous terms.** On the uniform mesh a boundary face takes its cross derivatives from the tangential difference of the face values (ghost and cell average), and the corner entries of the padded state are rebuilt (`fix_frame_corners`). The adaptive code has no padded array, so this becomes: the tangential neighbour code of the boundary leaf gives the neighbour's face value (its ghost state), and where that neighbour is itself across another physical side (a corner) the other side's map is applied, which is what the corner rebuild does. At a tangential level jump along the wall the neighbour's value is the virtual cell's.
* **The immersed boundary.** Unchanged: the ghost-cell wall already needs the finest level around the body, and the mask forcing does not touch the domain boundary.

## Part C: WENO5-Z

Larger, and independent of B:

* the stencil is six cells (`c-1 .. c4`), so `stencil_` widens from 4 to 6 entries, the virtual cells must exist two cells beyond the face on each side (a cell beyond a neighbour that is itself beyond a jump can be two levels away, which the recursive virtual cell already handles in principle), and the neighbour codes need a second ring;
* the halo is three leaves deep, which `face_dependencies` provides without a fixed constant. In the **uniform** solver WENO5 is serial because its MPI halo is two cells deep. If the adaptive path runs it on ranks, either the uniform path gets a three-deep halo too (a separate change in `Schemes`), or the two differ and the docs say so. I would do the first; it is not large.
* the uniform finding probably carries over: WENO5-Z with SSP-RK3 needs CFL 0.3 on the near-vacuum tube. How the adaptive step maps the CFL to the time step of each level was not checked here.
* the time seam: with a wider stencil more of the reconstruction reads the coarse level at the old time. This is the part I can least predict, and I would measure it with the Sod and vortex cases before claiming anything.

## Verification plan (the bands written first)

1. **Bitwise, level 0.** Extend `celladvance_level0_steps_match_uniform_steps_bitwise` to the new reconstructions and `boundary_order = 2`. For this the adaptive code must call the same inline `weno3z` / `weno5z` and apply the ghost recipe in the same order of operations as the uniform code.
2. **Bitwise, ranks.** Regression cases `amr_weno3_*` and `amr_bo2_*`, serial and 3 and 4 ranks (60 regression cases today; the count in the docs and the paper moves with each new one).
3. **Conservation** to 1e-14 in a closed box with remeshing (the check used for SSP-RK3), which the registers guarantee for any reconstruction as long as the same face flux feeds both sides.
4. **Order.** A manufactured-solution study on an adaptive mesh. None exists today (`mms*.py` have no AMR variant). It needs a *static* refined region, which the solver does not offer as a parameter; the unit tests build one by hand. Proposal: a test-only parameter that forces refinement inside a box. The expected result is a global L2 order of 2 to 3 with a refined strip, not the uniform values; I would write the band after the first run only in the sense of reporting it, and fix it from the uniform values minus the seam allowance before running.
5. **Cases.** Sod and the vortex (smooth) with WENO3-Z on an adaptive mesh against the uniform solver on the finest grid; the airfoil with `boundary_order = 2` and the ghost-cell wall, loads against the uniform run.

## Order of work and size

1. Part A with A1 (smallest: one formula, scalar path, Params and GUI rules, tests, docs).
2. Part B for the Euler walls and outflows, then the viscous cross derivatives, with the MMS study of step 4.
3. Part C, after measuring how much the wider stencil changes the seam.
4. A vector twin of the WENO face kernel, only if the profile shows the scalar path in the way.

Parts A and B are each a few days of work with their tests; C is larger and the least certain.

## Decisions for you

1. Coarse-fine faces with WENO: A1 (recommended) or A2.
2. Boundary ghosts across a level jump: interpolated or virtual values as in the interior (recommended), or the first-order boundary face at those leaves.
3. WENO5 on MPI: extend the uniform path to a three-deep halo as well (recommended), or only the adaptive one.
4. The static-refinement test hook (a parameter that forces a refined box): yes or no.
5. Whether to do C at all, given that B is the part that changes accuracy on real wall-bounded flows.
