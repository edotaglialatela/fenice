# WENO and second-order boundary faces on the adaptive mesh

Status: Part A (WENO3-Z, option A1) and Part B (`boundary_order = 2`, inviscid and viscous) are implemented; C is a proposal. SSP-RK3 with AMR is done (`f16cbd6`); what AMR still rejects (`Params.cpp`) is `reconstruction = weno3 | weno5` and `boundary_order = 2`. The method choices below are yours; each has a recommendation.

Everything here comes from reading `CellFlux.cpp`, `CellFlux.hpp`, `Schemes.cpp`, `Boundaries.hpp` and `docs/design/amr-cell-based.md`; no number is measured yet. A first version of this file was reviewed against the code; the corrections are listed at the end.

## What the adaptive residual looks like today

* **Stencils.** Every interior face has a fixed four-cell stencil `(c0, c1, c2, c3)` along its normal (`stencil_[f]`, built once per mesh change). A cell beyond a neighbour that is coarser is read by injection, one that is finer by the mean of the leaves under it (a virtual cell, recursive). The MPI halo is not a fixed depth: it is built from the leaves `face_dependencies` says the faces read, and `CellParallel::halo_reach` measures how far those reads reach, so a deeper stencil raises it in principle. That is the intent of the design, not something checked for a six-cell stencil: a leaf outside a rank's share turns into `kOutside` or a dropped virtual cell without an error, so a missing ring would make the result depend on the rank count. The guards are `halo_reach_bounds_the_halo_of_any_partition` and the few-roots MPI regression, and they have to be run for every new stencil.
* **Coarse-fine faces.** The fine side is reconstructed against the coarse value interpolated to the centre of the fine cell that would sit there (`interpolate_jumps`, limited slopes of the coarse leaf), and the coarse side by the usual limited formula against the mean of the fine leaves (`coarse_side_second_order`). Both sides are second order, from the same data as an interior face.
* **Time.** Recursive subcycling by 2, finer level first, flux registers on the coarse-fine faces, no time interpolation (the stencils are not time-consistent at the jump, measured as harmless with MUSCL). SSP-RK3 uses the same registers with the stage weights.
* **Physical boundary faces.** One ghost state from the boundary condition applied to the boundary leaf, converted through the primitive round trip; no stencil, so the face is first order. The interior face next to the wall, whose stencil reaches beyond the boundary leaf, replicates that leaf (`beyond()` returns the leaf itself for a boundary side). The viscous cross derivatives read the neighbour codes, where a code below zero is the diffusive ghost at a physical edge.
* **Virtual cells are rows of the primitive array.** `W_` holds the leaves, then the averaged virtual cells, then the interpolated ones, and `prepare` refreshes the rows each stage from the current state (`refresh_all` / `refresh_some`, `eval_interp`). A stencil entry is an index into that array.
* **Invariant to keep.** A level-0 mesh gives the uniform solver's residual bit for bit (`celladvance_level0_steps_match_uniform_steps_bitwise`), and the state does not depend on the rank count.

## Part A: WENO3-Z

WENO3-Z reads exactly the four cells of the existing stencil: the left state is `weno3z(c0, c1, c2)`, the right state `weno3z(c3, c2, c1)` (`Schemes.cpp`), where MUSCL uses `c1 + 0.5 phi(...)`. So for interior faces it replaces one formula by another with no new data, no new halo and no new virtual cells.

The decision is at the **coarse-fine faces**:

* **A1 (recommended): WENO on both sides, with the data MUSCL already uses.** Fine side `weno3z(c0, c1, c2)` (fine leaf on the left) or `weno3z(c3, c2, c1)` (fine leaf on the right), with the entry across the face the interpolated coarse value; coarse side `weno3z(wb, wc, wf)` with the coarse leaf on the left and `weno3z(wf, wc, wb)` on the right, `wf` being the mean of the fine leaves. No extra data. The interpolated value of the jump still uses the `muscl_limiter` slopes of the coarse leaf, so the limiter stays in the scheme near a seam under WENO. Third order in smooth flow on the seam, but the interpolated and averaged values are second order, so expect second order there.
* **A2: the limited MUSCL formula at the seam only.** Safest at a jump (a smooth-looking limiter at the seam), second order locally, and WENO3-Z everywhere else.

The Riemann batch in the engine (`muscl_face_flux_batch`, with an AVX2 twin) is written for the limiter formula. The first version would run WENO faces through the scalar path (`face_flux_batch` after reconstructing the states, as the coarse-fine faces do now); a vector twin comes later, only if the profile asks for it.

## Part B: `boundary_order = 2`

The uniform solver builds ghost layers: layer `l` is the boundary condition's ghost state of the interior cell `l` cells from the boundary (`fill_ghost_slab`). Two faces per boundary line read them, not one:

* the **boundary face** itself, reconstructed through them;
* the **first interior face**, whose stencil reaches one cell beyond the boundary leaf: in `Schemes::reconstruct` its `c0` is the ghost layer when the side has a halo, the edge cell otherwise. With WENO5 the first two interior faces read the layers. Today the adaptive code replicates the boundary leaf for that entry, which matches the uniform code only at `boundary_order = 1`. So the level-0 bitwise invariant needs the edge entry of those faces to switch from replication to the ghost as well, not only the boundary face to get a stencil.

**Proposed design (recommended): ghost layers as a third kind of virtual cell.** Add rows to `W_` after the interpolated ones: row `g` of a boundary leaf `c` and side `s` is the primitive state of `G_s(U[c])` (layer 0), and of `G_s(U[n1])` (layer 1, `n1` the interior neighbour of `c`, or its virtual or interpolated cell across a level jump), and of `G_s(U[n2])` for WENO5. They are refreshed each stage in `prepare`, like `eval_interp` does for the interpolated cells, with the stage's mass-flow target. Then:

* the boundary face and the first interior faces take plain `stencil_` indices (`beyond()` returns the ghost row instead of the leaf for a boundary side);
* the batch kernel and its AVX2 twin stay as they are, for MUSCL and later WENO;
* `face_dependencies` reaches the mirrored leaves through the ordinary stencil, with no second table;
* the rule for building the rows is the one `fill_ghost_slab` applies, so the bits can match.

(The alternative is a separate ghost-recipe table next to `stencil_`, evaluated inside `fluxes`; it keeps `W_` untouched but needs its own dependency code and a second path in the batch. I do not recommend it.)

Further points, each with a proposed answer:

* **The moving-wall ghost changes with `boundary_order = 2`**: it keeps the internal energy of the interior cell (`Boundaries.hpp`, `boundary_order` member), which the first-order ghost does not. The ghost rows of a moving wall must use that form, or the cavity and the other moving-wall cases lose level-0 equality.
* **A leaf whose interior neighbour is on another level.** Use the same virtual or interpolated cell an interior face uses. The wall face is then second order in smooth flow and the level jump next to the wall costs the order locally.
* **A side with too few cells.** The uniform code returns "no slab" when the side has fewer than `depth` cells; the adaptive code keeps the first-order boundary face and the replicated edge entry for those leaves.
* **Which boundary types.** Walls (slip, no-slip, moving), inlets, outlets and transmissive as in the uniform solver. Periodic is already an interior seam; non-reflecting keeps its treatment.
* **Viscous terms.** On the uniform mesh a boundary face takes its cross derivatives from the tangential difference of the face values (ghost and cell average), and `fix_frame_corners` rebuilds the corner entries of the padded state. In the adaptive code the tangential neighbour code of the boundary leaf gives the neighbour's face value (its ghost state); at a corner the neighbour is itself across another physical side, and the order of the two maps matters for the bits: the uniform rule takes the cell in the ghost row (the bottom or top map applied to the cell) and then applies the left or right map to that. The adaptive code must compose them in the same order. At a tangential level jump along the wall the neighbour's value is the virtual cell's.
* **The immersed boundary.** Unchanged: the ghost-cell wall already needs the finest level around the body, and the mask forcing does not touch the domain boundary.

## Part C: WENO5-Z

Larger, and independent of B:

* the stencil is six cells (`c-1 .. c4`), so `stencil_` widens from 4 to 6 entries, the virtual cells must exist two cells beyond the face on each side (a cell beyond a neighbour that is itself beyond a jump can be two levels away, which the recursive virtual cell already handles in principle), and the neighbour codes need a second ring;
* the halo has to reach three leaves; whether `halo_reach` and the share construction raise it by themselves for a six-cell stencil is to be tested (see the guards above), not assumed. In the **uniform** solver WENO5 is serial because its MPI halo is two cells deep. If the adaptive path runs it on ranks, either the uniform path gets a three-deep halo too (a separate change in `Schemes`, whose size I have not looked at), or the two differ and the docs say so. I would give the uniform path the deeper halo.
* the uniform finding probably carries over: WENO5-Z with SSP-RK3 needs CFL 0.3 on the near-vacuum tube. How the adaptive step maps the CFL to the time step of each level was not checked here.
* the time seam: with a wider stencil more of the reconstruction reads the coarse level at the old time. This is the part I can least predict, and I would measure it with the Sod and vortex cases before claiming anything.

## Verification plan (the bands written first)

1. **Bitwise, level 0.** Extend `celladvance_level0_steps_match_uniform_steps_bitwise` to the new reconstructions and `boundary_order = 2`. For this the adaptive code must call the same inline `weno3z` / `weno5z` and apply the ghost recipe in the same order of operations as the uniform code.
2. **Bitwise, ranks.** Regression cases `amr_weno3_*` and `amr_bo2_*`, serial and 3 and 4 ranks (60 regression cases today; the count in the docs and the paper moves with each new one).
3. **Conservation** to 1e-14 in a closed box with remeshing (the check used for SSP-RK3), which the registers guarantee for any reconstruction as long as the same face flux feeds both sides.
4. **Order.** The Part B order test is the existing set of manufactured solutions (`walls`, `outflow` and `box`, Euler and Navier-Stokes, in `mms_boundary.py`), run on an adaptive mesh. There is no AMR variant today, and it needs a *static* refined region, which the solver does not offer as a parameter (the unit tests build one by hand); proposal: a test-only parameter that forces refinement inside a box, a strip across the domain so the seam meets the walls. The uniform bands are L2 order at least 1.7 and L-infinity at least 1.3 on the finest pair, worst variable. For the adaptive runs I propose, before the first run, L2 at least 1.5 and L-infinity at least 1.0 (a seam allowance of 0.2 and 0.3); a result below them is reported as below, not moved. These numbers are mine and open to your change.
5. **Cases.** Sod and the vortex (smooth) with WENO3-Z on an adaptive mesh against the uniform solver on the finest grid. For Part B the lid-driven cavity on an adaptive mesh with `boundary_order = 2` (moving and no-slip walls, the Ghia reference, already one of the `gci.py` studies); the airfoil is a poor test of it, since its domain sides are far-field and the body wall is handled by the immersed boundary.

## Order of work and size

1. Part A with A1 (smallest: one formula, scalar path, Params and GUI rules, tests, docs).
2. Part B for the Euler walls and outflows, then the viscous cross derivatives, with the MMS study of step 4.
3. Part C, after measuring how much the wider stencil changes the seam.
4. A vector twin of the WENO face kernel, only if the profile shows the scalar path in the way.

Each step also updates `docs/design/amr-cell-based.md`, which still describes the old padded-state and cross-ghost rules.

Parts A and B are each a few days of work with their tests; C is larger and the least certain.

## Decisions for you

1. Coarse-fine faces with WENO: A1 (recommended) or A2.
2. Boundary ghosts across a level jump: interpolated or virtual values as in the interior (recommended), or the first-order boundary face at those leaves.
3. WENO5 on MPI: extend the uniform path to a three-deep halo as well (recommended), or only the adaptive one.
4. The static-refinement test hook (a parameter that forces a refined box): yes or no.
5. Whether to do C at all, given that B is the part that changes accuracy on real wall-bounded flows.

## Corrections made after review

The first version of this file said that only the boundary face needed a stencil (the first interior face reads the ghost layer too, and the moving-wall ghost form changes), proposed a separate ghost table (ghost rows in `W_` are simpler), called the halo automatic and the WENO5 uniform halo "not large" (neither was checked), gave the coarse-side WENO argument order for one orientation only, left the order bands to be fixed "after the first run", and proposed the airfoil as the boundary-order test.
