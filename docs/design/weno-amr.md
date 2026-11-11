# WENO5-Z on the adaptive mesh (2-D and 3-D): plan

Status: **not started; the three choices of section 3 are decided (2026-11-11), the phases of section 5 are the work.** It is the one open
feature of the adaptive engine: `reconstruction = weno3` and `boundary_order = 2` already run with AMR in 2-D and in 3-D (see the AMR page,
`docs/source/modules/AMR.md`), `weno5` is rejected with AMR (`Params.cpp`, and `CellFluxEngine::build` on an octree). The smaller open items
of the adaptive engine are listed at the end.

## 1. Where the engine stands

* **Stencils.** Every interior face has a fixed four-cell stencil `stencil_[f] = (c0, c1, c2, c3)` along its normal, built when the mesh
  changes; `c1` / `c2` are the leaves left / right of the face. The codes come from `CellFluxEngine::beyond`: a leaf, the leaf itself at a
  physical side (replication), a mean-type virtual cell across a finer side (`vkey_`), an interpolated virtual cell across a coarser side
  (`ikey_`, the coarse value at the centre of the fine cell that would sit there, with the `muscl_limiter` slopes of the coarse leaf), or
  `kOutside` in a rank's share. `nbr_` holds the same codes per leaf and side, `nbr_raw_` the same with the coarse value injected.
* **Rows.** The primitive array `W_` has the leaves, then the mean-type virtual cells, then the interpolated ones. `prepare(level)` refreshes
  the rows in `need_[level]` (the entries the stencils of the level's faces read) at every stage.
* **Flux path** (`CellFluxEngine::fluxes`, `CellFluxResidual.cpp`). MUSCL faces with a plain stencil go to the batch kernel
  (`muscl_face_flux_batch`, AVX2 twin). WENO3-Z faces, coarse-fine faces and faces that read a ghost layer take the scalar path (`recon4`),
  then `face_flux_batch`. At `boundary_order = 2` the ghost layers are evaluated where they are read: `ghost_w(side, U[c])` for the first
  layer, `ghost_w(side, U[n1])` for the second (`n1` the leaf behind the boundary leaf). No ghost table.
* **Coarse-fine faces.** The fine side is reconstructed against the interpolated cell across the face, the coarse side against the mean of
  the fine leaves (`coarse_side_second_order`), both with the scheme's formula (limiter or WENO3-Z).
* **Time.** Recursive subcycling by 2 with flux registers on the coarse-fine faces, no time interpolation; SSP-RK3 uses the registers with
  the stage weights.
* **MPI.** The halo is exactly what `face_dependencies` (`mark_face`) says the active faces read. The share is built from the leaves within
  `reach_rings` (`CellShare.cpp`, fixed constants measured with `CellParallel::halo_reach`) of the owned ones; a share that is too small makes
  `partition_local` return false and every rank retries with a wider reach.
* **Rebuild.** After an adapt (and, on a rank, when the new share is built over the old one) the tables of a clean leaf are copied. The
  rule: what `build` computes per leaf or face depends only on the leaf and the squares of its size across its sides, or is recomputed in
  full.
* **The uniform solver's WENO5-Z** (`MusclReconstruction::reconstruct_weno5`, `Schemes.cpp`): the six primitive cells of a face along its
  normal, `wl = weno5z(c0..c4)`, `wr = weno5z(c5..c1)` per variable, then `prim_to_cons`. Cells beyond a physical side replicate the edge
  cell; with `boundary_order = 2` the side has a ghost slab three layers deep (`BoundaryManager::fill_ghost_slab`, `Dh = 3`) and its
  boundary faces are reconstructed too; a periodic axis wraps with a three-deep halo if it has at least three cells, else it replicates.
  It runs the two-pass path (not the fused one), scalar, serial only (the uniform MPI halo is `kHaloDepth = 2`).

## 2. Two measured facts that shape the design

1. **The discretisation is second order whatever the reconstruction.** The flux balance uses face-centre values, so on the smooth
   manufactured solutions WENO5-Z gives an observed order of 2.00 in every variable and norm (`cases/validation/verification/mms_weno.py`).
   What WENO5-Z buys is a smaller error constant and less dissipation at shocks and vortices, not a higher order.
2. **A level jump already limits the adaptive error.** With WENO3-Z and with MUSCL the seam is second order in L2 and first order at the
   corner of a coarse and a fine cell; inside a refined square the error is 1.3 to 13.6 times that of the uniform fine mesh
   (`mms_amr_weno.py`). The error made at the seam is transported through the domain.

So a high-order closure of the seam cannot raise the order of the run, and the six-cell stencil pays only away from the seams.

## 3. Decisions

| | question | decision |
|---|---|---|
| 3.1 | a face whose six cells are not all leaves of its level | **A**: it keeps the four-cell stencil with the WENO3-Z formula |
| 3.2 | WENO5-Z on MPI ranks in the uniform solver | **not part of this work**: the uniform rejection stays, `weno5` + `use_mpi` runs with AMR only (section 8 is the later option) |
| 3.3 | faces next to the immersed body | **not wide**: a face with a solid or ghost leaf among its six cells keeps the four-cell stencil; the ghost band stays two cells |
| 4.4 | a periodic axis with fewer than three base cells | **rejected** with `weno5` and AMR (`Params.cpp`) |

The alternatives are kept below for the record of what was weighed.

**3.1 What a face does when its six cells are not all leaves of its level.**

* **A. Reduced stencil at jumps (chosen).** A face is *wide* when its six cells are leaves of the face's level (or replications /
  ghost layers at a physical side). A wide face uses WENO5-Z. Every other face (a coarse-fine face, or a same-level face with a virtual
  cell or an injected coarse leaf among its six cells) keeps today's four-cell stencil with the WENO3-Z formula. No new kind of virtual
  cell, no change at the seams (their bits and their time behaviour stay those of WENO3-Z), one more same-level ring in the MPI halo.
  Cost: the two faces on each side of a seam are WENO3-Z; by fact 2 the seam is the larger error there anyway.
* **B. Six cells across the jump.** Second-layer virtual cells: a second interpolated cell inside the coarse leaf (centre offset -0.25
  instead of +0.25 along the normal, key `(level, i, j, k, side, depth)`), mean-type blocks two cells away (`block_code` already recurses),
  and a second ring of injected codes for the coarse side (`nbr_raw_` at distance 2). The interpolated values stay limited-linear, so the
  fine side at a seam reads two second-order values: the formal order at the seam does not rise. More tables, a halo up to two coarse cells
  deeper, and more of the fine reconstruction reads coarse data frozen at the old time during the subcycles (to be measured).
* **C. B with a higher-order prolongation** (a conservative parabolic interpolation in the coarse leaf): reads the transverse and corner
  neighbours of the coarse leaf, so a wider dependency and a new limiter question. By fact 1 it cannot show up as order.

The plan below is written for A; section 9 says what B adds.

**3.2 WENO5-Z on MPI ranks in the uniform solver.** The adaptive engine does not depend on it. The uniform rejection stays as it is
(`weno5` + `use_mpi` works with AMR only, and the docs say so). Giving the uniform path a three-deep halo (section 8) would keep the two
solvers symmetric; it touches the uniform halo exchange, whose invariants (MPI == serial, strips of at least `kHaloDepth` cells) each
need their test extended, and it can be done at any time after this work.

**3.3 Faces next to the immersed body.** With the ghost-cell wall a ghost cell is a solid cell within two cells of a fluid cell; a wide
stencil reads three. A face with a solid or ghost leaf among its six cells is not wide: the wall treatment stays the one that is
validated, and the fluid faces within two cells of the body are WENO3-Z. The alternative, a ghost band three cells deep
(`ImmersedBoundary::enable_ghost_cells`, `ghost_reach`, the body band of the remesh), would change the uniform solver's IBM results as
well. The uniform solver has no special case here (its six-cell stencil reads the two ghost cells and a deep solid cell behind them), so
with a body the adaptive level 0 and the uniform solver differ under `weno5`, as they already do for the other schemes; phase W0 measures
by how much on the cylinder.

## 4. Design (choice A)

### 4.1 Tables

* `far_` : `std::vector<std::array<int, 2>>`, one entry per face: the codes of `c0'` (the cell behind `c0`) and `c5'` (the cell beyond
  `c3`), so that the six cells are `(far[0], s0, s1, s2, s3, far[1])`. Kept apart from `stencil_` so that the four-cell tables, their
  incremental copy and the MUSCL batch kernel are untouched. Built only when the scheme is WENO5-Z (empty otherwise: no memory, no work).
* `wide_` : `std::vector<char>`, one flag per face.
* Both are derived from `nbr_` and `stencil_` alone, in a pass over the faces at the end of `build` (two table reads per face, no lookup
  in the mesh):
  * `far[0] = s0 == s1 ? s1 : nbr_[ns * s0 + back]` and `far[1] = s3 == s2 ? s2 : nbr_[ns * s3 + fwd]`, where `s0 == s1` (resp. `s3 == s2`)
    is the replication at a physical side; a `-1` read from `nbr_` (the side of the domain behind `s0`) is replaced by `s0` (replication
    twice, as the uniform solver's clamp does).
  * `wide = face.kind == Interior`, and each of `s0`, `s3`, `far[0]`, `far[1]` is a leaf (`code < nleaf_`) of the face's level, and none
    of the six is `kOutside`, and (decision 3.3) none is solid or a ghost leaf (`solid_`, `ghost_leaf_mask_`). The ghost leaves are
    mapped after the build (`ghost_ready`), so with a ghost wall the pass runs after that mapping.
  * A face that is not wide has `far = {-1, -1}`.
* **They are recomputed in full at every build**, never copied: a far cell is two cells away, outside the squares the incremental rule
  covers (a refinement two cells away changes `far` without making the leaf dirty). The pass costs less than the copy would.
* `need_[level]` gets the far codes of the level's wide faces, so `prepare` refreshes their rows.
* `mark_face` marks the far codes of a wide face, so `face_dependencies`, the halo, `halo_reach` and the check in `localize` see them.
* `CellFluxEngine::localize` re-indexes `far_` with the leaf map (a far code that is not kept on a face the rank computes is an error, as
  for `stencil_`); `same_tables` compares `far_` and `wide_`; `memory_bytes` counts them.

### 4.2 Ghost layers at `boundary_order = 2`

A physical side that takes ghost layers (not periodic, not non-reflecting: `side_ok`) gives three layers, mirrored as the uniform slab:
layer `d` is `ghost_w(side, U[cell d])` of the `d`-th interior cell. For the three faces that read them:

* the boundary face of leaf `c`: cells `(n2, n1, c | g0, g1, g2)` with `n1 = nbr_[c, inward]`, `n2 = nbr_[n1, inward]`,
  `g_d = ghost_w(side, U[c, n1, n2])`. Wide if `n1` and `n2` are leaves of `c`'s level; else the existing four-cell boundary face;
* the first interior face (between `c` and `n1`): `far` on the boundary side is `g1`, `s0` is `g0`;
* the second interior face (between `n1` and `n2`): `far` on the boundary side is `g0`.

In `fluxes` this extends the existing replacement of a replicated entry (`gtmp0`, `gtmp3`) by two more temporaries per side; no table.
Without `boundary_order = 2`, and on periodic / non-reflecting sides, nothing changes: replication, and a first-order boundary face.

A moving wall keeps the internal energy of the interior cell (as `ghost_state` does today); `fill_ghost_slab` is the reference for the
bits, checked by the level-0 test of the cavity (section 6).

### 4.3 Flux path

* `MusclReconstruction::weno5_value(c0, c1, c2, c3, c4)` exposes the inline `weno5z` of `Schemes.cpp` (as `weno3_value` does), so the
  adaptive path calls the same function with the same `weno_eps_`.
* In `fluxes`: `const bool weno5 = scheme == Weno5`, and the WENO3-Z flag becomes `weno3 = scheme == Weno3 || weno5` for every face that is
  not wide. A wide face runs `recon6`: per variable `wl = weno5_value(a, b, c, d, e)`, `wr = weno5_value(f, e, d, c, b)` in the uniform
  loop order, then `prim_to_cons`, then the Riemann batch (`face_flux_batch`), like the other scalar faces.
* No vector twin. The uniform WENO5-Z is scalar as well; a twin only if a profile shows the reconstruction in the way, and then for both.
* 3-D: the same per face along its normal; nothing transverse is read.

### 4.4 Periodic axes shorter than three cells

The uniform solver replicates instead of wrapping when a periodic axis has fewer than three cells; the adaptive codes always wrap. For
the level-0 identity the adaptive engine rejects `weno5` on a periodic axis with fewer than three base cells, in `Params.cpp` (a mesh that
small has no use for a six-cell stencil; the alternative was `wide` false on such an axis).

### 4.5 MPI

* The reads of a wide face reach one more same-level cell on each side. `reach_rings` (2-D `d = 0`: 6, 3-D: 11) is measured, not derived:
  phase W4 runs `halo_reach` with `weno5` on the meshes of `halo_reach_bounds_the_halo_of_any_partition` and
  `cellparallel3d_halo_reach_lies_within_the_region_rings` and raises the constants where the measurement plus one ring asks for it (fixed
  constants per scheme, no parameter).
* Nothing else changes: the far leaves are same-level leaves, read in a stage of their own level, so the stage and final lists come out
  of `face_dependencies`; a share that is too small is detected by `partition_local` and retried.
* The uniform rejection in `Params.cpp` becomes `weno5 && use_mpi && !use_amr` (decision 3.2), and its message says that AMR runs do.

### 4.6 What does not change

The seams (coarse-fine faces, registers, subcycling), the virtual cells, the indicator, the remesh, checkpoints and hot starts, the viscous
terms, sources, body forces, the mass-flow and non-reflecting boundaries, and every result of MUSCL and WENO3-Z runs (their tables are not
touched: `far_` and `wide_` are empty).

## 5. Phases and exit criteria

| phase | work | exit |
|---|---|---|
| W0 | Read-only: the uniform solver with `weno5` next to a ghost wall and a mask body (cylinder, Cd against the WENO3-Z run), as the figure the adaptive run of W7 is compared with. | figure written here |
| W1 | `weno5_value`; `far_` / `wide_` in `build`, `same_tables`, `localize`, `memory_bytes`; `need_`; `mark_face`. No flux change yet. | all tests pass, no table of a MUSCL / WENO3-Z run changes (`same_tables`), incremental == full build with `weno5` |
| W2 | `recon6` in `fluxes`, replication at the sides, `boundary_order = 1`. Remove the 2-D rejection for serial runs. | level-0 bitwise against the uniform solver, 2-D, periodic and wall cases |
| W3 | Ghost layers three deep (`boundary_order = 2`), boundary faces. | level-0 bitwise with `boundary_order = 2` (cavity, moving wall, inlet / outlet) |
| W4 | Octree: remove the rejection in `build`; the same tests in 3-D. MPI: measure the reach, raise `reach_rings`, lift the rejection with AMR. | level-0 bitwise 3-D; serial == 1-7 ranks, no reach retry in the tests |
| W5 | Static and dynamic refinement: conservation, the seam tests, Sod on a refined band. | section 6, items 3-5 |
| W6 | Regression references `amr_weno5_*` (created with `--create-missing` after W2-W5 pass), `amr_mpi.py --check` cases, CI. | `ALL MATCH`, serial and 4 ranks |
| W7 | Validation studies and bands (section 7); `Params.cpp` messages, GUI (`gui/setup_dialog.py`), parameter page, AMR page, scheme table. | studies run, docs build |

W1 to W3 are the core and can be reviewed alone; each later phase leaves the tree releasable.

## 6. Tests (unit, written with the phase)

1. `cellflux_far_cells_equal_a_full_build`: after random adapts, `far_` / `wide_` of the incremental build equal those of a full build
   (2-D and 3-D); a refinement two cells from a face turns its `wide` off although both its leaves are clean.
2. `celladvance_level0_weno5_steps_match_uniform_steps_bitwise` (2-D) and the 3-D twin: periodic vortex, wall-bounded case with
   `boundary_order = 1` and `2`, a moving wall, inlet / outlet. The uniform references are the solver itself (its `vortex_weno5_rk3`,
   `tgv3d_weno5_rk3`, `cavity_weno5_b2`, `cavity3d_weno5_b2` regression cases cover the same configurations).
3. `celladvance_weno5_conserves_across_level_jumps`: closed box, static refined box (`amr_force_box`) and a remeshing run; mass, momentum
   and energy to 1e-14 relative (the control with the registers off must fail, as in the WENO3-Z test).
4. `celladvance_weno5_sod_on_a_static_refined_band`: no new extremum beyond the WENO3-Z run's, the same test shape as
   `celladvance_weno3_sod_on_a_static_refined_band`.
5. `cellflux_weno5_seam_faces_equal_weno3`: on a mesh with a jump, the flux of every face that is not wide equals the flux of a WENO3-Z
   engine on the same state, bit for bit (choice A's defining property).
6. Distributed: the existing emulated-rank tests (`FakeHub`, NaN-poisoned halos, `shadow`) with `weno5` in 2-D and 3-D, 1 to 7 ranks;
   `halo_reach` within `reach_rings`; `region_retries == 0`.
7. `Params`: `weno5` + `use_amr` accepted with SSP-RK3, rejected with RK2 and on a periodic axis shorter than three cells.

For every phase: the full regression first (`regression.py --mpi 4`), then `amr_mpi.py --check`, and the dumps of a MUSCL and a WENO3-Z
adaptive run before and after must have the same hash (also with `FENICE_NO_SIMD=1` and `FENICE_FULL_REBUILD=1`).

## 7. Validation (bands fixed here, before the first run)

1. **Manufactured solutions on a static refined square** (`mms_amr_weno.py` with a WENO5-Z row; periodic Euler and Navier-Stokes).
   Bands on the finest pair of the quick grids: L2 order at least 1.5, L-infinity at least 1.0 (the bands of the WENO3-Z rows: the seam is
   the same). The `--full` L-infinity result of WENO3-Z at the corner (0.60, 0.94) is expected again and is reported, not a failure of
   WENO5-Z. Reported without a band: the ratio of the L2 error to the WENO3-Z run on the same mesh, inside and outside the square.
2. **Uniform level** (already covered by `mms_weno.py`): unchanged, because level 0 is bit-identical.
3. **Isentropic vortex** on an adaptive mesh that follows the vortex: the density error at the final time must not exceed that of the
   WENO3-Z run on the same mesh (ratio at most 1.0); the ratio to the uniform WENO5-Z run on the finest grid is reported.
4. **Sod and Shu-Osher** on an adaptive mesh: the L1 density error must not exceed the WENO3-Z adaptive run's by more than 5 %; no new
   over- or undershoot above 1e-3 of the jump. The near-vacuum tube runs at CFL 0.3 as in the uniform solver.
5. **Lid-driven cavity** with `boundary_order = 2`, adaptive: centreline velocities within the band of the existing WENO3-Z adaptive
   cavity study.
6. **Cost**: wall time per step against WENO3-Z on the same mesh (the scalar WENO5-Z reconstruction is paid on the wide faces only),
   serial, 16 threads and 4 ranks; reported.

A result outside a band is reported as outside; the band is not moved after the run.

## 8. Later option: a three-deep halo in the uniform solver (not part of this work, decision 3.2)

* `MPIDecomposition::exchange_halos` takes the depth (2, or 3 for WENO5-Z); the strips then need at least three cells
  (`MPIDecomposition` constructor, its error message); `Halos::depth` already carries it to `reconstruct_weno5`.
* `Simulation::assemble` / `assemble_3d`: the interface faces take `(*halos->lo)[(D - 1) * nx + i]` with the run's depth; the viscous
  ring (`PaddedFrame`), the IBM halos and the mass-flow average keep their two layers.
* Tests: `MPI == serial` bit for bit for `vortex_weno5_rk3`, `tgv3d_weno5_rk3` and the cavity cases on 2 to 4 ranks, x- and y-strips; the
  MUSCL and WENO3-Z dumps unchanged. Then the uniform rejection in `Params.cpp` and the GUI message go.

## 9. What choice B would add to this plan

* Keys and rows for second-layer virtual cells (interpolated at depth 2, mean-type at distance 2), `nbr_raw_` at distance 2; all of them
  inside the incremental rule only if the dirty marking grows to two cells, otherwise recomputed in full.
* `fluxes`: a six-cell coarse-fine branch (fine side from three fine cells and two interpolated ones, coarse side from the coarse leaf,
  two cells behind it and two mean-type blocks ahead).
* `reach_rings` for the level differences 1 and 2, re-measured; a deeper halo in coarse cells.
* Tests: uniform flow preserved across every jump configuration (2:1 in both directions, corners, double jumps), conservation, and a
  measurement of the time seam (Sod and the vortex with one and with three levels, fixed CFL) before any claim.
* It is the larger half of the work and, by section 2, not expected to change the observed order.

## 10. Other open items of the adaptive engine

* **MPI remesh cost.** Each rank builds its new share over the old one: only the margin around its owned leaves is fetched and looked up,
  and the faces, neighbour codes and stencils of the leaves with no change next to them are copied. On the 720x360 airfoil (268 k leaves, 10
  remeshes that all change leaves, ranks pinned to physical cores) the remesh is 2.00 / 1.23 / 0.83 s on 2 / 4 / 8 ranks against 0.90 s
  serial. What is left, per rank on 2 ranks: the fetch of the region the indicator reads 0.25 s, the indicator 0.15 s, the ripple, merges
  and migration 0.18 s, the share (mask 0.08-0.13 s, mesh 0.14 s, engine 0.11 s, partition 0.22 s, pruning to the held leaves 0.28 s) and
  the halo lists 0.31 s. These figures include the second walk of the face dependencies in the pruning (about 0.15 s), which now runs
  only with `FENICE_CHECK_HALO=1` (the unit tests and `amr_mpi.py --check`). Candidates, in the order of what they would save:
  * the partition lists (`partition_local`, `rebuild_lists`) are made for every leaf and face of the share although only the margin and
    the changed leaves differ;
  * when no leaf changed since the last remesh (the steady 3-D sphere: 73 k leaves, 8 ranks, remesh 2.35 s of a 10.9 s loop, 1.3 s of it
    the fetch) the region and its fetch plan are the same: only the states need to travel, along the saved plan.
  On one node the eight ranks also slow each other down (about 1.35 times) and wait for each other in the collective calls (about 0.25 s
  of the 8-rank remesh); 16 OpenMP threads beat 8 ranks there (5.0 s against 10.9 s per 200 steps on the sphere), so the OpenMP time step
  matters more than the remesh.
* **OpenMP time step.** Virtual cells are evaluated for every level's call (about 15 % of the step; a per-level read list would gain a few
  percent on three levels and needs a parallel rebuild at each remesh); an AVX2 twin of the 3-D viscous kernels.
* **Validation.** The 3-D ghost-cell wall is validated against itself (self-convergence) and the octree against the uniform solver only, not against an
  external reference; 20 -> 40 cells/D octree runs (`cases/validation/amr/octree_q8.py run --include-40`, 1-2 h each); the ghost wall is first order on
  curved walls (see `docs/proposals/ibm-wall-treatment.md`).
* **Cost figure.** The octree-against-uniform cost ratio in the AMR page predates the 1.7x faster OpenMP step; `octree_q8.py timing` re-measures it.
