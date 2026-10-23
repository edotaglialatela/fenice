# Cell-based AMR for Fenice: design and plan (branch `amr-cell-based`)

Status: P1-P7 done, remesh speed-up and the P8 measurements in section 20 (P6 = MPI, section 18; P7 = the engine behind the CLI, section 19); P1, P2, P3 done (2026-10-22: `CellMesh` topology, `CellFluxEngine` convective residual, `CellAdvance` subcycling with flux replacement); the user accepted the P2 gate (section 9); P4 done (same day); P5 done (indicator, remesh, initial adaptation; section 13); jump closure (b) and the parent-level join done (sections 14-15), the second-order coarse side (c) is now the default (decision of the user, section 17). The user chose the face-variation indicator as the only one (section 11). Written 2026-10-22 after a review of the source of several adaptive-mesh codes.
`main` keeps the patch-based AMR (last good state: commit `f257ae1`); this branch replaces it. To go back: `git checkout main`.

## 1. Why

Goal (user): minimise total wall time on adaptive runs while keeping the MPI load balance acceptable.

What is wrong with the patch design (AMR.cpp, `Simulation::patch_residual`):

1. Cells inside a refined patch are refined whether they need it or not (patch = 8x8 .. 16x16 cells).
2. Every patch kernel looks up its ghost cells by geometry (`leaf_at`, `GridInterp2D`, `ghost_cell`) and has its own
   residual: two implementations of the physics (uniform `assemble`, AMR `patch_residual`) kept equal by tests.
3. Physical edges go through a separate path in which the boundary face is first order (MUSCL stencil not upgraded).
4. Conservation across coarse-fine seams needs flux registers and is exact only for Explicit Euler (O(dt^2) otherwise).

What the reference design does (a cell-based code whose source was read for this plan): a tree of cells plus an interface (face) list; each interface
computes its Riemann flux once and adds it to both cells (area / volume, times a time-weight coefficient on the coarse side),
so conservation follows from the data layout; refinement per cell from a face-variation indicator diffused twice; 2:1 inside
the refine/unrefine decision; recursive subcycling by 2, finer level first, no time interpolation; the coarse side of a
level jump uses its unreconstructed state; Morton keys for the domain decomposition; wall faces are second order.

## 2. Design

### 2.1 Cells
* The base mesh is the user's `nx x ny` level-0 grid (no patches, any `nx`, `ny`). Refinement ratio 2, `amr_max_level` levels.
* A cell is identified by `(level, i, j)` (integer coordinates at its level). Morton key `(level, morton(i, j))`.
* Only leaves exist as data. Flat arrays indexed by leaf: `level[]`, `ci[]`, `cj[]`, `U[]` (Field), kept sorted by
  `(level, morton)` so each level is a contiguous run and neighbours are close in memory. No per-cell objects.
* Parent of a leaf is implied by the key (`(level-1, i>>1, j>>1)`); the tree is not stored.
* Neighbour lookup happens only at remesh time, through a hash map key -> leaf index (never inside the residual, see 2.2).
  Across a side: same level (one leaf), coarser (the leaf of the parent-level neighbour), or finer (the two children
  touching the side).
* Balance: **corner (vertex) 2:1 balance**, not face balance only. With face balance alone the far children of the block
  beyond `c2` can be two levels finer, and the viscous cross terms, the IBM fringe and the halos reach diagonal neighbours.
  Corner balance keeps every stencil within one level of the cell that reads it, so the virtual values of 2.3 are defined by
  injection / the mean of the leaves under a block (recursive: the far children of a refined block may be finer, see section 9).

### 2.2 Faces
* A face belongs to the level of the finer of its two cells. Kinds: same-level, coarse-fine (the fine leaf and the coarse
  leaf; the coarse side is covered by two fine faces), boundary (one cell, a domain edge with its BC type).
* Face record: `left`, `right` (leaf index or -1), `axis`, `level`, `kind`, length. Built after each remesh.
* **Precomputed stencils.** At each remesh every face gets its stencil indices: `c0..c3` along the axis (with codes for
  virtual cells: "injection of leaf k", "mean of the four children of block b"), and the cells whose gradients enter the
  viscous cross terms. Every cell gets the same for its gradients and its indicator. The residual reads index arrays only:
  no hash map, no tree walk, no geometry in the time loop.
* Cell-centric gather: face fluxes are computed in a face loop into `F[face]` (parallel, no write conflicts), then each cell
  sums its own faces through a CSR list. Deterministic and bitwise independent of the thread count.

### 2.3 Reconstruction and fluxes (the bitwise anchor)
* MUSCL per face needs `c0..c3` along the axis: the left cell `c1`, the right cell `c2`, and their outer neighbours `c0`,
  `c3`. Same level: the real neighbours. Beyond a coarser cell: injection of the coarse value (virtual fine cell).
  Beyond a finer cell: the mean of the four children (virtual coarse cell).
* Therefore at a level jump the coarse side of a fine face is first order (`c3 - c2 = 0`), as in the reference design; the fine side keeps
  second order. Free-stream is preserved exactly across jumps.
* The limiter acts on raw differences `(c1 - c0, c2 - c1)` exactly as `MusclReconstruction::reconstruct` does today, and the
  cell residual uses the existing operation order `((0 - (fr - fl)) - (ft - fb)) / area`, so a tree with only level-0 leaves
  reproduces the uniform solver **bitwise**. (Dividing slopes by distances and multiplying back would change bits: do not.)
* **Parity checklist (what the level-0 equality has to reproduce), obtained by calling the existing code, not by
  rewriting it** (`RiemannSolver::face_flux`, `apply_limiter`, the BC ghost code of `BoundaryManager::apply_side`,
  `get_padded_U`, `CrossGhosts` rules):
  - no reconstruction on a boundary face, with the primitive round trip `prim_to_cons(cons_to_prim(.))` kept;
  - the edge-replicated `c0` / `c3` at the first interior face next to a physical boundary;
  - the wrap-around halos of periodic axes;
  - the Riemann ghost state of every BC type;
  - the viscous padded state (edge padding with BC overrides per BC type) and the `CrossGhosts` edge rules (periodic wrap,
    zero gradient at physical boundaries);
  - body force first, then source terms, in the residual.
  Consequence for the boundary: the AMR engine must first reproduce today's first-order boundary face (invariant 2). The
  second-order boundary face is a later switch applied to **both** engines.
* Viscous terms are already face-based in the solver (two-point normal derivative, cell-gradient average for the cross
  terms); they port directly, with the neighbour graph replacing the padded array.

### 2.4 Time stepping and conservation
* Same recursion as today: level `l` advances by `dt_l = dt_0 / 2^l`, the finer level first (two substeps), then the coarse
  level, with the coarse neighbour frozen at its start-of-step state (no time interpolation, like the reference design).
* Flux replacement instead of reflux: the fluxes of a coarse-fine face are computed on the fine side at every fine stage and
  accumulated with the time-scheme stage weights `b_i` (new `TimeScheme::stage_weights()`: Euler `[1]`, SSP-RK2 `[1/2, 1/2]`,
  RK4 `[1/6, 1/3, 1/3, 1/6]`) into a per-(coarse cell, side) register. The coarse cell uses the accumulated time-average
  flux for that side in every stage of its own advance, so the coarse update contains exactly the integral of the fine
  fluxes: conservation to roundoff for every scheme (the patch design is exact only for Explicit Euler).
* **Plumbing to add (it does not exist on the AMR path):** `advance_fixed_dt` takes no current time and never sets
  `TimeScheme::stage_time`; it must take the time of the step and set the stage times as `step()` does (needed by the
  unsteady sources that P4 allows with AMR). The same hook gives the residual its **stage index**, used to pick the weight
  `b_i` when accumulating the fine fluxes; a check asserts that the number of residual calls of a step equals the number of
  weights.
* **No snapshot copy.** Today every level pass copies all leaves into `U_ghost`. The other levels do not change while one
  level advances, so the face engine reads their `U` directly: one O(N) copy per pass saved.

### 2.5 Regridding
* Indicator per leaf: the Lohner ratio of the current engine, with neighbours from the neighbour graph (virtual cells at
  jumps). Refine above `amr_refine_threshold`; derefine a sibling group of four when all four are below
  `amr_derefine_threshold` and 2:1 allows it; `amr_buffer_layers` rings of neighbours (faces and diagonals) added around
  flagged cells. 2:1 is restored by rippling refinement to coarser neighbours not by forbidding refinement.
* Data transfer: conservative injection on refinement, area-weighted mean on derefinement (as today).
* Parameters kept: `use_amr`, `amr_max_level`, `amr_refine_threshold`, `amr_derefine_threshold`, `amr_refine_every`,
  `amr_init_levels`, `amr_buffer_layers`, `amr_refine_ibm`, `amr_indicator`. `amr_patch_nx/ny` disappear (decision 1).

### 2.6 MPI
* Replicated metadata (keys of all leaves, owners) on every rank, state only on owner and halo: as today.
* Baseline partition: each level cut into contiguous Morton blocks of equal count (every level is advanced separately by the
  recursion, so per-level balance is what keeps ranks busy). **Halo = exactly the set of cells referenced by the stencils
  (2.2) of the faces and cells this rank owns** (a geometric "two own cells" is ambiguous at level jumps). Exchanges per stage for the level being advanced, one buffer per
  rank pair, as `AMRParallel::exchange` does now. Requirement kept: MPI == serial bitwise for any rank count.
* Load balance is measured, not assumed (section 5). A weighted single-curve partition is an alternative, to be compared.

### 2.7 Physics ported
IBM (mask by cell centre, forcing, fringe/solid reset through the neighbour graph), body force, source terms (now allowed
with AMR: manufactured solutions on adaptive grids, also unsteady), all boundary conditions (the boundary face gets a
ghost state like any interface; the second-order boundary face of the uniform solver becomes possible here), non-reflecting
outlet (tangential derivatives from neighbours).

### 2.8 Output and restart
* VTK: one `.vtu` of leaf quads per rank per output (+ `.vtm`, `.pvd` as now). Python reader `fenice.read_vtu`. Before
  switching (P7) check what each AMR validation script reads: today `airfoil_amr.py`, `sod_amr.py` read `final.npz`
  (`U` and `leaf_meta` = bounds and level per leaf), not VTK.
* `dump_state` keeps `U` plus per-leaf metadata (bounds, level) so the validation scripts keep working. Hot start from a
  uniform checkpoint is interpolated onto the initial tree as today.

## 3. Invariants (must hold at every merge point)
1. All uniform references bitwise (the 32 non-AMR ones of the 38 in `regression.py`; the 6 AMR ones change, see below),
   serial and MPI.
2. A tree with only level-0 leaves equals the uniform solver bitwise (residual and steps), for every BC type.
3. Conservation to roundoff across level jumps, for Euler / SSP-RK2 / RK4, viscous included.
4. MPI == serial bitwise for any rank count; results independent of the OpenMP thread count.
5. `fp:precise`, no FMA contraction, `kOmpMinWork`, OpenMP 2.0 rules (thread-local max + critical).

### Invariants of the existing AMR unit tests that carry over
Conservation (viscous included; `amr_reflux_conservation`, `amr_viscous_reflux_conservation`), level-0 / tiled equals
uniform (`amr_root_patch_matches_uniform_residual`, `amr_same_level_patches_match_uniform_residual`,
`amr_tiled_matches_uniform_steps`), IBM masks and solid reset across levels (`amr_ibm_masks_and_reset_across_patches`), the
body kept at max level (`amr_body_kept_at_max_level`), marking, buffer and the anti-thrash guard (`amr_marking_and_buffer`,
`amr_error_indicator`), ownership and halos (`amr_ownership_and_halo`), conservative refine / derefine, 2:1 balance.

## 4. Phases and acceptance
| Phase | Content | Acceptance |
|---|---|---|
| P0 | Baseline numbers from the patch AMR on `main` (cells, wall time, errors, MPI scaling) and the flagged-versus-held measurement | numbers recorded in section 5 of this document (`results/` is gitignored) |
| P1 | Leaf arrays, keys, neighbour lookup, faces + CSR, refine/derefine, ripple 2:1 | tests: faces cover every side exactly, 2:1, conservative refine/derefine, Morton order |
| P2 | Convective residual on faces (Godunov, MUSCL, all limiters, Riemann variants, all BC types incl. periodic) | level-0 tree == uniform residual bitwise; free stream across jumps exact; **wall-time gate (section 5)** |
| P3 | Subcycling with flux replacement, stage weights | conservation to roundoff (Euler/RK2/RK4); level-0 steps == uniform steps; Sod vs exact |
| P4 | Viscous, body force, sources, IBM, NRBC | same tests with physics; MMS on an adaptive grid (steady and unsteady) |
| P5 | Indicator, buffer, balance, remesh, `amr_init_levels`, IBM body at max level, hot start | tests of marking/transfer; Sod and airfoil run |
| P6 | MPI | bitwise vs serial on 1..7 ranks; per-level balance report |
| P7 | VTK, restart, python readers, params, GUI keys, docs, validation scripts; delete the patch engine | all studies run; docs rewritten |
| P8 | Performance and load balance versus `main` | wall time per accuracy not worse on any study; report |
| P9 (separate decision) | 3-D octree | none yet |

The AMR regression references (6) change in P3..P5 because the numerics change (different coarse-fine closure, no patch
seams); they are regenerated only after validation against uniform fine runs and exact solutions, with your approval.

## 5. Measurements and the wall-time gate

### P0 results (patch AMR on `main`, the study configurations; taken 2026-10-22)
* Airfoil (`amr/airfoil_amr.py`, 192x96 base, 2 levels, 16x16 patches, indicator `density+velocity`, `amr_refine_ibm`):
  AMR 52 224 cells, 22.1 s; uniform 768x384 294 912 cells, 244.1 s; uniform 192x96 18 432 cells, 2.5 s.
* Over-refinement, measured on the final snapshot: each leaf above level 0 restricted 2x2 to the next coarser resolution, the
  Lohner indicator evaluated per cell there (threshold 0.3), flagged cells plus one buffer ring, 4 fine cells per flagged cell:
  - airfoil: level 1 holds 14 336 cells, 568 needed (4 %); level 2 holds 24 576, 5 112 needed (21 %); the refined levels
    together need 15 % of what they hold: 18 992 cells in total against 52 224 (2.7x fewer);
  - Sod (`amr/sod_amr.py`, 10x2 patches, 4 levels, density indicator): refined levels need 20 % of what they hold: 544
    cells against 2 720 (5x fewer).
  These are **upper bounds on the saving**: one end-of-run snapshot, the cells near an immersed body that `amr_refine_ibm`
  forces to the finest level are not counted as needed (a cell-based engine would refine a band around the surface, not the
  bounding box), and the history of the flagged region over the run is not included.
* MPI baseline (`amr/amr_mpi.py`, airfoil on a 720x360 base mesh with 24x24 patches, 50 steps, 16 logical processors
  of an 8-core machine; also in `docs/source/modules/AMR.md` on `main`): serial 16.4 s; OpenMP 2 / 4 / 8 / 16 threads
  8.8 / 5.1 / 3.2 / 2.9 s (speed-up 1.9 / 3.2 / 5.2 / 5.6); MPI 2 / 4 / 8 / 16 ranks 8.9 / 5.2 / 4.0 / 4.6 s (speed-up
  1.8 / 3.2 / 4.1 / 3.5), cells on the busiest rank 61 / 41 / 26 / 20 % (ideal 50 / 25 / 12.5 / 6.25 %). Every final state is
  bitwise the serial one. P8 must report the same table for the new engine.

### Wall-time gate at P2 (go / no-go)
The only goal is less total wall time. At the end of P2, measure the residual cost per cell of the face engine against the
uniform `assemble` on the same large mesh (same physics, same threads). Projected gain = (cells held / cells needed, from
above) / (per-cell cost ratio). **Stop before P3 if the projected gain is below the threshold agreed with the user** (open
decision 5; suggested: 1.5x on the airfoil study).

## 6. Risks
* Speed of a cell-gather loop against structured patch loops: mitigated by Morton-sorted flat arrays, a face loop without
  per-face objects, and measuring in P2 before going further.
* IBM fringe and solid reset use structured neighbour access today; the neighbour graph must reproduce the same sets.
* Viscous cross gradients at level jumps need a defined virtual-neighbour rule (injection / mean of children, as for MUSCL).
* Per-level partitions cost more halo traffic than one spatially compact partition: measure in P6/P8.
* The first-order coarse side at jumps is local; an interpolated second-order variant can be added later if the study shows
  the jump seams dominate the error.

## 7. Decisions (the user, 2026-10-22)
1. `amr_patch_nx/ny`: **removed** (the base mesh is the level-0 grid).
2. First-order coarse side at level jumps: **accepted** for now.
3. 3-D octree: decided after the performance phase (P8).
4. The AMR regression references (6) are regenerated after validation, with the user's approval.
5. Wall-time gate at P2: **no threshold fixed**. At the end of P2 report the measured projected gain; the user decides go / no-go.
6. Merging into `main`: **no** (not decided; the branch stays separate).

## 8. P1 as built (`CellMesh`, `cpp/include/fv/CellMesh.hpp`, `cpp/src/CellMesh.cpp`)
* Leaves are parallel arrays `level, ci, cj` (+ `U`) sorted by (level, Morton key); `level_begin` gives each level's run. A hash map
  key -> leaf exists for lookups at remesh time only; nothing iterates over it.
* `faces` (`CellFace`: `left/right`, `level` of the finer cell, `tpos`, `axis`, kind Interior / CoarseFine / Boundary, `length`), sorted
  by level (`face_level_begin`). Same-level faces are created by the lower-side leaf, coarse-fine faces by the fine leaf, boundary
  faces by the leaf at the edge. Periodic axes wrap (`left` = last column); `nx = 1` periodic is accepted (a leaf is its own
  neighbour), `nx = 2` periodic gives two faces between the same pair.
* Gather list `side_ptr / side_face`: for leaf `c` and side `s` (Left, Right, Bottom, Top) the faces on that side, one or two, lower
  transverse position first, so the P2 residual can keep the order `((0 - (fr - fl)) - (ft - fb)) / area`.
* `adapt(refine, coarsen)`: refine with a ripple (a leaf becoming level l+1 forces every corner neighbour of level < l to refine,
  to a fixpoint); merge four siblings (mean of the four, summed `((a+b)+(c+d))*0.25`) only if all four are flagged, none is refined,
  and no corner-neighbour block of the parent holds a leaf that is deeper than level l or is refined in the same call; refusals are
  decided on the state before the call (order independent). Refine then merge restores `U` bitwise.
* Stencils (`c0..c3`, virtual cells) are not in P1: they are P2, kept as parallel arrays so the face record does not change.
* Tests (`cellmesh_*` in `test_main.cpp`): an independent brute-force oracle in finest-level integer coordinates checks tiling,
  corner balance for every touching pair, exactly one face per shared segment (orientation, kind, level, boundary side), side
  coverage, ordering, neighbour summaries, for periodic and non-periodic axes, `nx = 1, 2`, random adapt sequences, plus conservation
  to 1e-12 on random states and hand-built ripple / merge-refusal cases.

## 9. P2 as built (`CellFluxEngine`, `cpp/include/fv/CellFlux.hpp`, `cpp/src/CellFlux.cpp`) and the wall-time gate
* Stencils `(c0..c3)` per face are index arrays into the primitive array `W` (leaf, or `N + virtual cell`). Beyond-cell rules: same
  level = the cell, coarser = injection, finer = virtual cell `((a+b)+(c+d))*0.25` of the leaves under the neighbour block (recursive,
  because the far children of a refined block may be finer than the near ones: corner balance does not forbid it), domain edge = replication,
  periodic = wrap. Coarse-fine face: the fine side is reconstructed against the injected coarse value, the coarse side keeps its cell
  value (`c3 = c2` / `c0 = c1` makes the same formula first order there).
* Boundary ghosts come from the existing code: `BoundaryManager::ghost_state` was extracted from `apply_side` (a pure move; the uniform
  path calls it too). Supported: wall, no-slip, moving wall, velocity inlet, mass-flow inlet (side average weighted by face length, which
  is the plain sum on a uniform side), outlet, transmissive, periodic. **Non-reflecting is rejected until P4** (it needs the tangential neighbours).
* Gather per leaf and side `G = sum(F * length)`, residual `((0 - (Gr - Gl)) - (Gt - Gb)) / area` as `assemble`.
* Tests: `cellflux_level0_matches_uniform_residual_bitwise` (7 boundary-type sets x 5 schemes incl. all limiters, the three Riemann
  solvers, first order; residual and `S_max` equal bit for bit to `Simulation::residual`), `cellflux_free_stream_is_exact_across_level_jumps`
  (R == 0 exactly on random adapted meshes, periodic and transmissive), `cellflux_residual_is_conservative_on_adapted_meshes`.
  Regression (`--mpi 4`) unchanged: ALL MATCH.
* Not in P2 (by design): `nx` or `ny` = 1 on a periodic axis is exact in the topology tests but not compared with the uniform solver (its
  MUSCL wrap needs 2 cells); no time stepping; no viscous / IBM / sources.

### Gate measurement (`build/release/bench_cellflux.exe 768 384 20`, MUSCL Van Leer, HLLC, Euler, 8-core machine)
The bench asserts that the face residual equals the uniform one bit for bit on the 768x384 mesh before timing. Timings fluctuate by
10-20 % between runs on this machine (8-thread uniform: 24-31 ns/cell), so ratios are quoted as ranges.

| | 1 thread | 8 threads |
|---|---|---|
| uniform `Simulation::residual`, 294 912 cells | 139 ns/cell | 24-31 ns/cell |
| face engine, same level-0 mesh | 124 ns/cell (0.90x) | 22-26 ns/cell (0.85-0.92x) |
| uniform at the adapted size (384x192, 73 728 cells) | 138 ns/cell | 31 ns/cell |
| face engine, adapted (68 283 leaves: 12 899 / 11 048 / 44 336 on levels 0 / 1 / 2, 731 virtual cells) | 126 ns/cell (0.91x of matched) | 26 ns/cell (0.86-1.1x of matched) |
| stencil build alone | 2.1 ms | 2-3 ms |
| no-change `adapt` + build (serial code) | 18 ms = 2.1 residuals | 17-20 ms = 7-13 residuals |
| remesh every 10 coarse steps (SSP-RK2, subcycling x2; 1 coarse step = 6.2 full-mesh residuals) | 3.3 % of the run | 13 % of the run |

Reading: the per-cell cost of the face engine is the same as the uniform kernel (0.85-1.1x, inside the noise at 8 threads; 0.9x at 1 thread).
**The remesh is serial and does not scale with the threads**: with 8 threads it costs about 13 % of the run at `amr_refine_every` = 10
(plus the indicator and the data transfer, not yet measured), against 3 % at 1 thread. A parallel / incremental rebuild is a P5/P8 item.

Projected gain over the patch AMR (P0 airfoil numbers of section 5). Wall time goes as sum(cells x 2^level) because the fine levels take 2^l substeps:
held 13 312 + 2 x 14 336 + 4 x 24 576 = 140 288; needed (upper bound) 13 312 + 2 x 568 + 4 x 5 112 = 34 896; weighted ratio 4.0x (the plain
cell-count ratio is 2.7x). Divided by the per-cell cost ratio (about 1) and reduced by the remesh share, the projected residual-cost gain is
about 3.5-4x at most at 1 thread and about 3.5x at 8 threads. It is an upper bound (end-of-run snapshot, no refined band around the body, no
history of the flagged region). The threshold is the user's decision (section 7, item 5).

## 10. P3 as built (`CellAdvance`, `cpp/include/fv/CellAdvance.hpp`)
* `TimeScheme::stage_weights()` / `stage_offsets()` (new virtuals, shared code; the patch engine is untouched): Euler `[1]`/`[0]`, SSP-RK2
  `[1/2,1/2]`/`[0,1]`, RK4 `[1/6,1/3,1/3,1/6]`/`[0,1/2,1/2,1]`. `advance_fixed_dt` is unchanged; `CellAdvance` sets `stage_time` itself
  in the residual callback (`t`, `t + 0.5 dt`, `t + dt`, as `step()` does), which is what the unsteady sources of P4 need. BDF2 stays rejected with AMR.
* `CellAdvance::step(dt, t)`: level l = one ODE system on its contiguous run of leaves, `advance_fixed_dt` on a copy of the slice, whose
  stage states are written into `mesh.U` for the neighbours, the final result copied back after the scheme returns. Finer level first,
  two substeps of `dt/2` (never clamped inside the recursion: `dt/2` is exact), the registers of level l+1 are zeroed before.
* Per-level engine pieces: `compute_level_fluxes` (faces of level l; the primitive state is refreshed only for the cells their stencils
  read, `need_[l]`), `accumulate_registers` (`reg += 0.5 b_i F_i` on the coarse-fine faces of the fine level, every stage), `gather_level`
  (a face of level l+1 on a side of a level-l leaf reads the register in every stage). `residual()` is the same two pieces over all faces
  (bit-identical, guarded by the P2 tests and the bench assert).
* Known time seam (as in the reference design, accepted): the fine level reads its coarse neighbours at the start of the coarse step, the coarse level reads
  (through its virtual cells and the first-order stencil of its same-level faces) the finer leaves at the END of the coarse step. The fluxes at
  the jump are exact in the time average (register), the reconstruction stencils are not time-consistent. Measured on Sod with waves crossing the jumps
  (below): no visible effect on the error, no overshoot.
* Tests: `celladvance_level0_steps_match_uniform_steps_bitwise` (Euler / RK2 / RK4, three boundary sets incl. periodic, 9x7 and 96x64, level-0
  and a `max_level = 2` mesh without refinement; 4 steps equal to `advance_fixed_dt` of the uniform solver bit for bit),
  `celladvance_conserves_across_level_jumps_and_the_control_fails` (random 3-level periodic mesh, 6 steps, relative error < 1e-13 for all three
  schemes; with the test hook that makes the coarse side read the last computed flux it is > 1e-8), `celladvance_free_stream_and_stage_times`
  (state unchanged bitwise for Euler / RK2, 1e-14 for RK4 whose weights sum to 1 only up to rounding; the sequence of (level, stage time) equals the
  expected recursion bitwise), `celladvance_sod_on_a_static_refined_band` (bands fixed beforehand: AMR error <= 1.6 x the uniform-200 error and
  < 0.5 x the uniform-50 error, against the exact Riemann solution). Measured: L1(rho) uniform 50 (200 cells) 9.96e-3, uniform 200 (3 200 cells) 2.910e-3,
  AMR base 50 + 2 levels in a static band (2 384 cells) 2.917e-3. Changing the register weight by 2 % makes three of the tests fail.
* Sod with waves crossing the jumps (static narrow bands, thresholds fixed before the run; the first one, band [0.55, 0.75], was first compared with
  the uniform-200 error in the band window and failed by design of the test, not of the scheme: the contact is smeared on the coarse grid before it enters the
  band and that smear persists, so the window error there equals the uniform-50 one, 1.73e-2 against 1.85e-2; the criterion was changed to "below uniform 50"
  and the band [0.4, 0.8] added, where the contact is fine from t = 0 and the rarefaction head and the shock cross the jumps moving out):
  global L1(rho) (a) 9.81e-3 (764 cells), (b) 4.39e-3 (1 424 cells), uniform 50 9.96e-3; error in the window [0.55, 0.75]: (b) 6.539e-3 against
  6.528e-3 for uniform 200 (the pre-existing static band [0.2, 0.92] keeps every wave fine and reproduces the uniform-200 error, 2.917e-3 against 2.910e-3).
  Maximum density 1.0000 (no overshoot), mass conserved to 1e-12.
* Not covered yet: thread-count independence on adapted meshes (all adapted-mesh tests are below `kOmpMinWork`); it is verified in P6 together with MPI == serial.

## 11. Decisions of 2026-10-22 (user): indicator
The user prefers how the reference design follows the gradients and wants ONE indicator: its face-variation indicator replaces the Lohner ratio of
the patch engine (no second indicator). The exact formulas and thresholds are shown to the user before they are fixed (P5).

**The indicator of the reference design** (read from its source):
1. `xi = 0` in every cell of the level.
2. For every face of the level and every enabled variable (density; pressure and velocity magnitude), with c_l, c_r the cell values: `m = min(|c_l|, |c_r|)`,
   floored at 1e-2 (0.1 for the velocity magnitude); `variation = |c_r - c_l| / m`; if `variation >= criteriaVar` both cells get `xi = 1`.
3. Diffusion, 2 iterations ("arbitrary number" in the source): for every face, if either cell has `xi > 0.05` (at a level jump only the FINER cell is tested) both
   cells receive `+0.1`; after all faces, `xi += received`.
4. A leaf is split if `xi >= xiSplit` and no neighbour is at a lower level (no ripple: the split is simply forbidden); a split cell is joined if `xi < xiJoin`, none of
   its children is split and no neighbour is at a higher level. `criteriaVar`, `xiSplit`, `xiJoin` are input parameters (no default in the source).
Reading: step 2 marks the cells on both sides of a sharp jump, step 3 spreads a band of about one (two for a small `xiSplit`) cells around them: that is why the
refinement follows shocks and contacts tightly without a separate buffer parameter, and why a smooth ramp whose per-cell variation stays below `criteriaVar` is not refined.

## 12. P4 as built
* Extra pieces of the shared code (pure moves, regression unchanged): `BoundaryManager::nonreflecting_ghost` (the characteristic outlet, with the
  tangential neighbours passed in `NrbcTangent`; `apply_side` builds them as before), `viscous_ghost` (2-D rule of `get_padded_U`; a test compares
  both for every boundary type), `NavierStokes::prandtl`, `ImmersedBoundary::rho_ref / p_ref`, `evaluate_source_points` (sources at arbitrary points).
* `CellPhysics` (set with `CellFluxEngine::set_physics` before `build`): viscous terms, body force, source expressions (steady cached per mesh,
  unsteady evaluated at `stage_time`, which `CellAdvance` copies from the scheme), the immersed body.
* Viscous: per-leaf neighbour codes (`nbr_`: leaf, virtual cell or physical edge) give the cell gradients (central differences of the four neighbour
  values, the diffusive ghost at an edge); the face flux is the same expression as `NavierStokes` with the normal derivative over the distance of the two
  cell centres (`0.5 (h_L + h_R)`, equal to `h` on a level) and cross terms from the two cell gradients. Viscous face fluxes enter the
  coarse-fine registers next to the convective ones. The viscous time-step speed is evaluated per leaf with its own cell size (a max over leaves).
* Immersed body: leaf masks are cut from the uniform mask of each level (`ibm_factory` on that level's mesh, cached); the forcing is applied to a forced
  copy refreshed per level after its update; `solid_reset` runs once per complete step (called by the driver, not by `CellAdvance::step`). A finer
  neighbour contributes the mean of its fluid leaves to the fringe; the body is meant to sit on the finest level (P5).
* Non-reflecting outlet: rejected in P2, supported now (tangential differences from the neighbour codes).
* Tests: bitwise equality with `Simulation::residual` for 14 viscous boundary combinations (first order and MUSCL, a 96x64 case above `kOmpMinWork`),
  body force, steady and unsteady sources (several stage times), the immersed body (forcing and the solid reset, with and without viscosity),
  the non-reflecting outlet (three transverse weights, with a periodic transverse axis and with viscosity). On adapted meshes: conservation to 1e-13 with
  viscosity (all three schemes; the control without registers fails; removing the viscous register fails the test), a uniform state does not move,
  and a mesh with every leaf at level 1 equals the uniform 64x64 mesh bit for bit (with viscosity and sources, two substeps).
* **Finding (not fixed, the user decides): the first-order jump closure dominates the error of a smooth flow on a half-refined mesh.** Manufactured
  solutions (steady viscous and unsteady Euler), base 32x32 with the lower half refined once, against the uniform 64x64 solution at t = 0.05: L1(rho)
  difference uniform 32 = 1.47e-3 / 1.53e-3, adaptive = 1.64e-3 / 1.95e-3 (the adaptive run is 12-27 % WORSE than the coarse grid although it has 2.5x
  the cells). Away from the two jump lines the adaptive error is 0.6 x the coarse one (4.6e-4 against 7.7e-4); within 0.12 of the jumps it is 1.7-2.0 x the
  coarse one (1.2e-3 against 6.9e-4). A five times smaller time step changes nothing (1.640e-3), so it is spatial, not the time seam. Cause: the coarse side
  of a jump is first order (decision 2 of section 7) and the fine side is reconstructed against the injected coarse value. The fix of the risk list
  (second-order jump closure by interpolating the coarse data) is the user's numerics decision; the test keeps the measured limits (far 0.8 x, near 3 x).
* Resolution study of the finding (`celladvance_jump_closure_study`, steady manufactured solution, base n x n with the lower half refined once, against
  the uniform 2n x 2n solution at t = 0.05; mean absolute density difference per unit area, adaptive / uniform n, in coarse cell widths from the nearest jump):

  | n | fine side, < 1 width | coarse side, < 1 width | fine side 1-2 | coarse side 1-2 | far from the jumps (fine side) |
  |---|---|---|---|---|---|
  | 16 | 8.1e-3 / 5.5e-3 (1.5x) | 1.1e-2 / 7.2e-3 (1.5x) | 0.46x | 0.91x | 0.02x |
  | 32 | 4.4e-3 / 1.6e-3 (2.8x) | 7.1e-3 / 1.8e-3 (4.0x) | 1.3x | 1.3x | 0.02x |
  | 64 | 2.5e-3 / 4.0e-4 (6.2x) | 4.3e-3 / 3.9e-4 (10.8x) | 2.0x | 2.2x | 0.19x |

  The error in the first coarse width next to a jump falls by only 1.5-1.7x per doubling (order about 0.6-0.7) while the uniform grid falls by 4x (order 2),
  so the adaptive solution converges more slowly than the base scheme and the seam share of the error grows with resolution. It is a loss of order at the
  jump, not a constant. Options for the user: (a) keep it, as the reference design does; (b) reconstruct the fine side against the coarse value interpolated to the virtual
  fine position and use the true centre distance (1.5 h_f, not h_f) in the fine cells' differences; (c) a full second-order coarse side. The choice interacts
  with the indicator: a face-variation indicator refines whole smooth gradients, so jump lines will often cross smooth regions, as here.

## 13. P5 as built (`CellIndicator`, `CellAdaptor`, `cpp/include/fv/CellRemesh.hpp`)
* `CellIndicator` implements the four steps of section 11 on the face list (the faces give both leaves of every pair, at the same level or across a jump; at a
  jump only the finer leaf decides whether to spread). Its settings live in a struct (`criterion`, variables, floors, `diffusion`, `diffusion_coeff`,
  `diffusion_trigger`, `split`, `join`); `Params`, `parameters.md` and the meaning of the `amr_*` keys are NOT touched until P7, because the patch engine of this branch
  still reads them. The defaults of the struct (criterion 0.1, density only, 2 iterations of 0.1 above 0.05, split 0.15, join 0.10) are a starting point for the user.
* Flags: refine at `xi >= split` below the finest level, join four siblings when all have `xi < join`; refinement ripples (`CellMesh::adapt`) instead of being
  forbidden next to a coarser leaf. `CellMesh::adapt` now returns whether the leaves changed and skips the rebuild when they did not.
* `CellAdaptor::remesh`: indicator, flags, `adapt`, `CellFluxEngine::build` (only if changed); `init_adapt(initial)`: fill the state at the leaf centres (each leaf on
  its own level's uniform mesh, not by injection), remesh, repeat until nothing changes (at most `max_level + 1` passes), fill once more.
* Immersed body: `set_body(factory, margin)` keeps every leaf whose footprint touches the body or `margin` finest cells around it (the fringe of the solid reset) on the
  finest level, never joined; the footprint is tested on the finest-level mask (not the leaf centre, not the bounding box).
* Tests: the indicator equals a plain structured 4-neighbour implementation on a level-0 mesh; the diffusion rule at level jumps (a coarse leaf does not spread
  into finer leaves that are below the trigger, a flagged fine leaf does); a remeshed step is deterministic, idempotent (a second remesh changes nothing, bitwise),
  conservative, and the finest level stays within the diffusion band; the body and its band sit on the finest level after the initial adaptation and a remesh;
  dynamic Sod (bands fixed beforehand: error <= 1.6 x the uniform-200 one, mean leaves < 0.6 x 3 200, shock on the finest level, mass to 1e-12) passes.
* Sod indicator study (`cellremesh_sod_indicator_study`, base 50 + 2 levels, remesh every 4 steps, L1 density against the exact solution; uniform 50 9.96e-3, uniform
  200 (3 200 cells) 2.91e-3):

  | criterion | density | density + pressure | density + velocity |
  |---|---|---|---|
  | 0.02 | 3.25e-3, 928 leaves | 3.10e-3, 966 | 3.10e-3, 999 |
  | 0.05 | 3.98e-3, 743 | 3.59e-3, 804 | 3.14e-3, 909 |
  | 0.10 | 4.33e-3, 570 | 4.56e-3, 643 | 3.43e-3, 797 |

  With the density alone and a large criterion the contact, smeared over several coarse cells, never shows a per-cell variation above the criterion and stays coarse
  (see `results/cell_amr/sod_cell_amr.png`); a smaller criterion or the velocity brings it in. The choice is the user's.
* Airfoil study (`cell_amr_demo "IBM Airfoil AMR" --vars rho,u --levels 2 --every 5 --tend 6`, base 192 x 96, NACA 0012, Re 1000, t = 6, criterion 0.1, split 0.15):
  22 290 leaves at the end (level 0: 18 084, 1: 454, 2: 3 752), mean 21 153, wall 11.6 s with all threads; against the patch engine on `main` (P0 numbers, same study): 52 224
  cells, 22.1 s, and uniform 768 x 384: 294 912 cells, 244 s. The refinement follows the body, the boundary layer and the near wake (`results/cell_amr/airfoil_cell_amr.png`).
  The accuracy (loads) is not yet compared; that is P8.
* Remesh cost (serial code; 255 remeshes in that run, 13 ms each): indicator 0.23 s, `CellMesh::adapt` 1.25 s, `CellFluxEngine::build` 1.89 s, together 29 % of the wall time. This
  is the first thing to optimise in P8 (incremental face / stencil update, no hash maps and ordered maps in the rebuild, parallel rebuild).
* Figures: `python cases/validation/amr/cell_amr_p5.py` (runs `cell_amr_demo`, writes `results/cell_amr/*.png`).
* **Deviations from the reference indicator** (the user accepts or rejects them): (1) refinement ripples to coarser neighbours instead of being forbidden next to one;
  (2) siblings are joined when all FOUR CHILDREN have `xi < join`, where the reference tests the parent's own `xi`, computed at the parent level; (3) pressure and velocity are
  always tested (the reference skips them where `xi` is already above 0.99, which does not change the result); (4) faces on a physical edge are ignored.
* **Finding: the mesh flickers at the edge of the refined region.** In the airfoil run 13 067 leaves were flagged for refinement and about 11 850 sibling groups joined over 255 remeshes
  (about 50 and 46 per remesh) in a nearly steady flow; the isolated level-1 island at (5.5, 3.1) of the figure has `xi = 0` at the end (it is not in the join / split
  hysteresis band: 25 of the 461 level-1 leaves are, 373 are below `join`). Likely cause, consistent with deviation (2): a marginal gradient is flagged on the coarse leaves,
  the refined children see half the variation per face (the same ramp over twice the faces) and fall below the criterion, they join at the next remesh, and the cycle repeats. Testing the
  parent's own `xi` at the parent level (what the reference does) keeps a refined cell refined while its coarse-level variation is above the criterion and should remove it; this is the
  proposed fix, not implemented (the user decides). The churn also feeds the remesh cost below and, through injection and averaging, a little numerical diffusion.
* **Airfoil accuracy against the cached studies** (`results/validation/airfoil_amr`, same case, t = 6, loads from the monitor / the net momentum flux into the solid leaves, mean over the last
  unit of time, then the value at t ~ 6; the instantaneous Cl at one time is noisy), all runs with all threads of the same 8-core machine:

  | run | cells at the end | wall | Cd mean | Cl mean | Cd(6) | Cl(6) |
  |---|---|---|---|---|---|---|
  | uniform 768 x 384 (reference) | 294 912 | 244 s | 0.1498 | 0.2171 | 0.1655 | 0.3540 |
  | uniform 384 x 192 | 73 728 | | 0.1557 | 0.1717 | 0.1652 | 0.3203 |
  | uniform 192 x 96 | 18 432 | 2.5 s | 0.1397 | 0.1027 | 0.1376 | 0.0910 |
  | patch engine (`main`), 192 x 96 + 2 levels | 52 224 | 22.1 s | 0.1483 | 0.2304 | 0.1542 | 0.2956 |
  | cell engine, criterion 0.1, density + velocity | 22 284 | 11.6 s (remesh 31 %) | 0.1511 | 0.2289 | 0.1656 | 0.3621 |
  | cell engine, criterion 0.05, density + velocity | 23 997 | 12.7 s (remesh 32 %) | 0.1496 | 0.2288 | 0.1628 | 0.3515 |

  Mean loads: Cd within 1 % of the reference for both AMR engines, Cl 5-6 % high for both; the cell engine uses 2.2x fewer leaves and 1.9x less wall time than the patch engine. Cl is the
  quantity that differs from the reference by the same amount for both engines, so that part is not specific to the new engine.
* Remesh cost, one more time: indicator 0.23 s, `CellMesh::adapt` 1.25 s, `CellFluxEngine::build` 1.96 s of a 11.6 s run; the churn above multiplies it.
* After a remesh that changed the leaves the wave-speed bound is recomputed (the viscous part scales with the level of each leaf); the tests and the demo do it.
* Not done in P5 (moved to P7 with the parameters): `amr_init_levels` (the initial adaptation makes `max_level + 1` passes) and the hot start from a uniform checkpoint.

## 14. Jump closure options tried (2026-10-22, after the user chose option (b))
* **Implemented (b)** (`CellFluxEngine::interpolate_jumps`, default on with MUSCL): a fine leaf sees the coarse leaf across a face as a virtual cell holding the coarse value
  interpolated, in primitive variables with the MUSCL limiter on the coarse leaf's own differences (neighbours injected, so no virtual depends on another), to the centre of the
  fine cell that would sit there (0.25 coarse widths from the coarse centre, towards the fine leaf in the normal direction and towards its half in the transverse one). It
  replaces the injected value in the fine-side stencil of coarse-fine faces, in the outer stencil points across a coarser neighbour of same-level faces, and in the gradient
  neighbours of the viscous terms; the coarse side of a coarse-fine face still reads exactly the coarse leaf's state. Without MUSCL the injection stays. Test: on a linear primitive
  field the interpolated values are within 2e-5 of the exact ones at the virtual centre (not roundoff: the coarse leaf's neighbour over the refined block is the mean of
  conservative fine values), against 2.75e-2 for the injected values.
* **Experiment (c)** (`coarse_side_second_order`, default off): the coarse side of a coarse-fine face is reconstructed too with the standard limited formula on the coarse leaf, its
  neighbour across the face being the mean of the fine leaves there.
* **Result** (`celladvance_jump_closure_study`, near-jump error of the adaptive run over the uniform base-grid error, within 2 coarse widths of a jump):

  | closure | n = 16 | n = 32 | n = 64 |
  |---|---|---|---|
  | injection (the original, section 12) | 1.15 | 2.47 | 5.42 |
  | (b) interpolated fine side | 1.15 | 2.52 | 5.88 |
  | (b) + (c) second-order coarse side | 0.78 | 1.06 | 1.51 |
  | (c) alone | 0.80 | 1.16 | 1.81 |

  (b) alone does nothing measurable: the error comes from the first-order coarse side, not from the injected fine-side value. (c) removes most of it (5.4 -> 1.8), (b) on top of it
  helps a little more (1.8 -> 1.5). The ratio still grows slowly with n for (b)+(c) (1.06 -> 1.51), so something second-order-breaking remains at the jump (the time seam, or the
  viscous cross terms; not investigated).
* Airfoil (same case as section 13, criterion 0.1, density + velocity; mean over the last unit of time, then at t ~ 6; uniform 768 x 384 reference: Cd 0.1498 / Cl 0.2171, at t = 6: 0.1655 / 0.3540):

  | closure | Cd mean | Cl mean | Cd(6) | Cl(6) | wall |
  |---|---|---|---|---|---|
  | injection | 0.1511 | 0.2289 | 0.1656 | 0.3621 | 11.6 s |
  | (b) | 0.1509 | 0.2295 | 0.1663 | 0.3621 | 12.7 s |
  | (b) + (c) | 0.1491 | 0.2249 | 0.1637 | 0.3538 | 13.1 s |

  With (b)+(c) the mean Cl error against the reference falls from 5.4 % to 3.6 % and Cl at t = 6 matches the reference (0.3538 against 0.3540), for 3 % more wall time.

## 15. Parent-level join (2026-10-22; deviation (2) of section 13 removed)
* `CellIndicator::evaluate` now works on the level hierarchy: the cells of level l are its leaves plus the split ancestors of the deeper leaves (a split cell holds the mean of its four
  children, `((a+b)+(c+d))*0.25`), stored sorted by Morton key and looked up by binary search (nothing iterates a hash map); xi of every level is computed with the four steps on that
  level's interfaces (two cells of the level, or a cell and a coarser leaf, where only the cell decides and receives; periodic wraps at every level). A leaf is refined on its own xi at its level;
  four sibling leaves are joined when the xi of their PARENT cell at the parent level is below `join` (`CellIndicator::parent_join = false` restores the old children rule, for the test).
  Consequence also named: a leaf next to a refined region now sees the split neighbour's mean at its own level instead of the fine leaves across the coarse-fine faces.
* Test (expectation fixed beforehand): a marginal ramp (coarse-face variation 0.14, fine-face 0.07, criterion 0.1) on a static field reaches a fixed point at once with the parent rule (376 leaves after
  every remesh) and flickers with the children rule (376 / 256 alternating over 10 remeshes), so the test can detect the defect.
* Effect, airfoil (same case, criterion 0.1, density + velocity, closure (b); before = section 13/14):

  | | refine flags | joined groups | final leaves | wall | remesh share | Cd mean | Cl mean | Cd(6) | Cl(6) |
  |---|---|---|---|---|---|---|---|---|---|
  | children rule | 13 067 | 11 827 | 22 329 | 12.7 s | 31 % | 0.1509 | 0.2295 | 0.1663 | 0.3621 |
  | parent rule | 2 601 | 944 | 23 481 | 13.8 s | 39 % | 0.1494 | 0.2300 | 0.1632 | 0.3525 |

  (reference, uniform 768 x 384: Cd mean 0.1498, Cl mean 0.2171, at t = 6 0.1655 / 0.3540.) The churn falls by a factor 5 (flags) and 12 (joins). Cd is now within 0.3 % of the reference; Cl
  is unchanged (+6 %). The indicator costs 7x more (0.23 s -> 1.71 s of the run: the level hierarchy is rebuilt per remesh, with a vector of states per level and binary searches); remesh is 39 % of
  the wall time (indicator 1.71 s, adapt 1.23 s, build 2.28 s), to be reduced in P8. With the second-order coarse side (experiment of section 14): Cd mean 0.1482, Cl mean 0.2268.
* Sod, base 50 + 2 levels, remesh every 4 steps (error of the density against the exact solution, mean leaves; before = section 13): the choice of variables and threshold now matters little.

  | criterion | density | density + pressure | density + velocity |
  |---|---|---|---|
  | 0.02 | 3.10e-3, 1 071 (before 3.25e-3, 928) | 3.09e-3, 1 091 (3.10e-3, 966) | 3.07e-3, 1 128 (3.10e-3, 999) |
  | 0.05 | 3.16e-3, 901 (3.98e-3, 743) | 3.12e-3, 1 030 (3.59e-3, 804) | 3.05e-3, 1 086 (3.14e-3, 909) |
  | 0.10 | 3.22e-3, 696 (4.33e-3, 570) | 3.12e-3, 783 (4.56e-3, 643) | 3.10e-3, 983 (3.43e-3, 797) |

  Uniform 200 (3 200 cells): 2.91e-3. The contact that the default 0.1 missed before is now kept refined.

## 16. Check of the second-order coarse side (c) on the other tests (default flipped to on, suite run, flipped back)
* Whole suite with `coarse_side_second_order = true`: all tests pass (102). Sod with waves crossing the jumps improves (global L1 9.91e-3 -> 9.36e-3 for the narrow band (a), window error
  1.78e-2 -> 1.36e-2), the maximum density stays below 1.01 and the minimum above 0.115 (checked in the test), dynamic Sod is unchanged (3.22e-3 -> 3.23e-3), conservation with and without
  viscosity and the free stream are unaffected (shared fluxes, zero slopes).
* The acceptance bound fixed before the first P4 run (adaptive manufactured-solution error below 0.75 x the coarse one, section 12), failed by injection (1.12 / 1.27 x), is MET with
  (b) + (c): steady viscous 1.049e-3 against 1.465e-3 (0.72 x), unsteady Euler 1.075e-3 against 1.534e-3 (0.70 x).
* `fenice_tests` takes 10.3 s (8.9 s before the closure study of four closures).

## 17. Decisions of the user after sections 14-16
* The jump closure (b) + (c) is the default (`interpolate_jumps` and `coarse_side_second_order` both on; the injection stays selectable for the tests). The manufactured-solution test now asserts the
  original bound (adaptive error below 0.75 x the coarse one) and a split guard (away from the jumps 0.8 x, near them 1.2 x of the coarse error).
* The indicator defaults are confirmed: criterion 0.1, density + velocity (density alone in the struct default; the demo and the studies use density + velocity for the airfoil), 2 diffusion
  iterations of 0.1 above 0.05, split 0.15, join 0.10, joins on the parent's xi.
* Default configuration on the airfoil (criterion 0.1, density + velocity, base 192 x 96 + 2 levels, t = 6): Cd mean 0.1482, Cl mean 0.2268 (reference 0.1498 / 0.2171), 23.5 k leaves, 14 s.

## 18. P6 as built (`CellParallel`, `cpp/include/fv/CellParallel.hpp`)
* **Replicated topology, distributed state.** Every rank holds the whole `CellMesh` (leaves, faces, stencils); the state of a leaf is valid on its owner and on the ranks that read it. Ownership: each
  level is cut into contiguous blocks of equal leaf count along its Morton order (per-level balance: every level is advanced separately by the recursion). The partition is deterministic
  from the mesh, so no communication is needed to agree on it.
* **Work.** A rank advances its owned leaves. A same-level face is computed by the owners of both its leaves (redundantly, same inputs, same bits); a coarse-fine face and a boundary face by the
  owner of the finer leaf. The coarse owner receives the registers (convective and viscous, time averaged) of its coarse-fine faces from the fine owners once per coarse step, before the
  coarse level advances (`exchange_registers`), so it never recomputes fine-level faces. Every rank runs the same recursion and calls every exchange at the same point, also a rank that owns nothing
  at a level (its `advance_fixed_dt` runs on an empty array and makes the declared number of residual calls).
* **Halo = what the engine reads.** `CellFluxEngine::face_dependencies` marks the leaves read by a face (stencil points, virtual cells and their leaves, the interpolated cells and the neighbours of
  their coarse leaf, the cell gradients of both cells, the non-reflecting tangential neighbours, the coarse side of a coarse-fine face); `CellParallel::partition` applies it to the faces of each rank,
  per level (`stage` lists: the owned leaves of a level other ranks read in that level's faces, pushed every stage) and overall (`final` lists, pushed when a level has advanced). The mass-flow
  inlet's side leaves are held on every rank at every stage (the average is summed in position order on each rank, never as partial sums), the solid reset's neighbours are in the final lists.
  The forced (immersed-body) copy is refreshed for the received leaves.
* **Remesh.** The state of every leaf is gathered on every rank (`gather_state`), the indicator, the flags, `adapt` and the engine rebuild run replicated (identical decisions, nothing to
  agree on), then `partition` again. This is O(N) per rank per remesh and does not shrink with the number of ranks: it caps the speed-up (see below); its optimisation (distributed indicator,
  incremental rebuild) is a P8 item.
* **Tests (`fenice_tests`, no MPI needed).** Ranks are emulated by threads with an in-process stand-in for the collectives (`FakeHub`; a message whose size differs from what the receiver expects
  throws, so asymmetric lists are caught). `cellparallel_equals_serial_bitwise_for_any_rank_count`: six cases (Euler periodic; viscous + body force + sources periodic with remesh; unsteady source with a
  no-slip / moving-wall mix; walls + outlet + non-reflecting + viscous with remesh; mass-flow inlet + velocity inlet; immersed body with viscosity, solid reset and remesh) on 1, 2, 3, 5 and 7
  ranks, 6 steps on random 3-level meshes, with the state of every leaf a rank does not hold set to NaN and re-poisoned after every remesh: the final state equals the serial one bit for bit, every rank
  agrees, a rank on 5-7 holds fewer leaves than the mesh and has a halo. Removing the register exchange, or the gradient neighbours from the dependencies, makes the test fail. Levels with fewer leaves
  than ranks, ranks without leaves at a level and periodic faces between ranks occur in these runs. `cell_engine_result_does_not_depend_on_the_thread_count`: an adapted mesh above `kOmpMinWork` with
  all the physics, 3 steps with 1 thread and with 8 threads, bit-identical (the promise of section 10).
* **Real MPI** (`python cases/validation/amr/cell_amr_mpi.py`, the demo under `mpiexec` with `--poison`; not part of the CI since the cell engine is not behind the CLI yet): dynamic Sod, the airfoil
  (immersed body, viscosity, non-reflecting outlet), and two periodic manufactured solutions (viscous, and unsteady Euler with a time-dependent source) with remesh: serial and 1..7 ranks give a
  byte-identical final state (`final.bin`) in all four cases.
* **Balance.** With equal leaf counts per level on every rank (to within one leaf) the weighted work (sum of 2^l N_l) is equal across ranks by construction (max / mean 1.000 on 2-16 ranks); this says how the work is
  cut, it is not a timing. Halo: on 2 ranks (268 362 leaves at the end) 134 181 owned each, 2 109 / 1 941 halo leaves (1.6 % / 1.4 %). **Storage is not distributed yet:** every rank stores the whole mesh, the
  whole state array and all the engine arrays; "valid" below counts the leaves whose state is up to date on a rank, not its memory (the patch engine's P0 column "max cells/rank" was storage). Distributed
  storage is a later item.
* **Scaling** (the study of `amr_mpi.py` in P0 with the cell engine: airfoil, 720 x 360 base, 2 levels, 50 steps, remesh every 5; 8-core machine with 16 logical processors, one thread per MPI rank; wall time of
  the time loop including the remesh; the final state is bitwise the serial one in every row):

  | run | wall [s] | speed-up | of which remesh [s] | leaves with a valid state on the busiest rank |
  |---|---|---|---|---|
  | serial (1 thread) | 10.94 | 1.00 | 2.83 | |
  | OpenMP 2 / 4 / 8 / 16 threads | 7.03 / 5.03 / 4.41 / 4.38 | 1.56 / 2.17 / 2.48 / 2.50 | 2.53 / 2.42 / 2.36 / 2.55 | |
  | MPI 2 ranks | 8.04 | 1.36 | 3.62 | 136 290 |
  | MPI 4 ranks | 6.56 | 1.67 | 3.81 | 69 721 |
  | MPI 8 ranks | 7.23 | 1.51 | 4.81 | 35 878 |
  | MPI 16 ranks | 11.22 | 0.98 | 7.39 | 18 892 |

  Patch engine on `main` (P0, section 5): serial 16.4 s; OpenMP 2 / 4 / 8 / 16: 8.8 / 5.1 / 3.2 / 2.9 s; MPI 2 / 4 / 8 / 16: 8.9 / 5.2 / 4.0 / 4.6 s. **Plainly: the cell engine is 1.5x faster in serial, about as fast
  on 2 ranks (8.0 against 8.9 s) and slower than the patch engine from 4 ranks on (6.6 against 5.2 s on 4, 7.2 against 4.0 s on 8, 11.2 against 4.6 s on 16), and from 4 threads on with OpenMP (5.0 / 4.4 / 4.4 against 5.1 / 3.2 /
  2.9 s).** Against the acceptance of P8 ("wall time per accuracy not worse on any study") the parallel runs currently fail; the remesh is a precondition for P8, not an optimisation.
* **Where the time goes** (measured, rank 0, seconds over the 10 remeshes of the run; the remesh column above is their sum plus the part of the time stepping lost waiting for it):

  | | gather | indicator | adapt | engine build | speed bound | partition |
  |---|---|---|---|---|---|---|
  | serial | | 0.87 | 0.80 | 0.97 | 0.69 | |
  | OpenMP 8 | | 0.92 | 0.82 | 0.97 | 0.15 | |
  | MPI 2 | 0.05 | 0.90 | 0.91 | 1.01 | 0.74 | 0.54 |
  | MPI 4 | 0.05 | 0.95 | 1.14 | 1.13 | 0.77 | 0.38 |
  | MPI 8 | 0.11 | 1.26 | 1.59 | 1.39 | 0.87 | 0.35 |
  | MPI 16 | 0.25 | 1.91 | 2.54 | 2.13 | 1.38 | 0.66 |

  The explanation I first wrote down (the gather and the O(ranks x levels x leaves) masks of `partition` grow with the ranks) was wrong: the gather is negligible (0.05-0.25 s), `partition` costs 0.35-0.66 s and does
  not grow steadily. The remesh is dominated by serial work that every rank REPEATS (indicator 0.9 s + adapt 0.8 s + build 1.0 s + speed bound 0.7-0.9 s = 3.3-3.4 s), which does not shrink with ranks or threads
  (only the speed bound, a parallel residual, shrinks with OpenMP: 0.69 -> 0.15 s), and it grows at 8-16 ranks where 8-16 processes run the same serial code on 8 cores (memory-bandwidth contention, then
  oversubscription). Time stepping alone (wall minus remesh): 8.1 s serial, 4.4 / 2.8 / 2.4 / 3.8 s on 2 / 4 / 8 / 16 ranks, i.e. speed-ups 1.8 / 2.9 / 3.4 / 2.1 (the last on an oversubscribed machine).
  P8 order of work: (1) do not replicate the remesh: the indicator over a distributed state, an incremental `adapt` and engine rebuild (only where leaves changed; churn is small after the parent-level join),
  (2) shrink `partition`, (3) distributed storage.
* `fenice_tests` takes 11.1 s now (8.9 s before P6: the multi-rank emulation).

## 19. P7 as built (the cell engine behind the CLI; the patch engine is gone)
* `use_amr` runs `Simulation::solve_amr` (setup builds `CellMesh`, `CellFluxEngine`, the physics, `CellIndicator`, `CellAdaptor`, the MPI hooks; `initialize_field` does the uniform
  pre-refinements, the initial adaptation, the solid reset and the partition). The patch engine (`AMR.cpp/hpp`, `restrict_patch`, the patch fields of `CrossGhosts` and `ViscousFaceFluxes`,
  `patch_residual`, `amr_patch_nx/ny`, `amr_buffer_layers`) and the development driver `cell_amr_demo` are removed. Removed keys give an error that names the replacement.
* Keys: `amr_max_level`, `amr_indicator` (variables joined by `+`), `amr_criterion`, `amr_refine_threshold` (0.15), `amr_derefine_threshold` (0.1, must not exceed the refine threshold),
  `amr_diffusion`, `amr_refine_every`, `amr_init_levels`, `amr_refine_ibm`. Periodic and mass-flow boundaries and source terms (not with the IBM) are now allowed with AMR; BDF2 is not.
* Output: `.vtu` (single rank) or per-rank `.vtu` + `.pvtu` (`export_cells`), `AMR_Level` field; `fenice.read_vtu`. Checkpoint: leaf list, `U`, `t`, step, lagged wave-speed bound, `.pvd` collection;
  resume is bitwise, also with another rank count; hot start interpolates at the leaf centres. `dump_state` / `final.npz`: per-leaf `U` and `leaf_meta`.
* **References.** The six AMR regression references were regenerated (their own commit) after the validation: each case against the uniform run at its finest level (relative L1 density difference
  of the AMR run / of the base mesh: sod_1st 3.3e-3 / 1.8e-2, sod_muscl_init 1.1e-3 / 1.3e-2, square_rk4 1.2e-2 / 5.0e-2, sod_few_roots 3.1e-3 / 3.8e-2, naca0012 1.9e-4 / 5.6e-4), the Sod and
  airfoil studies through the CLI, MPI == serial bitwise on 1-7 ranks and bitwise restart. The other 32 references are unchanged and match.
* **Studies through the CLI.** Sod, 10x2 base, 4 levels, remeshed every 2 steps: 2 312 leaves, L1 1.51e-2 (uniform 160x32: 5 120 cells, 1.43e-2; uniform 80x16: 2.20e-2). With the default remesh
  interval (10) the contact is under-resolved (848 leaves, 2.6e-2, and the criterion hardly matters), so the study sets 2. Airfoil, 192x96 + 2 levels, t = 6: 23 526 leaves, 14.2 s, rms error of
  Cd / Cl against the 768x384 run 0.011 / 0.054 (uniform 192x96: 0.034 / 0.36; 768x384: 294 912 cells, 244 s; criterion 0.03 and 0.05 change the errors by less than 1 %).
  The patch engine had 52 224 cells, 22 s, 0.007 / 0.039 (cached figures of `main`): the cell engine uses 2.2x fewer cells and 1.5x less time, with a larger error in the loads. This is reported, not
  resolved; mean Cl is 5-6 % high for both engines (section 13).
* **Real-MPI check**: `amr_mpi.py --check`: dynamic Sod, airfoil, viscous periodic MMS and unsteady MMS, serial and 1-7 ranks, bit for bit.
* **Scaling with the final code** (`amr_mpi.py`, 720x360, 2 levels, 50 steps, noisy machine): serial 11.5 s (remesh 2.9); OpenMP 2/4/8/16: 7.3/5.3/4.7/6.0 s; MPI 2/4/8/16: 9.1/8.7/14.2/29.9 s (remesh
  4.1/5.0/7.3/10.7). The patch engine's MPI was faster from 4 ranks on. This is the P8 problem and the next piece of work: a distributed or incremental remesh and distributed storage.

## 20. Remesh rework and P8 measurements (2026-10-23)
Rule of the work: nothing changes the numerics. Oracle = the six AMR references and the 32 others, `amr_mpi.py --check` (1-7 real ranks), and golden dumps of four adaptive cases (dynamic Sod, airfoil with the
body and the outlet, periodic viscous MMS, unsteady MMS) and of the 720x360 airfoil (268 362 leaves, 8 threads), all bitwise unchanged after every step.
* **Where the remesh time was** (airfoil 720x360, 2 levels, 50 steps, 10 remeshes, 1 thread, seconds): `CellMesh::rebuild` hash index 0.46 + faces 0.22 + sides 0.07; `adapt` ripple/merge 0.01,
  new-leaf list 0.04, global sort 0.10; `CellFluxEngine::build` stencils/codes 0.27, solid masks 0.08, `rebuild_lists` 0.74 (a comparison sort of ~4 x faces ints per level); indicator 0.87 (binary
  search of Morton keys per neighbour); wave-speed bound 0.71 (a full residual).
* **Changes.** (1) Leaf lookup = a dense table per level when nx_l ny_l <= 4N + 65536, else a binary search of the level's Morton keys (the hash map is gone). (2) `rebuild_lists`: sorted-unique by a
  mark array instead of std::sort. (3) Indicator: per-cell gather (each cell tests its own four sides, flag is symmetric; diffusion gathered per cell: every addend is the same number, so the sum does not
  depend on the order; `diffuse_cells`, tested bit for bit against the interface-list rule), per-cell primitives computed once, OpenMP. (4) Faces: count pass + prefix sum + fill pass (parallel, same
  order). (5) `adapt`: the new list is a merge of three Morton-sorted streams per level (kept / children / parents) holding source indices, the state is written once in parallel. (6) `build`: virtual cells are
  numbered in creation order, so the leaves with a Finer/Coarser side (and the faces touching them) are visited first, serially, in the original order, and everything else runs in parallel on the finished
  tables. (7) The wave-speed bound after a remesh is `CellFluxEngine::wave_speed` (the level by level flux pass of the time loop, same value as `residual()`, tested) and runs AFTER `partition()`, so under
  MPI each rank computes only its own leaves and the maximum is reduced (before, every rank ran a full-mesh residual).
* **Result** (same study, 10 remeshes; min of 3 on one thread): indicator 0.87 -> 0.28, adapt 0.84 -> 0.39, engine build 0.99 -> 0.47, wave-speed bound 0.71 -> 0.64 (1 thread; 0.15 on 8 threads and ~0.25 on 4 ranks),
  remesh total 2.9 -> 1.6 s serial, 2.45 -> 0.86 s on 8 threads. `amr_mpi.py` (wall of the 50 steps): serial 11.5 -> 9.8 s; OpenMP 2/4/8/16: 7.3/5.3/4.7/6.0 -> 5.5/3.5/2.9/2.7 s; MPI 2/4/8/16:
  9.1/8.7/14.2/29.9 -> 6.1/4.5/4.9/8.5 s (speed-ups 1.6 / 2.2 / 2.0 / 1.15). The remesh share is 16 % of the loop on one thread, 26 % on 8 threads, 26-58 % on 2-16 ranks (replicated work on 8-16 processes of an
  8-core machine contends for memory).
* **Time stepping is at the kernel limit.** Cost per leaf update 568 ns, the uniform solver's cost per cell 571 ns (same case, 720x360, 1 thread): the face-list engine has no overhead against the uniform
  kernel. 74 % of the step is the face flux (reconstruction + HLLC), 19 % the cell quantities and the viscous flux. A faster step needs a faster kernel, shared with the uniform solver (SIMD over faces keeps the bits only if every
  lane runs the same operations; not attempted).
* **Churn.** Per remesh about 0.13 % of the leaves change (50 steps: 3 256 refined + 274 joins over 10 remeshes of 268 k leaves; the full t = 6 airfoil: 2 651 + 978 over ~200 remeshes of 23.5 k leaves), yet the
  full topology and stencils are rebuilt each time. An incremental update (dirty leaves = the new ones and their face neighbours, clean faces copied with the leaf index remapped, virtual cells renumbered by
  key) would take adapt + build from ~0.85 s to an estimated 0.2 s per 10 remeshes; not done (see "Open" below).
* **Subtree ownership instead of per-level blocks** (what a local remesh with whole base-cell subtrees per rank would imply), measured offline on the final state of the airfoil run (268 362 leaves; levels
  258 416 / 866 / 9 080), base cells cut along the Morton order into N chunks of equal weighted work (sum of 2^l): balanced weighted work (1.00) but the per-level counts are not: max/mean leaves of level 0 / 1 / 2 =
  1.15 / 2.0 / 2.0 on 2 ranks, 1.15 / 2.3 / 2.1 on 4, 1.15 / 4.5 / 4.1 on 8, 1.15 / 5.8 / 7.3 on 16. Levels advance one after the other with exchanges, so the time is the sum over levels of the busiest rank:
  about 1.5x the ideal on 8 ranks for this case (the body and the wake are in a few base cells). The present per-level partition has 1.00 at every level but needs the whole topology on every rank.
* **Accuracy per cost against the patch engine** (built from `main` f257ae1 in a worktree, same machine, same case "IBM Airfoil AMR", t = 6, rms error of Cd(t), Cl(t) against uniform 768x384 for t >= 1):

  | run | cells at the end | wall [s] | Cd | Cl |
  |---|---|---|---|---|
  | uniform 192x96 | 18 432 | 2.5 (1.7) | 0.034 | 0.36 |
  | uniform 384x192 | 73 728 | 20.7 (13.3) | 0.0145 | 0.087 |
  | patch engine, 16x16 patches | 52 224 | 24.4 | 0.0074 | 0.0387 |
  | patch engine, 8x8 patches | | 20.3 | 0.0085 | 0.0457 |
  | cell engine, default (criterion 0.1) | 23 526 | 10.7 (7.8) | 0.0106 | 0.0541 |
  | cell engine, criterion 0.01, diffusion 2 | 36 885 | 15.8 (11.0) | 0.0098 | 0.0502 |
  | cell engine, criterion 0.01, diffusion 4 | 41 706 | 16.4 (12.1) | 0.0095 | 0.0486 |
  | cell engine, criterion 0.005, diffusion 4 | 50 262 | 24.7 (18.1) | 0.0078 | 0.0411 |
  | cell engine, criterion 0.002, diffusion 4 | 80 742 | 72.7 (53.6) | 0.0056 | 0.0301 |

  In parentheses: re-run on branch `amr-optimisation` (vector Riemann solver, incremental remesh update; same machine, MSVC, same errors
  to every digit; the patch engine is gone and was not re-measured). Then the cell engine reaches 0.0078 / 0.0411 in 18.1 s against the
  patch engine's 0.0074 / 0.0387 in 24.4 s (measured before the vector Riemann solver, which it did not have).
  **As first measured, at matched error the two engines cost the same on this study** (cell 0.0078 / 0.0411 in 24.7 s against patch 0.0074 / 0.0387 in 24.4 s); the cell engine is cheaper only at the lower accuracies (0.0106 / 0.054 in
  10.7 s; the patch engine with 8x8 patches, 0.0085 / 0.0457 in 20.3 s, lies between the cell engine's 16.4 s and 24.7 s points). Settings that do not move the error: the body margin (2, 6, 12 cells: identical to 3 digits), the remesh interval (every 2), the thresholds.
  75 % of the velocity error energy of the default run sits in the finest-level region (2 % of the area): the error is made where the resolution is the same as the reference's, so it is not a matter of covering
  more area with the finest level at fixed cost. Cost scales with sum n_l 2^l (50 k leaves with 29 k at level 2 cost 3.7x the default's 38.5 k updates, 2.4x its time). The acceptance rule "wall time per accuracy
  not worse on any study" is therefore met only against the uniform meshes, and met with equality (not better) against the patch engine on the airfoil.
* **Open.** (a) incremental topology update (estimated -0.6 s of a 1.6 s serial remesh; the wave-speed bound 0.64 s and the indicator 0.28 s remain; done on branch `amr-optimisation`, see `docs/design/amr-optimisation.md`); (b) distributed storage and a local remesh with ghost layers
  (needed for a mesh that does not fit one node; the per-level partition must then be kept for the time stepping, so the topology of a rank would be its owned leaves plus the halo of each level and the remesh
  would exchange flags between neighbouring ranks); (c) a SIMD flux kernel (shared with the uniform solver; done on `amr-optimisation`).
