# Cell-based AMR for Fenice: design and plan (branch `amr-cell-based`)

Status: P1, P2, P3 done (2026-10-22: `CellMesh` topology, `CellFluxEngine` convective residual, `CellAdvance` subcycling with flux replacement); the user accepted the P2 gate (section 9); P4 next (needs the user's go). Written 2026-10-22 after a review of AMReX, SAMRAI and the ECOGEN source.
`main` keeps the patch-based AMR (last good state: commit `f257ae1`); this branch replaces it. To go back: `git checkout main`.

## 1. Why

Goal (user): minimise total wall time on adaptive runs while keeping the MPI load balance acceptable.

What is wrong with the patch design (AMR.cpp, `Simulation::patch_residual`):

1. Cells inside a refined patch are refined whether they need it or not (patch = 8x8 .. 16x16 cells).
2. Every patch kernel looks up its ghost cells by geometry (`leaf_at`, `GridInterp2D`, `ghost_cell`) and has its own
   residual: two implementations of the physics (uniform `assemble`, AMR `patch_residual`) kept equal by tests.
3. Physical edges go through a separate path in which the boundary face is first order (MUSCL stencil not upgraded).
4. Conservation across coarse-fine seams needs flux registers and is exact only for Explicit Euler (O(dt^2) otherwise).

What ECOGEN does (read from `code-mphi/ECOGEN`, devel): a tree of cells plus an interface (face) list; each interface
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
* Therefore at a level jump the coarse side of a fine face is first order (`c3 - c2 = 0`), as in ECOGEN; the fine side keeps
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
  level, with the coarse neighbour frozen at its start-of-step state (no time interpolation, like ECOGEN).
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
  flagged cells. 2:1 is restored by rippling refinement to coarser neighbours (p4est style), not by forbidding refinement.
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
* Known time seam (as in ECOGEN, accepted): the fine level reads its coarse neighbours at the start of the coarse step, the coarse level reads
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
