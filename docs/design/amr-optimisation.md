# AMR optimisation: open problem and options

Branch `amr-optimisation`, started from `main` at the merge of `amr-cell-based` (commit 701048c).
Measurements and the reasoning are in `docs/design/amr-cell-based.md`, section 20.

## The problem

The adaptive engine is correct (bitwise equal to serial for any rank count, conservative, restartable) and
its time stepping costs the same per leaf as the uniform solver (568 ns against 571 ns). What limits it is the
**remesh**, the part of the run that rebuilds the mesh after the indicator has decided what to refine or join:

* It is **replicated**: every MPI rank gathers the whole state and rebuilds the whole topology (leaves, face
  list, stencils, virtual cells). It is OpenMP-parallel inside a rank but does not shrink with the number of ranks.
  On the airfoil (720x360 base, 2 levels, 50 steps, 268 k leaves) it is 16 % of the loop on one thread, 26 % on
  8 threads and 26 to 58 % on 2 to 16 ranks. MPI reaches 2.2x on 4 ranks and 1.15x on 16, where OpenMP gives 3.4x on 8 threads.
* **Storage is not distributed**: every rank holds the whole mesh and state, so MPI cannot be used to run a mesh
  that does not fit one node.
* It is **wasteful**: only about 0.13 % of the leaves change per remesh, yet topology and stencils are rebuilt
  for all of them.
* Accuracy per cost: on the airfoil the cell engine and the earlier patch-based engine cost the same at matched
  error (24.7 s against 24.4 s for Cd/Cl errors of 0.0078/0.0411 against 0.0074/0.0387). The cell engine is cheaper
  only at lower accuracy, so a gain has to come from the cost of the machinery, not from the indicator settings.

## Options

1. **Incremental topology update.** Mark the leaves that changed and their face neighbours as dirty, rebuild
   faces, stencils and virtual cells only there, and copy the rest with the leaf indices remapped (virtual cells
   renumbered by key). Estimated: adapt + engine build from about 0.85 s to 0.2 s per 10 remeshes, that is a
   serial remesh of 1.6 s down to about 1 s, with the wave-speed bound (0.64 s) and the indicator (0.28 s)
   remaining. Contained change, helps serial, OpenMP and MPI alike, keeps the replicated design. Does not help memory.
2. **Distributed storage with a local remesh and ghost layers.** Each rank keeps only its owned leaves plus the
   halo of each level; the indicator is exchanged between neighbouring ranks per smoothing iteration, split flags
   are exchanged, topology is rebuilt locally. The only option that lets a mesh exceed one node and makes
   the remesh scale with the ranks. Large rewrite of the indexing in `CellMesh` / `CellFluxEngine` / `CellParallel`.
   The per-level partition must be kept for time stepping: ownership of whole base-cell subtrees was measured to
   leave the fine levels on a few ranks (max/mean leaves of the finest level 4.1 on 8 ranks, about 1.5x slower
   stepping on the airfoil).
3. **SIMD flux kernel.** Time stepping is 74 % face flux (reconstruction + HLLC). Vectorising it over faces
   keeps the bits only if every lane performs exactly the same operations (no FMA contraction, same order), and it
   is shared with the uniform solver, whose references must also stay bitwise. Speeds up every run, adaptive or not,
   but does nothing for the remesh or for MPI scaling.

## Acceptance for any of them

Six AMR references and the 32 others unchanged, `python cases/validation/amr/amr_mpi.py --check` bitwise on 1 to 7
ranks, `python cases/validation/amr/amr_mpi.py` timings reported as the minimum of three runs, and the accuracy-per-cost
table of section 20 re-run.

## Decision and results (option 3 first)

Chosen first: the SIMD flux kernel. It is the largest share of the step (the Riemann solver was 56 % of the stepping of the
airfoil with the Low-Mach solver) and it helps uniform and adaptive runs alike. Gate before writing it: the airfoil 720x360, 50 steps,
1 thread, with `riemann_solver = HLLC` against Low-Mach differed by 2.15 s, the 59 M face fluxes times the 34 ns of the bench
difference predicted 2.0 s, so the solver cost in the real loop is compute-bound and not hidden by memory traffic.

Done (all bitwise identical: 38 references, `amr_mpi.py --check` on 1 to 7 ranks, golden dumps of five adaptive runs, the six
`bench_riemann` hashes scalar = batch, unit test `riemann_batch_equals_scalar_bit_for_bit` with NaN, floors, signed zeros and
wave speeds one ulp from zero; no `vfmadd` in the object file):

* `RiemannSolver::face_flux_batch`, AVX2 four faces at a time, three HLLC variants, ideal and stiffened gas: 46 / 80 / 60 ns per face
  became 19 / 25 / 22 ns. Used by the uniform solver (`compute_flux`) and by the AMR engine.
* `RiemannSolver::muscl_face_flux_batch`: the MUSCL reconstruction (four limiters) of the interior four-point faces fused with it.
  Boundary and coarse-fine faces still go through the scalar reconstruction and the batch Riemann solver.

| run (min of 3) | before | after |
|---|---:|---:|
| airfoil AMR 720x360, 50 steps, 1 thread, time loop | 10.0 s | 6.1 s |
| same, 8 threads | 3.2 s | 2.4 s |
| same, 4 MPI ranks | 4.5 s | 3.5 s |
| uniform airfoil 720x360, 50 steps, 1 thread | 7.4 s | 4.6 s |
| airfoil study, AMR 192x96 + 2 levels, t = 6 | 10.7 s | 8.0 s |

Cost of the choice: the stepping is now faster and the replicated remesh did not change, so it is a larger share of the loop (21 % on
one thread, 32 % on 8 threads, 34 to 65 % on 2 to 16 ranks) and the MPI speed-up is relatively worse (1.7x on 4 ranks against 2.6x for
8 threads). Memory is unaffected.

Left in the step (airfoil, 1 thread, seconds of 4.4): face fluxes 2.2 (of which the vector kernel about 1.3), viscous flux and cell
quantities 1.5, cell gather 0.7, state conversion 0.3. The viscous kernels are the next candidate for the same treatment; option 1
(incremental topology update) is now the most valuable item for the OpenMP and MPI scaling.

## Viscous kernels and option 1 (incremental topology update): results

Measured on a 4-core Linux container (Intel Xeon with AVX2, gcc 13, no MPI available there), airfoil AMR 720x360, 50 steps,
10 remeshes, minimum of three runs, the unmodified binary of this branch against the final one. The 8-core numbers above were not
re-measured. Bitwise after every step: the 38 references (serial; also with `FENICE_NO_SIMD=1`, `FENICE_FULL_REBUILD=1` and an
`FV_NO_SIMD` build), golden dumps of the four `amr_mpi.py --check` cases, of three longer viscous AMR runs (MMS 40 steps, cavity 200
steps, Sod remeshed every 2 steps) and of the 720x360 airfoil on 1 and 4 threads, and the unit tests (FakeHub 1-7 ranks). The real-MPI
oracles (`regression.py --mpi 4`, `amr_mpi.py --check`) could not run there.

* **Viscous kernels (AVX2): built, measured, removed.** `cell_quantities` and `viscous_speed` ran four consecutive leaves at a time
  (bit-identical twins, scalar loads, no gathers). In the container (gcc, Intel): cell quantities 0.93 -> 0.57 s, viscous speed bound
  0.57 -> 0.25 s. On the 8-core workstation (MSVC, Ryzen 7 7800X3D, 96 MB L3), bisected commit by commit, min of 3: stepping
  4.78 -> 5.20 s with them, the speed bound in the remesh 0.34 -> 0.28 s. Net loss on the main target, so they were reverted
  (the scalar code is the original). The face flux twin was slower everywhere (1.08-1.25 s against 0.88 s in the container).
  Lesson: these kernels read scattered data with little arithmetic, and the gain depends on compiler and caches; reducing the work
  (option 1) or vectorising arithmetic-heavy kernels (the Riemann solver) wins on every machine measured.
* **Option 1.** Step 1 numbers the virtual cells by key (block, resp. fine leaf and side) instead of creation order, which also lets
  the full build fill every table in parallel (engine build 1.34 -> 1.19 s). Step 2: `CellMesh::adapt` records the old index of every
  leaf and a dirty flag (new, or a square of its size across a side holds a new leaf, marked outward from the new leaves); the faces
  made by clean leaves are copied with the indices remapped; `CellFluxEngine::build` copies the codes of clean leaves, the stencils of
  faces between clean leaves, the solid mask and steady sources of kept leaves, and renumbers virtual cells by key. The work arrays
  are sized, not zero-filled (each entry is written before it is read in a step). Test `cellflux_incremental_rebuild_equals_full_rebuild`:
  after each of 14 random adapts in four setups, faces, side lists and every engine table equal a full rebuild.
  Engine build 1.41 -> 0.38 s, adaptation 0.59 -> 0.54 s (one run each).

Workstation (Ryzen 7 7800X3D, MSVC), serial, min of 3, per commit: remesh 1.23 s (base) -> 1.14 (virtual cells by key) -> 1.03
(incremental update) -> 0.93 s (dirty marking from the new leaves, no zero-fill); engine build 0.47 -> 0.20 s, adaptation 0.39 -> 0.35 s.
The time loop table of the AMR page is to be re-measured with the final code. Container numbers (4-core, gcc, min of 3) of the remesh
alone: 2.5 -> 1.5 s on one thread.

Left in the remesh (container, 1 thread, s of 1.5): wave-speed bound 0.6, indicator 0.4, adaptation 0.5 (new leaf list, face copy, side lists),
engine build 0.4 (code copy, per-level lists). The estimate of option 1 above (adapt + build 0.85 -> 0.2 s on the 8-core machine)
corresponds here to 2.0 -> 0.9 s. MPI timings and the 8-core table remain to be re-measured on the workstation.
