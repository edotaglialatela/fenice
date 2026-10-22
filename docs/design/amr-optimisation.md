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
