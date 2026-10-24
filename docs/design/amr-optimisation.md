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

Bitwise after every step: the 38 references (serial; also with `FENICE_NO_SIMD=1`, `FENICE_FULL_REBUILD=1` and an `FV_NO_SIMD` build;
serial and 4 MPI ranks in CI on Linux x64 / ARM64, macOS and Windows MSVC), golden dumps of the four `amr_mpi.py --check` cases, of
three longer viscous AMR runs (MMS 40 steps, cavity 200 steps, Sod remeshed every 2 steps) and of the 720x360 airfoil on 1 and 4
threads, the unit tests (FakeHub 1-7 ranks), and on the Threadripper the airfoil on 1-48 threads and 2-48 real MPI ranks.
Timings: airfoil AMR 720x360, 50 steps, 10 remeshes, minimum of three runs; stepping = time loop - remesh.

* **Viscous kernels (AVX2).** `viscous_cell_quantities_run` and `viscous_speed_range` run four consecutive leaves at a time with the
  same IEEE operations as the scalar twins (runs of leaves whose four neighbours are leaves; neighbours and level sizes by scalar
  loads, no gathers). Bisected per commit, one thread, min of 3:

  | machine, compiler | stepping before | with the kernels |
  |---|---:|---:|
  | Ryzen 7 7800X3D, MSVC, file without `/arch:AVX2` | 4.78 s | 5.20 s |
  | Ryzen 7 7800X3D, MSVC, file with `/arch:AVX2` | 4.81 s | 4.27 s |
  | Ryzen 7 7800X3D, gcc (WSL) | 4.73 s | 4.20 s |
  | Threadripper PRO 5965WX, gcc (WSL) | 4.77 s | 4.29 s |
  | 4-core Intel container, gcc | 8.8 s | 8.0 s |

  Without `/arch:AVX2` MSVC mixes SSE and AVX code around the intrinsics and the kernels lose; with it they gain as with gcc
  (`dumpbin /disasm` shows no `vfm`, CI bitwise on Windows). The flag applies to `SchemesAvx2.cpp` only; the CPU check moved to
  `Schemes.cpp` so that it is never compiled for AVX2. The face flux stays scalar everywhere: its vector twin (bit for bit equal) was
  slower in every variant tried (1.08-1.25 s against 0.88 s in the container); it is load-bound with three divisions per face.
  Unit test `viscous_cell_kernels_equal_scalar_bit_for_bit` (it says whether the vector path was built and ran).
* **Option 1.** Step 1 numbers the virtual cells by key (block, resp. fine leaf and side) instead of creation order, which also lets
  the full build fill every table in parallel. Step 2: `CellMesh::adapt` records the old index of every leaf and a dirty flag (new, or a
  square of its size across a side holds a new leaf, marked outward from the new leaves); the faces made by clean leaves are copied
  with the indices remapped; `CellFluxEngine::build` copies the codes of clean leaves, the stencils of faces between clean leaves, the
  solid mask and steady sources of kept leaves, and renumbers virtual cells by key. The work arrays are sized, not zero-filled (each
  entry is written before it is read in a step). Test `cellflux_incremental_rebuild_equals_full_rebuild`: after each of 14 random
  adapts in four setups, faces, side lists and every engine table equal a full rebuild. Serial remesh per commit on the 7800X3D with
  MSVC: 1.23 s (base) -> 1.14 (virtual cells by key) -> 1.03 (incremental update) -> 0.93 s (dirty marking, no zero-fill); engine build
  0.47 -> 0.20 s, adaptation 0.39 -> 0.35 s. With gcc: 1.22 -> 0.69 s (7800X3D), 1.24 -> 0.74 s (Threadripper).

| serial, time loop / remesh | before | after |
|---|---:|---:|
| Ryzen 7 7800X3D, MSVC | 6.01 / 1.23 s | 5.20 / 0.93 s |
| Ryzen 7 7800X3D, gcc | 5.95 / 1.22 s | 4.88 / 0.69 s |
| Threadripper PRO 5965WX, gcc | 6.01 / 1.24 s | 5.03 / 0.74 s |

Threadripper (gcc, kernels off): OpenMP 1 / 2 / 4 / 8 / 12 / 16 / 24 / 32 / 48 threads 5.50 / 3.17 / 2.13 / 1.65 / 1.47 / 1.45 / 1.63 /
1.74 / 1.80 s (remesh flat at 0.40 s from 8 threads); MPI 2 / 4 / 8 / 12 / 16 / 24 / 32 / 48 ranks 3.66 / 3.01 / 2.96 / 2.99 / 3.08 /
4.60 / 5.91 / 8.86 s, the replicated remesh 0.98 / 1.18 / 1.56 / 1.79 / 1.99 / 3.14 / 4.15 / 6.14 s. Left in the serial remesh (gcc,
7800X3D, s of 0.69): wave-speed bound 0.28, indicator 0.20, adaptation 0.19, engine build 0.18. Beyond a few ranks the replicated
remesh dominates: that is option 2.

Before the `/arch:AVX2` fix, on the 7800X3D with MSVC (`amr_mpi.py`, one run): serial 5.87 s (remesh 0.98), OpenMP 2 / 4 / 8 / 16 threads 3.54 / 2.29 /
2.08 / 2.02 s, MPI 2 / 4 / 8 / 16 ranks 3.87 / 3.05 / 3.77 / 7.46 s (remesh 1.12 / 1.26 / 2.04 / 4.40); `amr_mpi.py --check` bitwise
on 1-7 ranks, 94 unit tests. Under MPI each level is partitioned on its own, so the fine levels (866 and 9 080 leaves here) give a
rank a few hundred leaves and their exchanges are latency-bound; on this mesh the replicated remesh still costs more (4.4 of 7.5 s on
16 ranks).

**Accuracy per cost re-run** (`airfoil_amr.py --table`, 7800X3D, MSVC, scalar viscous loops, t = 6): the errors are those of section 20
of `amr-cell-based.md` to every printed digit (bitwise results); the wall times of the cell engine fell by 26 to 30 % (default 10.7 ->
7.8 s, criterion 0.005 / diffusion 4 24.7 -> 18.1 s, 0.002 / 4 72.7 -> 53.6 s). Against the patch engine as measured in section 20
(24.4 s for 0.0074 / 0.0387 with 16x16 patches, 20.3 s for 0.0085 / 0.0457 with 8x8; it is gone and was not re-measured, and it did
not have the vector Riemann solver), the cell engine now reaches 0.0078 / 0.0411 in 18.1 s.

## OpenMP scaling of the remesh (branch `amr-remesh-omp`)

After option 1 the remesh stopped scaling at 8 threads (Threadripper, gcc: 0.40 s from 8 threads on, 28 % of the loop at 16
threads). What was left was serial: the new leaf list of `CellMesh::adapt` (three streams per level, `push_back` and `std::merge`),
the side lists of `CellMesh::rebuild` (count / scatter over the faces), `CellFluxEngine::rebuild_lists` (per-level lists, sorted
by marks), the merge of the level cells and the flag loop of the indicator, and the serial allocation and fill of the face list and
of the engine tables. Now parallel, with integers and distinct keys only, so every table is the serial one for any thread count:

* `fv/ParallelScan.hpp`: a blocked prefix sum, a co-rank merge of two sorted runs (each output block placed by a binary search;
  `std::merge`'s output since the keys are distinct), the collection of the indices that satisfy a predicate and the compaction of
  marks set from many threads (relaxed atomic bytes). A fixed number of blocks (64), so the split does not depend on the thread
  count; inputs below 4 `kOmpMinWork` run the serial loop.
* `CellMesh::rebuild`: Morton keys and lookup tables in parallel; face offsets by the prefix sum; the face list before last is
  reused as the buffer; `face_level_begin` by `partition_point`; side lists by atomic counts, prefix sum and one slot per face (a
  side with two faces holds the children 2t and 2t + 1 of the transverse position t, even first: the order of the old serial swap;
  checked after the fill). The dirty marking visits only the new leaves.
* `CellMesh::adapt`: the three streams of every level counted per block, filled per block at their offsets and merged in parallel
  straight into the new list; the merge test runs in parallel (each group is tested by its anchor and writes its own entries).
* `CellFluxEngine`: `rebuild_lists` without a partition (the lists of a level are its ranges; `need_` and `vcells_` are sets marked
  from the faces), the partitioned path (MPI, one thread per rank) unchanged; `leaf_codes` / `face_stencil` write the boundary -1
  themselves, so an incremental build reuses the tables of the build before last instead of allocating and filling them; the special
  leaves of the virtual-cell keys collected in parallel.
* `CellIndicator`: the cells of each level by the co-rank merge, filled in parallel; `CellAdaptor::remesh` flags in parallel.

Bitwise after every commit: golden dumps of ten adaptive runs (the four `amr_mpi.py --check` cases, MMS 40 steps, cavity 200 steps,
Sod remeshed every 2 steps, the two AMR ghost-cell cases, the 720x360 airfoil) on 1 / 3 / 7 threads against the binary of `main`, the
38 references and the unit tests; on the final commit also on 1 / 2 / 3 / 4 / 7 threads, with `FENICE_FULL_REBUILD=1`,
`FENICE_NO_SIMD=1`, an `FV_NO_SIMD` build and clang without OpenMP;
no `vfm` in the objects; new unit tests `remesh_tables_do_not_depend_on_the_thread_count` (indicator, leaves, state, faces, gather
lists, incremental bookkeeping and every engine table with 1, 3 and 7 threads over 12 remeshes) and
`parallel_scan_helpers_equal_the_serial_loops` (the blocked paths against the serial loops). `rebuild_lists` was cross-checked
against the old serial code on every build of the tests and golden runs (temporary check, not committed).

Container (4-core Intel, gcc), airfoil AMR, min of 3 runs, time loop / remesh:

| run | `main` | `amr-remesh-omp` |
|---|---:|---:|
| 720x360, 50 steps, 1 thread | 10.55 / 1.60 s | 10.61 / 1.52 s |
| 720x360, 50 steps, 4 threads | 4.17 / 0.97 s | 4.12 / 0.60 s |
| default 192x96, t = 1.5, 1 thread | 5.75 / 0.77 s | 5.47 / 0.73 s |
| default 192x96, t = 1.5, 4 threads | 3.00 / 0.52 s | 2.82 / 0.37 s |

On 4 threads the remesh parts of the 720x360 run went from indicator 0.28, adaptation 0.47, engine build 0.34 s to 0.23, 0.23, 0.21 s
(the wave-speed bound, 0.20 s, was already parallel). Memory: the spare face list and engine tables stay alive between remeshes,
peak RSS 444 -> 458 MB (+3 %) on the 720x360 run, 41 -> 44 MB on the default one.

Threadripper PRO 5965WX (24 cores, gcc, WSL), same run (720x360, 50 steps, 10 remeshes), min of 3, interleaved, every state bitwise
equal to `main` on 1 thread (`amr_mpi.py --check` and `ibm_ghost.py --check` also passed there), time loop / remesh:

| threads | `main` | `amr-remesh-omp` |
|---:|---:|---:|
| 1 | 5.08 / 0.75 s | 5.05 / 0.73 s |
| 8 | 1.64 / 0.40 s | 1.44 / 0.21 s |
| 12 | 1.49 / 0.41 s | 1.25 / 0.19 s |
| 16 | 1.48 / 0.42 s | 1.20 / 0.18 s |
| 24 | 1.57 / 0.47 s | 1.30 / 0.19 s |
| 32 | 1.76 / 0.53 s | 1.36 / 0.20 s |

The remesh now goes 0.73 -> 0.18 s (4x) instead of 0.75 -> 0.40 s, and no longer grows beyond 16 threads (it did with `main`: 0.53 s on
32); on 16 threads it is 15 % of the loop instead of 28 %, and the loop is 19 % faster. Parts on 16 threads (`main` -> branch): indicator
0.13 -> 0.08, adaptation 0.22 -> 0.08, engine build 0.16 -> 0.07, wave-speed bound 0.05 -> 0.04 s. What is left serial is mainly the
allocation and zero-fill of the per-leaf work arrays of the indicator and of `adapt` (state, flags); not pursued.
