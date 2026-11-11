# Performance improvements: candidates after a scan of the hot paths

Status: proposal, nothing implemented. Measured on 2026-11-11 (Ryzen 7 7800X3D, 8 cores / 16 threads, MSVC, `FENICE_PROFILE=1`,
time loop only, minimum of 3 runs unless stated, ranks not pinned). The scripts, the golden hashes and the patch of the three small
experiments are in `results/perf_scan/` (`golden.sh`, `tm.sh`, `sha_base.txt`, `small_bit_identical_experiments.patch`).
Read for this scan: the time loops, the uniform kernels and their AVX2 twins, the AMR residual, advance, prepare and indicator. Not
read: the mesh and share builders, the MPI exchanges, IO, boundaries, case files, Python.

This note complements `memory-traffic-measurement.md` (stage-update fusion, non-temporal stores, reciprocal multiplication): those
candidates stand as written there and are not repeated.

## Where the time goes

All runs: MUSCL (Van Leer), low-Mach HLLC, SSP-RK2, viscous. Shares of the time loop.

| Case | 1 thread | 16 threads |
|---|---|---|
| 2-D Taylor-Green 1024^2, 40 steps | 9.14 s: convective 57 %, viscous 39 %, stage updates 3 % | 1.69 s: 53 % / 33 % / 12 % |
| 3-D Taylor-Green 96^3, 20 steps | 8.57 s: viscous 56 %, convective 37 %, divergence 4 % | 2.29 s: viscous 60 %, convective 24 %, divergence 8 % |
| AMR airfoil 720x360, 50 steps | 4.94 s: convective 46 %, remesh 18 %, viscous 18 %, gather 12 %, prepare + forcing 8 % | 1.46 s: remesh 27 %, convective 25 %, viscous 16 %, prepare + forcing 14 %, gather 12 % |
| Uniform IBM cylinder 1600x800, 40 steps | 10.66 s: convective 60 %, viscous 34 %, updates 3 %, forcing 2 % | 2.31 s: convective 48 %, viscous 28 %, updates 14 %, forcing 7.6 % (mask: 9 %) |

Per cell and stage on one thread: 2-D convective 62 ns, 2-D viscous 43 ns, 3-D convective 90 ns, 3-D viscous 136 ns (sweep 110,
gradients and copies 26). The Riemann solver alone (`bench_riemann`, batch mode, hash loop included) costs 22.3 ns per face with the
low-Mach scaling and 17.5 ns without; the uniform sweep spends 31 ns per face.

## Candidates

Ordered by expected gain per effort. "Measured" means built and timed; everything else is an estimate from the shares above.

### 1. Vector kernels for the 2-D uniform viscous sweep (bit-identical)

`NavierStokes::viscous_residual_rows` is scalar. Its per-cell part (`fill`: primitive state, temperature, four velocity derivatives; 12
divisions) is 26 % of the sweep, 10 % of the run: measured by running it twice per row (9.24 -> 10.16 s on 1 thread, 1.68 -> 1.83 s on
16). Its expressions are those of the AMR kernel `viscous_cells_avx2`, with the neighbours on contiguous rows instead of index codes.
The rest of the sweep (32 ns per cell) is the face fluxes and the speed bound: lambdas that return a `State` by value and test the
boundary cases (`lo`, `hi`, MPI ghosts, `bface2_`) for every face of the row. The row buffers (`u`, `v`, `T`, the derivatives) are
already contiguous per quantity, so the interior faces can be a branch-free loop with plain vector loads; the speed bound has the AMR
twin `viscous_speed_avx2`. The AMR face-flux twin was dropped because its loads were indexed; that reason does not apply to rows.

Expected: 15-20 % of a 2-D viscous run on 1 thread, about 10 % on 16. Risk: the rule "keep a vector kernel only if it is faster with
MSVC too". Test: `viscous_rows_equal_the_gradient_version_bit_for_bit` already covers the seams.

### 2. The 3-D viscous path (bit-identical, large)

The largest share of a 3-D run, and the part that scales worst (sweep 4.1x on 16 threads, convective 5.8x). It still does what the 2-D
path did before the row sweep: a padded copy, a primitive copy, three gradient fields, 13 per-cell scratch arrays, three face-flux
arrays, a separate balance pass and a separate speed pass. The 2-D rework gave 1.22-1.24x on many threads. A plane-by-plane sweep with
two planes of cell quantities in cache is the same idea in 3-D.

Expected: 1.3-1.5x on the viscous part, 15-20 % of a 3-D run. A free first step: `compute_gradients_3d` computes the gradient of the
energy slot, which nothing reads (3 of the 15 divisions per cell). The 3-D convective path also stores the face fluxes and balances them
in a second pass (`divergence`, 8 % on 16 threads); the 2-D fused residual shows the cure.

### 3. Round trips of the low-Mach face states (moves results at roundoff: the user's decision)

With MUSCL and the low-Mach solver a face goes: primitive (reconstruction) -> conservative -> primitive (`prepare4`) -> scaled ->
conservative -> primitive (`hllc4`). That is eight vector divisions per face that only convert back and forth, and the sound speed of
both sides is computed twice (it depends on rho and p only, which the scaling does not touch). Passing the primitive states through
gives the same flux to roundoff. The cost of the stage is visible in `bench_riemann`: 22.3 against 17.5 ns per face.

Expected: 10-15 % of the convective time, 5-8 % of any run, uniform and AMR (one kernel, scalar and AVX2 twins changed together).
Not bit-identical: the same trade as the division diet, with regenerated references.

### 4. The state copy of the uniform IBM (bit-identical)

`assemble` copies the whole state every stage (`Uf_ = U_in`, a serial copy) to overwrite the cells of the body: 624 ghost cells of
1.28 M in the cylinder run. It is 7.6 % of a 16-thread run with the ghost wall and 9 % with the mask. Making the copy parallel gains only
2 % (measured: 2.318 -> 2.272 s ghost, 3.046 -> 2.972 s mask): the pass is memory bound. For the ghost wall the copy can go: save the
ghost cells, write the ghost states in place, restore them after the residual. Expected: about 6 % on many threads. The MPI path has the
same copy (`forced = Uin` in `run_mpi`). The mask forcing converts every cell to primitive and back, fluid cells included, so it needs
the pass unless that round trip is dropped (roundoff, references).

### 5. AMR: passes that repeat or discard work (bit-identical, measured)

* `force_level` runs for a level at the start of the step, in `prepare` of every stage and after the advance. The one in `prepare` of the
  first stage reads a state the level was forced from already.
* `wave_speed` after a remesh runs `compute_level_fluxes`, which also computes the viscous cell quantities and face fluxes; only the
  speed is kept.

Both skipped (prototype in `small_bit_identical_experiments.patch`; the forcing skip is serial only, with MPI the halo leaves change in
`exchange_stage`): airfoil 4.92 -> 4.81 s on 1 thread, 1.399 -> 1.359 s on 16, wave-speed bound 0.29 -> 0.23 s. The final states of the
six golden cases are identical. 2-3 %: worth it only as part of other AMR work.

Open, not investigated: `prepare` + forcing are 14 % of a 16-thread step and scale 1.7x; the indicator scales 1.75x (the per-level
bounding-box loop, the front of the join test and the allocations are serial, and the level pass runs twice); `gather` is 12 %.

### 6. A line kernel for the uniform convective sweep (bit-identical, uncertain)

The sweep spends 31 ns per face against about 17 for the solver alone. Part of the gap is work (reconstruction: ten divisions per face
with Van Leer; the padded primitive pass; the balance), part is overhead of sharing the batch kernel with the AMR face list: 80 scalar
loads assembled into vectors per four faces, five index stores per face, about twenty blends per batch that select the axis, 24 scalar
stores, a scalar balance. On the uniform mesh every stencil of a line is contiguous if the primitive rows are stored per quantity, and
the axis is known per line. Expected: 10-20 % of the convective time. Prototype one axis and time it before deciding.

### 7. Usage trap: threads times ranks

`--threads N` on a case with `use_mpi = true` starts `n_cpus` ranks with N threads each: the 1600x800 cylinder took 11.7 s with 16 x 16
against 2.3 s. Clamp ranks x threads to the logical cores, or warn.

## Question: can operator splitting reduce the viscous cost?

Not the cost of an evaluation: splitting changes how often each operator runs and with which step, not what a sweep costs. Two things
it could buy, both changes of the time integration (the user's choice), both with new references:

* **Fewer viscous evaluations per step** (the viscous term evaluated once and held over the stages). Halves the viscous time with
  SSP-RK2: about 20 % of a 2-D run and 28 % of a 3-D one. The viscous term becomes first order in time, the observed orders of
  `mms_time.py` drop, and the stability budget has to be derived again.
* **A larger step where the viscous eigenvalue limits it.** The first step of each case with and without the viscous term:

  | Case | dt viscous | dt inviscid | viscous share of the speed bound |
  |---|---|---|---|
  | Taylor-Green 128^2 | 1.53e-3 | 1.91e-3 | 20 % |
  | Taylor-Green 512^2 | 2.41e-4 | 4.78e-4 | 50 % |
  | Taylor-Green 1024^2 | 8.0e-5 | 2.39e-4 | 66 % |
  | Lid cavity 128^2 | 2.18e-4 | 2.64e-4 | 17 % |
  | IBM cylinder 1600x800 | 6.6e-5 | 9.7e-5 | 32 % |
  | Taylor-Green 3-D 96^3 | 1.970e-3 | 1.984e-3 | < 1 % |

  With Strang splitting (half viscous step, convective step, half viscous step) the convective operator takes its own step and the
  viscous one is subcycled, or advanced with a stabilised multi-stage scheme whose stable step grows with the square of the stage
  count. At 1024^2 the convective evaluations would drop by 3x (57 % of the cost) while the viscous ones stay about the same: roughly
  1.6x for that run. It gives nothing where the share is small (the 3-D run, the cavity, coarse meshes).

Costs of either: a splitting error of second order in dt that also enters steady solutions (the cavity and Poiseuille validations would
depend on dt); the boundary conditions, the IBM forcing and the AMR flux registers must be applied per operator; the invariant "the
copies of the viscous time-step expression stay identical" and the shared budget are replaced; every viscous reference moves.

Recommendation: no splitting for speed. Reduce the cost per evaluation first (candidates 1 and 2). Splitting is a time-step lever for
fine, viscous-limited meshes only; if those runs matter, the stabilised multi-stage variant is the one to study, as a separate proposal.

## Question: OpenMP only?

On one node, yes for running; no for the code. Same binary, ranks x threads:

| Layout | Taylor-Green 1024^2 | AMR airfoil 720x360 |
|---|---|---|
| OpenMP, 16 threads | 1.63 s | 1.40 s |
| OpenMP, 8 threads | 1.87 s | 1.53 s |
| 16 ranks x 1 | 1.69 s | 2.33 s |
| 8 ranks x 1 | 1.90 s | 1.91 s |
| 8 ranks x 2 | 1.61 s | 2.08 s |
| 4 ranks x 4 | 1.60 s | 2.50 s |
| 2 ranks x 8 | 1.63 s | 3.28 s |

* The uniform solver has the same ceiling in every layout (memory-bound passes): MPI buys nothing on one node and costs nothing.
* The AMR engine is 1.4x faster with threads than with its best rank count, and the hybrid layouts are the slowest (2 x 8 is 2.3x
  slower than 16 threads): the per-rank work of the distributed mesh (share, exchanges, remesh) does not use the threads of a rank.
* MPI stays the only way to use more than one node, or more memory than one node has. One rank per socket with threads is the usual
  layout on multi-socket nodes, and it is the layout that is weak today for AMR.

Recommendation: keep MPI; make threads the documented and default choice on one node (candidate 7 guards the mix); spend no effort on
single-node MPI speed. If cluster runs with AMR are planned, threading the per-rank share and remesh work is the item to open.

## Question: can a 2-D run skip the z variables?

Partly today, fully only with a refactor. `rho*w` is not advanced by a pass of its own: it is the fifth slot of `State`, carried
through every kernel and every array. In 2-D it is exactly zero.

* **Skip the slot in the kernels (bit-identical).** Measured for the reconstruction only (the fifth variable's limiter, two of the ten
  divisions per face, and its loads): Taylor-Green 1024^2 9.48 -> 9.22 s on 1 thread (2.8 %), 1.72 -> 1.70 s on 16 (1.3 %), AMR airfoil
  5.02 -> 4.92 s (2 %); the five 2-D golden states are identical. Doing the same in the Riemann solver (flux, starred state, the
  selections and the shock blend of the fifth slot), the balance and the stage updates is about one fifth of the per-variable
  arithmetic: an estimated 4-6 % in total on one thread. It needs 2-D variants of the kernels (scalar and AVX2 twins together); the
  slot stays +0 because +0 plus a signed zero is +0, which the golden dumps must confirm for each kernel.
* **Store four slots in 2-D.** About 20 % less traffic in every whole-array pass; it matters on many threads and large meshes, where
  the stage updates alone are 12-14 % of a run. `State` has a compile-time size: this means kernels templated on the number of
  variables (or two builds), and changes in the checkpoint, VTK, halo and hot-start code. It is candidate 4 of
  `memory-traffic-measurement.md`; the measurement above does not change its ranking: the kernel-level skip is the cheap part.

## Suggested order

1. Candidate 1 (2-D viscous kernels): the best gain that keeps the bits.
2. Candidate 4 (IBM copy) and the kernel-level skip of the fifth slot: small, self-contained, bit-identical.
3. Candidate 3 (low-Mach round trips), if a regeneration of the references is accepted; reciprocal multiplication could share it.
4. Candidate 2 (3-D viscous), if 3-D viscous runs matter; candidate 6 after a one-axis prototype.

Verification for each: the protocol of `memory-traffic-measurement.md` (golden dumps before and after at 1 and 16 threads, also with
`FENICE_NO_SIMD=1` and `FENICE_FULL_REBUILD=1`; `fenice_tests`; `regression.py` serial and `--mpi 4`; `amr_mpi.py --check` for AMR
code; old and new binary timed side by side, kept only if it wins).
