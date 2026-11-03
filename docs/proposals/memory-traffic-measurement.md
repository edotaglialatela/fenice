# Uniform solver performance: what is left after the measurements

Status: measured on 2026-11-03 (Ryzen 7 7800X3D, 8 cores / 16 threads, MSVC; pass timings, thread scaling and A/B runs are in
`results/validation/performance/measurement-report.md`, made with `cpp/tests/profile_roofline.py`). The 2-D viscous sweep reads the
cell field and a ghost ring instead of a padded copy and a primitive copy (`get_padded_frame`, `viscous_residual_rows`): bit-identical,
1.22-1.24x on 2048^2 with 8-16 threads, 1.03x on one, no change on a cache-resident 512^2. What follows is what is open.

## What the measurements say

* **Bytes are modeled, not measured.** No hardware counter was available on this Windows workstation (no AMD uProf, no usable DRAM or
  L3-miss counter). The copy kernel (~50-55 GB/s on one core and on 16) is an empirical reference, not a DRAM ceiling. The modeled
  traffic of a viscous RK2 stage is ~440 B per cell (convective residual ~120, viscous sweep ~160, time update ~160).
* **A third of a large viscous stage does not scale with threads.** The passes that only copy, convert or update a whole array take the
  same time on 1, 4, 8 and 16 threads (`axpy` 4.5, SSP combine 3.25 ns/cell per step), while the compute passes scale (convective
  residual 68 -> 12 ns, viscous sweep 41 -> 5.8 ns). The stage as a whole runs at ~22 GB/s modeled, far below the reference, and
  reaches only ~4x on 8 cores; 16 threads add 6 % over 8 (SMT). Removing a whole-array pass pays on many threads and on meshes larger
  than the cache.
* **Short profiles of large meshes are skewed.** The `boundary` section at 4096^2 is a one-time first-touch cost (the face arrays are
  resized on the first call, `Boundaries.cpp`), about 2.1 ns/cell at every thread count and 0.2 s in total whatever the number of
  calls. Profile with enough steps or leave the first one out.
* **Small meshes:** at 64^2 and 128^2 viscous MUSCL is slower on 16 threads than on 8; at 512^2 they tie. This supports the fork / join
  overhead concern, not by itself a new `kOmpMinWork`.

## Candidates still open

Ordered by measured wall-time share; bit-identity is kept unless stated.

1. **Fuse the later Runge-Kutta stage updates into the residual.** The first stage cannot be fused (its `dt` needs the wave speed of the
   whole sweep) and `axpy` is that stage (7 % of a step). Only the SSP-RK2 combine (5 % of a step; the RK3 / RK4 combines likewise)
   can be applied row by row after the residual of that row, if the residual reads only the old stage values. Realistic gain 2-4 %;
   every residual path (MUSCL, first order, viscous accumulate, body force, sources, IBM) must write its rows first. Not worth the
   invasiveness on its own.
2. **Fewer fork / join regions** per stage: one parallel region with `nowait` loops where the data dependence allows. Mostly small meshes.
3. **Non-temporal stores for pure outputs** (removes the write-allocate read of `R`). Needs aligned rows; only justified by modeled bytes.
4. **A 4-slot state in 2-D** (`rho*w` is identically 0; about -20 % of state traffic). A large refactor of `State` / `Field`, the
   checkpoint writer and every kernel; only if hardware counters show that the state arrays dominate the traffic.
5. **Temporal blocking** (two stages per sweep of a tile): fights the global `dt` and the halo depth; research item.

## Not measured, and decisions that are the user's

* Real DRAM / L3 traffic per stage (needs a platform with counters: `perf stat` on Linux, or the vendor profiler) and the isolated
  per-pass microbenchmark (several production passes are not independently callable). Rank candidates 3-5 by bytes only after that.
* **Reciprocal multiplication** for the divisions by `dx`, `dy`, `2 dx` in the gradients and viscous fluxes (about ten per cell): moves
  results at roundoff level (the same trade as the 2026-11-03 division diet) and needs a deliberate regeneration of the references.
* **FMA or fast-math** in any form: breaks the scalar / AVX2 / serial / MPI / thread-count bit identity; only as a separate opt-in mode
  with its own references, if ever.

## Verification protocol for any candidate

1. Before editing: `dump_state` of the old binary for the golden set (512^2 periodic viscous, no-slip cavity, cavity with
   `boundary_order = 2`, Poiseuille, 2- and 3-rank MPI) at 1, 3 and 8 threads, and the 720x360 airfoil if AMR code is touched.
2. After: hashes identical; `fenice_tests`; `regression.py` serial and `--mpi 4`; `amr_mpi.py --check` for AMR code.
3. A unit test against the unfused reference on a mesh above `kOmpMinWork` (thread seams), periodic and wall sides, that fails when a
   seam row is deliberately wrong.
4. Timing: old and new binary side by side, min of 3, at 1024^2-2048^2 on 1 / 8 / 16 threads. Kept only if it wins; then it replaces
   the old path (no run-time switch; the unfused version stays only as the test reference).
