# Uniform solver performance: measuring the memory traffic, then reducing it

Status: measurements run on 2026-11-03. M1 profiling, integrated pass timing, the full regime sweep (M4), and repeated same-binary
A/B timing (M5) are recorded in [the measurement report](../../results/validation/performance/measurement-report.md). The
standalone pass microbenchmark described in M2 was not run because several production operations are not independently exposed.
M3 hardware traffic counters were unavailable on this Windows workstation, so byte counts remain modeled. Candidate 1 (no padded
copy, no primitive copy: the viscous sweep reads the cell field and a ghost ring) is implemented and bit-identical; the other
candidates are open.

**Caveat on the 4096^2 timings.** The `boundary` section of the 3-step profiles is a one-time cost, not a per-stage one: about
2.1 ns/cell at every thread count and 0.2 s in total whether the run makes 3 or 6 calls, which is the first-touch (zero fill) of
the face arrays resized on the first call (`Boundaries.cpp`, `UL.resize`). In a long run it is negligible, so shares of the stage
quoted below are slightly too small (the padded copy plus conversion is about 21 % of a steady stage, not 20 %), and short runs on
large meshes should be profiled with enough steps or with the first step left out.

## What is known

Measured with cpp/tests/profile_roofline.py on the workstation described in the measurement report:

* **Integrated pass timing at 4096^2, 16 threads:** an RK2 viscous step is 61.73 ns/cell/step (30.87 ns/cell/stage).
  Padded copy plus conservative-to-primitive conversion is about 6.15 ns/cell/stage, or 20% of the stage. Even if both
  costs disappeared entirely, Amdahl's law caps the speedup near 1.25x; a fused sweep would still need conversion work.
* **Runge-Kutta updates are visible:** axpy plus SSP-RK2 combine account for about 3.87 ns/cell/stage, around 13% of the
  step. This is a wall-time share, not a byte count.
* **Scaling:** at 4096^2, 16-thread speedups over one thread are 4.57x for first order, 4.15x for MUSCL RK2, and 4.01x for
  viscous MUSCL RK2. At 512^2, they are 4.26x, 4.63x, and 5.36x. At 64^2 and 128^2, viscous MUSCL is slower at 16 than at
  8 threads; at 512^2, 8 and 16 threads are effectively tied. This supports the small-mesh parallel-overhead concern but
  does not by itself justify changing kOmpMinWork.
* **Existing fusion A/B results at 2048^2:** disabling convective fusion makes the 16-thread first-order, MUSCL, and viscous
  MUSCL cases 1.88x, 1.69x, and 1.43x slower. Disabling only viscous fusion makes viscous MUSCL 1.09x slower at one thread
  and 1.34-1.36x slower at 8-16 threads. These are measured timing gains; the measurements do not prove that DRAM traffic
  caused them.
* **Optional work:** on the 128^2 manufactured-solution case, source evaluation and addition take about 97% of the step;
  on the selected IBM cylinder case, copying and applying the forcing takes about 79 us/step. These cases are diagnostic,
  not representative of a constant source or large mesh.
* The copy kernel reports about 50.5-55.6 GB/s at one thread and 52.1-53.2 GB/s at the configured maximum of 16 threads.
  This is an empirical reference only, not a measured DRAM ceiling.
* **The stage is not at that reference as a whole, but its streaming passes are.** The modeled ~680 B per 30.9 ns stage is about
  22 GB/s, and the compute-heavy passes sit far below it (convective residual 9.8 GB/s, viscous sweep 27). The passes that only
  copy, convert or update a whole array (padded copy 3.06, conversion 3.07, `axpy` 4.5, SSP combine 3.25 ns/cell) take the same
  time on 1, 4, 8 and 16 threads (a plateau at ~39-49 GB/s modeled), while the compute passes scale (residual 68 -> 12 ns,
  sweep 41 -> 5.8 ns). About a third of a large viscous stage is therefore flat in the thread count, which is why the whole stage
  only reaches ~4x on 8 cores (16 threads add 6 % over 8: SMT on a 7800X3D). Removing such a pass pays on many threads and on
  meshes larger than the cache, and not on a cache-resident 512^2.

## What is not known

The bytes per cell-stage that the solver really moves. profile_roofline.py prints an implied rate using an assumed byte count;
that rate is a model, not a measurement. AMD uProf was unavailable and Windows exposed no usable DRAM or L3-miss byte counter.
The elapsed-time difference from skipping a pass would measure its runtime cost, not the bytes it transferred, so it cannot
replace those counters. The traffic model (from reading Simulation::assemble; State is 40 B, even in 2-D) is:

| part of a stage (viscous RK2, per cell) | traffic estimate |
|---|---|
| convective residual, fused rows: read `U`, write `R` (write-allocate: the line is read first) | ~120 B |
| viscous row sweep: reads `U` (rows largely cached), `R` read + write | ~160 B |
| time update, per stage (`axpy` and `ssp_combine`, 160 B each, per step) | ~160 B |
| **total** | **~440 B** |

The viscous sweep used to read a padded copy of `U` and a padded primitive copy (written by two separate passes, ~240 B more);
the sweep now converts each row as it goes and reads only the ghost ring from a small array. The table is a source-level
estimate and may be off substantially because the integrated solver reuses data through cache. The timings above do not
replace this traffic model with measured bytes.

## Measurement plan

Nothing here changes the numerics. All of it can be kept as scripts under `cpp/tests/`.

**M1. Time every pass (completed).** Add profile sections (zero cost when `FENICE_PROFILE` is off) for what is not covered today: the time
update (`axpy` / `ssp_combine`, per scheme), `get_padded_U` + `fix_padded_corners`, the padded `cons_to_prim`, the viscous sweep
alone, and the body-force / source / IBM additions. The profile then adds up to the loop time and "everything else" disappears.

**M2. Each pass against the ceiling (integrated timings done; isolated microbenchmark not run).** For every pass of M1, a micro-benchmark (`cpp/apps/bench_passes.cpp`, next to `bench_riemann`)
runs it in isolation on an array larger than the L3 with 1 / 4 / 8 / 16 threads and reports ns per cell, the bytes it must move by
construction (from the table above, counted per array and written next to the code) and the resulting GB/s. A pass at 80-100 % of
the ceiling is done (only fewer bytes help); a pass at 30 % has a compute, latency or parallelism problem and is the target.

**M3. Bytes per cell-stage, measured (unavailable here).** Hardware counters where the platform offers them (DRAM or L3-miss
bytes per run; `perf stat` on Linux, the vendor's profiler on Windows). The Windows workstation exposed no suitable counter.
Skipping a pass can estimate timing impact but cannot measure its bytes, so that fallback is not used.

**M4. Regimes (completed).** Sweep the mesh from 64^2 to 4096^2 and the threads 1-16 for first order, MUSCL, and viscous; the
CSV reports ns per cell-stage. The small viscous cases confirm the 8-versus-16-thread overhead, but this sweep alone does not
select a new `kOmpMinWork`.

**M5. Repeatability (three runs and same-binary A/B completed).** The same Release binary is used for the A/B switches and the
thread count is recorded. WMI's 4201 MHz clock fields were not a dynamic frequency trace, so the in-run clock could not be
recorded. Compare gains against the raw three-run spread in the CSV.

Deliverable of this pass: measured timings and a ranking by wall-time share. A ranking by measured bytes saved still requires M3.

## Improvement candidates

Ordered by measured wall-time share and implementation risk. Traffic figures in parentheses remain estimates until M3 is available.
Each keeps the bit-identity contract unless stated.

Done: **no padded and no primitive copy in the 2-D viscous sweep.** `viscous_residual_rows` reads the cell field and the ghost
ring of the padded state (`BoundaryManager::get_padded_frame`, `fix_frame_corners`; the MPI layers are written into the ring);
a row's primitive state is converted with the row. Bit-identical (the 44 references, serial and 4 ranks; the unit test compares it
with the gradient version on the padded array for periodic, wall, second-order, and neighbour-rank rings, 1 / 3 / 7 threads).
Measured (viscous MUSCL RK2, 2048^2, min of 3, ns per cell-stage, before -> after): 1 thread 117 -> 114, 8 threads 29.2 -> 24.0,
16 threads 27.1 -> 21.9 (1.22-1.24x); 1024^2 16 threads 27.3 -> 23.1; 512^2 (in cache) unchanged. This is close to the Amdahl
bound of ~1.25-1.27x from the pass timings, so the conversion arithmetic that remains costs little.

1. **Fuse the later Runge-Kutta stage updates into the residual**. The first stage cannot be fused: its
   `dt` comes from the wave speed of the whole sweep, known only at the end, and `axpy` (4.5 ns, 7 % of a step) is that stage. Only
   the SSP-RK2 combine (3.25 ns, 5 % of a step; RK3 / RK4 combines likewise) knows `dt` and can be applied row by row after the
   residual of that row, if the residual reads only the old stage values. Realistic gain 2-4 %, not the 10-15 % first
   guessed; every residual path (MUSCL, first order, viscous accumulate, body force, sources, IBM) must write its rows
   before the update reads them. Not worth the invasiveness unless the update pass shows up elsewhere.
2. **Non-temporal stores for pure outputs** (up to ~1/3 of the modeled write traffic): removes the write-allocate read of `R`.
   Needs aligned rows; try on the AVX2 kernels. Only justified by modeled bytes until M3.
3. **Fewer fork / join regions** per stage (each pass is a separate `parallel for`): one parallel region per stage with `nowait`
   loops where the data dependence allows. Affects small meshes most (M4), where 16 threads are slower than 8.
4. **A 4-slot state in 2-D** (estimated -20 % of state traffic: `rho*w` is identically 0). A large refactor of `State` / `Field`
   indexing, the checkpoint writer and every kernel; only if M3 shows that the state arrays are most of the traffic. Padding
   `State` to 64 B is the opposite direction (+60 % traffic) and is not proposed.
5. **Temporal blocking** (two stages per sweep of a tile): consider only if M3 shows bandwidth is limiting. It fights the global
   `dt` and the halo depth, so it remains a research item.

## Decisions that belong to the user

* **Reciprocal multiplication** for the divisions by `dx`, `dy`, `2 dx` in the gradients and viscous fluxes (about ten divisions
  per cell): moves the results at roundoff level (the same trade as the 2026-11-03 division diet) and needs a deliberate
  regeneration of the references. Not done unasked.
* **FMA or fast-math** in any form: breaks the scalar / AVX2 / serial / MPI / thread-count bit identity that the tests assert;
  only as a separate opt-in mode with its own references, if ever. The timings show fusion gains on large cases, but without
  hardware counters they do not establish a memory limit or predict whether arithmetic changes would help.

## Verification protocol (every candidate)

1. Before editing: `dump_state` of the old binary for the golden set (512^2 periodic viscous, no-slip cavity, cavity with
   `boundary_order = 2`, Poiseuille, 2- and 3-rank MPI) at 1, 3 and 8 threads, and the 720x360 airfoil for the AMR code.
2. After each change: hashes identical; `fenice_tests`; `regression.py` serial and `--mpi 4`; `amr_mpi.py --check` if AMR code is touched.
3. A new unit test per fused pass that compares it with the unfused pass on a mesh above `kOmpMinWork` (so thread blocks have
   seams) with periodic and wall sides, and fails when the seam row is deliberately wrong.
4. Timing: the M5 protocol (old and new binary side by side, min of 3); the change is kept only if it wins on the workstation,
   and then it replaces the old path (no run-time switch: the unfused version stays only as the reference of its unit test).

## Expected outcome

Candidate 1 of the first plan (no padded / primitive copy) is in and measured at 1.2x on large meshes. What is left is small by
the pass timings: the SSP combine is 5 % of a step (2-4 % realistic), and the first-stage `axpy` cannot be fused. M3 is still
needed before ranking the remaining candidates by actual DRAM bytes saved; the current timings rank them by measured wall-time
share. M2's isolated microbenchmark remains outstanding. See the report for the pass breakdown, regime plot, and A/B results.
