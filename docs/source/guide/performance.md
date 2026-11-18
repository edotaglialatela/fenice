---
title: Performance and parallelism
nav: Performance
section: User guide
order: 50
---

> **In short.** Fenice uses vector instructions, OpenMP threads and MPI ranks together, and the result is the same bit for bit whichever you use.
> For one node use threads (the default); use MPI for several nodes or for more memory than one node has.
> The [practical notes](#practical-notes) are the quick answers; the tables are the measurements behind them.

Every kernel is native C++. Three levels of parallelism combine:

| Level | Mechanism | Used for |
|---|---|---|
| Vector | four faces or cells at a time: AVX2 (x86-64) or NEON (ARM64) | Riemann solvers, the MUSCL, WENO3-Z and WENO5-Z reconstructions (uniform meshes; the interior faces of AMR meshes), the viscous cell quantities and time-step bound of AMR meshes |
| Loop | OpenMP `parallel for` over faces / cells | Riemann fluxes, MUSCL, BCs, gradients, viscous terms, time-scheme updates |
| AMR faces | OpenMP over the faces and leaves of one refinement level | AMR runs |
| Domain | MPI strips with halo exchange | large uniform meshes, multiple nodes |
| AMR leaves | MPI over the leaves of each level, cut along a Morton curve | AMR runs (correct on any rank count; see the limits below) |

**Vector kernels.** On x86-64 CPUs with AVX2 and on ARM64 (NEON) the Riemann solver (the three HLLC variants, ideal and stiffened gas) runs four faces
at a time, and the MUSCL, WENO3-Z and WENO5-Z reconstructions run four faces at a time too (on AMR meshes for the interior faces whose stencil is made
of leaves; MUSCL is fused with the Riemann solver there). Every lane performs the same
IEEE operations in the same order as the scalar code (no fused multiply-add, no vector min / max, every branch computed and
selected by a mask), so the results are **bit-identical** with and without it, on any CPU.
* The scalar code is used on other CPUs
and for the leftover faces of a batch; `FENICE_NO_SIMD=1` switches the vector kernels off (to time them, or to check the bits).
* The Riemann solver costs 47 / 71 ns per face (HLLC / Low-Mach) as scalar code and 16 / 21 ns vectorised
(`bench_riemann`).
* On ARM64, where a value is two NEON registers, the gain is smaller: 55 / 84 against 41 / 54 ns on a hosted Linux
runner and 63 / 101 against 30 / 32 ns on an Apple-silicon one (shared machines: indicative figures).
* The viscous cell quantities (velocities, temperature and the gradients from the four neighbours) and the viscous time-step bound of
AMR meshes also run four leaves at a time, for runs of consecutive leaves whose neighbours are leaves (the others take the scalar
path).
* On the 720×360 airfoil (50 steps, one thread) all the vector kernels together save 3.6 s: 4.9 s against 8.5 s with `FENICE_NO_SIMD=1`
(Ryzen 7 7800X3D, MSVC).
* The kernels are written once, on a small set of operations (`SimdOps.hpp`: add, subtract, multiply, divide,
square root, comparisons that give a mask, bit logic and the selection of a lane) that each instruction set provides as one block of
that file: a value is four doubles, one AVX2 register or two NEON registers.
* On x86-64 the file of the kernels is compiled for AVX2 (`/arch:AVX2` with MSVC,
`-mavx2` with gcc / clang; FMA stays off): without `/arch:AVX2` the MSVC build of these kernels was 0.42 s *slower* than the scalar
loops.
* The CPU check that decides whether they run is compiled without AVX, so the program still starts on older CPUs.
* The viscous
face flux stays scalar: its vector version was slower than the scalar loop (1.1 to 1.25 s against 0.88 s), the kernel being dominated
by scattered loads with three divisions per face.

**Fused passes.** On a uniform mesh the residual is computed in one pass over rows of faces: the reconstruction (MUSCL, WENO3-Z or
WENO5-Z, or the cell states, first order), the Riemann solver and the flux balance of the cells run on a row at a time, so the face fluxes
and the left / right states never go to memory. `FENICE_NO_FUSED=1` takes the
separate passes; the two give the same bits (`fused_residual_passes_equal_the_two_pass_residual_bit_for_bit`).

* The 2-D viscous
residual is one sweep over rows as well: the velocity, temperature and their derivatives of a row and of the row above stay in a
small per-thread buffer, a cell's primitive state is converted with its row, and the padded state of the diffusive ghosts is never
built, only its ghost ring (`get_padded_frame`, a few entries per row and column; the interior is read from the cell field).
* On
large meshes, where the arrays do not fit in cache, this matters: viscous MUSCL RK2 on 2048^2 takes 109 / 22 / 20 ns per
cell-stage on 1 / 8 / 16 threads, about 1.2x less than with a padded and a primitive copy on 8-16 threads (1.03x on one); a 512^2
mesh, which stays in cache, takes the same time either way.
* Passes that only copy or convert a whole array run at the same
rate on one thread and on sixteen, so they limit the thread scaling of a large mesh; that is why removing them pays most on many
threads.
* The primitive
variables share one reciprocal of the density and the divergence multiplies by the reciprocal of the cell area, so the divisions
per face are few.
* Measured on the workstation (ns per cell-stage, 1 / 16 / 48 threads; $2048^2$ to $4096^2$ and $256^3$ meshes, SSP-RK2 for MUSCL and SSP-RK3 for WENO): MUSCL with the Euler equations 46 / 5.0 / 3.8
(the convective part of the viscous run), viscous Taylor-Green 66 / 8.4 / 6.9, 3-D viscous Taylor-Green 166 / 31 / 32, WENO3-Z 59 / 6.5 / 5.6, WENO5-Z 85 / 8.4 / 7.0,
WENO5-Z with the fourth-order flux 158 / 13 / 11. These are the numbers of the estimator below.

**Loop parallelism.** Loops shorter than `kOmpMinWork` (4096 items) run on one thread,
because OpenMP fork/join would cost more than the work. All reductions, such as the
maximum wave speed, are order-independent maxima. Results are
therefore **bit-identical for any thread count**.

**AMR.** A level advances with the other levels frozen, so the face and leaf loops of a level are
ordinary OpenMP loops (and, like the others, serial below `kOmpMinWork`). The result does not depend on
the thread count.

**AMR on MPI.** Under MPI the leaves of each level are distributed over the ranks along a Morton
(Z-order) curve. The result is identical to the serial one for any number of ranks, but the speed-up is
limited: the indicator and the flags are local to each rank, but the adaptation of the whole mesh (its leaves and face list) and the
build of the stencils are repeated on every rank, while between remeshes a rank stores only its leaves and the halo. On one rank or thread the update after a remesh is
incremental: only the leaves the adaptation touched and their face neighbours are recomputed, the rest is copied. OpenMP threads
are faster on one node; see [AMR]({{root}}modules/AMR.html) for the timings.

**MPI.** Each rank runs one OpenMP thread by default, to avoid oversubscription. Pass
`--threads N` or set `OMP_NUM_THREADS` for hybrid MPI + OpenMP runs. The MPI result is
identical to the serial one (see [MPIDecomposition]({{root}}modules/MPIDecomposition.html)).

## Cost of the reconstructions

Wall time of the whole run on the 24-core workstation of the next section (gcc, OpenMP, one process, 44 threads), cases as in the validation studies: the 2-D isentropic vortex
over one period (`nx = ny = 256`, 65 536 cells) and the 3-D Taylor-Green vortex at Re 1600 on $64^3$ to $t = 20$.
MUSCL uses the MC limiter with SSP-RK2, the WENO schemes SSP-RK3, CFL 0.5.

| Case | scheme | 1 thread | 44 threads | 44 threads vs 1 | $L_2$ density error / peak $\varepsilon$ |
|---|---|---:|---:|---:|---|
| vortex $256^2$ | MUSCL (MC) | 21.4 s | 2.6 s | 8.4× | $6.4\cdot10^{-5}$ |
| vortex $256^2$ | `weno3` | 45.3 s | 4.5 s | 10.0× | $4.8\cdot10^{-5}$ |
| vortex $256^2$ | `weno5` | 64.0 s | 6.0 s | 10.6× | $2.1\cdot10^{-6}$ |
| Taylor-Green $64^3$ | MUSCL (MC) | 470 s | 101 s | 4.7× | 0.01133 |
| Taylor-Green $64^3$ | `weno3` | 834 s | 155 s | 5.4× | 0.01091 |
| Taylor-Green $64^3$ | `weno5` | 1053 s | 188 s | 5.6× | 0.01267 |

`weno3` costs about 1.8 to 2.1 times and `weno5` about 2.2 to 3.0 times the MUSCL run on one thread (on the 2-D mesh of 65 536 cells the three schemes gain 8 to 11 times on 44 threads, on the 3-D mesh 4.7 to 5.6 times). The error is what the extra cost buys: on the smooth vortex `weno5` reaches at
$128^2$ (8.1 s on one thread) an error below that of MUSCL at $256^2$ (21.4 s), so for a given error it is the cheaper scheme there;
the Taylor-Green reference peak is 0.0127. The thread count never changes the result: all runs above are bitwise equal on 1 and 44 threads.

## Scaling on a 24-core workstation

The timings of the studies, here and on the [AMR]({{root}}modules/AMR.html) page, are measured on one workstation, a Threadripper PRO 5965WX (24 cores, 48 hardware threads, gcc, Linux). The solver was also built and run on other x86-64 machines (MSVC and gcc), on Apple M4 and on Linux ARM64, with the tests and the regression references passing, but no timing of those machines is quoted.

Strong scaling of the time loop, minimum of two runs, no output.
Each case is a fixed number of steps: the viscous Taylor-Green vortex on $4096^2$ (MUSCL, SSP-RK2, 40 steps) and on $2048^2$ (`weno3`,
SSP-RK3, 40 steps), the 3-D Taylor-Green vortex on $256^3$ (MUSCL, 10 steps), the AMR airfoil on a $1440\times720$ base mesh with 2 levels (50 steps, MUSCL and `weno5`)
and the octree sphere of the [sphere case]({{root}}cases/sphere.html) (60 steps). The speed-up is the time loop on one rank (or one thread) over the time loop of the run; ranks are bound to
physical cores up to 24, and the counts above 24 use both hardware threads of a core.

| Case | 1 rank | best MPI | best OpenMP | best ranks x threads |
|---|---:|---:|---:|---:|
| uniform 2-D MUSCL $4096^2$ | 89.1 s | 10.1x (48 ranks) | 9.6x (48 threads) | 11.8x (6 x 8) |
| uniform 2-D `weno3` $2048^2$ | 42.7 s | 10.8x (48 ranks) | 11.3x (48 threads) | 12.3x (4 x 12) |
| uniform 3-D MUSCL $256^3$ | 55.6 s | 7.3x (16 ranks) | 5.4x (16 threads) | 8.1x (12 x 4) |
| AMR airfoil, MUSCL | 19.2 s | 5.2x (48 ranks) | 5.5x (48 threads) | 4.8x (24 x 2) |
| AMR airfoil, `weno5` | 37.5 s | 7.7x (48 ranks) | 7.4x (48 threads) | 7.1x (24 x 2) |
| AMR octree sphere | 4.54 s | 2.5x (16 ranks) | 3.9x (16 threads) | 1.9x (12 x 2) |

<figure>
<img src="{{root}}img/workstation_speedup.png" alt="Speed-up of the six cases on MPI ranks, OpenMP threads and ranks x threads">
<figcaption>Speed-up of the time loop of the six cases on MPI ranks, OpenMP threads and ranks x threads (triangles, at the number of ranks). The grey band marks the counts above the 24 physical cores.</figcaption>
</figure>

<figure>
<img src="{{root}}img/workstation_efficiency.png" alt="Parallel efficiency of the six cases on MPI ranks, OpenMP threads and ranks x threads">
<figcaption>Parallel efficiency (speed-up divided by the number of workers) of the same runs.</figcaption>
</figure>

* **The uniform cases stop at 7 to 12 times the single-core speed**, whatever the mix of ranks and threads, and gain little past 16 to 24 workers: the same ceiling for MPI and OpenMP,
  as expected of passes over the whole array that are limited by the memory of the node. The hybrid runs (6 ranks of 8 threads, 4 of 12, 12 of 4) are 8 to 17 % faster than the best pure mode on the uniform meshes.
* **MPI against OpenMP** is close on the uniform cases (ranks ahead on the 3-D mesh, 7.3x against 5.4x on 16 workers). On the adaptive meshes OpenMP is ahead up to 16 workers, and the remesh grows with the ranks:
  on the octree sphere 16 ranks take 1.79 s with 1.23 s in the remesh, 48 ranks take 2.9 s with 2.4 s in it, and 16 threads take 1.16 s. Use MPI to go beyond one node; on one node prefer threads, or a few ranks of several threads.
* **`weno5` scales better than MUSCL on the adaptive airfoil** (7.7x against 5.2x on 48 ranks): the larger arithmetic cost per cell hides part of the remesh (34 % of the time loop against 43 % for MUSCL on 48 ranks). On the uniform meshes the schemes differ little (10.1x for MUSCL and 10.8x for `weno3` on 48 ranks).
* A mesh of a few thousand cells gains nothing from threads: loops under 4096 items run on one thread, so the fork/join of a longer loop would cost more than the work; MPI strips or one thread suit such cases.
* The results are bitwise equal for every rank and thread count; only the time changes.

The weak scaling keeps $1024\times512$ cells per rank (viscous Taylor-Green vortex, MUSCL, 40 steps): the time grows from 2.6 s on one rank to 3.4 s on 4, 4.7 s on 16 and 11.6 s on 48, an efficiency of 0.78, 0.55 and 0.23;
threads give 0.78, 0.53 and 0.20.

<figure>
<img src="{{root}}img/workstation_weak_scaling.png" alt="Weak scaling efficiency on MPI ranks and OpenMP threads">
<figcaption>Weak scaling (fixed cells per rank): $T(1)/T(N)$ on MPI ranks and OpenMP threads.</figcaption>
</figure>

<figure>
<img src="{{root}}img/workstation_time_split.png" alt="Time of each section over the time loop, for each case and number of MPI ranks">
<figcaption>Where the time goes: the sections of the time loop (they may nest, so a stack can exceed 1) for each case, against the number of MPI ranks (maximum over ranks).</figcaption>
</figure>

The table of every run, with the time split by section, is `results/validation/workstation_scaling/scaling.md` (written by the scaling stage of the workstation script, see `cases/validation/workstation_postprocess.py`).

{{widget:costEstimator}}

## Practical notes

* The final line of a run reports the **time-loop** duration. Snapshots are binary VTK
  by default (`vtk_format`); on very large meshes a high `save_interval` still keeps
  output from dominating the run time.
* `max_steps = N` stops a run after N steps, which is useful for timing.

**Where the time goes.** `FENICE_PROFILE=1` prints the total time step and residual sections on uniform runs, including the time updates, viscous padding and conversion, viscous sweep, IBM forcing, body force and sources.

* It also prints the output and remesh sections on AMR runs (the region the indicator reads, the flags, the topology, the migration of the state, the rebuild of the share and its engine tables, the partition; inside the rebuild the fetch of the leaves, the mask, the partial mesh, the engine build, the partition and the localisation), with the minimum and maximum over the ranks.
* Sections are timed once per step or remesh, never per face.
* Nested sections are inclusive, so the total and its children must not be summed together.
* With the switch off, profile sections only take the cached flag check.
* An AMR time step is also split into the stage copy, the face fluxes (primitive state and virtual cells, convective faces, viscous cell quantities, viscous faces), the flux registers, the gather into the cell residuals, the forcing and the halo exchanges (`amr: ...`), and `fetch_leaves` into its frontier, classification, exchange, serving and collection (`fetch: ...`).
* On one node a single process on all threads is faster than MPI ranks: on a 3-D sphere with 73 k leaves, 200 steps took 4.6 s on 16 threads against 10 s on 8 ranks (the remesh is 7 % of the first and 28 % of the second). Use MPI to go beyond one node, not within it.

## Adaptive meshes

The cost of an adaptive run against the uniform solver on the same finest mesh, its scaling on threads and ranks and the split of its time
(the step, the remesh, the fetch) are measured on the [AMR]({{root}}modules/AMR.html#cost-and-scaling) page.
