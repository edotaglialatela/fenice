---
title: Performance and parallelism
nav: Performance
section: Guide
order: 30
---

Every kernel is native C++. Three levels of parallelism combine:

| Level | Mechanism | Used for |
|---|---|---|
| Vector | AVX2, four faces or cells at a time (x86-64) | Riemann solvers, the MUSCL reconstruction of the interior faces of AMR meshes, the viscous cell quantities and time-step bound of AMR meshes |
| Loop | OpenMP `parallel for` over faces / cells | Riemann fluxes, MUSCL, BCs, gradients, viscous terms, time-scheme updates |
| AMR faces | OpenMP over the faces and leaves of one refinement level | AMR runs |
| Domain | MPI strips with halo exchange | large uniform meshes, multiple nodes |
| AMR leaves | MPI over the leaves of each level, cut along a Morton curve | AMR runs (correct on any rank count; see the limits below) |

**Vector kernels.** On x86-64 CPUs with AVX2 the Riemann solver (the three HLLC variants, ideal and stiffened gas) runs four faces
at a time, and on AMR meshes the MUSCL reconstruction of the interior faces is fused with it. Every lane performs the same
IEEE operations in the same order as the scalar code (no fused multiply-add, no vector min / max, every branch computed and
selected by a mask), so the results are **bit-identical** with and without it, on any CPU. The scalar code is used on other CPUs
and for the leftover faces of a batch; `FENICE_NO_SIMD=1` switches the vector kernels off (to time them, or to check the bits).
The Riemann solver costs 46 / 80 / 60 ns per face (HLLC / Low-Mach / Preconditioned) as scalar code and 19 / 25 / 22 ns vectorised
(`bench_riemann`).
The viscous cell quantities (velocities, temperature and the gradients from the four neighbours) and the viscous time-step bound of
AMR meshes also run four leaves at a time, for runs of consecutive leaves whose neighbours are leaves (the others take the scalar
path). On the 720×360 airfoil (one thread) they save about 0.5 s of stepping: 0.54 s on a Ryzen 7 7800X3D with MSVC, 0.53 s there
with gcc, 0.48 s on a Threadripper PRO 5965WX (gcc). The file of the vector kernels is compiled for AVX2 (`/arch:AVX2` with MSVC,
`-mavx2` with gcc / clang; FMA stays off): without `/arch:AVX2` the MSVC build of these kernels was 0.42 s *slower* than the scalar
loops. The CPU check that decides whether they run is compiled without AVX, so the program still starts on older CPUs. The viscous
face flux stays scalar: its vector version was slower than the scalar loop (1.1 to 1.25 s against 0.88 s), the kernel being dominated
by scattered loads with three divisions per face.

**Fused passes.** On a uniform mesh the residual is computed in one pass over rows of faces: the MUSCL reconstruction (or the
cell states, first order), the Riemann solver and, in 2-D, the flux balance of the cells run on a row at a time, so the face fluxes
and the left / right states never go to memory. `FENICE_NO_FUSED=1` takes the
separate passes; the two give the same bits (`fused_residual_passes_equal_the_two_pass_residual_bit_for_bit`). The 2-D viscous
residual is one sweep over rows as well: the velocity, temperature and their derivatives of a row and of the row above stay in a
small per-thread buffer, a cell's primitive state is converted with its row, and the padded state of the diffusive ghosts is never
built, only its ghost ring (`get_padded_frame`, a few entries per row and column; the interior is read from the cell field). On
large meshes, where the arrays do not fit in cache, this matters: viscous MUSCL RK2 on 2048^2 takes 114 / 24 / 22 ns per
cell-stage on 1 / 8 / 16 threads, about 1.2x less than with a padded and a primitive copy on 8-16 threads (1.03x on one); a 512^2
mesh, which stays in cache, takes the same time either way. Passes that only copy or convert a whole array run at the same
rate on one thread and on sixteen, so they limit the thread scaling of a large mesh; that is why removing them pays most on many
threads. The primitive
variables share one reciprocal of the density and the divergence multiplies by the reciprocal of the cell area, so the divisions
per face are few. Measured on a Ryzen 7 7800X3D (ns per cell-stage, 1 / 16 threads): Sod MUSCL 2000x200 54 / 11, first order
41 / 9, Kelvin-Helmholtz 512^2 58 / 13, viscous Taylor-Green 512^2 112 / 22, 3-D Taylor-Green 64^3 250 / 61.

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

## Benchmark

Time-loop wall time on a 16-thread workstation.

| Case | Steps | 1 thread | 16 threads | MPI 4×1 | 16 threads vs 1 |
|---|---:|---:|---:|---:|---:|
| IBM cylinder 400×200 | 20 | 0.529 s | 0.127 s | 0.151 s | 4.2× |
| Supersonic IBM 400×400 | 20 | 0.667 s | 0.173 s | 0.185 s | 3.9× |
| Shu-Osher 4000×2 | 200 | 0.367 s | 0.248 s | 0.097 s | 1.5× |
| Lid cavity SSP-RK3 128×128 | 20 | 0.095 s | 0.029 s | 0.029 s | 3.3× |

The Shu-Osher mesh is small (8 000 cells), so thread fork/join overhead weighs more and it
scales least with threads; MPI strips suit it better.

On a MacBook Air (Apple M4, 10 threads), which runs the scalar kernels, supersonic IBM 400×400 takes
0.138 s on 10 OpenMP threads, and the IBM cylinder 400×200 0.125 s on 4 MPI ranks.

## Practical notes

* The final line of a run reports the **time-loop** duration. Snapshots are binary VTK
  by default (`vtk_format`); on very large meshes a high `save_interval` still keeps
  output from dominating the run time.
* `max_steps = N` stops a run after N steps, which is useful for timing.

**Where the time goes.** `FENICE_PROFILE=1` prints the total time step and residual sections on uniform runs, including the time updates, viscous padding and conversion, viscous sweep, IBM forcing, body force and sources. It also prints the output and remesh sections on AMR runs (the region the indicator reads, the flags, the topology, the migration of the state, the rebuild of the share and its engine tables, the partition; inside the rebuild the fetch of the leaves, the mask, the partial mesh, the engine build, the partition and the localisation), with the minimum and maximum over the ranks. Sections are timed once per step or remesh, never per face. Nested sections are inclusive, so the total and its children must not be summed together. With the switch off, profile sections only take the cached flag check.
