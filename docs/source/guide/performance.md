---
title: Performance and parallelism
nav: Performance
section: Guide
order: 30
---

Every kernel is native C++. Three levels of parallelism combine:

| Level | Mechanism | Used for |
|---|---|---|
| Loop | OpenMP `parallel for` over faces / cells | Riemann fluxes, MUSCL, BCs, gradients, viscous terms, time-scheme updates |
| AMR faces | OpenMP over the faces and leaves of one refinement level | AMR runs |
| Domain | MPI strips with halo exchange | large uniform meshes, multiple nodes |
| AMR leaves | MPI over the leaves of each level, cut along a Morton curve | AMR runs (correct on any rank count; see the limits below) |

**Loop parallelism.** Loops shorter than `kOmpMinWork` (4096 items) run on one thread,
because OpenMP fork/join would cost more than the work. All reductions, such as the
maximum wave speed and the BDF2 residual norm, are order-independent maxima. Results are
therefore **bit-identical for any thread count**.

**AMR.** A level advances with the other levels frozen, so the face and leaf loops of a level are
ordinary OpenMP loops (and, like the others, serial below `kOmpMinWork`). The result does not depend on
the thread count.

**AMR on MPI.** Under MPI the leaves of each level are distributed over the ranks along a Morton
(Z-order) curve. The result is identical to the serial one for any number of ranks, but the speed-up is
limited: the remesh (indicator, adaptation, rebuild of the face list) is repeated on every rank, and every
rank stores the whole mesh. OpenMP threads are faster on one node; see [AMR]({{root}}modules/AMR.html) for the
timings.

**MPI.** Each rank runs one OpenMP thread by default, to avoid oversubscription. Pass
`--threads N` or set `OMP_NUM_THREADS` for hybrid MPI + OpenMP runs. The MPI result is
identical to the serial one (see [MPIDecomposition]({{root}}modules/MPIDecomposition.html)).

## Benchmark

Time-loop wall time on a 16-thread workstation.

| Case | Steps | 1 thread | 16 threads | MPI 4×1 | 16 threads vs 1 |
|---|---:|---:|---:|---:|---:|
| IBM cylinder 400×200 | 20 | 0.734 s | 0.136 s | 0.202 s | 5.4× |
| Supersonic IBM 400×400 | 20 | 0.843 s | 0.191 s | 0.222 s | 4.4× |
| Shu-Osher 4000×2 | 200 | 0.455 s | 0.206 s | 0.122 s | 2.2× |
| Lid cavity RK4 128×128 | 20 | 0.260 s | 0.042 s | 0.072 s | 6.2× |

The Shu-Osher mesh is small (8 000 cells), so thread fork/join overhead weighs more and it
scales least with threads; MPI strips suit it better.

On a MacBook Air (Apple M4, 10 threads) the same runs are faster: supersonic IBM 400×400
in 0.138 s on 10 OpenMP threads, and the IBM cylinder 400×200 in 0.125 s on 4 MPI ranks.

## Practical notes

* The final line of a run reports the **time-loop** duration. Snapshots are binary VTK
  by default (`vtk_format`); on very large meshes a high `save_interval` still keeps
  output from dominating the run time.
* `max_steps = N` stops a run after N steps, which is useful for timing.
* For low-Mach steady problems, `BDF2 Dual Time` with a convective $\Delta t_{\rm phys}$
  avoids the acoustic CFL limit; see [TimeScheme]({{root}}modules/TimeScheme.html).
