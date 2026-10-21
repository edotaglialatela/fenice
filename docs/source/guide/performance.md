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
| AMR patch | OpenMP over the patches of one refinement level | AMR runs |
| Domain | MPI strips with halo exchange | large uniform meshes, multiple nodes |

**Loop parallelism.** Loops shorter than `kOmpMinWork` (4096 items) run on one thread,
because OpenMP fork/join would cost more than the work. All reductions, such as the
maximum wave speed and the BDF2 residual norm, are order-independent maxima. Results are
therefore **bit-identical for any thread count**.

**AMR.** Within one level every patch reads the frozen neighbour snapshot, so the patch
updates are independent. They run concurrently, with one time-scheme clone (scratch
buffers) per thread. The OpenMP loops inside a patch are serial because they fall below
`kOmpMinWork`.

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
| AMR Sod, max_level 3 | 13 | 0.113 s | 0.024 s | – | 4.7× |

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
