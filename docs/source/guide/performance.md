---
title: Performance and parallelism
nav: Performance
section: Guide
order: 30
---

Every kernel is native C++. Three levels of parallelism combine:

| Level | Mechanism | Used for |
|---|---|---|
| Vector | AVX2, four faces or cells at a time (x86-64) | Riemann solvers, the MUSCL reconstruction of the interior faces of AMR meshes, the viscous cell quantities and viscous time-step bound of AMR meshes |
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
On AMR meshes the viscous cell quantities (velocities, temperature and the gradients from the four neighbours) and the viscous
time-step bound also run four leaves at a time, for runs of consecutive leaves whose neighbours are leaves (a leaf next to a virtual
cell or a physical edge takes the scalar path). They read the neighbours with scalar loads, not gather instructions, which are slow on
many CPUs. On the 720×360 airfoil (1 thread, 4-core Linux container, gcc) they went from 0.93 to 0.57 s and from 0.57 to 0.25 s of the
time loop. The viscous face flux stays scalar: a bit-identical vector version was slower than the scalar loop (1.1 to 1.25 s against
0.88 s), because the kernel is dominated by loads and has only three divisions per face to vectorise.

**Loop parallelism.** Loops shorter than `kOmpMinWork` (4096 items) run on one thread,
because OpenMP fork/join would cost more than the work. All reductions, such as the
maximum wave speed and the BDF2 residual norm, are order-independent maxima. Results are
therefore **bit-identical for any thread count**.

**AMR.** A level advances with the other levels frozen, so the face and leaf loops of a level are
ordinary OpenMP loops (and, like the others, serial below `kOmpMinWork`). The result does not depend on
the thread count.

**AMR on MPI.** Under MPI the leaves of each level are distributed over the ranks along a Morton
(Z-order) curve. The result is identical to the serial one for any number of ranks, but the speed-up is
limited: the remesh (indicator, adaptation, update of the face list and stencils) is repeated on every rank, and every
rank stores the whole mesh. The update after a remesh is incremental: only the leaves the adaptation touched and their face
neighbours are recomputed, the rest is copied. OpenMP threads are faster on one node; see [AMR]({{root}}modules/AMR.html) for the
timings.

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
| Lid cavity RK4 128×128 | 20 | 0.157 s | 0.031 s | 0.043 s | 5.1× |

The Shu-Osher mesh is small (8 000 cells), so thread fork/join overhead weighs more and it
scales least with threads; MPI strips suit it better.

On a MacBook Air (Apple M4, 10 threads), which runs the scalar kernels, supersonic IBM 400×400 takes
0.138 s on 10 OpenMP threads, and the IBM cylinder 400×200 0.125 s on 4 MPI ranks.

## Practical notes

* The final line of a run reports the **time-loop** duration. Snapshots are binary VTK
  by default (`vtk_format`); on very large meshes a high `save_interval` still keeps
  output from dominating the run time.
* `max_steps = N` stops a run after N steps, which is useful for timing.
* For low-Mach steady problems, `BDF2 Dual Time` with a convective $\Delta t_{\rm phys}$
  avoids the acoustic CFL limit; see [TimeScheme]({{root}}modules/TimeScheme.html).
