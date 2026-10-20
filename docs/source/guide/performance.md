---
title: Performance and parallelism
nav: Performance
section: Guide
order: 30
---

Every kernel is native C++ and runs at full speed from the first step. There is no
compilation threshold and no JIT warm-up. Three levels of parallelism combine:

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

Time-loop wall time on a 16-thread workstation, measured on 2026-10-20 at the C++ port.
The Python figures used its Numba kernels, which switched on automatically above 25 000
cells, and exclude JIT compilation.

| Case | Steps | Python | C++ 1 thread | C++ 16 threads | MPI 4×1 | Speed-up |
|---|---:|---:|---:|---:|---:|---:|
| IBM cylinder 400×200 | 20 | 2.18 s | 0.734 s | 0.136 s | 0.202 s | 16× |
| Supersonic IBM 400×400 | 20 | 2.76 s | 0.843 s | 0.191 s | 0.222 s | 14× |
| Shu-Osher 4000×2 | 200 | 2.49 s | 0.455 s | 0.206 s | 0.122 s | 12× |
| Lid cavity RK4 128×128 | 20 | 0.97 s | 0.260 s | 0.042 s | 0.072 s | 23× |
| AMR Sod, max_level 3 | 13 | 3.67 s | 0.113 s | 0.024 s | – | 153× |

<figure>
<img src="{{root}}img/cpp_speedup.png" style="width:100%">
<figcaption>Speed-up of the 16-thread C++ time loop over the Python solver.</figcaption>
</figure>

AMR gains the most: the Python version spent most of its time in per-patch bookkeeping
(neighbour searches, ghost filling, small NumPy calls), which native code removes
entirely.

## Practical notes

* The final line of a run reports the **time-loop** duration. VTK export is ASCII, which
  keeps it readable by the validation scripts; on very large meshes a high
  `save_interval` keeps output from dominating the run time.
* `max_steps = N` stops a run after N steps, which is useful for timing.
* For low-Mach steady problems, `BDF2 Dual Time` with a convective $\Delta t_{\rm phys}$
  avoids the acoustic CFL limit; see [TimeScheme]({{root}}modules/TimeScheme.html).
