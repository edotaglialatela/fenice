# Future-proofing the code: what limits it, and what would be substantial

Status: draft proposal, nothing implemented, no decision taken. Written on 2026-11-11 after the scan behind
`performance-improvements.md`. Read in full: the time loops, the uniform kernels and their AVX2 twins, the AMR residual, advance,
prepare and indicator. Read as headers or in part only: the mesh and share builders, the MPI exchanges, IO, boundaries, case files, the
Python package, the GUI. What could not be verified without a build experiment is marked *to check*. The decisions of `repository-review.md` (no templating on
the dimension in the AMR kernels, no portability layer for accelerators) are taken as given.

## What is already solid

* Bit-reproducibility as a contract: 75 reference states, serial equal to MPI and to any thread count, scalar equal to AVX2, with a test
  for each invariant. It makes every refactor checkable by a hash, which is the best protection a research code can have.
* Four platforms in CI on every push (Linux x64 and ARM64, macOS, Windows), with the MPI and AMR equivalence checks.
* Cases, parameters and switches come from one table each, and the documentation build fails on a missing key.
* Two vendored dependencies and an own checkpoint writer: nothing external can break the build.

## Structural limits (where the code stops, not where it is slow)

1. **One-direction MPI strips for the uniform solver.** `MPIDecomposition` cuts along y, else x; z is never split and a strip needs two
   cells. On many ranks the strips become slabs that are mostly halo, and the rank count is capped by `ny / 2`. A block decomposition in
   two or three directions is the change that decides whether the uniform solver can use a cluster. It touches the halo layouts, the
   cross ghosts of the viscous terms, the IBM halos and the mass-flow average: each has a serial-equal-MPI test already.
2. **Everything is gathered on rank 0 to be written.** The uniform snapshot and checkpoint (`decomp.gather`), the AMR checkpoint
   (`collect(..., root_only)`, see `distributed-checkpoint-write.md`), the post-processing outputs. It bounds the problem size by the
   memory of one rank and adds a serial phase to every output.
3. **32-bit indices.** `Mesh::num_cells`, `num_faces` and `cell` are `int`, and so are the leaf and face indices of the AMR engine. A 3-D
   mesh above about 700 M cells overflows the face count; only the two ghost-table builders check the product (`Physics.cpp`,
   `CellFlux.cpp`); the parameters are checked one by one (`Params.cpp`), not as a product, and the setup has no guard. A single
   check at setup that refuses such a mesh would turn a silent overflow into a message. With a block decomposition only the
   global indices need 64 bits.
4. **The state has a fixed size of five.** `State = std::array<double, 5>` appears through every kernel (about 100 uses of `kNVar` and
   as many of the fifth slot or `kMom`). It blocks the four-slot 2-D state, and any model with more transported variables; the species
   proposal works around it with a separate per-cell array. If species or turbulence variables are on the roadmap, decide once whether
   the number of variables becomes a property of the build (or of the run) before those features each add their own side array.
5. **Three copies of the physics.** The uniform 2-D path, the uniform 3-D path and the AMR face list each carry the reconstruction, the
   viscous expressions and the forcing, most of them with a scalar and a vector twin. The shared kernels header keeps the expressions
   identical, but every new model (a limiter, a flux, a wall treatment) is written and tested three to six times. The level-0 AMR mesh
   already reproduces the uniform bits without a body; the long-term question is whether the uniform solver stays a separate engine or
   becomes the fast path of one engine. This is the largest lever on future effort and the most expensive to pull.
6. **Vector kernels for one instruction set.** AVX2 on x86-64; ARM64 and macOS run the scalar code, which is 2.7x slower in the
   convective kernel on this machine (167 against 62 ns per cell). A thin layer over the eight or so operations the kernels use
   (add, sub, mul, div, sqrt, compare, blend) would let one source serve AVX2, NEON and wider registers with the same bit contract.
7. **MSVC keeps OpenMP at 2.0** (signed indices, no `max` reductions, no tasks, no `simd`). It costs a workaround in every reduction.
   The newer mode of the compiler (`/openmp:llvm`) may lift it: *to check*, including that the results stay identical.

## Maintainability

* **Long functions with captured state.** `Simulation::run_mpi` and `Simulation::solve_amr` are about 330 lines each, built from
  lambdas that share a dozen locals; the three time loops (serial, MPI, AMR) repeat the monitor, output, stop-file and post-processing
  logic. A small loop driver with the three engines behind one interface would remove the repetition and make a fourth engine cheap.
* **Module-level state.** `thread_local` scratch arrays (nine), the stop flag, the profile. Fine for one simulation per process; it
  rules out two `Simulation` objects in one process (a coupled run, an in-process parameter sweep, a Python binding).
* **Error paths under MPI** are handled: an exception on a rank is printed and followed by `MPI_Abort` (`main.cpp`), so the other
  ranks do not wait in a collective. Every failing rank prints its own message; nothing to do.
* **No sanitizer or static-analysis job.** CI builds Release only. A Debug build with the address and undefined-behaviour checks on
  the unit tests (Linux) would catch what bit-equality cannot: a read past a halo that happens to hold the right value.
* **Warnings are not errors**, and MSVC runs at `/W3`.
* **The Python package has no test suite of its own** (the known-answer checks live in `cases/validation/check_analysis.py`), and it
  declares numpy only, while `fenice.study` imports matplotlib and `fenice.uq` imports scipy inside the functions that need them:
  optional dependencies that `pyproject.toml` does not name (`prerequisites.py` installs them).
* **Reference states are opaque.** A reference that moves is accepted or not by a person. Recording, next to each reference, the
  physical quantity that validated it (an error norm, a drag coefficient) would let a later regeneration be judged by a number.

## Substantial improvements, in the order I would consider them

| Item | Why | Size | Moves the bits |
|---|---|---|---|
| Guard on index overflow; sanitizer job in CI; warnings as errors | cheap protection | small | no |
| Clamp ranks x threads (see `performance-improvements.md`) | a usage trap | small | no |
| One loop driver for the three time loops | less repeated logic before anything else is added | medium | no |
| Distributed write of checkpoints and snapshots | removes the rank-0 memory bound | medium | no |
| Block decomposition of the uniform solver | the condition for any cluster run | large | no (serial equal to MPI is the test) |
| Number of variables as a parameter of the build | species, turbulence, four-slot 2-D | large | no, if the five-variable build is unchanged |
| Portable vector layer | ARM64 and wider registers from one source | medium | no |
| One engine instead of two | halves the cost of every later model | very large | to decide |

Not proposed: accelerators (decided), a second mesh type (`unstructured-solver.md` owns that question), new numerical methods.

## Open questions for the owner

1. Is a cluster a target? If not, items 1 to 3 of the structural limits can stay documented limits.
2. Are species or other transported variables going ahead? That decides item 4.
3. Is the uniform solver kept as its own engine for good? That decides item 5 and how much of the kernel work in
   `performance-improvements.md` is worth doing twice.
