---
title: Testing
nav: Testing
section: Verification
order: 10
---

> **In short.** Two commands check that a change did not move the numerics: the unit tests and the regression against stored reference
> states. They are the first thing to run after touching the solver. Whether the solver is *right*, rather than unchanged, is the subject of
> [Validation]({{root}}guide/validation.html).

```powershell
.\cpp\build.ps1 -Test                          # build + unit tests
python cpp\tests\regression.py --mpi 4          # regression against the reference states
```

Run both after any change to the solver, or all the checks of CI in one command:

```powershell
python cases\validation\check_all.py            # unit tests, regression (serial, 4 ranks), AMR on 1-7 ranks, ghost-cell IBM, MMS, switch table
python cases\validation\check_all.py --full     # + the unit tests without SIMD, the replicated-storage variant of the AMR check
python cases\validation\check_all.py --golden OLD   # + golden dumps of this build compared bit for bit with the dump OLD
```

CI runs the AMR rank comparison on 1-4 ranks at every push (`amr_mpi.py --check --ranks 4`); the full 1-7 ranks (and the restarts
through 5-7 ranks) run weekly in `benchmarks.yml` and in `check_all.py`.

Each step is judged by its exit code and logs to `results/validation/check_all/`; the run stops at the first failure
(`--keep-going` runs the rest). OLD is made before the change with `cases/validation/amr/golden.py dump OLD --exe <old fenice>`;
`--full` also dumps with `FENICE_NO_SIMD=1` and `FENICE_FULL_REBUILD=1`, which must give the same states.

CI compiles with warnings as errors (`-DFV_WERROR=ON`, or `.\cpp\build.ps1 -Werror`) and runs the unit tests once more in a build
with the address and undefined-behaviour checks of the compiler (GCC or Clang):

```bash
cmake --preset sanitize && cmake --build --preset sanitize --target fenice_tests
./build/sanitize/fenice_tests                   # stops at the first invalid read or write, use after free or signed overflow
```

These checks see what a comparison of bits cannot: a read past a halo that happens to hold the right value.

## Unit tests (`fenice_tests`)

The tests in `cpp/tests/` use a self-contained harness (`test_common.hpp`), so no test framework is needed, and the runner
exits non-zero on failure. They are split by area: `test_core.cpp` (EOS, limiters, Riemann solvers, boundaries, time schemes, IBM,
mapping, I/O), `test_periodic.cpp` (periodic and 3-D flows), `test_casefile.cpp`, and `test_amr_mesh.cpp`, `test_amr_physics.cpp`,
`test_amr_remesh.cpp`, `test_amr_distributed.cpp` for the AMR (the helpers they share are in `test_amr_common.hpp`). With no
argument `fenice_tests` runs all 208 tests; `fenice_tests --list` prints their names and `fenice_tests NAME ..` runs those whose name
contains one of the arguments (a filter that matches nothing exits with code 2). They cover:

<details markdown="1">
<summary>What the unit tests check, area by area</summary>

| Area | What is checked |
|---|---|
| EOS | Round trips, floors, and the stiffened gas reducing to the ideal gas |
| Limiters | TVD bounds, symmetry, and Minmod being the most diffusive |
| Riemann solvers | Consistency, supersonic upwinding, mirror symmetry, Low-Mach = HLLC at Mach > 1, stiffened gas |
| Boundary conditions | Every BC type's ghost state, padded arrays, periodicity |
| Free-stream preservation | Every Riemann-solver / MUSCL / viscous combination |
| Sod | Positivity and mass conservation over 50 steps |
| MUSCL, gradients, viscous terms | MUSCL is exact on linear fields; gradients are exact on linear fields; Couette flow is steady |
| Time schemes | Order of accuracy, stage counts, CFL clamping |
| IBM | Body shapes, ghost cells and image points, the wall condition, deep-solid reset |
| I/O | Solution mapping, `.npz` round trips, checkpoint and hot-start edge cases, VTK / VTM / PVD layout and sanitisation |

</details>

The areas with more to say:

* **Source terms and the viscous time step.** One Euler step on a uniform state moves every cell by
  $\Delta t\,S(x, y, z)$ (expression parsing, state slots, cell-centre positions, 2-D and 3-D); a source that uses
  `t` is evaluated at each scheme's own stage times (Euler, trapezoid and Simpson checked against a cubic in $t$
  to $10^{-14}$, started from $t_0 \ne 0$) and `t` is detected as a whole identifier; sources are rejected
  with the IBM, and a Taylor-Green vortex at $\nu = 0.1$ and $0.2$ stays accurate at the default CFL
  (the convective and viscous limits share one budget, see [TimeScheme]({{root}}modules/TimeScheme.html)).
* **AMR.** Leaf mesh topology and 2:1 balance, the face list, the level-0 mesh equal to the uniform
  solver bit for bit, **conservation across level jumps** by flux replacement (with a negative control without
  it), the order of the jump closure on the manufactured solution, the indicator against a plain
  structured implementation, deterministic and idempotent remeshing, the body band, restart and
  hot start, and the MPI partition: an in-process emulation of 1 to 7 ranks with the state a rank does not hold set
  to NaN must reproduce the serial run bit for bit.
* **Periodic flows.**
  * Exact translation invariance on a periodic domain (MUSCL and viscous on and off).
  * [Taylor-Green]({{root}}cases/taylor-green.html) decay: the physical viscosity is
    recovered to 2 %, and the velocity error converges at 2nd order.
  * [Isentropic-vortex]({{root}}cases/isentropic-vortex.html) 2nd-order convergence.
* **Case files.** Every parameter has a section; parsing of every value type; the error
  messages (misspelt or misplaced keys, unknown sections and shapes, bad expressions,
  missing states, TOML syntax); regions (box, halfspace, sphere, cylinder, override order)
  and expressions checked exactly against hand-written loops; a profile with a region
  leaving the other cells bit-identical; save/load round trip; the shipped cases and
  `save_case`.
* **3-D.**
  * *Extrusion:* Sod (MUSCL), cavity, Poiseuille, the IBM cylinder and the isentropic
    vortex, run on `nz = 3` with periodic z, reproduce the 2-D solution in every layer
    (≤ 5e-14) with $w$ exactly zero.
  * *Rotated planes:* the 2-D Taylor-Green vortex placed in the x-z and y-z planes
    reproduces the x-y run (5e-15).
  * *Axis permutation:* relabelling x → y → z of a fully 3-D field relabels the solution
    (5e-15).
  * *Translation invariance* on a triply periodic box, bit for bit.
  * *Free stream* preserved by every Riemann solver, with and without viscosity.
  * *Taylor-Green 3-D:* the initial dissipation matches the exact $0.75\,\nu$ (0.5 %),
    and the default 32³ case decays monotonically (with a
    $\min(\Delta x,\Delta y,\Delta z)$ CFL length it would diverge after ~30 steps).

## Regression references (`regression.py`)

`cpp/tests/reference/` stores final states for 69 short configurations. Sixteen were
produced by an independent implementation of the same schemes. Each group is backed as follows:

| Group | Configurations | Backed by |
|---|---|---|
| Taylor-Green, isentropic vortex | | Validated against their exact solutions |
| 3-D (four) | Taylor-Green 3-D, an x-decomposed thin box, a no-slip cavity, Sod between slip walls in z | The 3-D unit tests above |
| Non-reflecting outlet (three) | 2-D vortex in Y- and X-decomposition, 3-D Taylor-Green with non-reflecting x and z sides | The `nonreflecting_outlet` test and the reflection measurements on the [Boundaries]({{root}}modules/Boundaries.html) page |
| Immersed wall (four) | No-slip cylinder, slip Mach 2 cylinder, NACA 0012 at 10°, a 3-D sphere | The `airfoil_geometry` test checks the airfoil geometry |
| AMR (thirteen) | First order and MUSCL Sod, an SSP-RK2, an SSP-RK3 and a WENO3-Z square, a reflecting-wall box and a WENO3-Z one with second-order boundary faces, a lid-driven cavity with second-order boundary faces, a hot start from a uniform and one from an AMR checkpoint, the NACA 0012 airfoil with the body at the finest level, a cylinder with a momentum source | Generated by the leaf engine, after a check against uniform runs at the finest level; also run on MPI ranks, which must give the same state |
| 3-D adaptive (four, octree leaves) | The Sod tube along x with three levels, the viscous manufactured solution with level jumps, Taylor-Green with a forced box that is off-axis in every direction, a sphere with the ghost-cell wall in its body band | The octree is checked against the uniform 3-D solver at level 0 and against the oracles of the unit tests; also run on MPI ranks |
| Manufactured solutions (seven) | 2-D Navier-Stokes, 2-D Euler on x-strips, 3-D Navier-Stokes, with a moving and a slip wall, with a no-slip wall and an outflow on x-strips, and the unsteady Euler solution with SSP-RK3 and with SSP-RK2 | The [order study]({{root}}cases/manufactured-solution.html); also run on MPI strips along x and y, so they cover the source terms under decomposition |
| Discontinuous benchmarks (five) | Short versions of Woodward-Colella blast waves, Kelvin-Helmholtz, Noh implosion, two 2-D Riemann configurations | They pin the limiter, Riemann solver and `shock_blend` path on strong shocks; also run on MPI ranks |

Together they cover:

* every Riemann solver, limiter and time scheme;
* stiffened gas, gravity, and periodic / inlet / outlet BCs;
* viscous flow and IBM;
* hot start;
* AMR with mixed refinement levels and derefinement;
* 3-D periodic, wall and no-slip front/back sides, 3-D IBM;
* the non-reflecting outlet, including its derivatives along sides split between MPI ranks.

`regression.py` runs the C++ solver on each configuration through the normal command
line and compares the final state cell by cell (tolerance 1e-8 relative). With `--mpi N`,
the 19 MPI-capable cases (6 of them 3-D) also run on N ranks, in both Y- and X-decomposition and with
periodic wrap-around. Every case matches bit for bit, except the mass-flow cases at
~1e-14 (sums associated differently).

`--create-missing` writes references for newly added cases only and never overwrites an
existing one.

A deliberate change to the numerics changes these states. Re-validate the solver
physically (`cases/validation/`) before regenerating the references.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request on GitHub Actions, except for changes that cannot move
the numerics: a push or pull request that touches only `docs/`, Markdown files, `LICENSE` or `.gitignore` does not start it.
Any other commit can skip it by putting `[skip ci]` (or `[ci skip]`) in the commit message. Use it for changes to figures,
plotting code, comments and the like; never for a change to the solver, the schemes, the case files or the references. On Linux
x64 (GCC), Linux ARM64 (GCC), macOS on Apple Silicon (Apple clang with Homebrew `libomp`)
and Windows (MSVC through `build.ps1`, MS-MPI), it builds the solver with OpenMP and MPI,
runs `fenice_tests` and the regression suite serially and on 4 MPI ranks. Every platform
must reproduce the same reference states. The hosted runners have 3 or 4 cores, so the
MPI runs oversubscribe them (Open MPI environment variables in the workflow).
Each platform then runs the quick mode of the [manufactured-solution study]({{root}}cases/manufactured-solution.html)
(`verification/mms.py`, needs sympy and matplotlib), which fails the job if an observed order drops below its band: the
bitwise references cannot see a loss of accuracy that is reproduced exactly.

The physical validation studies (exact solutions, grid convergence, benchmarks against data, fault injection and the mutation study)
are on the [Validation]({{root}}guide/validation.html) page.
