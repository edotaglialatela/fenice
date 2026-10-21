---
title: Testing and validation
nav: Testing
section: Guide
order: 50
---

```powershell
.\cpp\build.ps1 -Test                          # build + unit tests
python cpp\tests\regression.py --mpi 4          # regression against the reference states
```

Run both after any change to the solver.

## Unit tests (`fenice_tests`)

`cpp/tests/test_main.cpp` uses a self-contained harness, so no test framework is needed,
and exits non-zero on failure. Its 55 tests cover:

* **EOS.** Round trips, floors, and the stiffened gas reducing to the ideal gas.
* **Limiters.** TVD bounds, symmetry, and Minmod being the most diffusive.
* **Riemann solvers.** Consistency, supersonic upwinding, mirror symmetry,
  Low-Mach = HLLC at Mach > 1, preconditioning with Ma_cut = 1 = HLLC, stiffened gas.
* **Boundary conditions.** Every BC type's ghost state, padded arrays, periodicity.
* **Free-stream preservation** for every Riemann-solver / MUSCL / viscous combination.
* **Sod.** Positivity and mass conservation over 50 steps.
* **MUSCL, gradients, viscous terms.** MUSCL is exact on linear fields; gradients are
  exact on linear fields; Couette flow is steady.
* **Time schemes.** Order of accuracy, stage counts, CFL clamping, BDF2 convergence and
  its iteration cap.
* **IBM.** Masks, forcing, fringe and deep-solid reset.
* **AMR.** Tree structure, conservative refine/derefine, 2:1 balance, the indicator,
  marking with buffer and anti-thrash guard, patch coupling, linear ghost prolongation,
  and **exact reflux conservation**, together with a negative control without
  refluxing.
* **I/O.** Solution mapping, `.npz` round trips, checkpoint and hot-start edge cases,
  VTK / VTM / PVD layout and sanitisation.
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

`cpp/tests/reference/` stores final states for 29 short configurations. Sixteen were
produced by an independent implementation of the same schemes. The Taylor-Green and
isentropic-vortex references are validated against their exact solutions; the five
3-D references (Taylor-Green 3-D, an x-decomposed thin box, a no-slip cavity, Sod between
slip walls in z, and a viscous IBM sphere) are backed by the 3-D unit tests above; and the
three non-reflecting-outlet references (2-D vortex in Y- and X-decomposition, 3-D
Taylor-Green with non-reflecting x and z sides) by the `nonreflecting_outlet` test and the
reflection measurements on the [Boundaries]({{root}}modules/Boundaries.html) page; the IBM
airfoil reference (NACA 0012 at 10°) by the `airfoil_geometry` test; the AMR airfoil
reference (tiled root, body at the finest level) by `amr_tiled_matches_uniform_steps`, and all four AMR
references are also run on 2 to 7 MPI ranks, which must give the same state.
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
periodic wrap-around. Every case matches bit for bit, except the BDF2 / mass-flow cases at
~1e-14 (sums associated differently).

`--create-missing` writes references for newly added cases only and never overwrites an
existing one.

A deliberate change to the numerics changes these states. Re-validate the solver
physically (`validation/`) before regenerating the references.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request on GitHub Actions. On Linux
x64 (GCC), Linux ARM64 (GCC), macOS on Apple Silicon (Apple clang with Homebrew `libomp`)
and Windows (MSVC through `build.ps1`, MS-MPI), it builds the solver with OpenMP and MPI,
runs `fenice_tests` and the regression suite serially and on 4 MPI ranks. Every platform
must reproduce the same reference states. The hosted runners have 3 or 4 cores, so the
MPI runs oversubscribe them (Open MPI environment variables in the workflow).

## Physical validation (`validation/`)

Each script is one study: it runs the solver into `results/validation/<study>/`, compares
the result with an analytical or reference solution, prints the errors and saves its figures
there. Runs that already finished are reused, so a second call only redraws the figures
(`--force` re-runs them). Every script takes `--docs`, which also copies the figures into
`docs/img/`: the figures on the [test-case pages]({{root}}modules/TestCases.html) are made
this way. The shared helpers are in the `fenice` Python package (`python/fenice/`).

```powershell
python validation\run_all.py                  # the quick studies (minutes)
python validation\run_all.py --all --docs     # every study, refresh docs/img (hours: long MPI runs)
python validation\sod.py --force              # one study, re-running its simulations
```

| Script | Compares |
|---|---|
| `sod.py` | Sod vs the exact Riemann solution; 1st vs 2nd order; the four limiters |
| `sod_amr.py` | AMR vs uniform meshes at the two finest levels (cells and L1 error) |
| `shu_osher.py` | Shu-Osher density, MC vs Minmod, convergence against a 6400-cell reference |
| `isentropic_vortex.py` | isentropic-vortex convergence and limiter comparison |
| `taylor_green.py` | Taylor-Green kinetic-energy decay and grid convergence |
| `poiseuille.py` | Poiseuille parabola, grid convergence; serial vs 4 MPI ranks |
| `cavity.py` | cavity centrelines and vortex vs Ghia (1982), 32²-128², both orders |
| `taylor_green_3d.py` | 3-D Taylor-Green energy, dissipation, vorticity, Q-criterion animation |
| `ibm_loads.py cylinder` / `airfoil` | IBM loads from the monitor, Strouhal number |
| `cylinder_mesh_transition.py` | cylinder loads across a 400×200 → 800×400 hot start (animation) |
