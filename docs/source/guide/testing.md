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

`cpp/tests/reference/` stores final states for 31 short configurations. Sixteen were
produced by an independent implementation of the same schemes. The Taylor-Green and
isentropic-vortex references are validated against their exact solutions; the five
3-D references (Taylor-Green 3-D, an x-decomposed thin box, a no-slip cavity, Sod between
slip walls in z, and a viscous IBM sphere) are backed by the 3-D unit tests above; and the
three non-reflecting-outlet references (2-D vortex in Y- and X-decomposition, 3-D
Taylor-Green with non-reflecting x and z sides) by the `nonreflecting_outlet` test and the
reflection measurements on the [Boundaries]({{root}}modules/Boundaries.html) page; the IBM
airfoil reference (NACA 0012 at 10°) by the `airfoil_geometry` test; the AMR airfoil
reference (tiled root, body at the finest level) by `amr_tiled_matches_uniform_steps`, and all six AMR
references are also run on 2 to 7 MPI ranks, which must give the same state (one of them has only two
root patches, so most ranks own nothing at the coarse levels).
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
physically (`cases/validation/`) before regenerating the references.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request on GitHub Actions. On Linux
x64 (GCC), Linux ARM64 (GCC), macOS on Apple Silicon (Apple clang with Homebrew `libomp`)
and Windows (MSVC through `build.ps1`, MS-MPI), it builds the solver with OpenMP and MPI,
runs `fenice_tests` and the regression suite serially and on 4 MPI ranks. Every platform
must reproduce the same reference states. The hosted runners have 3 or 4 cores, so the
MPI runs oversubscribe them (Open MPI environment variables in the workflow).

## Physical validation (`cases/validation/`)

The vocabulary follows Roache (2009). *Verification* asks whether the equations are solved
correctly: exact solutions, order of accuracy, and the grid convergence of a result. *Validation*
asks whether the right equations are solved: comparison with reference data. The studies are
grouped the same way, one folder per group, so that the folder says what a study can establish.

Each script is one study: it runs the solver into `results/validation/<study>/`, compares
the result with an analytical or reference solution, prints the errors and saves its figures
there. Runs that already finished are reused, so a second call only redraws the figures
(`--force` re-runs them). Every script takes `--docs`, which also copies the figures into
`docs/img/`: the figures on the [test-case pages]({{root}}modules/TestCases.html) are made
this way. The numerics shared by the studies (error norms, observed order, convergence fits)
and the helpers that run and cache the solver are in the `fenice` Python package
(`python/fenice/`).

```powershell
python cases\validation\run_all.py                          # the quick studies of every group (minutes)
python cases\validation\run_all.py --all --docs             # every study, refresh docs/img (long MPI runs)
python cases\validation\run_all.py verification             # one group
python cases\validation\run_all.py --list                   # groups, studies, quick or long
python cases\validation\verification\sod.py --force          # one study, re-running its simulations
```

**`verification/`: is the code right?** An exact solution exists, so the error and the order of
accuracy are measured, not estimated.

| Script | Compares |
|---|---|
| `sod.py` | Sod vs the exact Riemann solution; 1st vs 2nd order; the four limiters |
| `isentropic_vortex.py` | isentropic-vortex convergence and limiter comparison |
| `taylor_green.py` | Taylor-Green kinetic-energy decay and grid convergence |
| `taylor_green_animation.py` | animation of the Taylor-Green vorticity surface decaying, against the exact solution |
| `poiseuille.py` | Poiseuille parabola, grid convergence; serial vs 4 MPI ranks |

**`benchmarks/`: is the answer right?** The reference is a literature table, a much finer
solution or a physical quantity without a closed form, so the comparison carries the
uncertainty of the reference too.

| Script | Compares |
|---|---|
| `shu_osher.py` | Shu-Osher density, MC vs Minmod, convergence against a 6400-cell reference |
| `cavity.py` | cavity centrelines and vortex vs Ghia (1982), 32²-128², both orders |
| `taylor_green_3d.py` | 3-D Taylor-Green energy, dissipation, vorticity, Q-criterion animation |
| `gci.py` | observed order, Grid Convergence Index and Richardson extrapolation of the cavity, airfoil and 3-D Taylor-Green results |
| `ibm_loads.py cylinder` / `airfoil` | IBM loads from the monitor, Strouhal number |

**`amr/`: does the adaptive mesh keep the answer and save the cost?**

| Script | Compares |
|---|---|
| `sod_amr.py` | AMR vs uniform meshes at the two finest levels (cells and L1 error) |
| `sod_amr_animation.py` | animation of the AMR patches following the waves of the Sod tube |
| `airfoil_amr.py` | AMR airfoil against uniform 192×96 and 768×384 meshes (cells, time, rms error of Cd and Cl) |
| `amr_mpi.py` | AMR airfoil on OpenMP threads and MPI ranks: states identical to serial, time and cells per rank |

**`features/`: demonstrations of solver features.**

| Script | Shows |
|---|---|
| `cylinder_mesh_transition.py` | cylinder loads across a 400×200 → 800×400 hot start (animation) |

### Grid convergence of the benchmarks (`benchmarks/gci.py`)

A benchmark without an exact solution gets an error band from a systematic grid refinement,
following Roache (2009, chapter 5). For three solutions $f_1, f_2, f_3$ (fine to coarse) on grids of
size $h_1 < h_2 < h_3$ the script computes

* the **apparent order** $p$ from $\varepsilon_{23}/(r_{23}^p - 1) = r_{12}^p\,\varepsilon_{12}/(r_{12}^p - 1)$, with
  $\varepsilon_{ij} = f_j - f_i$ (a closed form for a constant ratio, a damped iteration otherwise);
* the **fine-grid GCI**, $F_s\,|\varepsilon_{12}/f_1|/(r_{12}^p - 1)$, and the Richardson-extrapolated value
  $f_1 + (f_1 - f_2)/(r_{12}^p - 1)$;
* the **asymptotic-range check** $\mathrm{GCI}_{23}/(r_{12}^p\,\mathrm{GCI}_{12})$, which is 1 in the asymptotic range.

The factor of safety is $F_s = 1.25$ when the three grids give an apparent order within 25 % of the formal
one and 3 otherwise; an order above the formal one is replaced by the formal one. A triplet whose
differences do not shrink monotonically, or whose apparent order is not positive, gets no GCI: such grids are not
in the asymptotic range, and a number would be misleading. The refinement ratio is the same in every direction, the
loads are averaged over the same time window on every grid, and the time step shrinks with the mesh at a fixed CFL, so
the study measures the combined space-time convergence.

<figure>
<img src="{{root}}img/gci.png" style="width:100%">
<figcaption>Value against grid size for each functional of the finest triplet: the solver, the
Richardson-extrapolated value, the GCI band on the finest grid and the reference where there is one.
The star is the AMR airfoil run at its effective grid size (indicative only: changing thresholds is not a
systematic refinement).</figcaption>
</figure>

| study | quantity | grids (coarse to fine) | $p_{obs}$ | $F_s$ | fine-grid GCI | extrapolated | asymptotic ratio | reference |
|---|---|---|---:|---:|---:|---:|---:|---|
| cavity, MUSCL | $\psi_{\min}$ | -0.1058, -0.1123, -0.1136 | 2.24 | 1.25 | 0.5 % | -0.1141 | 1.18 | Ghia (1982) -0.1139; fine - ref = +2.70e-04, inside the GCI band |
| cavity, MUSCL | $u_{\min}$ on $x = 0.5$ | -0.2997, -0.3232, -0.3277 | 2.38 | 1.25 | 0.573 % | -0.3292 | 1.30 | Ghia (1982) -0.3273; fine - ref = -4.39e-04, inside the GCI band |
| cavity, MUSCL | $v_{\max}$ on $y = 0.5$ | 0.2725, 0.2975, 0.3025 | 2.30 | 1.25 | 0.7 % | 0.3042 | 1.23 | Ghia (1982) 0.302; fine - ref = +5.14e-04, inside the GCI band |
| cavity, MUSCL | $v_{\min}$ on $y = 0.5$ | -0.4073, -0.4447, -0.4519 | 2.38 | 1.25 | 0.663 % | -0.4543 | 1.30 | Ghia (1982) -0.4499; fine - ref = -1.94e-03, inside the GCI band |
| cavity, 1st order | $\psi_{\min}$ | -0.08475, -0.09796, -0.1056 | 0.79 | 1.25 | 12.5 % | -0.1161 | 1.00 | Ghia (1982) -0.1139; fine - ref = +8.30e-03, inside the GCI band |
| cavity, 1st order | $u_{\min}$ on $x = 0.5$ | -0.1982, -0.254, -0.2885 | 0.69 | 3 | 58.3 % | -0.3446 | 1.00 | Ghia (1982) -0.3273; fine - ref = +3.87e-02, inside the GCI band |
| cavity, 1st order | $v_{\max}$ on $y = 0.5$ | 0.1934, 0.2426, 0.2716 | 0.76 | 1.25 | 19.2 % | 0.3133 | 1.00 | Ghia (1982) 0.302; fine - ref = -3.04e-02, inside the GCI band |
| cavity, 1st order | $v_{\min}$ on $y = 0.5$ | -0.3193, -0.3829, -0.4177 | 0.87 | 1.25 | 12.6 % | -0.4597 | 1.00 | Ghia (1982) -0.4499; fine - ref = +3.22e-02, inside the GCI band |
| IBM airfoil, Re = 1000 | $\overline{C_d}$, $2 \leq t \leq 6$ | 0.1492, 0.1788, 0.1712 | - | 3 | - | - | - | not monotone: no observed order, no GCI (more or finer grids are needed) |
| IBM airfoil, Re = 1000 | $\overline{C_l}$, $2 \leq t \leq 6$ | 0.1094, 0.292, 0.3607 | 1.41 | 3 | 34.4 % | 0.402 | 1.00 | observed order 1.41 far from the formal 2: Fs = 3 |
| 3-D Taylor-Green, Re = 1600 | $E_k(20)/E_k(0)$ | 0.2312, 0.2064, 0.1883 | -0.41 | 3 | - | - | - | apparent order -0.41 <= 0: the solutions do not converge on these grids, no GCI |
| 3-D Taylor-Green, Re = 1600 | peak $\varepsilon$ | 0.01102, 0.01132, 0.01248 | -3.69 | 3 | - | - | - | DNS 512$^3$ 0.0127; fine - ref = -2.17e-04, no GCI band |

What the table says:

* **Cavity, second order.** The apparent orders (2.2 to 2.4) are close to the formal 2, so $F_s = 1.25$ and the
  fine-grid GCI is 0.5 to 0.7 %. All four of Ghia's values lie inside the band of the 128² solution, so the
  remaining difference to the benchmark is covered by the numerical uncertainty (Ghia's tables are samples of
  the extrema, so the centreline values are compared with the extremum of a local parabola through the solver's
  profile). The asymptotic ratio of 1.2 to 1.3 says the 32²-64²-128² triplet is close to, not exactly in, the
  asymptotic range.
* **Cavity, first order.** The apparent orders are 0.7 to 0.9 and the bands 12 to 58 %: the first-order scheme
  needs far finer grids, which is why the second-order scheme is the default.
* **IBM airfoil.** The mean drag is not monotone over 192×96, 384×192, 768×384 and the mean lift converges at
  order 1.4 with a 34 % band: a staircase body on an impulsive start is not in the asymptotic range on these
  grids, and the table says so. The AMR run lies between the two finest uniform meshes.
* **3-D Taylor-Green.** The energy at $t = 20$ and the peak dissipation do not converge on 32³, 64³ and 96³
  (negative apparent orders): the vortex is under-resolved, the peak rises towards the DNS value 0.0127
  (0.0125 at 96³) but no band can be given.

### References

* Roache, P. J. (2009). *Fundamentals of Verification and Validation*. Hermosa Publishers,
  Socorro, New Mexico. ISBN 978-0-913478-12-7.
