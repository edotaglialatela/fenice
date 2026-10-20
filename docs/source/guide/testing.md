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

## Unit tests (`fv_tests`)

`cpp/tests/test_main.cpp` uses a self-contained harness, so no test framework is needed,
and exits non-zero on failure. Its 49 tests cover:

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
    and the default 32³ case decays monotonically past the step where a
    $\min(\Delta x,\Delta y,\Delta z)$ CFL length made it diverge.

## Regression references (`regression.py`)

`cpp/tests/reference/` stores final states for 24 short configurations. Sixteen are
states of the original Python solver (the reference implementation, retired on
2026-10-20). The Taylor-Green and isentropic-vortex references were added later from the
C++ solver, after validation against their exact solutions. The five 3-D references
(Taylor-Green 3-D, an x-decomposed thin box, a no-slip cavity, Sod between slip walls in
z, and a viscous IBM sphere) were added after the 3-D unit tests above. Together they cover:

* every Riemann solver, limiter and time scheme;
* stiffened gas, gravity, and periodic / inlet / outlet BCs;
* viscous flow and IBM;
* hot start;
* AMR with mixed refinement levels and derefinement;
* 3-D periodic, wall and no-slip front/back sides, 3-D IBM.

`regression.py` runs the C++ solver on each configuration through the normal command
line and compares the final state cell by cell (tolerance 1e-8 relative). With `--mpi N`,
the 15 MPI-capable cases (5 of them 3-D) also run on N ranks, in both Y- and X-decomposition and with
periodic wrap-around. At retirement every case matched bit for bit, except the BDF2 /
mass-flow cases at ~1e-14.

`--create-missing` writes references for newly added cases only and never overwrites an
existing one.

A deliberate change to the numerics changes these states. Re-validate the solver
physically (`validation/`) before regenerating the references.

## Physical validation (`validation/`)

The scripts read the VTK output of a run and compare it with analytical or reference
solutions. The figures on the [test-case pages]({{root}}modules/TestCases.html) were
produced this way:

| Script | Compares |
|---|---|
| `validationSod.py`, `comparisonSecondOrder.py` | Sod vs the exact Riemann solution; 1st vs 2nd order, limiters |
| `amrComparisonSod.py` | uniform vs AMR Sod (cells vs L2 error) |
| `validationPoiseuille.py`, `mpiComparison.py` | Poiseuille parabola; serial vs MPI |
| `validationDrivenCavity.py` | cavity profiles vs Ghia (1982) |
| `validationShuOsher.py` | Shu-Osher density |
| `aeroCoeffCylinder.py` | cylinder C_L / C_D and Strouhal number |
| `meshMappingAnimation.py` | hot-start mapping animation |
| `validationTaylorGreen.py` | Taylor-Green kinetic-energy decay and grid convergence (runs the solver) |
| `validationIsentropicVortex.py` | isentropic-vortex convergence and limiter comparison (runs the solver) |
| `validationTaylorGreen3D.py` | 3-D Taylor-Green energy, dissipation and vorticity at 32³ / 64³ (runs the solver) |
