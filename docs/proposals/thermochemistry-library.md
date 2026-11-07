# Thermochemistry from an external library: proposal

Status: proposal, not started; the choices below are the user's. It refines one choice of `postprocessing-and-species.md` ("where the
thermochemistry comes from") and does not replace that file: the species array, the target cases G1, G2 and T, the bands and the order
of work stay as written there. Nothing here is measured; the cost figures marked *to measure* are the first thing to settle. Statements
about the library are from its documentation as remembered and are to be checked against the version chosen.

## What the question is

Multi-species and reacting flow needs four things the solver does not have: mixture thermodynamics (gamma, `cp(T)`, temperature from the
internal energy, the sound speed), species and temperature dependent transport coefficients, a source term from a reaction mechanism,
and independent reference data to check them against. A general-purpose thermochemistry library (C++, open source, mechanism files in a
standard text format, thermodynamic polynomials, a stiff reactor integrator) provides all four. The question is **which of the four the
solver should take from it, at build time or at run time, and which it should keep inside its own kernels.**

## Constraints from the solver

* **The face and cell loops cannot call a library.** The equation of state is evaluated per face (Riemann solvers, AVX2 twins) and per
  cell (primitive variables, time step). A call through a general object API there would cost orders of magnitude more than the flux and
  would not vectorise. Whatever the thermodynamics is, it must be inline arithmetic on arrays: for a mixture of polynomial species, the
  mass-fraction-weighted polynomial and a Newton iteration for `T(e, Y)`.
* **Bitwise reproducibility.** Serial, threaded and MPI runs agree bit for bit, the AVX2 kernels equal the scalar ones, and 64 references
  stay fixed. A library call inside a run adds a dependency of the result on the library's version, its compiler and its maths library;
  per-cell independence keeps thread and rank count out of the result, but references would depend on the build. Without species the
  five-variable path must stay untouched.
* **Thread safety.** The usual design of such libraries is one object per thread (the state of a mixture object is mutable); the cell
  loops are OpenMP and the ranks are MPI, so a library stage would hold one private object per thread, built once.
* **Build.** The C++ build and CI are self-contained today (CMake, no third-party dependency); the development machine is Windows and the
  workstation Linux. A library with its own build system and a numerical-solver dependency is a real cost on the first and on CI.
* **The explicit time step follows the speed of sound**, so the target is high-speed reacting flow (detonations, shock-induced ignition),
  where chemistry is stiff and local and the flow step is the explicit one.

## Four ways to use the library

| Use | In the solver run? | Cost on the build | Effect on bit reproducibility | Verdict (proposed) |
| --- | --- | --- | --- | --- |
| **U1. Reference data**: one script computes ignition delays, constant-volume histories, equilibrium and CJ states, and the ZND structure; the tables are committed (as the cavity data of the literature) | no | none for the solver, CI and docs; the library is needed only by whoever regenerates the tables | none | **yes, in any case** (already in the species proposal) |
| **U2. Mechanism converter**: a Python tool in `python/fenice/` reads a mechanism file with the library and writes the solver's own mechanism file (polynomial coefficients, molecular weights, reactions, third bodies, falloff) | no | none | none: the solver reads its own file | **yes**: the solver can run any mechanism the library can read, with no library at run time |
| **U3. Chemistry stage through the library**: per cell, per half step of the splitting, a reactor integrated by the library behind an optional build switch | yes | optional switch, one object per thread | references of reacting cases depend on the library build | **optional, later** (a cross-check of the solver's own integrator, and a route to mechanisms the built-in evaluator does not support) |
| **U4. Whole thermodynamics through the library** (per-cell calls for gamma and temperature) | yes | required | as U3 | **no**: the face loop forbids it |

U1 and U2 take everything useful from the library without touching the run. U3 is the only run-time use worth discussing, and only as a
second implementation of the chemistry stage selected at configuration time.

## Proposal

1. **Own mixture thermodynamics in the kernels** (as the species proposal assumes): the species data of the mechanism file (NASA-type
   polynomials), `cp(T, Y)`, `e(T, Y)`, sound speed from the frozen mixture gamma, and `T(e, Y)` by Newton from the previous temperature.
   Cases G1 (variants 1 to 3) decide whether the mixture gamma at a contact needs a correction.
2. **Own chemistry stage**: Arrhenius, third-body and falloff reaction rates, and a per-cell stiff implicit integrator, independent across
   cells, inside the Strang splitting (details and the AMR recursion issue are in the species proposal).
3. **Library outside the run (U1, U2)**: the reference tables, and the converter. The G2 test (zero-dimensional ignition) then compares
   the solver's integrator with the library's constant-volume reactor on the same mechanism; the solver's thermodynamics is compared with
   the library's at a grid of `(T, Y)` points, to roundoff for the polynomials.
4. **Transport** (only for viscous reacting flow, not for the detonation T): mixture-averaged viscosity, conductivity and diffusion
   coefficients are not cheap to evaluate. Options, a choice: (a) constant Prandtl and Lewis numbers with a power-law viscosity (the
   solver's present `transport_law` extended in `T`), checked against the library; (b) tables or fits in `(T, Y)` generated by the
   converter. Species diffusion with its enthalpy flux needs its own verification case (a one-dimensional laminar flame is the natural one,
   and it needs a low-Mach treatment that is out of scope here).
5. **U3 later**, only if there is a mechanism the built-in evaluator cannot do, or as an independent check of the integrator inside T.

### What each choice costs

| | Own thermodynamics and chemistry (proposed) | Library inside the run (U3 / U4) |
| --- | --- | --- |
| CI and Windows build | unchanged | a third-party build on both |
| references | independent of any library version | depend on the library build |
| speed | inline arithmetic, vectorisable thermodynamics | per-cell object calls (*to measure*: time per call against the cost of one cell of the flow step) |
| mechanisms | those the converter can express (to define: which reaction types are supported) | any the library supports |
| maintenance | rate evaluation and integrator are the solver's own, with their own tests | the library's |

## Verification of the interface with the library

* **Converter**: for the committed mechanism, every polynomial coefficient, molecular weight and reaction parameter in the written file
  equals the library's (a round-trip test that reads the written file back and compares); the library is needed only in this test, which
  is skipped when it is absent (the committed tables of U1 carry the CI).
* **Thermodynamics**: `cp`, `e`, `h`, `gamma` and `T(e, Y)` of the solver against the library on a grid of temperatures and compositions;
  band to be fixed before the run (roundoff for the polynomial evaluation, the Newton tolerance for `T`).
* **Chemistry**: G2 of the species proposal (ignition delay within 1 %, the histories, the equilibrium at fixed `e` and volume).
* **Detonation**: T of the species proposal, whose reference is the ZND and CJ table of U1.

## Relation to moment methods for a dispersed phase

A dispersed phase (droplets, bubbles, particles) would be a different extension: a population described by moments, with an inversion at
each cell and source terms in phase space. It shares the same structure as the species array (extra transported fields, a per-cell stage
that never reads another cell, a stage inside the splitting) but needs its own realizable reconstruction at the faces, which is the same
class of problem as the open positivity items (`positivity-corners-open-boundaries.md`). If it is ever taken up, it comes after the species
array exists, and the choice between a small own implementation and a library is made then, on the same grounds as above (build cost,
bitwise reproducibility, per-cell independence).

## Questions

1. Is the use of the library confirmed as U1 plus U2 (outside the run), with the chemistry kept inside the solver? Or is a run-time stage
   (U3) wanted from the start, accepting the build and reproducibility costs?
2. Which reaction types must the converter support first (elementary with Arrhenius, third body, falloff; pressure-dependent
   Chebyshev or PLOG are heavier)? Proposed: the first three, enough for the nine-species hydrogen mechanism.
3. Transport for the first reacting cases: none (the detonation is Euler with chemistry, as in the species proposal), or constant Pr and Le?
4. Is the library allowed as an optional dependency of the repository's tools (Python), with its version recorded in the table files, even
   though the C++ build stays free of it?
5. Is a mechanism file format of the solver's own (a small text file) acceptable, or should the solver read the library's own format?
