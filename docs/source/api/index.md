---
title: API reference
nav: Overview
section: API reference
order: 0
---

This section is for people who read or extend the C++ code. If you only run the solver, the
[User guide]({{root}}guide/getting-started.html) and the [Methods]({{root}}modules/overview.html) pages are what you need.

## How the reference is made

There is one page per public header in `cpp/include/fv/`. Each page lists the classes, functions, enums and
variables of that header, with the `//` comment written directly above the declaration. The pages are extracted from the headers
when the documentation is built, so they cannot go out of date. To document something, write a `//` comment above its
declaration. Each page starts with a link to the Methods page that explains what the module does.

| Header | Page | What it declares |
|---|---|---|
| `Core.hpp` | [Core]({{root}}api/Core.html) | mesh, equations of state, state layout |
| `Boundaries.hpp` | [Boundaries]({{root}}api/Boundaries.html) | boundary conditions and ghost cells |
| `Schemes.hpp` | [Schemes]({{root}}api/Schemes.html) | Riemann solvers, limiters, reconstruction |
| `Physics.hpp` | [Physics]({{root}}api/Physics.html) | viscous fluxes, body forces, immersed boundary |
| `TimeScheme.hpp` | [TimeScheme]({{root}}api/TimeScheme.html) | Runge-Kutta integrators |
| `CellMesh.hpp` | [CellMesh]({{root}}api/AMR.html) | adaptive leaf mesh |
| `SolutionMapping.hpp` | [SolutionMapping]({{root}}api/SolutionMapping.html) | interpolation and restriction |
| `IO.hpp` | [IO]({{root}}api/IO.html) | VTK output, checkpoints, monitors |
| `MPIDecomposition.hpp` | [MPIDecomposition]({{root}}api/MPIDecomposition.html) | domain decomposition |
| `Simulation.hpp` | [Simulation]({{root}}api/Simulation.html) | the run driver |
| `CaseFile.hpp` | [CaseFile]({{root}}api/CaseFile.html) | case file reader and writer |
| `TestCases.hpp` | [TestCases]({{root}}api/TestCases.html) | the case registry |

## Data conventions

* **State**: `Field = std::vector<State>`, `State = std::array<double,5>`. Conservative `[rho, rho*u, rho*v, rho*E, rho*w]`,
  primitive `[rho, u, v, p, w]`; `w` is stored last, and the momentum slot of each axis is `kMom = {1, 2, 4}`.
* **Cells** are row-major, `cell(i,j,k) = (k*ny + j)*nx + i`. **Faces** are numbered x-faces first, then y-faces, then (3-D)
  z-faces. `Mesh` stores sizes and origin only. Riemann solvers take the face axis: `face_flux(UL, UR, axis, smax)`.

## Repository layout

<details markdown="1">
<summary>The folders and what each holds</summary>

| Path | Content |
|---|---|
| `cpp/include/fv/`, `cpp/src/` | Solver library (one header per module page) |
| `cpp/apps/main.cpp` | `fenice` command-line driver |
| `cpp/tests/` | Unit tests (`fenice_tests`), regression references and `regression.py` |
| `cases/` | Test cases, one folder with a `case.toml` each ([Writing a test case]({{root}}guide/writing-cases.html)) |
| `cpp/third_party/` | toml++ and TinyExpr (case files) |
| `gui/` | PyQt6 workbench and launcher |
| `python/fenice/` | Python package shared by the GUI, validation and docs tools: finds and runs the solver, reads its output |
| `cases/validation/` | One script per validation study, grouped in `verification/`, `benchmarks/`, `amr/`, `features/`; results in `results/validation/<study>/` ([Validation]({{root}}guide/validation.html)) |
| `docs/` | This documentation: `docs/source/` and `docs/img/`, built into `docs/site/` by `python docs/build.py` |
| `build/` | Builds: `build/release/`, `build/debug/` |
| `results/` | Simulation output (not in git) |

</details>

The rules that keep the numerics reproducible (bit-identical results, vector kernels equal to their scalar twins, MPI equal to serial)
are described on the [Testing]({{root}}guide/testing.html) page.
