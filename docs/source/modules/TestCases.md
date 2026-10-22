---
title: Test cases
nav: Test cases (registry)
section: Modules
order: 110
header: TestCases.hpp
---

Every test case is a case file, `cases/<folder>/case.toml`, read at start-up (see
[Writing a test case]({{root}}guide/writing-cases.html)). `test_registry()` loads all of
them, sorted by their `[case] order`. `get_test_case()` finds a case by name, by folder
name, or by path. The nine benchmarks shipped with Fenice are such files. Their
defaults override the baseline defaults, and anything set in a config file or with `--set`
overrides those (see [Parameters]({{root}}guide/parameters.html)).

| Case (`--case`) | Validates | Page |
|---|---|---|
| `Sod Shock Tube` | Riemann solvers, limiters, MPI, AMR | [Sod]({{root}}cases/sod.html) |
| `Poiseuille Flow` | Viscous fluxes, inlet / outlet BCs, BDF2 | [Poiseuille]({{root}}cases/poiseuille.html) |
| `Lid Driven Cavity` | 2-D Navier-Stokes at Re = 400 (Ghia 1982) | [Cavity]({{root}}cases/cavity.html) |
| `IBM Flow over Cylinder` | IBM, vortex shedding at Re = 100, hot start | [Cylinder]({{root}}cases/cylinder.html) |
| `Shu-Osher Test` | Shock / entropy-wave interaction, MUSCL accuracy | [Shu-Osher]({{root}}cases/shu-osher.html) |
| `IBM Supersonic Flow over Cylinder` | IBM fringe treatment at Mach 2, bow shock | [Supersonic cylinder]({{root}}cases/supersonic-cylinder.html) |
| `Taylor-Green Vortex` | Viscous decay vs exact solution, periodic BCs | [Taylor-Green]({{root}}cases/taylor-green.html) |
| `Isentropic Vortex` | 2nd-order accuracy on smooth advection, periodic BCs | [Isentropic vortex]({{root}}cases/isentropic-vortex.html) |
| `Taylor-Green Vortex 3D` | 3-D solver: vortex stretching, transition, dissipation peak (Re = 1600) | [Taylor-Green 3-D]({{root}}cases/taylor-green-3d.html) |
| `Manufactured Solution`, `... Euler`, `... 3D` | Code verification: source terms, observed orders of accuracy (generated cases) | [Manufactured solutions]({{root}}cases/manufactured-solution.html) |

The 2-D cases also run on a 3-D mesh (`--set nz=...`). Their initial field is then
extruded unchanged along z. With periodic (default) front/back sides, every z-layer
reproduces the 2-D solution, which `fenice_tests` checks.

`fenice --list-cases` prints the registry, and
`fenice --case NAME --print-case-defaults` prints a case's own parameters. The tables on
the case pages are generated from that output.

A few analytic initial fields stay in C++ as **profiles** (`find_profile`), which a case
file selects with `[initial] profile = "NAME"`: `taylor_green`, `isentropic_vortex` and
`taylor_green_3d`. Everything else (uniform states, regions, analytic expressions) is
written in the case file.

**Adding a case.** Copy an existing one, `fenice --new-case NAME --from CASE`, then edit
`cases/<name>/case.toml`. No C++ or rebuild is needed, and the case appears in `--list-cases`
and in the GUI. A new initial field that cannot be written as regions and expressions can
be added as a profile in `cpp/src/TestCases.cpp`.

{{api}}
