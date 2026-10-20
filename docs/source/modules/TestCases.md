---
title: Test cases
nav: Test cases (registry)
section: Modules
order: 110
header: TestCases.hpp
---

Nine built-in benchmarks provide pre-configured initial fields, case-specific boundary
conditions and default parameters. Select one with `--case NAME`, or in the GUI. A case's
defaults override the baseline defaults, and anything set in a config file or with
`--set` overrides those (see [Parameters]({{root}}guide/parameters.html)).

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

The 2-D cases also run on a 3-D mesh (`--set nz=...`). Their initial field is then
extruded unchanged along z. With periodic (default) front/back sides, every z-layer
reproduces the 2-D solution, which `fenice_tests` checks.

`fenice --list-cases` prints the registry, and
`fenice --case NAME --print-case-defaults` prints a case's own defaults. The tables on
the case pages are generated from that output.

**Adding a case.** Write a function returning a `TestCase` (key, defaults,
`initialize`, optional `configure_boundaries`, `three_d = true` for a case that needs
`nz > 1`) in `cpp/src/TestCases.cpp` and add it to `test_registry()`. The GUI picks it up automatically.

{{api}}
