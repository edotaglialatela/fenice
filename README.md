# Fenice

**Fenice** (Italian for *phoenix*) is a compressible finite-volume solver for the Euler
and Navier-Stokes equations on structured Cartesian grids, in 2-D and 3-D. It is written
in C++17 with OpenMP threading and MPI domain decomposition, and comes with a PyQt6 setup
GUI, validation scripts and full documentation.

<p align="center">
  <img src="docs/img/taylor_green_3d_vorticity.png" width="90%"
       alt="Vorticity magnitude of the 3-D Taylor-Green vortex at four times">
</p>

## Features

- **Riemann solvers:** HLLC, Low-Mach HLLC (Thornber) and preconditioned HLLC (Weiss-Smith).
- **Reconstruction:** 2nd-order MUSCL with Minmod, Van Leer, MC and Koren limiters.
- **Physics:** ideal and stiffened-gas equations of state, Navier-Stokes viscous and heat
  fluxes, body forces, and an immersed boundary method (cylinder or sphere).
- **Time integration:** Explicit Euler, SSP-RK2, RK4, and BDF2 with dual time stepping.
- **Meshes:** uniform 2-D (`nz = 1`) or 3-D (`nz > 1`) grids, plus block-structured AMR in
  2-D with subcycling, refluxing and 2:1 balance.
- **Boundary conditions:** slip and no-slip walls, moving walls, velocity and mass-flow
  inlets, pressure outlets, transmissive and periodic.
- **Parallelism:** OpenMP everywhere, plus MPI strips whose results match the serial run
  exactly.
- **I/O:** VTK output for ParaView, `.npz` checkpoints with restart, and hot start that maps
  a solution onto a different resolution.

## Build

Windows (MSVC + Ninja from Visual Studio):

```powershell
.\cpp\build.ps1 -Test        # Release build + unit tests -> cpp\build\fenice.exe
```

MPI needs the MS-MPI SDK (`winget install Microsoft.msmpisdk`); without it the build is
OpenMP-only. Linux / macOS:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build
```

## Run

```powershell
cpp\build\fenice.exe --list-cases
cpp\build\fenice.exe --case "Sod Shock Tube"
cpp\build\fenice.exe --case "Taylor-Green Vortex 3D" --set monitor_interval=10
mpiexec -n 8 cpp\build\fenice.exe --case "IBM Supersonic Flow over Cylinder"
python gui\run.py            # GUI (python prerequisites.py installs its dependencies)
```

Output goes to `results/<case>/` as `.vtr` snapshots plus a `.pvd` time series for
ParaView, and a restart checkpoint.

## Test cases

| Case | Validates |
|---|---|
| Sod Shock Tube | Riemann solvers, limiters, MPI, AMR |
| Poiseuille Flow | Viscous fluxes, inlet / outlet BCs, BDF2 |
| Lid Driven Cavity | 2-D Navier-Stokes at Re = 400 vs Ghia et al. (1982) |
| IBM Flow over Cylinder | IBM, vortex shedding at Re = 100, hot start |
| Shu-Osher Test | Shock / entropy-wave interaction |
| IBM Supersonic Flow over Cylinder | IBM at Mach 2, bow shock |
| Taylor-Green Vortex | Viscous decay vs the exact solution |
| Isentropic Vortex | 2nd-order accuracy on smooth advection |
| Taylor-Green Vortex 3D | 3-D transition to turbulence at Re = 1600 |

## Testing

```powershell
cpp\build\fenice_tests.exe                  # unit tests
python cpp\tests\regression.py --mpi 4       # compare against the stored reference states
```

`cpp/tests/reference/` holds the final states of 24 reference configurations; every
change to the code must reproduce them (serial and on MPI ranks).

## Documentation

The HTML documentation is in [`docs/`](docs/index.html): open `docs/index.html` locally,
or rebuild it with `python docs/build.py --open`. It covers the governing equations, every
module, all parameters, and a validation page per test case.

## License

[MIT](LICENSE) © 2026 Edoardo Taglialatela
