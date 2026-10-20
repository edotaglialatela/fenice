# Fenice — C++ solver

The solver itself: a 2-D / 3-D compressible finite-volume code for the Euler and
Navier-Stokes equations, in C++17 with OpenMP and MPI (`nz = 1` is 2-D, `nz > 1` 3-D). The full documentation is in `docs/`
(`python docs/build.py --open`); this file is the short version.

## Layout

```
cpp/
  include/fv/   Core (types, mesh, EOS), Boundaries, Schemes (Riemann, limiters, MUSCL),
                Physics (gradients, Navier-Stokes, body force, IBM), TimeScheme, AMR,
                SolutionMapping, IO (VTK + npz), Params, TestCases, MPIDecomposition, Simulation
  src/          implementations (one .cpp per header)
  apps/main.cpp fenice command-line driver
  tests/        test_main.cpp (unit tests), regression.py + reference/ (reference states)
  third_party/  toml++ and TinyExpr (case-file reading and expressions)
  build.ps1     Windows build script (MSVC + Ninja)
```

## Build

```powershell
.\cpp\build.ps1 -Test        # Release build + unit tests -> cpp\build\fenice.exe
.\cpp\build.ps1 -NoMpi       # OpenMP only
```

MPI needs the MS-MPI SDK: `winget install Microsoft.msmpisdk`. You can also pass
`-MpiSdk "<dir with Include and Lib>"` for an unpacked copy. Linux / macOS:
`cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build cpp/build`.

## Run

```powershell
cpp\build\fenice.exe --list-cases
cpp\build\fenice.exe --case "Sod Shock Tube"
cpp\build\fenice.exe --case "Shu-Osher Test" --set nx=4000 --set muscl_limiter=MC
cpp\build\fenice.exe --config run.cfg --threads 8
mpiexec -n 8 cpp\build\fenice.exe --case "IBM Supersonic Flow over Cylinder"
cpp\build\fenice.exe --case "Taylor-Green Vortex 3D" --set monitor_interval=10   # 32^3
python gui\run.py                                   # GUI front end
```

Test cases are the case files in `../cases/` (one `case.toml` per folder), read at run
time; `--new-case NAME --from CASE` copies one. Parameters resolve in this order, later
sources winning: baseline defaults, then the case file, then `--config` (`key = value`), then `--set key=value`. Every key is
documented on the Parameters page of the docs. With `use_mpi = true`, the solver
re-launches itself under `mpiexec -n n_cpus`. Ctrl+C stops cleanly, writing the
checkpoint and the `.pvd` index.

Output goes to `results/<case>/`: ASCII `.vtr` / `.vtm` snapshots, a `.pvd` time series
for ParaView, and a `<case>_restart.npz` checkpoint for `resume` / `use_hot_start`.

## Tests

```powershell
cpp\build\fenice_tests.exe                       # 55 unit tests
python cpp\tests\regression.py --mpi 4       # 24 serial + 15 MPI runs vs reference states
```

`cpp/tests/reference/` stores final states for 24 configurations. Sixteen come from the
original Python solver, which was the reference implementation until 2026-10-20; the C++
solver reproduces them bit for bit (BDF2 / mass-flow cases to ~1e-14). Three more
(Taylor-Green, isentropic vortex) were added from the C++ solver after validation against
their exact solutions, and five 3-D ones after the 3-D unit tests. MPI runs reproduce the
serial runs exactly, in 2-D and 3-D.
