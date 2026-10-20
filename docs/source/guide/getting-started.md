---
title: Getting started
nav: Getting started
section: Guide
order: 10
---

## Build

### Windows (MSVC)

Requirements: Visual Studio 2022 or newer with the *Desktop development with C++*
workload (it provides `cl`, CMake and Ninja).

```powershell
.\cpp\build.ps1 -Test     # Release build + unit tests -> cpp\build\fenice.exe
.\cpp\build.ps1 -NoMpi    # build without MPI
```

`build.ps1` loads the Visual Studio x64 environment, configures CMake with Ninja
and builds two targets: `fenice` and the unit-test runner `fenice_tests`.

**MPI** needs the MS-MPI runtime (`mpiexec`) and the MS-MPI **SDK** (headers and
`msmpi.lib`):

```powershell
winget install Microsoft.msmpi Microsoft.msmpisdk
```

The SDK installer sets the `MSMPI_INC` / `MSMPI_LIB64` variables that CMake
uses. To build against an unpacked SDK instead, pass
`.\cpp\build.ps1 -MpiSdk "<folder containing Include and Lib>"`. Without the SDK the
build succeeds with OpenMP only, and a run requesting MPI falls back to one
process with all threads.

### Linux / macOS

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
```

OpenMP and MPI (OpenMPI or MPICH) are used automatically when CMake finds them.

### Build options

| CMake option | Default | Effect |
|---|---|---|
| `FV_USE_OPENMP` | `ON` | Multi-threaded kernels |
| `FV_USE_MPI` | `ON` | MPI domain decomposition (if an MPI SDK is found) |
| `FV_BUILD_TESTS` | `ON` | Build `fenice_tests` |

The code is compiled with `/fp:precise` (`-fno-fast-math -ffp-contract=off` on
GCC/Clang). This keeps the floating-point results bit-reproducible against the
regression references.

## Run

```powershell
cpp\build\fenice.exe --list-cases
cpp\build\fenice.exe --case "Sod Shock Tube"
cpp\build\fenice.exe --case "Shu-Osher Test" --set nx=4000 --set muscl_limiter=MC
cpp\build\fenice.exe --config run.cfg --threads 8
cpp\build\fenice.exe --case "Lid Driven Cavity" --print-config
```

| Option | Meaning |
|---|---|
| `--case NAME` | Test case: a name from `--list-cases`, or a path to a case folder ([Writing a test case]({{root}}guide/writing-cases.html)) |
| `--config FILE` | `key = value` parameter file (`#` starts a comment) |
| `--set KEY=VALUE` | Override one parameter (repeatable) |
| `--threads N` | OpenMP threads (default: all cores; 1 per rank under MPI) |
| `--list-cases` | List the built-in test cases |
| `--print-config` | Print the fully resolved parameters and exit |
| `--print-case-defaults` | Print only the test case's own defaults and exit |
| `--no-logo` | Do not print the banner |
| `--new-case NAME --from CASE` | Copy a case to `cases/<name>/case.toml` and exit |
| `--save-case TARGET` | Save the resolved run as a new case and exit (`--case-name` sets its name) |

Parameters resolve in this order, later sources winning: baseline defaults,
then the test case's file (`cases/<case>/case.toml`), then the `--config` file, then `--set` overrides. Every key is
listed on the [Parameters]({{root}}guide/parameters.html) page. Values are
validated at start-up, and a malformed number or unknown BC name stops the run with
an error.

**Ctrl+C** stops cleanly after the current step: the final snapshot, the
checkpoint and the `.pvd` index are written. A second Ctrl+C aborts immediately.

### MPI

With `use_mpi = true` and `n_cpus > 1`, `fenice` re-launches itself under
`mpiexec -n n_cpus`. You can also start it directly:

```powershell
mpiexec -n 8 cpp\build\fenice.exe --case "IBM Supersonic Flow over Cylinder"
```

Each rank owns a strip of rows or columns; see
[MPIDecomposition]({{root}}modules/MPIDecomposition.html). The MPI result is
identical to the serial one. AMR runs are serial (multi-threaded) and ignore
`use_mpi`.

### GUI

`python gui\run.py` opens the setup dialog. Accepting it runs the solver in the
same terminal with the chosen parameters; see [GUI]({{root}}guide/gui.html).

## Output

Each run writes to `<output_root>/<case>/`. The default `output_root` is
`results/` in the repository, and AMR runs use `<case>_amr/`.

| File | Content |
|---|---|
| `<case>_<step>.vtr` | Snapshot (VTK XML RectilinearGrid, ASCII): `Density`, `Pressure`, `Temperature_nondim`, `Mach`, `Velocity` |
| `<case>_<step>.vtm` + `<case>_<step>/p*.vtr` | AMR snapshot: one block per leaf patch, plus an `AMR_Level` field |
| `<case>.pvd` | Time-series index (open this in ParaView) |
| `<case>_restart.npz` | Checkpoint for `resume` and `use_hot_start` |

Snapshots and checkpoints are written every `save_interval` steps, and at the
end. Every file is written atomically (temporary file + rename), so ParaView
never sees a partially written file.

**Restart:** `resume = true` continues from the checkpoint of the same case.
**Hot start:** `use_hot_start = true` with `hot_start_file = <path.npz>` maps a
checkpoint of any resolution onto the current mesh by bilinear interpolation.
See [SolutionMapping]({{root}}modules/SolutionMapping.html).
