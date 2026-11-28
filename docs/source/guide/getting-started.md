---
title: Getting started
nav: Getting started
section: Start here
order: 10
---

This page takes you from a fresh checkout to a result on screen in five steps, then collects the reference tables
(command line, output files, build options, MPI) that you will come back to later.

## Your first run

**1. Build.** On Windows with Visual Studio 2022 or newer (*Desktop development with C++* workload):

```powershell
.\cpp\build.ps1 -Test        # Release build + unit tests -> build\release\fenice.exe
```

On Linux or macOS use `cmake --preset release && cmake --build --preset release`. Other options, and MPI, are
under [Build](#build) below.

**2. See what is available.**

```powershell
build\release\fenice.exe --list-cases
```

**3. Run a case.** The Sod shock tube (4000 cells, first order) finishes in well under a second:

```powershell
build\release\fenice.exe --case "Sod Shock Tube"
```

The solver prints the settings it resolved (mesh, Riemann solver, time scheme, CFL), then one progress line every
20 steps, and writes its output to `results/sod_shock_tube/`. The name of the folder is the case's name in lower case with
underscores; the `[EXPORTER]` line at the top of the output gives the exact path.

**4. Look at the result.** Open `results/sod_shock_tube/sod_shock_tube.pvd` in ParaView: it indexes all the snapshots in time.
The [Sod page]({{root}}cases/sod.html) shows what the profiles should look like and how they compare with the exact solution.

**5. Change something.** Every parameter can be overridden on the command line with `--set`, without editing any file:

```powershell
build\release\fenice.exe --case "Shu-Osher Test" --set nx=4000 --set muscl_limiter=MC
build\release\fenice.exe --case "Lid Driven Cavity" --print-config
```

`--print-config` shows every resolved parameter and exits without running. The parameters, with their meaning and
defaults, are on the [Parameters]({{root}}guide/parameters.html) page.

Prefer a window? `python gui\run.py` opens a setup dialog and `python gui\workbench.py` the full workbench; see
[GUI]({{root}}guide/gui.html).

### Where to go next

* Set up your own problem: [Writing a test case]({{root}}guide/writing-cases.html)
  (`--new-case "My Case" --from "Sod Shock Tube"` copies a case you can edit).
* Understand the options you just touched: [How Fenice works]({{root}}modules/overview.html).
* Make it faster or run it on many cores: [Performance]({{root}}guide/performance.html).

## Command line

```powershell
build\release\fenice.exe --config run.cfg --threads 8
```

<details markdown="1">
<summary>All command-line options</summary>

| Option | Meaning |
|---|---|
| `--case NAME` | Test case: a name from `--list-cases`, or a path to a case folder ([Writing a test case]({{root}}guide/writing-cases.html)) |
| `--config FILE` | `key = value` parameter file (`#` starts a comment) |
| `--set KEY=VALUE` | Override one parameter (repeatable) |
| `--threads N` | OpenMP threads (default: all cores; 1 per rank under MPI) |
| `--list-cases` | List the built-in test cases |
| `--list-gui-cases` | List the ones the GUI offers |
| `--print-config` | Print the fully resolved parameters and exit |
| `--print-case-defaults` | Print only the test case's own defaults and exit |
| `--explain` | With `--print-config`: show which layer (baseline, case, config, `--set`) sets each value |
| `--check-case` | Report errors, warnings, unused keys and baseline restatements of the resolved run and exit ([Checking a case]({{root}}guide/writing-cases.html#checking-a-case)) |
| `--list-templates` | List the starter templates; `--new-case NAME --from template:NAME` copies one |
| `--no-logo` | Do not print the banner (coloured on a terminal; set `NO_COLOR=1` for plain text) |
| `--new-case NAME --from CASE` | Copy a case to `cases/<name>/case.toml` and exit |
| `--save-case TARGET` | Save the resolved run as a new case and exit (`--case-name` sets its name) |

</details>

Parameters resolve in this order, later sources winning: baseline defaults,
then the test case's file (`cases/<case>/case.toml`), then the `--config` file, then `--set` overrides.
Values are validated at start-up. A misspelt key (`--set nX=200`, with a "did you mean" hint), a
malformed or fractional number where an integer is expected, an unknown name (BC,
limiter, EOS, IBM shape, scheme), a half-periodic axis or a non-positive `cfl` / `t_end`
stops the run with an error.

**Stopping a run.** Ctrl+C stops cleanly after the current step: the final snapshot, the
checkpoint and the `.pvd` index are written. A second Ctrl+C aborts immediately. Creating a file
named `STOP` in the output folder asks for the same clean stop (the solver removes a stale one at
the start); under MPI, where `mpiexec` aborts the job on Ctrl+C, it is the way to stop with a
checkpoint.

## Output

Each run writes to `<output_root>/<case>/`, where `<case>` is the case name in lower case with underscores
(`sod_shock_tube`). The default `output_root` is `results/` in the repository, and AMR runs use `<case>_amr/`.
Use `--set output_root=<folder>` to write somewhere else.

| File | Content |
|---|---|
| `<case>_<step>.vtr` | Snapshot (VTK XML RectilinearGrid; inline base64 binary by default, `vtk_format = ascii` for text): `Density`, `Pressure`, `Temperature_nondim`, `Mach`, `Velocity` |
| `<case>_<step>.vtu` (`.pvtu` + one `.vtu` per rank under MPI) | AMR snapshot: one quad (3-D: hexahedron) per leaf, plus an `AMR_Level` field |
| `<case>.pvd` | Time-series index (open this in ParaView) |
| `<case>_restart.npz` | Checkpoint for `resume` and `use_hot_start` |

Snapshots and checkpoints are written every `save_interval` steps, and at the
end. Every file is written atomically (temporary file + rename), so ParaView
never sees a partially written file.

**Restart:** `resume = true` continues from the checkpoint of the same case.
**Hot start:** `use_hot_start = true` with `hot_start_file = <path.npz>` maps a
checkpoint of any resolution onto the current mesh by bilinear interpolation.
See [SolutionMapping]({{root}}modules/SolutionMapping.html) and [IO]({{root}}modules/IO.html).

## Build

### Windows (MSVC)

Requirements: Visual Studio 2022 or newer with the *Desktop development with C++*
workload (it provides `cl`, CMake and Ninja).

```powershell
.\cpp\build.ps1 -Test     # Release build + unit tests -> build\release\fenice.exe
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
cmake --preset release
cmake --build --preset release
```

The presets use Ninja (`apt install ninja-build`, `brew install ninja`). OpenMP and MPI
(OpenMPI or MPICH) are used automatically when CMake finds them. On macOS, Apple's clang has
no OpenMP runtime: `brew install libomp open-mpi` provides both, and CMake finds Homebrew's
`libomp` by itself.

Every build goes to `build/release/` (`fenice`, `fenice_tests`), or `build/debug/` with the
`debug` preset; on Windows `build.ps1` uses the same `release` preset. The Python tools
(GUI, validation scripts, docs, regression) share the package in `python/`, which
`python prerequisites.py` installs in editable mode; it finds the solver in `build/release/`,
or wherever `FENICE_EXE` points.

Tested: Windows 11 (MSVC 19.50, MS-MPI 10.1), Ubuntu 26.04 (GCC 15.2, Open MPI 5.0) and
macOS 27 on Apple M4 (Apple clang 21.0, Open MPI 5.0.9). The regression references are
reproduced bit for bit on all three, serial and on 4 MPI ranks.

### Build options

| CMake option | Default | Effect |
|---|---|---|
| `FV_USE_OPENMP` | `ON` | Multi-threaded kernels |
| `FV_USE_MPI` | `ON` | MPI domain decomposition (if an MPI SDK is found) |
| `FV_BUILD_TESTS` | `ON` | Build `fenice_tests` |

The code is compiled with `/fp:precise` (`-fno-fast-math -ffp-contract=off` on
GCC/Clang). This keeps the floating-point results bit-reproducible against the
regression references.

## Running in parallel

**Threads.** By default a run uses all cores of the machine with OpenMP; `--threads N` sets the number.

**MPI.** With `use_mpi = true` and `n_cpus > 1`, `fenice` re-launches itself under
`mpiexec -n n_cpus`. You can also start it directly:

```powershell
mpiexec -n 8 build\release\fenice.exe --case "IBM Supersonic Flow over Cylinder"
```

Each rank runs one OpenMP thread unless `--threads N` (or `OMP_NUM_THREADS`) says otherwise; ranks times threads is
capped at the logical cores (a message tells when a request was cut), because more threads than cores run several times
slower. On one node threads alone (`use_mpi = false`) are as fast as ranks for uniform meshes and faster for AMR; use MPI
for more than one node or more memory than one node has.

Each rank owns a strip of rows or columns; see
[MPIDecomposition]({{root}}modules/MPIDecomposition.html). The MPI result is
identical to the serial one, also for AMR (the leaves of each level are distributed over
the ranks; see [AMR]({{root}}modules/AMR.html)).

## Graphical interface

`python gui\run.py` opens the setup dialog. Accepting it runs the solver in the
same terminal with the chosen parameters; see [GUI]({{root}}guide/gui.html).
