# AGENTS.md — Fenice

Structured-Cartesian compressible finite-volume solver for the Euler and Navier-Stokes equations in 2-D and 3-D
(`nz = 1` is 2-D, `nz > 1` is 3-D), C++17 (OpenMP + MPI). MUSCL / WENO reconstruction, HLLC solvers, Navier-Stokes,
body forces, Immersed Boundary Method, Euler / SSP-RK2 / SSP-RK3, cell-based AMR with subcycling (quadtree and octree),
VTK output, `.npz` checkpoints with hot start, a PyQt6 GUI. Entry points: `build/release/fenice.exe` (CLI),
`python gui/workbench.py` (workbench), `python gui/run.py` (single-window setup dialog); both talk to the CLI.

## Working rules

- Work on `main`; no leftover branches. Commit only when asked. A change that touches no numerics ends its message with `[skip ci]`.
- Keep each file's line endings. Most files are CRLF; Python text-mode writes and `sed` silently flip LF <-> CRLF (use binary mode).
- Do not name other software (other solvers, codes, tools) in docs, comments or tests.
- The user chooses numerical methods: diagnose and propose options, do not implement a new method unasked.
- No new AMR tunables: use fixed constants.
- Runs worth keeping go under `results/`; scratch checks go in a temporary directory.
- Docs describe the current solver only (no port or changelog narrative). Regenerate figures only when results changed.
- Reference states may be regenerated only after a deliberate, physically validated change, and the commit says so.

## Layout

```
cpp/include/fv/   headers, one per module (Core, Boundaries, Schemes, Physics, TimeScheme, Cell* = AMR, SolutionMapping,
                  IO, Params, TestCases, MPIDecomposition, Simulation)
cpp/src/          implementations; cpp/apps/main.cpp = CLI; Switches.cpp = the FENICE_* environment switches
cpp/tests/        unit tests (harness test_common.hpp, main in test_main.cpp), regression.py + reference/*.npz
cpp/third_party/  toml++ and TinyExpr, vendored unmodified
cases/            one folder per case with a case.toml (read at run time); cases/validation/ = studies
python/fenice/    shared Python package (pip install -e python): runner, readers, uq / modal / spectra / convergence / vv20
gui/              PyQt6 workbench + setup dialog (no physics; talks to fenice)
docs/             source/*.md -> docs/site/ (gitignored) via docs/build.py; design/ and proposals/ = local notes (gitignored)
results/, build/  output and builds, gitignored
```

## Commands

```powershell
.\cpp\build.ps1 -Test                      # build (MSVC + Ninja) + unit tests
python cpp\tests\regression.py --mpi 4     # regression vs references: must print ALL MATCH
build\release\fenice_tests.exe --list      # unit test names; `fenice_tests NAME` runs the tests containing NAME
python cases\validation\check_all.py       # every check CI runs, in one command
build\release\fenice.exe --case "Sod Shock Tube"     # --print-config, --print-parameters, --print-switches
python docs\build.py                       # build the docs (fails on an undocumented parameter)
```

Linux / macOS: `cmake --preset release && cmake --build build/release` (add `-DFV_USE_MPI=OFF` without an MPI SDK).
MPI needs the MS-MPI SDK; without it the build is OpenMP-only. `build.ps1 -MpiSdk <dir>` must not silently drop MPI.

## Data conventions

- **State**: `Field = std::vector<State>`, `State = std::array<double,5>`. Conservative `[rho, rho*u, rho*v, rho*E, rho*w]`,
  primitive `[rho, u, v, p, w]`. `w` is stored LAST so the first four slots keep the 2-D layout; momentum slot per axis is `kMom = {1, 2, 4}`.
- **Cells** are row-major `cell(i,j,k) = (k*ny + j)*nx + i`. **Faces**: x-faces, then y-faces, then (3-D) z-faces. `Mesh` stores sizes + origin only.
  Riemann solvers take the face axis: `face_flux(UL, UR, axis, smax)`.
- **Parameters** are `key = value` strings (`Params`) -> typed `SimParams`. Resolution order: `baseline_defaults()` -> case file -> `--config` -> `--set`.
  A new key goes into `parameter_table()` (`Params.cpp`: key, case-file section, default) AND into `docs/source/guide/parameters.md`.
- **Cases are files** (`cases/<folder>/case.toml`), never code; BCs come from the file plus overrides. New cases: `fenice --new-case NAME --from CASE`.
- **Checkpoints** are `results/{case}/{case}_restart.npz` (uncompressed zip, written by `IO.cpp`, no third-party code); `format_version` is checked on read.
  Hot start: 2-D->2-D, 2-D->3-D (extrusion) and 3-D->3-D are interpolated; 3-D->2-D is an error.
- **Writes are atomic**: VTK to `*.tmp` then rename; checkpoints to `*_tmp.npz` then rename.
- **VTK**: Python readers go through `fenice.vtk_array(node)` / `fenice.read_vtr(path)`, never `node.text`.

## Invariants (do not break; each has a test)

- **Bit-reproducibility.** `cpp/tests/reference/` holds the final states of every regression configuration; `regression.py` must match (tolerance 1e-8).
  Keep `/fp:precise` (no fast-math, no FMA contraction) and the operation order of existing kernels. Results are identical for any OpenMP thread count.
  After touching shared code, rerun the full regression first.
- **2-D stays bit-identical through the 3-D code.** `assemble` and `assemble_3d` are separate paths; keep 2-D expressions (`min(dx,dy)`, VTK layout) verbatim.
- **MPI == serial** bit for bit. Any new stencil that crosses a rank interface needs a halo (padded state, `CrossGhosts`, IBM halos, mass-flow average all-reduced).
  Local meshes use the exact global `dx/dy`. Strips need >= 2 cells; z is never split.
- **Fused paths agree.** The fused uniform residual and the two-pass path (`FENICE_NO_FUSED=1`) give identical bits; anything the residual reads goes in both.
  The fused 2-D viscous residual equals the gradient version `viscous_residual` (kept as the test reference).
- **Periodic seams are exact** (wrap-around halos, `CrossGhosts::periodic_*`); `periodic_translation_invariance` guards it.
- **3-D tests** (extrusion equals 2-D per layer; rotated-plane, axis permutation, translation invariance) must keep passing when touching 3-D stencils.
- **Time step**: `Simulation::dt_length` is `min(dx,dy)` in 2-D and `2/(1/dx+1/dy+1/dz)` in 3-D. The convective and viscous eigenvalues share one budget
  (the viscous speed counts twice); a max over cells is partition independent, a sum of per-partition maxima is not. The copies of the viscous expression must stay identical.
- **Shock-face dissipation** (`shock_blend`, default 0.2) lives in `RiemannSolver::hllc`; the AVX2 twin `hllc4` must stay bit-identical; `shock_blend = 0` is the older solver.
- **OpenMP**: loops below `kOmpMinWork` (4096) are serial (`if()` clause), keep it. MSVC is OpenMP 2.0: signed loop indices, no `max` reductions (thread-local max + critical).
- **AVX2 kernels** (`SchemesAvx2.cpp`, runtime-detected, `FENICE_NO_SIMD=1` disables) must be bit-identical to the scalar twin: only add / sub / mul / div / sqrt and masks,
  the scalar operation order, no FMA (check `dumpbin /disasm` or `objdump -d` for `vfm`), `std::min/max` as compare + blend, leftovers through the scalar function.
  Change a scalar kernel and its vector twin together. Keep CPU-independent code out of that file. Keep a new vector kernel only if it is faster with MSVC too.
- **IBM**: `apply_forcing` inside every residual (all RK stages); `apply_solid_reset` once per complete step only. Ghost-cell IBM (`ibm_method = ghost`, default `auto`)
  must keep `cases/validation/features/ibm_ghost.py --check` passing (serial vs 2-7 ranks, uniform and AMR).
- **Source terms**: `[source]` expressions of x, y, z, t and the run parameters; time-dependent ones are evaluated at `TimeScheme::stage_time`.
  The `manufactured_solution*` cases are GENERATED by `python -m fenice.mms --write`: never hand edit.

## AMR (cell-based face-list engine, 2-D quadtree and 3-D octree; description in `docs/source/modules/AMR.md`)

- Level 0 is bit-identical to the uniform solver; keep it that way when touching the physics.
- **Incremental rebuild**: anything `CellMesh::rebuild` / `CellFluxEngine::build` computes per leaf or face must depend only on the leaf and its neighbouring squares
  (copied for clean leaves) or be recomputed in full. `cellflux_incremental_rebuild_equals_full_rebuild` guards it; `FENICE_FULL_REBUILD=1` forces the full path.
- **Distributed state** (`CellParallel`): a rank stores only its share; the halo is exactly what `CellFluxEngine::face_dependencies` reads. Anything new that reads a leaf
  in the residual must be added to `face_dependencies`, or MPI diverges from serial. No rank holds the whole-mesh topology.
- Checkpoints hold the leaf list and resume bit for bit, also on another rank count. AMR runs write to `results/<case>_amr/`.
- WENO5 is not available with AMR yet. The AMR join criterion uses the parent's indicator times `CellIndicator::join_band`; do not make refine and join use the same criterion.
- The AMR engine's IBM is not bit-identical to the uniform solver's, even at level 0.

## GUI

A thin client: it holds no physics. Case list and defaults come from `fenice --list-gui-cases` / `--print-config`; `SetupDialog.sim_params` keys equal solver parameter keys.
Numerics for the analysis tools (spectra, modal decomposition, sensitivity, loads) belong in `python/fenice`, not in `gui/`; the GUI calls them.

## Testing and docs

- Unit tests: `TEST(name)` / `CHECK` / `CHECK_NEAR`; temporary files under `tmp_dir()`. Add a case to `CASES` in `regression.py` only together with a validated reference
  (`--create-missing` adds references without touching existing ones).
- For a change that must not move the bits, dump final states (`dump_state`) before and after and compare hashes (also with `FENICE_NO_SIMD=1` and `FENICE_FULL_REBUILD=1`).
- CI (`.github/workflows/ci.yml`, every push; Linux x64 / ARM64, macOS, Windows): unit tests, regression serial + 4 MPI ranks, `amr_mpi.py --check`, `ibm_ghost.py --check`, MMS quick study.
- Documentation: public C++ declarations carry a `//` comment directly above them (that is the API reference). Images live in `docs/img/`, prefixed by case.
  Validation figures come from `cases/validation/<group>/<study>.py --docs`. Never commit generated HTML.
