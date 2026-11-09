# Repository review: structural improvements and future-proofing

Status: items 4 and 7 implemented (2026-11-09), the CLAUDE.md part of item 8 done, items 5 and 6 decided (no work); item 1 done (viscous bound, step length), item 3 partly done, item 9 open. Based on the file sizes and layout of the repository at the 3-D octree Q7 commit and on the
invariants written down in `CLAUDE.md`; it is a map of candidates, not an audit line by line. Each item gives the evidence, the
benefit, the risk to the bit-reproducibility rule, and a size. The user decides what goes ahead.

## 1. Duplicated 2-D / 3-D kernels (largest recurring cost)

* Done for the viscous speed bound (2026-11-09): `fv/ViscousBound.hpp` (`viscous_nu_eff`, `viscous_speed_bound`) now serves the six scalar
  copies (`Physics.cpp` x3, `CellFlux.cpp` x3). Golden dumps (10 AMR runs, 1/3/7 threads) bitwise equal to the binary before the change in
  the default, `FENICE_NO_SIMD=1` and `FENICE_FULL_REBUILD=1` modes; regression and unit tests unchanged. The header must not be included
  by `SchemesAvx2.cpp`; the AVX2 twin stays hand-written. The time-step length and the 1/h^2 sums (`step_length_2d/3d`, `inv_h2_2d/3d`, 9 sites in `Simulation.cpp`, `Physics.cpp`, `CellFlux.cpp`) moved
  into the same header, same checks. The shock-blend sensor has one scalar copy (`Schemes.cpp`) and its AVX2 twin, which cannot share it
  (see above), so there was nothing to merge; the twin test `riemann_batch_equals_scalar_bit_for_bit` stays the guard.

* Evidence: `assemble` / `assemble_3d`, `viscous_residual` / `viscous_residual_3d`, the uniform and the AMR copies of the viscous
  eigenvalue expression ("six copies ... must stay identical"), the scalar and AVX2 twins of the Riemann and viscous kernels. Every
  physics change is made two to six times and guarded by a bit-for-bit test.
* Proposal: do not merge the 2-D path into the 3-D one (the 2-D bits are a requirement). Instead put each *expression that must stay
  identical* in one inline function in one header (viscous speed bound, shock-blend sensor, time-step length), called by every copy.
  Operation order is then fixed in one place, and the "keep the copies together" rule becomes a compile-time fact.
* Risk: low if each function is moved verbatim and the golden dumps (default, `NO_SIMD`, `FULL_REBUILD`) are compared. Size: small, one
  expression at a time.

## 2. Very large translation units

* Done (2026-11-09): `Simulation.cpp` (2823 lines) is now `Simulation.cpp` (setup, initial field, residual assembly), `SimulationRestart.cpp`
  (checkpoint, hot start, final dump) and `SimulationRun.cpp` (monitors, serial / AMR / MPI time loops), with the shared helpers and the
  post-processing writer in the internal headers `SimulationInternal.hpp` / `SimulationPostprocess.hpp`; `CellFlux.cpp` (2133) is now the
  face-list build (`CellFlux.cpp`) and the residual kernels (`CellFluxResidual.cpp`). Moves only: golden dumps bitwise equal in the three
  modes, regression and unit tests unchanged. `Physics.cpp` (1273) was left as is.

* Evidence: `Simulation.cpp` 2807 lines, `CellFlux.cpp` 2094, `Physics.cpp` 1273. `CellParallel` was already split in four files with
  good results.
* Proposal: split `Simulation.cpp` along its phases (setup and initial condition, the step loop, output and monitors, restart and hot
  start) and `CellFlux.cpp` into the face-list build and the residual kernels. No behaviour change; shorter incremental builds and
  reviews.
* Risk: none for the numerics (moves only). Do it when no long branch (the Q8 work) touches those files, to avoid conflicts.

## 3. One source of truth for parameters

* Done in part (2026-11-09): `parameter_table()` (`Params.cpp`: key, case-file section, baseline default) is the one table; `baseline_defaults()`
  and the case-file `section_table()` are built from it (resolved configuration of two cases identical before and after), `fenice
  --print-parameters` lists it, and `docs/build.py` fails when a key is missing from `guide/parameters.md`. Not done: the typed defaults in
  `SimParams::from` (fallbacks for absent keys, left as they are), one-line descriptions in the table, and generating the docs table or the
  GUI tooltips from it (the page carries prose and links a generated table would lose).

* Evidence: a new parameter must be added to `Params` / `SimParams`, `section_table()` in `CaseFile.cpp`, the table in
  `docs/source/guide/parameters.md`, and is picked up by the GUI through `--print-config`. A test enforces only the case-file part.
* Proposal: a single declarative table (name, type, default, section, one-line description, allowed values) from which the parser, the
  case-file sections, `--print-config`, the parameter table of the docs and the GUI tooltips are generated; the docs table becomes a
  `{{parameters}}` directive like `{{defaults:}}`. The `reconstruction` / `use_muscl` alias shows why: renames need a central place.
* Risk: low; behaviour is checked by the existing tests that read every parameter. Size: medium.

## 4. Versioned file formats

* Done: the checkpoint writes `format_version` (1); a file without it is read as version 1, a higher number is refused with a message
  (`checkpoint_edge_cases`). Not done: the same key in the `dump_state` file / `leaf_meta`, and fixture files per released version.

* Evidence: checkpoints and `final.npz` have grown by keys (`nz`, `leaf_k`, five columns in 3-D, the lagged wave-speed bound); old
  files are accepted by tolerating missing keys.
* Proposal: an explicit `format_version` key in the checkpoint, the dump and `leaf_meta`, with a single reader that maps old versions
  to the current one and refuses unknown future ones with a clear message. A test keeps one file of every released version.
* Risk: none for the numerics. Size: small.

## 5. Dimension as a compile-time parameter in the AMR kernels (only if 3-D AMR becomes a main use)

* Evidence: the engine is run-time `dim`; 3-D kernels sit beside the 2-D ones. Q8 will say how much the 3-D path costs against the
  uniform solver.
* Proposal: decide after Q8. If the 3-D kernels are slower for dimensional-branch reasons, templates on `dim` (with the 2-D
  instantiation required to give the same bits) are the tool; if the cost is memory traffic, they are not.
* Risk: medium; do not start before the Q8 measurements.
* Decision (2026-11-09): the Q8 measurements (design sections 19-20) put the cost of the 3-D octree in the remesh and in memory, not in
  dimensional branches, so no templating on `dim`.

## 6. Portability layer for accelerators

* Evidence: the solver is OpenMP + MPI with AVX2 intrinsics for two kernel families. The scalar fallback keeps ARM working.
* Decision (2026-11-09): nothing to build. If a GPU is ever wanted, the fused uniform residual (reconstruction + Riemann + balance per row of faces, no face
  storage) is the natural unit to port first; the face-list AMR engine is not. Write the decision down rather than design for it.

## 7. Reference states and golden dumps

* Done: `regression.py --update CASE...` refuses a dirty git tree (`--force` overrides) and writes a per-reference report (steps, max
  relative change) to paste into the commit message. Golden-dump comparison is still the procedure in `CLAUDE.md`.

* Evidence: 74 regression references, regenerated by hand after deliberate changes, golden-dump comparison done by an ad-hoc procedure
  described in prose (`amr/golden.py` exists).
* Proposal: one command that regenerates *only the named* references, writes the old-versus-new difference per reference to a report
  (which cases moved, by how much), and refuses to run on a dirty tree. The reasons for a regeneration are recorded in the commit
  message from that report. This makes the "only regenerate after a deliberate, validated change" rule enforceable.
* Risk: none. Size: small.

## 8. Documentation and project guidance

* `CLAUDE.md` is still untracked (ignored by .gitignore; tracking it is the owner's decision). It is long and had internal contradictions (reference counts 74 / 55 / 38, a path typo); the counts and
  the typo are fixed (the count is the number of files in `cpp/tests/reference/`). Open: keep the invariants (bit-reproducibility, the
  twin-kernel rules, the MPI == serial rule) in it, and move measurements and history to `docs/design/`.
* `docs/design/` holds five long files; `amr-distributed-handoff.md` (a hand-over note) is the candidate to fold into
  `amr-distributed.md` or retire after its content is checked. Nothing else is obviously stale.
* Validation pages repeat the same run commands and the same pass/fail bands in the page, in `testing.md` and in the script
  docstring. Proposal: the script docstring is the source and the pages link to it.

## 9. Continuous integration time

* Measured (2026-11-09, runs of 2026-11-08, `gh run view`; jobs run in parallel, so a push takes as long as the slowest): Windows 790-930 s,
  macOS 730-820 s, Linux x64 550-600 s, Linux ARM64 430-480 s. The largest steps per job: `amr_mpi.py --check` on 1-7 ranks 100-400 s
  (Windows 300-400 s, about 40 % of the job), the MMS quick study 90-180 s, build + unit tests 140-160 s on Windows (50 s + 55-95 s elsewhere),
  regression on 4 ranks 40-120 s, ghost-cell IBM check 20-80 s. Moving rank counts 5-7 of the AMR check to the nightly workflow would cut
  roughly 40 % of that step (an estimate: the time per rank count was not measured separately).

* Evidence: CI runs unit tests, regression serial and on 4 ranks, `amr_mpi.py --check`, `ibm_ghost.py --check` and the MMS quick
  study on four platforms. Each addition (the 3-D AMR cases) lengthens it.
* Proposal: measure the time per step of the workflow and move the slowest bitwise checks that do not depend on the platform
  (rank counts 5-7) to the nightly workflow, keeping 1, 2 and 4 on every push. Decide with the measured times.

## Suggested order

1. Items 1 and 7 (they make existing rules mechanical and cost little), then 4.
2. Items 3 and 8 together (one pass over parameters and project guidance).
3. Item 2, in a quiet period between the long AMR branches.
4. Items 5 and 6 only after Q8.
