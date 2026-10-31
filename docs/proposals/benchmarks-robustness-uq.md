# Compressible benchmarks, strong-shock robustness, UQ and modal analysis: one plan

Status: proposal, nothing implemented. Three independent tracks that can run in parallel; the order inside each track is fixed here,
the choices that are the user's (which limiter, which UQ back-end) are listed as open points and not decided.

Tracks:

* **A. Benchmark suite** (new compressible test cases, cost-tiered so that nothing big has to run while the workstation is busy).
* **B. Robustness at strong shocks** (opt-in positivity and carbuncle protection).
* **C. Python-only analysis layer**: parameter sweeps / UQ (C1) and modal decomposition of snapshots (C2).

A and C touch no solver code. B changes the solver but only behind new options, so every existing reference stays bit for bit
(`regression.py` ALL MATCH is the gate of every B step).

## Constraint: the workstation is often busy

The tests of track A must not need an idle machine at a time nobody can predict. Rules for the whole plan:

1. **Cost tiers.** Every study declares one of `quick` (under 2 min, one core), `medium` (under 30 min, 4 cores), `long` (hours or
   MPI). This extends the existing `long?` flag of `run_all.py` (a tuple field, or a dict with `tier` and `est_core_hours`).
   `run_all.py --list` prints the tier and the estimate; `--tier quick|medium|long` selects up to that tier. Default stays
   "quick only".
2. **Estimates before running.** `fenice --case X --set ... --dry-run-cost` (or a Python helper using `nx*ny*nz`, the CFL step
   count and a measured cells-per-second constant stored in the study) prints cells x steps and the estimated core-hours, so a
   study can be refused or postponed without starting it.
3. **Resumable.** Every medium/long run uses the existing checkpoint and hot start (`*_restart.npz`, resumes bit for bit) and a
   `--budget MINUTES` flag: the runner stops cleanly at the next checkpoint when the budget is spent and the next call continues
   from it. A half-finished study costs nothing but disk.
4. **Polite by default.** The runner starts the solver with `--threads N` (default 2, `FENICE_STUDY_THREADS`) and on Windows
   below-normal priority (`start /belowNormal` / `psutil.nice`), so the interactive work of the machine is not starved.
5. **A queue script.** `python cases/validation/queue.py [--tier medium] [--budget 120]` runs the pending studies in cost order,
   skips the ones whose cached result is current (hash of case file + solver version), and writes `results/validation/queue.json`
   (done / partial / pending, minutes used). It can be started at night or from a scheduled task and killed at any time.
6. **Off-machine.** The medium tier also gets a `workflow_dispatch` + nightly-schedule job in a new
   `.github/workflows/benchmarks.yml` (Linux x64), uploading `results/validation/*/summary.json` as an artifact. The long tier
   stays local. CI of every push keeps its current content (unit tests, regression, `amr_mpi --check`, `ibm_ghost --check`, MMS quick)
   and only gains the quick tier of track A (budget: under 3 minutes in total).
7. **Bands fixed beforehand**, as in `verification/mms.py`: each study states its acceptance numbers in the script before the first
   run; they are never fitted to the result. A band that turns out wrong is changed in a separate commit with the reason.
8. **Results are summarised, not committed.** Each study writes `results/validation/<study>/summary.json` (via `fenice.summary`);
   only the figures of `--docs` go to `docs/img/`.

## Track A: benchmark suite

Why: the validation set is strong on smooth flow (Taylor-Green, vortex, MMS, cavity) and thin on discontinuous and unsteady
compressible flow, where a bug in limiting, Riemann solvers or AMR shows. The cases are files (`cases/<name>/case.toml`), so adding
one needs no code unless a profile is missing.

| Step | Case / study | Check against | Tier | Notes |
|---|---|---|---|---|
| A1 | 2-D Riemann problems (Lax-Liu configurations, 4 of the 19) on 200x200 | published density contours: shock positions and extrema, bands on integral quantities (mass, total energy conserved to roundoff, density min/max inside the physical range) | quick (100x100) / medium (400x400) | uniform vs AMR: compare with the existing uniform-vs-AMR machinery |
| A2 | Woodward-Colella blast waves (1-D, 400 cells and a fine reference of the same code) | self-convergence plus conservation; positions of the contact and shocks from the Rankine-Hugoniot relations | quick | needs reflecting walls on both sides (exists) |
| A3 | Forward-facing step, Mach 3 | qualitative features + conservation, flow-through mass balance | medium | needs a solid corner: IBM box or a masked region, to be checked first |
| A4 | Double Mach reflection | shock foot position along the wall (known analytically), conservation | medium | inflow boundary with a moving shock: time-dependent boundary values; blocked by a missing capability, see open points |
| A5 | Kelvin-Helmholtz (smooth and sharp interface) | growth rate of the linear regime (analytic), L1 of the conserved mass | quick / medium | periodic; a smooth-initial-condition variant avoids a grid-dependent seed |
| A6 | Rayleigh-Taylor | linear growth rate, bubble front vs time | medium | needs a body force (exists) |
| A7 | Noh implosion and Sedov blast | exact self-similar solutions (`fenice.exact`) | quick (2-D Sedov on 200^2) | exposes wall heating and loss of symmetry; Sedov also tests AMR |
| A8 | 3-D compressible Taylor-Green at M = 0.3-0.5 | the incompressible TG DNS already in `taylor_green_3d.py` (dissipation rate curve), plus the compressibility drift | long (96^3, MPI) | the DNS reference exists in the repo; no new external data |
| A9 | Decaying compressible turbulence (box, M_t about 0.1-0.5) | kinetic-energy decay and spectrum slope; public DNS (JHTDB) only if the licence and size allow, otherwise self-convergence | long | first benchmark that needs the higher-order scheme to be fair; may become a reason for it (see Not in this plan) |

Order: A2, A7, A5, A1 (quick tiers first, they also feed track B), then A3/A6, A4 only once the boundary question is answered, A8 and
A9 last and only when the machine is free (queue script). Each step ends with: case file, study script in `verification/` or
`benchmarks/`, entry in `run_all.py` with its tier, a docs page under `docs/source/cases/` (no history, current solver only) and,
when the case is cheap and deterministic, a reference state in `cpp/tests/reference/` added with `--create-missing`.

Acceptance of the track: every step either passes its fixed bands or produces a written finding (like the MMS findings), which is
then an input of track B.

## Track B: robustness at strong shocks

Why: the HLLC + MUSCL path has no positivity protection and no carbuncle/shock-instability fix. Tests A1/A3/A7 (and anything at
Mach 10+ or with near-vacuum) are where it would fail. The track starts with a failing test, not with a method.

* **B0. Stress cases (quick tier).** Strong-shock tests whose pass criterion is "finishes with rho > 0 and p > 0 and conserves
  mass/energy": Sod with 1000:1 pressure ratio, the 123 problem (two rarefactions, near vacuum), Mach 10 and Mach 20 shock tube,
  Mach 10 flow over the IBM cylinder, the odd-even decoupling test (a planar shock on a grid with a tiny perturbation, Quirk). Record
  which fail today. These cases are the deliverable even if nothing else is built.
* **B1. Diagnose.** For each failure: where the negative value first appears (reconstruction, Riemann state, update, IBM reset),
  `FENICE_*` debug switch, no behaviour change.
* **B2. Options, one at a time, each default off.** Candidates, to be chosen by the user after B1:
  * reconstruct in primitive variables with a first-order fallback in cells where the reconstructed rho or p would be non-positive
    (cheapest, local, easy to keep bit-identical when never triggered);
  * a positivity-preserving flux limiter (Zhang-Shu type scaling of the update);
  * a carbuncle cure: pressure-based switch to a more dissipative flux (HLL or a Rusanov blend) at shock-aligned faces.
  Each lands as a `SimParams` key (+ `section_table()` entry + parameters.md row), in the scalar code first, then the AVX2 twin
  (bit identical to scalar, no FMA, `riemann_batch_equals_scalar_bit_for_bit`-style test), then AMR (face list) and MPI (halo through
  `face_dependencies`).
* **B3. Gates.** With the option off: regression ALL MATCH, `amr_mpi --check` and golden hashes unchanged. With it on: the B0 cases
  pass, the smooth cases (vortex, TG, MMS) keep their observed orders (the limiter must not trigger), and serial == MPI == AMR-MPI
  bitwise.

## Track C: Python analysis layer (no solver change)

Both parts live in `python/fenice/` and use only `fenice.run()` and the readers; optional heavy dependencies are optional
(`prerequisites.py` gains an extras group), the core package stays importable without them.

* **C1. Sweeps and UQ** (`fenice.uq`).
  * C1a: a sweep runner: parameter grid or sample set -> case overrides (`--set`) -> runs in parallel under a thread/core budget
    (same polite rules as above, same cached-result hash) -> one table (`summary.json` per run, quantities of interest from the
    monitor CSV or the final state).
  * C1b: sampling and indices with no extra dependency (Latin hypercube, Sobol sequence via `scipy.stats.qmc` if present, Saltelli
    estimator for first-order/total Sobol indices) and an optional adapter for UQpy / Dakota left as a thin function.
  * C1c: one worked study in `cases/validation/features/`: sensitivity of the cylinder Cd at Re 40 to inlet speed, resolution and
    limiter; a second, cheap one on Sod (L1 error vs CFL, limiter, resolution). Quick tier on small meshes, the larger sweep
    medium.
  * Optimisation (OpenMDAO) is deliberately out: it needs gradients or many runs; revisit after C1.
* **C2. Modal analysis** (`fenice.modal`).
  * C2a: a snapshot reader over a directory of `.vtr` / `.vtu` / `.npz` (uniform meshes first; AMR snapshots are resampled to the
    finest level, with the cost stated), returning a matrix (points x snapshots) of a chosen field, with cell-volume weights.
  * C2b: POD (SVD, snapshot method) and exact DMD in plain NumPy, SPOD behind the same interface later; mode export to `.vtr` so
    the workbench Postprocessing node can display modes.
  * C2c: a verification with a known answer before any physical use: DMD on a travelling wave and on a decaying standing wave
    (exact eigenvalues), POD on a sum of two orthogonal modes (exact energies). Then one application: cylinder shedding at Re 100
    (Strouhal from the DMD frequency vs from the lift signal; agreement within the bands set beforehand).
  * Needs a shedding run: medium tier, produced by the queue script, snapshots written with the existing output interval.

## Parallel schedule

The three tracks share only the tier machinery (rules 1-8 above), which therefore comes first and is small:

1. **Week 0 (infrastructure, both A and C use it):** tier field and `--list` columns in `run_all.py`, `--budget` / `--threads`
   on the study helper, `queue.py`, summary.json writer, cost estimate. Test: a fake slow study is stopped by the budget and resumed.
2. **Then in parallel, each on its own branch** (`benchmarks`, `robustness`, `analysis`): A2/A7/A5/A1 and B0 share the same quick
   cases (build B0 and A2/A7 once); C1a-b and C2a-c need no solver build at all and can proceed while the workstation is busy
   (only C2's application run and C1c's larger sweep go to the queue).
3. **Gates before any merge:** `.\cpp\build.ps1 -Test`, `python cpp\tests\regression.py --mpi 4` ALL MATCH, `python
   cases\validation\check_all.py`. Track B additionally the golden hashes. Nothing in A or C changes a solver byte.

## Open points (the user's)

* B2: which of the three protections, and whether any default may ever be turned on (it would regenerate references).
* A4 (double Mach reflection) needs a boundary value that depends on x and t (moving shock at the top boundary). The solver has time
  `t` only in source terms; adding it to boundary expressions is a small feature of its own, to be decided.
* A9 would be the natural argument for a low-dissipation / higher-order reconstruction; that is a separate proposal if A9 shows
  MUSCL is too dissipative. Not decided here.
* C1b: dependency policy (NumPy/SciPy only, or also UQpy).
* Whether the medium tier may use the GitHub nightly runner (cost/minutes), or stays local.

## Not in this plan

LES / turbulence models, moving immersed bodies, GPU backend, five-equation multiphase (see the candidate list in the session
that produced this plan); each would be its own proposal once this plan has produced its first findings.
