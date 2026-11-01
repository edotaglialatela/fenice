# Compressible benchmarks, strong-shock robustness, UQ, modal and spectral analysis: one plan

Status (2026-11-01): mostly implemented and merged; what is done and what is left is in "Status and leftover work" below. Four
independent tracks that can run in parallel; the order inside each track is fixed here, the choices that are the user's (which
limiter, which UQ back-end) are listed as open points, with the ones decided so far marked.

Tracks (D, spectral post-processing, is added at the end of the track C section):

* **A. Benchmark suite** (new compressible test cases, cost-tiered so that nothing big has to run while the workstation is busy).
* **B. Robustness at strong shocks** (opt-in positivity and carbuncle protection).
* **C. Python-only analysis layer**: parameter sweeps / UQ (C1) and modal decomposition of snapshots (C2).

A and C touch no solver code. B changes the solver but only behind new options, so every existing reference stays bit for bit
(`regression.py` ALL MATCH is the gate of every B step).

## Status and leftover work (2026-11-01)

**Done** (merged on main; every study has its bands fixed in its script and a page under `docs/source/`):

* Infrastructure (rules 1-8): tiers and `run_all.py --list`, cost estimate (`fenice.budget`), budgeted resumable runs
  (`check_budget.py`: resumed == uninterrupted, bit for bit), `--budget` / `--threads` on every study script, `study_queue.py`,
  `summary.json`, polite threads and priority, `.github/workflows/benchmarks.yml` (nightly, only when the sources changed, plus
  `workflow_dispatch`; results cached so a stopped study resumes). Measured: quick tier 6.6 min, medium +35 min (cavity 32); the
  per-push CI does not run them.
* Track A: A1 (`riemann_2d`, quick and `--full`, AMR against uniform), A2 (`blast_waves`), A5 (`kelvin_helmholtz`), A7 (`implosions`:
  Noh exact, Sedov scaling).
* Track B: B0 (`stress_shocks`: pressure ratio 1000, 123 with and without vacuum, Mach 10 / 20 / 100, Mach 10 cylinder, odd-even),
  B2 as a face-local HLLC to HLL blend (`shock_blend`, default 0.2, AVX2 twin bit-identical, `shock_blend.py` gate), B3 gates
  (regression, `amr_mpi`, unit tests; 16 references regenerated deliberately). Positivity code was left out on purpose: the TVD
  limiter on primitive variables keeps rho and p between neighbours, so a fallback in the reconstruction could never act.
* Track C: C1a-b (`fenice.uq`: grid, LHS, Sobol via SciPy when present, Saltelli / Jansen indices, cached sweeps), C2a-c
  (`fenice.modal`: load, POD, exact DMD, `delay_embed`, `.vtr` export; known-answer tests) and the C2 application
  (`benchmarks/cylinder_modal.py`, from existing runs: DMD St 0.1748 against 0.1746 from the lift spectrum).
* Track D: D1 (PSD, peaks, Strouhal), D2 (shell-averaged E(k), Helmholtz split), D5 for those (sine, Parseval, Taylor-Green
  initial spectrum, gradient / divergence-free fields) and the Re 100 lift against DMD.
* Prerequisites: SciPy extras group (`prerequisites.py`).

**Leftover, in the order I would take it:**

1. *Small fixes.*
   * Register `cylinder_low_blockage` in `run_all.py` (long tier): `cylinder_modal.py` needs its runs and tells the user to make
     them with `run_all.py cylinder_low_blockage`, which is not a known study yet.
   * Dispatch `Benchmarks` once on GitHub (the workflow parses but has not run there).
   * Re-run the Windows CI job that failed in `check_geometry` (1200 x 1800 supersonic cylinder, "solver None"; the script now
     prints the exit code and stderr, so the next failure says why).
   * Regression references (`--create-missing`) for the new cheap, deterministic cases (blast waves, Kelvin-Helmholtz, Noh,
     the 2-D Riemann configurations): none added yet.
   * Golden hashes of B3: the comparison of `shock_blend = 0` against the binary before the change (the four `amr_mpi.py --check`
     cases and the 720 x 360 airfoil, 1 and N threads, also `FENICE_NO_SIMD=1` / `FENICE_FULL_REBUILD=1`) was not run; the
     regression, unit tests and `amr_mpi` pass.
2. *Track C.*
   * C1c, the worked studies: cylinder Cd at Re 40 against inlet speed, resolution and limiter (quick on small meshes, the larger
     sweep medium), and a cheap Sod one (L1 against CFL, limiter, resolution).
   * Confidence intervals of the Sobol indices (a bootstrap in NumPy; `sobol_indices` returns point estimates only).
3. *Track D.*
   * D3: spectra of non-periodic data (window or even extension) and of adaptive-mesh data (resample to the finest level over a
     region, cost printed first, a leakage warning in the output).
   * D4: dissipation spectrum, enstrophy, Taylor micro-scale, integral scale, numerical-dissipation estimate from the departure
     of E(k) from the initial Taylor-Green spectrum.
   * D5, what remains: serial and MPI snapshots give identical spectra; the application to the 3-D Taylor-Green run (spectrum at
     the time of peak dissipation).
4. *Track A, benchmarks still to build.*
   * A3 forward-facing step Mach 3 (first check how to get the solid corner: IBM box or masked region).
   * A6 Rayleigh-Taylor (body force exists).
   * A8 3-D compressible Taylor-Green at M = 0.3-0.5 (96^3, MPI, long).
   * A9 decaying compressible turbulence (long); its run, with D2 / D4, is what decides the low-dissipation question below.
   * A4 double Mach reflection stays blocked on the boundary capability (open point below).
5. *Track B, only if a case asks for it.* B1 beyond the odd-even diagnosis (no stress case failed, so there was nothing else to
   trace) and a positivity-preserving update (Zhang-Shu type scaling): the risk that remains is in the time update, not in the
   reconstruction.

Not planned: running the fault-injection study inside the queue (it edits the sources and rebuilds: a disposable checkout only).

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
| A9 | Decaying compressible turbulence (box, M_t about 0.1-0.5) | kinetic-energy decay and spectrum slope; public DNS data only if the licence and size allow, otherwise self-convergence | long | first benchmark that needs the higher-order scheme to be fair; may become a reason for it (see Not in this plan) |

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
    estimator for first-order/total Sobol indices) and a thin function through which an external sampler can be plugged in.
  * C1c: one worked study in `cases/validation/features/`: sensitivity of the cylinder Cd at Re 40 to inlet speed, resolution and
    limiter; a second, cheap one on Sod (L1 error vs CFL, limiter, resolution). Quick tier on small meshes, the larger sweep
    medium.
  * Optimisation is deliberately out: it needs gradients or many runs; revisit after C1.
* **C2. Modal analysis** (`fenice.modal`).
  * C2a: a snapshot reader over a directory of `.vtr` / `.vtu` / `.npz` (uniform meshes first; AMR snapshots are resampled to the
    finest level, with the cost stated), returning a matrix (points x snapshots) of a chosen field, with cell-volume weights.
  * C2b: POD (SVD, snapshot method) and exact DMD in plain NumPy, SPOD behind the same interface later; mode export to `.vtr` so
    the workbench Postprocessing node can display modes.
  * C2c: a verification with a known answer before any physical use: DMD on a travelling wave and on a decaying standing wave
    (exact eigenvalues), POD on a sum of two orthogonal modes (exact energies). Then one application: cylinder shedding at Re 100
    (Strouhal from the DMD frequency vs from the lift signal; agreement within the bands set beforehand).
  * Needs a shedding run: medium tier, produced by the queue script, snapshots written with the existing output interval.

## Track D: spectral post-processing (Python, no solver change)

Why: C2 gives modes, but the first questions asked of a turbulent or unsteady run are spectral: how is the energy spread over
wavenumbers (is the inertial range there, where does the numerical dissipation bite), and at which frequencies does a signal
oscillate. A9 and A8 cannot be judged without it. `fenice.spectra`, NumPy FFT only, same readers and the same cost rules as C.

* **D1. Signals in time.** Power spectral density of probe, monitor and load series (lift, a probe velocity): detrend, window
  (Hann), averaged segments with overlap, peak finder with a frequency resolution stated; Strouhal number as a one-line helper.
  Series sampled at a variable step are first resampled onto a uniform time grid (the resampling and its error stated).
* **D2. Fields in space, uniform periodic meshes.** Shell-averaged energy spectrum E(k) of the velocity (2-D and 3-D), spectra of
  density and pressure, and for compressible flow the solenoidal / dilatational split (Helmholtz decomposition in Fourier space).
  Conventions fixed in the module docstring: wavenumber binning, normalisation so that the sum of E(k) over shells equals the mean
  kinetic energy (Parseval), the Nyquist cut, the density-weighted velocity option.
* **D3. Non-periodic and AMR data.** A window or an even extension before the transform on non-periodic directions, a line or a
  sub-box sample instead of the whole domain; AMR fields resampled to the finest level over the region asked for (cost printed
  first, same budget rules). Spectra of such data carry a warning in their output about leakage.
* **D4. Derived quantities.** Dissipation spectrum (k^2 E(k) for the incompressible part), transfer-free diagnostics only
  (no closure): enstrophy, Taylor micro-scale, integral scale, the numerical-dissipation estimate from the departure of E(k) from
  the exact initial spectrum of a Taylor-Green run.
* **D5. Verification before use** (answers known beforehand, bands fixed): a single sine wave gives one peak at the exact
  frequency / wavenumber and the exact amplitude; Parseval holds to roundoff for D1 and D2; the Taylor-Green initial field has the
  analytic spectrum (energy only in the lowest shell); the solenoidal part of a pure gradient field is zero and the dilatational
  part of a divergence-free field is zero to roundoff; serial and MPI snapshots give identical spectra. Application: the 3-D
  Taylor-Green run of `taylor_green_3d.py` (spectrum at the time of peak dissipation) and the cylinder lift series at Re 100
  (Strouhal against C2's DMD frequency).
* **D6. In-solver spectra (not planned).** A spectrum written during the run needs a distributed transform on the strips (halo and
  transposes) that must give the same bits for any rank count; the cost is large and nothing needs it. Offline from snapshots
  and from the probe files of `postprocessing-and-species.md` A.1 is enough; revisit only if snapshot I/O becomes the limit.

Order: D1 and D5 for D1 (needs only a CSV), then D2 on a periodic snapshot, then D4, D3 last. Tier: all quick except the
3-D application (medium, on the existing 96^3 data if the cache holds it).

## Parallel schedule

The four tracks share only the tier machinery (rules 1-8 above), which therefore comes first and is small:

1. **Week 0 (infrastructure, both A and C use it):** tier field and `--list` columns in `run_all.py`, `--budget` / `--threads`
   on the study helper, `queue.py`, summary.json writer, cost estimate. Test: a fake slow study is stopped by the budget and resumed.
2. **Then in parallel, each on its own branch** (`benchmarks`, `robustness`, `analysis`): A2/A7/A5/A1 and B0 share the same quick
   cases (build B0 and A2/A7 once); C1a-b, C2a-c and D1-D2 need no solver build at all and can proceed while the workstation is busy
   (only C2's application run and C1c's larger sweep go to the queue).
3. **Gates before any merge:** `.\cpp\build.ps1 -Test`, `python cpp\tests\regression.py --mpi 4` ALL MATCH, `python
   cases\validation\check_all.py`. Track B additionally the golden hashes. Nothing in A or C changes a solver byte.

## Open points (the user's)

* B2: **decided** (2026-10-31): the carbuncle / odd-even protection is the `shock_blend` flux blend, on by default at 0.2
  (references regenerated); no positivity code unless a case fails.
* A4 (double Mach reflection) needs a boundary value that depends on x and t (moving shock at the top boundary).
  `postprocessing-and-species.md` A.4 already proposes expressions of `t` for boundary values; A4 additionally needs x. Open.
* A9 would be the natural argument for a low-dissipation / higher-order reconstruction; that is a separate proposal if A9 shows
  MUSCL is too dissipative. **Decided to wait** for a larger turbulence-decay run (A8 / A9) before opening it.
* C1b: **decided**: NumPy core, SciPy optional (Sobol sequences), no external UQ package; one is used by passing its sampler
  function.
* Whether the medium tier may use the GitHub nightly runner: **decided**: yes, nightly and gated by a change of the sources (see
  Status); the long tier stays local.

## Not in this plan

LES / turbulence models, moving immersed bodies, GPU backend, five-equation multiphase (see the candidate list in the session
that produced this plan); each would be its own proposal once this plan has produced its first findings.
