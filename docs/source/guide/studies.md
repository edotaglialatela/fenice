---
title: Studies, cost tiers and the analysis layer
nav: Studies and analysis
section: Guide
order: 75
---

## Cost tiers and the queue

Every validation study declares a tier in `cases/validation/run_all.py`: `quick` (under 2 minutes on one core), `medium`
(under 30 minutes on 4 cores) or `long` (hours, or MPI). `run_all.py --list` prints the tier and the estimated core-hours;
`--tier medium` selects the studies up to that tier (default: quick), `--all` adds the long ones.

* **Budget.** `--budget MINUTES` (or `$FENICE_STUDY_BUDGET`) makes a study run its simulations in chunks of steps with the
  checkpoint and `resume`; when the budget is spent the study exits with code 75 and is reported `PART`. The next call resumes
  from the checkpoint, bit for bit (checked in `check_budget.py` against an uninterrupted run).
* **Polite by default.** Study runs use 2 OpenMP threads (`--threads`, `$FENICE_STUDY_THREADS`) and, on Windows, below-normal
  priority.
* **Queue.** `python cases/validation/study_queue.py --tier medium --budget 120` runs the pending studies in cost order,
  skips those whose result is current (hash of the script, all case files and the solver executable), records every study in
  `results/validation/queue.json` (done / partial / failed) and can be killed at any time.
* **Estimate.** `fenice.budget.estimate(cells, steps, stages)` gives cell updates and core-hours from a cells-per-second
  constant, and the tier they fit; `fenice.budget.steps_for` gives the step count from the CFL condition.
* **Summaries.** `fenice.write_summary(study, data)` writes `results/validation/<study>/summary.json`.

## Strong-shock stress cases

`verification/stress_shocks.py` runs 1-D tubes (Sod with a 1000:1 pressure ratio, the 123 problem and a variant that opens a
vacuum, shock tubes at Mach 10, 20 and 100), Mach 10 flow past the immersed cylinder and Quirk's odd-even test (a Mach 6 shock whose
centre row starts one cell ahead), each with 1st-order Godunov and with MUSCL (MC) + SSP-RK2, HLLC. The criterion is robustness:
the run finishes, every cell is finite with $\rho, p > 0$, and for the tubes the change of mass, momentum and energy equals the
time integral of the far-field boundary fluxes to $10^{-10}$. Every run finishes with positive density and pressure and
conserves to $10^{-13}$ or better (the smallest pressure in the vacuum run is $10^{-4}$ with MUSCL): the solver needs no
positivity protection at these settings. The odd-even test is the one finding: with first-order HLLC the shock front is kinked,
its position differing between rows by 9.6 cells (not only in the perturbed row), while with MUSCL the rows agree to 0.2 cells.
The 16 runs meet the robustness criterion; the odd-even spread is reported, not judged.

**Positivity.** MUSCL reconstructs the primitive variables with a TVD limiter, so each reconstructed density and pressure lies
between the two neighbouring cell values: with positive cells the reconstructed face states are positive by construction, and a
fallback to first order at those faces could never act. What remains is the update itself, which B0 did not break.

**Shock-face dissipation (`shock_blend`).** A face whose pressure jump `abs(pR - pL) / min(pL, pR)` exceeds `shock_blend`
(default 0.2) blends its HLLC flux toward HLL, fully at twice the value; HLL has no contact or shear wave to restore, which
is what lets row-to-row differences along a shock grow. The sensor reads the face's own two states, so the flux is still a
function of (UL, UR): no halo, no extra stencil, the same code in the uniform, adaptive and MPI paths and in the AVX2 twin
(bit-identical to the scalar one). `verification/shock_blend.py` gates it: the odd-even spread falls from 9.6 to 0.5 cells
(first order) and from 0.2 to 0.05 (MUSCL), the Sod L1 error grows by 1.0 % and 2.1 %, and all 12 strong-shock tube runs still
pass. A sweep over 0.2 to 5 showed that only 0.2 brought the first-order spread under one cell; larger thresholds leave the
shock-face smear partly undissipated. The price: on meshes so coarse that a smooth flow changes pressure by 20 % per cell the
sensor fires there too (the tiny regression meshes of the vortex and manufactured solutions moved by 1e-3 to 1e-1); on the
validation meshes the observed orders are unchanged.

## Analysis layer (Python, no solver change)

NumPy only; SciPy is optional (Sobol sequences). Each module is verified against exact answers in
`cases/validation/check_analysis.py`.

* `fenice.uq`: `grid`, `latin_hypercube`, `saltelli` / `sobol_indices` (first-order and total indices; Ishigami function
  reproduced to 0.03 with 20000 random points, to under 0.001 with Sobol points), `run_sweep` (parallel, cached by the hash of case file,
  overrides and solver; one `summary.json` per run). A sampler is any function `sampler(n, d, seed) -> array (n, d)`: a sampler from an external UQ package is used by passing
  that function, and the package itself is never imported or required (the Saltelli estimator is a few lines, checked against the
  analytic Ishigami indices).
* `fenice.modal`: `load_snapshots` (a `.pvd` series of `.vtr` or adaptive `.vtu` files, the latter resampled to the finest
  level; `tol` and the reported `jitter` for series that are uniform in time only to about $10^{-3}$), `pod` (snapshot method,
  weighted), `dmd` (exact), `delay_embed` (time-delay DMD of a scalar record such as a lift signal), `export_modes_vtr`. DMD
  recovers the exponents of a travelling wave, of a damped oscillation and of two damped oscillations in one delay-embedded
  record to $10^{-14}$; POD recovers the energies of two orthogonal modes to $10^{-15}$. Snapshots must be sampled above the
  Nyquist rate of the dynamics (more than two per period): the application on the cylinder wake
  (`benchmarks/cylinder_modal.py`, [cylinder page]({{root}}cases/cylinder.html)) uses the run with 3.15 snapshots per period.
* `fenice.spectra`: `psd` (Welch, Hann, detrend) with `peaks` and `strouhal`, `resample_uniform`, `energy_spectrum` (shell
  averaged, Parseval exact; the Taylor-Green field has all its energy in one shell), `helmholtz` (solenoidal / dilatational split).

Not provided: spectra of non-periodic or adaptive-mesh data, derived quantities (dissipation spectrum, micro-scale), and
the 3-D Taylor-Green spectrum at peak dissipation.
