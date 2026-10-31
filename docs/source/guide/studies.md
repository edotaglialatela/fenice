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
time integral of the far-field boundary fluxes to $10^{-10}$. **All 16 runs pass** (conservation errors $10^{-14}$ to
$10^{-15}$; the smallest pressure in the vacuum run is $10^{-4}$ with MUSCL). The odd-even test is reported, not judged: the
shock position differs between rows by 10 cells with Godunov and by 1 cell with MUSCL.

## Analysis layer (Python, no solver change)

NumPy only; SciPy is optional (Sobol sequences). Each module is verified against exact answers in
`cases/validation/check_analysis.py`.

* `fenice.uq`: `grid`, `latin_hypercube`, `saltelli` / `sobol_indices` (first-order and total indices; Ishigami function
  reproduced to 0.03 with 20000 random points, to under 0.001 with Sobol points), `run_sweep` (parallel, cached by the hash of case file,
  overrides and solver; one `summary.json` per run). A sampler is any function `sampler(n, d, seed) -> array (n, d)`.
* `fenice.modal`: `load_snapshots` (a `.pvd` series of `.vtr` or adaptive `.vtu` files, the latter resampled to the finest
  level), `pod` (snapshot method, weighted), `dmd` (exact), `export_modes_vtr`. DMD recovers the exponents of a travelling wave and of
  a damped oscillation to $10^{-14}$; POD recovers the energies of two orthogonal modes to $10^{-15}$.
* `fenice.spectra`: `psd` (Welch, Hann, detrend) with `peaks` and `strouhal`, `resample_uniform`, `energy_spectrum` (shell
  averaged, Parseval exact; the Taylor-Green field has all its energy in one shell), `helmholtz` (solenoidal / dilatational split).

Not provided: spectra of non-periodic or adaptive-mesh data, derived quantities (dissipation spectrum, micro-scale), and
the application runs (cylinder shedding at Re 100 against the lift signal, the 3-D Taylor-Green spectrum at peak dissipation).
