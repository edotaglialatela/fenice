# Benchmarks, robustness, UQ and spectral analysis: what is left

Status: the plan is implemented (the infrastructure for tiered, budgeted, resumable studies, `.github/workflows/benchmarks.yml` weekly and only when
the sources changed; the benchmarks A1 2-D Riemann problems, A2 blast waves, A5 Kelvin-Helmholtz, A7 Noh and Sedov; the strong-shock suite
`stress_shocks.py` and the `shock_blend` flux blend; the UQ layer `fenice.uq` and the modal layer `fenice.modal`; the spectral post-processing).
Each study has its bands fixed in its script and a page under `docs/source/`. What follows is what is open; the choices are the owner's.

## Benchmarks still to build

| Step | Case / study | Check against | Tier | Notes |
|---|---|---|---|---|
| A3 | Forward-facing step, Mach 3 | qualitative features, flow-through mass balance | medium | needs a solid corner: an IBM box or a masked region, to be checked first |
| A4 | Double Mach reflection | shock foot position along the wall (known analytically), conservation | medium | blocked: the inflow boundary has to follow a moving shock (boundary values depending on x and t; they accept expressions of t and tables today) |
| A6 | Rayleigh-Taylor | linear growth rate, bubble front against time | medium | the body force exists |
| A8 | 3-D compressible Taylor-Green at M = 0.3-0.5 | the incompressible Taylor-Green DNS already in `taylor_green_3d.py` (dissipation-rate curve), plus the compressibility drift | long (96^3, MPI) | no new external data |
| A9 | Decaying compressible turbulence (box, M_t about 0.1-0.5) | kinetic-energy decay and spectrum slope; self-convergence | long | with the spectra of A8 it decides whether a low-dissipation / higher-order reconstruction is needed: **decided to wait** for this run before opening that question |

Each step ends with a case file, a study script in `verification/` or `benchmarks/`, an entry in `run_all.py` with its tier, a docs page under
`docs/source/cases/` and, when the case is cheap and deterministic, a reference state in `cpp/tests/reference/` (`--create-missing`). Every step
either passes its fixed bands or produces a written finding.

## Other open items

* **Spectral post-processing.** The dissipation spectrum is per shell with the shell centre as |k|; a transfer-free refinement would need the modes'
  exact |k| per shell.
* **Robustness, only if a case asks for it.** A positivity-preserving update (Zhang-Shu type scaling): the remaining risk is in the time update, not in
  the reconstruction. No positivity code exists on purpose (the TVD limiter on primitive variables keeps rho and p between neighbours; WENO5-Z with
  SSP-RK3 needs CFL 0.3 on the near-vacuum tube, see the guide).
* **MUSCL at a transmissive outflow with `boundary_order = 2`** (manufactured outflow solution), left as it is: the order falls to 1.0-1.2
  between n = 64 and 128 with every limiter (Minmod, Van Leer, MC, Koren: they give a zero slope in the last cell, whose ghost neighbour is its own
  mirror), while WENO3-Z / WENO5-Z keep 2-2.5 and MUSCL on the walls case reaches 2. The error is a layer growing toward the outflow and building up
  in time (Van Leer, last four columns: order 2.0 at t = 0.02, 1.2 at 0.1, 1.0 at 0.5); the cause is not traced to a line of code. If it is picked up:
  change the slope of the last cell or the ghost extension and compare against the same table
  (`mms_boundary.py --schemes muscl --euler-only --boundary-order 2`).
* **Not planned:** running the fault-injection study inside the queue (it edits the sources and rebuilds: a disposable checkout only); LES /
  turbulence models, moving immersed bodies, a GPU backend, five-equation multiphase (each its own proposal).
