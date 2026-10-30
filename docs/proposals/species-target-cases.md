# Multi-species and reacting flow: target test cases

Status: proposal, nothing implemented. It fixes what "done" means before any code is written; the implementation choices of part B of
`postprocessing-and-species.md` (species storage, interface treatment, chemistry splitting, source of the thermochemistry) stay the
user's and are not decided here. Numbers marked *proposed* are starting values, to be confirmed before the first run (the convention of
`verification/mms.py`: bands are fixed beforehand, never fitted to the result).

## Aim

A reacting compressible flow that exercises everything the species work adds, with a reference that does not come from this solver:
a one-dimensional H2/O2/Ar detonation (case **T**), reached through two smaller cases that isolate its two new ingredients, the
mixture thermodynamics (**G1**) and the chemistry (**G2**). A case that cannot be checked against something independent is not a test.

Why a detonation, not a flame: it is compressible Euler plus stiff chemistry, which is what the solver already is (explicit time step
from the speed of sound, no species diffusion, no low-Mach treatment), and AMR has a real job on it (a reaction zone a fraction of a
millimetre thick inside a smooth flow). A laminar premixed or diffusion flame needs mixture-averaged transport and a low-Mach
formulation; it is out of scope here.

## The cases

### G1. Inert material interface (thermodynamics only)

One dimension, periodic, two gases of different molar mass and specific heats separated by a contact, uniform pressure `p0` and
velocity `u0`, advected one or more periods. Three variants, in this order of difficulty:

1. two calorically perfect gases (gamma depends on the mixture only through `Y`);
2. species with temperature-dependent `cp` (polynomial thermodynamics, as the chemistry will need);
3. the same with a smooth (tanh) and a sharp interface.

Reference: exact. Pressure and velocity stay `p0`, `u0`; the species fields are the initial ones translated.

| Quantity | Pass (proposed) |
| --- | --- |
| `max |p - p0| / p0`, `max |u - u0| / u0` after the run | below a band fixed per variant before the run; variant 1 expected at roundoff or a documented small value |
| `Y_k` stays in [0, 1], `sum Y_k = 1` | to roundoff (the species fluxes are consistent with the mass flux) |
| total mass of each species | conserved to roundoff, also across AMR coarse-fine faces and rank interfaces |
| MPI, threads, AMR level 0 | bit for bit the serial / uniform run |

This is the gate for the interface question of part B: if variant 2 fails the band, the fix (quasi-conservative energy update,
pressure-equilibrium correction, or accepting and documenting the error) is the user's choice, and the case then records the choice.
An optional G1b, the two-gas Sod problem, needs `fenice.exact` extended to a different gamma on each side of the contact.

### G2. Zero-dimensional ignition (chemistry only)

A uniform periodic domain at rest (a few cells), so the flow does nothing and the state evolves as an adiabatic constant-volume
reactor. Stoichiometric H2/air (or H2/O2/Ar), `p0` = 1 atm, `T0` = 1000 K (proposed; one case on each side of the ignition
temperature range is better than one), a published nine-species hydrogen mechanism of about twenty reactions.

Reference: a constant-volume reactor of a thermochemistry library with the same mechanism and the same thermodynamic data, integrated
to a tolerance far below the bands.

| Quantity | Pass (proposed) |
| --- | --- |
| ignition delay (maximum `dT/dt`) | within 1 % of the reference |
| species and temperature histories | relative difference below a band in log time, before and after ignition |
| final state | the equilibrium state at fixed internal energy and volume, to the band of the integrator's tolerance |
| element mass, total energy, density | conserved to roundoff / to the integrator tolerance (stated) |
| MPI, threads | bit for bit (the chemistry of a cell never reads another cell) |

Note what G2 does not test: with a uniform state the flow step does nothing, so the splitting error is zero here. The integrator is
checked against the reference; the splitting is checked in T.

### T. One-dimensional detonation (everything together)

H2/O2/Ar 2:1:7 by moles, `T0` = 300 K, `p0` = 6.67 kPa (proposed; a mixture and a pressure for which the reaction zone is millimetres
and the wave is regular), the same mechanism as G2.

Reference, computed by a Python tool in `python/fenice/` and stored as a small table in `cases/validation/` (see the next section):
the Chapman-Jouguet speed `D_CJ` and state from equilibrium, and the steady ZND structure (the frozen state behind the shock, then the
reaction integrated along the steady flow equations): the pressure, temperature and species profiles against the distance from the
shock, the induction length `L_ind` (to the maximum temperature gradient) and the reaction length.

Setup, two variants (a choice, see the questions):

* **travelling wave**: closed tube, initiation by a small region of burnt or hot gas at the closed end, AMR following the wave. The most
  realistic, and the one that shows what AMR is for; the speed comes from the arrival times at probes (item A1 of the other proposal)
  or the front position, over the second half of the tube, after the initial overdriven stage has relaxed.
* **steady frame**: the unburnt gas enters at `D_CJ`, the wave is held in place; cheaper, and the profile can be compared point by point,
  but the position of the front is only marginally stable and the inlet needs the composition.

| Quantity | Pass (proposed) |
| --- | --- |
| `D / D_CJ` (mean speed, finest resolution) | within 1 % |
| `L_ind` against the ZND value | converges with refinement (5, 10, 20, 40 cells per `L_ind`, observed order and GCI through `fenice.convergence`), finest within 5 % |
| burnt state at the end of the reaction zone | the CJ state, within a band |
| von Neumann pressure peak | reported with the resolution, no pass criterion (it is resolution limited) |
| AMR against the uniform run at the finest resolution | same `D`, same profile within the band; the AMR run must be cheaper (cells and time reported) |
| MPI, threads | bit for bit |

Risk: a one-dimensional detonation can pulsate for some mixtures. The Ar dilution is chosen to keep it regular; if it does not, the
speed is the mean over a stated window and the case documents the amplitude. A later case T2, the two-dimensional cellular structure
(cell width against published ranges), is the natural AMR showcase; it has no pass criterion until T is closed.

## What each case requires

| Needed | G1 | G2 | T |
| --- | --- | --- | --- |
| species array `rho Y_k` through the face engines, AMR registers, halos, checkpoints | yes | (cells only) | yes |
| mixture thermodynamics (gamma, `cp(T)`, sound speed, temperature from energy) | yes | yes | yes |
| species flux of the Riemann solver | yes | no | yes |
| species in the initial condition and boundaries | yes | yes | yes |
| mechanism data (rates, third bodies, falloff) and a stiff per-cell integrator | no | yes | yes |
| coupling of the chemistry with the time step, also inside the AMR recursion | no | no | yes |
| probes or front tracking | no | no | yes |

The five-variable path is untouched when no species are present, so the 38 references do not move; the new cases get their own
references (`--create-missing`) after validation against the independent data, as for the manufactured solutions.

## Reference data

The reference values come from a thermochemistry library run once by a script (`python/fenice/` tool, like `fenice.mms`), recorded
with the library and mechanism versions, and committed as small tables next to the studies (as the Ghia data for the cavity). The
studies, CI and the docs build then read the tables and need no library; only regenerating them does. Whatever the answer on where the
solver's own thermochemistry comes from, the reference stays independent of it: it is the check that the solver's rates and
thermodynamic fits (read from a generated mechanism file, or linked) reproduce the library's.

## Order

1. Fix the cases above (this document) and generate the reference tables, with the bands.
2. Passive species with the species flux; G1 variant 1, then thermodynamics and G1 variants 2 and 3 (the interface decision).
3. Mechanism and integrator; G2.
4. Coupling with the flow step (and the AMR recursion); T with the uniform mesh, then with AMR.
5. T2, the cellular detonation.

Each stage ends with the full oracle set (unit tests, regression, `amr_mpi.py --check` extended to the new cases, bitwise MPI).

## Questions

1. Is the long-term target combustion, or also multiphase (a volume fraction instead of species)? G1 serves both; T and G2 only the
   first.
2. Travelling wave or steady frame for T (or both)?
3. Mixture and mechanism: H2/O2/Ar and a nine-species mechanism as proposed, or another fuel from the start?
4. Are G1 and G2 required to pass before T is started (proposed), or may T run in parallel with them?
5. The bands above are proposed, not measured: confirm or replace them before the first run.
