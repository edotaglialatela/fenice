# Multi-species and reacting flow: proposal

Status: proposal, not started; the choices below are the user's. The target cases fix what "done" means. Numbers marked *proposed*
are starting values, to be confirmed before the first run (the convention of `verification/mms.py`: bands are fixed beforehand, never
fitted to the result).

## Starting point

* `State` has five slots everywhere (Riemann solvers, AVX2 kernels, AMR registers, MPI buffers, checkpoints).
* The equation of state is one struct (ideal or stiffened gas) with its laws exposed separately (`pressure_law`,
  `internal_energy_law`, `transport_law`). The IBM solid reset uses the ideal-gas pressure formula whatever the EOS; any refactor of the
  EOS must keep that.
* Right-hand-side contributions (body force, expression sources) are `RhsSourceTerms` of the simulation, applied once per mesh / range;
  reactions would be one more term, with no virtual call inside a face or cell loop. Any change here is bitwise neutral without species
  (regression, golden dumps, `FENICE_NO_SIMD=1`).
* `[[probe]]` / `[[sample]]` (primitive variables of the containing cell, without interpolation, same bytes on any rank count, AMR
  included), `[[force]]` and `[[average]]` exist, and so do boundary values that are expressions of `t` or tables.

## Choices that are the user's

* **Boundary conditions:** a total-pressure inlet, and an inlet/outlet condition that switches with the sign of the normal velocity.
* **Species.** Keep the five flow variables and carry the species as a separate per-cell array `rho Y_k`. The cost is in the paths
  that move a state: `assemble`, `assemble_3d`, the AMR face engine and registers, the MPI halos, checkpoints, the IBM reset, the
  viscous fluxes. With no species the five-variable path is untouched and the references do not move. Choices: reconstruct `Y_k` or
  `rho Y_k`; the species flux of HLLC (`rho* Y_k` from the upwind side of the contact is the usual one); the sound speed (frozen mixture
  gamma).
* **Material interfaces.** A conservative scheme with a mixture gamma that varies with `Y` produces pressure and velocity errors at a
  contact even when they start uniform. The test that decides is case G1 below. If it fails, the fix is a choice (a quasi-conservative
  energy update, a pressure-equilibrium correction, or accepting and documenting the error).
* **Chemistry.** Operator splitting (Strang: half a chemistry step, the flow step, half a chemistry step); a stiff implicit integrator per
  cell, independent across cells, so the result does not depend on the thread or rank count. With AMR the half steps sit per level and
  per stage of the recursion, and conservative flux replacement must stay intact: it needs its own design section. The explicit time
  step follows the speed of sound, so this suits high-speed reacting flow (detonations, shock-induced ignition, supersonic reacting flow
  behind a body), not slow flames.
* **Where the thermochemistry comes from.** Either link a thermochemistry library behind an optional build switch, or generate a
  mechanism file with a Python tool (like `fenice.mms` does for the manufactured solutions) holding the polynomial coefficients,
  molecular weights and reaction data, which the solver reads. The second keeps the C++ build and CI unchanged and keeps the references
  independent of the library's version; the solver then needs its own rate evaluation (Arrhenius, third bodies, falloff) and integrator. The two are compared use by use in `thermochemistry-library.md`.

## Target cases

A reacting compressible flow with a reference that does not come from this solver: a one-dimensional H2/O2/Ar detonation (**T**),
reached through two smaller cases that isolate its two new ingredients, the mixture thermodynamics (**G1**) and the chemistry (**G2**).
A case that cannot be checked against something independent is not a test. A detonation, not a flame, because it is compressible Euler
plus stiff chemistry (explicit time step from the speed of sound, no species diffusion, no low-Mach treatment), and AMR has a real job on
it (a reaction zone a fraction of a millimetre thick inside a smooth flow); a laminar flame needs mixture-averaged transport and a
low-Mach formulation, out of scope here.

**G1. Inert material interface (thermodynamics only).** One dimension, periodic, two gases of different molar mass and specific heats
separated by a contact, uniform pressure `p0` and velocity `u0`, advected one or more periods: (1) two calorically perfect gases;
(2) species with temperature-dependent `cp` (polynomial thermodynamics); (3) the same with a smooth (tanh) and a sharp interface.
Reference: exact (`p0`, `u0` stay, the species fields are translated).

| Quantity | Pass (proposed) |
| --- | --- |
| `max |p - p0| / p0`, `max |u - u0| / u0` | below a band fixed per variant before the run; variant 1 at roundoff or a documented small value |
| `Y_k` in [0, 1], `sum Y_k = 1` | to roundoff |
| total mass of each species | conserved to roundoff, also across AMR coarse-fine faces and rank interfaces |
| MPI, threads, AMR level 0 | bit for bit the serial / uniform run |

If variant 2 fails its band, the fix is the user's choice (see Material interfaces) and the case records it. An optional G1b, the two-gas
Sod problem, needs `fenice.exact` extended to a different gamma on each side of the contact.

**G2. Zero-dimensional ignition (chemistry only).** A uniform periodic domain at rest (a few cells): an adiabatic constant-volume reactor.
Stoichiometric H2/air (or H2/O2/Ar), `p0` = 1 atm, `T0` = 1000 K (proposed; one case on each side of the ignition temperature range is
better than one), a published nine-species hydrogen mechanism of about twenty reactions. Reference: a constant-volume reactor of a
thermochemistry library with the same mechanism and thermodynamic data, integrated far below the bands.

| Quantity | Pass (proposed) |
| --- | --- |
| ignition delay (maximum `dT/dt`) | within 1 % of the reference |
| species and temperature histories | relative difference below a band in log time, before and after ignition |
| final state | the equilibrium at fixed internal energy and volume, to the integrator's tolerance |
| element mass, total energy, density | conserved to roundoff / to the integrator tolerance (stated) |
| MPI, threads | bit for bit (the chemistry of a cell never reads another cell) |

G2 does not test the splitting (a uniform state makes the flow step do nothing); the integrator is checked here, the splitting in T.

**T. One-dimensional detonation (everything together).** H2/O2/Ar 2:1:7 by moles, `T0` = 300 K, `p0` = 6.67 kPa (proposed; a regular wave
with a millimetre reaction zone), the mechanism of G2. Reference: a Python tool in `python/fenice/` stores a small table in
`cases/validation/`: the Chapman-Jouguet speed `D_CJ` and state from equilibrium, and the steady ZND structure (profiles against the
distance from the shock, induction length `L_ind` to the maximum temperature gradient, reaction length). Setup, two variants (a choice):
a **travelling wave** (closed tube, initiation by a small hot region at the closed end, AMR following the wave; speed from probe arrival
times (`[[probe]]`) or the front position over the second half of the tube; the realistic one and the one that shows what AMR is for) or a
**steady frame** (unburnt gas enters at `D_CJ`; cheaper and comparable point by point, but the front is only marginally stable).

| Quantity | Pass (proposed) |
| --- | --- |
| `D / D_CJ` (mean speed, finest resolution) | within 1 % |
| `L_ind` against the ZND value | converges with refinement (5, 10, 20, 40 cells per `L_ind`, order and GCI through `fenice.convergence`), finest within 5 % |
| burnt state at the end of the reaction zone | the CJ state, within a band |
| von Neumann pressure peak | reported with the resolution, no pass criterion |
| AMR against the uniform run at the finest resolution | same `D`, same profile within the band; the AMR run cheaper (cells and time reported) |
| MPI, threads | bit for bit |

Risk: a 1-D detonation can pulsate for some mixtures; the Ar dilution keeps it regular, otherwise the speed is the mean over a stated
window and the amplitude is documented. A later T2, the 2-D cellular structure, is the natural AMR showcase; no pass criterion until T
is closed.

| Needed | G1 | G2 | T |
| --- | --- | --- | --- |
| species array `rho Y_k` through the face engines, AMR registers, halos, checkpoints | yes | (cells only) | yes |
| mixture thermodynamics (gamma, `cp(T)`, sound speed, temperature from energy) | yes | yes | yes |
| species flux of the Riemann solver | yes | no | yes |
| mechanism data and a stiff per-cell integrator | no | yes | yes |
| coupling of the chemistry with the time step, also inside the AMR recursion | no | no | yes |
| front tracking (probes exist) | no | no | yes |

The reference values come from a thermochemistry library run once by a script, recorded with the library and mechanism versions, and
committed as small tables (as the Ghia data for the cavity): studies, CI and the docs build read the tables and need no library. The
reference stays independent of the solver's own thermochemistry, which it checks. The five-variable path is untouched without species,
so the 44 references do not move; the new cases get their own (`--create-missing`) after validation against the independent data.

## Order

(1) fix the cases and generate the reference tables with the bands; (2) passive species with the species
flux, G1 variant 1, then thermodynamics and G1 variants 2 and 3 (the interface decision); (3) mechanism and integrator, G2; (4) coupling
with the flow step and the AMR recursion, T on the uniform mesh and then with AMR; (5) T2. A transported scalar stays in [0, 1] and sums
to one under a stated CFL. Each stage ends with the full oracle set (unit tests, regression, `amr_mpi.py --check` extended to the new
cases, bitwise MPI).

## Questions

1. Is the long-term target combustion, or also multiphase (a volume fraction instead of species)? G1 serves both; T and G2 only the first.
2. Travelling wave or steady frame for T (or both)?
3. Mixture and mechanism: H2/O2/Ar and a nine-species mechanism as proposed, or another fuel from the start?
4. Must G1 and G2 pass before T is started (proposed), or may T run in parallel?
5. The bands above are proposed, not measured: confirm or replace them before the first run.
