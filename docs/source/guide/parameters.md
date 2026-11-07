---
title: Parameters
nav: Parameters
section: Guide
order: 20
header: Params.hpp
---

Every run is described by the same set of `key = value` parameters. They can
come from a test case's file (`cases/<case>/case.toml`, see
[Writing a test case]({{root}}guide/writing-cases.html)), the GUI, a `--config` file or
`--set` overrides (see [Getting started]({{root}}guide/getting-started.html)). Booleans accept
`true/false`, `1/0`, `yes/no` and `on/off`. Choice values accept either the GUI label or the
short alias shown in brackets.

## Problem and mesh

| Key | Default | Meaning |
|---|---|---|
| `test_case` | `Sod Shock Tube` | Initial field and case-specific BCs (see [Test cases]({{root}}modules/TestCases.html)) |
| `Lx`, `Ly` | `1.0`, `1.0` | Domain size [m] |
| `nx`, `ny` | `100`, `100` | Cells in x and y |
| `nz` | `1` | Cells in z; `1` is a 2-D run, `> 1` a 3-D run (2-D cases are extruded along z) |
| `Lz` | `1.0` | Domain depth [m] (3-D only) |

## Thermodynamics

| Key | Default | Meaning |
|---|---|---|
| `eos_type` | `Ideal Gas` | `Ideal Gas` or `Stiffened Gas` |
| `gamma` | `1.4` | Heat capacity ratio |
| `p_inf` | `0.0` | Stiffened-gas pressure constant [Pa] (ignored for an ideal gas) |

## Physics

| Key | Default | Meaning |
|---|---|---|
| `use_viscous` | `false` | Navier-Stokes viscous and heat fluxes |
| `mu` | `1e-3` | Dynamic viscosity [Pa s] |
| `reynolds` | `0` | If > 0, sets `mu` = `rho_in` `u_in` L / `reynolds`, with L the IBM chord (airfoil) or diameter (cylinder, sphere); 0 keeps `mu` |
| `pr` | `0.72` | Prandtl number |
| `use_bodyforce` | `false` | Constant body force |
| `gx`, `gy` | `0.0`, `-9.81` | Body-force acceleration [m/s²] |
| `gz` | `0.0` | Body-force acceleration along z [m/s²] (3-D only) |

## Source terms

The case file section `[source]`. The equations solved become $\partial U/\partial t + \nabla\cdot(F - F_v) = S$,
with $U = [\rho, \rho u, \rho v, \rho E, \rho w]$. Each key is a number or an expression of `x`, `y`, `z` (cell
centre), `pi`, `e`, the time `t` and the numeric run parameters, added to the residual of every stage. Without `t` it is
evaluated once (steady); with `t` it is evaluated at the time of each stage of the time scheme, in absolute simulation
time. They exist for code verification by the method of manufactured solutions, see
[Manufactured solutions]({{root}}cases/manufactured-solution.html); the cases of that page set them. They also run with the
immersed boundary, uniform or adaptive, serial or on MPI ranks: like the body force they are added in every cell, and the solid
cells are reset to rest after the step, so the source acts on the fluid only.

| Key | Default | Meaning |
|---|---|---|
| `source_rho` | `0` | Mass source $S_\rho$ [kg/m³/s] |
| `source_mx`, `source_my` | `0`, `0` | Momentum sources $S_{\rho u}$, $S_{\rho v}$ |
| `source_mz` | `0` | Momentum source $S_{\rho w}$ (3-D) |
| `source_E` | `0` | Total-energy source $S_{\rho E}$ |

## Boundary conditions

| Key | Default | Meaning |
|---|---|---|
| `bc_left`, `bc_right`, `bc_bottom`, `bc_top` | `wall` | One of `wall` (*Inviscid Wall (Slip)*), `no_slip` (*Viscous Wall (No-Slip)*), `moving_wall`, `vel_inlet` (*Velocity Inlet*), `massflow_inlet` (*Mass Flow Inlet*), `outlet` (*Pressure Outlet*), `transmissive`, `nonreflecting` (*Non-Reflecting Outlet*), `periodic` (periodic on both sides of an axis or neither) |
| `bc_front`, `bc_back` | `periodic` | 3-D only: the $z = 0$ and $z = L_z$ sides, same choices |
| `u_in` | `0.0` | Inlet velocity / moving-wall x-velocity [m/s]; may be an expression of `t` |
| `v_in` | `0.0` | Moving-wall y-velocity [m/s]; may be an expression of `t` |
| `rho_in` | `1.0` | Mass-flow inlet density [kg/m³]; also the IBM reference density; may be an expression of `t` |
| `p_out` | `1.0` | Outlet pressure [Pa]; also the IBM reference pressure and the far-field pressure of `nonreflecting`; may be an expression of `t` |
| `u_in_table`, `v_in_table` | *(empty)* | Optional CSV path with `t,value` columns for the inlet / wall velocity component; linearly interpolated, holding the first and last values outside the table range |
| `rho_in_table`, `p_out_table` | *(empty)* | Optional CSV path with `t,value` columns for inlet density or outlet pressure; linearly interpolated, holding end values outside the table range |
| `nrbc_sigma` | `0.28` | `nonreflecting`: relaxation of the pressure towards `p_out` ($\sigma$; 0: none) |
| `nrbc_beta` | `-1` | `nonreflecting`: weight of the transverse terms in [0, 1] (1: none, 0: full); negative: local Mach number |

Each test case sets its boundary conditions in its case file, and `--config` and `--set`
override them. See the [BC formulas]({{root}}modules/Boundaries.html).

## Numerical schemes

| Key | Default | Meaning |
|---|---|---|
| `riemann_solver` | `HLLC` | `HLLC`, `HLLC (Low-Mach Thornber)` [`lowmach`], `HLLC (Preconditioned)` [`preconditioned`] |
| `use_muscl` | `false` | 2nd-order MUSCL reconstruction |
| `muscl_limiter` | `Van Leer` | `Van Leer`, `MC`, `Koren`, `Minmod` (case-insensitive); read when `reconstruction = muscl` |
| `reconstruction` | `muscl` | Face reconstruction of the 2nd-order path (`use_muscl = true`): `muscl` (limited, 2nd order), `weno3` (WENO3-Z, 3rd order) or `weno5` (WENO5-Z, 5th order). The WENO schemes need `time_scheme` `SSP Runge-Kutta 3` (SSP Runge-Kutta 2 is rejected), `weno3` also runs with AMR and MPI, `weno5` is serial (no MPI) and not available with AMR. See [Schemes]({{root}}modules/Schemes.html) |
| `weno_epsilon` | `1e-6` | WENO smoothness-indicator regularisation, relative to the data of the stencil: $\varepsilon = \texttt{weno\_epsilon}\cdot\overline{c^2}$. Larger values weigh the stencils closer to the linear ones (more accurate at smooth extrema, less non-oscillatory) |
| `boundary_order` | `2` | `2`: the faces on a physical boundary are reconstructed through ghost layers (the boundary condition's own extension), second order (`muscl`, `weno3`, `weno5`; with AMR for `muscl` and `weno3`: a leaf on a physical side takes its ghost layers from its own state and from the leaf or virtual cell behind it); on a viscous boundary face the cross derivatives are second order too, and the moving-wall ghost keeps the interior internal energy. `1`: the faces on a physical boundary keep the first-order ghost state (the behaviour of the references of the regression suite, which set it). With a first-order scheme (`use_muscl = false`) there is no reconstruction to extend and the order is 1 whatever was asked. See [manufactured solutions]({{root}}cases/manufactured-solution.html) |
| `time_scheme` | `Explicit Euler` | `Explicit Euler` [`euler`], `SSP Runge-Kutta 2` [`ssprk2`], `SSP Runge-Kutta 3` [`ssprk3`]. AMR runs accept all three |
| `cfl` | `0.5` | CFL number |
| `shock_blend` | `0.2` | Shock-face dissipation: a face whose pressure jump `abs(pR - pL) / min(pL, pR)` exceeds this blends its HLLC flux toward HLL (fully at twice the value). The sensor reads the face's own two states, so there is no extra stencil. 0 switches it off |

## Run control and output

| Key | Default | Meaning |
|---|---|---|
| `t_end` | `0.005` | End time [s] |
| `save_interval` | `1` | Snapshot + checkpoint every N steps |
| `write_interval_time` | `0` | If > 0, clip steps to snapshots at exact multiples of this time interval [s] (0: off); each such snapshot also writes a checkpoint |
| `dt_max` | `0` | Maximum physical time step [s] (0: off), applied in addition to the CFL limit and exact-time output clipping |
| `resume` | `false` | Continue from `<case>_restart.npz` (also with AMR: the leaves are restored exactly) |
| `output_root` | `<repo>/results` | Parent of the per-case output folder |
| `max_steps` | `0` | Stop after N steps (≤ 0: run to `t_end`) |
| `dump_state` | *(empty)* | Write the final state to this `.npz` (regression tests) |
| `monitor_interval` | `0` | Append domain averages (mass, kinetic and total energy), and with IBM the body loads (Fx, Fy, Cd, Cl, Cm), to `<case>_monitor.csv` every N steps (0: off) |
| `vtk_format` | `binary` | VTK data arrays: `binary` (inline base64 Float32: about half the size, faster to write) or `ascii` (plain text). Both are valid XML; `fenice.read_vtr` (Python package) reads either |

## Parallelism

| Key | Default | Meaning |
|---|---|---|
| `use_mpi` | `false` | Run under MPI (re-launch with `mpiexec`) |
| `n_cpus` | `2` | Number of MPI ranks |

The OpenMP thread count is set with `--threads` or `OMP_NUM_THREADS`.

## Immersed boundary

| Key | Default | Meaning |
|---|---|---|
| `use_ibm` | `false` | Enable the immersed obstacle |
| `ibm_shape` | `Cylinder` | `Cylinder` or `Sphere` (identical in 2-D; in 3-D a ball, or a cylinder through the whole depth), `NACA dddd` (4-digit section, e.g. `NACA 2412`) or `Airfoil` (from `ibm_file`); case-insensitive. Airfoils are extruded along z in 3-D |
| `ibm_radius` | `0.1` | Radius [m] (cylinder, sphere) |
| `ibm_cx`, `ibm_cy` | `0.5`, `0.5` | Centre [m]; for an airfoil, the quarter-chord point |
| `ibm_file` | (empty) | `Airfoil`: Selig or Lednicer coordinate file (UIUC format); a relative path is looked up next to the case's `case.toml`, then in the current folder |
| `ibm_chord` | `1.0` | Airfoil chord [m] (the file is normalised to unit chord first) |
| `ibm_aoa` | `0.0` | Airfoil angle of attack [deg], positive nose up for a flow along +x; rotation about the quarter chord |
| `ibm_cz` | `0.5` | Sphere centre z [m] (3-D only) |
| `ibm_method` | `auto` | `auto`: `ghost` wherever it is the default (2-D cylinder or airfoil; with AMR when `amr_refine_ibm` is on), else `mask` (3-D; `ghost` is available there for the sphere, a cylinder along z and an extruded airfoil, but not chosen by `auto` yet). `mask`: solid cells are cells whose centre is inside the body, velocity zeroed there (a staircase wall). `ghost`: sharp wall, ghost states mirrored across the true surface (no-slip with `use_viscous`, slip without), monitor loads from the momentum flux into the solid cells as with `mask`; 2-D cylinder or airfoil; serial, OpenMP and MPI, uniform or AMR (with `amr_refine_ibm`, the band at the finest level is widened to the ghost stencils) |
| `ibm_curvature` | `false` | Ghost wall with a slip condition only (`use_viscous = false`; no effect with no-slip). `true`: the ghost pressure follows the normal momentum equation at the wall, dp/dn = rho kappa u_t^2 (kappa: curvature of the wall, convex positive; exact for a cylinder, estimated over a window of three cells for a polygon), and the density follows constant entropy along the normal, instead of a zero normal gradient of both. The curved wall error falls from first order to about 1.6 (supersonic vortex, `verification/ibm_vortex.py`: 30 times smaller at 160 cells per radius); flat walls are unchanged. The change of a ghost pressure is limited to a factor of 0.5 to 2. The correction is weighted by how well the flow follows the wall (1 for a zero normal velocity at the image point, 0 at `abs(u_n)` = 0.2 `abs(u_t)`), so the free stream of an impulsive start, which hits the wall, is left alone: a Mach 2 cylinder started from a uniform state still detaches its bow shock |

## Solution mapping (hot start)

| Key | Default | Meaning |
|---|---|---|
| `use_hot_start` | `false` | Initialise by mapping a checkpoint onto this mesh |
| `hot_start_file` | *(empty)* | Source `.npz` checkpoint: of a uniform run (2-D or 3-D) or of a 2-D AMR run (any number of levels; its leaves are read as the finest-level field) |
| `hot_start_src_nx`, `hot_start_src_ny` | `0` | Source grid size for legacy files without metadata |

## Adaptive mesh refinement

| Key | Default | Meaning |
|---|---|---|
| `use_amr` | `false` | Adaptive mesh refinement on leaf cells (2-D only: rejected when `nz > 1`). Periodic and mass-flow boundaries, IBM, viscous terms, body force, source terms and MPI are supported. `nx` and `ny` are the level-0 mesh (any size) |
| `amr_max_level` | `2` | Refinement levels above the base mesh |
| `amr_indicator` | `density+velocity` | Variables whose variation across faces drives the refinement: `density`, `velocity` (the speed), `pressure`, or several joined by `+`. Use the speed for vortical, nearly incompressible flow such as a wake, where the density hardly varies |
| `amr_criterion` | `0.1` | Relative variation of a variable across a face that flags both cells |
| `amr_refine_threshold` | `0.15` | Refine a leaf when its smoothed indicator reaches this |
| `amr_derefine_threshold` | `0.1` | Join four siblings when the indicator of their parent cell is below this (must not exceed the refine threshold) |
| `amr_diffusion` | `2` | Smoothing iterations of the indicator: the refined band extends about this many cells beyond the flagged ones |
| `amr_refine_every` | `10` | Remesh every N coarse steps |
| `amr_init_levels` | `0` | Uniform pre-refinement levels at t = 0 |
| `amr_force_box` | (empty) | `x0 x1 y0 y1`: the leaves that overlap this box stay at `amr_max_level` whatever the indicator says, a static refined region (a test hook: the orders of accuracy across a level jump are measured with it). Empty: none |
| `amr_refine_ibm` | `true` | With `use_ibm`: leaves touching the body, and the cells around it that the solid reset uses, are refined to `amr_max_level` from t = 0 and never joined |

{{api}}
