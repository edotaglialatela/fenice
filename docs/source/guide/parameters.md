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

## Boundary conditions

| Key | Default | Meaning |
|---|---|---|
| `bc_left`, `bc_right`, `bc_bottom`, `bc_top` | `wall` | One of `wall` (*Inviscid Wall (Slip)*), `no_slip` (*Viscous Wall (No-Slip)*), `moving_wall`, `vel_inlet` (*Velocity Inlet*), `massflow_inlet` (*Mass Flow Inlet*), `outlet` (*Pressure Outlet*), `transmissive`, `nonreflecting` (*Non-Reflecting Outlet*), `periodic` (periodic on both sides of an axis or neither) |
| `bc_front`, `bc_back` | `periodic` | 3-D only: the $z = 0$ and $z = L_z$ sides, same choices |
| `u_in` | `0.0` | Inlet velocity / moving-wall velocity [m/s] |
| `rho_in` | `1.0` | Mass-flow inlet density [kg/m³]; also the IBM reference density |
| `p_out` | `1.0` | Outlet pressure [Pa]; also the IBM reference pressure and the far-field pressure of `nonreflecting` |
| `nrbc_sigma` | `0.28` | `nonreflecting`: relaxation of the pressure towards `p_out` ($\sigma$; 0: none) |
| `nrbc_beta` | `-1` | `nonreflecting`: weight of the transverse terms in [0, 1] (1: none, 0: full); negative: local Mach number |

Each test case sets its boundary conditions in its case file, and `--config` and `--set`
override them. See the [BC formulas]({{root}}modules/Boundaries.html).

## Numerical schemes

| Key | Default | Meaning |
|---|---|---|
| `riemann_solver` | `HLLC` | `HLLC`, `HLLC (Low-Mach Thornber)` [`lowmach`], `HLLC (Preconditioned)` [`preconditioned`] |
| `use_muscl` | `false` | 2nd-order MUSCL reconstruction |
| `muscl_limiter` | `Van Leer` | `Van Leer`, `MC`, `Koren`, `Minmod` (case-insensitive) |
| `time_scheme` | `Explicit Euler` | `Explicit Euler` [`euler`], `SSP Runge-Kutta 2` [`ssprk2`], `Runge-Kutta 4` [`rk4`], `BDF2 Dual Time` [`bdf2`] |
| `cfl` | `0.5` | CFL number (pseudo-time CFL for BDF2) |
| `dt_phys` | `0.1` | BDF2 physical time step [s] |
| `max_inner_iter` | `50` | BDF2 pseudo-time iterations per step |
| `inner_tol_rel` | `1e-3` | BDF2 relative convergence tolerance |

## Run control and output

| Key | Default | Meaning |
|---|---|---|
| `t_end` | `0.005` | End time [s] |
| `save_interval` | `1` | Snapshot + checkpoint every N steps |
| `resume` | `false` | Continue from `<case>_restart.npz` (not with AMR, which writes no checkpoint) |
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

## Solution mapping (hot start)

| Key | Default | Meaning |
|---|---|---|
| `use_hot_start` | `false` | Initialise by mapping a checkpoint onto this mesh |
| `hot_start_file` | *(empty)* | Source `.npz` checkpoint |
| `hot_start_src_nx`, `hot_start_src_ny` | `0` | Source grid size for legacy files without metadata |

## Adaptive mesh refinement

| Key | Default | Meaning |
|---|---|---|
| `use_amr` | `false` | Block-structured AMR (2-D only: rejected when `nz > 1`; no periodic or mass-flow boundaries, no BDF2). IBM, viscous terms, body force and MPI are supported. The level-0 patches tile the mesh: `nx` and `ny` must be multiples of `amr_patch_nx` and `amr_patch_ny`, and refinement splits one patch into four, so smaller patches refine more locally |
| `amr_max_level` | `2` | Refinement levels above the root |
| `amr_patch_nx`, `amr_patch_ny` | `8`, `8` | Cells per patch; must be **even** when `amr_max_level > 0` |
| `amr_refine_threshold` | `0.3` | Refine when the Löhner indicator η exceeds this |
| `amr_derefine_threshold` | `0.1` | Derefine when η of all four siblings is below this |
| `amr_refine_every` | `10` | Remesh every N coarse steps |
| `amr_init_levels` | `0` | Uniform pre-refinement levels at t = 0 |
| `amr_buffer_layers` | `1` | Rings of neighbours refined around flagged patches |
| `amr_refine_ibm` | `true` | With `use_ibm`: leaves whose box (grown by one cell) overlaps the body's bounding box are refined to `amr_max_level` from t = 0 and never coarsened |
| `amr_indicator` | `density` | Field of the Löhner refinement indicator: `density`, `velocity` (the speed) or `density+velocity` (the larger of the two). Use the speed for vortical, nearly incompressible flow such as a wake, where the density hardly varies |

{{api}}
