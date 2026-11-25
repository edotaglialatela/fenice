---
title: Parameters
nav: Parameters
section: User guide
order: 30
header: Params.hpp
---

> **In short.** Every setting of a run is a `key = value` pair, listed here by topic with its default. To change one for a single run use
> `--set key=value`; to keep the change, put it in the case file. `fenice --case NAME --print-config` shows the value every key
> has for a case.

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
| `nx`, `ny` | `100`, `100` | Cells in x and y. Indices are 32-bit: a mesh with more than 2^31 - 1 faces (about 700 million cells in 3-D, a billion in 2-D) is refused at setup |
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
| `bc_xmin`, `bc_xmax`, `bc_ymin`, `bc_ymax` | `wall` | One of `wall` (*Inviscid Wall (Slip)*), `no_slip` (*Viscous Wall (No-Slip)*), `moving_wall`, `vel_inlet` (*Velocity Inlet*), `massflow_inlet` (*Mass Flow Inlet*), `outlet` (*Pressure Outlet*), `transmissive`, `nonreflecting` (*Non-Reflecting Outlet*), `periodic` (periodic on both sides of an axis or neither) |
| `bc_zmin`, `bc_zmax` | `periodic` | 3-D only: the $z = 0$ and $z = L_z$ sides, same choices |
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
| `riemann_solver` | `HLLC` | `HLLC`, `HLLC (Low-Mach Thornber)` [`lowmach`] |
| `muscl_limiter` | `Van Leer` | `Van Leer`, `MC`, `Koren`, `Minmod` (case-insensitive); read when `reconstruction = muscl` |
| `reconstruction` | `none` | Face reconstruction: `none` (first order: the cell states, Godunov), `muscl` (limited, 2nd order, limiter `muscl_limiter`), `weno3` (WENO3-Z, 3rd order) or `weno5` (WENO5-Z, 5th order). The WENO schemes need `time_scheme` `SSP Runge-Kutta 3` (SSP Runge-Kutta 2 is rejected), both run with AMR, also on MPI ranks; on the uniform mesh `weno3` runs with MPI and `weno5` is serial. With AMR `weno5` needs at least three base cells along a periodic axis. See [Schemes]({{root}}modules/Schemes.html) |
| `flux_quadrature` | `midpoint` | Flux of a face on the uniform mesh: `midpoint` (the Riemann flux of the face-averaged states, second order in the flux balance in 2-D and 3-D whatever the reconstruction) or `fourth` (the states are the WENO face averages of the CONSERVED variables, reduced to face-centre values by the transverse second difference, `U - (1/24) d2_t U`, the Riemann solver runs on them and the flux is corrected back to a face average, `F + (1/24) d2_t F`: fourth order in space with `weno5`, one Riemann solve per face, conservative). Needs `reconstruction = weno3` or `weno5`; uniform mesh, OpenMP only (no MPI), no immersed body yet. A face keeps the flux of the averaged states where the correction cannot be made (a side without ghost layers, a transverse neighbour that does not exist, the shock sensor of the Riemann solver firing, a non-positive density or pressure of a point value); a reconstructed face whose density or pressure is not positive, or lies more than the width of its stencil's range outside that range, takes the two neighbouring cell states. The viscous terms, the source terms and the body force stay second order; initial data given at cell centres carry an `O(h^2)` error (the order shows with cell-averaged data, e.g. a restart file). See [Schemes]({{root}}modules/Schemes.html) |
| `use_muscl` | (none) | Retired key, still read for old case files: `true` means `reconstruction = muscl` (or the reconstruction given with it), `false` means `none`; `false` with `weno3` / `weno5` is an error. The solver prints a note. |
| `weno_epsilon` | `1e-6` | WENO smoothness-indicator regularisation, relative to the data of the stencil: $\varepsilon = \texttt{weno\_epsilon}\cdot\overline{c^2}$. Larger values weigh the stencils closer to the linear ones (more accurate at smooth extrema, less non-oscillatory) |
| `positivity_fallback` | `false` | **Experimental.** WENO (`weno3`, `weno5`) on the uniform mesh and on the adaptive mesh, in 2-D and 3-D, with OpenMP threads and MPI ranks (uniform `weno5` is serial only): after each time stage the new state is scanned, and if a cell has a density or a pressure of 0 or less, the stage is repeated with first-order faces (the two cell values as the states) on every face of those cells, and scanned again, until no new cell turns up (at most 8 passes; the run summary prints the number of passes). It costs one scan of the field per stage, and nothing else while no cell leaves the admissible set; with `false` the code of the stage is not touched. On the adaptive mesh the repair acts per level and stage on the owned leaves; a coarse leaf that loses positivity next to a finer level keeps the flux its coarse-fine faces got from the fine stage (those faces enter it through the flux register). Rejected with `flux_quadrature = fourth` and the other reconstructions. It repairs the update, not the states of the faces: the cells that lose positivity (the near-vacuum wake of a Mach 10 cylinder at CFL 0.3) have positive face states when they do |
| `boundary_order` | `2` | `2`: the faces on a physical boundary are reconstructed through ghost layers (the boundary condition's own extension), second order (`muscl`, `weno3`, `weno5`; with AMR too: a leaf on a physical side takes its ghost layers from its own state and from the leaf or virtual cell behind it); on a viscous boundary face the cross derivatives are second order too, and the moving-wall ghost keeps the interior internal energy. `1`: the faces on a physical boundary keep the first-order ghost state (the behaviour of the references of the regression suite, which set it). With a first-order scheme (`reconstruction = none`) there is no reconstruction to extend and the order is 1 whatever was asked. See [manufactured solutions]({{root}}cases/manufactured-solution.html) |
| `time_scheme` | `auto` | `auto` (`MUSCL-Hancock` with `reconstruction = muscl`, `Explicit Euler` otherwise), `Explicit Euler` [`euler`], `SSP Runge-Kutta 2` [`ssprk2`], `SSP Runge-Kutta 3` [`ssprk3`], `MUSCL-Hancock` [`hancock`]. AMR runs accept the first three and MUSCL-Hancock (quadtree and octree, with the immersed body, body force and sources, on one rank or on MPI ranks). MUSCL-Hancock is a one-residual predictor-corrector: each cell's limited face states are advanced by half a step with the cell's own flux difference and its body force and source terms (the sources at the middle of the step), then one Riemann solve per face; the viscous term is added to the balance only; the ghost state of a reflecting wall is made of the predicted state of the cell next to it, so a closed box keeps its mass and energy. On an adaptive mesh each level is predicted by half of its own step before the finer levels move, and the coarse leaf's prediction serves both sub-steps of the fine leaves across a jump; on MPI ranks the owners of the fine leaves hand the coarse side's limited state to the owner of the coarse leaf, and the owners hand the predicted side states of their leaves to the ranks that compute the face without owning the leaf (the other owner of a same-level face, the owners of the fine leaves of a jump). It needs `reconstruction = muscl` and runs on the uniform mesh in 2-D and 3-D (OpenMP, and MPI, with strips along y or along x: each rank predicts its own cells and swaps the states of its edge rows, or of its edge columns), with or without an immersed body (the cells of the body are not predicted, and a cell keeps its limited, not predicted, state on a side that faces a cell of the body, so the two states of a wall face belong to the time level of the ghost cells); the time step comes from the wave speed of the previous step, which a checkpoint keeps |
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
| `ibm_vx`, `ibm_vy`, `ibm_vz` | `0.0`, `0.0`, `0.0` | Prescribed translation velocity of the body [m/s], constant, from its placement at t = 0. Any nonzero value (or `ibm_omega`) makes the body move: uniform or adaptive mesh (serial or MPI, the same bits for any rank count), 2-D, or a sphere in 3-D, uniform or octree (translation, `ibm_vz`, and spin, `ibm_omega`, `ibm_omega_x`, `ibm_omega_y`), no `ibm_curvature`; a resumed run continues bit for bit, also on another rank count (the checkpoint keeps the pose the geometry was last built at and, for a body driven by the flow, its position, velocity and last acceleration). The pose is fixed during a step and the ghost-cell wall carries the wall velocity of the rigid motion; a body may not travel more than one cell (on an adaptive mesh, one finest cell) in a step |
| `ibm_omega` | `0.0` | Prescribed angular velocity of the body [rad/s], counter-clockwise, about (`ibm_cx`, `ibm_cy`) (the quarter chord of an airfoil); a circle only translates. For a sphere in 3-D it is the component about the z axis through its centre (the sphere does not change, the wall carries the velocity $\boldsymbol\omega\times\mathbf r$) |
| `ibm_omega_x`, `ibm_omega_y` | `0.0`, `0.0` | The other two components of the prescribed angular velocity of a sphere in 3-D [rad/s], about the axes through its centre; `ibm_omega_from` / `ibm_omega_until` window all three. Not for a body driven by the flow |
| `ibm_omega_from`, `ibm_omega_until` | `0.0`, `0.0` | Times between which the prescribed rotation `ibm_omega` runs; outside them the angle is held (a pitch-up to a fixed incidence: `ibm_omega = -w` from `t0` to `t0 + T` gives a nose-up change of `w T` rad, nose-up being clockwise). `ibm_omega_until = 0.0`: the rotation never stops. Need `ibm_omega` |
| `ibm_mass` | `0.0` | Mass of the body [kg per unit depth in 2-D, kg in 3-D]. Above zero the body is driven by the flow instead of following a prescribed motion: its translation is free, `ibm_vx`, `ibm_vy`, `ibm_vz` are then its initial velocity, and each step the loads of the state at the start of the step (the force on the body, summed in a fixed global order, the same bits for any rank count or thread count; on a uniform mesh they are those of the first stage, with AMR one residual of the whole mesh is added per step) advance it with a second-order explicit update (Adams-Bashforth for the velocity, trapezoid for the position). With AMR the wall velocity follows every step and the geometry every quarter of a finest cell. A body driven by the flow needs a Runge-Kutta or Euler time scheme, no `positivity_fallback`, no `ibm_omega`. A body that travels more than one cell in a step stops the run; the loose coupling is tested down to a body 0.3 times as heavy as the fluid it displaces |
| `ibm_stiff_x`, `ibm_stiff_y`, `ibm_stiff_z` | `0.0`, `0.0`, `0.0` | Spring stiffness of a body driven by the flow [N/m per unit depth in 2-D]: the force `-k x` pulls it back to its placement along each axis (needs `ibm_mass`) |
| `ibm_damp_x`, `ibm_damp_y`, `ibm_damp_z` | `0.0`, `0.0`, `0.0` | Damping coefficient of a body driven by the flow [N s/m per unit depth in 2-D]: the force `-c v` along each axis (needs `ibm_mass`). With `use_bodyforce` the weight `m g` acts on the body (the buoyancy comes from the pressure the wall sees; checked against the added-mass ODE for a cylinder released in a stratified gas: `ibm_gravity.py`; a sphere in 3-D: `ibm_gravity3d.py`; the ghost pressure of the wall follows the hydrostatic gradient of the gas, see the ghost-cell wall) |
| `ibm_method` | (none) | Retired key, still read for old case files: the ghost-cell wall is the only wall. `auto` and `ghost` are accepted and do nothing; any other value is an error. |
| `ibm_curvature` | `false` | Slip walls in 2-D only (`use_viscous = false`; no effect with no-slip). `true`: the ghost pressure follows the normal momentum equation at the wall, dp/dn = rho kappa u_t^2 (kappa: curvature of the wall, convex positive; exact for a cylinder, estimated over a window of three cells for a polygon), and the density follows constant entropy along the normal, instead of a zero normal gradient of both. The curved wall error falls from first order to about 1.6 (supersonic vortex, `verification/ibm_vortex.py`: 30 times smaller at 160 cells per radius); flat walls are unchanged. The change of a ghost pressure is limited to a factor of 0.5 to 2. The correction is weighted by how well the flow follows the wall (1 for a zero normal velocity at the image point, 0 at `abs(u_n)` = 0.2 `abs(u_t)`), so the free stream of an impulsive start, which hits the wall, is left alone: a Mach 2 cylinder started from a uniform state still detaches its bow shock |

## Solution mapping (hot start)

| Key | Default | Meaning |
|---|---|---|
| `use_hot_start` | `false` | Initialise by mapping a checkpoint onto this mesh |
| `hot_start_file` | *(empty)* | Source `.npz` checkpoint: of a uniform run (2-D or 3-D) or of an AMR run (a 2-D quadtree for a 2-D run, a 3-D octree for a 3-D run; any number of levels; its leaves are read as the finest-level field) |
| `hot_start_src_nx`, `hot_start_src_ny` | `0` | Source grid size for legacy files without metadata |

## Adaptive mesh refinement

| Key | Default | Meaning |
|---|---|---|
| `use_amr` | `false` | Adaptive mesh refinement on leaf cells, in 2-D (quadtree) and 3-D (octree, `nz > 1`). Periodic and mass-flow boundaries, IBM, viscous terms, body force, source terms and MPI are supported. `nx`, `ny` (and `nz`) are the level-0 mesh (any size). With `use_ibm` the body is kept at the finest level |
| `amr_max_level` | `2` | Refinement levels above the base mesh |
| `amr_indicator` | `density+velocity` | Variables whose variation across faces drives the refinement: `density`, `velocity` (the speed), `pressure`, or several joined by `+`. Use the speed for vortical, nearly incompressible flow such as a wake, where the density hardly varies |
| `amr_criterion` | `0.1` | Relative variation of a variable across a face that flags both cells |
| `amr_refine_threshold` | `0.15` | Refine a leaf when its smoothed indicator reaches this |
| `amr_derefine_threshold` | `0.1` | Join four siblings when the indicator of their parent cell is below this (must not exceed the refine threshold) |
| `amr_diffusion` | `2` | Smoothing iterations of the indicator: the refined band extends about this many cells beyond the flagged ones |
| `amr_refine_every` | `10` | Remesh every N coarse steps |
| `amr_init_levels` | `0` | Uniform pre-refinement levels at t = 0 |
| `amr_force_box` | (empty) | `x0 x1 y0 y1` (3-D: also `z0 z1`; four values span every z): the leaves that overlap this box stay at `amr_max_level` whatever the indicator says, a static refined region (a test hook: the orders of accuracy across a level jump are measured with it). Empty: none |
| `amr_refine_ibm` | (none) | Retired key, still read for old case files: with `use_ibm` the body is always kept at the finest level (leaves touching the body and the band the ghost-cell wall reads, 6 finest cells in 3-D, are refined to `amr_max_level` from t = 0 and never joined; the deep interior is not refined). `true` is accepted and does nothing; `false` is an error. |

{{api}}
