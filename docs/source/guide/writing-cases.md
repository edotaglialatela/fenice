---
title: Writing a test case
nav: Writing a test case
section: Guide
order: 25
---

A test case is a folder under `cases/` holding one file, `case.toml`. The file describes
the mesh, physics, numerics, boundary conditions, run control and initial conditions. The
solver reads it at start-up, so adding or changing a case needs no C++ and no
recompilation. The approach follows ECOGEN's input files and OpenFOAM's case folders:

```text
cases/
  sod_shock_tube/case.toml
  lid_driven_cavity/case.toml
  ...
  my_new_case/case.toml        <- yours
```

The fastest way to start is to copy an existing case:

```powershell
cpp\build\fenice.exe --new-case "Shock Bubble" --from "Sod Shock Tube"   # writes cases/shock_bubble/case.toml
# edit cases/shock_bubble/case.toml, then
cpp\build\fenice.exe --case "Shock Bubble"
```

In the GUI, configure a run and press **Save as case…**, which does the same with the
dialog's settings (see [GUI]({{root}}guide/gui.html)).

## A complete example

A shock wave hitting a light circular bubble. This is the Sod case, with a bubble added:

```toml
# cases/shock_bubble/case.toml
[case]
name = "Shock Bubble"                       # what --case and the GUI list show
description = "Shock wave hitting a light circular bubble"
order = 100                                 # position in the case list

[mesh]
Lx = 1.0
Ly = 0.5
nx = 300
ny = 150

[boundaries]
bc_left = "transmissive"
bc_right = "transmissive"
bc_bottom = "wall"
bc_top = "wall"

[numerics]
riemann_solver = "HLLC"
use_muscl = true
time_scheme = "SSP Runge-Kutta 2"
cfl = 0.5

[run]
t_end = 0.25
save_interval = 50

[initial]
state = { rho = 0.125, u = 0.0, v = 0.0, p = 0.1 }   # background: the low-pressure gas

[[initial.region]]          # high-pressure driver, x <= 0.3
shape = "halfspace"
point = [0.3, 0.0]
normal = [-1.0, 0.0]
state = { rho = 1.0, p = 1.0 }

[[initial.region]]          # light bubble, in pressure equilibrium with its surroundings
shape = "sphere"            # a circle in 2-D
center = [0.55, "Ly/2"]
radius = 0.1
state = { rho = 0.02 }
```

Anything a case file leaves out takes its default from the
[Parameters]({{root}}guide/parameters.html) page. `--config` files and `--set` still
override the case file, as before: `fenice --case "Shock Bubble" --set nx=600 --set ny=300`.

## Sections

**`[case]`** describes the case itself:

| Key | Meaning |
|---|---|
| `name` | Name used by `--case`, `--list-cases` and the GUI (required, unique) |
| `description` | One line, shown in the docs |
| `order` | Sort position in the case list (default 1000; the built-in cases use 10 … 90) |
| `requires_viscous` | `true`: warn if the run has Navier-Stokes switched off |
| `requires_3d` | `true`: the initial condition needs `nz > 1` (otherwise a 2-D case, extruded along z on a 3-D mesh) |

**Parameter sections** hold the run parameters, with exactly the names on the
[Parameters]({{root}}guide/parameters.html) page. Each parameter has one home section:

| Section | Parameters |
|---|---|
| `[mesh]` | `Lx`, `Ly`, `Lz`, `nx`, `ny`, `nz` |
| `[thermo]` | `eos_type`, `gamma`, `p_inf` |
| `[physics]` | `use_viscous`, `mu`, `pr`, `use_bodyforce`, `gx`, `gy`, `gz` |
| `[boundaries]` | `bc_left`, `bc_right`, `bc_bottom`, `bc_top`, `bc_front`, `bc_back`, `u_in`, `rho_in`, `p_out` |
| `[numerics]` | `riemann_solver`, `use_muscl`, `muscl_limiter`, `time_scheme`, `cfl`, `dt_phys`, `max_inner_iter`, `inner_tol_rel` |
| `[run]` | `t_end`, `save_interval`, `monitor_interval`, `max_steps`, `resume`, `output_root`, `dump_state` |
| `[parallel]` | `use_mpi`, `n_cpus` |
| `[ibm]` | `use_ibm`, `ibm_shape`, `ibm_radius`, `ibm_cx`, `ibm_cy`, `ibm_cz` |
| `[hot_start]` | `use_hot_start`, `hot_start_file`, `hot_start_src_nx`, `hot_start_src_ny` |
| `[amr]` | `use_amr`, `amr_*` |

Values use TOML types: numbers (`nx = 200`, `mu = 2.5e-3`), booleans (`use_muscl = true`)
and strings (`riemann_solver = "HLLC"`). The boundary conditions come from the case file
(and `--config` / `--set`) only: no case forces its own in code.

**`[initial]`** sets the initial condition (next section).

## Initial conditions

The initial field is built per cell, at the cell centre. Every case starts from a
background, given either as a **`state`** or as a built-in **`profile`**. Then the
**regions** are applied in order, each one overriding the variables it lists in the cells
it contains:

```toml
[initial]
state = { rho = 1.0, u = 0.0, v = 0.0, w = 0.0, p = 1.0 }   # rho and p required; u, v, w default to 0

[[initial.region]]
shape = "box"
min = [0.2, 0.0]
max = [0.4, "Ly"]
state = { rho = 2.0 }        # only rho changes here; u, v, w, p come from the background
```

States use the primitive variables `rho`, `u`, `v`, `w` (z-velocity, 3-D) and `p`. The
solver converts them to conservative form with the case's equation of state.

### Region shapes

| Shape | Keys | Contains the cell centre **c** when |
|---|---|---|
| `box` | `min`, `max` | `min <= c <= max` in every direction given (a missing z bound is unbounded) |
| `halfspace` | `point`, `normal` | `(c - point) · normal >= 0`: the side the normal points to |
| `sphere` | `center`, `radius` | the distance ‖c − center‖ is `<= radius` (a circle in 2-D; 3-D needs a 3-component center) |
| `cylinder` | `center`, `radius`, `axis` (`"x"`, `"y"` or `"z"`, default `"z"`) | the distance from the axis through `center` is `<= radius` (a circle in 2-D) |

Vectors have 2 or 3 entries. In 2-D only x and y are used. Geometry entries can be
expressions of the run parameters, such as `"Lx/2"` or `"Ly - 0.1"`, so a region follows
the domain when you change it with `--set`.

### Expressions

Any state value or geometry entry can be a string holding a formula instead of a number:

```toml
[initial]
state = { rho = "1 + 0.2*sin(5*(x - 5))", u = 0.0, v = 0.0, p = 1.0 }        # Shu-Osher entropy wave
state = { rho = "rho_in", u = "u_in", v = 0.0, p = "p_out + 12*mu*u_in/(Ly*Ly)*(Lx - x)" }   # Poiseuille
state = { rho = 1.0, u = "sin(x)*cos(y)*cos(z)", v = "-cos(x)*sin(y)*cos(z)", p = "1/(gamma*0.1*0.1)" }
```

| Available | |
|---|---|
| Coordinates | `x`, `y`, `z`: the cell centre (`z = 0` in 2-D) |
| Run parameters | every numeric parameter by name: `Lx`, `Ly`, `Lz`, `nx`, `mu`, `gamma`, `u_in`, `rho_in`, `p_out`, `gx`, ... (resolved values, so `--set` overrides apply) |
| Constants | `pi`, `e` |
| Functions | `sin cos tan asin acos atan atan2 sinh cosh tanh exp sqrt pow abs floor ceil`, `ln` (natural log); note that `log` is base 10, like `log10` |
| Operators | `+ - * / ^` and `%` (remainder), with the usual precedence |

Expressions are evaluated in double precision by [TinyExpr](https://github.com/codeplea/tinyexpr).
An expression evaluates bit for bit like the same arithmetic written in C++ in the same
order, so an analytic initial condition needs no C++ profile to be exact (the built-in
Poiseuille, Shu-Osher and 3-D Taylor-Green cases are written this way).

### Built-in profiles

A few analytic fields stay in C++ (`cpp/src/TestCases.cpp`) and are selected by name:

| `profile = ` | Field |
|---|---|
| `"taylor_green"` | 2-D Taylor-Green vortex at t = 0 (periodic square, U₀ = 1, ρ₀ = 1, p₀ = 100) |
| `"isentropic_vortex"` | Isentropic vortex of strength β = 5 in a (1, 1) free stream |
| `"taylor_green_3d"` | 3-D Taylor-Green vortex (also written with expressions in its case file) |

Regions can be added on top of a profile. Cells outside every region keep the profile's
state exactly.

## Checks and error messages

Every case file is checked when it loads, before anything runs. Mistakes stop with the
file, the line and a suggestion:

```text
cases/my_case/case.toml:15: unknown parameter 'gama' (did you mean 'gamma'?)
cases/my_case/case.toml:32: 'nx' belongs in [mesh], not [run]
cases/my_case/case.toml:26: unknown section [numeric] (did you mean 'numerics'?)
cases/my_case/case.toml:43: region 2: unknown shape 'sphear' (did you mean 'sphere'?)
cases/my_case/case.toml: region 2.state.rho: cannot evaluate "1 + sin(xx)" (error at character 13) (did you mean 'x'?)
cases/my_case/case.toml: [initial] needs a background state with at least rho and p, ...
```

A case file that fails to load is skipped by `--list-cases` and the GUI, with its error
printed. `--case` on that case shows the error.

## Command line

| Option | Effect |
|---|---|
| `--case NAME` | Run a case by its `[case] name` or its folder name |
| `--case PATH` | Run a case folder (or `case.toml`) anywhere, without adding it to `cases/` |
| `--new-case NAME --from CASE` | Copy `CASE` to `cases/<name>/case.toml` with the `[case] name` `NAME` |
| `--save-case TARGET` | Save the resolved run (case + `--config` + `--set`) as a new case. `TARGET` is a name (written under `cases/`) or a folder path |
| `--case-name NAME` | `[case] name` for `--save-case` (default: from `TARGET`) |
| `--list-cases` | List all cases in order |

Saving never overwrites an existing case. A saved case keeps the parameters its source sets
plus every parameter you changed, and copies the source's initial conditions.
`FENICE_CASES=<folder>` points the solver at a different case folder.
