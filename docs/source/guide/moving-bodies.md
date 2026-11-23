---
title: Moving bodies
nav: Moving bodies
section: User guide
order: 45
---

> **In short.** The immersed body can move: along a prescribed path, or driven by the flow (mass, spring, damper, weight). The ghost-cell wall follows the body,
> cells the body leaves are refilled from the wall layer, and every run (serial, any rank count, uniform or adaptive mesh) gives the same bits.

## What moves

| Body | Mesh | Motion |
|---|---|---|
| cylinder, polygon (airfoil) | uniform or adaptive, 2-D | translation (`ibm_vx`, `ibm_vy`); prescribed rotation (`ibm_omega`) |
| sphere | uniform, 3-D | translation (`ibm_vx`, `ibm_vy`, `ibm_vz`) |

A body driven by the flow is set with `ibm_mass > 0` (kg per unit depth in 2-D, kg in 3-D); `ibm_vx/vy/vz` are then its initial velocity. Springs
(`ibm_stiff_*`) pull it back to its placement, dampers (`ibm_damp_*`) oppose its velocity, and with `use_bodyforce` its weight `m g` acts on it.
A driven body translates only (`ibm_omega` must be 0). Not available: the adaptive mesh in 3-D, `ibm_curvature`, the Hancock scheme and `positivity_fallback`
with a driven body. All keys are in the [parameter reference](parameters.html).

## How a step goes

1. The pose is fixed during a step. The geometry (mask, ghost cells, image points) is rebuilt whenever the body has moved a quarter of a (finest) cell since
   the last build; in between only the wall velocity is refreshed.
2. Cells that were solid and are fluid now are filled from an image point in the fluid along the wall normal (wall-layer state), so no stale solid value enters the flow.
3. A driven body takes the force of the state at the start of the step (the first Runge-Kutta stage on a uniform mesh; one extra residual per step on an
   adaptive mesh). The terms of the solid cells are summed in one fixed global order, so every rank advances the same body.
4. The velocity is advanced by Adams-Bashforth 2 (Euler on the first step), the position by the trapezoid rule.

A body may not travel more than one cell (one finest cell with AMR) in a step: the run stops otherwise. Lower the CFL number.

## Checkpoints

The checkpoint keeps the pose the geometry was last built at and, for a driven body, its position, velocity and last acceleration. A resumed run continues bit
for bit, also on another rank count.

## Validation

| Study | What it checks |
|---|---|
| `features/ibm_ghost.py --check` | serial == 2-7 ranks bit for bit, uniform and adaptive, prescribed and driven bodies (a sphere with strips along y and x); resume with `--only resume` |
| `features/ibm_spring.py` | a body on a spring in a gas against the added-mass oscillator: frequency within 0.4 %, stable down to a body 0.3 times as heavy as the fluid it displaces |
| `features/ibm_gravity.py`, `ibm_gravity3d.py` | weight and buoyancy in a stratified gas against `(m + m_a) a = -(m - rho_c V) g` |

The gravity studies agree for sinking bodies (cylinder 0.8 %, sphere 0.3 % at the finest mesh) and the neutral body stays. A body lighter than the gas it
displaces rises about 12 % faster than the added-mass reference in both the cylinder and the sphere; the cause is not isolated, and that case is reported but not gated.
