# Manufactured solutions with non-periodic sides: before the second-order boundary closure

Measured on 2026-11-02 (commit `a723c28`), t = 0.5, 2-D, grids 16², 32², 64², 128², CFL as in `verification/mms.py`
(default of the case), error against the manufactured solution at the cell centres. The physical-boundary faces keep the
first-order ghost states (the face on the boundary is not reconstructed). Kept as the reference for the
second-order boundary closure (`mms_boundary.py` repeats it).

Observed order of the finest pair of each step (16→32, 32→64, 64→128) of the worst variable.

## Manufactured Solution Walls

| scheme | norm | worst variable | orders per grid pair | finest rho error |
|---|---|---|---|---|
| MUSCL VL, SSP-RK2 | L2 | rho | 1.61, 1.35, 1.09 | 4.93e-04 |
| MUSCL VL, SSP-RK2 | Linf | rho | 0.79, 0.83, 0.83 | 5.16e-03 |
| WENO3-Z, SSP-RK3 | L2 | rho | 1.80, 1.38, 1.04 | 5.11e-04 |
| WENO3-Z, SSP-RK3 | Linf | rho | 1.11, 0.78, 0.74 | 5.50e-03 |
| WENO5-Z, RK4 | L2 | rho | 1.21, 1.27, 1.02 | 5.18e-04 |
| WENO5-Z, RK4 | Linf | rho | 0.61, 0.81, 0.73 | 5.58e-03 |

## Manufactured Solution Outflow

| scheme | norm | worst variable | orders per grid pair | finest rho error |
|---|---|---|---|---|
| MUSCL VL, SSP-RK2 | L2 | u | 2.10, 1.55, 1.01 | 1.66e-04 |
| MUSCL VL, SSP-RK2 | Linf | u | 1.62, 1.18, 0.73 | 2.93e-03 |
| WENO3-Z, SSP-RK3 | L2 | u | 2.17, 1.42, 1.20 | 1.85e-04 |
| WENO3-Z, SSP-RK3 | Linf | rho | 0.96, 0.79, 0.81 | 3.37e-03 |
| WENO5-Z, RK4 | L2 | u | 1.39, 1.21, 1.06 | 1.88e-04 |
| WENO5-Z, RK4 | Linf | rho | 0.58, 0.79, 0.79 | 3.44e-03 |

All three reconstructions end at order about 1 (L2) and below 1 (Linf) and give nearly the same error on the finest grid:
the boundary closure dominates the error, and the interior reconstruction does not show.
