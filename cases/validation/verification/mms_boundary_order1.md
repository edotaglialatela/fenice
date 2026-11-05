# Manufactured solutions with non-periodic sides, boundary_order = 1 (the physical-boundary faces keep the first-order ghost states)

Measured 2026-11-02 with `mms_boundary.py --boundary-order 1` (commit `a723c28` plus the boundary_order key, default 1:
the results are those of the solver before the key). t = 0.5, grids 16 -> 32 -> 64 -> 128, observed order of the pair of
grids in the worst variable. 'band' compares the finest pair with 1.7 (L2) and 1.3 (Linf). All three reconstructions end at order
about 1 and give nearly the same error on the finest grid: the boundary closure dominates the error.

| solution | scheme | norm | worst variable | orders per grid pair | finest rho error | band |
|---|---|---|---|---|---|---|
| Manufactured Solution Walls Euler | MUSCL VL, SSP-RK2 | L2 | u | 1.01, 1.12, 1.24 | 4.81e-04 | below (1.7) |
| Manufactured Solution Walls Euler | MUSCL VL, SSP-RK2 | Linf | u | -0.07, 0.31, 0.52 | 1.85e-02 | below (1.3) |
| Manufactured Solution Walls Euler | WENO3-Z, SSP-RK3 | L2 | u | 0.97, 1.10, 1.20 | 4.98e-04 | below (1.7) |
| Manufactured Solution Walls Euler | WENO3-Z, SSP-RK3 | Linf | u | -0.00, 0.34, 0.51 | 1.88e-02 | below (1.3) |
| Manufactured Solution Walls Euler | WENO5-Z, SSP-RK3 | L2 | u | 1.00, 1.12, 1.20 | 5.14e-04 | below (1.7) |
| Manufactured Solution Walls Euler | WENO5-Z, SSP-RK3 | Linf | u | 0.21, 0.38, 0.52 | 1.90e-02 | below (1.3) |
| Manufactured Solution Walls | MUSCL VL, SSP-RK2 | L2 | rho | 1.61, 1.35, 1.09 | 4.93e-04 | below (1.7) |
| Manufactured Solution Walls | MUSCL VL, SSP-RK2 | Linf | rho | 0.79, 0.83, 0.83 | 5.16e-03 | below (1.3) |
| Manufactured Solution Walls | WENO3-Z, SSP-RK3 | L2 | rho | 1.80, 1.38, 1.04 | 5.11e-04 | below (1.7) |
| Manufactured Solution Walls | WENO3-Z, SSP-RK3 | Linf | rho | 1.11, 0.78, 0.74 | 5.50e-03 | below (1.3) |
| Manufactured Solution Walls | WENO5-Z, SSP-RK3 | L2 | rho | 1.21, 1.27, 1.02 | 5.18e-04 | below (1.7) |
| Manufactured Solution Walls | WENO5-Z, SSP-RK3 | Linf | rho | 0.61, 0.81, 0.73 | 5.58e-03 | below (1.3) |
| Manufactured Solution Outflow Euler | MUSCL VL, SSP-RK2 | L2 | p | 1.88, 1.63, 1.23 | 2.78e-04 | below (1.7) |
| Manufactured Solution Outflow Euler | MUSCL VL, SSP-RK2 | Linf | p | 1.37, 1.00, 0.72 | 2.45e-03 | below (1.3) |
| Manufactured Solution Outflow Euler | WENO3-Z, SSP-RK3 | L2 | v | 1.38, 1.30, 1.35 | 1.64e-04 | below (1.7) |
| Manufactured Solution Outflow Euler | WENO3-Z, SSP-RK3 | Linf | p | 0.87, 1.13, 0.69 | 2.70e-03 | below (1.3) |
| Manufactured Solution Outflow Euler | WENO5-Z, SSP-RK3 | L2 | p | 1.10, 1.19, 1.16 | 1.87e-04 | below (1.7) |
| Manufactured Solution Outflow Euler | WENO5-Z, SSP-RK3 | Linf | p | 0.34, 0.56, 0.69 | 2.77e-03 | below (1.3) |
| Manufactured Solution Outflow | MUSCL VL, SSP-RK2 | L2 | u | 2.10, 1.55, 1.01 | 1.66e-04 | below (1.7) |
| Manufactured Solution Outflow | MUSCL VL, SSP-RK2 | Linf | u | 1.62, 1.18, 0.73 | 2.93e-03 | below (1.3) |
| Manufactured Solution Outflow | WENO3-Z, SSP-RK3 | L2 | u | 2.17, 1.42, 1.20 | 1.85e-04 | below (1.7) |
| Manufactured Solution Outflow | WENO3-Z, SSP-RK3 | Linf | rho | 0.96, 0.79, 0.81 | 3.37e-03 | below (1.3) |
| Manufactured Solution Outflow | WENO5-Z, SSP-RK3 | L2 | u | 1.39, 1.21, 1.06 | 1.88e-04 | below (1.7) |
| Manufactured Solution Outflow | WENO5-Z, SSP-RK3 | Linf | rho | 0.58, 0.79, 0.79 | 3.44e-03 | below (1.3) |
