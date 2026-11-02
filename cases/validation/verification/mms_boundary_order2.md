# Manufactured solutions with non-periodic sides, boundary_order = 2 (ghost layers, faces on the boundary reconstructed)

Measured 2026-11-02 with `mms_boundary.py --boundary-order 2 --schemes muscl,weno3` (WENO5 is not available with this
closure yet). Same setup as `mms_boundary_order1.md`; compare row by row. The Euler solutions isolate the convective
closure; the Navier-Stokes ones also include the viscous closure of the walls (moving-wall ghost energy fixed under the
same key; the viscous face treatment is still first order at the boundary).

| solution | scheme | norm | worst variable | orders per grid pair | finest rho error | band |
|---|---|---|---|---|---|---|
| Manufactured Solution Walls Euler | MUSCL VL, SSP-RK2 | L2 | p | 2.47, 2.06, 2.08 | 8.40e-05 | ok (1.7) |
| Manufactured Solution Walls Euler | MUSCL VL, SSP-RK2 | Linf | v | 2.01, 1.63, 1.47 | 5.06e-04 | ok (1.3) |
| Manufactured Solution Walls Euler | WENO3-Z, SSP-RK3 | L2 | v | 2.14, 2.14, 2.45 | 5.11e-05 | ok (1.7) |
| Manufactured Solution Walls Euler | WENO3-Z, SSP-RK3 | Linf | v | 1.62, 1.79, 1.78 | 3.22e-04 | ok (1.3) |
| Manufactured Solution Walls | MUSCL VL, SSP-RK2 | L2 | v | 1.78, 1.14, 1.04 | 1.23e-04 | below (1.7) |
| Manufactured Solution Walls | MUSCL VL, SSP-RK2 | Linf | u | 2.12, 0.69, 0.76 | 7.31e-04 | below (1.3) |
| Manufactured Solution Walls | WENO3-Z, SSP-RK3 | L2 | v | 2.06, 1.09, 1.00 | 1.13e-04 | below (1.7) |
| Manufactured Solution Walls | WENO3-Z, SSP-RK3 | Linf | u | 1.84, 0.61, 0.87 | 7.65e-04 | below (1.3) |
| Manufactured Solution Outflow Euler | MUSCL VL, SSP-RK2 | L2 | v | 1.94, 1.50, 1.15 | 2.59e-04 | below (1.7) |
| Manufactured Solution Outflow Euler | MUSCL VL, SSP-RK2 | Linf | u | 1.84, 1.49, 0.90 | 1.41e-03 | below (1.3) |
| Manufactured Solution Outflow Euler | WENO3-Z, SSP-RK3 | L2 | u | 1.99, 2.15, 2.32 | 5.85e-05 | ok (1.7) |
| Manufactured Solution Outflow Euler | WENO3-Z, SSP-RK3 | Linf | u | 1.60, 1.98, 1.83 | 4.85e-04 | ok (1.3) |
| Manufactured Solution Outflow | MUSCL VL, SSP-RK2 | L2 | u | 2.01, 1.48, 0.99 | 9.51e-05 | below (1.7) |
| Manufactured Solution Outflow | MUSCL VL, SSP-RK2 | Linf | u | 1.62, 1.18, 0.73 | 7.23e-04 | below (1.3) |
| Manufactured Solution Outflow | WENO3-Z, SSP-RK3 | L2 | u | 2.04, 1.73, 2.52 | 4.84e-05 | ok (1.7) |
| Manufactured Solution Outflow | WENO3-Z, SSP-RK3 | Linf | p | 1.17, 2.04, 2.08 | 5.92e-04 | ok (1.3) |
