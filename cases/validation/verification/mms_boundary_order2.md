# Manufactured solutions with non-periodic sides, boundary_order = 2 (ghost layers, faces on the boundary reconstructed)

Measured 2026-11-02 with `mms_boundary.py --boundary-order 2` (MUSCL, WENO3-Z, WENO5-Z). Same setup as
`mms_boundary_order1.md`; compare row by row. The Euler solutions isolate the convective closure; the Navier-Stokes ones also include the
viscous closure of the walls (moving-wall ghost energy and the cross derivatives on the boundary faces, both under the same key).

| solution | scheme | norm | worst variable | orders per grid pair | finest rho error | band |
|---|---|---|---|---|---|---|
| Manufactured Solution Walls Euler | MUSCL VL, SSP-RK2 | L2 | p | 2.47, 2.06, 2.08 | 8.40e-05 | ok (1.7) |
| Manufactured Solution Walls Euler | MUSCL VL, SSP-RK2 | Linf | v | 2.01, 1.63, 1.47 | 5.06e-04 | ok (1.3) |
| Manufactured Solution Walls Euler | WENO3-Z, SSP-RK3 | L2 | v | 2.14, 2.14, 2.45 | 5.11e-05 | ok (1.7) |
| Manufactured Solution Walls Euler | WENO3-Z, SSP-RK3 | Linf | v | 1.62, 1.79, 1.78 | 3.22e-04 | ok (1.3) |
| Manufactured Solution Walls Euler | WENO5-Z, SSP-RK3 | L2 | p | 2.17, 1.95, 1.99 | 4.36e-05 | ok (1.7) |
| Manufactured Solution Walls Euler | WENO5-Z, SSP-RK3 | Linf | u | 1.55, 2.04, 1.95 | 2.36e-04 | ok (1.3) |
| Manufactured Solution Walls | MUSCL VL, SSP-RK2 | L2 | v | 2.40, 2.04, 1.95 | 6.62e-05 | ok (1.7) |
| Manufactured Solution Walls | MUSCL VL, SSP-RK2 | Linf | v | 2.64, 1.80, 1.90 | 3.73e-04 | ok (1.3) |
| Manufactured Solution Walls | WENO3-Z, SSP-RK3 | L2 | v | 2.54, 2.24, 2.03 | 4.74e-05 | ok (1.7) |
| Manufactured Solution Walls | WENO3-Z, SSP-RK3 | Linf | p | 1.96, 1.83, 1.86 | 4.54e-04 | ok (1.3) |
| Manufactured Solution Walls | WENO5-Z, SSP-RK3 | L2 | p | 1.87, 1.97, 2.00 | 2.86e-05 | ok (1.7) |
| Manufactured Solution Walls | WENO5-Z, SSP-RK3 | Linf | p | 1.93, 1.96, 1.99 | 6.95e-05 | ok (1.3) |
| Manufactured Solution Outflow Euler | MUSCL VL, SSP-RK2 | L2 | v | 1.94, 1.50, 1.15 | 2.59e-04 | below (1.7) |
| Manufactured Solution Outflow Euler | MUSCL VL, SSP-RK2 | Linf | u | 1.84, 1.49, 0.90 | 1.41e-03 | below (1.3) |
| Manufactured Solution Outflow Euler | WENO3-Z, SSP-RK3 | L2 | u | 1.99, 2.15, 2.32 | 5.85e-05 | ok (1.7) |
| Manufactured Solution Outflow Euler | WENO3-Z, SSP-RK3 | Linf | u | 1.60, 1.98, 1.83 | 4.85e-04 | ok (1.3) |
| Manufactured Solution Outflow Euler | WENO5-Z, SSP-RK3 | L2 | u | 2.02, 1.98, 1.99 | 3.42e-05 | ok (1.7) |
| Manufactured Solution Outflow Euler | WENO5-Z, SSP-RK3 | Linf | u | 1.87, 1.83, 1.96 | 1.26e-04 | ok (1.3) |
| Manufactured Solution Outflow | MUSCL VL, SSP-RK2 | L2 | u | 2.00, 1.47, 0.99 | 9.57e-05 | below (1.7) |
| Manufactured Solution Outflow | MUSCL VL, SSP-RK2 | Linf | u | 1.58, 1.19, 0.76 | 7.13e-04 | below (1.3) |
| Manufactured Solution Outflow | WENO3-Z, SSP-RK3 | L2 | u | 2.04, 1.71, 2.48 | 4.84e-05 | ok (1.7) |
| Manufactured Solution Outflow | WENO3-Z, SSP-RK3 | Linf | p | 1.17, 2.04, 2.09 | 5.84e-04 | ok (1.3) |
| Manufactured Solution Outflow | WENO5-Z, SSP-RK3 | L2 | rho | 2.08, 2.02, 1.99 | 1.94e-05 | ok (1.7) |
| Manufactured Solution Outflow | WENO5-Z, SSP-RK3 | Linf | v | 2.33, 1.78, 1.88 | 5.78e-05 | ok (1.3) |
