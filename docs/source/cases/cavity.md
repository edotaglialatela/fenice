---
title: Lid-driven cavity
nav: Lid-driven cavity
section: Test cases
order: 30
---

A square box ($L_x = L_y = 1$) whose top wall moves at $u_{\rm lid}$ drives a primary
recirculation vortex. The Reynolds number is

$$\mathrm{Re} = \frac{\rho\,u_{\rm lid}\,L}{\mu} = 400$$

with the defaults ($\rho = 1$, $u_{\rm lid} = 1$, $\mu = 2.5\times10^{-3}$), a regime where
the flow is steady with a single dominant vortex. The ambient pressure
$p_{\rm out} = 100$ suppresses compressibility: $c = \sqrt{\gamma p/\rho} \approx 11.8$ gives
$\mathrm{Ma} \approx 0.085$. Density variations are therefore $O(\mathrm{Ma}^2) \approx 0.7\%$,
and the results compare directly with the incompressible Ghia (1982) reference. The
walls are no-slip, and the lid is a `moving_wall` with velocity `u_in`.

```powershell
build\release\fenice.exe --case "Lid Driven Cavity"
python cases\validation\benchmarks\cavity.py         # 32², 64², 128², both orders
```

The validation script runs every grid to the steady state ($t = 40$), with 1st-order
Godunov and with MUSCL, and compares the centreline velocities with Ghia's tables and the
primary vortex with Ghia's stream function.

## Default parameters

{{defaults:Lid Driven Cavity}}

## Validation results

<figure>
<img src="{{root}}img/cavity_validation.png" style="width:100%">
<figcaption>Left and centre: u on the vertical centreline and v on the horizontal one, against
Ghia et al. (1982). Right: RMS difference from Ghia's points over the grids; MUSCL converges
at 2nd order (slope −2.1), Godunov at 1st order (slope −1.0).</figcaption>
</figure>

| Grid | Scheme | RMS error u | RMS error v | Vortex centre | ψ<sub>min</sub> |
|---|---|---:|---:|---|---:|
| 32² | 1st order | 0.064 | 0.078 | (0.609, 0.641) | −0.0847 |
| 128² | 1st order | 0.018 | 0.020 | (0.559, 0.605) | −0.1056 |
| 32² | MUSCL | 0.024 | 0.028 | (0.547, 0.609) | −0.1057 |
| 64² | MUSCL | 0.0052 | 0.0045 | (0.555, 0.602) | −0.1123 |
| 128² | MUSCL | **0.0013** | **0.0016** | **(0.551, 0.605)** | **−0.1136** |
| Ghia et al. | | | | (0.5547, 0.6055) | −0.1139 |

At 128² with MUSCL the primary vortex sits within one cell of Ghia's and its stream
function within 0.3 %. One point of Ghia's v table, −0.238 at x = 0.9063, is left out of the
v error: it cannot lie on a smooth profile between its neighbours (−0.450 at 0.8594, −0.228 at
0.9453), and every grid converges to about −0.39 there, so it looks like a misprint. It is
still plotted (red cross).

<figure>
<img src="{{root}}img/cavity_streamlines.png" style="width:70%; display:block; margin:0 auto;">
<figcaption>Streamlines and speed at 128² (MUSCL): the primary vortex, its centre against
Ghia's, and the secondary vortex in the bottom-right corner.</figcaption>
</figure>

The 1st-order runs show why the default 30 × 30 grid is a quick check, not a converged
answer: numerical viscosity weakens the vortex (ψ<sub>min</sub> = −0.085 at 32²) and pushes it
towards the top-right. Use MUSCL and 64² or more for quantitative work.

## Reconstruction and boundary order

RMS error of the centreline u / v against Ghia et al. (Re 400), MUSCL (MC, SSP-RK2), `weno3` and `weno5` (SSP-RK3), with
`boundary_order` 1 and 2 (single runs, `t_end = 40`):

| scheme | `boundary_order` | $32^2$ | $64^2$ | $128^2$ |
|---|---|---|---|---|
| MUSCL (MC) | 1 | 2.24e-02 / 2.53e-02 | 4.90e-03 / 3.88e-03 | 1.31e-03 / 1.63e-03 |
| MUSCL (MC) | 2 | 1.79e-02 / 2.27e-02 | 3.92e-03 / 3.12e-03 | 1.40e-03 / 1.86e-03 |
| `weno3` | 1 | 2.59e-02 / 3.11e-02 | 5.61e-03 / 5.32e-03 | 1.18e-03 / 1.70e-03 |
| `weno3` | 2 | 1.70e-02 / 2.24e-02 | 3.81e-03 / 3.73e-03 | 1.24e-03 / 1.97e-03 |
| `weno5` | 1 | 2.64e-02 / 3.06e-02 | 5.47e-03 / 5.05e-03 | 1.17e-03 / 1.74e-03 |
| `weno5` | 2 | 1.39e-02 / 1.73e-02 | 3.25e-03 / 2.81e-03 | 1.26e-03 / 2.05e-03 |

Second-order boundary faces lower the error on $32^2$ and $64^2$ for all three schemes but raise it by 5 to 8 % (u) on $128^2$, so the observed
order of the last pair falls from 1.9-2.3 to 1.4-1.6. On the fine grid the three schemes are within 12 % of each other: on this smooth,
weakly compressible flow the reconstruction order does not show, and the level of about 1e-3 is not set by the reconstruction (the tabulated reference
data, the finite Mach number and the boundary treatment were not separated).

## Reference

* Ghia, U., Ghia, K.N. & Shin, C.T. (1982). High-Re solutions for incompressible flow
  using the Navier-Stokes equations and a multigrid method. *J. Comput. Phys.* 48(3), 387-411.
