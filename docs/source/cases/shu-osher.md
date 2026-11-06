---
title: Shu-Osher shock / entropy-wave interaction
nav: Shu-Osher
section: Test cases
order: 50
---

The standard benchmark for a scheme's ability to capture a *strong shock* while
resolving smooth *high-frequency entropy waves* behind it (Shu & Osher 1989). A Mach-3
shock propagates rightward into a sinusoidal density perturbation, and the post-shock
density develops fine-scale oscillations that need low numerical dissipation.

The domain is $x \in [0, 10]$ with the shock at $x_s = 1$. The $y$ direction has only
$n_y = 2$ cells between slip walls, so the problem is effectively 1-D.

**Left, post-shock state** ($x < x_s$, Rankine-Hugoniot for Ma = 3, $\gamma = 1.4$):

$$\rho_L = 3.857143, \quad u_L = 2.629369, \quad v_L = 0, \quad p_L = 10.33333$$

**Right, pre-shock state** ($x \ge x_s$):

$$\rho_R(x) = 1 + \varepsilon\sin\bigl(k(x - x_0)\bigr), \quad u_R = 0, \quad p_R = 1, \qquad
  \varepsilon = 0.2,\ k = 5,\ x_0 = 5$$

The fine-scale post-shock oscillations are physical: they appear in the exact reference of
Titarev & Toro (2004). A too-diffusive scheme smears them, and a non-TVD scheme adds
spurious oscillations at the shock. Good results need a high-resolution MUSCL limiter (MC
or Van Leer), a TVD-preserving integrator (SSP-RK2) and $n_x \ge 400$. With $n_y = 2$, MPI
uses the **X-direction** decomposition.

```powershell
build\release\fenice.exe --case "Shu-Osher Test"
build\release\fenice.exe --case "Shu-Osher Test" --set muscl_limiter=Minmod
python cases\validation\benchmarks\shu_osher.py      # MC and Minmod, 200-1600 cells vs a 6400-cell reference
```

## Default parameters

{{defaults:Shu-Osher Test}}

## Validation results

There is no closed-form solution, so the reference is the MC run on 6400 cells, and the
error of each grid is measured against the reference averaged onto it.

<figure>
<img src="{{root}}img/shu_osher.png" style="width:100%">
<figcaption>Density at t = 1.8 with MC and Minmod on 400 (open markers) and 800 cells (filled); the inset zooms on the
post-shock entropy waves, which Minmod flattens more. The script prints the L1 error against the reference
(MC: 0.77, 0.55, 0.22, 0.085 on 200, 400, 800, 1600 cells; Minmod: 0.85, 0.69, 0.44, 0.19).
The shock limits both limiters to about first order; MC stays about 2× more accurate.</figcaption>
</figure>

## Reconstruction: limiters and WENO

`benchmarks/shu_osher_schemes.py` adds the two WENO-Z reconstructions (SSP-RK3) to the two limiters above, on the same four grids. Two
references are used, both on 6400 cells: the MC run of the study above and a WENO5-Z run (CFL 0.3); the error of every run differs by
at most 8 % between them (the table is in `results/validation/shu_osher_schemes/`), so the ranking does not depend on the reference.
L1 density error against the WENO5-Z reference:

| scheme | 200 | 400 | 800 | 1600 |
|---|---:|---:|---:|---:|
| MUSCL Minmod | 0.846 | 0.695 | 0.444 | 0.199 |
| MUSCL MC | 0.767 | 0.556 | 0.228 | 0.092 |
| WENO3-Z | 0.790 | 0.590 | 0.227 | 0.077 |
| WENO5-Z | 0.540 | 0.149 | 0.089 | 0.042 |

<figure>
<img src="{{root}}img/shu_osher_schemes.png" style="width:100%">
<figcaption>Left: the post-shock entropy waves on 400 cells with each reconstruction against the 6400-cell WENO5-Z reference.
Right: $L_1$ density error against the number of cells; the dashed line is first order.</figcaption>
</figure>

<figure>
<img src="{{root}}img/shu_osher_schemes_profile.png" style="width:100%">
<figcaption>The whole profile on 400 cells.</figcaption>
</figure>

* **WENO5-Z** is the only scheme that resolves the amplitude of the entropy waves on 400 cells; the others have damped them
  to about 4.1-4.3 against 4.6. Its error on 400 cells is below that of MC on 800, and on 800 cells it equals that of MC on 1600: the
  fifth-order reconstruction is worth about a factor two in resolution here.
* **WENO3-Z** is no better than MC on 200-800 cells (0.79 against 0.77, 0.59 against 0.56) and 16 % better on 1600; it has the stencil
  of MUSCL, so on this problem it only buys what the third-order interior brings.
* **Minmod** is the most diffusive on every grid.
* **Order.** The observed orders between successive grids are scattered (0.3-1.9) and no scheme is asymptotic: the shock limits all of
  them, and the entropy waves are under-resolved on the coarse grids. The comparison is of the error at a given resolution, not of an order.
* The WENO runs use SSP-RK3 and the case's CFL (0.4); the MUSCL ones SSP-RK2, as in the study above.

## References

* Shu, C.-W. & Osher, S. (1989). Efficient implementation of essentially non-oscillatory
  shock-capturing schemes, II. *J. Comput. Phys.* 83(1), 32-78.
* Titarev, V.A. & Toro, E.F. (2004). Finite-volume WENO schemes for three-dimensional
  conservation laws. *J. Comput. Phys.* 201(1), 238-260.
