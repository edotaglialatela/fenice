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
spurious oscillations at the shock. Good results need a high-resolution reconstruction (MUSCL with the MC
or Van Leer limiter, or WENO), a TVD-preserving integrator (SSP-RK2 / RK3) and $n_x \ge 400$. With $n_y = 2$, MPI
uses the **X-direction** decomposition.

```powershell
build\release\fenice.exe --case "Shu-Osher Test"
build\release\fenice.exe --case "Shu-Osher Test" --set reconstruction=weno5 --set "time_scheme=SSP Runge-Kutta 3"
python cases\validation\benchmarks\shu_osher_schemes.py   # MUSCL-MC, WENO3-Z, WENO5-Z, 200-1600 cells vs 6400 cells
```

## Default parameters

{{defaults:Shu-Osher Test}}

## Validation results: MUSCL-MC, WENO3-Z and WENO5-Z

There is no closed-form solution, so `benchmarks/shu_osher_schemes.py` compares MUSCL with the MC limiter (SSP-RK2) with the two
WENO-Z reconstructions (SSP-RK3) on 200, 400, 800 and 1600 cells. Two references are used, both on 6400 cells: an MC run and a
WENO5-Z run (CFL 0.3); the error of every run, measured against the reference averaged onto its grid, differs by
at most 8 % between them (the table is in `results/validation/shu_osher_schemes/`), so the ranking does not depend on the reference.
L1 density error against the WENO5-Z reference:

| scheme | 200 | 400 | 800 | 1600 |
|---|---:|---:|---:|---:|
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
* **Order.** The observed orders between successive grids are scattered (0.3-1.9) and no scheme is asymptotic: the shock limits all of
  them, and the entropy waves are under-resolved on the coarse grids. The comparison is of the error at a given resolution, not of an order.
* The WENO runs use SSP-RK3 and the case's CFL (0.4); the MUSCL ones SSP-RK2.

## References

* Shu, C.-W. & Osher, S. (1989). Efficient implementation of essentially non-oscillatory
  shock-capturing schemes, II. *J. Comput. Phys.* 83(1), 32-78.
* Titarev, V.A. & Toro, E.F. (2004). Finite-volume WENO schemes for three-dimensional
  conservation laws. *J. Comput. Phys.* 201(1), 238-260.
