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
python validation\shu_osher.py      # MC and Minmod, 200-1600 cells vs a 6400-cell reference
```

## Default parameters

{{defaults:Shu-Osher Test}}

## Validation results

There is no closed-form solution, so the reference is the MC run on 6400 cells, and the
error of each grid is measured against the reference averaged onto it.

<figure>
<img src="{{root}}img/shu_osher.png" style="width:100%">
<figcaption>Density at t = 1.8 with MC and Minmod on 400 cells; the inset zooms on the
post-shock entropy waves, which Minmod flattens more. Right: L1 error against the reference.
The shock limits both limiters to about first order; MC stays about 2× more accurate.</figcaption>
</figure>

## References

* Shu, C.-W. & Osher, S. (1989). Efficient implementation of essentially non-oscillatory
  shock-capturing schemes, II. *J. Comput. Phys.* 83(1), 32-78.
* Titarev, V.A. & Toro, E.F. (2004). Finite-volume WENO schemes for three-dimensional
  conservation laws. *J. Comput. Phys.* 201(1), 238-260.
