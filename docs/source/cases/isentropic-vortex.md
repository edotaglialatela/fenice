---
title: Isentropic vortex - advection accuracy
nav: Isentropic vortex
section: Test cases
order: 80
---

A smooth, exact solution of the Euler equations (Shu 1998). A vortex is superposed on a
uniform free stream $(u_\infty, v_\infty) = (1, 1)$ on the periodic square $[0, 10]^2$ and
is advected without change of shape. The exact solution at any time is therefore the
initial vortex moved to $(5 + t, 5 + t)$, and after $t = 10$ it is back where it started.
With $r$ the distance to the vortex centre and strength $\beta = 5$:

$$\delta u = -\frac{\beta}{2\pi}\,e^{(1 - r^2)/2}\,(y - y_c), \qquad
  \delta v = \frac{\beta}{2\pi}\,e^{(1 - r^2)/2}\,(x - x_c)$$

$$T = 1 - \frac{(\gamma - 1)\beta^2}{8\gamma\pi^2}\,e^{1 - r^2}, \qquad
  \rho = T^{1/(\gamma - 1)}, \qquad p = \rho T$$

The entropy $p/\rho^\gamma$ is uniform and nothing should dissipate it. Every deviation
from the exact field is therefore numerical error, which makes this the standard check of
the **order of accuracy** of the spatial and temporal schemes. Unlike Sod or Shu-Osher it
has no shocks, so the limiters should only clip at the smooth extrema.

```powershell
build\release\fenice.exe --case "Isentropic Vortex"
python cases\validation\isentropic_vortex.py
```

## Default parameters

{{defaults:Isentropic Vortex}}

## Validation results

<figure>
<img src="{{root}}img/isentropic_vortex_validation.png" style="width:100%">
<figcaption>After one full period (t = 10). Left: centreline density at N = 40, 80, 160 against the
exact profile. Right: L1 density error (MC limiter) with observed orders 1.8, 2.25 and 1.95, and the
other limiters at N = 80. MC is the most accurate, followed by Van Leer, Koren and Minmod.</figcaption>
</figure>

The automated check `isentropic_vortex_convergence` in `fenice_tests` runs N = 20, 40, 80 to
t = 1 and requires an order above 1.7 (measured 2.2). The regression suite also runs the
vortex under MPI in both decomposition directions, including a thin 64×6 mesh that forces
column strips with periodic wrap-around.

## Reference

* Shu, C.-W. (1998). Essentially non-oscillatory and weighted essentially non-oscillatory
  schemes for hyperbolic conservation laws. In *Advanced Numerical Approximation of
  Nonlinear Hyperbolic Equations*, Lecture Notes in Mathematics 1697, Springer, 325-432.
