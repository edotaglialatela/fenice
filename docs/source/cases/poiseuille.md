---
title: Hagen-Poiseuille channel flow
nav: Poiseuille flow
section: Test cases
order: 20
---

Laminar flow between two infinite parallel plates (no-slip top and bottom), driven by a
constant pressure gradient. The exact steady solution is the **Poiseuille parabola**:

$$u(y) = \frac{1}{2\mu}\Bigl(-\frac{\partial p}{\partial x}\Bigr)\Bigl(\frac{L_y^2}{4} - y^2\Bigr), \qquad
  u_{\max} = \tfrac32\,\bar u, \qquad -\frac{\partial p}{\partial x} = \frac{12\,\mu\,\bar u}{L_y^2}$$

Here $\bar u$ is the bulk (inlet) velocity and $y \in [-L_y/2, L_y/2]$. The run starts
from a uniform velocity $u = u_{\rm in}$ with the **exact steady pressure profile**
$p(x) = p_{\rm out} + 12\mu u_{\rm in}(L_x - x)/L_y^2$, which suppresses the long acoustic
transient otherwise needed to build the gradient.

The case forces a **mass-flow inlet** on the left ($\rho_0 = 1$, $u_{\rm in} = 0.01$), a
**pressure outlet** on the right ($p_{\rm out} = 1$) and **no-slip walls**. With the
defaults, $\mathrm{Re} = \rho_0 u_{\rm in} L_y/\mu = 0.5$. The flow is very low-Mach,
which makes it a good target for `HLLC (Low-Mach Thornber)` and `BDF2 Dual Time`.

```powershell
build\release\fenice.exe --case "Poiseuille Flow"
build\release\fenice.exe --case "Poiseuille Flow" --set time_scheme=bdf2
python cases\validation\poiseuille.py     # grid convergence + serial vs 4 MPI ranks
```

## Default parameters

{{defaults:Poiseuille Flow}}

## Validation results

<figure>
<img src="{{root}}img/poiseuille_convergence.png" style="width:100%">
<figcaption>Velocity profile at x = 0.95 L<sub>x</sub> on n<sub>y</sub> = 10, 20, 40 and the
relative L2 error against the parabola: second order (observed 2.10 and 1.93).</figcaption>
</figure>

<figure>
<img src="{{root}}img/poiseuille_mpi.png" style="width:100%">
<figcaption>Serial vs 4 MPI ranks (y-strips, dotted): the fields agree to 1.8e-15, the
round-off of the all-reduced mass-flow inlet average.</figcaption>
</figure>
