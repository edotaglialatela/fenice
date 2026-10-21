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
python validation\validationPoiseuille.py
python validation\mpiComparison.py      # serial vs MPI (results/poiseuille_flow[_mpi])
```

## Default parameters

{{defaults:Poiseuille Flow}}

## Validation results

<figure>
<img src="{{root}}img/convergence_poiseuille.png" style="width:100%">
<figcaption>L2 error of the velocity profile against the Poiseuille parabola as a function of grid resolution.</figcaption>
</figure>

<figure>
<img src="{{root}}img/poiseuille_mpi_comparison.png" style="width:100%">
<figcaption>Serial vs 4-rank MPI after 19 214 steps: the profiles are identical (maximum difference 0).</figcaption>
</figure>
