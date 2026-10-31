---
title: Noh implosion and Sedov-Taylor blast
nav: Noh and Sedov
section: Test cases
order: 54
---

Two point-symmetric problems on one quarter of the plane (symmetry walls on $x = 0$ and $y = 0$, transmissive far sides).

**Noh implosion** (Noh 1987), $\gamma = 5/3$: cold gas ($p = 10^{-6}$) flows radially inward at unit speed. A shock moves out at
speed $1/3$; the exact solution has density 16 and pressure $16/3$ behind it ($r < t/3$) and density $1 + t/r$ ahead. The cold
inflow carries no signal, so the transmissive far sides are exact. The numerical solution shows the known *wall heating*: the
density is low in a small region around the origin, which the bands exclude ($r \ge 0.04$).

**Sedov-Taylor blast**, $\gamma = 1.4$: unit energy (full plane) in a disc of 3.5 cells at the origin of a cold uniform gas.
The shock radius follows $R = (E t^2 / \alpha \rho)^{1/4}$, i.e. $R \propto t^{1/2}$. The constant $\alpha$ is not computed
here, so the study checks the scaling law, the peak density (strong-shock limit 6), energy conservation and symmetry, not the
prefactor (the runs give $\alpha \approx 1.0$).

```powershell
build\release\fenice.exe --case "Noh Implosion"
python cases\validation\verification\implosions.py      # Noh 100, 200; Sedov 200, 400 at four times (about 3 minutes)
```

## Default parameters

{{defaults:Noh Implosion}}

## Validation results

Bands, fixed before the first run. Noh: shock radius within 3 cells of $t/3$, mean density on $0.08 \le r \le 0.15$ within 10 %
of 16, relative L1 density error on $0.04 \le r \le 0.5$ below 8 %, the error falling from 100 to 200 cells, $x \leftrightarrow y$
symmetry to $10^{-9}$. Sedov: energy conserved to $10^{-11}$, exponent of $R(t)$ in $0.5 \pm 0.03$, peak radially averaged
density in $[3.5, 6.2]$, $R(0.2)$ of 200 and 400 cells within two coarse cells.

<figure>
<img src="{{root}}img/implosions.png" style="width:100%">
<figcaption>Left: Noh density against the radius at t = 0.6 (the shock radius is within a cell of 0.2; plateau 15.6 and 15.8 on
100 and 200 cells, relative L1 error 2.9 % and 1.6 %). Right: Sedov radially averaged density at t = 0.2 (exponent 0.495
and 0.506, peak density 4.6 and 5.1 on 200 and 400 cells per unit length; symmetry exact).</figcaption>
</figure>

## References

* Noh, W.F. (1987). Errors for calculations of strong shocks using an artificial viscosity and an artificial heat flux.
  *J. Comput. Phys.* 72(1), 78-120.
* Sedov, L.I. (1959). *Similarity and Dimensional Methods in Mechanics*. Academic Press.
