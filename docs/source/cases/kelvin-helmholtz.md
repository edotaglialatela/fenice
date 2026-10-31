---
title: Kelvin-Helmholtz instability
nav: Kelvin-Helmholtz
section: Test cases
order: 56
---

A smooth shear layer $u = U \tanh\bigl((y - 8)/\delta\bigr)$ with $U = 0.2$, $\delta = 1$, density 1 and sound speed 1 (Mach 0.2
on each side) is periodic in $x$ and closed by slip walls at $y = 0$ and $16$. The box is one wavelength of the most unstable
mode of the tanh layer, $\alpha\delta = 0.4446$ ($L_x = 14.1319$); a small transverse velocity of that wavelength seeds the
growth. The incompressible layer grows at $\sigma = 0.1897\,U/\delta$ (Michalke 1964), here $\sigma = 0.03794$.

```powershell
build\release\fenice.exe --case "Kelvin-Helmholtz"
python cases\validation\verification\kelvin_helmholtz.py      # 96 x 109 and 144 x 163 cells (about 10 s)
```

## Default parameters

{{defaults:Kelvin-Helmholtz}}

## Validation results

The transverse kinetic energy $\sum \rho v^2$ is fitted as $\exp(2\sigma t)$ on $30 \le t \le 80$. Bands, fixed before the first
run: $\sigma$ within 20 % of the incompressible value on the finer grid; the two grids within 10 % of each other; mass and
energy conserved to $10^{-11}$; $\rho, p > 0$.

<figure>
<img src="{{root}}img/kelvin_helmholtz.png" style="width:70%">
<figcaption>Growth of the transverse kinetic energy on two grids, with the analytic slope. The fitted rates are 0.03427
(96 x 109) and 0.03410 (144 x 163), 10 % below the incompressible value on both grids. They agree with each other to 0.5 %,
so it is not set by the grid resolution at these sizes. The incompressible formula ignores the finite Mach number and the
walls; whether they account for the 10 % was not tested here (it would take a run at lower Mach number and a taller box).</figcaption>
</figure>

## References

* Michalke, A. (1964). On the inviscid instability of the hyperbolic-tangent velocity profile. *J. Fluid Mech.* 19(4), 543-556.
