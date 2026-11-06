---
title: Woodward-Colella interacting blast waves
nav: Blast waves
section: Test cases
order: 52
---

Two strong shocks (pressure 1000 and 100 against 0.01, $\gamma = 1.4$) start at the two ends of a tube closed by reflecting
walls. They run into each other, and their contacts interact: a severe test of shock capturing, of positivity (the pressure
ratio is $10^5$) and of conservation (Woodward & Colella 1984). The $y$ direction has four cells between slip walls, so the
problem is effectively 1-D. There is no exact solution; the study compares coarse runs with a fine run of the same code.

```powershell
build\release\fenice.exe --case "Woodward-Colella Blast Waves"
python cases\validation\verification\blast_waves.py      # 400, 800, 1600 cells vs a 3200-cell reference (about 30 s)
```

## Default parameters

{{defaults:Woodward-Colella Blast Waves}}

## Validation results

The bands are fixed in the script before the first run: mass and energy conserved to a relative $10^{-11}$ (walls on every
side), $\rho, p > 0$, the L1 density difference to the reference decreasing with an observed rate of at least 0.4. The position of the density
maximum is reported but is not a band (see below).

<figure>
<img src="{{root}}img/blast_waves.png" style="width:100%">
<figcaption>Density and pressure at t = 0.038 on 400, 800 and 1600 cells and the 3200-cell reference. Conservation errors
are 2-6e-16; the L1 density differences to the reference are 0.109, 0.053 and 0.034 (observed rates 1.0 and 0.6).</figcaption>
</figure>

**Sensitivity to rounding.** This problem amplifies rounding-level differences. The whole study was repeated six times with the
CFL changed by 1e-12 (relative), which changes the time step by rounding only, once with `shock_blend = 0.2` (the default) and once
with `shock_blend = 0`. The 3200-cell reference was recomputed each time.

| `shock_blend` | L1 at 400 | L1 at 800 | L1 at 1600 | observed rates | rate band (0.4) |
|---|---|---|---|---|---|
| 0.2 | 0.101 to 0.118 | 0.053 to 0.063 | 0.020 (3 runs) or 0.034 (3 runs) | 0.68 to 1.07 and 0.63 to 1.67 | 6 of 6 pass |
| 0 | 0.133 to 0.193 | 0.082 to 0.163 | 0.033 to 0.036 (3 runs) or 0.10 (3 runs) | 0.03 to 1.23 and 0.36 to 2.31 | 1 of 6 pass (one run has a non-monotone error, no rate) |

The error at the finest level has two values, not a spread around one. A change of a single rounding in the flux (the replacement of
divisions by multiplications by a reciprocal) moved the default from 0.018 to 0.034 at 1600 cells. With the dissipation on,
the observed rate stays above the band in every run; with it off the rate band fails in five of six, so the blend is the more robust choice here. The error at
the finest level is bimodal either way (0.020 or 0.034 with it on, 0.033 to 0.036 or 0.10 with it off), so no single run says more
than which of the two values it landed on.

The density maximum is a spike of one or two cells, and two spikes have nearly the same height, so the position of the maximum
jumps between them under the same perturbations: its distance to the reference's is 0.003 or 0.12 at 400 cells (0.006 to 0.009
at 800, 0.0008 to 0.0073 at 1600). It converges in physical distance, not in a fixed number of cells, and is quantised to the cell
centres of each grid. It is printed by the study and is not a pass or fail criterion.

## WENO3-Z against the MUSCL run

```powershell
python cases\validation\verification\blast_waves_weno.py      # about 4 min on 2 threads
```

The study above uses MUSCL (MC limiter, SSP-RK2). `blast_waves_weno.py` repeats it with `reconstruction = weno3` and SSP-RK3
(everything else unchanged: HLLC, `shock_blend = 0.2`, CFL 0.4) on 400, 800, 1600 cells and a 3200-cell WENO3-Z reference, and
compares the solutions with the 800-cell MUSCL run. The bands are those of the MUSCL study (conservation 1e-11, positivity,
self-convergence rate at least 0.4); the comparison with MUSCL is reported, not a band, because the two schemes are different.
Differences are the mean $|\rho_a - \rho_b|$ after averaging both rows onto the coarser mesh.

| WENO3-Z cells | L1 to its own 3200-cell reference | L1 to the MUSCL 800-cell run | conservation | min $\rho$, min $p$ |
|---|---|---|---|---|
| 400 | 0.103 | 0.069 | 9e-14 | 0.149, 18.9 |
| 800 | 0.049 | 0.036 | 2e-13 | 0.147, 19.1 |
| 1600 | 0.018 | 0.043 | 4e-13 | 0.146, 19.2 |

The observed rates are 1.09 and 1.44 and the study passes. For reference, the MUSCL 800-cell run is 0.053 from its own 3200-cell
reference, and the two 3200-cell references (WENO3-Z and MUSCL) differ by 0.013.

<figure>
<img src="{{root}}img/blast_waves_weno.png" alt="Blast waves: WENO3-Z on three meshes and the MUSCL 800-cell run">
<figcaption>Density at t = 0.038 (left, and a zoom on the right) for WENO3-Z on 400, 800 and 1600 cells, MUSCL on 800 cells and
both 3200-cell references; below, the density difference of WENO3-Z 800 and 1600 to MUSCL 800 (both averaged onto 800 cells) and
of the two references.</figcaption>
</figure>

* At the same 800 cells WENO3-Z is slightly closer to its reference than MUSCL is to its own (0.049 against 0.053); the two
  800-cell solutions differ from each other by 0.036, about as much as either differs from its reference, so at this
  resolution the scheme is not what separates them from the converged answer, the mesh is. They disagree most at the density
  peak ($x \approx 0.76$-$0.80$) and, less, at the left foot of the structure ($x \approx 0.6$), where WENO3-Z is steeper.
* The distance of the 1600-cell WENO3-Z run to the 800-cell MUSCL run (0.043) is larger than at 800 cells because it is
  dominated by the error of the 800-cell MUSCL run itself (0.053), not by a disagreement of the schemes: the references differ
  by 0.013.
* **Sensitivity.** The 1600-cell run and its reference were repeated with the CFL changed by 1e-12 (relative) in both
  directions: the error is 0.017854, 0.017857 and 0.017858. The error does not jump between two values as the MUSCL error
  does (0.020 or 0.034); with WENO3-Z at this resolution the study is insensitive to the perturbation. That is two perturbations
  at one resolution, with the dissipation on only, a smaller sweep than the six runs of the MUSCL study.
* The 1600-cell WENO3-Z error (0.018) is about that of the lower MUSCL value (0.020), so WENO3-Z lands where the better of the
  two MUSCL outcomes does, without the other outcome appearing.

## References

* Woodward, P. & Colella, P. (1984). The numerical simulation of two-dimensional fluid flow with strong shocks.
  *J. Comput. Phys.* 54(1), 115-173.
