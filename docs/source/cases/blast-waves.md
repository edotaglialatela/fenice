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
side), $\rho, p > 0$, the L1 density difference to the reference decreasing with an observed rate of at least 0.4, and the
position of the density maximum within 0.02 of the reference's and not moving away with refinement.

<figure>
<img src="{{root}}img/blast_waves.png" style="width:100%">
<figcaption>Density and pressure at t = 0.038 on 400, 800 and 1600 cells and the 3200-cell reference. Conservation errors
are 2-5e-16; the L1 density differences to the reference are 9.6e-2, 4.5e-2 and 1.8e-2 (observed rates 1.1 and 1.3).</figcaption>
</figure>

The density maximum is a spike of one or two cells: its position converges in physical distance (0.0095, 0.0077, 0.0036 from
400 to 1600 cells), not in a fixed number of cells.

## References

* Woodward, P. & Colella, P. (1984). The numerical simulation of two-dimensional fluid flow with strong shocks.
  *J. Comput. Phys.* 54(1), 115-173.
