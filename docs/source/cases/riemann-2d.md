---
title: Two-dimensional Riemann problems
nav: 2-D Riemann
section: Test cases
order: 58
---

The unit square is cut at $(0.5, 0.5)$ into four constant states (quadrants NE, NW, SW, SE). Four configurations of the Lax-Liu
family (3, 4, 6 and 12) are run on 100, 200 and 400 cells per side, transmissive on all sides, $\gamma = 1.4$, MUSCL (MC) with
SSP-RK2 and HLLC. The initial data are entered in `cases/validation/verification/riemann_2d.py` as they are commonly tabulated;
they were not compared with the original papers, and the repository holds no published contour data, so the study makes **no
claim about published shock positions**.

```powershell
python cases\validation\verification\riemann_2d.py           # 100, 200 and AMR on configuration 3 (7 s)
python cases\validation\verification\riemann_2d.py --full    # + the 400 x 400 reference and the convergence rates
```

## Validation results

Bands, fixed before the first run: finite, $\rho, p > 0$ everywhere; configurations 3 and 4 symmetric under $x \leftrightarrow y$
to $10^{-9}$ (configuration 12 is not symmetric: its NW and SE densities differ); with `--full`, the mean density difference
to the 400 x 400 run decreases at an observed rate of at least 0.3 from 100 to 200; AMR on configuration 3 (base 100 x 100, one
level) is closer to the uniform 200 x 200 run than the uniform 100 x 100 run is, by a factor of at least 0.75.

<figure>
<img src="{{root}}img/riemann_2d.png" style="width:100%">
<figcaption>Density contours on 400 x 400 cells. Observed rates 0.90, 1.08, 1.24 and 1.34 (configurations 3, 4, 6, 12);
symmetry exact (0) on both grids; the AMR run differs from the uniform 200 x 200 run by 1.0e-3, the uniform 100 x 100 run
by 1.2e-2.</figcaption>
</figure>
