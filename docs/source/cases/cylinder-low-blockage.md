---
title: Flow over a cylinder in a large box - AMR, ghost-cell wall
nav: Cylinder, low blockage (AMR)
section: Test cases
order: 41
---

The [cylinder case]({{root}}cases/cylinder.html) runs in a $4\times2$ box, so the slip walls block 10 % of the channel,
which raises both the drag and the Strouhal number above the unbounded values. This case removes the confinement: the same
cylinder ($r = 0.1$, $D = 0.2$, $\mathrm{Re} = 100$, $\mathrm{Ma} \approx 0.085$) at $(3, 6)$ in a $16\times12$ box,
$80\times60$ diameters, with a blockage $D/L_y$ of 1.7 %. Covering that box with 40 cells per diameter takes
$3200\times2400 = 7.7$ million cells, so the case uses the [cell-based AMR]({{root}}modules/AMR.html):

* a $200\times150$ base mesh and three refinement levels (20 cells per diameter at the body; `amr_max_level = 4` gives 40),
* the body kept at the finest level (`amr_refine_ibm`) with the **ghost-cell wall** (`ibm_method = ghost`), which needs about
  20 cells per diameter at $\mathrm{Re} = 40$ where the mask needs four times that per direction,
* the wake followed by the speed indicator (`amr_indicator = "velocity"`: the density hardly varies at this Mach number),
* a velocity inlet, a **transmissive** right side and slip walls above and below.
  The `nonreflecting` outlet was tried first and was unstable at this pressure ($p = 100$, sound speed 11.8):
  the pressure grew from 100 to over 300 within eight time units,
* a small transverse kick behind the cylinder (`[[initial.region]]`) that starts the shedding at once; a symmetric start takes
  about 25 time units.

```powershell
build\release\fenice.exe --case "IBM Cylinder Low Blockage AMR"                    # 20 cells per diameter, t = 60
build\release\fenice.exe --case "IBM Cylinder Low Blockage AMR" --set amr_max_level=4   # 40 cells per diameter
python cases\validation\benchmarks\cylinder_low_blockage.py                          # both runs, loads, spectra, figure
```

## Default parameters

{{defaults:IBM Cylinder Low Blockage AMR}}

## Results

Over $25 \le t \le 60$ (28 shedding periods), 3 levels (20 cells per diameter) and 4 levels (40 cells per diameter):

| | 20 cells/D | 40 cells/D | literature, Re = 100 |
|---|---:|---:|---:|
| $\overline{C_d}$ | 1.421 | 1.367 | 1.33-1.35 |
| lift amplitude | 0.429 | 0.362 | about 0.33 |
| St (spectrum of $C_l$) | 0.1657 | 0.1659 | 0.164-0.166 |
| St (zero crossings of $C_l$) | 0.1657 | 0.1659 | |
| leaves at the end | 36 186 | 42 126 | |

<figure>
<img src="{{root}}img/cylinder_low_blockage.png" style="width:100%">
<figcaption>Re = 100 in the $16\times12$ box, 40 cells per diameter at the body: vorticity at $t = 60$ (the velocity of the
leaves is smoothed over one coarse wake cell before the derivative), the AMR level of the same window, the lift of both
resolutions and its spectrum. The final mesh has 42 126 leaves in place of 7.7 million cells;
the run takes about 6 minutes on 16 OpenMP threads at 20 cells per diameter and about 13 minutes on 8 MPI ranks at 40
(Ryzen 7 7800X3D).</figcaption>
</figure>

The Strouhal number is $0.166$ at both resolutions, inside the literature band, and the frequency is the same to the
fourth digit, whereas the 10 %-blockage run gives $0.175$: the excess of the confined case is a blockage effect,
not a resolution one. The drag and the lift amplitude are more sensitive to the resolution: the mean drag falls from 1.42 to
1.37 and the lift amplitude from 0.43 to 0.36 when the cells are halved, towards the literature values (1.33-1.35 and
about 0.33); the two runs do not allow an extrapolation (two grids give no observed order), so no further digit is claimed.

The wake is carried on level 1 of the mesh (a vortex is about 7 of its cells across) and the vortices lose strength faster than at the
body resolution; the loads depend on the near body, where the finest level is kept.

**Limits.** A hot start (`use_hot_start`) maps a checkpoint of a uniform mesh; it cannot take the checkpoint of an AMR run with
a different number of levels (`Hot-start state has 36186 cells, expected 30000`), so the 40-cells-per-diameter run starts
from the initial condition. Resuming an AMR run on its own mesh (`resume = true`) works.
