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

## The mesh

The $200\times150$ base mesh (level 0) has cells of size $h_0 = 16/200 = 0.08$, 2.5 per diameter. Every level halves the cell
size, $h_\ell = 0.08/2^\ell$:

| level | cell size | cells per diameter | where it is used |
|---|---:|---:|---|
| 0 | 0.08 | 2.5 | far field, inlet side, above and below the wake (29 506 of 42 126 leaves at the end of the 4-level run) |
| 1 | 0.04 | 5 | the vortex street downstream of $x \approx 5$ |
| 2 | 0.02 | 10 | transition between the body and the wake |
| 3 | 0.01 | 20 | the body of the 3-level run (`amr_max_level = 3`); a transition level in the 4-level run |
| 4 | 0.005 | 40 | the body of the 4-level run (8 412 leaves, around the body and the first part of the wake) |

* The cells touching the body, and a band of 6 finest-level cells around it (what the ghost-cell wall reads), are on the
  finest level from $t = 0$ and are never joined (`amr_refine_ibm`).
* Elsewhere a leaf is refined when the relative variation of the **speed** across its faces reaches
  `amr_refine_threshold` (0.15); the indicator is smoothed `amr_diffusion = 2` times, which widens the band, and four siblings are joined
  when the indicator of their parent is below `amr_derefine_threshold` (0.1). The mesh is adapted every
  `amr_refine_every = 5` coarse steps, so the fine regions follow the vortices.
* Each level advances with its own time step $\Delta t_0/2^\ell$ (subcycling), finer levels first, and the fluxes through the faces
  between levels are replaced through flux registers, so the scheme stays conservative.
* In the picture below the AMR-level panel near the body shows this: the finest level around the body, a step down through the near wake,
  and level 1 for the rest of the street (the street farther downstream is not in the zoom; level 1 holds the vortices there, as the table says).

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
leaves is smoothed over one coarse wake cell before the derivative; the body is drawn), a zoom on the body at the finest
cell size (smoothing 2.5 cells, so the boundary layer is not widened; vorticity saturated at $\pm 20$) with the AMR level
there, the lift of both resolutions and its spectrum around the peak (the literature band is drawn at its true width). The final mesh has 42 126 leaves in place of 7.7 million cells;
the run takes about 6 minutes on 16 OpenMP threads at 20 cells per diameter and about 13 minutes on 8 MPI ranks at 40
(Ryzen 7 7800X3D).</figcaption>
</figure>

The Strouhal number is $0.166$ at both resolutions, inside the literature band, and the frequency is the same to the
fourth digit, whereas the 10 %-blockage run gives $0.175$: the excess of the confined case is a blockage effect,
not a resolution one. The drag and the lift amplitude are more sensitive to the resolution: the mean drag falls from 1.42 to
1.37 and the lift amplitude from 0.43 to 0.36 when the cells are halved, towards the literature values (1.33-1.35 and
about 0.33); the two runs do not allow an extrapolation (two grids give no observed order), so no further digit is claimed.

## Spectra of the loads

The record $25 \le t \le 60$ (35 time units, 29 shedding periods; frequency resolution $D/T = 0.0057$ in St) is analysed with a
Hann window and zero padding.

<figure>
<img src="{{root}}img/cylinder_low_blockage_spectra.png" style="width:100%">
<figcaption>Top: lift spectrum and a detail of its peak on a linear scale, where the literature band $0.164$-$0.166$ is visible.
Bottom: drag spectrum and the Strouhal number from a sliding window of 12 time units, at 20 and 40 cells per diameter.</figcaption>
</figure>

The lift peaks at $\mathrm{St} = 0.1657$ (20 cells per diameter, amplitude 0.39) and $0.1659$ (40, amplitude 0.33) with a third
harmonic at $0.495$ and $0.497$. The drag peaks at $2.000$ and $2.001$ times the shedding frequency, and has no component at
the shedding frequency itself, as the symmetry of the street requires. A window of 12 time units gives $0.1656$-$0.1658$ and
$0.1658$-$0.1662$ along the record: the signal is stationary, unlike that of the confined case, where the lift amplitude was
still decaying.

The wake is carried on level 1 of the mesh (a vortex is about 7 of its cells across) and the vortices lose strength faster than at the
body resolution; the loads depend on the near body, where the finest level is kept.

**Limits.** A hot start (`use_hot_start`) maps a checkpoint of a uniform mesh; it cannot take the checkpoint of an AMR run with
a different number of levels (`Hot-start state has 36186 cells, expected 30000`), so the 40-cells-per-diameter run starts
from the initial condition. Resuming an AMR run on its own mesh (`resume = true`) works.
