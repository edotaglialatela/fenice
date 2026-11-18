---
title: Sphere in a channel - 3-D immersed boundary
nav: Sphere (3-D IBM)
section: Test cases
order: 95
---

> **Run it:** `build\release\fenice.exe --case "IBM Sphere"` (output in `results/`; [how to open it]({{root}}guide/getting-started.html#your-first-run)).

A sphere of diameter $D = 1$ in a $6 \times 3 \times 3$ channel with walls on four sides (8.7 % blockage), inflow at $x = 0$, Reynolds
number $\mathrm{Re} = 20$ and Mach 0.3, run to $t = 30$, when the flow is steady. It is the 3-D check of the immersed boundary: the
**ghost-cell wall** with trilinear image points over eight cells.

There is no exact 3-D reference for this confined flow. An unbounded sphere at Re = 20 has $C_d \approx 2.6$ (Schiller-Naumann); the
walls raise it, so the check is a sanity range ($2 \le C_d \le 4$) plus **self-convergence**: how little the drag changes
between two meshes.

```powershell
python cases\validation\features\ibm_sphere.py --docs      # 10 and 20 cells per diameter
```

The study writes its own case file (`results/validation/ibm_sphere/`). The showcase case `cases/ibm_sphere` (`fenice --case "IBM Sphere"`, also in the GUI) is the Re = 100 flow on the octree, 20 cells per diameter, snapshots every 0.5.

## Results

<figure>
<img src="{{root}}img/ibm_sphere.png" style="width:100%">
<figcaption>Left: drag history at 10 and 20 cells per diameter. Middle: $C_d$ at $t = 30$ against the cell size.
Right: speed on the mid-plane $z = L_z/2$ at 20 cells per diameter (the circle is the sphere); the wake is
steady and symmetric.</figcaption>
</figure>

| $C_d$, 10 cells / D | $C_d$, 20 cells / D | change |
|---:|---:|---:|
| 3.460 | 3.514 | 0.054 (1.5 %) |

* Both runs are settled ($|\Delta C_d| / C_d < 10^{-4}$ over the last rows).
* The drag still moves upward with refinement, by 1.5 %. Two meshes are not a convergence proof: a third mesh would be needed to
  say where it is going. The study's band (a change of at most 3 %) is a regression guard set after the run.
* The serial and distributed runs of the sphere agree bit for bit (`ibm_ghost.py --check`, 2 to 7 ranks).

On the adaptive octree (`cases/validation/amr/octree_q8.py`) the same sphere at an equal finest resolution of 20 cells per diameter gives $C_d$ within about 0.1 % of the uniform solver (Re 20 and 100), in 0.3-0.5 of the time (whole runs on 16 threads) and with 76 k leaves against 432 k cells; see [AMR]({{root}}modules/AMR.html).

## Related

The sphere is also a regression reference of the sharp wall and of the 3-D adaptive mesh (see [AMR]({{root}}modules/AMR.html)).
