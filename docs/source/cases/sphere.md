---
title: Sphere in a channel - 3-D immersed boundary
nav: Sphere (3-D IBM)
section: Test cases
order: 95
---

A sphere of diameter $D = 1$ in a $6 \times 3 \times 3$ channel with walls on four sides (8.7 % blockage), inflow at $x = 0$, Reynolds
number $\mathrm{Re} = 20$ and Mach 0.3, run to $t = 30$, when the flow is steady. It is the 3-D check of the immersed boundary: the
staircase **mask** against the sharp **ghost-cell wall** (`ibm_method = ghost`, trilinear image points over eight cells).

There is no exact 3-D reference for this confined flow. An unbounded sphere at Re = 20 has $C_d \approx 2.6$ (Schiller-Naumann); the
walls raise it, so the check is a sanity range ($2 \le C_d \le 4$) plus **self-convergence**: the ghost wall should change less
between two meshes than the mask, whose effective size depends on the resolution.

```powershell
python cases\validation\features\ibm_sphere.py --docs      # 10 and 20 cells per diameter, mask and ghost
```

The study writes its own case file (`results/validation/ibm_sphere/`); there is no sphere folder under `cases/`.

## Results

<figure>
<img src="{{root}}img/ibm_sphere.png" style="width:100%">
<figcaption>Left: drag history for the two methods at 10 and 20 cells per diameter. Middle: $C_d$ at $t = 30$ against the cell size.
Right: speed on the mid-plane $z = L_z/2$ with the ghost wall at 20 cells per diameter (the circle is the sphere); the wake is
steady and symmetric.</figcaption>
</figure>

| method | $C_d$, 10 cells / D | $C_d$, 20 cells / D | change |
|---|---:|---:|---:|
| mask | 3.293 | 3.385 | 0.093 |
| ghost wall | 3.460 | 3.514 | 0.054 |

* Every run is settled ($|\Delta C_d| / C_d < 10^{-4}$ over the last rows).
* The change of the ghost wall is 0.58 of the mask's. The band was fixed at 0.6 before the run, so the result is **marginal**: it
  supports the ghost wall being less dependent on the mesh, but two meshes are not a convergence proof, and both methods still move
  upward with refinement (the ghost wall by 1.6 %, the mask by 2.8 %).
* The two methods have not met at 20 cells per diameter (3.51 against 3.39); a third mesh would be needed to say where each is going.
* The serial and distributed runs of the sphere agree bit for bit (`ibm_ghost.py --check`, 2 to 7 ranks).

## Related

The sphere is also a regression reference of the sharp wall and of the 3-D adaptive mesh (see [AMR]({{root}}modules/AMR.html)).
