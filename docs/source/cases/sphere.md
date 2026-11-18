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
python cases\validation\amr\octree_q8.py long              # 80 cells per diameter on the octree, to t = 60 (hours)
```

The study writes its own case file (`results/validation/ibm_sphere/`). The showcase case `cases/ibm_sphere` (`fenice --case "IBM Sphere"`, also in the GUI) is the Re = 100 flow on the octree, 20 cells per diameter, snapshots every 0.5.

## Results

<figure>
<img src="{{root}}img/ibm_sphere.png" style="width:100%">
<figcaption>Uniform meshes. Left: drag history at 10 and 20 cells per diameter. Right: $C_d$ at $t = 30$ against the cell size.
The flow field at 80 cells per diameter is in the figure of the next section.</figcaption>
</figure>

| $C_d$, 10 cells / D | $C_d$, 20 cells / D | change |
|---:|---:|---:|
| 3.460 | 3.514 | 0.054 (1.5 %) |

* Both runs are settled ($|\Delta C_d| / C_d < 10^{-4}$ over the last rows).
* The drag still moves upward with refinement, by 1.5 %. Two meshes are not a convergence proof: a third mesh would be needed to
  say where it is going. The study's band (a change of at most 3 %) is a regression guard set after the run.
* The serial and distributed runs of the sphere agree bit for bit (`ibm_ghost.py --check`, 2 to 7 ranks).

### Finer meshes on the octree

The octree reaches finest cell sizes that the uniform solver cannot afford here (80 cells per diameter would be 28 million cells). Workstation, Re 20 (the 80-cell run on 48 threads):

```powershell
python cases\validation\amr\octree_q8.py run --include-40                       # 20 and 40 cells per diameter, to t = 30 (1-2 h each at 40)
python cases\validation\amr\octree_q8.py long                                   # 80 cells per diameter (base 10, 3 levels), Re 20, to t = 60 (hours, 1.9 GB)
python cases\validation\amr\octree_q8.py long --re 100 --t-end 30 --levels 2    # another Reynolds number, end time or finest level (finest = 10 x 2^levels)
python cases\validation\amr\octree_sphere.py --docs                              # the figure below, from those runs
```

The runs are cached by their tag in `results/validation/amr_octree_q8/` (a rerun resumes or reuses them) and use 16 threads; `FENICE_OCTREE_THREADS` sets the count (48 for the table below).

| finest cells / D | base, levels | leaves at the end | run to | $C_d$ at $t = 30$ | time loop |
|---:|---|---:|---:|---:|---:|
| 20 | 5, 2 | 76 120 | 30 | 3.5124 | 870 s |
| 40 | 10, 2 | 283 544 | 30 | 3.5456 | 10 980 s |
| 80 | 10, 3 | 913 236 | 60 | 3.5598 | 28 761 s (66 085 steps) |

The 80-cell run was continued to $t = 60$, where $C_d = 3.5517$; its peak memory is 1.9 GB.

<figure>
<img src="{{root}}img/ibm_sphere_octree.png" style="width:100%">
<figcaption>The octree runs (Re 20, <code>cases/validation/amr/octree_sphere.py</code>). Top: drag histories at 20, 40 and 80 cells per diameter (the 80-cell run to $t = 60$), and $C_d$ at $t = 30$ against the finest cell size.
Bottom: speed on the mid-plane $z = L_z/2$ of the 80-cell run at $t = 60$ (the circle is the sphere), and a zoom around the body with the boundaries between the refinement levels of the leaves (913 236 leaves; the body and its band stay at the finest level).
The wake is steady and symmetric; the blocks far from the sphere are the coarse leaves.</figcaption>
</figure>

* Over the three meshes at $t = 30$ the observed order is 1.23, far from the formal 2, so $F_s = 3$: fine-grid GCI 0.9 %, extrapolated $C_d$ 3.570.
* The finest run is **not settled at $t = 30$**: its drag falls by 0.0027 (0.08 %) every 10 time units, 3.5598 at $t = 30$, 3.5571 at 40, 3.5544 at 50 and 3.5517 at 60, with no sign of slowing,
  so the 0.23 % between $t = 30$ and $t = 60$ is as large as the GCI above. The two coarser runs stop at $t = 30$ and may drift in the same way, so the
  extrapolated value is indicative; a longer run would be needed to say where the drag at 80 cells per diameter settles.

On the adaptive octree (`cases/validation/amr/octree_q8.py`) the same sphere at an equal finest resolution of 20 cells per diameter gives $C_d$ within about 0.1 % of the uniform solver (Re 20 and 100), in 0.3-0.5 of the time (whole runs on 16 threads) and with 76 k leaves against 432 k cells; see [AMR]({{root}}modules/AMR.html).

## Related

The sphere is also a regression reference of the sharp wall and of the 3-D adaptive mesh (see [AMR]({{root}}modules/AMR.html)).
