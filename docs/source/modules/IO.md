---
title: IO - VTK output and checkpoints
nav: IO
section: Modules
order: 80
header: IO.hpp
---

## VTK output

`VTKExporter` writes ParaView-readable VTK XML in ASCII. Every file is written to
`<file>.tmp` and then renamed, so ParaView never sees a partially written file, even if
the run is interrupted mid-write.

* **Uniform mesh.** One `RectilinearGrid` `<case>_<step>.vtr` per snapshot, with the
  cell fields `Density`, `Pressure`, `Temperature_nondim` ($p/\rho$), `Mach` and
  `Velocity` (3 components). A 2-D grid is one cell layer (z extent `0 0`, $w = 0$); a 3-D grid
  has the z coordinates $0 \ldots L_z$ and the real $w$.
* **AMR.** One `UnstructuredGrid` `<case>_<step>.vtu` with a quad per leaf and the same cell
  fields (`export_cells`), plus `AMR_Level`. Under MPI every rank writes the `.vtu` of its owned leaves and
  rank 0 a `.pvtu` that lists them.
* **Time series.** `write_pvd` writes `<case>.pvd` with one entry per snapshot (the
  `.vtr`, `.vtu` or `.pvtu`).

Values are sanitised for ParaView's float32 parser: NaN becomes 0, ±Inf becomes
±$10^{30}$, and $|x| < 10^{-30}$ becomes 0. The `cases/validation/` scripts parse these files
directly with `xml.etree`.

## Checkpoints (`.npz`)

Checkpoints are NumPy `.npz` archives (uncompressed zip of `.npy` arrays), written and
read without third-party libraries, so `numpy.load` reads them directly:

| Key | Content |
|---|---|
| `U` | state, `(nx*ny, 4)` float64 in 2-D, `(nx*ny*nz, 5)` in 3-D (columns $\rho, \rho u, \rho v, \rho E, \rho w$) |
| `t`, `step` | physical time and step |
| `nx`, `ny`, `Lx`, `Ly` | grid metadata (makes checkpoints self-describing for hot start) |
| `nz`, `Lz` | grid depth (`nz = 1` for 2-D; missing in older files, read as 1) |
| `collection_time`, `collection_file` | `.pvd` index, restored on resume |

The reader handles zip64 headers as written by `numpy.savez`. Archives that store the
`.pvd` index as a pickled `collection` object instead of the two arrays are also accepted
for resume and hot start; the pickled index is skipped. Compressed archives (`savez_compressed`) are rejected
with an explicit error.

```python
import numpy as np
d = np.load("results/sod_shock_tube/sod_shock_tube_restart.npz")
rho = d["U"][:, 0].reshape(int(d["ny"]), int(d["nx"]))
# 3-D: rho = d["U"][:, 0].reshape(int(d["nz"]), int(d["ny"]), int(d["nx"]))
```

## Monitor (`monitor_interval`)

With `monitor_interval = N > 0`, the run appends one line every $N$ steps (plus the first
and last step) to `<case>_monitor.csv`:

```text
step,t,mass,kinetic_energy,total_energy
```

The three columns are cell averages of $\rho$, $\tfrac12\rho|\mathbf u|^2$ and $\rho E$ over
the whole domain. Under MPI they are all-reduced, so the file matches the serial one. The
monitor is much cheaper than snapshots for integral histories. The 3-D Taylor-Green
validation computes its dissipation rate $-dE_k/dt$ from it.

With an immersed body (`use_ibm`) five columns follow: `Fx,Fy,Cd,Cl,Cm`, the loads of the
fluid on the body and their coefficients (see
[Physics]({{root}}modules/Physics.html#loads-on-the-body)). They are computed from one extra
residual evaluation of the current state at each monitor step, so they do not change the
solution.

Three more columns, `res_rho,res_mom,res_E`, always end the row: the residual norms. With
$R = \mathrm{d}U/\mathrm{d}t$ the right-hand side of the semi-discrete equations at the current
state, each is the root mean square of $R$ over the fluid cells (weighted by the cell area or
volume; solid cells of an immersed body are left out), for $ho$, for the momentum (the
magnitude of the vector) and for $ho E$. They tend to zero as a flow becomes steady; they are
raw values, to be normalised by the reader (the GUI divides by the first row). They come from the
same extra residual evaluation, now made at every monitor step; under MPI they are all-reduced.

{{api}}
