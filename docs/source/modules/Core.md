---
title: Core - mesh and equations of state
nav: Core (mesh, EOS)
section: Modules
order: 10
header: Core.hpp
---

## Cartesian mesh

`Mesh` describes a uniform $n_x \times n_y \times n_z$ grid of
$[x_0, x_0+L_x]\times[y_0, y_0+L_y]\times[z_0, z_0+L_z]$ with $\Delta x = L_x/n_x$,
$\Delta y = L_y/n_y$ and $\Delta z = L_z/n_z$. With $n_z = 1$ (`is3d()` false) the mesh is
2-D: there are no z-faces, the cell "volume" is the area $\Delta x\,\Delta y$ and the
face "area" is the edge length. The topology is implicit, so no per-face arrays are stored
and the same code serves the full domain, the MPI strips and the level-0 mesh of the AMR run. `generate_cartesian_mesh_3d` builds either kind
(`generate_cartesian_mesh` is its 2-D shorthand).

**Cell indexing** is row-major, `cell(i, j, k) = (k*ny + j)*nx + i`, with column $i$ (x),
row $j$ (y) and layer $k$ (z). A 2-D index is the $k = 0$ layer, `j*nx + i`. Cell centres
are $x_i = (i+\tfrac12)\Delta x + x_0$ and likewise for $y_j$, $z_k$.

**Face ordering.** Three blocks follow each other:

| Block | Count | Normal | Index |
|---|---|---|---|
| x-faces | $(n_x+1)\,n_y\,n_z$ | $(1,0,0)$ | `xface(i,j,k) = (k*ny + j)*(nx+1) + i` |
| y-faces | $n_x(n_y+1)\,n_z$ | $(0,1,0)$ | `yface(i,j,k) = num_xfaces + (k*(ny+1) + j)*nx + i` |
| z-faces (3-D) | $n_x n_y (n_z+1)$ | $(0,0,1)$ | `zface(i,j,k) = num_xfaces + num_yfaces + (k*ny + j)*nx + i` |

Face $i$ of a row separates cells $i-1$ and $i$, and likewise along y and z. The first and
last face of each line lie on the physical boundary, where a ghost state is supplied by the
[boundary conditions]({{root}}modules/Boundaries.html). The sides are `Face::Left/Right`
(x), `Bottom/Top` (y) and, in 3-D, `Front` ($z = z_0$) / `Back` ($z = z_0 + L_z$).
`Mesh::boundary_face` and `Mesh::boundary_cell` give the $m$-th boundary face of a side and
the cell next to it, with $m$ running over the side's cells in index order.

## Equations of state

Both EOS use the conservative vector
$\mathbf{U} = [\rho,\ \rho u,\ \rho v,\ \rho E,\ \rho w]^\top$ and the primitive vector
$\mathbf{W} = [\rho,\ u,\ v,\ p,\ w]^\top$, with total specific energy
$E = e_{\rm int} + \tfrac12|\mathbf{u}|^2$, $|\mathbf{u}|^2 = u^2+v^2+w^2$. The
z-components sit last so that the first four entries keep the 2-D layout; `kMom = {1, 2, 4}`
lists the momentum slots per axis.

**Ideal gas** (calorically perfect):

$$p = (\gamma - 1)\,\rho\, e_{\rm int}, \qquad c = \sqrt{\frac{\gamma p}{\rho}}, \qquad
  \rho E = \frac{p}{\gamma-1} + \tfrac12\rho|\mathbf{u}|^2$$

The default $\gamma = 1.4$ is air.

**Stiffened gas** (liquids and high-pressure fluids; Menikoff & Plohr 1989). A constant
pressure offset $p_\infty$ models intermolecular repulsion:

$$p + p_\infty = (\gamma - 1)\,\rho\, e_{\rm int}, \qquad c = \sqrt{\frac{\gamma (p + p_\infty)}{\rho}}, \qquad
  \rho E = \frac{p + \gamma p_\infty}{\gamma-1} + \tfrac12\rho|\mathbf{u}|^2$$

$p_\infty = 0$ recovers the ideal gas exactly. The defaults $\gamma = 4.4$ and
$p_\infty = 6\times10^8$ Pa model liquid water.

**Floors.** `cons_to_prim` floors density and pressure at
`kEosEps` $= 10^{-10}$, which prevents division by zero and negative sound speeds in
cells that become unphysical during transients.

### References

* Menikoff, R. & Plohr, B.J. (1989). The Riemann problem for fluid flow of real
  materials. *Rev. Mod. Phys.* 61(1), 75-130.

{{api}}
