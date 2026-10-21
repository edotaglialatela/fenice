---
title: Physics - viscous fluxes, gradients, body forces, IBM
nav: Physics
section: Modules
order: 40
header: Physics.hpp
---

## Cell-centre gradients

`compute_gradients` evaluates central differences on the BC-aware padded state:

$$\left.\frac{\partial\phi}{\partial x}\right|_i = \frac{\phi_{i+1} - \phi_{i-1}}{2\Delta x}$$

(`compute_gradients_3d` adds $\partial/\partial z$ on 3-D meshes).

On a uniform orthogonal grid this is identical to both the Green-Gauss face average and
the least-squares reconstruction. Near walls the stencil uses the
[diffusive ghost cells]({{root}}modules/Boundaries.html), so wall velocities enter the
gradients correctly.

## Navier-Stokes viscous residual

For a Newtonian fluid under Stokes' hypothesis the viscous stress tensor is

$$\tau_{xx} = \mu\Bigl(\tfrac43\frac{\partial u}{\partial x} - \tfrac23\frac{\partial v}{\partial y}\Bigr), \quad
  \tau_{yy} = \mu\Bigl(\tfrac43\frac{\partial v}{\partial y} - \tfrac23\frac{\partial u}{\partial x}\Bigr), \quad
  \tau_{xy} = \mu\Bigl(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x}\Bigr)$$

In 3-D (`viscous_residual_3d`) the full tensor is used,

$$\tau_{ij} = \mu\Bigl(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}
  - \tfrac23\,\delta_{ij}\,\nabla\cdot\mathbf u\Bigr)$$

which reduces to the expressions above when $w = 0$ and nothing varies along z.

The Fourier heat flux uses the conductivity $k = \mu\gamma/[\Pr(\gamma-1)]$:

$$\mathbf q = -k\,\nabla T, \qquad T = p/\rho$$

The residual is the divergence of the viscous flux, assembled face by face:
$[0,\ \tau_{xx},\ \tau_{xy},\ u\tau_{xx} + v\tau_{xy} + w\tau_{xz} - q_x,\ \tau_{xz}]$ on
x-faces and the analogues on y- and z-faces, scaled by the face area and divided by the
cell volume.

**Stencils.**

* *Normal derivatives* use a compact 2-point difference across the face,
  $(\phi_{i+1} - \phi_i)/\Delta x$. This gives a proper 3-point Laplacian, which damps the
  velocity checkerboard mode.
* *Cross derivatives*, such as $\partial^2u/\partial x\partial y$, use the face average
  of the cell-centre gradients.
* *At walls*, the ghost velocities and temperatures come from the padded state. The
  wall shear is therefore correct for no-slip walls, and the adiabatic ghost
  temperature gives zero wall heat flux.
* *Across an MPI interface*, the neighbour's gradients are rebuilt from the halo
  (`CrossGhosts`), so the stencil matches the serial run.
* *At a periodic boundary*, the cross derivatives use the opposite edge cell. The
  stencil is then the same as in the interior, and a periodic domain is exactly
  translation-invariant (checked bit for bit in 2-D and 3-D).

**Time-step limit.** The von Neumann condition of the heat equation gives

$$\Delta t_{\rm visc} = \frac14\,\frac{\min(\Delta x, \Delta y)^2}{\max(\nu, \alpha_{\rm th})}\ \ \text{(2-D)}, \qquad
  \Delta t_{\rm visc} = \frac{1}{2\max(\nu, \alpha_{\rm th})\sum_d \Delta x_d^{-2}}\ \ \text{(3-D)}$$

with $\nu = \mu/\rho_{\min}$ and $\alpha_{\rm th} = k/(\rho_{\min} c_p)$. On square/cubic cells
these are $h^2/4\nu$ and $h^2/6\nu$, the forward-Euler limits of the 2-D and 3-D heat
equation. The limit enters the CFL estimate as an equivalent speed
$S_{\rm visc} = \mathrm{CFL}\,h/\Delta t_{\rm visc}$, with the same length $h$ as the
convective step (see [Overview]({{root}}index.html#how-a-time-step-works)).

## Body forces

`ConstantBodyForce` adds a uniform acceleration $\mathbf g = (g_x, g_y, g_z)$ (gravity or an
imposed pressure gradient; $g_z$ acts only on 3-D meshes):

$$\mathbf S = \bigl[0,\ \rho g_x,\ \rho g_y,\ \rho\,\mathbf u\cdot\mathbf g,\ \rho g_z\bigr]^\top$$

The energy component is the power delivered by the force, $\mathbf u\cdot\rho\mathbf g$.

## Immersed boundary method

A direct-forcing IBM. A cell is **solid** when its centre lies inside the body:

* **Circle / sphere.** In 2-D `Cylinder` and `Sphere` are the same circle. In 3-D, `Sphere`
  is a ball centred at (`ibm_cx`, `ibm_cy`, `ibm_cz`), and `Cylinder` is a circular
  cylinder running through the whole depth along z.
* **Airfoil.** A closed polygon (`Geometry.hpp`): a `NACA dddd` section from the 4-digit
  formulas (closed trailing edge, 200 cosine-spaced points per surface), or `Airfoil` read
  from `ibm_file` in the Selig or Lednicer format of the UIUC database. The section is
  normalised to unit chord, scaled to `ibm_chord`, rotated by `ibm_aoa` (nose up) about its
  quarter chord and placed with the quarter chord at (`ibm_cx`, `ibm_cy`); the point test is
  the even-odd rule. In 3-D it is extruded along z.

The mask is a staircase: the wall is known to within half a cell. A sharp trailing edge is
only a few cells thick, so airfoil lift converges slowly with the grid; resolve the chord
with at least ~60 cells.

* **Forcing** (`apply_forcing`) runs inside every residual evaluation, so it applies
  at every Runge-Kutta stage. It zeroes the velocity of solid cells and leaves
  density and pressure untouched, which keeps the multi-stage step thermodynamically
  consistent.
* **Solid reset** (`apply_solid_reset`) runs once per complete time step:
  * **Fringe cells** are solid cells with at least one fluid neighbour. They take
    $\rho$ and $p$ from that neighbour (a Neumann, zero-gradient condition). This
    suppresses the pressure well that would otherwise form at the surface, and avoids
    artificial mass sources at the boundary.
  * **Deep solid cells** have no fluid neighbour. They are frozen to
    $(\rho_{\rm ref}, p_{\rm ref}) = $ (`rho_in`, `p_out`) to stop long-term drift.
  * The velocity is zero throughout the obstacle.
  * The energy of reset cells uses the ideal-gas relation $\rho E = p/(\gamma-1)$ for any
    EOS (with a stiffened gas the $p_\infty$ term is not included).

The reset must never run inside RK stages. Doing so introduces hard density and
pressure discontinuities, which the Riemann solver amplifies into spurious shocks.
Under MPI the fringe test sees the neighbours across rank interfaces
(`SolidResetHalo`), and the mask is cut from the global mask, so the obstacle is
identical for any number of ranks. The 3-D fringe test also looks at the z-neighbours.

### Loads on the body

The force of the fluid on the body is the net momentum flux into the solid cells: summing
their residuals, the fluxes through faces between two solid cells cancel and only the faces
on the body surface remain,

$$\mathbf F = \sum_{c\ \in\ \rm solid} \mathbf R_c\,V_c, \qquad
M_z = \sum_{c\ \in\ \rm solid} (\mathbf x_c - \mathbf x_{\rm ref}) \times \mathbf R_c\,V_c,$$

with $\mathbf R$ the momentum residual of the forced state (body force removed) and
$\mathbf x_{\rm ref}$ = (`ibm_cx`, `ibm_cy`). It contains the pressure and the viscous
stresses alike, for any shape. With `monitor_interval` the monitor gets $F_x$, $F_y$ and
$C_d = F_x/(q A)$, $C_l = F_y/(q A)$, $C_m = -M_z/(q A L)$ (nose-up positive), with
$q = \tfrac12\rho_{\rm in}u_{\rm in}^2$, $L$ the chord or diameter and $A = L$ per unit depth
in 2-D ($L L_z$ in 3-D, $\pi r^2$ for a sphere). On the cylinder at Re = 100 (400 × 200) this
gives $\overline{C_d} = 1.43$, a lift amplitude of 0.37 and St = 0.177, against
$C_d \approx 1.35$ and $C_l' \approx 0.33$ for an unconfined cylinder; the 10 % blockage of the
channel raises both.

{{widget:IBM}}

{{api}}
