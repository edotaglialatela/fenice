---
title: Boundaries - ghost-cell boundary conditions
nav: Boundaries
section: Modules
order: 20
header: Boundaries.hpp
---

Every boundary condition is applied through a **ghost cell**. For each
boundary face, a fictitious exterior state $\mathbf{U}_g$ is chosen so that the
Riemann problem between the interior state $\mathbf{U}_i$ and $\mathbf{U}_g$
produces the desired physical flux. `BoundaryManager` keeps two consistent
representations:

* **Convective ghosts.** `apply_boundary_conditions` fills the face arrays
  $\mathbf{U}_L$, $\mathbf{U}_R$ fed to the Riemann solver. On a boundary face one
  side is the interior cell and the other side is the ghost.
* **Diffusive ghosts.** `get_padded_U` returns the state padded to
  $(n_y+2)\times(n_x+2)$, or $(n_z+2)\times(n_y+2)\times(n_x+2)$ in 3-D. The gradient
  calculator and the viscous flux use it, so they see the same wall values as the
  convective solver.

A 2-D mesh has four sides (left, right, bottom, top). A 3-D mesh adds **front**
($z = 0$, `bc_front`) and **back** ($z = L_z$, `bc_back`), which default to periodic.

## Boundary-condition types

For a face with outward normal $\hat n$:

| Type | Ghost state |
|---|---|
| `wall`: inviscid slip wall | normal velocity reflected, tangential kept: $\mathbf{u}_g = \mathbf{u}_i - 2(\mathbf{u}_i\cdot\hat n)\,\hat n$ |
| `no_slip`: viscous wall | $\mathbf{u}_g = -\mathbf{u}_i$, so the face-average velocity is zero |
| `moving_wall`: moving viscous wall | $\mathbf{u}_g = 2\mathbf{u}_w - \mathbf{u}_i$ with $\mathbf{u}_w = (u_{\rm in}, v_{\rm in}, 0)$ |
| `vel_inlet`: velocity inlet | $\mathbf{u}_g = 2\mathbf{u}_{\rm in} - \mathbf{u}_i$ with $\mathbf{u}_{\rm in} = (u_{\rm in}, v_{\rm in}, 0)$; $\rho$ and $p$ extrapolated from the interior |
| `massflow_inlet`: mass-flow inlet | target velocity $u_t = \rho_{\rm in}u_{\rm in}/\langle\rho\rangle$, with $\langle\rho\rangle$ the inlet-averaged interior density; $u_g = 2u_t - u_i$ |
| `outlet`: pressure outlet | $p_g = p_{\rm out}$; outflow velocity kept, backflow ($V_n < 0$) clamped to zero |
| `transmissive`: zero gradient | $\mathbf{U}_g = \mathbf{U}_i$ (far field / outflow) |
| `periodic` | ghost taken from the cell on the opposite boundary; MUSCL and the viscous cross-derivatives also wrap around, so a periodic seam is identical to an interior face |

All velocity-based ghosts are built in primitive variables and converted back
with the EOS. The padded array applies the matching momentum rules: no-slip
negates both momenta, a slip wall negates only the normal momentum, and a moving
wall sets the tangential momentum to $2\rho u_w - \rho u$. The energy is left
unchanged, which makes the walls adiabatic. In 3-D the same rules act on all three
momenta: the wall-normal one is negated and the tangential ones follow the BC.

The prescribed inlet and wall velocities have no z-component, so on the front and back
sides `vel_inlet`, `massflow_inlet` and `moving_wall` act as walls with tangential motion.
Feed 3-D inflows through the x or y sides.

Under MPI the mass-flow average $\langle\rho\rangle$ is reduced over every rank
sharing the inlet, so the inlet behaves exactly as in a serial run.

{{api}}
