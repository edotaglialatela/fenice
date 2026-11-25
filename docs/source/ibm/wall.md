---
title: The immersed-boundary wall
nav: The wall
section: Immersed boundary
order: 10
---

> **In short.** A body is placed on the Cartesian mesh without meshing it (`use_ibm`). The cells under the body are removed from the flow, and the
> cells next to the wall (**ghost cells**) are given a state that makes the wall condition hold on the true surface, not on the staircase. The force on the
> body is the momentum flux into its cells. This page states the problem and the discrete wall; [Moving bodies](moving-bodies.html) adds the motion and the coupling.

## The problem

Let $\Omega$ be the computational box, $\Omega_b \subset \Omega$ the body, $\Gamma = \partial\Omega_b$ its surface with the unit normal $\mathbf n$ pointing
**into the fluid**, and $\Omega_f = \Omega\setminus\overline{\Omega_b}$. In $\Omega_f$ the solver integrates the compressible Navier-Stokes equations (the Euler equations
for `use_viscous = false`)

$$\partial_t \mathbf U + \nabla\!\cdot\!\mathbf F(\mathbf U) = \nabla\!\cdot\!\mathbf F_v(\mathbf U,\nabla\mathbf U) + \mathbf S, \qquad
\mathbf U = \bigl[\rho,\ \rho\mathbf u,\ \rho E\bigr]^\top,$$

with the fluxes of the [Physics]({{root}}modules/Physics.html) page and $\mathbf S$ the body force and source terms. On $\Gamma$ the wall conditions are

$$\begin{aligned}
\text{no-slip } (\texttt{use\_viscous}):\quad & \mathbf u = \mathbf u_w, \\
\text{slip (inviscid)}:\quad & (\mathbf u-\mathbf u_w)\cdot\mathbf n = 0, \\
\text{both}:\quad & \partial_n\rho = \partial_n p = 0,
\end{aligned}$$

where $\mathbf u_w(\mathbf x,t)$ is the velocity of the wall: zero for a fixed body, the rigid-body velocity of a moving one. The loads the fluid exerts on the body are

$$\mathbf F_f = -\oint_\Gamma \bigl(p\,\mathbf n - \boldsymbol\tau\,\mathbf n\bigr)\,\mathrm dS, \qquad
M_z = -\oint_\Gamma (\mathbf x-\mathbf x_{\rm ref})\times\bigl(p\,\mathbf n - \boldsymbol\tau\,\mathbf n\bigr)\,\mathrm dS .$$

## Bodies

A cell is **solid** when its centre lies in $\Omega_b$:

* **Circle / sphere.** In 2-D `Cylinder` and `Sphere` are the same circle. In 3-D, `Sphere` is a ball centred at (`ibm_cx`, `ibm_cy`, `ibm_cz`), and `Cylinder` is a
  circular cylinder running through the whole depth along z.
* **Airfoil.** A closed polygon (`Geometry.hpp`): a `NACA dddd` section from the 4-digit formulas (closed trailing edge, 200 cosine-spaced points per surface), or
  `Airfoil` read from `ibm_file` in the Selig or Lednicer format of the UIUC database. The section is normalised to unit chord, scaled to `ibm_chord`, rotated by
  `ibm_aoa` (nose up) about its quarter chord and placed with the quarter chord at (`ibm_cx`, `ibm_cy`); the point test is the even-odd rule. In 3-D it is extruded along z.

A sharp trailing edge is only a few cells thick, so airfoil lift converges slowly with the grid; resolve the chord with at least ~60 cells.

## The ghost-cell wall

The wall exists for the cylinder and the airfoil (2-D, uniform or AMR, or extruded along z in 3-D) and for the 3-D sphere, serial, OpenMP or MPI.

**Ghost cells.** Let $h$ be the cell size. The solid cells that the fluid stencils read, those within two cells of a fluid cell, form the set $\mathcal G$. For $g\in\mathcal G$ at
$\mathbf x_g$ the solver finds

$$\mathbf x_w = \operatorname*{arg\,min}_{\mathbf x\in\Gamma}|\mathbf x-\mathbf x_g|,\qquad d_g = |\mathbf x_g-\mathbf x_w|,\qquad
\mathbf x_i = \mathbf x_w + d_i\,\mathbf n,\qquad d_i \ge \max(1.5\,h,\ d_g+h),$$

the closest point of the true surface, the distance to it, and the **image point**: the mirror of the ghost cell across the wall, pushed out to at least 1.5 cells into the fluid (and
farther in half-cell steps, up to 4 cells, if the cells around it are not all fluid). The state at $\mathbf x_i$ is interpolated from the fluid cells around it,

$$\mathbf W_i = \sum_k w_k\,\mathbf W_k,$$

bilinearly in 2-D, trilinearly over the eight cells around the image point in 3-D with the solid ones left out and the weights renormalised (a body extruded along z: in the layer of the ghost cell, so every
layer is the 2-D run). With $\varsigma = d_g/d_i$ and the relative velocity of the image $\mathbf r_i = \mathbf u_i-\mathbf u_w$, the ghost state is

$$\rho_g=\rho_i,\qquad p_g=p_i,\qquad
\mathbf u_g=\begin{cases}\mathbf u_w-\varsigma\,\mathbf r_i & \text{no-slip}\\[2pt]
\mathbf u_w+\mathbf r_i-(1+\varsigma)\,(\mathbf r_i\!\cdot\!\mathbf n)\,\mathbf n & \text{slip}\end{cases}$$

For a fixed body $\mathbf u_w=0$ and no-slip gives $\mathbf u_g=-\varsigma\,\mathbf u_i$. The construction puts the wall conditions on the real surface: the line through $\mathbf x_g$ (at
$-d_g$ along the normal) and $\mathbf x_i$ (at $+d_i$) takes at the wall the value

$$\frac{d_i\,\mathbf u_g+d_g\,\mathbf u_i}{d_i+d_g}=\mathbf u_w \quad(\text{no-slip}),\qquad
\text{and, for the normal part of } \mathbf u-\mathbf u_w,\ \ \frac{d_i(-\varsigma r_n)+d_g\,r_n}{d_i+d_g}=0 \quad(\text{slip}),$$

and $\rho,p$ have zero slope across $\Gamma$ by $\rho_g=\rho_i$, $p_g=p_i$.

**Gravity (`use_bodyforce`).** A gas at rest under the body force $\mathbf g$ has the pressure gradient $\nabla p=\rho\,\mathbf g$, not zero. The ghost pressure then follows it from the
image point to the ghost cell, $p_g=p_i-\rho_i\,(d_g+d_i)\,\mathbf g\!\cdot\!\mathbf n$ ($\mathbf n$ points into the fluid), so that the wall pressure of a hydrostatic gas is the hydrostatic one
and the load on a held body is the weight of the gas it displaces from the first step. With the plain copy a held cylinder saw a vertical load 12 % above that weight (7 % at twice the resolution) for the
first few sound crossing times, a kick that every released body took. Without gravity the ghost state is the plain copy above.

<img src="{{root}}img/ibm_ghost_geometry.png" style="width:100%">

*Left: a ghost cell $\mathbf x_g$, its closest wall point $\mathbf x_w$ and its image point $\mathbf x_i$. The image state is interpolated from the four fluid cells around $\mathbf x_i$ (green). Ghost cells are the solid cells within two cells of the fluid; deeper solid cells are reset. Right: along the normal, the ghost velocity is chosen so that the line from $\mathbf u_g$ to $\mathbf u_i$ passes through the wall velocity on the true surface.*

 In 3-D the velocity has three components: slip removes only the normal part (a spanwise velocity along an
extruded body survives), no-slip scales all three.

**Per step.**

* The deeper solid cells (not in $\mathcal G$) stay at rest and are reset to the reference state (`rho_in`, `p_out`, energy from the EOS) each step.
* The ghost cells are filled inside every residual evaluation (`apply_forcing`), so at every Runge-Kutta stage, and once more after the complete step together with the
  reset of the deeper cells (`apply_solid_reset`).
* A ghost cell for which no image point with fluid cells around it exists (a slot one or two cells wide) copies the nearest fluid cell and gets a zero velocity; the solver prints how many there are.

**Properties.**

* The wall position is exact, so no staircase waves are shed and the solution converges fast with the grid: on a cylinder at Re = 40 the drag is within 0.15 % of its converged value at 20 cells per diameter.
* The wall is not conservative: in a closed inviscid box the fluid mass grows by about 0.2 % by $t=4$.
* Under MPI each rank fills its own ghost cells and fetches the few cells of other strips their image points read, so the result is identical for any number of ranks (also in 3-D). With AMR the
  body is kept at the finest level in a band as wide as the ghost stencils (2-D; in 3-D the band is six finest cells wide, and the wall is active once no leaf coarser than the finest level lies
  within that reach of the surface, a fact the ranks settle together; a run stops if that cannot be reached).
* `reconstruction = weno5` on a uniform mesh uses the four-cell WENO3-Z stencil on every face that has a solid cell among its six, as the adaptive mesh does.
* In 3-D there are no surface samples (`surface_loads` is zero): the monitor loads are the momentum flux into the solid cells, which is dimension independent.

<details markdown="1">
<summary>Curvature correction (`ibm_curvature`, 2-D slip walls)</summary>

**Curvature correction (`ibm_curvature`, 2-D slip walls).** The zero normal gradient of the pressure is a first-order error on a curved wall, where
the normal momentum equation gives $\partial p/\partial n = \rho\,\kappa\,u_t^2$ ($\kappa$: curvature of the wall, positive for a convex body; $1/R$ for a
cylinder, estimated from the turning of the normal over three cells for a polygon). With `ibm_curvature = true` the ghost pressure is
$p_{\rm image} - \rho\,\kappa\,u_t^2\,(d_{\rm image} + d_{\rm ghost})$, limited to a factor 0.5 to 2, and the density follows constant entropy along the normal. The
balance holds where the flow follows the wall, so the correction is weighted by $\max(0, 1 - |u_n|/(0.2\,|u_t|))$ at the image point: unweighted, the
bow shock of an impulsively started Mach 2 cylinder stays pinned to the wall. It is opt-in, rejected in 3-D and for a moving body, and leaves flat walls unchanged.

</details>

<details markdown="1">
<summary>Evidence for the ghost-cell wall</summary>

**Evidence.** Mach 2 flow along a 10 degree wedge (`verification/ibm_wedge.py`, exact oblique shock, faces inclined to the grid): the pressure error of the
ghost wall is 2.3e-2, 7.7e-4 and 6.3e-5 at 50, 100 and 200 cells per unit length (order 3.6). The supersonic vortex around a cylinder (`verification/ibm_vortex.py`, exact solution): the ghost wall's pressure error
falls only as $h^1$ (2.3e-2 at 160 cells per radius), with the correction it is 8.7e-4 and falls as $h^{1.5}$. In 3-D the evidence is the exactness of the
trilinear image state on a linear field (unit test), the equality of extruded bodies with the 2-D run, bitwise MPI equality, and the self-convergence of
the sphere's drag (`features/ibm_sphere.py`: Cd 3.46 and 3.51 at 10 and 20 cells per diameter, a change of 1.5 %; weak evidence, with no exact reference).

</details>

## Loads on the body

<img src="{{root}}img/ibm_loads.png" style="width:85%; display:block; margin:0 auto;">

*Summing the momentum residuals of the solid cells. Each face between two solid cells appears twice with opposite signs (grey). Only the faces between fluid and solid (red) are left, and they carry the force the fluid exerts on the body.*

Write the semi-discrete update of a cell as $V_c\,\dot{\mathbf U}_c = -\sum_{f\in\partial c}\mathbf F_f\!\cdot\!\mathbf n_f A_f + V_c\mathbf S_c$, with residual
$\mathbf R_c=\dot{\mathbf U}_c$. Sum the momentum rows over all solid cells: the fluxes through faces between two solid cells cancel in pairs and only the faces
on the body surface remain, which carry the momentum that the fluid hands to the body, pressure and viscous stress alike and for any shape,

$$\mathbf F_f \simeq \sum_{c\ \in\ \rm solid} V_c\,\mathbf R_c^{\rm mom}, \qquad
M_z \simeq \sum_{c\ \in\ \rm solid} V_c\,(\mathbf x_c - \mathbf x_{\rm ref}) \times \mathbf R_c^{\rm mom},$$

with $\mathbf R^{\rm mom}$ the momentum residual of the forced state (body force removed, so that $\mathbf F_f$ is the surface force alone) and
$\mathbf x_{\rm ref}$ = (`ibm_cx`, `ibm_cy`), or the body's own reference point if it moves. With `monitor_interval` the monitor gets $F_x$, $F_y$ and
$C_d = F_x/(q A)$, $C_l = F_y/(q A)$, $C_m = -M_z/(q A L)$ (nose-up positive), with
$q = \tfrac12\rho_{\rm in}u_{\rm in}^2$, $L$ the chord or diameter and $A = L$ per unit depth
in 2-D ($L L_z$ in 3-D, $\pi r^2$ for a sphere). On the cylinder at Re = 100 (400 × 200) this
gives $\overline{C_d} = 1.54$, a lift amplitude of 0.46 at its peak and St = 0.178 over
$50 \le t \le 80$, against
$C_d \approx 1.35$ and $C_l' \approx 0.33$ for an unconfined cylinder; the 10 % blockage of the
channel raises both. `ImmersedBoundary::surface_loads` integrates the true surface from probes at $d$ and $2d$ along the normal (for surface distributions; that integral is 1-3 % low).

{{widget:IBM}}

---

*Declarations: [`fv/Physics.hpp`]({{root}}api/Physics.html).*
