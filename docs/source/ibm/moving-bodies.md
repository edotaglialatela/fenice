---
title: Moving bodies and fluid-structure coupling
nav: Moving bodies and FSI
section: Immersed boundary
order: 20
---

> **In short.** The body can follow a prescribed path, or be a rigid body driven by the flow: mass $m$, spring $K$, damper $C$ and weight, advanced by the force the
> fluid exerts on it. Fluid and body are coupled explicitly, once per step. The [ghost-cell wall]({{root}}ibm/wall.html) follows the body, and every run (serial, any rank count,
> uniform or adaptive mesh) gives the same bits.

## Kinematics

The body has a pose, the position of its reference point $\mathbf x_b(t)$ (its placement at $t=0$: `ibm_cx`, `ibm_cy`, `ibm_cz`) and, for a polygon, an angle $\theta(t)$ about the pivot
$\mathbf x_p$ (the quarter chord of an airfoil). The surface moves as a rigid body, so the wall velocity that enters the wall condition of the [ghost-cell wall]({{root}}ibm/wall.html) is

$$\mathbf u_w(\mathbf x,t)=\mathbf U_b(t)+\Omega(t)\,\hat{\mathbf z}\times(\mathbf x-\mathbf x_p),\qquad \mathbf U_b=\dot{\mathbf x}_b,\ \ \Omega=\dot\theta .$$

**Prescribed motion:** $\mathbf x_b = \mathbf x_b(0) + \mathbf U t$ and $\theta=\Omega t$ with constant $\mathbf U=$ (`ibm_vx`, `ibm_vy`, `ibm_vz`) and $\Omega=$ `ibm_omega`. Any
nonzero value makes the body move. A circle only translates. A sphere also spins, about any axis through its centre: with $\boldsymbol\Omega=$ (`ibm_omega_x`, `ibm_omega_y`, `ibm_omega`) the wall velocity is

$$\mathbf u_w=\mathbf U_b+\boldsymbol\Omega\times(\mathbf x-\mathbf x_b),$$

the sphere itself does not change, so a spin never rebuilds the geometry (only the wall velocity of the ghost cells and of the cells inside the body).

The rotation can be limited in time: with `ibm_omega_from` $=t_0$ and `ibm_omega_until` $=t_1$ the angle is
$\theta(t)=\Omega\,\bigl(\min(\max(t,t_0),t_1)-t_0\bigr)$ and $\Omega(t)=\Omega$ only for $t_0\le t<t_1$: the body rests, turns, and holds the new angle (`ibm_omega_until = 0`: it never stops).
A nose-up pitch is clockwise, so `ibm_omega` is negative. The case [`IBM Airfoil Pitch`]({{root}}cases/airfoil-pitch.html) uses it for a NACA 0012 at Re = 1000 that rests at 0 degrees, pitches to 15 degrees and holds.

<img src="{{root}}img/ibm_kinematics.png" style="width:100%">

*Left: the wall velocity $\mathbf u_w$ along the surface of an airfoil that rises at $\mathbf U_b$ and pitches nose up ($\Omega<0$) about its quarter chord. Right: the angle of a limited rotation: at rest until $t_0$, turning at $\Omega$ until $t_1$, then held.*

## The coupled system

With a body driven by the flow (`ibm_mass` $= m > 0$) the unknowns are the fluid state $\mathbf U(\mathbf x,t)$ on the moving domain $\Omega_f(t)=\Omega\setminus\Omega_b(t)$ and the body
state $(\mathbf x_b,\mathbf U_b)$. The system solved is

$$\begin{aligned}
&\partial_t \mathbf U + \nabla\!\cdot\!\mathbf F(\mathbf U) = \nabla\!\cdot\!\mathbf F_v(\mathbf U,\nabla\mathbf U) + \rho\,[0,\ \mathbf g,\ \mathbf u\!\cdot\!\mathbf g]^\top && \text{in } \Omega_f(t),\\
&\mathbf u=\mathbf U_b\ \ (\text{no-slip}),\quad (\mathbf u-\mathbf U_b)\!\cdot\!\mathbf n=0\ \ (\text{slip}),\quad \partial_n\rho=\partial_n p=0 && \text{on } \Gamma(t)=\partial\Omega_b(t),\\[4pt]
&m\,\dot{\mathbf U}_b = \mathbf F_f(t) - \mathsf K\,(\mathbf x_b-\mathbf x_b(0)) - \mathsf C\,\mathbf U_b + m\,\mathbf g, \qquad \dot{\mathbf x}_b=\mathbf U_b, &&\\
&\mathbf F_f=-\oint_{\Gamma(t)}\bigl(p\,\mathbf n-\boldsymbol\tau\,\mathbf n\bigr)\,\mathrm dS, &&
\end{aligned}$$

with $\mathbf g$ the gravity of `use_bodyforce` (zero without it), $\mathsf K=\operatorname{diag}(k_x,k_y,k_z)$ and $\mathsf C=\operatorname{diag}(c_x,c_y,c_z)$. The two
subsystems communicate through the surface only: the fluid sees the pose $\mathbf x_b(t)$ (the shape of $\Omega_f$) and the velocity $\mathbf U_b(t)$ (the wall condition); the body sees the load $\mathbf F_f$.

**A rotation driven by the flow** (`ibm_inertia` $=I>0$: a polygon or a cylinder in 2-D, a sphere in 3-D) adds the angle $\theta$ about the pivot $\mathbf x_p$ (`ibm_cx`, `ibm_cy`) to the body state:

$$I\,\ddot\theta = M_f(t) - k_\theta\,\theta - c_\theta\,\dot\theta,\qquad M_f=-\oint_{\Gamma(t)}(\mathbf x-\mathbf x_p)\times\bigl(p\,\mathbf n-\boldsymbol\tau\,\mathbf n\bigr)\,\mathrm dS ,$$

the moment of the same surface load, summed over the solid cells about the pivot of the current pose in the same global order as the force. The body is either pinned at its placement (`ibm_mass` $=0$:
it only turns, about the pivot) or, for a cylinder and a sphere, free to translate too (the pivot is its centre of mass, so the two motions are not coupled). A body that translates freely and turns about a pivot
other than its centre of mass needs the coupling of the two (static imbalance), which is not built, and is rejected.

A **sphere** has the isotropic inertia $I$ and no angle (its geometry does not change): the state is the angular velocity vector $\boldsymbol\Omega=(\Omega_x,\Omega_y,\Omega_z)$, started from (`ibm_omega_x`, `ibm_omega_y`, `ibm_omega`),
$I\,\dot{\boldsymbol\Omega}=\mathbf M_f-c_\theta\boldsymbol\Omega$ (no gyroscopic term, as $I$ is a scalar; no spring: `ibm_stiff_theta` is rejected), and the three moments are summed in the same global order.

<img src="{{root}}img/ibm_coupling.png" style="width:100%">

*The coupled system. The body hands the fluid its pose and velocity, and the fluid hands the body its load. The spring, damper and weight act on the body only. Buoyancy needs no term of its own: it is part of the pressure load $\mathbf F_f$.*

| Symbol | Key | Meaning |
|---|---|---|
| $m$ | `ibm_mass` | mass [kg per unit depth in 2-D, kg in 3-D]; $>0$ switches the mode on |
| $\mathbf U_b(0)$ | `ibm_vx`, `ibm_vy`, `ibm_vz` | initial velocity (the prescribed velocity when $m=0$) |
| $\mathsf K$ | `ibm_stiff_x`, `ibm_stiff_y`, `ibm_stiff_z` | spring stiffness, force $-k\,(x_b-x_b(0))$ toward the placement |
| $\mathsf C$ | `ibm_damp_x`, `ibm_damp_y`, `ibm_damp_z` | damping, force $-c\,U_b$ |
| $\mathbf g$ | `gx`, `gy`, `gz` with `use_bodyforce` | weight $m\mathbf g$ on the body, and the gravity of the fluid |
| $I$ | `ibm_inertia` | moment of inertia about the pivot [kg m$^2$ per unit depth]; $>0$ frees the rotation; `ibm_omega` is then its initial angular velocity |
| $k_\theta$, $c_\theta$ | `ibm_stiff_theta`, `ibm_damp_theta` | torsional spring (moment $-k_\theta\theta$ toward the placement) and damper (moment $-c_\theta\dot\theta$) |

**Buoyancy needs no model.** In a gas at rest, $\nabla p_0=\rho_0\mathbf g$, the pressure part of $\mathbf F_f$ is $-\oint p_0\,\mathbf n\,\mathrm dS=-\int_{\Omega_b}\nabla p_0\,\mathrm dV=-\rho_c V\,\mathbf g$
($\rho_c$: the gas density at the body, $V$: its volume), Archimedes' principle; the solver adds only the weight $m\mathbf g$. The net force is $(m-\rho_c V)\,\mathbf g$.

## One step

Let $\mathbf x_b^n,\mathbf U_b^n$ be the body at the start of step $n$ and $\Delta t=t^{n+1}-t^n$.

<img src="{{root}}img/ibm_step.png" style="width:100%">

*One coupled step. The fluid advances with the body frozen at its state at $t^n$. The load is read from the first stage, and only then does the body advance. The fluid is not solved again with the new pose within the step.*

1. **Pose.** The pose is fixed during the step. The geometry (solid mask, ghost cells, image points) is rebuilt when $\max(|\mathbf x_b-\mathbf x_b^{\rm built}|,\ |\theta-\theta^{\rm built}|\,r_{\max})>h/4$,
   with $r_{\max}$ the largest distance of the body from its pivot ($h$: the finest cell size); in between only the wall velocity $\mathbf u_w$ is refreshed. A body may not travel more than one cell in a step:
   $\Delta t\,\max_d(|U_{b,d}|/h_d)\le1$, otherwise the run stops (lower the CFL number).
2. **Fresh cells.** A cell that was solid at the old pose and is fluid at the new one has no valid state. It is filled from the image point of its own wall: with the closest wall point
   $\mathbf x_w$, $d_c=|\mathbf x_c-\mathbf x_w|$ and the image at $d_i$,
   $\rho_c=\rho_i,\ p_c=p_i,\ \mathbf u_c=\mathbf u_w+(d_c/d_i)(\mathbf u_i-\mathbf u_w)$, a linear profile from the wall value (a cell with no usable image point copies the nearest fluid cell at the wall velocity).
3. **Fluid step.** The usual Runge-Kutta step with the wall at $\mathbf x_b^n$, $\mathbf U_b^n$.
4. **Loads.** $\mathbf F_f^n=\sum_{c\in{\rm solid}}V_c\,\mathbf R_c^{\rm mom}(\mathbf U^n)$ of the [loads]({{root}}ibm/wall.html#loads-on-the-body), taken from the first Runge-Kutta stage on a uniform mesh. With AMR (subcycling has no single first stage) one
   extra residual of the state at the start of the step is evaluated. The terms of the solid cells are summed in one fixed global cell order, gathered on every rank, so every rank adds the same bits and
   advances its own copy of the body identically.
5. **Body update.** With $\mathbf a^n=\bigl(\mathbf F_f^n-\mathsf K(\mathbf x_b^n-\mathbf x_b(0))-\mathsf C\mathbf U_b^n\bigr)/m+\mathbf g$ and $r=\Delta t^n/\Delta t^{n-1}$ (Adams-Bashforth 2 for a variable step; Euler on the first step and after a step more than twice the previous one):

$$\mathbf U_b^{n+1}=\mathbf U_b^n+\Delta t\Bigl[\bigl(1+\tfrac r2\bigr)\mathbf a^n-\tfrac r2\,\mathbf a^{n-1}\Bigr],\qquad
\mathbf x_b^{n+1}=\mathbf x_b^n+\tfrac{\Delta t}{2}\bigl(\mathbf U_b^n+\mathbf U_b^{n+1}\bigr).$$

A driven rotation is advanced by the same law, with $\alpha^n=(M_f^n-k_\theta\theta^n-c_\theta\Omega^n)/I$ in place of $\mathbf a^n$ and $\theta,\Omega$ in place of $\mathbf x_b,\mathbf U_b$ (a sphere: each component of $\boldsymbol\Omega$). The wall
velocity of the rotation is refreshed every step (the geometry only follows the quarter-cell rule above), and a rotation counts in the one-cell-per-step limit through $\Delta t\,|\Omega|\,r_{\max}/h$.

<img src="{{root}}img/ibm_fresh_cells.png" style="width:100%">

*Step 2 on a cylinder moving right (the displacement is exaggerated: the geometry is rebuilt once the body has moved $h/4$). The cells it leaves behind (blue) were solid and hold no flow state. Each one is filled from the image point of its own wall, like a ghost cell, but on the fluid side. The cells it covers (hatched) join the solid.*

The coupling is explicit and loose: one fluid solve per step, no sub-iterations. It is stable for the bodies tested, down to a mass of 0.3 times that of the fluid the body displaces. The body law is integrated with a second-order scheme; the order of the coupled
system as a whole is not established. The body needs a Runge-Kutta or Euler time scheme, no
`positivity_fallback`, and no `ibm_curvature`.

**A light rotating body.** The torque of a body turning in a viscous fluid has no inertial part (a sphere or a cylinder has no added moment of inertia: the torque grows only like $\sqrt{s}$ at high
frequency, a history term), yet the explicit coupling is unstable for a light enough body. Measured on a disk released with a spin: at CFL 0.4 it is stable down to $\rho_b/\rho\approx0.4\,h/R$ ($h$: cell size, $R$: radius)
and at 20 cells per diameter follows the exact angular velocity to 1 % of its initial value (from $t=0.01$ on; 4.5 % at 10 cells per diameter); below that it grows without bound. A smaller CFL number lowers the threshold (at $h/R=0.1$ a disk of $\rho_b/\rho=0.02$ is stable and accurate at CFL 0.2 and
diverges at CFL 0.4), so a lighter body needs a smaller time step or an implicit coupling (not built).

## What is supported

| Body | Mesh | Prescribed | Driven by the flow |
|---|---|---|---|
| cylinder, airfoil | uniform, 2-D (serial, MPI) | translation, rotation (polygon) | translation; rotation (pinned airfoil, cylinder with free translation) |
| cylinder, airfoil | adaptive, 2-D (serial, MPI) | translation, rotation (polygon) | translation; rotation (pinned airfoil, cylinder with free translation) |
| sphere | uniform, 3-D (serial, MPI) | translation, spin about any axis | translation; spin about any axis |
| sphere | adaptive, 3-D octree (serial, MPI) | translation, spin about any axis | translation; spin about any axis |

Not available: free translation together with rotation about a pivot other than the centre of mass, a rotation driven in 3-D for a body other than a sphere, `ibm_curvature`, the Hancock scheme, `positivity_fallback`.

**Checkpoints.** The checkpoint keeps the pose the geometry was last built at and, for a driven body, its position, velocity and last acceleration, and for a driven rotation its angle, angular velocity and last angular acceleration (the key `body_pose`). A resumed run continues bit
for bit, also on another rank count.

## Validation

Two reference problems have a closed-form answer through the added-mass oscillator: the fluid moving with a slowly accelerating body behaves as an extra inertia $m_a$ (cylinder
$\rho\pi R^2$ in an incompressible fluid, $1.0535\,\rho\pi R^2$ at $kR=0.169$ in a gas; sphere $\tfrac12\rho V$),

$$(m+m_a)\,\ddot x + k\,x = 0 \ \ \Rightarrow\ \ \omega^2=\frac{k}{m+m_a},\qquad\qquad (m+m_a)\,\ddot{\mathbf x}=-(m-\rho_cV)\,\mathbf g .$$

| Study | What it checks |
|---|---|
| `features/ibm_ghost.py --check` | serial $=$ 2-7 ranks bit for bit, uniform and adaptive (quadtree and octree), prescribed and driven bodies (a sphere with strips along y and along x); a sphere started in a gas at rest on the octree against the uniform mesh of its finest resolution (the 40-step mean of the force within 2 % of the peak); resume with `--only resume` |
| `features/ibm_spring.py` | the spring oscillator $(m+m_a)\ddot x+kx=0$: frequency within 0.4 %, stable down to $m=0.3\rho V$ |
| `features/ibm_gravity.py`, `ibm_gravity3d.py` | $(m+m_a)\ddot{\mathbf x}=-(m-\rho_cV)\mathbf g$ in a stratified isothermal gas: cylinder and sphere at mass ratios 3, 1 and 0.5 |
| `features/ibm_spin_sphere.py` | a sphere spun up in a viscous gas at rest: the torque against the exact Stokes transient (within 2 % at 10 cells per diameter), spin about x and y equal to spin about z with the axes relabelled (roundoff) |
| `features/ibm_pitch_response.py` | a sprung NACA 0012 released in the flow, its pitch free: damping, effective stiffness, and the divergence boundary $K<0.075$ against the static polar (two meshes) ([pitching airfoil]({{root}}cases/airfoil-pitch.html)) |
| `features/ibm_spin_down.py` | a cylinder released with a spin in a viscous gas at rest, its rotation free: the angular velocity against the exact transient $\hat\omega=I\omega_0/\bigl(Is+2\pi\mu R^2(2+xK_0(x)/K_1(x))\bigr)$, $x=R\sqrt{s/\nu}$ (within 0.3 % of $\omega_0$ from $t=0.1$ and within 2.5 % from $t=0.01$, at $\rho_b/\rho=10$, 1 and 0.1); a sphere with $\hat\omega=I\omega_0/\bigl(Is+\tfrac{8\pi}{3}\mu R^3\,\tfrac{x^2+3x+3}{x+1}\bigr)$ (within 1.4 % of $\omega_0$ from $t=0.1$ at 10 cells per diameter, 7.7 % from $t=0.01$: the first steps after an impulsive release are not resolved; spin about x and about z equal); the stability limit of the loose coupling in $\rho_b/\rho$ against $h/R$ and the CFL number |
| `features/ibm_falling_sphere.py` | a sphere falling to its terminal speed (Re 32, Mach 0.13, octree, 30 diameters of fall): terminal speed 7 % below the Schiller-Naumann reference (band 10 %), the load of a sphere prescribed at that speed equal to the free fall's within 1 % |

The gravity studies agree for the sinking bodies (cylinder 0.8 %, sphere 0.3 % at the finest mesh) and the neutral body stays. A body lighter than the gas it displaces rises about 12 % faster than the
reference in both the cylinder and the sphere. The cause is not isolated (it does not depend on the time step, and the 3-D start-up transient decays faster than the 2-D one), so that case is reported, not gated.
