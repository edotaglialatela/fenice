---
title: Taylor-Green vortex 3-D - transition to turbulence
nav: Taylor-Green vortex 3-D
section: Test cases
order: 90
---

The 3-D Taylor-Green vortex is the standard check for 3-D compressible Navier-Stokes
solvers (the "TGV" of the High-Order CFD Workshops). A single large-scale vortex
arrangement on the periodic box $[0, 2\pi]^3$,

$$u = U_0 \sin x \cos y \cos z, \qquad v = -U_0 \cos x \sin y \cos z, \qquad w = 0$$

$$p = p_0 + \frac{\rho_0 U_0^2}{16}\bigl(\cos 2x + \cos 2y\bigr)\bigl(\cos 2z + 2\bigr), \qquad \rho = \rho_0$$

has no exact solution. Vortex stretching, which exists only in 3-D, transfers energy to
ever smaller scales, the flow becomes turbulent around $t \approx 9$, and the kinetic
energy then decays. The rate of decay $\varepsilon = -dE_k/dt$ peaks at
$\varepsilon \approx 0.0127$ near $t \approx 9$ in the spectral DNS at $512^3$
(van Rees et al. 2011).

The case uses $U_0 = \rho_0 = 1$, $\mathrm{Ma} = 0.1$ ($p_0 = 1/(\gamma\,\mathrm{Ma}^2)$),
$\mathrm{Re} = \rho_0 U_0 L/\mu = 1600$ ($\mu = 6.25\times10^{-4}$, $L = 1$), Low-Mach
HLLC, MUSCL with Van Leer, SSP-RK2 and periodic boundaries on all six sides.

**What it validates.** The default **32³** grid is deliberately coarse: the case is a
quick check of the 3-D solver, not a DNS. What it exercises:

* **The z-direction everywhere**: z-faces, front/back periodic halos, 3-D MUSCL, the full
  viscous stress tensor and the 3-D time-step limits.
* **Vortex stretching**: the flow only becomes turbulent if all three directions are
  coupled correctly. A 2-D-like bug (missing $w$ coupling, a wrong cross-derivative)
  leaves it laminar.
* **The dissipation balance**: at $t = 0$ the exact dissipation is
  $2\nu\zeta = \nu\langle|\boldsymbol\omega|^2\rangle = 0.75\,\nu$.

```powershell
cpp\build\fenice.exe --case "Taylor-Green Vortex 3D" --set monitor_interval=10
python validation\validationTaylorGreen3D.py            # 32³ + 64³ (~6 min with 16 threads)
python validation\validationTaylorGreen3D.py --skip-64  # 32³ only (~20 s)
python validation\validationTaylorGreen3D.py --grids 32 64 96 --vorticity-n 96 --mpi 16 --q-animation
```

The last line is what produced the figures below (about 40 min on 16 cores); `--q-animation`
needs PyVista (`python prerequisites.py --viz3d`).

`monitor_interval` writes the domain-averaged mass and kinetic / total energy to
`<case>_monitor.csv` (see [IO]({{root}}modules/IO.html#monitor-monitor_interval)), which
is all a kinetic-energy history needs.

## Default parameters

{{defaults:Taylor-Green Vortex 3D}}

## Results

<figure>
<img src="{{root}}img/taylor_green_3d_validation.png" style="width:100%">
<figcaption>Left: kinetic energy. Right: total dissipation rate −dE<sub>k</sub>/dt from the monitor,
the resolved physical part 2νζ (enstrophy) from the 32³ and 96³ snapshots, and the DNS peak.
Peak dissipation: 0.0110 at t = 6.6 (32³), 0.0113 at t = 8.5 (64³) and 0.0125 at t = 8.3 (96³),
against 0.0127 at t ≈ 9 for the DNS; E<sub>k</sub>(20)/E<sub>k</sub>(0) = 0.23, 0.21 and 0.19.</figcaption>
</figure>

<figure>
<img src="{{root}}img/taylor_green_3d_vorticity.png" style="width:100%">
<figcaption>Vorticity magnitude at 96³ on three faces of the box (top) and on the periodic
plane x = 0 (bottom). The initial large-scale vortices (t = 0) roll up into vortex sheets
(t ≈ 4), the sheets break up near the dissipation peak (t ≈ 8), and small-scale structures
fill the box (t ≈ 12).</figcaption>
</figure>

<figure>
<img src="{{root}}img/taylor_green_3d_q.gif" style="width:70%; display:block; margin:0 auto;">
<figcaption>Coherent vortices at 96³: iso-surface Q = 3 of the Q-criterion
Q = ½(‖Ω‖² − ‖S‖²) (rotation dominating strain), coloured by the vorticity ω<sub>z</sub>
(blue negative, red positive), from t = 3.5, when Q first exceeds 3, to t = 20. Vortex tubes
form and pair, reconnect into the ring-like lattice of the dissipation peak (t ≈ 8), break
into small-scale turbulence and decay. The camera turns 60° over the animation.</figcaption>
</figure>

Reading the curves:

* **Physical transition.** Every grid follows the textbook sequence: a laminar phase with
  slowly rising dissipation, a peak, then turbulent decay.
* **Grid convergence.** At 32³ the scheme's numerical dissipation adds to the viscous
  one, so the energy drops too early and the peak comes too soon. Refining moves the peak
  later and higher, toward the DNS: at 96³ it is 2 % below the DNS value, 0.7 time units
  early. This is the expected convergence of a 2nd-order finite-volume scheme on the TGV.
* **Resolved vs total dissipation.** $2\nu\zeta$ equals $-dE_k/dt$ at $t = 0$ (the exact
  $0.75\,\nu$). Later it is only a small part of the total at 32³, so most of the
  dissipation there is numerical: the smallest eddies are far below the grid scale. At 96³
  the resolved part is about three times larger and accounts for over a third of the
  total at the peak.
* **Small oscillations** of the 32³ $-dE_k/dt$ come from acoustic waves at Ma 0.1
  exchanging kinetic and internal energy; the total energy is conserved to round-off.

The automated 3-D checks are listed under [Testing]({{root}}guide/testing.html). They include
the initial dissipation (0.5 % from $0.75\,\nu$) and a stability check of this default case.

## References

* Brachet, M.E. et al. (1983). Small-scale structure of the Taylor-Green vortex.
  *J. Fluid Mech.* 130, 411-452.
* van Rees, W.M., Leonard, A., Pullin, D.I. & Koumoutsakos, P. (2011). A comparison of
  vortex and pseudo-spectral methods for the simulation of periodic vortical flows at
  high Reynolds numbers. *J. Comput. Phys.* 230(8), 2794-2805.
* Wang, Z.J. et al. (2013). High-order CFD methods: current status and perspective.
  *Int. J. Numer. Meth. Fluids* 72(8), 811-845.
