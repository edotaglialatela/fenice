---
title: Pitching airfoil - a NACA 0012 turned about its quarter chord
nav: Pitching airfoil (IBM, Re = 1000)
section: Test cases
order: 46
---

> **Run it** (output in `results/`; [how to open it]({{root}}guide/getting-started.html#your-first-run)):
>
> * `build\release\fenice.exe --case "IBM Airfoil Pitch"`

A NACA 0012 at Re = $\rho\,u_{\rm in}\,c/\mu = 1000$ and Ma = 0.2 that rests at 0° until $t = 15$ (the wake develops), turns nose-up about its quarter chord to a higher angle and holds it. It exercises the
[moving immersed boundary]({{root}}ibm/moving-bodies.html) with a prescribed rotation on an [adaptive mesh]({{root}}modules/AMR.html), and it is the reference for what the wall should reproduce after the motion stops: a static airfoil at the final angle.
Lengths are in chords, times in $c/u_{\rm in}$; the box is $12c \times 6c$ with the quarter chord at $(3c, 3c)$, a velocity inlet on the left, the
[non-reflecting outlet]({{root}}modules/Boundaries.html#non-reflecting-outlet) on the right and slip walls three chords above and below (a blockage of 1/6).

The rotation is set by three keys: `ibm_omega` [rad per unit time, negative = nose-up], `ibm_omega_from` and `ibm_omega_until` (see [Moving bodies]({{root}}ibm/moving-bodies.html#kinematics)). The default is the fast pitch-up of the figures below, 0 to 15° in 1.5 time units:

```powershell
build\release\fenice.exe --case "IBM Airfoil Pitch"                              # fast pitch-up to 15 degrees
build\release\fenice.exe --case "IBM Airfoil Pitch" --set ibm_omega=-0.017453292519943 --set ibm_omega_until=21   # slow pitch-up to 6 degrees (1 degree per time unit)
build\release\fenice.exe --case "IBM Airfoil Pitch" --set ibm_omega=0 --set ibm_omega_from=0 --set ibm_omega_until=0 --set ibm_aoa=6   # the static airfoil at 6 degrees
build\release\fenice.exe --case "IBM Airfoil Pitch" --set amr_max_level=3        # the resolution of the results below
```

The base mesh is $720 \times 360$ (60 cells per chord). The default has two refinement levels (240 cells per chord at the body, which is always at the finest level); the results below use three (480 cells per chord, about 3·10⁵ leaves).
A run to $t = 45$ at three levels takes 1.1 to 1.2 h on 8 cores of the workstation (16 threads); the same run at two levels takes 21 min.

## Default parameters

{{defaults:IBM Airfoil Pitch}}

## Static airfoil

The static airfoil at five angles (three levels, run to $t = 40$; 3°, 6° and 10° continued to $t = 100$). Means and oscillation amplitudes (half the peak-to-peak) over the last 10 time units of each run:

| angle | $C_l$ | $C_l$ amplitude | $C_d$ | $C_d$ amplitude | $C_m$ (quarter chord) |
|---:|---:|---:|---:|---:|---:|
| 0° | 0.0000 | 0.0000 | 0.1228 | 0.0014 | 0.0000 |
| 3° | 0.1610 | 0.057 | 0.1257 | 0.0033 | +0.0079 |
| 6° | 0.2836 | 0.060 | 0.1352 | 0.0057 | +0.0147 |
| 10° | 0.4278 | 0.029 | 0.1724 | 0.0046 | +0.0107 |
| 15° | 0.7073 | 0.246 | 0.2929 | 0.0444 | −0.0425 |

At 0° the lift is zero to round-off ($|\overline{C_l}| = 6\cdot10^{-16}$): the symmetric airfoil in a symmetric box gives a symmetric flow.

<figure>
<img src="{{root}}img/airfoil_pitch_polar.png" style="width:100%">
<figcaption>Static polar of the NACA 0012 at Re = 1000. Markers: means over the last 10 time units of each run; bars: oscillation amplitude in that window. The red triangle is the slow pitch-up
to 6° after the hold (below). The dotted line marks 8°, near which published two-dimensional studies of this flow report the onset of unsteadiness (free air, incompressible: the comparison is qualitative here, with
slip walls at ±3 chords and Ma = 0.2).</figcaption>
</figure>

**Steady below, shedding above.** At 3°, 6° and 10° the lift oscillation that follows the impulsive start decays; at 15° it settles into a vortex street.

<figure>
<img src="{{root}}img/airfoil_pitch_decay.png" style="width:70%">
<figcaption>Amplitude of the lift oscillation (half the peak-to-peak) in windows of 10 time units, after the impulsive start of the static runs. 3°, 6° and 10° decay; 15° is a sustained cycle
(amplitude 0.25 and $f\,c/u_{\rm in} = 0.70$).</figcaption>
</figure>

| angle | amplitude, $t = 30$–40 | $t = 60$–70 | $t = 90$–100 |
|---:|---:|---:|---:|
| 3° | 0.084 | 0.063 | 0.057 |
| 6° | 0.131 | 0.082 | 0.060 |
| 10° | 0.120 | 0.054 | 0.029 |
| 15° | 0.246 | – | – |

The mean lift does not move while the oscillation decays (0.161, 0.284, 0.428). At 3° the decay has slowed to about 0.3 % per time unit by $t = 100$, so that run has not reached a steady state in the window; at 6° and 10° the amplitude is still falling.
The transition from decaying to sustained oscillation lies between 10° and 15°: the angles in between were not run, so the onset is not bracketed.

**The oscillation at $f\,c/u_{\rm in} = 2.7$ is acoustic.** The decaying oscillation at 3°, 6° and 10° has the same dominant frequency, 2.7 (at 10° the dominant peak moves to 0.9 once the amplitude is small, after $t = 70$). It is not vortex shedding: it does not depend on the remesh interval (`amr_refine_every` 5 or 10: same frequency, same amplitude to four digits) or on the time step (the whole 6° run at CFL 0.5 and at CFL 0.25: the same dominant frequency and amplitudes within 0.3 % in every window up to $t = 40$), but it scales with the sound speed. At Ma = 0.3 the 6° run oscillates at $f\,c/u_{\rm in} = 1.8$ over $t = 10$-20, against 2.7 at Ma = 0.2, so $f\,\mathrm{Ma} = 0.54$ in both and the wavelength is $1/(f\,\mathrm{Ma}) = 1.85$ chords. The mean lift hardly changes (0.281 at Ma = 0.3, 0.286 at Ma = 0.2 over the same window). The acoustic path that sets the 1.85 chords is not identified.

## Pitching: the moving wall

Both runs below start at 0° and hold the final angle. The wall velocity of the rotation enters the ghost-cell fill, and the geometry is rebuilt as the body turns.

**Slow pitch-up to 6°** ($t = 15$ to 21, 1° per time unit). After the hold the flow settles on the static 6° solution: $C_l = 0.2829$ against 0.2835 (0.19 %) and $C_d = 0.1352$ against 0.1353 (0.06 %), over $t = 35$–45.
Going from two to three levels changes the settled loads by 1.05 % ($C_l$) and 0.79 % ($C_d$). During the ramp $C_l$ runs above the static polar at the instantaneous angle, by 0.055 on average and 0.104 at the end of the ramp: the lag of an unsteady
flow behind the quasi-steady value.

<figure>
<img src="{{root}}img/airfoil_pitch_slow.png" style="width:100%">
<figcaption>Slow pitch-up from 0 to 6° ($t = 15$ to 21), at two and three levels. Top left: angle. Top right and bottom left: $C_l$ and $C_d$ against the static 6° values (dotted). Bottom right: $C_l$ against the angle
during the ramp, with the static polar.</figcaption>
</figure>

**Fast pitch-up to 15°** ($t = 15$ to 16.5, 10° per time unit). The lift overshoots the static value while a leading-edge vortex forms, then the flow turns into the shedding of the static 15° airfoil. $C_l$ peaks at 1.44 at
$t = 16.25$, 1.25 time units after the start of the rotation, against the static mean of 0.71; $C_d$ peaks at 0.40 against 0.29.

<figure>
<img src="{{root}}img/airfoil_pitch_fast.png" style="width:100%">
<figcaption>Fast pitch-up from 0 to 15° ($t = 15$ to 16.5), then hold. Angle, $C_l$, $C_d$ and $C_m$; the dotted lines are the means of the static 15° run.</figcaption>
</figure>

<figure>
<img src="{{root}}img/airfoil_pitch_vorticity.png" style="width:100%">
<figcaption>Vorticity (red: counter-clockwise) after the fast pitch-up, at $t = 18.5$ (the leading-edge vortex and the starting wake) and at $t = 28.9$ (the established vortex street at 15°). Three levels.</figcaption>
</figure>

<figure>
<img src="{{root}}img/airfoil_pitch_vorticity.gif" style="width:100%">
<figcaption>The same run, $t = 14$ to 27.</figcaption>
</figure>

## Adaptive mesh

The airfoil is kept at the finest level and the indicator (`density+velocity`) follows the shear layers and the vortices, so the refined region turns with the airfoil and then trails the leading-edge vortex. At $t = 20$, with the vortex over the airfoil,
the mesh has 331 164 leaves: 256 758 at level 0, 3 162 at level 1, 11 484 at level 2 and 59 760 at level 3. A uniform mesh at the finest resolution would have $720 \times 360 \times 64 = 16.6$ million cells, so the leaves are 2 % of it.

<figure>
<img src="{{root}}img/airfoil_pitch_mesh.gif" style="width:100%">
<figcaption>The leaves around the airfoil during the fast pitch-up, $t = 14$ to 34. Colour: refinement level (white: level 0, then three darker blues; the airfoil is black); the lines are the leaf edges.</figcaption>
</figure>

## What the runs check

| Check | Result |
|---|---|
| symmetry at 0° | $\overline{C_l} = -6\cdot10^{-16}$ |
| rest phase of the pitching runs equals the static 0° run | identical (difference 0) |
| the moving wall after the hold equals the static wall at the same angle | $C_l$ 0.19 %, $C_d$ 0.06 % (slow run, 6°) |
| mesh: two levels against three, settled loads | $C_l$ 1.05 %, $C_d$ 0.79 % |

The runs used here are workstation runs (24 cores); the case is a few-hours job at three levels and is not part of the automatic checks. The prescribed-rotation schedule itself is covered by a unit test and by the serial / MPI checks of
`features/ibm_ghost.py --check` ([Moving bodies]({{root}}ibm/moving-bodies.html#validation)).

**Limits.** Slip walls at ±3 chords and Ma = 0.2 differ from the free-air incompressible set-up of most published results; the rotation starts and stops impulsively (constant rate); the angle at the start of the rotation is exactly 0.
