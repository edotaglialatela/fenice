# Product

## Register

product

## Users

CFD researchers and the solver's own author — expert users who already understand
the underlying physics (compressible Euler / Navier-Stokes, Riemann solvers, MUSCL,
IBM, AMR, MPI). They sit down to configure and launch a simulation run, not to
learn the field. Context is a desktop workstation, often before a long compute job:
they want to set mesh dimensions, thermodynamics, boundary conditions, numerical
schemes, and advanced features (immersed boundaries, solution mapping/hot-start,
AMR) correctly and confidently, then get out of the way of the solver.

The primary surface is the `SetupDialog` (PyQt6, `gui/setup_dialog.py`, launching the C++ solver): a four-tab
configuration form — Configuration, Immersed Boundaries, Solution Mapping, AMR —
with a logo panel. Every screen is a setup task; success is a correctly specified
run that launches without a misconfiguration surprise.

## Product Purpose

A structured-Cartesian 2D compressible finite-volume solver. The GUI's job is to
turn a dense set of physical and numerical parameters into a fast, error-resistant
setup experience. It exists so that an expert can express a precise simulation
intent — domain, EOS, BCs, scheme, parallelism, advanced physics — without hand-
editing dictionaries or second-guessing which fields interact. Success looks like:
the right controls visible at the right time, invalid combinations caught before
launch, and a panel that feels like a trustworthy instrument rather than a default
toolkit dialog.

## Brand Personality

**Precise, quiet, instrument-like.** Calm, exact, restrained. The interface is a
well-tuned instrument panel: it states values clearly, conveys state without
shouting, and lets the physics be the star. Voice is technical and unembellished —
correct units, exact labels, no marketing tone. Emotional goal: the confidence of
working with a precision tool that won't surprise you.

## Anti-references

- **Bloated enterprise CAD/CAE** (ANSYS Fluent, Star-CCM+ style): ribbon bars,
  over-chromed panels, nested toolbars, decorative gradients, heavy framing. Avoid
  the dense, busy, "everything always visible" enterprise weight.
- Raw unstyled default Qt — cramped, gray, no hierarchy — is the current floor to
  rise above, not a destination.
- Consumer/playful or gamified aesthetics; this must stay credible and technical.

## Design Principles

1. **Instrument, not toolbox.** Every control reads like a labeled dial on a
   precision panel — exact, calm, unambiguous. Restraint is the default.
2. **Progressive disclosure over density.** Advanced and conditional options
   (viscous terms, body forces, MPI, IBM geometry, hot-start fallbacks) appear only
   when relevant. Don't show the whole enterprise console at once.
3. **Catch errors before launch, not after.** Validation, sensible defaults, and
   clear interdependency cues (e.g. AMR vs MPI mutual exclusion) prevent costly
   misconfigured runs. The UI guards the long compute job behind it.
4. **State legible at a glance.** Enabled/disabled, valid/invalid, default/overridden
   should be readable without hunting. Quiet, not flat.
5. **Units and labels are exact.** Physical correctness in the copy is part of the
   craft: right symbols, right units, no approximations in nomenclature.

## Accessibility & Inclusion

Target WCAG 2.1 AA. Body text ≥4.5:1, large/bold text ≥3:1, placeholders held to
the same body bar. Clear, visible keyboard focus states across the form. Legible
at standard desktop DPI. No special assistive requirements beyond the AA baseline,
but state and validation cues should remain understandable without relying on color
alone where it's cheap to add an icon or text label.
