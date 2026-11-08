# Paper restructure: outline proposal

Status: proposal only; `fenice_paper/main.tex` is not touched. The paper's counts (references, tests, AMR figures) will change with
the Q8 results of the 3-D octree work, so the rewrite should start after them. The current sections (from `main.tex`) are:
Introduction, Software structure, Governing equations and discretisation (equations, mesh, solvers and reconstruction, fully discrete
scheme, time integration, immersed boundaries), Bitwise reproducibility, Cell-based AMR, Code verification (MMS, time accuracy, exact
solutions, strong shocks), Validation (cavity and GCI, 3-D Taylor-Green, cylinder and airfoil), Graphical interface, Performance
(with the DG comparison), Limitations and outlook, Conclusions, and appendices (orders, sources in the code, WENO).

## Problems of the present order

1. The strongest idea of the code, *the same answer for any thread / rank count and a test that proves it*, sits in section 4,
   after the discretisation, and the AMR section (the largest) does not say it extends that guarantee to a distributed adaptive mesh.
2. Features added since (WENO3/5-Z, `boundary_order = 2`, the sharp ghost wall, the 3-D octree with MPI, spectral post-processing, the
   mutation / fault-injection studies) live in an appendix or are missing from the main line; the reader meets them as exceptions.
3. Verification and validation are separated from the feature they test, so the discretisation sections make claims that are
   evidenced 15 pages later.
4. The GUI section interrupts the numerical argument.

## Proposed thread

Claim: *a compact compressible solver whose every feature (higher-order reconstruction, immersed bodies, adaptive meshes in 2-D and
3-D, MPI) is added under one rule, bitwise equality with the simpler path it extends, and is verified by tests that can fail.*

1. **Introduction**: the gap, the claim above, the contributions as a list the later sections discharge one by one.
2. **The design rule**: bitwise reproducibility as a requirement; the ladder of equalities (thread count, rank count, uniform vs level-0 AMR,
   2-D vs extruded 3-D, fused vs two-pass residual, scalar vs vector kernels, restart and rank-count change). Short, with the test names.
3. **Numerical method** (one section): equations, solvers, reconstruction (MUSCL, WENO-Z, boundary order), time integration and
   the combined viscous time-step budget, immersed boundaries (mask and ghost wall).
4. **Adaptive meshes in 2-D and 3-D**: leaf-based face-list engine, jump closure, subcycling and registers, the octree as the same
   engine with $d = 3$ (table of what changes), distributed state and remesh. Put the 3-D measurements (Q8) here.
5. **Verification**: MMS (space and time, with WENO and AMR), exact solutions, shocks, the fault-injection and mutation studies as
   the evidence that the tests detect defects.
6. **Validation**: cavity with GCI and V&V 20, Taylor-Green 2-D and 3-D (with spectra up to Nyquist and the numerical-dissipation share),
   cylinder, airfoil, sphere; each result next to the claim it supports.
7. **Performance**: uniform, AMR against uniform at equal accuracy, scaling, memory per rank; the DG comparison.
8. **Limitations and outlook**; **Conclusions**.
   Appendices: observed orders, WENO literature and limits, sources in the code. The GUI moves to a short paragraph in section 2 or an appendix.

## Order of work

1. Wait for Q8 and for the 3-D Taylor-Green / sphere presentation to settle.
2. Rewrite sections 2 and 4 first (they carry the new thread); then fold the appendix material into 3 and 5.
3. Regenerate every count from the repository (tests, references, studies) in one pass, at the end.

Open choice for the author: whether the target journal accepts a longer paper that keeps all of the above, or whether verification
should be cut to the MMS, the exact solutions and the mutation study with the rest in a supplement.
