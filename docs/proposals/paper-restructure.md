# Paper restructure: outline proposal

Status: approved by the owner on 2026-11-09 and carried out (section "Carried out" at the end), docs site first (part A below), then the paper (part B). Decisions:
keep all verification and validation, shown shorter where less relevant (a table instead of a figure); the viscous time-step coefficient in
the paper is corrected to the solver's. The Q8 results of the octree work are in, so the counts can be regenerated at the end. The current sections (from `main.tex`) are:
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

---

# Part A: the documentation site (docs/source)

Measured first (`shingles.py`, word 7-grams shared between files): verbatim repetition is small (one block shared by `cylinder.md` and
`SolutionMapping.md`, a few lines between `parameters.md` and `studies.md`). The real problems are (1) `index.md` carries a second, older
feature list (it still says "MUSCL, four limiters", "optional MUSCL", a "setup dialog") and the repository layout that the README also
has; (2) `testing.md` (484 lines) mixes two audiences: how to run the tests and CI, and the physical-validation studies (GCI, V&V 20, fault
injection, mutation); (3) validation pages repeat the run command, quick/full mode and the bands of the same study in the page, in
`testing.md` and in the script docstring; (4) one broken anchor (`parameters.html#source-terms-manufactured-solutions`).

Target map (file names and URLs stay, except one new page):

| Page | Role after the restructure |
|---|---|
| `index.md` | Landing page: what Fenice is (current features, short), quick start, a "where to read what" map; equations, time step and data conventions stay here (unique content); the repository layout table goes to the README only |
| `guide/getting-started.md`, `parameters.md`, `writing-cases.md`, `gui.md`, `switches.md`, `performance.md` | unchanged in role |
| `guide/testing.md` | How the solver is tested: unit tests, regression references, CI, `check_all.py`, golden dumps |
| `guide/validation.md` (new) | Physical validation: the one place for how to run the studies (`run_all.py`, tiers, `--full`, `--docs`), a table study -> quantity -> band -> page, and the GCI, V&V 20, fault-injection and mutation sections moved from `testing.md` |
| `guide/studies.md` | Queue, stress cases, analysis layer (unchanged; the shock-blend paragraph links to `parameters.md`) |
| `modules/*.md` | One fact one page: hot start only in `SolutionMapping.md` (its figure lives with the cylinder case), IBM only in `Physics.md`, AMR only in `AMR.md` |
| `cases/*.md` | Result and figure of each case; the run command stays, bands are stated once in the validation table and linked |

Checks for every group of pages: `docs/build.py` (also fails on an undocumented parameter), a link and anchor check of `docs/site`
(baseline: 1 broken anchor, fixed), `check_switches.py`, and a "nothing lost" diff of the numbers in `docs/source` before and after; every
number that disappears is a deliberate cut and is listed in the commit message.

# Part B: the paper (outside the repository)

Back up `fenice_paper/` first; reorder with a script that cuts at the section markers and checks that the multiset of lines is unchanged and
that every `\label` and `\cite` key survives; only then merge duplicates. Build with pdflatex, bibtex, pdflatex, pdflatex. Counts are
regenerated last (reference states 75, unit tests 192, the fault-injection numbers), and the sources appendix lists the tracked files
including the split ones (`Simulation*.cpp`, `CellFluxResidual.cpp`, `ViscousBound.hpp`, `parameter_table`).

# Carried out (2026-11-09)

* Part A: `guide/validation.md` split from `testing.md`; `index.md` current and with a reading map; README Run / GUI / Adding-a-case
  sections shortened; the hot-start figure and the shock-blend definition kept on one page each; one broken anchor fixed. The site
  builds and all 3122 links and anchors resolve. Verbatim repetition was small, so the gain is mostly structure and removed stale text.
* Part B: the paper (outside the repository, backup `fenice_paper_backup_2026-11-09`) now reads Introduction, Bitwise reproducibility,
  Software structure (with the graphical interface as a subsection), Numerical method, Cell-based AMR (new subsection on the octree with
  the Q8 measurements), Code verification, Validation, Performance, Limitations, Conclusions. Counts (75 reference states), the abstract,
  highlights, contributions, limitations and the sources appendix were updated; 40 pages, no undefined references. The viscous
  time-step coefficient was already the solver's. Not done: re-running the fault-injection and mutation studies on the current sources
  (their numbers are those of 2026-11-01 and 2026-11-03), and the Trixi.jl comparison results.
