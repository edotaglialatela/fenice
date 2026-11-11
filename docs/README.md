# docs/

* `source/` the documentation site (Markdown, built by `python docs/build.py` into the gitignored `site/`, published by `.github/workflows/docs.yml`).
  It describes the solver as it is.
* `design/` plans for work that is not done yet; each starts with a status line.
  * `weno-amr.md` WENO5-Z on the adaptive mesh, and the other open items of the adaptive engine (remesh cost on MPI, OpenMP step, validation).
* `proposals/` ideas and studies that are open or partly done, each starting with the status of what became of it. Finished ones are removed (git
  history keeps them).
  * Partly done: `ibm-wall-treatment.md` (the ghost-cell wall and the curvature term are in; the study and what is left),
    `benchmarks-robustness-uq.md` (the benchmarks still to build), `memory-traffic-measurement.md` (what is left after the uniform-solver
    measurements), `repository-review.md` (what was left on purpose).
  * Not started, the owner's choice: `postprocessing-and-species.md`, `thermochemistry-library.md`, `unstructured-solver.md`,
    `distributed-checkpoint-write.md`, `gui-workbench.md` (the graph exists; the thermochemistry node and saving are proposals),
    `performance-improvements.md` (measured candidates for the kernels, and the questions of operator splitting, threads against ranks
    and the fifth state slot in 2-D), `future-proofing.md` (structural limits, maintainability, and the larger changes ranked).
