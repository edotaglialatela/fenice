# docs/

* `source/` the documentation site (Markdown, built by `python docs/build.py` into the gitignored `site/`, published by `.github/workflows/docs.yml`).
* `design/` how the adaptive-mesh engine was designed and built, one file per stage; each starts with a status line.
  * `amr-cell-based.md` the 2-D cell-based engine (leaves, face list, subcycling, MPI, indicator). Done, merged.
  * `amr-optimisation.md` the three options to speed it up (incremental rebuild, distributed storage, SIMD kernel). All done.
  * `amr-distributed.md` distributed storage and the local remesh. Done (phases 0-6).
  * `amr-octree.md` the 3-D octree engine, phases Q0-Q8b with the measurements. Done; the 40 cells/D runs are optional.
* `proposals/` ideas and studies, each with the status of what became of it.
  * Implemented: `amr-weno-boundary-order.md` (WENO5 on AMR is the open part), `viscous-timestep-stability.md`, `ibm-wall-treatment.md`
    (open: curvature term, conservative wall flux), `benchmarks-robustness-uq.md` (mostly), `memory-traffic-measurement.md` (what is left
    is listed in it), `positivity-corners-open-boundaries.md` (decided), `repository-review.md` (items 1, 2, 4, 7, 9 done; 3 partly).
  * Not started, the owner's choice: `postprocessing-and-species.md`, `thermochemistry-library.md`, `unstructured-solver.md`,
    `distributed-checkpoint-write.md`, `gui-workbench.md` (the graph exists; the thermochemistry node and saving are proposals),
    `paper-restructure.md` (after the Q8 results).
