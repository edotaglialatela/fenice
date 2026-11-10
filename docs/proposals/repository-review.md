# Repository structure: what is left

Status: the structural work of the review is done (one header for the expressions that must stay identical between the 2-D / 3-D, uniform /
AMR and scalar / AVX2 copies; `Simulation.cpp` and `CellFlux.cpp` split by phase; `parameter_table()`; `format_version` on the checkpoint; a
`regression.py --update` that refuses a dirty tree and writes a report; the documentation index; CI ranks 1-4 on every push). What follows is
what was left on purpose or decided against. The owner decides what goes ahead.

## Open

* **Parameters from one table.** `parameter_table()` (`Params.cpp`: key, case-file section, baseline default) feeds the baseline defaults and the
  case-file sections, `fenice --print-parameters` lists it, and `docs/build.py` fails when a key is missing from `guide/parameters.md`. Not done: the
  typed fallbacks in `SimParams::from` for absent keys, a one-line description per key in the table, and generating the docs table or the GUI
  tooltips from it (the page carries prose and links that a generated table would lose). Size: medium, low risk.
* **`format_version` in the `dump_state` file.** Only the checkpoint carries it (the file that is read back). The dump / `leaf_meta` file is
  diagnostic output read by key by the regression and the Python tools; a new key would change the hash of every golden dump. No fixture files per
  released version (the unit test writes the old and the future layouts itself).
* **`AGENTS.md` is tracked** (the shared rules; `CLAUDE.md` imports it, `CLAUDE.local.md` is the ignored private layer).

## Decided: no work

* **Dimension as a compile-time parameter in the AMR kernels.** The octree measurements put the cost of 3-D in the remesh and in memory, not in
  dimensional branches, so no templating on `dim`.
* **Portability layer for accelerators.** If a GPU is ever wanted, the fused uniform residual (reconstruction, Riemann and balance per row of faces,
  no face storage) is the natural unit to port first; the face-list AMR engine is not.
