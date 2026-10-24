# Handoff: distributed AMR storage, start of phase 2

Paste this into the new chat, together with `CLAUDE.md` (it is gitignored, so the new session cannot see it otherwise).

## Where things are

* Repository `edotaglialatela/fenice`, branch **`amr-distributed`**. PR **#6** (phases 0 and 1) is open, not merged, and CI is
  green on Linux x64 / ARM64, macOS and Windows MSVC, including real MPI on 1-7 ranks.
* The plan, the decisions and every result are in **`docs/design/amr-distributed.md`**. Read sections 0 (rules, oracles, files),
  3 (design), 4 (phases), 5 (shadow oracle), 7 (decisions), 8 (phase-0 results) and 9 (phase 1) before writing code.
* If PR #6 gets merged before phase 2 starts, start phase 2 on a fresh branch from `main`. If it is still open, continue on
  `amr-distributed`.

## Done

* **Phase 0** (measurements; the numerics did not change):
  * `CellParallel::halo_reach` measures the halo reach and `FENICE_HALO_REACH=1` prints it.
  * The guard test `halo_reach_bounds_the_halo_of_any_partition` checks the real halo against that reach.
  * `memory_usage()`; an AMR run ends with a resident-memory report per rank and a table-size report.
  * `cases/validation/amr/amr_memory.py` is the memory / time study.
* **Phase 1** (same lists as before; the old algorithm ran alongside in every unit test):
  * Ownership is described by per-level splitters (`owner_of(level, key)`).
  * `partition()` works out only its own rank's reads, sends (level, i, j) requests to the owners (two exchanges), and is collective.
  * Registers are listed by (fine leaf, side) on both ranks.

## Key measurements (user's MSVC machine, airfoil, 50 steps)

* **Memory:** about 1.1-1.5 kB per leaf on every rank, and the peak per rank is the same on 1 and 16 ranks:
  * 720 x 360: 387 MB serial, 380 MB on 16 ranks.
  * 1440 x 720: 1.51 GB serial, 1.49 GB on 16 ranks.
  * The engine is three quarters of it. The per-face `F_`, `Fv_`, `reg_` and `regv_` are sized for every face, but `reg_` / `regv_`
    are needed only for coarse-fine faces and `Fv_` only when viscous: a cheap saving with no change to the bits.
* **Time, 720 x 360:** wall 5.25 / 3.86 / 2.89 / 3.42 / 5.74 s on 1 / 2 / 4 / 8 / 16 ranks. The remesh, repeated on every rank,
  goes from 0.79 to 3.36 s.
* **Time, 1440 x 720:** wall 20.3 / 13.8 / 11.4 / 13.9 / 24.2 s. The remesh goes from 3.1 to 13.9 s.
* The best rank count is 4. The repeated O(N) work (indicator, adapt, build, partition) slows down 2.5-5x from 2 to 16 ranks
  because the ranks share the memory bandwidth.
* **Phase 1 did not make `partition()` faster** (0.72 s on 16 ranks at 720, 2.92 s at 1440). The R x levels x N masks were
  not the cost; the O(N) loops of the replicated mesh are, and they go away in phases 2 and 4.
* **Halo reach**, in rings of the owning leaf's level:
  * same level 5; finer leaves 2-3; coarser leaves 6-7 on the airfoil, 10 on the four-level test meshes;
  * ghost images 7;
  * use the per-level-difference table from `halo_reach`, not a constant.

## User decisions (section 7 of the plan)

* **D1:** checkpoints become per-rank files, and a resume works on the same rank count only. `dump_state` keeps a gather to
  rank 0. VTK stays one `.vtu` per rank.
* **D2:** monitor sums become per-rank sums plus an allreduce. They then agree only to roundoff across rank counts; the states
  stay bitwise. This needs a sum-allreduce in `CellComm`.
* **D3:** after a remesh, allgather the new keys. No sample sort unless a measurement shows the allgather matters.
* **D4:** the target case is the airfoil, 720 x 360 and 1440 x 720.
* **D5:** keep `FENICE_REPLICATED_REMESH=1` as the old path during phases 2-5, and remove it after the last phase once everything works.

## Next: phase 2 (the large one)

Local storage for time stepping:

* After the remesh, which stays replicated for now and gathers as today, each rank keeps only its owned leaves and its halo.
* Each rank builds a partial local `CellMesh` and `CellFluxEngine`, sorted by (level, key), owned or halo flagged.
* Sides that look outside the local set are `Outside`; reading one from an active face must throw.
* The global mesh is dropped until the next remesh. The next remesh gathers again, and the gather carries leaf keys plus state.
* The lists of `CellParallel` are already keyed, and their order is (level, key) on both sides, so they carry over.
* Use the shadow oracle of section 5: in the `FakeHub` tests, compare the local tables with the replicated ones by key.
* Take the per-face register saving here.
* Bits must not move.
* Consult the advisor on the approach before writing code. It is a rewrite of the indexing of `CellMesh`, `CellFluxEngine` and
  `CellParallel`, plus their use in `Simulation::solve_amr` (monitor, loads, export, checkpoint, wave-speed bound).

## Working rules in practice (cloud container)

* **Build:**
  * The container is Linux with 4 cores and no MPI: `cmake --preset release -DFV_USE_MPI=OFF && cmake --build build/release -j4`.
  * `build/nosimd` is a configured scalar build (`-DFV_NO_SIMD`): `cmake --build build/nosimd`.
* **Golden reference: freeze it before the first edit.**
  * Build the branch head, copy `build/release/fenice` to the scratchpad, and run
    `python cases/validation/amr/golden.py dump <scratch>/old --exe <copy> --quick` (about 4 min).
  * After each step, run `... dump <scratch>/new --quick` and `... compare <scratch>/old <scratch>/new`.
  * Golden runs are serial / OpenMP only (`use_mpi=false`). Multi-rank correctness comes from the `FakeHub` unit tests
    (NaN-poisoned halos) and from CI's real MPI.
* **Every commit:**
  * `build/release/fenice_tests`;
  * `FENICE_EXE=build/release/fenice python cpp/tests/regression.py` (must print ALL MATCH);
  * the golden compare (also with `--env FENICE_FULL_REBUILD=1` on a subset);
  * the nosimd tests.
* **Development trick that worked in phase 1:** keep the old algorithm temporarily, run it alongside the new one behind an
  environment variable, compare every table in the unit tests, then delete it before committing.
* **GitHub:**
  * `gh` GraphQL is blocked: use REST (`gh api repos/edotaglialatela/fenice/pulls...`).
  * `gh run list --branch amr-distributed` and `gh run view <id> --json conclusion,jobs` work.
  * Push, wait for CI, then update PR #6 (or open the next PR) when a phase is done.
* **Timings and memory on many ranks** come from the user: `python cases/validation/amr/amr_memory.py --force` on the MSVC
  machine (Ryzen 7 7800X3D) and the Threadripper (WSL / gcc).

## Lines for CLAUDE.md (the user adds them at the end)

* `halo_reach_bounds_the_halo_of_any_partition` guards the halo reach. Its pinned values change only together with the halo of
  the distributed AMR (`docs/design/amr-distributed.md`).
* `FENICE_HALO_REACH=1` prints `CellParallel::halo_reach`. AMR runs end with the resident memory per rank and the table sizes;
  `cases/validation/amr/amr_memory.py` is the memory / time baseline.
* `CellParallel::partition()` is collective. Halo requests travel as (level, i, j) and the lists are ordered by (level, key);
  registers are listed by (fine leaf, side). Nothing that crosses ranks is a leaf index.
