# AMR with distributed storage and a local remesh (plan)

Status: **approved (D1-D5 answered, section 7); phase 0 done (section 8), phase 1 next.** Branch `amr-distributed`, from `main`
at 883c45d (both PRs #4 / #5 merged). This is option 2 of `docs/design/amr-optimisation.md`, as a step towards 3-D octree AMR.
Every number below that was not measured is marked *(estimate)*.

## 0. Start here (for a fresh session)

**Rules** (the user's, and CLAUDE.md, which is gitignored: ask the user to paste it if it is not in the context):

* Nothing may change the numerics unless the user says so: **bitwise** oracles after every step. MPI == serial bit for bit for any
  rank count is the central invariant of the AMR engine and stays so.
* Windows / MSVC is the main target (OpenMP 2.0: signed loop indices, no `min` / `max` reductions, no `atomic capture` / `write`;
  `/fp:precise`, no FMA: check `objdump -d <obj> | grep vfm`). Must also build with gcc / clang and on ARM (scalar; `-DFV_NO_SIMD`).
  Loops below `kOmpMinWork` stay serial (`if()` clause).
* Small commits on the branch, push, PR when a phase is done; the user runs timings on a Ryzen 7 7800X3D (MSVC) and a Threadripper
  PRO 5965WX (24 cores, WSL / gcc). The cloud container is Linux, 4 cores, **no MPI**: build with `-DFV_USE_MPI=OFF`, use the
  `FakeHub` unit tests for multi-rank logic, CI for real MPI, the user's machines for timings (container timings only before / after
  on the same binary set).

**Oracles** (all must stay green after every commit):

| oracle | command | what |
|---|---|---|
| unit tests | `build/release/fenice_tests` | incl. `cellparallel_equals_serial_bitwise_for_any_rank_count` (1, 2, 3, 5, 7 emulated ranks, NaN-poisoned halos), `cellflux_incremental_rebuild_equals_full_rebuild`, `remesh_tables_do_not_depend_on_the_thread_count` |
| references | `FENICE_EXE=build/release/fenice python cpp/tests/regression.py` | 38 states (6 AMR) |
| golden dumps | `python cases/validation/amr/golden.py dump OLD --exe <main binary>`, `... dump NEW`, `... compare OLD NEW` | ten adaptive runs, 1-7 threads, monitor on; also `--env FENICE_FULL_REBUILD=1`, `--env FENICE_NO_SIMD=1` |
| real MPI (CI) | `python cases/validation/amr/amr_mpi.py --check`, `python cases/validation/features/ibm_ghost.py --check` | 1-7 ranks bitwise equal to serial |
| halo guard | `build/release/fenice_tests` (`halo_reach_bounds_the_halo_of_any_partition`) | the real halo of 1-16 emulated ranks lies in the reach `CellParallel::halo_reach` measures; the reach is pinned |
| memory / time baseline (user) | `python cases/validation/amr/amr_memory.py` (`--ranks`, `--only`, `--steps`, `--reach`) | RSS per rank, table bytes, remesh parts on 1-16 ranks (section 8) |
| timings (user) | a script like the one used for PR #4 (clone, build `main` and the branch, interleaved runs, min of 3) | |

**Files.** `cpp/include/fv/CellMesh.hpp` (leaves, faces, side lists, adapt, incremental bookkeeping), `CellFlux.hpp` (engine:
stencils, virtual cells, per-level lists, `face_dependencies`, IBM / ghost leaves), `CellAdvance.hpp` (recursion, subcycling,
registers), `CellRemesh.hpp` (indicator, flags, body band, `CellAdaptor::remesh`), `CellParallel.hpp` (partition, halo lists,
exchanges, `gather_state`), `ParallelScan.hpp` (parallel helpers of the remesh), `cpp/src/Simulation.cpp` (`solve_amr`, setup,
`fill_cells`, I/O, monitor). Design history: `docs/design/amr-cell-based.md` (section 18: MPI as built; 20: remesh measurements),
`docs/design/amr-optimisation.md`.

## 1. Where we are

Replicated topology, distributed state (`CellParallel`): every rank holds the whole mesh, faces, stencils and the whole state array;
a leaf's state is *valid* on its owner and its halo. Each level is cut into contiguous Morton blocks of equal count (per-level
balance: the levels advance one after the other). Same-level faces are computed by both owners (same bits), coarse-fine and boundary
faces by the fine owner; registers go to the coarse owner. The remesh gathers the whole state on every rank and repeats indicator,
adapt and engine build there.

Measured consequences (Threadripper, airfoil 720x360, 50 steps, before PR #4; `amr-optimisation.md`): MPI 2 / 4 / 8 / 16 ranks
3.66 / 3.01 / 2.96 / 3.08 s against 5.50 s serial, the replicated remesh 0.98 / 1.18 / 1.56 / 1.99 s of it; beyond 16 ranks the run
gets slower (8.86 s on 48). PR #4 made the remesh OpenMP-parallel (0.18 s on 16 threads), which does not help MPI ranks (one thread
each). And no rank can hold less than the whole mesh.

**What is global on every rank today** (to be removed; sizes for the 720x360 airfoil, 268 k leaves, finest level 2880 x 1440; from
reading the code; measured in section 8.1):

| item | where | size |
|---|---|---|
| leaf arrays, faces, side lists, engine tables, work arrays, spare buffers of PR #4 | `CellMesh`, `CellFluxEngine` | O(N) |
| the state of every leaf, `gather_state` (allgatherv of all leaves before every remesh and every output) | `CellParallel` | O(N) |
| `need_stage` masks in `partition()` | `CellParallel` | R x levels x N bytes: grows with the rank count |
| initial state on the whole uniform mesh of each level | `Simulation::fill_cells` | O(finest uniform): 4.1 M states, 165 MB |
| IBM mask of the whole uniform mesh of every level | `CellFluxEngine::level_mask_` | O(finest uniform) bytes |
| body band (finest-level mask dilated) | `CellAdaptor::band_` | O(finest uniform) bytes |
| ghost-cell table on the whole finest uniform mesh | `CellFluxEngine::ghost_ib_` (+ the mesh at `Simulation.cpp:404`) | O(finest uniform) |
| hot-start interpolant | `Simulation::hot_interp_` | O(source mesh) |
| checkpoint, `dump_state`, `export_state` | gather to rank 0 | O(N) on rank 0 |

## 2. Goal and acceptance

* Each rank stores its owned leaves (the per-level Morton blocks, kept) plus a halo; nothing O(N) or O(finest uniform) per rank
  except what the user accepts for I/O (decision D1).
* The remesh runs on owned data with neighbour exchanges: indicator, flags, ripple, merges, adapt, re-partition, halo rebuild, engine
  build.
* **Bitwise**: every oracle of section 0, plus during the transition a shadow check (the replicated remesh run alongside in the
  tests; section 5).
* **Memory**: peak RSS per rank about (N / R + halo) x bytes per leaf, measured on 1-16 ranks on a case larger than the airfoil
  (decision D4).
* **Time**: the remesh shrinks with the ranks; MPI time stepping not slower than now. *(Expected, not measured: the replicated part,
  1.56 s of the 2.96 s on 8 ranks before PR #4, should drop to the exchange latency plus N / R work.)*

## 3. Design

### 3.1 Identity and indexing

* A leaf is identified globally by `(level, Morton key)`; local indices are per rank. Every list that crosses ranks (halo
  requests, send lists, migration, flags, registers) is keyed, never by global leaf index. The partition is described by per-level
  splitters: rank r owns at level l the keys in `[split[l][r], split[l][r + 1])` (R + 1 keys per level, allgathered), so the owner
  of any leaf follows from a binary search.
* The local mesh of a rank = its owned leaves (all levels) + its halo, sorted by (level, key) like today, flagged owned / halo.
  `CellMesh` gets a "partial" mode: sides of halo leaves that look outside the local set are `Outside` (an error if a stencil of an
  active face reads them). Faces are built for what the engine evaluates (owned leaves' faces as now, plus the halo leaves' sides the
  stencils and gradients need); `face_dependencies` stays the definition of what is read.

### 3.2 Halo

* **Geometric halo**: all leaves (any level) intersecting the dilation of each owned leaf by `H` cells of its own level, plus the
  leaves of the levels around a jump that the virtual cells use (the means of finer leaves, the coarse leaf of an interpolated cell
  and its neighbours). `H` is measured, not guessed (phase 0): MUSCL (2), gradients for viscous cross terms (1 more), interpolated
  jump cells (coarse neighbours), non-reflecting tangential neighbours, the ghost-cell images (up to 6 finest cells for the
  cylinder and the airfoil, `ghost_reach`), mass-flow side leaves (a 1-D set on the inlet side, kept on every rank as now: small).
  **Measured in phase 0** (section 8.2): the reach depends on the level difference between the driver and the leaf read, so the
  geometric halo is defined per level difference d = level(owned) - level(read): a leaf of level t is in the halo if it lies
  within `H[d]` rings of an owned leaf (rings of the owned leaf's level, `CellParallel::ring`). On the airfoil `H` is 5 for the
  same level, 3 for a finer leaf, 6 / 7 for a leaf one / two levels coarser (the virtual and interpolated cells of the jumps
  cascade: the coarse leaf's neighbours, the coarser leaf beyond it and its neighbours), 7 for the ghost images; the random
  four-level test meshes reach 10 (three jumps in a row). `CellParallel::halo_reach` computes the table for any mesh, so the halo
  by region can take `H[d]` from it (or from the pinned bounds of the guard test) instead of a constant.
* Halo by region: a rank computes the key ranges its dilated region covers per level, sends the requests to the owners (from the
  splitters), the owners reply with the leaves (level, key, state). Exchanges per stage / level reuse the existing pattern (send
  lists built once per partition from the requests, `CellComm::exchange`).
* Guard test (kept forever; the time-stepping part is in, `halo_reach_bounds_the_halo_of_any_partition`): on a replicated mesh,
  for every emulated rank, every dependency of every face it evaluates, of the ghost fill, the solid reset, the indicator, the
  ripple and the merge test lies in its geometric halo. The indicator, ripple and merge parts are added with phase 3.

### 3.3 Indicator (distributed)

* The level cells (leaves of a level and split ancestors holding the mean of their four children) are built bottom-up. With
  per-level Morton blocks, the children of a split cell of level l are level-(l+1) cells owned by the level-(l+1) blocks, and the
  four siblings can straddle a rank boundary. Rule: **the rank that owns the first child (in Morton order) forms the mean, with the
  same expression and order as today** `((a + b) + (c + d)) * 0.25`, after receiving the missing children; then split cells
  are distributed like leaves of level l (owner by the level-l splitters, so a parent may move to the level-l owner of its key).
  One exchange per level, finest first.
* Primitives per cell as now; `xi` per level needs the level's neighbours across rank boundaries (halo of level cells, width 1),
  and `diffusion` iterations either exchange `xi` once per iteration (2 exchanges by default) or use a halo widened by
  `diffusion` cells and compute redundantly. *(Choice: exchange per iteration; fewer cells, same bits since every cell is computed by
  its owner.)*
* Result per owned leaf: `xi`, `xi_parent` (the parent cell may be owned elsewhere: exchanged).

### 3.4 Flags, ripple, merges

* Refine flags from `xi` and the body band on owned leaves. **Ripple** (corner 2:1): a refined leaf of level l forces its coarser
  corner neighbours (level l - 1) to refine; the chain drops one level per step, so it ends after at most `max_level` steps. Run a
  fixed number of rounds (`max_level`), each: local propagation to a fixed point, then exchange of the new flags of halo leaves to
  their owners. Every rank makes the same number of collective calls.
* **Merge test** (four siblings, the parent's corner neighbourhood, nobody refined there): needs the final refine flags of the halo
  (one more exchange) and the siblings (possibly on two ranks: decided by the owner of the anchor, as now, reading the sibling flags
  from the halo).
* Shadow check during development: the flags and the new leaf set equal the replicated ones on every remesh (section 5).

### 3.5 Adapt, re-partition, migration

* Each rank applies the flags to its owned leaves: children (injection) of refined leaves, parents (mean of four, the same
  expression as now) of merged groups formed by the anchor's owner from its own and the halo siblings, kept leaves.
* The new leaves of level t come from the owners of level t - 1 (children) and t + 1 (parents), i.e. from the wrong blocks. New
  global positions: per level, the kept leaves are already distributed in Morton order (rank r's kept leaves precede rank r+1's);
  the new leaves are few in steady runs (0.13 % of the leaves per remesh on the airfoil, measured in section 20 of the cell design)
  and can be allgathered as keys only (level, key, 12 bytes), so every rank computes the global position of each of its leaves (kept
  count of lower ranks + local rank + new keys below it) and the new equal-count splitters. Leaves whose owner changes migrate
  point-to-point (key + state; also the per-leaf tables a clean leaf keeps: solid flag, steady source).
* **Large churn** (`init_adapt` from the base mesh, a restart on another rank count, a remesh after a shock enters): allgathering
  the new keys is O(new leaves) per rank. Acceptable for the sizes in sight *(estimate: 12 bytes x new leaves; 3 MB for a full
  268 k remesh)*; a sample sort replaces it if the user wants meshes where that is too much (decision D3).
* After migration: halo by region (3.2), local topology (3.1), engine build. The incremental update of PR #4 / option 1 (`prev_leaf`,
  `topo_dirty`) carries over per rank with local indices (phase 6).

### 3.6 Physics and the rest

* IBM: the solid flag of a leaf is the centre-inside test of the body at the leaf's own level (the same function as the uniform mask,
  per point instead of per uniform mesh: same bits); the body band per leaf from the finest-level cells of the leaf's footprint
  dilated by the margin (same test). The ghost-cell table only for the finest-level cells in the owned region plus reach. This
  removes `level_mask_`, `band_` and the whole-mesh ghost table.
* `fill_cells`: evaluate the initial condition per leaf (the same per-cell expression; check the bits against the per-level mesh
  path: the initial state is part of the references).
* Mass-flow inlet: its side leaves on every rank as now (1-D).
* Wave-speed bound, maxima: allreduce max (order independent), as now.
* **Monitor sums** (volume averages, loads): today root sums over the gathered leaves; distributed, they become per-rank sums plus
  an allreduce: equal to roundoff across rank counts, not bitwise (as the uniform MPI path already is; `ibm_ghost.py --check`
  compares loads with rtol 1e-10). The states remain the bitwise oracle. Decision D2.
* I/O: VTK is already one `.vtu` per rank (`.pvtu`). Checkpoint, `dump_state`, `export_state` gather to rank 0 today. Decision D1.

### 3.7 Dimension-agnostic layer (towards 3-D)

Key arithmetic (parent, children, the 8 (2-D) / 26 (3-D) neighbour offsets, the key range of a region per level), the splitters,
migration, halo-by-region and the ripple rounds go behind a small layer templated on the dimension, used by the 2-D `CellMesh` /
`CellFluxEngine` (which stay 2-D). Note for later: a 64-bit 3-D Morton key leaves 21 bits per axis (2 M cells per axis at the finest
level) against the 29 bits per axis `kKeyBits` allows in 2-D.

## 4. Phases (each mergeable with all oracles green)

| phase | content | touches | size *(estimate)* |
|---|---|---|---|
| 0 | Measurements: peak RSS per rank and remesh / stepping times on 1-16 ranks for the airfoil and a larger case; the halo reach test on replicated meshes (sets H) | a study script, a unit test | small |
| 1 | Keyed lists: halo and send lists, registers and solid exchanges by (level, key); `partition()` O(local) instead of the R x levels x N masks | `CellParallel` | medium |
| 2 | Local storage for time stepping: after the (still replicated) remesh each rank keeps only its owned leaves + halo, rebuilds a partial local mesh and engine; the global mesh is dropped until the next remesh (which gathers as now) | `CellMesh` partial mode, `CellFluxEngine`, `CellParallel`, `Simulation` | large |
| 3 | Distributed indicator and flags (ripple rounds, merge test), checked against the replicated flags in the tests | `CellRemesh`, `CellParallel` | large |
| 4 | Distributed adapt, re-partition, migration, halo by region: the global mesh is gone | `CellMesh`, `CellParallel`, `Simulation` | large |
| 5 | Local IBM mask, body band, ghost table, `fill_cells`, hot start; I/O per decision D1 | `CellFlux`, `CellRemesh`, `Physics`, `Simulation`, `IO` | medium |
| 6 | Incremental local rebuild, performance, the timings on both machines | all of the above | medium |

Phase 2 already distributes the memory of the time stepping (the largest arrays) while the remesh is still replicated; phases 3-4
remove the replicated remesh. MPI == serial is checked after every phase with the existing tests; phases 2-4 add the shadow checks.

## 5. Development oracle (shadow)

While phases 2-4 are in progress, the `FakeHub` tests keep a replicated mesh alongside: after every remesh the distributed run's leaf
keys, flags, per-rank owned blocks and engine tables (gathered, by key) must equal the replicated ones. A runtime switch
(`FENICE_REPLICATED_REMESH=1`, like `FENICE_FULL_REBUILD`) keeps the old path for A/B and bisecting until phase 6; then it is
retired (decision D5).

## 6. Risks

* Partial-mesh topology: a stencil reaching outside the local set at a corner case (level jump at a rank boundary next to a
  periodic seam or a body). Mitigation: the guard test of 3.2 on random meshes and every check case, `Outside` sides that throw.
* Split cells straddling ranks in the indicator: the mean must be formed from the same four values in the same order.
* Collective-call symmetry (every rank in every exchange, also ranks that own nothing at a level), as today.
* Memory of the transition (phase 2 keeps the replicated remesh, so the peak at a remesh is still O(N) per rank until phase 4).
* Effort: phases 2-4 are a rewrite of the indexing of three classes *(estimate: the largest piece of work since P6)*.

## 7. Decisions (answered by the user, 2026-10-24)

* **D1 I/O**: "do what is best; resume on the same number of ranks is enough". Chosen: the checkpoint becomes per-rank files
  (each rank writes its owned leaves by key with their state, plus the shared header: t, step, wave-speed bound, rank count);
  a resume reads them on the same rank count and refuses another one with a clear message (today's resume on another rank count
  is dropped). `dump_state` (the test / validation dump) keeps a gather to rank 0, streamed rank by rank: it is opt-in, and the
  regression and golden oracles compare it. VTK stays one `.vtu` per rank.
* **D2 monitor sums**: per-rank sums plus an allreduce (the user, after phase 0). Each rank sums the volume terms and the loads
  of its owned leaves in leaf order, one allreduce (sum) per monitor row: no O(N) on root. The monitor values then equal the
  serial ones to roundoff, not bit for bit, across rank counts (as on the uniform MPI path); the states stay the bitwise oracle,
  and the checks that compare monitor columns or loads across rank counts use a relative tolerance (`ibm_ghost.py --check`
  uses rtol 1e-10). Needs a sum-allreduce in `CellComm`.
* **D3 large churn**: "what is best for computational time". Chosen: allgather of the new keys (12 bytes each; 0.13 % of the
  leaves per remesh in steady runs, 3 MB for a full 268 k remesh): one collective, no extra rounds. A sample sort only if a
  measurement shows the allgather to matter (it would at init on meshes much larger than 1 M leaves).
* **D4 target case**: the airfoil: 720 x 360 and 1440 x 720 (two levels), `amr_memory.py`.
* **D5** the replicated path: kept as `FENICE_REPLICATED_REMESH=1` during phases 2-5 for A/B and bisecting, removed after the
  last phase once everything works.
* Scope check: per-level partition kept (the subtree alternative measured ~1.5x slower stepping on 8 ranks, section 20).

## 8. Phase 0 results

### 8.1 Memory and time (serial, measured in the container)

The solver now prints at the end of an AMR run the resident memory of every rank (min / max over the ranks: after setup, after
the initial adaptation, at the end, the peak) and the bytes of the tables of rank 0 (container capacities). Serial runs, one
thread, 50 steps, cloud container (Linux, gcc; times only comparable with each other):

| airfoil | leaves | RSS setup | after init | end | peak | mesh | engine | uniform-mesh arrays | body band | wall | remesh (indicator / adapt / build) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 720 x 360 | 268 k | 41 MB | 276 MB | 396 MB | 458 MB | 107 MB | 321 MB | 5 MB | 4 MB | 11.0 s | 1.59 s (0.51 / 0.49 / 0.36) |
| 1440 x 720 | 1.06 M | 145 MB | 977 MB | 1352 MB | 1609 MB | 427 MB | 1280 MB | 21 MB | 16 MB | 44.1 s | 7.73 s (2.32 / 3.57 / 1.44) |

* About **1.5 kB per leaf** at the end, every rank today; the peak (+15-20 %) is the initial adaptation (`fill_cells` evaluates
  the initial condition on the whole uniform mesh of every level: 4.1 M / 16.6 M states at the finest level) and the remesh
  temporaries. The engine is three quarters of it: per face `F_`, `Fv_`, `reg_`, `regv_` (40 bytes each, 1.04 M faces at 720:
  160 MB; `reg_` / `regv_` are needed for coarse-fine faces only, `Fv_` only when viscous), `W_`, `forced_`, `src_` (allocated
  without sources), the stencils and their spare copies of PR #4. Phase 2 makes all of it local; the per-face registers sized for
  every face are a cheap saving to take there (bits unchanged).
* The O(finest uniform) arrays are small at two levels (5 + 4 MB at 720: masks are bytes) but grow by 4 per level; the
  `fill_cells` transient (40 bytes per finest cell) is the largest of them.
* MPI (the user's machines): `python cases/validation/amr/amr_memory.py` runs both meshes on 1, 2, 4, 8, 16 ranks; every rank is
  expected at the serial figures (replicated) plus the `partition()` masks (R x levels x N bytes: 13 MB at 720 on 16 ranks).
  The 1440 mesh needs about 1.6 GB per rank at the peak, 26 GB on 16 ranks: use `--ranks 1 2 4 8` or `--only airfoil_720` on
  a machine with less memory. *(Results to be added here.)*

### 8.2 Halo reach (measured, independent of the partition)

`CellParallel::halo_reach(mesh, engine)` measures every read of the time stepping from the leaf whose owner makes it (both leaves
of a same-level face, the fine leaf of a coarse-fine face, the leaf of a boundary face, the solid leaf of the solid reset; for
the ghost images the driver of the finest-level face that reads the ghost leaf), in rings of the driver's level (ring 1 = touching,
even at a corner; periodic axes wrap), per level difference. `FENICE_HALO_REACH=1` prints it for the initial and the final mesh.

| reads (720 x 360 airfoil, 2 levels) | finer by 2 | finer by 1 | same level | coarser by 1 | coarser by 2 |
|---|---|---|---|---|---|
| same-level faces | 2 | 3 | 5 | 4 | 4 |
| coarse-fine faces (from the fine leaf) | - | 3 | 4 | 6 | 7 |
| boundary faces | - | - | 1 | - | - |
| ghost images (`ibm_method = ghost`) | - | - | 7 | - | - |
| solid reset (ghost: images from the ghost leaf) | - | - | 1 (5) | - | - |

The same on 1440 x 720. 99.9 % of the reads are within ring 2 (reads per ring at 720: 3.2 M / 9.7 M / 2.2 M / 16 k / 2.5 k / 120 /
12 / 30 for rings 0-7); the far ones are the cascades of virtual and interpolated cells at the level jumps. The random four-level
meshes of the unit test reach 10 for coarse-fine faces (three jumps in a row) and 4 for boundary faces (a boundary leaf's
neighbours through an interpolated cell); the test pins interior 5, coarse-fine 10, boundary 4, ghost images 7, solid reset 5, and
checks on six physics configurations and 1, 2, 3, 5, 7, 16 emulated ranks that the real halo of `partition()` lies within the
measured reach (the mass-flow side leaves apart, read on every rank).

Reach of the remesh (from the code, for phase 3; checked by the guard test then):

* Indicator: one ring of level cells per pass (the face test, then each of the `amr_diffusion` smoothing iterations), plus the
  coarser leaf across a side; with an exchange of `xi` per iteration the halo of level cells is one ring wide.
* Ripple: the eight corner neighbours (ring 1 of the refined leaf's level), up to `max_level` rounds.
* Merge test: the parent's eight corner neighbour blocks at the parent level: ring 3 of the anchor's level from the anchor (2
  from the sibling block), and the final refine flags there.
