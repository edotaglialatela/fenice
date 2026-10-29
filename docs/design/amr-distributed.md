# AMR with distributed storage and a local remesh (plan)

Status: **approved (D1-D5 answered, section 7); phases 0 (section 8), 1 (section 9), 2 (section 10) 3 (section 11), 4 and 5 (sections 12 and 13; the topology of the whole mesh is kept on every rank, without faces) done.** Branch `amr-distributed`, from `main`
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
| 2 | Local storage for time stepping: after the (still replicated) remesh each rank keeps only its owned leaves + halo (**done by extraction**: the whole-mesh tables are re-indexed to the held leaves, section 10; the direct build of a partial mesh and engine, with the by-key shadow comparison of section 5, moves to phase 4, which needs it); the global mesh is dropped until the next remesh (which gathers as now) | `CellMesh` partial mode, `CellFluxEngine`, `CellParallel`, `Simulation` | large |
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
  a machine with less memory.

Measured by the user (`amr_memory.py`, MSVC build on Windows, 50 steps; RSS min-max over the ranks, which agree within 1 MB):

| airfoil | ranks | wall | remesh | gather / indicator / adapt / build / partition | RSS after init | end | peak |
|---|---|---|---|---|---|---|---|
| 720 x 360 | 1 | 5.25 s | 0.79 s | 0 / 0.29 / 0.23 / 0.14 / 0 | 222 MB | 296 MB | 387 MB |
| | 2 | 4.10 s | 1.13 s | 0.05 / 0.35 / 0.29 / 0.18 / 0.24 | 228 MB | 287 MB | 380 MB |
| | 4 | 2.99 s | 1.14 s | 0.05 / 0.35 / 0.32 / 0.21 / 0.24 | 225 MB | 277 MB | 381 MB |
| | 8 | 3.65 s | 1.79 s | 0.12 / 0.58 / 0.63 / 0.36 / 0.37 | 224 MB | 273 MB | 381 MB |
| | 16 | 6.09 s | 3.41 s | 0.35 / 1.26 / 1.45 / 0.59 / 0.56 | 224 MB | 273 MB | 381 MB |
| 1440 x 720 | 1 | 20.5 s | 3.13 s | 0 / 1.08 / 0.89 / 0.55 / 0 | 853 MB | 1142 MB | 1509 MB |
| | 2 | 14.1 s | 3.77 s | 0.20 / 1.17 / 1.00 / 0.60 / 0.79 | 877 MB | 1107 MB | 1487 MB |
| | 4 | 11.9 s | 4.43 s | 0.28 / 1.55 / 1.39 / 0.78 / 0.87 | 865 MB | 1066 MB | 1487 MB |
| | 8 | 14.5 s | 7.04 s | 0.50 / 2.53 / 2.38 / 1.44 / 1.29 | 860 MB | 1047 MB | 1487 MB |
| | 16 | 23.9 s | 13.9 s | 1.08 / 4.84 / 5.76 / 2.21 / 2.32 | 860 MB | 1040 MB | 1487 MB |

* The memory per rank does not fall with the ranks at all: the peak is the same on 1 and 16 ranks (the stepping arrays shrink
  a little, the replicated mesh and engine do not). Tables of a rank with MSVC: mesh 87 / 346 MB, engine 242 / 967 MB.
* The replicated remesh does not just stay constant: indicator, adapt and build get slower with the ranks (2.5-5x from 1 to 16
  ranks: every rank does the same memory-bound work at the same time, sharing the memory bandwidth), and `partition()` (the
  R x levels x N masks) grows from 0.24 to 0.56 s (720) and 0.79 to 2.32 s (1440). Remesh is 56-58 % of the 16-rank runs.
  The best rank count is 4 for both meshes.

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

## 9. Phase 1 (done): keyed lists, partition without the R x levels x N masks

* Ownership is described by per-level splitters (`CellParallel::owner_of(level, key)`: the first Morton key of each rank's block,
  R + 1 per level; computed locally while the mesh is replicated, allgathered once the blocks move in phase 4).
* `partition()` works out only this rank's reads: the faces it computes (`face_dependencies`), the ghost images, the mass-flow
  side leaves, the solid reset of its own solid leaves; two O(N) byte masks (held, read in a stage of its own level) instead of
  R x (levels + 1) x N. The receive lists follow from them; the leaves owned elsewhere go to their owners as (2 level + stage
  flag, i, j) requests (one exchange of the counts, one of the requests), and the owners build the send lists from the requests
  they get (stage, final, solid). Order of every list: the leaf order, i.e. (level, key), on both sides.
* Registers: both ranks list a coarse-fine face by (fine leaf, its side) from their own leaves: the fine owner from its fine
  leaves' coarser sides, the coarse owner from its coarse leaves' finer sides, sorted by (fine leaf, side). No face index crosses.
* Checked: during development the old algorithm ran alongside in every unit test (1-16 emulated ranks, all physics, ghost IBM)
  and every list, the held set and the owners were identical; the bitwise `FakeHub` tests, the halo guard (now on threaded ranks,
  since `partition()` communicates), regression, golden dumps and CI (`amr_mpi.py --check`, `ibm_ghost.py --check`, 1-7 real
  ranks) are green.
* Still O(N) per rank while the mesh is replicated: `owner_` and `held_` (a cache over the replicated leaves), the two masks,
  `ob_` / `oe_` (used by `gather_state`). Phase 2 makes them local.
* **Measured by the user after phase 1** (`amr_memory.py --force`, same machine as 8.1): `partition()` did **not** get
  faster: 0.24 / 0.23 / 0.31 / 0.72 s on 2 / 4 / 8 / 16 ranks (720), 0.79 / 0.83 / 1.17 / 2.92 s (1440), against 0.24 / 0.24 /
  0.37 / 0.56 and 0.79 / 0.87 / 1.29 / 2.32 s before (run-to-run noise is about 10-20 % at 16 ranks; wall 5.74 / 24.2 s on 16
  ranks against 6.09 / 23.9). So the R x levels x N masks were not what made it grow: what remains is O(N) work repeated on every
  rank (the owner of every leaf, the masks, the loop over every face, `CellFluxEngine::set_partition` over the whole mesh), and
  like the indicator / adapt / build it slows down 2.5-4x from 2 to 16 ranks as the ranks share the memory bandwidth. Phase 1
  removed the R factor and every global index between ranks (needed for phase 2), not the O(N) per rank; that goes with the
  replicated mesh in phases 2 and 4. Memory unchanged, as expected.

## 10. Phase 2 (done): local storage for the time stepping

Commits: 2a registers / `Fv_` saving, 2b the monitor without a gather (D2), 2c localize.

* **2a** `reg_` / `regv_` hold one entry per coarse-fine face (`reg_slot_`, `reg_level_begin_`), `Fv_` exists only with viscous terms.
  Bits unchanged (golden dumps, regression).
* **2b (D2)** `CellComm::allreduce_sum` (MPI and `FakeHub`). The monitor row is the sum over the owned leaves in leaf order, one
  allreduce per value: one rank gives the serial bits (the golden monitors are identical), several ranks agree to roundoff. The
  loads of a partitioned engine come from the level passes (`compute_level_fluxes`, the coarse-fine fluxes handed to the coarse owners
  as registers of weight 1, `gather_level`), not from a full residual; `Simulation::cell_loads` sums the owned solid leaves.
* **2c** After the (still replicated) remesh and `partition()`, `CellParallel::localize()` keeps what the rank holds:
  * `CellMesh::keep_partial(keep_leaf, keep_face)` drops the other leaves and faces in place (same order, so the (level, key) order and the
    level runs stay) and returns the maps. A face keeps `kOutside` (-2) for a leaf that is gone; a side whose faces are not all kept has an
    empty gather list and `neighbour()` says `Outside`. Kept faces: those the rank computes, every face of an owned leaf (the gather, the
    registers) and the mass-flow faces.
  * `CellFluxEngine::localize` re-indexes every table of the build and of the partition (stencils, neighbour codes, solid flags,
    steady sources, mass-flow faces, per-level lists, ghost tables) with the maps. A virtual cell stays if every leaf of its tree is kept
    (an interpolated one also if the cells its slopes read are); any code that points elsewhere is `kOutside`. It then runs
    `face_dependencies` on every face the rank computes and **throws** if one reads `kOutside` or a leaf that is not kept
    (`a_partial_mesh_throws_when_a_face_reads_a_leaf_it_does_not_keep` shrinks the halo by one leaf and expects it). Nothing is
    recomputed, so the bits hold by construction.
  * `CellParallel` re-indexes its lists (leaf indices by the leaf map, register lists by the face map); `level_present(l)` keeps the
    whole mesh's answer to "does level l have leaves", which `CellAdvance` and the monitor use: a rank may hold no leaf of a level that
    others have and must still make the collective calls of that level (this hung the hot-start regression on 4 ranks before).
  * The remesh calls `globalize()` first: the owned leaves of all ranks (level, i, j, state: 8 doubles) are allgathered and
    `assign_leaves` rebuilds the whole mesh; then `remesh()` (which builds the engine in full, the old incremental base is gone),
    `partition()`, `localize()`. If the remesh changed nothing the engine is built anyway (its tables were freed).
  * Checkpoint, `dump_state`, the leaf counts and the final report go through `CellParallel::collect` (allgather, sorted by (level, key);
    the file formats are unchanged). `FENICE_REPLICATED_REMESH=1` skips `localize()` (D5).
  * `Simulation::solve_amr` takes the wave-speed bound after a remesh from `force_state` + `wave_speed` on both paths; the golden dumps
    stay bitwise (the old code used the forced state of the old indexing, which the maximum did not depend on).
* **Checks**: unit tests (new: `cellparallel_localized_run_equals_serial_bitwise_and_keeps_only_its_halo`, 1-7 emulated ranks, every
  physics, remesh included; `a_partial_mesh_throws...`), regression serial and 4 ranks ALL MATCH, golden dumps ALL BITWISE, real MPI
  (`amr_mpi.py --check`, `ibm_ghost.py --check`) 1-7 ranks bitwise.
* **Measured** (this machine, Ryzen 7 7800X3D, MSVC, airfoil 720 x 360, 50 steps, `amr_memory.py --force`; before phase 2 in section 8.1):

| ranks | wall [s] | remesh [s] (gather / indicator / adapt / build / partition) | RSS after init / end [MB] | peak [MB] |
|---|---|---|---|---|
| 1 (serial) | 5.06 | 0.80 (0.00 / 0.28 / 0.23 / 0.14 / 0.00) | 184 / 257 | 349 |
| 2 | 4.99 | 2.31 (0.53 / 0.34 / 0.32 / 0.47 / 0.62) | 101 / 133 | 342 |
| 4 | 3.71 | 2.23 (0.59 / 0.37 / 0.39 / 0.53 / 0.43) | 74 / 91 | 342 |
| 8 | 4.25 | 3.07 (0.84 / 0.55 / 0.58 / 0.72 / 0.57) | 60 / 70 | 342 |
| 16 | 7.53 | 5.63 (1.71 / 1.06 / 1.35 / 1.25 / 0.86) | 54 / 60 | 342 |

  Steady memory per rank now falls with the ranks (end of the run 257 MB serial, 133 / 91 / 70 / 60 MB on 2 / 4 / 8 / 16; the
  replicated storage was 239 MB on 4 ranks). The **peak does not**: it is the remesh, where every rank still holds the whole mesh and
  engine (phases 3-4 remove it). **The wall time got worse** (4 ranks 3.71 s against 2.89 s, 16 ranks 7.53 against 5.74): every remesh
  now rebuilds the whole mesh and does a full engine build on every rank (the incremental rebuild of PR #4 needs the tables that
  `localize` frees), so the "gather" and "build" columns grow. This is the expected price of phase 2 and goes with phases 3-4 (the
  remesh on owned data) and 6 (the incremental local rebuild); `FENICE_REPLICATED_REMESH=1` is the fast path until then.

**Not done in phase 2, and why.** The plan said each rank *builds* a partial mesh and engine; phase 2 *extracts* them from the whole-mesh
build instead (`keep_partial`, `localize`): nothing is recomputed, so the bits hold by construction and the plumbing (`Simulation`,
`CellAdvance`, `CellParallel`) could be debugged against an independent oracle. The direct build (lookups with three outcomes: leaf /
subdivided / not held, an engine build that tolerates `Outside`) is what phase 4 needs once the whole mesh is gone; the by-key shadow
comparison of section 5 (its tables against these extracted ones) belongs there, and is cheap to add then because the extraction is the
reference. What phase 2 checks instead: bitwise states on 1-16 emulated ranks and 1-7 real ranks (every physics, remesh included),
the validation in `CellFluxEngine::localize` (throws, with a negative test), and a resume on 4 ranks equal to a straight run
(30 steps = 15 + checkpoint + 15, `dump_state` bit for bit).

**Decided by the user (2026-10-29).** CI is green; `localize` stays the default (`FENICE_REPLICATED_REMESH=1` is the A/B path). The wave-speed
bound after a remesh is taken from the freshly forced state on every path (`force_state` + `wave_speed`): bit-identical on every oracle
(golden, regression, MPI) and it makes the serial and the localized runs use the same rule.

## 11. Phase 3 (done, as built): the indicator, the flags and the state of the new leaves are local

* **Built** (`CellParallel::remesh(topo, adaptor)`, for a localized run; the replicated path is unchanged):
  * Every rank keeps `topo`, the topology of the whole mesh (leaves, faces, no state). It is what makes the ripple, the merge test and
    the new leaf list identical on every rank without exchanging them: the flags of all ranks (one number per leaf, an allgatherv of
    the owned leaves in (level, key) order) are applied by `CellMesh::adapt` on `topo` on every rank, and `CellMesh::source` (set with
    `record_sources`) says for every new leaf which old leaves its state comes from (a kept leaf, the parent of a child, the four
    siblings of a merged group).
  * Indicator: `CellIndicator::region` lists the leaves the xi and xi_parent of the owned leaves read: the leaves of the cells within
    `diffusion + 1` cells of the leaf at its level and of its parent at the parent level (the face test of a cell reads its
    neighbours, each smoothing iteration the neighbours' xi), plus the leaves under a split cell and the coarser leaf covering a cell.
    A bitmap per level dilates the cells of the owned leaves, so the cost is that of the owned leaves and their margin. The state of
    the leaves a rank does not hold comes from their owners (`fetch_states`: two exchanges); `CellIndicator::evaluate` runs on that
    partial mesh (a split cell of an incomplete sibling group has no parent cell: its values are not used) and `CellAdaptor::flags`
    takes the decisions of `remesh()` (the same `decide`).
  * State: the new owned leaves (the equal-count blocks of the new mesh) read their old leaves (fetched where not held) and build
    injection or the mean `((a + b) + (c + d)) * 0.25`, the expression of `adapt`. Then `mesh_ = topo`, the full engine build,
    `partition()`, `localize()` and `exchange_final` for every level (the halo state). If the flags change nothing, nothing else runs
    (before, every remesh globalized and rebuilt).
* **Checks**: `local_indicator_equals_the_whole_mesh_indicator_for_any_partition` (xi and xi_parent of every owned leaf, 1-16 partitions,
  four periodicities, three smoothing depths; with the region one cell smaller it fails: 146 checks), `cellparallel_distributed_remesh_equals_serial_bitwise`
  (four physics, 1-16 emulated ranks; fails with the refine flags dropped), regression serial and 4 ranks, `amr_mpi.py --check` and
  `ibm_ghost.py --check` 1-7 ranks, golden dumps (the entries both dumps have are bitwise).
* **Measured** (same machine and script as section 10; wall [s] / remesh [s] (gather = region and fetch, indicator = flags, adapt =
  topology adapt and migration, build, partition = partition, localize, halo) / RSS at the end [MB]):

| ranks | wall | remesh (gather / indicator / adapt / build / partition) | RSS end |
|---|---|---|---|
| 1 | 5.00 | 0.78 (0.00 / 0.28 / 0.22 / 0.13 / 0.00) | 256 |
| 2 | 3.99 | 1.69 (0.10 / 0.13 / 0.29 / 0.45 / 0.53) | 229 |
| 4 | 2.93 | 1.57 (0.09 / 0.09 / 0.33 / 0.54 / 0.42) | 184 |
| 8 | 3.38 | 2.26 (0.10 / 0.14 / 0.57 / 0.85 / 0.49) | 160 |
| 16 | 5.32 | 4.07 (0.17 / 0.13 / 1.16 / 1.67 / 0.85) | 151 |

  Against phase 2 (section 10): 4 ranks 3.71 -> 2.93 s, 16 ranks 7.53 -> 5.32 s; gather plus indicator 0.96 -> 0.18 s on 4 ranks.
  Per remesh on 4 ranks (rank 0): copy of `topo` 12 ms, engine build 41 ms, partition 17 ms, localize 23 ms, topology adapt and migration
  31 ms, region and indicator about 14 ms. What is left is O(N) work on every rank (adapt, copy, build, partition, localize), which is
  why it grows from 4 to 16 ranks (the ranks share the memory bandwidth), and the whole-mesh `topo` (87 MB) costs memory: the steady
  RSS per rank is 184 MB on 4 ranks (91 in phase 2), the peak is unchanged (342 MB).
* **Left for phase 4**: the whole-mesh topology, its adapt (ripple and merges are cheap, the face list update is not), the full
  engine build on a mesh the rank then mostly drops, and the partition over all faces are the remaining replicated work. Plan: build
  the partial mesh directly from the leaves of the geometric halo (rings per level difference from `halo_reach`, section 3.2), make
  `CellFluxEngine::build` tolerate `Outside` on the part of the mesh nobody reads, and take the exact halo from `partition()` on that
  mesh. Then `topo` can shrink to the keys (or go), the ripple and merge test use the halo flags (section 3.4), and the memory and the
  remesh time fall with the rank count.

## 12. Phase 4: the share built directly (done)

* `CellParallel::region_mask`: from the owned leaves of the new topology, the leaves within a ring count per level difference (4 rings for a
  coarser owner, 6 same level, 8 / 9 / 11 for finer ones, more with a body, and the leaves along a mass-flow inlet) plus `extra` rings.
  `CellMesh::build_from` makes the partial mesh (leaves, faces made by the kept leaves with `kOutside` at the far end, a side without all
  its faces empty) from the whole topology; `CellFluxEngine::build(mesh, &whole)` tolerates `Outside` (incomplete virtual cells are
  dropped, `ghost_ready_` is a fact of the whole mesh); `partition_local(&whole)` returns false, before anything is communicated, if a face
  an owned leaf computes reads what the rank does not keep. All ranks then retry with `extra += 2` (an allreduce), up to 15.
* The topology `topo` is faceless (`CellMesh::set_faceless`): `adapt` does no face work and `topo` holds only leaves and lookups.
* Test: with `CellParallel::shadow` every remesh of `cellparallel_distributed_remesh_equals_serial_bitwise` also builds the share the old
  way (whole mesh and engine, then `localize`) and throws unless mesh, engine tables (`same_tables`, now including the ghost maps and the
  masks; `special_` of leaves with an unknown side is not compared) and lists are the same. Specs added: ghost-cell body, mass-flow inlet,
  unsteady source.
* Airfoil 720x360, 50 steps (wall / remesh [s], RSS at end [MB]): 2 ranks 4.09 / 1.80 / 184, 4 ranks 2.57 / 1.20 / 139, 8 ranks
  2.33 / 1.19 / 116, 16 ranks 2.86 / 1.57 / 105 (phase 3: 3.99, 2.93, 3.38, 5.32 s for 2, 4, 8, 16 ranks). The peak (342 MB) is unchanged:
  the state is still gathered once at a checkpoint and the setup builds the whole mesh.
* Left: the topology adapt (0.65 s at 16 ranks) and the engine build on the share are still O(share) plus O(N) lookups on every rank; the
  topology could shrink to keys, and the state of the initial mesh is still distributed from a whole mesh.

## 13. Phase 5: the first mesh without a whole mesh, windowed initial condition (done)

* The peak memory per rank was 342 MB at every rank count: the initial adaptation ran on the whole mesh and engine on every rank (`init_adapt`),
  and `fill_cells` evaluated the initial condition on the whole uniform field of each level (166 MB on the finest level of the airfoil).
* `Mesh::win_i0 / win_i1 / win_j0 / win_j1`: an initial-condition evaluation covers a window of cells (the 2-D profiles and
  `evaluate_initial`), indexed relative to it, with the coordinates of the whole mesh, so the bits are the same. `fill_cells` evaluates the box
  that holds a level's leaves: serial peak 358 -> 242 MB during the initial adaptation, golden dumps unchanged.
* `CellParallel::init_adapt(topo, ad, fill)` (several ranks, localized storage): `topo` is the faceless base mesh (refined to
  `amr_init_levels`), every rank builds its share of it with `build_share` (the code of the remesh, factored out) and the passes of
  `CellAdaptor::init_adapt` run through `remesh()` with the initial condition filled on all the leaves of the share (the state the remesh
  migrates is overwritten, as it is in the serial passes). The solid reset follows with `exchange_solid`. A resume still reads the whole
  checkpoint, then localizes.
* Test: `cellparallel_distributed_init_equals_serial_init_bitwise` (leaves, passes and state bit for bit, 1-8 emulated ranks, with a body,
  ghost cells and a mass-flow inlet, first reach too small to force the retry); `amr_mpi.py --check` and `ibm_ghost.py --check` use it on
  real MPI.
* Airfoil 720x360, 50 steps (wall [s], RSS at the end / peak [MB]): 2 ranks 4.14, 171 / 227; 4 ranks 2.56, 121 / 158; 8 ranks 2.25, 95 / 128;
  16 ranks 2.62, 83 / 116 (serial 5.03, 257 / 348). Split of the share build on 4 ranks (50 steps plus the initial passes): region 0.19 s,
  partial mesh 0.10, engine 0.14, partition 0.15, localize 0.20, all O(share).
* Checkpoint: `CellParallel::collect(..., root_only)` sends the records to rank 0 only (`exchange`), the only rank that writes; with a
  checkpoint every 10 steps the peak was 172-176 / 129-137 MB (4 / 16 ranks) on every rank with the gather to all, now 159-176 / 115-137
  (rank 0 holds the state). The timers of the remesh are reset after the initial passes. Real MPI: `amr_mpi.py --check` has a case with
  `amr_init_levels=1`; a hot start (uniform checkpoint mapped on the airfoil, 20 178 leaves) gave the serial bits on 1, 2, 4, 5 ranks.
  `FENICE_HALO_REACH=1` needs the whole mesh (serial or `FENICE_REPLICATED_REMESH=1`); on a distributed start it says so.
* Left: the topology adapt (0.13 s on 4 ranks, 0.50 s on 16: the same O(N) work, slower with the cores shared), which would need a
  distributed topology; an incremental share build (reuse of the previous local mesh and engine); gather to one rank for the checkpoint.

## 14. Phase 6: the topology adapt without a whole-mesh topology (plan, in progress)

Goal: no rank holds the whole topology (`topo`) any more; the remesh costs and the memory per rank fall with the rank count.

**Whole-mesh dependencies of `remesh(topo, ad)` and what replaces them**

| dependency | replacement |
|---|---|
| `owned_run`, `owned_leaves` (equal-count run of `topo.level_begin`) | the owned leaves of the share (`owner_ == me`) |
| `CellIndicator::region(topo, owned)`, `fetch_states` | a fetched work set W of old leaves with states (`fetch_leaves`), same region |
| flags allgather (one number per leaf of the whole mesh) | flags of the owned leaves; the flags / R of the margin leaves of W from their owners along the fetch lists |
| `topo.adapt` (ripple, merges), `topo.source`, `nU(N)` | the same rules on W: ripple rounds with an exchange of R and an allreduce of "changed", merges evaluated by every rank for the groups touching its owned leaves, new leaves built by their creator with the injection / `((a+b)+(c+d))*0.25` expressions |
| `partition_local`: `level_begin` runs, `leaf_key(g, s0)`, `level_present_`, `ob_`/`oe_` | `GlobalFacts`: per-level counts and the splitters, found by a distributed bisection on the key (the k-th smallest key per level and boundary, exact), `level_present_` from the counts |
| `region_mask(topo)`, `build_from(topo)`, `engine_.build(mesh, &topo)` | the same functions on a fetched local set T (the leaves of the region and of the mass-flow sides); `ghost_ready` by a collective count of the referenced finest-level cells |
| the new ownership | new splitters, then the new leaves and their states migrate to their owners (count handshake and `exchange`) |
| `init_adapt` | builds the first share from a transient base topology (setup only), frees it, then the passes use the distributed remesh |

**Primitive** `fetch_leaves`: the leaves of the current mesh that intersect a region (items: level, box of cells, range of leaf levels), answered by the
owners (routed by the splitters; per level, coarse to fine, a cell covered by a leaf already found is not asked again), with their states if
wanted; it records the lists of leaves exchanged with each rank (the mirror lists carry the flags later). The indicator region, the
adapt neighbourhood W (rings of the leaf's level: 3, the merge test reaches the parent's neighbours) and the input of `build_share`
all use it.

**Stages** (each committed, the oracles after each): 1 the primitive and its test against `topo`; 2 `GlobalFacts` and the distributed
splitters (equal to `partition()` on the whole mesh); 3 `build_share` from a fetched T; 4 the distributed adapt, compared with
`topo.adapt` at every remesh in the tests (shadow topology, leaves only; the states by the serial-versus-ranks bitwise test); 5 `topo`
dropped from `Simulation`; `FENICE_REPLICATED_TOPOLOGY=1` keeps the old remesh for comparison.
