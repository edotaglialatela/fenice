# AMR with distributed storage and a local remesh (plan)

Status: **plan, awaiting the user's approval; no code yet.** Branch `amr-distributed`, from `main` at 883c45d (both PRs #4 / #5
merged). This is option 2 of `docs/design/amr-optimisation.md`, as a step towards 3-D octree AMR. Every number below that was not
measured is marked *(estimate)*.

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
reading the code, to be confirmed by the measurements of phase 0):

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
  *(Estimate: H = 3 at the leaf's level plus the ghost reach near a body.)*
* Halo by region: a rank computes the key ranges its dilated region covers per level, sends the requests to the owners (from the
  splitters), the owners reply with the leaves (level, key, state). Exchanges per stage / level reuse the existing pattern (send
  lists built once per partition from the requests, `CellComm::exchange`).
* Guard test (kept forever): on a replicated mesh, for every emulated rank, every dependency of every face it evaluates, of the
  ghost fill, the solid reset, the indicator, the ripple and the merge test lies in its geometric halo.

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

## 7. Decisions for the user

* **D1 I/O**: may checkpoint / `dump_state` / `export_state` keep a gather to rank 0 (simple; rank 0 then needs O(N) memory at
  output time), or per-rank files (a restart reads them by key, any rank count)? Restart on another rank count: required?
* **D2 monitor sums**: volume averages and loads equal to roundoff across rank counts (states stay bitwise): acceptable?
* **D3 large churn**: allgather of new keys (simple, O(new leaves) per rank) or a distributed sample sort from the start?
* **D4 target case** for the memory demonstration (e.g. the airfoil at 1440 x 720 with 3 levels, or a case the user cares about).
* **D5** the replicated path: keep it as a fallback (`FENICE_REPLICATED_REMESH`) or remove it after phase 6?
* Scope check: per-level partition kept (the subtree alternative measured ~1.5x slower stepping on 8 ranks, section 20).
