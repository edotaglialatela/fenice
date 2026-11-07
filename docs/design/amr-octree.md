# 3-D adaptive mesh refinement (octree leaves): design and plan

Status: **decisions answered; Q0 (measurements), Q1 (dimension layer and `CellMesh` in 3-D) and Q2 (the convective residual in 3-D) done.** Written 2026-11-07 after a read of the 2-D leaf engine
(`docs/design/amr-cell-based.md`, `amr-distributed.md`), a grep inventory of its source and the residual of the 3-D uniform solver. Every
number that was not measured is marked *(estimate)*. The decisions are in section 8 (the user took every recommendation, 2026-11-07); the Q0
numbers are in section 11, Q1 as built in section 12, Q2 in section 13. Each phase of section 7 ends with a stop for the user's go-ahead, as the 2-D rebuild did.

This is phase P9 of `amr-cell-based.md` (the 3-D octree) and the "dimension-agnostic layer" of section 3.7 of `amr-distributed.md`.

## 1. Goal and rules

Goal (user, 2026-11-07): a **generalised** AMR engine that runs in 3-D. The user accepts a risk to the 2-D bits if it only means
regenerating references; what must not be lost is correctness, conservation and the equality of MPI and serial runs.

Rules that carry over unchanged:

* MPI == serial bit for bit for any rank count, and results independent of the OpenMP thread count (a mesh above `kOmpMinWork`).
* `/fp:precise`, no FMA contraction, no `min`/`max` reductions in OpenMP loops (OpenMP 2.0), operation order kept where a bitwise oracle
  depends on it.
* Conservation to round-off across level jumps for Euler, SSP-RK2 and SSP-RK3, viscous terms included.
* Docs in sync with the parameters; no other software named; each file keeps its line endings.

What is new: the 2-D bitwise anchors are **preferred, not mandatory** (section 5.4). A reference of the 2-D AMR set may be regenerated, one
at a time, with a stated reason and the P7 validation gate.

## 2. What exists and what is dimension-specific (measured inventory)

The engine is `CellMesh` (leaves, faces, side lists, adapt), `CellFluxEngine` (stencils, virtual cells, fluxes), `CellAdvance`
(subcycling and registers), `CellIndicator`/`CellAdaptor` (remesh), `CellParallel`/`CellFetch`/`CellShare`/`CellParallelRemesh`
(partition, halos, topology-free remesh), and the `solve_amr` paths of `Simulation.cpp`. The face and cell physics (Riemann solvers,
reconstruction, boundary ghost states, EOS, 5-component state) is shared with the uniform solver and already handles three
dimensions; the 3-D uniform path exists, is tested and runs on MPI. So the work is in the **topology and in the tables built from it**, not in
the flux kernels: a face flux is computed from four stencil indices along the face normal, whatever the dimension.

First-pass inventory (`grep` on the 2-D idioms: `nbr_[4`, `side_ptr[4`, `4 * c`, `morton_key(`, `)*0.25`, `ci`/`cj`, `dy`, `CellBox`/`dilate_cells`;
the counts are occurrences, a pointer to where the 2-D shape is written into the code, not a count of lines to change; the classification is my
reading, and the first task of phase Q1 is a line-by-line audit that replaces this table):

| file | lines | `ci/cj` | `morton` | `4*c` sides | `*0.25` | `dy` | box | class |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `CellMesh.cpp` / `.hpp` | 869 / 201 | 86 / 6 | 8 / 1 | 2 / 2 | 1 | 3 / 5 | 0 / 2 | redesign: keys, find, faces, side lists, balance, adapt, merge |
| `CellFlux.cpp` / `.hpp` | 1736 / 303 | 14 | 0 | 4 | 1 | 16 / 0 | 0 | mechanical (neighbour codes, gradients) + redesign (virtual cells, interpolated jump cells, viscous cell kernels, IBM masks) |
| `CellRemesh.cpp` / `.hpp` | 475 / 126 | 38 | 3 | 2 | 1 | 2 | 5 / 2 | redesign: level hierarchy of the indicator, diffusion, body band |
| `CellParallel.cpp` / `.hpp` | 600 / 243 | 14 / 4 | 2 | 0 | 0 | 0 / 0 | 0 / 1 | mostly agnostic (keys, splitters, lists); mechanical where a (level, i, j) triple is built |
| `CellParallelRemesh.cpp` | 298 | 16 | 2 | 0 | 1 | 0 | 1 | mechanical: rings and neighbour offsets 8 -> 26, regions as boxes |
| `CellFetch.cpp`, `CellShare.cpp` | 257, 258 | 0, 18 | 1, 1 | 0 | 0 | 0 | 0, 3 | mechanical: region boxes and their key ranges |
| `CellAdvance.cpp`/`.hpp`, `ParallelScan.hpp`, `CellParallelDetail.hpp` | 60/54, 113, 17 | 0, 0, 2 | 0, 0, 1 | 0 | 0 | 0 | 0 | agnostic (levels and registers only) |
| `Simulation.cpp` (`solve_amr`, `fill_cells`, monitor, dump) | 2710 total | | | | | | | mechanical: leaf coordinates, `leaf_meta` has 5 columns, export |
| `IO.cpp` (`export_cells`) | | | | | | | | mechanical: cell type and connectivity of the VTU writer |
| `Physics.cpp` (`ImmersedBoundary::ghost_reach`, masks), `Params.cpp` | | | | | | | | redesign (IBM, section 4.9); the guard `use_amr` and `nz > 1` is one line |

5 610 lines in the `Cell*` files and `ParallelScan.hpp`. The 2-D idioms sit in a minority of the lines but in the data structures that everything
else indexes, so the change is wide, not deep.

## 3. Architecture: how one engine serves two and three dimensions (decision D1)

The user wants a generalised engine. The options:

| | A1 runtime dimension | A2 template everything on `D` | A3 separate 3-D engine |
|---|---|---|---|
| idea | `CellMesh` and `CellFluxEngine` hold `dim_` (2 or 3); loops run over `2*dim_` sides and `1<<dim_` children; key/child/neighbour arithmetic in a small header templated on `D` | the classes themselves are templates | copy and adapt the 2-D classes for 3-D |
| 2-D cost | a runtime loop bound in the topology and table builders (memory-bound, expected below 1 %); the flux loops read stencil indices and do not change | none at run time; two instantiations of 5 600 lines (compile time, test time) | none |
| code to maintain | one | one, two instantiations | two, which drift |
| fits the user's goal | yes | yes | no |

**Recommendation: A1**, with the templated key layer that `amr-distributed.md` section 3.7 already planned. The gate is measured, not assumed: the
`bench_cellflux` per-leaf cost in 2-D after phase Q2 must stay within 5 % of today's (section 9 of `amr-cell-based.md` records the baseline: 0.85-1.1x
of the uniform kernel). If it does not, the hot loops move to A2 for the flux kernels only. A3 is listed because it is the lowest-risk way to protect
the 2-D engine, and the cost is the duplicate that the user has said they do not want.

**Operation order (decision D2).** The 3-D uniform solver does not share expressions with the 2-D one. The convective gather of the 3-D uniform
residual (`Simulation.cpp`, the divergence loop) is

    r[q] = (((0 - (fr*Ax - fl*Ax)) - (ft*Ay - fb*Ay)) - (fk*Az - ff*Az)) * inv_vol,    inv_vol = 1/cell_volume

and the 3-D viscous residual (`NavierStokes::viscous_residual_3d`) builds the full velocity-gradient tensor per cell (13 scratch fields: three
velocities, T and nine gradients), face-averages the cell gradients for the cross terms, and has its own padded-state and `CrossGhosts` handling
(periodic z, MPI cross ghosts). The 2-D engine reproduces the 2-D kernels (`viscous_residual`, four gradients). The existing extrusion tests show that
the 3-D path matches the 2-D one only to round-off (about 5e-14), not bitwise. Therefore one engine cannot be bitwise equal to both uniform paths
unless it keeps **two expression families**, chosen by `dim_`:

* D2a (recommended): the 2-D family stays as it is (2-D AMR level-0 == uniform 2-D, bitwise, and all 2-D bits untouched where the order is kept); the 3-D
  family is new and is the one the 3-D level-0 oracle checks against `viscous_residual_3d` and the divergence loop above. Costs a branch in the cell
  kernels (`ViscousCellArrays`, `viscous_cell_quantities*` get a `w` and nine gradients in the 3-D family; the AVX2 runs get a 3-D twin).
* D2b: one 3-D-style expression family for both; the 2-D AMR references and the "AMR level-0 == uniform 2-D" oracle become round-off level and the
  2-D references are regenerated. Simpler code, weaker 2-D oracles.

## 4. What changes in three dimensions (and the traps)

### 4.1 Keys (decision D3)
A leaf is `(level, i, j, k)`; the Morton key interleaves three coordinates. A 64-bit key leaves 21 bits per axis, so `nx << max_level` must stay
below 2^21 = 2 097 152 cells per axis at the finest level (2-D allows 2^29, `kKeyBits`). A 3-D mesh with 2 M cells per axis has 10^19 cells, which no run
will reach, so the limit is a check with a clear message, not a constraint. A 128-bit key would touch the per-level splitters, the bisection of
`GlobalFacts` and every sort; there is no case for it. **Recommendation: 64-bit key, 21 bits per axis in 3-D, the 2-D width unchanged.** `ck` is stored for
every leaf (zero in 2-D, 4 bytes per leaf).

### 4.2 Topology
* Sides: 6 (`Face::Left..Back` already exist). `side_ptr`/`side_face` are indexed `2*dim*c + s`. A side holds **1 face, or 4 fine faces** (2 in 2-D) in
  a fixed order (transverse position `t2` major, `t1` minor); the gather sums them in that order. `CellFace::tpos` becomes a pair (`t1`, `t2`).
* Children: 8. The conservative mean of a merge extends the 2-D sum order as `((a+b)+(c+d))` for each z-plane and then `(plane0 + plane1) * 0.125`
  (in 2-D the `*0.25` expression is unchanged). Injection on refine is unchanged.
* **Balance (decision D4).** Vertex 2:1 balance in 3-D means all **26** neighbours (6 faces, 12 edges, 8 corners). It is the direct generalisation
  of the invariant the 2-D engine is proven on and every stencil reach (virtual cells, cross-derivative cells, the halo reach) is argued from it.
  Face-only or face-and-edge balance would leave levels two apart at the corner of a stencil. **Recommendation: full 26-neighbour balance first**; a
  weaker balance is an optimisation to be measured after the engine is correct (it saves leaves, at the price of re-proving the stencil reach).
  Expected consequence *(estimate)*: the ripple adds more refined leaves than in 2-D around a refined region (a buffer of one leaf in all 26 directions).
* Lookup: a dense table per level while `nx_l*ny_l*nz_l <= 4N + 65536`, else a binary search of the level's keys; unchanged in form, but note the dense table is
  cubic in the level, so it stops being used one or two levels earlier than in 2-D.
* Brute-force oracle: the `cellmesh_*` tests check the topology in finest-level integer coordinates (tiling, balance for every touching pair, one face per
  shared segment, side coverage, ordering). It is extended to 3-D (shared squares instead of segments) and run in 2-D and 3-D; it is the first thing written.

### 4.3 Stencils, virtual cells and the jump closure
* A stencil is still four indices along the normal. Beyond a coarser cell: injection. Beyond a finer cell: the mean of the leaves under it, a **virtual cell
  of 8 children** (recursive, because the far children of a refined block may be finer: the same argument as in 2-D, with 4 far children instead of 2).
* **The jump closure (b)+(c)** (section 14 and decision 17 of `amr-cell-based.md`, the default) needs a 3-D form that is not a mechanical copy. The fine
  side sees the coarse leaf as a virtual cell with the coarse value interpolated, with the coarse leaf's own limited slopes, to the centre of the fine cell
  that would sit across the face: 0.25 coarse widths from the coarse centre towards the fine leaf in the **normal** direction and +-0.25 in **each of the two
  transverse** ones (2-D has one transverse offset). A coarse face is covered by **four** fine faces. The coarse side (c) is reconstructed against the mean
  of the fine leaves behind the face. The ratio of the face areas and the register weights follow (section 4.7). The 3-D closure is validated by the
  order study across a jump (section 5.2), not by analogy.
* Boundary closure: virtual cells at a physical edge keep the replicate rule; the `boundary_order = 2` ghost states are functions of one cell and are
  evaluated where read (section 21 of `amr-cell-based.md`), so they extend without tables.

### 4.4 The residual to reproduce
The anchor oracle of phase Q2 is "a tree with only level-0 leaves equals the 3-D uniform solver bitwise, for every boundary type": the divergence
expression of section 3, the 3-D Riemann/reconstruction calls, the primitive round trip on boundary faces, the periodic wraps of three axes, and the
`CrossGhosts` rules of `viscous_residual_3d`. Phase Q2 reads the three call sites in the uniform path and writes this list the way section 2.3 of
`amr-cell-based.md` does for 2-D, before the engine is written. (Not done in this document: the viscous padded-state and `CrossGhosts` rules of the 3-D path
are read in Q2, not described here.)

### 4.5 Viscous terms
The cell-centre gradients come from six neighbours (central differences; virtual or interpolated cells across a jump; the diffusive ghost at a physical
edge); each cell carries `u, v, w, T` and nine gradients (the 2-D engine: `u, v, T` and four), so the per-leaf viscous arrays grow by about 2x
*(estimate)*. The viscous time-step bound uses the 3-D sum `2 L D sum 1/h^2` over three axes. The cross terms of a face use the two cell gradients as
now.

### 4.6 Boundaries
Six sides. The non-reflecting outlet has **two** tangential directions (`NrbcTangent` in the 2-D engine carries one pair); the mass-flow inlet's side
average is over a 2-D face, summed in position order on every rank (never as partial sums), which fixes the order in 3-D too; the corner composition rules
of `boundary_order = 2` (a y-face composes the left/right and bottom/top maps in the other order, section 21) become **edge and corner** rules in 3-D: the
rule is derived from the 3-D uniform code (`fill_ghost_slab` and the padded frame), not guessed. Periodic wraps on three axes.

### 4.7 Time stepping and conservation
Unchanged in structure: level `l` advances by `dt_0/2^l`, the finer level first, registers on the coarse-fine faces accumulated with the stage weights. A
coarse side now has up to **4** fine faces, each with its own register, and the coarse cell sums them in the fixed order of 4.2; the face areas of fine
and coarse faces differ by 4 (2 in 2-D), which the area factor carries. Conservation to round-off is the oracle, with the control that removing a
register fails it, as in 2-D.

### 4.8 Indicator (decision D5)
The face-variation indicator and its parent-level join generalise face by face (a leaf has 6 sides). One point is a numerics choice, not mechanics: the
diffusion step adds 0.1 to both cells of a face whose cell is above the trigger, `amr_diffusion` times. A cell now has 6 faces, so for the same input the
band spreads further in 3-D than in 2-D (a cell next to a flagged one receives 0.1 per flagged neighbour: up to 6 against 4). Options: (a) keep the
coefficient (a thicker band in 3-D, simplest, what the extrusion check below needs to be understood against); (b) scale it by 4/6 so a plane front gives the
same band; (c) a new parameter. **Recommendation: (a) first, measure the band on a plane shock, decide with the user.** Note for the oracle: a z-invariant
3-D run does not reproduce the 2-D flags with (a) (the z neighbours add to xi), so the extrusion oracle of section 5 fixes the leaves by `amr_force_box`
and compares the physics, and the indicator is tested separately against a plain structured 6-neighbour implementation (as the 2-D one was).

### 4.9 Immersed boundary: the memory trap (decision D6)
Today the masks per level are cut from a **uniform mesh of that level** (`ibm_factory(generate_cartesian_mesh(Lx, Ly, nx_at(l), ny_at(l)))` in
`CellFlux.cpp`), the body band is found on the finest-level mask, and the ghost-wall table is built on the uniform finest mesh and held on every rank
(`ghost_ib_`). In 2-D that is an O(finest-level cells) array that is affordable (2880 x 1440 = 4 M cells for the large airfoil). In 3-D it is
`nx_l*ny_l*nz_l` per level: 64^3 with 4 levels is 10^9 cells at the finest level. It would defeat the purpose of AMR and must be redesigned, not ported.

**Proposed design.** A **per-leaf geometric classification** from a distance bound: every shape has an exact or Lipschitz-1 signed distance (ball: exact;
cylinder and extruded polygon: the 2-D distance; `closest_wall_point3` exists since the 3-D ghost wall). A box of any level is classified by the
distance at its centre against its half-diagonal: entirely solid, entirely fluid, or cut. The tree descends only through cut boxes, so the work is
O(surface cells), and the mask of a leaf is the centre test of the same shape function the uniform mask uses (the same expression, so a level-0 tree has
the identical mask). The ghost table is built from the leaves within `ghost_reach` of the surface (the band of finest-level leaves that the body
refinement already forces), distributed with the leaves, not global. The 3-D ghost wall (`ibm_method = ghost`, merged 2026-11-07) provides the
per-cell ghost state; what changes is where its table lives. `amr_refine_ibm` keeps the body and a margin on the finest level. This is an engineering piece of
its own (phase Q4) and the largest redesign outside the topology.

### 4.10 MPI and the distributed remesh (the cost trap)
Everything of the 2-D MPI layer applies: per-level Morton blocks, halo = the cells the stencils read, registers from the fine owners, fetch regions,
topology-free adapt with ripple rounds, splitters by bisection on the key. The pieces that depend on dimension are the region boxes (`CellBox`: w x h
becomes w x h x d), the ring neighbourhoods of the adapt region (the 8 neighbour offsets become 26, rings of the leaf's level: 3), the dependency marking
(`face_dependencies`: six neighbour codes) and the registers (4 per coarse side). It is mechanical, but the **cost does not scale like the code**:

* The 2-D distributed remesh already costs 1.6-1.9x the wall time of the replicated one it replaced (`amr-distributed.md` section 14), and that cost is the
  fetch regions, which are surface-like.
* Surface-to-volume is worse in 3-D. For a rank block of `n` leaves per level the halo of depth `r` rings is about `2 dim r n^((dim-1)/dim)` *(estimate,
  compact blocks)*. At `n = 10^5` and depth 2 that is about 2.5 % of the owned leaves in 2-D and about 26 % in 3-D; for the 3-ring adapt region 3.8 % against about
  39 %. So a 3-D rank pays roughly **ten times** the halo traffic of a 2-D rank at equal leaf count, and the per-rank block must be larger to amortise it.
* This is why the per-level partition (every level advanced separately) deserves a measurement in phase Q0 on a 3-D proxy before the MPI phase is
  scheduled, and why decision D7 (below) lets the single-rank phases go first.

### 4.11 Output, restart, frontend
* `.vtu` of leaf hexahedra (cell type voxel/hexahedron instead of pixel/quad; per-rank `.vtu` + `.pvtu`), `fenice.read_vtu` extended.
* `dump_state`/`final.npz`: `leaf_meta` has five columns (bounds and level); 3-D adds the z bounds (seven). The Python readers of the AMR validation scripts
  are updated together.
* Checkpoint: leaf list gains `k`; resume stays bitwise, also across rank counts. `leaves_to_uniform` (hot start from an AMR checkpoint) gets a 3-D version;
  hot start from a uniform checkpoint interpolates at leaf centres in 3-D.
* `Params`: the guard "AMR is 2-D only" is removed in the phase that opens the CLI (Q6); parameters, GUI keys, `parameters.md` and the docs pages follow.

### 4.12 Memory *(estimate)*
The 2-D engine costs 1.1-1.5 kB per leaf on every rank (measured, `amr-distributed-handoff.md`; three quarters of it is the engine's tables). In 3-D a leaf has
6 sides, about 3 faces, and carries nine gradients instead of four: **about 2-2.5 kB per leaf** *(estimate)*. One million leaves is then about 2.5 GB
in total, so distributed storage (already built) is not optional in 3-D, and the saving of registers only on coarse-fine faces (named in the handoff as
a cheap saving) becomes worth doing early.

## 5. Oracles

The 2-D engine earned its trust from oracles that did not depend on the engine itself. The same, for every phase:

### 5.1 Topology
The brute-force finest-level-coordinate oracle of section 4.2 in 2-D and 3-D; refine-then-merge restores `U` bitwise; random adapt sequences; periodic
and non-periodic axes, `nx = 1, 2`.

### 5.2 Numerics
1. **Level-0 3-D tree == 3-D uniform solver, bitwise**, residual and steps, every boundary type, Euler/RK2/RK3, MUSCL with all limiters, WENO3-Z, the three
   Riemann solvers, `boundary_order` 1 and 2, with viscosity, body force, sources and the IBM.
2. **z-invariant 3-D AMR == 2-D AMR per layer.** A periodic z, a large `Lz` and a fixed set of leaves (`amr_force_box` spanning all z), as
   `extrusion_matches_2d` does for the uniform solver; every 2-D adaptive test becomes a 3-D test of the same physics at round-off level (not bitwise,
   because of the two expression families, section 3). The cheapest strong check, and it reuses every 2-D validation.
3. Free stream exact across jumps (R == 0 exactly); conservation to 1e-13 for the three schemes, viscous included, with the register control failing.
4. Permutation symmetry: a planar Sod tube along x, along y and along z on the same refined slab, each against the exact solution and against each other
   (equal up to the permutation to round-off in the smooth part).
5. Manufactured solutions on an adapted 3-D mesh (the 3-D Navier-Stokes solution of `mms_ns_3d` with a static refined box, `amr_force_box`): observed order
   away from the jumps and the near-jump table of the closure (section 12 and 14 of `amr-cell-based.md`) rebuilt for 3-D.
6. Uniform 3-D finest-level run against the adaptive one (Sod, 3-D Taylor-Green, the sphere): L1 differences, the P7 gate.

### 5.3 Parallel
`FakeHub` emulated ranks (1-16, NaN-poisoned halos) and real MPI on 1-7 ranks, serial == ranks bit for bit, for the 3-D cases; the thread-count
independence on a mesh above `kOmpMinWork`; `amr_mpi.py --check` and `ibm_ghost.py --check` extended with 3-D adaptive cases; the halo-reach guard test
extended (the reach is pinned per dimension).

### 5.4 The 2-D references
The AMR regression references of the 2-D set (the `amr_*` cases and the hot starts from AMR) may be regenerated. The policy: a phase **tries** to keep the
2-D bits (the golden dumps of `cases/validation/amr/golden.py`, ten adaptive runs, are compared after every phase); a reference is regenerated only when
a change of operation order or balance makes the bits differ, with the reason written in the commit, **in its own commit**, and only after the same
gate as P7 (each case against the uniform run at its finest level, error against the base mesh not worse, MPI == serial on 1-7 ranks, bitwise restart).
The 32+ uniform references are not touched: the uniform solver is not changed by this work.

## 6. Where the effort goes

The 2-D leaf engine took phases P1-P8 and, for the distributed storage and the topology-free remesh, phases 0-6 of `amr-distributed.md`. The 3-D
generalisation reuses the structure, so the work is smaller than that sum, but its cost is concentrated in: the topology (keys, faces, balance, merge),
the virtual-cell/jump-closure geometry, the viscous kernels (and their AVX2 twin), the IBM redesign of 4.9, the generalisation of the fetch regions, and
the validation. The flux kernels, the time recursion, the registers and the Riemann/reconstruction code change little.

## 7. Phases (each ends with a stop for the user's go-ahead)

| Phase | Content | Acceptance |
|---|---|---|
| Q0 | Measurements, no engine change: the 2-D baseline of record (bench, goldens, memory); the 3-D uniform costs of the target cases (sphere Re 20-100, 3-D Taylor-Green) and the leaf counts an octree would need for them (an offline estimate from the uniform solution, as P0 did); the halo fraction of Morton blocks in 3-D on a proxy; the decision on the two expression families | numbers in this document; D1-D9 answered. **Done 2026-11-07, section 11** |
| Q1 | Dimension layer (key arithmetic, children, 26 offsets, key ranges of a region) and `CellMesh` in 2-D and 3-D: leaves, faces, side lists (1 or 4), balance, adapt (refine to 8, merge, ripple); the brute-force oracle in both dimensions | oracle tests in 2-D (existing ones unchanged) and 3-D; golden dumps of 2-D equal. **Done 2026-11-07, section 12** |
| Q2 | `CellFluxEngine` convective residual in 3-D: stencils with virtual cells of 8, the 3-D jump closure (b)+(c), boundary ghosts on six sides, WENO3-Z, `boundary_order = 2`; the 2-D per-leaf cost gate | level-0 3-D == uniform 3-D bitwise; free stream exact; the near-jump order table; **2-D per-leaf cost within 5 %**. **Done 2026-11-07, section 13** |
| Q3 | `CellAdvance`: 4 registers per coarse side; subcycling in 3-D | conservation for the three schemes; level-0 steps == uniform 3-D steps; permutation Sod; the z-invariant check |
| Q4 | Physics: viscous 3-D family (and AVX2 twin), body force, sources, NRBC with two tangential directions, mass-flow inlet; **the IBM redesign** (per-leaf classification, the ghost table from the band leaves) | bitwise level-0 tests with each physics; MMS on an adapted 3-D mesh; IBM masks equal to the uniform ones on the same cells; ghost wall equal to the uniform 3-D ghost wall |
| Q5 | Indicator, flags, remesh, initial adaptation, body band, hot starts in 3-D (serial and OpenMP, through the whole-mesh `CellMesh::adapt` that stays as the oracle) | indicator equals a structured 6-neighbour implementation; remesh deterministic, idempotent, conservative; a 3-D Sod and Taylor-Green with AMR against uniform |
| Q6 | MPI: partition per level, halo lists, registers, region boxes and 26 rings in the topology-free remesh; **the CLI opens** (`use_amr` with `nz > 1`) | serial == 1-7 ranks bitwise (FakeHub and real MPI); the halo-reach guard; thread independence; memory per rank |
| Q7 | I/O (hexahedra, 3-D `leaf_meta`, checkpoint, readers), parameters, GUI keys, docs, CI, the 3-D regression references | restart bitwise; docs and parameters in sync; CI green |
| Q8 | Performance and the deferred 3-D validation: cost per accuracy against uniform 3-D on the sphere and the 3-D Taylor-Green; scaling; `auto` choosing the ghost wall in 3-D if the evidence supports it | report; the user decides what ships |

The 2-D cost gate in Q2 is the go/no-go for A1, and the Q0 estimate of the leaf count is the go/no-go for the whole project: if an octree needs more than
about half the cells of the uniform mesh on the target cases, the gain does not pay for the subcycling and the table memory, and the user should know
before Q1.

## 8. Decisions for the user

Answered by the user on 2026-11-07: **every recommendation is taken** (last column). D5 is settled by the Q0 measurement (section 11.4).

| # | Question | Options | Decided |
|---|---|---|---|
| D1 | Architecture | A1 runtime `dim_` + templated key layer; A2 template everything; A3 separate 3-D engine | **A1**, with the 5 % gate in Q2 (protocol in 11.5) |
| D2 | Expression families | D2a keep the 2-D kernels and add the 3-D ones (3-D level-0 bitwise vs 3-D uniform, 2-D untouched); D2b one family, 2-D regenerated | **D2a** |
| D3 | Key | 64-bit, 21 bits per axis in 3-D; 128-bit | **64-bit** |
| D4 | 2:1 balance in 3-D | all 26 neighbours; face + edge; face only | **all 26** (costs < 0.2 % of the leaves on the sphere, 11.2) |
| D5 | Indicator diffusion in 3-D | keep 0.1 per face; scale by 4/6; new parameter | **(a) keep 0.1**: a plane front gives the same band in 2-D and 3-D (11.4) |
| D6 | IBM in the octree | per-leaf distance classification and distributed ghost table (4.9); mask only in the first release | **the classification** |
| D7 | Order of the MPI work | single-rank phases first (Q1-Q5 test through the whole-mesh adapt) and the CLI opens at Q6; or distributed from the first phase | **single-rank first, CLI at Q6** |
| D8 | Closure and indicator defaults | carry over unchanged ((b)+(c) closure, criterion 0.1, density + velocity, parent-level join), verified in 3-D by the order table | **carry over, verify** |
| D9 | First-release scope | WENO3-Z and `boundary_order = 2` included (they are in the engine); WENO5 stays out as in 2-D; no surface samples in 3-D (as for the uniform ghost wall); hot start from a uniform checkpoint included | **as listed** |
| D10 | Stop rule | go/no-go in Q0 on the leaf-count estimate and at the end of each phase | **as in the 2-D rebuild**; the Q0 verdict is in 11.1 |

## 9. Risks

* **Cost of the distributed remesh in 3-D** (4.10): the fetch regions are surface-dominated. Mitigation: Q0 measurement, the single-rank phases first, the
  "fetch only around the border leaves" idea recorded in `amr-distributed.md` section 14 as not tried.
* **The IBM redesign** (4.9) is new code with no 2-D equivalent; the ghost table has to be distributed with the leaves.
* **2-D performance regression** from the runtime dimension: gated in Q2.
* **Two expression families** are two sets of kernels to keep correct (and their AVX2 twins); the extrusion oracle ties them at round-off.
* **The jump closure in 3-D** is not a mechanical copy (two transverse offsets, four fine faces): it is validated by its own order table.
* **Memory** (4.12): distributed storage is a precondition for useful 3-D runs.
* **Validation depth.** The 3-D ghost wall has only self-convergence evidence today (`docs/source/modules/Physics.md`); the 3-D validation deferred for
  that reason is scheduled in Q8, after the engine, not before.

## 10. What this document does not know

* The viscous padded-state and `CrossGhosts` rules of the 3-D path, the edge/corner composition of `boundary_order = 2` for the viscous terms in
  3-D, and the 3-D `NrbcTangent` plumbing are described by their role, not read in detail; Q4 reads them first. (The convective residual has no
  edge or corner composition: section 13.1.)
* The inventory counts are a first-pass `grep`; they bound the work, they do not measure it.
* The 3-D memory per leaf and the 3-D per-leaf cost of the face engine remain estimates until Q2-Q4 (there is no 3-D engine to measure).
  The leaf counts and the halo fractions were measured offline in Q0 (section 11); they are the engine's rules applied to uniform
  solutions, not runs of an engine.

## 11. Phase Q0 results (2026-11-07)

Binary of record: `main` at `e5bc73f` (no engine change in Q0), release build, Ryzen 7 7800X3D (8 cores, 16 logical), 31 GB. The binaries
are kept in `results/validation/amr_octree_q0/bin/` (with `COMMIT.txt`) so that Q1-Q2 compare against the same build. Script:
`cases/validation/amr/octree_q0.py` (stages `run`, `estimate`, `halo`, `cost`, `bench`; each writes `summary.json` under
`results/validation/amr_octree_q0/<stage>/`).

### 11.1 Verdict (D10)

| target | octree leaves / uniform cells | work ratio sum N_l 2^l / (N_u 2^M) | verdict |
|---|---:|---:|---|
| sphere Re 20, `max_level` 2 (base 5 cells/D, finest 20) | 0.18 | 0.16 | **go** |
| sphere Re 20, `max_level` 1 | 0.25 | 0.20 | go |
| sphere Re 100, `max_level` 2 | 0.24 | 0.23 | **go** |
| sphere Re 100, `max_level` 1 | 0.31 | 0.26 | go |
| 3-D Taylor-Green 96^3, `max_level` 1 or 2, default indicator | 0.994 (time mean; never below 0.935) | 0.994 | **no-go** as a performance case |
| the same, criterion 0.3 | 0.87 (time mean; 1.0 at t = 9.3) | 0.86 | no-go |

The gate of section 7 (an octree needing more than about half the uniform cells does not pay) is passed by the sphere at both Reynolds numbers
and both depths, with margin, and failed by the Taylor-Green vortex at any setting: the vortex fills the periodic box and the default
indicator flags nearly every cell from t = 0 (raising the criterion to 0.3 does not change that past the start). The 3-D Taylor-Green stays in
the plan as a **correctness** case (level-0 and z-invariant oracles, Q2-Q3), not as a cost case in Q8. **Recommendation: go to Q1**, with the
sphere (external flow with a body and a wake) as the performance target.

The work ratio assumes that the finest level's substep equals the uniform finest-mesh step. It does: the engine's bound uses each leaf's own
lengths (`viscous_cell_speed` in `CellFlux.cpp`: `|u| + c + 2 L sum 1/h^2 nu_eff` with `h` of the leaf's level), so the viscous part of a level-l
leaf is 2^l that of level 0 and `dt_0 / 2^M` equals the uniform `dt` for the same worst state (both sphere runs are partly viscous-limited: Re 20
took twice the steps of Re 100). The 3-D bound (4.5) keeps this form.

Projected wall-time gain on the sphere *(estimate)*: work ratio x per-leaf cost ratio of the face engine. The 2-D ratio measured today is 1.39
(11.5; the 3-D one is unknown until Q2), so the residual cost is 0.22-0.32 of the uniform run at `max_level` 2 (a gain of about 3-4.5x) and 0.27-0.37
at `max_level` 1, before the remesh (2.4 % of the run in 2-D at 1 thread, `bench_cellflux`). Memory: 106 k leaves at 2-2.5 kB *(estimate, 4.12)*
= 210-265 MB against 497 MB measured for the uniform 432 k cells.

Limits of the estimate: one end state per case (both sphere flows are steady; the history of the start-up is not counted); every level of the
hierarchy evaluated on the whole restricted grid (the engine sees a coarser leaf's value where a level is absent: it moves where the band ends by
a cell, not its size); the sphere sits in a 6 x 3 x 3 channel with 8.7 % blockage, so the far field, the part an octree coarsens, is small (a
larger domain only lowers the ratios); the finest resolution is 20 cells per diameter, the one the uniform fields exist at.

### 11.2 Leaf counts per level (sphere, ghost-wall band of 6 finest cells, 26-neighbour balance)

| case | M | level 0 | level 1 | level 2 | total | uniform |
|---|---|---:|---:|---:|---:|---:|
| Re 20 | 1 | 46 376 | 60 992 | | 107 368 | 432 000 |
| Re 20 | 2 | 4 264 | 12 264 | 60 992 | 77 520 | 432 000 |
| Re 100 | 1 | 42 436 | 92 512 | | 134 948 | 432 000 |
| Re 100 | 2 | 4 191 | 8 908 | 92 512 | 105 611 | 432 000 |

* The finest level holds 79-88 % of the leaves: the body band and the wake decide the cost, so the band width matters more than the coarse levels.
  Band of 2 (mask) instead of 6 (ghost wall): 0.175 / 0.234 instead of 0.179 / 0.244 (Re 20 / 100, M = 2).
* **D4:** the 26-neighbour ripple adds at most 0.2 % of the leaves (Re 20, M = 2: 77 520 against 77 380 without balance; Re 100: 105 611 against
  105 415). Full balance costs nothing measurable here.
* The fields are `results/validation/ibm_sphere/ghost_20` (Re 20) and a new uniform run at Re 100 (ghost wall, 120 x 60 x 60, t = 30, 11 060 steps,
  Cd settled at 1.364, 643 s on 16 threads).
* The same flows on their mid-plane z = cz as 2-D quadtrees give 0.40-0.53 of the uniform cells at M = 2 and 0.47-0.60 at M = 1: the third dimension roughly halves the ratio
  for a compact body, as expected.

### 11.3 Halo fraction (Morton blocks of equal leaf count per level; halo leaves of other ranks / owned leaves)

Two neighbourhoods, at the owner's level: **stencil** = the cells a residual reads (+-1, +-2 along each axis, and the edge cells (+-1, +-1) of each
pair of axes for the viscous cross terms: 18 cells in 3-D, 12 in 2-D); **r = 3** = three rings in all 26 (8) directions, the shape of the adapt
region of the topology-free remesh (4.10). (Two full rings, 124 cells in 3-D, overstate the stencil: in the summary as `2`.)

| mesh | leaves per rank | stencil: mean / max | r = 3: mean / max |
|---|---:|---:|---:|
| uniform 2-D 2048^2 on 32 ranks (compact blocks) | 131 072 | 0.020 / 0.023 | 0.029 / 0.035 |
| uniform 3-D 128^3 on 16 ranks (compact blocks) | 131 072 | 0.158 / 0.190 | 0.250 / 0.301 |
| 2-D airfoil 720 x 360 after 50 steps (golden dump, `max_level` 2), 16 ranks | 16 763 | 0.090 / 0.213 | 0.155 / 0.429 |
| 3-D sphere Re 20 estimate (M = 2), 4 ranks | 19 380 | 0.606 / 0.842 | 1.506 / 2.073 |
| the same, 8 / 16 ranks | 9 690 / 4 845 | 1.03 / 1.63, 1.43 / 2.17 | 2.90 / 4.06, 4.79 / 7.26 |

* Compact blocks: 3-D costs 8x the halo of 2-D at equal leaves per rank (the estimate of 4.10 said about 10x: 2.5 % and 26 %; measured 2.0 % and
  16 %).
* Real trees: at equal leaves per rank (about 17-19 k) the 3-D sphere has 7x the stencil halo of the 2-D airfoil (mean) and **about 2x what a
  compact 3-D block of that size would have** (0.30, scaled from the 131 k proxy as n^(-1/3)): the per-level partition cuts each level separately,
  so a rank's share of a small level is a thin, scattered piece of the Morton curve. The adapt region (r = 3) already exceeds the owned leaves
  at 4 ranks.
* Consequence for Q6 (not decided now): 3-D MPI needs about 10^5 leaves per rank and per level, or a partition that is not cut per level (one
  Morton curve over all levels with work weights 2^l). The single-rank phases (D7) do not depend on it.

### 11.4 Indicator diffusion (D5)

Plane fronts (density jump, periodic 64^d grid; refined band width in cells per front cell): aligned 2.00 in 2-D and 2.00 in 3-D with 0.1 and with
0.1 x 4/6; front diagonal in two axes 2.27 / 2.26 (0.1) and 1.97 / 1.97 (4/6); diagonal in three axes 2.23 (0.1) and 2.22 (4/6) in 3-D. The third pair of faces
raises xi but **does not widen the band** of a plane front; the coefficient changes 2-D and 3-D alike. On the sphere, 4/6 saves 3-7 % of the leaves (Re 20,
M = 2: 0.167 against 0.179). Decision (a), keep 0.1, as recommended: no 3-D-specific constant. The z-invariant oracle still fixes the leaves with
`amr_force_box`, because xi differs (not the flags of a plane front, but the values).

### 11.5 2-D baseline of record

* **Golden dumps:** `results/validation/amr_octree_q0/golden_q0/` (`golden.py dump`, the ten adaptive runs, 49 files), every run equal to its 1-thread run.
  Every later phase runs `golden.py compare results/validation/amr_octree_q0/golden_q0 NEW` (policy of 5.4).
* **`bench_cellflux 768 384 20`**, 5 interleaved repeats (ns per cell and residual, min / max):

  | | 1 thread | 8 threads |
  |---|---|---|
  | uniform `Simulation::residual`, 294 912 cells | 51.6 / 52.8 | 8.1 / 8.7 |
  | face engine, level-0 same mesh | **72.6** / 73.7 (1.39-1.41x of uniform) | 11.1 / 16.2 |
  | face engine, adapted (68 283 leaves) | 74.6 / 78.3 | 13.6 / 18.5 |
  | stencil build (ms) | 9.2 / 9.7 | 3.4 / 4.2 |

  The uniform kernel has become 2.7x faster since the P2 gate of `amr-cell-based.md` section 9 (139 -> 52 ns/cell, the fused uniform residual among
  the later changes; not bisected) and the face engine 1.7x (124 -> 73), so the face engine now costs **1.4x the uniform kernel per cell** at level 0
  (it was 0.9x). This is the factor of 11.1; closing that gap is a 2-D optimisation, outside this plan.
* **Protocol of the Q2 5 % gate:** `bench_cellflux 768 384 20`, the Q0 binary and the new one interleaved, 5 repeats each, 1 thread, compare the
  minima of the level-0 face engine (spread 1.5 % at 1 thread, so 5 % is resolvable). At 8 threads the spread is 36-78 % on this machine: the
  8-thread figure is reported, not gated. Nothing else may run on the machine meanwhile.
* **Memory:** `amr_memory.py --only airfoil_720 --ranks 1`: 268 209 leaves after 50 steps, resident 266 MB at the end and 358 MB at the peak (1.0 /
  1.34 kB per leaf), tables mesh 87 + engine 184 MB.

### 11.6 3-D uniform cost (the solver the octree competes with)

`octree_q0.py cost`: wall time of 40 minus 20 steps (setup removed), min of 3, no output, peak working set:

| case | cells | 1 thread | 8 threads | 16 threads | peak memory |
|---|---:|---:|---:|---:|---:|
| sphere Re 100, 20 cells/D, ghost wall (SSP-RK2, MUSCL, viscous) | 432 000 | 437 ns/cell/step | 140 | 131 | 498 MB (1.2 kB/cell) |
| the same, mask | 432 000 | 449 | 147 | 136 | 498 MB |
| Taylor-Green 64^3 (case defaults) | 262 144 | 500 | 116 | 98 | 264-296 MB |
| Taylor-Green 96^3 | 884 736 | 487 | 129 | 120 | 967 MB (1.15 kB/cell) |

The ghost wall costs the same as the mask (3 %). The sphere scales 3.1x from 1 to 8 threads (Taylor-Green 3.8x): the 3-D uniform path is
memory-bound on this machine, which an octree with fewer cells also helps. A steady sphere at Re 100 takes 11 060 steps, about 10 min on 16 threads;
at Re 20, 22 292 steps.

### 11.7 Expression families (D2)

Decided D2a: the 2-D kernels stay; the 3-D family is new and checked bitwise against `viscous_residual_3d` and the 3-D divergence loop at level 0.
No measurement was needed for it in Q0.

Note for later phases: the Q0 binary has the case folder of its (removed) build tree compiled in; run it with `FENICE_CASES=<repo>/cases`.

## 12. Phase Q1 as built (2026-11-07): the dimension layer and `CellMesh` in 2-D and 3-D

* **Dimension layer** `cpp/include/fv/CellKey.hpp` (header only): `morton_key` (2-D, 29 bits per axis, moved here unchanged and inline) and
  `morton_key3` (3-D, 21 bits per axis, `x` lowest bit, `key >> 3` = the parent), `cell_key(dim, i, j, k)`, `cell_key_bits`, `child_offset`
  (children in key order: x fastest, then y, then z), `neighbour_offset` (the 3^D - 1 offsets from a constant table, dz outer, dy, dx inner: the
  2-D loop order), `transverse1/2` (the transverse axes of a face: (y, z), (x, z), (x, y)). The key-range helper of the fetch regions is left to
  Q6, its only user.
* **`CellMesh`** (A1, runtime `dim`): `CellMesh::make_3d(Lx, Ly, Lz, nx, ny, nz, max_level, px, py, pz, x0, y0, z0)`; the 2-D constructor and every
  2-D call site are unchanged (`k` arguments default to 0). New: `dim`, `ck` (0 for every 2-D leaf, 4 bytes per leaf), `nz`/`Lz`/`z0`/`periodic_z`,
  `dz`, `zc`, `volume` (the area in 2-D), `nz_at` (1 in 2-D), `key_of`, `sides()` / `children()` / `fine_faces()` (4/4/2 or 6/8/4).
  `CellFace::tpos2` (second transverse position, 0 in 2-D); `CellFace::length` is the face measure (`Mesh::face_area`: `dy dz`, `dx dz`, `dx dy` in 3-D).
  `CellNeighbour::c, d` (the third and fourth finer leaf in 3-D). Gather lists are indexed `sides() * c + s`; a side towards finer leaves holds
  `fine_faces()` faces in the slot `(tpos & 1) + 2 (tpos2 & 1)` (tpos2 major), checked at every rebuild. `source` holds 8 entries (-1 beyond the
  children). One face maker (`face_of`) serves `rebuild` and `build_from`; `assign_leaves` / `assign_partial_leaves` take `ks` (3-D).
* **Adapt in 3-D:** refine to 8 children by injection, the ripple over all 26 neighbours (vertex balance, D4), merge of 8 siblings with the mean
  `(((a+b)+(c+d)) + ((e+f)+(g+h))) * 0.125` (each z-plane in the 2-D order), the merge test over the 26 neighbour blocks of the parent and the
  children of each that touch it, `forced_by_finer` over the level-l+1 ring of the 2^3 block, `has_leaf_under` over 8 children, the incremental
  rebuild's dirty marking over 6 sides and the 4 children across a side. The 2-D operation order is kept everywhere (face order, slots, the merge
  mean, the order of the checks that can throw).
* **Tests** (`cpp/tests/test_amr_octree.cpp`, 9 tests; the generic oracle `check_cell_mesh_nd` in `test_amr_common.hpp`): the key functions against
  the bit definition, child and neighbour order, the 21-bit limit; level 0 against `generate_cartesian_mesh_3d` (centres, volumes, face areas, face
  counts) for the 8 periodicity combinations; the brute-force oracle (tiling, order, vertex balance for every touching pair, one face per shared
  square with orientation / kind / level / both transverse positions / area, gather lists and neighbour summaries) on random adapt sequences,
  periodic domains of 1 and 2 cells per axis; conservation to 1e-12; the incremental rebuild equal to a full rebuild (`same_arrays`, also in 2-D);
  refine-then-merge restores `U` bitwise; a hand case of the ripple across a corner and edges and of merges refused across a face, an edge and a
  corner; four finer neighbours in tpos2-major order; the `forced_by_finer` closure equal to the ripple; `build_from` equal to `keep_partial`;
  `assign_leaves` order independence and the tiling check. The generic oracle is also run on 2-D meshes next to the 2-D one. Two mutations were
  caught (a face-only ripple: 9 tests fail in 2-D and 3-D; swapped slots: 8 tests fail).
* **2-D unchanged:** unit tests 139/139 (the 2-D `cellmesh_*` tests untouched), regression ALL MATCH serial and on 4 ranks, the golden dumps of
  section 11.5 (49 files) **bitwise equal** to the Q0 binary's (`golden.py compare`).
* **`CellFace` stays 32 bytes** (`static_assert`): `tpos2` would have made it 40 (`Face` was int-sized), costing 12.9 MB of mesh tables on the
  720 x 360 airfoil (+15 %; the face list is held twice, `faces` and the reused buffer `face_spare_`). `enum class Face` now has a one-byte
  underlying type (`Core.hpp`; nothing depended on its size). Memory on that airfoil (serial, 50 steps): mesh tables 86.7 -> 87.7 MB (the `ck`
  array), peak 357.6 -> 358.5 MB.
* **2-D cost** (min of 5 interleaved runs against the Q0 binary, 1 thread): the residual of `bench_cellflux 768 384 20` (the Q2-gate protocol)
  is unchanged: level-0 face engine 74.1 -> 73.4 ns/cell (0.99x), adapted 76.1 -> 75.1 (0.99x), the untouched uniform kernel 1.006x (the noise
  floor); the engine's stencil build 9.4 -> 10.0 ms (+6 %, `CellFlux` reads the mesh through the generalised lookups). On the serial 720 x 360
  airfoil (60 steps, a remesh every 2 steps) the adapt part 0.59 -> 0.63 s (+7 %, about 1.3 ms per remesh of 268 k leaves: the generic loops), the
  indicator 0.72 -> 0.69 s (the key is now inline), the time loop 7.59 -> 7.62 s (+0.4 %).
* **CI suite run locally** (`check_all.py --golden`, Windows MSVC): unit tests, regression serial and 4 ranks, `amr_mpi.py --check` (real MPI, 1-7
  ranks, adapting meshes: the partial-mesh and remesh paths of `CellMesh`), `ibm_ghost.py --check`, post-processing, MMS quick, golden compare:
  all passed. The other CI platforms (Linux x64 / ARM64, macOS) run on the push.
* **Not in Q1:** nothing reads a 3-D mesh yet (the engine, the indicator, `CellParallel` and the I/O stay 2-D; `use_amr` with `nz > 1` is still
  rejected). `CellBox` and `dilate_cells` stay 2-D until Q5/Q6.

## 13. Phase Q2 as built (2026-11-07): the convective residual of `CellFluxEngine` in 3-D

### 13.1 The 3-D uniform residual the level-0 tree reproduces (the list section 4.4 asked for)
Read in `Simulation::assemble_3d`, `MusclReconstruction::reconstruct_3d` / the fused line kernel and `RiemannSolver::face_flux(_batch)`:
* **Face states.** Interior faces: the four cells along the normal, `wl = c1 + 0.5 lim(c1 - c0, c2 - c1)`, `wr = c2 - 0.5 lim(c3 - c2, c2 - c1)` in
  primitive variables (WENO3-Z: `weno3z(c0, c1, c2)` / `weno3z(c3, c2, c1)`), then `prim_to_cons`; the same formulas on all three axes. First order:
  the cell states, no round trip. Boundary faces without a halo: the BC ghost state against the cell, both round-tripped (`prim_to_cons(cons_to_prim)`)
  with MUSCL. Periodic axes: wrap-around halos of depth 2 (the boundary faces are interior faces). `boundary_order = 2`: ghost slabs from
  `fill_ghost_slab` (layer d = the BC ghost of the cell at depth d), skipped on periodic and non-reflecting sides; along a normal only cells of that
  line are read, so there is **no edge or corner composition in the convective residual** (it matters for the viscous padded frame, Q4).
* **Riemann.** `face_flux(UL, UR, axis)` (HLLC, low-Mach, preconditioned; the shock blend reads only the face's states), axis 2 included; batches of
  four with AVX2, bit-identical to the scalar calls.
* **Divergence.** `r = (((0 - (fr Ax - fl Ax)) - (ft Ay - fb Ay)) - (fk Az - ff Az)) * inv_vol`, `Ax = dy dz`, `Ay = dx dz`, `Az = dx dy`,
  `inv_vol = 1 / (dx dy dz)`: each flux times its area, then the differences. The engine's gather does exactly this at level 0.
* **Speed.** The maximum of the face speeds (no viscous part in Q2).

### 13.2 What changed in the engine
* Neighbour codes `sides()` per leaf (`ns_`), virtual-cell nodes with up to 8 children (`std::array<int, 8>`, the first `nch_` used; the 3-D mean
  `(((a+b)+(c+d)) + ((e+f)+(g+h))) * 0.125`, the order of CellMesh's merge; the 2-D `((a+b)+(c+d)) * 0.25` unchanged), keys with `k`
  (`vkey_` = (l, i, j, k), `ikey_` = (l, i, j, k, side); with k = 0 the 2-D sort order is unchanged), the face's back / forward sides
  `2 axis` / `2 axis + 1` (z-faces run the same reconstruction, `boundary_order = 2` and coarse-side code).
* **Jump closure (b)+(c) in 3-D:** the interpolated cell is `Wc + dn sn + dt st + dt2 st2`, the limited slopes of the coarse leaf along the normal and
  the two transverse axes (`transverse1/2`), offsets +-0.25 coarse widths towards the fine leaf's half on each; the coarse side reads the mean of the
  8 children of the block across the face. 2-D keeps `Wc + dn sn + dt st`.
* **Gather in 3-D:** `G_s` = one `F * area`, or four summed in pairs in slot order `(F0 a + F1 a) + (F2 a + F3 a)` (a fixed order; four equal values
  add up exactly either way), `r = (((0 - (G1 - G0)) - (G3 - G2)) - (G5 - G4)) * (1 / volume)`. The 2-D branch is untouched.
* **Rejected on a 3-D mesh by `build()` until Q4** (clear messages): viscous terms, body force, sources, immersed bodies, mass-flow inlets,
  non-reflecting outlets; WENO5-Z as in 2-D. The partition / `localize` code is generalised (sides, children) but untested in 3-D until Q6.

### 13.3 Tests (`cpp/tests/test_amr_octree_flux.cpp`, 8 tests)
* `cellflux3d_level0_matches_uniform_residual_bitwise`: a level-0 octree against `Simulation::residual` of the 3-D uniform solver, **bit for bit**
  (residual and speed), 6 boundary sets (wall, transmissive, outlet, no-slip, moving wall, velocity inlet, periodic on 1-3 axes) x 6 schemes (first
  order, MUSCL Van Leer / Minmod / MC / Koren, WENO3-Z; HLLC, low-Mach, preconditioned) x `boundary_order` 1 and 2: 66 cases; plus two meshes above
  `kOmpMinWork` (parallel loops).
* `cellflux3d_invariant_axis_equals_the_2d_engine`: a field independent of axis a on the 2-D adapted tree extruded along a (periodic), for a = x, y,
  z: the 3-D residual equals the 2-D engine's to 1e-12 (relative), the a-momentum residual is exactly 0; first order, MUSCL with `boundary_order` 1
  and 2, WENO3-Z with walls. Over 1000 coarse-fine faces normal to the other axes; this is the oracle of the 3-D closure (a sign error in `dt2` fails it).
* `cellflux3d_free_stream_is_exact_across_level_jumps`: R == 0 exactly on random octrees (transmissive, partly and fully periodic; first order,
  MUSCL, WENO3-Z; `boundary_order` 1 and 2). One exception, documented in the test: WENO3-Z does not reproduce a constant to the last bit and a
  first-order boundary face uses the round-tripped state, so with `boundary_order = 1` the leaves on a physical side carry round-off (2e-15), as on a
  uniform mesh; every other leaf, the level jumps included, is exact.
* `cellflux3d_residual_is_conservative_on_adapted_meshes` (sum of R volume <= 1e-13 of its magnitude), `cellflux3d_incremental_build_equals_full_build_and_threads_agree`
  (`same_tables` after 6 random adapts; 1 thread and all threads give the same bits on a mesh above `kOmpMinWork`),
  `cellflux3d_rejects_what_it_does_not_do_yet`, `cellflux3d_level_pieces_equal_the_whole_residual` (the pieces of the subcycled advance:
  `compute_level_fluxes` of each of three levels + `gather_level` equal the slices of `residual()` bit for bit, `wave_speed` its speed; MUSCL and
  WENO3-Z with `boundary_order = 2`; they read the per-level lists `need_`, `lface_` and the owned ranges, which `residual()` does not).
* `cellflux3d_near_jump_truncation_table`: the residual's truncation error against `-div F` of a monotone polynomial field (exact by central differences
  of the physical flux), base n^d on [0, 4]^d with the box [1, 2]^d one level finer, mean normalised error within 2 base widths of the jump and on the
  coarse leaves more than 3 widths from it. Bands fixed before the first run: (1) far order (32 -> 64) >= 1.8 in 2-D and 3-D, (2) 3-D near-jump order
  >= the 2-D one - 0.3, (3) 3-D near / far ratio at 64 <= 3x the 2-D one.

  | | near jump n = 16 / 32 / 64 | orders | far n = 16 / 32 / 64 | orders | near / far at 64 |
  |---|---|---|---|---|---|
  | 2-D | 3.60e-2 / 2.47e-2 / 2.23e-2 | 0.54, 0.15 | 1.75e-4 / 3.76e-5 / 8.53e-6 | 2.22, 2.14 | 2610 |
  | 3-D | 3.54e-2 / 2.26e-2 / 1.87e-2 | 0.65, 0.27 | 1.40e-4 / 3.02e-5 / 7.05e-6 | 2.21, 2.10 | 2656 |

  All three bands pass. The far errors equal those of the uniform base mesh (checked with the box not refined): the jump's influence is local, as the
  stencil reach says. At the jump the residual's truncation error is O(1) in both dimensions (the usual behaviour of a non-conforming interface; what
  the solution converges at is the time-run question of Q3), and the 3-D closure is no worse than the accepted 2-D one. History of the measure, for the
  record: the first run failed band (1) in **both** dimensions (far order about 1); the cause was the measure, not the code (a field whose derivatives
  vanished at the domain edge made the limiter clip there, and a domain too small for the far region to hold the same kind of cell at every n); the
  field and the domain were changed, the bands were not.
* Mutations: the sign of `dt2` flipped is caught (invariance test); summing the four fine faces sequentially is not a defect (exact as well).

### 13.4 Gates
* **2-D per-leaf cost (the A1 gate, protocol of 11.5):** `bench_cellflux 768 384 20`, 1 thread, min of 5 interleaved runs against the Q0 binary:
  level-0 face engine 73.1 -> 73.0 ns/cell (**0.999x**), adapted 74.7 -> 74.1 (0.992x), uniform 0.998x (the noise floor): **passed** (limit 1.05x).
  The engine's stencil build 9.3 -> 10.1 ms (+8.6 %: wider keys and node records; Q1 alone was +6 %). Memory on the 720 x 360 airfoil (serial,
  50 steps): engine tables 183.5 MB as with Q0, mesh 87.7 MB (Q1's `ck`), peak 357.6 -> 358.7 MB.
* **2-D bits:** the 49 golden dumps of 11.5 bitwise equal to the Q0 binary's, and the quick sets with `FENICE_FULL_REBUILD=1` and
  `FENICE_NO_SIMD=1` as well (`golden.py compare`).
* Unit tests 146/146; the CI suite locally (`check_all.py --golden`: regression serial and 4 ranks, `amr_mpi.py --check` 1-7 ranks,
  `ibm_ghost.py --check`, MMS, post-processing): all passed.
* **Not in Q2:** time stepping on a 3-D tree (`CellAdvance`, registers of 4 per coarse side, the subcycled conservation and the time-run order: Q3),
  the physics of Q4, anything 3-D in the CLI.
