# WENO5-Z on the adaptive mesh (2-D and 3-D): plan

Status: **not started; the owner decides whether to do it.** It is the one open feature of the adaptive engine: `reconstruction = weno3` and
`boundary_order = 2` already run with AMR in 2-D and in 3-D (see the AMR page, `docs/source/modules/AMR.md`), `weno5` is rejected with AMR
(`CellFluxEngine`, `Params.cpp`). The same file lists the smaller open items of the adaptive engine at the end.

## Where the engine stands

* Every interior face has a fixed four-cell stencil `(c0, c1, c2, c3)` along its normal, built once per mesh change (`stencil_[f]`). A cell
  beyond a coarser neighbour is read by injection, one beyond a finer neighbour as the mean of the leaves under it (a virtual cell, recursive).
  The rows of the primitive array `W_` are the leaves, then the averaged virtual cells, then the interpolated ones (the coarse value
  at the centre of the fine cell that would sit across a jump); `prepare` refreshes them each stage. At `boundary_order = 2` `CellFluxEngine::fluxes`
  evaluates the ghost layers where they are read (`ghost_state` of the boundary leaf, and of the leaf or virtual cell behind it; no table).
* WENO3-Z reads exactly the four cells of that stencil, so it needed no new data. On a coarse-fine face both sides use the WENO formula
  with the interpolated / averaged value across the seam; the interpolated value still uses the `muscl_limiter` slopes of the coarse leaf.
  WENO faces take the scalar flux path (the batch kernel and its AVX2 twin are written for the limiter formula).
* Time: recursive subcycling by 2 with flux registers on the coarse-fine faces and no time interpolation; SSP-RK3 uses the same registers with the
  stage weights.
* The MPI halo is built from the leaves `face_dependencies` says the faces read; `CellParallel::halo_reach` measures how far those reads reach.
  A leaf outside a rank's share becomes a dropped virtual cell without an error, so a missing ring would make the result depend on the rank count.
* Invariants to keep: a level-0 mesh gives the uniform solver's residual bit for bit (`celladvance_level0_steps_match_uniform_steps_bitwise`), and
  the state does not depend on the rank count (`amr_mpi.py --check`).

## What WENO5-Z needs

1. **A six-cell stencil.** `stencil_` widens from 4 to 6 entries, the virtual cells must exist two cells beyond the face on each side (a cell
   beyond a neighbour that is itself beyond a jump can be two levels away, which the recursive virtual cell handles in principle), and the neighbour
   codes need a second ring. In 3-D the same per face along its normal axis; the octree keys give the neighbours of the neighbour.
2. **Ghost layers of depth three.** At a boundary the first two interior faces read ghost layers (the uniform solver builds `Dh = 3` layers for
   WENO5): the states `G_s(U[c])`, `G_s(U[n1])`, `G_s(U[n2])` per boundary leaf, built with the same rule so the bits can match; a
   moving wall keeps the internal energy of the interior cell at `boundary_order = 2`.
3. **A halo three leaves deep.** The uniform solver runs WENO5 serial because its MPI halo is two cells deep. Either the uniform path gets a
   three-deep halo too (a change in `Schemes` and the halo exchange whose size is not looked at; recommended, so the two solvers agree), or the
   adaptive path alone does and the docs say so. Whether `halo_reach` and the share construction raise the reach by themselves for a six-cell stencil is
   to be tested (`halo_reach_bounds_the_halo_of_any_partition`, the few-roots MPI regression), not assumed.
4. **Time step.** WENO5-Z with SSP-RK3 needed CFL 0.3 on the near-vacuum tube in the uniform solver; how the adaptive step maps the CFL to the
   time step of each level is to be checked. With a wider stencil more of the reconstruction reads the coarse level at the old time at a seam:
   measure it on Sod and the vortex before claiming anything.
5. **Params, GUI and docs**: remove the AMR (and, with 3, the MPI) rejection of `weno5`, update the scheme table and the parameter page.

## Verification (bands fixed before the first run)

1. Bitwise at level 0 against the uniform solver (extend `celladvance_level0_steps_match_uniform_steps_bitwise` to `weno5` and `boundary_order = 2`).
2. Bitwise ranks: regression cases `amr_weno5_*` in 2-D and 3-D, serial and several ranks; `amr_mpi.py --check`.
3. Conservation to 1e-14 in a closed box with remeshing (the registers guarantee it while the same face flux feeds both sides).
4. Order on the manufactured solutions on an adaptive mesh with a static refined strip (`amr_force_box`): the uniform bands are L2 order at least 1.7 and
   L-infinity at least 1.3 on the finest pair; for the adaptive runs L2 at least 1.5 and L-infinity at least 1.0 (a seam allowance), a result below
   them is reported as below, not moved.
5. Cases: Sod and the vortex with WENO5-Z on an adaptive mesh against the uniform solver on the finest grid; the lid-driven cavity with
   `boundary_order = 2`.

Size: larger than WENO3-Z, and the least certain part is the time seam. Order of work: stencil and virtual cells, ghost rows, halo (with the uniform
path), tests, then the cases. A vector twin of the WENO face kernel only if the profile shows the scalar path in the way.

## Other open items of the adaptive engine

* **MPI remesh cost.** On 8 ranks the share rebuild is about half of the remesh (3.9 s of 7.5 s on the 72 k-leaf sphere): editing the share
  instead of rebuilding it would help multi-node runs. On one node 16 OpenMP threads beat 8 ranks (10.2 s against 16.1 s per 200 steps there), so
  the OpenMP time step matters more than the remesh. A border-only fetch was judged not worth it (the region is mostly halo).
* **OpenMP time step.** Virtual cells are evaluated for every level's call (about 15 % of the step; a per-level read list would gain a few
  percent on three levels and needs a parallel rebuild at each remesh); an AVX2 twin of the 3-D viscous kernels.
* **Validation.** The 3-D ghost-cell wall is validated against itself (self-convergence) and the octree against the uniform solver only, not against an
  external reference; 20 -> 40 cells/D octree runs (`cases/validation/amr/octree_q8.py run --include-40`, 1-2 h each); the ghost wall is first order on
  curved walls (see `docs/proposals/ibm-wall-treatment.md`).
* **Cost figure.** The octree-against-uniform cost ratio in the AMR page predates the 1.7x faster OpenMP step; `octree_q8.py timing` re-measures it.
