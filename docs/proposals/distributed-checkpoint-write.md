# Writing the checkpoint of a distributed AMR run

Status: proposal, nothing implemented. The restart already reads only each rank's share (`3dbadeb`); this is about the write.

## What happens today

`Simulation::save_checkpoint` (also `export_state` and the final dump) takes the whole leaf list through `CellParallel::collect(..., root_only = true)`: every rank sends its owned leaves (8 doubles per leaf: level, i, j, five conserved variables), rank 0 sorts them by (level, Morton key) and writes one `.npz`. The other ranks wait.

Counted from the code (not measured with a profiler):

* **Memory on rank 0**, on top of its own share: about 130 to 140 bytes per leaf at the peak (the receive buffer of 64 bytes, the sort keys and order of 16, the sorted level/i/j/U arrays of 52, then the double copies of the three index arrays and the `.npy` bodies at write time). 1 million leaves cost about 0.14 GB, 10 million about 1.4 GB.
* **File size limit.** `write_npz` stores entries without zip64 and refuses an archive of 4 GB. The checkpoint holds 56 bytes per leaf (`U` 32 + three index arrays 24), so the limit is about 76 million leaves. The reader accepts zip64; only the writer does not.
* **Time**: the transfer is one `allgatherv` of 64 bytes per leaf and one sequential write; for the runs made so far (the airfoil at 268 thousand leaves) it is not visible in the profile.

So the limit is real but far above what has been run (the largest AMR mesh so far has 268 thousand leaves). It will matter above roughly 10 million leaves.

## Options

1. **Leave as is and state the numbers** (this file). No code, no risk.
2. **Stream through rank 0, same file.** Ranks send their sorted blocks one after the other (or in bounded chunks); rank 0 writes the entries incrementally, accumulating the CRC and patching the local headers, with zip64 sizes. The file is byte-identical to today's, restart is unchanged, and the peak on rank 0 falls from about 135 B per leaf to the size of one chunk. Output is still serialised through rank 0 (bandwidth, not memory). Needs a streaming zip writer in `IO.cpp` and a global order that is known before the data arrive (counts per level from an `allgather`, then each rank's rows follow from the sort key).
3. **Parallel write of one file** (MPI-IO at precomputed offsets, CRC combined from per-rank CRCs). Same bytes as option 2, with the writing spread over ranks. Needs MPI-IO in the communicator layer, which does not exist yet, and a serial fallback.
4. **One file per rank plus a manifest.** Scales best, but is a format change and a restart on another rank count has to read several pieces; it also touches every tool that reads a checkpoint (hot start, GUI, post-processing).

## Recommendation

Option 1 now; option 2 when a run above a few million leaves is planned (it fixes both the memory and the 4 GB limit without changing the format). Options 3 and 4 only if the serialised write ever shows up in a profile.
