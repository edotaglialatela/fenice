# An unstructured-mesh solver: proposal

Status: proposal, nothing implemented, no decision taken. Whether to do it, and in which order against the species work
(`species-target-cases.md`), is the user's.

## Why, and why not

The solver is structured Cartesian with cell-based AMR and an immersed boundary for bodies. That is strong for shock-dominated flow
on simple domains and weak where the geometry must be resolved by the mesh: boundary layers on curved walls, complex shapes, meshes
made by an external mesher. An unstructured solver would read such meshes (`.msh` files) and give body-fitted walls.

What it costs: it is a second solver, not a generalisation of the first. The numerics of the Cartesian one depend on the axes
(`face_flux(UL, UR, axis, smax)`, directional MUSCL, the viscous cross-derivatives, the AVX2 kernels), and the AMR and the
parallel partition depend on the Morton ordering of square cells. The Cartesian solver would stay as it is, with its references
(38 bitwise states) untouched; the new one would be a separate path with its own verification.

What it does not give: AMR (no counterpart on unstructured meshes in this plan), the vectorised kernels, the bitwise parallel
guarantees until they are re-earned. For the shock and detonation cases the Cartesian solver is the better tool; the case for the
new solver is the body-fitted viscous wall.

A cheaper route to part of the same goal: improve the immersed-boundary wall treatment and read general polygons from a contour file
(today the shapes are a cylinder, a sphere, 4-digit NACA sections and an airfoil file). Compare before committing.

## What can be reused

* The Riemann solvers, through a rotated frame: rotate the left and right states into the face-normal frame, call the existing
  axis-0 flux, rotate the flux back. The flux function itself stays bit for bit the same.
* The equation of state, the time schemes (they act on a field and a residual), the parameters and case files, checkpoints, VTK
  (`.vtu`) output, the GUI as a client.
* The structure of the face-list engine of the AMR (each face computed once, added to both cells), as a template.

## What is new

* A mesh reader and the mesh data: cells (triangles and quads in 2-D), faces with area and normal, cell volumes and centroids, the
  neighbour lists; boundary patches mapped to the existing boundary conditions by name.
* Second-order reconstruction: a gradient (least squares or Green-Gauss) and a limiter. Which ones is a numerical choice and is the
  user's: options are to be presented with a verification of each (order on a smooth solution, behaviour at a shock) before one is kept.
* Viscous fluxes on non-orthogonal faces (face gradient with a correction).
* Parallel decomposition with a graph partitioner and halo layers; the deterministic-reduction rule (maxima only, no sums across
  ranks) carries over, since MPI must equal serial bit for bit.
* Verification from scratch: the manufactured solution (the generator would produce a source for a general mesh), the isentropic
  vortex on triangles and quads (observed order), and benchmarks with exact or reference data: a supersonic wedge (oblique shock
  angle), a supersonic cylinder (shock stand-off distance), and later a viscous flat plate or airfoil.

## Stages

1. 2-D, Euler, explicit, serial: reader, geometry, rotated-frame flux, reconstruction, boundary conditions, `.vtu` output.
   Gate: observed order on the vortex, the wedge angle, the cylinder stand-off, symmetry under mesh rotation.
2. Navier-Stokes (viscous fluxes, no-slip walls), then the existing time schemes. Gate: a viscous manufactured solution, a flat plate.
3. MPI with a partitioner. Gate: bit for bit against serial on several rank counts.
4. 3-D (tetrahedra, hexahedra, prisms for boundary layers).
5. Optional: GUI side (mesher launch, patch naming), species on top of the new solver.

Each stage ends with the full oracle set of the Cartesian solver untouched, and its own new references (`--create-missing`) only after
validation against the independent data.

## Questions

1. Which flows need it: viscous boundary layers on curved walls, complex geometry, or meshes that already exist? If none, the
   cheaper route above may be enough.
2. Order against the species work: they are independent (species first on the Cartesian line is proposed).
3. 2-D only at first, or 3-D from the start of the design (it changes the data structures, not the first stage)?
4. The reconstruction and the limiter (stage 1): a choice to be made after the options are compared.
