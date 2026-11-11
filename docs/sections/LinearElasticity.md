# Linear elasticity

`raisim::fem::LinearElasticity` solves the static equilibrium of a small-strain,
isotropic elastic solid using four-node tetrahedra. It is independent of a
`raisim::World` and the time-stepped deformable-body solver. The public header is
`raisim/fem/LinearElasticity.hpp`.

```cpp
#include "raisim/fem/LinearElasticity.hpp"

using raisim::fem::LinearElasticity;
using raisim::fem::TetMesh;

LinearElasticity solid(TetMesh::box({1.0, 0.2, 0.2}, {20, 4, 4}));
solid.setMaterial({70e9, 0.3, 2700.0});
for (unsigned int i = 0; i < solid.mesh().vertices.size(); ++i) {
  if (solid.mesh().vertices[i][0] == 0.0) solid.fix(i);
}
solid.setGravity({0.0, 0.0, -9.81});
const auto& stats = solid.solve();
solid.writeVtk("beam.vtk");
```

All quantities use SI units: metres, newtons, pascals, and kg/m³. Imported meshes
can be scaled with `TetMesh::load(path, metresPerFileUnit)`. Element ordering and
boundary-face indices remain stable during load edits. Element results use Voigt
order `xx, yy, zz, xy, yz, zx`; shear strains are engineering shear, twice the
corresponding strain tensor entries. Stress shears are physical stress components.

## Supports and loads

`fix(node)` constrains all three displacement components to zero.
`prescribe(node, axis, value)` sets one displacement component, including a
nonzero value; other axes remain free. `release(node, axis)` removes that support.

`addNodalForce` applies a force in newtons. `addSurfaceTraction` integrates a
uniform traction in Pa over a boundary triangle, distributing its resultant
equally among the triangle's three nodes. Positive `addSurfacePressure` acts
inward. `setGravity` applies acceleration using the element densities and
volume-lumped nodal masses. `clearLoads` clears forces, surface loads, and gravity.

`setElementMaterials` accepts one material per element for heterogeneous solids.
`mesh().regions` preserves Gmsh 2.2 physical tags, Gmsh 4.1 volume entity tags, or
Medit reference numbers for applications that assign materials by region.

After a successful solve, `displacements`, `reactions`, and `elementResults`
provide results. Element fields include stress, strain, principal stresses,
von Mises stress, and strain energy. Getters reject results invalidated by a
setter. Reactions include the residual on supported degrees of freedom and physical spring reactions. For multipoint constraints this includes constraint forces on the participating nodes; internal tie forces cancel in the global resultant.

## Efficiency and validation

Shape gradients and volumes are computed once. The solver assembles the lower
triangle of the sparse stiffness using analytical 3×3 element blocks, eliminates
prescribed degrees of freedom exactly, equilibrates the reduced matrix, and
uses sparse LDLT for small models and bolt connectors. At 5,000 free degrees of
freedom, models without bolts automatically use conjugate gradients preconditioned
by smoothed-aggregation algebraic multigrid (multigrid CG), including directional
fixtures, multipoint constraints and elastic foundations. The multigrid hierarchy
aggregates the unknowns of neighbouring vertices and spans each aggregate's
coarse space with the six rigid-body modes; one symmetric V-cycle with
Gauss-Seidel smoothing preconditions each CG iteration. Iteration counts stay
almost constant as meshes grow: on a 258,000-unknown CAD part, multigrid CG needed
62 iterations where incomplete-Cholesky CG (ICCG) needed 3,868, and the solve took
3.4 s instead of 60 s. Both stop at the same relative residual, 1e-11.
Axis-aligned fixture equations use Cartesian elimination.
Load, prescribed motion, spring rest displacement and density-only edits reuse the
factorization or preconditioner. Material stiffness edits reuse symbolic analysis
when the reduced sparsity pattern is unchanged. A cache of up to eight independent
load responses accelerates repeated changes of load direction. Solver instances
are single-threaded.

`statistics()` reports assembly, factorization, solve-and-recovery timings,
matrix/factor sizes, cache reuse, strain energy, and a dimensionless backward
equilibrium error based on `|f| + |K| |u|`, plus a load-relative force residual.
Every solver path checks rigid modes of face-connected components, compatibility
at shared nodes, and global force/moment balance. A free component is identified
by component number and mesh region. A pivot or converged iteration alone does
not establish that a structure is adequately supported. `linear_fem_benchmark` measures
construction and repeated load cases, checking axial extension and factor reuse.

Tests cover uniaxial tension and Poisson contraction, affine strain patches,
engineering shear stress, hydrostatic pressure, reaction forces and moments,
gravity, layered materials, prescribed motion, factor reuse and failure retries,
unsupported components, invalid meshes, overflow, import formats and locale independence.

The implementation was written for RaiSim after studying
[MFEM's linear elasticity example](https://github.com/mfem/mfem/blob/master/examples/ex2.cpp)
and elasticity integrator (BSD-3-Clause), and
[SOFA's tetrahedral FEM formulation](https://github.com/sofa-framework/sofa/tree/master/Sofa/Component/SolidMechanics/FEM/Elastic)
(LGPL-2.1-or-later). Their implementation code is not incorporated into this solver.

Try the [Rayrai FEM Workbench](./examples/rayrai/rayrai_fem_workbench.md) to open
STEP, IGES or BREP solids and apply supports and loads to their faces interactively
on desktop or on the website.

## General fixtures and connectors

`addLinearConstraint(terms, value)` imposes `sum(weight*u[node][axis]) = value`
exactly using a sparse nullspace reduction. `addDirectionalConstraint` is a
convenience for an arbitrary unit direction. Consistent redundant equations are
accepted; conflicting values fail. Constraint value edits reuse the reduced
factorization. The workbench builds symmetry, rollers, guides, remote motion,
bearings, spherical supports and bonded surface ties from these equations.

`addElasticSupport(node, direction, stiffness, rest)` adds a physical directional
spring. `setElasticSupportRestDisplacement` changes its rest position without
rebuilding stiffness. `setElementMaterials` replaces the element material array;
`setMaterial` restores one uniform material for the entire mesh. `addBolt(headPatch, nutPatch, properties)` couples two noncollinear
surface patches using weighted translation and rotation averages. It transfers
axial force, shear, bending and torsion using a circular Timoshenko beam, without
making the patches rigid. Results include connector force, physical head/nut
moments, axial extension and energy. All connected bodies must collectively
have their six global rigid modes restrained.

Positive bolt preload is a reference-state tension: it produces that tension
at zero relative head motion. Equilibrium tension changes with compliance of
the connected parts. It does not model tightening sequences, thread stresses,
friction, separation or unilateral contact. Bonded ties are bilateral projected
node-to-surface displacement coupling, not mortar contact. Use coincident
interfaces: a tolerated geometric gap approximates the displacement connection
and does not preserve rigid rotations across the gap.

`setSolverBackend(Automatic/SparseLDLT/IncompleteCholeskyCG/AlgebraicMultigridCG)`
controls the backend. Both iterative backends accept Cartesian and general linear
constraints and springs; bolt connectors use LDLT. Iteration counts and the
selected backend (`statistics().iterative`, `statistics().multigrid`) are reported
in statistics. Independent load directions initially require iterations; later
linear combinations can reuse the response cache. Direct solvers can still be
faster for many unrelated loads. See the reproducible
[solver benchmarks](./LinearElasticityBenchmarks.md).

`setProgressCallback(callback)` reports the progress of `solve()`: the callback
receives a `SolveStage` (`Assembly`, `Factorization`, `Iteration`, `Results`) and
the completed fraction of that stage in [0, 1]. Iterative solves report their iterations as the
logarithmic reduction of the residual toward the tolerance, at most every 50 ms;
direct factorizations report only their start and end. The callback runs on the
thread that calls `solve()` and does not change the solution. An exception thrown
by the callback stops `solve()` and propagates, which lets an application cancel
a long solve; the next `solve()` repeats the interrupted work.

```cpp
solid.setProgressCallback([&](LinearElasticity::SolveStage stage, double fraction) {
  if (cancelled) throw std::runtime_error("Solve cancelled");
  showProgress(stage, fraction);
});
```

## Error estimation and adaptive refinement

`raisim/fem/ErrorEstimation.hpp` estimates the discretization error of a solved
model. `estimateRecoveryError(mesh, elementResults, materials)` recovers a
continuous stress by volume-weighted averaging of the element stresses at each
vertex, separately per region, and integrates the energy of its difference to the
element stress exactly over every tetrahedron (a Zienkiewicz-Zhu estimate).
It returns the squared error of each element and `relative()`, the error relative
to the estimated exact energy norm. On a cantilever the estimate is within 30% of
the true energy error and approaches it as the mesh is refined.

`refinementTargets(mesh, estimate, minimumEdge, fraction, maximumCells)` marks the
elements with the largest errors that together hold `fraction` of the error
(Doerfler marking, default one half) and gives them half their longest edge as a
target. `raisim/fem/MeshResolution.hpp` then bisects them:
`refineElements(mesh, elementTargets, maximumCells, &parents)` splits every
tetrahedron longer than its target across its longest edge, and splits longer
neighbouring edges first, which keeps the mesh conforming and bounds element
quality. Existing vertices keep their indices, regions are copied, and `parents`
lists the two end vertices of every new vertex, so boundary conditions and fields
can be mapped exactly. `refineToEdgeTargets` takes per-vertex targets instead, and
`refineToMaximumEdge` refines uniformly.

```cpp
solid.solve();
for (int step = 0; step < 3; ++step) {
  const auto estimate = raisim::fem::estimateRecoveryError(
      solid.mesh(), solid.elementResults(), {material});
  if (estimate.relative() < 0.05) break;
  auto mesh = raisim::fem::refineElements(
      solid.mesh(), raisim::fem::refinementTargets(solid.mesh(), estimate, 0.));
  // Build a new LinearElasticity on mesh, apply the same supports and loads,
  // and solve again.
}
```

Where the error concentrates, the adaptive mesh is far smaller than a uniform one
of the same accuracy: under a pressure patch on a block, three refinement steps
reach a lower energy error with 11,620 tetrahedra than uniform refinement with
41,472. A sharp re-entrant corner or a fully clamped edge has infinite stress,
and refinement keeps concentrating there; limit the steps or the element size.

The optional `RAISIM_FEM_REFERENCE_TESTS` tests compare displacement, reaction,
engineering strain, stress, principal stress, von Mises stress and energy against
scikit-fem/SciPy on identical distorted and rotated meshes, with mixed materials,
nonzero displacement values, gravity, pressure, traction and multipoint equations.
The optional `RAISIM_FEM_MFEM_BENCHMARK` builds an independent MFEM comparison.
These libraries are validation dependencies, not RaiSim runtime dependencies.
