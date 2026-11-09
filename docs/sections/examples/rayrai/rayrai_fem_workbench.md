# Rayrai Example: FEM Workbench

![Linear elastic stress analysis in the Rayrai FEM Workbench](../../../../rsc/docs/image/rayrai/rayrai_fem_workbench.png)

Import a solid, select supports and loads, and inspect displacement, stress, and
strain. The example uses RaiSim's single-threaded linear static FEM solver and
renders its results with Rayrai. It opens with a solved aluminium cantilever.

Use this page's **Run on web** button to run the same C++ workbench in the browser.
The site uses its existing `web-examples/example.html?example=rayrai_fem_workbench`
launcher. Uploaded models and calculations remain on the visitor's computer.

## Workflow

1. Drop a file on the viewport or choose **Import**. Set metres per file unit
   before importing a surface mesh or BREP: use `0.001` for millimetres. STEP/IGES
   read their declared units. Element size is in metres. The camera orbits the
   imported bounds center, and directional lighting distinguishes solid faces.
   Surface files appear immediately without volume meshing. CAD opens after surface
   tessellation; it does not generate tetrahedra until you solve.
2. Enter Young's modulus, Poisson's ratio, and density. The initial values are
   representative aluminium properties; enter the properties of your material.
3. Choose **Triangle**, **Planar patch**, **Smooth surface**, or **Box** to select
   a region. Planar selection stays on the connected patch. Smooth selection
   follows adjacent triangles within 35 degrees, including cylindrical walls. Box
   selection includes nodes through the solid. Shift adds to the selection.
   Plane selection provides exact minimum/maximum faces or a coordinate and tolerance.
4. Add fixed supports, axis-specific supports, or prescribed displacements.
   Uncheck unconstrained axes for a roller support. Add total forces, surface
   traction, inward pressure, or gravity. Conditions can be enabled, edited, and deleted.
5. Set **Max element edge (m)**, then choose **Solve**. The workbench generates a
   volume mesh from the original geometry using the current settings, transfers
   supports and loads, and solves. Every tetrahedron edge is bounded by the requested
   maximum; small features and geometric accuracy can require smaller elements.
   Change the size and solve again to regenerate a finer or coarser mesh. **Live
   solve** recalculates edits when the current mesh is valid. Changing only loads
   and prescribed values reuses the stiffness factorization or iterative preconditioner.
6. Select a stress or strain component, principal stress, von Mises stress, or
   displacement magnitude. Deformation magnification changes the display only.
7. Save a `.rfem` project, including its original geometry, meshing settings,
   material, conditions and generated mesh, or export
   VTK displacement/reaction point fields and stress/strain element fields.
   Browser exports download files; desktop exports use the entered paths.

Under **Mesher / file formats**, **Surface error, 0=auto (m)** controls geometric
approximation separately from element size. Automatic accuracy is the smaller of
1% of the requested edge and 0.1% of the original geometry's bounding diagonal.
CAD is tessellated again at that accuracy on Solve. A tighter error can increase
meshing time even with a coarse element size.

Imported volume meshes retain their original nodes, part labels and internal
interfaces. Solve refines the original volume to meet a smaller maximum edge; a
larger setting retains an already finer supplied mesh. Surface and CAD imports
can be regenerated at either finer or coarser sizes.

Surface conditions retain their original geometric regions and capture accuracy
across remeshing and project reload. Transfers check coverage in both directions
so a load or fixture cannot silently lose an unresolved part of its region.
A partial patch boundary must conform to the generated mesh: select a complete
surface, split the CAD face, or supply a conforming volume mesh if the workbench
cannot transfer it exactly. Node-only selections represent discrete locations;
use a surface selection for continuous support. Failed transfers leave the
previous model intact.

## Files

Both builds read linear tetrahedral volume meshes in ASCII Gmsh 2.2/4.1 (`.msh`),
Medit (`.mesh`), and legacy ASCII VTK (`.vtk`). Higher-order tetrahedra and other
volume element types are rejected. Binary variants must be exported as ASCII.

Watertight, consistently oriented STL, OBJ, PLY, glTF/GLB, and DAE surface meshes
are tetrahedralized before solving. Include glTF buffers and other companion
files when choosing browser files. The browser uses the separately loaded,
single-threaded fTetWild WebAssembly mesher (MPL-2.0), built from the same pinned
source and conservative envelope optimization as the native helper. Dependency
notices, modified MPL source, and build recipes are included in the browser bundle. Raw STL does not preserve assembly
part identities; use STEP solids or a volume mesh with distinct parts for
connectors between contacting bodies.

Both builds import STEP, IGES and BREP. The browser translates CAD with the
separately loaded occt-import-js/Open CASCADE WebAssembly library and meshes its
surface with fTetWild. The unmodified translator and its LGPL/OCCT notices and
source references are included in the bundle. Each CAD solid is meshed
separately and retained as a distinct region, including touching assembly parts.
Desktop uses the bundled native fTetWild helper and shared Open CASCADE libraries;
Gmsh and Python are unnecessary. The source build pins the same fTetWild revision
as the browser and installs dependency notices. STEP and IGES units are read from
the CAD file and converted to metres. BREP has no unit metadata; set metres per
file unit before importing it.

External Gmsh remains available under **Mesher / file formats → Use external
Gmsh**. Its path can be entered there or set with `RAISIM_FEM_GMSH`. For a custom
native helper location, set `RAISIM_FEM_MESHER`. Import errors distinguish a missing
executable or script interpreter from a geometry failure and retain the mesher log.

Native SolidWorks `.sldprt` and `.sldasm` files require a
separate proprietary-format translator; SolidWorks can export STEP or STL.

## Run on desktop

Build the RaiSim SDK with `RAISIM_BUILD_FEM_MESHER=ON` (the default for native
builds with CMake 3.20 or newer). This builds the pinned meshing dependencies from
source and installs `bin/raisim_fem_mesher`, its shared CAD libraries and
notices. Linux builds use Clang 20. The workbench finds the SDK helper
automatically; a sibling RaiSim source checkout also supports building it locally.
`RAISIM_FEM_MESHER_EXECUTABLE` selects an existing helper at example configuration.

From the `raisim2Lib` directory on Linux:

```bash
cmake -S examples -B build-examples -DCMAKE_PREFIX_PATH=/path/to/raisim-sdk \
  -DCMAKE_C_COMPILER=clang-20 -DCMAKE_CXX_COMPILER=clang++-20 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-examples --target rayrai_fem_workbench -j12
./build-examples/rayrai_fem_workbench
./build-examples/rayrai_fem_workbench part.step
```

On Windows the executable is `build-examples/bin/rayrai_fem_workbench.exe`.
`--self-test` checks the workbench model, load edits, factor reuse, project
round-trip, VTK export, and translated/tiny/large mesh camera framing without a window. `--frames N --capture image.png`
captures the desktop app for a fixed run. `--test-drop file` exercises its SDL
file-drop path.

The native integration suite checks surface validity, positive tetrahedral volumes,
CAD units, independent assembly regions, and failure cleanup. Run these from the
RaiSim source checkout:

```bash
python3 test/fem/test_native_mesher.py --mesher /path/to/raisim_fem_mesher -v
python3 benchmark/fem_mesher_benchmark.py --mesher /path/to/raisim_fem_mesher
```

The native helper uses conservative geometric certificates to avoid dense envelope
sampling when containment can be established directly. Uncertain cases retain the
original sampler and tolerance. Use `--compare-sampling` to compare both paths;
the benchmark checks identical deterministic native meshes. `--geometry all`
adds a hollow cylinder, sphere and supplied volume refinement to the STEP assembly.
`--preview` also measures surface-only CAD preparation with zero tetrahedra. An optional
`--gmsh /path/to/gmsh` includes Gmsh. Results include time, volume error and
minimum tetrahedron mean ratio, and actual maximum edge. Requested element sizes can produce different
meshes across backends. Timed-out benchmark cases and accuracy-target misses are
reported explicitly. See the [native meshing measurements](../../LinearElasticityBenchmarks.md#native-meshing)
for timings and geometric errors.

CMake target: `rayrai_fem_workbench`. Source:
`examples/src/rayrai/fem/rayrai_fem_workbench.cpp`.

## Model limits

This is small-displacement, small-strain isotropic linear elasticity, with
four-node tetrahedra. It does not model plasticity, fracture, large rotations,
or contact between separate components. The supports and connections must eliminate all unrestrained rigid modes
of the complete assembly. Zero-stiffness mechanisms and invalid meshes
produce errors. Linear tetrahedra require mesh refinement for bending and can
lock for nearly incompressible materials. The material editor warns above
Poisson ratio 0.45. See the [bending convergence measurements](../../LinearElasticityBenchmarks.md#bending-convergence)
before choosing a mesh. Local stress peaks at sharp corners,
point loads, and ideal supports require convergence checks.

See [Linear elasticity](../../LinearElasticity.md) for the C++ API and validation.

## Fixture methods

Use **More fixture methods** after selecting a region. Selection can use a
triangle, a connected planar patch, a smooth surface, a box, a coordinate slab, or a cylindrical
surface defined by its axis, center, radius, length and tolerance.

| Method | Behavior |
| --- | --- |
| Fixed / axis displacement | Restrain chosen global axes, with optional nonzero motion. |
| Directional / symmetry | Prescribe displacement along an arbitrary direction. |
| Normal roller | Restrain selected-surface normal motion; sharp edges get separate normal constraints for each face group. Use a cylindrical bearing when roll must be free. |
| Slider / guide | Restrain two directions perpendicular to the guide axis. |
| Cylindrical bearing | Restrain radial motion and optionally the average axial center; roll is free by default. Other supports must remove any remaining global rigid rotation for a unique static solution. |
| Grounded bolt hole | Restrain shaft radial/tangential motion and a separately captured axial seat region. |
| Bolt connector | Connect two captured head/nut surface patches with diameter, material, grip length and reference preload. Reports force and end moments. |
| Remote displacement | Impose translation and small rotation around an entered center. |
| Ball / spherical support | Restrain patch-average translation while permitting rotation and local deformation. |
| Elastic foundation | Distribute a specified total directional stiffness over a surface. |
| Bonded surface tie | Project slave nodes onto a captured master triangle surface within tolerance and couple their displacements. |

For a bolt connector, capture the first patch when adding the condition, select
the second region, then use **Capture second region** in the condition editor.
Grounded bolt holes use the second region for the axial seat. Bonded ties use it
as the master surface. Save all fixture definitions in the RFEM project. Version 4
projects retain original geometry, settings and geometric selection references for
remeshing; versions 1–3 remain readable. Legacy projects retain their existing
mesh resolution. Tie projections are cached across load edits.

Bolt preload means tension at zero relative head motion; the resulting tension
also depends on the compliance of the connected parts. These are linear beam
connectors, without thread contact, friction or joint separation. Bonded ties
use node-to-surface interpolation and can have interpolation error for unrelated
triangulations; refine the interface and check convergence. Keep tied surfaces
coincident; a finite projection gap approximates their connection.
