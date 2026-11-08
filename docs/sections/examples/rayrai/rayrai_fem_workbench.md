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
   before importing a mesh: use `0.001` for millimetres. Element size is in metres.
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
5. Choose **Solve**. **Live solve** recalculates after edits. Changing only loads
   and prescribed values reuses the stiffness factorization or iterative preconditioner.
6. Select a stress or strain component, principal stress, von Mises stress, or
   displacement magnitude. Deformation magnification changes the display only.
7. Save a `.rfem` project, including its mesh, material and conditions, or export
   VTK displacement/reaction point fields and stress/strain element fields.
   Browser exports download files; desktop exports use the entered paths.

## Files

Both builds read linear tetrahedral volume meshes in ASCII Gmsh 2.2/4.1 (`.msh`),
Medit (`.mesh`), and legacy ASCII VTK (`.vtk`). Higher-order tetrahedra and other
volume element types are rejected. Binary variants must be exported as ASCII.

Watertight, consistently oriented STL, OBJ, PLY, glTF/GLB, and DAE surface meshes
are tetrahedralized before solving. Include glTF buffers and other companion
files when choosing browser files. The browser uses the separately loaded,
single-threaded fTetWild WebAssembly mesher (MPL-2.0); its license and source
reference are included in the browser bundle. Raw STL does not preserve assembly
part identities; use STEP solids or a volume mesh with distinct parts for
connectors between contacting bodies.

Both builds import STEP, IGES and BREP. The browser translates CAD with the
separately loaded occt-import-js/Open CASCADE WebAssembly library and meshes its
surface with fTetWild. The unmodified translator and its LGPL/OCCT notices and
source references are included in the bundle. Each CAD solid is meshed
separately and retained as a distinct region, including touching assembly parts. Desktop uses a separately installed
Gmsh executable. Put `gmsh` on PATH, set `RAISIM_FEM_GMSH`, or enter its
path under **Mesher / file formats**. STEP and IGES units are read from the CAD
file and converted to metres. BREP has no unit metadata; provide geometry in metres. Native SolidWorks `.sldprt` and `.sldasm` files require a
separate proprietary-format translator; SolidWorks can export STEP or STL.

## Run on desktop

```bash
cmake --build build-examples --target rayrai_fem_workbench -j 12
./build-examples/examples/rayrai_fem_workbench
./build-examples/examples/rayrai_fem_workbench part.msh
```

On Windows the executable is `build-examples/bin/rayrai_fem_workbench.exe`.
`--self-test` checks the workbench model, load edits, factor reuse, project
round-trip and VTK export without a window. `--frames N --capture image.png`
captures the desktop app for a fixed run. `--test-drop file` exercises its SDL
file-drop path.

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
as the master surface. Save all fixture definitions in the RFEM project. New version 3 projects store
boundary triangles by their vertex IDs, and older version 1/2 projects remain
readable. Tie projections are cached across load edits.

Bolt preload means tension at zero relative head motion; the resulting tension
also depends on the compliance of the connected parts. These are linear beam
connectors, without thread contact, friction or joint separation. Bonded ties
use node-to-surface interpolation and can have interpolation error for unrelated
triangulations; refine the interface and check convergence. Keep tied surfaces
coincident; a finite projection gap approximates their connection.
