###################################
Server Example: Deformable Objects
###################################

Overview
========
Drapes a soft cloth over a raised static sphere and, to the side, builds a
stack of randomly oriented deformable cubes from OBJ meshes. All cubes use
resampled surface particles with automatically generated internal struts, so
they share the same volumetric behavior.

Use this example as the starting point for soft-body setup. It shows explicit
cloth topology and deformable-object construction from closed OBJ meshes. See
:doc:`../../DeformableObject`.

.. image:: ../../../../rsc/docs/image/deformable_objects.png
   :alt: deformable_objects example
   :width: 100%

Target
======
CMake target: ``deformable_objects``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/deformable_objects

On Windows, run ``deformable_objects.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port
8080. The simulation starts once a viewer connects and the example exits when
the viewer disconnects.

Details
=======
- Demonstrates ``World::addDeformableCloth`` with explicit vertices and
  triangle indices.
- Demonstrates OBJ particle generation and automatic internal struts.
- Generates deterministic random cube orientations at startup, so repeated runs
  show the same pile while still exercising non-axis-aligned contacts.
- Uses ``MeshParticleOptions::spacing`` so RaiSim resamples the mesh surface and
  builds a dense particle/sphere contact proxy instead of relying only on the
  raw OBJ vertices. During mesh loading, RaiSim also raises the deformable
  collision radius to at least ``0.58 * spacing`` so the generated spheres cover
  the surface without particle-scale holes.
- Uses ``distanceCompliance`` and ``bendCompliance`` for the cloth, and
  ``distanceCompliance`` plus internal struts for the mesh objects.
- Tunes the cloth to be highly deformable, with high bend and stretch
  compliance so it visibly sags and wraps around the sphere.
- Uses identical mesh settings for every cube and dedicated
  ``ground``/``deformable_cube`` and ``deformable_cube``/``deformable_cube``
  material pairs.
- Keeps the generated cloth and cube particle counts moderate so the example
  remains interactive while still showing visible deformation.
- Writes temporary closed cube OBJ files to the system temporary directory at
  startup and removes them on exit, so no extra mesh asset is required.

Construction modes
==================
The example covers:

- **Cloth**: user-provided vertices and triangle indices define a sheet. This
  is the right model for flags, hanging cloth, and thin surfaces.
- **Mesh object**: particles are generated from a closed OBJ mesh and
  connected with internal struts. This is the starting point for deformable
  objects that should compress and spring back.

Material parameters
===================
The material controls the XPBD/PBD response:

- ``distanceCompliance`` controls stretch resistance.
- ``bendCompliance`` controls bending resistance for cloth-like surfaces.
- ``distanceCompliance`` also controls the elastic response of the mesh-based
  cubes in this example.

Lower compliance means stiffer constraints. More particles or lower compliance
make each step more expensive, so start performance comparisons at a lower
resolution.

The cloth in this example uses deliberately soft values so it visibly sags and
wraps over the raised sphere. The mesh-based deformable cubes are stacked to
the side so they do not hide the cloth-sphere interaction. The cube material
uses internal struts and damping so the dropped cubes deform, settle, and
recover. The example sets the cube ``collisionRadius`` from the requested
particle spacing. For a stiffer fabric, reduce ``distanceCompliance`` and
``bendCompliance`` or increase the solver iteration count.

Visualization
=============
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` before running it:

.. code-block:: bash

   # Terminal 1
   ./build-examples/examples/rayrai_tcp_viewer

   # Terminal 2
   ./build-examples/examples/deformable_objects

The viewer receives the deformable topology and vertex updates through the
server stream.

