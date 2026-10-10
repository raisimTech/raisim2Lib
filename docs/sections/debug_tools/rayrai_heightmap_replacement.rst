######################################
Debug Tool: Heightmap Replacement
######################################

Overview
========
Checks rayrai depth images for stale geometry after removing heightmaps and
primitives. The scene
places a PD-controlled ANYmal on a flat terrain patch and renders its front and
rear depth cameras into separate ImGui panels, and draws the pixels of each
camera projected into the world as small instanced cubes. Press Space to create a visibly
different heightmap with broad mounds that shift across both camera frames, then
replace randomized populations of simple primitives and delete the old scene.

Target
======
CMake target: ``rayrai_heightmap_replacement``.
Source: ``examples/tools/debug/rayrai_heightmap_replacement.cpp``.

Run
====
Build with ``RAISIM_DEBUG_TOOLS=ON`` as described in :doc:`../DebugTools`,
then run the build-tree executable:

.. code-block:: bash

   ./build-debug/examples/rayrai_heightmap_replacement

On Windows, run ``build-debug\bin\rayrai_heightmap_replacement.exe`` instead.
This tool renders in process with rayrai and does not need ``rayrai_tcp_viewer``.

How to reproduce
================
#. Watch the mounds in the front and rear depth images shown at the upper left.
#. Tap Space several times. Each press removes the current heightmap and creates
   a new generation with a random number of boxes, spheres, and cylinders in each
   camera frame.
#. Compare the depth image with the current colored terrain in the main view. A
   mound or primitive from an earlier generation indicates stale deleted geometry
   in the depth pass.
#. The projected pixel cubes must lie on the current terrain and primitives. Cubes
   floating in the air or buried in the ground show that the depth image does not
   match the scene.

Details
=======
- Loads the sensored ANYmal used by ``sensor_suite`` and selects its front and
  rear depth cameras.
- Uses PD gains and a nominal standing target to hold the twelve leg joints.
- Keeps a radius of 1.15 m around the robot exactly flat, with a smooth transition
  to rolling terrain.
- Moves broad near-field mounds left and right across both camera views on every
  terrain generation.
- Places two to nine deterministic pseudo-random static boxes, spheres, and
  cylinders in each camera's near field.
- Projects each depth image into world-frame points with
  ``DepthCamera::depthToPointCloud`` and draws every sixth pixel in each
  direction as a small cube, one ``InstancedVisuals`` batch per camera in the
  color of its frustum (orange front, blue rear), darker with distance. The
  batches are not detectable, so the cameras never see the cubes.
- Allocates the new heightmap and primitives before calling
  ``World::removeObject`` on the old scene, guaranteeing distinct object
  addresses while still exercising deletion.
