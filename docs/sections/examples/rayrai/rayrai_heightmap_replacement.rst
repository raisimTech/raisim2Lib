######################################
Rayrai Example: Heightmap Replacement
######################################

Overview
========
Visualizes stale deleted-heightmap geometry in a rayrai depth image. The scene
places a PD-controlled ANYmal on a flat terrain patch and renders its front and
rear depth cameras into separate ImGui panels. Press Space to create a visibly
different heightmap with broad mounds that shift across both camera frames, then
replace randomized populations of simple primitives and delete the old scene.

Target
======
CMake target: ``rayrai_heightmap_replacement``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/rayrai_heightmap_replacement

On Windows, run ``rayrai_heightmap_replacement.exe`` instead.
This example renders in process with rayrai and does not need ``rayrai_tcp_viewer``.

How to reproduce
================
#. Watch the mounds in the front and rear depth images shown at the upper left.
#. Tap Space several times. Each press removes the current heightmap and creates
   a new generation with a random number of boxes, spheres, and cylinders in each
   camera frame.
#. Compare the depth image with the current colored terrain in the main view. A
   mound or primitive from an earlier generation indicates stale deleted geometry
   in the depth pass.

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
- Allocates the new heightmap and primitives before calling
  ``World::removeObject`` on the old scene, guaranteeing distinct object
  addresses while still exercising deletion.
