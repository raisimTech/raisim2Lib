############################
Map Example: Hill1 Heightmap
############################

Overview
========
Loads the hill1 heightmap and drops Aliengo high above the terrain to demonstrate heightmap placement and scale. The map is set to "hill1" in the server.

Screenshot
==========
.. image:: ../../../image/map_hill1.png

Binary
======
CMake target and executable name: ``map_hill1_heightmap``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target map_hill1_heightmap
   ./map_hill1_heightmap

On Windows, run ``map_hill1_heightmap.exe`` instead.
This example uses RaisimServer. Start a visualizer client (RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) and connect to port 8080.


Details
=======
- Loads the hill1 heightmap PNG with explicit scale/offset and hides the mesh.
- Drops Aliengo from height and holds posture with PD gains.
- Sets the Unreal map to ``hill1`` and focuses the camera on the robot.

Source
======
.. literalinclude:: ../../../../examples/src/maps/map_hill1_heightmap.cpp
   :language: cpp
