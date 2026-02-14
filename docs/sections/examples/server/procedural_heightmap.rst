####################################
Server Example: Procedural Heightmap
####################################

Overview
========
Creates a procedural heightmap and places ANYmal on top. Use this to see how to configure fractal terrain properties and run a robot on generated terrain.

Screenshot
==========
.. image:: ../../../image/heightMapUsingTerrainGenerator.gif

Binary
======
CMake target and executable name: ``procedural_heightmap``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target procedural_heightmap
   ./procedural_heightmap

On Windows, run ``procedural_heightmap.exe`` instead.
This example uses RaisimServer. Start a visualizer client (RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) and connect to port 8080.


Details
=======
- Generates terrain from ``TerrainProperties`` (fractal noise).
- Spawns ANYmal on the heightmap with PD control.
- Demonstrates procedural heightmap creation and appearance.

Source
======
.. literalinclude:: ../../../../examples/src/server/procedural_heightmap.cpp
   :language: cpp
