##################################
Server Example: Dynamic Heightmap
##################################

Overview
========
Updates a heightmap height field and color map every frame to animate terrain. It can be used with RaisimUnity, RaisimUnreal, or the rayrai TCP viewer.

Screenshot
==========
.. image:: ../../../image/dynamic_heightmap_unreal.png

Binary
======
CMake target and executable name: ``dynamic_heightmap``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target dynamic_heightmap
   ./dynamic_heightmap

On Windows, run ``dynamic_heightmap.exe`` instead.
This example uses RaisimServer. Start a visualizer client (RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) and connect to port 8080.


Details
=======
- Creates a heightmap grid and updates heights and colors every frame.
- Uses visualization mutex locking while updating the heightmap.
- Intended for dynamic heightmap rendering in visualizers.

Source
======
.. literalinclude:: ../../../../examples/src/server/dynamic_heightmap.cpp
   :language: cpp
