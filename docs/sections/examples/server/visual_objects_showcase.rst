#######################################
Server Example: Visual Objects Showcase
#######################################

Overview
========
Adds visual primitives, meshes, arrows, polylines, dynamic meshes, and a visual heightmap through the server API. It demonstrates the visualization helpers and dynamic updates.

Screenshot
==========
.. image:: ../../../image/visuals.gif

Binary
======
CMake target and executable name: ``visual_objects_showcase``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target visual_objects_showcase
   ./visual_objects_showcase

On Windows, run ``visual_objects_showcase.exe`` instead.
This example uses RaisimServer. Start a visualizer client (RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) and connect to port 8080.


Details
=======
- Adds visual-only primitives, meshes, arrows, polylines, and a visual heightmap.
- Updates colors/sizes and dynamic mesh data every frame.
- Shows visual articulated systems and custom mesh streaming.

Source
======
.. literalinclude:: ../../../../examples/src/server/visual_objects_showcase.cpp
   :language: cpp
