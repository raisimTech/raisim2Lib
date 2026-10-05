#######################################
Server Example: Visual Objects Showcase
#######################################

Overview
========
Adds visual primitives, meshes, arrows, polylines, dynamic meshes, and a visual heightmap through the server API. It demonstrates the visualization helpers and dynamic updates.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/visual_objects_showcase.png
   :alt: visual_objects_showcase example
   :width: 100%

Target
======
CMake target: ``visual_objects_showcase``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/visual_objects_showcase

On Windows, run ``visual_objects_showcase.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.


Details
=======
- Adds visual-only primitives, meshes, arrows, polylines, and a visual heightmap.
- Updates colors, sizes, dynamic mesh data, and the visual heightmap every
  loop while holding ``lockVisualizationServerMutex()``.
- Shows visual articulated systems and custom mesh streaming.
- Does not integrate the world; only the visual objects change.

