##################################
Server Example: Dynamic Heightmap
##################################

Overview
========
Updates a heightmap height field and color map every frame to animate terrain.
Use ``rayrai_tcp_viewer`` for supported visualization.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/dynamic_heightmap.png
   :alt: dynamic_heightmap example
   :width: 100%

Target
======
CMake target: ``dynamic_heightmap``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/dynamic_heightmap

On Windows, run ``dynamic_heightmap.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.


Details
=======
- Creates a 100 × 100 heightmap and updates its heights and colors every step.
- Calls ``HeightMap::update`` and ``HeightMap::setColor`` inside the
  ``integrateWorldThreadSafe`` callback, which runs under the server's world
  mutex; the viewer can still pause and step the loop.
- Intended for dynamic heightmap rendering in ``rayrai_tcp_viewer``.

