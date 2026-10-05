###########################
Server Example: Ray Casting
###########################

Overview
========
Casts a ray from a fixed origin and visualizes the hit point with a polyline
and a sphere. Use it as the simplest reference for ray casting; see
:doc:`../../RayTest`.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/ray_casting.png
   :alt: ray_casting example
   :width: 100%

Target
======
CMake target: ``ray_casting``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/ray_casting

On Windows, run ``ray_casting.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.


Details
=======
- Casts a single ray from a fixed origin every step; the ray direction sweeps
  around the vertical so the hit point traces a spiral.
- Visualizes the hit point with a polyline and marker sphere.
- Uses ``World::rayTest`` against terrain and primitives.

