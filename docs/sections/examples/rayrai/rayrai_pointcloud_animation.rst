####################################
Rayrai Example: Pointcloud Animation
####################################

Overview
========
Generates animated ring point clouds to show per-point color updates and dynamic point buffer uploads.

Screenshot
==========
.. image:: ../../../image/rayrai_pointcloud_animation.png

Binary
======
CMake target and executable name: ``rayrai_pointcloud_animation``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target rayrai_pointcloud_animation
   ./rayrai_pointcloud_animation

On Windows, run ``rayrai_pointcloud_animation.exe`` instead.
This example uses the in-process rayrai renderer (no external client required).


Details
=======
- Builds a synthetic multi-ring point cloud with per-ring colors.
- Updates positions every frame to create an orbiting animation.
- Good for testing point-cloud upload and rendering.

Source
======
.. literalinclude:: ../../../../examples/src/rayrai/rayrai_pointcloud_animation.cpp
   :language: cpp
