##############################
Rayrai Example: Custom Visuals
##############################

Overview
========
Demonstrates custom visual primitives (sphere, box, cylinder, capsule, mesh) and simple animation of positions and orientations.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/rayrai_custom_visuals.png
   :alt: rayrai_custom_visuals example
   :width: 100%

Target
======
CMake target: ``rayrai_custom_visuals``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/rayrai_custom_visuals

On Windows, run ``rayrai_custom_visuals.exe`` instead.
This example renders in process with rayrai and does not need ``rayrai_tcp_viewer``.


Details
=======
- Creates custom visuals (sphere/box/cylinder/capsule/mesh) outside the physics world.
- Animates visual positions and orientations each frame.
- Demonstrates ``setDetectable(true)`` for custom visuals that should appear in
  external-camera and other detectable-only capture render passes.
  Detectability is render filtering metadata; it does not add physics,
  collision, or a RaiSim object.
