###############################
Rayrai Example: Instancing Grid
###############################

Overview
========
Renders a large grid of instanced boxes to demonstrate instancing performance and per-instance weighting.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/rayrai_instancing_grid.png
   :alt: rayrai_instancing_grid example
   :width: 100%

Target
======
CMake target: ``rayrai_instancing_grid``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/rayrai_instancing_grid

On Windows, run ``rayrai_instancing_grid.exe`` instead.
This example renders in process with rayrai and does not need ``rayrai_tcp_viewer``.


Details
=======
- Creates a 300 × 300 grid (90,000 instances) of instanced boxes with
  per-instance color weights.
- Animates one instance to demonstrate dynamic updates.
- Uses ``InstancedVisuals`` for efficient bulk rendering.

