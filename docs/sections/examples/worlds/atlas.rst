#####################
Server Example: Atlas
#####################

Overview
========
Spawns an Atlas humanoid and applies an external force and torque to its base
every step. The example demonstrates a larger articulated system streamed
through RaisimServer.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/atlas.png
   :alt: atlas example
   :width: 100%

Target
======
CMake target: ``atlas``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/atlas

On Windows, run ``atlas.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.


Details
=======
- Spawns one Atlas (the grid size ``N`` in the source is 1) and initializes
  its base pose with zero joint torques.
- Applies an external force and torque to the base every step to perturb the
  robot.
- Uses a checkerboard ground so the TCP viewer's reflective ground option is visible.
- Focuses the TCP viewer on the robot.

