#############################
Server Example: Office1 Scene
#############################

Overview
========
Loads the office1 XML world, adds a dynamic ball, and spawns an Aliengo
quadruped with PD control.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/office1_scene.png
   :alt: office1_scene example
   :width: 100%

Target
======
CMake target: ``office1_scene``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/office1_scene

On Windows, run ``office1_scene.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.


Details
=======
- Loads ``rsc/maps/office1.xml`` with the ``raisim::World`` constructor, adds
  a grid-textured ground, and launches a sphere with an initial velocity.
- Spawns Aliengo with PD posture control on top of the scene.
- Focuses the camera on the robot.

