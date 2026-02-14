#######################################
Server Example: Templated Tracked Robot
#######################################

Overview
========
Loads a templated tracked robot URDF with parameter overrides and drives wheels/flippers using PD targets. It shows how to customize URDF parameters programmatically.

Screenshot
==========
.. image:: ../../../image/trackedRobot.gif

Binary
======
CMake target and executable name: ``templated_tracked_robot``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target templated_tracked_robot
   ./templated_tracked_robot

On Windows, run ``templated_tracked_robot.exe`` instead.
This example uses RaisimServer. Start a visualizer client (RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) and connect to port 8080.


Details
=======
- Instantiates a templated tracked robot URDF with parameter overrides.
- Applies PD control to wheels and flippers while resetting track joints.
- Adds a sequence of box obstacles in front of the robot.

Source
======
.. literalinclude:: ../../../../examples/src/server/templated_tracked_robot.cpp
   :language: cpp
