#################################
Server Example: NVIDIA USD Robots
#################################

Overview
========
Loads three bundled Isaac Sim robot USD files with
``World::addUsdArticulatedSystem`` and publishes them together through
``RaisimServer``. The example is intentionally limited to assets that were
smoke-tested as RaiSim articulated systems with supported collision bodies.
See :doc:`../../OpenUSD`.

.. image:: ../../../../rsc/docs/image/rayrai/rayrai_usd_nvidia_robots.png
   :alt: Collision bodies from three Isaac Sim USD robot assets imported into RaiSim
   :width: 100%

Target
======
CMake target: ``nvidia_usd_robots`` (C++20). RaiSim loads USD through its
bundled OpenUSD runtime on every supported platform, so no extra build switch
is needed.

Assets
======
The bundled assets are:

- ``create3``: ``rsc/isaac/Robots/iRobot/Create3/create_3.usd`` (BSD-3)
- ``jetbot``: ``rsc/isaac/Robots/NVIDIA/Robomaker/aws_robomaker_jetbot.usd`` (MIT)
- ``ant``: ``rsc/isaac/Robots/IsaacSim/Ant/ant.usd`` (MIT)

Run
===
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/nvidia_usd_robots

On Windows, run ``nvidia_usd_robots.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.

Details
=======
- Loads each USD file with ``World::addUsdArticulatedSystem`` and simulates at
  500 Hz.
- Throws an error if a USD file does not import as an articulated system or if
  no supported collision bodies are imported.
- Places the imported floating bases above a checkerboard ground and applies a
  shared ``nvidia_usd_robot`` collision material.
