####################################
Server Example: Shadow Hand USD Cube
####################################

Overview
========
Loads NVIDIA Isaac Sim's Shadow Robot ShadowHand USD scene directly with
``World(shadow_hand.usd)`` and places a RaiSim rigid-body cube at the Isaac Lab
in-hand cube start pose. The constructor imports the USD Physics rigid bodies
and joints as a RaiSim articulated system. See :doc:`../../OpenUSD`.

The cube is a native RaiSim box. The example fails fast if the ShadowHand
articulation, collision bodies, or visual meshes are not imported.

.. image:: ../../../../rsc/docs/image/rayrai/rayrai_usd_shadow_hand_cube.png
   :alt: ShadowHand USD collision geometry with a cube above the hand
   :width: 100%

Target
======
CMake target: ``shadow_hand_usd_cube`` (C++20).

Run
===
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/shadow_hand_usd_cube

On Windows, run ``shadow_hand_usd_cube.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect
to port 8080. The viewer remains interactive while the simulation runs.

Details
=======
- Loads ``rsc/isaac/Robots/ShadowRobot/ShadowHand/shadow_hand.usd`` with
  the ``World`` constructor.
- Uses RaiSim's bundled OpenUSD runtime directly; no Assimp USD importer or
  USD-specific build switch is required.
- Throws an error if the imported scene does not contain an articulated system
  named ``shadow_hand`` with collision bodies and visual meshes.
- Places the ShadowHand base at ``(0, 0, 0.5)``.
- Drives the hand toward the imported nominal configuration with PD control.
  While the viewer has the simulation paused, joint positions edited in the
  viewer become the new PD targets.
- Creates a dynamic 6.5 cm cube at ``(0, -0.39, 1.05)``, matching the Isaac Lab
  Shadow Hand cube task's initial object pose.
