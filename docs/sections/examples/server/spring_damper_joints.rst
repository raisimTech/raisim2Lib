####################################
Server Example: Spring Damper Joints
####################################

Overview
========
Loads URDFs with spring and damper joints (cartpole and chain) to visualize joint compliance. This example focuses on spring-damper behavior.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/spring_damper_joints.png
   :alt: spring_damper_joints example
   :width: 100%

Target
======
CMake target: ``spring_damper_joints``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/spring_damper_joints

On Windows, run ``spring_damper_joints.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.


Details
=======
- Loads ``rsc/springDamper/cartpole.urdf`` (revolute and prismatic joints) and
  ``rsc/springDamper/chainSpringed.urdf`` (ball joints), both with
  spring/damper joint parameters.
- Demonstrates URDF-based joint spring/damper behavior.
- Focuses the camera on the cartpole.

