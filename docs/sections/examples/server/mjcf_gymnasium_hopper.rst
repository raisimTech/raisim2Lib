#####################################
Server Example: MJCF Gymnasium Hopper
#####################################

.. image:: ../../../../rsc/docs/image/mjcf_gymnasium_hopper.png
   :alt: mjcf_gymnasium_hopper example
   :width: 100%

Overview
========
Loads the Gymnasium Hopper MuJoCo XML asset through ``raisim::World`` and
publishes it through ``raisim::RaisimServer``. The planar model's root moves
on two slide joints and one hinge, and the source MJCF uses capsule geoms, a
plane, defaults, materials, and actuators.

Target
======
CMake target: ``mjcf_gymnasium_hopper``.

Run
===
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/mjcf_gymnasium_hopper

On Windows, run ``mjcf_gymnasium_hopper.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.

Details
=======
- Loads ``rsc/mjcf/gymnasium/hopper.xml`` with the ``raisim::World``
  constructor and prints the DOF, generalized-coordinate dimension, and
  collision-body count.
- Retrieves the loaded articulated system by its MJCF root body name
  (``torso``).
- Applies a small sinusoidal torque pattern to the leg joints in
  ``FORCE_AND_TORQUE`` control mode.
