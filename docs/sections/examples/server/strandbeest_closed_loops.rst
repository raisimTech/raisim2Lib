#########################################
Server Example: Strandbeest Closed Loops
#########################################

Overview
========
A 12-legged Strandbeest walks, driven by one crank. Its 36 closed kinematic loops are closed by
URDF ``<equality>`` constraints (see :doc:`../../articulated_system/ClosedLoopSystems`).

Screenshot
==========
.. image:: ../../../../rsc/docs/image/rayrai/constraints_strandbeest.png
   :alt: strandbeest_closed_loops example
   :width: 70%

Target
======
CMake target: ``strandbeest_closed_loops``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/strandbeest_closed_loops

On Windows, run ``strandbeest_closed_loops.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.

Details
=======
- Loads ``rsc/strandbeest/strandbeest.urdf``, converted from the MIT-licensed
  `StrandbeestRobot.jl <https://github.com/rdeits/StrandbeestRobot.jl>`__ model: 79 degrees of
  freedom and 36 loop constraints.
- Every loop is planar, so each loop hinge is an equality constraint along the two axes normal to
  its hinge axis. Closing them with two pins per hinge instead gives the same motion, but doubles
  the number of constraint rows (half of them redundant) and made a step about 3.5 times slower.
- The constraints are eliminated exactly every step, so the contact solver only iterates on the
  feet's ground contacts (about two sweeps per step).
- The crank ``joint_crossbar_crank`` turns at one revolution per second through a velocity-only
  PD; every other joint is passive. The Strandbeest walks about 1.3 m per crank revolution.
