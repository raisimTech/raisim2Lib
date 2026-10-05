###########################
Server Example: ANYmal Pair
###########################

Overview
========
Runs two ANYmal variants (B and C) with PD control in a world scene. This example is a compact reference for loading multiple quadruped URDFs and visualizing them through RaisimServer.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/anymal_pair.png
   :alt: anymal_pair example
   :width: 100%

Target
======
CMake target: ``anymal_pair``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/anymal_pair

On Windows, run ``anymal_pair.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.


Details
=======
- Loads ANYmal B and C URDFs and applies PD targets for all leg joints.
- Uses a checkerboard ground and focuses on anymalC.
- Runs the world through the standard RaisimServer object stream.

