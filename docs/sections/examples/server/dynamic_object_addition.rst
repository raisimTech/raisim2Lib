#######################################
Server Example: Dynamic Object Addition
#######################################

Overview
========
Runs an ANYmal with PD control and periodically throws balls at it. It shows
how to add objects while the simulation and the server are running.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/dynamic_object_addition.png
   :alt: dynamic_object_addition example
   :width: 100%

Target
======
CMake target: ``dynamic_object_addition``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/dynamic_object_addition

On Windows, run ``dynamic_object_addition.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.


Details
=======
- Spawns ANYmal with PD control and adds a sphere every 600 steps, up to ten.
- Sets an initial velocity on each new sphere to create a "ball throw" effect.
- Adds the spheres inside the ``integrateWorldThreadSafe`` callback, which runs
  under the server's world mutex, so the world can change safely while the
  server is streaming it.

