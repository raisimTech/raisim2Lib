################################
Server Example: Sim Control Demo
################################

Overview
========

Demonstrates the interactive simulation-control requests supported by
``RaisimServer`` and the rayrai TCP viewer: pause, resume, single- or multi-step,
external force and torque application, single-body pose edits, and articulated
generalized-coordinate edits.

Target
======
CMake target: ``sim_control_demo``. The source is
``examples/src/server/sim_control_demo.cpp``.

Run
===
Start the viewer, then run the build-tree example:

.. code-block:: bash

   # Terminal 1
   ./build-examples/examples/rayrai_tcp_viewer

   # Terminal 2
   ./build-examples/examples/sim_control_demo

On Windows, run ``sim_control_demo.exe`` instead. Select ``anymal`` or
``poke_box`` in the viewer. Use the Objects tab to pause, resume, and step, or
use the force and pose tools described in :doc:`../../RayraiTcpViewer`.

Security Boundary
=================

The server binds to ``127.0.0.1`` by default. Simulation-control requests are
not authenticated: a client that can reach the server can change the world.
Keep the loopback default unless every host on the reachable network is
trusted. Calling ``server.setBindLoopbackOnly(false)`` explicitly opts into
listening on all interfaces.

Details
=======

* Creates a PD-controlled ANYmal named ``anymal`` for generalized-coordinate
  edits.
* Creates a dynamic box named ``poke_box`` for force, torque, and pose edits.
* Uses ``RaisimServer::integrateWorldThreadSafe()``; negotiated viewer requests
  are applied inside the server's synchronization boundary before integration.
* Runs until terminated by the user.
