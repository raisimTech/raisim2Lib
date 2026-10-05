#########################################
Server Example: Synchronous Server Update
#########################################

Overview
========
Drives ``RaisimServer`` from the simulation loop instead of a server thread:
the world advances one step, then the server answers one viewer request. This
shows how to keep visualization and camera-sensor updates in lockstep with the
simulation.

Target
======
CMake target: ``synchronous_server_update``. The source is
``examples/src/server/synchronous_server_update.cpp``.

Run
===
Start the example, then connect the viewer:

.. code-block:: bash

   # Terminal 1
   ./build-examples/examples/synchronous_server_update

   # Terminal 2
   ./build-examples/examples/rayrai_tcp_viewer

On Windows, run ``synchronous_server_update.exe`` instead. The example waits
for a viewer on port 8080 before it starts simulating.

Details
=======
- Uses ``setupSocket()``, ``acceptConnection()``, ``waitForMessageFromClient()``,
  ``processRequests()``, and ``closeConnection()`` instead of
  ``launchServer()``.
- Each loop applies the viewer's drag force with ``applyInteractionForce()``,
  integrates the world, waits up to 1 s for a viewer request, and processes
  it. If processing fails, it waits for a new connection.
- Loads a sensored ANYmal and sets its RGB and depth cameras to
  ``MeasurementSource::MANUAL`` so the viewer renders their frames on request.
