#############################
Raisim Server
#############################

``raisim::RaisimServer`` serializes a ``raisim::World`` and streams it to a
client over TCP. The supported client is ``rayrai_tcp_viewer`` (see
:doc:`RayraiTcpViewer`). For older Unity or Unreal workflows, see
:doc:`LegacyIntegrations`. For a comparison between server-based visualization
and in-process rayrai rendering, see :doc:`Visualization`.

Besides the world itself, ``RaisimServer`` can stream visual-only objects that
take no part in the simulation, such as visuals, polylines, visual articulated
systems and charts. The animation below shows them in a legacy visualizer; see
:doc:`Examples` for runnable examples.

.. image:: ../../rsc/docs/image/visuals.gif

Typical usage
=========================
Create the server, launch it, and advance the world through the thread-safe
integration helper:

.. code-block:: cpp

  raisim::World world;
  raisim::RaisimServer server(&world);
  server.launchServer(8080);
  for (;;) {
    server.integrateWorldThreadSafe();
  }

``integrateWorldThreadSafe()`` locks the world mutex and applies the pending
client pose and force requests. It then decides from the pause and step state
whether this call advances the world. On an advancing call it applies the
interaction-wire and client forces. It runs the callback, if one was given,
integrates the world when allowed, and unlocks the mutex.

``launchServer(port)`` binds its socket before it returns. If ``port`` is
already in use, the server tries the next 63 ports, logs
``Port <requested> is in use; bound to port <actual> instead``, and reports an
error only when none of them is free. ``getPort()`` returns the port actually
bound, and the discovery beacons advertise it (see `Discovery beacons`_). The
server serves one client at a time and accepts the next one after the current
client disconnects.

Thread safety and lifecycle
===========================
The server reads the world state from a background thread. If you modify the
world from another thread while the server is running, hold the world mutex
(``lockVisualizationServerMutex()`` / ``unlockVisualizationServerMutex()``)
to avoid races. ``integrateWorldThreadSafe()`` takes it for you.

``hibernate()`` keeps the client connected but stops sending world updates,
and ``wakeup()`` resumes them. This does not pause the simulation; for that,
see `Interactive sim control`_. Call ``killServer()`` to stop the server
thread, disconnect the client and close the socket. Call it before the server
object is destroyed.

Sensor measurements
==================================
``RaisimServer`` can have the connected rayrai TCP viewer render the images of
RGB and depth cameras. When a ``MeasurementSource::MANUAL`` camera reaches its
update period, the regular scene reply includes its pose, lens model,
resolution and clipping range. The viewer renders the camera image and returns
BGRA pixels or metric depth values in a ``REQUEST_SENSOR_UPDATE`` frame. The
server validates every entry before it replaces sensor buffers and updates
timestamps.

IMU and spinning LiDAR measurements are always computed by RaiSim; their
metadata is streamed for inspection, but the viewer does not return samples
for them. Use ``MeasurementSource::RAISIM`` for RaiSim-computed sensors, and
manual RGB/depth cameras when the connected viewer should render the image. See
:ref:`tcp-viewer-sensor-round-trip` on the TCP viewer page.

Streamed scene notes
====================
* Spatial tendons, including wires added with ``addStiffWire``,
  ``addCompliantWire`` or ``addCustomWire``, are streamed as polylines. The TCP
  viewer includes them in the camera images it renders for the server.
* An object whose appearance is ``"hidden"`` or ``"invisible"``
  (case-insensitive) is still streamed, but rayrai and the TCP viewer do not draw
  it and it cannot be selected.

Sharing a RaiSim Engine scene
=============================
A world built from a ``.rscene`` file (``raisim::World("scene.rscene")``) streams its
physics bodies like any other world. The rest of the scene (visual-only objects, foliage,
lights, sky, terrain textures and the saved camera) lives only in the file. So the server
also offers the viewer the scene file and every file it uses: meshes, ``.rasset``
descriptors and their parts, textures, HDR images, and the buffers and images that glTF,
OBJ and COLLADA meshes reference. ``raisim::rscene::referencedFiles()`` returns that list.

The viewer looks for each file on its own computer first and asks its user before
downloading anything; see :ref:`sections/RayraiTcpViewer:RaiSim Engine scenes`. Sharing is
on by default:

.. code-block:: cpp

  raisim::World world("forest.rscene");
  raisim::RaisimServer server(&world);
  server.setSceneFileSharing(false); // a remote viewer then shows the bodies only
  server.launchServer();

* Only the files in the list can be read, and a client addresses them by their index in
  it, never by path.
* The server hashes the files on its network thread, at most 64 MiB per update, so a large
  scene never delays a frame for long. A file edited while the server runs is hashed again
  for the next client.
* Downloads travel in the update replies, at most 8 MiB per reply, and leave room for the
  world update.
* Viewers built before this feature, and worlds that were not built from a ``.rscene``
  file, are not affected.

``examples/src/server/visualization/rscene_server.cpp`` serves the rayrai forest scene
(:doc:`examples/server/rscene_server`).

Synchronous updates (optional)
==============================
``processRequests()`` handles one request/response exchange with the
connected client. The thread started by ``launchServer()`` calls it in a loop,
so most programs never call it. A program that wants to serve the client from
its own loop instead skips ``launchServer()`` and uses ``setupSocket()``,
``acceptConnection()``, ``waitForMessageFromClient()`` and
``processRequests()`` directly; see
``examples/src/server/synchronous_server_update.cpp``. In this mode no discovery
beacons are sent, so the viewer must connect by host and port. A loop that
calls ``world.integrate()`` itself also never applies the client's pause, step,
force, torque, pose or generalized-coordinate requests, which only
``integrateWorldThreadSafe()`` applies.

``processRequests()`` returns ``false`` if the client does not respond, uses
another protocol version, or sends a request frame the server rejects (see
`Request validation`_).

Protocol versioning and deformable streaming
============================================
The TCP wire protocol is versioned. Each request starts with a protocol header
that advertises the client's feature bits. The server requires the client's
protocol version to equal its own and rejects unknown feature bits: it logs a
warning and closes the connection rather than misparse the stream. The feature
bits are:

* ``PROTOCOL_FEATURE_EXPLICIT_HEADER``: every request and reply starts with
  the explicit protocol version and feature header.
* ``PROTOCOL_FEATURE_DEFORMABLE_DELTA``: deformable mesh topology is sent only
  at initialization or when the topology changes; normal frames send vertex
  positions only.
* ``PROTOCOL_FEATURE_SIM_CONTROL``: when both ends advertise this bit, the
  client may pause, resume and step the simulation and push external forces,
  torques, body poses and articulated-system generalized coordinates. The
  server applies no authentication: if the connection is open, the client can
  drive the simulation. See `Interactive sim control`_ below.
* ``PROTOCOL_FEATURE_CONTACT_OBJECT_TAGS``: each streamed contact identifies
  both participating objects, enabling per-object contact counts and plots.
* ``PROTOCOL_FEATURE_ACTUATOR_STATE``: the detailed state of the selected
  articulated system also carries its actuators (see :doc:`Actuators`): motor
  parameters, the motor state of the last step, and the velocity and actuator
  torque of every driven joint. The viewer plots these in the selected
  object's **Actuators** tab.
* ``PROTOCOL_FEATURE_SCENE_FILES``: every reply carries the state of the
  world's ``.rscene`` file list, and the client may ask for the list and for
  file chunks (see `Sharing a RaiSim Engine scene`_).

The rayrai TCP viewer negotiates all six bits automatically.
``sendUpdateRequest()`` in ``rayrai/RaisimTcpCommon.hpp`` advertises every bit
the client build supports; a custom client that writes its own request frames
may advertise any subset (see :doc:`RayraiTcpViewer`).

Interactive sim control
=======================
When ``PROTOCOL_FEATURE_SIM_CONTROL`` is negotiated, a connected client such as
``rayrai_tcp_viewer`` can drive the simulation instead of only observing it.
The viewer's **Objects** tab has a *Simulation* row with icon buttons to pause
or resume, step one frame and step ten frames.

Client requests added by this feature:

* ``CR_PAUSE`` / ``CR_RESUME``: set or clear a pause flag that
  ``integrateWorldThreadSafe()`` checks. While paused, ``world_->integrate()``
  is not called, but the network thread keeps streaming the state.
  ``CR_RESUME`` also drops queued steps.
* ``CR_STEP_N``: queue *N* single steps to advance while paused.
* ``CR_APPLY_FORCE``: apply an external force at a world-space point on a
  body. The server multiplies the force command by the body's mass, so the
  command acts as an acceleration.
* ``CR_APPLY_TORQUE``: apply an external torque on a body (not scaled by
  mass).
* ``CR_SET_POSE``: teleport a single-body object to a position and a
  quaternion in (w, x, y, z) order.
* ``CR_SET_GC``: set the generalized coordinate of an articulated system.

Scene-editing requests predate this feature and are accepted from any client
with the same protocol version, whether or not ``PROTOCOL_FEATURE_SIM_CONTROL``
is negotiated: ``CR_SPAWN_*`` (box, sphere, cylinder, capsule, mesh,
articulated system, ground plane, PNG height map), ``CR_REMOVE_OBJECT`` (an
object or a spatial tendon, by visual tag), ``CR_SAVE_THE_WORLD`` (export the
world as XML), and the interaction wire ``CR_ATTACH_WIRE`` /
``CR_DRAG_OBJECT``. File paths in these requests are opened and written on the
server host.

There is no authentication or per-client authorization: once the connection is
open, the client can issue any request negotiated by both ends, plus every
scene-editing request. The bind address is the only access control the server
provides:

.. code-block:: cpp

  raisim::RaisimServer server(&world);
  server.setBindLoopbackOnly(false);  // listen on all interfaces; call before launchServer()
  server.launchServer();

Keep the default (``127.0.0.1`` only) for development. Open the bind to a wider
network only when you trust every host that can reach the port.

Your own simulation code can use the same pause and step state, for example
to drive a headless server:

.. code-block:: cpp

  server.pauseSimulation();
  // ... the world is not integrated while paused; state streaming keeps running ...
  server.stepSimulation(10);   // queue 10 single steps
  // ... or resume completely ...
  server.resumeSimulation();
  if (server.isSimulationPaused()) { /* … */ }

Pose and generalized-coordinate requests are applied at the next
``integrateWorldThreadSafe()`` call, also while paused. A force or torque
request starts or refreshes a client force that is held for 0.12 s of
simulation time and applied on every integration step until it expires. While
paused, the force can be refreshed, but it is not applied until a step or a
resume lets ``integrateWorldThreadSafe()`` integrate the world.
``stepSimulation(N)`` queues N single steps; each paused
``integrateWorldThreadSafe()`` call consumes one. Unlike ``CR_RESUME``,
``resumeSimulation()`` only clears the pause flag: steps queued earlier stay
queued and run the next time the simulation is paused.

Request validation
------------------
The server decodes and validates a whole request frame before applying any of
it. It rejects the frame when a request contains, for example:

* a non-finite number, or a quaternion that is zero or not of unit length;
* a non-positive mass or dimension in a spawn request (a capsule's height may
  be zero);
* a file that does not exist on the server; height maps must be PNG,
  articulated systems URDF or XML, and meshes need a file extension;
* a force or torque target that does not exist, or a body index that does not
  exist on a force, torque or wire target (a wire whose target no longer
  exists is simply detached);
* a pose target that is not a single body, or a generalized-coordinate vector
  whose size does not match the articulated system;
* a sim-control request on a connection that did not negotiate
  ``PROTOCOL_FEATURE_SIM_CONTROL``, or a step count outside 1 to 1,000,000;
* a ``CR_SAVE_THE_WORLD`` request that is not the only request in its frame.

A rejected frame is logged as ``Rejecting malformed client frame: <reason>``,
nothing from it is applied, and the client is disconnected. If a model file
passes validation but then fails to load, the server removes the objects that
frame had already spawned, logs the failure and disconnects the client.

Custom mutation under the world mutex
-------------------------------------
Programs that spawn, move or modify world objects every tick can pass a
callback to ``integrateWorldThreadSafe()``. The callback runs under the world
mutex on every call, also while paused. It runs after the client requests and
forces are applied, and before ``world_->integrate()`` on calls that advance
the world:

.. code-block:: cpp

  for (int i = 0;; i++) {
    server.integrateWorldThreadSafe([&]() {
      if (i % 600 == 0) {
        auto* ball = world.addSphere(0.1, 1.0);
        ball->setPosition(0, -2, 0.8);
        ball->setVelocity(0, 10, 0, 0, 0, 0);
      }
    });
  }

The callback overload keeps all the pause, step, force and pose behavior of the
overload without arguments, so the viewer can still drive the simulation while
the program changes the world every tick. See
``examples/src/server/basics/dynamic_object_addition.cpp`` and
``examples/src/server/terrain/dynamic_heightmap.cpp`` for working uses, and
``examples/src/server/sim_control_demo.cpp`` for a minimal scene to try the
viewer's simulation controls on.

Viewer requests
===============
The server can also ask the connected viewer to act. ``rayrai_tcp_viewer``
carries these requests out on the next scene update (see :doc:`RayraiTcpViewer`):

* ``requestSaveScreenshot()``: save a PNG of the scene in the viewer's output
  folder, as its ``F12`` key does.
* ``startRecordingVideo(name)`` / ``stopRecordingVideo()``: record the scene
  to the viewer's output folder, using the file name and extension of
  ``name`` (``.mp4`` when it has none). A viewer without ``ffmpeg`` writes
  numbered PNGs to ``<stem>_frames/`` in the output folder instead. The
  viewer ignores the request while a recording started in the viewer is
  running.
* ``setCameraPositionAndLookAt(pos, lookAt)``: place the viewer camera at
  ``pos``, aimed at the point ``lookAt``.
* ``focusOn(object)``: frame ``object`` in the viewer and select it.

The viewer's output folder is its ``--screenshot-dir`` or the folder chosen in
its **Record** tab. Screenshot and recording requests stored in a recorded
session are ignored when the session is replayed.

Default bind address
====================
``RaisimServer`` binds to ``127.0.0.1`` by default rather than ``INADDR_ANY``,
which prevents accidental exposure on a shared LAN. To accept connections from
other machines, call this before ``launchServer()``:

.. code-block:: cpp

  server.setBindLoopbackOnly(false);

Treat the bind address as a security boundary: once the server is reachable
from a wider network, any client on that network can pause, step, apply
forces, spawn and remove objects. There is no token or authorization layer to
fall back on.

Discovery beacons
=================
While the server is running, ``RaisimServer`` advertises itself with a UDP
beacon once per second on port ``59312``. The beacon payload contains the
RaiSim TCP protocol version, TCP port, host name, executable name (on Linux;
other platforms send ``RaisimServer``), bind mode (``loopback`` or ``all``),
and connection status. ``rayrai_tcp_viewer`` lists only beacons whose protocol
version equals its own, in the server table of a pane without a session and
in the endpoint popup of the **Connection** tab.

With the default loopback bind, beacons are sent only to ``127.0.0.1``. After
``server.setBindLoopbackOnly(false)``, they are broadcast on the local network
so another machine can discover the server. Discovery is only a convenience
layer: clients may always connect directly to the TCP endpoint with
``--connect host:port`` or equivalent custom-client code.

For Windows LAN use, allow both the TCP server port (``8080`` by default) and
UDP port ``59312`` through the firewall. If UDP broadcast is blocked, manual
TCP connection still works.

RaisimServer API
=========================

.. doxygenclass:: raisim::RaisimServer
   :members:

Visuals API
=========================

.. doxygenstruct:: raisim::Visuals
   :members:

Polyline API
=========================

.. doxygenstruct:: raisim::PolyLine
   :members:

ArticulatedSystemVisual API
============================

.. doxygenstruct:: raisim::ArticulatedSystemVisual
   :members:
