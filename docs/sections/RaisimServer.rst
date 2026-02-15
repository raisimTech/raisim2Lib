#############################
Raisim Server
#############################

RaisimServer serializes ``raisim::World`` and streams the data to clients via TCP/IP.
We provide the RaisimUnity (`doc <https://raisim.com/sections/RaisimUnity.html>`_) client, which visualizes a ``raisim::World``.
Basic usage is described in the `documentation <https://raisim.com/sections/RaisimUnity.html#how-to-use-raisimunity>`_.

In addition to visualizing a ``raisim::World``, ``raisim::RaisimServer`` can visualize additional objects.
An example can be found `here <https://github.com/raisimTech/raisim2Lib/blob/master/examples/src/server/visual_objects_showcase.cpp>`__.
The example is displayed as follows:

.. image:: ../image/visuals.gif

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

``integrateWorldThreadSafe()`` locks the world mutex, applies any user
interaction force coming from the visualizer, integrates the world, and
unlocks the mutex.

Thread safety and lifecycle
===========================
The server reads the world state from a background thread. If you modify the
world manually while the server is running, guard it with the visualization
mutex (``lockVisualizationServerMutex()`` / ``unlockVisualizationServerMutex()``)
to avoid races.

The server can be paused with ``hibernate()`` and resumed with ``wakeup()``.
Call ``killServer()`` to stop the server thread and disconnect the client.

Sensor updates from the visualizer
==================================
If a sensor's measurement source is set to
``Sensor::MeasurementSource::VISUALIZER``, the server requests updates only
when the sensor's update period elapses. The visualizer sends the rendered
RGB/depth buffers back to the server, which then writes them into the sensor's
internal buffers and updates the timestamp. Sensors with non-visualizer
measurement sources are serialized directly from RaiSim.

Synchronous updates (optional)
==============================
``processRequests()`` implements a synchronous request/response loop used by
clients that explicitly pull updates. It returns ``false`` if the client does not
respond, rejects the protocol version, or fails to deliver sensor data in time.


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
