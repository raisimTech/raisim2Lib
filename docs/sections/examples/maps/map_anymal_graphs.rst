##########################
Map Example: Anymal Graphs
##########################

Overview
========
Runs two ANYmal variants (B and C) with PD control on a map and streams joint position, velocity, and torque graphs to the visualizer. It is a good reference for time-series graphing.

Screenshot
==========
.. image:: ../../../image/map_anymal_graphs.png

Binary
======
CMake target and executable name: ``map_anymal_graphs``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target map_anymal_graphs
   ./map_anymal_graphs

On Windows, run ``map_anymal_graphs.exe`` instead.
This example uses RaisimServer. Start a visualizer client (RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) and connect to port 8080.


Details
=======
- Loads ANYmal B and C URDFs and applies PD targets for all leg joints.
- Sets the Unreal map to ``wheat`` and hides the ground; focuses on anymalC.
- Streams joint position/velocity/torque into three time-series graphs.

Source
======
.. literalinclude:: ../../../../examples/src/maps/map_anymal_graphs.cpp
   :language: cpp
