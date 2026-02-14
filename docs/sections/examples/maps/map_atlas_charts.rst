#########################
Map Example: Atlas Charts
#########################

Overview
========
Spawns Atlas on a map and demonstrates RaisimServer charting with time-series plots and bar charts. Use this to see how to publish telemetry to the visualizer.

Screenshot
==========
.. image:: ../../../image/map_atlas_chart.png

Binary
======
CMake target and executable name: ``map_atlas_charts``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target map_atlas_charts
   ./map_atlas_charts

On Windows, run ``map_atlas_charts.exe`` instead.
This example uses RaisimServer. Start a visualizer client (RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) and connect to port 8080.


Details
=======
- Spawns Atlas robots and initializes the base pose with zero joint torques.
- Applies external force/torque each frame to perturb the robot.
- Demonstrates time-series and bar chart overlays (Unreal visualization).

Source
======
.. literalinclude:: ../../../../examples/src/maps/map_atlas_charts.cpp
   :language: cpp
