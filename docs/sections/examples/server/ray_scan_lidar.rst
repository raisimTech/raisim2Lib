##############################
Server Example: Ray Scan LiDAR
##############################

Overview
========
Sweeps rays from a robot frame to emulate a LiDAR scan and visualizes hits with instanced boxes. This is a compact example of ray-based sensing.

Screenshot
==========
.. image:: ../../../image/huskyScan.gif

Binary
======
CMake target and executable name: ``ray_scan_lidar``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target ray_scan_lidar
   ./ray_scan_lidar

On Windows, run ``ray_scan_lidar.exe`` instead.
This example uses RaisimServer. Start a visualizer client (RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) and connect to port 8080.


Details
=======
- Sweeps a yaw/pitch grid and uses ``rayTest`` to emulate LiDAR.
- Displays scan points as instanced boxes colored by range.
- Uses an IMU frame for ray orientation and resets the robot when out of bounds.

Source
======
.. literalinclude:: ../../../../examples/src/server/ray_scan_lidar.cpp
   :language: cpp
