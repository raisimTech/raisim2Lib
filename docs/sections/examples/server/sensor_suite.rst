############################
Server Example: Sensor Suite
############################

Overview
========
Demonstrates the RGB camera, depth camera, IMU, and LiDAR sensors on ANYmal,
including depth-to-point-cloud conversion and point-cloud visualization. It is
the main reference for the sensor APIs; see :doc:`../../Sensors`.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/sensors_cpp.png
   :alt: sensor_suite example
   :width: 100%

Target
======
CMake target: ``sensor_suite``. The source is
``examples/src/server/sensor_suite.cpp``.

Run
===
Start the viewer, then run the build-tree example:

.. code-block:: bash

   # Terminal 1
   ./build-examples/examples/rayrai_tcp_viewer

   # Terminal 2
   ./build-examples/examples/sensor_suite

On Windows, run ``sensor_suite.exe`` instead. The example uses RaisimServer on
port 8080.

Details
=======
- Loads ``anymal_c/urdf/anymal_sensored.urdf``, which carries front and rear
  RGB/depth cameras, an IMU, and a spinning LiDAR.
- Configures the RGB and depth cameras with ``MeasurementSource::MANUAL``.
  The TCP viewer renders these camera frames on request, returns BGRA and
  metric depth buffers to ``RaisimServer``, and shows previews in the
  **Sensors** tab of the selected ANYmal. The source includes a commented-out
  line that switches the front depth camera to RaiSim CPU depth.
- Select the ANYmal, open **Sensors**, and use **Show frustum** to inspect the
  camera pose and range in the main scene.
- Converts the front depth array to 3D points with
  ``DepthCamera::depthToPointCloud`` and reads the LiDAR scan.
- Streams the latest LiDAR scan as a point cloud and two marker spheres through
  ``RaisimServer``.
