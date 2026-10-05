################################
Rayrai Example: LiDAR Pointcloud
################################

Overview
========
Attaches a Livox LiDAR module to Go1 and visualizes the scan as a point cloud
every frame. Nearby primitives and a static mesh generate richer returns.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/rayrai_lidar_pointcloud.png
   :alt: rayrai_lidar_pointcloud example
   :width: 100%

Target
======
CMake target: ``rayrai_lidar_pointcloud``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/rayrai_lidar_pointcloud

On Windows, run ``rayrai_lidar_pointcloud.exe`` instead.
This example renders in process with rayrai and does not need ``rayrai_tcp_viewer``.


Details
=======
- Loads Go1 with the ``livox_lidar`` module and updates the scan each frame.
- Transforms LiDAR points from the sensor frame to the world frame.
- Visualizes the scan as a ``raisin::PointCloud``; the point size is set through
  ``PointCloud::pointSize``.

