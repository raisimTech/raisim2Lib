############################
Rayrai Example: Depth Camera
############################

Overview
========
Renders a linear depth texture from the Go1 depth camera and shows it in ImGui
with a frustum overlay. This is the recommended RGB/depth sensor path when
rayrai is available. The active code path reads rendered depth from rayrai.
The source also keeps a disabled (``#if 0``) ``World::captureDepthCamera``
block as a deterministic CPU fallback for headless ray-query use.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/rayrai_depth_camera.png
   :alt: rayrai_depth_camera example
   :width: 100%

Target
======
CMake target: ``rayrai_depth_camera``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/rayrai_depth_camera

On Windows, run ``rayrai_depth_camera.exe`` instead.
This example renders in process with rayrai and does not need ``rayrai_tcp_viewer``.


Details
=======
- Loads Go1 with the D455 module and fetches the depth sensor.
- Renders a linear depth texture and shows it in an ImGui window.
- Reads the rayrai depth buffer with ``raisin::Camera::getRawImage``.
- Keeps a disabled CPU ``World::captureDepthCamera`` fallback that returns
  depth, a segmentation object id and an optional hit point per pixel, a
  timestamp, and deterministic depth noise when rayrai rendering is not
  available.
- Places a sphere and a box in front of the camera using its pose.

