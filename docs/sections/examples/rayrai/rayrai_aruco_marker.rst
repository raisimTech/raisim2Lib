############################
Rayrai Example: Aruco Marker
############################

.. image:: ../../../../rsc/docs/image/rayrai_aruco_marker.png
   :alt: rayrai_aruco_marker example
   :width: 100%


Overview
========
Uses an orthographic camera and flat lighting to render a textured mesh marker for detection-style pipelines. It is tuned for consistent marker appearance.

Target
======
CMake target: ``rayrai_aruco_marker``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/rayrai_aruco_marker

On Windows, run ``rayrai_aruco_marker.exe`` instead.
This example renders in process with rayrai and does not need ``rayrai_tcp_viewer``.


Details
=======
- Loads the textured ``aruco_marker/aruco_marker.dae`` mesh as a visual and
  rotates it to face the camera.
- Configures a directional light and disables shadows for flat lighting.
- Uses an orthographic camera to render a marker-like view.

