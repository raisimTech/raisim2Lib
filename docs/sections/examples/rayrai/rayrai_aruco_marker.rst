############################
Rayrai Example: Aruco Marker
############################

Overview
========
Uses an orthographic camera and flat lighting to render a textured mesh marker for detection-style pipelines. It is tuned for consistent marker appearance.

Binary
======
CMake target and executable name: ``rayrai_aruco_marker``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target rayrai_aruco_marker
   ./rayrai_aruco_marker

On Windows, run ``rayrai_aruco_marker.exe`` instead.
This example uses the in-process rayrai renderer (no external client required).


Details
=======
- Adds a mesh visual as a stand-in marker and rotates it to face the camera.
- Configures a directional light and disables shadows for flat lighting.
- Uses an orthographic camera to render a marker-like view.

Source
======
.. literalinclude:: ../../../../examples/src/rayrai/rayrai_aruco_marker.cpp
   :language: cpp
