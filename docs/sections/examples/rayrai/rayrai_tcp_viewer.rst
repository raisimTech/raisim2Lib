##########################
Rayrai Example: TCP Viewer
##########################

Overview
========
Standalone TCP viewer for ``raisim::RaisimServer`` scenes. It connects to a
running server, receives the scene, and renders it with rayrai and ImGui
controls. Build the ``rayrai_tcp_viewer`` target from this repository; it also
has command-line options for batch runs and diagnostics.
:doc:`../../RayraiTcpViewer` is the full reference for its panels, options and
wire format.

Target
======
CMake target: ``rayrai_tcp_viewer``.

Source
======
``examples/src/rayrai/tools/rayrai_tcp_viewer.cpp`` is the viewer's main
source file; the ``TcpViewer*`` files next to it hold the rest of the
implementation. ``examples/CMakeLists.txt`` builds the target from these files.
The platform install scripts refresh them to match the installed rayrai
library.

Run
====
Run the build-tree executable while a ``RaisimServer`` application is running:

.. code-block:: bash

   ./build-examples/examples/rayrai_tcp_viewer

On Windows, the executable is ``.\build-examples\bin\rayrai_tcp_viewer.exe``.
The viewer is a separate process; in-process rayrai examples open their own
renderer window and do not need it.

Useful options
==============

.. code-block:: bash

   rayrai_tcp_viewer --connect 127.0.0.1:8080 --auto-frame
   rayrai_tcp_viewer --host 192.168.1.42 --port 8081 --auto-connect
   rayrai_tcp_viewer --connect '[2001:db8::10]:8080'
   rayrai_tcp_viewer --simulate /path/to/world.xml
   rayrai_tcp_viewer --inspect /path/to/robot.urdf
   rayrai_tcp_viewer --resource-dir /path/to/rsc --window-size 1600x900
   rayrai_tcp_viewer --fullscreen --minimize-panels --keep-overlay-open
   rayrai_tcp_viewer --no-pre-warm
   rayrai_tcp_viewer --warm-at-startup
   rayrai_tcp_viewer --no-save-settings
   rayrai_tcp_viewer --camera-lookat 3,-4,2,0,0,0 --force-camera-lookat
   rayrai_tcp_viewer --camera-offset 2,-3,1
   rayrai_tcp_viewer --screenshot /tmp/rayrai_tcp_viewer.png
   rayrai_tcp_viewer --screenshot-dir /tmp/rayrai_frames
   rayrai_tcp_viewer --record-session /tmp/session.rrtcs
   rayrai_tcp_viewer --update-rate 30
   rayrai_tcp_viewer --replay-session /tmp/session.rrtcs --replay-speed 0.5 --replay-loop
   rayrai_tcp_viewer --trajectory-csv /tmp/poses.csv --export-scene /tmp/scene.json
   rayrai_tcp_viewer --server-list /tmp/servers.txt --wait-for-server 10 --exit-after 30

Run ``rayrai_tcp_viewer --help`` for the full option list.

Details
=======
- Connects to a running ``RaisimServer`` over TCP and receives the scene.
- Renders the streamed scene with rayrai and exposes an ImGui control panel
  with Connection, Options, Record, Render, Objects, Diagnostics and Help tabs.
- Supports explicit host/port selection, the ``--connect`` endpoint shortcut,
  and discovery of ``RaisimServer`` instances through UDP beacons.
- Splits its window into independent panes, each with its own connection, and
  restores the layout and endpoints on the next launch.
- Runs a dropped RaiSim world XML (or ``--simulate FILE``) in a child
  simulation process, and opens dropped URDF/MJCF files in a kinematic
  inspector.
- Supports repeatable ``--resource-dir`` entries for resolving streamed mesh and
  texture resources.
- Provides camera framing, orthographic view snaps, fullscreen startup, panel
  minimization, screenshots, MP4/MOV/MKV video recording through ``ffmpeg``,
  PNG sequence output, raw TCP session recording, offline replay with a
  timeline, trajectory CSV logging, target update-rate control, and
  scene/object JSON diagnostics.
- Interactive tools include simulation pause and step, Shift+left-drag force
  application, Ctrl+left-drag (Cmd on macOS) interaction-wire pulling, a
  2-point ruler, 3-point angle measurement, pose grabber axes, body frames, COM
  markers, scene editing (spawn, delete, world export), live signal plots with
  CSV export, and a Help tab listing the keyboard shortcuts.
- Rebuild the ``rayrai_tcp_viewer`` target after changing its source or the
  installed rayrai library.
- Renders manual RGB/depth cameras on request from ``RaisimServer``, returns
  the BGRA/metric-depth images, and shows previews, camera frustums and camera
  frames in the selected-object Sensors tab.
- Shows the actuators of a selected articulated system, with motor operating
  region plots, in the selected-object Actuators tab.
- Prefer in-process rayrai when the application must own the render context or
  access textures without a viewer process and TCP round trip.
