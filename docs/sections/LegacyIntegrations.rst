#############################
Legacy Integrations
#############################

RaisimUnity and RaisimUnreal are no longer supported in the current binary distribution.
Use :doc:`Rayrai` for visualization and :doc:`Visualization` for choosing between
the rayrai TCP viewer and in-process rendering.

Current replacement paths
=========================

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Older workflow
     - Current workflow
   * - Launch a Unity or Unreal visualizer for a ``RaisimServer`` application
     - Start ``rayrai_tcp_viewer`` and run the same
       ``raisim::RaisimServer`` application.
   * - Record visualized policy rollouts
     - Call ``RaisimServer::startRecordingVideo`` and ``stopRecordingVideo``;
       ``rayrai_tcp_viewer`` encodes the video with ``ffmpeg`` or writes a PNG
       sequence when ``ffmpeg`` is not available. For in-process rendering, use
       rayrai screenshot or offscreen capture.
   * - Render RGB/depth images for sensors
     - Use in-process rayrai sensor rendering with ``raisin::RayraiWindow``,
       or set the sensors to ``Sensor::MeasurementSource::MANUAL`` so
       ``rayrai_tcp_viewer`` renders them for a ``RaisimServer`` application.
       Use RaiSim CPU depth only when rayrai is unavailable or a deterministic
       headless ray-query fallback is explicitly required.
   * - Inspect glTF, PBR, HDR, or OpenUSD assets
     - Use the rayrai asset-inspection examples listed in :doc:`Examples`.

Kept pages
==========

The old page names are kept only as short redirects for users following stale
links:

* :doc:`RaisimUnity`
* :doc:`RaisimUnreal`
