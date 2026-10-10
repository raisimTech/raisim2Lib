#######################
Sensors and depth/LiDAR
#######################

This page covers aligning rayrai to RaiSim camera sensors, fisheye lenses,
CPU readback, rendering depth and LiDAR data, and the TCP viewer protocol. For
the general camera control and picking APIs (free-fly, orbit, picking from the
viewer) see :doc:`Capture`.

Sensor alignment
================
rayrai can align rendering to RaiSim camera sensors. Constructing a
``raisin::Camera`` from a RaiSim ``RGBCamera`` or ``DepthCamera`` copies the
sensor's resolution, clip planes, horizontal field of view, and lens model (a GL
context must be current). The sensor overloads of ``renderWithExternalCamera``
and ``renderDepthPlaneDistance`` then copy the sensor's world pose into that
camera on every call; ``syncRaisimCameraPose`` does the same for other render
paths. RaiSim sensor rendering is world-object-only: custom visuals, instanced
visuals, point clouds, coordinate frames, and other viewer-only helpers are
excluded from RGB, depth, and LiDAR data-generation passes. Use generic
external-camera rendering only when you intentionally want a viewer render that
includes visualization objects.
Note: ``syncRaisimCameraPose`` updates ``Camera::position/front/up`` directly;
avoid calling ``Camera::update()`` immediately afterward unless you also update
``yaw``/``pitch``.

For runnable coverage, see
:doc:`Rayrai RGB camera <../examples/rayrai/rayrai_rgb_camera>`,
:doc:`Rayrai depth camera <../examples/rayrai/rayrai_depth_camera>`,
:doc:`Rayrai LiDAR point cloud <../examples/rayrai/rayrai_lidar_pointcloud>`, and
:doc:`Rayrai ArUco marker <../examples/rayrai/rayrai_aruco_marker>` for dedicated
sensor examples. ``rayrai_complete_showcase`` combines RGB/depth cameras, LiDAR
visualization, camera frustums, raw buffer readback, and custom visuals in one
runnable scene. The sensor overview in :doc:`Sensors <../Sensors>` includes a
longer RGB/depth readback example.

To investigate stale depth geometry after removing terrain or primitives, use
the optional :doc:`../debug_tools/rayrai_heightmap_replacement` tool.

RGB/Depth camera workflow (manual source + external camera):

.. code-block:: cpp

    auto rgbCam = anymal->getSensorSet("d455_front")->getSensor<raisim::RGBCamera>("color");
    auto depthCam = anymal->getSensorSet("d455_front")->getSensor<raisim::DepthCamera>("depth");

    rgbCam->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);
    depthCam->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);

    raisin::Camera rgbCamera(*rgbCam);
    raisin::Camera depthCamera(*depthCam);

    viewer.renderWithExternalCamera(*rgbCam, rgbCamera, {});   // color image
    viewer.renderWithExternalCamera(*depthCam, depthCamera, {}); // optional color view of the depth sensor
    viewer.renderDepthPlaneDistance(*depthCam, depthCamera);    // linear depth

``renderDepthPlaneDistance`` does not need the color render before it; render
the depth camera's color image only when you want to show it.

You can read back the camera buffers on CPU. ``getRawImage`` reads the scene
color target as 8-bit BGRA (``width * height * 4`` bytes); the sensor resolution
must match the camera's render target. ``SensorStorageMode::SENSOR_SETTER``
(the default) writes into the RaiSim sensor instead of a caller buffer:

.. code-block:: cpp

    const auto& prop = rgbCam->getProperties();
    const int width = std::max(1, prop.width);
    const int height = std::max(1, prop.height);
    std::vector<char> bgra(size_t(width) * size_t(height) * 4);
    rgbCamera.getRawImage(*rgbCam, raisin::Camera::SensorStorageMode::CUSTOM_BUFFER,
      bgra.data(), bgra.size(), /*flipVertical=*/false);

Depth uses a ``float`` buffer with ``width * height`` entries. Constructing a
``Camera`` from a ``DepthCamera`` allocates only the linear-depth target; the
color and post-processing targets are allocated if the camera is later used
for a color render. Keep the camera alive across frames to reuse its buffers,
and destroy it while its owning GL context is current. ``Camera`` cannot be
copied or moved because it owns GL handles.

The ``RGBCamera``/``DepthCamera`` overloads of ``renderWithExternalCamera``
always render without MSAA, temporal AA, or viewer upscaling, whatever the
quality preset. They do run the built-in postprocessing unless
``RenderOverrides::postProcess`` is ``false``, so enabling
``setLinearHdrRenderingEnabled(true)`` (see :doc:`Capture`) also changes RGB
sensor images; pass ``postProcess = false`` to keep the previous output.

Asynchronous readback
=====================
``getRawImage`` is synchronous. For high-rate capture,
``Camera::setAsyncReadbackEnabled(true, ringSize)`` (default 3; values below 2
are raised to 2) enables a ring of pixel-buffer objects:
``readSceneColorRgbaAsync(rgba)`` (RGBA8, unlike the BGRA of ``getRawImage``)
and ``readLinearDepthAsync(depth)`` start a readback of the current frame and
return ``true`` once they have copied an earlier one. The first ``ringSize``
calls return ``false`` while the ring fills, and the copied frame then lags the
latest render by ``ringSize`` calls.

.. code-block:: cpp

    rgbCamera.setAsyncReadbackEnabled(true, 3);
    std::vector<unsigned char> rgba;
    viewer.renderWithExternalCamera(*rgbCam, rgbCamera, {});
    if (rgbCamera.readSceneColorRgbaAsync(rgba, /*flipVertical=*/true)) {
      // rgba holds the frame rendered three calls earlier.
    }

Fisheye lenses
==============
A RaiSim camera whose ``lens`` property is an equidistant fisheye model
(``raisim::CameraLensModel::setOpenCvFisheye(fx, fy, cx, cy, k1, k2, k3, k4)``)
is honored by ``raisin::Camera``. The ``RGBCamera`` and ``DepthCamera``
constructors copy the lens, ``isFisheyeLens()`` reports it, and
``renderWithExternalCamera`` resamples the rendered color image through the
OpenCV/ROS equidistant model with ``k1``–``k4`` distortion. The source view is
rasterized with a horizontal field of view of at most 179°. Only the color
image is remapped; depth from ``renderDepthPlaneDistance`` stays rectilinear.
For a camera that is not built from a RaiSim sensor, call
``Camera::setLensModel(lens, hFovRad)`` and set ``zoom`` (the vertical field of
view of the rasterized source view) yourself.

TCP viewer protocol
===================
The rayrai TCP viewer protocol is explicitly versioned. The current viewer sends a
protocol header with feature bits before each request, and the server replies with the
negotiated feature set. A viewer rejects newer unsupported protocol versions with a clear
error instead of attempting to parse an incompatible stream.

Current feature bits cover the explicit header, deformable delta streaming, sim
control, contact ownership tags, and actuator state. Deformable objects send
mesh topology during initialization or topology changes; ordinary update
frames send vertex positions only. This keeps dynamic cloth/cube streaming
cheaper while avoiding binary compression until network bandwidth is measured
as a bottleneck. Sim-control messages share the same feature-negotiated request
path.

The protocol constants are defined in ``rayrai/TcpProtocolReader.hpp``, which
``rayrai/RaisimTcpCommon.hpp`` includes (namespace ``raisin::tcp_viewer``):

* ``kDefaultPort`` — default ``RaisimServer`` port the viewer connects to
  (8080).
* ``kProtocolVersion`` — the current wire version. Mismatched versions cause
  the viewer to disconnect with a versioned-protocol error.
* ``kProtocolFeatureExplicitHeader``, ``kProtocolFeatureDeformableDelta``,
  ``kProtocolFeatureSimControl``, ``kProtocolFeatureContactObjectTags``, and
  ``kProtocolFeatureActuatorState`` — the currently negotiated feature bits;
  ``kProtocolSupportedFeatures`` is the OR of all bits this build understands.
  The viewer rejects a server update that sets any other bit.
* ``kMaxMessageBytes`` — maximum accepted message size (default 64 MiB),
  overridable at build time via the
  ``RAISIM_TCP_VIEWER_MAX_MESSAGE_BYTES`` preprocessor define when very large
  scenes need a larger frame budget.

The wire format is a native-endian binary stream. Each TCP frame begins with
an ``int32_t`` total-frame-size header (including the 4-byte header itself).
Strings in both directions, including sensor names in sensor-update messages,
carry ``int32_t`` byte-length prefixes.

Custom TCP clients should use ``raisin::tcp_viewer::BufferReader`` to decode
frames. It is a non-owning view over the received byte buffer (the buffer must
outlive the reader) with bounds-checked accessors:

.. code-block:: cpp

    raisin::tcp_viewer::BufferReader reader(buffer);  // const std::vector<char>&
    auto version = reader.read<int>();
    auto features = reader.read<std::uint64_t>();
    auto name = reader.readString();
    auto positions = reader.readFloatVector();
    if (!reader.ok) {
      // malformed frame; drop the connection
    }

Each read advances the cursor (``reader.offset()``) and clears ``reader.ok`` if
there is not enough data left or a length prefix is invalid. The flag is
sticky, so callers can decode an entire frame and check ``ok`` at the end
rather than after every field. Besides ``read<T>()`` and ``readString()``, the
reader provides ``readBool``, ``readVec3f``, ``readVec4f``, ``readQuatWxyz``,
``readFloatVector``, ``readIntVector``, ``readByteVector``, and
``readColorMap``.

The current viewer also services ``MeasurementSource::MANUAL`` RGB/depth
requests received in the scene stream. It renders from the streamed camera
pose and lens, returns BGRA or metric-depth buffers in a sensor-update message,
and exposes the latest preview in the selected object's Sensors tab. IMU and
spinning-LiDAR values remain RaiSim-side. See :doc:`../RayraiTcpViewer` for the
full request/response sequence and troubleshooting guidance.

Depth and LiDAR
===============
The renderer supports a linear depth plane and a GPU LiDAR. The depth pass renders RaiSim world
objects only; visualization-only objects are intentionally ignored so they cannot leak into
training observations. The GPU LiDAR scans the collision geometry, exactly like the CPU LiDAR.

* ``renderDepthPlaneDistance`` renders camera-plane distance into the camera's
  ``R32F`` linear-depth texture; pixels without geometry are ``0``. Its
  optional ``drawVisualizationObjects`` argument (default ``false``) adds custom
  and instanced visuals; only detectable ones are included unless
  ``visualizationObjectsMustBeDetectable`` is ``false``.
* ``measureSpinningLidarSingleDrawGPU`` advances the LiDAR head with
  ``SpinningLidar::advanceSweep`` and casts the swept rays on the GPU against
  the collision bodies of the world (``World::getRayCastBodies``): primitives,
  heightmaps and meshes, intersected exactly. It returns the points of
  ``SpinningLidar::update``, ray by ray and in the same order, to float
  precision (about 1e-5 of the range), and stores them in the sensor frame with
  ``SpinningLidar::setScan``. As on the CPU, rays start at the minimum range and
  the body carrying the sensor is ignored. Pass ``objectToExclude`` to leave out
  every body of one more object.

Read the depth texture of an external camera with
``Camera::getLinearDepthTexture()``; ``getDepthPlaneTexture()`` returns the one
of the viewer's internal camera.

The two LiDAR paths share the head state, so a LiDAR can switch between them at
any update. Both return the same points; they differ in cost. A GPU update pays
a fixed synchronous readback of about 0.1 ms and then casts thousands of rays
almost for free; a CPU update costs about 0.1 to 0.3 µs per ray. For an Ouster
OS1-64 (32,768 rays per turn) over a heightmap, updating at the sensor's 100 Hz
rate, the GPU path is about 8 to 9 times faster; called every 1 ms physics step
(330 rays per update) the two are about even.

1) GPU: ``measureSpinningLidarSingleDrawGPU``, preferred when rayrai is
   available and each update casts a few thousand rays or more.
2) CPU: ``SpinningLidar::update``, needs no GL context.

GPU example:

.. code-block:: cpp

    lidar->updatePose();
    const glm::dvec3 posW = raisin::toGlm(lidar->getPosition());
    const glm::dmat3 rotW = raisin::toGlm(lidar->getOrientation());
    viewer.measureSpinningLidarSingleDrawGPU(*lidar, posW, rotW);
