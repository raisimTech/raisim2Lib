#############################
Sensors
#############################

RaiSim supports these sensor workflows:

* **URDF-attached sensors** for robots. Sensor metadata is loaded from a sensor
  XML file attached to a URDF link. This is the right workflow for IMU, RGB,
  depth, and spinning LiDAR sensors that should move with an articulated system.
* **rayrai-rendered RGB/depth sensors** for camera observations. When rayrai is
  available, use in-process rayrai rendering with
  ``Sensor::MeasurementSource::MANUAL`` for RGB/depth sensor buffers. This is
  the recommended path for rendered images, screenshots, and dataset generation.
* **World-level CPU depth-camera capture** only as a deterministic headless
  fallback. ``World::captureDepthCamera`` casts CPU rays from an arbitrary camera
  pose and returns depth, object segmentation, optional hit points, and a
  timestamp without requiring a renderer.

The CPU ray-query paths are single-threaded and deterministic, but they are not
the recommended RGB/depth image path when rayrai sensors are available. RaiSim
itself does not support sensor measurement updates from the TCP visualizer.
Renderer-produced images must be consumed in the rayrai process or copied into a
sensor buffer by user code with ``MeasurementSource::MANUAL``.


Sensor frames and camera convention
====================================

Camera and LiDAR sensors use the RaiSim sensor-frame convention:

* ``+X``: forward
* ``+Y``: left
* ``+Z``: up

Camera depth is the distance along the camera ``+X`` axis, not the Euclidean ray
length. This matches the existing ``DepthCamera`` convention and is convenient for
pinhole camera models.


How to attach a sensor to a link
====================================

Before further explanations, we clarify some terms used throughout this page.
**sensor** means one measurement stream, such as an RGB camera, a depth camera,
an IMU, or a LiDAR scan. **sensor_set** means a set of sensors contained in one
link. For example, an Intel RealSense-like link can contain RGB, depth, and IMU
sensors.

Create a link for a sensor set and give it a ``sensor`` attribute:

.. code-block:: xml

    <link name="realsense_d435" sensor="realsense435.xml"/>

Such a link must be empty: its inertial, visual and collision elements are
defined in the sensor XML file. The sensor XML file should be stored in the same
directory as the URDF file. If it is not found, RaiSim searches
``[urdf_dir]/sensor``, ``[urdf_dir]/sensors``, ``[urdf_dir]/..``, and
``[urdf_dir]/../sensors``. For a URDF given as a string, the resource directory
passed to ``World::addArticulatedSystem()`` takes the place of ``[urdf_dir]``.
An example sensor file is
`realsense435.xml <https://github.com/raisimTech/raisim2Lib/blob/master/rsc/anymal_c/sensors/realsense435.xml>`_.
An example URDF is
`anymal_sensored.urdf <https://github.com/raisimTech/raisim2Lib/blob/master/rsc/anymal_c/urdf/anymal_sensored.urdf>`__.


Measurement source and update modes
====================================

The update method is selected with ``raisim::Sensor::setMeasurementSource``:

* ``RAISIM``: measurement is computed by RaiSim. The owning articulated system
  refreshes it during ``World::integrate()`` whenever ``1 / update_rate`` of
  simulated time has passed since the last update. This is the default for IMU
  and spinning LiDAR sensors. ``DepthCamera`` supports RaiSim-side CPU ray
  updates, but use this for headless fallback rather than for rendered camera
  images when rayrai is available. ``RGBCamera`` cannot use it.
* ``MANUAL``: user code writes the buffers. This is the default for RGB and
  depth cameras. Use it for in-process rayrai RGB/depth rendering or real
  hardware integration.

There is no ``VISUALIZER`` measurement source in the RaiSim API. RaiSim does not
request RGB or depth frames from a TCP visualizer and does not update sensor
buffers from the visualizer.

Recommended RGB/depth usage with rayrai sets camera sensors to ``MANUAL`` and
lets the renderer fill or read back the buffers. See `Using sensors with rayrai`_
for a complete example. Use ``RAISIM`` CPU depth updates only when rayrai is not
available or when a deterministic headless ray-query fallback is required.

For RGB sensors, RaiSim does not synthesize color in the physics engine. Write
the buffer manually, preferably from in-process rayrai rendering when it is
available, or from real hardware:

.. code-block:: cpp

    auto* rgb = sensorSet->getSensor<raisim::RGBCamera>("color");
    rgb->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);
    std::vector<char> rgba(size_t(rgb->getProperties().width) *
                           size_t(rgb->getProperties().height) * 4);
    rgb->setImageBuffer(rgba);

When reading sensor buffers from another user-managed thread, use
``std::scoped_lock``:

.. code-block:: cpp

    {
      std::scoped_lock lock(*depth);
      const auto& image = depth->getDepthArray();
      // read image here
    }


Depth camera
====================================

``raisim::DepthCamera`` stores a float array with ``width * height`` entries.
Finite values are plane depth in meters. Missing pixels are ``NaN``.

CPU-only fallback update:

.. code-block:: cpp

    auto* depth = anymal->getSensorSet("depth_camera_front_camera_parent")
                    ->getSensor<raisim::DepthCamera>("depth");
    depth->setMeasurementSource(raisim::Sensor::MeasurementSource::RAISIM);

    world.integrate();     // refreshes the image at the sensor's update rate
    depth->update(world);  // optional: ray-cast a new image now

    const auto& z = depth->getDepthArray();
    const int w = depth->getProperties().width;
    const int h = depth->getProperties().height;
    float centerDepth = z[(h / 2) * w + (w / 2)];

Convert depth to a point cloud:

.. code-block:: cpp

    std::vector<raisim::Vec<3>> pointsWorld;
    depth->depthToPointCloud(depth->getDepthArray(), pointsWorld, false);

    std::vector<raisim::Vec<3>> pointsSensor;
    depth->depthToPointCloud(depth->getDepthArray(), pointsSensor, true);

The camera intrinsics are defined by width, height, horizontal field of view
(``hFOV``), and optional pixel offsets. If you modify camera properties after
loading, call ``updateRayDirections()`` before the next update.


RGB camera
====================================

``raisim::RGBCamera`` stores an image buffer with ``width * height * 4`` bytes
(4 bytes per pixel, rows contiguous). RaiSim itself does not render RGB and does
not interpret the channel order; rayrai's ``Camera::getRawImage()`` writes BGRA.
Use rayrai or manual hardware input; ``update()`` is a fatal error for an RGB
camera.

Manual buffer update:

.. code-block:: cpp

    auto* rgb = anymal->getSensorSet("depth_camera_front_camera_parent")
                  ->getSensor<raisim::RGBCamera>("color");

    auto& prop = rgb->getProperties();
    std::vector<char> rgba(size_t(prop.width) * size_t(prop.height) * 4);

    // Fill rgba as R, G, B, A bytes.
    for (int y = 0; y < prop.height; ++y) {
      for (int x = 0; x < prop.width; ++x) {
        const size_t k = 4 * (size_t(y) * size_t(prop.width) + size_t(x));
        rgba[k + 0] = char(255);
        rgba[k + 1] = char(0);
        rgba[k + 2] = char(0);
        rgba[k + 3] = char(255);
      }
    }

    rgb->setImageBuffer(rgba);


Spinning LiDAR
====================================

``raisim::SpinningLidar`` uses RaiSim ray casting and returns hit points in the
sensor frame. The world helper ``rayTestLidar`` performs one frustum-style
broadphase for the scan, then raycasts only the candidate bodies.

Typical robot-attached LiDAR update:

.. code-block:: cpp

    auto* lidar = anymal->getSensorSet("lidar_link")
                    ->getSensor<raisim::SpinningLidar>("lidar");
    lidar->setMeasurementSource(raisim::Sensor::MeasurementSource::RAISIM);  // the default

    world.integrate();     // refreshes the scan at the sensor's update rate
    lidar->update(world);  // optional: scan now

    const auto& scanS = lidar->getScan(); // points in sensor frame
    lidar->updatePose();
    const auto& p_WS = lidar->getPosition();
    const auto& R_WS = lidar->getOrientation();

    std::vector<raisim::Vec<3>> scanW;
    scanW.reserve(scanS.size());
    for (const auto& p_S : scanS) {
      scanW.push_back(p_WS + R_WS * p_S);
    }

For custom scanning patterns, use ``World::rayTestLidar`` directly. Yaw and
pitch are each specified as ``start angle, signed increment, count``. See
:doc:`RayTest` for the exact signature, sampling order, compact-output behavior,
and performance considerations.

``update()`` first advances the spinning head with ``advanceSweep(worldTime)``,
which returns the yaw columns swept since the last update (``first``, ``count``
and ``direction``; a sweep longer than one turn repeats columns) and sets the
current yaw and timestamps, and then casts those columns. Another ray tracer can
call ``advanceSweep()`` itself, cast the columns against
``World::getRayCastBodies()`` (the collision bodies the ray tests see, with
their world poses) and store the points with ``setScan()``. rayrai's GPU LiDAR
does exactly this and returns the same points as ``update()``.


IMU
====================================

``raisim::InertialMeasurementUnit`` is intended for robot-attached sensor sets
loaded from URDF sensor XML. With ``RAISIM``, the default, the owning articulated
system writes the measurements during ``World::integrate()`` at the sensor's
update rate; ``update()`` does nothing. This requires inverse dynamics, which the
URDF and world-XML loaders enable when they create an IMU. Read the values after
stepping the world:

.. code-block:: cpp

    auto* imu = anymal->getSensorSet("lidar_link")
                  ->getSensor<raisim::InertialMeasurementUnit>("imu");

    world.integrate();

    const Eigen::Vector3d& acc = imu->getLinearAcceleration();  // specific force, IMU frame [m/s^2]
    const Eigen::Vector3d& angVel = imu->getAngularVelocity();  // IMU frame [rad/s]
    const raisim::Vec<4>& quat = imu->getOrientation();         // IMU frame in world, (w, x, y, z)

Like a real accelerometer, the IMU reports the specific force (acceleration
minus gravity): an IMU at rest reads the negated gravity vector in the IMU frame.
With ``MANUAL``, set the values from hardware with ``setLinearAcceleration()``,
``setAngularVelocity()`` and ``setOrientation()``.


CPU depth camera fallback
====================================

Prefer rayrai sensor rendering for RGB/depth observations when rayrai is
available. ``World::captureDepthCamera`` is a world-level CPU depth camera for
deterministic headless fallback and ray-query validation. It does not require a
robot-mounted sensor, renderer, or visualizer.
It casts one ray per pixel and can return:

* depth image as ``std::vector<float>``
* object segmentation ids as ``std::vector<int>``
* optional world-frame hit points
* capture timestamp from ``World::getWorldTime()``
* optional deterministic Gaussian or uniform depth noise

The segmentation value is the hit object's ``getIndexInWorld()``. Background is
``-1``. Depth and points are ``NaN`` for background pixels.

The camera frame follows the same convention as RaiSim depth sensors:
``+X`` is forward, ``+Y`` is left, and ``+Z`` is up. The rotation passed to
``captureDepthCamera`` maps camera-frame vectors into the world frame.

CPU fallback scene and capture setup:

.. code-block:: cpp

    raisim::World world;
    world.setWorldTime(1.25);
    world.addGround(0.0);
    auto* sphere = world.addSphere(0.5, 1.0);
    sphere->setPosition(0.0, 0.0, 0.5);

    raisim::World::DepthCameraProperties prop;
    prop.width = 64;
    prop.height = 48;
    prop.clipNear = 0.01;
    prop.clipFar = 10.0;
    prop.hFOV = M_PI / 2.0;
    prop.captureDepth = true;
    prop.captureSegmentation = true;
    prop.capturePoints = true;

    // Camera at z=2 looking down. Columns are camera axes in world frame:
    // camera +X -> world -Z, camera +Y -> world +Y, camera +Z -> world +X.
    raisim::Mat<3, 3> R_WC;
    R_WC.setZero();
    R_WC(0, 2) = 1.0;
    R_WC(1, 1) = 1.0;
    R_WC(2, 0) = -1.0;

    raisim::World::DepthCameraFrame frame;
    world.captureDepthCamera({0.0, 0.0, 2.0}, R_WC, prop, frame);

Depth image
------------------------------------

Enable ``captureDepth`` and read ``frame.depth``. The vector has
``frame.width * frame.height`` entries in row-major order. Each value is the
distance along the camera forward axis, not the Euclidean ray length. Pixels
with no hit or hits outside ``[clipNear, clipFar]`` are ``NaN``.

.. code-block:: cpp

    const int center = (prop.height / 2) * prop.width + (prop.width / 2);
    if (std::isfinite(frame.depth[center])) {
      float centerDepthMeters = frame.depth[center];
    }

Segmentation object id
------------------------------------

Enable ``captureSegmentation`` and read ``frame.segmentation``. A valid pixel
stores the hit object's ``getIndexInWorld()``. Background pixels store ``-1``.
You can compare this value to object ids kept by the application.

.. code-block:: cpp

    prop.captureSegmentation = true;
    raisim::World::DepthCameraFrame frame;
    world.captureDepthCamera(cameraPos, cameraRot, prop, frame);

    const int center = (frame.height / 2) * frame.width + (frame.width / 2);
    int objectId = frame.segmentation[center];
    if (objectId == static_cast<int>(sphere->getIndexInWorld())) {
      // The center pixel hit the sphere.
    }

Optional hit point per pixel
------------------------------------

Set ``capturePoints`` to ``true`` to fill ``frame.points``. Each entry is the
world-frame hit point for the corresponding pixel. This is optional because it
uses more memory than depth-only or segmentation-only capture.

.. code-block:: cpp

    prop.capturePoints = true;
    raisim::World::DepthCameraFrame frame;
    world.captureDepthCamera(cameraPos, cameraRot, prop, frame);

    const int center = (frame.height / 2) * frame.width + (frame.width / 2);
    const raisim::Vec<3>& hitPointW = frame.points[center];
    if (std::isfinite(hitPointW[0])) {
      std::cout << "hit point in world: "
                << hitPointW[0] << ", "
                << hitPointW[1] << ", "
                << hitPointW[2] << std::endl;
    }

Timestamp
------------------------------------

Each frame stores the current ``World`` time in ``frame.timeStamp``. This is
useful when camera captures are interleaved with control, logging, or dataset
generation.

.. code-block:: cpp

    world.setWorldTime(1.25);

    raisim::World::DepthCameraFrame frame;
    world.captureDepthCamera(cameraPos, cameraRot, prop, frame);

    double captureTimeSeconds = frame.timeStamp;  // 1.25 in this example.

Optional deterministic depth noise
------------------------------------

Depth noise is disabled by default. To add reproducible noise, set
``depthNoiseType``, ``depthNoiseMean``, ``depthNoiseStd``, and
``depthNoiseSeed``. The same world state, camera pose, properties, and seed
produce the same noisy depth image.

.. code-block:: cpp

    prop.depthNoiseType =
        raisim::World::DepthCameraProperties::DepthNoiseType::GAUSSIAN;
    prop.depthNoiseMean = 0.0;
    prop.depthNoiseStd = 0.002;
    prop.depthNoiseSeed = 42;

    raisim::World::DepthCameraFrame noisyA;
    raisim::World::DepthCameraFrame noisyB;
    world.captureDepthCamera(cameraPos, cameraRot, prop, noisyA);
    world.captureDepthCamera(cameraPos, cameraRot, prop, noisyB);

    const int center = (prop.height / 2) * prop.width + (prop.width / 2);
    assert(noisyA.depth[center] == noisyB.depth[center]);

Use ``DepthNoiseType::UNIFORM`` for bounded uniform noise in
``[mean - std, mean + std]``.

Self-filtering uses the same ``objectId`` and ``localId`` convention as
``rayTest``. This is useful for cameras mounted on a robot:

.. code-block:: cpp

    world.captureDepthCamera(cameraPos,
                             cameraRot,
                             prop,
                             frame,
                             robot->getIndexInWorld(),
                             cameraParentLocalBodyId);

Using sensors with rayrai
====================================

rayrai can render RaiSim camera sensors in process. This is the recommended
RGB/depth sensor path when rayrai is available because it uses the same sensor
pose and intrinsics while producing renderer-backed color and depth buffers.
Use the CPU camera capture API only when rayrai is unavailable or when a
CPU-only deterministic ray-query result is specifically required.

Current runnable coverage:

* :doc:`Rayrai RGB camera <examples/rayrai/rayrai_rgb_camera>` uses the Go1
  ``d455_front/color`` RGB sensor and renders it through ``raisin::Camera``.
* :doc:`Rayrai depth camera <examples/rayrai/rayrai_depth_camera>` uses the Go1
  ``d455_front/depth`` depth sensor, renders the linear depth plane, reads
  a ``float`` depth buffer back to the CPU, and draws the projected points as
  instanced cubes.
* :doc:`Rayrai LiDAR point cloud <examples/rayrai/rayrai_lidar_pointcloud>`
  shows a robot-mounted ``SpinningLidar`` and visualizes its scan as a rayrai
  point cloud.
* :doc:`Rayrai ArUco marker <examples/rayrai/rayrai_aruco_marker>` covers
  marker rendering and camera-facing visual output.

Use the package examples above as public reference programs for sensor
rendering workflows.

A typical RGB/depth setup has four parts: load a URDF with sensors, switch the
camera sensors to manual measurements, create rayrai camera objects from those
sensors, and render the sensor views every frame.

.. code-block:: cpp

    auto world = std::make_shared<raisim::World>();
    world->addGround();

    std::vector<std::string> modules = {"d455"};
    auto* robot = world->addArticulatedSystem(go1Urdf, modules, go1ResourceDir);

    auto* rgbCam = robot->getSensorSet("d455_front")
                   ->getSensor<raisim::RGBCamera>("color");
    auto* depthCam = robot->getSensorSet("d455_front")
                     ->getSensor<raisim::DepthCamera>("depth");

    rgbCam->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);
    depthCam->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);

    auto viewer = std::make_shared<raisin::RayraiWindow>(world, 1280, 720);
    auto rgbCamera = std::make_shared<raisin::Camera>(*rgbCam);
    auto depthCamera = std::make_shared<raisin::Camera>(*depthCam);

    const auto& rgbProp = rgbCam->getProperties();
    const int rgbWidth = std::max(1, rgbProp.width);
    const int rgbHeight = std::max(1, rgbProp.height);
    std::vector<char> rgbBuffer(size_t(rgbWidth) * size_t(rgbHeight) * 4);

    const auto& depthProp = depthCam->getProperties();
    const int depthWidth = std::max(1, depthProp.width);
    const int depthHeight = std::max(1, depthProp.height);
    std::vector<float> depthBuffer(size_t(depthWidth) * size_t(depthHeight));

    while (running) {
      world->integrate();

      viewer->renderWithExternalCamera(*rgbCam, *rgbCamera, {});
      viewer->renderWithExternalCamera(*depthCam, *depthCamera, {});
      viewer->renderDepthPlaneDistance(*depthCam, *depthCamera);

      rgbCamera->getRawImage(*rgbCam,
                             raisin::Camera::SensorStorageMode::CUSTOM_BUFFER,
                             rgbBuffer.data(),
                             rgbBuffer.size(),
                             /*flipVertical=*/false);
      depthCamera->getRawImage(*depthCam,
                               raisin::Camera::SensorStorageMode::CUSTOM_BUFFER,
                               depthBuffer.data(),
                               depthBuffer.size(),
                               /*flipVertical=*/false);
    }

``renderWithExternalCamera`` renders from the sensor pose; the ``raisin::Camera``
constructed from the sensor carries its intrinsics and resolution.
``renderDepthPlaneDistance`` converts the depth camera render into a linear
plane-distance texture and the ``float`` readback buffer contains one depth
value per pixel. The RGB readback buffer contains four bytes per pixel in BGRA
order.

For LiDAR, ``rayrai_lidar_pointcloud`` computes the scan of the robot-mounted
``SpinningLidar`` on the CPU with ``update(world)``, transforms the sensor-frame
points to world coordinates and displays them as a rayrai point cloud. To
measure the scan on the GPU instead, call ``measureSpinningLidarSingleDrawGPU``,
which casts the same rays against the same collision geometry and stores the
same sensor-frame points in the LiDAR with ``setScan()``:

.. code-block:: cpp

    auto* lidar = robot->getSensorSet("livox_lidar_0")
                  ->getSensor<raisim::SpinningLidar>("lidar");

    lidar->updatePose();
    // toGlm(): small conversion helper defined in rayrai_lidar_pointcloud.cpp
    const glm::dvec3 posW = toGlm(lidar->getPosition());
    const glm::dmat3 rotW = toGlm(lidar->getOrientation());
    viewer->measureSpinningLidarSingleDrawGPU(*lidar, posW, rotW);


Examples
====================================

* Use the dedicated rayrai RGB, depth, LiDAR, and ArUco examples above for
  robot-attached rendered sensor buffers.
* Use ``rayrai_complete_showcase`` when you need one runnable scene that
  combines RGB/depth cameras, LiDAR visualization, and custom visuals.
* ``World::captureDepthCamera`` supports depth, segmentation, optional hit
  points, and timestamps without a visualizer, but it is the CPU fallback path.


Parent Class API
=====================

.. doxygenclass:: raisim::Sensor
   :members:

Depth Camera API
=====================

.. doxygenclass:: raisim::DepthCamera
   :members:

RGB Camera API
=====================

.. doxygenclass:: raisim::RGBCamera
   :members:

Inertial Measurement Unit API
======================================

.. doxygenclass:: raisim::InertialMeasurementUnit
   :members:
