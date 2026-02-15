#############################
Sensors
#############################

This module provides a way to specify sensor properties using a URDF-like format.
We provide a few implementation examples for depth camera, RGB camera, and IMU.

We use RaiSim to interface with the real hardware.
This makes sense because we want to use the same code to control both the simulated and real robot.
This sensor package is also designed for real robots.
This might be confusing, but it keeps the code simpler and shorter.


How to attach a sensor to a link
====================================
Before further explanations, we clarify some terms that will be used throughout this documentation.
**"sensor"** is a part that outputs a single type of information. For example, RGB cameras, depth cameras, gyroscopes, etc.
**"sensor_set"** is a set of sensors contained in a single link.
For example, Intel RealSense (it contains IMU+RGB+depth) and IMU (gyro and accelerometer) are "sensor_set".
A sensor_set can contain only a single sensor.

Create a link for a sensor_set and give it a ``sensor`` attribute as

.. code-block:: xml

    <link name="realsense_d435" sensor="realsense435.xml"/>

The `realsense435.xml <https://github.com/raisimTech/raisim2Lib/blob/master/rsc/anymal_c/sensors/realsense435.xml>`_ file should specify all necessary details of the sensor.
It should be stored in the same directory as the URDF file.
If it is not found, RaiSim will search the following directories in order: ``[urdf_dir]/sensor``, ``[urdf_dir]/sensors``, ``[urdf_dir]/..``, and ``[urdf_dir]/../sensors``.
An example URDF file can be found `here <https://github.com/raisimTech/raisim2Lib/blob/master/rsc/anymal_c/urdf/anymal_sensored.urdf>`__.


Update
====================================
The sensor update method can be set using ``raisim::Sensor::setMeasurementSource``.
Available sources are:

* ``RAISIM``: measurement is updated from the simulation side. This is the default
  for IMU and spinning LiDAR sensors, and can be invoked manually via
  ``sensor->update(*world)`` where supported.
* ``VISUALIZER``: a visualizer fills the camera buffers (RaisimUnity/Unreal or the
  rayrai TCP viewer).
* ``MANUAL``: user code is responsible for writing the buffers (recommended for
  in-process rayrai rendering or real hardware integration).

Notes by sensor type (per the public headers):

* ``RGBCamera`` does not implement a RaiSim-side update. Use a visualizer or
  manual updates (rayrai + ``raisin::Camera::getRawImage``).
* ``DepthCamera`` can be updated by RaiSim (``DepthCamera::update(*world)``) or by
  visualizers/manual updates.
* ``InertialMeasurementUnit`` is updated by RaiSim and rejects visualizer updates.

The update is performed at the specified frequency (i.e., ``update_rate`` in the sensor XML).
When reading sensor data from another thread, use RAII locking via the built-in
``lock()``/``unlock()`` hooks:

.. code-block:: cpp

    {
      std::lock_guard<raisim::Sensor> lock(*depthSensor);
      const auto &depth = depthSensor->getDepthArray();
      // use depth here
    }

Example
====================================
Example: https://github.com/raisimTech/raisim2Lib/blob/master/examples/src/server/sensor_suite.cpp

Using sensors with rayrai
====================================
rayrai can render RaiSim camera sensors in-process. The rayrai complete showcase
(``examples/src/rayrai/rayrai_complete_showcase.cpp``) demonstrates how to wire RGB, depth,
and LiDAR sensors together. The key steps are:

1) Load a sensored URDF and fetch the sensors by name:

.. code-block:: cpp

    auto anymal = world->addArticulatedSystem(anymalUrdf);
    auto lidar = anymal->getSensorSet("lidar_link")
                   ->getSensor<raisim::SpinningLidar>("lidar");
    auto rgbCam = anymal->getSensorSet("depth_camera_front_camera_parent")
                   ->getSensor<raisim::RGBCamera>("color");
    auto depthCam = anymal->getSensorSet("depth_camera_front_camera_parent")
                     ->getSensor<raisim::DepthCamera>("depth");

2) Let rayrai control sensor updates by setting the measurement source to MANUAL:

.. code-block:: cpp

    lidar->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);
    rgbCam->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);
    depthCam->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);

3) For RGB/depth cameras, render with rayrai every frame:

.. code-block:: cpp

    auto viewer = std::make_shared<raisin::RayraiWindow>(world, 1280, 720);
    auto rgbCamera = std::make_shared<raisin::Camera>(*rgbCam);
    auto depthCamera = std::make_shared<raisin::Camera>(*depthCam);

    viewer->renderWithExternalCamera(*rgbCam, *rgbCamera, {});
    viewer->renderWithExternalCamera(*depthCam, *depthCamera, {});
    viewer->renderDepthPlaneDistance(*depthCam, *depthCamera);

``renderWithExternalCamera`` renders with the sensor intrinsics and pose.
If you need CPU access to the rendered buffers, call ``raisin::Camera::getRawImage``
to populate either the sensor buffers or a custom buffer. ``renderDepthPlaneDistance``
produces a linear depth texture for depth sensors.

4) To read raw images on the CPU, allocate a buffer using the sensor properties
and call ``raisin::Camera::getRawImage``:

.. code-block:: cpp

    const int w = std::max(1, rgbCam->getProperties().width);
    const int h = std::max(1, rgbCam->getProperties().height);
    std::vector<char> rgbBuffer(size_t(w) * size_t(h) * 4);
    rgbCamera->getRawImage(*rgbCam, raisin::Camera::SensorStorageMode::CUSTOM_BUFFER,
      rgbBuffer.data(), rgbBuffer.size(), false);

Depth uses a ``float`` buffer with ``width * height`` entries.

5) For LiDAR, update the pose and request a scan from RaiSim, then transform
points from the sensor frame to world coordinates:

.. code-block:: cpp

    lidar->updatePose();
    const auto sensorPosW = lidar->getPosition();
    const auto sensorRotW = lidar->getOrientation();
    lidar->update(*world);
    const auto& scanS = lidar->getScan();
    // pW = R_W * pS + t_W

The example converts the scan into a rayrai ``PointCloud`` for visualization.

Notes:

- For camera sensors, use rayrai for updates (manual source) instead of calling
  ``sensor->update(*world)``.
- The sensor update rate in the XML is still respected by RaiSim sensors; if you
  want to throttle updates in rayrai, skip rendering on frames that are too close.

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
