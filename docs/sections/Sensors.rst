#############################
Sensors
#############################

RaiSim supports these sensor workflows:

* **URDF-attached sensors** for robots. Sensor metadata is loaded from a sensor
  XML file attached to a URDF link. This is the right workflow for IMU, RGB,
  depth, and spinning LiDAR sensors that should move with an articulated system.
* **rayrai-rendered RGB/depth/segmentation sensors** for camera observations. When rayrai is
  available, use in-process rayrai rendering with
  ``Sensor::MeasurementSource::MANUAL`` for RGB/depth sensor buffers. This is
  the recommended path for rendered images, screenshots, and dataset generation.
* **World-level CPU depth-camera capture** only as a deterministic headless
  fallback. ``World::captureDepthCamera`` casts CPU rays from an arbitrary camera
  pose and returns depth, object segmentation, optional hit points, and a
  timestamp without requiring a renderer.

The CPU ray-query paths are single-threaded and deterministic. Use in-process
rayrai rendering for RGB/depth images when it is available. A connected TCP
visualizer can also render due ``MANUAL`` cameras and return their images to
``RaisimServer``; the server applies the sensor noise when committing those buffers.


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

There is no ``VISUALIZER`` measurement source in the RaiSim API. TCP camera
requests use ``MANUAL`` sensors: a connected visualizer renders requested frames,
and ``RaisimServer`` writes the returned images and acquisition timestamps.

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
Finite values are plane depth in meters. CPU ray captures use ``NaN`` for
missing pixels; clean raw renderer captures may retain a different missing-return
encoding. Active noise models normalize invalid depths to ``NaN``.

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


.. _segmentation-camera:

Segmentation camera
====================================

``raisim::SegmentationCamera`` reports, for every pixel, which object, which body of that object
and which semantic label the camera sees. It fills up to three ``int32`` images of
``width * height`` values, row-major with row 0 at the top of the image:

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Image
     - Value
     - Nothing hit
   * - ``getObjectImage()``
     - World index of the object, ``World::getObject(index)`` (``Object::getIndexInWorld()``)
     - ``-1``
   * - ``getBodyImage()``
     - Body index within that object: an ``ArticulatedSystem`` body
       (``getBodyNames()`` order); ``0`` for single bodies, compounds, terrain and deformables
     - ``-1``
   * - ``getLabelImage()``
     - Semantic label of that body; ``0`` for geometry without a label
     - ``-1``

``-1`` (``SegmentationCamera::kBackground``) in every image means nothing was hit within
``[clipNear, clipFar]``. The object and body images hold ``-2``
(``SegmentationCamera::kNotWorldObject``) for rayrai visuals that are not RaiSim objects; they
appear only when a renderer is asked to draw visualization objects. Number your own classes from
1: label 0 means unlabeled.

The segmentation type selects the images, as Gazebo's ``segmentation_type`` does:

* ``SEMANTIC``: the label image;
* ``INSTANCE``: the object and body images;
* ``PANOPTIC`` (default): all three.

The images a type does not select are empty (``None`` in Python).

Labels
------

Labels are plain non-negative integers set on objects. An articulated system's bodies take its
label, unless a body has its own; a body label takes precedence whatever the order of the calls.
Links merged into their parent through fixed joints are part of the parent body.

.. code-block:: cpp

    table->setSegmentationLabel(3);
    robot->setSegmentationLabel(1);                       // every body of the robot
    robot->setBodySegmentationLabel("gripper_link", 2);  // except the gripper
    robot->clearBodySegmentationLabels();                // all bodies take the object label again

``World`` keeps optional class names for labels, like Isaac Lab's ``idToLabels``:

.. code-block:: cpp

    box->setSegmentationLabel(world.addSegmentationClass("box"));  // 1 the first time, then the same label
    const std::string& name = world.getSegmentationClassName(label);  // "" for an unnamed label

World XML files store labels and classes. An object takes ``segmentation_label="3"`` or
``segmentation_class="box"``; an articulated system also takes
``<body_segmentation body="gripper_link" label="2"/>`` (or ``class="gripper"``) children; the
world's names are kept in ``<segmentation_classes><class label="1" name="box"/></segmentation_classes>``.
``.rscene`` files label objects and articulated systems with their ``semanticClass`` key, which
RaiSim Engine 2 writes.

Attaching a segmentation camera
-------------------------------

The sensor XML matches the RGB and depth cameras. The optional ``<segmentation type="..."/>``
element takes ``semantic``, ``instance`` or ``panoptic``; other values are rejected.

.. code-block:: xml

    <sensor name="segmentation" type="segmentation" update_rate="30">
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <camera horizontal_fov="1.5" projection="pinhole">
        <image width="640" height="480"/>
        <clip near="0.1" far="20"/>
        <segmentation type="panoptic"/>
      </camera>
    </sensor>

Give it the pose and camera parameters of an RGB camera to get labels aligned with its image.

Measuring
---------

* **rayrai (recommended)**: keep the sensor ``MANUAL`` and render it on the GPU from the visual
  geometry; see :ref:`rayrai segmentation <rayrai-segmentation>`.

  .. code-block:: cpp

      raisin::Camera segmentationCamera(*segmentation);
      viewer.renderSegmentation(*segmentation, segmentationCamera);
      segmentationCamera.getRawImage(*segmentation);  // fills the sensor's images

* **CPU ray cast**: with ``MeasurementSource::RAISIM`` each update casts one ray per pixel against
  the collision geometry (owning body excluded), like a ``RAISIM`` depth camera. Rays start on the
  near plane and end on the far plane, so clipping matches the renderer's. It needs no GPU and is
  deterministic, but costs one ray test per pixel.

  .. code-block:: cpp

      segmentation->setMeasurementSource(raisim::Sensor::MeasurementSource::RAISIM);
      world.integrate();  // updates the images at the sensor's update rate

* **TCP viewer**: a ``RaisimServer`` asks a connected ``rayrai_tcp_viewer`` to render due ``MANUAL``
  segmentation cameras, as it does for RGB and depth cameras, and stores the returned images. The
  viewer shows them in its sensor pane in id colors.

The renderer draws visual geometry and the ray cast collision geometry, so the two differ where
those differ. Granular particles and objects hidden in the viewer are ray-cast but not drawn by
rayrai, and the ray cast skips the camera's own body while the renderer draws it. World indices
refer to the world at the measurement time: ``World::removeObject()`` moves the last object into
the freed index.

Display colors
--------------

``SegmentationCamera::colorize(ids, rgba, background)`` gives each non-negative id a fixed,
well-separated color that does not depend on the other ids, and negative ids the background color
(transparent black by default), like Drake's ``ColorizeLabelImage``:

.. code-block:: cpp

    std::vector<unsigned char> rgba;
    raisim::SegmentationCamera::colorize(segmentation->getLabelImage(), rgba);

Python
------

.. code-block:: python

    camera = robot.getSensorSet("camera_link").getSensor("segmentation")
    camera.setMeasurementSource(raisim.SensorMeasurementSource.RAISIM)
    world.integrate()
    objects = camera.getObjectImage()   # (H, W) int32, -1 where nothing was hit
    labels = camera.getLabelImage()     # (H, W) int32
    colors = raisim.SegmentationCamera.colorize(labels)  # (H, W, 4) uint8
    mask = objects == table.getIndexInWorld()

raisimPy does not bind rayrai: Python programs use the CPU ray cast or a TCP viewer.

Coming from another simulator
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Simulator
     - RaiSim equivalent
   * - MuJoCo ``Renderer.enable_segmentation_rendering()``, ``(H, W, 2)`` ``[objid, objtype]``
     - ``getObjectImage()`` and ``getBodyImage()``; ``-1`` background in both
   * - PyBullet ``getCameraImage`` with ``ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX``
     - ``uid`` → ``getObjectImage()``, ``link`` → ``getBodyImage()`` (two images instead of
       ``uid + ((link + 1) << 24)``); ``-1`` background
   * - Isaac Lab ``data_types``: ``semantic_segmentation`` / ``instance_id_segmentation_fast``
     - ``SEMANTIC`` / ``INSTANCE`` cameras; ``idToLabels`` → ``World::getSegmentationClassName()``.
       Isaac's ``BACKGROUND = 0`` and ``UNLABELLED = 1`` are ``label + 1``
   * - Gazebo ``<sensor type="segmentation">``, ``<segmentation_type>``, label system ``<label>``
     - ``<sensor type="segmentation">`` with ``<segmentation type="..."/>``;
       ``Object::setSegmentationLabel()``; panoptic gives labels and instances
   * - Drake ``RgbdSensor`` label image, ``RenderLabel``, ``ColorizeLabelImage``
     - ``getLabelImage()`` with integer labels; ``SegmentationCamera::colorize()``
   * - Genesis ``segmentation_level='entity' | 'link'``, ``cam.render(segmentation=True)``
     - entity → ``getObjectImage()``, link → ``getObjectImage()`` with ``getBodyImage()``
   * - ManiSkill / SAPIEN actor or link ids, ``segmentation_id_map``
     - ``getObjectImage()`` with ``World::getObject(index)``; ``getBodyImage()`` for links
   * - Habitat ``SensorType.SEMANTIC`` with ``semantic_id``
     - ``SEMANTIC`` camera with ``Object::setSegmentationLabel()``
   * - CARLA semantic / instance segmentation cameras
     - ``SEMANTIC`` / ``PANOPTIC`` cameras; tags are labels


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


.. _sensor-noise-models:

Sensor noise
====================================

Robot-mounted RGB, depth, and spinning LiDAR sensors support optional measurement
noise. Defaults preserve the clean measurements. Each sensor owns a random stream;
``resetNoise()`` restarts it. Seed zero derives a deterministic stream from the
sensor's full name and its owning object's stable World id, decorrelating sensors
on one or several robots. An unnamed detached sensor has no identity to mix.
Nonzero ``seed`` values select explicit streams, so matching explicit seeds and
models replay the same draws, independently of sensor names.

``World::setSensorNoiseSeed(uint64_t)`` mixes an environment seed into all attached
sensor streams. Call it before adding robots, or between complete physics steps
to restart existing random streams. Measurements, IMU bias and filters, the
acquisition clock and delayed samples are preserved. Use ``imu->reset()`` when
restarting the complete IMU model is intended. The default World seed is zero;
cloned Worlds with the same seeds and object creation order deliberately replay. In parallel environments,
set a distinct World seed from the environment index. All 64 seed bits participate;
this does not alter individual sensor property seeds. World XML and checkpoints
save the environment seed, and checkpoint restoration resumes the saved streams.
Resetting properties also resets the stream; for depth, call
``updateRayDirections()`` after editing properties. Repeatability is for the same
build and sequence of captures; floating-point math can differ between platforms.

.. list-table:: Measurement models
   :header-rows: 1
   :widths: 15 40 45

   * - Sensor
     - Properties
     - Meaning
   * - RGB
     - ``noiseType``, ``mean``, ``std``
     - Independent Gaussian or uniform read noise per color channel, in normalized
       intensity units: 0 is black, 1 is full scale. Alpha is preserved.
   * - RGB
     - ``shotNoise``, ``rowNoiseStd``
     - Independent signal-dependent Gaussian noise with variance
       ``shotNoise * intensity``, plus a Gaussian offset shared across each row
       and its three color channels.
   * - Depth
     - ``noiseType``, ``mean``, ``std``, ``stdDepthSquared``
     - Bias in meters and Gaussian/uniform noise with standard deviation
       ``std + stdDepthSquared * z * z``, where ``z`` is clean optical depth in meters.
   * - Depth
     - ``resolution``, ``dropoutProbability``
     - Quantization step in meters and independent per-valid-pixel dropout
       probability. Dropped or out-of-range measurements become NaN.
   * - LiDAR
     - ``rangeBias``, ``rangeNoiseStd``, ``angularNoiseStd``
     - Range bias and Gaussian range noise in meters; independent Gaussian yaw
       and pitch errors in radians, applied to the returned sensor-frame hits.
   * - LiDAR
     - ``resolution``, ``dropoutProbability``
     - Range quantization in meters and independent per-hit dropout. Dropped and
       out-of-range hits are omitted, preserving the order of surviving hits.

The legacy ``World::captureDepthCamera`` has a separate, reseeded-per-capture
noise API and uses a uniform half-range; see `Optional deterministic depth noise`_.
For attached sensors, Gaussian and uniform ``std`` both mean standard deviation: the uniform interval
has half-width ``sqrt(3) * std``. ``NO_NOISE`` ignores the additive ``mean``/``std``
parameters; RGB shot/row effects and depth quantization/dropout remain independently
configurable. RGB is rounded and clamped to byte values after all noise is added.
Depth and LiDAR are quantized after noise, then checked against their configured
range. Noisy depths outside the finite float32 storage range become NaN before
conversion. Non-finite and missing returns never turn into valid hits. Noise does
not change the acquisition timestamp, pose, scan sweep, or alpha channel.

RGB ``shotNoise`` is a high photon-count Gaussian approximation in the **rendered
intensity space**, rather than a calibrated photon/electron simulation. It does
not model low-light Poisson statistics or an image signal processor. LiDAR angle
noise perturbs a hit's reported direction; it does not widen the beam or cast a
second ray. These models do not add rolling shutter, motion distortion, multipath,
material-dependent return probabilities, or camera/LiDAR latency.

Noise is reproducible on the same platform and build when the scene, capture
order, properties and seed match. Floating-point Gaussian transforms may differ
between platforms. LiDAR draws are assigned in retained-hit order: a missing or
extra hit shifts the draws of subsequent hits, so CPU/GPU scans with different
hit sets need not share per-ray errors.

Shipped RealSense sensor definitions explicitly disable camera noise to preserve
clean robot captures. Set ``type="gaussian"`` or ``type="uniform"`` with a nonzero
``stddev`` to enable it. Third-party files with existing nonzero noise settings
now activate those settings; use ``type="none"`` to retain clean captures.
Unknown camera noise types raise ``std::invalid_argument``.

Disabled depth models preserve raw input bytes, including the renderer's missing
return encoding. Active models normalize non-positive and out-of-clip inputs to
NaN. Depth ``resolution`` is a quantization step in meters, independent of image
``width`` and ``height``.

Configure a depth camera in C++:

.. code-block:: cpp

    auto& p = depth->getProperties();
    p.noiseType = decltype(p.noiseType)::GAUSSIAN;
    p.std = 0.002;                 // 2 mm at zero distance
    p.stdDepthSquared = 0.001;     // sigma is 18 mm at 4 m
    p.resolution = 0.001;          // 1 mm depth steps
    p.dropoutProbability = 0.01;
    p.seed = 101;
    depth->updateRayDirections();

For RGB, copy the immutable properties and pass them to ``setProperties()``:

.. code-block:: cpp

    auto p = rgb->getProperties();
    p.noiseType = decltype(p.noiseType)::GAUSSIAN;
    p.std = 0.01;                 // about 2.55 byte levels
    p.shotNoise = 0.0001;
    p.rowNoiseStd = 0.002;
    p.seed = 43;
    rgb->setProperties(p);

In URDF sensor XML, camera noise settings belong on an optional
``<camera><noise>`` element; LiDAR takes an optional ``<noise>`` child of
``<sensor>``. Omitted attributes retain their clean defaults, so effect-only
settings such as ``<noise shot_noise="0.001"/>`` or
``<noise dropout_probability="0.01"/>`` need no dummy read-noise fields.
Read noise defaults to ``NO_NOISE``; set ``type="gaussian"`` or ``type="uniform"``
when configuring additive read noise. Unknown noise types are rejected.

.. code-block:: xml

    <!-- Inside a depth sensor's <camera>: -->
    <noise type="gaussian" mean="0" stddev="0.002"
           std_depth_squared="0.001" resolution="0.001"
           dropout_probability="0.01" seed="101"/>
    <!-- Inside an RGB sensor's <camera>: -->
    <noise type="gaussian" mean="0" stddev="0.01"
           shot_noise="0.0001" row_noise_std="0.002" seed="43"/>
    <!-- Inside a spinning_lidar sensor: -->
    <noise range_bias="0.002" range_noise_std="0.01"
           angular_noise_std="0.001" resolution="0.001"
           dropout_probability="0.01" seed="987"/>

Noise is assigned to physical pixels in top-left image order. For a raw
bottom-left buffer, call ``applyNoise(data, size, false)``; Rayrai passes its
``flipVertical`` choice automatically. Resetting the stream before captures in
opposite layouts produces the same seeded measurements with reversed row storage.
Depth point-cloud conversion expects top-left rows; fisheye rays with a nonpositive
forward component have no valid positive optical depth and produce NaN points.

CPU depth/LiDAR ``update()`` and Rayrai's sensor-aware ``getRawImage()`` / GPU
LiDAR capture apply noise automatically, including custom output buffers. CPU
``DepthCamera::get3DPoints()`` follows the noisy depth image, including its NaNs.
For Rayrai, use ``depthToPointCloud()`` to reconstruct measured points; readback
does not populate ``get3DPoints()``. The point conversion handles pinhole and
fisheye optical depth.

Manual setters such as ``setImageBuffer()``, ``setDepthArray()``, and ``setScan()``
keep hardware/playback data unchanged. Call ``applyNoise()`` explicitly after a
manual setter to synthesize one measurement. Do not call it again after an
automatic capture. Raw ``Camera`` readbacks without a RaiSim sensor remain clean.
Asynchronous raw readbacks can be processed explicitly with the sensor's
``applyNoise(pointer, size)`` overload when the returned frame is consumed.

For TCP cameras, the visualizer sends clean pixels and RaiSim applies the model
when it commits the received image. The protocol is unchanged; the visualizer's
preview remains clean while the sensor buffer contains the noisy measurement.
There are no per-pixel allocations; all-disabled models return before processing
pixels. Rendering resolution, materials, lighting, and shadow quality are unchanged.

See :doc:`the sensor noise example <examples/rayrai/rayrai_sensor_noise>` for
reproducible RGB/depth comparison images and the exact capture calls.

.. _imu-sensor-model:

IMU
====================================

``raisim::InertialMeasurementUnit`` produces accelerometer and gyroscope
measurements for a frame attached to an articulated system. The default is an
ideal sensor; bias, noise, bandwidth, quantization, saturation and delivery
latency (0 to 60 seconds) are configurable. Sensor measurements do not require enabling inverse
dynamics on the robot.

Start with the runnable :doc:`examples/server/imu_sensor_model` example. It
compares an ideal sensor and a modeled sensor at the same mounting point and
writes CSV without a viewer. :doc:`examples/server/sensor_suite` demonstrates
using an IMU alongside rendered cameras and LiDAR.

Reading measurements
------------------------------------

With ``MeasurementSource::RAISIM`` (the default), read the IMU after
``world.integrate()``. Getters hold the latest delivered sample between updates;
a changed measurement value is not a reliable way to detect a new sample.

.. code-block:: cpp

    auto* imu = robot->getSensorSet("base")
                    ->getSensor<raisim::InertialMeasurementUnit>("imu");
    double lastAcquisition = -1.0;

    // Inside the simulation loop:
    world.integrate();
    const double acquired = imu->getUpdateTimeStamp();
    if (acquired >= 0.0 && acquired != lastAcquisition) {
      lastAcquisition = acquired;
      const Eigen::Vector3d acc = imu->getLinearAcceleration(); // IMU frame, m/s^2
      const Eigen::Vector3d gyro = imu->getAngularVelocity();   // IMU frame, rad/s
      const double delivered = imu->getDeliveryTimeStamp();   // simulation seconds
      // Consume acc/gyro with their acquisition timestamp, not the current world time.
    }

With ``MANUAL``, RaiSim leaves the measurements unchanged. User code can write
hardware data with ``setLinearAcceleration()``, ``setAngularVelocity()`` and
``setOrientation()``, and manage its timestamps through ``setUpdateTimeStamp()``.

How the ideal reading is computed
------------------------------------

The IMU uses the completed physics step's pose and angular velocity. Generalized
acceleration is obtained from the solver's velocity change over a physics step
of duration :math:`h`:

.. math::

    \dot v \simeq \frac{v_{\mathrm{after}}-v_{\mathrm{before}}}{h}.

A forward acceleration pass through the joint tree adds joint motion, rotational
and Coriolis terms. For an IMU at world-frame offset :math:`r` from its body's
reference point, the resulting acceleration is

.. math::

    a_{\mathrm{IMU}} = a_{\mathrm{body}}
      + \alpha\times r + \omega\times(\omega\times r).

The last two terms account for tangential and centripetal acceleration. A sensor
away from a rotation axis therefore measures acceleration even when the body's
reference point is stationary.

Let :math:`R_{WS}` map sensor coordinates into world coordinates. The ideal
accelerometer reports **specific force**, and the gyro reports angular velocity,
both expressed in the IMU's own axes:

.. math::

    f = R_{WS}^{T}(a_{\mathrm{IMU}}-g), \qquad
    \omega_S = R_{WS}^{T}\omega_W.

For an aligned, upright sensor with gravity :math:`g=(0,0,-9.81)`:

.. list-table:: Ideal accelerometer checks
   :header-rows: 1

   * - Motion
     - Expected reading
   * - Stationary and supported
     - :math:`(0,0,9.81)` m/s²
   * - Free fall without rotation
     - :math:`(0,0,0)` m/s²
   * - Constant rotation about +Z, zero gravity, sensor 0.1 m along the body's +X axis, :math:`\omega_Z=2` rad/s
     - :math:`(-0.4,0,0)` m/s²; the gyro reads :math:`(0,0,2)` rad/s

Rotating the sensor mount rotates these vectors into the sensor frame. The
accelerometer is not a gravity-removed linear-acceleration output.

``getOrientation()`` and ``getGroundTruthOrientation()`` return the same
**ground-truth** sensor-to-world quaternion in ``(w, x, y, z)`` order. It belongs
to the delivered sample's acquisition time, including when delivery is delayed.
It is not an attitude estimate derived from noisy accelerometer and gyro data;
adding channel noise does not make this quaternion noisy. Validate an attitude
estimator using the raw accelerometer and gyro channels instead.

Sampling and delivery
------------------------------------

The acquisition clock retains its phase when a deadline falls between physics
steps. Acquisition takes place at the first completed step at or after the
deadline; the next deadline advances from the original clock rather than the
rounded acquisition time. There is at most one acquisition per physics step,
so a requested rate above the physics rate cannot create additional motion data.

For a simulation starting at time zero with 1 ms physics, a 400 Hz IMU and 3 ms
latency, the first samples are:

.. list-table:: Simulation times in milliseconds
   :header-rows: 1

   * - Acquisition
     - Delivery
   * - 1
     - 4
   * - 4
     - 7
   * - 6
     - 9
   * - 9
     - 12
   * - 11
     - 14

The acquisition intervals alternate between 2 ms and 3 ms. The mean rate remains
400 Hz. ``getUpdateTimeStamp()`` is the **acquisition** time of the most recently
delivered sample. ``getDeliveryTimeStamp()`` is when that sample became available.
Both are simulation times, not wall-clock times.

For latency :math:`L`, a sample acquired at :math:`t_s` becomes available at the
first physics step with :math:`t\ge t_s+L`. Delivery is checked every physics
step, including steps without a new acquisition. Latency is therefore rounded
up to the available physics times. Acceleration, gyro and ground-truth orientation
are queued together. Before the first delivery, timestamps are negative and the
quaternion is the identity; do not treat those initial buffers as measurements.

Queue storage grows with acquired samples, up to 65,536 pending measurements
(about 6 MiB per IMU). If this capacity is exhausted, or growth cannot allocate
memory, new samples are dropped until room becomes available. Existing queued
samples keep their acquisition times and delivery latency. A warning is emitted
on the first loss after ``reset()``; ``getDroppedMeasurementCount()`` reports the
cumulative loss and is preserved by checkpoints. Reduce latency or update rate if
this count is nonzero. The physics timestep continues normally.

Sleeping and fixed bodies continue to produce samples, including specific force
from gravity and the configured stochastic errors. With filtering enabled, the
filter advances every physics step, even when no new output sample is due.

Python RGB, LiDAR and IMU ``getProperties()`` return independent copies. Edit the
copy and call ``setProperties()`` to validate and apply it; a rejected edit leaves
the sensor unchanged. Depth retains mutable properties and requires
``updateRayDirections()`` after edits.

Configuring the sensor model
------------------------------------

Accelerometer and gyro settings are independent. Prefer const access when copying
properties, then apply the complete configuration with ``setProperties()``:

.. code-block:: cpp

    #include <utility> // std::as_const

    auto properties = std::as_const(*imu).getProperties();
    properties.seed = 43;
    properties.accelerometer.bias = Eigen::Vector3d(0.01, -0.02, 0.03);
    properties.accelerometer.noiseDensity = 0.002;
    properties.accelerometer.biasRandomWalk = 0.0001;
    properties.accelerometer.cutoffFrequency = 30.0;
    properties.accelerometer.resolution = 0.001;
    properties.gyroscope.bias = Eigen::Vector3d(0.001, -0.002, 0.003);
    properties.gyroscope.noiseDensity = 0.0002;
    properties.gyroscope.biasRandomWalk = 0.00001;
    properties.gyroscope.cutoffFrequency = 50.0;
    properties.gyroscope.resolution = 0.0001;
    properties.maxAcc = 16.0 * 9.81; // 16 g, independently on each axis
    properties.maxAngVel = 34.906585; // approximately 2000 degrees/s, in rad/s
    properties.latency = 0.003;
    imu->setProperties(properties);
    imu->setUpdateRate(400.0);

These are illustrative settings, not a calibrated preset for a specific device.
In the table below, a "channel unit" is m/s² for the accelerometer and rad/s for
the gyro. All channel error settings default to zero (disabled).

.. list-table:: Model parameters
   :header-rows: 1
   :widths: 25 25 50

   * - Property
     - Unit
     - Meaning
   * - ``bias``
     - Channel units, three axes
     - Initial offset in sensor coordinates.
   * - ``noiseDensity``
     - Channel units / sqrt(Hz)
     - Continuous white-noise strength; not the per-sample standard deviation.
   * - ``biasRandomWalk``
     - Channel units / sqrt(s)
     - Strength of the Brownian bias increments.
   * - ``cutoffFrequency``
     - Hz
     - First-order low-pass cutoff; zero bypasses filtering.
   * - ``resolution``
     - Channel units
     - Quantization step; zero disables quantization.
   * - ``maxAcc``, ``maxAngVel``
     - m/s², rad/s
     - Positive symmetric per-axis clipping ranges. Defaults are effectively unlimited.
   * - ``latency``
     - Seconds
     - Delay between acquisition and availability, from 0 to 60 seconds, shared by all reported channels.
   * - ``seed``
     - Unsigned 32-bit integer
     - Zero derives a stream from sensor identity; nonzero selects an explicit stream. The World environment seed is mixed into both.

Bias and unfiltered noise
------------------------------------

For each axis, the initial bias is the configured ``bias``. With bias random-walk
strength :math:`\sigma_b`, a bias update over elapsed time :math:`\Delta t` is

.. math::

    b_{k+1}=b_k+\sigma_b\sqrt{\Delta t}\,\xi_k,
    \qquad \xi_k\sim\mathcal N(0,1).

Bias is stepped at physics rate for filtered channels and at acquisition times
for unfiltered channels, using the corresponding elapsed time. Its uncertainty
grows with elapsed time; it is not independent additive noise at each sample.
``getAccelerometerBias()`` and ``getGyroscopeBias()`` expose the current simulated
bias state for diagnostics. Those diagnostic states are not delayed to match a
previously acquired measurement.

When ``cutoffFrequency`` is zero, the ideal signal is sampled at acquisition and
the output before quantization/clipping is

.. math::

    u_k=x_k+b_k+\frac{\sigma_n}{\sqrt{\Delta t_s}}\,\eta_k,
    \qquad \eta_k\sim\mathcal N(0,1),

where :math:`\sigma_n` is ``noiseDensity`` and :math:`\Delta t_s` is the actual
acquisition interval. This noise-variance convention assumes ideal noise
averaging/decimation, as in
`Kalibr's IMU noise model <https://github.com/ethz-asl/kalibr/wiki/IMU-Noise-Model>`_.
It does not average the ideal motion signal over that interval. Enable a
suitable cutoff when signal bandwidth and filtering before downsampling matter.

For example, noise density 0.002 m/s²/sqrt(Hz) at a uniform 400 Hz acquisition
rate gives a sample standard deviation of 0.04 m/s² without the explicit filter.
With rounded acquisition intervals, the variance follows each actual interval.
Accelerometer and gyro axes use independent random draws, with separate channel
parameters.

Low-pass filtering and filtered noise
-------------------------------------

For a positive cutoff :math:`f_c`, define :math:`\lambda=2\pi f_c` and
:math:`a=\exp(-\lambda h)`, where :math:`h` is the physics step. The signal and
bias are filtered together:

.. math::

    s_{k+1}=a\,s_k+(1-a)(x_{k+1}+b_{k+1}).

This is the exact first-order response for a held input over the step. The signal
filter initializes from the first input to avoid an artificial startup transient.
White noise passing through the same bandwidth is advanced separately:

.. math::

    z_{k+1}=a\,z_k+
      \sigma_n\sqrt{\frac{\lambda}{2}(1-a^2)}\,\eta_k.

The filtered reading is :math:`u=s+z`. This is the exact discrete
Ornstein–Uhlenbeck transition for that continuous white-noise model. Its
steady-state variance is :math:`\sigma_n^2\pi f_c`, and its one-physics-step
correlation is :math:`\exp(-2\pi f_c h)`. Unlike drawing independent output noise,
it preserves the temporal correlation created by the sensor bandwidth. The noise
state starts at zero and approaches this stationary distribution after reset.

Filtering advances before output downsampling. Coefficients are cached for a
constant physics timestep and recomputed when the step changes. "Exact" here
refers to these filter transitions; the simulated motion and stepped bias remain
limited by the physics resolution and the held-input assumption.

Quantization and saturation
------------------------------------

At acquisition, each output axis is quantized to step :math:`q`, when configured,
then independently clipped to range :math:`[-M,M]`:

.. math::

    y_i=\operatorname{clip}\left(q\operatorname{round}(u_i/q),-M,M\right).

With ``resolution=0``, the rounding stage is skipped. ``maxAcc`` and ``maxAngVel``
are axis limits, so saturation on one axis does not reduce another axis. This
replaces the previous vector-norm scaling behavior.

The legacy ``noiseType``, ``mean`` and ``std`` settings now add output sample noise
to both channels after the explicit filter and before quantization. Uniform noise
has half-width ``sqrt(3) * std``. Prefer the independent channel noise densities
for new models; configuring both mechanisms adds both noise contributions.

Reset, replay and XML
------------------------------------

``setProperties()`` rejects invalid model parameters before applying them and
resets measurements, bias, filters, sampling clock, pending delivery queue and
random stream. Sensor ``name`` and ``full_name`` are immutable after construction;
changing them through ``setProperties()`` is rejected.
``resetNoise()`` restarts only the random stream, preserving measurements and model
history. ``reset()`` restarts the current model from its configured biases
and seed; it does not rewind the robot or an application controller. Legacy
mutable ``getProperties()`` access is supported: valid edits reset the model on
the next update. Invalid edits, including identity changes, restore the last valid
configuration and emit a warning, so they cannot interrupt a world step. Use
``setProperties()`` for immediate validation and exceptions before stepping.
Obtaining a mutable reference permanently enables property comparisons on each
update, even when it was obtained only to copy the settings. References remain
tracked after checkpoint restore. Prefer ``std::as_const(*imu).getProperties()``
for reading, or call through a const reference.

Repeated motion with the same seed on the same platform/build reproduces the
stochastic output. Cross-platform floating-point results need not be bit-identical.
The IMU uses the same full-width seeded stream and Gaussian sampler as the
camera/LiDAR models, so high seed bits also select different streams. World
checkpoints preserve the complete random state,
bias, filter state, acquisition clock, pending delayed samples and delivery timestamps.
For replay of a running simulation, restore a world checkpoint rather than merely
reapplying the same seed. Application controller and random state still belong to
the application.

URDF declarations support the same configuration:

.. code-block:: xml

    <sensor name="imu" type="imu" update_rate="400" seed="43" latency="0.003">
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
      <limit acc_limit="156.96" ang_vel_limit="34.906585"/>
      <accelerometer bias="0.01 -0.02 0.03" noise_density="0.002"
                     bias_random_walk="0.0001" cutoff_frequency="30" resolution="0.001"/>
      <gyroscope bias="0.001 -0.002 0.003" noise_density="0.0002"
                 bias_random_walk="0.00001" cutoff_frequency="50" resolution="0.0001"/>
    </sensor>

Use the channel child elements above as the canonical URDF form. Prefixed
``acc_*``/``gyro_*`` attributes remain accepted for serialized world XML; avoid
mixing those attributes with child elements in authored URDF. Legacy additive
sample noise can be configured with a ``<noise type="gaussian" mean="0"
stddev="0.01"/>`` child.

World XML export preserves this configuration, not the evolving stochastic state.
The model fields require matching RaiSim headers and binaries that provide
``InertialMeasurementUnit::ChannelProperties``; older binary packages do not
implement this API.

Accuracy and performance limits
------------------------------------

* Contact impulses contribute acceleration averaged over one physics step.
  Smaller timesteps may be required to resolve short impacts or vibration.
* A first-order filter is a configurable sensor approximation. There is no
  device-specific transfer function, temperature dependence, scale-factor error,
  cross-axis calibration error or attitude-estimation algorithm in this model.
* Delivery occurs only on physics steps. There is no wall-clock scheduling jitter
  or packet-loss model.
* The ideal path bypasses random draws and filtering. The zero-latency path needs
  no delay buffer; positive latency uses reusable queue storage. Acceleration is
  propagated once per articulated system when needed by its IMUs, without the
  inverse-dynamics force sweep.

For engine-side timing, the ``imu_sensor_benchmark`` target compares baseline
integration, ideal IMUs and the full model on one thread:
``build-benchmark/benchmark/imu_sensor_benchmark --steps=500000 --robots=16``.


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

Depth noise is disabled by default.

This legacy world-level capture API is separate from the attached
:ref:`sensor-noise-models` above. In this API, uniform ``depthNoiseStd`` is the
**half-range**, while attached ``DepthCamera::DepthCameraProperties::std`` is a
**standard deviation** (uniform half-range ``sqrt(3) * std``). Its generator is
reseeded on each capture: repeat captures with the same input repeat the noise,
rather than advancing an attached sensor's stream. The legacy standard-library
distributions reproduce within a platform/build; their Gaussian samples can differ
between C++ standard libraries. Use attached depth sensors for the shared model,
quantization, dropout, environment seeds and an advancing stream.

To enable it, set ``depthNoiseType``, ``depthNoiseMean``, ``depthNoiseStd``, and
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
