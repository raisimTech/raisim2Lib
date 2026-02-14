#############################
rayrai Visualizer
#############################

Overview
========
rayrai is an in-process C++ renderer for RaiSim. It renders into an offscreen OpenGL
texture and is designed to embed in custom UIs (ImGui, Qt, etc.) or headless pipelines.
Unlike RaisimUnity or RaisimUnreal, rayrai runs inside the simulation process and exposes
direct access to render targets, picking, and custom visuals.

Key characteristics:

* In-process rendering (no server required)
* Offscreen textures for UI embedding or sensor simulation
* Lightweight scene management for RaiSim objects and custom visuals
* Support for point clouds, coordinate frames, and picking
* Direct ImGui/SDL2 integration patterns and headless/offscreen workflows

The public API lives in the `raisin` namespace (note the spelling). The primary entry
point is ``raisin::RayraiWindow``.

Dependencies
============
rayrai depends on SDL2, OpenGL, glbinding, glm, assimp, stb, and imgui. The installed
package in ``rayrai/<OS>`` ships the headers and CMake config for these dependencies.

Included binary (recommended)
=============================
For most users, the easiest way to use rayrai is the included TCP viewer binary.
It lives at ``rayrai/<OS>/bin/rayrai_raisim_tcp_viewer`` (Linux packages ship it;
other platforms may need to build it) and works with RaisimServer-based
simulations. Run the binary while your simulation is running, and it will connect
to the server to display the scene.

You can start it directly from the repo:

.. code-block:: bash

    ./rayrai/<OS>/bin/rayrai_raisim_tcp_viewer

Run it with ``--help`` to see available options. On Windows, use the ``.exe``
binary.
If you are running a RaisimServer-based example, start the example first so the
server is listening (by default on port 8080), then launch the viewer to connect.

Build and link
==============
rayrai installs a CMake package under ``rayrai/<OS>``. Add it to
``CMAKE_PREFIX_PATH`` and link the ``rayrai`` target.

.. code-block:: cmake

    find_package(rayrai CONFIG REQUIRED)

    add_executable(my_app main.cpp)
    target_link_libraries(my_app PRIVATE rayrai)

In practice, you will also set the Raisim prefix (for example, ``-DCMAKE_PREFIX_PATH``)
to include both ``raisim/<OS>`` and ``rayrai/<OS>``.

Minimal usage
=============
The typical workflow is:

1. Create a RaiSim world.
2. Construct a ``raisin::RayraiWindow`` for that world.
3. Update the renderer each frame and consume the output texture.

.. code-block:: cpp

    #include <memory>

    #include <raisim/World.hpp>
    #include <rayrai/RayraiWindow.hpp>

    int main() {
      auto world = std::make_shared<raisim::World>();
      raisin::RayraiWindow viewer(world, 1280, 720);

      auto sphere = viewer.addVisualSphere("goal", 0.2, 0.9f, 0.2f, 0.2f, 1.0f);
      sphere->setPosition(0.0, 0.0, 1.0);

      while (true) {
        world->integrate();
        viewer.update(1280, 720, false, 0, 0, false);
        unsigned int tex = viewer.getImageTexture();
        (void)tex; // use the texture in your UI or pipeline
      }
    }

If you are integrating with an existing OpenGL context, you can use
``RayraiWindow::createOffscreenGlContext`` and
``RayraiWindow::makeOffscreenContextCurrent`` to control the context setup.

Example: custom visuals + background color
==========================================
This example adds a custom box, renders an RGB texture each frame, and reads the
image texture handle for UI integration. ``setBackgroundColor`` expects RGBA in
0..255 space (matching the example sources).

.. code-block:: cpp

    #include <memory>

    #include <raisim/World.hpp>
    #include <rayrai/RayraiWindow.hpp>

    int main() {
      auto world = std::make_shared<raisim::World>();
      world->addGround();

      raisin::RayraiWindow viewer(world, 1280, 720);
      viewer.setBackgroundColor({40, 45, 55, 255});

      auto box = viewer.addVisualBox("marker", 0.4, 0.2, 0.1, 0.9f, 0.6f, 0.1f, 1.0f);
      box->setPosition(1.0, 0.0, 0.3);

      while (true) {
        world->integrate();
        viewer.update(1280, 720, false, 0, 0, false);
        unsigned int colorTex = viewer.getImageTexture();
        (void)colorTex; // feed into your UI or pipeline
      }
    }

Shadows
=======
rayrai uses a directional light for shadows. By default, the shadow center tracks a
point in front of the camera and the shadow box is fixed in size. You can customize
both via ``RayraiWindow``:

.. code-block:: cpp

    // shadow center is N meters ahead of the camera (default: 10.0)
    viewer.setShadowCenterOffset(12.0f);
    // shadow box (default: halfSize=12.5, near=0.1, far=55.0)
    viewer.setShadowOrtho(20.0f, 0.1f, 80.0f);

If you need a fully custom shadow view/projection, use the lower-level
``raisin::Light`` API directly.

Examples
========
Rayrai examples are documented in :doc:`Examples <Examples>`. Each example page
includes a short explanation and the full source code, along with build and
run notes.

Quick map to the shipped examples (``examples/src/rayrai``):

* ``rayrai_custom_visuals.cpp``: custom visuals (sphere/box/cylinder/capsule/mesh).
* ``rayrai_instancing_grid.cpp``: instanced visuals and per-instance updates.
* ``rayrai_pointcloud_animation.cpp``: point cloud updates and GPU uploads.
* ``rayrai_rgb_camera.cpp``: RGBCamera rendering and ImGui preview.
* ``rayrai_depth_camera.cpp``: DepthCamera rendering + linear depth preview.
* ``rayrai_lidar_pointcloud.cpp``: LiDAR scan to point cloud visualization.
* ``rayrai_complete_showcase.cpp``: end-to-end demo with sensors, UI, and overlays.
* ``rayrai_tcp_viewer.cpp``: TCP viewer for RaisimServer scenes.

ImGui integration (SDL2 + OpenGL)
=================================
rayrai is designed to be embedded in custom UI. The repository ships a minimal
SDL2/ImGui helper in ``rayrai/example_common.hpp``. The core idea is:

1) Create an OpenGL context (SDL2 here).
2) Render the RayraiWindow texture into an ImGui ``Image``.
3) Forward hover and cursor data to ``RayraiWindow::update``.

Minimal pattern (trimmed from the examples):

.. code-block:: cpp

    ExampleApp app;
    if (!app.init("rayrai_example", 1280, 720))
      return -1;

    auto world = std::make_shared<raisim::World>();
    auto viewer = std::make_shared<raisin::RayraiWindow>(world, 1280, 720);

    while (!app.quit) {
      app.processEvents();
      world->integrate();

      app.beginFrame();
      app.renderViewer(*viewer); // ImGui::Image + viewer.update(...)
      app.endFrame();
    }

The ``renderViewer`` helper uses ``ImGui::IsItemHovered()`` and mouse positions
to drive camera interaction and picking. See ``rayrai_rgb_camera.cpp`` and
``rayrai_depth_camera.cpp`` for overlay windows showing camera outputs.

Headless/offscreen OpenGL context
=================================
If you already manage an OpenGL context (or want a headless one), use the static
helpers to create and bind a hidden SDL window context:

.. code-block:: cpp

    SDL_Window* window = nullptr;
    SDL_GLContext glContext = nullptr;
    raisin::RayraiWindow::createOffscreenGlContext(window, glContext, "rayrai_offscreen");
    raisin::RayraiWindow::makeOffscreenContextCurrent(window, glContext);

    auto world = std::make_shared<raisim::World>();
    raisin::RayraiWindow viewer(world, 640, 480);

Custom visuals and instancing
=============================
rayrai renders two categories of content:

* **RaiSim objects**: The renderer mirrors the objects already in the world.
* **Custom visuals**: Extra visuals you add explicitly (spheres, boxes, meshes, etc.).

Custom visuals are created through ``RayraiWindow`` and returned as ``Visuals``:

.. code-block:: cpp

    auto box = viewer.addVisualBox("marker", 0.1, 0.1, 0.1, 0.2f, 0.6f, 1.0f, 1.0f);
    box->setPosition(1.0, 0.0, 0.5);

For repeated geometry, use ``InstancedVisuals`` to reduce draw overhead:

.. code-block:: cpp

    auto instanced = viewer.addInstancedVisuals(
      "boxes", raisim::Shape::Box, glm::vec3(0.1f, 0.1f, 0.1f),
      glm::vec4(1.f, 0.2f, 0.2f, 1.f), glm::vec4(0.2f, 0.2f, 1.f, 1.f));
    instanced->addInstance(glm::vec3(0.0f, 0.0f, 0.1f), 0.0f);
    instanced->addInstance(glm::vec3(0.2f, 0.0f, 0.1f), 1.0f);

If you want to load meshes once and share them across visuals, use
``raisin::RayraiGlobalAsset`` and ``addVisualCustomMesh``:

.. code-block:: cpp

    auto assets = std::make_shared<raisin::RayraiGlobalAsset>();
    auto meshes = assets->getMeshes("/path/to/model.obj");
    auto custom = viewer.addVisualCustomMesh("custom", meshes, glm::vec4(0.9f, 0.9f, 1.0f, 1.0f));
    custom->setPosition(0.0, 1.0, 0.5);

Point clouds and coordinate frames
==================================
Point clouds and coordinate frames are lightweight debug aids:

.. code-block:: cpp

    auto cloud = viewer.addPointCloud("scan");
    auto frame = viewer.addCoordinateFrame("robot_frame");

These objects are rendered alongside the world and can be updated every frame.
Call ``updatePointBuffer()`` after changing point cloud data.

.. code-block:: cpp

    cloud->positions = {glm::vec3(0, 0, 1), glm::vec3(0.2f, 0.1f, 1.1f)};
    cloud->colors = {glm::vec4(0, 1, 0, 1), glm::vec4(1, 0, 0, 1)};
    cloud->updatePointBuffer();

Camera control and picking
==========================
``RayraiWindow`` manages an internal camera for offscreen rendering.
You can access or override it through ``getCamera()`` or by using
``renderWithExternalCamera`` when you want explicit control.

Picking is available through ``pickWithExternalCamera``. It renders a
selection pass and returns the encoded object id for a pixel.

The internal ``update`` call drives camera input and picking:

.. code-block:: cpp

    // Cursor is in framebuffer coordinates of the render area
    viewer.update(width, height, isHovered, cursorX, cursorY, shouldClick);

Sensor alignment
================
rayrai can align rendering to RaiSim camera sensors. Use
``syncRaisimCameraPose`` and ``renderWithExternalCamera`` to ensure the
render camera matches the sensor pose and intrinsics.
Note: ``syncRaisimCameraPose`` updates ``Camera::position/front/up`` directly;
avoid calling ``Camera::update()`` immediately afterward unless you also update
``yaw``/``pitch``.

RGB/Depth camera workflow (manual source + external camera):

.. code-block:: cpp

    auto rgbCam = anymal->getSensorSet("d455_front")->getSensor<raisim::RGBCamera>("color");
    auto depthCam = anymal->getSensorSet("d455_front")->getSensor<raisim::DepthCamera>("depth");

    rgbCam->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);
    depthCam->setMeasurementSource(raisim::Sensor::MeasurementSource::MANUAL);

    raisin::Camera rgbCamera(*rgbCam);
    raisin::Camera depthCamera(*depthCam);

    viewer.renderWithExternalCamera(*rgbCam, rgbCamera, {});
    viewer.renderWithExternalCamera(*depthCam, depthCamera, {});
    viewer.renderDepthPlaneDistance(*depthCam, depthCamera);

You can read back the camera buffers on CPU:

.. code-block:: cpp

    const auto& prop = rgbCam->getProperties();
    const int width = std::max(1, prop.width);
    const int height = std::max(1, prop.height);
    std::vector<char> rgba(size_t(width) * size_t(height) * 4);
    rgbCamera.getRawImage(*rgbCam, raisin::Camera::SensorStorageMode::CUSTOM_BUFFER,
      rgba.data(), rgba.size(), /*flipVertical=*/false);

Depth uses a ``float`` buffer with ``width * height`` entries.

Depth and LiDAR
===============
The renderer supports a linear depth plane and a GPU-assisted LiDAR pass:

* ``renderDepthPlaneDistance`` renders a linear depth texture.
* ``measureSpinningLidarSingleDrawGPU`` renders a LiDAR slice using a
  spherical chunk shader.

You can retrieve the depth texture via ``getDepthPlaneTexture()``.

LiDAR usage has two paths:

1) CPU-based scan via RaiSim (``SpinningLidar::update``), then visualize with a point cloud.
2) GPU slice rendering via ``measureSpinningLidarSingleDrawGPU`` for fast incremental updates.

GPU slice example:

.. code-block:: cpp

    lidar->updatePose();
    const glm::dvec3 posW = raisin::toGlm(lidar->getPosition());
    const glm::dmat3 rotW = raisin::toGlm(lidar->getOrientation());
    viewer.measureSpinningLidarSingleDrawGPU(*lidar, posW, rotW);

Additional tips
===============

* If you add/remove RaiSim objects after constructing the viewer, call
  ``RayraiWindow::updateObjectLists()`` so the renderer refreshes its cache.
* Use ``setShowCollisionBodies(true)`` for debug visualization of collision shapes.
* If you want rayrai overlays to use your ImGui font, pass it via ``setExternalFont``.

API
====

Core types
**********

.. doxygenclass:: raisin::RayraiWindow
   :members:

.. doxygenclass:: raisin::Camera
   :members:

.. doxygenclass:: raisin::RayraiGlobalAsset
   :members:

Visuals and geometry
********************

.. doxygenclass:: raisin::Visuals
   :members:

.. doxygenclass:: raisin::InstancedVisuals
   :members:

.. doxygenclass:: OpenGLMesh
   :members:

.. doxygennamespace:: raisin::assimp
   :members:

Scene helpers
*************

.. doxygenclass:: raisin::PointCloud
   :members:

.. doxygenclass:: raisin::CoordinateFrame
   :members:

.. doxygenclass:: raisin::CameraFrustum
   :members:

.. doxygenenum:: raisin::VisualCategory

Rendering and materials
************************

.. doxygenclass:: raisin::Light
   :members:

.. doxygenclass:: raisin::Material
   :members:

RaiSim integration
******************

.. doxygenclass:: raisin::RaisimObject
   :members:

.. doxygenstruct:: raisin::tcp_viewer::BufferReader
   :members:
