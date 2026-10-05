#############################
Visualization
#############################

RaiSim has two visualization workflows. Choose one based on where rendering
should happen.

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Workflow
     - Use it when
     - Main executable/API
   * - ``RaisimServer`` + TCP viewer
     - Your simulation should publish world state and a separate viewer should
       display it.
     - ``raisim::RaisimServer`` and ``rayrai_tcp_viewer`` (built with the
       examples)
   * - In-process rayrai
     - Your application needs direct access to OpenGL textures, RGB/depth
       images, screenshots, PBR assets, or custom UI embedding.
     - ``raisin::RayraiWindow``

For older Unity or Unreal workflows, see :doc:`LegacyIntegrations`.

RaisimServer + TCP Viewer
=========================

This is the simplest way to inspect a simulation while keeping the renderer out
of the simulation process. Your application owns the world and starts a
``RaisimServer``, which listens on ``127.0.0.1:8080`` by default:

.. code-block:: cpp

    raisim::World world;
    raisim::RaisimServer server(&world);
    server.launchServer(8080);

    while (running) {
      server.integrateWorldThreadSafe();
    }

Start the viewer, which is built with the examples, in a terminal where
``raisim_env.sh`` is sourced:

.. code-block:: bash

    source ./raisim_env.sh
    ./build-examples/examples/rayrai_tcp_viewer

Then run a server-based example in a second terminal, also sourced:

.. code-block:: bash

    source ./raisim_env.sh
    ./build-examples/examples/primitive_grid

The viewer connects to ``127.0.0.1:8080`` by default. Use this workflow for
everyday debug visualization, object inspection, interactive camera control,
simulation control (pause, step, push and reposition bodies), and
viewer-rendered manual RGB/depth cameras. Camera requests and the returned BGRA
or metric-depth images share the TCP connection with the scene updates. See
:doc:`RayraiTcpViewer` for the viewer itself.

Use in-process rayrai when the application must own the OpenGL context, access
textures without a network round trip, include custom render-only objects, or
keep producing camera frames without a viewer process. Use RaiSim's CPU depth
computation when rayrai is unavailable, or when a deterministic headless ray
query is explicitly required.

In-Process rayrai
=================

Use in-process rayrai when rendering is part of the application:

.. code-block:: cpp

    auto world = std::make_shared<raisim::World>();
    raisin::RayraiWindow viewer(world, 1280, 720);

    while (running) {
      world->integrate();
      viewer.update(1280, 720, false, 0, 0, false);
      unsigned int colorTexture = viewer.getImageTexture();
      (void)colorTexture;
    }

Examples:

.. code-block:: bash

    source ./raisim_env.sh
    ./build-examples/examples/rayrai_basic_scene
    ./build-examples/examples/rayrai_pbr_material_grid
    ./build-examples/examples/rayrai_pbr_texture_maps

Use this workflow for screenshots, offscreen rendering, dataset generation,
custom ImGui/Qt tools, PBR visual inspection, glTF visual import, picking,
and direct RGB/depth readback.

Which Page Next?
================

* :doc:`RayraiTcpViewer` documents the TCP viewer: panels, controls,
  command-line options and the wire format for custom clients.
* :doc:`RaisimServer` documents the server lifecycle, thread-safety boundary,
  simulation control, synchronous request loop, and server-side visual helper
  API.
* :doc:`Rayrai` documents ``RayraiWindow``, render-quality controls, custom
  visuals, glTF/Blender import, offscreen contexts, sensors, and rayrai API
  reference.
* :doc:`Sensors` documents the recommended rayrai RGB/depth sensor workflow,
  manual sensor buffers, and CPU-only fallback paths.
