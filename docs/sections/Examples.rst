#############################
Examples
#############################

Overview
========
The ``raisim2Lib`` distribution ships prebuilt RaiSim and rayrai libraries
together with C++ example sources. Build the examples with CMake. The install
target does not copy the example executables, and the package does not include
a prebuilt TCP viewer: the ``rayrai_tcp_viewer`` target builds it from the
example sources.

The examples fall into three kinds:

* **Server examples**, such as ``primitive_grid``, build a ``raisim::World``
  and publish it through ``raisim::RaisimServer``. View them with
  ``rayrai_tcp_viewer``.
* **Rayrai examples**, such as ``rayrai_pbr_material_grid``, render in process
  with ``raisin::RayraiWindow``.
* **Non-visual tools and benchmarks**, such as ``model_asset_pipeline`` and
  ``articulated_system_benchmark``, print results or write files.

Together they cover the RaiSim physics APIs, mesh import and export, OpenUSD
loading, and rayrai rendering and asset inspection. RaisimUnity and RaisimUnreal
are no longer supported; see :doc:`LegacyIntegrations`.

.. image:: ../../rsc/docs/image/examples_overview.png
   :alt: Overview of RaiSim and rayrai examples
   :width: 100%

.. toctree::
   :hidden:

   examples/overview

Run
===
After the top-level build from :doc:`BuildAndTest`, run examples from the build
tree:

.. code-block:: bash

    ./build-examples/examples/primitive_grid
    ./build-examples/examples/rayrai_pbr_material_grid

On Windows, the executables are in ``build-examples\bin``:

.. code-block:: powershell

    .\build-examples\bin\primitive_grid.exe

If the runtime loader cannot find the shared libraries, set up the environment
before running examples. On Linux and macOS, ``raisim_env.sh`` sets
``LD_LIBRARY_PATH`` or ``DYLD_LIBRARY_PATH``:

.. code-block:: bash

    source /path/to/raisim2Lib/raisim_env.sh

On Windows, CMake copies the runtime DLLs next to the executables. If a DLL is
still missing, add the package directories to ``PATH`` from PowerShell or from
the Command Prompt:

.. code-block:: powershell

    .\raisim_env.ps1

.. code-block:: batch

    raisim_env.bat

Visualization modes
===================
There are two visualization paths. See :doc:`Visualization` for the
full workflow comparison.

RaisimServer examples
---------------------
Server examples such as ``primitive_grid`` create a RaiSim world and publish it
through ``raisim::RaisimServer``. They do not open a renderer window themselves.
Start the source-built viewer, then run the server example:

.. code-block:: bash

    # Terminal 1
    ./build-examples/examples/rayrai_tcp_viewer

    # Terminal 2
    ./build-examples/examples/primitive_grid

The default server port is ``8080`` unless the example changes it. Use this
path when you want to inspect the same simulation data that a normal
RaisimServer application publishes.

Rayrai examples
---------------
Examples such as ``rayrai_pbr_material_grid``, ``rayrai_pbr_texture_maps``, and
``rayrai_visual_asset_support`` create or use a ``raisin::RayraiWindow``
directly and render in process. They do not need the TCP viewer:

.. code-block:: bash

    ./build-examples/examples/rayrai_pbr_material_grid

Prefer these examples when you need camera images, GPU/offscreen rendering, PBR
materials, glTF visual import, or standalone rayrai feature inspection. USD
visual meshes can also be loaded through ``RayraiWindow::addVisualMesh``; see
:doc:`OpenUSD` for the importer scope.

Non-visual examples
-------------------
Some examples print output or create files instead of showing a window.
``model_asset_pipeline`` writes preprocessed and exported OBJ files to
``raisim_model_asset_pipeline_example`` in the system temporary directory
(``/tmp`` on Linux). ``articulated_system_benchmark`` and
``anymal_standing_benchmark`` print timing results.

Example layout
==============
The examples project groups targets by executable behavior:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Group
     - Purpose
   * - Server examples
     - RaiSim physics, contact, terrain, sensor, robot, XML/MJCF, and OpenUSD
       scene workflows published through ``RaisimServer``.
   * - ``rayrai_*``
     - In-process rayrai renderer examples and the ``rayrai_tcp_viewer`` tool.
   * - Benchmark examples
     - Timing runs for articulated-system, contact, and sleeping-island
       workloads.

Choosing an example
===================
Start with these targets when learning a specific feature:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Target
     - Demonstrates
   * - ``primitive_grid``
     - Basic rigid primitive creation and RaisimServer publishing.
   * - ``model_asset_pipeline``
     - Mesh preprocessing, content-hash cache reuse, ``addMesh`` with processed
       assets, and OBJ export from a world.
   * - ``shadow_hand_usd_cube``
     - Loading an OpenUSD ShadowHand scene through ``World(shadow_hand.usd)``,
       adding a native RaiSim cube, and publishing both through
       ``RaisimServer``.
   * - ``nvidia_usd_robots``
     - Loading vetted NVIDIA Isaac Sim robot USD files with
       ``World::addUsdArticulatedSystem`` and publishing them through
       ``RaisimServer``.
   * - ``rayrai_pbr_material_grid``
     - Inspecting metallic-roughness PBR material behavior under rayrai.
   * - ``rayrai_visual_asset_support``
     - Rendering textured URDF visual meshes (OBJ/DAE with image textures)
       separately from their collision geometry.
   * - ``rayrai_coacd_mesh_approximation``
     - Visually comparing original meshes and CoACD convex approximation
       parts generated through ``World::addMesh``. The first run can take a
       few minutes while CoACD generates and caches the parts.
   * - ``rayrai_tcp_viewer``
     - The TCP visualizer used by RaisimServer examples.
   * - ``tendon_elastic``, ``tendon_pulleys``, ``tendon_coupling``
     - Elastic suspensions, routed guides/pulley branches, and joint
       transmissions. See :doc:`tendons/Examples`.
   * - ``rayrai_tendons``
     - Automatic tendon rendering with scene selection, pause/step/reset,
       and live transmission values. See :doc:`Tendons`.
   * - ``rayrai_motor_operating_region``
     - Actuators with motor operating regions (EM-MOR), linked from the URDF
       as actuator files, with live torque-speed plots for the twelve motors
       of a randomly actuated quadruped rig. See
       :doc:`examples/rayrai/rayrai_motor_operating_region`.
   * - ``rayrai_forest``
     - Dense instanced vegetation on a heightmap with automatic mesh LOD,
       foliage wind and shadows, and asynchronous loading. See
       :doc:`examples/rayrai/rayrai_forest` for the required rayrai version.
   * - ``rayrai_forest_from_rscene``
     - The same forest built from its saved RaiSim Engine ``.rscene`` file
       with ``raisim::World(path)`` and ``raisin::applyRscene``. Requires
       RaiSim and rayrai v2.7.1 (not yet released); see
       :doc:`examples/rayrai/rayrai_forest_from_rscene` and :doc:`RsceneFile`.
   * - ``rayrai_city``
     - A photoreal city of modular buildings, streets and parked cars loaded
       from ``rsc/city/rayrai_city.rscene``, with an ANYmal C quadruped added
       in C++. See :doc:`examples/rayrai/rayrai_city`.

Targets without a dedicated page
--------------------------------
These targets are built with the others but are documented only here or on a
feature page:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Target
     - Demonstrates
   * - ``heightmap_primitive_drop``
     - 432 boxes, spheres, capsules, and cylinders dropped onto a fractal
       heightmap.
   * - ``heightmap_mesh_drop``
     - Twenty monkey meshes with triangle-mesh collision dropped one at a time
       onto a colored procedural heightmap.
   * - ``blocky_heightmap_drop``
     - 900 primitives and convex-hull meshes dropped onto a heightmap made of
       flat 0.4 m tiles.
   * - ``rotated_blocky_heightmap_drop``
     - The ``blocky_heightmap_drop`` scene with the heightmap tilted by
       15 degrees about the x axis (see :doc:`HeightMap`): the same bodies land
       on the slope, and the spheres and cylinders roll down it.
   * - ``large_scale_ray_test``
     - Repeated ray queries against walls and pillars; see :doc:`RayTest`.
   * - ``rayrai_blue_wall_scene``
     - An authored glTF interior with its light sidecar and HDR environment,
       described by ``rsc/rayrai/blue_wall/blue_wall.rscene``. The example
       reads that file with its own parser; ``raisim::World(path)`` and
       ``raisin::applyRscene`` load it as well (see :doc:`RsceneFile`).
       ``--screenshot PNG`` renders in a hidden window, saves the image, and
       exits.

Runtime assets
==============
Some targets depend on bundled assets or platform runtime packages:

* OpenUSD loading uses the package's bundled OpenUSD runtime and USD files; keep
  the ``openusd`` runtime directory and ``rsc`` assets with the installed
  package.
* Rayrai examples require SDL2/OpenGL and rayrai runtime libraries. See
  :ref:`rayrai-platform-support` for the OpenGL requirements and the reduced
  macOS feature set.
* Poly Haven and PBR asset examples require the corresponding assets under
  ``rsc``. CMake copies ``rsc`` next to the executables
  (``build-examples/examples/rsc``, or ``build-examples/bin/rsc`` on Windows).
* ``rayrai_forest`` reads its assets from ``rsc/forest`` in the
  configured checkout and writes regenerable ``rayrai_cache_*.lods`` files
  beside them. ``rayrai_forest_from_rscene`` reads
  ``rsc/forest/rayrai_forest.rscene`` from the ``rsc`` copy next to the
  executables, and the scene references the same assets.
* ``rayrai_city`` reads ``rsc/city/rayrai_city.rscene`` from the ``rsc`` copy
  next to the executables (``--scene`` picks another file). Its street trees
  are the forest's, so it also needs ``rsc/forest``.
* ``rayrai_coacd_mesh_approximation`` writes ``raisim_coacd_*`` cache files
  beside the YCB meshes in the build tree's ``rsc`` copy.

After building, list the example executables from the platform-specific output
directory. Linux and macOS builds place them under ``examples``:

.. code-block:: bash

    ls build-examples/examples/

Windows builds place them under ``bin``:

.. code-block:: powershell

    Get-ChildItem .\build-examples\bin\*.exe

Benchmark Examples
==================

.. toctree::
   :maxdepth: 1

   examples/benchmark/articulated_system_benchmark
   examples/benchmark/anymal_standing_benchmark
   examples/server/island_sleep_benchmark

Rayrai Tools And Examples
=========================

.. toctree::
   :maxdepth: 1

   examples/rayrai/rayrai_basic_scene
   examples/rayrai/rayrai_complete_showcase
   examples/rayrai/rayrai_blender_scene_import
   examples/rayrai/rayrai_rgb_camera
   examples/rayrai/rayrai_depth_camera
   examples/rayrai/rayrai_heightmap_replacement
   examples/rayrai/rayrai_lidar_pointcloud
   examples/rayrai/rayrai_aruco_marker
   examples/rayrai/rayrai_custom_visuals
   examples/rayrai/rayrai_instancing_grid
   examples/rayrai/rayrai_forest
   examples/rayrai/rayrai_pointcloud_animation
   examples/rayrai/rayrai_pbr_material_grid
   examples/rayrai/rayrai_pbr_texture_maps
   examples/rayrai/rayrai_nested_glass
   examples/rayrai/rayrai_quality_lighting
   examples/rayrai/rayrai_visual_asset_support
   examples/rayrai/rayrai_coacd_mesh_approximation
   examples/rayrai/rayrai_runtime_scene_editing
   examples/rayrai/rayrai_rolling_spinning_friction
   examples/rayrai/rayrai_motor_operating_region
   examples/rayrai/rayrai_swept_ccd
   examples/rayrai/rayrai_forest_from_rscene
   examples/rayrai/rayrai_city
   examples/rayrai/rayrai_tcp_viewer

Server Examples
===============

.. toctree::
   :maxdepth: 1

   examples/server/compound_object
   examples/server/deformable_objects
   examples/server/dynamic_heightmap
   examples/server/dynamic_object_addition
   examples/server/dzhanibekov_effect
   examples/server/granular_media
   examples/server/heightmap_from_png
   examples/server/inverse_dynamics
   examples/server/kinematic_platform
   examples/server/length_constraints_newtons_cradle
   examples/server/material_restitution
   examples/server/material_static_friction
   examples/server/mesh_stack
   examples/server/minitaur_pd
   examples/server/mjcf_gymnasium_hopper
   examples/server/mjcf_gymnasium_humanoid
   examples/server/mjcf_gymnasium_walker2d
   examples/server/model_asset_pipeline
   examples/server/nvidia_usd_robots
   examples/server/object_lifecycle_stress
   examples/server/primitive_grid
   examples/server/procedural_heightmap
   examples/server/ray_casting
   examples/server/ray_scan_lidar
   examples/server/robotiq_gripper_mimic
   examples/server/rscene_server
   examples/server/sensor_suite
   examples/server/shadow_hand_usd_cube
   examples/server/sim_control_demo
   examples/server/sphere_drop
   examples/server/spring_damper_joints
   examples/server/strandbeest_closed_loops
   examples/server/synchronous_server_update
   examples/server/templated_tracked_robot
   examples/server/visual_objects_showcase
   examples/server/wheeled_robot_force_control
   examples/server/ycb_objects
   examples/worlds/anymal_pair
   examples/worlds/atlas
   examples/worlds/kinova_arm
   examples/worlds/office1_scene

XML Examples
============

.. toctree::
   :maxdepth: 1

   examples/xml/xml_templated_world
   examples/xml/xml_world_loader
