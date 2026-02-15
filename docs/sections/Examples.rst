#############################
Examples
#############################

Overview
========
The ``examples/src`` directory contains runnable samples covering basic
simulation setup, RaisimServer visualization, maps, sensors, rayrai rendering,
and XML-based world configuration. Each example has its own page with a short
description and the full source.

Most ``server`` and ``maps`` examples require a visualizer connection
(RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) to see the scene.

Build and run
=============
Enable examples during configuration:

.. code-block:: bash

    cmake -S . -B build -DRAISIM_EXAMPLE=ON
    cmake --build build --target primitive_grid

Run examples from the build output directory so the ``rsc`` assets are found:

.. code-block:: bash

    cd build/examples
    ./primitive_grid

On Windows with multi-config generators (Visual Studio), build and run the
``Release`` binary:

.. code-block:: powershell

    cmake --build build --target primitive_grid --config Release
    .\build\examples\Release\primitive_grid.exe

If the runtime loader cannot find shared libraries, use the platform-specific
environment setup:

.. code-block:: bash

    source /path/to/raisim2Lib/raisim_env.sh

This script sets ``LD_LIBRARY_PATH`` on Linux and ``DYLD_LIBRARY_PATH`` on macOS.

.. code-block:: powershell

    .\raisim_env.ps1

.. code-block:: batch

    raisim_env.bat

XML
===

.. toctree::
   :maxdepth: 1

   examples/xml/xml_world_loader
   examples/xml/xml_templated_world

Maps
====

.. toctree::
   :maxdepth: 1

   examples/maps/map_mountain1_heightmap
   examples/maps/map_office1_scene
   examples/maps/map_lake1_heightmap
   examples/maps/map_hill1_heightmap
   examples/maps/map_atlas_charts
   examples/maps/map_anymal_graphs
   examples/maps/map_kinova_arm

Rayrai
======

.. toctree::
   :maxdepth: 1

   examples/rayrai/rayrai_complete_showcase
   examples/rayrai/rayrai_lidar_pointcloud
   examples/rayrai/rayrai_rgb_camera
   examples/rayrai/rayrai_depth_camera
   examples/rayrai/rayrai_custom_visuals
   examples/rayrai/rayrai_pointcloud_animation
   examples/rayrai/rayrai_instancing_grid
   examples/rayrai/rayrai_aruco_marker
   examples/rayrai/rayrai_basic_scene
   examples/rayrai/rayrai_tcp_viewer

RaisimServer
============

.. toctree::
   :maxdepth: 1

   examples/server/primitive_grid
   examples/server/mesh_stack
   examples/server/dynamic_object_addition
   examples/server/procedural_heightmap
   examples/server/material_restitution
   examples/server/material_static_friction
   examples/server/heightmap_from_png
   examples/server/dynamic_heightmap_unreal
   examples/server/spring_damper_joints
   examples/server/templated_tracked_robot
   examples/server/kinematic_platform
   examples/server/sensor_suite
   examples/server/synchronous_server_update
   examples/server/visual_objects_showcase
   examples/server/ray_casting
   examples/server/ray_scan_lidar
   examples/server/compound_object
   examples/server/wheeled_robot_force_control
   examples/server/length_constraints_newtons_cradle
   examples/server/inverse_dynamics
   examples/server/minitaur_pd
   examples/server/dzhanibekov_effect
   examples/server/object_lifecycle_stress
   examples/server/island_sleep_benchmark
