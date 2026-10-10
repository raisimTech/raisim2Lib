cmake_minimum_required(VERSION 3.10)
project(raisim_examples LANGUAGES CXX)

#==
# Dependencies
#==

if(RAISIM_PREFIX)
    list(APPEND CMAKE_PREFIX_PATH "${RAISIM_PREFIX}")
endif()
if(RAYRAI_PREFIX)
    list(APPEND CMAKE_PREFIX_PATH "${RAYRAI_PREFIX}")
endif()

find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})
find_package(rayrai CONFIG REQUIRED)

if(TARGET glm::glm AND NOT TARGET glm)
    add_library(glm ALIAS glm::glm)
endif()

if(WIN32)
    # Keep example executables inside the build tree (not under the prebuilt package folders).
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../rsc" DESTINATION "${CMAKE_BINARY_DIR}/bin")

    function(copy_runtime_dlls_from _source_dir)
        if(EXISTS "${_source_dir}")
            file(GLOB _runtime_dlls "${_source_dir}/*.dll")
            if(_runtime_dlls)
                file(COPY ${_runtime_dlls} DESTINATION "${CMAKE_BINARY_DIR}/bin")
            endif()
            if(EXISTS "${_source_dir}/openusd")
                file(COPY "${_source_dir}/openusd" DESTINATION "${CMAKE_BINARY_DIR}/bin")

                # OpenUSD plugInfo paths resolve plugin DLLs relative to this subtree.
                file(GLOB _openusd_runtime_dlls "${_source_dir}/usd_*.dll")
                if(_openusd_runtime_dlls)
                    file(COPY ${_openusd_runtime_dlls} DESTINATION "${CMAKE_BINARY_DIR}/bin/openusd")
                endif()
            endif()
        endif()
    endfunction()

    # Make examples runnable without sourcing env scripts by copying runtime DLLs next to executables.
    if(DEFINED RAISIM_PREFIX)
        copy_runtime_dlls_from("${RAISIM_PREFIX}")
        copy_runtime_dlls_from("${RAISIM_PREFIX}/bin")
        copy_runtime_dlls_from("${RAISIM_PREFIX}/lib")
    endif()
    if(DEFINED RAYRAI_PREFIX AND NOT RAYRAI_PREFIX STREQUAL "")
        copy_runtime_dlls_from("${RAYRAI_PREFIX}")
        copy_runtime_dlls_from("${RAYRAI_PREFIX}/bin")
        copy_runtime_dlls_from("${RAYRAI_PREFIX}/lib")
    endif()

    set(_vcpkg_root "$ENV{VCPKG_ROOT}")
    if(_vcpkg_root STREQUAL "" AND EXISTS "C:/vcpkg")
        set(_vcpkg_root "C:/vcpkg")
    endif()
    if(NOT _vcpkg_root STREQUAL "")
        copy_runtime_dlls_from("${_vcpkg_root}/installed/x64-windows/bin")
        copy_runtime_dlls_from("${_vcpkg_root}/installed/x64-windows/debug/bin")
    endif()
else()
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/../rsc" DESTINATION ${CMAKE_BINARY_DIR}/examples)
endif()

include_directories ( ${EIGEN3_INCLUDE_DIRS} )

#==
# Build Targets
#==

function(create_executable app_name file_name)
    add_executable(${app_name} ${file_name})
    set_target_properties(${app_name} PROPERTIES MACOSX_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/../raisim/mac/lib")
    if(WIN32)
        target_link_libraries(${app_name} PUBLIC raisim::raisim Ws2_32 Winmm)
        target_compile_definitions(${app_name} PRIVATE EIGEN_HAS_STD_RESULT_OF=0)
        target_compile_options(${app_name} PRIVATE "/MP")
    else()
        target_link_libraries(${app_name} PUBLIC raisim::raisim pthread)
        target_compile_options(${app_name} PRIVATE "$<$<CONFIG:RELEASE>:-O3>")
        target_compile_options(${app_name} PRIVATE "$<$<CONFIG:DEBUG>:-O0>")
        if(EXAMPLES_RPATH)
            set_target_properties(${app_name} PROPERTIES
                BUILD_RPATH "${EXAMPLES_RPATH}"
            )
        endif()
    endif()

    if(APPLE)
        execute_process(COMMAND sysctl -q hw.optional.arm64
                OUTPUT_VARIABLE _sysctl_stdout
                ERROR_VARIABLE _sysctl_stderr
                RESULT_VARIABLE _sysctl_result
                )
        if(_sysctl_result EQUAL 0 AND _sysctl_stdout MATCHES "hw.optional.arm64: 1")
            target_compile_options(${app_name} PRIVATE -mcpu=apple-m1)
        endif()
    endif()

    target_include_directories(${app_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endfunction()

function(create_rayrai_executable app_name file_name)
    create_executable(${app_name} ${file_name})
    target_link_libraries(${app_name} PRIVATE rayrai)
endfunction()

# server examples
# Basics and object lifecycle.
create_executable(primitive_grid src/server/basics/primitive_grid.cpp)
create_executable(mesh_stack src/server/basics/mesh_stack.cpp)
create_executable(compound_object src/server/basics/compound_object.cpp)
create_executable(dynamic_object_addition src/server/basics/dynamic_object_addition.cpp)
create_executable(ycb_objects src/server/basics/ycb_objects.cpp)
create_executable(model_asset_pipeline src/server/assets/model_asset_pipeline.cpp)

# Materials and terrain.
create_executable(material_restitution src/server/materials/material_restitution.cpp)
create_executable(material_static_friction src/server/materials/material_static_friction.cpp)
create_executable(procedural_heightmap src/server/terrain/procedural_heightmap.cpp)
create_executable(heightmap_from_png src/server/terrain/heightmap_from_png.cpp)
create_executable(dynamic_heightmap src/server/terrain/dynamic_heightmap.cpp)
create_executable(heightmap_mesh_drop src/server/terrain/heightmap_mesh_drop.cpp)
create_executable(blocky_heightmap_drop src/server/terrain/blocky_heightmap_drop.cpp)

# Sensors.
create_executable(ray_casting src/server/sensors/ray_casting.cpp)
create_executable(large_scale_ray_test src/server/sensors/large_scale_ray_test.cpp)
create_executable(ray_scan_lidar src/server/sensors/ray_scan_lidar.cpp)
create_executable(sensor_suite src/server/sensor_suite.cpp)

# Robots and control.
create_executable(wheeled_robot_force_control src/server/robots/wheeled_robot_force_control.cpp)
create_executable(templated_tracked_robot src/server/robots/templated_tracked_robot.cpp)
create_executable(minitaur_pd src/server/robots/minitaur_pd.cpp)
create_executable(robotiq_gripper_mimic src/server/robots/robotiq_gripper_mimic.cpp)
create_executable(strandbeest_closed_loops src/server/robots/strandbeest_closed_loops.cpp)
create_executable(inverse_dynamics src/server/robots/inverse_dynamics.cpp)
create_executable(shadow_hand_usd_cube src/server/robots/shadow_hand_usd_cube.cpp)
create_executable(nvidia_usd_robots src/server/robots/nvidia_usd_robots.cpp)
set_target_properties(robotiq_gripper_mimic shadow_hand_usd_cube nvidia_usd_robots PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

# Dynamics and constraints.
create_executable(length_constraints_newtons_cradle src/server/dynamics/length_constraints_newtons_cradle.cpp)
create_executable(spring_damper_joints src/server/dynamics/spring_damper_joints.cpp)
create_executable(kinematic_platform src/server/dynamics/kinematic_platform.cpp)
create_executable(dzhanibekov_effect src/server/dynamics/dzhanibekov_effect.cpp)
create_executable(tendon_elastic src/server/dynamics/tendon_elastic.cpp)
create_executable(tendon_pulleys src/server/dynamics/tendon_pulleys.cpp)
create_executable(tendon_coupling src/server/dynamics/tendon_coupling.cpp)
set_target_properties(tendon_elastic tendon_pulleys tendon_coupling PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

# Visualization.
create_executable(visual_objects_showcase src/server/visualization/visual_objects_showcase.cpp)
create_executable(synchronous_server_update src/server/synchronous_server_update.cpp)
create_executable(sim_control_demo src/server/sim_control_demo.cpp)
# Serving a .rscene scene, with its files shared to remote viewers, is new in v2.7.1.
get_target_property(_raisim_include_dirs raisim::raisim INTERFACE_INCLUDE_DIRECTORIES)
set(_raisim_shares_scene_files OFF)
foreach(_raisim_include_dir IN LISTS _raisim_include_dirs)
    if(EXISTS "${_raisim_include_dir}/raisim/server/SceneFileServer.hpp")
        set(_raisim_shares_scene_files ON)
    endif()
endforeach()
if(_raisim_shares_scene_files)
    create_executable(rscene_server src/server/visualization/rscene_server.cpp)
else()
    message(STATUS "Skipping rscene_server: the installed raisim cannot share .rscene files "
                   "with the TCP viewer (raisim/server/SceneFileServer.hpp, new in v2.7.1)")
endif()

# Performance and deformables.
create_executable(island_sleep_benchmark src/server/performance/island_sleep_benchmark.cpp)
create_executable(deformable_objects src/server/deformable/deformable_objects.cpp)

# benchmark examples
create_executable(articulated_system_benchmark src/benchmark/articulated_system_benchmark.cpp)
create_executable(anymal_standing_benchmark src/benchmark/anymal_standing_benchmark.cpp)
create_executable(granular_media src/benchmark/granular_media.cpp)

# xml reader
create_executable(xml_world_loader src/xml/xml_world_loader.cpp)
create_executable(xml_templated_world src/xml/xml_templated_world.cpp)
create_executable(mjcf_gymnasium_hopper src/server/mjcf/mjcf_gymnasium_hopper.cpp)
create_executable(mjcf_gymnasium_walker2d src/server/mjcf/mjcf_gymnasium_walker2d.cpp)
create_executable(mjcf_gymnasium_humanoid src/server/mjcf/mjcf_gymnasium_humanoid.cpp)

# Packaged robot scenes.
create_executable(kinova_arm src/worlds/kinova_arm.cpp)

# rayrai examples
# Entry points.
create_rayrai_executable(rayrai_basic_scene src/rayrai/getting_started/rayrai_basic_scene.cpp)
create_rayrai_executable(rayrai_complete_showcase src/rayrai/getting_started/rayrai_complete_showcase.cpp)

# Sensors.
create_rayrai_executable(rayrai_rgb_camera src/rayrai/sensors/rayrai_rgb_camera.cpp)
create_rayrai_executable(rayrai_depth_camera src/rayrai/sensors/rayrai_depth_camera.cpp)
create_rayrai_executable(rayrai_lidar_pointcloud src/rayrai/sensors/rayrai_lidar_pointcloud.cpp)
create_rayrai_executable(rayrai_aruco_marker src/rayrai/sensors/rayrai_aruco_marker.cpp)

# Visual-only overlays and GPU visual primitives.
create_rayrai_executable(rayrai_custom_visuals src/rayrai/visuals/rayrai_custom_visuals.cpp)
create_rayrai_executable(rayrai_instancing_grid src/rayrai/visuals/rayrai_instancing_grid.cpp)
create_rayrai_executable(rayrai_pointcloud_animation src/rayrai/visuals/rayrai_pointcloud_animation.cpp)

# Dynamics.
create_rayrai_executable(rayrai_rolling_spinning_friction src/rayrai/dynamics/rayrai_rolling_spinning_friction.cpp)
create_rayrai_executable(rayrai_motor_operating_region src/rayrai/dynamics/rayrai_motor_operating_region.cpp)
create_rayrai_executable(rayrai_tendons src/rayrai/dynamics/rayrai_tendons.cpp)
set_target_properties(rayrai_tendons PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON)

# Materials and assets.
create_rayrai_executable(rayrai_pbr_material_grid src/rayrai/materials/rayrai_pbr_material_grid.cpp)
create_rayrai_executable(rayrai_pbr_texture_maps src/rayrai/materials/rayrai_pbr_texture_maps.cpp)
create_rayrai_executable(rayrai_quality_lighting src/rayrai/materials/rayrai_quality_lighting.cpp)
create_rayrai_executable(rayrai_nested_glass src/rayrai/materials/rayrai_nested_glass.cpp)
create_rayrai_executable(rayrai_coacd_mesh_approximation src/rayrai/assets/rayrai_coacd_mesh_approximation.cpp)
create_rayrai_executable(rayrai_visual_asset_support src/rayrai/assets/rayrai_visual_asset_support.cpp)
create_rayrai_executable(rayrai_blue_wall_scene src/rayrai/assets/rayrai_blue_wall_scene.cpp)

# Runtime scene editing and collision detection.
create_rayrai_executable(rayrai_runtime_scene_editing src/rayrai/runtime/rayrai_runtime_scene_editing.cpp)
create_rayrai_executable(rayrai_swept_ccd src/rayrai/collision/rayrai_swept_ccd.cpp)

# Tools.
create_rayrai_executable(rayrai_tcp_viewer src/rayrai/tools/rayrai_tcp_viewer.cpp)
target_sources(rayrai_tcp_viewer PRIVATE
    src/rayrai/tools/TcpViewerSceneFiles.cpp
    src/rayrai/tools/TcpViewerActuators.cpp
    src/rayrai/tools/TcpViewerVideo.cpp
    src/rayrai/tools/TcpViewerSignals.cpp
    src/rayrai/tools/TcpViewerPaneLayout.cpp
    src/rayrai/tools/TcpViewerSimulationWorker.cpp
    src/rayrai/tools/TcpViewerSimulation.cpp
    src/rayrai/tools/TcpViewerScreenshot.cpp
    src/rayrai/tools/TcpViewerSession.cpp
    src/rayrai/tools/TcpViewerDiscovery.cpp
    src/rayrai/tools/TcpViewerSensors.cpp
    src/rayrai/tools/TcpViewerSettings.cpp)

# Optional desktop integration: install a launcher for the viewer into
# ~/.local/share/applications and pin it to the GNOME / Ubuntu dock after each
# Release build. Off by default because a build should not normally rearrange
# the user's desktop; the flag is cached, so configuring once with
# -DRAISIM_EXAMPLE_DESKTOP_LAUNCHER=ON keeps it enabled for that build tree.
# The script skips itself on non-Linux hosts, on non-Release configurations and
# on machines with no graphical session, and never fails the build.
option(RAISIM_EXAMPLE_DESKTOP_LAUNCHER
    "Install and pin a desktop launcher for rayrai_tcp_viewer (Linux, Release builds)" OFF)

if(RAISIM_EXAMPLE_DESKTOP_LAUNCHER AND UNIX AND NOT APPLE)
    set(_viewer_launcher_script
        "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/install_rayrai_viewer_launcher.sh")
    if(EXISTS "${_viewer_launcher_script}")
        add_custom_command(TARGET rayrai_tcp_viewer POST_BUILD
            COMMAND "${_viewer_launcher_script}"
                    --viewer "$<TARGET_FILE:rayrai_tcp_viewer>"
                    --repo "${CMAKE_CURRENT_SOURCE_DIR}/.."
                    --config "$<CONFIG>"
                    --pin
            COMMENT "Installing rayrai_tcp_viewer desktop launcher"
            VERBATIM)
    else()
        message(WARNING
            "RAISIM_EXAMPLE_DESKTOP_LAUNCHER is ON but ${_viewer_launcher_script} is missing")
    endif()
endif()

if(WIN32)
  target_link_libraries(rayrai_tcp_viewer PRIVATE shell32 ws2_32)
endif()

# Terrain-grounded mixed forest. Assets stay in the checkout; --assets overrides this path.
create_rayrai_executable(rayrai_forest src/rayrai/worlds/rayrai_forest.cpp)
target_compile_features(rayrai_forest PRIVATE cxx_std_20)
target_compile_definitions(rayrai_forest PRIVATE
    FOREST_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../rsc/forest")
# The same forest built by raisim::World from its .rscene file. The .rscene reader is new in
# v2.7.1, so a package without its rayrai half (RsceneVisuals.hpp) skips this example.
get_target_property(_rayrai_include_dirs rayrai INTERFACE_INCLUDE_DIRECTORIES)
set(_rayrai_has_rscene OFF)
foreach(_rayrai_include_dir IN LISTS _rayrai_include_dirs)
    if(EXISTS "${_rayrai_include_dir}/rayrai/RsceneVisuals.hpp")
        set(_rayrai_has_rscene ON)
    endif()
endforeach()
if(_rayrai_has_rscene)
    create_rayrai_executable(rayrai_forest_from_rscene src/rayrai/worlds/rayrai_forest_from_rscene.cpp)
    target_compile_features(rayrai_forest_from_rscene PRIVATE cxx_std_20)
else()
    message(STATUS "Skipping rayrai_forest_from_rscene: the installed rayrai has no .rscene "
                   "support (rayrai/RsceneVisuals.hpp, new in v2.7.1)")
endif()
# Photoreal city loaded from rsc/city/rayrai_city.rscene, with an ANYmal C added in C++.
if(_rayrai_has_rscene)
    create_rayrai_executable(rayrai_city src/rayrai/worlds/rayrai_city.cpp)
    target_compile_features(rayrai_city PRIVATE cxx_std_20)
else()
    message(STATUS "Skipping rayrai_city: the installed rayrai has no .rscene support")
endif()
# Photoreal warehouse loaded from rsc/warehouse/rayrai_warehouse.rscene, with an ANYmal C added in C++.
if(_rayrai_has_rscene)
    create_rayrai_executable(rayrai_warehouse src/rayrai/worlds/rayrai_warehouse.cpp)
    target_compile_features(rayrai_warehouse PRIVATE cxx_std_20)
else()
    message(STATUS "Skipping rayrai_warehouse: the installed rayrai has no .rscene support")
endif()
option(RAISIM_FOREST_COLLISION_BENCHMARK "Build the forest collision benchmark" OFF)
if(RAISIM_FOREST_COLLISION_BENCHMARK)
    create_executable(forest_collision_benchmark tools/benchmark_forest_collision.cpp)
    target_compile_features(forest_collision_benchmark PRIVATE cxx_std_20)
    target_include_directories(forest_collision_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/rayrai/worlds)
    target_compile_definitions(forest_collision_benchmark PRIVATE
        FOREST_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../rsc/forest")
endif()

# Manual regression and debug programs are opt-in.
option(RAISIM_DEBUG_TOOLS "Build manual regression and debug tools" OFF)
if(RAISIM_DEBUG_TOOLS)
    create_executable(object_lifecycle_stress tools/debug/object_lifecycle_stress.cpp)
    create_rayrai_executable(rayrai_heightmap_replacement tools/debug/rayrai_heightmap_replacement.cpp)
endif()
