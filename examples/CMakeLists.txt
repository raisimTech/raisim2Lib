cmake_minimum_required(VERSION 3.10)
project(raisim_examples LANGUAGES CXX)

#==
# Dependencies
#==

find_package(raisim CONFIG REQUIRED)
find_package(Eigen3 REQUIRED HINTS ${Eigen3_HINT})
find_package(rayrai CONFIG REQUIRED)

if(RAYRAI_PREFIX)
    list(APPEND CMAKE_PREFIX_PATH "${RAYRAI_PREFIX}")
endif()

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
create_executable(primitive_grid src/server/primitive_grid.cpp)
create_executable(mesh_stack src/server/mesh_stack.cpp)
create_executable(dynamic_object_addition src/server/dynamic_object_addition.cpp)
create_executable(procedural_heightmap src/server/procedural_heightmap.cpp)
create_executable(material_restitution src/server/material_restitution.cpp)
create_executable(material_static_friction src/server/material_static_friction.cpp)
create_executable(heightmap_from_png src/server/heightmap_from_png.cpp)
create_executable(dynamic_heightmap src/server/dynamic_heightmap.cpp)
create_executable(wheeled_robot_force_control src/server/wheeled_robot_force_control.cpp)
create_executable(length_constraints_newtons_cradle src/server/length_constraints_newtons_cradle.cpp)
create_executable(visual_objects_showcase src/server/visual_objects_showcase.cpp)
create_executable(ray_casting src/server/ray_casting.cpp)
create_executable(ray_scan_lidar src/server/ray_scan_lidar.cpp)
create_executable(compound_object src/server/compound_object.cpp)
create_executable(spring_damper_joints src/server/spring_damper_joints.cpp)
create_executable(templated_tracked_robot src/server/templated_tracked_robot.cpp)
create_executable(kinematic_platform src/server/kinematic_platform.cpp)
create_executable(sensor_suite src/server/sensor_suite.cpp)
create_executable(synchronous_server_update src/server/synchronous_server_update.cpp)
create_executable(object_lifecycle_stress src/server/object_lifecycle_stress.cpp)
create_executable(inverse_dynamics src/server/inverse_dynamics.cpp)
create_executable(dzhanibekov_effect src/server/dzhanibekov_effect.cpp)
create_executable(minitaur_pd src/server/minitaur_pd.cpp)
create_executable(ycb_objects src/server/ycb_objects.cpp)
create_executable(island_sleep_benchmark src/server/island_sleep_benchmark.cpp)

# benchmark examples
create_executable(articulated_system_benchmark src/benchmark/articulated_system_benchmark.cpp)

# xml reader
create_executable(xml_world_loader src/xml/xml_world_loader.cpp)
create_executable(xml_templated_world src/xml/xml_templated_world.cpp)

# maps
create_executable(map_mountain1_heightmap src/maps/map_mountain1_heightmap.cpp)
create_executable(map_office1_scene src/maps/map_office1_scene.cpp)
create_executable(map_lake1_heightmap src/maps/map_lake1_heightmap.cpp)
create_executable(map_atlas_charts src/maps/map_atlas_charts.cpp)
create_executable(map_anymal_graphs src/maps/map_anymal_graphs.cpp)
create_executable(map_hill1_heightmap src/maps/map_hill1_heightmap.cpp)
create_executable(map_kinova_arm src/maps/map_kinova_arm.cpp)

# rayrai examples
create_rayrai_executable(rayrai_complete_showcase src/rayrai/rayrai_complete_showcase.cpp)
create_rayrai_executable(rayrai_lidar_pointcloud src/rayrai/rayrai_lidar_pointcloud.cpp)
create_rayrai_executable(rayrai_custom_visuals src/rayrai/rayrai_custom_visuals.cpp)
create_rayrai_executable(rayrai_pointcloud_animation src/rayrai/rayrai_pointcloud_animation.cpp)
create_rayrai_executable(rayrai_basic_scene src/rayrai/rayrai_basic_scene.cpp)
create_rayrai_executable(rayrai_depth_camera src/rayrai/rayrai_depth_camera.cpp)
create_rayrai_executable(rayrai_rgb_camera src/rayrai/rayrai_rgb_camera.cpp)
create_rayrai_executable(rayrai_instancing_grid src/rayrai/rayrai_instancing_grid.cpp)
create_rayrai_executable(rayrai_tcp_viewer src/rayrai/rayrai_tcp_viewer.cpp)
create_rayrai_executable(rayrai_aruco_marker src/rayrai/rayrai_aruco_marker.cpp)
