cmake_minimum_required(VERSION 3.18)
if(NOT DEFINED RAISIM_VERSION)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../CMakeLists.txt" _raisim_root_cmake)
    if(_raisim_root_cmake MATCHES "project\\(raisim VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(RAISIM_VERSION "${CMAKE_MATCH_1}")
    elseif(_raisim_root_cmake MATCHES "set\\(RAISIM_VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(RAISIM_VERSION "${CMAKE_MATCH_1}")
    else()
        message(FATAL_ERROR "RAISIM_VERSION is not defined and could not be parsed from ../CMakeLists.txt")
    endif()
endif()
project (raisim_wrapper VERSION ${RAISIM_VERSION})
set(CMAKE_CXX_STANDARD 20)

# CMake >= (policy introduction) warns unless the behavior is selected.
# Keep compatibility with older CMake by guarding the policy.
if(POLICY CMP0177)
    cmake_policy(SET CMP0177 NEW)
endif()

# find the various packages
find_package(Eigen3 REQUIRED)
find_package(raisim CONFIG REQUIRED)
if (CMAKE_VERSION VERSION_LESS 3.18)
    set(RAISIM_PY_PYTHON_DEV_COMPONENT Development)
else()
    set(RAISIM_PY_PYTHON_DEV_COMPONENT Development.Module)
endif()
find_package(Python REQUIRED COMPONENTS Interpreter ${RAISIM_PY_PYTHON_DEV_COMPONENT})
if(NOT COMMAND nanobind_add_module)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../thirdParty/nanobind"
                     "${CMAKE_CURRENT_BINARY_DIR}/nanobind")
endif()

# Newer packages expose force/tactile sensors; the 2.12 package does not.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES raisim::raisim)
unset(RAISIMPY_HAS_WRENCH_SENSORS CACHE)
check_cxx_source_compiles("
    #include <raisim/sensors/ForceSensor.hpp>
    #include <raisim/sensors/TactileSensor.hpp>
    int main() {
        auto force = raisim::Sensor::Type::FORCE;
        auto tactile = raisim::Sensor::Type::TACTILE;
        return int(force) == int(tactile);
    }" RAISIMPY_HAS_WRENCH_SENSORS)
unset(RAISIMPY_HAS_SEGMENTATION CACHE)
check_cxx_source_compiles("
    #include <raisim/World.hpp>
    #include <raisim/sensors/SegmentationSensor.hpp>
    int main() {
        raisim::World* world = nullptr;
        auto label = &raisim::World::addSegmentationClass;
        return world != nullptr && label != nullptr &&
               raisim::Sensor::Type::SEGMENTATION != raisim::Sensor::Type::SEGMENTATION;
    }" RAISIMPY_HAS_SEGMENTATION)
unset(CMAKE_REQUIRED_LIBRARIES)

# source files
file(GLOB SOURCES CONFIGURE_DEPENDS "src/*.cpp")

nanobind_add_module(raisimpy ${SOURCES})
target_link_libraries(raisimpy PRIVATE raisim::raisim)
if(UNIX)
    if(APPLE)
        set(_raisimpy_origin "@loader_path")
    else()
        set(_raisimpy_origin "$ORIGIN")
    endif()
    set_target_properties(raisimpy PROPERTIES
        INSTALL_RPATH "${_raisimpy_origin};$<TARGET_FILE_DIR:raisim::raisim>")
endif()
target_compile_definitions(raisimpy PRIVATE
    RAISIMPY_VERSION="${raisim_VERSION}"
    RAISIMPY_HAS_WRENCH_SENSORS=$<BOOL:${RAISIMPY_HAS_WRENCH_SENSORS}>
    RAISIMPY_HAS_SEGMENTATION=$<BOOL:${RAISIMPY_HAS_SEGMENTATION}>)

option(RAISIM_PY_RAYRAI "Build the in-process Rayrai Python API" ON)
if(RAISIM_PY_RAYRAI)
    option(RAISIMPY_RAYRAI_TEST_FAULTS "Compile Rayrai frame fault injection for regression tests" OFF)
    find_package(rayrai CONFIG REQUIRED HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../rayrai/lib/cmake/rayrai")
    file(GLOB RAYRAI_PY_SOURCES CONFIGURE_DEPENDS "rayrai/*.cpp" "rayrai/generated/*.cpp")
    nanobind_add_module(raisimpy_rayrai ${RAYRAI_PY_SOURCES})
    if(RAISIMPY_RAYRAI_TEST_FAULTS)
        target_compile_definitions(raisimpy_rayrai PRIVATE RAISIMPY_RAYRAI_TEST_FAULTS=1)
    endif()
    target_include_directories(raisimpy_rayrai PRIVATE include rayrai ${EIGEN3_INCLUDE_DIRS})
    target_link_libraries(raisimpy_rayrai PRIVATE rayrai raisim::raisimPng)
    if(UNIX AND NOT APPLE)
        # Keep helper libraries as direct dependencies: the renderer SDK's
        # RUNPATH is relative to its own prefix, while these live with RaiSim.
        target_link_options(raisimpy_rayrai PRIVATE "LINKER:--no-as-needed")
        target_link_libraries(raisimpy_rayrai PRIVATE
            raisim::raisimPng raisim::raisimMine raisim::raisimZ)
    endif()
    if(UNIX)
        set_target_properties(raisimpy_rayrai PROPERTIES
            INSTALL_RPATH "${_raisimpy_origin};$<TARGET_FILE_DIR:raisim::raisim>;$<TARGET_FILE_DIR:rayrai>")
    endif()
    target_precompile_headers(raisimpy_rayrai PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/rayrai/runtime.hpp")
    set(CMAKE_REQUIRED_LIBRARIES rayrai)
    unset(RAISIMPY_HAS_LAYER_CACHE_INVALIDATION CACHE)
    check_cxx_source_compiles("#include <rayrai/RayraiWindow.hpp>
        int main(int argc, char** argv) { reinterpret_cast<raisin::RayraiWindow*>(argv)->invalidateExternalFrameCache(argc > 0); }"
        RAISIMPY_HAS_LAYER_CACHE_INVALIDATION)
    unset(CMAKE_REQUIRED_LIBRARIES)
    target_compile_definitions(raisimpy_rayrai PRIVATE RAISIMPY_HAS_LAYER_CACHE_INVALIDATION=$<BOOL:${RAISIMPY_HAS_LAYER_CACHE_INVALIDATION}>)
    add_dependencies(raisimpy raisimpy_rayrai)
    target_compile_definitions(raisimpy PRIVATE RAISIMPY_HAS_RAYRAI=1)
else()
    target_compile_definitions(raisimpy PRIVATE RAISIMPY_HAS_RAYRAI=0)
endif()

# header files
target_include_directories(raisimpy PUBLIC include ${EIGEN3_INCLUDE_DIRS})
message("-- python executable: ${Python_EXECUTABLE}")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    include(CTest)
endif()
if(BUILD_TESTING)
    add_test(NAME raisimpy_runtime
        COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py"
            --module-dir "$<TARGET_FILE_DIR:raisimpy>"
            --library-dir "$<TARGET_FILE_DIR:raisim::raisim>")
    set_tests_properties(raisimpy_runtime PROPERTIES TIMEOUT 240)
endif()

# Install precompiled Python modules (raisimpy)
execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "if True:
            import sysconfig
            print(sysconfig.get_paths()[\"platlib\"])"
        OUTPUT_VARIABLE PYTHON_SITE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
message("-- python site is ${PYTHON_SITE}")

if(NOT RAISIM_PREFIX)
    get_filename_component(RAISIM_PREFIX "${raisim_DIR}/../../.." ABSOLUTE)
endif()
if(UNIX)
    set(INSTALL_SITE "${RAISIM_PREFIX}/lib")
elseif(WIN32)
    target_link_libraries(raisimpy PRIVATE Ws2_32)
    file(GLOB_RECURSE DLL_FILES "${RAISIM_PREFIX}/bin/*.dll")
    install(FILES ${DLL_FILES}
            DESTINATION ${PYTHON_SITE}
    )
    set(INSTALL_SITE "${RAISIM_PREFIX}/bin")
endif()

install(
        TARGETS
        raisimpy
        DESTINATION
        "${PYTHON_SITE}"
)

install(
        TARGETS
        raisimpy
        DESTINATION
        "${INSTALL_SITE}"
)

if(RAISIM_PY_RAYRAI)
    install(TARGETS raisimpy_rayrai DESTINATION "${PYTHON_SITE}")
    install(TARGETS raisimpy_rayrai DESTINATION "${INSTALL_SITE}")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/typings/raisimpy_rayrai"
        DESTINATION "${PYTHON_SITE}" FILES_MATCHING PATTERN "*.pyi")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/typings/raisimpy"
        DESTINATION "${PYTHON_SITE}" FILES_MATCHING PATTERN "*.pyi")
endif()
