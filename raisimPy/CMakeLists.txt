cmake_minimum_required(VERSION 3.18)
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

# source files
file(GLOB SOURCES "src/*.cpp")

nanobind_add_module(raisimpy ${SOURCES})
target_link_libraries(raisimpy PRIVATE raisim::raisim)

# header files
target_include_directories(raisimpy PUBLIC include ${EIGEN3_INCLUDE_DIRS})
message("-- python executable: ${Python_EXECUTABLE}")

# Install precompiled Python modules (raisimpy)
execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "if True:
            import sysconfig
            print(sysconfig.get_paths()[\"purelib\"])"
        OUTPUT_VARIABLE PYTHON_SITE
        OUTPUT_STRIP_TRAILING_WHITESPACE)
message("-- python site is ${PYTHON_SITE}")

if(UNIX)
    set(INSTALL_SITE "${CMAKE_CURRENT_SOURCE_DIR}/../raisim/${RAISIM_OS}/lib")
elseif(WIN32)
    target_link_libraries(raisimpy PRIVATE Ws2_32)
    file(GLOB_RECURSE DLL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../raisim/win32/bin/*.dll")
    install(FILES ${DLL_FILES}
            DESTINATION ${PYTHON_SITE}
    )
    set(INSTALL_SITE "${CMAKE_CURRENT_SOURCE_DIR}/../raisim/${RAISIM_OS}/bin")
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
