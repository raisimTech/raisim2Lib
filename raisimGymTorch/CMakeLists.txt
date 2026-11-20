cmake_minimum_required(VERSION 3.10)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
    execute_process(COMMAND sysctl -q hw.optional.arm64
            OUTPUT_VARIABLE _raisim_sysctl_stdout
            ERROR_VARIABLE _raisim_sysctl_stderr
            RESULT_VARIABLE _raisim_sysctl_result
            )
    execute_process(COMMAND uname -m
            OUTPUT_VARIABLE _raisim_uname_machine
            OUTPUT_STRIP_TRAILING_WHITESPACE
            )
    if((_raisim_sysctl_result EQUAL 0 AND _raisim_sysctl_stdout MATCHES "hw.optional.arm64: 1")
            OR _raisim_uname_machine MATCHES "^(aarch64|arm64)$")
        set(_raisim_apple_silicon TRUE)
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architecture for macOS packages." FORCE)
    endif()
endif()
project(_raisim_gym_torch)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)
option(RAISIMGYM_TORCH_WITH_LIBTORCH "Enable the generic TorchScript policy runner" ON)
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

set(_raisim_expected_root "${CMAKE_CURRENT_SOURCE_DIR}/../raisim")
set(_rayrai_expected_root "${CMAKE_CURRENT_SOURCE_DIR}/../rayrai")

if(NOT EXISTS "${_raisim_expected_root}" OR NOT EXISTS "${_rayrai_expected_root}")
    if(WIN32)
        set(_raisim_asset_platform "windows-x86")
    elseif(APPLE)
        if(_raisim_apple_silicon
                OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
            set(_raisim_asset_platform "macos-arm64")
        else()
            set(_raisim_asset_platform "macos-x86_64")
        endif()
    elseif(UNIX AND NOT APPLE AND NOT ANDROID AND NOT WEBGL)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
            set(_raisim_asset_platform "linux-arm64")
        else()
            set(_raisim_asset_platform "linux-x86")
        endif()
    else()
        message(FATAL_ERROR "Automatic raisim release download is not supported on this platform.")
    endif()

    set(_raisim_asset_file "${_raisim_asset_platform}-${RAISIM_VERSION}.zip")
    set(_raisim_download_dir "${CMAKE_CURRENT_BINARY_DIR}/downloads")
    set(_raisim_asset_path "${_raisim_download_dir}/${_raisim_asset_file}")
    file(MAKE_DIRECTORY "${_raisim_download_dir}")

    set(_raisim_download_success FALSE)
    set(_raisim_last_download_error "")
    set(_raisim_download_urls
        "https://github.com/raisimTech/raisim2Lib/releases/download/v${RAISIM_VERSION}/${_raisim_asset_file}"
        "https://github.com/raisimTech/raisim2Lib/releases/download/${RAISIM_VERSION}/${_raisim_asset_file}"
    )

    foreach(_raisim_url IN LISTS _raisim_download_urls)
        message(STATUS "Downloading raisim release asset: ${_raisim_url}")
        file(DOWNLOAD
            "${_raisim_url}"
            "${_raisim_asset_path}"
            STATUS _raisim_download_status
            SHOW_PROGRESS
            TLS_VERIFY ON
        )
        list(GET _raisim_download_status 0 _raisim_download_code)
        if(_raisim_download_code EQUAL 0)
            set(_raisim_download_success TRUE)
            break()
        else()
            list(GET _raisim_download_status 1 _raisim_download_message)
            set(_raisim_last_download_error "${_raisim_download_code}: ${_raisim_download_message}")
        endif()
    endforeach()

    if(NOT _raisim_download_success)
        message(FATAL_ERROR "Failed to download ${_raisim_asset_file} from raisim releases. Last error: ${_raisim_last_download_error}")
    endif()

    set(_raisim_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/extract-${RAISIM_VERSION}")
    file(REMOVE_RECURSE "${_raisim_extract_dir}")
    file(MAKE_DIRECTORY "${_raisim_extract_dir}")
    file(ARCHIVE_EXTRACT INPUT "${_raisim_asset_path}" DESTINATION "${_raisim_extract_dir}")

    set(_resolved_raisim_dir "")
    set(_resolved_rayrai_dir "")
    if(EXISTS "${_raisim_extract_dir}/raisim")
        set(_resolved_raisim_dir "${_raisim_extract_dir}/raisim")
    endif()
    if(EXISTS "${_raisim_extract_dir}/rayrai")
        set(_resolved_rayrai_dir "${_raisim_extract_dir}/rayrai")
    endif()

    if(_resolved_raisim_dir STREQUAL "")
        file(GLOB _raisim_extract_children LIST_DIRECTORIES TRUE "${_raisim_extract_dir}/*")
        foreach(_child IN LISTS _raisim_extract_children)
            if(EXISTS "${_child}/raisim")
                set(_resolved_raisim_dir "${_child}/raisim")
            endif()
            if(EXISTS "${_child}/rayrai")
                set(_resolved_rayrai_dir "${_child}/rayrai")
            endif()
        endforeach()
    endif()

    if(_resolved_raisim_dir STREQUAL "")
        message(FATAL_ERROR "Archive extraction completed but could not find a 'raisim' directory in ${_raisim_extract_dir}.")
    endif()

    if(NOT EXISTS "${_raisim_expected_root}")
        file(COPY "${_resolved_raisim_dir}" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/..")
    endif()
    if(NOT EXISTS "${_rayrai_expected_root}")
        if(_resolved_rayrai_dir STREQUAL "")
            message(FATAL_ERROR "Archive extraction completed but could not find a 'rayrai' directory in ${_raisim_extract_dir}.")
        endif()
        file(COPY "${_resolved_rayrai_dir}" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/..")
    endif()
endif()

######################
### help functions ###
######################
MACRO(SUBDIRLIST result curdir)
    FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
    SET(dirlist "")
    FOREACH(child ${children})
        IF(IS_DIRECTORY ${curdir}/${child})
            LIST(APPEND dirlist ${child})
        ENDIF()
    ENDFOREACH()
    SET(${result} ${dirlist})
ENDMACRO()

####################
### dependencies ###
####################
set(Dependencies)

add_subdirectory(thirdParty/nanobind nanobind)
if(NOT Python_INCLUDE_DIRS AND TARGET Python::Module)
    get_target_property(_python_module_include_dirs Python::Module INTERFACE_INCLUDE_DIRECTORIES)
    if(_python_module_include_dirs)
        set(Python_INCLUDE_DIRS "${_python_module_include_dirs}")
    endif()
endif()
if(NOT Python_INCLUDE_DIRS AND DEFINED _Python_INCLUDE_DIR)
    set(Python_INCLUDE_DIRS "${_Python_INCLUDE_DIR}")
endif()
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../thirdParty/Eigen3)
find_package(Eigen3 CONFIG REQUIRED)

if (APPLE)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        list(APPEND Dependencies OpenMP::OpenMP_CXX)
    else()
        add_compile_definitions(RAISIMGYM_NO_OPENMP)
    endif()
else ()
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

if (UNIX AND NOT APPLE AND NOT ANDROID AND NOT WEBGL AND NOT WIN32)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(RAISIM_OS linux-arm)
    else()
        set(RAISIM_OS linux)
    endif()
elseif(APPLE)
    # on macOS "uname -m" returns the architecture (x86_64 or arm64)
    execute_process(
            COMMAND uname -m
            RESULT_VARIABLE result
            OUTPUT_VARIABLE OSX_NATIVE_ARCHITECTURE
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(RAISIM_ARCH ${OSX_NATIVE_ARCHITECTURE} CACHE STRING "ARCH")

    if("${RAISIM_ARCH}" STREQUAL "arm64")
        set(RAISIM_OS m1)
    else()
        set(RAISIM_OS mac)
    endif()

elseif(WIN32)
    set(RAISIM_OS win32)
    list(APPEND CMAKE_PREFIX_PATH ../thirdParty)
    list(APPEND Dependencies Ws2_32)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/bin)
endif()

set(RAISIM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../raisim")
set(RAISIM_PREFIX "")
if(EXISTS "${RAISIM_ROOT}/${RAISIM_OS}/lib/cmake/raisim/raisimConfig.cmake")
    set(RAISIM_PREFIX "${RAISIM_ROOT}/${RAISIM_OS}")
elseif(EXISTS "${RAISIM_ROOT}/lib/cmake/raisim/raisimConfig.cmake")
    set(RAISIM_PREFIX "${RAISIM_ROOT}")
endif()
if(RAISIM_PREFIX STREQUAL "")
    message(FATAL_ERROR "Could not find raisim package config under ${RAISIM_ROOT}.")
endif()
list(APPEND CMAKE_PREFIX_PATH "${RAISIM_PREFIX}")

find_package(raisim CONFIG REQUIRED)
set(RAYRAI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../rayrai")
set(RAYRAI_PREFIX "")
if(EXISTS "${RAYRAI_ROOT}/${RAISIM_OS}/lib/cmake/rayrai/rayraiConfig.cmake")
    set(RAYRAI_PREFIX "${RAYRAI_ROOT}/${RAISIM_OS}")
elseif(EXISTS "${RAYRAI_ROOT}/lib/cmake/rayrai/rayraiConfig.cmake")
    set(RAYRAI_PREFIX "${RAYRAI_ROOT}")
endif()
if(RAYRAI_PREFIX STREQUAL "")
    message(FATAL_ERROR "Could not find rayrai package config under ${RAYRAI_ROOT}.")
endif()
list(APPEND CMAKE_PREFIX_PATH "${RAYRAI_PREFIX}")
find_package(rayrai CONFIG REQUIRED)
set(RAYRAI_INCLUDE_DIR "${RAYRAI_PREFIX}/include")
find_library(RAYRAI_LIB
    NAMES rayrai librayrai
    PATHS "${RAYRAI_PREFIX}/lib"
    NO_DEFAULT_PATH
)
if(NOT RAYRAI_LIB)
    message(FATAL_ERROR "rayrai library not found under ${RAYRAI_PREFIX}/lib")
endif()

add_library(rayrai_shared SHARED IMPORTED)
set_target_properties(rayrai_shared PROPERTIES IMPORTED_LOCATION "${RAYRAI_LIB}")

if (RAISIMGYM_TORCH_WITH_LIBTORCH)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent); print(int(torch.compiled_with_cxx11_abi()))"
        RESULT_VARIABLE _torch_query_result
        OUTPUT_VARIABLE _torch_query_output
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT _torch_query_result EQUAL 0)
        message(FATAL_ERROR "PyTorch must be installed in the active Python environment")
    endif()
    string(REPLACE "\n" ";" _torch_query_values "${_torch_query_output}")
    list(GET _torch_query_values 0 TORCH_PYTHON_ROOT)
    list(GET _torch_query_values 1 TORCH_CXX11_ABI)
    set(TORCH_INCLUDE_DIRS
        "${TORCH_PYTHON_ROOT}/include"
        "${TORCH_PYTHON_ROOT}/include/torch/csrc/api/include")
    set(TORCH_LIBRARY_DIR "${TORCH_PYTHON_ROOT}/lib")

    # find_library() caches successful results. setup.py deliberately reuses
    # its CMake build directory, so a different virtual environment (or a
    # removed standalone LibTorch installation) could otherwise leave these
    # variables pointing at libraries which no longer exist. Always resolve
    # them from the torch package belonging to Python_EXECUTABLE.
    unset(TORCH_LIBRARY CACHE)
    unset(TORCH_CPU_LIBRARY CACHE)
    unset(C10_LIBRARY CACHE)
    unset(TORCH_LIBRARY)
    unset(TORCH_CPU_LIBRARY)
    unset(C10_LIBRARY)
    find_library(TORCH_LIBRARY NAMES torch
        PATHS "${TORCH_LIBRARY_DIR}" NO_DEFAULT_PATH)
    find_library(TORCH_CPU_LIBRARY NAMES torch_cpu
        PATHS "${TORCH_LIBRARY_DIR}" NO_DEFAULT_PATH)
    find_library(C10_LIBRARY NAMES c10
        PATHS "${TORCH_LIBRARY_DIR}" NO_DEFAULT_PATH)
    if(NOT TORCH_LIBRARY OR NOT TORCH_CPU_LIBRARY OR NOT C10_LIBRARY)
        message(FATAL_ERROR
            "PyTorch C++ libraries were not found under ${TORCH_LIBRARY_DIR}")
    endif()
    set(TORCH_LIBRARIES
        "${TORCH_LIBRARY}" "${TORCH_CPU_LIBRARY}" "${C10_LIBRARY}")
endif()

#######################
### src and linking ###
#######################
SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/envs)
set(RAISIMGYM_ENV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/envs)

set(RAISIMGYM_TORCH_SOURCES)
if(RAISIMGYM_TORCH_WITH_LIBTORCH)
    set(_raisimgym_torch_bridge
        "${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env/TorchPolicyBridge.cpp")
    list(APPEND RAISIMGYM_TORCH_SOURCES "${_raisimgym_torch_bridge}")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set_source_files_properties(${_raisimgym_torch_bridge} PROPERTIES
            COMPILE_DEFINITIONS "_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}")
    endif()
endif()

######## Env ##########
FOREACH(subdir ${SUBDIRS})
    nanobind_add_module(${subdir} raisimGymTorch/env/raisim_gym.cpp raisimGymTorch/env/Yaml.cpp raisimGymTorch/env/VideoRecorder.cpp ${RAISIMGYM_TORCH_SOURCES})
    target_link_libraries(${subdir} PRIVATE raisim::raisim rayrai ${Dependencies})
    # Evaluation videos are rendered in-process with rayrai (VideoRecorder.cpp).
    target_compile_definitions(${subdir} PRIVATE RAISIMGYM_WITH_RAYRAI)
    if(UNIX AND NOT APPLE)
        # rayrai's RUNPATH points at its own prefix, but these helper libraries live with
        # RaiSim. Keep them direct dependencies so this module's RUNPATH finds them (as in raisimPy).
        target_link_options(${subdir} PRIVATE "LINKER:--no-as-needed")
        target_link_libraries(${subdir} PRIVATE raisim::raisimPng raisim::raisimMine raisim::raisimZ)
    endif()
    target_include_directories(${subdir} PUBLIC ${EIGEN3_INCLUDE_DIRS} ${RAISIMGYM_ENV_DIR}/${subdir} ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env ${RAYRAI_INCLUDE_DIR} ${RAYRAI_INCLUDE_DIR}/SDL2)
    target_compile_options(${subdir} PRIVATE -fPIC -O3 -flto)
    if (NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        target_compile_options(${subdir} PRIVATE -mno-avx2)
    endif()
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${subdir} PRIVATE -Wno-deprecated-enum-enum-conversion)
    endif()
    if (RAISIMGYM_TORCH_WITH_LIBTORCH)
        target_compile_definitions(${subdir} PRIVATE RAISIMGYM_TORCH_WITH_LIBTORCH)
        target_include_directories(${subdir} SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
        set_target_properties(${subdir} PROPERTIES
            BUILD_RPATH "${TORCH_LIBRARY_DIR}"
            INSTALL_RPATH "${TORCH_LIBRARY_DIR}")
        target_link_libraries(${subdir} PRIVATE ${TORCH_LIBRARIES})
    endif()
    target_compile_definitions(${subdir} PRIVATE "-DRAISIMGYM_TORCH_ENV_NAME=${subdir}")
    target_compile_definitions(${subdir} PRIVATE "$<$<CONFIG:Release>:EIGEN_NO_DEBUG>")
    target_compile_definitions(${subdir} PRIVATE EIGEN_DONT_PARALLELIZE)
    target_compile_definitions(${subdir} PRIVATE EIGEN_HAS_STD_RESULT_OF=0)

    message("raisim os ${RAISIM_OS}")
    message("raisim arch ${RAISIM_ARCH}")

    if("${RAISIM_OS}" STREQUAL "m1")
        if("${RAISIM_ARCH}" STREQUAL "arm64")
            message("building with correct lib")
            target_compile_options(${subdir} PRIVATE -mcpu=apple-m1)
        endif()
    endif()

    if (CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("[RAISIM_GYM] BUILDING THE DEBUG APP for ${subdir}")
        add_executable(${subdir}_debug_app raisimGymTorch/env/debug_app.cpp raisimGymTorch/env/Yaml.cpp)
        target_link_libraries(${subdir}_debug_app PRIVATE raisim::raisim)
        target_include_directories(${subdir}_debug_app PUBLIC raisimGymTorch/env/envs/${subdir} ${EIGEN3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/raisimGymTorch/env)
        if(WIN32)
            target_link_libraries(${subdir}_debug_app PRIVATE Ws2_32)
        else()
            target_compile_options(${subdir}_debug_app PRIVATE -fPIC -g -O0)
            target_compile_definitions(${subdir}_debug_app PRIVATE EIGEN_HAS_STD_RESULT_OF=0)
            if (NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
                target_compile_options(${subdir}_debug_app PRIVATE -mno-avx2)
            endif()
            if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(${subdir}_debug_app PRIVATE -Wno-deprecated-enum-enum-conversion)
            endif()
        endif()
    endif()
ENDFOREACH()
