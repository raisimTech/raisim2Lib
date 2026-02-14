cmake_minimum_required(VERSION 3.18)
set(RAISIM_VERSION 2.0.0)
project(raisim VERSION ${RAISIM_VERSION} LANGUAGES CXX)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Include CMake helper modules
include(GNUInstallDirs)

# Build options
option(RAISIM_EXAMPLE "Build example applications." TRUE)
option(RAISIM_MATLAB "Build a matlab wrapper." FALSE)
option(RAISIM_PY "Build a python wrapper." FALSE)
option(RAISIM_DOC "Build a raisim documentation. You can find the latest doc in raisim.com." FALSE)

if(WIN32)
    set(_raisim_asset_platform "windows-x86")
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

file(REMOVE_RECURSE
    "${CMAKE_CURRENT_SOURCE_DIR}/raisim"
    "${CMAKE_CURRENT_SOURCE_DIR}/rayrai"
)

file(COPY "${_resolved_raisim_dir}" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
if(NOT _resolved_rayrai_dir STREQUAL "")
    file(COPY "${_resolved_rayrai_dir}" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Clean up extracted release wrapper directory from the source tree, if present.
file(REMOVE_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/${_raisim_asset_platform}-${RAISIM_VERSION}")

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/raisim")
    message(FATAL_ERROR "Archive extraction completed but 'raisim' directory was not found.")
endif()

if (UNIX AND NOT APPLE AND NOT ANDROID AND NOT WEBGL AND NOT WIN32)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(RAISIM_OS linux-arm)
    else ()
        set(RAISIM_OS linux)
    endif ()
elseif(APPLE)
    execute_process(COMMAND sysctl -q hw.optional.arm64
            OUTPUT_VARIABLE _sysctl_stdout
            ERROR_VARIABLE _sysctl_stderr
            RESULT_VARIABLE _sysctl_result
            )
    if(_sysctl_result EQUAL 0 AND _sysctl_stdout MATCHES "hw.optional.arm64: 1")
        set(RAISIM_OS m1)
    else()
        set(RAISIM_OS mac)
    endif()
elseif(WIN32)
    set(RAISIM_OS win32)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/Eigen3)
endif()

set(RAISIM_PREFIX "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/raisim/${RAISIM_OS}/lib/cmake/raisim/raisimConfig.cmake")
    set(RAISIM_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/raisim/${RAISIM_OS}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/raisim/lib/cmake/raisim/raisimConfig.cmake")
    set(RAISIM_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/raisim")
endif()

if(RAISIM_PREFIX STREQUAL "")
    message(FATAL_ERROR "Could not find raisim package config after extraction.")
endif()
list(APPEND CMAKE_PREFIX_PATH "${RAISIM_PREFIX}")
message(STATUS "Using raisim package prefix: ${RAISIM_PREFIX}")

set(RAYRAI_PREFIX "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rayrai/${RAISIM_OS}/lib/cmake/rayrai/rayraiConfig.cmake")
    set(RAYRAI_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/rayrai/${RAISIM_OS})
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rayrai/lib/cmake/rayrai/rayraiConfig.cmake")
    set(RAYRAI_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/rayrai")
endif()

if(NOT RAYRAI_PREFIX STREQUAL "")
    list(APPEND CMAKE_PREFIX_PATH ${RAYRAI_PREFIX})
    message(STATUS "Using rayrai package prefix: ${RAYRAI_PREFIX}")
endif()

#==
# Package installation
#==
set(RAISIM_BIN_DIR ${RAISIM_OS})
message("-- Detected OS/Architecture is ${RAISIM_OS}")

if(WIN32)
    install(
        DIRECTORY
        ${RAISIM_PREFIX}/bin
        DESTINATION
        ${CMAKE_INSTALL_PREFIX}
    )
else()
    install(
        DIRECTORY
        ${RAISIM_PREFIX}/lib
        DESTINATION
        ${CMAKE_INSTALL_PREFIX}
    )
endif()

install(
        DIRECTORY
        ${RAISIM_PREFIX}/include
        DESTINATION
        ${CMAKE_INSTALL_PREFIX}
)

if(NOT RAYRAI_PREFIX STREQUAL "")
    install(
        DIRECTORY
        ${RAYRAI_PREFIX}/lib
        DESTINATION
        ${CMAKE_INSTALL_PREFIX}
    )

    install(
        DIRECTORY
        ${RAYRAI_PREFIX}/include
        DESTINATION
        ${CMAKE_INSTALL_PREFIX}
    )

    if(EXISTS ${RAYRAI_PREFIX}/share)
        install(
            DIRECTORY
            ${RAYRAI_PREFIX}/share
            DESTINATION
            ${CMAKE_INSTALL_PREFIX}
        )
    endif()

    if(EXISTS ${RAYRAI_PREFIX}/bin)
        install(
            DIRECTORY
            ${RAYRAI_PREFIX}/bin
            DESTINATION
            ${CMAKE_INSTALL_PREFIX}
        )
    endif()

    if(EXISTS ${RAYRAI_PREFIX}/cmake)
        install(
            DIRECTORY
            ${RAYRAI_PREFIX}/cmake
            DESTINATION
            ${CMAKE_INSTALL_PREFIX}
        )
    endif()
endif()

install(FILES package.xml DESTINATION "${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}")

if(RAISIM_ALL)
    set(RAISIM_EXAMPLE TRUE)
    set(RAISIM_PY TRUE)
    set(RAISIM_DOC TRUE)
endif()

if(RAISIM_EXAMPLE)
    add_subdirectory(examples)
endif()

if(RAISIM_PY)
    add_subdirectory(thirdParty/nanobind)
    add_subdirectory(raisimPy)
endif()

if(RAISIM_DOC)
    add_subdirectory(docs)
endif()
