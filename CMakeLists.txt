cmake_minimum_required(VERSION 3.18)
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

set(_raisim_expected_version 2.5.7)
if(DEFINED CACHE{RAISIM_VERSION} AND NOT "${RAISIM_VERSION}" VERSION_EQUAL "${_raisim_expected_version}")
    message(FATAL_ERROR
        "RAISIM_VERSION is ${RAISIM_VERSION}, but this checkout expects ${_raisim_expected_version}. "
        "Remove the stale CMake cache or configure with -DRAISIM_VERSION=${_raisim_expected_version}."
    )
endif()
set(RAISIM_VERSION 2.5.7 CACHE STRING "Raisim release version to download.")
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

if (UNIX AND NOT APPLE AND NOT ANDROID AND NOT WEBGL AND NOT WIN32)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(RAISIM_OS linux-arm)
    else ()
        set(RAISIM_OS linux)
    endif ()
elseif(APPLE)
    if(_raisim_apple_silicon
            OR CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(RAISIM_OS m1)
    else()
        set(RAISIM_OS mac)
    endif()
elseif(WIN32)
    set(RAISIM_OS win32)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/Eigen3)

    set(_vcpkg_root "$ENV{VCPKG_ROOT}")
    if(_vcpkg_root STREQUAL "" AND EXISTS "C:/vcpkg")
        set(_vcpkg_root "C:/vcpkg")
    endif()
    if(NOT _vcpkg_root STREQUAL "")
        set(_vcpkg_triplet "$ENV{VCPKG_DEFAULT_TRIPLET}")
        if(_vcpkg_triplet STREQUAL "")
            set(_vcpkg_triplet "x64-windows")
        endif()

        set(_vcpkg_installed_prefix "${_vcpkg_root}/installed/${_vcpkg_triplet}")
        if(EXISTS "${_vcpkg_installed_prefix}")
            list(APPEND CMAKE_PREFIX_PATH "${_vcpkg_installed_prefix}")
            if(NOT DEFINED SDL2_DIR AND EXISTS "${_vcpkg_installed_prefix}/share/sdl2/SDL2Config.cmake")
                set(SDL2_DIR "${_vcpkg_installed_prefix}/share/sdl2")
            endif()
        endif()
    endif()
endif()

function(_raisim_find_package_prefix out_var)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/raisim/${RAISIM_OS}/lib/cmake/raisim/raisimConfig.cmake")
        set(${out_var} "${CMAKE_CURRENT_SOURCE_DIR}/raisim/${RAISIM_OS}" PARENT_SCOPE)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/raisim/lib/cmake/raisim/raisimConfig.cmake")
        set(${out_var} "${CMAKE_CURRENT_SOURCE_DIR}/raisim" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

function(_raisim_read_package_version prefix out_var)
    set(_raisim_version_file "")
    if(EXISTS "${prefix}/lib/cmake/raisim/raisimConfig-version.cmake")
        set(_raisim_version_file "${prefix}/lib/cmake/raisim/raisimConfig-version.cmake")
    endif()

    if(_raisim_version_file STREQUAL "")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    file(READ "${_raisim_version_file}" _raisim_version_contents)
    if(_raisim_version_contents MATCHES "set\\(PACKAGE_VERSION \"([^\"]+)\"\\)")
        set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

function(_raisim_download_release version)
    set(_raisim_asset_file "${_raisim_asset_platform}-${version}.zip")
    set(_raisim_download_dir "${CMAKE_CURRENT_BINARY_DIR}/downloads")
    set(_raisim_asset_path "${_raisim_download_dir}/${_raisim_asset_file}")
    file(MAKE_DIRECTORY "${_raisim_download_dir}")

    set(_raisim_download_success FALSE)
    set(_raisim_last_download_error "")
    set(_raisim_download_urls
        "https://github.com/raisimTech/raisim2Lib/releases/download/v${version}/${_raisim_asset_file}"
        "https://github.com/raisimTech/raisim2Lib/releases/download/${version}/${_raisim_asset_file}"
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

    set(_raisim_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/extract-${version}")
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
    file(REMOVE_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/${_raisim_asset_platform}-${version}")

    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/raisim")
        message(FATAL_ERROR "Archive extraction completed but 'raisim' directory was not found.")
    endif()
endfunction()

_raisim_find_package_prefix(RAISIM_PREFIX)
set(_raisim_download_required FALSE)
set(_raisim_download_reason "")

if(RAISIM_PREFIX STREQUAL "")
    set(_raisim_download_required TRUE)
    set(_raisim_download_reason "No usable raisim package found.")
else()
    _raisim_read_package_version("${RAISIM_PREFIX}" _raisim_installed_version)
    if(_raisim_installed_version STREQUAL "")
        set(_raisim_download_required TRUE)
        set(_raisim_download_reason "Found raisim package at ${RAISIM_PREFIX}, but could not determine its version.")
    elseif(NOT "${_raisim_installed_version}" VERSION_EQUAL "${RAISIM_VERSION}")
        set(_raisim_download_required TRUE)
        set(_raisim_download_reason "Found raisim package ${_raisim_installed_version} at ${RAISIM_PREFIX}, but this checkout requires ${RAISIM_VERSION}.")
    else()
        message(STATUS "Found required raisim package ${_raisim_installed_version}: ${RAISIM_PREFIX}")
    endif()
endif()

if(_raisim_download_required)
    message(STATUS "${_raisim_download_reason} Downloading raisim ${RAISIM_VERSION}.")
    _raisim_download_release("${RAISIM_VERSION}")
    _raisim_find_package_prefix(RAISIM_PREFIX)

    if(RAISIM_PREFIX STREQUAL "")
        message(FATAL_ERROR "Could not find raisim package config after extraction.")
    endif()

    _raisim_read_package_version("${RAISIM_PREFIX}" _raisim_installed_version)
    if(_raisim_installed_version STREQUAL "")
        message(FATAL_ERROR "Downloaded raisim ${RAISIM_VERSION}, but could not determine the installed package version.")
    endif()

    if(NOT "${_raisim_installed_version}" VERSION_EQUAL "${RAISIM_VERSION}")
        message(FATAL_ERROR
            "Downloaded raisim package ${_raisim_installed_version}, but this checkout requires ${RAISIM_VERSION}."
        )
    endif()

    message(STATUS "Installed required raisim package ${_raisim_installed_version}: ${RAISIM_PREFIX}")
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
        "${RAISIM_PREFIX}/bin"
        DESTINATION
        "."
    )
else()
    install(
        DIRECTORY
        "${RAISIM_PREFIX}/lib"
        DESTINATION
        "."
    )
endif()

install(
        DIRECTORY
        "${RAISIM_PREFIX}/include"
        DESTINATION
        "."
)

if(NOT RAYRAI_PREFIX STREQUAL "")
    set(_rayrai_lib_install_options)
    set(_raisim_openusd_dir "${RAISIM_PREFIX}/lib/openusd")
    set(_rayrai_openusd_dir "${RAYRAI_PREFIX}/lib/openusd")

    if(IS_SYMLINK "${_rayrai_openusd_dir}")
        if(NOT EXISTS "${_rayrai_openusd_dir}")
            message(FATAL_ERROR
                "rayrai OpenUSD runtime symlink is dangling: ${_rayrai_openusd_dir}"
            )
        endif()

        get_filename_component(_raisim_openusd_realpath "${_raisim_openusd_dir}" REALPATH)
        get_filename_component(_rayrai_openusd_realpath "${_rayrai_openusd_dir}" REALPATH)
        if(_rayrai_openusd_realpath STREQUAL _raisim_openusd_realpath)
            # The release archive reuses the RaiSim OpenUSD runtime through this
            # symlink. RaiSim's lib directory is installed above, so copying the
            # alias would collide with the real destination directory.
            list(APPEND _rayrai_lib_install_options PATTERN "openusd" EXCLUDE)
        else()
            message(FATAL_ERROR
                "rayrai OpenUSD runtime symlink must resolve to ${_raisim_openusd_dir}, "
                "but it resolves to ${_rayrai_openusd_realpath}"
            )
        endif()
    endif()

    install(
        DIRECTORY
        "${RAYRAI_PREFIX}/lib"
        DESTINATION
        "."
        ${_rayrai_lib_install_options}
    )

    install(
        DIRECTORY
        "${RAYRAI_PREFIX}/include"
        DESTINATION
        "."
    )

    if(EXISTS "${RAYRAI_PREFIX}/share")
        install(
            DIRECTORY
            "${RAYRAI_PREFIX}/share"
            DESTINATION
            "."
        )
    endif()

    if(EXISTS "${RAYRAI_PREFIX}/bin")
        install(
            DIRECTORY
            "${RAYRAI_PREFIX}/bin"
            DESTINATION
            "."
        )
    endif()

    if(EXISTS "${RAYRAI_PREFIX}/cmake")
        install(
            DIRECTORY
            "${RAYRAI_PREFIX}/cmake"
            DESTINATION
            "."
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
