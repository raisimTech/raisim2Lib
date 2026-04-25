cmake_minimum_required(VERSION 3.18)
set(RAISIM_VERSION 2.0.0 CACHE STRING "Default raisim release version to download when raisim is missing.")
project(raisim VERSION ${RAISIM_VERSION} LANGUAGES CXX)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Include CMake helper modules
include(GNUInstallDirs)

# Build options
option(RAISIM_EXAMPLE "Build example applications." TRUE)
option(RAISIM_MATLAB "Build a matlab wrapper." FALSE)
option(RAISIM_PY "Build a python wrapper." FALSE)
option(RAISIM_DOC "Build a raisim documentation. You can find the latest doc in raisim.com." FALSE)
option(RAISIM_CHECK_FOR_UPDATES "Check GitHub for newer raisim releases when configuring." TRUE)
set(RAISIM_UPGRADE "" CACHE STRING "Set to ON to upgrade an existing raisim directory to the latest available release, or OFF to skip the prompt.")
set_property(CACHE RAISIM_UPGRADE PROPERTY STRINGS "" ON OFF)

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

function(_raisim_fetch_latest_release out_var error_var)
    set(_raisim_latest_json "${CMAKE_CURRENT_BINARY_DIR}/downloads/raisim-latest-release.json")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/downloads")
    file(DOWNLOAD
        "https://api.github.com/repos/raisimTech/raisim2Lib/releases/latest"
        "${_raisim_latest_json}"
        STATUS _raisim_latest_status
        TIMEOUT 5
        TLS_VERIFY ON
        HTTPHEADER "Accept: application/vnd.github+json"
    )

    list(GET _raisim_latest_status 0 _raisim_latest_code)
    if(NOT _raisim_latest_code EQUAL 0)
        list(GET _raisim_latest_status 1 _raisim_latest_message)
        set(${out_var} "" PARENT_SCOPE)
        set(${error_var} "${_raisim_latest_code}: ${_raisim_latest_message}" PARENT_SCOPE)
        return()
    endif()

    file(READ "${_raisim_latest_json}" _raisim_latest_contents)
    if(_raisim_latest_contents MATCHES "\"tag_name\"[ \t\r\n]*:[ \t\r\n]*\"v?([0-9][^\"]*)\"")
        set(${out_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
        set(${error_var} "" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
        set(${error_var} "Could not parse latest release tag from GitHub response." PARENT_SCOPE)
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
set(_raisim_latest_version "")
set(_raisim_latest_error "")

if(RAISIM_CHECK_FOR_UPDATES)
    _raisim_fetch_latest_release(_raisim_latest_version _raisim_latest_error)
    if(_raisim_latest_version STREQUAL "")
        message(STATUS "Skipping raisim update check: ${_raisim_latest_error}")
    else()
        message(STATUS "Latest raisim release available online: ${_raisim_latest_version}")
    endif()
endif()

if(RAISIM_PREFIX STREQUAL "")
    if(NOT _raisim_latest_version STREQUAL "")
        set(RAISIM_VERSION "${_raisim_latest_version}")
    endif()
    message(STATUS "No usable raisim package found. Downloading raisim ${RAISIM_VERSION}.")
    _raisim_download_release("${RAISIM_VERSION}")
    _raisim_find_package_prefix(RAISIM_PREFIX)
else()
    _raisim_read_package_version("${RAISIM_PREFIX}" _raisim_installed_version)
    if(_raisim_installed_version STREQUAL "")
        message(STATUS "Using existing raisim package; could not determine installed version.")
    else()
        message(STATUS "Found existing raisim package ${_raisim_installed_version}: ${RAISIM_PREFIX}")
    endif()

    if(NOT _raisim_latest_version STREQUAL ""
            AND NOT _raisim_installed_version STREQUAL ""
            AND _raisim_installed_version VERSION_LESS _raisim_latest_version)
        if(RAISIM_UPGRADE)
            set(RAISIM_VERSION "${_raisim_latest_version}")
            message(STATUS "Upgrading raisim from ${_raisim_installed_version} to ${RAISIM_VERSION}.")
            _raisim_download_release("${RAISIM_VERSION}")
            _raisim_find_package_prefix(RAISIM_PREFIX)
        elseif(RAISIM_UPGRADE STREQUAL "")
            message(STATUS
                "A newer raisim release is available (${_raisim_installed_version} -> ${_raisim_latest_version}). "
                "Would you like to upgrade? Configure with -DRAISIM_UPGRADE=ON to upgrade, "
                "or -DRAISIM_UPGRADE=OFF to keep the existing copy without this prompt."
            )
        else()
            message(STATUS "Keeping existing raisim ${_raisim_installed_version}; latest online release is ${_raisim_latest_version}.")
        endif()
    endif()
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
