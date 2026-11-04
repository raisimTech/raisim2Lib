cmake_minimum_required(VERSION 3.18)
project(raisim_docs LANGUAGES CXX)

get_filename_component(RAISIM_DOCS_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

find_package(Doxygen REQUIRED)
find_package(raisim CONFIG REQUIRED)

set(RAISIM_DOCS_RAISIM_INCLUDE_DIR "" CACHE PATH "Override path to RaiSim public headers directory (the folder containing raiSim headers, usually .../include/raisim)")
set(RAISIM_DOCS_RAYRAI_INCLUDE_DIR "" CACHE PATH "Override path to Rayrai public headers directory (usually .../include/rayrai)")

# The site is built with VitePress (Node.js) from the Markdown pages in this
# directory; scripts/generate_api_docs.py turns the Doxygen XML into the API pages.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(RAISIM_DOCS_NODE NAMES node)
if(WIN32)
  # Node.js for Windows also ships an extensionless npm shell script, which
  # Windows cannot run.
  find_program(RAISIM_DOCS_NPM NAMES npm.cmd)
else()
  find_program(RAISIM_DOCS_NPM NAMES npm)
endif()
if(NOT RAISIM_DOCS_NODE OR NOT RAISIM_DOCS_NPM)
  message(FATAL_ERROR "Node.js and npm not found. Install Node.js 18 or newer to build the documentation site.")
endif()
execute_process(COMMAND "${RAISIM_DOCS_NODE}" --version
                OUTPUT_VARIABLE RAISIM_DOCS_NODE_VERSION
                RESULT_VARIABLE _RAISIM_DOCS_NODE_RESULT
                OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
string(REGEX REPLACE "^v" "" RAISIM_DOCS_NODE_VERSION "${RAISIM_DOCS_NODE_VERSION}")
if(NOT _RAISIM_DOCS_NODE_RESULT EQUAL 0 OR
   NOT RAISIM_DOCS_NODE_VERSION MATCHES "^[0-9]+\\." OR
   RAISIM_DOCS_NODE_VERSION VERSION_LESS 18)
  message(FATAL_ERROR
    "The documentation site needs Node.js 18 or newer (VitePress); "
    "${RAISIM_DOCS_NODE} reports version '${RAISIM_DOCS_NODE_VERSION}'.")
endif()

set(_RAISIM_INCLUDE_CANDIDATES)
if(RAISIM_DOCS_RAISIM_INCLUDE_DIR)
  list(APPEND _RAISIM_INCLUDE_CANDIDATES "${RAISIM_DOCS_RAISIM_INCLUDE_DIR}")
endif()
list(APPEND _RAISIM_INCLUDE_CANDIDATES
  "${RAISIM_DOCS_REPO_ROOT}/raisim/include/raisim"
  "${RAISIM_DOCS_REPO_ROOT}/raisim/linux/include/raisim"
)
foreach(_RAISIM_BASE_INCLUDE IN LISTS raisim_INCLUDE_DIRS)
  list(APPEND _RAISIM_INCLUDE_CANDIDATES "${_RAISIM_BASE_INCLUDE}/raisim")
endforeach()

set(RAISIM_INCLUDE_DIR "")
foreach(_RAISIM_CANDIDATE IN LISTS _RAISIM_INCLUDE_CANDIDATES)
  if(EXISTS "${_RAISIM_CANDIDATE}")
    set(RAISIM_INCLUDE_DIR "${_RAISIM_CANDIDATE}")
    break()
  endif()
endforeach()

set(_RAYRAI_INCLUDE_CANDIDATES)
if(RAISIM_DOCS_RAYRAI_INCLUDE_DIR)
  list(APPEND _RAYRAI_INCLUDE_CANDIDATES "${RAISIM_DOCS_RAYRAI_INCLUDE_DIR}")
endif()
list(APPEND _RAYRAI_INCLUDE_CANDIDATES
  "${RAISIM_DOCS_REPO_ROOT}/rayrai/include/rayrai"
  "${RAISIM_DOCS_REPO_ROOT}/rayrai/linux/include/rayrai"
)
set(RAYRAI_INCLUDE_DIR "")
foreach(_RAYRAI_CANDIDATE IN LISTS _RAYRAI_INCLUDE_CANDIDATES)
  if(EXISTS "${_RAYRAI_CANDIDATE}")
    set(RAYRAI_INCLUDE_DIR "${_RAYRAI_CANDIDATE}")
    break()
  endif()
endforeach()

if(NOT RAISIM_INCLUDE_DIR)
  string(REPLACE ";" "\n  - " _RAISIM_INCLUDE_CANDIDATES_FMT "${_RAISIM_INCLUDE_CANDIDATES}")
  message(FATAL_ERROR
    "RaiSim headers not found. Checked:\n"
    "  - ${_RAISIM_INCLUDE_CANDIDATES_FMT}\n"
    "Ensure the prebuilt package is present or set the include path explicitly.")
endif()
if(NOT RAYRAI_INCLUDE_DIR)
  string(REPLACE ";" "\n  - " _RAYRAI_INCLUDE_CANDIDATES_FMT "${_RAYRAI_INCLUDE_CANDIDATES}")
  message(FATAL_ERROR
    "Rayrai headers not found. Checked:\n"
    "  - ${_RAYRAI_INCLUDE_CANDIDATES_FMT}\n"
    "Ensure the prebuilt package is present or set the include path explicitly.")
endif()

set(DOXYGEN_INPUT_DIR "${RAISIM_INCLUDE_DIR} ${RAYRAI_INCLUDE_DIR}")

file(GLOB_RECURSE RAISIM_PUBLIC_HEADERS CONFIGURE_DEPENDS
  "${RAISIM_INCLUDE_DIR}/*.hpp"
  "${RAISIM_INCLUDE_DIR}/*.h"
  "${RAYRAI_INCLUDE_DIR}/*.hpp"
  "${RAYRAI_INCLUDE_DIR}/*.h"
)

set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Replace variables inside @@ with the current values. A change to
# Doxyfile.in reconfigures the project, which rewrites the Doxyfile.
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

# Doxygen writes only XML (Doxyfile.in), which generate_api_docs.py reads.
add_custom_target(doxygen_output ALL
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
        DEPENDS ${RAISIM_PUBLIC_HEADERS} ${DOXYFILE_OUT}
        COMMENT "Generating the Doxygen XML"
        VERBATIM)

# Optional: build and run small C++ programs that regenerate the rayrai
# documentation images. Off by default, so the docs use the checked-in images.
# When ON, the image generators rewrite the images in rsc/docs/image/rayrai/
# and the docs_site target depends on them, so a site rebuild refreshes the
# images first. Regenerating needs rayrai, SDL2 and an OpenGL context.
option(RAISIM_DOCS_BUILD_RAYRAI_IMAGES
    "Build and run the rayrai documentation image generators before the site build"
    OFF)

set(_RAYRAI_DOC_IMAGE_DEPS "")
if(RAISIM_DOCS_BUILD_RAYRAI_IMAGES AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/image_generators/CMakeLists.txt")
  add_subdirectory(image_generators)
  if(DEFINED RAYRAI_DOC_IMAGE_TARGET AND TARGET ${RAYRAI_DOC_IMAGE_TARGET})
    set(_RAYRAI_DOC_IMAGE_DEPS ${RAYRAI_DOC_IMAGE_TARGET})
  endif()
endif()

# The site is built from a copy of the sources in the build directory, so a
# build writes nothing into the source tree and two build directories do not
# share any output. cmake/StageSiteSources.cmake copies this directory fresh to
# ${RAISIM_DOCS_STAGE_DIR}/docs on every build and links the repository's
# rsc/, examples/ and scripts/ next to it, where the relative paths of the
# pages expect them.
set(RAISIM_DOCS_STAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/site-src)
add_custom_target(docs_stage
        COMMAND ${CMAKE_COMMAND}
                -DDOCS_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                -DREPO_ROOT=${RAISIM_DOCS_REPO_ROOT}
                -DSTAGE_DIR=${RAISIM_DOCS_STAGE_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/StageSiteSources.cmake
        COMMENT "Staging the documentation sources in ${RAISIM_DOCS_STAGE_DIR}"
        VERBATIM)

# API pages: Markdown generated from the Doxygen XML into the staged sources.
add_custom_target(docs_api
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_api_docs.py
                --xml ${DOXYGEN_OUTPUT_DIR}/xml --out ${RAISIM_DOCS_STAGE_DIR}/docs/api
        COMMENT "Generating the API reference pages"
        VERBATIM)
add_dependencies(docs_api doxygen_output docs_stage)

# Node packages, installed from package-lock.json whenever it changes. They go
# to ${CMAKE_CURRENT_BINARY_DIR}/node_modules, a parent of the staged sources,
# where Node.js finds them.
set(RAISIM_DOCS_NPM_STAMP ${CMAKE_CURRENT_BINARY_DIR}/npm_ci.stamp)
add_custom_command(
        OUTPUT ${RAISIM_DOCS_NPM_STAMP}
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/package.json
                ${CMAKE_CURRENT_SOURCE_DIR}/package-lock.json
                ${CMAKE_CURRENT_BINARY_DIR}
        COMMAND ${RAISIM_DOCS_NPM} ci --no-audit --no-fund
        COMMAND ${CMAKE_COMMAND} -E touch ${RAISIM_DOCS_NPM_STAMP}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/package.json ${CMAKE_CURRENT_SOURCE_DIR}/package-lock.json
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Installing the documentation site's Node packages"
        VERBATIM)

# The static site, written to ${RAISIM_DOCS_SITE_DIR}. VitePress fails on dead
# links, so a page that links to a missing page or API entry stops the build.
# The browser examples (web-examples/) and the games (games/) join the site after
# it is built, so links into them are not checked here.
set(RAISIM_DOCS_SITE_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/site)
add_custom_target(docs_site ALL
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${RAISIM_DOCS_SITE_DIR}
        COMMAND ${RAISIM_DOCS_NODE} ${CMAKE_CURRENT_BINARY_DIR}/node_modules/vitepress/bin/vitepress.js
                build ${RAISIM_DOCS_STAGE_DIR}/docs --outDir ${RAISIM_DOCS_SITE_DIR}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_links.py ${RAISIM_DOCS_SITE_DIR}
                --ignore /web-examples/ --ignore /games/
        DEPENDS ${RAISIM_DOCS_NPM_STAMP}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Building the documentation site with VitePress"
        VERBATIM)
add_dependencies(docs_site docs_api ${_RAYRAI_DOC_IMAGE_DEPS})

add_custom_target(Doxygen ALL DEPENDS doxygen_output)
