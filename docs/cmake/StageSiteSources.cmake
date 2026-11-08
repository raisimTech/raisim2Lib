# Stages the sources of the documentation site in the build directory, so that
# a site build writes nothing into the source tree:
#
#   cmake -DDOCS_SOURCE_DIR=<raisim2Lib/docs> -DREPO_ROOT=<raisim2Lib>
#         -DSTAGE_DIR=<build>/site-src -P StageSiteSources.cmake
#
# It leaves this layout, which the relative paths of the pages expect:
#
#   STAGE_DIR/docs            a fresh copy of DOCS_SOURCE_DIR without
#                             node_modules, api, .venv, _build, the VitePress
#                             cache and output, and generator leftovers
#   STAGE_DIR/CMakeLists.txt  a copy of the root CMakeLists.txt, where
#                             .vitepress/config.mts reads the release version
#   STAGE_DIR/rsc             links to the directories of REPO_ROOT that pages
#   STAGE_DIR/examples        reference as ../../rsc/... and
#   STAGE_DIR/scripts         <<< @/../examples/... (symbolic links on POSIX,
#                             directory junctions on Windows, which need no
#                             administrator rights)
#
# Every run removes the previous staging first, so pages deleted from the
# sources do not linger. The links are removed before anything else, with
# commands that never descend into a link, so no recursive delete can reach
# the directories they point to.

cmake_minimum_required(VERSION 3.18)

foreach(_var IN ITEMS DOCS_SOURCE_DIR REPO_ROOT STAGE_DIR)
  if(NOT ${_var})
    message(FATAL_ERROR "StageSiteSources.cmake: set -D${_var}=<path>")
  endif()
endforeach()

set(_linked_dirs rsc examples scripts)
# Entries of DOCS_SOURCE_DIR that are not copied (anywhere in the tree for
# node_modules and __pycache__).
set(_skipped_entries api node_modules .venv _build)

# Removes a link that an earlier run created; never what the link points to.
function(_remove_staged_link link)
  if(NOT EXISTS "${link}" AND NOT IS_SYMLINK "${link}")
    return()
  endif()
  if(CMAKE_HOST_WIN32)
    # rmdir without /s removes a junction itself and refuses a real,
    # non-empty directory instead of deleting its contents.
    file(TO_NATIVE_PATH "${link}" _native)
    execute_process(COMMAND cmd /c rmdir "${_native}"
                    RESULT_VARIABLE _result ERROR_VARIABLE _error)
  elseif(IS_SYMLINK "${link}")
    file(REMOVE "${link}")
    set(_result 0)
  else()
    set(_result 1)
    set(_error "not a symbolic link")
  endif()
  if(NOT _result EQUAL 0 OR EXISTS "${link}" OR IS_SYMLINK "${link}")
    message(FATAL_ERROR
      "Cannot remove ${link} (${_error}). The documentation build expects a "
      "link it created there; remove the path by hand.")
  endif()
endfunction()

foreach(_name IN LISTS _linked_dirs)
  _remove_staged_link("${STAGE_DIR}/${_name}")
endforeach()
file(REMOVE_RECURSE "${STAGE_DIR}/docs")
file(REMOVE "${STAGE_DIR}/CMakeLists.txt")

file(MAKE_DIRECTORY "${STAGE_DIR}/docs")
file(GLOB _entries LIST_DIRECTORIES true RELATIVE "${DOCS_SOURCE_DIR}"
     "${DOCS_SOURCE_DIR}/*")
foreach(_entry IN LISTS _entries)
  # .api.new-* and .api.old-* are siblings that generate_api_docs.py writes.
  if(_entry IN_LIST _skipped_entries OR _entry MATCHES "^\\.api\\.")
    continue()
  endif()
  file(COPY "${DOCS_SOURCE_DIR}/${_entry}" DESTINATION "${STAGE_DIR}/docs"
       PATTERN node_modules EXCLUDE
       PATTERN __pycache__ EXCLUDE
       REGEX "/\\.vitepress/(\\.temp|cache|dist)$" EXCLUDE)
endforeach()
file(COPY "${REPO_ROOT}/CMakeLists.txt" DESTINATION "${STAGE_DIR}")

foreach(_name IN LISTS _linked_dirs)
  set(_target "${REPO_ROOT}/${_name}")
  set(_link "${STAGE_DIR}/${_name}")
  if(NOT IS_DIRECTORY "${_target}")
    message(FATAL_ERROR "The documentation needs ${_target}, which is missing.")
  endif()
  if(CMAKE_HOST_WIN32)
    file(TO_NATIVE_PATH "${_target}" _target_native)
    file(TO_NATIVE_PATH "${_link}" _link_native)
    execute_process(COMMAND cmd /c mklink /J "${_link_native}" "${_target_native}"
                    RESULT_VARIABLE _result OUTPUT_QUIET ERROR_VARIABLE _error)
  else()
    file(CREATE_LINK "${_target}" "${_link}" RESULT _result SYMBOLIC)
    set(_error "${_result}")
  endif()
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Cannot link ${_link} to ${_target}: ${_error}")
  endif()
endforeach()
