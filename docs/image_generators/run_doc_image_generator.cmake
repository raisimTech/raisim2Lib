# Runs one documentation image generator with a small retry budget.
# Headless SDL/GL setup can fail transiently when many generators are run
# back-to-back from the CMake docs target.

cmake_minimum_required(VERSION 3.18)

foreach(_required DOC_IMAGE_EXECUTABLE DOC_IMAGE_OUTPUT_DIR DOC_IMAGE_RUNTIME_ENV_VAR)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()

if(NOT DEFINED DOC_IMAGE_NAME)
  get_filename_component(DOC_IMAGE_NAME "${DOC_IMAGE_EXECUTABLE}" NAME)
endif()
if(NOT DEFINED DOC_IMAGE_ATTEMPTS)
  set(DOC_IMAGE_ATTEMPTS 3)
endif()
if(NOT DEFINED DOC_IMAGE_RUNTIME_ENV_VALUE)
  set(DOC_IMAGE_RUNTIME_ENV_VALUE "")
endif()

set(_doc_image_env
    --unset=MAKEFLAGS
    --unset=MFLAGS
    --unset=MAKELEVEL)
if(NOT DOC_IMAGE_RUNTIME_ENV_VAR STREQUAL "")
  list(APPEND _doc_image_env
      "${DOC_IMAGE_RUNTIME_ENV_VAR}=${DOC_IMAGE_RUNTIME_ENV_VALUE}")
endif()
if(DEFINED DOC_IMAGE_ROLLING_FRAMES_DIR AND NOT DOC_IMAGE_ROLLING_FRAMES_DIR STREQUAL "")
  list(APPEND _doc_image_env
      "RAISIM_DOC_ROLLING_FRICTION_FRAMES_DIR=${DOC_IMAGE_ROLLING_FRAMES_DIR}")
endif()
if(DEFINED DOC_IMAGE_STRANDBEEST_FRAMES_DIR AND NOT DOC_IMAGE_STRANDBEEST_FRAMES_DIR STREQUAL "")
  list(APPEND _doc_image_env
      "RAISIM_DOC_STRANDBEEST_FRAMES_DIR=${DOC_IMAGE_STRANDBEEST_FRAMES_DIR}")
endif()
if(DEFINED DOC_IMAGE_LIDAR_FRAMES_DIR AND NOT DOC_IMAGE_LIDAR_FRAMES_DIR STREQUAL "")
  list(APPEND _doc_image_env
      "RAISIM_DOC_RAY_SCAN_LIDAR_FRAMES_DIR=${DOC_IMAGE_LIDAR_FRAMES_DIR}")
endif()
if(DEFINED DOC_IMAGE_GALLERY_FRAMES_DIR AND NOT DOC_IMAGE_GALLERY_FRAMES_DIR STREQUAL "")
  list(APPEND _doc_image_env
      "RAISIM_DOC_GALLERY_FRAMES_DIR=${DOC_IMAGE_GALLERY_FRAMES_DIR}")
endif()
if(NOT DEFINED DOC_IMAGE_ARGUMENTS OR DOC_IMAGE_ARGUMENTS STREQUAL "")
  set(DOC_IMAGE_ARGUMENTS "${DOC_IMAGE_OUTPUT_DIR}")
endif()

set(_attempt 1)
while(_attempt LESS_EQUAL DOC_IMAGE_ATTEMPTS)
  execute_process(
      COMMAND "${CMAKE_COMMAND}" -E env ${_doc_image_env}
              -- "${DOC_IMAGE_EXECUTABLE}" ${DOC_IMAGE_ARGUMENTS}
      WORKING_DIRECTORY "${DOC_IMAGE_OUTPUT_DIR}"
      RESULT_VARIABLE _result)
  if(_result STREQUAL "0")
    return()
  endif()

  if(_attempt LESS DOC_IMAGE_ATTEMPTS)
    message(WARNING
        "doc_image: ${DOC_IMAGE_NAME} failed with ${_result}; "
        "retrying (${_attempt}/${DOC_IMAGE_ATTEMPTS})")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1)
  endif()
  math(EXPR _attempt "${_attempt} + 1")
endwhile()

message(FATAL_ERROR
    "doc_image: ${DOC_IMAGE_NAME} failed after ${DOC_IMAGE_ATTEMPTS} "
    "attempts (last result: ${_result})")
