find_package(Doxygen REQUIRED)
find_package(raisim CONFIG REQUIRED)

option(RAISIM_DOCS_BOOTSTRAP_VENV "Automatically create docs/.venv and install Sphinx requirements when missing" ON)
set(RAISIM_DOCS_RAISIM_INCLUDE_DIR "" CACHE PATH "Override path to RaiSim public headers directory (the folder containing raiSim headers, usually .../include/raisim)")
set(RAISIM_DOCS_RAYRAI_INCLUDE_DIR "" CACHE PATH "Override path to Rayrai public headers directory (usually .../include/rayrai)")

set(RAISIM_DOCS_VENV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.venv")
if(WIN32)
  set(RAISIM_DOCS_VENV_SPHINX "${RAISIM_DOCS_VENV_DIR}/Scripts/sphinx-build.exe")
  set(RAISIM_DOCS_VENV_PYTHON "${RAISIM_DOCS_VENV_DIR}/Scripts/python.exe")
else()
  set(RAISIM_DOCS_VENV_SPHINX "${RAISIM_DOCS_VENV_DIR}/bin/sphinx-build")
  set(RAISIM_DOCS_VENV_PYTHON "${RAISIM_DOCS_VENV_DIR}/bin/python")
endif()

if(EXISTS "${RAISIM_DOCS_VENV_SPHINX}")
  set(SPHINX_EXECUTABLE "${RAISIM_DOCS_VENV_SPHINX}" CACHE FILEPATH "Sphinx executable for docs build" FORCE)
else()
  set(_RAISIM_SPHINX_READY OFF)

  find_package(Sphinx QUIET)
  if(SPHINX_FOUND AND SPHINX_EXECUTABLE AND NOT SPHINX_EXECUTABLE STREQUAL "SPHINX_EXECUTABLE-NOTFOUND")
    set(_RAISIM_SPHINX_READY ON)
  endif()

  if(NOT _RAISIM_SPHINX_READY AND RAISIM_DOCS_BOOTSTRAP_VENV)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    if(NOT EXISTS "${RAISIM_DOCS_VENV_PYTHON}")
      message(STATUS "Sphinx not found; creating docs virtualenv at ${RAISIM_DOCS_VENV_DIR}")
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m venv "${RAISIM_DOCS_VENV_DIR}"
        RESULT_VARIABLE _RAISIM_VENV_RESULT
      )
      if(NOT _RAISIM_VENV_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to create docs venv at ${RAISIM_DOCS_VENV_DIR}. Install python3-venv and retry.")
      endif()
    endif()

    message(STATUS "Installing docs Python requirements from ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt")
    execute_process(
      COMMAND "${RAISIM_DOCS_VENV_PYTHON}" -m pip install -r "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt"
      RESULT_VARIABLE _RAISIM_PIP_RESULT
    )
    if(NOT _RAISIM_PIP_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to install docs requirements into ${RAISIM_DOCS_VENV_DIR}.")
    endif()

    if(EXISTS "${RAISIM_DOCS_VENV_SPHINX}")
      set(SPHINX_EXECUTABLE "${RAISIM_DOCS_VENV_SPHINX}" CACHE FILEPATH "Sphinx executable for docs build" FORCE)
      set(_RAISIM_SPHINX_READY ON)
    endif()
  endif()

  if(NOT _RAISIM_SPHINX_READY)
    message(FATAL_ERROR
      "Sphinx not found. Enable RAISIM_DOCS_BOOTSTRAP_VENV=ON to auto-bootstrap docs/.venv,\n"
      "create it manually, or provide SPHINX_EXECUTABLE explicitly.")
  endif()
endif()

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)

set(_RAISIM_INCLUDE_CANDIDATES)
if(RAISIM_DOCS_RAISIM_INCLUDE_DIR)
  list(APPEND _RAISIM_INCLUDE_CANDIDATES "${RAISIM_DOCS_RAISIM_INCLUDE_DIR}")
endif()
list(APPEND _RAISIM_INCLUDE_CANDIDATES
  "${PROJECT_SOURCE_DIR}/raisim/include/raisim"
  "${PROJECT_SOURCE_DIR}/raisim/linux/include/raisim"
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
  "${PROJECT_SOURCE_DIR}/rayrai/include/rayrai"
  "${PROJECT_SOURCE_DIR}/rayrai/linux/include/rayrai"
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

set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

#Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

add_custom_target(doxygen_output ALL
        DEPENDS ${RAISIM_PUBLIC_HEADERS}
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
        MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
        COMMENT "Generating docs")

add_custom_target(Sphinx ALL
        DEPENDS ${RAISIM_PUBLIC_HEADERS} doxygen_output
        COMMAND ${SPHINX_EXECUTABLE} -b html -E -a
        # Tell Breathe where to find the Doxygen output
        -Dbreathe_projects.raisim=${DOXYGEN_OUTPUT_DIR}/xml
        ${SPHINX_SOURCE} ${SPHINX_BUILD}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating documentation with Sphinx")

add_custom_target(Doxygen ALL DEPENDS doxygen_output)
