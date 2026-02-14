find_package(Doxygen REQUIRED)
find_package(raisim CONFIG REQUIRED)

set(RAISIM_DOCS_VENV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.venv")
if(WIN32)
  set(RAISIM_DOCS_VENV_SPHINX "${RAISIM_DOCS_VENV_DIR}/Scripts/sphinx-build.exe")
else()
  set(RAISIM_DOCS_VENV_SPHINX "${RAISIM_DOCS_VENV_DIR}/bin/sphinx-build")
endif()

if(EXISTS "${RAISIM_DOCS_VENV_SPHINX}")
  set(SPHINX_EXECUTABLE "${RAISIM_DOCS_VENV_SPHINX}" CACHE FILEPATH "Sphinx executable for docs build" FORCE)
else()
  find_package(Sphinx)
  if(NOT SPHINX_FOUND OR NOT SPHINX_EXECUTABLE OR SPHINX_EXECUTABLE STREQUAL "SPHINX_EXECUTABLE-NOTFOUND")
    message(FATAL_ERROR
      "Sphinx not found. Create a local venv in docs/ and install requirements:\n"
      "  Ubuntu/Debian: sudo apt install python3-venv\n"
      "  python3 -m venv ${CMAKE_CURRENT_SOURCE_DIR}/.venv\n"
      "  . ${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/activate\n"
      "  python -m pip install -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt\n"
      "Or provide SPHINX_EXECUTABLE explicitly.")
  endif()
endif()

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)

set(RAISIM_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/raisim/linux/include/raisim")
set(RAYRAI_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/rayrai/linux/include/rayrai")
set(DOXYGEN_INPUT_DIR "${RAISIM_INCLUDE_DIR} ${RAYRAI_INCLUDE_DIR}")

if(NOT EXISTS "${RAISIM_INCLUDE_DIR}")
  message(FATAL_ERROR
    "RaiSim headers not found at ${RAISIM_INCLUDE_DIR}. "
    "Ensure the prebuilt package exists under raisim/linux or update the path.")
endif()
if(NOT EXISTS "${RAYRAI_INCLUDE_DIR}")
  message(FATAL_ERROR
    "Rayrai headers not found at ${RAYRAI_INCLUDE_DIR}. "
    "Ensure the prebuilt package exists under rayrai/linux or update the path.")
endif()

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
