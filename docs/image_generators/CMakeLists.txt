# Doc image generators for rayrai.
#
# Each binary renders a small headless scene and writes one or more PNGs into
# ${RAISIM_DOC_IMAGE_OUTPUT_DIR} (default: ${CMAKE_CURRENT_SOURCE_DIR}/../../rsc/docs/image/rayrai).
# The docs/CMakeLists.txt wires a custom target that runs every generator
# before sphinx-build, so a fresh doc build always uses up-to-date images.

cmake_minimum_required(VERSION 3.18)

if(NOT TARGET rayrai)
  find_package(rayrai CONFIG QUIET)
endif()
if(NOT TARGET rayrai)
  message(STATUS "rayrai not available; skipping doc image generators")
  return()
endif()

if(NOT TARGET raisim::raisim)
  find_package(raisim CONFIG QUIET)
endif()
if(NOT TARGET raisim::raisim)
  message(STATUS "raisim not available; skipping doc image generators")
  return()
endif()

find_package(SDL2 QUIET)
if(NOT TARGET SDL2::SDL2)
  message(STATUS "SDL2 not available; skipping doc image generators")
  return()
endif()

set(RAISIM_DOC_IMAGE_OUTPUT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../rsc/docs/image/rayrai"
    CACHE PATH "Directory where doc image generators write PNGs")
if(RAISIM_DOC_IMAGE_OUTPUT_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/../image/rayrai")
  set(RAISIM_DOC_IMAGE_OUTPUT_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/../../rsc/docs/image/rayrai"
      CACHE PATH "Directory where doc image generators write PNGs" FORCE)
endif()
file(MAKE_DIRECTORY "${RAISIM_DOC_IMAGE_OUTPUT_DIR}")
get_filename_component(RAISIM_DOC_IMAGE_OUTPUT_DIR
    "${RAISIM_DOC_IMAGE_OUTPUT_DIR}" REALPATH)

# stb_image_write.h ships in the rayrai install tree under
# <rayrai_prefix>/include/stb. The rayrai CONFIG target already exposes that
# path on its INTERFACE_INCLUDE_DIRECTORIES, so we do not need to add it
# explicitly.

set(_doc_image_generators
    doc_image_minimal_usage
    doc_image_custom_visuals
    doc_image_quality_presets
    doc_image_tone_mapping
    doc_image_color_grading
    doc_image_weather_presets
    doc_image_material_factories
    doc_image_ssr_ssao
    doc_image_cinematic
    doc_image_volumetric_fog
    doc_image_wet_snow
    doc_image_lights
    doc_image_rolling_friction
    doc_image_usd_examples
    doc_image_rscene_minimal
    doc_image_rscene_warehouse
    doc_image_rscene_terrain
    doc_image_rscene_articulated
    doc_image_rscene_atmosphere
    doc_image_rscene_features
    doc_image_constraints
    doc_image_strandbeest
    doc_image_benchmark_scenes
    doc_image_heightmap_rotation)
set(_gallery_scenes rayrai_blue_wall_scene granular_media_showcase
                    tendon_pulleys rayrai_nested_glass_showcase rayrai_city)
foreach(_scene IN LISTS _gallery_scenes)
  list(APPEND _doc_image_generators "doc_image_${_scene}")
endforeach()

set(_doc_image_outputs "")
set(_doc_image_targets "")
set(_doc_image_runner "${CMAKE_CURRENT_SOURCE_DIR}/run_doc_image_generator.cmake")

# Collect the rayrai and raisim install lib directories so we can put them on
# LD_LIBRARY_PATH / DYLD_LIBRARY_PATH / PATH when invoking the generator
# binaries. RUNPATH on the executables covers direct dependencies, but
# transitive loads (e.g. libassimp from librayrai) still need a fallback.
#
# RAYRAI_OVERRIDE_LIB_DIR lets a developer point doc-image generation at a
# freshly-built librayrai.so before the installed package has been updated.
# When set, that directory goes to the front of the runtime search path so
# its rayrai library wins over the installed one.
set(RAYRAI_OVERRIDE_LIB_DIR "" CACHE PATH
    "Optional path to a librayrai.so that overrides the installed copy when running doc image generators")

set(_doc_image_lib_dirs "")
if(RAYRAI_OVERRIDE_LIB_DIR AND EXISTS "${RAYRAI_OVERRIDE_LIB_DIR}")
  list(APPEND _doc_image_lib_dirs "${RAYRAI_OVERRIDE_LIB_DIR}")
endif()
foreach(_dep_target rayrai raisim::raisim)
  if(TARGET ${_dep_target})
    foreach(_prop IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE
            IMPORTED_LOCATION_DEBUG IMPORTED_LOCATION_NOCONFIG
            IMPORTED_LOCATION_RELWITHDEBINFO)
      get_target_property(_loc ${_dep_target} ${_prop})
      if(_loc AND EXISTS "${_loc}")
        get_filename_component(_loc_dir "${_loc}" DIRECTORY)
        list(APPEND _doc_image_lib_dirs "${_loc_dir}")
        foreach(_openusd_dir "${_loc_dir}/openusd" "${_loc_dir}/openusd/lib")
          if(EXISTS "${_openusd_dir}")
            list(APPEND _doc_image_lib_dirs "${_openusd_dir}")
          endif()
        endforeach()
      endif()
    endforeach()
  endif()
endforeach()
list(REMOVE_DUPLICATES _doc_image_lib_dirs)

if(WIN32)
  set(_doc_image_runtime_env_var "PATH")
  set(_doc_image_path_sep ";")
elseif(APPLE)
  set(_doc_image_runtime_env_var "DYLD_LIBRARY_PATH")
  set(_doc_image_path_sep ":")
else()
  set(_doc_image_runtime_env_var "LD_LIBRARY_PATH")
  set(_doc_image_path_sep ":")
endif()
string(REPLACE ";" "${_doc_image_path_sep}" _doc_image_lib_dirs_str "${_doc_image_lib_dirs}")

add_executable(doc_image_shader_warmup
    "${CMAKE_CURRENT_SOURCE_DIR}/doc_image_shader_warmup.cpp")
target_link_libraries(doc_image_shader_warmup PRIVATE rayrai raisim::raisim SDL2::SDL2)
target_compile_features(doc_image_shader_warmup PRIVATE cxx_std_17)
if(MSVC)
  target_compile_definitions(doc_image_shader_warmup PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

set(_doc_image_shader_warmup_stamp
    "${CMAKE_CURRENT_BINARY_DIR}/doc_image_shader_warmup.stamp")
set(_doc_image_shader_warmup_env
    "${_doc_image_runtime_env_var}=${_doc_image_lib_dirs_str}$<$<BOOL:$ENV{${_doc_image_runtime_env_var}}>:${_doc_image_path_sep}$ENV{${_doc_image_runtime_env_var}}>")
add_custom_command(
    OUTPUT "${_doc_image_shader_warmup_stamp}"
    COMMAND ${CMAKE_COMMAND} -E env
            ${_doc_image_shader_warmup_env}
            $<TARGET_FILE:doc_image_shader_warmup>
    COMMAND ${CMAKE_COMMAND} -E touch "${_doc_image_shader_warmup_stamp}"
    DEPENDS doc_image_shader_warmup
            "${CMAKE_CURRENT_SOURCE_DIR}/doc_image_shader_warmup.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/doc_image_common.hpp"
    WORKING_DIRECTORY "${RAISIM_DOC_IMAGE_OUTPUT_DIR}"
    COMMENT "Prewarming rayrai shader cache for doc image generators"
    VERBATIM)

set(_rolling_frames_dir "${CMAKE_CURRENT_BINARY_DIR}/rolling_friction_frames")
set(_strandbeest_frames_dir "${CMAKE_CURRENT_BINARY_DIR}/strandbeest_frames")
set(_previous_stamp "${_doc_image_shader_warmup_stamp}")
list(APPEND _doc_image_outputs "${_doc_image_shader_warmup_stamp}")
foreach(_gen IN LISTS _doc_image_generators)
  set(_gen_source "${CMAKE_CURRENT_SOURCE_DIR}/${_gen}.cpp")
  if(_gen STREQUAL "doc_image_rayrai_blue_wall_scene")
    set(_gen_source "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/src/rayrai/assets/rayrai_blue_wall_scene.cpp")
  elseif(_gen STREQUAL "doc_image_rayrai_nested_glass_showcase")
    set(_gen_source "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/src/rayrai/materials/rayrai_nested_glass.cpp")
  elseif(_gen STREQUAL "doc_image_granular_media_showcase" OR _gen STREQUAL "doc_image_tendon_pulleys"
         OR _gen STREQUAL "doc_image_rayrai_city")
    set(_gen_source "${CMAKE_CURRENT_SOURCE_DIR}/doc_image_home_dynamics.cpp")
  endif()
  add_executable(${_gen} "${_gen_source}")
  string(REGEX REPLACE "^doc_image_" "" _scene_name "${_gen}")
  if(_scene_name IN_LIST _gallery_scenes)
    target_include_directories(${_gen} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/include")
    target_compile_definitions(${_gen} PRIVATE DOC_IMAGE_SCENE="${_scene_name}")
  endif()
  target_link_libraries(${_gen} PRIVATE rayrai raisim::raisim SDL2::SDL2)
  target_compile_features(${_gen} PRIVATE cxx_std_17)
  target_compile_definitions(${_gen} PRIVATE
      DOC_IMAGE_OUTPUT_DIR="${RAISIM_DOC_IMAGE_OUTPUT_DIR}")
  if(MSVC)
    target_compile_definitions(${_gen} PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()

  # Sentinel stamp file so CMake only re-runs the generator when its source
  # changes. The generator itself writes the PNGs; we touch the stamp on
  # success. Stamps are chained so that the binaries run one at a time even
  # under `make -jN` — concurrent SDL/GL initialization on the same display
  # is racy.
  set(_stamp "${CMAKE_CURRENT_BINARY_DIR}/${_gen}.stamp")
  set(_doc_image_prepare_command "")
  set(_doc_image_rolling_frames_dir_arg "")
  set(_doc_image_strandbeest_frames_dir_arg "")
  set(_doc_image_gallery_frames_dir_arg "")
  set(_doc_image_arguments "")
  set(_doc_image_model_dependencies "")
  if(_gen STREQUAL "doc_image_rolling_friction")
    set(_doc_image_rolling_frames_dir_arg "${_rolling_frames_dir}")
    set(_doc_image_prepare_command
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_rolling_frames_dir}")
  elseif(_gen STREQUAL "doc_image_strandbeest")
    set(_doc_image_strandbeest_frames_dir_arg "${_strandbeest_frames_dir}")
    set(_doc_image_prepare_command
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_strandbeest_frames_dir}")
    set(_doc_image_model_dependencies
        "${CMAKE_CURRENT_SOURCE_DIR}/../../rsc/strandbeest/strandbeest.urdf")
  elseif(_scene_name IN_LIST _gallery_scenes)
    set(_doc_image_gallery_frames_dir_arg "${CMAKE_CURRENT_BINARY_DIR}/${_gen}_frames")
    set(_doc_image_prepare_command
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_doc_image_gallery_frames_dir_arg}")
    if(_scene_name STREQUAL "rayrai_blue_wall_scene")
      set(_doc_image_arguments "--record;${_doc_image_gallery_frames_dir_arg}")
      set(_blue_wall "${CMAKE_CURRENT_SOURCE_DIR}/../../rsc/rayrai/blue_wall")
      set(_doc_image_model_dependencies
          "${_blue_wall}/blue_wall.rscene" "${_blue_wall}/blue_wall.gltf"
          "${_blue_wall}/blue_wall.bin" "${_blue_wall}/blue_wall.gltf.rayrai_lights.json"
          "${_blue_wall}/hamburg_canal_1k.hdr")
      # The example finds its scene with exampleRscPath(), i.e. in rsc/ next to its executable.
      set(_blue_wall_rsc "$<TARGET_FILE_DIR:${_gen}>/rsc/rayrai/blue_wall")
      list(APPEND _doc_image_prepare_command
          COMMAND ${CMAKE_COMMAND} -E make_directory "${_blue_wall_rsc}"
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  ${_doc_image_model_dependencies} "${_blue_wall_rsc}")
    elseif(_scene_name STREQUAL "rayrai_city")
      set(_doc_image_model_dependencies
          "${CMAKE_CURRENT_SOURCE_DIR}/../../rsc/city/rayrai_city.rscene"
          "${CMAKE_CURRENT_SOURCE_DIR}/../../rsc/anymal_c/urdf/anymal.urdf")
    elseif(_scene_name STREQUAL "rayrai_nested_glass_showcase")
      set(_doc_image_arguments "--record=${_doc_image_gallery_frames_dir_arg};--frames=60;--width=640;--height=400;--samples=32;--backend=portable")
      set(_doc_image_model_dependencies
          "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/src/rayrai/materials/glass_geometry.hpp")
    else()
      set(_doc_image_model_dependencies
          "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/include/tendon_scenes.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/../../examples/include/tendon_models.hpp"
          "${CMAKE_CURRENT_SOURCE_DIR}/../../rsc/anymal/urdf/anymal.urdf")
    endif()
  endif()
  add_custom_command(
      OUTPUT "${_stamp}"
      ${_doc_image_prepare_command}
      COMMAND ${CMAKE_COMMAND}
              -D "DOC_IMAGE_NAME=${_gen}"
              -D "DOC_IMAGE_EXECUTABLE=$<TARGET_FILE:${_gen}>"
              -D "DOC_IMAGE_OUTPUT_DIR=${RAISIM_DOC_IMAGE_OUTPUT_DIR}"
              -D "DOC_IMAGE_RUNTIME_ENV_VAR=${_doc_image_runtime_env_var}"
              -D "DOC_IMAGE_RUNTIME_ENV_VALUE=${_doc_image_lib_dirs_str}"
              -D "DOC_IMAGE_ROLLING_FRAMES_DIR=${_doc_image_rolling_frames_dir_arg}"
              -D "DOC_IMAGE_STRANDBEEST_FRAMES_DIR=${_doc_image_strandbeest_frames_dir_arg}"
              -D "DOC_IMAGE_GALLERY_FRAMES_DIR=${_doc_image_gallery_frames_dir_arg}"
              -D "DOC_IMAGE_ARGUMENTS=${_doc_image_arguments}"
              -D "DOC_IMAGE_ATTEMPTS=3"
              -P "${_doc_image_runner}"
      COMMAND ${CMAKE_COMMAND} -E touch "${_stamp}"
      DEPENDS ${_gen} "${_gen_source}"
              "${CMAKE_CURRENT_SOURCE_DIR}/doc_image_common.hpp"
              "${_doc_image_runner}"
              ${_doc_image_model_dependencies}
              ${_previous_stamp}
      WORKING_DIRECTORY "${RAISIM_DOC_IMAGE_OUTPUT_DIR}"
      COMMENT "Generating rayrai doc images: ${_gen}"
      VERBATIM)
  list(APPEND _doc_image_outputs "${_stamp}")
  list(APPEND _doc_image_targets "${_gen}")
  set(_previous_stamp "${_stamp}")
endforeach()

# All home-gallery captures use the same size, duration and GIF encoding.
find_program(FFMPEG_EXECUTABLE ffmpeg)
if(FFMPEG_EXECUTABLE)
  foreach(_scene IN LISTS _gallery_scenes)
    set(_gif_name "${_scene}")
    if(_scene STREQUAL "granular_media_showcase")
      set(_gif_name "granular_media_kicking")
    endif()
    set(_gif "${RAISIM_DOC_IMAGE_OUTPUT_DIR}/${_gif_name}.gif")
    set(_palette "palettegen=stats_mode=diff")
    if(_scene STREQUAL "rayrai_city")
      # The camera moves through a detailed street, so frames share almost no pixels;
      # 64 colors keep the GIF near the size of the others (about 7.5 MB instead of 11.7).
      set(_palette "palettegen=stats_mode=diff:max_colors=64")
    endif()
    add_custom_command(
        OUTPUT "${_gif}"
        COMMAND ${FFMPEG_EXECUTABLE} -y -loglevel error -framerate 15
                -i "${CMAKE_CURRENT_BINARY_DIR}/doc_image_${_scene}_frames/frame_%03d.png"
                -filter_complex "[0:v] scale=640:-1:flags=lanczos,split [a][b]; [a] ${_palette} [p]; [b][p] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
                -loop 0 "${_gif}.tmp.gif"
        COMMAND ${CMAKE_COMMAND} -E rename "${_gif}.tmp.gif" "${_gif}"
        DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/doc_image_${_scene}.stamp"
        COMMENT "Compositing home-gallery GIF: ${_scene}"
        VERBATIM)
    list(APPEND _doc_image_outputs "${_gif}")
  endforeach()
endif()

# Composite the recorded frame sequences into GIFs if ffmpeg is available.
# Each rule depends on its generator's stamp so the frames are refreshed first.
find_program(FFMPEG_EXECUTABLE ffmpeg)
set(_rolling_gif "${RAISIM_DOC_IMAGE_OUTPUT_DIR}/rayrai_rolling_spinning_friction.gif")
if(FFMPEG_EXECUTABLE)
  set(_rolling_stamp "${CMAKE_CURRENT_BINARY_DIR}/doc_image_rolling_friction.stamp")
  add_custom_command(
      OUTPUT "${_rolling_gif}"
      # Generate an indexed-palette GIF so the file is small.
      COMMAND ${FFMPEG_EXECUTABLE} -y -loglevel error
              -framerate 25
              -i "${_rolling_frames_dir}/frame_%03d.png"
              -filter_complex "[0:v] palettegen=stats_mode=diff [p]; [0:v][p] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
              -loop 0
              "${_rolling_gif}"
      DEPENDS "${_rolling_stamp}"
      WORKING_DIRECTORY "${RAISIM_DOC_IMAGE_OUTPUT_DIR}"
      COMMENT "Compositing rolling-friction frames into GIF"
      VERBATIM)
  list(APPEND _doc_image_outputs "${_rolling_gif}")
  set(_strandbeest_gif "${RAISIM_DOC_IMAGE_OUTPUT_DIR}/strandbeest_closed_loops.gif")
  add_custom_command(
      OUTPUT "${_strandbeest_gif}"
      COMMAND ${FFMPEG_EXECUTABLE} -y -loglevel error
              -framerate 25
              -i "${_strandbeest_frames_dir}/frame_%03d.png"
              -filter_complex "[0:v] scale=640:-1:flags=lanczos,split [a][b]; [a] palettegen=stats_mode=diff [p]; [b][p] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
              -loop 0
              "${_strandbeest_gif}"
      DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/doc_image_strandbeest.stamp"
      WORKING_DIRECTORY "${RAISIM_DOC_IMAGE_OUTPUT_DIR}"
      COMMENT "Compositing Strandbeest frames into GIF"
      VERBATIM)
  list(APPEND _doc_image_outputs "${_strandbeest_gif}")
else()
  message(STATUS "ffmpeg not found; skipping documentation GIF compositing")
endif()

add_custom_target(rayrai_doc_images
    DEPENDS ${_doc_image_outputs}
    COMMENT "All rayrai doc images regenerated")

# Re-export the list so the parent docs CMakeLists can attach Sphinx to it.
set(RAYRAI_DOC_IMAGE_TARGET rayrai_doc_images PARENT_SCOPE)
set(RAYRAI_DOC_IMAGE_OUTPUTS ${_doc_image_outputs} PARENT_SCOPE)
