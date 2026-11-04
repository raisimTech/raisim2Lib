# Examples of programs built on the RaiSim WebAssembly SDK. The same sources build for the desktop
# against the native RaiSim and Rayrai packages; see README.md.
cmake_minimum_required(VERSION 3.20)
project(raisim_wasm_examples CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(RAISIM2LIB_RSC "${CMAKE_CURRENT_SOURCE_DIR}/../rsc")
if(EMSCRIPTEN)
    set(RAISIM_WASM_SDK "${CMAKE_CURRENT_SOURCE_DIR}/sdk" CACHE PATH "The unpacked RaiSim WebAssembly SDK")
    find_package(rayrai CONFIG REQUIRED PATHS "${RAISIM_WASM_SDK}" NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
else()
    find_package(rayrai CONFIG REQUIRED)
    if(NOT TARGET rayrai::rayrai)
        add_library(rayrai::rayrai ALIAS rayrai)
    endif()
endif()

# Physics only: a shell blows a crate stack apart; the program prints where the crates land. In
# WebAssembly it runs under Node.js.
add_executable(crate_blast src/crate_blast.cpp)
target_link_libraries(crate_blast PRIVATE raisim::raisim)
if(EMSCRIPTEN)
    target_link_options(crate_blast PRIVATE -sENVIRONMENT=node -sEXIT_RUNTIME=1 -sALLOW_MEMORY_GROWTH=1)
    raisim_web_add_license(crate_blast)
endif()

# A tank survival game drawn by Rayrai. It reads its models, textures and font from raisim2Lib/rsc. On the
# web page they are files next to the page (tank_survival-assets/, about 110 MB without Rayrai's local
# mesh caches) that it downloads into /assets before the game starts; the page is tank_survival.html,
# and TANK_SURVIVAL_URL is the address it is published at, for the card a shared link shows. The web
# page plants conifers only: the broadleaf tree's mesh alone is a 95 MB download.
set(tank_survival_assets tank_survival fonts/roboto/Roboto-SemiBold.ttf offroad_buggy/terrain/textures
    forest/pine_sapling_small forest/fir_sapling forest/rock_moss_set_01 forest/grass_medium_01
    forest/grass_medium_02 forest/grass_bermuda_01 forest/dandelion_01 forest/fern_02)
foreach(rock RANGE 5)
    list(APPEND tank_survival_assets forest/rock_moss_${rock})
endforeach()
set(TANK_SURVIVAL_URL "https://raisim.com/games/tank-survival/" CACHE STRING "Address the tank survival page is published at")
add_executable(tank_survival src/tank_survival/main.cpp)
if(EMSCRIPTEN)
    set(mounts "")
    foreach(asset IN LISTS tank_survival_assets)
        list(APPEND mounts "${RAISIM2LIB_RSC}/${asset}@/assets/${asset}")
    endforeach()
    rayrai_add_web_app(tank_survival TITLE "RaiSim Tank Survival"
        DESCRIPTION "Hold out against waves of armored vehicles in a tank battle that runs in your browser. Every shell, blast and wading tank is simulated by RaiSim and drawn by Rayrai."
        URL "${TANK_SURVIVAL_URL}" IMAGE "${RAISIM2LIB_RSC}/docs/image/tank_survival_card.jpg"
        ASSETS ${mounts} EXCLUDE "rayrai_cache_*")
    target_compile_definitions(tank_survival PRIVATE TANK_SURVIVAL_ASSETS="/assets" TANK_SURVIVAL_BROADLEAF_TREES=0)
else()
    target_link_libraries(tank_survival PRIVATE rayrai::rayrai)
    target_compile_definitions(tank_survival PRIVATE TANK_SURVIVAL_ASSETS="${RAISIM2LIB_RSC}")
endif()

# Checks of the game's logic that need no window (desktop only).
if(NOT EMSCRIPTEN)
    enable_testing()
    add_executable(tank_survival_test tests/tank_survival_test.cpp)
    target_include_directories(tank_survival_test PRIVATE src/tank_survival)
    target_link_libraries(tank_survival_test PRIVATE rayrai::rayrai)
    add_test(NAME tank_survival_test COMMAND tank_survival_test)
endif()
