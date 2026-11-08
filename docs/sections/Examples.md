# Examples

Examples are grouped by the feature they demonstrate, with clips recorded from
the running programs. Choose a topic, then select an example to open its page;
[Run](#run) explains how to build and run them.

<ExampleGallery />

## Overview

The `raisim2Lib` distribution ships prebuilt RaiSim and rayrai libraries
together with C++ example sources. Build the examples with CMake. The install
target does not copy the example executables, and the package does not include
a prebuilt TCP viewer: the `rayrai_tcp_viewer` target builds it from the
example sources.

The examples fall into three kinds:

- **Server examples**, such as `primitive_grid`, build a `raisim::World`
  and publish it through `raisim::RaisimServer`. View them with
  `rayrai_tcp_viewer`.
- **Rayrai examples**, such as `rayrai_pbr_material_grid`, render in process
  with `raisin::RayraiWindow`.
- **Non-visual tools and benchmarks**, such as `model_asset_pipeline` and
  `articulated_system_benchmark`, print results or write files.

Together they cover the RaiSim physics APIs, mesh import and export, OpenUSD
loading, and rayrai rendering and asset inspection. RaisimUnity and RaisimUnreal
are no longer supported; see [Legacy Integrations](./LegacyIntegrations.md).

For optional manual regression and debug programs, see [Regression and Debug Tools](./DebugTools.md).

## Run

After the top-level build from [Build, Test, and Benchmark](./BuildAndTest.md), run examples from the build
tree:

```bash
./build-examples/examples/primitive_grid
./build-examples/examples/rayrai_pbr_material_grid
```

On Windows, the executables are in `build-examples\bin`:

```powershell
.\build-examples\bin\primitive_grid.exe
```

If the runtime loader cannot find the shared libraries, set up the environment
before running examples. On Linux and macOS, `raisim_env.sh` sets
`LD_LIBRARY_PATH` or `DYLD_LIBRARY_PATH`:

```bash
source /path/to/raisim2Lib/raisim_env.sh
```

On Windows, CMake copies the runtime DLLs next to the executables. If a DLL is
still missing, add the package directories to `PATH` from PowerShell or from
the Command Prompt:

```powershell
.\raisim_env.ps1
```

```batch
raisim_env.bat
```

## Visualization modes

There are two visualization paths. See [Visualization](./Visualization.md) for the
full workflow comparison.

### RaisimServer examples

Server examples such as `primitive_grid` create a RaiSim world and publish it
through `raisim::RaisimServer`. They do not open a renderer window themselves.
Start the source-built viewer, then run the server example:

```bash
# Terminal 1
./build-examples/examples/rayrai_tcp_viewer

# Terminal 2
./build-examples/examples/primitive_grid
```

The default server port is `8080` unless the example changes it. Use this
path when you want to inspect the same simulation data that a normal
RaisimServer application publishes.

### Rayrai examples

Examples such as `rayrai_pbr_material_grid`, `rayrai_pbr_texture_maps`, and
`rayrai_visual_asset_support` create or use a `raisin::RayraiWindow`
directly and render in process. They do not need the TCP viewer:

```bash
./build-examples/examples/rayrai_pbr_material_grid
```

Prefer these examples when you need camera images, GPU/offscreen rendering, PBR
materials, glTF visual import, or standalone rayrai feature inspection. USD
visual meshes can also be loaded through `RayraiWindow::addVisualMesh`; see
[OpenUSD Loading](./OpenUSD.md) for the importer scope.

### Non-visual examples

Some examples print output or create files instead of showing a window.
`model_asset_pipeline` writes preprocessed and exported OBJ files to
`raisim_model_asset_pipeline_example` in the system temporary directory
(`/tmp` on Linux). `articulated_system_benchmark` and
`anymal_standing_benchmark` print timing results.

## Example layout

The examples project groups targets by executable behavior:

| Group | Purpose |
| --- | --- |
| Server examples | RaiSim physics, contact, terrain, sensor, robot, XML/MJCF, and OpenUSD scene workflows published through `RaisimServer`. |
| `rayrai_*` | In-process rayrai renderer examples and the `rayrai_tcp_viewer` tool. |
| Benchmark examples | Timing runs for articulated-system, contact, and sleeping-island workloads. |

## Choosing an example

Start with these targets when learning a specific feature:

| Target | Demonstrates |
| --- | --- |
| `primitive_grid` | Basic rigid primitive creation and RaisimServer publishing. |
| `model_asset_pipeline` | Mesh preprocessing, content-hash cache reuse, `addMesh` with processed assets, and OBJ export from a world. |
| `shadow_hand_usd_cube` | Loading an OpenUSD ShadowHand scene through `World(shadow_hand.usd)`, adding a native RaiSim cube, and publishing both through `RaisimServer`. |
| `nvidia_usd_robots` | Loading vetted NVIDIA Isaac Sim robot USD files with `World::addUsdArticulatedSystem` and publishing them through `RaisimServer`. |
| `rayrai_pbr_material_grid` | Inspecting metallic-roughness PBR material behavior under rayrai. |
| `rayrai_visual_asset_support` | Rendering textured URDF visual meshes (OBJ/DAE with image textures) separately from their collision geometry. |
| `rayrai_coacd_mesh_approximation` | Visually comparing original meshes and CoACD convex approximation parts generated through `World::addMesh`. The first run can take a few minutes while CoACD generates and caches the parts. |
| `rayrai_tcp_viewer` | The TCP visualizer used by RaisimServer examples. |
| `tendon_elastic`, `tendon_pulleys`, `tendon_coupling` | Elastic suspensions, routed guides/pulley branches, and joint transmissions. See [Tendon examples and Rayrai](./tendons/Examples.md). |
| `rayrai_tendons` | Automatic tendon rendering with scene selection, pause/step/reset, and live transmission values. See [Tendons](./Tendons.md). |
| `rayrai_inverse_kinematics` | Mouse-driven inverse kinematics: drag and turn a target, and `ArticulatedSystem::solveIK()` drives a simulated Kinova arm with a Robotiq gripper after it, through joint PD targets, to push, grasp and lift cubes. See [Rayrai Example: Inverse Kinematics](./examples/rayrai/rayrai_inverse_kinematics.md). |
| `robotiq_cloth_fold` | A Kinova arm with a Robotiq 2F-85 lays a cloth on the table and folds it in half: deformable cloth in contact with gripper pads, a table and itself, and fold waypoints solved with `ArticulatedSystem::solveIK()` from where the cloth landed. See [Server Example: Robotiq Cloth Fold](./examples/server/robotiq_cloth_fold.md). |
| `heavy_objects_on_cloth` | A 10 kg ball on a cloth, a 5 kg box on two stacked cloths, and a 5 kg blanket over a 1 kg box, with the cloth solved together with the bodies, so each load rests on what carries the cloth. See [Server Example: Heavy Objects on Cloth](./examples/server/heavy_objects_on_cloth.md). |
| `cloth_on_slope` | A box on a cloth on two ramps: below the friction angle both stay, above it they slide together. See [Server Example: Cloth on a Slope](./examples/server/cloth_on_slope.md). |
| `tablecloth_pull` | The tablecloth trick: a sharp pull whips a cloth out from under three dishes, which stay on the table. See [Server Example: Tablecloth Pull](./examples/server/tablecloth_pull.md). |
| `rayrai_deformable_terrain` | An off-road buggy drives across deformable soil, leaving permanent wheel ruts. See [Rayrai Example: Deformable Terrain](./examples/rayrai/rayrai_deformable_terrain.md). |
| `rayrai_motor_operating_region` | Actuators with motor operating regions (EM-MOR), linked from the URDF as actuator files, with live torque-speed plots for the twelve motors of a randomly actuated quadruped rig. See [Rayrai Example: Motor Operating Region](./examples/rayrai/rayrai_motor_operating_region.md). |
| `rayrai_forest` | Dense instanced vegetation on a heightmap with automatic mesh LOD, foliage wind and shadows, and asynchronous loading. See [Rayrai Example: Forest](./examples/rayrai/rayrai_forest.md) for the required rayrai version. |
| `rayrai_forest_from_rscene` | The same forest built from its saved RaiSim Engine `.rscene` file with `raisim::World(path)` and shown by a `RayraiWindow`. Requires RaiSim and rayrai v2.8.2 (not yet released); see [Rayrai Example: Forest From .rscene](./examples/rayrai/rayrai_forest_from_rscene.md) and [.rscene File](./RsceneFile.md). |
| `rayrai_city` | A photoreal city of modular buildings, streets and parked cars loaded from `rsc/city/rayrai_city.rscene`, with an ANYmal C quadruped added in C++. See [Rayrai Example: City](./examples/rayrai/rayrai_city.md). |
| `rayrai_city_from_xml` | The same city and robot from one RaiSim XML world file, which describes the ANYmal C and includes the city's `.rscene` file. See [Rayrai Example: City From XML](./examples/rayrai/rayrai_city_from_xml.md). |
| `rayrai_warehouse` | A photoreal warehouse of pallet racks, forklifts and a dock area loaded from `rsc/warehouse/rayrai_warehouse.rscene`, with an ANYmal C quadruped added in C++. See [Rayrai Example: Warehouse](./examples/rayrai/rayrai_warehouse.md). |
| `rayrai_water_pond` | A pond in a meadow drawn with rayrai water: ripples, the sun glittering on the surface, reflections of the banks and a green tint that hides the floor with depth, with grass, trees, rocks and a wooden jetty on its banks. See [Rayrai Example: Water Pond](./examples/rayrai/rayrai_water_pond.md). |
| `rayrai_water_stream` | A creek running down a hillside into a pool, simulated as shallow water on the GPU: the water parts around mossy rocks, foams where it runs fast and fills the pool, and wooden balls ride the current with buoyancy and drag from `sampleWaterFlow()`. See [Rayrai Example: Water Stream](./examples/rayrai/rayrai_water_stream.md). |
| `rayrai_offroad_buggy` | An off-road buggy with double-wishbone suspension, coilovers and rack-and-pinion steering, all in one URDF closed by pin, equality and mimic constraints, driven with the arrow keys on a dirt race circuit with pits, tyre walls, hay bales, cones and spectator stands; `--validate` checks its physics. See [Rayrai Example: Off-Road Buggy](./examples/rayrai/rayrai_offroad_buggy.md). |

### Targets without a dedicated page

These targets are built with the others but are documented only here or on a
feature page:

| Target | Demonstrates |
| --- | --- |
| `heightmap_mesh_drop` | Twenty monkey meshes with triangle-mesh collision dropped one at a time onto a colored procedural heightmap. |
| `blocky_heightmap_drop` | 900 primitives and convex-hull meshes dropped onto a heightmap made of flat 0.4 m tiles. |
| `large_scale_ray_test` | Repeated ray queries against walls and pillars; see [Ray Test](./RayTest.md). |
| `rayrai_blue_wall_scene` | An authored glTF interior with its light sidecar and HDR environment, described by `rsc/rayrai/blue_wall/blue_wall.rscene`. The example reads that file with its own parser; `raisim::World(path)` and a `RayraiWindow` on that world load it as well (see [.rscene File](./RsceneFile.md)). `--screenshot PNG` renders in a hidden window, saves the image, and exits. |

## Runtime assets

Some targets depend on bundled assets or platform runtime packages:

- OpenUSD loading uses the package's bundled OpenUSD runtime and USD files; keep
  the `openusd` runtime directory and `rsc` assets with the installed
  package.
- Rayrai examples require SDL2/OpenGL and rayrai runtime libraries. See
  [Platform support](./Rayrai.md#platform-support) for the OpenGL requirements and the reduced
  macOS feature set.
- Poly Haven and PBR asset examples require the corresponding assets under
  `rsc`. CMake copies `rsc` next to the executables
  (`build-examples/examples/rsc`, or `build-examples/bin/rsc` on Windows).
- `rayrai_forest` reads its assets from `rsc/forest` in the
  configured checkout and writes regenerable `rayrai_cache_*.lods` files
  beside them. `rayrai_forest_from_rscene` reads
  `rsc/forest/rayrai_forest.rscene` from the `rsc` copy next to the
  executables, and the scene references the same assets.
- `rayrai_city` reads `rsc/city/rayrai_city.rscene` from the `rsc` copy
  next to the executables (`--scene` picks another file). Its street trees
  are the forest's, so it also needs `rsc/forest`. `rayrai_city_from_xml`
  reads `rsc/xmlScripts/rscene/anymal_in_city.xml`, which includes the same
  scene.
- `rayrai_warehouse` reads `rsc/warehouse/rayrai_warehouse.rscene` the same
  way. It reuses the city's sky, asphalt and traffic cones, so it also needs
  `rsc/city`.
- `rayrai_offroad_buggy` reads `rsc/offroad_buggy` (the car, its URDF, the
  terrain and the race venue's props), places the trees, rocks and grass of
  `rsc/forest`, and uses the sky and cars of `rsc/city` and the barrels of
  `rsc/warehouse`.
- `rayrai_water_pond` places the plants and rocks of `rsc/forest`.
- `rayrai_water_stream` places the plants and rocks of `rsc/forest` and gives
  the rocks in the creek colliders from the same models.
- `rayrai_coacd_mesh_approximation` writes `raisim_coacd_*` cache files
  beside the YCB meshes in the build tree's `rsc` copy.

After building, list the example executables from the platform-specific output
directory. Linux and macOS builds place them under `examples`:

```bash
ls build-examples/examples/
```

Windows builds place them under `bin`:

```powershell
Get-ChildItem .\build-examples\bin\*.exe
```
