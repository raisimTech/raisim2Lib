# Example Source Layout

The source tree is grouped by the subsystem or workflow each example demonstrates.
Executable target names are intentionally stable even when source files move.

## Running examples

Build from the `raisim2Lib` root:

```bash
cmake -S . -B build-examples \
  -DRAISIM_EXAMPLE=ON \
  -DRAISIM_CHECK_FOR_UPDATES=OFF
cmake --build build-examples -j
cd build-examples/examples
```

On Windows with Visual Studio or MSBuild, select the release configuration in
the build command and run examples from `build-examples\bin`:

```powershell
cmake --build build-examples --config Release --parallel
cd build-examples\bin
```

If a source-built viewer asks for `SDL2d.dll`, it was built from a debug-flavored
configuration; rebuild with `--config Release` to use the release SDL2 runtime.

Most `server/*` examples publish to `raisim::RaisimServer`. Build the
`rayrai_tcp_viewer` target and start the resulting build-tree executable in
another terminal to visualize them.
Most `rayrai/*` examples create an in-process rayrai window and do not need the
TCP viewer.

Examples find their models, maps and textures with
`exampleRscPath(argv[0], "relative/path")` from
`examples/include/example_resources.hpp`. It looks for the `rsc` folder CMake
copies next to the executables, so examples run from any working directory
without arguments.

rayrai examples that load meshes asynchronously show a loading bar with
`RayraiLoadingProgress` from `examples/include/rayrai_loading_progress.hpp`:
pass it `RayraiWindow::pendingAsyncMeshLoadCount()` every frame and draw it
after the viewer. The fill uses the lime, coral, magenta, and violet gradient
from `rsc/docs/image/logo.png`, with deep-purple text on a white band.

## Directory groups

- `server/basics`: primitive objects, mesh objects,
  compound objects, dynamic object addition, and YCB object loading.
- `server/assets`: model import, mesh preprocessing, cache reuse, `addMesh`
  workflows, and mesh asset export.
- `server/materials`: contact material effects such as restitution and static
  friction.
- `server/terrain`: procedural, image, and dynamic height map examples.
- `server/sensors`: CPU ray casting, ray-scan lidar, and a headless IMU sensor model example.
- `server/robots`: articulated robot examples and robot control workflows.
- `server/dynamics`: constraints, moving platforms, springs, and rigid-body
  dynamics demos.
- `server/visualization`: server-driven visual output and synchronous update
  flow.
- `server/performance`: island sleeping benchmark.
- `server/deformable`: cloth, surface-mesh deformables, filled deformables,
  internal struts, compliance, and Young's modulus examples.
- `server/mjcf`: MJCF world loading and articulated Gymnasium models.
- `benchmark`: timing-oriented examples for common RaiSim workloads.
- `rayrai/getting_started`: minimal and complete rayrai entry points.
- `rayrai/sensors`: rayrai RGB camera, depth camera, lidar point cloud, and
  marker examples.
- `rayrai/visuals`: custom visuals, instancing, and point cloud examples.
- `rayrai/dynamics`: in-process rigid-body dynamics visualizations.
- `rayrai/robots`: interactive robot examples, such as a manipulator driven by
  inverse kinematics.
- `rayrai/materials`: PBR and texture material examples.
- `rayrai/assets`: mesh and asset-processing visualization examples, including
  visual-only glTF assets kept separate from collision meshes.
- `rayrai/runtime`: runtime scene editing using stable object ids, snapshots,
  collision filters, cloning, and removal.
- `rayrai/collision`: collision detection examples such as swept continuous
  collision detection.
- `rayrai/tools`: standalone rayrai tools such as the TCP viewer.
- `rayrai/worlds`: large rayrai worlds such as the instanced forest and its
  `.rscene` version, the photoreal city and warehouse loaded from
  `.rscene` with an ANYmal C on the street or in an aisle, and the off-road
  buggy with its simulated suspension.
- `worlds`: packaged manipulator scene.
- `xml`: XML world loading and templated XML world examples.

## Useful starting targets

- `primitive_grid`: basic server-side simulation and visualization.
- `rayrai_basic_scene`: minimal in-process rayrai rendering.
- `rayrai_complete_showcase`: broad rayrai feature overview.
- `rayrai_depth_camera`: rayrai depth capture plus CPU depth-camera comparison.
- `rayrai_rgb_camera`: in-process rayrai RGB capture.
- `rayrai_rolling_spinning_friction`: rolling and spinning friction on a grid
  of spheres and cylinders.
- `rayrai_motor_operating_region`: torque-speed plots of twelve randomly
  actuated motors whose operating regions (bus voltage and peak torque) come
  from actuator files linked in the URDF, including KAIST-Hound-style coupled
  hip/knee actuators.
- `rayrai_inverse_kinematics`: drag a target with the mouse and a simulated Kinova
  arm with a Robotiq 2F-85 follows it: `ArticulatedSystem::solveIK()` turns the
  target pose into joint PD targets every frame, and the arm can push, grasp and
  lift the cubes on its workbench.
- `robotiq_gripper_mimic`: a Kinova arm with a wrist-mounted Robotiq 2F-85 uses PD
  joint poses to grasp a cube at one table marker, carry it to another, release it,
  then reset and repeat.
- `tendon_elastic`, `tendon_pulleys`, and `tendon_coupling`: procedural tendon
  simulations streamed to the TCP viewer.
- `rayrai_tendons`: the same tendon scenes with automatic local drawing and
  interactive controls. See the [tendon examples guide](../TENDONS.md).
- `deformable_objects`: cloth, mesh deformables, filled particles, struts,
  compliance, and elastic modulus.
- `robotiq_cloth_fold`: a Kinova arm with a wrist-mounted Robotiq 2F-85 holds a
  cloth by one corner, lays half of it on the table, folds the other half over
  it, then unfolds and lifts it, and repeats.
- `heavy_objects_on_cloth`: a 10 kg ball on a cloth, a 5 kg box on two stacked
  cloths, and a 5 kg blanket over a 1 kg box, with the cloth solved together with
  the bodies (`World::setMonolithicDeformables()`).
- `cloth_on_slope`: a box on a cloth on two ramps; below the friction angle both
  stay, above it they slide together.
- `tablecloth_pull`: a sharp pull whips a cloth out from under three dishes.
- `imu_sensor_model`: ideal versus modeled IMU CSV at matching acquisition times;
  includes seeded errors, bandwidth, quantization, and latency. It takes no
  arguments and prints the ideal reference alongside modeled samples. No viewer is needed.
- `model_asset_pipeline`: mesh preprocessing and OBJ asset export.
- `mjcf_gymnasium_hopper`: load and actuate Gymnasium's Hopper MJCF model.
- `mjcf_gymnasium_walker2d`: load and actuate Gymnasium's Walker2d MJCF model.
- `mjcf_gymnasium_humanoid`: load Gymnasium's Humanoid MJCF model and drop it
  from a raised arbitrary configuration.
- `anymal_standing_benchmark`: run the native ANYmal PD-standing benchmark and
  report simulation throughput plus average contacts.
- `articulated_system_benchmark`: run standalone timing scenes for ANYmal,
  Atlas, and chain articulated systems.
- `dynamic_heightmap`: animate a heightmap and color map through RaisimServer.
- `blocky_heightmap_drop`: drop 900 mixed bodies (boxes, spheres, capsules, cylinders,
  and monkey meshes) onto a 250x250-sample, 20m x 20m
  height map whose 5x5 sample blocks share one height drawn uniformly from -0.15m to 0.15m.
- `rayrai_coacd_mesh_approximation`: original mesh versus CoACD convex approximation mesh
  collision parts through `World::addMesh`.
- `rayrai_visual_asset_support`: inspect realistic textured URDF assets while
  keeping visual and collision geometry separate.
- `rayrai_blue_wall_scene`: explore the furnished Poly Haven Blue Wall scene
  with imported lights and HDR environment lighting.
- `rayrai_forest_from_rscene`: build the `rayrai_forest` world from its saved
  RaiSim Engine `.rscene` file with `raisim::World(path)`, shown by a
  `raisin::RayraiWindow` on that world.
- `rayrai_city`: a photoreal city of modular buildings, streets and parked cars
  loaded from `rsc/city/rayrai_city.rscene`, with an ANYmal C standing on the
  street.
- `rayrai_city_from_xml`: the same city and ANYmal C from one RaiSim XML world
  file (`rsc/xmlScripts/rscene/anymal_in_city.xml`) that describes the robot and
  includes the city's `.rscene` file.
- `rayrai_warehouse`: a photoreal warehouse of pallet racks, forklifts and a
  dock area loaded from `rsc/warehouse/rayrai_warehouse.rscene`, with an
  ANYmal C standing in an aisle.
- `rayrai_deformable_terrain`: drive the full off-road buggy with
  double-wishbone suspension over a deformable mesh soil bed, leaving permanent
  wheel tracks. Supports `--headless`,
  `--steps=N`, and `--export=DIR`.
- `rayrai_offroad_buggy`: an off-road buggy with double-wishbone suspension,
  coilovers and rack-and-pinion steering, all in one URDF closed by pin,
  equality and mimic constraints, driven with the arrow keys on a dirt race
  circuit with a pit lane, tyre walls, hay bales and cones it can knock over,
  spectator stands, trees and boulders. The springs are coloured by their
  compression; `--validate` runs physics checks without a window.
- `rayrai_water_pond`: a pond in a meadow drawn with rayrai water. Grass, trees,
  rocks and a wooden jetty stand on its banks, light ripples move over it and the
  sun glitters on its surface. Uses the Poly Haven plants and rocks in
  `rsc/forest`.
- `rayrai_water_stream`: a creek running down a hillside into a pool, simulated
  as shallow water by rayrai. The water parts around mossy rocks and foams where
  it runs fast, and wooden balls float down it with buoyancy and drag from
  `sampleWaterFlow()`. Uses the Poly Haven plants and rocks in `rsc/forest`.
- `rayrai_runtime_scene_editing`: stable ids, snapshots, collision filters,
  cloning, and removal.
- `rayrai_swept_ccd`: swept CCD settings for a fast falling sphere.

Some targets are guarded by installed RaiSim API availability. If CMake prints a
`Skipping ...` message, install a newer RaiSim/rayrai package and reconfigure.

## Physics snapshots in rayrai examples

15 of the 31 rayrai demos use `PhysicsSnapshotCapture` and
`RayraiWindow::applyPhysicsSnapshot`:

- `rayrai_basic_scene`, `rayrai_rgb_camera`, `rayrai_depth_camera`
- `rayrai_custom_visuals`, `rayrai_instancing_grid`, `rayrai_pointcloud_animation`
- `rayrai_rolling_spinning_friction`, `rayrai_tendons`
- `rayrai_pbr_material_grid`, `rayrai_quality_lighting`
- `rayrai_coacd_mesh_approximation`, `rayrai_visual_asset_support`
- `rayrai_runtime_scene_editing`, `rayrai_swept_ccd`, `rayrai_inverse_kinematics`

`capture()` returns `std::shared_ptr<const PhysicsSnapshot>`; pass that owner directly
into `applyPhysicsSnapshot()` and retain it while another thread consumes the frame:

```cpp
const auto frame = snapshotCapture.capture(*world);  // physics thread, World protected
viewer->applyPhysicsSnapshot(frame);                // render thread, GL context current
```

Frames store contiguous object state, shared persistent render descriptions and
one pose per articulated link. Geometry and appearance have separate revisions.
Released frame buffers are reused; a frame held by any camera stays unchanged.
Object membership is reconciled only when its topology token changes. Deformable
vertices share their unchanged indices. Heightmap patches carry vertex rows and a
base revision; a renderer that skips that base uses the complete latest vertices.

These demos bind the renderer to its own empty world and copy the source physics
state after integration. The RGB/depth demos also copy calibration and sensor
poses, use the Camera-only render overloads, and read pixels into independent
buffers. Depth points are reconstructed from copied rays and the captured pose.
The remaining 14 demos use the direct-world API. Server demos, benchmarks and
the TCP viewer tool are outside this count.

Capture checks deformable vertex positions, visual heightmap samples and mutable
articulated visuals before reusing object state. It leaves shared heightmap dirty
flags untouched. Each changed snapshot carries a render-state revision independent
of the source object's revision, so geometry and appearance edits invalidate render
caches. Switching an existing scene to snapshots replaces its scene-owned visuals
and preserves unrelated visual overlays.

The loops are sequential to keep each example focused. For separate physics and
camera threads, protect integration, snapshot capture and sensor pose copying
with the physics lock. Pass the immutable snapshot and copied pose to the render
thread, release the physics lock, then protect snapshot application and camera
rendering with the window lock. All GL work stays on its owning thread. Capture
at the sensor's scheduled simulation time; do not wait for rendering while
holding the physics lock. Render cameras use copied poses, with no live
`targetObject`.

The snapshot API is included in the normal rayrai library build. These example
targets are always created, with no feature switch or availability check. Build
them against the updated rayrai package containing `rayrai/PhysicsSnapshot.hpp`.

## Regression and debug tools

Manual diagnostic programs live under `examples/tools/debug` and are built with
`-DRAISIM_DEBUG_TOOLS=ON` when building the examples project. The option is off
by default. See the [debug tools guide](../../docs/sections/DebugTools.md) for
build and run instructions.

- `object_lifecycle_stress`: repeatedly create, simulate, and remove primitives
  to inspect object lifecycle and memory handling.
- `rayrai_heightmap_replacement`: inspect front and rear depth images while
  replacing terrain and primitives to detect stale deleted geometry.

## Blue Wall RayRai scene

![Blue Wall scene from the example's initial camera](../../rsc/docs/image/rayrai_blue_wall_scene.png)

The initial camera faces the blue wall, painting, dresser, bookshelf, and chair.
The example reads the packaged `blue_wall.rscene` for its camera, environment,
and mesh path. Its lossless glTF payload, light metadata, and HDR map are under
`rsc/rayrai/blue_wall`. From the `raisim2Lib` root, build and run:

```sh
cmake --build build-examples --target rayrai_blue_wall_scene -j12
./build-examples/examples/rayrai_blue_wall_scene
```

Use `--screenshot output.png` to save this view. The adjacent Raisim test build
registers the asset check; `RAISIM_EXAMPLE_GPU_TESTS=ON` there also registers
camera and image-quality checks against the original GLB render;
`--benchmark-frames N` measures N rendered frames.

`rayrai/sensors/rayrai_sensor_noise.cpp` saves clean/noisy RGB and depth comparisons offscreen with seeded read, shot, row and depth noise, quantization, and dropout. Run `rayrai_sensor_noise` without arguments in the desired output directory, using a current SDK exposing the sensor-noise API.
