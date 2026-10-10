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
after the viewer.

## Directory groups

- `server/basics`: primitive objects, mesh objects,
  compound objects, dynamic object addition, and YCB object loading.
- `server/assets`: model import, mesh preprocessing, cache reuse, `addMesh`
  workflows, and mesh asset export.
- `server/materials`: contact material effects such as restitution and static
  friction.
- `server/terrain`: procedural, image, and dynamic height map examples.
- `server/sensors`: CPU ray casting and ray-scan lidar examples.
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
- `rayrai/materials`: PBR and texture material examples.
- `rayrai/assets`: mesh and asset-processing visualization examples, including
  visual-only glTF assets kept separate from collision meshes.
- `rayrai/runtime`: runtime scene editing using stable object ids, snapshots,
  collision filters, cloning, and removal.
- `rayrai/collision`: collision detection examples such as swept continuous
  collision detection.
- `rayrai/tools`: standalone rayrai tools such as the TCP viewer.
- `rayrai/worlds`: large rayrai worlds such as the instanced forest and its
  `.rscene` version, and the photoreal city and warehouse loaded from
  `.rscene` with an ANYmal C on the street or in an aisle.
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
- `robotiq_gripper_mimic`: a Kinova arm with a wrist-mounted Robotiq 2F-85 uses PD
  joint poses to grasp a cube at one table marker, carry it to another, release it,
  then reset and repeat. Use `--headless --cycles 3` for task checks or
  `--benchmark --cycles 10` for single-threaded timing.
- `tendon_elastic`, `tendon_pulleys`, and `tendon_coupling`: procedural tendon
  simulations streamed to the TCP viewer.
- `rayrai_tendons`: the same tendon scenes with automatic local drawing and
  interactive controls. See the [tendon examples guide](../TENDONS.md).
- `deformable_objects`: cloth, mesh deformables, filled particles, struts,
  compliance, and elastic modulus.
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
  RaiSim Engine `.rscene` file with `raisim::World(path)` and
  `raisin::applyRscene`.
- `rayrai_city`: a photoreal city of modular buildings, streets and parked cars
  loaded from `rsc/city/rayrai_city.rscene`, with an ANYmal C standing on the
  street.
- `rayrai_warehouse`: a photoreal warehouse of pallet racks, forklifts and a
  dock area loaded from `rsc/warehouse/rayrai_warehouse.rscene`, with an
  ANYmal C standing in an aisle.
- `rayrai_runtime_scene_editing`: stable ids, snapshots, collision filters,
  cloning, and removal.
- `rayrai_swept_ccd`: swept CCD settings for a fast falling sphere.

Some targets are guarded by installed RaiSim API availability. If CMake prints a
`Skipping ...` message, install a newer RaiSim/rayrai package and reconfigure.

## Regression and debug tools

Manual diagnostic programs live under `examples/tools/debug` and are built with
`-DRAISIM_DEBUG_TOOLS=ON` when building the examples project. The option is off
by default. See the [debug tools guide](../../docs/sections/DebugTools.rst) for
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
