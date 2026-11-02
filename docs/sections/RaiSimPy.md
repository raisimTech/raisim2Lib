# RaisimPy

RaisimPy is the nanobind-based Python wrapper built from the public
`raisim2Lib/raisimPy` sources. It follows the C++ RaiSim API closely while
using NumPy arrays for vectors and matrices. The wrapper sources are MIT
licensed; the RaiSim binary they use remains subject to the RaiSim license.
NumPy is required for the Rayrai API, examples, and runtime tests.

## Build

Build the wrapper with the same Python interpreter that will import it:

```bash
cd /path/to/raisim2Lib
source ./raisim_env.sh
cmake -S . -B build-py \
  -DCMAKE_BUILD_TYPE=Release \
  -DRAISIM_EXAMPLE=OFF \
  -DRAISIM_PY=ON \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build build-py --parallel 12
```

Use a fresh build directory when upgrading from an older package, or update
the cached `RAISIM_VERSION` to the version required by the root CMake file.
The extension exposes the linked package version as `raisimpy.__version__`.
Force/tactile sensor bindings are enabled when that package contains their
headers and sensor enum values; check `raisimpy.hasWrenchSensors` before using
them. They are unavailable in the 2.12.0 package.

The wrapper also supports a standalone CMake build with
`cmake -S raisimPy -B build-python -DCMAKE_PREFIX_PATH="$PWD/raisim"`.
Use the same Python executable option as above.

On Linux and macOS, the extension is emitted under `build-py/raisimPy`. Its
RPATH already locates the RaiSim and Rayrai libraries, so it only has to be on
`PYTHONPATH`, which `raisim_env.sh` sets up. In a shell where it was sourced,
run an example directly from the build tree with:

```bash
python raisimPy/examples/robots.py
```

Set `RAISIMPY_BUILD_DIR` before sourcing to use a different build directory.
To import `raisimpy` with no environment setup at all, install it into the
site-packages of the Python selected at configure time, and repeat this after
each rebuild:

```bash
cmake --install build-py/raisimPy
```

On Windows, select the intended configuration explicitly with
`cmake --build build-py --config Release --parallel 12`. `raisim_env.ps1` and
`raisim_env.bat` add `build-py\raisimPy` and its `Release` folder to
`PYTHONPATH`. Python 3.8+ no longer searches `PATH` for an extension's DLLs, so
prefer `cmake --install build-py/raisimPy --config Release`, which also copies
the RaiSim DLLs next to the installed module. Use the same Debug/Release
configuration as the imported RaiSim binary selected by CMake.

## Activation And Minimal Use

`World.setActivationKey` is the preferred spelling. `World.setLicenseFile`
remains as a compatibility alias.
The examples accept `--activation-key /path/to/activation.raisim`. Otherwise
they use `RAISIM_ACTIVATION_KEY`, `rsc/activation.raisim`, or
`~/.raisim/activation.raisim`, in that order.

```python
import time
import raisimpy as raisim

raisim.World.setActivationKey("/absolute/path/to/activation.raisim")

world = raisim.World()
world.setTimeStep(0.001)
world.addGround()
ball = world.addSphere(0.2, 1.0)
ball.setPosition(0.0, 0.0, 1.0)

server = raisim.RaisimServer(world)
server.launchServer(8080)
for _ in range(10_000):
    server.integrateWorldThreadSafe()
    time.sleep(world.getTimeStep())  # run at roughly real time
server.killServer()
```

Start `build-examples/examples/rayrai_tcp_viewer` in another terminal to view
this server-based example.

## In-process Rayrai

The default build includes the nanobind renderer extension `raisimpy_rayrai`.
Use it through `raisimpy.rayrai`; `raisimpy.RayraiWindow` and
`raisimpy.RayraiViewer` are aliases. `raisimpy.hasRayrai` reports whether the
backend was enabled. The physics extension loads without importing OpenGL;
renderer libraries load when the Rayrai API is first accessed. To build only
physics, configure with `-DRAISIM_PY_RAYRAI=OFF`.

```python
import raisimpy as raisim
from raisimpy import rayrai

world = raisim.World()
world.addGround()
world.addSphere(0.3, 1.0).setPosition(0.0, 0.0, 2.0)

with rayrai.RayraiWindow(world, 800, 600, visible=True) as viewer:
    viewer.setRenderQualityPreset(rayrai.RenderQualityPreset.Balanced)
    for step in range(1000):
        viewer.integrateWorldThreadSafe()
        if step % 16 == 0 and not viewer.render():
            break
    rgba = viewer.readRgba()  # owned uint8 array, (height, width, 4)
```

No separate viewer process or TCP server is needed. Use `visible=False` for a
hidden OpenGL context. A hidden context still needs a working graphics driver.
On Linux with Mesa, `SDL_VIDEODRIVER=offscreen` can provide a display-free
context. Context creation failures raise `RuntimeError`.

The bindings cover the application-facing public `RayraiWindow` interface,
including its nested records and enums, and the related camera, visual,
material, mesh, lighting, environment, and settings interfaces:

| Area | Entry points |
| --- | --- |
| Window/context | `RayraiWindow`, `render`, `pollEvents`, `resize`, `setVisible`, `close`, context manager, `createOffscreenGlContext` |
| Cameras | `getCamera`, `Camera(viewer, ...)`, `renderWithExternalCamera`, projection/fisheye lens settings, picking, RGB/depth sensor overloads |
| Capture | `readRgba`, `saveScreenshot`, `readSceneColorBgra`, `readLinearDepth`, async readback, supersampling, debug-pass captures and GPU timings |
| Visuals | Primitive/custom meshes, `makeVisualMesh` builder, instancing, point clouds, coordinate frames, camera frustums, visibility and LOD controls |
| Materials and lights | `Material`, texture/filter/sampler controls, `OpenGLMesh`, `Shader`, `getLight`, `AdditionalLight`, PBR environments |
| Environment | Weather transitions/callbacks, fog, clouds, water bodies/flow/waves, reflection probes, irradiance and sky visibility grids |
| Performance/import | Render presets/settings, diagnostics, caches, async model import, mesh preprocessing, scene/sidecar import/export, physics snapshots |

C++ method and property names are preserved; Python keywords gain a trailing
underscore (`lambda_`, `FoliageType.None_`). Nested types are available on their
owning class, such as `RayraiWindow.RenderOverrides` and `Material.AlphaMode`.
Window settings types also have module-level aliases. Checked-in nanobind type
stubs are under `raisimPy/typings`.

### Cameras and NumPy capture

```python
properties = raisim.RGBCameraProperties()
properties.width, properties.height = 320, 240
properties.hFOV = 1.2
camera = rayrai.Camera(viewer, properties)
camera.position = [4.0, 3.0, 3.0]
camera.yaw, camera.pitch = -135.0, -25.0
camera.update(False)
viewer.renderWithExternalCamera(camera, rayrai.RenderOverrides())
rgba = camera.readRgba()
viewer.renderDepthPlaneDistance(camera)
depth = camera.readLinearDepth()  # owned float32 array, (240, 320)
```

`getRawImage(sensor)` fills a RaiSim RGB/depth sensor through the native sensor
setter, including its configured noise. Passing `SensorStorageMode.CUSTOM_BUFFER`
returns an owned BGRA/depth NumPy array instead of accepting a raw pointer.
`readSceneColorRgbaAsync` and `readLinearDepthAsync` return `None` while a
readback is pending, then an owned array. Color reads use RGBA except the
explicit BGRA and RGB sensor APIs. `saveScreenshot` writes PNG using the SDK image library.
Capture-record `rgba` fields remain flat byte sequences; reshape with
`numpy.asarray(capture.rgba, dtype=numpy.uint8).reshape(capture.height, capture.width, 4)`.
`syncRaisimCameraPose(sensor, camera)` supports RGB and depth sensors and locks
the camera viewer's World while copying the sensor pose. Use a sensor from that World.
It sets the camera direction vectors directly; do not call `camera.update()`
between synchronization and rendering, since that recalculates them from yaw/pitch.

Vectors accept sequences or NumPy arrays. GLM matrix arrays use `(rows, columns)`;
quaternions use `(w, x, y, z)`. Array getters copy their data. For collections
such as point-cloud positions, assign the edited collection back, then call
`updatePointBuffer`. Settings/material getters copy values; edit and pass the
value to the corresponding setter. This also applies to `mutableMaterialOverride`:
use `setMaterialOverride` to commit the edited material.

Output overloads that would otherwise have the same Python argument list get
explicit names: `assimp.loadModelWithSourceFiles(path)` returns `(meshes, files)`;
`assets.getMeshesWithCanonicalKey(path)` returns `(meshes, key)`. Their ordinary
`loadModel`/`getMeshes` forms return just the meshes.
`importVisualScene(name, path, options, report=None)` takes the output report
explicitly; options do not store a raw report pointer. Model source-file `time`
properties are integer Unix timestamps in nanoseconds. Texture-slot metadata
names are read-only.

Native scalar/vector output references become return values. For example,
`visual.approximateBounds()` returns `(valid, center, radius)` and
`warmupNextShader()` returns `(elapsed, done)`. Stream diagnostics return a tuple
containing the output string. Native object/report output parameters remain
writable Python records. Native raw SDL handles are replaced with managed
viewers; `makeCurrent` and `detachOffscreenGlContext` manage their context.

### Ownership and threads

A viewer retains its World and owns its SDL/OpenGL/ImGui contexts. Renderer-owned
Python objects retain the viewer state. `close()` is idempotent and releases GPU
resources with their context current; retained handles raise `RuntimeError`
after close. Captured images and copied settings remain usable. Multiple
viewers can coexist, and each call activates the correct context. Objects from
different viewers cannot be mixed.
`RenderOverrides` retains its assigned shader handles and validates their viewer
at render time. Target visuals retain shared ownership, including standalone
visuals supplied to `setTargetVisual`; a retained returned visual remains usable
after removal from the viewer. Internal frame-cache records with native pointers
are not Python data types.

Create, use, and close each viewer on its creating thread. Physics snapshots
can be captured separately and supplied through `applyPhysicsSnapshot`; protect
the World while capturing it. Rendering and `integrateWorldThreadSafe` lock the
World. Rendering retains the GIL; waiting to acquire a World lock releases it, so
physics/control threads can unlock the World. Python render/control calls are serialized;
this API does not promise parallel Python rendering. All viewers in a process
must use the same native `ThreadingMode` (the Python default is `MultiThread`).
Visible viewers must be created on Python's main thread; on macOS this applies
to hidden viewers too. Hidden worker-thread contexts on other platforms depend
on SDL and the graphics driver supporting that usage.
Use `lockVisualizationServerMutex` / `unlockVisualizationServerMutex` when
explicitly protecting viewer operations; these track locks across viewers
sharing a World. Do not render while holding a direct `World.lock()`.
Closing releases a viewer's explicit World lock; one matching later unlock
(for example in `finally`) is accepted.
Native fatal diagnostics on the calling thread become Python exceptions.
For deterministic shutdown, use `with` or call `close` explicitly. Weather
callbacks receive copied diagnostics; defer rendering, new viewers, and
World/topology changes until the callback returns. Nested render calls raise
`RuntimeError` instead of waiting for a World lock held by the outer render. To stop from a callback, call `requestClose()` and close the
viewer after the render call returns.
Callbacks live in the Python viewer's GC-visible dictionary, so capturing the
viewer does not create an uncollectable native reference cycle. They stop firing
if that Python viewer is collected while child handles retain its native state.

### Maintaining API coverage

`raisimPy/rayrai/api.json` records the SDK declarations. `coverage.json` records
each generated/manual binding and explains exclusions: C++ ownership tokens,
assignment/copy operations, and private acceleration/row-allocation interfaces.
Template helpers are listed as manual concrete instantiations or exclusions.
Raw GPU resources always pass through the managed context layer. The generator
produces the checked-in C++ files, so normal builds need no Clang installation:

```bash
python raisimPy/tools/generate_rayrai.py --check
# After changing SDK headers, with libclang installed:
python raisimPy/tools/extract_rayrai_api.py --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15
python raisimPy/tools/generate_rayrai.py
python raisimPy/tools/generate_rayrai_stubs.py --module-dir build-py/raisimPy
```

The tests compare all exposed names with this inventory and check SDK header
hashes. A CMake link check handles SDKs whose headers declare layer-selective
cache invalidation but whose binary exports only full invalidation; both Python
call forms remain available, with full invalidation as the fallback.
Maintainers can build with `-DRAISIMPY_RAYRAI_TEST_FAULTS=ON` and run the tests
with `RAISIMPY_TEST_FRAME_FAULTS=1` to verify recovery from an exception during
a visible ImGui frame. Fault injection is disabled in normal builds.

## Examples And API Coverage

The examples under `raisimPy/examples` cover articulated robots, joint
springs, heightmaps, constraints, ray-cast scans, charts, point clouds,
instanced visuals, dynamic visual meshes, and server-side visual objects.
Every example accepts `--steps N`, `--headless`, `--no-sleep`, and `--port`.
Headless runs do not open sockets or sleep. A port of zero selects a free
loopback port. Examples stop their server on completion and on Python exit.

```bash
PYTHONPATH="$PWD/build-py/raisimPy" \
  python raisimPy/examples/robots.py --headless --steps 200
```

The current bindings also expose:

| API | Python entry points |
| --- | --- |
| Inverse kinematics | `ArticulatedSystem.solveIK`, `IKOptions`, `IKResult`; frame names or indices, position and orientation targets |
| DC actuators and tracks | `DcMotorParameters`, `ActuatorDefinition`, torque commands/bounds/states, bus voltage, `getTracks` and belt speed |
| Deformables | `World.addDeformableCloth`, `addDeformableVolume`, `addDeformableObject`; nested material/build options, pins, transforms and particle arrays |
| Native granular systems | `World.addGranularParticles`, `addGranularBox`; material/box options, emission, removal, boundaries, state files and step statistics |
| Deformable soil | `World.addDeformableTerrain`; nested material/state types, pressure, reset and contact statistics |
| Replay and reset | `World.captureCheckpoint`, `restoreCheckpoint`, single-body snapshots and primitive cloning |
| World controls | Sleeping/waking, contact/broadphase settings, swept CCD, collision counters, stable object IDs, material and collision-filter edits |
| Import | `World.addUsdArticulatedSystem`, URDF modules, MJCF loading and mesh asset export |
| Server controls | Pause/resume/single-step, actual bound port, loopback binding and scene-file sharing |

`actuatedSlider.py` combines IK and a torque-limited actuator. `particles.py`
shows a pinned cloth and native grains; `deformableTerrain.py` shows soil
under a rigid load. Deformable soil cannot share a world with deformable or
granular particle objects in the current engine; the Python factories reject
that combination with `ValueError`. `checkpoints.py` replays a
bouncing ball from a captured checkpoint.

`restoreCheckpoint(checkpoint)` returns a boolean; use
`raiseOnError=True` to raise `ValueError` with the rejection reason. Capture
and restore between complete steps. Checkpoints belong to their original
world, and topology changes invalidate them. `captureSingleBodySnapshot`
returns a snapshot directly; restore accepts an object or stable object ID.

Python method names normally match C++. The physics bindings are under
`raisimPy/src`; renderer bindings and their inventory are under `raisimPy/rayrai`.

NumPy vectors use shape `(N,)`, particle positions use `(N, 3)`, and matrices
use `(rows, columns)`. The bindings use contiguous float64 arrays and reject
incorrect vector/matrix shapes. Particle-state arrays and the new material,
actuator, and world-settings getters return copies; use the corresponding
setters to modify the engine.
World object getters borrow objects and retain the world, and a server
retains its world. Removing an object invalidates wrappers to that object;
use stable IDs and `isObjectAlive` when objects may be removed.

### City, warehouse, forest, and buggy

`city.py` and `warehouse.py` load the same `.rscene` assets and standing
ANYmal C configurations as their native examples. `forest.py` loads the
forest terrain, tree/rock colliders, and saved visual scene. Each accepts
`--scene /path/to/scene.rscene` to select another scene file.

```bash
python raisimPy/examples/city.py
python raisimPy/examples/warehouse.py
python raisimPy/examples/forest.py
```

These four examples open an in-process Rayrai window by default when the
backend is built. Use `--renderer tcp` and run
`build-examples/examples/rayrai_tcp_viewer` separately for the TCP viewer.
The renderer reads the scene's saved camera, environment, and visual assets.
The examples use the existing assets under `rsc`; preparation tools are in
`examples/tools`. `--headless` runs physics without a renderer.

`rayrai_render.py` demonstrates direct rendering, material overrides, RGBA and
linear depth capture. Its `--headless` option creates a hidden GL context;
`--output image.png` saves a screenshot.

`buggy.py` loads `rsc/offroad_buggy/buggy.urdf` and the matching 16-bit
race-track heightmap. The URDF supplies the 60-DoF suspension and steering
mechanism. Rear-wheel drive has torque/power limits and a limited-slip
differential; the controller also supplies braking and rate-limited power
steering. It holds the brakes for the first simulated second to settle the
suspension, then reads the viewer's **Throttle**, **Steering**, and **Brake**
sliders. Command-line options set their initial values:

```bash
python raisimPy/examples/buggy.py
python raisimPy/examples/buggy.py --throttle 0.35 --steer 0.15
python raisimPy/examples/buggy.py --headless --steps 3000 --throttle 0.35
```

Throttle and brake range from 0 to 1; steering ranges from -1 (right) to 1
(left). `--reverse` selects reverse drive. The common `--steps`, `--port`,
`--headless`, and activation options apply to all four examples.

## Tests

With `RAISIM_PY=ON` and `BUILD_TESTING=ON` (the default), CMake registers the
`raisimpy_runtime` CTest. It runs the Python API regressions and all examples
with bounded headless loops, including the cradle's tendon removal and a
checkpoint restore:

```bash
RAISIM_ACTIVATION_KEY="$HOME/.raisim/activation.raisim" \
  ctest --test-dir build-py -R raisimpy_runtime --output-on-failure
```

To exercise server startup and shutdown too, set `RAISIMPY_TEST_SERVER=1`.
Those tests bind ephemeral loopback ports and need permission to open local
sockets. They do not validate rendered viewer output.


To run the real in-process renderer tests and capture example:

```bash
RAISIMPY_TEST_RAYRAI=1 SDL_VIDEODRIVER=offscreen \
  ctest --test-dir build-py -R raisimpy_runtime --output-on-failure
```

These exercise actual rendering/readback, multiple contexts, RGB/depth,
visual/material/point-cloud/instance edits, thread rejection, fatal-error
recovery, and handle invalidation after close. Set `RAISIMPY_TEST_SERVER=1`
separately to test the TCP path.
