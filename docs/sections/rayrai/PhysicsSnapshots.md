# Physics snapshots

<a id="rayrai-physics-snapshots"></a>

Physics snapshots let a camera render a complete physics state while the
simulation continues on another thread. The physics thread copies poses,
geometry and appearance after integration. The render thread applies that
immutable frame to its `RayraiWindow` and renders with copied camera state.
The World mutex protects the physics copy; the window mutex protects snapshot
application, rendering and readback. These phases do not hold both mutexes.

Snapshots are also how rayrai draws a World bound to the viewer: each render
copies it with `updateObjectLists()` and draws the copy. Applying a snapshot
replaces that copy with yours, so both cases share one rendering path.

The API is always included in the normal rayrai library build. Include
`rayrai/PhysicsSnapshot.hpp` and link the `rayrai` target, as described in
[rayrai Visualizer](../Rayrai.md).

## Capture and apply

Use one `raisin::PhysicsSnapshotCapture` on the physics thread and keep it
between frames so it can reuse its buffers. `capture(world)` takes no locks
and performs no OpenGL work. The caller protects the World and captures after
`integrate()` returns, or immediately after `integrate2()` when using split
integration. Sensor poses copied for the image must come from that same step.

`capture()` returns `std::shared_ptr<const raisin::PhysicsSnapshot>`. Pass
this shared owner directly to `RayraiWindow::applyPhysicsSnapshot()` on the
render thread, with the window's OpenGL context current. Subsequent renders,
including shadows, culling and picking, use the copied physics state. An
application that builds or edits snapshots itself gets `std::invalid_argument`
for an object without its description or mesh, malformed vertex blocks, or
indices past the vertices; the viewer then keeps showing the last complete
frame. The renderer can be constructed with its own empty World:

```cpp
#include <memory>
#include <mutex>
#include <raisim/World.hpp>
#include <rayrai/PhysicsSnapshot.hpp>
#include <rayrai/RayraiWindow.hpp>

// Application setup; the renderer's GL context is current here.
auto sourceWorld = std::make_shared<raisim::World>();
sourceWorld->addGround();
raisin::PhysicsSnapshotCapture capture;
raisin::RayraiWindow viewer(std::make_shared<raisim::World>(), 640, 480);
std::mutex windowMutex;
std::shared_ptr<const raisin::PhysicsSnapshot> frame;

// Physics phase.
{
  std::lock_guard<raisim::World> physicsLock(*sourceWorld);
  sourceWorld->integrate();
  frame = capture.capture(*sourceWorld);
}

// Render phase, on the thread owning the GL context.
{
  std::lock_guard<std::mutex> windowLock(windowMutex);
  viewer.applyPhysicsSnapshot(frame);
  viewer.update(640, 480, false, 0, 0, false);
}
```

This sequence also works in a single-threaded loop. For independent threads,
publish the frame through a synchronized queue or mailbox after releasing the
World mutex. With one renderer, `ThreadingMode::SingleThread` remains
appropriate: all its GL operations stay on one thread. Multiple render workers
must follow the context and threading-mode rules in [rayrai Visualizer](../Rayrai.md).

## Ownership and scheduling

Keep a shared owner while a camera consumes a frame. The capture writes into a
frame again only after the last consumer releases it, using an explicit
release/acquire handoff. Expired weak owners cannot reacquire a frame that is
being filled. Heightmap and deformable vertices are immutable blocks shared
between frames, so an edit allocates only the blocks it changes. Several cameras
can share one frame, and retaining an older frame keeps its poses and geometry
unchanged even after the source objects move or are removed.

Animating an articulated visual's mesh scale, offset, orientation or link
attachment updates its model transform, culling bounds and shadows while
retaining its loaded mesh. For file-mesh visuals, edit `VisObject::scale`
for per-axis scaling, `offset` for local translation, `rot` for local
rotation, and `localIdx` for the attached link. Capture after the edit and
apply that frame on the render thread; a viewer drawing its bound World picks
up the edit on its next render. These transform edits keep loaded meshes in
both synchronous and asynchronous loading modes, so animation does not
restart file loading. This also applies to the off-road buggy's spring and
drive-shaft animation.

Sphere, box and cylinder dimensions also update model transforms using shared
unit meshes. Capsules bake their radius and length into vertices, so changing
those dimensions rebuilds capsule geometry. Changing a mesh filename loads
the replacement asset. Both kinds of geometry edit retain unrelated model
assets while acquiring the new geometry.

A mailbox containing only the latest pending frame bounds the amount of queued
work. Replacing its pending owner drops an obsolete image request; a renderer
that already took the previous owner can finish safely. Every published frame
contains the complete render state, so consuming the newest one does not require
applying the skipped frames first.

Capture at the camera's scheduled simulation time, after the corresponding
physics step. A 30 Hz camera does not need a new snapshot at every 1 kHz physics
step. Multiple cameras due at the same step can share the physics frame and
publish their individual poses with it. `PhysicsSnapshot::simTime` records
the source simulation time; use it when timestamping the resulting image.

Release a mailbox lock before applying or rendering a frame. The physics thread
must release the World mutex before publishing and must not wait for a window
lock, a GPU fence or pixel readback while holding it.

## RGB camera on a separate thread

Copy `RGBCameraProperties` under the World mutex during setup, then construct
`raisin::Camera` from that value on the render thread. Publish the camera pose
alongside each physics frame. RaiSim sensor +X points forward and +Z points up.
The following mailbox example assumes the sensor is already attached to its
robot and the World contains that robot:

```cpp
#include <utility>
#include <vector>
#include <glm/glm.hpp>

struct CapturedRgbFrame {
  std::shared_ptr<const raisin::PhysicsSnapshot> physics;
  glm::vec3 position{0.f};
  glm::mat3 rotation{1.f};
};

struct RgbMailbox {
  std::mutex mutex;
  std::shared_ptr<const CapturedRgbFrame> pending;
};

// Called only on the physics thread.
void stepAndPublish(raisim::World& world, raisim::RGBCamera& sensor,
                    raisin::PhysicsSnapshotCapture& capture, RgbMailbox& mailbox) {
  auto frame = std::make_shared<CapturedRgbFrame>();
  {
    std::lock_guard<raisim::World> physicsLock(world);
    world.integrate();
    frame->physics = capture.capture(world);
    sensor.updatePose();
    const auto& position = sensor.getPosition();
    const auto& rotation = sensor.getOrientation();
    frame->position = {position[0], position[1], position[2]};
    for (int column = 0; column < 3; ++column)
      for (int row = 0; row < 3; ++row)
        frame->rotation[column][row] = rotation(row, column);
  }
  std::lock_guard<std::mutex> publishLock(mailbox.mutex);
  mailbox.pending = std::move(frame);
}

// Called only on the render thread, with its GL context current.
bool renderNewest(raisin::RayraiWindow& viewer, raisin::Camera& camera,
                  std::mutex& windowMutex, RgbMailbox& mailbox,
                  std::vector<char>& bgra) {
  std::shared_ptr<const CapturedRgbFrame> frame;
  {
    std::lock_guard<std::mutex> mailboxLock(mailbox.mutex);
    frame = std::exchange(mailbox.pending, nullptr);
  }
  if (!frame) return false;

  std::lock_guard<std::mutex> windowLock(windowMutex);
  viewer.applyPhysicsSnapshot(frame->physics);
  camera.targetObject = nullptr;
  camera.position = frame->position;
  camera.front = frame->rotation[0];
  camera.up = frame->rotation[2];
  camera.right = glm::normalize(glm::cross(camera.front, camera.up));

  raisin::RayraiWindow::RenderOverrides overrides;
  overrides.drawCoordinateFrames = false;
  overrides.drawPointClouds = false;
  overrides.drawVisualizationObjects = false;
  overrides.drawWeatherEffects = false;
  overrides.allowViewerMsaa = false;
  overrides.allowTemporalAa = false;
  overrides.allowViewerUpscale = false;
  viewer.renderWithExternalCamera(camera, overrides);
  bgra.resize(size_t(camera.rtWidth()) * size_t(camera.rtHeight()) * 4);
  camera.readSceneColorBgra(bgra.data(), bgra.size());
  return true;
}
```

Call `stepAndPublish()` for steps at which a camera image is due. On other
steps, integrate under the World mutex without capturing or publishing a frame.
The example leaves the camera's simulation-time schedule to the application.

Create and retain the render camera as follows; `rgbProperties` is the value
copied from `sensor.getProperties()` during protected setup:

```cpp
// Render thread, GL context current.
raisin::Camera camera(rgbProperties);
```

The Camera-only overloads and `readSceneColorBgra()` do not access the live
sensor. Keep `targetObject` null and set the copied axes directly; calling
`Camera::update()` afterward would recompute the view from its input/orbit
state. The image buffer contains BGRA8 in top-to-bottom row order by default.

Camera-only rendering uses the generic external-camera capture policy. The
example disables custom visuals, instanced visuals and viewer effects to render
physics objects. Set `drawVisualizationObjects = true` when the image should
also include detectable scene visuals, such as visual-only `.rscene` objects
or automatic tendon visuals. See [Sensors and depth/LiDAR](./Sensors.md) for the other capture policies.

## Depth camera

Copy `DepthCameraProperties` and the depth sensor pose in the same way. Create
its `Camera` on the render thread and apply the shared physics frame before
rendering. Use an independent float buffer for readback:

```cpp
raisin::Camera depthCamera(depthProperties);  // copied calibration, GL context current
// Set position/front/up/right from the captured depth sensor pose.
viewer.applyPhysicsSnapshot(frame);
viewer.renderDepthPlaneDistance(depthCamera);
std::vector<float> depth(size_t(depthCamera.rtWidth()) * size_t(depthCamera.rtHeight()));
depthCamera.readLinearDepth(depth.data(), depth.size());
```

Depth values are camera-plane distances in metres; pixels without geometry are
zero. The default depth call excludes visualization objects. Pass
`nullptr, true` after the camera argument to include detectable custom and
instanced visuals. Reconstruct point clouds using the copied sensor rays and
captured pose, as demonstrated in
[Rayrai Example: Depth Camera](../examples/rayrai/rayrai_depth_camera.md).

GPU spinning lidar currently casts against the bound live World. Its sensor
overload raises `std::invalid_argument` in snapshot mode; use a live-world
renderer protected by the World mutex for that API.

## Changes, geometry and lifetime

Frames store contiguous object state and one pose per articulated link. Shapes,
mesh filenames, scales and local visual transforms are shared across pose and
appearance updates. The render thread composes cached visual offsets with link
poses. Geometry and appearance revisions are tracked separately, including
edits through mutable `getVisOb()` and `getVisColOb()` arrays.

Object membership is reconciled only when identities or order change. An object
whose render revision did not advance keeps its state from the previous frame;
the revision advances for pose and state changes, visual-only heightmap edits
and deformable integration. An articulated system's link poses are read again
only when its revision advances, while its mutable `getVisOb()` and
`getVisColOb()` arrays are compared on every capture, so edits through
retained references are detected. Deformable integration updates vertex
geometry while retaining unchanged triangle indices.

A heightmap mesh has one vertex block per visual row. Capture reads only the
sample rows whose revision (`HeightMap::getHeightRowRevisions()`) advanced, so
it sees visual-only edits and patches even if another consumer cleared the
source dirty flags, and leaves those flags untouched. It refills the visual rows
those samples reach, including the margin needed for normals, and shares the
other rows with the previous frame. The renderer uploads the rows that differ
from the ones it shows, so applying a frame after skipping others costs no more.
Indices are uploaded only when their triangle topology changes. Geometry caches
follow object IDs across membership changes rather than world-list positions.

A frame shares its World's immutable `.rscene` scene instead of copying it.
Applying a frame with a different scene replaces the previous scene-owned
visuals and preserves unrelated overlays; a frame without a scene restores the
preceding viewer settings. Snapshots retain CPU render data, rather than owning
the source World or its objects; `world` and `object` pointers are identity
keys. A separate `worldIdentity` token distinguishes World lifetimes even when
an allocator reuses the same address. Applying and rendering never dereference
those keys: each `RaisimObject` holds a copy of its object's state. A camera
whose `targetObject` is set follows that object's position in the rendered
frame and is released when the object leaves it. Mesh files, textures and other
external resources must remain available for asset loading.

`viewer.clearPhysicsSnapshot()` shows the World bound to the viewer again and
copies it immediately, so protect that World as for a render. If the viewer was
bound to an empty World, that remains its bound World. Use
`viewer.swapWorld(sourceWorld)` to switch to the live source World; this also
clears the snapshot.

## Examples and measurement

The snapshot workflow is used by 15 of the 31 rayrai demos, including
[Rayrai Example: Basic Scene](../examples/rayrai/rayrai_basic_scene.md),
[Rayrai Example: RGB Camera](../examples/rayrai/rayrai_rgb_camera.md),
[Rayrai Example: Depth Camera](../examples/rayrai/rayrai_depth_camera.md), and
[Rayrai Example: Runtime Scene Editing](../examples/rayrai/rayrai_runtime_scene_editing.md).

The source target `rayrai_physics_snapshot_benchmark` measures capture and
apply for 1,000 unchanged and moving boxes after warmup, excluding rendering
and readback; `rayrai_world_sync_benchmark` measures the per-frame copy and
render of a World bound to the viewer. It also measures capture of 100 idle heightmaps and 100 idle cloths.
Pose-only updates in these scenes capture into released frames, allocating only
the shared owner of each frame. First capture, membership changes, geometry edits
and additional simultaneous frame owners can allocate new storage. Image
frequency also depends on rendering and readback costs; see [Performance notes and the C++ API reference](./Performance.md).
