# Performance notes and the C++ API reference

## Performance notes

rayrai defaults to one simulation/render thread. Applications that explicitly
opt into `ThreadingMode::MultiThread` use one renderer and current OpenGL
context per worker, as described in [rayrai Visualizer](../Rayrai.md). The renderer keeps the
main performance controls explicit:

- Disable shadows for high-throughput camera observations when shadows are not part
  of the desired output. Use `RenderOverrides::doShadows = false` or disable the
  light's shadow map globally.
- Prefer `addInstancedVisuals` for repeated primitives or meshes. A single
  instanced visual renders many copies with one instance buffer instead of many
  separate visual objects.
- Opaque RaiSim single-body primitives (spheres, boxes, cylinders, capsules) are
  batched into instanced draws in shadow-map passes and in simple-shading color
  passes. With PBR primitives (`Balanced` and above) the color pass batches
  them only in renders without shadows.
- TCP viewer scenes synthesize repeated articulated mesh visuals into internal
  `InstancedVisuals` batches when mesh path, scale, color policy, and color match.
  `RemoteScene::tcpMeshBatchCount()` and
  `tcpMeshBatchMeshColorCount()` expose diagnostics for this path.
- Use `InstancedVisuals::setMaxRenderedInstances` and
  `PointCloud::setMaxRenderedPoints` as simple LOD caps for debug overlays,
  particles, scans, or dense markers that do not need full density in every frame.
- Keep non-visible debug geometry outside the camera frustum when possible. rayrai
  performs coarse frustum culling for RaiSim objects, custom visuals, instanced
  visuals, and point clouds, so off-camera content is skipped before draw submission.
  Mesh-local bounds are used when available, so offset meshes, articulated links,
  deformables, and heightmaps cull and frame from their rendered bounds rather
  than only from the object origin.
- Scenes with many local lights can give point, spot and area lights a finite
  range with `additionalLightContributionThreshold` (see [Render quality, tone mapping, color grading](./RenderQuality.md)).
- `updateObjectLists()` copies only what changed in the bound World: an unchanged
  object costs a revision check, and the object cache is rebuilt only when objects
  are added or removed. A heightmap edit copies and uploads only the rows it
  changes. `rayrai_world_sync_benchmark` measures the copy and a small frame for
  moving boxes, robots, edited heightmaps and a deformable terrain.
- RGB/depth CPU readback supports an optional PBO-backed asynchronous path. The
  synchronous path remains the deterministic default for simple RL loops.
- Dynamic point clouds and instanced visuals support partial buffer updates for
  streaming changes.
- Deformable TCP streaming sends topology/indices only during initialization or topology
  changes; normal frames send vertex positions only.

## Physics snapshots for independent cameras

Use [Physics snapshots](./PhysicsSnapshots.md) to separate physics capture from camera rendering.
Capture a shared immutable frame immediately after integration under the World
mutex, then apply and render it on the owning GL thread after releasing that
mutex. The capture pool reuses buffers only after consumers release them.
Persistent shape descriptions are shared, articulated frames contain one pose
per link, and geometry and appearance have separate revisions. Membership is
reconciled only when it changes; dynamic mesh updates preserve unchanged
indices and heightmap patch ranges.

The source target `rayrai_physics_snapshot_benchmark` measures capture/apply
for 1,000 unchanged and moving boxes after warmup, excluding rendering and
readback. The dedicated guide includes API examples, the latest-frame mailbox
pattern, copied RGB/depth camera state, and lifetime rules.

Animated articulated visuals retain their loaded meshes when only scale,
local pose, link attachment or color changes. Candidate bounds and shadows
follow transform edits. Replacing a mesh file or resizing a capsule retains
unrelated model assets while acquiring the new geometry; see
[Physics snapshots](./PhysicsSnapshots.md) for the visual fields to edit.

The engine source target `rayrai_animated_mesh_reuse_test --benchmark`
measures capture/apply while a robot's authored mesh scale changes over 1,000
updates. It releases frame-scoped asset owners as rendering does and reports
`animated_mesh_apply_us` and `model_reloads`. With unchanged model files,
`model_reloads` should remain zero. Run with one thread and a working
OpenGL context, for example from a standalone visualizer build:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 \
  ./build-visualizer/rayrai/rayrai_animated_mesh_reuse_test --benchmark
```

The test without `--benchmark` checks synchronous and asynchronous loading,
both applied snapshots and a bound World, rendered depth, capsule resizing,
and mesh-file replacement.

## GPU paths and OpenGL requirements

Several instanced-rendering paths run on the GPU when the context supports them
and fall back automatically otherwise:

- Instanced visuals with 4096 or more rendered instances are culled with a
  compute shader (OpenGL 4.3). Without it, culling runs on the CPU.
- Instanced meshes made of several submeshes can be submitted with
  multi-draw-indirect (OpenGL 4.3 or `GL_ARB_multi_draw_indirect`); otherwise
  each submesh is drawn separately.
- Foliage visuals with 4096 or more instances are occlusion-culled against
  opaque geometry already drawn in the frame, using a hierarchical depth
  pyramid or occlusion queries. The compute-shader pyramid needs OpenGL 4.3.
- With OpenGL 4.3 and MSAA, foliage is also culled against the depth rendered
  so far in the same frame (see [Dense foliage](#dense-foliage)).

Most of these fallbacks print a one-time `WARN` line to stderr; the matching
`RAYRAI_SUPPRESS_*_FALLBACK_WARNINGS` variable silences it (see
[Environment variables](#environment-variables)).

macOS provides OpenGL 4.1 (no compute shaders or multi-draw-indirect), so it
uses the fallback of every path above; see [Platform support](../Rayrai.md#platform-support). The
renderer also selects a full, compact or limited shader tier from the number of
fragment texture units when it is constructed; the lower tiers render fewer
material and post-process features (see
[GPU capability tiers](./Materials.md#gpu-capability-tiers)).

## Dense foliage

Instanced visuals with alpha-masked foliage materials use a dedicated path with
no extra API calls:

- Compatible foliage draws in the color and shadow passes are merged into
  batched runs. Other objects between them split a run instead of disabling
  batching.
- Visibility, and for larger plants the LOD level, are chosen per instance and
  per submesh from the automatic mesh LOD chain. A level is used when its
  projected simplification error is at most 2.25 pixels, or 2.5 shadow-map
  texels in shadow passes. `RAYRAI_FOLIAGE_LOD_PIXEL_ERROR` and
  `RAYRAI_FOLIAGE_SHADOW_LOD_PIXEL_ERROR` override these budgets for the whole
  process (values in `(0, 8]`).
- With OpenGL 4.3 and MSAA, passes that contain a foliage visual with 2048 or
  more instances cull foliage against the depth already rendered in the same
  frame, tested in 8-pixel tiles that cover every sample.
- Foliage casts shadows by default. Call `setCastsShadows(false)` on an
  `InstancedVisuals` to exclude it; wind and grass-patch configuration keep
  that choice. When the main viewer refreshes cascaded shadows every frame,
  foliage that cannot shadow anything inside a cascade's part of the view is
  skipped for that cascade.
- With asynchronous mesh loading ([Capture, diagnostics, and headless rendering](./Capture.md)), file imports, texture
  decoding, base geometry and instanced LOD chains are prepared on worker
  threads: imports and texture decodes share up to eight, up to half the
  hardware threads prepare geometry and LOD chains of different assets at
  once, and a large LOD chain builds up to three of its levels at once. Larger
  models go first: their textures decode first and their stages run first on
  the render thread, which only builds materials and issues uploads. With
  OpenGL 4.4 (or `GL_ARB_buffer_storage`) the workers copy textures and
  vertex buffers into persistently mapped staging buffers, so an upload is a
  GPU copy; a context holds up to two 128 MiB blocks of them while it loads.
  Each `update()` and
  external-camera render spends up to 8 ms of render-thread time on loading
  (one stage runs even when it takes longer), so a scene of many assets loads
  in a few frames. Each
  `pollAsyncMeshLoads(n)` call advances at most `n` import/upload stages
  (`0` does nothing) and uploads at most one prepared submesh per pending
  instanced visual; `pendingAsyncMeshLoadCount()` includes these jobs.
  Prepared LOD chains are cached on disk (see [On-disk caches](#on-disk-caches)).
- `rayrai/Bc7TextureStorage.hpp` provides an opt-in load-time helper,
  `raisin::compressRgbTextureBc7(texture)`, that re-stores an RGB8/SRGB8
  texture and all of its mip levels as BC7. It needs OpenGL 4.2 or
  `GL_ARB_texture_compression_bptc` (check `supportsBc7TextureStorage()`)
  and is lossy. Alpha, float, immutable and already compressed textures are
  left unchanged.

## Frame submission and measurement

The renderer applies these submission optimizations automatically; none of
them changes the rendered image:

- After shadow maps or the planar reflection were rendered in a frame, one
  `glFlush` submits them so the GPU works on them while the CPU encodes the
  scene pass. This lowers frame time at the cost of some process CPU time.
  The `RAYRAI_DEBUG_DISABLE_PASS_FLUSH=1` debug switch skips this flush.
- The environment background is drawn inside the scene pass. After an opaque
  depth prepass it is depth-tested, so only pixels that no surface covers are
  shaded. `environmentBackgroundDiagnostics()` reports how the last scene pass
  drew it.
- The transparent pass skips the proximity-fade depth copy and the screen-space
  refraction capture when no drawn material samples them.
- An external `Camera` keeps its multisampled render target between frames;
  the next render of that camera without MSAA releases it.
  `Camera::setSceneMsaaSamples(samples, keepAllocation)` exposes the same
  behavior.

While `RenderQualitySettings::geometryRefraction` is on, the frame caches that
let unchanged external-camera renders reuse earlier results are bypassed, so
every render is drawn in full.

`captureViewerPassTimings(width, height)` renders the built-in viewer camera
once with blocking GPU timer queries (the renderer's context must be current)
and returns the `shadow`, `scene_color_and_resolve` and `postprocess`
passes. `captureRenderPassTimings` does the same for an external camera. When
foliage instanced visuals are present, both render a second frame and append
`shadow_foliage*` and `scene_foliage*` entries that are nested inside the
coarse passes. `shadowCacheDiagnostics()`, `foliageRenderDiagnostics()` and
`conservativeOcclusionCullingDiagnostics()` report shadow-map reuse and
culling without extra GPU work.

## Shared mesh buffers across contexts

OpenGL contexts in one share group reuse the immutable vertex/index buffers of
the built-in primitives and of file-backed meshes whose materials use no
textures and no `nextPass` chain. To create such contexts, make the root SDL
OpenGL context current before calling `RayraiWindow::createOffscreenGlContext`
for each child. Each call makes the new context current, so rebind the root
when creating the next child. Creating a context on a thread with no current
context creates an independent share group and cannot provide this reuse.

Textures are not shared. rayrai sets filtering and anisotropy on each texture
object per material and per quality preset, so every context decodes and
uploads its own copy of each image file, and textured meshes are uploaded per
context as well.

Keep each renderer's context current for rendering and resource destruction.
Do not use one context concurrently on multiple threads. VAOs, framebuffers,
and camera targets remain specific to each context. A context created on the
main thread stays current there; release it with
`RayraiWindow::detachOffscreenGlContext(window)` before a worker adopts it
with `makeOffscreenContextCurrent`. The cache retains weak references, so
reuse requires an existing owner to keep the mesh alive.

The public diagnostics can be read without changing rendering behavior:

```cpp
const auto stats = raisin::RayraiGlobalAsset::sharedGpuAssetStats();
const auto uploadedBytes = stats.gpuBufferBytesUploaded;
const auto reusedBytes = stats.gpuBufferBytesReused;
const auto templates = stats.cachedSnapshots;
```

The byte counts are cumulative mesh-buffer counters, not total or live GPU
memory. Textures, VAOs, and render targets are excluded. `cachedSnapshots` is
the number of published mesh templates currently held, one per distinct asset
and share group. Use `resetSharedGpuAssetStats()` before a controlled
measurement. The cache is enabled by default;
`setSharedGpuAssetCacheEnabled(false)` makes every later mesh lookup upload
its own buffers.

## Shader binary cache and multi-threaded prewarming

rayrai's PBR shaders compile in 10-30 seconds on first run depending on the
GL driver. Two features (available since v2.3.0) remove that wait in
production RL pipelines:

1. **Persistent shader binary cache.** Compiled GL programs are written to a
   per-driver cache directory and reloaded on subsequent runs.
2. **Static prewarm helper** for multi-threaded RL setups where many worker
   threads will each construct their own `RayraiWindow` — one thread can
   pre-compile every shader, then every worker thread reads from the cache.

Both are wired through the `RayraiWindow` constructor:

```cpp
raisin::RayraiWindow viewer(
  world,
  /*width=*/1280, /*height=*/720,
  raisin::RayraiWindow::ThreadingMode::SingleThread,
  /*shaderCompileThreadCount=*/1,            // 1 = matches rayrai's default
  /*shaderBinaryCacheEnabled=*/true,         // on by default
  /*shaderBinaryCacheDirectory=*/"",          // empty → default location
  /*logShaderBinaryCache=*/false);
```

- `shaderCompileThreadCount` is passed to `glMaxShaderCompilerThreadsARB`
  on drivers that expose `GL_ARB_parallel_shader_compile` and is ignored
  elsewhere. Raise it only when the host application is itself multi-threaded.
- `shaderBinaryCacheEnabled = true` (the default) writes program binaries
  under the cache directory keyed by GL vendor / renderer / version + GLSL
  source, when the driver reports at least one program binary format. On the
  next run, identical configurations are loaded directly, skipping the GLSL
  compile entirely. Several processes can share one cache directory.
- `shaderBinaryCacheDirectory = ""` uses the first writable directory in this
  order, then falls back to `<temp>/raisim/rayrai`:
  - Linux: `$XDG_CACHE_HOME/raisim/rayrai`, `~/.cache/raisim/rayrai`,
    `~/.raisim/rayrai`
  - macOS: `$XDG_CACHE_HOME/raisim/rayrai`, `~/Library/Caches/raisim/rayrai`,
    `~/.cache/raisim/rayrai`, `~/.raisim/rayrai`
  - Windows: `%LOCALAPPDATA%\RaiSim\rayrai`, `%HOME%\.raisim\rayrai` (when
    `HOME` is set), `%USERPROFILE%\AppData\Local\RaiSim\rayrai`,
    `%USERPROFILE%\.raisim\rayrai`
- `logShaderBinaryCache = true` prints every hit/miss/store to stderr —
  useful for verifying the cache is actually being consulted.

The compile thread count and the cache settings are process-wide: each
`RayraiWindow` constructor applies its arguments to the whole process, so pass
the same values to every renderer, including a prewarmer.

Cache statistics (per process, across all `Shader::compile` calls):

```cpp
auto s = raisin::Shader::binaryCacheStats();
std::printf("hits=%llu misses=%llu stores=%llu coordinated_waits=%llu\n",
            static_cast<unsigned long long>(s.hits),
            static_cast<unsigned long long>(s.misses),
            static_cast<unsigned long long>(s.stores),
            static_cast<unsigned long long>(s.coordinatedWaits));
raisin::Shader::resetBinaryCacheStats();  // scope a measurement window
```

Pre-warming for parallel RL can be done with the `rayrai_shader_prewarm`
utility when it is available in the rayrai package. The utility creates an
offscreen context and compiles every built-in shader into the persistent binary
cache:

```bash
rayrai_shader_prewarm --cache-dir /tmp/rayrai-cache --compile-threads 4 --multi-thread
rayrai_shader_prewarm --log-cache
```

For an embedded application, create an offscreen context on the main thread,
release it, and call `prewarmShadersForCurrentContext` once from a background
thread that adopts it. SDL windows must be created and destroyed on the main
thread, and every renderer in the process, including the prewarmer, must use
the same `ThreadingMode`. Worker threads that later construct their own
`RayraiWindow` will find the heavy shaders already cached (or briefly block on
a per-shader compile mutex if the warm-up is still in flight):

```cpp
// Main thread: create the prewarm context, then release it.
SDL_Window* w = nullptr;
SDL_GLContext gl = nullptr;
raisin::RayraiWindow::createOffscreenGlContext(w, gl, "rayrai_prewarm");
raisin::RayraiWindow::detachOffscreenGlContext(w);

std::thread prewarm([w, gl] {
  raisin::RayraiWindow::makeOffscreenContextCurrent(w, gl);
  raisin::RayraiWindow::prewarmShadersForCurrentContext(
    raisin::RayraiWindow::ThreadingMode::MultiThread);
  raisin::RayraiWindow::detachOffscreenGlContext(w);
});

// ... start the rendering workers ...

prewarm.join();  // main thread
SDL_GL_DeleteContext(gl);
SDL_DestroyWindow(w);
```

Pass the workers' cache directory as the second argument of
`prewarmShadersForCurrentContext` when they do not use the default location.
With the binary cache enabled, each registered program is compiled from GLSL
once per process, however many worker threads construct renderers.

When you only want to warm a specific subset:

```cpp
// Names of every program registered with the running RayraiWindow.
auto names = viewer.linkedShaderNames();

// Compile the heaviest authored-content shader explicitly.
long long ms = viewer.compileShaderByName("pbrMeshHigh");
std::printf("pbrMeshHigh warmup: %lld ms\n", ms);

// Iterate every program, one per call. `done` flips true when all are
// linked; useful for spreading shader work across multiple frames.
bool done = false;
while (!done) {
  viewer.warmupNextShader(done);
}
```

For diagnostics on the cost of the lazy-compile pipeline (how long each
program took on first use, how many are linked so far), call
`shaderWarmupDiagnostics()`.

## On-disk caches

Besides the shader binary cache, rayrai keeps these caches. All of them are
regenerated when missing, so the files can be deleted at any time. The mesh
caches are also checked against their source and rebuilt when stale.

| Contents | Default location | Override |
| --- | --- | --- |
| Imported-mesh metadata and generated LOD chains | `<temp>/rayrai_mesh_preprocess_cache` | `RAYRAI_MESH_PREPROCESS_CACHE_DIR` |
| LOD chains prepared by asynchronous loading of instanced meshes, and the prepared geometry and import description of asynchronously loaded models of at least 65,536 vertices | `rayrai_cache_<file>.lods`, `.rgeo` and `.rimp` next to the source asset, else `<temp>/rayrai_async_lod_cache` | `RAYRAI_ASYNC_LOD_CACHE_DIR` |
| Procedural cloud noise textures | `$HOME/.raisim/rayrai`, then `%USERPROFILE%\.raisim\rayrai` (Windows), then `<temp>/raisim/rayrai` | `RAYRAI_CLOUD_TEXTURE_CACHE_DIR` |

`<temp>` is the system temporary directory. The asynchronous LOD and
prepared-geometry sidecars store full vertex data and can be large for dense
assets; an import description holds the model's materials, its embedded
textures and what the import measured of its meshes.

## Environment variables

rayrai reads these variables from the process environment; set them before the
renderer starts. Boolean switches are on when set to a non-empty value that does
not start with `0`, unless noted otherwise.

- `RAYRAI_MESH_PREPROCESS_CACHE_DIR`, `RAYRAI_ASYNC_LOD_CACHE_DIR`,
  `RAYRAI_CLOUD_TEXTURE_CACHE_DIR`: cache directories (see
  [On-disk caches](#on-disk-caches)).
- `RAYRAI_FOLIAGE_LOD_PIXEL_ERROR` and
  `RAYRAI_FOLIAGE_SHADOW_LOD_PIXEL_ERROR`: foliage LOD error budgets
  (defaults `2.25` pixels and `2.5` texels; read once per process). The
  shadow budget follows the color budget when only the latter is set.
- `RAYRAI_SUPPRESS_<PATH>_FALLBACK_WARNINGS`: silence the one-time warning of
  a GPU path fallback, for example
  `RAYRAI_SUPPRESS_GPU_INSTANCE_CULLING_FALLBACK_WARNINGS`.
- `RAYRAI_FORCE_COMPACT_PBR_SAMPLER_FALLBACK` and
  `RAYRAI_FORCE_LIMITED_PBR_SAMPLER_FALLBACK`: select the compact or limited
  shader tier on a larger GPU, for testing.
- `RAYRAI_EAGER_SHADER_WARMUP`: compile every registered shader when a
  renderer is constructed instead of on first use.
- `RAYRAI_PBR_DEBUG_OUTPUT`: PBR debug visualization channel (`0` is off).
  `RayraiWindow::setPbrDebugOutputMode` overrides it for one renderer, and
  `captureDebugPasses` overrides it for the frames it renders.

The TCP viewer reads its own `RAYRAI_TCP_VIEWER_*` variables
([Rayrai TCP Viewer](../RayraiTcpViewer.md)). Variables named `RAYRAI_DEBUG_*` and
`RAYRAI_LOG_*` are renderer-development diagnostics (debug switches and
logging) and may change between releases. Apart from these debug switches,
optimizations have no switches to turn them off.

## Additional tips

- Every `update` and external-camera render calls
  `RayraiWindow::updateObjectLists()`, so RaiSim objects added to or removed
  from the world are picked up automatically. Call it yourself only when the
  renderer's object lists must be current before the next render.
- Use `setShowCollisionBodies(true)` for debug visualization of collision shapes.
- If you want rayrai overlays to use your ImGui font, pass it via `setExternalFont`.

## API

### Core types

- [`raisin::RayraiWindow`](/api/raisin/RayraiWindow.md)
- [`raisin::Camera`](/api/raisin/Camera.md)
- [`raisin::RayraiGlobalAsset`](/api/raisin/RayraiGlobalAsset.md)

### Visuals and geometry

- [`raisin::Visuals`](/api/raisin/Visuals.md)
- [`raisin::InstancedVisuals`](/api/raisin/InstancedVisuals.md)
- [`OpenGLMesh`](/api/OpenGLMesh.md)
- [`raisin::assimp`](/api/raisin/assimp/index.md)

### Scene helpers

- [`raisin::PointCloud`](/api/raisin/PointCloud.md)
- [`raisin::CoordinateFrame`](/api/raisin/CoordinateFrame.md)
- [`raisin::CameraFrustum`](/api/raisin/CameraFrustum.md)
- [`raisin::VisualCategory`](/api/raisin/index.md#visualcategory)

### Rendering and materials

- [`raisin::Light`](/api/raisin/Light.md)
- [`raisin::Material`](/api/raisin/Material.md)
- [`raisin::RenderQualitySettings`](/api/raisin/RenderQualitySettings.md)

The per-render toggles passed to the external-camera, capture, and timing
APIs are the nested struct [`raisin::RayraiWindow::RenderOverrides`](/api/raisin/RayraiWindow/RenderOverrides.md),
documented with `raisin::RayraiWindow` above.

- [`raisin::RenderQualityPreset`](/api/raisin/index.md#renderqualitypreset)
- [`raisin::ViewerColorMode`](/api/raisin/index.md#viewercolormode)
- [`raisin::ColorGradePreset`](/api/raisin/index.md#colorgradepreset)
- [`raisin::PostProcessDebugMode`](/api/raisin/index.md#postprocessdebugmode)
- [`raisin::GeometryRefractionBackend`](/api/raisin/index.md#geometryrefractionbackend)
- [`raisin::GeometryRefractionDiagnostics`](/api/raisin/GeometryRefractionDiagnostics.md)
- [`raisin::GeometryRefractionSamplingStatistics`](/api/raisin/GeometryRefractionSamplingStatistics.md)

### Weather, scene effects, and reflections

- [`raisin::WeatherSettings`](/api/raisin/WeatherSettings.md)
- [`raisin::WeatherPreset`](/api/raisin/index.md#weatherpreset)
- [`raisin::LocalFogVolume`](/api/raisin/LocalFogVolume.md)
- [`raisin::ProjectedDecal`](/api/raisin/ProjectedDecal.md)
- [`raisin::IrradianceVolume`](/api/raisin/IrradianceVolume.md)
- [`raisin::ReflectionProbe`](/api/raisin/ReflectionProbe.md)
- [`raisin::ReflectionProbeCaptureSettings`](/api/raisin/ReflectionProbeCaptureSettings.md)
- [`raisin::ReflectionProbeFilterSettings`](/api/raisin/ReflectionProbeFilterSettings.md)
- [`raisin::ReflectionProbeBlend`](/api/raisin/ReflectionProbeBlend.md)
- [`raisin::ReflectionProbePlacementSuggestion`](/api/raisin/ReflectionProbePlacementSuggestion.md)

### Water

- [`raisin::WaterBody`](/api/raisin/WaterBody.md)
- [`raisin::GerstnerWave`](/api/raisin/GerstnerWave.md)
- [`raisin::WaterWaveSpectrum`](/api/raisin/WaterWaveSpectrum.md)
- [`raisin::WaterFlow`](/api/raisin/WaterFlow.md)
- [`raisin::WaterFlowSource`](/api/raisin/WaterFlowSource.md)
- [`raisin::WaterFlowSample`](/api/raisin/WaterFlowSample.md)
- [`raisin::WaterSurfaceSample`](/api/raisin/WaterSurfaceSample.md)
- [`raisin::WaterDiagnostics`](/api/raisin/WaterDiagnostics.md)
- [`raisin::WaterBodyShape`](/api/raisin/index.md#waterbodyshape)
- [`raisin::WaterPreset`](/api/raisin/index.md#waterpreset)
- [`raisin::WaterFlowEdges`](/api/raisin/index.md#waterflowedges)

The water functions (`generateGerstnerWaves`, `defaultWaterBody`, `validateWaterBody`,
`sampleWaterSurface`, `maxWaterWaveHeight`) are in `rayrai/Water.hpp`; see [Water](./Water.md).

Render quality, weather, capture, reflection-probe, and diagnostics helpers are
exposed through `raisin::RayraiWindow` and the public headers included by
`rayrai/RayraiWindow.hpp`. The sections above document the intended entry
points and the fast-path boundaries for those APIs.

### RaiSim integration

- [`raisin::RaisimObject`](/api/raisin/RaisimObject.md)
