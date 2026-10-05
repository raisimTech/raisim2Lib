###########################################
Performance notes and the C++ API reference
###########################################

Performance notes
=================

rayrai defaults to one simulation/render thread. Applications that explicitly
opt into ``ThreadingMode::MultiThread`` use one renderer and current OpenGL
context per worker, as described in :doc:`../Rayrai`. The renderer keeps the
main performance controls explicit:

* Disable shadows for high-throughput camera observations when shadows are not part
  of the desired output. Use ``RenderOverrides::doShadows = false`` or disable the
  light's shadow map globally.
* Prefer ``addInstancedVisuals`` for repeated primitives or meshes. A single
  instanced visual renders many copies with one instance buffer instead of many
  separate visual objects.
* Opaque RaiSim single-body primitives (spheres, boxes, cylinders, capsules) are
  batched into instanced draws in shadow-map passes and in simple-shading color
  passes. With PBR primitives (``Balanced`` and above) the color pass batches
  them only in renders without shadows.
* TCP viewer scenes synthesize repeated articulated mesh visuals into internal
  ``InstancedVisuals`` batches when mesh path, scale, color policy, and color match.
  ``RemoteScene::tcpMeshBatchCount()`` and
  ``tcpMeshBatchMeshColorCount()`` expose diagnostics for this path.
* Use ``InstancedVisuals::setMaxRenderedInstances`` and
  ``PointCloud::setMaxRenderedPoints`` as simple LOD caps for debug overlays,
  particles, scans, or dense markers that do not need full density in every frame.
* Keep non-visible debug geometry outside the camera frustum when possible. rayrai
  performs coarse frustum culling for RaiSim objects, custom visuals, instanced
  visuals, and point clouds, so off-camera content is skipped before draw submission.
  Mesh-local bounds are used when available, so offset meshes, articulated links,
  deformables, and heightmaps cull and frame from their rendered bounds rather
  than only from the object origin.
* Scenes with many local lights can give point, spot and area lights a finite
  range with ``additionalLightContributionThreshold`` (see :doc:`RenderQuality`).
* If the world topology is stable, repeated ``updateObjectLists()`` calls are cheap:
  rayrai refreshes appearances without rebuilding the object cache unless objects
  were added or removed.
* RGB/depth CPU readback supports an optional PBO-backed asynchronous path. The
  synchronous path remains the deterministic default for simple RL loops.
* Dynamic point clouds and instanced visuals support partial buffer updates for
  streaming changes.
* Deformable TCP streaming sends topology/indices only during initialization or topology
  changes; normal frames send vertex positions only.

GPU paths and OpenGL requirements
=================================
Several instanced-rendering paths run on the GPU when the context supports them
and fall back automatically otherwise:

* Instanced visuals with 4096 or more rendered instances are culled with a
  compute shader (OpenGL 4.3). Without it, culling runs on the CPU.
* Instanced meshes made of several submeshes can be submitted with
  multi-draw-indirect (OpenGL 4.3 or ``GL_ARB_multi_draw_indirect``); otherwise
  each submesh is drawn separately.
* Foliage visuals with 4096 or more instances are occlusion-culled against
  opaque geometry already drawn in the frame, using a hierarchical depth
  pyramid or occlusion queries. The compute-shader pyramid needs OpenGL 4.3.
* With OpenGL 4.3 and MSAA, foliage is also culled against the depth rendered
  so far in the same frame (see `Dense foliage`_).

Most of these fallbacks print a one-time ``WARN`` line to stderr; the matching
``RAYRAI_SUPPRESS_*_FALLBACK_WARNINGS`` variable silences it (see
`Environment variables`_).

macOS provides OpenGL 4.1 (no compute shaders or multi-draw-indirect), so it
uses the fallback of every path above; see :ref:`rayrai-platform-support`. The
renderer also selects a full, compact or limited shader tier from the number of
fragment texture units when it is constructed; the lower tiers render fewer
material and post-process features (see
:ref:`GPU capability tiers <sections/rayrai/Materials:GPU capability tiers>`).

Dense foliage
=============
Instanced visuals with alpha-masked foliage materials use a dedicated path with
no extra API calls:

* Compatible foliage draws in the color and shadow passes are merged into
  batched runs. Other objects between them split a run instead of disabling
  batching.
* Visibility, and for larger plants the LOD level, are chosen per instance and
  per submesh from the automatic mesh LOD chain. A level is used when its
  projected simplification error is at most 2.25 pixels, or 2.5 shadow-map
  texels in shadow passes. ``RAYRAI_FOLIAGE_LOD_PIXEL_ERROR`` and
  ``RAYRAI_FOLIAGE_SHADOW_LOD_PIXEL_ERROR`` override these budgets for the whole
  process (values in ``(0, 8]``).
* With OpenGL 4.3 and MSAA, passes that contain a foliage visual with 2048 or
  more instances cull foliage against the depth already rendered in the same
  frame, tested in 8-pixel tiles that cover every sample.
* Foliage casts shadows by default. Call ``setCastsShadows(false)`` on an
  ``InstancedVisuals`` to exclude it; wind and grass-patch configuration keep
  that choice. When the main viewer refreshes cascaded shadows every frame,
  foliage that cannot shadow anything inside a cascade's part of the view is
  skipped for that cascade.
* With asynchronous mesh loading (:doc:`Capture`), base geometry and instanced
  LOD chains are prepared on worker threads. Each ``pollAsyncMeshLoads(n)`` call
  advances at most ``n`` import/upload stages (``0`` does nothing) and uploads
  at most one prepared submesh per pending instanced visual;
  ``pendingAsyncMeshLoadCount()`` includes these jobs. Prepared LOD chains are
  cached on disk (see `On-disk caches`_).
* ``rayrai/Bc7TextureStorage.hpp`` provides an opt-in load-time helper,
  ``raisin::compressRgbTextureBc7(texture)``, that re-stores an RGB8/SRGB8
  texture and all of its mip levels as BC7. It needs OpenGL 4.2 or
  ``GL_ARB_texture_compression_bptc`` (check ``supportsBc7TextureStorage()``)
  and is lossy. Alpha, float, immutable and already compressed textures are
  left unchanged.

Frame submission and measurement
================================
The renderer applies these submission optimizations automatically; none of
them changes the rendered image:

* After shadow maps or the planar reflection were rendered in a frame, one
  ``glFlush`` submits them so the GPU works on them while the CPU encodes the
  scene pass. This lowers frame time at the cost of some process CPU time.
  The ``RAYRAI_DEBUG_DISABLE_PASS_FLUSH=1`` debug switch skips this flush.
* The environment background is drawn inside the scene pass. After an opaque
  depth prepass it is depth-tested, so only pixels that no surface covers are
  shaded. ``environmentBackgroundDiagnostics()`` reports how the last scene pass
  drew it.
* The transparent pass skips the proximity-fade depth copy and the screen-space
  refraction capture when no drawn material samples them.
* An external ``Camera`` keeps its multisampled render target between frames;
  the next render of that camera without MSAA releases it.
  ``Camera::setSceneMsaaSamples(samples, keepAllocation)`` exposes the same
  behavior.

While ``RenderQualitySettings::geometryRefraction`` is on, the frame caches that
let unchanged external-camera renders reuse earlier results are bypassed, so
every render is drawn in full.

``captureViewerPassTimings(width, height)`` renders the built-in viewer camera
once with blocking GPU timer queries (the renderer's context must be current)
and returns the ``shadow``, ``scene_color_and_resolve`` and ``postprocess``
passes. ``captureRenderPassTimings`` does the same for an external camera. When
foliage instanced visuals are present, both render a second frame and append
``shadow_foliage*`` and ``scene_foliage*`` entries that are nested inside the
coarse passes. ``shadowCacheDiagnostics()``, ``foliageRenderDiagnostics()`` and
``conservativeOcclusionCullingDiagnostics()`` report shadow-map reuse and
culling without extra GPU work.

Shared mesh buffers across contexts
===================================

OpenGL contexts in one share group reuse the immutable vertex/index buffers of
the built-in primitives and of file-backed meshes whose materials use no
textures and no ``nextPass`` chain. To create such contexts, make the root SDL
OpenGL context current before calling ``RayraiWindow::createOffscreenGlContext``
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
``RayraiWindow::detachOffscreenGlContext(window)`` before a worker adopts it
with ``makeOffscreenContextCurrent``. The cache retains weak references, so
reuse requires an existing owner to keep the mesh alive.

The public diagnostics can be read without changing rendering behavior:

.. code-block:: cpp

    const auto stats = raisin::RayraiGlobalAsset::sharedGpuAssetStats();
    const auto uploadedBytes = stats.gpuBufferBytesUploaded;
    const auto reusedBytes = stats.gpuBufferBytesReused;
    const auto templates = stats.cachedSnapshots;

The byte counts are cumulative mesh-buffer counters, not total or live GPU
memory. Textures, VAOs, and render targets are excluded. ``cachedSnapshots`` is
the number of published mesh templates currently held, one per distinct asset
and share group. Use ``resetSharedGpuAssetStats()`` before a controlled
measurement. The cache is enabled by default;
``setSharedGpuAssetCacheEnabled(false)`` makes every later mesh lookup upload
its own buffers.

Shader binary cache and multi-threaded prewarming
=================================================

rayrai's PBR shaders compile in 10-30 seconds on first run depending on the
GL driver. Two features (available since v2.3.0) remove that wait in
production RL pipelines:

1. **Persistent shader binary cache.** Compiled GL programs are written to a
   per-driver cache directory and reloaded on subsequent runs.
2. **Static prewarm helper** for multi-threaded RL setups where many worker
   threads will each construct their own ``RayraiWindow`` — one thread can
   pre-compile every shader, then every worker thread reads from the cache.

Both are wired through the ``RayraiWindow`` constructor:

.. code-block:: cpp

    raisin::RayraiWindow viewer(
      world,
      /*width=*/1280, /*height=*/720,
      raisin::RayraiWindow::ThreadingMode::SingleThread,
      /*shaderCompileThreadCount=*/1,            // 1 = matches rayrai's default
      /*shaderBinaryCacheEnabled=*/true,         // on by default
      /*shaderBinaryCacheDirectory=*/"",          // empty → default location
      /*logShaderBinaryCache=*/false);

* ``shaderCompileThreadCount`` is passed to ``glMaxShaderCompilerThreadsARB``
  on drivers that expose ``GL_ARB_parallel_shader_compile`` and is ignored
  elsewhere. Raise it only when the host application is itself multi-threaded.
* ``shaderBinaryCacheEnabled = true`` (the default) writes program binaries
  under the cache directory keyed by GL vendor / renderer / version + GLSL
  source, when the driver reports at least one program binary format. On the
  next run, identical configurations are loaded directly, skipping the GLSL
  compile entirely. Several processes can share one cache directory.
* ``shaderBinaryCacheDirectory = ""`` uses the first writable directory in this
  order, then falls back to ``<temp>/raisim/rayrai``:

  * Linux: ``$XDG_CACHE_HOME/raisim/rayrai``, ``~/.cache/raisim/rayrai``,
    ``~/.raisim/rayrai``
  * macOS: ``$XDG_CACHE_HOME/raisim/rayrai``, ``~/Library/Caches/raisim/rayrai``,
    ``~/.cache/raisim/rayrai``, ``~/.raisim/rayrai``
  * Windows: ``%LOCALAPPDATA%\RaiSim\rayrai``, ``%HOME%\.raisim\rayrai`` (when
    ``HOME`` is set), ``%USERPROFILE%\AppData\Local\RaiSim\rayrai``,
    ``%USERPROFILE%\.raisim\rayrai``

* ``logShaderBinaryCache = true`` prints every hit/miss/store to stderr —
  useful for verifying the cache is actually being consulted.

The compile thread count and the cache settings are process-wide: each
``RayraiWindow`` constructor applies its arguments to the whole process, so pass
the same values to every renderer, including a prewarmer.

Cache statistics (per process, across all ``Shader::compile`` calls):

.. code-block:: cpp

    auto s = raisin::Shader::binaryCacheStats();
    std::printf("hits=%llu misses=%llu stores=%llu coordinated_waits=%llu\n",
                static_cast<unsigned long long>(s.hits),
                static_cast<unsigned long long>(s.misses),
                static_cast<unsigned long long>(s.stores),
                static_cast<unsigned long long>(s.coordinatedWaits));
    raisin::Shader::resetBinaryCacheStats();  // scope a measurement window

Pre-warming for parallel RL can be done with the ``rayrai_shader_prewarm``
utility when it is available in the rayrai package. The utility creates an
offscreen context and compiles every built-in shader into the persistent binary
cache:

.. code-block:: bash

    rayrai_shader_prewarm --cache-dir /tmp/rayrai-cache --compile-threads 4 --multi-thread
    rayrai_shader_prewarm --log-cache

For an embedded application, create an offscreen context on the main thread,
release it, and call ``prewarmShadersForCurrentContext`` once from a background
thread that adopts it. SDL windows must be created and destroyed on the main
thread, and every renderer in the process, including the prewarmer, must use
the same ``ThreadingMode``. Worker threads that later construct their own
``RayraiWindow`` will find the heavy shaders already cached (or briefly block on
a per-shader compile mutex if the warm-up is still in flight):

.. code-block:: cpp

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

Pass the workers' cache directory as the second argument of
``prewarmShadersForCurrentContext`` when they do not use the default location.
With the binary cache enabled, each registered program is compiled from GLSL
once per process, however many worker threads construct renderers.

When you only want to warm a specific subset:

.. code-block:: cpp

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

For diagnostics on the cost of the lazy-compile pipeline (how long each
program took on first use, how many are linked so far), call
``shaderWarmupDiagnostics()``.

On-disk caches
==============
Besides the shader binary cache, rayrai keeps these caches. All of them are
regenerated when missing, so the files can be deleted at any time. The mesh
caches are also checked against their source and rebuilt when stale.

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Contents
     - Default location
     - Override
   * - Imported-mesh metadata and generated LOD chains
     - ``<temp>/rayrai_mesh_preprocess_cache``
     - ``RAYRAI_MESH_PREPROCESS_CACHE_DIR``
   * - LOD chains prepared by asynchronous loading of instanced meshes
     - ``rayrai_cache_<file>.lods`` next to the source asset, else
       ``<temp>/rayrai_async_lod_cache``
     - ``RAYRAI_ASYNC_LOD_CACHE_DIR``
   * - Procedural cloud noise textures
     - ``$HOME/.raisim/rayrai``, then ``%USERPROFILE%\.raisim\rayrai``
       (Windows), then ``<temp>/raisim/rayrai``
     - ``RAYRAI_CLOUD_TEXTURE_CACHE_DIR``

``<temp>`` is the system temporary directory. The asynchronous LOD sidecars
store full vertex data and can be large for dense assets.

Environment variables
=====================
rayrai reads these variables from the process environment; set them before the
renderer starts. Boolean switches are on when set to a non-empty value that does
not start with ``0``, unless noted otherwise.

* ``RAYRAI_MESH_PREPROCESS_CACHE_DIR``, ``RAYRAI_ASYNC_LOD_CACHE_DIR``,
  ``RAYRAI_CLOUD_TEXTURE_CACHE_DIR``: cache directories (see
  `On-disk caches`_).
* ``RAYRAI_FOLIAGE_LOD_PIXEL_ERROR`` and
  ``RAYRAI_FOLIAGE_SHADOW_LOD_PIXEL_ERROR``: foliage LOD error budgets
  (defaults ``2.25`` pixels and ``2.5`` texels; read once per process). The
  shadow budget follows the color budget when only the latter is set.
* ``RAYRAI_SUPPRESS_<PATH>_FALLBACK_WARNINGS``: silence the one-time warning of
  a GPU path fallback, for example
  ``RAYRAI_SUPPRESS_GPU_INSTANCE_CULLING_FALLBACK_WARNINGS``.
* ``RAYRAI_FORCE_COMPACT_PBR_SAMPLER_FALLBACK`` and
  ``RAYRAI_FORCE_LIMITED_PBR_SAMPLER_FALLBACK``: select the compact or limited
  shader tier on a larger GPU, for testing.
* ``RAYRAI_EAGER_SHADER_WARMUP``: compile every registered shader when a
  renderer is constructed instead of on first use.
* ``RAYRAI_PBR_DEBUG_OUTPUT``: PBR debug visualization channel (``0`` is off).
  ``RayraiWindow::setPbrDebugOutputMode`` overrides it for one renderer, and
  ``captureDebugPasses`` overrides it for the frames it renders.

The TCP viewer reads its own ``RAYRAI_TCP_VIEWER_*`` variables
(:doc:`../RayraiTcpViewer`). Variables named ``RAYRAI_DEBUG_*`` and
``RAYRAI_LOG_*`` are renderer-development diagnostics (debug switches and
logging) and may change between releases. Apart from these debug switches,
optimizations have no switches to turn them off.

Additional tips
===============

* Every ``update`` and external-camera render calls
  ``RayraiWindow::updateObjectLists()``, so RaiSim objects added to or removed
  from the world are picked up automatically. Call it yourself only when the
  renderer's object lists must be current before the next render.
* Use ``setShowCollisionBodies(true)`` for debug visualization of collision shapes.
* If you want rayrai overlays to use your ImGui font, pass it via ``setExternalFont``.

API
====

Core types
**********

.. doxygenclass:: raisin::RayraiWindow
   :members:

.. doxygenclass:: raisin::Camera
   :members:

.. doxygenclass:: raisin::RayraiGlobalAsset
   :members:

Visuals and geometry
********************

.. doxygenclass:: raisin::Visuals
   :members:

.. doxygenclass:: raisin::InstancedVisuals
   :members:

.. doxygenclass:: OpenGLMesh
   :members:

.. doxygennamespace:: raisin::assimp
   :members:

Scene helpers
*************

.. doxygenclass:: raisin::PointCloud
   :members:

.. doxygenclass:: raisin::CoordinateFrame
   :members:

.. doxygenclass:: raisin::CameraFrustum
   :members:

.. doxygenenum:: raisin::VisualCategory

Rendering and materials
************************

.. doxygenclass:: raisin::Light
   :members:

.. doxygenclass:: raisin::Material
   :members:

.. doxygenstruct:: raisin::RenderQualitySettings
   :members:

The per-render toggles passed to the external-camera, capture, and timing
APIs are the nested struct :cpp:struct:`raisin::RayraiWindow::RenderOverrides`,
documented with ``raisin::RayraiWindow`` above.

.. doxygenenum:: raisin::RenderQualityPreset

.. doxygenenum:: raisin::ViewerColorMode

.. doxygenenum:: raisin::ColorGradePreset

.. doxygenenum:: raisin::PostProcessDebugMode

.. doxygenenum:: raisin::GeometryRefractionBackend

.. doxygenstruct:: raisin::GeometryRefractionDiagnostics
   :members:

.. doxygenstruct:: raisin::GeometryRefractionSamplingStatistics
   :members:

Weather, scene effects, and reflections
***************************************

.. doxygenstruct:: raisin::WeatherSettings
   :members:

.. doxygenenum:: raisin::WeatherPreset

.. doxygenstruct:: raisin::LocalFogVolume
   :members:

.. doxygenstruct:: raisin::ProjectedDecal
   :members:

.. doxygenstruct:: raisin::IrradianceVolume
   :members:

.. doxygenstruct:: raisin::ReflectionProbe
   :members:

.. doxygenstruct:: raisin::ReflectionProbeCaptureSettings
   :members:

.. doxygenstruct:: raisin::ReflectionProbeFilterSettings
   :members:

.. doxygenstruct:: raisin::ReflectionProbeBlend
   :members:

.. doxygenstruct:: raisin::ReflectionProbePlacementSuggestion
   :members:

Render quality, weather, capture, reflection-probe, and diagnostics helpers are
exposed through ``raisin::RayraiWindow`` and the public headers included by
``rayrai/RayraiWindow.hpp``. The sections above document the intended entry
points and the fast-path boundaries for those APIs.

RaiSim integration
******************

.. doxygenclass:: raisin::RaisimObject
   :members:

.. (BufferReader doxygen entry moved to :doc:`../RayraiTcpViewer`.)
