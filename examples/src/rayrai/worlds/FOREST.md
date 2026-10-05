# Dense forest terrain and physics

![Grass-covered forest terrain with Raisim objects](../../../../rsc/docs/image/forest.png)

The saved-scene viewer uses the same forest assets and placements:

![Forest loaded from rayrai_forest.rscene](../../../../rsc/docs/image/forest_rscene.png)

`rayrai_forest` is a small native Rayrai example with an **80 × 80 m** rolling
heightmap, hills, a gully, and continuous grass coverage. The terrain is static;
“dynamic” here means varied terrain shape, with animated foliage and simulated
objects. The same 161 × 161 heightmap supplies both collision and plant heights.

- **1,888 trees:** 880 pine saplings, 880 fir saplings, and 128 broadleaf wild syringa.
- **61,200 ground-cover instances:** 18,000 each of Bermuda grass and the two
  medium grasses, plus 1,800 each of fern,
  dandelion, nettle, and periwinkle. This adds the requested two trees and five
  ground-cover types to the original forest's one tree and two grasses.
- **180 mossy rocks:** six Poly Haven shapes with varied sizes and orientations,
  shared textures, instancing, automatic LOD, and shadows.
- **100 dynamic Raisim primitives:** the original six crates and six balls,
  72 colorful boxes, spheres, and capsules stacked along the camera's trail,
  and 16 balls dropped over nearby tree trunks to show capsule collisions.
- Instanced batches, automatic mesh LOD, projected-size thinning, foliage shadow
  LOD, shadow casting enabled by default for every vegetation batch, three directional
  shadow cascades, 4× MSAA, wind, and brown soil with scattered leaf litter
  ([Mud Forest](https://polyhaven.com/a/mud_forest)). The ground uses subdued
  normal relief and high roughness to blend with the vegetation and mossy rocks.
- Strong direct sunlight (diffuse/specular strength 18, previously 5)
  and base exposure 1.1 give exposed leaves bright, near-white highlights under
  the ACES tone curve. Ambient fill stays unchanged to preserve deep canopy
  shadows; no light-shaft effect is added.
- A subtle cool-gray haze separates distant foliage, using 1,200 m weather
  visibility and the existing height fog. Nearby leaves retain their contrast.

The collision debug capture shows the 100 primitives from the saved scene with
ground cover and broadleaf visuals hidden for clarity. All tree and rock
collision bodies remain active in this view.

![Colorful forest primitives along the trail](../../../../rsc/docs/image/forest_collision_primitives.png)

Grass and small ground cover remain visual-only. Each of the 1,888 trees has one
to three hidden static capsules fitted to its woody centerline, and each of the
180 rocks has a convex collision mesh with about 100 triangles. The terrain and 100 dynamic physics objects retain their own
collision shapes. Each tree and rock type ships a `model.rasset` descriptor that
pairs `model.gltf` with its local collision shape. The scene loads nine descriptors
once, then applies the same scatter transforms to the visuals and proxies.
Every imported plant is individually rooted at local Z=0.
Display-lineup translations are removed before scattering, preventing floating
clusters on hillsides. Grass and ground cover span the entire heightmap, including
the former trail, steep slopes, terrain edges and the area around the physics
objects. The terrain has no contrasting trail tint. Grass increases from 43,200
to 54,000 clumps to maintain density across the larger covered area. Trees stay
upright and avoid steep slopes, the view corridor and the physics clearing.
Crowns overlap for a dense canopy,
while tree roots remain at least 0.8 m apart. Trees extend closer to the
view corridor than in the original sparse layout. Rocks avoid tree trunks and each
other; their rounded bases are embedded using terrain samples around the contact
patch. More rocks appear along the view corridor. Their complete footprints
stay outside the corridor and physics clearing, which now also contain grass.

The example consists of [the viewer](rayrai_forest.cpp),
[viewer setup](forest_viewer.hpp), and a
[scene helper](forest_scene.hpp), with a
[tree proximity index](forest_tree_index.hpp) for placement. The `.rasset` format is a versioned XML
asset descriptor: one `<visual file="..."/>` and one or more local
`<collision type="cylinder|capsule|sphere|box|convex_mesh" .../>` elements.
Paths resolve relative to the descriptor. Collision shapes are invisible in
Rayrai and stay in the physics world; missing assets and invalid dimensions
are errors. `examples/tools/generate_forest_rassets.py` regenerates tree capsule
fits from the woody glTF primitives and convex rock OBJ proxies. The rock
triangle target defaults to 100 and can be changed with `--rock-triangles`;
the actual face count can differ slightly because a convex hull has a discrete
number of vertices. Asset
preparation requires NumPy and SciPy; running the packaged example does not.
The same forest is also saved as
[`rayrai_forest.rscene`](../../../../rsc/forest/rayrai_forest.rscene). It contains
the sampled terrain, exact deterministic plant and rock transforms, 2,068
hidden `.rasset` collision objects, 16 instanced visual batches, and the
100 dynamic primitives. Its roughly 15 MiB of scene text references the asset
files in this directory without copying meshes or textures. A matching copy in the adjacent Raisim checkout
is the Engine2 project `raisim_engine2/examples/rayrai_forest/`, whose default
scene and asset directory point back to these assets. The `rayrai_forest_rscene`
executable in that checkout opens the project through Engine2 so it can be
compared with this native viewer.
Open either scene in Engine2 to edit collision assets and visual batches.
Neither viewer needs runtime Python. A small
[loading overlay](../../../include/rayrai_loading_progress.hpp)
shows asset progress across the top while meshes import asynchronously and
finish GPU uploads. It disappears automatically once all assets are ready,
and does not intercept camera input. Physics waits until loading finishes.
Scattering uses a fixed seed and portable integer
random-number conversion. Simulation runs eight 2 ms steps per rendered frame
on the calling thread, including during benchmarks. The example does not add
physics worker threads.

Tree spacing and rock/tree exclusion use a spatial index to avoid scanning
unrelated plants. Every candidate still uses the original `std::hypot` and
strict radius comparison. Random draws, accepted transforms, instance order,
physics and renderer settings remain the same. This reduces scene construction
time; it does not change per-frame rendering or its quality. The regression
check compares every transform field bit-for-bit with the original scatter,
including mixed plant order and floating-point bin/radius boundaries.

The renderer also reuses exact instance transform/color records across changing
scene and shadow selections for single-part batches of up to 32,768 instances.
This includes each 18,000-instance grass batch. The cache stores the existing
computed values; instance and distance-fade edits invalidate them. It adds up to
2.5 MiB of record storage per eligible single-part batch (about 1.37 MiB for
18,000 instances), plus validity flags. Geometry, draw order, shaders, LOD,
wind, lighting and MSAA settings are unchanged.

When these cached records are collected into a draw batch, the renderer copies
complete records directly into the retained batch bytes. Compile-time size and
member-offset checks keep the source and destination layouts identical. This
removes an intermediate record without recomputing transforms, changing
floating-point arithmetic or adding cache storage. Invalidated records retain
the original computation path.

Individually selected foliage LODs also reuse their exact color-LOD choice
across passes sharing the same camera and LOD inputs. Each shadow cascade still
performs its own visibility tests and shadow LOD selection. Camera position,
FOV/viewport size, instance edits, source bounds, wind inflation, triangle
counts and simplification errors invalidate the color choice when needed.
The cache adds one byte per rendered instance and source mesh part; it preserves
the original distance arithmetic and LOD thresholds.

Large ground-cover groups with 8,192–32,768 visible instances use packed depth
keys and instance IDs during sorting on 64-bit hosts. Later radix passes read
those records sequentially instead of loading and converting depths again.
The existing scratch vectors hold the packed records, and the final draw IDs
retain the exact depth/ID ordering, including ties and signed zero. Other group
sizes and 32-bit hosts retain the existing ID-based radix path. For packed groups, unsorted IDs,
IDs above the 32-bit range and NaNs retain the comparison-sort path.
`rayrai_foliage_instance_sort_test` checks both size boundaries and arbitrary
float patterns against comparison sorting; its `--benchmark` option measures
sorting with one worker. The GPU preparation test also checks fully visible
groups across the boundaries against an independent depth/ID comparison sort,
including mirrored scales, edits, and exact color and depth output.

For meshes with several parts, visibility diagnostics count each instance once
across overlapping LOD groups. Batches of up to 1,024 candidate instances
(after the render-count limit, before culling and stride) use byte flags, covering the forest's tree batches. Larger batches
retain packed flags. Each eligible batch uses at most 1 KiB of byte flags; the
count and draw records remain exact. The preparation test checks both sides
of the boundary, shrink/regrow, render limits, strides and cached selections.
This reduces CPU bookkeeping.

The renderer's `rayrai_foliage_preparation_test` compares cached and uncached
GPU records, RGBA bytes and depth bytes exactly. It covers the 18,000-instance
batch size, both sides of the cache limits, edits and cache re-enabling. Its
`--benchmark-selection` option measures CPU preparation with a moving camera
and four scene/shadow passes. `--benchmark-lod-search` exercises detailed,
mixed-LOD instances; `--benchmark-shadow-lods` includes independent shadow LOD
selection. The benchmark moves both the frustum and the LOD camera position. Run benchmarks individually with one worker;
this preparation timing is not an end-to-end forest FPS measurement.
`rayrai_foliage_color_lod_cache_gpu_test` runs the color-cache byte comparisons
independently via `rayrai_foliage_preparation_test --color-lod-cache`.
`rayrai_foliage_color_lod_cache_test` checks cache reuse and every invalidating
input, including rounding boundaries, tied/nonmonotonic LOD chains and special
floating-point values. The preparation GPU test additionally compares index
geometry, instance records, color and depth exactly across color/shadow passes,
camera and policy changes, mirrored scales and instance edits.

## Build and run

Use the checkout's Raisim/Rayrai packages and a valid Raisim activation key.
From the `raisim2Lib` root on Linux:

```sh
cmake -S examples -B /tmp/raisim-forest-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-20 \
  -DRAISIM_PREFIX="$PWD/raisim" -DRAYRAI_PREFIX="$PWD/rayrai" \
  -DRAISIM_FOREST_COLLISION_BENCHMARK=ON
cmake --build /tmp/raisim-forest-build --target rayrai_forest forest_collision_benchmark -j12
/tmp/raisim-forest-build/rayrai_forest
```

Forest tests live in the adjacent `raisim/test/examples` directory and are
registered by the Raisim root build when this checkout is present. Configure
that build with `RAISIM_TEST=ON` and `RAISIM_RAYRAI_TEST=ON`; set
`RAISIM_FOREST_GPU_TESTS=ON` to include GPU checks.

```sh
cmake -S ../raisim -B /tmp/raisim-forest-tests -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++-20 -DRAISIM_TEST=ON \
  -DRAISIM_RAYRAI_TEST=ON -DRAISIM_FOREST_GPU_TESTS=ON
cmake --build /tmp/raisim-forest-tests --target \
  forest_scene_test forest_scatter_test forest_loading_test forest_render_test forest_shadow_test -j12
ctest --test-dir /tmp/raisim-forest-tests -j12 --output-on-failure -R '^forest_'
```

The target also participates in the normal top-level examples build. Use the
platform's normal compiler/generator on macOS and Windows; the example has no
Linux-specific API or architecture-specific flags. Those platforms have not
been runtime-tested for this addition.

With the Raisim checkout beside `raisim2Lib`, build its
`rayrai_forest_rscene` target and run it to view the saved scene. The
`rayrai_forest_rscene_export` target regenerates both `.rscene` files from
this example's `forest_scene.hpp`; see
`raisim_engine2/examples/rayrai_forest/README.md` in that checkout for commands.

The asset path defaults to this checkout's `../../../../rsc/forest`. To relocate
an executable, copy that directory and pass `--assets /path/to/forest`.
Use the viewer's standard mouse/keyboard camera controls to explore.

Bounded rendering, the placement benchmark and the collision benchmark run from
the test and example builds above:

```sh
/tmp/raisim-forest-tests/test/examples/forest_render_test --frames 3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/raisim-forest-tests/test/examples/forest_scatter_test --benchmark
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/raisim-forest-build/forest_collision_benchmark
```

After preparing the source glTF files, regenerate collision proxies with
`python3 examples/tools/generate_forest_rassets.py rsc/forest --rock-triangles 100`.

GPU tests and hidden runs still require a desktop OpenGL context. Run GPU tests
serially and one viewer at a time. The completed-frame FPS that
`forest_render_test` prints includes simulation, rendering, UI, swap, and GPU
completion, after all assets finish loading and 60 subsequent warm-up frames;
loading is excluded. The first load can take tens of seconds because Rayrai
builds LODs for the high-detail meshes. Those exact prepared meshes are saved
beside each asset as `rayrai_cache_model.gltf.lods`. Later launches read the
cached levels on a worker instead of rebuilding them. Async loading and the
progress bar remain active throughout.

The 16 optional cache files add **282 MiB** to the original 131.5 MiB bundle.
They are generated locally, ignored by Git, and can be deleted to reclaim space;
Rayrai recreates them on demand. Full geometry fingerprints detect changed
external buffers and LOD-affecting material settings. Other material and texture
properties come from the current import. Invalid or damaged caches are ignored.
A read-only asset directory uses the system temporary cache directory instead.
No vertex precision, geometry, LOD settings or rendering quality is reduced.

Set `RAYRAI_ASYNC_LOD_CACHE_DIR` to choose a cache directory; a new, empty
directory reproduces a first launch.

Import, base-geometry preparation, and instanced LOD generation run on workers.
Material/texture resolution and incremental GPU uploads use the render thread.
The loading bar includes all preparation stages. `forest_render_test` rejects
a loading update longer than one second. Short upload or texture-resolution
hitches remain possible.
The supplied preview shows the scene shortly after the objects drop. The example
renders interactively and has no capture, frame limit, hidden mode, or timing
code. Bounded rendering and benchmarks use the separate `forest_render_test`
runner, sharing the same scene setup.

## Loading the forest from `rayrai_forest.rscene`

`rayrai_forest` places every tree, plant and rock in C++. The
[`rayrai_forest_from_rscene`](rayrai_forest_from_rscene.cpp) example builds the
same world from the saved
[`rayrai_forest.rscene`](../../../../rsc/forest/rayrai_forest.rscene)
in a few lines, without RaiSim Engine:

```cpp
auto world = std::make_shared<raisim::World>("rayrai_forest.rscene");
raisin::RayraiWindow viewer(world, 1280, 800);
raisin::applyRscene(*world->getRscene(), viewer);
```

`raisim::World(path)` creates the terrain, the 2,068 hidden `.rasset` tree
and rock bodies and the 100 props. `raisin::applyRscene` applies the render
settings, sky, sun, terrain texture, 16 instanced batches and saved camera.
The example takes no command-line arguments; like the other Rayrai examples,
it finds `rsc/forest/rayrai_forest.rscene` with `exampleRscPath`. The file's
render settings can be changed in C++ first:

```cpp
auto render = raisin::rsceneRenderSettings(*world->getRscene());
render.quality.viewerMsaaSamples = 8;
raisin::applyRscene(*world->getRscene(), viewer, render);
```

The scene stores `raisim::World`'s default solver settings (ERP 1.5, 150
iterations, fixed iteration order, and the default contact material), so the
props fall exactly as in the native example. The Raisim checkout's
`forest_rscene_reader` test compares terrain samples, bodies, every instance
transform and the props' positions after 400 steps against the C++ forest
and requires them to match. Renders match Engine2's `rayrai_forest_rscene`
viewer pixel for pixel once Engine2's float rounding of the sun direction is
applied. Engine2 stores the direction as a quaternion, which changes its last
bit.

`World(path)` takes 277 ms for this scene (146 ms parsing the 15 MiB file,
111 ms creating bodies; `benchmarks --bench rscene_import`, single thread,
median of 5). Content the reader cannot reproduce exactly, such as
articulated systems, parented nodes or non-directional lights, is a fatal
error naming the line. See `docs/rscene.md` in the Raisim checkout.

## Assets and reproduction

The shipped resources occupy approximately **131.5 MiB**. The broadleaf tree
accounts for most of this: its original geometry contains about two million
triangles. Rayrai builds lower-detail representations for rendering. Textures
are 1K, and only one plant from each original model lineup is retained.

All models and terrain textures are from [Poly Haven](https://polyhaven.com),
under its [CC0 asset license](https://polyhaven.com/license).
Powered by Poly Haven. Original download URLs and SHA-256 hashes are recorded
in [sources.json](../../../../rsc/forest/sources.json); the prepared bundle's hashes are in
[manifest.json](../../../../rsc/forest/manifest.json). Asset IDs correspond to
`https://polyhaven.com/a/<asset-id>`.

To reproduce the shipped resources using Python's standard library:

```sh
python3 examples/tools/download_forest_assets.py /tmp/forest-assets
python3 examples/tools/prepare_forest_assets.py /tmp/forest-assets
python3 ../raisim/test/examples/test_forest_assets.py /tmp/forest-assets/prepared
```

Run the viewer with `--assets /tmp/forest-assets/prepared` to inspect the result.
The preparation scripts select a single plant, remove unused plant geometry,
rotate Y-up meshes to Z-up, align the lowest vertex to zero, and add leaf
transmission metadata. Pine/fir twig geometry uses opaque rendering because
its source color texture has no alpha. The shaders keep the actual needles.
Six rock shapes share their original buffers and textures, with display offsets
removed and each shape normalized to a one-metre horizontal bounding radius.
Regeneration downloads the current Poly Haven versions; the shipped source
hashes identify the exact versions used here.

## Validation

- Clang 20 Release build on Linux x86-64.
- CTest: terrain-grid indexing, >7 m relief, all 63,088 root placements,
  grass coverage in every terrain cell, along the former trail and at all edges,
  tree clearing exclusions, minimum tree spacing, type counts, all 180 rock placements
  and footprint exclusions, and two seconds of physics.
- Asset test: texture/buffer references, finite vertices, actual mesh-root
  heights, normalized rock bounds, and every shipped resource checksum.
- Loading UI test: visible on the first loading frame, advances with completed
  assets, stays above the scene, and produces no UI geometry after completion.
- Native GPU smoke test, loading-loop responsiveness regression, and inspected
  interactive preview (captured externally).
- Rendered foliage-shadow regression: every vegetation batch must cast shadows,
  and enabling those shadows must darken at least 0.5% of the image by more
  than 12 average RGB levels, compared with the same frozen scene without them.

Rayrai culls small foliage individually, orders it front to back, batches
compatible color/shadow draws, and reuses their GPU buffers. Large foliage
groups skip completely hidden instances using conservative same-frame depth
that preserves every MSAA sample. The 8-pixel depth tiles and cached camera calculations avoid repeated work.
Tighter source-mesh boxes, transformed for instance rotation/scale and expanded
for wind, now reject more fully hidden plants. Visible geometry, LOD, wind, lighting,
shadows and antialiasing stay unchanged. The depth path requires OpenGL 4.3 and
supported MSAA storage; other contexts retain the previous renderer.

Eligible foliage now stores each uniform vertex color once and fetches full-precision
tangents alongside positions, normals and UVs. Every float value is preserved;
varying colors and older OpenGL contexts retain their previous layout.
