######################
Rayrai Example: Forest
######################

.. image:: ../../../../rsc/docs/image/forest.png
   :alt: rayrai_forest example
   :width: 100%

Overview
========
Renders an 80 m × 80 m rolling heightmap covered by instanced Poly Haven
vegetation: 1,888 trees, 61,200 ground-cover instances, and 180 mossy rocks.
One hundred dynamic props (crates, balls, capsules, and spheres dropped onto
tree crowns) fall onto the same heightmap that places the plants. Ground cover
is visual-only. Trees and rocks collide through static collision proxies
loaded from each asset's ``model.rasset``.

Target
======
CMake target: ``rayrai_forest`` (C++20).

The example needs rayrai 2.7.0 or newer. On Linux, the 2.7.0 package can throw
``std::bad_alloc`` from ``InstancedVisuals::addInstances`` with newer libstdc++
versions; the fix is listed for the next release in :doc:`../../changelog/v2.7`.

Run
===
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/rayrai_forest
   ./build-examples/examples/rayrai_forest --assets /path/to/forest

On Windows, run ``rayrai_forest.exe`` instead. The asset directory defaults to
``rsc/forest`` in the checkout used to configure the build. Copy that
directory and pass ``--assets`` when you move the executable. The example
renders in process and does not need ``rayrai_tcp_viewer``.

Loading the forest from a scene file
====================================
The same world is saved as ``rsc/forest/rayrai_forest.rscene``.
:doc:`rayrai_forest_from_rscene` builds it from that file with
``raisim::World(path)`` and ``raisin::applyRscene`` instead of placing it in
C++, and simulates identically. The reader is new in v2.7.1 (not yet
released); :doc:`../../RsceneFile` describes the file format.

Loading and caches
==================
Meshes load asynchronously while a progress bar across the top of the window
counts the finished assets; physics starts once loading completes. The first
launch builds mesh LODs for the high-detail plants and can take tens of
seconds. Rayrai saves the prepared levels beside each model as
``rayrai_cache_model.gltf.lods`` and reads them on later launches. When the
asset directory is read-only, the cache goes to the system temporary directory
instead. Set ``RAYRAI_ASYNC_LOD_CACHE_DIR`` to choose a cache directory. The
cache files are ignored by Git and can be deleted at any time.

Instance transform records are also cached for single-part foliage batches
with up to 32,768 instances, including the 18,000-instance grass batches.
The cache reuses the original computed values across scene and shadow
selections and is invalidated by instance or distance-fade edits. Each
18,000-instance batch uses about 1.37 MiB for these records, plus validity
flags. Geometry, draw order, shader arithmetic and quality settings stay the
same. This reduces CPU preparation work.

When collecting cached records into draw batches, the renderer copies complete
records directly into the retained batch bytes. Compile-time size and member
offset checks ensure identical source and destination layouts. This removes an
intermediate record without changing floating-point arithmetic or cache storage;
invalidated records still follow the original computation path.

For individually selected foliage LODs, the renderer also reuses the exact
color-LOD choice across passes with identical camera and LOD inputs. Each
shadow cascade retains its own visibility tests and shadow LOD choice. Camera,
instance, bounds, wind and LOD-policy changes invalidate this cache. It stores
one byte per rendered instance and mesh part and keeps the original arithmetic
and thresholds.

On 64-bit hosts, ground-cover groups with 8,192–32,768 visible instances sort
packed depth keys and instance IDs in the existing scratch vectors. This
avoids repeated depth loads and conversion while preserving the exact final
depth/ID ordering, including tied depths and signed zero. Other sizes and
32-bit hosts keep the existing ID-based radix sort. For packed groups, unsorted IDs, IDs above
the 32-bit range and NaNs use the comparison-sort path. The CPU sorting test
covers both size boundaries and arbitrary float patterns; the GPU preparation
test also checks fully visible groups for exact draw order, color and depth.
This reduces sorting work.

Visibility diagnostics for meshes with several parts count each instance once
across overlapping LOD groups. Batches with up to 1,024 candidate instances
(after the render-count limit, before culling and stride) use byte flags, including the forest's tree batches; larger batches
retain packed flags. This adds at most 1 KiB of byte flags per eligible batch
and preserves exact counts and draw records. The preparation test covers both
sides of the boundary, shrink/regrow, render limits, strides and cached
selections. This reduces CPU bookkeeping.

Details
=======
- Uses one 161 × 161 heightmap for both collision and plant placement, with a
  fixed-seed scatter.
- Adds the tree and rock collision proxies with ``raisim::Rasset::load`` and
  ``Rasset::addStaticColliders``.
- Renders ten plant types and six rock shapes as ``InstancedVisuals`` with
  automatic mesh LOD, projected-size thinning, foliage shadow LOD, per-type
  wind, and shadows enabled for every batch.
- Uses the ``High`` preset with ACES tone mapping, 4× MSAA, three directional
  shadow cascades out to 80 m, and clear weather with a distance haze.
- Integrates eight 2 ms physics steps per rendered frame.

Tests
=====
The forest checks live in the RaiSim source repository (``test/examples``) and
are registered when that repository is configured with ``-DRAISIM_TEST=ON``
next to this checkout. They cover terrain placement and physics, the
``.rscene`` reader, and asset integrity (needs Python 3); with
``-DRAISIM_RAYRAI_TEST=ON`` they also cover the loading overlay.
``-DRAISIM_FOREST_GPU_TESTS=ON`` adds rendering smoke tests and foliage-shadow
checks, which need a display and OpenGL. ``FOREST.md`` lists the build and
``ctest`` commands. The scatter regression compares transform fields
bit-for-bit with the original placement algorithm. The Rayrai
``rayrai_foliage_preparation_test`` also compares cached and uncached GPU
instance records, color bytes and depth bytes exactly at the forest batch
size and cache boundaries, including edits and cache re-enabling. Detailed
foliage checks include exact index geometry across color and shadow passes
with changing cameras and LOD policies. The CPU
``rayrai_foliage_color_lod_cache_test`` covers reuse, all invalidating inputs
and floating-point boundaries.
The separate ``rayrai_foliage_color_lod_cache_gpu_test`` runs the color-cache
byte comparisons via ``rayrai_foliage_preparation_test --color-lod-cache``.

The assets are from Poly Haven under CC0; see
``rsc/forest/ATTRIBUTION.md``. ``examples/src/rayrai/worlds/FOREST.md``
documents asset preparation, the build and test commands, and the LOD cache.
