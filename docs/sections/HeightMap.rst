#############################
Height Map
#############################

A height map is a grid of points that are triangulated to form a surface.
Since it is axis-aligned, collision checking is very efficient and it is a recommended way to create terrain.
A height map is always a static object. Height samples are stored row by row:
the sample at grid index ``(x, y)`` is ``height[y * xSamples + x]``.

Heightmap collision requires identity orientation. Translating a heightmap is
supported: ``setPosition(x, y, z)`` moves its center to ``(x, y)`` and offsets
all heights by ``z``. ``setOrientation`` accepts only the identity, and
collision detection raises a fatal error and terminates the simulation process
if a heightmap is rotated. These checks are active in every build
configuration. Use a mesh when the terrain surface itself must be rotated.

You can query the surface at a world-frame ``(x, y)`` position:

* height: :code:`getHeight(x, y)` (visual height) and
  :code:`getContactHeight(x, y)` (collision surface; the two differ only after
  a visual-only update)
* normal vector: :code:`getNormal(x, y, normal)` (collision surface, written to
  the output argument)

Runtime updates
===============

``update(...)`` changes the visual and collision height data; it cannot change
the number of samples. The supplied height vector must contain exactly
``getXSamples() * getYSamples()`` finite entries; invalid input is ignored with
a warning and leaves the existing map unchanged.

For visualization-only streaming, ``updateVisualHeight(...)`` replaces the
full visual height array and ``updateVisualHeightPatch(...)`` updates an
inclusive rectangular sample range without rebuilding collision. Both require
the center and size arguments to match the current ones. The patch method
receives the full row-major height array and copies only the requested range. Per-vertex colors use ``setColor(...)``; after a full color map has been
initialized, ``setColorPatch(...)`` updates a tightly packed rectangular color
range.

Example
=============================

.. toctree::
   :maxdepth: 2

   HeightMap_example_png
   HeightMap_example_raw_values
   HeightMap_example_terrain_generator
   HeightMap_example_txt

API
====

HeightMap
**********

.. doxygenclass:: raisim::HeightMap
   :members:

Terrain properties
******************

.. doxygenstruct:: raisim::TerrainProperties
   :members:
