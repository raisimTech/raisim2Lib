#############################
Height Map
#############################

A height map is a grid of points that are triangulated to form a surface.
Since it is a regular grid, collision checking is very efficient and it is a recommended way to create terrain.
A height map is always a static object. Height samples are stored row by row:
the sample at grid index ``(x, y)`` is ``height[y * xSamples + x]``.

Placement
=========

The grid lives in the height map's own frame: the center of the grid is the
origin and the heights rise along the local z axis. ``setPosition(x, y, z)``
moves the center to ``(x, y)`` and offsets all heights by ``z``.
``setOrientation(...)`` turns the whole surface about its center, so a tilted
map is a slope, a map rotated 90 degrees about x is a wall, and one rotated 180
degrees is a ceiling.

.. image:: ../../rsc/docs/image/rayrai/heightmap_rotation.png
   :alt: A rough height map tilted into a slope and a bumpy height map stood up as a wall
   :width: 100%

Two rotated height maps: a slope tilted 20 degrees and a wall at its foot. The
box and the cylinder rest on the slope, held by friction, while the ball (its
descent shown light to dark), the capsule and the small sphere roll and slide
down and stop against the wall.

Everything that touches the surface follows the orientation: contacts with
every shape, ray tests and the sensors built on them, swept CCD, granular
particles, and rendering in rayrai and the TCP viewer. A scene on a rotated map
moves like the same scene on the unrotated map, with every body and gravity
rotated along with it; the two agree to rounding error. A height map is one-sided whatever its
orientation: everything below the surface, along the local -z axis, is inside
the terrain, and a body that ends up there is pushed back out through the
surface.

You can query the surface at a world-frame ``(x, y)`` position. These queries
ignore the orientation: they describe the map with its position applied but
not rotated.

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
