#############################
Height Map
#############################

A height map is a grid of points that are triangulated to form a surface.
Since it is axis-aligned, collision checking is very efficient and it is a recommended way to create terrain.

You can get the following two properties from a height map instance:

* height: :code:`getHeight()`
* normal vector: :code:`getNormal()`

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
