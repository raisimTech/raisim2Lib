#############################
Single-Body Objects
#############################

``raisim::SingleBodyObject`` is an object with only one rigid body.
``raisim::Compound`` is also a SingleBodyObject because its components move together as one rigid body.
All SingleBodyObjects have **6 degrees of freedom**: 3 for position and 3 for orientation.

Primitives
=========================
The following five primitive shapes are supported in RaiSim.

.. image:: ../image/SingleBodyObjects.png


Compound
===================================
An example can be found `here <https://raisim.com/sections/RaisimUnity.html>`__.

``raisim::Compound`` has multiple primitive shapes that are rigidly attached to each other to form a single rigid body.
The shapes do not have to overlap to stay attached.

A compound object can be added to the world using the method ``raisim::World::addCompound``.
This method takes a vector of children, which have their own shape, material, position, and orientation.
The shape can be specified by a geometric type (i.e., ``raisim::ObjectType``) and its size parameters (``objectParam``).
The ``objectParam`` follows a standard way to represent the size of a primitive in RaiSim:

*  Sphere: radius, 0, 0, 0
*  Box: x, y, z, 0
*  Capsule and cylinder: radius, height, 0, 0

The ``objectParam`` is an instance of ``raisim::Vec<4>``.
All shapes require fewer than four parameters and the unused elements (i.e., the zeroes above) are ignored.

The ``trans`` member defines the position and orientation of the child in the body frame.
It is a ``struct`` with public ``rot`` and ``pos`` members.

The ``mass``, ``COM``, and ``inertia`` arguments specify the dynamical properties of the combined body.

Mesh
===================================
.. image:: ../image/mesh.png
  :alt: mesh
  :width: 300

``raisim::Mesh`` represents a single-body object defined by a triangle mesh
(typically from an OBJ file). It is commonly used for static or kinematic
geometry such as props, terrains, or scanned environments. Meshes are added
via ``raisim::World::addMesh`` with several overloads:

* Explicit inertia/COM: ``addMesh(meshFile, mass, inertia, com, scale, material, collisionMode, collisionGroup, collisionMask)``.
* Auto inertia/COM: ``addMesh(meshFile, mass, scale, material, collisionMode, collisionGroup, collisionMask)``,
  which estimates inertia and COM from the scaled axis-aligned bounding box.
* Clone: ``addMesh(existingMesh, material, collisionGroup, collisionMask)`` to reuse a loaded mesh.

``collisionMode`` selects the mesh collision representation:

* ``MeshCollisionMode::ORIGINAL_MESH``: uses the original triangle mesh.
* ``MeshCollisionMode::CONVEX_HULL``: uses a single convex hull.

The ``addMesh`` overloads default to ``MeshCollisionMode::ORIGINAL_MESH`` unless specified.
Convex decomposition (formerly ``CONVEX_SPLIT``) is no longer available in the API.
If you need a decomposed collision shape, pre-decompose the mesh and load the
parts as multiple meshes (or as a Compound) instead.

Mesh collision is supported for OBJ files; keep collision meshes reasonably sized for performance.




SingleBodyObject API (Parent class)
======================================

.. doxygenclass:: raisim::SingleBodyObject
   :members:


Compound API
=========================

.. doxygenclass:: raisim::Compound
   :members:

Sphere API
=========================

.. doxygenclass:: raisim::Sphere
   :members:

Box API
=========================

.. doxygenclass:: raisim::Box
   :members:

Capsule API
=========================

.. doxygenclass:: raisim::Capsule
   :members:

Cylinder API
=========================

.. doxygenclass:: raisim::Cylinder
   :members:

Ground API
=========================

.. doxygenclass:: raisim::Ground
   :members:

Mesh API
=========================

.. doxygenclass:: raisim::Mesh
   :members:
