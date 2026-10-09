#################################
Collision Detection and Colliders
#################################

Overview
========
RaiSim performs collision detection in two stages: an AABB-based broadphase
filters candidate pairs, followed by a pair-specific narrowphase that generates
contact points. The narrowphase algorithm depends on the shape pair (analytic
tests, SAT, MPR, or GJK/EPA). Pairs rejected by the collision group and mask
(see :doc:`Contact`) and pairs of two static bodies never reach the narrowphase.

The broadphase is selected with ``contact::BroadphaseSettings::type``
(``World::setBroadphaseSettings``):

* ``contact::BroadphaseType::Sap3Axis`` (default): sweep and prune on three axes.
* ``contact::BroadphaseType::MultiBoxPrune``: multi-box pruning on a uniform
  grid; the ``mbp*`` fields set the grid bounds and cell size.
* ``contact::BroadphaseType::None``: no AABB culling; every pair goes to the
  narrowphase. Use it only for debugging.

Scenes with fewer than 35 collision bodies use an all-pairs AABB test
regardless of this setting.

Global contact limits and merging
=================================
* **Per-pair limit**: :code:`ContactSettings::maxContactsPerPair` (default 8)
  is clamped to **[1, 8]** inside the contact detector. Any pair listed below
  is capped by this limit, even if the narrowphase could generate more points.
* **Merging for mesh/heightmap pairs**: contacts produced by triangle-based
  tests are merged to reduce redundancy. The merge uses a distance threshold
  proportional to the smaller shape size (about 0.105 * characteristic length;
  a heightmap counts as infinitely large, so the other shape sets the scale),
  a normal alignment threshold of about 4 degrees, and a small positional
  epsilon (~1e-4) to coalesce nearly identical points. Mesh vs mesh pairs only
  coalesce points closer than 1e-4.
* **Heightmap manifold reduction**: heightmap pairs first collect up to 8
  contacts. RaiSim then drops contacts that duplicate another contact or lie
  between two contacts with nearly identical normals and add no extra depth,
  and keeps the deepest remaining contacts up to the per-pair limit.
* **Special caps**: some generators have additional caps. Box-cylinder can
  produce up to 16 contacts internally but is still clamped to the per-pair
  limit; cylinder-heightmap accepts at most 3 contacts from each terrain
  triangle.

Shape categories
================
* **Primitive**: sphere, box, capsule, cylinder, plane. A plane is the
  ``World::addGround`` ground. Compound children are ordinary primitive bodies
  and follow the primitive tables. Cones are not supported.
* **ConvexMesh**: convex mesh representation (``MeshCollisionMode::CONVEX_HULL``
  or each part of a ``MeshCollisionMode::CONVEXIFY`` decomposition).
* **Mesh**: non-convex triangle mesh (``MeshCollisionMode::ORIGINAL_MESH``).
* **Heightmap**: grid terrain represented by triangles. It can be translated and rotated
  arbitrarily; contacts are computed in the heightmap's own frame and mapped back to the world.
* **Ray**: query-only shape used by ray tests.

Pair list (narrowphase algorithms and contact counts)
=====================================================
Pairs are symmetric; a single entry covers both A vs B and B vs A.

Primitive vs Primitive
----------------------
.. list-table::
   :header-rows: 1
   :widths: 20 45 20 35

   * - Pair
     - Narrowphase algorithm
     - Contact points
     - Notes
   * - Sphere vs Sphere
     - Analytic distance check; contact on line between centers.
     - 1
     - Normal along center line.
   * - Sphere vs Box
     - Closest point on oriented box (analytic).
     - 1
     - Uses box clamp in local frame.
   * - Sphere vs Capsule
     - Closest point on capsule axis segment, then sphere-sphere contact.
     - 1
     - Segment endpoints are capsule caps.
   * - Sphere vs Cylinder
     - Analytic finite-cylinder test (side and caps).
     - 1
     - Handles endcaps and side wall.
   * - Box vs Box
     - SAT on 15 axes (3 + 3 + 9 cross), face/edge clipping.
     - Up to 8
     - Edge-edge yields 1; face-face yields a clipped manifold.
   * - Box vs Capsule
     - Segment-box closest approach; fallback to cap sphere-box tests.
     - Up to 2
     - 1 if segment interior hits; otherwise up to 2 cap contacts.
   * - Box vs Cylinder
     - SAT + clipping (cylinder vs box face/circle); MPR if clipping yields no
       point.
     - Up to 16 (clamped to the per-pair limit)
     - Produces 1-2 for edge hits, more for face overlap.
   * - Capsule vs Capsule
     - Segment-segment closest points; parallel overlap uses both ends of the
       overlapping axis interval.
     - Up to 2
     - Parallel, overlapping axes can emit two contacts.
   * - Capsule vs Cylinder
     - MPR (Minkowski Portal Refinement) support mapping.
     - 1
     - Generic convex-convex contact.
   * - Cylinder vs Cylinder
     - Nearly parallel axes in end-to-end (cap-to-cap) contact: analytic rim
       sampling; otherwise MPR.
     - Up to 8 (cap-to-cap), otherwise 1
     - Side-by-side contact and non-parallel axes use MPR.

Primitive vs Plane
------------------
Plane contacts use the plane's world-frame normal. The ground plane created by
``World::addGround`` has the normal +Z (world up).

.. list-table::
   :header-rows: 1
   :widths: 20 45 20 35

   * - Pair
     - Narrowphase algorithm
     - Contact points
     - Notes
   * - Sphere vs Plane
     - Analytic projection to plane.
     - 1
     - Contact at the sphere point deepest along the plane normal.
   * - Box vs Plane
     - Projected box corners and face selection.
     - Up to 4
     - Chooses deepest corner then up to 3 additional points.
   * - Capsule vs Plane
     - Analytic endcap projection.
     - 1
     - Uses the lower end of the capsule axis, even when the capsule lies flat.
   * - Cylinder vs Plane
     - Analytic: rim contacts if the axis is parallel to the plane normal or
       the whole lower rim is below the plane; otherwise two edge points.
     - Up to 4 (parallel or fully submerged rim), otherwise up to 2
     - Parallel means the cylinder axis is aligned with the plane normal.

ConvexMesh vs Primitive/ConvexMesh
----------------------------------
.. list-table::
   :header-rows: 1
   :widths: 22 45 18 35

   * - Pair
     - Narrowphase algorithm
     - Contact points
     - Notes
   * - ConvexMesh vs Sphere/Box/Capsule/Cylinder
     - MPR (support mapping).
     - 1
     - Uses convex support points; no manifold.
   * - ConvexMesh vs ConvexMesh
     - GJK for intersection + EPA for penetration.
     - 1
     - Contact point from support point along penetration normal.

Convex (primitive/convex mesh) vs Mesh (triangle mesh)
------------------------------------------------------
Mesh pairs use a BVH to find candidate triangles, then test each triangle
against the convex shape. Contacts are merged by distance/normal and capped
by the per-pair limit.

.. list-table::
   :header-rows: 1
   :widths: 22 45 18 35

   * - Pair
     - Narrowphase algorithm
     - Contact points
     - Notes
   * - Sphere vs Mesh
     - Closest point on triangle (distance check).
     - Up to 8
     - One contact per penetrating triangle, then merged.
   * - Capsule vs Mesh
     - Segment-triangle closest points.
     - Up to 8
     - Uses capsule axis segment and radius.
   * - Cylinder vs Mesh
     - Segment-triangle closest points.
     - Up to 8
     - Uses cylinder axis segment and radius (the cylinder is treated like a
       capsule).
   * - Box vs Mesh
     - Closest-feature tests + triangle plane test; GJK/EPA fallback.
     - Up to 8
     - Box-triangle contact uses multiple feature checks.
   * - ConvexMesh vs Mesh
     - GJK/EPA between convex mesh and triangle.
     - Up to 8
     - One contact per triangle candidate, then merged.

Mesh vs Mesh
------------
.. list-table::
   :header-rows: 1
   :widths: 22 45 18 35

   * - Pair
     - Narrowphase algorithm
     - Contact points
     - Notes
   * - Mesh vs Mesh
     - BVH traversal + GJK/EPA on triangle pairs.
     - Up to 8
     - Only nearly coincident points (closer than 1e-4) are merged.

Heightmap interactions
----------------------
Heightmap contacts operate on grid triangles. A rotated heightmap is handled in its own frame:
the other body's pose (and, for swept CCD, its motion) is expressed in the unrotated map's frame,
the contacts are found there and their points and normals are rotated back into the world. When
the footprint of a box, capsule, cylinder, or convex mesh lies over an exactly level patch of the
heightmap, RaiSim reuses the corresponding plane routine (see the plane tables). All heightmap
pairs go through the manifold reduction described above.

.. list-table::
   :header-rows: 1
   :widths: 22 45 18 35

   * - Pair
     - Narrowphase algorithm
     - Contact points
     - Notes
   * - Sphere vs Heightmap
     - Closest points on the terrain triangles within reach of the sphere.
     - Up to 8 (1 on a planar patch)
     - Keeps the deepest contact and adds contacts from surfaces whose normals
       differ from it by more than 20 degrees (for example, walls in a corner).
   * - Box vs Heightmap
     - Per-cell plane-box contacts.
     - Up to 8
     - Contacts merged by distance/normal.
   * - Capsule vs Heightmap
     - Per-cell plane-capsule contacts.
     - Up to 8
     - Contacts merged by distance/normal.
   * - Cylinder vs Heightmap
     - Per-cell plane-cylinder contacts.
     - Up to 8
     - At most 3 contacts per terrain triangle; merged by distance/normal.
   * - ConvexMesh vs Heightmap
     - Vertex vs triangle plane tests.
     - Up to 8
     - Uses convex mesh vertices to generate contacts.
   * - Mesh vs Heightmap
     - BVH traversal + GJK intersection test on mesh triangle vs heightmap
       triangle; contact from triangle-triangle closest points.
     - Up to 8
     - Normal from the terrain triangle; a recovery pass handles mesh vertices
       that are already below the terrain.

Plane vs Mesh / ConvexMesh
--------------------------
.. list-table::
   :header-rows: 1
   :widths: 22 45 18 35

   * - Pair
     - Narrowphase algorithm
     - Contact points
     - Notes
   * - Plane vs Mesh
     - Penetrating vertices/edges binned by angle around centroid.
     - Up to 8
     - Produces a coarse contact ring.
   * - Plane vs ConvexMesh
     - Vertex binning around centroid.
     - Up to 3
     - Uses deepest vertices below plane.

Ray tests (query-only)
----------------------
Ray pairs are used by ray tests and return a single hit (if any) with zero
penetration.

.. list-table::
   :header-rows: 1
   :widths: 22 45 18 35

   * - Pair
     - Algorithm
     - Contact points
     - Notes
   * - Ray vs Sphere
     - Analytic ray-sphere intersection.
     - 1
     - Returns closest hit along ray.
   * - Ray vs Box
     - Slab intersection in box local frame.
     - 1
     - Normal from hit face.
   * - Ray vs Capsule
     - Analytic ray vs capsule (cylinder + caps).
     - 1
     - Handles inside-origin cases.
   * - Ray vs Cylinder
     - Analytic ray vs finite cylinder.
     - 1
     - Includes caps.
   * - Ray vs Plane
     - Line-plane intersection.
     - 1
     - Uses the plane normal, flipped for back-side hits.
   * - Ray vs Mesh/ConvexMesh
     - BVH query + ray-triangle test.
     - 1
     - Closest triangle hit.
   * - Ray vs Heightmap
     - Grid traversal / height test.
     - 1
     - Returns first hit if any.
   * - Ray vs Ray
     - Closest segment-segment distance.
     - 1
     - Contact if distance <= 1e-6.

Swept CCD
=========
Swept continuous collision detection is opt-in through ``contact::ContactSettings``. It is intended
for fast bodies that may pass through thin static terrain within one time step. It covers dynamic
sphere, capsule, box, and cylinder collision bodies against static ground planes and heightmaps;
other pairs use only discrete detection.

.. code-block:: cpp

    auto settings = world.getContactSettings();
    settings.sweptCcdEnabled = true;            // default: false
    settings.sweptCcdMinSpeed = 8.0;            // default: 0.0 (sweep every moving body)
    settings.sweptCcdSpeculativeMargin = 1.0e-4; // default: 1e-4 m
    world.setContactSettings(settings);

When enabled, RaiSim adds speculative contacts for bodies whose speed (linear speed plus angular
speed times the sweep radius) is at least ``sweptCcdMinSpeed``.
Use ``World::setCollisionCountersEnabled(true)`` and ``getLastCollisionCounters()`` to inspect
``sweptCcdCandidates`` and ``sweptCcdContacts`` in benchmark or diagnostic code. Deformable particles
do not use swept CCD: their contacts include speculative ones for every body a particle can reach within
the step (see :doc:`DeformableObject`).

Collision counters
==================
``World::CollisionCounters`` exposes optional diagnostic counters, reset at every contact detection:
deformable particle contact statistics (``deformableParticleCandidates``,
``deformableParticleAabbTests``, ``deformableParticleContacts``, ``deformableParticleGridCells``)
and swept CCD counts (``sweptCcdCandidates``, ``sweptCcdContacts``). Counters are disabled by
default to keep the normal simulation path cheap.

.. code-block:: cpp

    world.setCollisionCountersEnabled(true);
    world.integrate();
    const auto& counters = world.getLastCollisionCounters();

Unsupported or no-op pairs
==========================
* **Plane vs Plane**, **Plane vs Heightmap**, **Heightmap vs Heightmap**: no contact generation
  (pairs of static bodies are never tested).
* **Cone**: not supported; creating a cone is a fatal error.
* **Ray vs Plane/Primitive/Mesh/Heightmap**: query-only (not solved as rigid contacts).
