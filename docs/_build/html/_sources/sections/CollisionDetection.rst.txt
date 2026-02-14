#################################
Collision Detection and Colliders
#################################

Overview
========
RaiSim performs collision detection in two stages: an AABB-based broadphase
filters candidate pairs, followed by a pair-specific narrowphase that generates
contact points. The narrowphase algorithm depends on the shape pair (analytic
tests, SAT, MPR, or GJK/EPA).

Global contact limits and merging
=================================
* **Per-pair limit**: :code:`ContactSettings::maxContactsPerPair` is clamped to
  **[1, 8]** inside the contact detector. Any pair listed below is capped by
  this limit, even if the narrowphase could generate more points.
* **Merging for mesh/heightmap pairs**: contacts produced by triangle-based
  tests are merged to reduce redundancy. The merge uses a distance threshold
  proportional to the smaller shape size (about 0.105 * characteristic length),
  a normal alignment threshold of about 4 degrees, and a small positional
  epsilon (~1e-4) to coalesce nearly identical points.
* **Special caps**: some generators have additional caps (e.g., box-cylinder
  can produce up to 16 contacts internally, but is still clamped to the global
  limit; cylinder-heightmap is reduced to at most 3 contacts after merging).

Shape categories
================
* **Primitive**: sphere, box, capsule, cylinder, plane.
* **ConvexMesh**: convexified mesh (convex hull or convex decomposition).
* **Mesh**: triangle mesh (non-convex).
* **Heightmap**: grid terrain represented by triangles (rotation is not
  supported in collision).
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
     - SAT + clipping (cylinder vs box face/circle).
     - Up to 16 (clamped to global limit)
     - Produces 1-2 for edge hits, more for face overlap.
   * - Capsule vs Capsule
     - Segment-segment closest points; parallel overlap uses two caps.
     - Up to 2
     - Parallel, overlapping axes can emit two contacts.
   * - Capsule vs Cylinder
     - MPR (Minkowski Portal Refinement) support mapping.
     - 1
     - Generic convex-convex contact.
   * - Cylinder vs Cylinder
     - Parallel case: analytic rim sampling; otherwise MPR fallback.
     - Up to 8 (parallel), otherwise 1
     - Non-parallel or end-cap cases use MPR.

Primitive vs Plane
------------------
Plane contacts use a fixed plane normal of +Z (world up).

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
     - Contact at sphere bottom along +Z.
   * - Box vs Plane
     - Projected box corners and face selection.
     - Up to 4
     - Chooses deepest corner then up to 3 additional points.
   * - Capsule vs Plane
     - Analytic endcap projection.
     - 1
     - Uses lower capsule cap.
   * - Cylinder vs Plane
     - Analytic: cap contacts if parallel, otherwise two edge points.
     - Up to 4 (parallel), otherwise up to 2
     - Parallel means cylinder axis aligned with +Z.

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
by the global limit.

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
     - Uses cylinder axis segment and radius.
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
     - Contacts merged; deepest contact is retained.

Heightmap interactions
----------------------
Heightmap contacts operate on grid triangles; heightmap rotation is not
supported (rotation must be identity).

.. list-table::
   :header-rows: 1
   :widths: 22 45 18 35

   * - Pair
     - Narrowphase algorithm
     - Contact points
     - Notes
   * - Sphere vs Heightmap
     - Local triangle plane at sampled (x, y).
     - 1
     - Uses normal of the triangle under the sphere center.
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
     - Up to 3
     - Merged then reduced to at most 3 points.
   * - ConvexMesh vs Heightmap
     - Vertex vs triangle plane tests.
     - Up to 8
     - Uses convex mesh vertices to generate contacts.
   * - Mesh vs Heightmap
     - BVH traversal + GJK/EPA on mesh triangle vs heightmap triangle.
     - Up to 8
     - Contacts merged by distance/normal.
   * - Plane vs Heightmap
     - Plane-mesh style vertex/edge binning.
     - Up to 8
     - Same logic as plane vs mesh.

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
     - Uses plane normal +Z.
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
     - Contact if distance < 1e-6.

Unsupported or no-op pairs
==========================
* **Plane vs Plane**: no contact generation.
* **Ray vs Plane/Primitive/Mesh/Heightmap**: query-only (not solved as rigid contacts).
