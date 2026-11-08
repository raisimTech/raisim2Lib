# Collision Detection and Colliders

## Overview

RaiSim performs collision detection in two stages: an AABB-based broadphase
filters candidate pairs, followed by a pair-specific narrowphase that generates
contact points. The narrowphase algorithm depends on the shape pair (analytic
tests, SAT, MPR, or GJK/EPA). Pairs rejected by the collision group and mask
(see [Contact and Collision](./Contact.md)) and pairs of two static bodies never reach the narrowphase.

The broadphase is selected with `contact::BroadphaseSettings::type`
(`World::setBroadphaseSettings`):

- `contact::BroadphaseType::Sap3Axis` (default): sweep and prune on three axes.
- `contact::BroadphaseType::MultiBoxPrune`: multi-box pruning on a uniform
  grid; the `mbp*` fields set the grid bounds and cell size.
- `contact::BroadphaseType::None`: no AABB culling; every pair goes to the
  narrowphase. Use it only for debugging.

Scenes with fewer than 35 collision bodies use an all-pairs AABB test
regardless of this setting.

A body whose bounding box is no longer finite, because its pose became NaN or
infinite (the simulation diverged), stops the simulation with a fatal error
whichever of these runs. Bodies that collide with nothing (collision group or mask 0) are
not checked. A deformable particle that diverged makes no contact instead (see
[Deformable Objects](./DeformableObject.md)).

## Global contact limits and merging

- **Per-pair limit**: `ContactSettings::maxContactsPerPair` (default 8)
  is clamped to **[1, 8\]** inside the contact detector. Any pair listed below
  is capped by this limit, even if the narrowphase could generate more points.
- **Merging for mesh/heightmap pairs**: contacts produced by triangle-based
  tests are merged to reduce redundancy. The merge uses a distance threshold
  proportional to the smaller shape size (about 0.105 \* characteristic length;
  a heightmap counts as infinitely large, so the other shape sets the scale),
  a normal alignment threshold of about 4 degrees, and a small positional
  epsilon (~1e-4) to coalesce nearly identical points. Mesh vs mesh pairs only
  coalesce points closer than 1e-4.
- **Heightmap manifold reduction**: heightmap pairs first collect up to 8
  contacts. RaiSim then drops contacts that duplicate another contact or lie
  between two contacts with nearly identical normals and add no extra depth,
  and keeps the deepest remaining contacts up to the per-pair limit.
- **Special caps**: some generators have additional caps. Box-cylinder can
  produce up to 16 contacts internally but is still clamped to the per-pair
  limit; cylinder-heightmap accepts at most 3 contacts from each terrain
  triangle.

## Shape categories

- **Primitive**: sphere, box, capsule, cylinder, plane. A plane is the
  `World::addGround` ground. Compound children are ordinary primitive bodies
  and follow the primitive tables. Cones are not supported.
- **ConvexMesh**: convex mesh representation (`MeshCollisionMode::CONVEX_HULL`
  or each part of a `MeshCollisionMode::CONVEXIFY` decomposition).
- **Mesh**: non-convex triangle mesh (`MeshCollisionMode::ORIGINAL_MESH`).
- **Heightmap**: grid terrain represented by triangles. It can be translated and rotated
  arbitrarily; contacts are computed in the heightmap's own frame and mapped back to the world.
- **Ray**: query-only shape used by ray tests.

## Pair list (narrowphase algorithms and contact counts)

Pairs are symmetric; a single entry covers both A vs B and B vs A.

### Primitive vs Primitive

| Pair | Narrowphase algorithm | Contact points | Notes |
| --- | --- | --- | --- |
| Sphere vs Sphere | Analytic distance check; contact on line between centers. | 1 | Normal along center line. |
| Sphere vs Box | Closest point on oriented box (analytic). | 1 | Uses box clamp in local frame. |
| Sphere vs Capsule | Closest point on capsule axis segment, then sphere-sphere contact. | 1 | Segment endpoints are capsule caps. |
| Sphere vs Cylinder | Analytic finite-cylinder test (side and caps). | 1 | Handles endcaps and side wall. |
| Box vs Box | SAT on 15 axes (3 + 3 + 9 cross), face/edge clipping. | Up to 8 | Edge-edge yields 1; face-face yields a clipped manifold. |
| Box vs Capsule | Segment-box closest approach; fallback to cap sphere-box tests. | Up to 2 | 1 if segment interior hits; otherwise up to 2 cap contacts. |
| Box vs Cylinder | SAT + clipping (cylinder vs box face/circle); MPR if clipping yields no point. | Up to 16 (clamped to the per-pair limit) | Produces 1-2 for edge hits, more for face overlap. |
| Capsule vs Capsule | Segment-segment closest points; parallel overlap uses both ends of the overlapping axis interval. | Up to 2 | Parallel, overlapping axes can emit two contacts. |
| Capsule vs Cylinder | MPR (Minkowski Portal Refinement) support mapping, then the touching flat parts (see below). | Up to 2 | A capsule lying on a cap or along the side gets both ends of the line where they touch. |
| Cylinder vs Cylinder | Nearly parallel axes in end-to-end (cap-to-cap) contact: analytic rim sampling; otherwise MPR, then the touching flat parts (see below). | Up to 8 (cap-to-cap), otherwise up to 4 | A cylinder lying on another's cap or side gets both ends of the line where they touch. |

### Primitive vs Plane

Plane contacts use the plane's world-frame normal. The ground plane created by
`World::addGround` has the normal +Z (world up).

| Pair | Narrowphase algorithm | Contact points | Notes |
| --- | --- | --- | --- |
| Sphere vs Plane | Analytic projection to plane. | 1 | Contact at the sphere point deepest along the plane normal. |
| Box vs Plane | Projected box corners and face selection. | Up to 4 | Chooses deepest corner then up to 3 additional points. |
| Capsule vs Plane | Analytic endcap projection. | Up to 2 | One contact under each end of the capsule axis that reaches the plane, the deeper first: a capsule lying on the plane rests on both ends. The same holds on a level part of a heightmap, which uses this pair. |
| Cylinder vs Plane | Analytic: rim contacts if the axis is parallel to the plane normal or the whole lower rim is below the plane; otherwise two edge points. | Up to 4 (parallel or fully submerged rim), otherwise up to 2 | Parallel means the cylinder axis is aligned with the plane normal. |

MPR finds the deepest point and the normal of a pair. Where the part of each
body facing the other along the normal is flat -- a box face or edge, a cylinder
cap (as the octagon inscribed in its rim) or the line along its side, the line
along a capsule's side -- and lies within about 3 degrees of facing the normal,
the pair gets the corners of the overlap of the two parts that the surfaces
reach, up to four of them, spread over the overlap, each at its own depth and
halfway between the surfaces. A convex mesh's part is the outline of its
vertices that reach as deep as the contact, which on a face is the face; its
height between them is interpolated, so its contacts are never deeper than the
mesh. Over a face, the face's own normal is the contact normal. MPR's deepest
point is kept as well where the corners fall short of it, such as between two
nearly parallel lying cylinders or past the octagon of a cap. Otherwise, such as for two cylinders lying across each
other, the pair keeps MPR's one contact, on the first body.

### ConvexMesh vs Primitive/ConvexMesh

| Pair | Narrowphase algorithm | Contact points | Notes |
| --- | --- | --- | --- |
| ConvexMesh vs Sphere/Box/Capsule/Cylinder | MPR (support mapping), then the touching flat parts (see below). | Up to 4 | A hull face on a box face or a cylinder cap gets the corners of their overlap; a sphere gets one contact. |
| ConvexMesh vs ConvexMesh | GJK for intersection + EPA for penetration. | Up to 4 | Faces touching face to face get up to four corners of their overlap, spread over it: the vertices of each body near its deepest plane along the normal that lie over the other's, and the crossings of the two outlines, each with its own depth. A vertex or edge contact gets the EPA witness point of the first body. |

### Convex (primitive/convex mesh) vs Mesh (triangle mesh)

Mesh pairs use a BVH to find candidate triangles, then test each triangle
against the convex shape. Contacts are merged by distance/normal and capped
by the per-pair limit.

Sphere, capsule and cylinder queries also reject BVH nodes whose triangles
cannot lie within the radius of the center or axis segment, and cylinder
queries those lying past a cap. Once the contact set of a sphere or capsule is
full, nodes that cannot produce a deeper retained contact are skipped.
Box queries reject nodes separated from the oriented box. These checks reduce
triangle work for dense meshes passing close to a primitive while preserving
candidate order and contact merging.

A box and a mesh triangle touch only where they overlap, and the box is pushed
out of the triangle the shortest way. Two exceptions keep a mesh surface
behaving like one surface rather than a collection of triangles:

- Where the shortest way out would only slide the box along a flat mesh surface
  onto the next triangles (at the rim of a box lying on a triangulated floor, or
  across a sliver), the box is held off along the surface's normal instead.
- A triangle crossing a box face at an angle, where a curved mesh such as a
  hand presses into the face, holds the box off along the triangle's own normal.

A box lying face down on a mesh face touches it at the corners of the faces'
overlap, so it rests flat even on a single large triangle. The contacts of a
box with a mesh keep every surface the box lies against (the floor and both
walls of a mesh bin's corner) and are spread over each, deepest first.

A cylinder and a mesh triangle also touch only where they overlap, with the
same two exceptions. A cylinder standing on a mesh face touches it around its
rim, one lying on it along the lowest line of its side, and one tipped onto
its rim at the rim's lowest point, each contact at its own depth, so it rests
and rolls on a triangulated floor as on a box or the ground. On a flat surface
its contacts are spread around the rim before other points of the overlap, and
they come deepest first.

Three cases are handled less well than for a box:

- In a gently curved hollow, such as a shallow mesh bowl, the rim's contacts
  can gather on one side of the rim: a thin disc set in such a bowl sinks a few
  millimetres and keeps rocking instead of coming to rest.
- A cylinder's rim pressed against a convex crease of a mesh, such as a curb or
  a stair nosing, rests a few millimetres deep and is pushed partly along the
  crease.
- A thin wheel pressed into a mesh floor deeper than half its width is pushed
  back along a direction up to about 8 degrees off the floor's normal.

A cylinder lying along a mesh edge parallel to its axis, such as a pipe lying in
a slot on its two edges, touches each edge at both ends of the stretch it lies
on.

These checks apply automatically to non-convex mesh collision bodies, including
articulated self-collision. See [Benchmark](./Benchmark.md) for the dense-mesh proximity
and G1 hand self-collision workloads.

| Pair | Narrowphase algorithm | Contact points | Notes |
| --- | --- | --- | --- |
| Sphere vs Mesh | Closest point on triangle (distance check). | Up to 8 | One contact per penetrating triangle, then merged. |
| Capsule vs Mesh | Segment-triangle closest points. | Up to 8 | Uses capsule axis segment and radius. |
| Cylinder vs Mesh | Separating axes of the cylinder and each triangle. | Up to 8 | A cap against a face touches around its rim, the side along its lowest line; contacts come deepest first (see above). |
| Box vs Mesh | Separating axes of the box and each triangle. | Up to 8 | Faces against faces touch at the corners of their overlap; an edge against an edge at one point. Contacts come deepest first (see above). |
| ConvexMesh vs Mesh | GJK/EPA between convex mesh and triangle. | Up to 8 | One contact per triangle candidate, midway between the EPA witness points, then merged. Faces touching flat get their contacts inside the overlap of the faces. |

### Mesh vs Mesh

| Pair | Narrowphase algorithm | Contact points | Notes |
| --- | --- | --- | --- |
| Mesh vs Mesh | BVH traversal + GJK/EPA on triangle pairs. | Up to 8 | One contact per triangle pair, midway between the EPA witness points. Only nearly coincident points (closer than 1e-4) are merged. |

### Heightmap interactions

Heightmap contacts operate on grid triangles. A rotated heightmap is handled in its own frame:
the other body's pose (and, for swept CCD, its motion) is expressed in the unrotated map's frame,
the contacts are found there and their points and normals are rotated back into the world. When
the footprint of a box, capsule, cylinder, or convex mesh lies over an exactly level patch of the
heightmap, RaiSim reuses the corresponding plane routine (see the plane tables). All heightmap
pairs go through the manifold reduction described above.

| Pair | Narrowphase algorithm | Contact points | Notes |
| --- | --- | --- | --- |
| Sphere vs Heightmap | Closest points on the terrain triangles within reach of the sphere. | Up to 8 (1 on a planar patch) | Keeps the deepest contact and adds contacts from surfaces whose normals differ from it by more than 20 degrees (for example, walls in a corner). |
| Box vs Heightmap | Per-cell plane-box contacts. | Up to 8 | Contacts merged by distance/normal. |
| Capsule vs Heightmap | Per-cell plane-capsule contacts. | Up to 8 | Contacts merged by distance/normal. |
| Cylinder vs Heightmap | Per-cell plane-cylinder contacts. | Up to 8 | At most 3 contacts per terrain triangle; merged by distance/normal. |
| ConvexMesh vs Heightmap | Vertex vs triangle plane tests. | Up to 8 | Uses convex mesh vertices to generate contacts. |
| Mesh vs Heightmap | BVH traversal + GJK intersection test on mesh triangle vs heightmap triangle; contact from triangle-triangle closest points. | Up to 8 | Normal from the terrain triangle; a recovery pass handles mesh vertices that are already below the terrain. |

### Plane vs Mesh / ConvexMesh

| Pair | Narrowphase algorithm | Contact points | Notes |
| --- | --- | --- | --- |
| Plane vs Mesh | Penetrating vertices/edges binned by angle around centroid. | Up to 8 | Produces a coarse contact ring. |
| Plane vs ConvexMesh | Vertex binning around centroid. | Up to 3 | Uses deepest vertices below plane. |

### Ray tests (query-only)

Ray pairs are used by ray tests and return a single hit (if any) with zero
penetration.

| Pair | Algorithm | Contact points | Notes |
| --- | --- | --- | --- |
| Ray vs Sphere | Analytic ray-sphere intersection. | 1 | Returns closest hit along ray. |
| Ray vs Box | Slab intersection in box local frame. | 1 | Normal from hit face. |
| Ray vs Capsule | Analytic ray vs capsule (cylinder + caps). | 1 | Handles inside-origin cases. |
| Ray vs Cylinder | Analytic ray vs finite cylinder. | 1 | Includes caps. |
| Ray vs Plane | Line-plane intersection. | 1 | Uses the plane normal, flipped for back-side hits. |
| Ray vs Mesh/ConvexMesh | BVH query + ray-triangle test. | 1 | Closest triangle hit. |
| Ray vs Heightmap | Grid traversal / height test. | 1 | Returns first hit if any. |
| Ray vs Ray | Closest segment-segment distance. | 1 | Contact if distance <= 1e-6. |

## Swept CCD

Swept continuous collision detection is opt-in through `contact::ContactSettings`. It is intended
for fast bodies that may pass through thin static terrain within one time step. It covers dynamic
sphere, capsule, box, and cylinder collision bodies against static ground planes and heightmaps;
other pairs use only discrete detection.

```cpp
auto settings = world.getContactSettings();
settings.sweptCcdEnabled = true;            // default: false
settings.sweptCcdMinSpeed = 8.0;            // default: 0.0 (sweep every moving body)
settings.sweptCcdSpeculativeMargin = 1.0e-4; // default: 1e-4 m
world.setContactSettings(settings);
```

When enabled, RaiSim adds speculative contacts for bodies whose speed (linear speed plus angular
speed times the sweep radius) is at least `sweptCcdMinSpeed`. Such a contact lets the body approach
the surface at up to the speed that carries it there by the end of the step. A body approaching
faster reaches the surface within the step: without restitution it stops there, and with
restitution it bounces off with the surface's restitution, starting up to the gap early (its
height is then off by up to the gap times one plus the restitution). A body within
`sweptCcdSpeculativeMargin` of the terrain keeps speculative contacts whatever its speed, where
touching contacts would be (a box lying flat keeps its four corners), so a body stopped at the
surface stays on it when it is pushed down (a foot under its leg). A speculative contact holds a
resting body the error-reduction rate times the time step times the margin above the surface, so
this needs `World::setERP()` well below one over the time step: at 0.95 over the time step a box
already sank into the ground under a push (the default, 1.5/s, is far below). An
articulated system integrated with `TRAPEZOID` moves by the mean of its start and end speeds, so
on an impact it goes in by up to half the step's travel, swept CCD or not, and comes back out at
the error-reduction rate; the default `SEMI_IMPLICIT` stops at the surface.
Use `World::setCollisionCountersEnabled(true)` and `getLastCollisionCounters()` to inspect
`sweptCcdCandidates` and `sweptCcdContacts` in benchmark or diagnostic code. Deformable particles
do not use swept CCD: their contacts include speculative ones for every body a particle can reach within
the step (see [Deformable Objects](./DeformableObject.md)).

## Collision counters

`World::CollisionCounters` exposes optional diagnostic counters, reset at every contact detection:
deformable particle contact statistics (`deformableParticleCandidates`,
`deformableParticleAabbTests`, `deformableParticleContacts`, `deformableParticleGridCells`)
and swept CCD counts (`sweptCcdCandidates`, `sweptCcdContacts`). Counters are disabled by
default to keep the normal simulation path cheap.

```cpp
world.setCollisionCountersEnabled(true);
world.integrate();
const auto& counters = world.getLastCollisionCounters();
```

## Unsupported or no-op pairs

- **Plane vs Plane**, **Plane vs Heightmap**, **Heightmap vs Heightmap**: no contact generation
  (pairs of static bodies are never tested).
- **Cone**: not supported; creating a cone is a fatal error.
- **Ray vs Plane/Primitive/Mesh/Heightmap**: query-only (not solved as rigid contacts).
