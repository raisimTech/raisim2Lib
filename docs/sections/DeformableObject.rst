#############################
Deformable Objects
#############################

``raisim::DeformableObject`` is RaiSim's XPBD/PBD deformable-body object.
It is intended for fast simulation and data-generation workloads that need
cloth, soft shells, or coarse volumetric proxies without leaving RaiSim's
single-threaded stepping model.

The implementation uses distance constraints between particles and optional
isometric bending constraints between adjacent triangles. Collision is handled
by one aggregate deformable object in the broad phase, with spherical particle
proxies used internally for contact generation. This keeps deformable objects
cheap to register in the world while still allowing particle-level contacts
against the ground, rigid bodies, and other deformable objects. The
``deformable_objects`` example (:doc:`examples/server/deformable_objects`)
drapes a cloth over a sphere and stacks mesh-based soft cubes.

API Surface
===========

The public constructors are exposed through ``raisim::World``:

* ``World::addDeformableCloth(vertices, triangles, pinnedVertices, material, contactMaterial, collisionGroup, collisionMask)``
  creates a deformable cloth or shell directly from world-space particle
  positions and triangle indices.
* ``World::addDeformableCloth(meshFileInObjFormat, scale, pinnedVertices, material, contactMaterial, collisionGroup, collisionMask)``
  loads an OBJ mesh as a surface cloth/shell and uses its vertices directly as
  particles.
* ``World::addDeformableObject(meshFileInObjFormat, material, options, pinnedVertices, contactMaterial, collisionGroup, collisionMask)``
  loads an OBJ mesh with ``MeshBuildOptions`` for resampled surface particles,
  filled particles, and optional internal struts.

Trailing arguments have defaults: scale ``1``, no pinned vertices, a
default-constructed ``Material``, contact material ``"default"``, collision
group ``1``, and a collision mask that accepts every group.
``addDeformableObject`` always requires its material and build options. All
three creation paths register one ``DeformableObject`` in the world.
Particle-level operations use local particle indices. Pinned particles report
``BodyType::STATIC`` through ``getBodyType(localIdx)``; all other particles
report ``BodyType::DYNAMIC``.

Data Model
==========

The simulated state is particle based:

* ``getNumParticles()`` returns the number of particles.
* ``getPositions()`` returns world-space particle positions.
* ``getTriangles()`` returns the visual/surface triangle topology.
* ``getVisualPosition(idx)`` returns the particle position plus its visual
  offset. For a closed triangle mesh, construction moves each surface particle
  inward by about ``collisionRadius`` so that the collision envelope matches the
  input surface; the visual offset undoes that shift. For an open mesh the
  offset is zero.
* ``getDistanceConstraintCount()`` and ``getBendingConstraintCount()`` expose
  the generated constraint counts for diagnostics.
* ``getCollisionBodyCount()`` is normally ``1`` because one aggregate collision
  body represents the deformable object in the broad phase.
* ``setPositionOffset(offset)`` translates all particles, including the
  positions that pinned particles are held at.
  ``applyRigidTransform(translation, rotation)`` rotates all particles about the
  world origin and then translates them; velocities and pending forces are
  rotated as well.

The collision proxy radius is not the visual mesh thickness. It controls the
spherical particle contacts used by the solver. If visual geometry appears to
touch but contacts are weak or delayed, check ``collisionRadius`` and particle
spacing before increasing stiffness.

Creating A Deformable Object
============================

Explicit vertices
-----------------

Use ``World::addDeformableCloth`` when you already have particle positions and
surface triangle indices. The vertices are in world coordinates.

.. code-block:: cpp

    std::vector<raisim::Vec<3>> vertices = {
        {-0.5, -0.5, 1.0},
        { 0.5, -0.5, 1.0},
        { 0.5,  0.5, 1.0},
        {-0.5,  0.5, 1.0}};

    std::vector<raisim::DeformableObject::Triangle> triangles = {
        {0, 1, 2},
        {0, 2, 3}};

    raisim::DeformableObject::Material material;
    material.totalMass = 1.0;
    material.distanceCompliance = 1.0e-5;
    material.bendCompliance = 1.0e-5;
    material.damping = 0.02;
    material.collisionRadius = 0.01;
    material.iterations = 6;

    auto* cloth = world.addDeformableCloth(vertices, triangles, {}, material);

Mesh surface particles
----------------------

Use ``World::addDeformableObject`` with ``MeshParticleOptions::Mode::Surface``
to resample an OBJ surface. Every triangle is subdivided so that none of its
edges is longer than ``spacing`` (default 0.05 m), and the resulting vertices
become the particles; triangles that are already finer keep their original
vertices. To use the OBJ vertices directly without resampling, call
``World::addDeformableCloth(meshFile, scale, ...)`` instead.

.. code-block:: cpp

    raisim::DeformableObject::MeshBuildOptions build;
    build.particles.mode = raisim::DeformableObject::MeshParticleOptions::Mode::Surface;
    build.particles.scale = 1.0;

    auto* shell = world.addDeformableObject("soft_shell.obj", material, build);
    shell->setPositionOffset({0.0, 0.0, 1.0});

OBJ polygon faces are triangulated as a fan. Texture and normal indices are
ignored. This mode is suitable for cloth and shell meshes.

``addDeformableObject`` raises ``Material::collisionRadius`` to at least
``0.58 * spacing`` in both Surface and Filled modes so that neighboring
particle spheres overlap. Pinned-vertex indices passed to
``addDeformableObject`` refer to the generated particle list, not to the
original OBJ vertex numbering.

Filled mesh particles
---------------------

Use ``MeshParticleOptions::Mode::Filled`` for a closed OBJ triangle mesh that
should receive interior particles. RaiSim resamples the surface as in Surface
mode and adds interior particles on a regular grid with the same ``spacing``.

.. code-block:: cpp

    raisim::DeformableObject::MeshBuildOptions build;
    build.particles.mode = raisim::DeformableObject::MeshParticleOptions::Mode::Filled;
    build.particles.scale = 1.0;
    build.particles.spacing = 0.05;
    build.particles.maxFillParticles = 50000;

    auto* softBody = world.addDeformableObject("closed_cube.obj", material, build);
    softBody->setPositionOffset({0.0, 0.0, 1.0});

Filled mode requires a closed (watertight) triangle mesh with a meaningful
inside/outside volume: interior particles are selected by inside/outside tests
against that surface, which are not meaningful for an open mesh.

When using filled mode, tune ``spacing`` and ``maxFillParticles`` (default
50000) together. Smaller spacing increases particle count and usually improves
shape support, but it also increases contact and constraint cost.
``maxFillParticles`` limits the number of interior particles; if the requested
spacing would exceed it, use a larger spacing or a simpler mesh rather than
raising the limit for an unexpectedly large model.

Internal struts
---------------

Internal struts are additional distance constraints used to preserve a rest
shape and provide bounce/recovery for soft shells or filled objects.
``PairsWithinRadius`` connects every particle pair whose distance is at most
``radius``; pairs that already share a constraint are skipped. Generation tests
every particle pair (quadratic in the particle count), and a large radius
creates many constraints that each cost solver time.

.. code-block:: cpp

    raisim::DeformableObject::MeshBuildOptions build;
    build.particles.mode = raisim::DeformableObject::MeshParticleOptions::Mode::Filled;
    build.particles.spacing = 0.05;
    build.internalStruts.mode =
        raisim::DeformableObject::InternalStrutOptions::Mode::PairsWithinRadius;
    build.internalStruts.radius = 0.09;

    auto* softBody = world.addDeformableObject("closed_cube.obj", material, build);

You can also add constraints after construction:

.. code-block:: cpp

    softBody->addDistanceConstraint(0, 7);
    softBody->addDistanceConstraints({{0, 6}, {1, 7}});
    softBody->addInternalStruts(build.internalStruts);

Manual constraints are useful for adding diagonal support to coarse shells.
Their rest length is the current distance between the two particles, so add them
immediately after construction or after intentionally placing the object in its
desired rest pose. ``addDistanceConstraints`` skips pairs that already have a
constraint; ``addDistanceConstraint`` does not check for duplicates.

Material Parameters
===================

``DeformableObject::Material`` controls mass, solver behavior, collision proxy
size, and elastic response. Defaults are shown in parentheses:

* ``totalMass`` (1 kg): total mass distributed uniformly over all particles.
* ``distanceCompliance`` (-1): XPBD distance compliance. ``0`` is rigid; larger
  values are softer. Negative values preserve legacy behavior by using
  ``1 / distanceStiffness``, so the default compliance is ``1e-4``.
* ``distanceStiffness`` (1e4): legacy stiffness parameter. Prefer
  ``distanceCompliance`` for new code.
* ``bendCompliance`` (-1): XPBD isometric bending compliance for adjacent
  triangle pairs. ``0`` is rigid; larger values allow easier folding. Negative
  values use ``1 / bendStiffness`` when ``bendStiffness`` is positive and
  otherwise disable bending, so bending is off by default.
* ``bendStiffness`` (0): legacy bending stiffness parameter. Prefer
  ``bendCompliance`` for new code.
* ``youngsModulus`` (-1, unused): Young's modulus in Pa. If positive, per-edge
  XPBD compliance is derived from rest length and effective area, and
  ``distanceCompliance`` is ignored.
* ``poissonRatio`` (0.3): stored for elastic material definitions and, when
  ``youngsModulus`` is positive, required to lie between -1 and 0.5. The
  current distance-constraint model does not implement a volumetric
  shear/Poisson model.
* ``thickness`` (0.01 m): surface thickness used to derive an effective area
  from each edge length when ``youngsModulus`` is positive and
  ``crossSectionArea`` is not set.
* ``crossSectionArea`` (-1, unused): effective area for each distance
  constraint. If positive, this overrides the thickness-based estimate.
* ``damping`` (0.02): fraction of particle velocity removed per world step,
  clamped to ``[0, 1]``. The result does not depend on ``substeps``.
* ``collisionRadius`` (0.01 m): radius of each internal particle collision
  proxy.
* ``iterations`` (6): number of constraint projection iterations per substep.
* ``substeps`` (1): number of deformable substeps per RaiSim world step.
* ``solverMode`` (``XPBD``): ``XPBD`` or ``PBD``. ``PBD`` projects every
  constraint rigidly and ignores the compliance magnitudes, so its effective
  stiffness depends on ``iterations`` and the time step. Whether bending
  constraints exist is still decided by ``bendCompliance``/``bendStiffness``.

Elastic Modulus
===============

For engineering-style material input, set ``youngsModulus`` instead of
``distanceCompliance``:

.. code-block:: cpp

    raisim::DeformableObject::Material material;
    material.totalMass = 1.0;
    material.youngsModulus = 5.0e4;
    material.poissonRatio = 0.3;
    material.thickness = 0.02;
    material.collisionRadius = 0.01;

When ``youngsModulus > 0``, RaiSim computes each distance-constraint compliance
as:

.. math::

    \alpha = \frac{L}{E A}

where ``L`` is the rest length, ``E`` is ``youngsModulus``, and ``A`` is either
``crossSectionArea`` or the thickness-derived area ``thickness * L``. With the
thickness-derived area the compliance reduces to ``1 / (E * thickness)`` for
every edge. Larger ``youngsModulus`` therefore produces a stiffer object.

Pinned Vertices And Forces
==========================

Pinned vertices are fixed in world coordinates, have infinite mass, and are
exposed to collision as static proxies. They can be specified at construction
time or changed later; ``pinVertex`` holds the particle at its current
position:

.. code-block:: cpp

    cloth->pinVertex(0);
    cloth->unpinVertex(0, 0.01);

``unpinVertex(idx, mass)`` restores a finite, positive mass for that particle.
Use a mass consistent with the surrounding particles (``totalMass`` divided by
the particle count). Constraint corrections are distributed in proportion to
inverse mass: a much lighter particle absorbs most of each correction and
reacts strongly to forces, while a much heavier one barely moves and drags its
neighbors as if it were nearly pinned.

External forces can be applied per particle with the standard ``Object``
interface. They act on the next step only, and calls on pinned particles are
ignored. Particles have no rotational state, so ``setExternalTorque`` has no
effect:

.. code-block:: cpp

    softBody->setExternalForce(3, {0.0, 0.0, 2.0});
    softBody->clearExternalForcesAndTorques();

Solver Tuning
=============

Start with the largest stable time step your application needs, then tune in
this order:

1. Choose a particle spacing that resolves the shape at the level your task
   actually observes.
2. Set ``collisionRadius`` so neighboring particle proxies cover the surface
   without making the visual mesh appear inflated.
3. Increase ``iterations`` until distance constraints converge enough for the
   task.
4. Increase ``substeps`` when contacts are jittery or high-speed impacts create
   excessive penetration.
5. Reduce ``distanceCompliance`` or increase ``youngsModulus`` only after the
   discretization and solver budget are reasonable.

``XPBD`` is the recommended default because its compliance gives a material
stiffness that depends far less on the iteration count and time step than
``PBD``, where stiffness is only a by-product of how many rigid projections are
performed. Use ``PBD`` mainly for backward-compatible behavior or controlled
comparisons.

Common failure modes:

* Exploding or jittering contacts usually mean the time step is too large for
  the stiffness/contact radius combination.
* A cloth that stretches too much needs lower ``distanceCompliance`` or more
  iterations.
* A cloth that folds too easily needs bending constraints (a nonnegative
  ``bendCompliance``; bending is off by default) and then a lower
  ``bendCompliance``.
* Filled objects that collapse need more internal struts, smaller spacing, or
  higher elastic stiffness.

Rayrai Visualization
====================

A local ``RayraiWindow`` draws deformable objects in its world automatically.
``RaisimServer`` serializes them as dynamic ``Shape::Mesh`` visuals for the
Rayrai TCP viewer. The first visualizer packet sends the object name, triangle
topology, and current vertex positions. Later packets stream updated vertex
positions and resend the triangle list only if the topology changes. Rayrai
rebuilds the custom OpenGL mesh, recomputes normals from the triangle list, and
renders the deformable surface in world coordinates. Both paths draw the
vertices at ``getVisualPosition()``.

The particle collision proxies are an internal physics representation and are
not rendered as one sphere per vertex. For an open cloth, the drawn surface
passes through the particle centers, so the collision envelope extends about
``collisionRadius`` to either side of it. For a closed mesh, the inward particle
shift and the visual offset make the drawn surface approximately coincide with
the collision envelope. For stacked soft cubes or other closed shapes, choose
particle spacing and ``collisionRadius`` together: neighboring particle spheres
should cover the surface without leaving gaps, but the radius should stay small
relative to the object's features.

World XML
=========

``World::exportToXml`` writes each deformable object as a ``<deformable>``
object with its material, contact material, collision group and mask, every
particle's position, rest position, velocity, visual offset, inverse mass, and
pinned flag, the triangle list, and the distance constraints. Loading the file
with ``raisim::World(path)`` recreates the object; bending constraints are
rebuilt from the triangles and material. The element layout is intended for
this export/import round trip; the loader requires the attributes that the
exporter writes.

Limitations
===========

This is still an experimental implementation. It does not include tetrahedral
finite elements, collisions between particles of the same object
(self-collision), or topology changes such as tearing or cutting. Granular
systems do not collide with deformable objects. Swept CCD is available for
rigid-body contact settings, but deformable particle contacts use the discrete
collision path described above.

API Reference
=============

.. doxygenclass:: raisim::DeformableObject
   :members:
