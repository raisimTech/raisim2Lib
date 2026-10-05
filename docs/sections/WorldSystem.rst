#############################
World
#############################
The :code:`raisim::World` class owns simulation resources: objects, constraints,
materials, collision detection, contact solving, sensors, time, and gravity. All
objects within a single World instance can collide with one another unless their
collision group and mask settings disable that pair. See :doc:`Contact` and
:doc:`CollisionDetection` for contact and collision details.

A ``raisim::World`` can be instantiated in four ways. The constructor that takes
a path chooses the loader from the file extension and content.

#. **From a USD file.** Pass a ``.usd``, ``.usda``, ``.usdc``, or ``.usdz``
   path to the constructor. RaiSim opens the USD stage and imports one USD
   Physics articulation (the default prim if it has
   ``PhysicsArticulationRootAPI``, otherwise the first such prim) as an
   ``ArticulatedSystem``, with its rigid bodies, joints, and supported
   collision shapes, and applies the stage's physics-scene gravity and
   physics materials when present. Other prims, such as free rigid bodies outside that
   articulation, are not imported, so add the ground and other objects in
   code. A stage without an articulation root throws ``std::runtime_error``.
   See :doc:`OpenUSD` for the full import semantics and tooling.

   .. code-block:: cpp

     raisim::World world("robot.usd");
     world.addGround();

   Use this entry point for robots authored in USD-native tools such as Isaac
   Sim, Omniverse, and Blender.

#. **From a RaiSim XML or MuJoCo MJCF configuration file.** Pass the file path
   to the same constructor. The file content, a ``<raisim>`` or a ``<mujoco>``
   element, selects the loader; the extension does not matter. Use this when
   the scene is hand-edited, template-driven, or already lives in one of those
   formats. See :doc:`WorldConfigurationFile` for the RaiSim XML schema and
   the MJCF notes below.

#. **From a RaiSim Engine scene file** (from v2.7.1, not yet released; the
   v2.7.0 package does not include the reader). Pass a ``.rscene`` path to the
   same constructor. RaiSim reads the solver settings and contact materials,
   sampled terrain, primitive, mesh and ``.rasset`` bodies, and wires;
   :doc:`RsceneFile` documents the format. The parsed scene stays available
   through ``World::getRscene()``, so rayrai can apply the rest of the scene,
   as RaiSim Engine 2's viewport shows it (render settings, sky, lights, fog,
   visual-only objects, instanced visuals, terrain foliage and textures, and
   the camera), in one call:

   .. code-block:: cpp

     auto world = std::make_shared<raisim::World>("forest.rscene");
     raisin::RayraiWindow viewer(world, 1280, 800);
     raisin::applyRscene(*world->getRscene(), viewer);

   Change the file's render settings in C++ before applying them:

   .. code-block:: cpp

     auto render = raisin::rsceneRenderSettings(*world->getRscene());
     render.quality.viewerMsaaSamples = 8;
     raisin::applyRscene(*world->getRscene(), viewer, render);

   Content the reader cannot reproduce exactly, such as articulated systems,
   compounds or nodes that inherit another node's transform, is a fatal error
   naming the line rather than being dropped. The scene's solver settings replace the
   world's defaults; Engine 2's defaults differ from ``raisim::World``'s (see
   :ref:`sections/RsceneFile:Solver defaults`). See
   :doc:`examples/rayrai/rayrai_forest_from_rscene` for a complete example.

#. **Programmatically.** Default-construct an empty world and add objects
   in code:

   .. code-block:: cpp

     raisim::World world;
     world.addGround();
     auto* box = world.addBox(1.0, 1.0, 1.0, 1.0);

These methods can be combined: load an initial file and then add or remove
objects in code before stepping.

MJCF files
==========

The MJCF (MuJoCo file format) reader is experimental. MJCF files are loaded
with the same ``raisim::World`` constructor used for RaiSim XML:

.. code-block:: cpp

  raisim::World world("rsc/mjcf/gymnasium/hopper.xml");

The reader supports the subset used by the bundled MJCF examples: static
``worldbody`` geoms, articulated bodies (each top-level ``body`` becomes one
``ArticulatedSystem``), free, ball, slide, and hinge joints, primitive geoms,
mesh assets, inertial tags, defaults, compiler ``angle``/``eulerseq``
settings, ``compiler meshdir``, material colors, ``option timestep``,
``equality connect`` constraints between bodies of the same top-level body,
and tendons (see :doc:`tendons/Reference`). Mesh asset paths are resolved
relative to the MJCF file directory, or relative to ``compiler meshdir`` when it
is provided. MJCF files without an ``asset`` block are accepted. Following the
MJCF default, a ``geom`` with no ``type`` attribute is treated as a sphere and
must provide at least one ``size`` value for its radius.

It is not a complete MuJoCo replacement. ``<actuator>`` elements are not
imported (the example programs apply joint torques from C++), ``<include>`` is
not supported, equality types other than ``connect`` and tendon equalities are
fatal errors, and integrators other than Euler are ignored with a warning.
Validate other MuJoCo features before relying on them.

Example targets:

* ``mjcf_gymnasium_hopper`` loads and actuates Gymnasium's Hopper model.
* ``mjcf_gymnasium_walker2d`` loads and actuates Gymnasium's Walker2d model.
* ``mjcf_gymnasium_humanoid`` loads Gymnasium's Humanoid model and drops it
  from a raised arbitrary configuration.

For callers that already hold a constructed ``raisim::World`` and want to
populate it from an MJCF file at runtime (instead of constructing the world
from the file path), use ``World::loadMjcfFile``:

.. code-block:: cpp

  raisim::World world;
  world.loadMjcfFile("rsc/mjcf/gymnasium/hopper.xml");

This is the same loader that ``World(configFile)`` calls internally for a
``<mujoco>`` file. The world should normally be empty before the call: MJCF
files often declare ground planes, ``worldbody`` geoms, and ``option`` blocks
that change world-level state such as the time step. The rayrai TCP viewer uses
this entry point in its drag-and-drop articulated-system inspector mode (see
:doc:`RayraiTcpViewer`).

Adding New Objects
============================
To add a new object of type X, utilize the :code:`addX` method.
For example, to add a sphere:

.. code-block:: cpp

  raisim::World world;
  auto sphere = world.addSphere(0.5, 1.0);

:code:`sphere` is a pointer to the object, which the world owns. Use it to read
and modify the object's state. The pointer becomes invalid when the object is
removed with ``World::removeObject``.

Most object-creation methods accept optional :code:`material`, :code:`collisionGroup`, and :code:`collisionMask` arguments.
A ground (``addGround``) has a fixed collision group and exposes only a
collision mask. Height maps (``addHeightMap``) accept a group that defaults to
the static collision group ``RAISIM_STATIC_COLLISION_GROUP``.
Collision groups and masks are described in :doc:`Contact`.
The :code:`material` argument names the material that governs contact dynamics; see :doc:`MaterialSystem`.

The object types are described in :doc:`Object`.

Upon object addition, a name may be assigned:

.. code-block:: cpp

  sphere->setName("ball");

An object pointer can be retrieved by name:

.. code-block:: cpp

  auto* ball = world.getObject("ball");

An object may consist of multiple bodies (e.g., an articulated system).
A **local index** is used to designate individual bodies.
To maintain API consistency, many methods require the local index argument even for single-body objects.
For single-body objects, the local index is ignored, and users may pass 0 to comply with the API.

Stable object identifiers
-------------------------
``Object::Id`` (a ``uint64_t``) is assigned when an object is added to a world
and remains valid for the lifetime of that object even when other objects are
removed; ids are not reused within a world. ``World::getObjectById(id)``
returns a pointer, or ``nullptr`` when the object has been removed;
``isObjectAlive(id)`` returns the same information as a ``bool``.
``getIndexInWorld()`` is fine for steady-state lookups, but in code that adds
and removes objects (RL resets, asset streaming) prefer the stable id: when an
object is removed, the last object in the world's list moves into the removed
object's index.

Snapshots, checkpoints, and cloning
-----------------------------------
Single-body objects support a fast capture/restore workflow for RL resets and
rollback debugging. ``captureSingleBodySnapshot(obj, snapshotOut)`` fills a
``SingleBodySnapshot`` (object id, type, body type, pose, linear/angular
velocity, material, and collision group/mask). ``restoreSingleBodySnapshot``
reapplies it later, either by pointer to the same object or by stable
``ObjectId``; pass ``restoreCollisionProperties=false`` to skip material and
collision filter restoration. The snapshot is plain data and cheap to copy and
reuse. ``cloneSingleBodyObject(source, name)`` produces a new single-body
object with the same primitive shape, mass, pose, velocity, body type,
material, collision filter, and appearance. It supports spheres, boxes,
cylinders, and capsules and returns ``nullptr`` for other types.

To rewind the whole world, ``World::captureCheckpoint()`` returns an in-memory
``World::Checkpoint`` of the runtime state, including controls, pending forces,
sensors, sleeping islands, and contact-solver history, and
``restoreCheckpoint(checkpoint)`` restores it. Capture between complete steps
(not between ``integrate1()`` and ``integrate2()``) and keep the
object/constraint/sensor topology unchanged until the restore; otherwise the
restore is rejected and returns ``false``. Checkpoints belong to the world that
captured them and are not files.

.. code-block:: cpp

    raisim::World::SingleBodySnapshot snapshot;
    world.captureSingleBodySnapshot(sphere, snapshot);
    // ... simulate, then reset:
    world.restoreSingleBodySnapshot(sphere, snapshot);

    auto* twin = world.cloneSingleBodyObject(sphere, "sphere_copy");

Saving the World to an XML File
================================
``raisim::World::exportToXml(dir, file)`` (or ``exportToXml(path)``) saves the
current world to a RaiSim XML file that ``raisim::World(path)`` can load again.
The file stores gravity, the time step, solver, contact and sleeping settings,
materials, objects with their current state, and tendons (see
:doc:`WorldConfigurationFile`). Unnamed objects receive generated names. An
articulated system that was not loaded from a file is written as a URDF file
(``model_<index>.urdf``) in ``dir``, next to the XML file.

Stepping and time
=================

``World::integrate()`` advances the world by one timestep. It is equivalent to
calling ``integrate1()`` and then ``integrate2()``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Method
     - Work performed
   * - ``setTimeStep(dt)``
     - Updates the world timestep, contact solver timestep, and object-local
       timestep state. A non-finite or non-positive ``dt`` is a fatal error.
   * - ``integrate1()``
     - Clears previous contacts, runs collision detection, registers contacts,
       and calls each object's first pre-solver update hook.
   * - ``integrate2()``
     - Runs the second pre-solver update hook, solves contacts, integrates
       object state, and advances the world time.
   * - ``integrate()``
     - Runs ``integrate1(); integrate2();``.
   * - ``integrateNoContactDetection()``
     - Replaces ``integrate1()`` when contacts are not needed: clears previous
       contacts and updates collision geometry and the first pre-solver hook,
       but skips collision detection and contact-problem construction. Call
       ``integrate2()`` afterwards to complete the step.

``getWorldTime()`` returns the integrated simulation time, and
``setWorldTime(time)`` can manually adjust it. For visualization through
``RaisimServer``, prefer ``RaisimServer::integrateWorldThreadSafe()`` so the
server's background reader and user interactions are synchronized.

Collision Detection and Broadphase
==================================
RaiSim performs collision detection in two stages: a broadphase pass that filters
candidate pairs using axis-aligned bounding boxes (AABBs), followed by a
narrowphase pass (analytic tests, SAT, MPR, and GJK/EPA depending on the shape pair)
to generate contact points. You can configure the
contact detector via :code:`setContactSettings` or adjust only the broadphase
via :code:`setBroadphaseSettings`. These settings should be updated when the
world is not stepping.

Broadphase options are defined by :code:`contact::BroadphaseType`:

* :code:`None` (brute-force pairs, useful for debugging or very small scenes)
* :code:`Sap3Axis` (sweep-and-prune, default)
* :code:`MultiBoxPrune` (grid-based broadphase for large worlds)

The current ``contact::ContactSettings`` defaults are:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Setting
     - Default
   * - ``gjkMaxIterations``, ``gjkTolerance``
     - ``32``, ``1e-6``
   * - ``epaMaxIterations``, ``epaTolerance``
     - ``64``, ``1e-4``
   * - ``maxContactsPerPair``
     - ``8``
   * - ``sweptCcdEnabled``
     - ``false``
   * - ``sweptCcdMinSpeed``
     - ``0.0``
   * - ``sweptCcdSpeculativeMargin``
     - ``1e-4``
   * - ``broadphase.type``
     - ``Sap3Axis``
   * - ``broadphase.mbpUseWorldBounds``
     - ``false`` (the grid covers the padded bounds of all bodies)
   * - ``broadphase.mbpWorldMin`` / ``mbpWorldMax``
     - ``{-100, -100, -100}`` / ``{100, 100, 100}``; used only when
       ``mbpUseWorldBounds`` is ``true``
   * - ``broadphase.mbpCellSize``
     - ``{1, 1, 1}``
   * - ``broadphase.mbpPadding``
     - ``0.5``
   * - ``broadphase.mbpMaxCellsPerAxis`` / ``mbpMaxCellsPerObject``
     - ``128`` / ``64``

Example broadphase configuration (MultiBoxPrune):

.. code-block:: cpp

  #include <raisim/World.hpp>
  #include <raisim/contact_engine/contact_engine.h>

  raisim::World world;
  auto settings = world.getContactSettings();
  settings.broadphase.type = contact::BroadphaseType::MultiBoxPrune;
  settings.broadphase.mbpWorldMin = {-50.0, -50.0, -2.0};
  settings.broadphase.mbpWorldMax = { 50.0,  50.0, 10.0};
  settings.broadphase.mbpCellSize = {  1.0,   1.0,  1.0};
  settings.broadphase.mbpUseWorldBounds = true;
  world.setContactSettings(settings);

Collision groups and masks still gate which pairs are considered in both
broadphase and narrowphase.

Contact Solver Settings
=======================

``World::setERP(erp, erp2)`` updates the contact solver's error-reduction
parameters. ``World::setContactSolverParam(alpha_init, alpha_min, alpha_decay,
maxIter, threshold)`` updates the solver configuration. The alpha arguments are
kept for compatibility and ignored; the effective parameters are ``maxIter`` and
``threshold``. The solver defaults are ``maxIteration = 150`` and
``error_to_terminate = 1e-8``.

The contact solver sweeps through the contacts forward or backward and flips the
direction after every solve, so consecutive steps alternate.
``World::setContactSolverIterationOrder(order)`` sets the direction of the next
solve (``true``: forward). Call it before a step when the result must not
depend on how many solves preceded it, for example when replaying from a
restored state; see :doc:`Determinism`. Articulated-system loop constraints
(pins and equality constraints) are eliminated before the solve and do not
affect the sweep; only pin rows added directly to the solver force forward
sweeps.

Sleeping islands
================
RaiSim can skip simulation for *sleeping islands*: groups of dynamic objects
connected by contacts. Sleeping is **enabled by default**. An island goes to
sleep when all objects in the island remain quiet for a configurable number of
consecutive steps (:code:`quietSteps`, default **5**) and their maximum linear
and angular velocities stay below the configured thresholds (defaults:
**linear 0.002 m/s**, **angular 0.01 rad/s**). Disabling sleeping wakes every
sleeping object.

Notes:

* The quiet counter resets whenever either speed threshold is exceeded.
* The quiet duration is ``quietSteps * timeStep`` (5 ms at a 1 ms timestep).
  A pendulum can remain below both thresholds near a turning point for five
  steps and go to sleep. Use a longer window for slower oscillations or smaller
  timesteps when this occurs.
* Only **dynamic** objects participate in sleeping islands.
* Any user modification (e.g., changing state) keeps the island awake.
* Contacts between awake and sleeping islands will wake the sleeping island
  on the next step.

Configuration API:

.. code-block:: cpp

  world.setSleepingEnabled(true);
  world.setSleepingParameters(/*linear*/ 0.002, /*angular*/ 0.01, /*quietSteps*/ 5);
  world.setSleepingVelocityThresholds(0.002, 0.01);
  world.wakeObject(obj);   // wakes the object's island
  world.wakeAll();

You can query whether a specific object is sleeping with
:code:`isObjectSleeping`.

API
=========

.. doxygenclass:: raisim::World
   :members:
