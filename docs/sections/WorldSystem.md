# World

The `raisim::World` class owns simulation resources: objects, constraints,
materials, collision detection, contact solving, sensors, time, and gravity. All
objects within a single World instance can collide with one another unless their
collision group and mask settings disable that pair. See [Contact and Collision](./Contact.md) and
[Collision Detection and Colliders](./CollisionDetection.md) for contact and collision details.

A `raisim::World` can be instantiated in four ways. The constructor that takes
a path chooses the loader from the file extension and content.

1. **From a USD file.** Pass a `.usd`, `.usda`, `.usdc`, or `.usdz`
   path to the constructor. RaiSim opens the USD stage and imports one USD
   Physics articulation (the default prim if it has
   `PhysicsArticulationRootAPI`, otherwise the first such prim) as an
   `ArticulatedSystem`, with its rigid bodies, joints, and supported
   collision shapes, and applies the stage's physics-scene gravity and
   physics materials when present. Other prims, such as free rigid bodies outside that
   articulation, are not imported, so add the ground and other objects in
   code. A stage without an articulation root throws `std::runtime_error`.
   See [OpenUSD Loading](./OpenUSD.md) for the full import semantics and tooling.

   ```cpp
   raisim::World world("robot.usd");
   world.addGround();
   ```

   Use this entry point for robots authored in USD-native tools such as Isaac
   Sim, Omniverse, and Blender.

2. **From a RaiSim XML or MuJoCo MJCF configuration file.** Pass the file path
   to the same constructor. The file content, a `<raisim>` or a `<mujoco>`
   element, selects the loader; the extension does not matter. Use this when
   the scene is hand-edited, template-driven, or already lives in one of those
   formats. See [World Configuration File](./WorldConfigurationFile.md) for the RaiSim XML schema and
   the MJCF notes below.

3. **From a RaiSim Engine scene file** (from v2.8.0; earlier packages do not
   include the reader). Pass a `.rscene` path to the
   same constructor. RaiSim reads the solver settings and contact materials,
   sampled terrain, primitive, mesh and `.rasset` bodies, and wires;
   [.rscene File](./RsceneFile.md) documents the format. The parsed scene stays available
   through `World::getRscene()`, so rayrai can apply the rest of the scene,
   as RaiSim Engine 2's viewport shows it (render settings, sky, lights, fog,
   visual-only objects, instanced visuals, terrain foliage and textures, and
   the camera). From v2.8.2 (not yet released), a `RayraiWindow` applies it
   when it is constructed:

   ```cpp
   auto world = std::make_shared<raisim::World>("forest.rscene");
   raisin::RayraiWindow viewer(world, 1280, 800);  // shows the whole scene
   ```

   To change the file's render settings in C++:

   ```cpp
   auto render = raisin::rsceneRenderSettings(*world->getRscene());
   render.quality.viewerMsaaSamples = 8;
   raisin::setRsceneRenderSettings(viewer, render);
   ```

   Content the reader cannot reproduce exactly, such as articulated systems,
   compounds or nodes that inherit another node's transform, is a fatal error
   naming the line rather than being dropped. The scene's solver settings replace the
   world's defaults; Engine 2's defaults differ from `raisim::World`'s (see
   [Solver defaults](./RsceneFile.md#solver-defaults)). See
   [Rayrai Example: Forest From .rscene](./examples/rayrai/rayrai_forest_from_rscene.md) for a complete example.
   An XML world configuration file can include a scene
   (`<include value="[THIS_DIR]/scene.rscene"/>`) and add a robot to it; see
   [A robot in a scene](./RsceneFile.md#a-robot-in-a-scene).

4. **Programmatically.** Default-construct an empty world and add objects
   in code:

   ```cpp
   raisim::World world;
   world.addGround();
   auto* box = world.addBox(1.0, 1.0, 1.0, 1.0);
   ```

These methods can be combined: load an initial file and then add or remove
objects in code before stepping.

## MJCF files

The MJCF (MuJoCo file format) reader is experimental. MJCF files are loaded
with the same `raisim::World` constructor used for RaiSim XML:

```cpp
raisim::World world("rsc/mjcf/gymnasium/hopper.xml");
```

The reader supports the subset used by the bundled MJCF examples: static
`worldbody` geoms, articulated bodies (each top-level `body` becomes one
`ArticulatedSystem`), free, ball, slide, and hinge joints, primitive geoms,
mesh assets, inertial tags, defaults (`class`, `childclass` for the
elements below a body, and `main` for the main class; a `freejoint` takes
no defaults, as in MuJoCo), compiler `angle`/`eulerseq` settings, `compiler meshdir`, `compiler settotalmass` (every body's mass and inertia scaled by
one factor), material colors, `option timestep`, `equality connect`
constraints between bodies of the same top-level body (the `anchor` in
`body1`'s frame, as in MuJoCo), and tendons (see [Tendon API and model files](./tendons/Reference.md)).
Mesh asset paths are resolved relative to the MJCF file directory, or relative
to `compiler meshdir` when it is provided. MJCF files without an `asset`
block are accepted. Following the MJCF default, a `geom` with no `type`
attribute is treated as a sphere and must provide at least one `size` value
for its radius. As in MuJoCo, a shape takes the first values of its size: a
geom's own size replaces the first values of an inherited one and may give no
more values than its shape has, and a mesh geom ignores a size (with a warning
when the geom states it). A default class may give each element type once. A geom's `mass`
replaces its `density`; a geom without volume, such as a plane, ignores it as
in MuJoCo, and so does a mesh geom, whose volume RaiSim does not compute (give
its body an `inertial` element instead). A joint with `limited="false"`
ignores its `range`.

A `worldbody` plane whose normal is turned off the vertical is held by a
fixed articulated system, named after the geom if the geom has a name and added
after the model's bodies; a level one is RaiSim's ground. Like any fixed
articulated system, it reports contacts with the static `worldbody` geoms it
touches, which MuJoCo does not; they carry no impulse.

`<include file="..."/>` is replaced by the elements inside the root of the
named file, as in MuJoCo, and included files may include others. As in MuJoCo,
a relative path, of an included file or of a mesh file in one, is looked for
in the directory of the main file (for a mesh, its `meshdir`) and then in the
directory of the file that has it. A file that includes itself, directly or
through other files, an `include` with children and a file that cannot be
read are errors. Sections repeated by the includes, such as a scene's
`worldbody` and the included robot's, are merged into one. A later top-level
`default` adds to the main class, but, as in MuJoCo, not to the classes
defined before it, which are copies of the main class as it was where they were
defined.

Geoms pair up for contact as in MuJoCo: two geoms touch when the `contype`
of either shares a bit with the `conaffinity` of the other, so a model whose
bodies have `conaffinity="0"` (Gymnasium's Ant, HalfCheetah, Walker2d) touches
its floor but not itself. A geom with both `contype` and `conaffinity` zero
touches nothing. The loader turns each distinct `contype`/`conaffinity`
pair into a RaiSim collision group bit and a mask that pairs it accordingly
(see [Contact and Collision](./Contact.md)). The bits belong to the world: a file loaded later reuses
the bits of the pairs it shares with earlier files and extends their masks, so
geoms of different files pair up as in MuJoCo too, and loading a model again
takes no new bits. MuJoCo's default pair (1, 1) becomes RaiSim's default group
with every mask bit, as RaiSim's default objects have. A geom that MuJoCo would
let touch a default geom touches RaiSim's default objects, its ground and its
heightmaps. Other RaiSim bodies touch MJCF geoms only through the default and
static groups, as long as no group bit they use is shared with a pair. A new
pair takes a bit that no other body of the world uses, and only when every bit
is taken does it share one, with a warning: a body in that group then touches
the geoms the pair touches, where the body's mask allows it. The MJCF files of
a world can have 62 distinct pairs besides (1, 1), one for each bit that is
neither the default nor the static group, counting those of removed models,
whose bits stay taken.

It is not a complete MuJoCo replacement. `<actuator>` elements are not
imported (the example programs apply joint torques from C++), equality types
other than `connect`, `joint` and tendon equalities are fatal errors, and
integrators other than Euler are ignored with a warning.
Validate other MuJoCo features before relying on them.

Example targets:

- `mjcf_gymnasium_hopper` loads and actuates Gymnasium's Hopper model.
- `mjcf_gymnasium_walker2d` loads and actuates Gymnasium's Walker2d model.
- `mjcf_gymnasium_humanoid` loads Gymnasium's Humanoid model and drops it
  from a raised arbitrary configuration.

For callers that already hold a constructed `raisim::World` and want to
populate it from an MJCF file at runtime (instead of constructing the world
from the file path), use `World::loadMjcfFile`:

```cpp
raisim::World world;
world.loadMjcfFile("rsc/mjcf/gymnasium/hopper.xml");
```

This is the same loader that `World(configFile)` calls internally for a
`<mujoco>` file. The world should normally be empty before the call: MJCF
files often declare ground planes, `worldbody` geoms, and `option` blocks
that change world-level state such as the time step. The rayrai TCP viewer uses
this entry point in its drag-and-drop articulated-system inspector mode (see
[Rayrai TCP Viewer](./RayraiTcpViewer.md)).

## Adding New Objects

To add a new object of type X, utilize the `addX` method.
For example, to add a sphere:

```cpp
raisim::World world;
auto sphere = world.addSphere(0.5, 1.0);
```

`sphere` is a pointer to the object, which the world owns. Use it to read
and modify the object's state. The pointer becomes invalid when the object is
removed with `World::removeObject`.

Most object-creation methods accept optional `material`, `collisionGroup`, and `collisionMask` arguments.
A ground (`addGround`) has a fixed collision group and exposes only a
collision mask. Height maps (`addHeightMap`) accept a group that defaults to
the static collision group `RAISIM_STATIC_COLLISION_GROUP`.
Collision groups and masks are described in [Contact and Collision](./Contact.md).
The `material` argument names the material that governs contact dynamics; see [Material System](./MaterialSystem.md).

The object types are described in [Objects](./Object.md).

Upon object addition, a name may be assigned:

```cpp
sphere->setName("ball");
```

An object pointer can be retrieved by name:

```cpp
auto* ball = world.getObject("ball");
```

An object may consist of multiple bodies (e.g., an articulated system).
A **local index** is used to designate individual bodies.
To maintain API consistency, many methods require the local index argument even for single-body objects.
For single-body objects, the local index is ignored, and users may pass 0 to comply with the API.

### Stable object identifiers

`Object::Id` (a `uint64_t`) is assigned when an object is added to a world
and remains valid for the lifetime of that object even when other objects are
removed; ids are not reused within a world. `World::getObjectById(id)`
returns a pointer, or `nullptr` when the object has been removed;
`isObjectAlive(id)` returns the same information as a `bool`.
`getIndexInWorld()` is fine for steady-state lookups, but in code that adds
and removes objects (RL resets, asset streaming) prefer the stable id: when an
object is removed, the last object in the world's list moves into the removed
object's index.

### Snapshots, checkpoints, and cloning

Single-body objects support a fast capture/restore workflow for RL resets and
rollback debugging. `captureSingleBodySnapshot(obj, snapshotOut)` fills a
`SingleBodySnapshot` (object id, type, body type, pose, linear/angular
velocity, material, and collision group/mask). `restoreSingleBodySnapshot`
reapplies it later, either by pointer to the same object or by stable
`ObjectId`; pass `restoreCollisionProperties=false` to skip material and
collision filter restoration. The snapshot is plain data and cheap to copy and
reuse. `cloneSingleBodyObject(source, name)` produces a new single-body
object with the same primitive shape, mass, pose, velocity, body type,
material, collision filter, and appearance. It supports spheres, boxes,
cylinders, and capsules and returns `nullptr` for other types.

To rewind the whole world, `World::captureCheckpoint()` returns an in-memory
`World::Checkpoint` of the runtime state, including controls, pending forces,
sensors, sleeping islands, and contact-solver history, and
`restoreCheckpoint(checkpoint)` restores it. Capture between complete steps
(not between `integrate1()` and `integrate2()`) and keep the
object/constraint/sensor topology unchanged until the restore; otherwise the
restore is rejected and returns `false`. Checkpoints belong to the world that
captured them and are not files.

```cpp
raisim::World::SingleBodySnapshot snapshot;
world.captureSingleBodySnapshot(sphere, snapshot);
// ... simulate, then reset:
world.restoreSingleBodySnapshot(sphere, snapshot);

auto* twin = world.cloneSingleBodyObject(sphere, "sphere_copy");
```

## Saving the World to an XML File

`raisim::World::exportToXml(dir, file)` (or `exportToXml(path)`) saves the
current world to a RaiSim XML file that `raisim::World(path)` can load again.
The file stores gravity, the time step, solver, contact and sleeping settings,
materials, objects with their current state, and tendons (see
[World Configuration File](./WorldConfigurationFile.md)). Unnamed objects receive generated names. An
articulated system that was not loaded from a file is written as a URDF file
(`model_<index>.urdf`) in `dir`, next to the XML file.

## Stepping and time

`World::integrate()` advances the world by one timestep. It is equivalent to
calling `integrate1()` and then `integrate2()`.

| Method | Work performed |
| --- | --- |
| `setTimeStep(dt)` | Updates the world timestep, contact solver timestep, and object-local timestep state. A non-finite or non-positive `dt` is a fatal error. |
| `integrate1()` | Clears previous contacts, runs collision detection, registers contacts, and calls each object's first pre-solver update hook. |
| `integrate2()` | Runs the second pre-solver update hook, solves contacts, integrates object state, and advances the world time. |
| `integrate()` | Runs `integrate1(); integrate2();`. |
| `integrateNoContactDetection()` | Replaces `integrate1()` when contacts are not needed: clears previous contacts and updates collision geometry and the first pre-solver hook, but skips collision detection and contact-problem construction. Call `integrate2()` afterwards to complete the step. |

Between `integrate1()` and `integrate2()`, which works on the contacts
and states `integrate1()` made for every object, adding or removing an
object, a tendon or a wire throws `std::logic_error` and leaves the world as
it was. The step stays open until `integrate2()` (or the next
`integrate()`) completes it, so code that calls `integrate1()` only to read
contacts must complete the step before it edits the scene.

`getWorldTime()` returns the integrated simulation time, and
`setWorldTime(time)` can manually adjust it. For visualization through
`RaisimServer`, prefer `RaisimServer::integrateWorldThreadSafe()` so the
server's background reader and user interactions are synchronized.

## Collision Detection and Broadphase

RaiSim performs collision detection in two stages: a broadphase pass that filters
candidate pairs using axis-aligned bounding boxes (AABBs), followed by a
narrowphase pass (analytic tests, SAT, MPR, and GJK/EPA depending on the shape pair)
to generate contact points. You can configure the
contact detector via `setContactSettings` or adjust only the broadphase
via `setBroadphaseSettings`. These settings should be updated when the
world is not stepping.

Broadphase options are defined by `contact::BroadphaseType`:

- `None` (brute-force pairs, useful for debugging or very small scenes)
- `Sap3Axis` (sweep-and-prune, default)
- `MultiBoxPrune` (grid-based broadphase for large worlds)

The current `contact::ContactSettings` defaults are:

| Setting | Default |
| --- | --- |
| `gjkMaxIterations`, `gjkTolerance` | `32`, `1e-6` |
| `epaMaxIterations`, `epaTolerance` | `64`, `1e-4` |
| `maxContactsPerPair` | `8` |
| `sweptCcdEnabled` | `false` |
| `sweptCcdMinSpeed` | `0.0` |
| `sweptCcdSpeculativeMargin` | `1e-4` |
| `broadphase.type` | `Sap3Axis` |
| `broadphase.mbpUseWorldBounds` | `false` (the grid covers the padded bounds of all bodies) |
| `broadphase.mbpWorldMin` / `mbpWorldMax` | `{-100, -100, -100}` / `{100, 100, 100}`; used only when `mbpUseWorldBounds` is `true` |
| `broadphase.mbpCellSize` | `{1, 1, 1}` |
| `broadphase.mbpPadding` | `0.5` |
| `broadphase.mbpMaxCellsPerAxis` / `mbpMaxCellsPerObject` | `128` / `64` |

Settings RaiSim cannot use are fatal errors, from `setContactSettings()`, a
world file or a scene alike, and the world keeps its settings: fewer than one
GJK or EPA iteration, a negative or non-finite tolerance, a non-finite
swept-CCD speed or margin (negative ones count as zero), an unknown broadphase
type, fewer than one cell per axis or per object, a cell size or padding that
is not finite (or a negative padding), and, with `mbpUseWorldBounds`, world
bounds that are not finite or whose min is above max. A cell size that is not
positive is chosen from the span of the grid. The grid has at most 2^24 cells
(the default 128 per axis makes 2^21): a finer one gets larger cells, shared
alike by the axes that span more than one.

Example broadphase configuration (MultiBoxPrune):

```cpp
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
```

Collision groups and masks still gate which pairs are considered in both
broadphase and narrowphase.

## Contact Solver Settings

`World::setERP(erp, erp2)` updates the contact solver's error-reduction
parameters. `World::setContactSolverParam(alpha_init, alpha_min, alpha_decay, maxIter, threshold)` updates the solver configuration. The alpha arguments are
kept for compatibility and ignored; the effective parameters are `maxIter` and
`threshold`. The solver defaults are `maxIteration = 150` and
`error_to_terminate = 1e-8`.

The contact solver sweeps through the contacts forward or backward and flips the
direction after every solve, so consecutive steps alternate.
`World::setContactSolverIterationOrder(order)` sets the direction of the next
solve (`true`: forward). Call it before a step when the result must not
depend on how many solves preceded it, for example when replaying from a
restored state; see [Determinism](./Determinism.md). Articulated-system loop constraints
(pins and equality constraints) are eliminated before the solve and do not
affect the sweep; only pin rows added directly to the solver force forward
sweeps.

## Sleeping islands

RaiSim can skip simulation for *sleeping islands*: groups of dynamic objects
connected by contacts. Sleeping is **enabled by default**. An island goes to
sleep when all objects in the island remain quiet for a configurable number of
consecutive steps (`quietSteps`, default **5**) and their maximum linear
and angular velocities stay below the configured thresholds (defaults:
**linear 0.002 m/s**, **angular 0.01 rad/s**). Disabling sleeping wakes every
sleeping object.

Notes:

- The quiet counter resets whenever either speed threshold is exceeded.
- The quiet duration is `quietSteps * timeStep` (5 ms at a 1 ms timestep).
  A pendulum can remain below both thresholds near a turning point for five
  steps and go to sleep. Use a longer window for slower oscillations or smaller
  timesteps when this occurs.
- Only **dynamic** objects participate in sleeping islands. A sleeping object
  switched to `BodyType::KINEMATIC` stops sleeping.
- Any user modification (e.g., changing state) keeps the island awake.
- Contacts between awake and sleeping islands will wake the sleeping island
  on the next step.
- A moving kinematic body (non-zero velocity, or a pose set since the last
  step) wakes the sleeping islands its collision bodies reach: anywhere between
  where it was at the last step and where its velocity carries it in this step,
  so a platform moved away by `setPosition()` wakes what slept on it. It also
  keeps the islands it touches awake. A kinematic body at rest lets the bodies
  on it sleep.

Configuration API:

```cpp
world.setSleepingEnabled(true);
world.setSleepingParameters(/*linear*/ 0.002, /*angular*/ 0.01, /*quietSteps*/ 5);
world.setSleepingVelocityThresholds(0.002, 0.01);
world.wakeObject(obj);   // wakes the object's island
world.wakeAll();
```

You can query whether a specific object is sleeping with
`isObjectSleeping`.

## API

- [`raisim::World`](/api/raisim/World.md)
