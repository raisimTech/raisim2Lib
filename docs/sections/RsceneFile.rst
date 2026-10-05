############
.rscene File
############

``.rscene`` is the text scene format written by RaiSim Engine 2, the internal
world-authoring tool (see :doc:`RaisimEngine2`). RaiSim builds the physics
world from it, and rayrai shows the rest of the scene the way Engine 2's editor
viewport shows it: render settings and sky, lights, local fog, reflection
probes, visual-only objects, instanced visuals, terrain foliage and textures,
and the camera. This page describes the format as these two readers see it:
the syntax, every record and field they read, how references to other files
are resolved, and what they reject or ignore.

Availability
============

The reader is new in v2.7.1, which is not released yet (see
:doc:`changelog/v2.7`). It is part of the RaiSim and rayrai libraries:

* ``raisim/Rscene.hpp`` (RaiSim): ``raisim::World(path)`` for a path that ends
  in ``.rscene`` (in any letter case), ``World::getRscene()``, and the
  lower-level ``raisim::rscene::load``, ``rscene::applyPhysics`` and
  ``rscene::addToWorld``.
* ``rayrai/RsceneVisuals.hpp`` (rayrai): ``raisin::applyRscene`` and
  ``raisin::rsceneRenderSettings``.

The v2.7.0 package ships neither header, so code that uses them does not
compile against it. Its ``World(path)`` constructor does not recognize the
extension either: it searches the file for a ``<raisim>`` or ``<mujoco>``
element, finds neither, and returns an empty world without an error.

Engine 2 itself is not public, so scenes reach you as files, for example
``rsc/forest/rayrai_forest.rscene`` or ``rsc/rayrai/blue_wall/blue_wall.rscene``.
You can also write a file by hand or generate one from a script; the rules on
this page are what the readers enforce. `Example file`_ is a complete scene
that loads.

Loading a scene
===============

Physics only
------------

.. code-block:: cpp

   #include "raisim/World.hpp"

   raisim::World world("scene.rscene");
   for (int i = 0; i < 1000; ++i) world.integrate();

``World(path)`` parses and checks the whole file first, then applies the solver
settings and contact materials and adds the terrain, the bodies and the wires.
The parsed scene stays available through ``world.getRscene()``, which returns
``nullptr`` for a world built any other way. A file that the reader rejects is
a RaiSim fatal error, which exits the process unless you install a fatal
callback (see :doc:`LoggingSystem`). If the callback returns instead of
throwing, the world keeps the bodies added before the error and
``getRscene()`` returns ``nullptr``.

To add a scene to a world you already have, or to receive errors as
exceptions, call the steps yourself:

.. code-block:: cpp

   #include "raisim/Rscene.hpp"
   #include "raisim/World.hpp"

   raisim::World world;
   try {
     const auto scene = raisim::rscene::load("scene.rscene");      // parse and check
     const auto report = raisim::rscene::addToWorld(scene, world); // settings and bodies
     std::cout << report.bodies.size() << " bodies, " << report.colliders << " colliders\n";
   } catch (const std::exception& error) {
     std::cerr << error.what() << '\n';  // e.g. "scene.rscene: line 12 (object): ..."
   }

``rscene::load`` reads and checks the file without touching a world.
``rscene::addToWorld`` calls ``rscene::applyPhysics(scene.physics, world)``,
which overwrites the world's solver settings and sets the contact material
pairs, then adds one height map per ``terrain_region`` record, one body per
``object`` record that has a physics body (one body per collision shape for a
``.rasset`` object; ``report.bodies`` holds its first part and
``report.colliders`` counts them all) and one spatial tendon per ``wire``
record (``report.wires``). A world filled this way does not keep the scene, so
``world.getRscene()`` stays ``nullptr``: keep the ``Scene`` yourself if you
also want to render it.

Physics and rayrai rendering
----------------------------

.. code-block:: cpp

   #include "raisim/World.hpp"
   #include "rayrai/RayraiWindow.hpp"
   #include "rayrai/RsceneVisuals.hpp"

   auto world = std::make_shared<raisim::World>("scene.rscene");
   raisin::RayraiWindow viewer(world, 1280, 800);

   // The file's render settings, optionally changed in C++ before they are applied.
   auto render = raisin::rsceneRenderSettings(*world->getRscene());
   render.quality.addViewerFillLights = false;
   const auto visuals = raisin::applyRscene(*world->getRscene(), viewer, render);

   while (running) {
     world->integrate();
     viewer.update(1280, 800, false, 0, 0, false);
   }

``applyRscene(scene, viewer)`` without the third argument applies
``rsceneRenderSettings(scene)`` unchanged. Either form applies, in this order,
the quality preset and render settings (``rayrai_render`` and
``environment``), the background (a plain color, an HDR image, or the rayrai
sky configured by ``weather``), the objects the RaiSim world does not draw
(visual-only objects and the visual meshes of visible ``.rasset`` objects,
returned in ``visuals.objects``), the lights, the local fog volumes, the
reflection probes, one ``InstancedVisuals`` batch per enabled
``instanced_visual`` record (``visuals.instancedVisuals``, in file order), the
terrain foliage (``visuals.foliage``), the terrain textures and the first
enabled camera. Bodies and height maps are drawn from the world like any other
RaiSim object; the hidden ``.rasset`` colliders are not drawn.

* A render record the file omits, and every key it omits, takes Engine 2's
  default, so a file without ``environment``, ``weather`` or
  ``rayrai_render`` records still applies.
* RaiSim stores ``environment``, ``weather`` and ``rayrai_render`` without
  interpreting them, so rayrai is the first to check their keys.
* ``applyRscene`` calls ``setRenderQualitySettings`` and replaces the viewer's
  additional lights, local fog volumes, projected decals and irradiance
  volumes, so add your own afterwards. Call it once per viewer: batch and
  visual names must be unique, so applying a scene to the same viewer twice is
  a fatal error.
* With weather enabled, rayrai derives exposure, environment intensity, fog and
  shadow strength from the base settings and the sky. Change the base values in
  ``render.quality`` and ``render.weather`` before ``applyRscene`` instead of
  reading them back from the viewer afterwards. After ``applyRscene``, the
  viewer's own setters (``setRenderQualitySettings``, ``getLight``,
  ``getCamera``) still change anything.
* Both functions throw ``std::runtime_error`` for a malformed render record, a
  missing texture, mesh or HDR file, or content that rayrai cannot reproduce
  (see `Rejected content`_).

:doc:`examples/rayrai/rayrai_forest_from_rscene` loads the shipped forest scene
this way. Every ``.rscene`` file in the RaiSim and ``raisim2Lib`` repositories
loads both ways, and the image matches Engine 2's viewport; only two Engine 2
test inputs, which are deliberately not loadable, are excluded.

Physics on a server, rendering in the TCP viewer
------------------------------------------------

.. code-block:: cpp

   #include "raisim/RaisimServer.hpp"

   raisim::World world("scene.rscene");
   raisim::RaisimServer server(&world);
   server.launchServer();
   while (running) server.integrateWorldThreadSafe();

``rayrai_tcp_viewer`` draws the bodies the server streams. It also applies the
scene with ``applyRscene``, from the same file on the server's computer or from
a copy elsewhere, so it shows what the in-process viewer above shows. A viewer
on another computer that lacks the scene's files asks its user, then
downloads them from the server (see :doc:`RaisimServer` and
:ref:`sections/RayraiTcpViewer:RaiSim Engine scenes`).
:doc:`examples/server/rscene_server` serves the forest scene.

File syntax
===========

A ``.rscene`` file is text with one record per line. A trailing ``\r``
(Windows line ending) is treated as white space.

.. code-block:: text

   tag positional0 positional1 ... key=value key=value ...

* **Header.** The first record must be ``raisim_engine_scene 2`` or, in an
  older file, ``raisim_engine_scene 1`` (see `Versioning`_). Blank lines and
  comments may come before it.
* **Tokens** are separated by spaces, tabs or carriage returns. The first token
  is the record tag. A token without ``=`` is a positional field; a token with
  ``=`` is a ``key=value`` field, split at its first ``=``. All positional
  fields must come before the first ``key=value`` field, and a key may appear
  only once in a record. This page numbers positional fields from ``#0``, the
  first field after the tag.
* **Escaping.** ``%XX`` (two hexadecimal digits, in either case) encodes one
  byte, for example ``Forest%20mud`` for ``Forest mud``. Spaces and tabs must
  be escaped because they separate tokens. A value that is exactly ``-`` is
  the empty string, and so is nothing after ``=``; a value that really is a
  dash is written ``%2D``. Engine 2 escapes every byte up to ``0x20`` as well
  as ``%``, ``=`` and ``;``, and writes a lone dash as ``%2D``.
* **Comments.** A line whose first character other than white space is ``#``
  is a comment and is skipped without being split into tokens, so it may
  contain any text. There are no end-of-line comments: a ``#`` after a
  record's fields is read as another field. Engine 2 only accepts a ``#`` in
  the first column, so start comments there in files Engine 2 must also read.
* **Order.** After the header, records may appear in any order, except that a
  ``terrain_splat_layer`` or ``terrain_foliage_layer`` record must follow the
  ``terrain_region`` it names. A repeated ``time_step``, ``gravity``,
  ``solver``, ``asset_root``, ``scene_graph``, ``environment``, ``weather`` or
  ``rayrai_render`` record replaces the earlier one; the other records
  accumulate.
* **Required and optional fields.** The positional fields listed for a record
  are required; a few records accept the shorter layouts of older files, as
  noted. Every ``key=value`` field is optional: a key that is absent or empty
  takes Engine 2's default, listed with the record, exactly as Engine 2's own
  loader does. A present value of the wrong type is an error, where Engine 2
  would silently keep its default. Keys the readers do not use, and positional
  fields after the ones they use, are ignored.

Value types
-----------

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Type
     - Syntax
   * - number
     - A finite decimal floating-point number in the C locale, such as ``0.5``,
       ``-9.81`` or ``1e-08``. ``nan``, ``inf`` and trailing characters are
       errors.
   * - integer
     - A number without a fractional part, in the 32-bit signed range.
   * - flag
     - ``true``, ``1``, ``yes`` or ``on``, or ``false``, ``0``, ``no`` or
       ``off``. Engine 2 writes ``true`` and ``false``.
   * - group
     - An unsigned 64-bit decimal integer, used for collision groups and masks.
       ``18446744073709551615`` sets all 64 bits.
   * - vec2, vec3
     - Two or three positional numbers (``0 0 -9.81``) or, as a key value, two
       or three comma-separated numbers (``1,0,0``).
   * - quat
     - Four positional numbers in ``w x y z`` order.
   * - color
     - Three or four comma-separated numbers, ``r,g,b`` or ``r,g,b,a``; alpha
       defaults to 1.
   * - list
     - Numbers separated by ``,`` or ``;`` (the two are interchangeable), with
       no empty entry and no trailing separator. An empty value is an empty
       list.
   * - name
     - One of the names listed with the field. Names are matched in any letter
       case, and the aliases Engine 2 accepts are accepted too; any other name
       is an error.
   * - text
     - Any escaped string. ``-`` is the empty string.

Versioning
----------

The number in the header is the format version. Engine 2 writes version 2 and
reads versions 1 and 2; the RaiSim reader reads both as well. Within a version,
Engine 2 has added keys over time; a file that lacks a newer key takes the
default listed here.

The two versions differ in one place, the base of ``asset_root``:

* **Version 2**: ``asset_root`` is relative to the directory that contains the
  ``.rscene`` file (or absolute), in Engine 2 and in RaiSim. See
  `Paths and assets`_.
* **Version 1**: Engine 2 read ``asset_root`` relative to the scene file,
  except for a scene inside an Engine 2 project, where it was relative to the
  project root (with ``.`` meaning the project's asset directory) and Engine 2
  also looked for references in the project root. The RaiSim reader does not
  read Engine 2 project files, so it accepts a version-1 file only when its
  ``asset_root`` is absent, ``.`` or absolute, which means the same in both
  cases. Any other version-1 ``asset_root`` is an error. Open and save such a
  scene in Engine 2, which writes version 2, or change the header to version 2
  if the path is relative to the scene file.

Every scene shipped in the repositories is a version-1 file with
``asset_root .`` whose references are relative to the scene file, so both
versions read them the same way.

Conventions
===========

* **Units** are SI: metres, kilograms and seconds, and m/s for speeds. Camera
  fields of view, cone angles and the weather's angles are in degrees.
* **Frame.** Poses are in RaiSim's world frame: right-handed with ``+z`` up.
* **Rotations** are quaternions in ``w x y z`` order. Primitive bodies
  normalize them; a ``.rasset`` object needs a unit quaternion (squared norm
  within ``1e-5`` of 1); instanced-visual quaternions are normalized by rayrai.
* **Node paths** such as ``/World/Props/Crate`` are names. The path becomes
  the RaiSim object name (``world.getObject("/World/Props/Crate")``) or the
  rayrai visual or batch name. A node also has an ``id`` (default: a
  record-specific prefix and the path, such as ``object_/World/Props/Crate``);
  other records refer to it by id or path.
* **Folders and parents.** ``group`` records are folders: they organize the
  scene tree and carry no transform. A node's ``parentGroupId`` names its
  parent by id or path. A node in a folder, or whose parent does not exist, is
  placed in the world frame, as in Engine 2. A node whose parent is another
  node (an object, light, terrain and so on) would inherit that node's
  transform, which the readers do not evaluate, so it is an error, unless the
  ``scene_graph`` record sets ``inheritedTransforms=false``.
* **Colors** are linear RGB or RGBA with components in [0, 1].
* **Primitive sizes** come from an ``object`` record's scale, radius and height;
  see `object record`_.
* **Height maps** store ``xSamples * ySamples`` heights row by row: the sample
  at grid index ``(x, y)`` is ``heights[y * xSamples + x]``, located at
  ``centerX - xSize/2 + x * xSize/(xSamples - 1)`` and
  ``centerY - ySize/2 + y * ySize/(ySamples - 1)``. Heights are relative to the
  record's center ``z``. Per-sample paint lists use the same order. See
  :doc:`HeightMap`.
* **Cameras** look along their local ``+x`` axis with local ``+z`` up, so the
  identity quaternion looks along world ``+x``.
* **Light directions** point the way the light travels, from the light toward
  the scene. They need not be unit length.

Paths and assets
================

File references (meshes, ``.rasset`` descriptors, textures and HDR images) are
resolved by ``Scene::resolve``, with the same rule Engine 2 uses for a
version-2 scene:

#. An empty or absolute path is used as it is.
#. A relative path is first tried against the directory that contains the
   ``.rscene`` file, and used if that file exists.
#. Otherwise the path is resolved against ``<scene directory>/<asset_root>``,
   whether or not the file exists there; a missing file is reported when it is
   opened. An absolute ``asset_root`` replaces the scene directory.

Without an ``asset_root`` record, or with ``asset_root .``, both steps use the
scene directory. Forward slashes work on every platform.

When Engine 2 saves a scene to a file, it makes the file resolve on its own:

* It writes version 2, with ``asset_root`` relative to the saved file.
* Inside an Engine 2 project, it first copies assets from outside the project
  into the project's asset directory, and ``asset_root`` names that directory
  (for example ``../assets`` for a scene in ``<project>/scenes``) unless the
  scene has its own asset root.
* It rewrites every reference that would no longer resolve to the same file,
  for example a path relative to the project root or to the directory the
  scene was loaded from, as a path relative to the saved file.

A scene saved by Engine 2, inside a project or not, therefore loads in RaiSim
and rayrai with the same files Engine 2 uses. A version-1 scene saved by an
older Engine 2 inside a project may need to be saved again (see
`Versioning`_).

To copy a scene, ``raisim::rscene::referencedFiles(scene)`` lists every file
RaiSim or rayrai reads for it, as sorted absolute paths: the scene file; each
existing file a record references; the visual and collision meshes of
``.rasset`` descriptors; the buffers and images a glTF, GLB, OBJ (through its
``.mtl`` files) or COLLADA mesh references; and the images in a mesh's
``textures``, ``materials`` and ``materials/textures`` folders, which rayrai
uses for a mesh whose materials name no texture. A reference to a missing file
is left out, so loading the scene still reports it. Keeping the files' layout
below their deepest common directory keeps relative references working.
``raisim::rscene::forEachFileReference(scene, visit)`` visits every reference
as written in the file and can rewrite it, for example to point an absolute
path at a local copy. ``RaisimServer`` uses both to share a scene with a
remote TCP viewer.

.rasset descriptors
-------------------

A ``mesh`` object can refer to a ``.rasset`` file, a small XML descriptor that
pairs one visual mesh with collision shapes in the mesh's local frame
(``raisim/Rasset.hpp``):

.. code-block:: xml

   <rasset version="1">
     <visual file="model.gltf"/>
     <collision type="capsule" radius="0.027" height="0.99"
                position="0.001 0.002 0.495" quaternion="1 0 0 0"/>
   </rasset>

Paths inside a descriptor are relative to the descriptor. Collision types are
``box`` (``size``: three full side lengths), ``sphere`` (``radius``),
``cylinder`` and ``capsule`` (``radius`` and ``height``; a capsule's height is
the distance between its hemisphere centers), and ``convex_mesh`` (``file``,
whose convex hull is used). ``position`` defaults to zero and ``quaternion``
(``w x y z``) to identity.

For a ``mesh`` object with a descriptor, RaiSim creates one hidden static body
per collision shape, placed by the object's position, rotation and scale. The
first body is named after the object path and the others
``<path> collision <i>``. RaiSim never loads the visual mesh: rayrai draws it
for a visible object, and an ``instanced_visual`` batch whose ``meshPath``
names the descriptor draws many copies at once. The forest scene draws its
2,068 hidden tree and rock colliders this way with nine batches; seven more
batches draw ground plants from plain glTF files that have no colliders.

Records
=======

.. list-table::
   :header-rows: 1
   :widths: 22 33 45

   * - Record
     - RaiSim
     - rayrai
   * - ``raisim_engine_scene``
     - Header (required, first record).
     -
   * - ``time_step``, ``gravity``, ``solver``
     - Solver, contact and sleeping settings.
     -
   * - ``contact_material``
     - A material pair of the world.
     -
   * - ``asset_root``
     - Resolves mesh and ``.rasset`` paths.
     - Resolves mesh, texture and HDR paths.
   * - ``scene_graph``
     - Whether parents pass on their transforms.
     -
   * - ``group``
     - A folder; no transform.
     -
   * - ``material``
     - Color of the bodies and terrain that reference it.
     - The material of the objects rayrai draws.
   * - ``asset``
     - Nothing.
     - The default material of its objects.
   * - ``terrain_texture``
     - Tints that color the height map.
     - Albedo, normal and displacement textures.
   * - ``terrain_region``
     - A static height map, colored by its paint.
     - Drawn from the world.
   * - ``terrain_splat_layer``
     - Weights that color the height map.
     -
   * - ``terrain_foliage_layer``
     - Stored.
     - Scattered instanced foliage.
   * - ``object``
     - A ground, box, sphere, cylinder, capsule or mesh body, unless
       visual-only.
     - Visual-only objects and visible ``.rasset`` objects.
   * - ``wire``
     - A spatial tendon between two bodies.
     - Drawn from the world.
   * - ``sensor``
     - Nothing (see `sensor record`_).
     -
   * - ``instanced_visual``
     - Stored (enabled batches only).
     - One instanced batch each.
   * - ``light``
     - Stored (visible lights only).
     - The main light and additional lights.
   * - ``local_fog``
     - Stored (enabled volumes only).
     - Local fog volumes.
   * - ``reflection_probe``
     - Stored (enabled probes only).
     - Reflection probes.
   * - ``camera``
     - Stored (enabled cameras only).
     - The first one becomes the viewer camera.
   * - ``environment``
     - Stored.
     - Render settings and background.
   * - ``weather``
     - Stored.
     - Sky and weather.
   * - ``rayrai_render``
     - Stored.
     - Quality preset and renderer settings.
   * - ``snapping``, ``editor_ux``, ``render_bake``, ``prefab_override``
     - Engine 2 editor state; skipped without being checked.
     -

Every other record tag is an error (see `Rejected content`_). In the tables
below, **RaiSim** means the field is applied by ``World(path)`` or
``rscene::addToWorld``, **rayrai** that it is applied by ``applyRscene``,
**check** that the value is checked but not applied, and **ignored** that
Engine 2 writes it but neither reader uses it. The default column lists the
value of an absent key.

Header record
-------------

``raisim_engine_scene <version>``: exactly one positional field, ``1`` or
``2`` (see `Versioning`_).

time_step record
----------------

``time_step <dt>``: the integration time step in seconds (number, **RaiSim**,
``World::setTimeStep``). Without the record, the time step is ``0.0025``.

gravity record
--------------

``gravity <x> <y> <z>``: gravity in m/s² (vec3, **RaiSim**,
``World::setGravity``). Without the record, gravity is ``0 0 -9.81``.

solver record
-------------

``solver <iterations> <tolerance> <erp> <mode> key=value ...``. Without the
record, the reader applies the Engine 2 defaults below, not those of
``raisim::World``; see `Solver defaults`_.

.. list-table::
   :header-rows: 1
   :widths: 30 12 14 44

   * - Field
     - Type
     - Default
     - Use (**RaiSim** unless noted)
   * - ``#0`` iterations
     - integer
     - ``80``
     - Contact-solver iteration limit, see ``contactMaxIterations``.
   * - ``#1`` tolerance
     - number
     - ``1e-7``
     - Contact-solver termination threshold, see ``contactThreshold``.
   * - ``#2`` erp
     - number
     - ``0.2``
     - ``World::setERP(erp, erp2)``.
   * - ``#3`` mode
     - text
     - ``accurate``
     - **ignored**
   * - ``erp2``
     - number
     - ``0``
     - Second argument of ``World::setERP``.
   * - ``defaultFriction``, ``defaultRestitution``,
       ``defaultRestitutionThreshold``, ``defaultStaticFriction``,
       ``defaultStaticFrictionVelocityThreshold``,
       ``defaultRollingFriction``, ``defaultSpinningFriction``
     - number
     - ``0.8``, ``0``, ``0.001``, ``0.8``, ``0.001``, ``0``, ``0``
     - ``World::setDefaultMaterial`` in this order (thresholds in m/s). See
       :doc:`MaterialSystem`.
   * - ``contactAlphaInit``, ``contactAlphaMin``, ``contactAlphaDecay``
     - number
     - ``1``, ``1``, ``1``
     - Passed to ``World::setContactSolverParam``; the solver does not use them.
   * - ``contactMaxIterations``
     - integer
     - ``80``
     - Contact-solver iteration limit, but only when it differs from ``80``;
       otherwise ``#0`` is used.
   * - ``contactThreshold``
     - number
     - ``1e-7``
     - Termination threshold, but only when it differs from ``1e-7``;
       otherwise ``#1`` is used.
   * - ``contactGjkMaxIterations``, ``contactGjkTolerance``,
       ``contactEpaMaxIterations``, ``contactEpaTolerance``
     - integer, number
     - ``32``, ``1e-6``, ``64``, ``1e-4``
     - GJK and EPA limits in ``World::setContactSettings``.
   * - ``maxContactsPerPair``
     - integer
     - ``8``
     - Contact points kept per colliding pair.
   * - ``sweptCcdEnabled``, ``sweptCcdMinSpeed``, ``sweptCcdSpeculativeMargin``
     - flag, number
     - ``false``, ``0``, ``1e-4``
     - Swept continuous collision detection (m/s, m).
   * - ``broadphaseType``
     - name
     - ``sap3_axis``
     - ``sap3_axis``; ``multi_box_prune`` (or ``multiboxprune``, ``mbp``);
       ``none`` (or ``disabled``), the names Engine 2's RaiSim bridge
       recognizes. Engine 2 uses ``sap3_axis`` for any other name; the reader
       rejects it.
   * - ``broadphaseWorldMin``, ``broadphaseWorldMax``, ``broadphaseCellSize``
     - vec3
     - ``-100,-100,-100``, ``100,100,100``, ``1,1,1``
     - Multi-box-prune grid bounds and cell size (m).
   * - ``broadphaseUseWorldBounds``, ``broadphasePadding``,
       ``broadphaseMaxCellsPerAxis``, ``broadphaseMaxCellsPerObject``
     - flag, number, integer
     - ``false``, ``0.5``, ``128``, ``64``
     - Multi-box-prune settings.
   * - ``sleepingEnabled``
     - flag
     - ``true``
     - ``World::setSleepingEnabled``.
   * - ``sleepingLinearVelocityThreshold``,
       ``sleepingAngularVelocityThreshold``, ``sleepingQuietSteps``
     - number, integer
     - ``0.002``, ``0.01``, ``5``
     - ``World::setSleepingParameters`` (m/s, rad/s, steps).
   * - ``worldTime``
     - number
     - ``0``
     - ``World::setWorldTime`` (s).
   * - ``fixedContactSolverIterationOrder``
     - flag
     - ``false``
     - ``World::setContactSolverIterationOrder``. The solver alternates its
       sweep direction every step either way; the value sets the direction of
       the next sweep (``true``: forward, as in a new ``raisim::World``).

Solver defaults
^^^^^^^^^^^^^^^

Engine 2's solver defaults, which the reader also uses for a file without a
``solver`` record, differ from those of a default-constructed
``raisim::World``. A scene that should simulate exactly like C++ code that
uses ``raisim::World``'s defaults must store the right column; the shipped
forest scene does.

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Setting
     - Engine 2 default
     - ``raisim::World`` default
   * - ``time_step``
     - ``0.0025``
     - ``0.005``
   * - ``#0`` iterations (contact-solver iteration limit)
     - ``80``
     - ``150``
   * - ``#1`` tolerance (contact-solver threshold)
     - ``1e-7``
     - ``1e-8``
   * - ``#2`` erp, ``erp2``
     - ``0.2``, ``0``
     - ``1.5``, ``0.002``
   * - ``defaultRestitutionThreshold``
     - ``0.001``
     - ``0.01``
   * - ``defaultStaticFrictionVelocityThreshold``
     - ``0.001``
     - ``1``
   * - ``fixedContactSolverIterationOrder``
     - ``false``
     - ``true``

All other solver keys have the same defaults in both. The `Example file`_
uses the ``raisim::World`` values.

contact_material record
-----------------------

``contact_material <materialA> <materialB> <friction> <restitution> <threshold> <staticFriction> <staticVelocity> <rolling> <spinning>``:
the properties of a pair of RaiSim contact materials (**RaiSim**,
``World::setMaterialPairProp`` with these nine arguments, applied in file
order after the default material). The thresholds are in m/s; see
:doc:`MaterialSystem`. ``id`` is **ignored**.

asset_root record
-----------------

``asset_root <path>``: the directory, relative to the scene file or absolute,
that is searched for references not found next to the scene file (text). A
version-1 file accepts only ``.`` or an absolute path. See `Paths and assets`_
and `Versioning`_.

scene_graph record
------------------

``scene_graph key=value ...``: ``inheritedTransforms`` (flag, default
``true``) says whether a node parented to another node inherits its
transform; see `Conventions`_. The other keys are Engine 2 editor settings and
are **ignored**.

group record
------------

``group <path> key=value ...``: a folder of the scene tree. ``id`` (default
``folder_<path>``) is how nodes refer to it in ``parentGroupId``. A folder
carries no transform. ``parentId``, ``visible``, ``locked`` and ``expanded``
are editor state and are **ignored**, as in Engine 2's viewport.

material record
---------------

``material <name> <r> <g> <b> <a> <metallic> <roughness> [<er> <eg> <eb> <emissiveStrength> <doubleSided> <six textures>] key=value ...``.
The first seven positional fields are required; the next eleven are read when
the record has all of them. A scene without ``material`` records gets
Engine 2's default material ``engine_default`` (color ``0.55, 0.62, 0.72, 1``,
roughness ``0.55``).

RaiSim uses only the color: a body or terrain that references the material
gets the appearance string ``"r, g, b, a"``, and one whose material is empty or
names an unknown id gets ``0.55, 0.62, 0.72, 1.0``. rayrai builds a full
material from the fields below for the objects it draws itself (see
`object record`_), as Engine 2 does; bodies in the physics world are drawn
with their appearance color.

.. list-table::
   :header-rows: 1
   :widths: 34 16 16 34

   * - Field
     - Type
     - Default
     - Use
   * - ``#0`` name
     - text
     -
     - Display name, and the default id.
   * - ``#1`` to ``#4``
     - numbers
     -
     - **RaiSim**, **rayrai**: base color RGBA.
   * - ``#5`` metallic, ``#6`` roughness
     - number
     -
     - **rayrai**
   * - ``#7`` to ``#9`` emissive, ``#10`` emissive strength, ``#11`` double sided
     - numbers, flag
     - ``0,0,0``, ``0``, ``false``
     - **rayrai**
   * - ``#12`` to ``#17`` textures
     - text
     -
     - **rayrai**, **Engine 2**: albedo, normal, metallic, roughness, AO and emissive images.
       Additional material texture keys are preserved and resolved too.
   * - ``id``
     - text
     - ``#0``
     - The identifier that ``object``, ``terrain_region`` and ``asset``
       records reference.
   * - ``alphaMode``
     - name
     - ``opaque``
     - **rayrai**: ``opaque``, ``mask`` (``masked``, ``alpha_scissor``),
       ``hash`` (``alpha_hash``) or ``blend`` (``transparent``).
   * - ``blendMode``
     - name
     - ``mix``
     - **rayrai**: ``mix``, ``add`` (``additive``), ``subtract`` (``sub``),
       ``multiply`` (``mul``) or ``premultiplied_alpha`` (``premultiplied``,
       ``premul``).
   * - ``alphaAntialiasing``
     - name
     - ``off``
     - **rayrai**: ``off``, ``alpha_to_coverage`` (``coverage``) or
       ``alpha_to_coverage_and_to_one`` (``coverage_to_one``).
   * - ``distanceFadeMode``
     - name
     - ``disabled``
     - **rayrai**: ``disabled``, ``pixel_alpha`` (``alpha``), ``pixel_dither``
       (``dither``) or ``object_dither``.
   * - ``normalConvention``, ``emissionOperator``
     - name
     - ``opengl``, ``multiply``
     - **rayrai**: ``opengl`` or ``directx``; ``multiply`` or ``add``.
   * - ``diffuseMode``, ``specularMode``
     - name
     - ``lambert``, ``schlick_ggx``
     - **rayrai**: ``lambert``, ``burley``, ``lambert_wrap`` (``wrap``) or
       ``toon``; ``schlick_ggx``, ``toon`` or ``disabled`` (``off``,
       ``none``).
   * - ``useVertexColorAsAlbedo``, ``vertexColorIsSrgb``,
       ``proximityFadeEnabled``
     - flag
     - ``false``
     - **rayrai**
   * - ``textureBlendFactor``, ``ao``, ``renderPriority``, ``alphaCutoff``,
       ``alphaHashScale``, ``alphaAntialiasingEdge``, ``distanceFadeMin``,
       ``distanceFadeMax``, ``proximityFadeDistance``, ``normalStrength``,
       ``bentNormalStrength``, ``aoStrength``, ``aoLightAffect``,
       ``specularFactor``
     - number
     - ``0``, ``1``, ``0``, ``0.5``, ``1``, ``0.3``, ``0``, ``0``, ``1``,
       ``1``, ``1``, ``1``, ``0``, ``1``
     - **rayrai**: the ``raisin::Material`` fields of the same names.
   * - ``specularColor``
     - color
     - ``1,1,1``
     - **rayrai**
   * - ``textureBlendTexture``
     - text (path)
     - none
     - **rayrai**: texture that blends the material over the mesh's own.
   * - ``albedoTransform*``, ``normalTransform*``,
       ``metallicRoughnessTransform*`` (``Authored``, ``Scale``,
       ``Offset``, ``Rotation``)
     - flag, vec2, number
     - not authored
     - **rayrai**: UV transforms of those texture slots.

The other material keys (more than a hundred PBR settings) are **ignored**, as
in Engine 2's viewport.

asset record
------------

``asset <name> <primitive> <mesh> key=value ...``: an importable asset of the
Engine 2 library. It creates nothing. ``id`` (default ``#0``) and
``defaultMaterial`` (default ``engine_default``) decide whether an object
placed from it (``assetId``) keeps its mesh's own materials (see
`object record`_); every other field is **ignored**.

terrain_texture record
----------------------

``terrain_texture <slot> <id> <name> key=value ...``: one texture layer. A
scene without ``terrain_texture`` records gets Engine 2's two default layers:
``terrain_grass`` (slot 0, color ``0.28,0.43,0.20``) and ``terrain_rock``
(slot 1, color ``0.42,0.40,0.36``, roughness ``0.92``).

.. list-table::
   :header-rows: 1
   :widths: 24 14 16 46

   * - Field
     - Type
     - Default
     - Use
   * - ``#0`` slot
     - integer
     -
     - Layer index that the terrain's ``baseTexture``, paint and splat layers
       refer to.
   * - ``#1`` id, ``#2`` name
     - text
     -
     - **ignored**
   * - ``color``
     - color
     - ``0.28,0.43,0.20``
     - **RaiSim**: the layer's tint in the height map's vertex colors (see
       ``HeightMap::setColor``). ``RaisimServer`` streams them to its
       viewers; in-process rayrai shades a textured terrain with its albedo
       texture instead.
   * - ``albedo``, ``normal``
     - text (path)
     - none
     - **rayrai**: albedo and normal textures.
   * - ``displacementTexture``
     - text (path)
     - none
     - **rayrai**: height texture for parallax mapping. ``displacementMap``,
       ``height``, ``heightTexture``, ``heightmapTexture`` and ``disp`` are
       read too; the first non-empty one counts.
   * - ``normalDepth``
     - number
     - ``1``
     - **rayrai**: normal-map strength (negative values act as 0).
   * - ``roughness``
     - number
     - ``0.82``
     - **rayrai**: roughness, clamped to [0.04, 1].
   * - ``uvScale``
     - number
     - ``4``
     - **rayrai**: tiling of the two layers when a second layer is blended in
       (the texture repeats every ``uvScale`` metres); otherwise rayrai tiles
       the base texture in world metres.

``aoStrength``, ``detile`` and ``displacement`` are **ignored**.

rayrai gives every height map one terrain material, as Engine 2 does: the base
layer of the first terrain whose base layer has an albedo provides the albedo,
normal and displacement textures, normal strength and roughness. When no base
layer has one, the first layer with an albedo provides only the albedo. The
first layer with an albedo and a slot other than the base slot is blended over
the base layer where the terrain's per-sample paint shows it (see
`terrain_region record`_).

terrain_region record
---------------------

``terrain_region <path> <xSamples> <ySamples> <xSize> <ySize> <cx> <cy> <cz> key=value ...``:
a sampled height map. RaiSim adds it with ``World::addHeightMap`` as a static
height map in collision group ``1 << 63`` (bit 63) that collides with every
group, so an object whose mask clears bit 63 passes through the terrain.

.. list-table::
   :header-rows: 1
   :widths: 24 14 14 48

   * - Field
     - Type
     - Default
     - Use
   * - ``#0`` path
     - text
     -
     - **RaiSim**: the height map's name.
   * - ``#1``, ``#2`` samples
     - integer
     -
     - **RaiSim**: samples along x and y, at least 2 each, at most 16384 each
       and 64 Mi in total.
   * - ``#3``, ``#4`` size
     - number (m)
     -
     - **RaiSim**: extent along x and y, positive.
   * - ``#5`` to ``#7`` center
     - vec3 (m)
     -
     - **RaiSim**: center x and y, and the height offset added to every sample.
   * - ``heights``
     - list (m)
     - all ``0``
     - **RaiSim**: ``xSamples * ySamples`` heights relative to the center
       ``z``, in the order described in `Conventions`_.
   * - ``material``
     - text
     - ``engine_default``
     - **RaiSim**: ``material`` id for the appearance color.
   * - ``contactMaterial``
     - text
     - ``default``
     - **RaiSim**: RaiSim material name for contacts (see
       :doc:`MaterialSystem`).
   * - ``baseTexture``
     - integer
     - ``0``
     - **RaiSim**, **rayrai**: slot of the base ``terrain_texture`` layer,
       clamped to [0, 31].
   * - ``texturePrimary``, ``textureSecondary``
     - list of slots (0 to 255)
     - the base slot
     - **RaiSim**, **rayrai**: per-sample paint: the two layers a sample
       shows.
   * - ``textureBlend``
     - list
     - all ``0``
     - **RaiSim**, **rayrai**: per-sample blend from the primary to the
       secondary layer, clamped to [0, 1].
   * - ``vertexColors``
     - colors separated by ``;``
     - all white
     - **RaiSim**: per-sample color multiplier.
   * - ``wetness``
     - list
     - all ``0``
     - **RaiSim**: per-sample wetness; a wet sample is up to 18 % darker.
   * - ``holes``
     - list (0 to 255)
     - all ``0``
     - **RaiSim**: a non-zero sample is colored almost black. The collision
       surface stays solid, as in Engine 2.
   * - ``textureSplattingEnabled``
     - flag
     - ``true``
     - **RaiSim**: whether the splat layers decide the colors.
   * - ``sourceKind``
     - name
     - ``samples``
     - **check**: ``samples`` (``height_samples``, ``inline``).
   * - ``visible``, ``collidable``
     - flag
     - ``true``
     - **check**: must be ``true``.
   * - ``id``, ``parentGroupId``
     - text
     - ``terrain_<path>``, none
     - See `Conventions`_.

A paint list has one value per sample or is empty. RaiSim colors each sample
like Engine 2: with splatting enabled, the tints of the splat layers weighted
by their weights and strengths; where the weights add up to zero, or without
splatting, the primary layer's tint blended toward the secondary layer's;
then multiplied by the vertex color, darkened by wetness, and holes made dark.
A slot without a ``terrain_texture`` layer has the tint ``0.28,0.43,0.20``.

``location``, ``castShadow``, ``receiveShadow``, ``sourcePath``,
``cloneSource``, the ``png*`` and ``procedural*`` keys, ``lodLevels`` and
``collisionShapeSize`` are **ignored**.

terrain_splat_layer record
--------------------------

``terrain_splat_layer <terrain> key=value ...``: a weighted texture layer of
the terrain whose path or id is ``#0``; the ``terrain_region`` must come
first. Keys: ``slot`` (integer, default ``0``), ``enabled`` (flag, ``true``),
``strength`` (number, ``1``) and ``weights`` (list, one weight per sample;
default ``1`` for the base slot and ``0`` otherwise). **RaiSim** uses the
weights for the vertex colors; rayrai's terrain texture follows the per-sample
paint, as in Engine 2.

Without splat layers a terrain has one base layer of weight 1. The first
``terrain_splat_layer`` of a terrain replaces that default layer. Engine 2
gives a terrain's only layer the base slot when all its weights are 1, and the
reader does the same.

terrain_foliage_layer record
----------------------------

``terrain_foliage_layer <terrain> key=value ...``: plants or rocks scattered
over the terrain whose path or id is ``#0`` (the ``terrain_region`` must come
first). RaiSim stores the layer; **rayrai** scatters it exactly as Engine 2
does, so a scene shows the same instances in both.

.. list-table::
   :header-rows: 1
   :widths: 32 14 18 36

   * - Key
     - Type
     - Default
     - Use (**rayrai**)
   * - ``id``, ``name``, ``enabled``
     - text, text, flag
     - ``foliage_<n>``, the id, ``true``
     - A disabled layer is skipped.
   * - ``primitive``, ``meshPath``
     - name, text (path)
     - ``mesh``, none
     - A mesh file or ``.rasset`` descriptor; a ``mesh`` layer without a path,
       or a ``box``, ``sphere``, ``cylinder`` or ``capsule`` layer, draws that
       primitive.
   * - ``size``, ``colorA``, ``colorB``
     - vec3, color, color
     - ``0.25,0.25,0.65``, ``0.18,0.42,0.16``, ``0.42,0.64,0.24``
     - Base size and the two colors each instance blends.
   * - ``density``
     - list
     - all ``0``
     - Per-sample density in [0, 1].
   * - ``densityPerSquareMeter``, ``maxInstances``
     - number, integer
     - ``1``, ``20000``
     - Instances per square metre at density 1, and the limit.
   * - ``minScale``, ``maxScale``
     - number
     - ``0.8``, ``1.25``
     - Random uniform scale.
   * - ``randomYawDegrees``, ``randomPitchDegrees``, ``randomRollDegrees``,
       ``alignToNormal``
     - number, flag
     - ``180``, ``0``, ``0``, ``false``
     - Random rotation spreads.
   * - ``minHeight``, ``maxHeight``, ``minSlopeDegrees``, ``maxSlopeDegrees``
     - number
     - ``-1e6``, ``1e6``, ``0``, ``90``
     - Where instances are placed.
   * - ``seed``
     - integer
     - ``1``
     - Random seed.
   * - ``castShadows``, ``detectable``, ``automaticMeshLod``,
       ``maxRenderedInstances``, ``renderedInstanceStride``,
       ``doubleBufferedInstanceUploads``, ``sortTransparentInstances``
     - flag, integer
     - ``true``, ``false``, ``true``, ``0``, ``1``, ``true``, ``false``
     - The matching ``InstancedVisuals`` setters.
   * - ``projectedLod``, ``projectedLodMinRadiusPixels``,
       ``projectedLodMaxStride``
     - flag, number, integer
     - ``true``, ``1.5``, ``8``
     - ``InstancedVisuals::setProjectedLodPolicy``.
   * - ``foliageRootHeight``, ``foliageTipHeight``, ``foliageWindStrength``,
       ``foliageStiffness``, ``foliageFlutterWeight``
     - number
     - ``0``, ``1``, ``0.65``, ``0.55``, ``0.6``
     - ``InstancedVisuals::configureFoliageWind``.

Each terrain sample places ``density * densityPerSquareMeter * cell area``
instances on average, rounded up or down by the seeded random numbers, jittered
within the cell around the sample, scaled and turned by the random spreads. A
mesh layer is first turned 90 degrees about x, for meshes modelled with ``+y``
up. Each instance stands on the terrain surface (its mesh bounds rest on the
ground) unless it falls outside the height range; a sample outside the slope
range gets none. rayrai keeps a layer in one batch named
``<terrain path>/Foliage/<id>``, or, for large layers, in 16 m chunks named
``<terrain path>/Foliage/<id>/Chunk_<x>_<y>``.

object record
-------------

``object`` has 25 positional fields followed by keys. Older files have 19
positional fields (``#17`` visible, ``#18`` fixed, meaning a static or dynamic
body) or 17 (a visible dynamic body).

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Field
     - Type
     - Use
   * - ``#0`` path
     - text
     - **RaiSim**: object name; **rayrai**: visual name.
   * - ``#1`` primitive
     - name
     - ``ground`` (``plane``), ``box`` (``cube``), ``sphere``, ``cylinder``,
       ``capsule`` or ``mesh`` (``static_mesh``).
   * - ``#2`` to ``#4`` position
     - vec3 (m)
     - World position. For a ``ground``, the physics plane's height is ``z``.
   * - ``#5`` to ``#8`` rotation
     - quat
     - World orientation. Keep the identity for ``ground``.
   * - ``#9`` to ``#11`` scale
     - vec3
     - ``box``: full side lengths in metres. ``sphere``: radius multiplier
       (x). ``cylinder``, ``capsule``: radius multiplier (x) and height
       multiplier (z). ``mesh``: **RaiSim** uses the mean of the three, as
       Engine 2 does; rayrai scales the drawn mesh per axis. ``ground``:
       the size of a drawn plane.
   * - ``#12`` radius, ``#13`` height
     - number (m)
     - Sphere, cylinder and capsule size before scaling. A capsule's height is
       the distance between its hemisphere centers.
   * - ``#14`` mass
     - number (kg)
     - **RaiSim**: must be positive for a dynamic body; other modes use 1 kg
       for a non-positive mass.
   * - ``#15`` contactMaterial
     - text
     - **RaiSim**: RaiSim material name for contacts.
   * - ``#16`` material
     - text
     - ``material`` id: the appearance color in RaiSim, the material in
       rayrai.
   * - ``#17`` visual only, ``#19`` locked
     - flag
     - Used only when ``#21`` is not a known mode: a visual-only flag makes a
       visual-only object, a locked flag a static body.
   * - ``#18`` visible
     - flag
     - ``false``: **RaiSim** sets the appearance ``hidden`` (the body still
       collides) and rayrai draws nothing for the object.
   * - ``#20`` mesh
     - text (path)
     - The render mesh, unless the ``renderMeshPath`` key is present.
   * - ``#21`` body mode
     - name
     - ``static`` (``fixed``), ``kinematic``, ``dynamic`` (``movable``) or
       ``visual_only`` (``visualonly``, ``visual``).
   * - ``#22`` collidable
     - flag
     - ``false`` makes the object visual-only.
   * - ``#23`` group, ``#24`` mask
     - group
     - **RaiSim**: collision group and mask (see :doc:`Contact`). A ground
       uses only the mask.

An object has a physics body unless it is visual-only or not collidable.
RaiSim gives the body the record's path as its name, its body mode as
``BodyType``, its position and rotation, and the appearance of its material.
A ``ground`` is always a static half space, as in Engine 2. A ``mesh`` body
uses ``collisionMeshPath``, or the render mesh when that is empty:

* a ``.rasset`` descriptor makes static bodies from its collision shapes
  (see `.rasset descriptors`_); a descriptor object must be static;
* any other mesh file becomes one ``World::addMesh`` body with the collision
  mode of ``collisionMode`` (or ``meshCollision``): ``convex_hull``
  (``convexhull``, ``convex``, and ``primitive``, ``none`` or ``default``,
  which Engine 2 builds as a hull), ``original_mesh`` (``original``,
  ``triangle_mesh``, ``trimesh``) or ``convexify`` (``coacd``,
  ``convex_decomposition``), with the ``coacd*`` keys as the
  ``raisim::CoacdOptions`` fields of the same names. A dynamic mesh with
  ``customInertia=true`` uses ``inertiaDiagonal`` and ``centerOfMass``.

rayrai draws the objects the RaiSim world does not show: visible visual-only
objects, and the visual mesh of a visible ``.rasset`` object (its colliders
are hidden). A primitive is drawn at the size RaiSim would build; a mesh file
or descriptor's visual mesh at the object's scale. A mesh keeps its own
materials when ``#16`` is empty, ``engine_default`` or the default material of
its ``assetId``; otherwise the scene material replaces them. With
``materialRemaps`` (``source:target`` items separated by ``;``; ``->`` and
``=`` also separate the pair), each named source material of the mesh is
replaced by the target scene material instead, and the object's material
replaces the rest. The ``visual*`` keys set the drawn visual: ``visualUseMeshColor``,
``visualFlatShading``, ``visualTwoSided``, ``visualDetectable``,
``visualShadowCasting`` (``inherit``, ``on``, ``off``, ``double_sided`` or
``shadows_only``; ``inherit`` follows ``castShadow``), ``visualTransparency``,
``visualRangeBegin``, ``visualRangeEnd``, ``visualRangeBeginMargin``,
``visualRangeEndMargin``, ``visualRangeFade`` (``disabled``, ``self`` or
``dependencies``), ``visualAutomaticMeshLod``, ``visualAutomaticMeshLodBias``,
``visualCustomBounds`` with ``visualCustomBoundsCenter`` and
``visualCustomBoundsRadius``, ``visualTransparentSortOffset``,
``visualTransparentSortUsesBoundsCenter``, ``visualMaterialOverlay`` (a
material id drawn over the visual) and the ``visualPbrEnv*`` keys (an HDR
environment, irradiance and prefiltered map and a BRDF table for the visual).
Their defaults are those of ``raisin::Visuals``.

``id`` (default ``object_<path>``) and ``parentGroupId`` follow `Conventions`_.
The ``raisim*`` tuning keys, ``semanticClass``, ``instanceId``,
``segmentationColor`` and ``receiveShadow`` are **ignored**, as by Engine 2's
bridges.

wire record
-----------

``wire <path> <kind> <bodyA> <bodyB> <length> key=value ...``: a distance
constraint between two bodies, named by object id or path (a terrain counts
too), that both have physics bodies. **RaiSim** adds it with
``World::addSpatialTendon`` under the wire's path, as Engine 2's bridge does,
from ``localPositionA`` on body part ``localIndexA`` to ``localPositionB`` on
part ``localIndexB`` (defaults ``0,0,0`` and ``0``):

* ``stiff`` (``stiff_wire``): the length is the tendon's upper limit;
* ``compliant`` (``soft``, ``compliant_wire``): a spring over ``[0, length]``
  with ``stiffness`` (N/m, default ``1000``);
* ``custom`` (``custom_wire``): a slack tendon of that rest length; its force
  comes from user code.

``damping`` (default ``0``) and ``compliance`` (the limit compliance, default
``0``) apply to every kind. ``visualizationWidth`` (default ``0.01``) is the
drawn width; ``0`` hides the wire. ``enabled=false`` skips the record, and
``id`` is **ignored**.

sensor record
-------------

``sensor <path> <kind> <parent> <x> <y> <z> <qw> <qx> <qy> <qz> key=value ...``:
a camera, depth camera, lidar or IMU mounted on a node. The record is parsed
and creates nothing. RaiSim sensors live on articulated systems, which the
reader does not create, and Engine 2 also creates nothing for a sensor on a
rigid body; the frames and frusta Engine 2 draws for sensors are editor
overlays.

instanced_visual record
-----------------------

``instanced_visual <path> <primitive> key=value ...``: render-only copies of
one mesh or primitive, without physics. A record with ``enabled=false`` is
skipped.

.. list-table::
   :header-rows: 1
   :widths: 30 14 18 38

   * - Field
     - Type
     - Default
     - Use (**rayrai**)
   * - ``#0`` path
     - text
     -
     - Batch name; must be unique in the viewer.
   * - ``#1`` primitive
     - name
     -
     - ``box``, ``sphere``, ``cylinder``, ``capsule`` or ``mesh``.
   * - ``meshPath``
     - text (path)
     - none
     - For ``mesh``: a mesh file, or a ``.rasset`` descriptor whose visual
       mesh is used.
   * - ``size``
     - vec3
     - ``1,1,1``
     - Base size of the batch.
   * - ``colorA``, ``colorB``
     - color
     - ``0.55,0.62,0.72``, ``0.35,0.48,0.66``
     - The two colors that each instance blends.
   * - ``instances``
     - list
     - none
     - Ten numbers per instance: position ``x y z`` (m), quaternion
       ``w x y z`` and scale ``x y z``. Engine 2 separates instances with
       ``;``.
   * - ``colorWeights``
     - list
     - none
     - Per-instance blend weight between ``colorA`` and ``colorB``; instances
       without a weight use 0.
   * - ``castShadows``, ``detectable``, ``automaticMeshLod``,
       ``maxRenderedInstances``, ``renderedInstanceStride``,
       ``doubleBufferedInstanceUploads``, ``sortTransparentInstances``
     - flag, integer
     - ``true``, ``false``, ``false``, ``0``, ``1``, ``false``, ``false``
     - The matching ``InstancedVisuals`` setters
       (``setCastsShadows``, ``setDetectable``,
       ``setAutomaticMeshLodEnabled`` and so on).
   * - ``projectedLod``, ``projectedLodMinRadiusPixels``,
       ``projectedLodMaxStride``
     - flag, number, integer
     - ``false``, ``2``, ``8``
     - ``InstancedVisuals::setProjectedLodPolicy``.
   * - ``shadowFoliageLod``, ``shadowFoliageLodMinRadiusPixels``,
       ``shadowFoliageLodMaxStride``
     - flag, number, integer
     - ``false``, ``2.5``, ``16``
     - ``InstancedVisuals::setShadowFoliageLodPolicy``.
   * - ``foliageWindEnabled``, ``grassBladeRootHeight``,
       ``grassBladeTipHeight``, ``grassWindStrength``, ``grassStiffness``,
       ``grassFlutterWeight``
     - flag, number
     - ``false``, ``0``, ``1``, ``1``, ``0.35``, ``0.15``
     - When enabled, ``InstancedVisuals::configureFoliageWind``
       (root height, tip height, strength, stiffness, flutter weight).
   * - ``grassPatch``
     - flag
     - ``false``
     - **check**: must be ``false``.

``terrainBinding``, ``terrainFoliageLayer``, ``densityPreview`` and the
``scatter*`` and other ``grass*`` keys are **ignored**, as in Engine 2's
viewport. ``id`` and ``parentGroupId`` follow `Conventions`_. See
:doc:`rayrai/Foliage` for the level-of-detail and wind settings.

light record
------------

``light <path> <dx> <dy> <dz> <intensity> key=value ...``. A scene may contain
any number of lights. A light with ``visible=false`` is skipped. **rayrai**
makes the best shadow-casting light the viewer's main light, preferring a
directional light, then an area, a spot and a point light, and the first of
its kind; every other light becomes an additional light (rayrai keeps up to
16). The light's frame turns its local ``-z`` axis onto the direction; the area
axes are given in that frame.

.. list-table::
   :header-rows: 1
   :widths: 30 14 18 38

   * - Field
     - Type
     - Default
     - Use (**rayrai**)
   * - ``#1`` to ``#3`` direction
     - vec3
     -
     - Travel direction, normalized (directional and spot lights).
   * - ``#4`` intensity
     - number
     -
     - diffuse = ``color * intensity``; specular =
       ``specularColor * intensity * specularIntensity``.
   * - ``type``
     - name
     - ``directional``
     - ``directional`` (``sun``), ``point``, ``spot`` or ``area``.
   * - ``position``
     - vec3 (m)
     - ``0,0,3``
     - Position of a point, spot or area light.
   * - ``color``, ``specularColor``, ``specularIntensity``
     - color, color, number
     - white, white, ``1``
     - See ``#4``.
   * - ``ambientColor``
     - color
     - black
     - The light's ambient term. For the main light, black means the
       ``environment`` ambient (``#3`` to ``#5``).
   * - ``attenuationConstant``, ``attenuationLinear``,
       ``attenuationQuadratic``, ``radius``
     - number
     - ``1``, ``0.04``, ``0.012``, ``0``
     - Distance falloff and source size (see ``raisin::AdditionalLight``).
   * - ``coneAngle``, ``innerConeAngle``
     - number (deg)
     - ``30``, ``18``
     - Spot cone half angles.
   * - ``areaSize``, ``areaRight``, ``areaUp``
     - vec3
     - ``1,1,0``, ``1,0,0``, ``0,1,0``
     - Area-light width and height (x and y of ``areaSize``) and axes.
   * - ``projectorTexture``, ``projectorStrength``, ``projectorUvScale``,
       ``projectorUvOffset``
     - text (path), number, vec2, vec2
     - none, ``0``, ``1,1``, ``0,0``
     - A texture projected by the light.
   * - ``negative``, ``temperatureEnabled``, ``colorTemperature``
     - flag, flag, number (K)
     - ``false``, ``false``, ``6500``
     - Subtracting light, and a tint by color temperature.
   * - ``distanceFadeEnabled``, ``distanceFadeBegin``,
       ``distanceFadeShadow``, ``distanceFadeLength``
     - flag, number (m)
     - ``false``, ``40``, ``50``, ``10``
     - Fading by camera distance.
   * - ``shadows``
     - flag
     - ``true``
     - Whether the light casts shadows.
   * - ``shadowResolution``
     - integer (px)
     - ``2048``
     - The main light's shadow-map size; it overrides the render settings'
       shadow resolution for this light.
   * - ``shadowBias``, ``shadowStrength``, ``shadowPcfRadius``
     - number
     - ``0.0008``, ``0.85``, ``1.25``
     - The main light's ``Light::setShadowParams`` (PCF radius in texels).
   * - ``shadowOrthoHalfSize``, ``shadowNear``, ``shadowFar``
     - number (m)
     - ``5``, ``0.1``, ``30``
     - The main light's ``RayraiWindow::setShadowOrtho``.
   * - ``shadowCenter``, ``shadowPosition``, ``shadowUseCustomPosition``
     - vec3, vec3, flag
     - ``0,0,0``, ``0,0,0``, ``false``
     - The main light's shadow-map placement.

``#0`` path and ``id`` name the light (default id ``light_<path>``), and
``parentGroupId`` follows `Conventions`_. ``range`` is **ignored**: the
attenuation coefficients decide the falloff, in Engine 2 as well.

local_fog record
----------------

``local_fog <path> <cx> <cy> <cz> <radius> key=value ...``: a spherical fog
volume (**rayrai**, ``RayraiWindow::addLocalFogVolume``). Keys: ``color``
(default ``0.72,0.80,0.90``), ``density`` (``0.12``), ``edgeFade``
(``0.35``), ``noiseScale`` (``5``) and ``noiseStrength`` (``0.55``).
``enabled=false`` skips the record.

reflection_probe record
-----------------------

``reflection_probe <path> <x> <y> <z> <radius> key=value ...``: a local
reflection probe (**rayrai**, ``RayraiWindow::addReflectionProbe``) with
``strength`` (default ``1``) and box projection (``boxProjection``,
``boxMin``, ``boxMax``; default off, ``-5,-5,-5``, ``5,5,5``).
``enabled=false`` skips the record.

With ``captureOnApply=true``, ``applyRscene`` captures the scene into the
probe, after the objects and lights and before the instanced visuals and
foliage, as Engine 2 does: ``captureCached`` (default ``true``) uses rayrai's
capture cache, ``cubemapSize`` (``256``) the cube face size,
``captureFiltered`` (``true``) also filters the image with the
``irradianceResolution``, ``irradianceSamples``, ``prefilteredResolution``,
``prefilteredMipLevels``, ``prefilteredSamples``, ``brdfLutSize`` and
``brdfLutSamples`` settings, and the ``captureDraw*``, ``captureShadows``,
``captureEnvironmentBackground`` and ``captureEnvironmentExposure`` keys set
what the capture draws. A probe that is not captured on apply holds no image
and has no effect, in Engine 2 as well.

camera record
-------------

``camera <path> <x> <y> <z> <qw> <qx> <qy> <qz> <vfov> <near> <far> <width> <height> <mode> <enabled> key=value ...``.
A camera with ``#14`` set to ``false`` is skipped. RaiSim stores the enabled
cameras in ``Scene::cameras``; rayrai applies the first one to the viewer
camera.

.. list-table::
   :header-rows: 1
   :widths: 30 16 54

   * - Field
     - Type
     - Use
   * - ``#1`` to ``#3`` position, ``#4`` to ``#7`` rotation
     - vec3 (m), quat
     - **rayrai**: camera pose; it looks along its local ``+x`` with local
       ``+z`` up.
   * - ``#8`` vertical field of view
     - number (deg)
     - **rayrai**: clamped to [1, 175].
   * - ``#9`` near, ``#10`` far
     - number (m)
     - **rayrai**: clip distances.
   * - ``#11`` width, ``#12`` height
     - integer (px)
     - **rayrai**: only their ratio is used, as the aspect ratio.
   * - ``#13`` render mode
     - text
     - **ignored** (for example ``rgb``).
   * - ``#14`` enabled
     - flag
     - ``false`` skips the record.
   * - ``horizontalFov``
     - number (deg)
     - **rayrai**: when positive, the horizontal field of view; otherwise
       derived from the vertical one and the aspect ratio. Default ``0``.
   * - ``projection``
     - text
     - **rayrai**: ``perspective`` (default) or ``orthographic``.

``id``, ``parentGroupId`` (see `Conventions`_), the ``preview*`` keys and
``outputPath`` are **ignored**.

environment record
------------------

``environment <r> <g> <b> <ar> <ag> <ab> <fog> <shadows> <grid> <gridSize> [<hdr>] key=value ...``.
RaiSim only stores this record. rayrai starts from the preset defaults of
``rayrai_render`` and applies these fields over them; field names on the right
are ``RenderQualitySettings`` members (see :doc:`rayrai/RenderQuality`).
Without the record, every field takes its default.

.. list-table::
   :header-rows: 1
   :widths: 30 16 18 36

   * - Field
     - Type
     - Default
     - Use (**rayrai**)
   * - ``#0`` to ``#2`` background
     - numbers
     - ``0.38,0.38,0.38``
     - Background color (linear RGB), also ``backgroundColorRgb255``.
   * - ``#3`` to ``#5`` ambient
     - numbers
     - ``0.15,0.16,0.20``
     - ``mainLightAmbient``, and the main light's ambient when its
       ``ambientColor`` is black.
   * - ``#6`` fog density
     - number (1/m)
     - ``0``
     - ``fogDensity``. Outside the sky background, a positive value also
       turns on ``fogColorOverrideEnabled``.
   * - ``#7`` shadows
     - flag
     - ``true``
     - ``shadowsEnabled``.
   * - ``#8``, ``#9``
     - flag, number
     -
     - **ignored** (Engine 2 editor grid).
   * - ``#10`` environment map
     - text (path)
     - none
     - Equirectangular HDR image for ``backgroundMode=hdr``.
   * - ``backgroundMode``
     - name
     - ``plain_color``
     - ``plain_color`` (``plain``, ``color``, ``clear_color``, ``solid``),
       ``hdr`` (``environment``, ``environment_map``, ``cubemap``; the image in
       ``#10``) or ``rayrai_sky`` (``rayraisky``, ``rayrai``, ``sky``,
       ``procedural_sky``; the sky generated from the ``weather`` record).
   * - ``fogColor``
     - color
     - ``0.55,0.60,0.68``
     - ``fogColor``.
   * - ``exposure``, ``gamma``
     - number
     - ``1``, ``2.2``
     - ``pbrExposure``, ``gamma``.
   * - ``bloomIntensity``, ``bloomThreshold``, ``bloomRadius``
     - number
     - ``0``, ``0.82``, ``4``
     - ``bloomStrength`` (a positive value turns on ``bloomEnabled``),
       ``bloomThreshold``, ``bloomRadius``.
   * - ``shadowMapSize``, ``shadowedLightBudget``
     - integer
     - ``2048``, ``1``
     - ``shadowResolution``, ``shadowedLightBudget``.
   * - ``colorMode``
     - name
     - ``unreal_preview``
     - ``colorMode``: ``fast_linear`` (or ``linear``), ``aces_approx``
       (``aces``), ``filmic_approx`` (``filmic``), ``agx_approx`` (``agx``),
       or ``unreal_preview_approx`` (``unreal``, ``unreal_preview``).
   * - ``pbrEnvironmentLightingTint``
     - color
     - ``1,1,1``
     - ``pbrEnvironmentLightingTint``.
   * - ``pbrEnvironmentIntensity``
     - number
     - ``1``
     - ``pbrEnvironmentIntensity``, and the intensity of the HDR or sky
       background (at least 0.05 for the sky).
   * - ``fxaa``, ``ssao``
     - flag
     - ``true``, ``true``
     - ``fxaaEnabled``, ``screenSpaceAoEnabled``.
   * - ``heightFogEnabled``, ``heightFogDensity``, ``heightFogBaseHeight``,
       ``heightFogFalloff``
     - flag, number
     - ``false``, ``0``, ``0``, ``0.35``
     - Height fog settings of the same names.

weather record
--------------

``weather [<enabled>] key=value ...``. rayrai reads this record when the
``environment`` background is ``rayrai_sky``; without the record, the sky
uses the defaults below. See :doc:`rayrai/Weather`.

``#0`` is ``WeatherSettings::enabled`` (default ``true``). Each key sets the
``WeatherSettings`` member of the same name, except that ``timeOfDay`` sets
``timeOfDayHours``. The defaults are Engine 2's: ``preset`` ``clear`` (or
``hazy``, ``overcast``, ``fog``, ``rain``, ``heavy_rain``, ``snow``,
``storm``, ``night_clear``, ``night_rain`` or ``custom``), ``quality``
``high`` (or ``low``, ``medium`` or ``ultra``), ``seed`` ``1``, ``timeOfDay``
``13``, ``latitude`` ``37``, ``longitude`` ``127``, ``year`` ``2026``,
``month`` ``5``, ``day`` ``8``, ``windDirection`` ``1,0.25,0``, ``windSpeed``
``1.5``, ``transitionSeconds`` ``0``, ``affectSensors`` ``false``,
``cloudCoverage`` ``0.05``, ``cloudDensity`` ``0.05``, ``precipitationRate``
``0``, ``rainOcclusionStrength`` ``0``, ``fogDensity`` ``0``,
``visibilityMeters`` ``10000``, ``fogColor`` ``0.72,0.80,0.90``,
``fogAnisotropy`` ``0``, ``humidity`` ``0.35``, ``wetness`` ``0``,
``wetnessAccumulationEnabled`` ``true``, ``wetnessAccumulationRate``
``0.35``, ``wetnessDryingRate`` ``0.1``, ``snowCoverage`` ``0``,
``lightningRate`` ``0``, ``lightningLocalPointLightEnabled`` ``true``,
``lightningLocalPointDistance`` ``18``, ``lightningLocalPointRadius`` ``28``,
``lightningLocalPointIntensity`` ``2.2``, ``thunderSpeedOfSound`` ``343``,
``airTurbidity`` ``2``, ``groundAlbedo`` ``0.35``, ``useExplicitSunAngles``
``false``, ``sunAzimuthDegrees`` ``180``, ``sunElevationDegrees`` ``42``,
``sunDiskSize`` ``0.018``, ``moonDiskSize`` ``0.014``,
``cloudAltitudeMeters`` ``850``, ``cloudThicknessMeters`` ``180``,
``cloudShadowStrength`` ``0.04``, ``cloudScale`` ``0.18``,
``cloudAnimationSpeed`` ``0``, ``lensDropletsEnabled`` ``false`` and
``lensDropletStrength`` ``1``.

The file stores no UTC offset, so rayrai's default (an explicit offset of
9 hours) applies to ``timeOfDay``. The wind also drives foliage: with the sky
background, a non-zero ``windDirection`` sets the render settings'
``foliageWindDirection`` (its x and y), and a positive ``windSpeed`` turns on
``foliageWindEnabled`` with that speed.

rayrai_render record
--------------------

``rayrai_render key=value ...``. RaiSim only stores this record. ``preset``
(default ``ultra``; ``fast`` or ``low``, ``balanced`` or ``medium``,
``high``, ``ultra`` or ``custom``) selects
``RayraiWindow::defaultRenderQualitySettings(preset)`` as the starting point.
When ``custom`` (default ``false``) is ``true`` or the preset is ``custom``,
the keys below are applied after the ``environment`` record; otherwise they
are ignored and the preset's values apply, including
``addViewerFillLights = true``, which adds two viewer fill lights.

.. list-table::
   :header-rows: 1
   :widths: 40 18 42

   * - Key
     - Default
     - Use (**rayrai**)
   * - ``viewerMsaaSamples``
     - ``0``
     - A positive value is rounded down to 1, 2, 4 or 8; ``0`` keeps the
       preset's value.
   * - ``textureAnisotropy``
     - ``0``
     - A positive value is clamped to [1, 16]; ``0`` keeps the preset's value.
   * - ``shadowResolution``, ``shadowBias``, ``shadowStrength``,
       ``shadowPcfRadius``
     - ``2048``, ``0.0008``, ``0.6``, ``1.25``
     - Shadow settings; ``shadowResolution`` replaces ``environment``'s
       ``shadowMapSize``.
   * - ``directionalShadowCascadeCount``, ``directionalShadowCascadeLambda``,
       ``directionalShadowCascadeMaxDistance``
     - ``1``, ``0.7``, ``0``
     - Cascades, clamped to [1, 4], [0, 1] and at least 0 m.
   * - ``highFidelityPbr``, ``pbrToneMapping``
     - ``false``, ``false``
     - Settings of the same names.
   * - ``colorMode``, ``pbrExposure``
     - ``unreal_preview``, ``1``
     - Replace ``environment``'s ``colorMode`` and ``exposure``.
   * - ``pbrEnvironmentIntensity``, ``addViewerFillLights``,
       ``reflectiveGround``
     - none
     - Each replaces the value set so far only when present.
   * - ``autoExposure``, ``autoExposureKey``, ``autoExposureSpeed``
     - ``false``, ``0.18``, ``0.05``
     - ``autoExposureEnabled``, ``autoExposureKey``, ``autoExposureSpeed``.
   * - ``volumetricFog``, ``volumetricFogDensity``, ``volumetricLightStrength``
     - ``false``, ``0``, ``0``
     - ``volumetricFogEnabled``, ``volumetricFogDensity``,
       ``volumetricLightStrength``.
   * - ``contactShadows``, ``contactShadowsLength``, ``contactShadowsStrength``
     - ``false``, ``0.12``, ``0.7``
     - ``contactShadowsEnabled``, ``contactShadowsLength``,
       ``contactShadowsStrength``.
   * - ``screenSpaceReflections``, ``screenSpaceReflectionStrength``
     - ``false``, ``0``
     - ``ssrEnabled``, ``ssrStrength``.
   * - ``motionBlur``, ``motionBlurDirection`` (vec2), ``motionBlurStrength``
     - ``false``, ``0,0``, ``0``
     - ``motionBlurEnabled``, ``motionBlurDirection``, ``motionBlurStrength``.
   * - ``depthOfField``, ``depthOfFieldFocusDistance``,
       ``depthOfFieldAperture``
     - ``false``, ``5``, ``0``
     - ``depthOfFieldEnabled``, ``depthOfFieldFocusDistance``,
       ``depthOfFieldMaxRadius``.

``diagnostics`` is **ignored**.

Rejected content
================

The readers refuse content they cannot reproduce instead of dropping it.
Errors found while parsing name the file, the line and the record tag; the
others name the object or file involved.

While parsing (``rscene::load``):

* a missing or wrong header, or an empty file;
* a version-1 ``asset_root`` other than ``.`` or an absolute path (see
  `Versioning`_);
* a positional field after a ``key=value`` field, a repeated key, a missing
  positional field, or a value of the wrong type, including a name that is
  not listed for its field;
* any record tag not listed in `Records`_, for example
  ``articulated_resource``, ``articulated``, ``articulated_ik``,
  ``compound``, ``compound_child``, ``deformable``, ``granular``,
  ``projected_decal``, ``irradiance_volume`` and ``point_cloud``;
* a node whose ``parentGroupId`` names another node while transforms are
  inherited (see `Conventions`_);
* a terrain that is not sampled, is hidden or not collidable, has fewer than
  two samples per axis, a non-positive size, or a list with the wrong number
  of values, and a ``terrain_splat_layer`` or ``terrain_foliage_layer`` whose
  terrain does not come before it;
* a dynamic object without positive mass;
* an enabled instanced visual that is a grass patch or a ``ground``, or whose
  instance list is not a multiple of ten numbers;
* an enabled wire with a negative length or body index.

While adding bodies (``rscene::addToWorld``):

* a ``mesh`` object whose mesh file is missing;
* a ``.rasset`` object that is not static or has ``customInertia=true``, a
  descriptor that cannot be read or is invalid, or a rotation that is not a
  unit quaternion;
* a wire whose body has no physics body, or whose tendon RaiSim rejects.

While applying to rayrai (``rsceneRenderSettings`` and ``applyRscene``):

* an unknown preset, color mode, background mode, weather preset or weather
  quality, or a malformed key in these records;
* an HDR background without an image, or an image that cannot be loaded, or a
  sky that cannot be generated;
* a missing texture (terrain layer, material blend texture, light projector
  or visual environment map) or a missing mesh (visual-only object, ``.rasset``
  visual, instanced visual or foliage layer);
* an unknown material mode, ``visualShadowCasting`` or ``visualRangeFade``
  name, or a material that ``visualMaterialOverlay`` or ``materialRemaps``
  names but the scene does not define;
* a reflection probe whose capture fails;
* a batch or visual name already in use (a fatal error).

``rscene::load`` and ``rsceneRenderSettings`` throw ``std::runtime_error``.
``rscene::addToWorld`` throws ``std::runtime_error``, or
``std::invalid_argument`` for an invalid ``.rasset`` instance, so catch
``std::exception``. ``World(path)`` reports any of them as a RaiSim fatal
error.

Two RaiSim Engine 2 test inputs in the repository are deliberately not
loadable: ``raisim_engine2/tests/data/missing_mesh_scene.rscene`` names a mesh
that does not exist, and
``raisim_engine2/tests/data/full_authoring_scene.rscene``, a round-trip
fixture of Engine 2's own loader, uses every node kind, including the
unsupported ones, and names files that do not exist.

Example file
============

A crate and a ball dropped into a 4 m × 4 m height-map bowl, lit by one sun
and seen from a camera at ``(-4, 0, 1.5)`` looking along ``+x``. The ``solver`` record stores the
``raisim::World`` defaults (see `Solver defaults`_). Records must stay on one
line, so the long ones run past the right edge of the box:

.. code-block:: text

   raisim_engine_scene 2
   # A crate and a ball dropped into a small height-map bowl, lit by one sun.
   time_step 0.001
   gravity 0 0 -9.81
   solver 150 1e-08 1.5 accurate erp2=0.002 defaultFriction=0.8 defaultRestitution=0 defaultRestitutionThreshold=0.01 defaultStaticFriction=0.8 defaultStaticFrictionVelocityThreshold=1 defaultRollingFriction=0 defaultSpinningFriction=0 contactAlphaInit=1 contactAlphaMin=1 contactAlphaDecay=1 contactMaxIterations=80 contactThreshold=1e-07 contactGjkMaxIterations=32 contactGjkTolerance=1e-06 contactEpaMaxIterations=64 contactEpaTolerance=1e-04 maxContactsPerPair=8 sweptCcdEnabled=false sweptCcdMinSpeed=0 sweptCcdSpeculativeMargin=1e-04 sleepingEnabled=true sleepingLinearVelocityThreshold=0.002 sleepingAngularVelocityThreshold=0.01 sleepingQuietSteps=5 broadphaseType=sap3_axis broadphaseWorldMin=-100,-100,-100 broadphaseWorldMax=100,100,100 broadphaseCellSize=1,1,1 broadphaseUseWorldBounds=false broadphasePadding=0.5 broadphaseMaxCellsPerAxis=128 broadphaseMaxCellsPerObject=64 worldTime=0 fixedContactSolverIterationOrder=true
   asset_root .
   environment 0.6 0.7 0.85 0.3 0.3 0.3 0 true true 10 - fogColor=0.6,0.7,0.85 exposure=1 gamma=2.2 bloomIntensity=0 bloomThreshold=0.8 bloomRadius=4 shadowMapSize=2048 shadowedLightBudget=2 backgroundMode=plain_color colorMode=aces_approx pbrEnvironmentLightingTint=1,1,1 pbrEnvironmentIntensity=1 fxaa=true ssao=false heightFogEnabled=false heightFogDensity=0 heightFogBaseHeight=0 heightFogFalloff=0.35
   rayrai_render preset=balanced custom=false
   material crate 0.8 0.5 0.2 1 0 0.6 id=crate
   terrain_texture 0 grass Grass color=0.3,0.45,0.2,1 albedo=- normal=- uvScale=1 normalDepth=1 roughness=0.9
   terrain_region /World/Terrain 3 3 4 4 0 0 0 parentGroupId=- material=- contactMaterial=default baseTexture=0 visible=true collidable=true sourceKind=samples texturePrimary=- textureSecondary=- textureBlend=- wetness=- holes=- vertexColors=- heights=0.3,0.15,0.3,0.15,0,0.15,0.3,0.15,0.3
   object /World/Crate box 0 0 1 1 0 0 0 0.5 0.5 0.5 0.5 1 2 default crate false true false - dynamic true 1 18446744073709551615 parentGroupId=- renderMeshPath=- collisionMeshPath=- materialRemaps=-
   object /World/Ball sphere 1.2 1.2 1 1 0 0 0 1 1 1 0.15 1 0.5 default - false true false - dynamic true 1 18446744073709551615 parentGroupId=- renderMeshPath=- collisionMeshPath=- materialRemaps=-
   light /World/Sun -0.4 0.3 -0.85 3 parentGroupId=- type=directional visible=true shadows=true color=1,0.95,0.85 ambientColor=0,0,0 specularColor=1,1,1 specularIntensity=1 shadowResolution=2048 shadowBias=0.0008 shadowStrength=1 shadowPcfRadius=1 shadowOrthoHalfSize=10 shadowNear=0.1 shadowFar=50
   camera /World/Camera -4 0 1.5 0.991445 0 0.130526 0 55 0.05 100 1280 800 rgb true parentGroupId=- projection=perspective horizontalFov=0

``World(path)`` on this file creates three objects: the height map
``/World/Terrain`` and the dynamic bodies ``/World/Crate`` (a 0.5 m, 2 kg box
colored by the ``crate`` material) and ``/World/Ball`` (a 0.15 m, 0.5 kg sphere
with the default color). The terrain's empty ``material`` takes the default
id ``engine_default``, which this file does not define, so the terrain has
the default color as well. Because ``contactMaxIterations`` and
``contactThreshold`` hold their Engine 2 defaults, the contact solver uses the
``#0`` and ``#1`` values, 150 iterations and ``1e-8``. The camera quaternion
is a 15° rotation about ``+y``, which tilts its ``+x`` view direction down.
With ``custom=false``, rayrai uses the ``balanced`` preset's settings over the
``environment`` record, including its two viewer fill lights; the C++ snippet
in `Physics and rayrai rendering`_ turns them off.

API reference
=============

.. doxygenfunction:: raisim::rscene::load

.. doxygenfunction:: raisim::rscene::applyPhysics

.. doxygenfunction:: raisim::rscene::addToWorld

.. doxygenfunction:: raisim::rscene::referencedFiles

.. doxygenfunction:: raisim::rscene::forEachFileReference

.. doxygenstruct:: raisim::rscene::Scene
   :members:

.. doxygenstruct:: raisim::rscene::WorldReport
   :members:

.. doxygenfunction:: raisin::rsceneRenderSettings

.. doxygenfunction:: raisin::applyRscene(const raisim::rscene::Scene&, RayraiWindow&)

.. doxygenfunction:: raisin::applyRscene(const raisim::rscene::Scene&, RayraiWindow&, const RsceneRenderSettings&)

.. doxygenstruct:: raisin::RsceneRenderSettings
   :members:

.. doxygenstruct:: raisin::RsceneVisuals
   :members:

The parsed records (``raisim::rscene::Physics``, ``ContactMaterial``,
``Material``, ``Terrain``, ``Object``, ``Wire``, ``Light``, ``Camera``,
``LocalFog``, ``ReflectionProbe``, ``InstancedVisual`` and the others) are
declared in ``raisim/Rscene.hpp``.

Other scene inputs
==================

* RaiSim world XML for hand-authored or template-driven worlds
  (:doc:`WorldConfigurationFile`);
* MJCF for supported MuJoCo model imports (:doc:`WorldSystem`);
* OpenUSD for USD Physics articulations (:doc:`OpenUSD`); or
* the C++ or RaisimPy API to construct a world programmatically.
