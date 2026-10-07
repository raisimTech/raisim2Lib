########################
World Configuration File
########################

A RaiSim world configuration file is an XML file with a ``<raisim>`` root
element. Load it with ``raisim::World world("path/to/world.xml")``; see
:doc:`WorldSystem`. Example files are in
`rsc/xmlScripts <https://github.com/raisimTech/raisim2Lib/tree/main/rsc/xmlScripts>`__.

The following sections describe the elements the loader reads.
Elements and attributes marked (optional) may be omitted.
Unmarked elements and attributes are required when their parent element is present.
Elements marked (multiple) may appear more than once under the same parent.

Vector-valued attributes (``Vec<3>``, ``Vec<4>``, ``VecDyn``) are lists of
numbers separated by whitespace or by commas, e.g. ``pos="0 0 1"`` or
``pos="0, 0, 1"``. The token ``[THIS_DIR]`` anywhere in the file is replaced by
the directory that contains the file, so resource paths can be written relative
to the file, e.g. ``urdf_path="[THIS_DIR]/../anymal/urdf/anymal.urdf"``.
Relative paths without ``[THIS_DIR]`` are resolved against the process working
directory.

1. ``raisim``: the root element.

   1. <attribute> (optional) ``version`` [string]: the RaiSim version that wrote
      the file. The loader does not check it; ``exportToXml`` writes ``2.0.0``.
   2. <child> (optional) ``material``: material pair properties. See
      :doc:`MaterialSystem` for their meaning.

      1. <child> (optional) ``default``: the properties used for material pairs
         without a ``pair_prop`` entry. If omitted, the built-in defaults are used.

         1. <attribute> ``friction`` [float]
         2. <attribute> ``restitution`` [float]
         3. <attribute> ``restitution_threshold`` [float]
         4. <attribute> (optional, default = ``friction``) ``static_friction`` [float]
         5. <attribute> (optional, default = 1) ``static_friction_velocity_threshold`` [float]
         6. <attribute> (optional, default = 0) ``rolling_friction`` [float]
         7. <attribute> (optional, default = 0) ``spinning_friction`` [float]

      2. <child> (optional, multiple) ``pair_prop``

         1. <attribute> ``name1`` [string]
         2. <attribute> ``name2`` [string]
         3. <attribute> ``friction`` [float]
         4. <attribute> ``restitution`` [float]
         5. <attribute> ``restitution_threshold`` [float]
         6. <attribute> (optional, default = ``friction``) ``static_friction`` [float]
         7. <attribute> (required when ``static_friction`` is present, otherwise
            1e-3) ``static_friction_velocity_threshold`` [float]
         8. <attribute> (optional, default = 0) ``rolling_friction`` [float]
         9. <attribute> (optional, default = 0) ``spinning_friction`` [float]

   3. <child> (optional) ``gravity``

      1. <attribute> ``value`` [Vec<3>]: default = ``0, 0, -9.81``

   4. <child> (optional) ``timeStep``. Only this camelCase tag name is read.

      1. <attribute> ``value`` [float]: default = 0.005

   5. <child> (optional) ``erp``: (advanced) the spring and damping terms of the
      contact error dynamics; see ``World::setERP``.

      1. <attribute> ``erp`` [float]: spring term
      2. <attribute> ``erp2`` [float]: damping term

   6. <child> (optional) ``simulation``: solver, contact-detection and sleeping
      settings. See `Simulation settings`_.
   7. <child> (optional) ``object_class``: reusable object templates. See
      `object_class`_.
   8. <child> (optional) ``objects``: the objects of the world. See
      `Object XML Description`_.
   9. <child> (optional, multiple) ``tendon`` and ``tendon_coupling``: see the
      native world XML section of :doc:`tendons/Reference`.
   10. <child> (optional, multiple) ``wire``: legacy length constraint. The
       loader converts each wire into a two-site tendon; exports write
       ``tendon`` elements instead. See :doc:`Constraints` and the
       `wire example <https://github.com/raisimTech/raisim2Lib/blob/main/rsc/xmlScripts/wire/newtonsCradle.xml>`__.

       1. <attribute> ``name`` [string]: wire (tendon) name.
       2. <attribute> ``type`` [string]: ``stiff``, ``compliant``, or ``custom``.
       3. <attribute> ``length`` [float]: nominal length (m), at least 0.
       4. <attribute> (optional) ``stretch_type`` [string]: ``stretch_only``,
          ``compression_only``, or ``both``. Default: ``stretch_only`` for
          stiff and compliant wires, ``compression_only`` for custom wires.
       5. <attribute> (required for ``compliant``) ``stiffness`` [float]
       6. <attribute> (optional, ``custom`` only, default = 0) ``tension`` [float]
       7. <attribute> (optional, default = 0.02) ``width`` [float] and
          (optional, default = ``1, 1, 1, 1``) ``color`` [Vec<4>]: appearance.
       8. <child> ``object1`` and ``object2``: the two attachment points. Each
          has a ``name`` [string] attribute naming the attached object and one
          of the following:

          * ``local_index`` [int] and ``pos`` [Vec<3>]: the local body index
            (0 for a single-body object) and the attachment point in that
            body's frame; or
          * ``frame`` [string] (articulated systems only): the name of the
            frame to attach to.

   11. <child> (optional, multiple) ``include`` and ``params``: see
       `Configuration Template`_.

Object XML Description
----------------------------

Each child element of ``objects`` creates one object. The element name selects
the object type (``sphere``, ``box``, ``capsule``, ``cylinder``, ``compound``,
``mesh``, ``ground``, ``heightmap``, ``articulatedSystem``, ``deformable``,
``granular``) or names an ``object_class`` defined in the same file.

Every object requires a ``name`` attribute, and an object with
``exist="false"`` is skipped (see `Configuration Template`_).

The XML reader accepts both the snake_case names documented below and the camelCase aliases used by older examples for common attributes, such as ``collisionGroup``, ``collisionMask``, ``xSample``, ``ySample``, ``xSize``, ``ySize``, ``centerX``, ``centerY``, ``resDir``, ``urdfPath``, ``fileName``, ``linVel``, and ``angVel``.

``collision_group`` and ``collision_mask`` accept two notations:

* a plain decimal integer is the raw bitmask, so ``collision_group="1"`` is
  bit 0 and ``"-1"`` sets every bit;
* ``collision[i|j|...]`` sets bits ``i``, ``j``, ...; for example,
  ``collision[1|4|6]`` is ``(1 << 1) | (1 << 4) | (1 << 6)``, and
  ``collision[-1]`` sets every bit.

Two objects can collide only if each object's group shares a bit with the other
object's mask. See :doc:`Contact` for details.

Single bodies
^^^^^^^^^^^^^
`Example <https://github.com/raisimTech/raisim2Lib/blob/main/rsc/xmlScripts/objects/SingleBodies.xml>`__

``sphere``, ``box``, ``capsule``, ``cylinder``, ``compound``, and ``mesh``
share the following attributes and children. The type-specific parts follow in
the next sections.

**attributes**: ``name`` [string], ``mass`` [float],
(optional, default=1) ``collision_group``,
(optional, default=-1) ``collision_mask``,
(optional, default ``default``) ``material`` [string] (ignored by ``compound``,
whose children carry their own materials),
(optional) ``appearance`` [string],
(optional, default ``dynamic``) ``body_type`` [string]: ``dynamic``, ``kinematic``, or ``static``,
(optional for primitives, default ``0, 0, 0``) ``com`` [Vec<3>]: the center of
mass in the body frame

1. <child> ``inertia``: <attribute> ``xx``, ``xy``, ``xz``, ``yy``, ``yz``,
   ``zz`` [float], the inertia about the center of mass in the body frame.
   Optional for ``sphere``, ``box``, ``capsule``, and ``cylinder``, which
   otherwise derive it from the geometry assuming uniform density. Required for
   ``mesh`` and ``compound``: without it their inertia is zero, which is
   rejected as not positive definite.

2. <child> ``state``: <attribute> ``pos`` [Vec<3>], (optional, default ``1, 0, 0, 0``)
   ``quat`` [Vec<4>], (optional, default ``0, 0, 0``) ``lin_vel`` [Vec<3>],
   (optional, default ``0, 0, 0``) ``ang_vel`` [Vec<3>]

sphere
^^^^^^^^^^^^^

1. <child> ``dim``: <attribute> ``radius`` [float]

capsule and cylinder
^^^^^^^^^^^^^^^^^^^^^^^

1. <child> ``dim``: <attribute> ``radius`` [float], ``height`` [float]

box
^^^^^^^^^^^^^^^^^^^^^^^

1. <child> ``dim``: <attribute> ``x`` [float], ``y`` [float], ``z`` [float]:
   the full side lengths along the body axes

compound
^^^^^^^^^^^^^^^^^^^^^^^

A single rigid body made of several primitive collision shapes. ``com`` and the
``inertia`` child are required.

1. <child> ``children``: one element per shape, in any order. Each element is
   ``sphere``, ``box``, ``capsule``, ``cylinder``, or the name of a compound
   ``object_class`` (whose shapes are inserted at this pose). Other attributes
   of the shape elements, such as ``name`` or ``mass``, are ignored.

   ::

     <attribute> (optional, default=default) material [string]
     <attribute> (optional) appearance [string]
     <child> dim               (not used for an object_class child)
       sphere:             <attribute> radius [float]
       capsule, cylinder:  <attribute> radius [float], height [float]
       box:                <attribute> x [float], y [float], z [float]
     <child> state             (pose of the shape in the compound body frame)
       <attribute> pos [Vec<3>]
       <attribute> (optional, default=1,0,0,0) quat [Vec<4>]

mesh
^^^^^^^^^^^^^^^^^^^^^^^

``com`` and the ``inertia`` child are required.

**attributes**:
``file_name`` [string]: the mesh file,
``scale`` [Vec<3>]: mesh scale; only the first component is used (uniform scale),
(optional, default=0) ``collision_mode`` [int]: ``0`` keeps the triangle mesh,
``1`` uses its convex hull, ``2`` uses a CoACD convex decomposition

1. <child> (optional) ``coacd``: CoACD parameters used with ``collision_mode="2"``.
   The attributes ``threshold``, ``max_hulls``, ``preprocess``,
   ``prep_resolution``, ``sample_resolution``, ``mcts_nodes``,
   ``mcts_iterations``, ``mcts_depth``, ``pca``, ``merge``, ``decimate``,
   ``max_vertices``, ``extrude``, ``extrude_margin``, ``approximation``,
   ``seed``, and ``real_metric`` map to the fields of ``raisim::CoacdOptions``;
   omitted attributes keep their defaults.

ground
^^^^^^^^^^^^^^^^^^^^^^^
`Example <https://github.com/raisimTech/raisim2Lib/blob/main/rsc/xmlScripts/material/material.xml>`__

An infinite static plane. Its collision group is fixed, so only the mask can be set.

**attributes**: ``name`` [string], (optional, default=0) ``height`` [float],
(optional, default ``default``) ``material`` [string], (optional, default=-1)
``collision_mask``, (optional) ``appearance`` [string]

heightmap
^^^^^^^^^^^^^^^^^^^^^^^
`Examples <https://github.com/raisimTech/raisim2Lib/tree/main/rsc/xmlScripts/heightMaps>`__

**attributes (all options)**: ``name`` [string],
(optional, default ``default``) ``material`` [string],
(optional) ``appearance`` [string],
(optional, default=1) ``collision_group``, (optional, default=-1) ``collision_mask``,
(optional, default=0) ``center_x``, ``center_y``, ``center_z`` [float],
(optional, default=identity) ``quat`` [w, x, y, z].
``center_x`` and ``center_y`` place the grid center in the height map's frame, ``center_z``
sets the position of that frame along z, and ``quat`` its orientation about the frame origin
(see :doc:`HeightMap`).

The height data comes from exactly one of the following options. If several
are present, the first one in this list is used.

1. Raw heights: ``x_sample`` [int], ``y_sample`` [int], (optional, default=10)
   ``x_size`` and ``y_size`` [float], and ``height`` [list of float] with
   ``x_sample * y_sample`` values.

2. Terrain generator: ``x_sample`` [int], ``y_sample`` [int], (optional,
   default=10) ``x_size`` and ``y_size`` [float], and a child
   ``terrain_properties`` (alias ``terrainProperties``) with the attributes
   ``frequency`` [float], ``z_scale`` [float], ``fractal_octaves`` [int],
   ``fractal_lacunarity`` [float], ``fractal_gain`` [float], ``step_size``
   [float], ``seed`` [int], and (optional, default=0) ``height_offset`` [float].
   This option currently ignores ``collision_group`` and ``collision_mask``.

3. PNG image: ``png`` [string], (optional, default=10) ``x_size`` and ``y_size``
   [float], ``z_scale`` (alias ``height_scale``) [float], and ``z_offset``
   (alias ``height_offset``) [float]. The sample counts come from the image
   resolution.

4. Text file: ``text`` [string], the path of a height-map text file, which also
   stores the sample counts and sizes.

articulatedSystem / articulated_system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`Example <https://github.com/raisimTech/raisim2Lib/blob/main/rsc/xmlScripts/heightMaps/heightMapUsingPng.xml>`__

Both tag names are accepted by the XML reader. ``articulatedSystem`` is the canonical exported form.

**attributes**: ``name`` [string], ``urdf_path`` [string],
(optional, default=URDF directory) ``res_dir`` [string],
(optional, default=1) ``collision_group``, (optional, default=-1) ``collision_mask``,
(optional) ``modules`` [space-separated string list],
(optional) ``joint_order`` [space-separated joint names]: redefines the joint
order (a child joint cannot precede its parent),
(optional, default=0) ``collide_parent`` [int]: 1 enables collisions between a
link and its parent,
(optional, default=0) ``convexify_meshes`` [int]: 1 treats the URDF collision
meshes as convex shapes

Use an absolute ``urdf_path`` or one that starts with ``[THIS_DIR]``.

1. <child> ``state``: <attribute> ``qpos`` [VecDyn], the generalized coordinate,
   (optional, default=zeros) ``qvel`` [VecDyn], the generalized velocity

object_class
^^^^^^^^^^^^^^^^^^^^^

``object_class`` defines reusable object templates under the top-level ``raisim`` node. Each child node name becomes a class name that can be used as an object tag under ``objects``.

Common class attributes are ``type`` [string] (an object type name such as
``sphere``), (required for ``sphere``, ``box``, ``capsule``, ``cylinder``,
``mesh``, and ``compound``) ``mass`` [float], (optional) ``material``
[string], (optional) ``com`` [Vec<3>], and the optional child ``inertia`` with
attributes ``xx``, ``xy``, ``xz``, ``yy``, ``yz``, and ``zz``.

Primitive classes use the child ``dimension`` with the same shape parameters as
``dim``. Mesh classes use the child ``file`` with the attribute ``name`` and the
child ``scale`` with the attributes ``x``, ``y``, and ``z`` (only ``x`` is used).
Articulated-system classes use the attributes ``urdf_path`` and optional
``res_dir``. Compound classes use the child ``children``; each child element is
named after a primitive type or another compound class and uses the attributes
``object_param`` [VecDyn] (the shape parameters: radius; radius and height; or
x, y and z), ``pos`` [Vec<3>], optional ``quat`` [Vec<4>], optional
``material``, and optional ``appearance``.

An instance of a class under ``objects`` takes ``name``, the collision
attributes, ``body_type``, ``state``, and ``exist`` like the corresponding
built-in type; its shape, mass, and material come from the class.

Round-trip elements
^^^^^^^^^^^^^^^^^^^

``World::exportToXml`` also writes the following elements so that an exported
world reloads in the same state. They are rarely written by hand; generate them
with ``exportToXml``.

* ``deformable`` and ``granular`` objects, with their particles, constraints,
  and particle materials.
* ``tuning`` child of an object: damping and gyroscopic mode of single bodies,
  the pose of ``ground`` and ``heightmap`` objects, and the integration
  scheme, control mode, PD gains, targets, and feedforward force of
  articulated systems (``kp``, ``kd``, ``qref``, ``uref``, and ``feedforward``
  are then required).
* ``sensors`` and ``actuators`` children of an articulated system. See
  :doc:`Sensors` and :doc:`Actuators`.

Simulation settings
-------------------

The optional ``simulation`` element under ``raisim`` sets solver,
contact-detection, and sleeping parameters. Every attribute is optional;
omitted attributes keep the current values. Boolean attributes are written as
``0`` or ``1``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Attributes
     - Setting
   * - ``world_time``
     - ``World::setWorldTime``
   * - ``solver_order``
     - ``World::setContactSolverIterationOrder``
   * - ``solver_iterations``, ``solver_threshold``
     - maximum contact-solver iterations and termination threshold
       (``World::setContactSolverParam``)
   * - ``alpha_init``, ``alpha_low``, ``alpha_decay``
     - accepted for compatibility; the solver ignores them
   * - ``gjk_iterations``, ``gjk_tolerance``, ``epa_iterations``,
       ``epa_tolerance``, ``max_contacts``
     - ``gjkMaxIterations``, ``gjkTolerance``, ``epaMaxIterations``,
       ``epaTolerance``, ``maxContactsPerPair`` of ``contact::ContactSettings``
   * - ``swept_ccd``, ``ccd_min_speed``, ``ccd_margin``
     - ``sweptCcdEnabled``, ``sweptCcdMinSpeed``, ``sweptCcdSpeculativeMargin``
   * - ``broadphase``
     - broadphase type: ``0`` = ``None``, ``1`` = ``Sap3Axis``, ``2`` =
       ``MultiBoxPrune``
   * - ``mbp_min_x/y/z``, ``mbp_max_x/y/z``, ``mbp_cell_x/y/z``,
       ``mbp_bounds``, ``mbp_padding``, ``mbp_max_axis``, ``mbp_max_object``
     - ``mbpWorldMin``, ``mbpWorldMax``, ``mbpCellSize``,
       ``mbpUseWorldBounds``, ``mbpPadding``, ``mbpMaxCellsPerAxis``,
       ``mbpMaxCellsPerObject`` of ``contact::BroadphaseSettings``
   * - ``sleeping``, ``sleep_linear``, ``sleep_angular``, ``sleep_steps``
     - ``World::setSleepingEnabled`` and ``World::setSleepingParameters``

See :doc:`WorldSystem` for the meaning and defaults of these settings.


Configuration Template
----------------------------
Configuration templates build worlds systematically from parameters, loops, and
reusable files. See
`templatedWorld.xml <https://github.com/raisimTech/raisim2Lib/blob/main/rsc/xmlScripts/templatedWorld/templatedWorld.xml>`__
and the ``xml_templated_world`` example, which loads it.

**Includes.** ``<include value="[THIS_DIR]/terrain.xml"/>`` loads another world
file into the same world before the including file's own content. Use it to
combine sub-worlds.

**Parameters.** A token ``@@name`` is replaced by the value of the parameter
``name``. Define values in the file:

.. code-block:: xml

   <params>
       <object_spawn_height value="0.3"/>
   </params>

or pass them to the constructor:

.. code-block:: cpp

   std::vector<raisim::World::ParameterContainer> params = {{"floor_height", "-1"}};
   raisim::World world("templatedWorld.xml", params);

``<params>`` values apply only to the file that defines them, while constructor
values also apply to included files. A ``<params>`` value takes precedence over
a constructor value with the same name. An ``@@`` token that is still
unresolved after substitution is a fatal error.

**Loops.** ``<array idx="@@i" start="0" end="9" increment="1"> ... </array>``
repeats its content for ``i = start, start + increment, ...`` up to and
including ``end``, replacing every occurrence of the ``idx`` symbol with the
integer value. ``idx``, ``start``, ``end``, and ``increment`` are required;
``start``, ``end``, and ``increment`` must be integers (possibly supplied by
parameters), ``start`` must not exceed ``end``, and ``increment`` must be
positive. Arrays can be nested. See
`spheres.xml <https://github.com/raisimTech/raisim2Lib/blob/main/rsc/xmlScripts/templatedWorld/spheres.xml>`__.

**Math expressions.** Text enclosed in ``{}`` is evaluated after parameters and
loops are expanded and replaced by the result, e.g.
``pos="{cos(0.1*@@i)} {sin(0.1*@@i)} 1"``. Expressions support ``+``, ``-``,
``*``, ``/`` with the usual precedence, parentheses, the functions ``sin``,
``cos``, ``exp``, and ``log``, and the constant ``pi``. Write numbers in plain
decimal notation; exponent notation such as ``1e-3`` is not supported.

**Conditional objects.** An object with ``exist="false"`` (or ``"False"``) is
skipped. Combined with a parameter, e.g. ``exist="@@spawn_box"``, this enables
or disables objects at load time.
