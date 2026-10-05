#############################
Tendon API and model files
#############################

All names below refer to the current C++ tendon API. Include
``<raisim/World.hpp>``. :doc:`Physics` gives the equations and sign conventions;
:doc:`CodeExamples` provides focused, tested C++ recipes, and
:doc:`Examples` provides complete runnable scenes.

World API
=========

.. code-block:: cpp

   auto* spatial = world.addSpatialTendon(name, path, properties);
   auto* fixed = world.addFixedTendon(name, jointTerms, properties);
   auto* equality = world.addTendonCoupling(name, first, second, couplingProperties);

   auto* found = world.getTendon(name);  // nullptr when absent
   const auto& tendons = world.getTendons();
   const auto& couplings = world.getTendonCouplings();

   world.removeTendonCoupling(equality);
   world.removeTendon(spatial);

The properties arguments can be omitted to use defaults. ``getTendons()`` and
``getTendonCouplings()`` return const references to vectors of world-owned
``unique_ptr`` objects. ``getTendon()`` looks up a tendon by name; enumerate
couplings and inspect their names to find a coupling. See :doc:`../Tendons` for
ownership, deletion, enable state, and split-integration restrictions.

Sites, paths, and joints
========================

``Tendon::Site`` contains ``object``, ``localIndex``, and ``position``:

.. list-table::
   :header-rows: 1
   :widths: 26 26 48

   * - Attachment
     - ``localIndex``
     - ``position``
   * - World (``object == nullptr``)
     - 0
     - World coordinates.
   * - Single rigid body
     - 0
     - Body-local coordinates, not a world-space point.
   * - Articulated system
     - Simulated body index, e.g. ``getBodyIdx(linkName)``
     - Coordinates in that body's frame. A frame index and a body index are
       different concepts.
   * - Deformable/granular system
     - Particle index
     - World-aligned offset from the particle center (particles have no
       orientation). Use zero to attach directly to the particle.

For an imported fixed frame, resolve its simulated parent body and express the
attachment point in that parent's frame. Do not use a fixed-frame index as a
body index. All referenced objects must belong to the same world as the tendon.

Build a spatial path with ``Tendon::PathElement``:

.. code-block:: cpp

   using E = raisim::Tendon::PathElement;
   auto anchor = E::via({nullptr, 0, {0.0, 0.0, 3.0}});
   auto endpoint = E::via({load, 0, {0.0, 0.0, 0.1}});
   auto guide = E::cylinder({guideBody}, 0.2, {0.0, 1.0, 0.0});
   guide.withSideSite({nullptr, 0, {0.0, 0.0, 3.5}});
   auto* tendon = world.addSpatialTendon("route", {anchor, guide, endpoint});

The example fragment assumes ``load`` and ``guideBody`` already exist and have
suitable exterior placement. ``sphere(center, radius)`` supplies a sphere guide.
``cylinder(center, radius, axis)`` defaults to the local +Z axis. The axis is
normalized during validation; it must be finite and nonzero. Both radii and
pulley divisors must be finite and strictly positive. A guide center must be in
the world frame, on a single rigid body, or on an articulated-system body;
particles cannot carry a wrapping guide.

Every branch needs at least two sites, starting and ending with sites. A wrap
must have an immediate site on both sides. Consequently two consecutive wraps
need an intervening via-point. ``E::pulley(divisor)`` starts a new branch; a
leading pulley is permitted. A path ending in a pulley or a wrap is rejected.

A fixed tendon uses ``Tendon::JointTerm{system, jointName, coefficient}``:

.. code-block:: cpp

   auto* transmission = world.addFixedTendon("finger", {
       {robot, "proximal_joint", 1.0},
       {robot, "distal_joint", 0.65}});

Only revolute and prismatic joint names are accepted. At least one term is
required. ``getPath()`` and ``getJoints()`` expose const route/term lists.
``getType()`` distinguishes ``Tendon::Type::Spatial`` and ``Fixed``.
``getName()`` reads the name and ``setName()`` renames a tendon, checking that
its new name is nonempty and unique.

After a geometry refresh, the diagnostic records expose:

* ``PointGradient::site``: the owning attachment/body and body-local application
  point; ``position``: its world-space application point; ``gradient``: the
  world-space length derivative, including branch scaling. For a wrapped guide,
  records identify the instantaneous tangency points on that guide.
* ``JointGradient::system``, ``coordinateIndex``, ``velocityIndex``, and
  ``coefficient``: the resolved scalar-joint transmission term. The solver sums
  repeated contributions even though the returned term list retains them.
* ``VisualSegment::start`` and ``end``: world-space endpoints of one drawing
  segment; ``wrapped``: whether it tessellates the curved surface portion.
  ``getVisualSegments()`` returns the records cached by the last
  ``updateGeometry(true)`` call. ``computeVisualSegments(segments)`` instead
  fills a caller-owned vector at the current poses without modifying the
  tendon; renderers use this const function.

Tendon properties
=================

Read the current value, modify a copy, and call ``setProperties()``. Constructing
a fresh ``Properties`` instead resets any fields you do not explicitly set.
The defaults below describe a newly constructed properties object.

.. list-table::
   :header-rows: 1
   :widths: 26 19 55

   * - C++ field
     - Default
     - Meaning
   * - ``stiffness``
     - 0
     - Nonnegative spring stiffness outside the rest interval.
   * - ``damping``
     - 0
     - Nonnegative viscous resistance to length velocity, including inside the
       spring's slack interval.
   * - ``springLower``, ``springUpper``
     - NaN, NaN
     - Both NaN resolve to the creation reference length. Otherwise endpoints must be ordered; the lower endpoint may be
       -infinity and the upper endpoint may be +infinity. Equal endpoints give a rest length; unequal endpoints
       give a zero-spring-force interval.
   * - ``lowerLimit``, ``upperLimit``
     - -infinity, +infinity
     - Ordered constraint bounds. Equal finite values lock length. These do
       not set the spring rest interval or the drive target.
   * - ``limitMargin``
     - 0
     - Nonnegative inward offset of unilateral constraint boundaries. Ignored
       for an equal-bound lock.
   * - ``limitCompliance``
     - 0
     - Nonnegative length-constraint softness, used as compliance / timestep².
       See the correction/compliance relation in :doc:`Physics`.
   * - ``frictionLoss``
     - 0
     - Nonnegative dry-friction force threshold for transmission motion.
   * - ``frictionCompliance``
     - 0
     - Nonnegative friction-row regularization, used as compliance / timestep².
       Positive values permit creep under load.
   * - ``armature``
     - 0
     - Nonnegative added inertia along the transmission.
   * - ``positionCorrection``
     - 0.2
     - Length-constraint position-correction fraction in [0, 1]. Couplings have
       their own independent correction property.
   * - ``actuationLower``, ``actuationUpper``
     - -infinity, +infinity
     - Ordered bounds on combined actuator force. The interval need not contain
       zero. Passive and constraint forces are excluded.
   * - ``width``
     - 0.004
     - Strictly positive visualization radius in metres; no collision thickness.
   * - ``color``
     - {0.9, 0.35, 0.08, 1.0}
     - Finite RGBA components. Use values in [0, 1] for conventional rendering.

All nonnegative scalar fields must also be finite. Spring/length/actuation bounds may
be unbounded only in their corresponding outward directions. NaN is rejected
except for the paired automatic spring endpoints. Invalid settings throw
``std::invalid_argument``. Properties are shared by spatial and fixed tendons;
choose the coordinate units consistently as explained in :doc:`Physics`.

Drive properties
================

``setDrive()`` replaces the drive configuration; ``getDrive()`` returns a const
reference. Copy it before changing individual fields.

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - ``Tendon::Drive`` field
     - Default
     - Meaning
   * - ``force``
     - 0
     - Finite signed feedforward force; positive lengthens the transmission.
   * - ``targetLength``
     - 0
     - Finite position target. It is not automatically initialized to the
       creation reference. Set it before enabling a position servo.
   * - ``targetVelocity``
     - 0
     - Finite desired transmission velocity.
   * - ``positionGain``
     - 0
     - Finite nonnegative implicit position-feedback gain.
   * - ``velocityGain``
     - 0
     - Finite nonnegative implicit velocity-feedback gain.
   * - ``activationTime``
     - 0
     - Finite nonnegative feedforward activation time constant. Zero applies
       feedforward commands immediately during the next step.

.. code-block:: cpp

   raisim::Tendon::Drive drive;
   drive.targetLength = transmission->getLength();
   drive.positionGain = 120.0;
   drive.velocityGain = 12.0;
   transmission->setDrive(drive);

   auto p = transmission->getProperties();
   p.actuationLower = -50.0;
   p.actuationUpper = 0.0;  // Actuator may only pull toward smaller length.
   transmission->setProperties(p);

``setActuationForce(value)`` changes only ``Drive::force``.
``setTension(value)`` changes it to ``-value``. They leave servo gains, targets,
and activation time unchanged. Force commands take effect on simulation steps.

Coupling properties and API
===========================

``TendonCoupling::Properties`` contains:

.. list-table::
   :header-rows: 1
   :widths: 29 22 49

   * - Field
     - Default
     - Meaning
   * - ``coefficients``
     - {0, 1, 0, 0, 0}
     - Five finite coefficients, constant term first. Default couples equal
       changes from the two creation references.
   * - ``compliance``
     - 0
     - Finite nonnegative coupling-row compliance.
   * - ``positionCorrection``
     - 0.2
     - Finite position-correction fraction in [0, 1].
   * - ``enabled``
     - true
     - Whether the coupling contributes, provided both tendons are enabled.

.. code-block:: cpp

   raisim::TendonCoupling::Properties relation;
   relation.coefficients = {0.0, 2.0, 0.1, 0.0, 0.0};
   auto* coupling = world.addTendonCoupling("ratio", first, second, relation);
   // deltaLengthFirst = 2*x + 0.1*x*x, x = deltaLengthSecond

``getFirst()`` and ``getSecond()`` return borrowed tendon pointers; the second
may be null. ``getName()``, ``getProperties()``, ``setProperties()``, and
``getForce()`` provide inspection/configuration. There is no independent spatial
path for a coupling. Its force is the multiplier for the first tendon, with the
second receiving the derivative-scaled reaction described in :doc:`Physics`.

Native world XML
================

Native XML places each ``<tendon>`` and ``<tendon_coupling>`` directly under
``<raisim>``, alongside ``<objects>``. This differs from the MJCF container
syntax. Here is a complete world corresponding to the introductory suspension:

.. literalinclude:: ../../code/tendon_world.xml
   :language: xml
   :caption: tendon_world.xml

Download :download:`the native XML <../../code/tendon_world.xml>`. Load it with
``raisim::World world(xmlPath)`` or drop it onto the TCP viewer to simulate it.

Tendon attributes include ``name``, ``type="spatial|fixed"``, ``enabled``,
``reference_length``, and ``activation``. Missing reference length defaults to
the imported tendon's initial length. Missing activation defaults to zero.
Numeric property names follow this mapping:

.. code-block:: text

   stiffness             damping               spring_lower
   spring_upper          lower_limit           upper_limit
   limit_margin          limit_compliance      friction_loss
   friction_compliance   armature              position_correction
   actuation_lower       actuation_upper       width
   color

The optional ``<drive>`` child accepts ``force``, ``target_length``,
``target_velocity``, ``position_gain``, ``velocity_gain``, and
``activation_time``. Unspecified properties use the C++ defaults.

Spatial path children remain in routing order:

* ``<site object="load" local_index="0" pos="0, 0, 0.1"/>`` is an attachment.
  Omit ``object`` for a world site; omitted index and position default to zero.
* ``<sphere ... radius="0.2"/>`` and ``<cylinder ... radius="0.2"
  axis="0, 1, 0"/>`` use the same site attributes to locate their centers.
  An optional ``<side .../>`` child defines the side site with the same site
  attributes. A missing cylinder axis defaults to local +Z.
* ``<pulley divisor="2"/>`` splits branches. Its divisor is required.

A fixed tendon instead contains ``<joint object="robot" joint="joint_name"
coefficient="0.65"/>`` children. The object name must identify an articulated
system; joint names identify its revolute/prismatic coordinates.

A coupling uses the required ``name`` and ``first`` attributes and the optional
``second``, ``coefficients="c0, c1, c2, c3, c4"``, ``compliance``,
``position_correction``, and ``enabled`` attributes. The referenced tendons must
exist. Omitting the second fixes the first relative to its reference length.

Give referenced objects unique names. Unknown or ambiguous object names and
unknown tendon references are rejected. Export writes the current reference and
activation state as well as properties and commands. It does not serialize an
arbitrary C++ controller or a future schedule of changing servo targets. A
tendon that backs a legacy C++ wire adapter is exported with additional
``legacy_*`` attributes, so loading the file restores the wire view as well.

For a complete scene, prefer ``World::exportToXml``. The example programs expose
``--export PATH`` and export initial scenes. Their embedded articulated models
produce URDF sidecar files, with absolute references in the XML. Keep those
assets and update paths if moving the exported scene. The example exporter
passes an absolute directory with a trailing separator to the directory/filename
overload so embedded URDF files are written beside the XML. Native export also
depends on the objects themselves being exportable; the tendon exporter does
not reconstruct URDF descriptions for arbitrary ``Child``-built articulations.

MJCF import
===========

RaiSim imports tendon definitions through its existing MJCF world loader.
Supported tendon elements and attribute mappings are:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - MJCF construct
     - RaiSim interpretation
   * - ``<tendon><spatial>``
     - Ordered named sites, sphere/cylinder ``geom`` guides, optional
       ``sidesite``, and ``pulley divisor`` branches.
   * - ``<tendon><fixed>``
     - Named scalar joints with ``coef`` coefficients.
   * - Tendon defaults/classes
     - Inherited tendon properties are resolved by the model loader.
   * - ``springlength``
     - One value sets a rest length; two values set an interval. ``-1`` or
       ``-1 -1`` selects the initial length.
   * - ``stiffness``, ``damping``, ``armature``, ``frictionloss``
     - Corresponding tendon properties.
   * - ``limited``, ``range``, ``margin``
     - Length bounds and margin. ``auto`` uses a supplied range;
       ``limited="false"`` disables those bounds.
   * - ``actuatorfrclimited``, ``actuatorfrcrange``
     - Bounds on the tendon actuator contribution. ``auto`` uses a supplied
       force range; explicit ``false`` disables the bounds.
   * - ``width``, ``rgba``
     - Drawing radius and color.
   * - ``<equality><tendon>``
     - ``tendon1``, optional ``tendon2``, ``polycoef``, and ``active`` map to a
       polynomial coupling and its enable state.

Routing guides are resolved from declared sphere/cylinder geometry even when
that geometry is excluded from collision detection. These remain the
frictionless routing primitives described in :doc:`Physics`.

The importer does not load the separate MuJoCo actuator subsystem into tendon
``Drive`` objects. Configure actuation through the C++ API after import.
MuJoCo's ``solref``/``solimp`` constraint response is not reproduced by RaiSim's
compliance/correction model. Explicit tendon ``solreflimit``, ``solimplimit``,
``solreffriction``, or ``solimpfriction`` parameters trigger a warning about this
difference. Equality solver parameters also do not establish equivalent RaiSim
constraint behavior. Supported model syntax therefore does not imply identical
force histories, solver convergence, or trajectories across engines.
