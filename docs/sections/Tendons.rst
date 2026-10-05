#############################
Tendons
#############################

A RaiSim tendon defines one scalar transmission coordinate called its **length**.
A spatial tendon obtains that coordinate from a routed path through attachment
sites and wrapping guides. A fixed tendon obtains it from a weighted sum of
joint coordinates. Springs, damping, actuation, friction, inertia, limits, and
couplings all act through this same coordinate.

The public types are ``raisim::Tendon`` and ``raisim::TendonCoupling``, available
through ``<raisim/World.hpp>``. The world owns them and advances their physics
with the contact solver. Spatial tendons are drawn automatically by Rayrai and
by the TCP viewer connected to ``RaisimServer``.

.. image:: ../../rsc/docs/image/tendons.png
   :alt: Elastic suspensions, cylinder and sphere wrapping, and coupled joints in Rayrai
   :width: 100%

Straight length constraints use this same tendon API with two sites. The legacy
C++ wire functions (``addStiffWire``, ``addCompliantWire``, ``addCustomWire``)
remain as source-compatible adapters: each wire is a view of a two-site spatial
tendon, returned by ``LengthConstraint::getTendon()``. See :doc:`Constraints`
for the mapping from wire behavior to tendon properties and for legacy XML
import.

.. toctree::
   :maxdepth: 2

   tendons/CodeExamples
   tendons/Physics
   tendons/Reference
   tendons/Examples

What the model represents
=========================

Use spatial tendons for cables, belts, ligaments, routed elastic elements, or
other mechanisms whose force depends on a path length. Sites can attach to
rigid bodies, articulated links, individual deformable or granular particles,
or the world. One tendon can span multiple objects and articulated systems.

Use fixed tendons for differential transmissions or coordinated joint actuation.
For example, ``shoulder + 0.65 * elbow`` is a valid tendon coordinate. A fixed
tendon does not lock its joints: the word *fixed* describes constant transmission
coefficients. Locking a length requires a length limit or a coupling.

Both types share the following features:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Feature
     - Physical role
   * - Spring and damping
     - Elastic force around a rest length or outside a slack interval;
       viscous resistance to length change. Integrated implicitly.
   * - Drive
     - Signed feedforward force, position/velocity feedback, optional
       first-order force activation, and limits on the combined actuator force.
   * - Length bounds
     - Independent lower and upper constraints, or a bilateral length lock,
       with position correction and optional compliance.
   * - Dry friction
     - A bounded impulse opposes transmission motion and can hold it at rest.
   * - Armature
     - Additional inertia along the length coordinate, including curvature
       effects for a spatial route.
   * - Coupling
     - A polynomial equality between changes in two tendon lengths, or a
       constraint on one tendon relative to its initial reference length.

The :doc:`tendons/Physics` chapter derives these behaviors and their discrete
solver equations. :doc:`tendons/Reference` lists every property, its default,
and the native XML and MJCF interfaces.

Massless routing and its limits
-------------------------------

The routed geometry is massless. It has no particles of its own, sag under its
own weight, bending stiffness, torsional stiffness, cable thickness in collision
detection, or finite-speed wave propagation. ``armature`` adds transmission
inertia; it does not distribute material mass along the drawn cable.

A spatial tendon interacts with the sites and guides explicitly listed in its
path. It does not discover arbitrary obstacles, collide with other tendons, or
wrap around meshes or cylinder end caps. Sphere and infinite-cylinder guides
are frictionless. ``frictionLoss`` resists the scalar transmission; it does not
implement capstan friction or different tensions on opposite sides of a guide.

The drawing radius is a visualization parameter. To keep a cable visibly outside
a collision surface, set the wrapping radius to the desired cable centerline
radius, accounting for visual thickness yourself.

A short working example
=======================

This complete program suspends a sphere from a world anchor. The spring is slack
up to 1 m, the hard maximum length is 1.5 m, and a small feedforward tension pulls
upward in addition to the spring. Normal RaiSim license discovery applies; an
optional first argument supplies an activation-key path.

.. literalinclude:: ../code/tendon_quickstart.cpp
   :language: cpp
   :caption: tendon_quickstart.cpp

Download :download:`the complete source <../code/tendon_quickstart.cpp>`.
For focused, tested C++ recipes, including current-length locks and straight
two-site connections, see :doc:`tendons/CodeExamples`.
Build and run the larger ``tendon_elastic``, ``tendon_pulleys``,
``tendon_coupling``, and ``rayrai_tendons`` programs using
:doc:`tendons/Examples`.

Three lengths with different meanings
=====================================

Keep the following quantities distinct when configuring a mechanism:

* ``getLength()`` is the current geometric or weighted-joint coordinate.
* ``getReferenceLength()`` is captured when the tendon is created. Polynomial
  couplings use changes relative to this reference.
* ``springLower`` and ``springUpper`` define the spring's rest interval.
  They default to the initial reference length, but can be set independently.

Likewise, ``Drive::targetLength`` is an actuator command, and
``lowerLimit``/``upperLimit`` are constraint bounds. Changing one does not move
the others. Setting the spring endpoints to NaN later restores the spring
interval to the original reference, rather than capturing a new configuration.
Create a tendon after setting its objects' initial poses and joint coordinates.

Force signs and diagnostics
===========================

``getForce()`` is signed: positive force acts to **increase** length.
``getTension()`` is exactly its negative. ``setTension(5.0)`` writes a feedforward
actuation command of ``-5.0``; it does not prescribe the total solved tension.
Other configured force components and actuator limits still apply.

``setTension()`` and ``setActuationForce()`` retain the current servo gains,
targets, and activation time. To switch to a pure force command, explicitly set
both servo gains to zero. A negative reported tension is possible because the
scalar model can transmit forces in either direction.

A spring interval ``[0, restLength]`` makes the *spring term* pull-only for a
nonnegative spatial length. Damping, friction, armature, drives, and lower
limits can still generate forces of either sign. A shortening-only actuator
uses ``actuationUpper = 0`` with a nonpositive lower force bound; that bound
clamps only the actuator contribution.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Getter
     - Meaning and timing
   * - ``getLength()``
     - Cached length at the last geometry refresh. Geometry is refreshed before
       solving a step; refresh again to inspect the newly integrated pose.
   * - ``getVelocity()``
     - Final solved transmission velocity after a step, evaluated with that
       step's Jacobian. After ``updateGeometry()``, velocity at the current pose.
   * - ``getForce()``, ``getTension()``
     - Last-step total force, including passive, actuator, armature, friction,
       limit, and coupling contributions. Force equals solved impulse / timestep.
   * - ``getActuationForce()``
     - Last-step actuator contribution after filtering, implicit feedback, and
       actuator-force saturation.
   * - ``getLimitForce()``, ``getFrictionForce()``
     - Signed sums of the corresponding last-step constraint contributions.
   * - ``getPotentialEnergy()``
     - Spring energy at the cached length; excludes bodies, actuators, and
       compliant constraint rows.
   * - ``getKineticEnergy()``
     - Armature energy at the cached velocity; excludes body kinetic energy.
   * - ``getPointGradients()``, ``getJointGradients()``
     - Current transmission derivatives and application points/indices. Point
       gradients include guide reactions and pulley scaling.

Use ``updateGeometry()`` after manually assigning poses or after integration
when current geometric measurements are needed. It updates length, velocity,
and gradients without advancing the simulation or changing last-step forces.
``updateGeometry(true)`` also refreshes ``getVisualSegments()``. A refresh with
``false`` does not rebuild the stored drawing segments.

Ownership and runtime changes
=============================

Create, look up, enumerate, and remove tendons through ``World``. The returned
pointers are borrowed; do not delete them. Tendon names must be nonempty and
unique among tendons. Coupling names must be nonempty and unique among couplings.

Removing a tendon removes its dependent couplings. Removing an object removes
tendons that reference it, including references through wrapping guides or side
sites, and removes their dependent couplings. Discard borrowed pointers after
such removals. A path and its joint-term list are immutable through the public
API: remove and recreate a tendon to change its topology.

For particle sites, ``localIndex`` identifies one particle, not a stable
handle. ``GranularSystem::removeParticle()`` (also used by the bulk removal
functions) moves the last particle into the removed index. The world then
removes tendons attached to the removed particle, together with their couplings,
and remaps sites on the moved particle to its new index. Other operations that
replace or reorder particles, such as ``GranularSystem::loadGranularState()``,
are not tracked: remove the affected tendons first and recreate them afterward.

``setEnabled(false)`` disables the tendon's solver contributions and, after
scene synchronization, its drawing. Couplings involving that tendon also stop
contributing. Geometry remains inspectable. The next simulated step clears its
reported force components; disabling does not retroactively change previous
step results. The force-activation state is retained and is not advanced while
disabled. Re-enabling resumes from that state.

With sleeping enabled (the default), bodies connected by an enabled tendon or
coupling belong to one sleeping island and sleep and wake together. A tendon
wakes its attachments while its length is changing, while its drive is active
(nonzero feedforward force, activation state, or gain), or while a length lock
has not converged. A tendon that only holds a steady load lets its bodies sleep,
as a resting contact does. Changing a tendon's properties, drive, or enable state
wakes its attached objects. Participating articulated systems use semi-implicit
position integration for the constrained step so their positions advance
consistently with the solved end-step velocities.

Make tendon topology changes between complete steps. Adding/removing tendons or
couplings between ``integrate1()`` and ``integrate2()`` is rejected. Apply
configuration/control changes before a complete step as well. In a server
application, protect edits with the world/server mutex and use
``integrateWorldThreadSafe`` for stepping and control updates. The examples
show this pattern; the tendon physics itself introduces no simulation threads.

Checkpoint capture/restore preserves tendon properties, enable state, drive
commands, references, and activation state along with simulation state.
Topology changes invalidate earlier checkpoints. See :doc:`WorldSystem` for
the surrounding world workflow and :doc:`tendons/Reference` for XML persistence.

Choosing and tuning a model
===========================

Start with a rest interval and a spring when elastic deflection is part of the
mechanism. Use a length bound when the mechanism must prevent extension or
compression. Use a coupling when a relationship between transmission coordinates
must be enforced. Multiple effects can coexist, but incompatible locks, joint
limits, contacts, and couplings can create an unsatisfiable system.

Choose a timestep that resolves the mechanism's motion, especially contact and
wrap/release transitions. Implicit springs improve stiffness stability but do
not guarantee accurate motion at arbitrary timesteps. Increase solver iterations
or tighten the solver tolerance when interacting contacts and tendon rows have
not converged. Check current length errors, actuator saturation, and separate
force components before changing physical parameters.

Worlds with no tendons allocate no tendon-system storage and use the ordinary
contact-solver specialization. Removing the last tendon releases that optional
storage. Enabled tendons add work for geometry, participating body dynamics,
constraint rows, and solver iterations. Spatial armature also evaluates a
curvature bias. Visual tessellation is requested by rendering/streaming, rather
than generated during ordinary physics integration. Measure representative
scenes with the serial headless commands in :doc:`tendons/Examples`.
