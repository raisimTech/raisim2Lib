#############################
Constraints
#############################

Length constraints use tendons
==============================

A straight length constraint is a spatial tendon with two sites. Use
``World::addSpatialTendon`` for hard limits, a length lock, springs, or commanded
tension. Routed cables and fixed joint transmissions use the same properties,
solver, inspection, rendering, checkpoint, and removal interfaces. See
:doc:`Tendons`, :doc:`tendons/CodeExamples`, and :doc:`tendons/Physics`.

Migrating the former wire API
=============================

New code should use ``addSpatialTendon``, ``getTendon``, ``getTendons``, and
``removeTendon``. The Python bindings no longer provide ``addStiffWire``,
``addCompliantWire``, ``addCustomWire``, ``getWire``, ``getWires``, or the
``LengthConstraint`` types.

In C++, these functions and classes remain only as source-compatibility
adapters. Each ``add*Wire`` call creates an ordinary two-site tendon (named
``legacy_wire_<n>``) and returns a thin ``LengthConstraint`` view of it;
``LengthConstraint::getTendon()`` returns the underlying tendon, and removing
the view removes the tendon. The tendon physics described below applies to
both. Recompile applications and bindings against matching headers and
libraries.

.. list-table:: Configure a two-site Tendon with target length L
   :header-rows: 1
   :widths: 40 60

   * - Former behavior
     - Tendon properties or command
   * - Stiff, stretch only
     - ``upperLimit = L``
   * - Stiff, compression only
     - ``lowerLimit = L``
   * - Stiff, both directions
     - ``lowerLimit = upperLimit = L``
   * - Compliant, stretch only
     - ``stiffness = k; springLower = 0; springUpper = L``
   * - Compliant, compression only
     - ``stiffness = k; springLower = L; springUpper = +infinity``
   * - Compliant, both directions
     - ``stiffness = k; springLower = springUpper = L``
   * - Custom tension
     - ``tendon->setTension(T)``; positive T pulls

Other properties keep their defaults in this table. A spring interval imposes
no hard bound. Add limits explicitly if the model needs both elasticity and a
maximum or minimum length. ``getLength()`` reports the current transmission
coordinate; the configured target is in ``getProperties()``. Call
``updateGeometry()`` after manually changing object positions before reading it.
``getTension()`` is a signed scalar, positive in tension. ``getForce()`` has the
opposite sign. Appearance is set through ``Properties::width`` and ``color``.
The world owns tendons; removal invalidates their pointers and also removes
couplings that reference them. Removing an attached object removes its tendons.

Existing RaiSim XML ``<wire>`` elements remain readable. The loader converts
them immediately into ordinary two-site tendons, including stretch mode, body
attachments, nominal length, stiffness, and appearance. A custom wire's optional
``tension`` attribute becomes its tension command; absent commands default to
zero. In C++, each converted wire is also reachable by its name through the
legacy ``World::getWire`` view. New XML exports write only ``<tendon>``
elements; converted wires carry ``legacy_*`` attributes so that the view
survives a reload.

The mapping preserves the intended physical model, but does not promise
identical trajectories. Former compliant wires applied explicit spring forces;
tendon springs and damping use the implicit velocity solve described in
:doc:`tendons/Physics`. Hard bounds also use tendon constraint rows now. Recheck
timestep, damping, and tolerances when migrating a tuned simulation.

Historical class links
----------------------

.. toctree::
   :maxdepth: 1

   StiffLengthConstraint
   CompliantLengthConstraint
   CustomLengthConstraint

Pin constraints
===============

Pin and equality constraints close the loops of a closed-loop articulated
system and are specified under URDF ``<constraints>``. A pin keeps two attachment
points coincident in all three directions; an equality constraint keeps them
coincident along one or two axes fixed in its first body. They are eliminated
exactly before the contact solve rather than iterated by it; see
:doc:`articulated_system/ClosedLoopSystems`. They retain their independent API
because they constrain vector position, whereas a tendon constrains one scalar
transmission coordinate.
