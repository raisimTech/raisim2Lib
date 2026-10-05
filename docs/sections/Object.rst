#############################
Objects
#############################

Body types
===============

RaiSim uses ``BodyType`` to describe whether a body or particle is integrated
as a dynamic state, moved kinematically, or treated as static collision
geometry. There are three available body types:

1. ``DYNAMIC``: can have a velocity, has finite mass
2. ``KINEMATIC``: can have a velocity, has infinite mass (e.g., conveyor belt)
3. ``STATIC``: cannot have a velocity, has infinite mass (e.g., wall)

``SingleBodyObject`` instances can be any of the three body types.
``ArticulatedSystem`` instances are dynamic, except fixed-base systems can
report static base bodies. ``DeformableObject`` particles are dynamic unless
they are pinned, in which case the pinned particle reports ``STATIC``.
``GranularSystem`` particles are dynamic unless they are marked fixed, in which
case the fixed particle reports ``STATIC``.

Every object provides ``getBodyType()`` and ``getBodyType(localIdx)``. For
multi-body or multi-particle objects, pass the local body or particle index to
``getBodyType(localIdx)`` to query a specific element. ``SingleBodyObject``
additionally provides ``setBodyType(BodyType type)``.

Object Identity And Indices
===========================

Every object has both a world index and a stable id:

* ``getIndexInWorld()`` is the object's current position in the world's
  object list (``World::getObjList()``, ``World::getObject(index)``). It is
  useful for iterating the current world state, but it can change when an
  object is removed: the last object in the list moves into the removed
  object's index.
* ``getId()`` is assigned by ``World`` and remains stable for the lifetime of
  the object. Use it for runtime scene editing, visualizer bookkeeping, or
  application-side maps that must survive object removal.

Local indices are separate from both of these. A ``SingleBodyObject`` has local
index ``0``. ``ArticulatedSystem`` local indices refer to links. ``DeformableObject``
and ``GranularSystem`` local indices refer to particles.

Name
===============

All objects can be named.
These names are used by visualizers.
``raisim::World::getObject(name)`` retrieves an object by name and returns
``nullptr`` if no object has that name.
Here is an example.

.. code:: cpp

    auto sphere = world.addSphere(1,1);
    sphere->setName("sphere");
    std::string name = sphere->getName();
    auto same_sphere = world.getObject("sphere");

Types
===============

The common base class is ``raisim::Object``. Concrete object families include:

* ``ArticulatedSystem``: URDF/MJCF-style multi-body robots and mechanisms
  (:doc:`ArticulatedSystem`).
* ``SingleBodyObject``: primitive, compound, and mesh rigid bodies with one
  body index (:doc:`SingleBodyObjects`). The static ``Ground`` and
  ``HeightMap`` terrain objects are single-body objects as well
  (:doc:`HeightMap`).
* ``DeformableObject``: XPBD/PBD cloth, shell, and coarse soft-body objects
  (:doc:`DeformableObject`).
* ``GranularSystem``: many spherical grains stored and stepped as one object
  (:doc:`GranularMedia`).

Contacts And External Forces
============================

``getContacts()`` returns the contacts accumulated on an object during the last
world step. For rigid ``SingleBodyObject`` instances and articulated systems,
contact point ids stored in the solver correspond to entries in this contact
list. For particle-like objects such as ``DeformableObject`` and
``GranularSystem``, solver contact point ids identify particles instead. Do not
assume a solver point id can always be used as an index into ``getContacts()``.

``getContactPointVel(pointId, vel)`` returns the contact-point velocity that
the contact solver works with (the velocity being solved for, not the state
before the step). Pass the solver point id described above. For single bodies,
articulated systems, and deformable objects, the velocity is expressed in the
contact frame, whose z axis is the contact normal; for a granular system it is
the particle's linear velocity in the world frame. For world-frame body
velocities, use ``getVelocity(localIdx, vel_w)`` or
``getVelocity(localIdx, pos_b, vel_w)``. For particle-like objects, local body
and force APIs use particle indices.

External forces and torques use local body or particle indices. Forces and
torques are expressed in the world frame and act during the next world step:

.. code-block:: cpp

    object->setExternalForce(localIdx, {0.0, 0.0, 5.0});   // at the center of mass
    object->setExternalTorque(localIdx, {0.0, 0.0, 0.2});
    object->clearExternalForcesAndTorques();

For rigid bodies and articulated links, ``setExternalForce(localIdx, pos, force)``
applies a world-frame force at a position expressed in the body frame. For
particle-like objects such as deformables and granular systems, use the particle
index as ``localIdx``.

Threaded Visualization
======================

``Object`` exposes ``lockMutex``/``unlockMutex`` and ``lock``/``unlock`` for
applications that update object state while a visualizer or another thread is
reading it. Prefer ``std::scoped_lock`` in user code so exceptions or early
returns do not leave the object locked:

.. code-block:: cpp

    std::scoped_lock guard(*object);
    object->setName("updated_name");

API
=========

.. doxygenclass:: raisim::Object
   :members:
