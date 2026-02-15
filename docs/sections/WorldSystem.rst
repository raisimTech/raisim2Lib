#############################
World
#############################
The :code:`raisim::World` class is responsible for the creation and management of all simulation resources.
All objects within a single World instance are capable of collision with one another, provided their collision mask and group settings do not explicitly disable such interactions (refer to the ``Contact and Collision`` section).

A World instance can be instantiated in two ways.
The first method involves loading a RaiSim world configuration file in XML format.
Further details regarding the XML format are available in the ``World Configuration File`` section.

The second method is to dynamically generate the world via code.

These methods can be combined by loading an initial XML configuration and subsequently adding objects dynamically.

An experimental MJCF (MuJoCo file format) reader is also available, allowing for the loading of simple models.

Adding New Objects
============================
To add a new object of type X, utilize the :code:`addX` method.
For example, to add a sphere:

.. code-block:: cpp

  raisim::World world;
  auto sphere = world.addSphere(0.5, 1.0);

:code:`sphere` represents a pointer to the internal resource.
It facilitates access to and modification of internal variables.

Most object-creation methods accept optional :code:`material`, :code:`collisionGroup`, and :code:`collisionMask` arguments.
Static objects such as ground and heightmaps use a fixed collision group and expose only a collision mask.
Collision variables are detailed in the "Contact and Collision" section.
The :code:`material` argument specifies the material governing contact dynamics, as further explained in the "Material System" chapter.

A comprehensive list of objects is provided in the "Object" chapter.

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

Saving the World to an XML File
================================
``raisim::World::exportToXml(dir, file)`` (or ``exportToXml(path)``) saves the current world state to an XML file.
`EXAMPLES <https://github.com/raisimTech/raisim2Lib/blob/master/examples/src/server/length_constraints_newtons_cradle.cpp>`__

Collision detection and broadphase
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

Example configuration (MultiBoxPrune):

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
broadphase and narrowphase. See the "Contact and Collision" section for the
group/mask rules and the default static collision group.

Sleeping islands
================
RaiSim can skip simulation for *sleeping islands*: groups of dynamic objects
connected by contacts. Sleeping is **enabled by default**. An island goes to
sleep when all objects in the island remain quiet for a configurable number of
consecutive steps (:code:`quietSteps`, default **2**) and their maximum linear
and angular velocities stay below the configured thresholds (defaults:
**linear 0.001**, **angular 0.005**).

Notes:

* Only **dynamic** objects participate in sleeping islands.
* Any user modification (e.g., changing state) keeps the island awake.
* Contacts between awake and sleeping islands will wake the sleeping island
  on the next step.

Configuration API:

.. code-block:: cpp

  world.setSleepingEnabled(true);
  world.setSleepingParameters(/*linear*/ 0.001, /*angular*/ 0.005, /*quietSteps*/ 2);
  world.setSleepingVelocityThresholds(0.001, 0.005);
  world.wakeObject(obj);   // wakes the object's island
  world.wakeAll();

You can query whether a specific object is sleeping with
:code:`isObjectSleeping`.

API
=========

.. doxygenclass:: raisim::World
   :members:
