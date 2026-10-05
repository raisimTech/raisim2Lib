#############################
Contact and Collision
#############################

Collision Group and Mask
=========================

Collision groups and masks in RaiSim are 64-bit bit sets (``raisim::CollisionGroup``), as in most other physics engines.
``raisim::COLLISION(n)`` returns the bit for group ``n`` (``1 << n``). Consider the following example:

.. code-block:: cpp

  raisim::World world;
  auto sphere0 = world.addSphere(1, 1, "default", raisim::COLLISION(0), raisim::COLLISION(0) | raisim::COLLISION(1));
  auto sphere1 = world.addSphere(1, 1, "default", raisim::COLLISION(1), raisim::COLLISION(0) | raisim::COLLISION(1));
  auto sphere2 = world.addSphere(1, 1, "default", raisim::COLLISION(2), raisim::COLLISION(1));
  auto sphere3 = world.addSphere(1, 1, "default", raisim::COLLISION(3), -1);

In the above example, ``sphere0`` is in collision group 0 and can collide with collision groups 0 and 1.
``sphere1`` is in collision group 1 and can collide with collision groups 0 and 1.
``sphere2`` is in collision group 2 and can collide with collision group 1.
``sphere3`` is in collision group 3 and its mask accepts every group (-1 sets all bits).

**The collision group and mask use AND logic**.
For A and B to collide, A's group must be in B's mask and B's group must be in A's mask.

``sphere0`` can collide with ``sphere1``.
``sphere1`` cannot collide with ``sphere2`` (``sphere1``'s mask does not include group 2).
``sphere3`` cannot collide with any of the other spheres, because none of their masks includes group 3.
It can still collide with a ground added with ``world.addGround()``, whose default mask is -1.

By default, movable objects use ``collisionGroup = 1`` (that is, ``COLLISION(0)``) and ``collisionMask = -1``, so they collide with everything.
Static objects (e.g., ground and heightmap) are in the static collision group ``RAISIM_STATIC_COLLISION_GROUP``
(bit 63; bit 31 on Windows) and their default mask is -1.

Contacts
=========================

:code:`raisim::Object` (and thus :code:`raisim::ArticulatedSystem` and :code:`raisim::SingleBodyObject`) have a method :code:`getContacts` which returns the list of contacts.
For example,

.. code-block:: cpp

  auto& contactsOnAnymal = anymal->getContacts();

The Contact class is header-only and can be found at :code:`include/raisim/contact/Contact.hpp`.

Each contact has two participants, objectA and objectB. The position (``getPosition()``) and
normal (``getNormal()``) are expressed in the world frame; the normal points from objectB to objectA.
The impulse (``getImpulse()``) is the impulse acting on objectA, expressed in the **contact frame**;
objectB receives the negated impulse. Divide it by the time step to obtain a force.
A contact frame is defined such that its z-axis is collinear with the contact normal and its origin is at the contact point.
Its x- and y-axes are chosen arbitrarily.
``getContactFrame()`` returns the transpose of the contact frame's rotation: each row is a contact-frame axis in world coordinates.
Here is a more detailed example:

.. code-block:: cpp

  /// Check all contact impulses acting on "LF_SHANK"
  auto footIndex = anymal->getBodyIdx("LF_SHANK");

  /// For all contacts on the robot, check ...
  for(auto& contact: anymal->getContacts()) {
    if (contact.skip()) continue; /// the second entry of a self-collision point is set to 'skip'
    if ( footIndex == contact.getlocalBodyIndex() ) {
      std::cout<<"Contact impulse in the contact frame: "<<contact.getImpulse().e().transpose()<<std::endl;
      /// the impulse acts on objectA. You can check if this object is objectA or B by:
      std::cout<<"is ObjectA: "<<contact.isObjectA()<<std::endl;
      std::cout<<"Contact frame: \n"<<contact.getContactFrame().e().transpose()<<std::endl;
      /// getContactFrame() is transposed, so transpose it back to map contact-frame vectors to the world frame.
      std::cout<<"Contact impulse in the world frame: "<<(contact.getContactFrame().e().transpose() * contact.getImpulse().e()).transpose()<<std::endl;
      std::cout<<"Contact Normal in the world frame: "<<contact.getNormal().e().transpose()<<std::endl;
      std::cout<<"Contact position in the world frame: "<<contact.getPosition().e().transpose()<<std::endl;
      std::cout<<"It collides with: "<<world.getObject(contact.getPairObjectIndex())->getName()<<std::endl;
      if (contact.getPairObjectBodyType() != raisim::BodyType::STATIC) {
        /// Static objects do not store contacts, so check whether the pair object is static.
        /// This saves computation in RaiSim.
        world.getObject(contact.getPairObjectIndex())->getContacts(); /// You can use the same methods on the pair object
      }
      std::cout<<"See Contact.hpp for the full list of methods"<<std::endl;
    }
  }

``getImpulse()`` dereferences an internal pointer that is set only for contacts of awake dynamic
objects during ``World::integrate()``. Check ``getImpulsePtr()`` for ``nullptr`` before reading
impulses on kinematic objects or before the first integration step.

Collision detection details
===========================
For a detailed breakdown of collision pairs, narrowphase algorithms, and
per-pair contact counts, see the :doc:`CollisionDetection` section.

Contact Solver Notes
====================

RaiSim uses a bisection-based per-contact solver for rigid contacts and tendon
constraint rows. Closed-loop pin and equality constraints of articulated systems
are eliminated exactly before the solve (see
:doc:`articulated_system/ClosedLoopSystems`). ``World::setContactSolverParam``
still accepts the historical alpha arguments, but the current implementation
keeps the three alpha values fixed internally; only ``maxIter`` (default 150)
and ``threshold`` (default 1e-8 per constraint row) affect the solver
configuration. ``World::setERP(erp, erp2)`` controls the position-error
reduction terms used by the contact solve.

The solver warm-starts ordinary rigid contacts from the previous converged
solve when the object pair matches and a previous contact lies within 1 cm of
the new one. Contacts that are approaching quickly (impacts) start from zero.
The cached impulse is stored in world coordinates and projected into the
current contact frame and friction cone before it is applied. Warm-start data
is intentionally not reused after a non-converged or stalled solve.

Particle-style contacts are handled differently. Deformable and granular
contact points are particles or mesh vertices, so their positions do not
identify a persistent contact. The contact solver therefore does not
warm-start deformable or granular contacts; they are solved from zero
impulses each step.

``World::setContactSolverIterationOrder(order)`` sets the starting sweep
direction for the next contact solve (``true`` is forward). The bisection
solver then flips the stored direction after each solve, so subsequent solves
alternate between forward and reverse sweeps unless the application sets the
starting direction again. Pin rows added directly to the solver make every solve
sweep forward; articulated-system loop constraints are eliminated before the solve
and do not.

API
=========
You can get a vector of contacts on an object using ``raisim::Object::getContacts``.
Each element in the vector has the following API:

.. doxygenclass:: raisim::Contact
   :members:
