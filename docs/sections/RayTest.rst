#############################
Ray Test
#############################

.. image:: ../image/raytest.gif

A ray test is a collision check between the world and a ray.
Users specify the starting point, direction, and length. It returns the closest hit
along the ray (if any).

Arguments
=========
Single-ray test (:code:`rayTest`)
---------------------------------
Signature (C++):

.. code-block:: cpp

    const RayCollisionList& rayTest(const Vec<3>& start,
                                    const Vec<3>& direction,
                                    double length,
                                    size_t objectId = size_t(-10),
                                    size_t localId = size_t(-10),
                                    CollisionGroup collisionMask = CollisionGroup(-1));

* :code:`start`: ray origin in world frame.
* :code:`direction`: ray direction in world frame. It does not need to be normalized.
* :code:`length`: ray length in meters. Only hits within [0, length] are returned.
* :code:`objectId` / :code:`localId`: optional self-filter. Collisions with the given
  object and local body id pair are ignored (useful for sensors attached to robots).
  The defaults disable this filter.
* :code:`collisionMask`: collision mask to filter which groups the ray can hit.
  The default (-1) allows all groups.

The returned :code:`RayCollisionList` stores only the closest hit (0 or 1 item).

Example
========================================

An example is provided `here <https://github.com/raisimTech/raisim2Lib/blob/master/examples/src/server/ray_casting.cpp>`__.
Only the following line

.. code-block:: cpp

    auto& col = world.rayTest({0,0,5}, direction, 50.);

performs the ray test. The rest of the code is for demonstration only. The ray test stores only the closest hit.

Getting the hit object (example)
--------------------------------
To identify what was hit, check the returned :code:`RayCollisionList`.

.. code-block:: cpp

    const auto& hits = world.rayTest({0, 0, 5}, direction, 50.0);
    if (hits.size() == 0) {
      std::cout << "No hit\n";
    } else {
      const auto& hit = hits[0];
      const auto* obj = hit.getObject();
      if (obj) {
        std::cout << "Hit object: " << obj->getName()
                  << " (id=" << obj->getIndexInWorld() << ")\n";
      }
      std::cout << "Hit position: " << hit.getPosition().transpose() << "\n";
      const auto body = hit.getCollisionBody();
      if (body) {
        std::cout << "Collision body: " << body->name << "\n";
      }
    }

Batch ray tests (lidar)
=======================
For spinning lidar-style scans, use :code:`World::rayTestLidar(...)` instead of
calling :code:`rayTest(...)` for each ray. It is significantly faster because it:

* Performs a **single frustum-based broadphase** to collect candidate bodies for
  the whole scan (yaw/pitch/range window), instead of repeating a full world
  scan per ray.
* Reuses the internal ray body and avoids per-ray allocations.
* Performs per-ray AABB checks only on the prefiltered candidate list.

Minimal example:

.. code-block:: cpp

    std::vector<raisim::Vec<3>, raisim::AlignedAllocator<raisim::Vec<3>, 32>> scan;
    world.rayTestLidar(rot, pos,
                       yawStartCount, yawEndCount, yawIncrement, spinDirection,
                       pitchSamples, pitchMinAngle, pitchIncrement,
                       rangeMin, rangeMax,
                       objectId, localId, collisionMask,
                       scan);

The output :code:`scan` contains hit points in the sensor frame (one per ray that hits).

Arguments (batched lidar scan)
------------------------------
* :code:`rot`: sensor orientation (sensor frame to world frame).
* :code:`pos`: sensor position in world frame.
* :code:`yawStartCount`, :code:`yawEndCount`, :code:`yawIncrement`: yaw sweep range
  and step (in radians). The scan iterates from start to end using :code:`spinDirection`.
* :code:`spinDirection`: +1 or -1, controlling yaw sweep direction.
* :code:`pitchSamples`, :code:`pitchMinAngle`, :code:`pitchIncrement`: pitch sampling
  configuration (in radians).
* :code:`rangeMin`, :code:`rangeMax`: min/max range in meters.
* :code:`objectId` / :code:`localId`: optional self-filter (same as :code:`rayTest`).
* :code:`collisionMask`: collision mask to filter which groups the rays can hit.
* :code:`scan`: output hit positions in the sensor frame.


Details
=======
RayCollisionItem summary
------------------------
Each hit entry stores:

* :code:`getObject()`: pointer to the hit :code:`raisim::Object` (or nullptr if none).
* :code:`getPosition()`: world-frame hit position (contact point).
* :code:`getCollisionBody()`: collision body handle for the hit (may be null for some objects).

RayCollisionList summary
------------------------
Container semantics:

* :code:`size()`: number of valid hits in the list.
* :code:`operator[](i)`: random access to the i-th hit.
* :code:`begin()`, :code:`end()`: iterators over valid hits.
* :code:`back()`: iterator to the last valid hit.
* :code:`setSize(n)`: updates the number of valid hits (used internally).
* :code:`resize(n)`, :code:`reserve(n)`: storage management.

Notes
=====
* :code:`RayCollisionList` is a lightweight container reused across calls. It is
  cleared and populated by :code:`rayTest(...)`.
* It contains **at most one item** because RaiSim keeps only the closest hit
  along the ray. Use :code:`list.size()` to check whether a hit occurred (0 or 1).

API
====

RayCollisionItem
----------------

.. doxygenclass:: raisim::RayCollisionItem
   :members:

RayCollisionList
----------------

.. doxygenclass:: raisim::RayCollisionList
   :members:
