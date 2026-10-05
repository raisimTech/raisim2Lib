#############################
Ray Test
#############################

.. image:: ../../rsc/docs/image/raytest.gif

A ray test is a collision check between the world and a ray.
Users specify the starting point, direction, and length. It returns the closest hit
along the ray (if any).

Arguments
=========
Single-ray test (:code:`rayTest`)
---------------------------------
Signature (C++):

.. code-block:: cpp

    const RayCollisionList& rayTest(const Eigen::Ref<const Eigen::Vector3d, Eigen::Unaligned>& start,
                                    const Eigen::Ref<const Eigen::Vector3d, Eigen::Unaligned>& direction,
                                    double length,
                                    size_t objectId = size_t(-10),
                                    size_t localId = size_t(-10),
                                    CollisionGroup collisionMask = CollisionGroup(-1));

Overloads accept ``{x, y, z}`` initializer lists for ``start``, ``direction``,
or both. Pass a ``raisim::Vec<3>`` as ``vec.e()``.

* :code:`start`: ray origin in world frame.
* :code:`direction`: ray direction in world frame. It is normalized internally,
  so it does not need to be a unit vector; a zero vector is treated as +z.
* :code:`length`: ray length in meters. Only hits within [0, length] are returned;
  a non-positive length returns no hit.
* :code:`objectId` / :code:`localId`: optional self-filter. Collisions with the body
  whose object world index (``Object::getIndexInWorld()``) and local body index
  match are ignored (useful for sensors attached to robots). Both must be set for
  the filter to take effect; the defaults disable it.
* :code:`collisionMask`: collision mask to filter which groups the ray can hit.
  The default (-1) allows all groups.

The returned :code:`RayCollisionList` stores only the closest hit (0 or 1 item).

Example
========================================

The ray test itself is a single call:

.. code-block:: cpp

    auto& col = world.rayTest({0,0,5}, direction, 50.);

The ``ray_casting`` example sweeps this ray over the ground and visualizes the
hit point through ``RaisimServer``.

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
For spinning lidar-style scans, :code:`World::rayTestLidar(...)` generates a
rectangular yaw/pitch sampling pattern and performs the scan in one call.

Signature (C++):

.. code-block:: cpp

    void rayTestLidar(
        const Mat<3, 3>& rot,
        const Vec<3>& pos,
        double yawStartAngle,
        double yawIncrement,
        size_t yawCount,
        double pitchStartAngle,
        double pitchIncrement,
        size_t pitchCount,
        double rangeMin,
        double rangeMax,
        size_t objectId,
        size_t localId,
        CollisionGroup collisionMask,
        std::vector<Vec<3>, AlignedAllocator<Vec<3>, 32>>& scan);

Both angular axes use the same :code:`start angle, increment, count` convention.
For sample indices :math:`y` and :math:`p`, the angles are

.. math::

    \mathrm{yaw}(y) = \mathrm{yawStartAngle} + y\,\mathrm{yawIncrement}

.. math::

    \mathrm{pitch}(p) = \mathrm{pitchStartAngle} + p\,\mathrm{pitchIncrement}

where :math:`0 \le y < \mathrm{yawCount}` and
:math:`0 \le p < \mathrm{pitchCount}`. Increments are signed: use a negative
increment to scan an axis in the decreasing-angle direction. A single-sample
axis normally uses an increment of zero. If either count is zero, an angle is
not finite, or :code:`rangeMax <= rangeMin`, :code:`scan` is cleared and no
rays are cast.

Each ray direction in the sensor frame is

.. math::

    \boldsymbol{d} = (\cos\mathrm{pitch}\cos\mathrm{yaw},\;
                  \cos\mathrm{pitch}\sin\mathrm{yaw},\;
                  \sin\mathrm{pitch}),

so yaw rotates about the sensor z axis starting from the sensor x axis, and a
positive pitch tilts the ray toward the sensor +z axis.

Minimal example for a complete yaw revolution with 16 pitch channels
(``rot``, ``pos``, ``rangeMin``, ``rangeMax``, ``objectId``, ``localId``, and
``collisionMask`` come from the application):

.. code-block:: cpp

    constexpr double pi = 3.14159265358979323846;
    constexpr size_t yawCount = 1024;
    constexpr size_t pitchCount = 16;
    constexpr double yawStartAngle = -pi;
    constexpr double yawIncrement = 2.0 * pi / double(yawCount);
    constexpr double pitchStartAngle = -15.0 * pi / 180.0;
    constexpr double pitchEndAngle = 15.0 * pi / 180.0;
    constexpr double pitchIncrement =
        (pitchEndAngle - pitchStartAngle) / double(pitchCount - 1);

    std::vector<raisim::Vec<3>, raisim::AlignedAllocator<raisim::Vec<3>, 32>> scan;
    world.rayTestLidar(rot, pos,
                       yawStartAngle, yawIncrement, yawCount,
                       pitchStartAngle, pitchIncrement, pitchCount,
                       rangeMin, rangeMax,
                       objectId, localId, collisionMask,
                       scan);

Yaw is the outer loop and pitch is the inner loop. Consequently, rays are
generated in this order:

.. code-block:: text

    (yaw[0], pitch[0]), (yaw[0], pitch[1]), ...,
    (yaw[1], pitch[0]), (yaw[1], pitch[1]), ...

The output :code:`scan` contains hit points in the sensor frame. It is compact:
only rays that hit contribute a point, so a miss does not create a placeholder
entry and indices after a miss no longer map directly to angular sample indices.

Arguments (batched lidar scan)
------------------------------
* :code:`rot`: sensor orientation (sensor frame to world frame).
* :code:`pos`: sensor position in world frame.
* :code:`yawStartAngle`, :code:`yawIncrement`, :code:`yawCount`: first yaw angle,
  signed yaw step, and number of yaw samples.
* :code:`pitchStartAngle`, :code:`pitchIncrement`, :code:`pitchCount`: first pitch
  angle, signed pitch step, and number of pitch samples.
* :code:`rangeMin`, :code:`rangeMax`: minimum and maximum sensor-relative range
  in meters. Each collision ray starts at :code:`rangeMin` and has length
  :code:`rangeMax - rangeMin`.
* :code:`objectId` / :code:`localId`: optional self-filter (same as :code:`rayTest`).
* :code:`collisionMask`: collision mask to filter which groups the rays can hit.
* :code:`scan`: output hit positions in the sensor frame.

Performance notes
-----------------
:code:`rayTestLidar` is optimized for repeated structured scans. It caches the
sensor-frame directions and their normalized forms for an unchanged yaw/pitch
pattern. In a sufficiently large, spatially sparse scene it also builds
conservative angular candidate buckets, so each ray normally checks only nearby
bodies. The buckets are reused while the scene, sensor pose, ranges, self-filter,
and collision mask are unchanged; changes to any of those inputs invalidate the
relevant cache. Bucket bounds are conservative and every retained candidate
still goes through an exact shape intersection. Sphere candidates use the same
quadratic roots and range rules as the contact engine but omit the unused
contact-normal calculation. No path approximates hit positions.

Dense or unsupported angular configurations automatically use the exact ray
BVH instead. Small scenes use a scan-wide conservative cull and a compact flat
candidate list. These fallbacks preserve the same closest-hit and compact-output
semantics.

Use the public ``ray_scan_lidar`` example to inspect structured scans and
``large_scale_ray_test`` to inspect repeated ray queries. Build them together
with the TCP viewer and start the viewer:

.. code-block:: bash

    cmake --build build-examples --target ray_scan_lidar large_scale_ray_test rayrai_tcp_viewer --parallel 12
    ./build-examples/examples/rayrai_tcp_viewer

Then start either simulation (for example,
``./build-examples/examples/ray_scan_lidar``) in another terminal. These examples visualize the APIs;
they are not timing comparisons. For application benchmarks, time
``World::rayTestLidar`` and repeated ``World::rayTest`` calls with identical
sensor poses, rays, ranges, and collision filters on one thread, and verify the
hit positions before comparing timings. Relative speed depends on scene layout,
motion, filters, and scan shape.


Details
=======
RayCollisionItem summary
------------------------
Each hit entry stores:

* :code:`getObject()`: pointer to the hit :code:`raisim::Object` (or nullptr if none).
* :code:`getPosition()`: world-frame hit position as an ``Eigen::Vector3d``.
* :code:`getCollisionBody()`: pointer to the collision body that was hit (may be null for some objects).

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
* :code:`RayCollisionList` is a lightweight container owned by the world and
  reused across calls. Every :code:`rayTest(...)` call overwrites it, so copy
  any hit you need before the next ray test.
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
