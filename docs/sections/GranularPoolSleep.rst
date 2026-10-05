#############################
Granular Pool Sleep
#############################

``raisim::GranularPoolSleepController`` is an opt-in approximation for a
settled granular pool that another object drops into or moves through. It
temporarily fixes quiet grains that are outside a wake sphere around each
incoming object, so only the grains near the disturbance are simulated. The
default simulation does not use it.

Include ``<raisim/object/granular/GranularPoolSleepController.hpp>``. The
controller is header-only and works on one :doc:`GranularMedia` system at a
time.

Usage
=====

Settle the pool first, then call ``arm()``. Call ``update()`` before every world
step with one wake sphere per moving object, and call ``release()`` before
changing the particle count or when sleeping is no longer wanted.

.. code-block:: cpp

    #include <array>
    #include <raisim/object/granular/GranularPoolSleepController.hpp>

    raisim::GranularPoolSleepController sleep;
    sleep.arm(*grains);
    for (int step = 0; step < steps; ++step) {
      raisim::Vec<3> center;
      intruder->getPosition(0, center);
      const std::array sources{
          raisim::GranularPoolSleepController::WakeSphere{center, intruderBoundingRadius}};
      sleep.update(*grains, sources);
      world.integrate();
    }
    sleep.release(*grains);

``arm()`` puts every grain that is currently quiet to sleep. ``update()`` wakes
every sleeping grain whose center is within ``WakeSphere::radius + halo`` of a
wake sphere, then puts other grains to sleep according to the policy.
``release()`` wakes all sleeping grains and detaches the controller.
``sleepingCount()`` returns the number of grains asleep, and
``wakeTransitions()`` counts sleep-to-awake transitions since the last ``arm()``.

A sleeping grain is a fixed grain. It loses its velocity when it is frozen.
Waking it restores dynamic status, not its previous velocity or contact history.
Grains that were already fixed before ``arm()`` are never changed.

Options
=======

Pass ``GranularPoolSleepController::Options`` to the constructor. Every distance
and threshold must be finite and nonnegative; otherwise the constructor throws
``std::invalid_argument``.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Meaning
   * - ``policy``
     - ``QUIET``
     - When awake grains go back to sleep (see below).
   * - ``halo``
     - 0.35 m
     - Distance added to each wake sphere's radius.
   * - ``speedThreshold``
     - 0.02 m/s
     - A grain is quiet only when its linear speed is below this value.
   * - ``angularSpeedThreshold``
     - 2.0 rad/s
     - A grain is quiet only when its angular speed is below this value.
   * - ``quietSteps``
     - 20
     - Consecutive quiet updates before a grain sleeps under ``QUIET``.
   * - ``extraWakeDistance``
     - 0.08 m
     - Extra wake reach under ``WAKE_ONCE``.
   * - ``keepAboveZ``
     - unset
     - When set, grains whose center is at or above this height stay awake.

The policies are:

* ``RADIUS``: every grain outside the wake reach sleeps immediately.
* ``QUIET``: a grain outside the wake reach sleeps after ``quietSteps``
  consecutive quiet updates.
* ``WAKE_ONCE``: a woken grain never goes back to sleep. The wake reach is
  extended by ``extraWakeDistance``.

.. code-block:: cpp

    raisim::GranularPoolSleepController::Options options;
    options.policy = raisim::GranularPoolSleepController::Policy::QUIET;
    options.halo = 0.35;
    options.keepAboveZ = poolTopZ - 4.0 * grainRadius;  // keep the top two layers awake
    raisim::GranularPoolSleepController sleep(options);

Rules
=====

* The wake sphere radius must contain the entire object at every orientation.
* When a fast object can reach a grain before the next ``update()``, add the
  largest expected per-step motion to the halo.
* Include every externally driven object and every other disturbance as a wake
  source. Grains are woken only by wake spheres and ``keepAboveZ``.
* Do not change the pool's particle count while the controller is armed.
  ``update()`` and ``release()`` throw ``std::logic_error`` when the count
  changed. Call ``release()`` first.
* Do not save a checkpoint while the controller is armed; the checkpoint would
  record sleeping grains as fixed grains.
* ``arm()`` throws ``std::logic_error`` when the controller is already armed.

Accuracy
========

No policy reproduces a fully dynamic pool exactly. Frozen grains do not move
internally and do not transmit forces, so the trajectories of the grains and of
the incoming object drift from a fully dynamic simulation. A smaller halo puts
more grains to sleep, which is faster but less accurate. ``WAKE_ONCE`` can
reduce the grain error compared with ``QUIET`` but saves less time, because
grains stay awake after the object passes. Keeping the top layers awake with
``keepAboveZ`` can improve the incoming object's path for some shapes, at the
cost of more awake grains.

Choose the halo and policy from the error your scene can accept, and leave
sleeping disabled when exact replay matters.

API Reference
=============

.. doxygenclass:: raisim::GranularPoolSleepController
   :members:
