##########################################
Benchmark Example: ANYmal Standing
##########################################

Overview
========
Headless timing run for the ANYmal PD-standing contact workload. It prints the
simulation throughput and the average number of contacts.

Target
======
CMake target: ``anymal_standing_benchmark``.

Run
===
Run it on one thread for benchmark comparisons:

.. code-block:: bash

   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./build-examples/examples/anymal_standing_benchmark --fast

On Windows, run ``anymal_standing_benchmark.exe`` instead. The example does not
use RaisimServer.

Options
=======
* ``--steps N`` or ``--loops N`` sets the number of integration steps (default
  1,000,000).
* ``--fast`` or ``--quick`` caps the step count at 20,000 for smoke tests; an
  explicit ``--steps`` still takes precedence.
* ``ANYMAL_STEPS`` sets the default step count from the environment.
* ``RAISIM_ANYMAL_PHASE_PROFILE=1`` also prints the average ``integrate1`` and
  ``integrate2`` time per step.

Details
=======
- Uses the packaged ANYmal URDF under ``rsc/anymal``.
- Disables sleeping so the measured workload stays active for the whole run.
