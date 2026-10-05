############################################
Benchmark Example: Articulated System
############################################

Overview
========
Headless timing run for articulated-dynamics workloads. It runs several fixed
scenes and prints the time per step of each through ``raisim::print_timediff``.
Use it for articulated-system timing checks.

Target
======
CMake target: ``articulated_system_benchmark``.

Run
===
Run it on one thread for benchmark comparisons:

.. code-block:: bash

   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./build-examples/examples/articulated_system_benchmark

On Windows, run ``articulated_system_benchmark.exe`` instead. The example takes
no arguments and does not use RaisimServer.

Details
=======
The example times these scenes in order:

- ANYmal standing on the ground, then the same robot after the ground is
  removed (1,000,000 steps each).
- Atlas falling with and without the ground (100,000 steps each); the average
  contact count is printed as well.
- Chains of 10 and 20 spring-loaded ball-joint links (30 and 60 DOF; 100,000
  steps each).
