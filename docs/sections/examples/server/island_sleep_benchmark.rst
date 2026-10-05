##########################################
Benchmark Example: Island Sleep
##########################################

.. image:: ../../../../rsc/docs/image/island_sleep_benchmark.png
   :alt: island_sleep_benchmark example
   :width: 100%

Overview
========
Visualizes the sleeping-island system. A sleeping island is a connected group
of bodies that has come to rest (low velocity and low contact activity). The
simulator marks it as sleeping and skips its updates until something wakes it.

The scene builds four separate 3 × 3 × 3 box stacks (islands). Boxes turn blue
while sleeping and red while active. At step 2500 (5 s), a heavy sphere is
launched into one stack to wake that island.

Target
======
CMake target: ``island_sleep_benchmark``.

Run
===
Start the viewer, then run the example on one thread:

.. code-block:: bash

    # Terminal 1
    ./build-examples/examples/rayrai_tcp_viewer

    # Terminal 2
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      ./build-examples/examples/island_sleep_benchmark --steps=12000

On Windows, run ``island_sleep_benchmark.exe`` instead. This example uses
RaisimServer on port 8080. When the run ends, it prints the number of steps
and the achieved step rate. Each step also sleeps 2 ms so the run can be
watched in real time, so the printed rate is not a raw solver throughput.

Arguments
=========

* ``--steps=N``: number of simulation steps to run (default: 12000).
* ``--hit-step=N``: step at which to launch the wake sphere (default: 2500;
  clamped to the last step for shorter runs).
