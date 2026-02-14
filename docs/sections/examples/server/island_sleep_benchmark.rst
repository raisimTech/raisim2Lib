########################
Island Sleep Benchmark
########################

This example visualizes the sleeping-island system. A "sleeping island" is a
connected group of bodies that has come to rest (low velocity and low contact
activity) and is marked as sleeping by the simulator to skip expensive updates
until something wakes it.

The scene builds several separated box stacks (islands). Boxes turn blue when
sleeping and red when active. Midway through the run, a large sphere is launched
into one stack to wake that island and propagate motion.

Run
====

Build and run the example (it always launches the RaiSim TCP server):

.. code-block:: bash

    cmake --build build --target island_sleep_benchmark
    ./build/examples/island_sleep_benchmark --steps=12000

Arguments
=========

* ``--steps=``: number of simulation steps to run (default: 12000).
* ``--hit-step=``: step index to launch the wake sphere (default: steps/2).

Source
======

.. literalinclude:: ../../../../examples/src/server/island_sleep_benchmark.cpp
   :language: c++
