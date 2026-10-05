#######################################
Server Example: Object Lifecycle Stress
#######################################

Overview
========
Stress-tests the object lifecycle by creating many primitives, simulating
briefly, and then removing them, in an endless loop. The example is grouped
with the server examples but runs headless.

Target
======
CMake target: ``object_lifecycle_stress``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/object_lifecycle_stress

On Windows, run ``object_lifecycle_stress.exe`` instead.
This example runs headless and does not use RaisimServer. It runs until you
stop it.

Details
=======
- Each iteration creates a 6 × 6 × 6 grid of boxes, spheres, capsules, and
  cylinders, integrates five steps, and removes them again.
- Intended as a stress test for object lifecycle and memory handling.

