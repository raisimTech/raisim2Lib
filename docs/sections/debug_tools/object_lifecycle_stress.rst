#######################################
Debug Tool: Object Lifecycle Stress
#######################################

Overview
========
Stress-tests the object lifecycle by creating many primitives, simulating
briefly, and then removing them, in an endless loop. This manual diagnostic
tool runs headless and can be used to inspect memory handling over time.

Target
======
CMake target: ``object_lifecycle_stress``.
Source: ``examples/tools/debug/object_lifecycle_stress.cpp``.

Run
====
Build with ``RAISIM_DEBUG_TOOLS=ON`` as described in :doc:`../DebugTools`,
then run the build-tree executable:

.. code-block:: bash

   ./build-debug/examples/object_lifecycle_stress

On Windows, run ``build-debug\bin\object_lifecycle_stress.exe`` instead.
The tool runs until you stop it with Ctrl+C.

Details
=======
- Each iteration creates a 6 × 6 × 6 grid of boxes, spheres, capsules, and
  cylinders, integrates five steps, and removes them again.
- Intended as a stress test for object lifecycle and memory handling.
