#################################
Regression and Debug Tools
#################################

These manual tools help investigate object lifecycle and rendering regressions.
Their sources live in ``examples/tools/debug``. Enable the optional
``RAISIM_DEBUG_TOOLS`` CMake option to build them with the examples project.
The option is disabled by default.

Build
=====

From the ``raisim2Lib`` root:

.. code-block:: bash

   cmake -S . -B build-debug \
     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DRAISIM_EXAMPLE=ON \
     -DRAISIM_DEBUG_TOOLS=ON
   cmake --build build-debug --config RelWithDebInfo \
     --target object_lifecycle_stress rayrai_heightmap_replacement --parallel

For a standalone build of ``examples`` against installed RaiSim and rayrai
packages, add ``-DRAISIM_DEBUG_TOOLS=ON`` to the configure command in
:doc:`BuildAndTest`.

Linux and macOS place the top-level build's executables in
``build-debug/examples``; Windows places them in ``build-debug\bin``.
They use the same package libraries and runtime resources as the examples.
See :doc:`Installation` for activation and :doc:`BuildAndTest` for package
build requirements.

Tools
=====

``object_lifecycle_stress`` repeatedly creates and removes primitives in a
headless simulation. It runs until interrupted, allowing memory and lifecycle
behavior to be inspected over time.

``rayrai_heightmap_replacement`` is an interactive rendering regression tool.
Replace terrain and primitives with Space, then check that depth images and
projected points match the current scene. It requires a working rayrai graphics
context.

.. toctree::
   :maxdepth: 1

   debug_tools/object_lifecycle_stress
   debug_tools/rayrai_heightmap_replacement
