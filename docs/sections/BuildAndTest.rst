#############################
Build, Test, and Benchmark
#############################

This page describes the public ``raisim2Lib`` package workflow. The workspace
downloads RaiSim and rayrai as binary packages, then builds examples and
optional wrappers against those packages. It does not contain the RaiSim engine
source tree. For installation and activation, see :doc:`Installation`. For
repository layout and build output paths, see :doc:`ProjectLayout`.

Common CMake options
====================

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Option
     - Meaning
   * - ``RAISIM_EXAMPLE``
     - Build C++ examples. Enabled by default.
   * - ``RAISIM_PY``
     - Build the Python wrapper (``raisimPy``). Disabled by default.
   * - ``RAISIM_DOC``
     - Build documentation through CMake. Disabled by default.
   * - ``RAISIM_ALL``
     - Shortcut that enables ``RAISIM_EXAMPLE``, ``RAISIM_PY``, and
       ``RAISIM_DOC``.
   * - ``RAISIM_MATLAB``
     - Reserved compatibility option. The current public workspace does not
       add a MATLAB wrapper subdirectory or build target.
   * - ``RAISIM_EXAMPLE_DESKTOP_LAUNCHER``
     - Linux only. After each Release build of ``rayrai_tcp_viewer``, install a
       desktop launcher for it and pin it to the GNOME / Ubuntu dock. Disabled
       by default; see :doc:`RayraiTcpViewer`.
   * - ``RAISIM_FOREST_COLLISION_BENCHMARK``
     - Also build the ``forest_collision_benchmark`` example tool. Disabled by
       default.
   * - ``RAISIM_DEBUG_TOOLS``
     - Build manual regression and debug tools with the examples project
       (``RAISIM_EXAMPLE=ON``). Disabled by default; see :doc:`DebugTools`.

The RaiSim version is pinned by the checkout (``RAISIM_VERSION`` in the
top-level ``CMakeLists.txt``). Configuring with a different
``-DRAISIM_VERSION`` is an error; update the checkout instead.

Build examples
==============

Build the package workspace with examples enabled:

.. code-block:: bash

    cd /path/to/raisim2Lib
    cmake -S . -B build-examples \
      -DCMAKE_BUILD_TYPE=Release \
      -DRAISIM_EXAMPLE=ON
    cmake --build build-examples --parallel 12

You can also build only the example CMake project against an installed RaiSim
and rayrai package:

.. code-block:: bash

    cmake -S examples -B /tmp/raisim2lib-examples \
      -DCMAKE_BUILD_TYPE=Release \
      -DRAISIM_PREFIX=/path/to/raisim \
      -DRAYRAI_PREFIX=/path/to/rayrai
    cmake --build /tmp/raisim2lib-examples --parallel 12

Run source-built examples from the build directory. They are not installed by
the ``raisim2Lib`` install target. A top-level build places them under
``<build-dir>/examples``; a standalone build of ``examples`` places them
directly in its build directory:

.. code-block:: bash

    ./build-examples/examples/rayrai_tcp_viewer
    ./build-examples/examples/primitive_grid
    /tmp/raisim2lib-examples/rayrai_tcp_viewer
    /tmp/raisim2lib-examples/primitive_grid

The release package does not include a prebuilt viewer. The
``rayrai_tcp_viewer`` target is built from the viewer sources owned by this
repository and links against the packaged RaiSim and rayrai libraries.

Debug and Release package variants
==================================

The current RaiSim package can contain Debug and Release libraries in one
prefix. Its exported targets map a Debug consumer to the Debug library and
Release, RelWithDebInfo, and MinSizeRel consumers to a non-Debug library.
``RSDEBUG`` follows the selected binary automatically. If the prefix contains
only one variant, every consumer configuration maps to that available variant;
building a downstream project in Debug does not manufacture RaiSim debug
assertions when only the Release library is installed.

Timing examples
===============

``raisim2Lib`` includes standalone timing-oriented example executables. The
closed-source engine benchmark runner used for RaiSim release validation is not
shipped in this public binary-package workspace. Run package timing examples on
one thread and repeat the same command several times when comparing package or
scene changes:

.. code-block:: bash

    cmake --build build-examples \
      --target anymal_standing_benchmark articulated_system_benchmark --parallel 12
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      ./build-examples/examples/anymal_standing_benchmark --fast
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      ./build-examples/examples/articulated_system_benchmark

Use :doc:`Performance` for scene-level tuning guidance and for choosing a
representative package example before changing solver settings or sensor
workloads in a downstream application.

Tests
=====

For manual object lifecycle and rendering regression checks, see
:doc:`DebugTools`.

The public workspace has no CTest suite; the engine's own test suite is not
shipped. The examples double as smoke tests, and ``raisimPy`` includes a
``unittest`` module for its tendon bindings. Run it after building with
``-DRAISIM_PY=ON``, with ``PYTHONPATH`` pointing to the directory that contains
the built ``raisimpy`` module:

.. code-block:: bash

    PYTHONPATH=build/raisimPy RAISIM_ACTIVATION_KEY=$HOME/.raisim/activation.raisim \
      python3 -m unittest raisimPy/tests/test_tendons.py

Build the documentation
=======================

Prefer the CMake documentation target when you need API reference blocks,
because CMake can generate Doxygen XML and pass the Breathe project path to
Sphinx:

.. code-block:: bash

    cmake -S /path/to/raisim2Lib/docs -B /tmp/raisim2lib-docs \
      -DCMAKE_PREFIX_PATH="/path/to/raisim2Lib/raisim;/path/to/raisim2Lib/rayrai" \
      -DRAISIM_DOCS_BUILD_RAYRAI_IMAGES=OFF \
      -DRAISIM_DOCS_RAISIM_INCLUDE_DIR=/path/to/raisim2Lib/raisim/include/raisim \
      -DRAISIM_DOCS_RAYRAI_INCLUDE_DIR=/path/to/raisim2Lib/rayrai/include/rayrai
    cmake --build /tmp/raisim2lib-docs --parallel 12

This generates API references from the distributed headers and reuses the
checked-in documentation images. Enable ``RAISIM_DOCS_BUILD_RAYRAI_IMAGES``
when you have a working OpenGL context and want to regenerate those images.

For prose and link checks, the documentation can also be built directly with
Sphinx. When it cannot find Sphinx, the CMake docs configure creates
``docs/.venv`` with the packages from ``docs/requirements.txt``
(``RAISIM_DOCS_BOOTSTRAP_VENV``, on by default); any Python environment with
those packages works as well:

.. code-block:: bash

    /path/to/raisim2Lib/docs/.venv/bin/sphinx-build \
      -b html /path/to/raisim2Lib/docs /tmp/raisim2lib-docs-html

Direct Sphinx builds do not generate Doxygen XML by themselves. They are useful
for quick RST validation, but Breathe API directives will warn unless
``breathe_projects.raisim`` points to generated Doxygen XML.
See :doc:`Troubleshooting` for common warning modes.
