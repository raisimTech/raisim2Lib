#############################
Project Layout
#############################

``raisim2Lib`` is the public package workspace. It downloads and uses binary
RaiSim and rayrai packages, then provides examples, wrappers, resources, and
documentation around those packages. It is not the RaiSim engine source tree,
so engine-internal benchmark sources and implementation files are not available
here. This page maps the directories users most often need while installing the
package and building examples.

Public Workspace Directories
============================

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Path
     - Purpose
   * - ``raisim``
     - Platform-specific binary RaiSim package with headers, libraries, and
       CMake config files. Downloaded by the first CMake configure (or
       ``raisim_upgrade``); not tracked in Git.
   * - ``rayrai``
     - Platform-specific binary rayrai package with the renderer library,
       headers, and CMake config files. Downloaded together with ``raisim``.
   * - ``examples``
     - C++ example sources and their CMake project. CMake target names include
       ``primitive_grid`` and ``rayrai_tcp_viewer``.
   * - ``examples/tools/debug``
     - Optional manual regression and debug tools; see :doc:`DebugTools`.
   * - ``rsc``
     - Runtime resources such as robot models, meshes, textures, USD/glTF
       assets, and example data.
   * - ``raisimPy``
     - Optional Python wrapper build for the binary RaiSim package.
   * - ``raisimGymTorch``
     - Reinforcement-learning package built against the binary RaiSim package.
   * - ``docs``
     - Sphinx documentation sources and generated example pages.
   * - ``thirdParty``
     - Bundled third-party sources: Eigen3 (used by Windows builds) and
       nanobind (used by ``raisimPy``).
   * - ``cmake``
     - CMake find modules for the build (``FindSphinx.cmake`` for the
       documentation build).
   * - ``scripts``
     - Helper scripts: the Blender scene exporter
       (``export_blender_scene.py``) and the Linux desktop-launcher installer
       for ``rayrai_tcp_viewer``.
   * - ``raisim_env.*``, ``raisim_upgrade.*``
     - Environment scripts that add the package library directories to the
       library search path (``.sh``, ``.bat``, ``.ps1``) and scripts that
       download a release package into ``raisim`` and ``rayrai``
       (``.sh``, ``.ps1``).

Build Directories
=================

Build directories are generated locally and are not part of the repository. The docs use these names for
clarity:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Build directory
     - Typical use
   * - ``build``
     - Default local build for examples and optional wrappers.
   * - ``build-examples``
     - Top-level build with examples enabled; the example commands in these
       docs use it.
   * - ``build-debug``
     - Debug build for local debugging.
   * - ``build-docs``
     - CMake-driven docs build.

On Linux and macOS, a top-level build places example executables, together
with a copy of ``rsc``, under the ``examples`` subdirectory of the build tree.
The install target does not copy these source-built examples, and the release
package does not ship a separate prebuilt TCP viewer. For example:

.. code-block:: bash

    ./build-examples/examples/primitive_grid
    ./build-examples/examples/rayrai_tcp_viewer

On Windows, CMake places the example executables under ``<build-dir>/bin``,
together with a copy of ``rsc`` and the runtime DLLs.

The package example tree documented under :doc:`Examples` contains grouped
source directories such as ``src/server``, ``src/rayrai``, ``src/worlds``,
``src/xml``, and ``src/benchmark``. Target names, not source directory names, are the stable
user-facing interface.

Installed Package Layout
========================

The unpacked release contains separate package prefixes for RaiSim and rayrai:

.. code-block:: text

    <raisim2Lib>/raisim
    <raisim2Lib>/rayrai

Downstream projects use ``CMAKE_PREFIX_PATH`` to find these installed packages.
See :doc:`Installation` for environment setup and activation.

Where To Add New Things
=======================

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Change
     - Usual location
   * - New user-facing C++ example
     - ``examples/src`` plus a target registration in ``examples/CMakeLists.txt``.
   * - New example resource
     - ``rsc``. Put example-only resources under a relevant subdirectory there.
   * - New Python wrapper behavior
     - ``raisimPy``.
   * - New RL package behavior
     - ``raisimGymTorch``.
   * - New docs page
     - ``docs/sections`` and the relevant ``toctree`` in ``index.rst`` or a
       section index page.

Keep examples focused on demonstrating the public binary-package APIs.
