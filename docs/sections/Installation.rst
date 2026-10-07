#############################
Installation
#############################

What This Page Covers
=====================

This page covers installed package paths, environment variables, build commands
for examples and ``raisimPy``, dependencies, and license activation for the
binary RaiSim distribution. Get it from
``https://github.com/raisimTech/raisim2Lib``: the repository provides example,
wrapper, and rayrai viewer sources, resources, and documentation, and its
release archives provide the prebuilt RaiSim and rayrai headers and libraries.

Dependencies
============

Minimum requirements:

* A supported 64-bit operating system.
* CMake 3.18 or newer for the top-level package workspace.
* A C++20-capable compiler. The exported ``raisim::raisim`` CMake target
  requires C++20, for the examples as well as for your own applications.
* Eigen3. On Linux and macOS, install it with the system package manager
  (for example ``libeigen3-dev`` on Ubuntu); Windows builds use the copy in
  ``thirdParty/Eigen3``.
* SDL2 with its CMake package files, required by ``find_package(rayrai)`` and
  therefore by the examples, unless it is bundled with the rayrai package. On
  Linux and macOS, install the development package (for example
  ``libsdl2-dev`` on Ubuntu); on Windows, install ``sdl2`` with vcpkg. The
  top-level CMake configure adds the vcpkg tree from ``VCPKG_ROOT`` (or
  ``C:\vcpkg``) to the search path.
* OpenGL 3.3 core profile or newer for rayrai. OpenGL 4.3 enables the full
  renderer; macOS provides OpenGL 4.1 and runs a reduced feature set (see
  :ref:`rayrai-platform-support`).
* Visual Studio 2019 or newer on Windows.

The binary package includes RaiSim and rayrai libraries and headers. See the
license files distributed with the package for third-party license details.

Download
========

Clone the public repository to obtain the example and wrapper sources, runtime
resources, environment scripts, and documentation:

.. code-block:: bash

    git clone https://github.com/raisimTech/raisim2Lib.git
    cd raisim2Lib

Each checkout pins one RaiSim release (``RAISIM_VERSION`` in the top-level
``CMakeLists.txt``). The top-level CMake configure downloads that release into
``raisim/`` and ``rayrai/`` when they are missing or hold a different version.
For a manual installation, download the asset for your OS and CPU architecture
from the repository's Releases page, then copy its ``raisim`` and ``rayrai``
directories together into this checkout. Release archives contain compiled
packages; the public Git checkout supplies examples, wrappers, and ``rsc``.

``./raisim_upgrade.sh [version]`` (Linux and macOS) and ``raisim_upgrade.ps1``
(Windows) download a release asset and replace ``raisim/`` and ``rayrai/``.
Without a version they offer the latest release; pass ``-y`` (``-Yes`` for
the PowerShell script) to skip the prompts. Install the version that the checkout pins: the next CMake configure
replaces any other version with the pinned one.

Current asset names are ``linux-x86-<version>.zip`` for Linux x86-64,
``linux-arm64-<version>.zip`` for Linux ARM64,
``macos-arm64-<version>.zip`` for Apple Silicon,
``macos-x86_64-<version>.zip`` for Intel Macs when available, and
``windows-x86-<version>.zip`` for Windows x86-64.

Local Install Layout
====================

The unpacked release is already a usable local package tree. Its current flat
layout is:

.. code-block:: text

    <raisim2Lib>/raisim
    <raisim2Lib>/rayrai

RaiSim and rayrai are installed as CMake packages. Downstream projects that use
only physics should point ``CMAKE_PREFIX_PATH`` at the RaiSim package prefix.
Projects that use rayrai should include both prefixes.

Older releases also supported architecture subdirectories such as
``raisim/linux`` and ``rayrai/linux``. The current CMake and environment scripts
still recognize that legacy layout, but new documentation and release archives
use the flat prefixes above.

Environment Setup
=================

.. tabs::
  .. group-tab:: Linux

    .. code-block:: bash

        cd $HOME/raisim2Lib
        source ./raisim_env.sh

    ``raisim_env.sh`` must be sourced, not executed. It sets ``RAISIM_DIR``
    and ``RAISIM_OS`` and adds the matching RaiSim and rayrai library
    directories to ``LD_LIBRARY_PATH``. Current flat packages use
    ``raisim/lib`` and ``rayrai/lib``; legacy packages use the
    ``$RAISIM_OS/lib`` directories below those prefixes. It only adds
    directories that exist, so source it again after the first configure has
    downloaded the packages. Configure and build this public workspace with
    examples and ``raisimPy`` enabled:

    .. code-block:: bash

        cmake -S . -B build \
          -DCMAKE_BUILD_TYPE=Release \
          -DRAISIM_EXAMPLE=ON \
          -DRAISIM_PY=ON
        cmake --build build --parallel 12

    ``RAISIM_EXAMPLE`` is enabled by default. ``RAISIM_PY`` must be enabled
    explicitly when you want the Python wrapper.

  .. group-tab:: macOS

    .. code-block:: bash

        cd $HOME/raisim2Lib
        source ./raisim_env.sh

    ``raisim_env.sh`` adds the matching RaiSim and rayrai library directories
    to ``DYLD_LIBRARY_PATH``. It selects the legacy ``$RAISIM_OS/lib``
    directories when present and otherwise uses the current flat ``lib``
    directories. Configure and build this public workspace in Release mode:

    .. code-block:: bash

        cmake -S . -B build \
          -DCMAKE_BUILD_TYPE=Release \
          -DRAISIM_EXAMPLE=ON
        cmake --build build --parallel 12

    ``RAISIM_EXAMPLE`` is enabled by default. If ``raisim/`` is missing or its
    version differs from ``RAISIM_VERSION``, CMake downloads the pinned release
    asset and replaces both ``raisim/`` and ``rayrai/``. To replace the
    packages explicitly, use ``raisim_upgrade.sh`` as described in
    `Download`_.

    Apple Silicon hosts select ``macos-arm64``, even when CMake or the shell is
    running under Rosetta. Intel Mac builds select ``macos-x86_64`` when that
    release asset is available. ``RAISIM_PY`` must be enabled explicitly when
    you want the Python wrapper.

  .. group-tab:: Windows

    In PowerShell, dot-source ``raisim_env.ps1`` (``. .\raisim_env.ps1``); in
    ``cmd.exe``, run ``raisim_env.bat``. Both add the RaiSim and rayrai ``bin``
    and ``lib`` directories, and vcpkg's ``bin`` directories when vcpkg is
    found, to ``Path`` for the current session. Running ``raisim_env.bat`` from
    PowerShell does not change the PowerShell session. The example build also
    copies the runtime DLLs next to the example executables. Configure and
    build this public workspace with examples and ``raisimPy`` enabled:

    .. code-block:: powershell

        cmake -S . -B build -DRAISIM_EXAMPLE=ON -DRAISIM_PY=ON
        cmake --build build --config Release --parallel 12

Build And Install
=================

Use this command sequence when you want examples and ``raisimPy`` from the local
``raisim2Lib`` tree:

.. code-block:: bash

    cd $HOME/raisim2Lib
    source ./raisim_env.sh
    cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DRAISIM_EXAMPLE=ON \
      -DRAISIM_PY=ON
    cmake --build build --parallel 12

On Windows, omit ``CMAKE_BUILD_TYPE`` and pass ``--config Release`` to the
build command instead, as shown in `Environment Setup`_. If ``raisim/`` is
missing or its version differs from ``RAISIM_VERSION``, the configure step
downloads the pinned package for the host platform.

Installation is optional for running examples from the build tree. The install
step copies package headers, libraries, and CMake files; it does not install the
example executables built from ``examples/``, including ``rayrai_tcp_viewer``.
If you do install, choose a prefix you can write to instead of relying on
CMake's default ``/usr/local``:

.. code-block:: bash

    cmake --install build --prefix $HOME/raisim2Lib/install

For downstream C++ projects that use both RaiSim and rayrai, configure with both
package prefixes from the unpacked ``raisim2Lib`` tree:

.. code-block:: bash

    export RAISIM_ROOT=$HOME/raisim2Lib
    cmake -S . -B build -DCMAKE_PREFIX_PATH="$RAISIM_ROOT/raisim;$RAISIM_ROOT/rayrai"
    cmake --build build --parallel 12

Activation Key
==============

Rename the activation key received by email to ``activation.raisim`` and place
it in:

* Linux and macOS: ``$HOME/.raisim/activation.raisim``
* Windows: ``C:\Users\<YOUR-USERNAME>\.raisim\activation.raisim``

RaiSim first checks the path passed to
``raisim::World::setActivationKey()``. If that file is not found, it falls back
to the user-directory location above (``$HOME`` on Linux and macOS,
``%USERPROFILE%`` on Windows) and then to ``raisim/activation.raisim`` in the
same user directory. The user directory is also taken from the operating
system's account records, so a process started without ``$HOME``, for example
by a service manager or a launcher that clears the environment, still finds the
key. If the current user has no key, RaiSim checks the same locations in the
other users' home directories on the machine and prints which key it uses. If
no key is found, RaiSim prints the checked paths and your machine id and stops
with a fatal error. RaiSim locates and validates the
key once per process, when the first ``raisim::World`` is constructed, so call
``setActivationKey()`` before creating any world.

Rayrai
======

rayrai is the supported visualizer for current RaiSim. There are two usage
modes:

* The source-built ``rayrai_tcp_viewer`` connects to applications that publish
  a ``raisim::World`` through ``raisim::RaisimServer``.
* In-process rayrai examples create ``raisin::RayraiWindow`` directly for
  screenshots, RGB/depth rendering, PBR assets, HDR lighting, and offscreen
  workflows.

RaisimUnity and RaisimUnreal are legacy integrations and are no longer the
supported visualization path.

Blender and glTF Assets
=======================

For Blender-authored scenes, use the general exporter:

.. code-block:: bash

    blender --background scene.blend \
      --python $HOME/raisim2Lib/scripts/export_blender_scene.py \
      -- --format glb --output /path/to/scene.glb

The exporter writes renderable glTF/GLB scenes (``--format`` also accepts
``gltf`` and ``obj``), keeps Z-up coordinates for RaiSim/rayrai, and emits
rayrai sidecars for Blender area lights and authored reflection probes next to
the output file, e.g. ``scene.glb.rayrai_lights.json`` and
``scene.glb.rayrai_probes.json``.
