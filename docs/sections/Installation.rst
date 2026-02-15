#############################
Installation
#############################

Setup
========

Local installation is recommended to prevent conflicts with other libraries.
The following variables are used throughout this guide:

* ``RAISIM_DIR``: The directory where the raisim2Lib repository has been cloned.
* ``LOCAL_INSTALL``: The installation directory for exported CMake packages (e.g., ~/raisim_build).

These may be exported as environment variables or replaced with absolute paths in the subsequent commands.

Dependencies
============

Ensure the following packages are installed prior to using RaiSim:

* Eigen
* CMake 3.15 or newer
* For Windows users: Visual Studio 2019 (support is officially for 2019, though newer versions may be compatible). **Ensure the C++ workload is installed during setup.**

RaiSim incorporates several open-source libraries. Please refer to the COPYING file for a complete list.
The source code for Eigen and nanobind is provided in the ``thirdParty`` directory.

RaiSim Installation
===================

Clone the raisim2Lib repository from https://github.com/raisimTech/raisim2Lib.

The cloned repository includes prebuilt RaiSim packages.
The installed CMake packages can be found under ``raisim2Lib/raisim/<OS-TYPE>``.

To integrate RaiSim into your project, add the CMake package path and the shared library path to the corresponding environment variables as follows:


.. tabs::
  .. group-tab:: Linux
    Add the following lines to your ``~/.bashrc`` file:

    .. code-block:: bash

        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WORKSPACE/raisim/linux/lib
        export PYTHONPATH=$PYTHONPATH:$WORKSPACE/raisim/linux/lib

    This configuration enables the Linux linker to locate the RaiSim shared libraries.
    To allow your project's CMake configuration to locate RaiSim, pass ``-DCMAKE_PREFIX_PATH=$WORKSPACE/raisim/linux``.

  .. group-tab:: Mac
    Add the following lines to your ``.zshrc`` file:

    .. code-block:: bash

        export DYLD_LIBRARY_PATH=DYLD_LIBRARY_PATH:$WORKSPACE/raisim/mac/lib
        export PYTHONPATH=$PYTHONPATH:$WORKSPACE/raisim/mac/lib

    For Apple silicon Macs, utilize the ``m1`` directory instead of ``mac``.
    Avoid using ``~`` to reference the home directory; explicitly state the full path.
    This configuration enables the macOS linker to locate the RaiSim shared libraries.
    To allow your project's CMake configuration to locate RaiSim, pass ``-DCMAKE_PREFIX_PATH=$WORKSPACE/raisim/mac``.
    As noted, utilize ``m1`` for Apple silicon architectures.


  .. group-tab:: Windows
    Add the win32 CMake package directory to your `Path` environment variable.
    To achieve this, follow these steps:

    1. Open "Edit the system environment variables".
    2. In the "Advanced" tab, select "Environment Variables".
    3. Select the "Path" variable in the "System variables" list and click "Edit".
    4. Append ``$WORKSPACE/raisim/win32/``.
    5. To inform your project of RaiSim's location, pass ``-DCMAKE_PREFIX_PATH=$WORKSPACE/raisim/win32``. (Note: Release and debug builds share a single directory.)

Rayrai Visualizer (In-Process)
================================

The Rayrai visualizer is located in the ``visualizer/`` directory and is built as a standalone CMake project.
It depends on the installed RaiSim package (and its Assimp dependency); therefore, install RaiSim first,
then set ``CMAKE_PREFIX_PATH`` to point to the RaiSim installation directory.

Rayrai Dependencies
--------------------

* OpenGL development headers and libraries

On Ubuntu, this typically requires the OpenGL packages (e.g.,
``libgl1-mesa-dev`` and ``libglu1-mesa-dev``).

Build Steps
-------------

.. tabs::
  .. group-tab:: Linux or Mac

    .. code-block:: bash

        cd $RAISIM_DIR
        cmake -S visualizer -B build-visualizer \
          -DCMAKE_PREFIX_PATH=$RAISIM_DIR/raisim/linux
        cmake --build build-visualizer
        cmake --install build-visualizer --prefix $RAISIM_DIR/rayrai/linux

  .. group-tab:: Windows

    SDL is bundled with rayrai on Windows, so no separate SDL2 installation is
    required. Configure the visualizer:

    .. code-block:: powershell

        cmake -S visualizer -B build-visualizer ^
          -DCMAKE_PREFIX_PATH=C:\path\to\raisim2Lib\raisim\win32
        cmake --build build-visualizer --config Release
        cmake --install build-visualizer --config Release --prefix C:\path\to\raisim2Lib\rayrai\win32


Building RaiSim Examples and Installing RaisimPy
====================================================

Use the ``CMakeLists.txt`` file in the ``raisim2Lib`` directory to build RaiSim examples and other modules.
The following options are available:

* ``RAISIM_EXAMPLE`` : Compile C++ RaiSim examples.
* ``RAISIM_MATLAB`` : Compile RaisimMatlab (a compiled binary is also provided). This option requires MATLAB.
* ``RAISIM_PY`` : Compile RaisimPy. Upon installation, RaisimPy files are copied to the Python site-packages directory.

Build files can be generated using CMake as follows.

To build RaisimPy for the specific Python version, activate the corresponding conda environment before invoking CMake.

.. tabs::
  .. group-tab:: Linux or Mac

    .. code-block:: bash

      cd $WORKSPACE
      mkdir build
      cd build
      cmake .. -DCMAKE_INSTALL_PREFIX=$LOCAL_INSTALL -DRAISIM_EXAMPLE=ON -DRAISIM_PY=ON -DPYTHON_EXECUTABLE=$(which python)
      make install -j4

  .. group-tab:: Windows

    Using the CMake GUI is recommended.
    After generating the build files with the CMake GUI, RaiSim can be built and installed using Visual Studio.
    Alternatively, use the following commands in Windows PowerShell:

    .. code-block::

      cd $WORKSPACE
      mkdir build
      cd build
      cmake .. -DCMAKE_INSTALL_PREFIX=$LOCAL_INSTALL -DRAISIM_EXAMPLE=ON -DRAISIM_PY=ON -DPYTHON_EXECUTABLE=$(which python)
      cmake --build . --target install --config Release


.. note::
    **For Linux Users**
    To utilize the Vulkan version of RaiSimUnity on Linux, install ``minizip``, ``ffmpeg``, and
    ``vulkan``. If RaisimGym usage is planned, install the GPU driver version recommended by
    PyTorch. Otherwise, install the latest available driver.

    To install ``minizip`` and ``ffmpeg``:

    .. code-block:: bash

        sudo apt install minizip ffmpeg

    Vulkan installation varies by OS distribution; refer to the specific distribution's documentation.

    Ubuntu 22.04 users must execute the following command prior to running RaiSimUnity:

    .. code-block:: bash

        sudo ln -s /usr/lib/x86_64-linux-gnu/libdl.so.2 /usr/lib/x86_64-linux-gnu/libdl.so

    For other Ubuntu versions, this symlink is automatically provided.

    If RaiSimUnity remains non-functional, the driver likely lacks Vulkan support.
    In such cases, use RaiSimUnityOpenGL, which provides minimal graphics capabilities.

Activation Key
================

Rename the activation key received via email to ``activation.raisim``.
Save this file in ``<YOUR-HOME-DIR>/.raisim``.
* Linux and macOS: ``/home/<YOUR-USERNAME>/.raisim``
* Windows: ``C:\Users\<YOUR-USERNAME>\.raisim`` (Ensure ``C`` matches your home drive).

RaiSim also checks the path specified by ``raisim::World::setActivationKey()``.
If the file is not found there, it will search the user directory where the ``activation.raisim`` file was saved.

Examples
===============

Built examples are stored in the ``examples`` directory.
On Windows, execute examples via PowerShell rather than by double-clicking the icons.
In the event of an installation failure, an error message will be displayed.

Run the RaiSimUnity executable located in ``raisimUnity/<OS>/RaiSimUnity`` prior to running the examples.
