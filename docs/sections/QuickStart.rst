#############################
Quick Start
#############################

This page takes a ``raisim2Lib`` checkout to a running example. For package
layout, dependencies, environment variables, and activation details, see
:doc:`Installation`.

1. Install and activate RaiSim
==============================

Clone ``raisim2Lib`` and place the activation file at the default location:

.. code-block:: text

    Linux/macOS: $HOME/.raisim/activation.raisim
    Windows:     C:\Users\<YOUR-USERNAME>\.raisim\activation.raisim

Build the examples. The first configure downloads the RaiSim and rayrai binary
packages that match the checkout into ``raisim/`` and ``rayrai/``:

.. code-block:: bash

    git clone https://github.com/raisimTech/raisim2Lib.git
    cd raisim2Lib
    cmake -S . -B build-examples \
      -DCMAKE_BUILD_TYPE=Release \
      -DRAISIM_EXAMPLE=ON
    cmake --build build-examples --parallel 12
    source ./raisim_env.sh

``raisim_env.sh`` adds the RaiSim and rayrai library directories to the library
search path. It only adds directories that exist, so source it after the first
configure has downloaded the packages.

On Linux and macOS, example executables are then under
``build-examples/examples``. On Windows, configure without
``CMAKE_BUILD_TYPE`` and build with ``--config Release``; executables are under
``build-examples\bin``, next to copies of the runtime DLLs.

2. Run a server-based example
=============================

Start the source-built TCP viewer in one terminal:

.. code-block:: bash

    ./build-examples/examples/rayrai_tcp_viewer

Run a server example in another terminal:

.. code-block:: bash

    ./build-examples/examples/primitive_grid

``primitive_grid`` and the other server examples publish their world through a
``raisim::RaisimServer``, which listens on port ``8080`` by default. On first
launch the viewer connects to ``127.0.0.1:8080``; if an application uses another
host or port, pass ``--connect host:port`` to the viewer or edit the endpoint in
its **Connection** tab (see :doc:`RayraiTcpViewer`).

3. Run an in-process rayrai example
===================================

Rayrai examples create their own window or offscreen OpenGL context and do not
need the TCP viewer:

.. code-block:: bash

    ./build-examples/examples/rayrai_pbr_material_grid
    ./build-examples/examples/rayrai_visual_asset_support

4. Run a non-visual example
===========================

.. code-block:: bash

    ./build-examples/examples/model_asset_pipeline

5. Run an OpenUSD mesh-loading example
======================================

As in step 2, start the TCP viewer in one terminal and the ShadowHand USD
example in another:

.. code-block:: bash

    ./build-examples/examples/rayrai_tcp_viewer
    ./build-examples/examples/shadow_hand_usd_cube

Next steps
==========

* Use :doc:`Examples` to choose a target by feature.
* Use :doc:`OpenUSD` for USD mesh-loading scope, runtime layout, and
  troubleshooting.
* Use :doc:`Visualization` to choose between the TCP viewer and in-process
  rayrai.
* Use :doc:`Troubleshooting` for common runtime, viewer, and activation issues.
* Use :doc:`Rayrai` for renderer controls, offscreen rendering, and the TCP
  viewer.
