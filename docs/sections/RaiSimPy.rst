#############################
RaisimPy
#############################

RaisimPy is the nanobind-based Python wrapper built from the public
``raisim2Lib/raisimPy`` sources. It follows the C++ RaiSim API closely while
using NumPy arrays for vectors and matrices. The wrapper sources are MIT
licensed; the RaiSim binary they use remains subject to the RaiSim license.

Build
=====

Build the wrapper with the same Python interpreter that will import it:

.. code-block:: bash

    cd /path/to/raisim2Lib
    source ./raisim_env.sh
    cmake -S . -B build-py \
      -DCMAKE_BUILD_TYPE=Release \
      -DRAISIM_EXAMPLE=OFF \
      -DRAISIM_PY=ON \
      -DPython_EXECUTABLE="$(command -v python)"
    cmake --build build-py --parallel 12

On Linux and macOS, the extension is emitted under ``build-py/raisimPy``. Run
an example directly from the build tree with:

.. code-block:: bash

    PYTHONPATH="$PWD/build-py/raisimPy${PYTHONPATH:+:$PYTHONPATH}" \
      python raisimPy/examples/robots.py

On Windows, select the intended configuration explicitly with
``cmake --build build-py --config Release --parallel 12`` and add the directory
containing ``raisimpy.pyd`` to ``PYTHONPATH``. Use the same Debug/Release
configuration as the imported RaiSim binary selected by CMake.

Activation And Minimal Use
==========================

``World.setActivationKey`` is the preferred spelling. ``World.setLicenseFile``
remains as a compatibility alias.

.. code-block:: python

    import time
    import raisimpy as raisim

    raisim.World.setActivationKey("/absolute/path/to/activation.raisim")

    world = raisim.World()
    world.setTimeStep(0.001)
    world.addGround()
    ball = world.addSphere(0.2, 1.0)
    ball.setPosition(0.0, 0.0, 1.0)

    server = raisim.RaisimServer(world)
    server.launchServer(8080)
    for _ in range(10_000):
        server.integrateWorldThreadSafe()
        time.sleep(world.getTimeStep())  # run at roughly real time
    server.killServer()

Start ``build-examples/examples/rayrai_tcp_viewer`` in another terminal to view
this server-based example.

Examples And API Coverage
=========================

The examples under ``raisimPy/examples`` cover articulated robots, joint
springs, heightmaps, constraints, ray-cast scans, charts, point clouds,
instanced visuals, dynamic visual meshes, and server-side visual objects. The
Python method names normally match C++, but the binding surface is defined by
the files under ``raisimPy/src``; a C++ method that is not bound there is not
available in Python.

NumPy arrays passed to pose, state, force, and geometry methods must have the
shape and dtype expected by that binding. Keep referenced RaiSim objects alive
for as long as Python wrappers returned by the world or server are in use.
