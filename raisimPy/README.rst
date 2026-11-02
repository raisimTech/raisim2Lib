RaiSimPy: Python wrapper for RaiSim
===================================

Build and test
~~~~~~~~~~~~~~

From the repository root, with the intended Python environment active::

    source ./raisim_env.sh
    cmake -S . -B build-py -DCMAKE_BUILD_TYPE=Release -DRAISIM_PY=ON -DRAISIM_EXAMPLE=OFF -DPython_EXECUTABLE="$(command -v python)"
    cmake --build build-py --target raisimpy --parallel 4
    RAISIM_ACTIVATION_KEY="$HOME/.raisim/activation.raisim" ctest --test-dir build-py -R raisimpy_runtime --output-on-failure
    python raisimPy/examples/robots.py --headless --steps 200

``raisim_env.sh`` puts ``build-py/raisimPy`` (or ``$RAISIMPY_BUILD_DIR``) on
``PYTHONPATH``. To import ``raisimpy`` without it, run
``cmake --install build-py/raisimPy`` to install into the configured Python's
site-packages; repeat after each rebuild.

NumPy is required. Use a fresh build directory after a package version change.
The extension reports the linked engine version as ``raisimpy.__version__``.
``raisimpy.hasWrenchSensors`` indicates whether the package supports the optional
force/tactile sensor bindings.

All examples accept ``--steps``, ``--headless``, ``--no-sleep``, ``--port``, and
``--activation-key``. Without an explicit key, they try ``RAISIM_ACTIVATION_KEY``,
the repository's ``rsc/activation.raisim``, then ``~/.raisim/activation.raisim``.

The wrapper includes IK, actuators/tracks, deformable and granular objects,
deformable terrain, checkpoints/snapshots, USD import, sleeping, collision
settings and viewer controls. See ``docs/sections/RaiSimPy.md`` for usage and
the ownership/array conventions.

In-process rendering
~~~~~~~~~~~~~~~~~~~~

The default build also produces ``raisimpy_rayrai``. Use it through
``raisimpy.rayrai`` or ``raisimpy.RayraiWindow``. It includes managed contexts,
cameras and NumPy RGB/depth capture, visual/mesh/material/light APIs, render
settings, water/weather/reflections, imports, snapshots and diagnostics.
``raisimpy.hasRayrai`` reports build support. Disable it with
``-DRAISIM_PY_RAYRAI=OFF`` for physics-only installations. Use and close a
viewer on its creating thread; use a ``with`` block for deterministic cleanup.
Create visible viewers on Python's main thread (all viewers on macOS).
Explicit viewer locking uses ``lockVisualizationServerMutex``; do not render
while holding a direct ``World.lock()``.

See ``docs/sections/RaiSimPy.md`` for the API guide and coverage inventory.
Run the capture example with a hidden OpenGL context::

    python raisimPy/examples/rayrai_render.py --headless --steps 200 --output image.png

Set ``RAISIMPY_TEST_RAYRAI=1`` to enable actual rendering tests. On Linux,
``SDL_VIDEODRIVER=offscreen`` can provide a display-free Mesa GL context.
Checked-in Python stubs are under ``typings``; maintainers regenerate the
C++ bindings with ``python raisimPy/tools/generate_rayrai.py``.

World examples
~~~~~~~~~~~~~~

``city.py``, ``warehouse.py``, and ``forest.py`` load the corresponding scenes
under ``rsc``. City and warehouse include a standing ANYmal C. They open an
in-process Rayrai window when built; pass ``--renderer tcp`` to use
``build-examples/examples/rayrai_tcp_viewer`` in another terminal.

``buggy.py`` runs the native 60-DoF buggy URDF on its race-track heightmap.
Drive with the viewer's Throttle, Steering and Brake sliders, or set their
initial values with ``--throttle``, ``--steer`` and ``--brake``; ``--reverse``
selects reverse drive. For a bounded simulation without a viewer::

    python raisimPy/examples/buggy.py --headless --steps 3000 --throttle 0.35

LICENSE
~~~~~~~

All files in this folder are distributed under the `MIT <https://choosealicense.com/licenses/mit/>`_ License,
however the RaiSim software is under the End-User License Agreement that you can find 
`here <raisim.com>`_.



