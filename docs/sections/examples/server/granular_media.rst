##############################
Server Example: Granular Media
##############################

.. image:: ../../../../rsc/docs/image/granular_media.png
   :alt: granular_media example
   :width: 100%

Overview
========
Stands a PD-controlled ANYmal on a native granular bed created with
``World::addGranularParticles`` (``raisim::GranularSystem``). The grains are
streamed to the viewer as instanced sphere visuals. See :doc:`../../GranularMedia`.

Target
======
CMake target: ``granular_media``. The source is
``examples/src/benchmark/granular_media.cpp``.

Run
===
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/granular_media

On Windows, run ``granular_media.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port
8080. Without ``--steps``, the simulation starts once a viewer connects and
runs until you stop it.

Options
=======
* ``--steps N`` runs a fixed number of steps without waiting for a viewer,
  then prints the time per step and averaged contact statistics. The exit code
  is 2 if the bed became unstable (non-finite positions or grains faster than
  ``--max-allowed-speed``).
* ``--port=N`` changes the server port.
* ``--resolution``, ``--layers``, ``--radius``, ``--spacing``,
  ``--bed-length``, and ``--bed-width`` set the bed geometry; ``--substeps``,
  ``--stiffness``, ``--damping``, ``--tangential-stiffness``,
  ``--tangential-damping``, ``--friction``, ``--rolling-friction``, and
  ``--hertz`` set the granular contact model.
* ``--settle-steps`` and ``--robot-settle-steps`` set how many steps run
  before the server starts, to let the bed and then the robot settle.

