#######################################
Server Example: MJCF Gymnasium Humanoid
#######################################

.. image:: ../../../../rsc/docs/image/mjcf_gymnasium_humanoid.png
   :alt: mjcf_gymnasium_humanoid example
   :width: 100%

Overview
========
Loads the Gymnasium Humanoid MuJoCo XML asset through ``raisim::World`` and
publishes it through ``raisim::RaisimServer``. The humanoid starts in a
nonzero joint configuration with its base about 1 m above the standing pose
and then drops under gravity with zero applied generalized force.

Target
======
CMake target: ``mjcf_gymnasium_humanoid``.

Run
===
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/mjcf_gymnasium_humanoid

On Windows, run ``mjcf_gymnasium_humanoid.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.

Details
=======
- Loads ``rsc/mjcf/gymnasium/humanoid.xml`` with the ``raisim::World``
  constructor and retrieves the articulated system by its root body name
  (``torso``).
- Handles a larger MJCF articulated system with a free root and many child
  bodies.
- Sets an initial free-root pose and a nonzero joint configuration before
  dropping the model under gravity.
