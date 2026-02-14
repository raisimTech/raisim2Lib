##################################
Server Example: Kinematic Platform
##################################

Overview
========
Creates a kinematic ground platform that moves sinusoidally under an ANYmal. It demonstrates kinematic bodies and their interaction with dynamic robots.

Screenshot
==========
.. image:: ../../../image/kinematic_platform.png

Binary
======
CMake target and executable name: ``kinematic_platform``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target kinematic_platform
   ./kinematic_platform

On Windows, run ``kinematic_platform.exe`` instead.
This example uses RaisimServer. Start a visualizer client (RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) and connect to port 8080.


Details
=======
- Creates a kinematic ground platform (infinite mass) and moves it sinusoidally.
- Places ANYmal on top with PD posture control.
- Demonstrates ``BodyType::KINEMATIC`` and prescribed motion.

Source
======
.. literalinclude:: ../../../../examples/src/server/kinematic_platform.cpp
   :language: cpp
