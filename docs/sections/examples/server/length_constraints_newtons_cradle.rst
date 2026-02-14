#################################################
Server Example: Length Constraints Newtons Cradle
#################################################

Overview
========
Creates a Newton's cradle with stiff and compliant wires, adds robots, and exports the world to XML. It is the main reference for length constraints and wire APIs.

Screenshot
==========
.. image:: ../../../image/constraints.gif

Binary
======
CMake target and executable name: ``length_constraints_newtons_cradle``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target length_constraints_newtons_cradle
   ./length_constraints_newtons_cradle

On Windows, run ``length_constraints_newtons_cradle.exe`` instead.
This example uses RaisimServer. Start a visualizer client (RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) and connect to port 8080.


Details
=======
- Builds a Newton's cradle with stiff wires and steel balls.
- Adds compliant and custom wires attached to a box and robots.
- Exports the world to XML and removes a custom wire mid-simulation.

Source
======
.. literalinclude:: ../../../../examples/src/server/length_constraints_newtons_cradle.cpp
   :language: cpp
