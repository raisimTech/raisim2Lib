##################################################
Server Example: Length Constraints Newton's Cradle
##################################################

Overview
========
Hangs a Newton's cradle from spatial tendons, attaches a box and two ANYmal
robots with spring and tension-driven tendons, and exports the world to XML.
Despite the target name, the example uses the tendon API
(``World::addSpatialTendon``); the older wire API now wraps tendons. See
:doc:`../../Tendons`.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/length_constraints_newtons_cradle.png
   :alt: length_constraints_newtons_cradle example
   :width: 100%

Target
======
CMake target: ``length_constraints_newtons_cradle``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/length_constraints_newtons_cradle

On Windows, run ``length_constraints_newtons_cradle.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.


Details
=======
- Hangs four steel balls from static pins with cable tendons that have a
  2 m upper length limit; the first cable is drawn wider and in red. The last
  ball starts pulled out to the side.
- Hangs a box and an ANYmal C from spring tendons (stiffness 200 and 1000).
- Lifts an ANYmal B with a tendon driven by ``Tendon::setTension`` and removes
  that tendon with ``World::removeTendon`` after 5000 loop iterations.
- Exports the world to ``exportedWorld.xml`` next to the executable.
- Integrates only while a viewer is connected.

