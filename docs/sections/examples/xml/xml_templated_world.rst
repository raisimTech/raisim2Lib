############################
XML Example: Templated World
############################

.. image:: ../../../../rsc/docs/image/xml_templated_world.png
   :alt: xml_templated_world example
   :width: 100%


Overview
========
Instantiates a templated XML world with parameter overrides (spawn options,
counts, offsets). Use it to see how a parameterized XML file can generate
variants of a scene without duplicating the XML. See
:doc:`../../WorldConfigurationFile`.

Target
======
CMake target: ``xml_templated_world``.

Run
====
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/xml_templated_world

On Windows, run ``xml_templated_world.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.


Details
=======
- Loads ``rsc/xmlScripts/templatedWorld/templatedWorld.xml`` and passes the
  parameter overrides to the ``raisim::World`` constructor.
- Uses ``World::ParameterContainer`` entries to set spawn flags, the sphere
  count, the sphere height offset, the Laikago start position, and the floor
  height.
- Runs the scene with RaisimServer for visualization.

