#########################
XML Example: World Loader
#########################

Overview
========
Loads a world XML file and publishes it through ``RaisimServer``. With no
argument it loads ``objects/SingleBodies.xml``; pass another file to load it
instead. See :doc:`../../WorldConfigurationFile` for the XML format.

Screenshot
==========
.. image:: ../../../../rsc/docs/image/heightMapUsingPNG.gif
   :alt: xml_world_loader running heightMaps/heightMapUsingPng.xml

The screenshot shows ``heightMaps/heightMapUsingPng.xml``.

Target
======
CMake target: ``xml_world_loader``.

Run
===
Run the build-tree executable, optionally with an XML file:

.. code-block:: bash

   ./build-examples/examples/xml_world_loader
   ./build-examples/examples/xml_world_loader heightMaps/heightMapUsingPng.xml
   ./build-examples/examples/xml_world_loader /absolute/path/to/world.xml

On Windows, run ``xml_world_loader.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.
``--help`` prints the usage.

Details
=======
- Uses the argument as given if that file exists; otherwise looks for it under
  the ``rsc/xmlScripts`` copy next to the executable. It exits with an error if
  neither exists.
- The bundled worlds under ``rsc/xmlScripts`` include ``objects/``,
  ``heightMaps/``, ``material/``, ``templatedWorld/``, and ``wire/`` examples.
- Constructs ``raisim::World`` from the resolved path and publishes it through
  ``RaisimServer``.

For application code that already has a path, construct the world directly:

.. code-block:: cpp

   raisim::World world("/absolute/path/to/world.xml");
