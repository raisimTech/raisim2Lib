#########################
XML Example: World Loader
#########################

Overview
========
Loads a world directly from an XML file passed on the command line and runs a RaisimServer session. It also records a video, making it a reference for XML-driven scene loading and logging.

Screenshot
==========
.. image:: ../../../image/heightMapUsingPNG.gif

Binary
======
CMake target and executable name: ``xml_world_loader``.

Run
====
Build and run from your build directory:

.. code-block:: bash

   cmake --build . --target xml_world_loader
   ./xml_world_loader /full/path/to/world.xml

On Windows, run ``xml_world_loader.exe`` instead.
The XML loader expects an absolute path to the world file.
This example uses RaisimServer. Start a visualizer client (RaisimUnity, RaisimUnreal, or the rayrai TCP viewer) and connect to port 8080.


Details
=======
- Loads a world directly from an XML path provided on the command line.
- Starts RaisimServer and records a video output file.
- Useful as a reference for XML-driven world creation.

Source
======
.. literalinclude:: ../../../../examples/src/xml/xml_world_loader.cpp
   :language: cpp
