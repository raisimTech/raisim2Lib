###########################
Server Example: YCB Objects
###########################

.. image:: ../../../../rsc/docs/image/ycb_objects.png
   :alt: ycb_objects example
   :width: 100%

Overview
========
Loads four YCB objects from their URDF files and drops them on a ground plane.
It is a small reference for loading mesh-based objects described by URDF.

Target
======
CMake target: ``ycb_objects``.

Run
===
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/ycb_objects

On Windows, run ``ycb_objects.exe`` instead.
This example uses RaisimServer. Start ``rayrai_tcp_viewer`` and connect to port 8080.

Details
=======
- Loads ``002_master_chef_can.urdf``, ``007_tuna_fish_can.urdf``,
  ``012_strawberry.urdf``, and ``013_apple.urdf`` from ``rsc/ycb`` as
  articulated systems and places them in a row.
- Focuses the viewer on the first object.

