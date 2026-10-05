#####################################
Rayrai Example: Visual Asset Support
#####################################

.. image:: ../../../../rsc/docs/image/rayrai_visual_asset_support.png
   :alt: rayrai_visual_asset_support example
   :width: 100%

Overview
========
Shows visual asset loading on realistic URDF assets with image textures. The
example loads the textured ANYmal C URDF and several YCB object URDFs whose OBJ
or DAE visual meshes reference texture image files, then renders them in rayrai.

Target
======
CMake target: ``rayrai_visual_asset_support``.

Run
===
Run the build-tree executable:

.. code-block:: bash

   ./build-examples/examples/rayrai_visual_asset_support

On Windows, run ``rayrai_visual_asset_support.exe`` instead.
This example renders in process with rayrai and does not need ``rayrai_tcp_viewer``.

Details
=======
- Loads ``rsc/anymal_c/urdf/anymal.urdf`` and four YCB object URDFs from
  ``rsc/ycb``.
- Loads the textured visual meshes referenced from URDF ``visual`` tags.
- Keeps visual assets and collision assets separate through URDF ``visual``
  and ``collision`` tags.
- Renders image textures from packaged mesh assets such as
  ``rsc/anymal_c/meshes/*.jpg`` and ``rsc/ycb/*/google_16k/texture_map.png``.
- Uses renderer-facing mesh assets for rayrai while RaiSim continues to use
  the collision geometry defined by the URDF.
