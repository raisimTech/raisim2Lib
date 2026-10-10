#############################
RaiSim |raisim_version_title|
#############################

.. list-table::
   :widths: 50 50
   :align: center
   :class: home-gallery

   * - .. image:: ../rsc/docs/image/rayrai_complete_showcase.gif
          :alt: rayrai_complete_showcase animated example
          :width: 100%
     - .. image:: ../rsc/docs/image/forest_pan.gif
          :alt: Camera panning across forest trees and crates in rayrai
          :width: 100%
   * - .. image:: ../rsc/docs/image/deformable_objects.gif
          :alt: deformable_objects animated example
          :width: 100%
     - .. image:: ../rsc/docs/image/procedural_heightmap.gif
          :alt: procedural_heightmap animated example
          :width: 100%
   * - .. image:: ../rsc/docs/image/rayrai/rayrai_blue_wall_scene.gif
          :alt: Camera orbiting the Blue Wall rayrai scene
          :width: 100%
     - .. image:: ../rsc/docs/image/rayrai/granular_media_kicking.gif
          :alt: ANYmal kicking simulated sand with noisy PD joint targets
          :width: 100%
   * - .. image:: ../rsc/docs/image/rayrai/tendon_pulleys.gif
          :alt: Tendon-driven loads moving around wrapped pulleys
          :width: 100%
     - .. image:: ../rsc/docs/image/rayrai/rayrai_nested_glass_showcase.gif
          :alt: Camera orbiting nested and overlapping glass solids
          :width: 100%
   * - .. image:: ../rsc/docs/image/rayrai/strandbeest_closed_loops.gif
          :alt: 12-legged Strandbeest walking, driven by one crank through 36 closed kinematic loops
          :width: 100%
          :target: sections/examples/server/strandbeest_closed_loops.html
     - .. image:: ../rsc/docs/image/rayrai/rayrai_city.gif
          :alt: Camera moving around ANYmal C standing on a photoreal city street next to roadworks
          :width: 100%
          :target: sections/examples/rayrai/rayrai_city.html

RaiSim is a cross-platform multi-body physics engine for robotics and AI. The
binary package provides rigid bodies, articulated systems, deformable bodies,
granular particles, rayrai RGB/depth sensor rendering, deterministic CPU ray
sensors for headless use, ``RaisimServer`` streaming, and the rayrai OpenGL
visualizer.

If you are new to RaiSim, start with the page that matches your goal:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Goal
     - Start here
   * - Install and run one example
     - :doc:`sections/QuickStart`
   * - Find the right example target
     - :doc:`sections/Examples`
   * - Visualize a running ``RaisimServer`` scene
     - :doc:`sections/Visualization`, :doc:`sections/RaisimServer`, and
       :doc:`sections/Rayrai`
   * - Add objects, contacts, sensors, or materials
     - :doc:`sections/WorldSystem`, :doc:`sections/Object`,
       :doc:`sections/Contact`, :doc:`sections/Sensors`, and
       :doc:`sections/MaterialSystem`
   * - Build examples, run timing examples, or tune a binary package scene
     - :doc:`sections/FeatureMap`, :doc:`sections/BuildAndTest`,
       :doc:`sections/ProjectLayout`, and :doc:`sections/Performance`

.. image:: ../rsc/docs/image/examples_overview.png
  :alt: Current RaiSim and rayrai example overview
  :width: 100%

.. toctree::
   :maxdepth: 1
   :caption: Get started

   sections/QuickStart
   sections/Installation
   sections/Visualization
   sections/Examples
   sections/FeatureMap
   sections/BuildAndTest
   sections/DebugTools
   sections/ProjectLayout
   sections/Performance
   sections/Benchmark
   sections/Troubleshooting
   sections/Changelog
   sections/License
   sections/Support
   sections/Acknowledgement

.. toctree::
   :maxdepth: 1
   :caption: RaiSim C++

   sections/Introduction
   sections/ConventionsAndNotations
   sections/Determinism
   sections/Math
   sections/LoggingSystem
   sections/WorldSystem
   sections/WorldConfigurationFile
   sections/RsceneFile
   sections/OpenUSD
   sections/RaisimServer
   sections/Object
   sections/ArticulatedSystem
   sections/SingleBodyObjects
   sections/DeformableObject
   sections/GranularMedia
   sections/Contact
   sections/CollisionDetection
   sections/MaterialSystem
   sections/HeightMap
   sections/Constraints
   sections/Tendons
   sections/RayTest

.. toctree::
   :maxdepth: 1
   :caption: Related software

   sections/Rayrai
   sections/RayraiTcpViewer
   sections/RaisimEngine2
   sections/RaisimGymTorch
   sections/RaiSimPy
   sections/RaiSimMatlab
   sections/LegacyIntegrations
