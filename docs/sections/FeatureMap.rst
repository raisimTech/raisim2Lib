#############################
Feature Map
#############################

Use this page when you know what you want to build, but not which RaiSim API or
documentation page owns it. For runnable programs, start with :doc:`Examples`.
For build, test, benchmark, and documentation commands, use :doc:`BuildAndTest`.

World and bodies
================

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Goal
     - Main API
     - Documentation
   * - Create and step a simulation
     - ``raisim::World``, ``World::integrate``
     - :doc:`WorldSystem`, :doc:`QuickStart`
   * - Add primitive rigid bodies
     - ``World::addGround``, ``addSphere``, ``addBox``, ``addCylinder``,
       ``addCapsule``
     - :doc:`SingleBodyObjects`
   * - Add a robot from URDF or MJCF
     - ``World::addArticulatedSystem``
     - :doc:`ArticulatedSystem`
   * - Model geared or coupled actuators whose torques are limited to their motor operating regions
     - URDF ``<actuator>``, ``ArticulatedSystem::setActuatorTorques``,
       ``getActuatorTorqueBounds``, ``getActuatorStates``, ``setBusVoltage``
     - :doc:`Actuators`
   * - Add joint friction and damping
     - URDF ``<dynamics damping friction>``, ``ArticulatedSystem::setJointDamping``
     - :doc:`articulated_system/JointDampingAndFriction`
   * - Control joints with the built-in PD controller or a generalized force
     - ``ArticulatedSystem::setPdGains``, ``setPdTarget``, ``setGeneralizedForce``
     - :doc:`articulated_system/DynamicsAndControl`
   * - Add mesh collision geometry
     - ``World::addMesh``, mesh preprocessing options, mesh collision modes
     - :doc:`SingleBodyObjects`, :doc:`Examples`
   * - Add terrain
     - ``World::addHeightMap``, ``raisim::HeightMap``
     - :doc:`HeightMap`
   * - Add deformable cloth or meshes
     - ``World::addDeformableCloth``, ``World::addDeformableObject``
     - :doc:`DeformableObject`
   * - Add granular particles
     - ``World::addGranularBox``, ``World::addGranularParticles``,
       ``raisim::GranularSystem``
     - :doc:`GranularMedia`
   * - Add length constraints, tendons, or loop constraints
     - ``addSpatialTendon``, ``addFixedTendon``, ``addTendonCoupling``,
       articulated-system pin and equality constraints
     - :doc:`Constraints`, :doc:`articulated_system/ClosedLoopSystems`
   * - Make one joint follow another (mimic)
     - URDF ``<mimic>``, MJCF ``<equality><joint>``,
       ``ArticulatedSystem::addMimicConstraint``
     - :doc:`articulated_system/MimicJoints`
   * - Route cables or couple joints with tendons
     - ``World::addSpatialTendon``, ``addFixedTendon``, ``addTendonCoupling``
     - :doc:`Tendons`, :doc:`tendons/Physics`, :doc:`tendons/Examples`

Contacts, collision, and materials
==================================

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Goal
     - Main API
     - Documentation
   * - Inspect contacts after a step
     - ``raisim::Contact``, object contact accessors
     - :doc:`Contact`, :doc:`Object`
   * - Configure contact material behavior
     - ``raisim::MaterialManager``, ``World::setMaterialPairProp``,
       ``World::updateMaterialProp``, default material properties,
       rolling/spinning friction coefficients
     - :doc:`MaterialSystem`
   * - Change collision filtering or contact settings
     - object collision and contact-setting APIs
     - :doc:`CollisionDetection`, :doc:`Object`
   * - Work with object state and appearance
     - ``raisim::Object`` and derived body classes
     - :doc:`Object`, :doc:`SingleBodyObjects`, :doc:`ArticulatedSystem`

Sensors and ray queries
=======================

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Goal
     - Main API
     - Documentation
   * - Cast a single CPU ray
     - ``World::rayTest``
     - :doc:`RayTest`
   * - Simulate lidar rays
     - ``World::rayTestLidar``, ``raisim::SpinningLidar``
     - :doc:`Sensors`, :doc:`RayTest`
   * - Render RGB or depth sensor images
     - ``raisin::RayraiWindow``, manual sensor measurement sources
     - :doc:`Rayrai`, :doc:`Visualization`, :doc:`Sensors`
   * - Capture depth without rayrai
     - ``World::captureDepthCamera``
     - :doc:`Sensors`
   * - Read inertial measurements
     - ``raisim::InertialMeasurementUnit``
     - :doc:`Sensors`

Visualization and tools
=======================

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Goal
     - Main API or target
     - Documentation
   * - View a ``RaisimServer`` application
     - ``raisim::RaisimServer``, ``rayrai_tcp_viewer``
     - :doc:`Visualization`, :doc:`RaisimServer`
   * - Render in process
     - ``raisin::RayraiWindow``
     - :doc:`Rayrai`, :doc:`Visualization`
   * - Inspect PBR, glTF, HDR, OpenUSD, and textured URDF assets
     - ``rayrai_pbr_material_grid``,
       ``rayrai_pbr_texture_maps``,
       ``rayrai_visual_asset_support``
     - :doc:`Examples`, :doc:`Rayrai`, :doc:`OpenUSD`
   * - Render dense vegetation
     - ``InstancedVisuals`` automatic mesh LOD, foliage wind, and shadow LOD;
       ``rayrai_forest``
     - :doc:`rayrai/Visuals`, :doc:`examples/rayrai/rayrai_forest`
   * - Load a RaiSim Engine ``.rscene`` scene (from v2.7.1, not yet released)
     - ``raisim::World(path)``, ``raisin::applyRscene``,
       ``raisin::rsceneRenderSettings``; ``rayrai_forest_from_rscene``
     - :doc:`RsceneFile`, :doc:`WorldSystem`,
       :doc:`examples/rayrai/rayrai_forest_from_rscene`
   * - Render glass
     - ``Material::glass``, ``RenderQualitySettings::screenSpaceRefraction``,
       ``RenderQualitySettings::geometryRefraction``
     - :doc:`rayrai/Materials`, :doc:`rayrai/PostProcess`
   * - Visualize tendons
     - ``RayraiWindow::getTendonVisual``, ``rayrai_tendons``
     - :doc:`tendons/Examples`
   * - Exercise sensor and rayrai rendering examples
     - ``rayrai_rgb_camera``, ``rayrai_depth_camera``,
       ``rayrai_lidar_pointcloud``, ``rayrai_complete_showcase``
     - :doc:`Examples`, :doc:`BuildAndTest`
   * - Tune simulation performance
     - timing examples, collision settings, sleeping islands, solver settings
     - :doc:`Performance`, :doc:`BuildAndTest`

Build and repository workflow
=============================

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Goal
     - Documentation
   * - Build the public distribution examples and wrappers
     - :doc:`QuickStart`, :doc:`BuildAndTest`
   * - Run C++ tests, benchmarks, and docs builds
     - :doc:`BuildAndTest`
   * - Find generated executables or source directories
     - :doc:`ProjectLayout`
   * - Diagnose missing targets, libraries, activation keys, viewer
       connections, or documentation warnings
     - :doc:`Troubleshooting`
