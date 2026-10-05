#############################
Performance
#############################

This page is a practical tuning guide for users of the ``raisim2Lib`` binary
package workspace and for downstream applications linked against the downloaded
RaiSim and rayrai packages. For commands that build and run package examples,
see :doc:`BuildAndTest`.

Fast scenes usually come from three choices:

* keep collision geometry simple and intentional;
* avoid creating more contacts than the task needs;
* measure with timing-oriented package examples or your downstream workload on
  one thread before changing solver or modeling choices.

What Usually Costs Time
=======================

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Cost center
     - What increases it
     - What to do first
   * - Broadphase collision detection
     - Many active collision bodies, large worlds, or dense moving object sets.
     - Use collision groups/masks and choose a broadphase that matches the scene.
       See :doc:`WorldSystem` and :doc:`CollisionDetection`.
   * - Narrowphase collision detection
     - Expensive shape pairs, dense mesh contact, or geometry that creates many
       candidate contacts.
     - Use primitives, height maps, simplified convex collision assets, or cached
       mesh preprocessing where possible. See :doc:`SingleBodyObjects`.
   * - Contact count
     - Many articulated bodies touching many other bodies.
     - Remove unnecessary collision shapes and split non-interacting objects with
       collision masks.
   * - Contact solver
     - Stacks, pinched contacts, highly coupled contacts, or excessive contact
       manifolds.
     - Reduce redundant contacts before tuning solver iterations.
   * - Sensors
     - Large CPU depth images, many lidar rays, or high sensor update rates.
     - Use in-process rayrai for RGB/depth sensor images when rayrai is
       available. Lower CPU ray-query resolution/update rate only for headless
       fallback or deterministic physics-ray workloads. See :doc:`Sensors`.
   * - Visualization
     - Rendering in the simulation process, high-quality PBR settings, or
       synchronous screen capture.
     - Use ``RaisimServer`` + ``rayrai_tcp_viewer`` for debugging, and
       in-process rayrai only when the application needs rendered images. See
       :doc:`Visualization`.

Collision Modeling
==================

Collision geometry is often the largest performance lever. Visual meshes should
not automatically become collision meshes. Prefer this order:

1. primitive shapes for robot links and simple objects;
2. height maps for terrain;
3. preprocessed convex collision assets for irregular objects;
4. mesh collision only when the task really needs mesh-level contact.

When a mesh collision asset is needed repeatedly, reuse the CoACD decomposition
cache that ``World::addMesh`` writes and the normalized OBJ cache produced by
``Mesh::preprocessMesh`` (see :doc:`SingleBodyObjects`). The default broadphase
is ``contact::BroadphaseType::Sap3Axis``. For broadphase-heavy scenes, try
``contact::BroadphaseType::MultiBoxPrune`` through
``World::setBroadphaseSettings``, with grid bounds and cell sizes
(``mbpWorldMin``, ``mbpWorldMax``, ``mbpCellSize``) that match the active world
volume, and measure both. Scenes with fewer than 35 collision bodies already use
an all-pairs AABB test regardless of the setting. ``BroadphaseType::None``
skips AABB culling entirely and is only meant for debugging.

Contacts and Solver Work
========================

The contact solver cost depends on how coupled the contacts are. Ten isolated
contacts are very different from ten contacts in a compressed stack. Before
tuning solver parameters, check whether the model is generating contacts that do
not matter for the task:

* remove decorative collision bodies;
* simplify foot, gripper, and terrain collision geometry;
* avoid overlapping collision shapes in one body unless they are intentional;
* use collision groups and masks to skip pairs that should never interact;
* keep timesteps and material parameters within the stability range required by
  the task.

``World::setContactSolverParam`` exposes solver parameters for advanced users
(only the maximum iteration count and the termination threshold take effect),
but it is usually a later step. Reducing redundant contacts tends to be more
robust than asking the solver to process a harder problem faster. See
:doc:`Contact` for the solver notes.

Sleeping Islands
================

Sleeping is enabled by default. RaiSim skips simulation work for a dynamic
island once every body in it stays below the linear (0.002 m/s) and angular
(0.01 rad/s) speed thresholds for 5 consecutive steps; these are the defaults.
This helps scenes with piles, props, or objects that settle and then stay quiet.
Disabling sleeping wakes every sleeping object.

.. code-block:: cpp

  world.setSleepingEnabled(true);
  world.setSleepingParameters(/*linear*/ 0.002, /*angular*/ 0.01, /*quietSteps*/ 5);  // defaults
  world.wakeAll();  // wake every object on the next step

Disable sleeping when every object must remain numerically active every step, or
when a benchmark is intended to measure the awake dynamics path. The public
``island_sleep_benchmark`` demonstrates sleep/wake behavior visually; it includes
a server and a fixed 2 ms delay per step, so its elapsed time is not a measure
of raw physics throughput.

Package Timing Examples
=======================

``raisim2Lib`` ships source for standalone timing-oriented examples for binary
package users. Maintainers use a separate unified benchmark runner from the
closed-source RaiSim engine repository for release validation; that runner is
not available in this public workspace. Build the public examples in Release
mode and run timing commands on one thread:

.. code-block:: bash

  cmake -S . -B build-examples \
      -DCMAKE_BUILD_TYPE=Release \
      -DRAISIM_EXAMPLE=ON
  cmake --build build-examples \
      --target anymal_standing_benchmark articulated_system_benchmark --parallel 12
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      ./build-examples/examples/anymal_standing_benchmark --fast
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      ./build-examples/examples/articulated_system_benchmark

The public workspace does not install these example executables; run them from
the build tree shown above (``build-examples\bin`` on Windows).
``--fast`` caps ``anymal_standing_benchmark`` at 20,000 steps; ``--steps=N``
sets the step count explicitly.

Public timing and visualization examples include:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Executable
     - Use it for
   * - ``anymal_standing_benchmark``
     - Headless articulated robot contact scenario.
   * - ``articulated_system_benchmark``
     - Articulated-system stepping cost.
   * - ``granular_media``
     - Visual granular-bed interaction demo with a server and pacing delays.
   * - ``island_sleep_benchmark``
     - Visual sleep/wake demo with a fixed delay per step; not raw timing.

For stable comparisons, run the same executable, package version, compiler,
build type, CPU governor, benchmark arguments, and repeat count. Run one timing
process at a time. Avoid comparing a visualized run against a headless run
unless visualization is the measured workload.

Threading and Determinism
=========================

RaiSim simulation performance should be measured on one simulation thread unless
the experiment explicitly studies external parallelism. Running many worlds in
parallel is an application-level design choice; it is different from changing the
deterministic stepping behavior of one ``raisim::World``.

``World::setContactSolverIterationOrder(order)`` sets the starting sweep
direction for the next contact solve. The solver flips the stored direction
after each solve, so call it before each measured step only if the experiment
requires the same starting sweep every time. For reproducible benchmark numbers,
also control random seeds in the benchmark or application, keep visualization
disabled unless measured, and pin the same benchmark arguments.

Algorithm Background
====================

RaiSim uses an articulated-system dynamics path designed to avoid forming and
inverting a full mass matrix for ordinary forward dynamics. The practical effect
is that articulated dynamics without heavy contact work scales much better with
degrees of freedom than a pipeline that explicitly forms and factorizes the mass
matrix every step.

That does not make every simulation linear in model size. Collision detection,
contact generation, solver coupling, deformable bodies, granular media, sensors,
and rendering can dominate the wall time. Treat old headline kHz numbers as
historical context, not as a guarantee for a current scene. Measure the package
example or downstream application you actually run, with the compiler, hardware,
and build options in front of you.
