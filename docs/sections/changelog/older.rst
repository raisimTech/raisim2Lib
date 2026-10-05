Changelog: v1.x and earlier
===========================

Brief per-version notes reconstructed from the RaiSim source history.

v1.1.8 (2024-04-07)
-------------------

* Sensors: IMU methods, sensor getters/setters, modifiable sensor properties; sensor update no longer user-callable; depth order, IMU and timing fixes.
* Mutexes for all objects and the world; visual heightmap.
* Internal contact-count limit, joint velocity limit getter, named generalized-velocity indices; ``base_link`` accepted as fixed-base link.
* Contact solver tuned for pin joints and closed loops; new Cholesky inverse; RK4 (single body), inverse dynamics and fixed-base nonlinearity fixes.
* Fixed ``setExternalForce`` transpose, a contact segfault and transparency; ``setupSocket`` takes a port; dynamic visual meshes.
* GCC builds, Linux debug build, macOS/M1 install fixes.

v1.1.7 (2023-03-31)
-------------------

* IMU sensor and inverse dynamics.
* Dynamic heightmaps with per-vertex colours; collision shape parameter fix.
* Configurable wire width in visualization.
* Server protocol update; sensor freeze and parsing fixes.

v1.1.6 (2022-12-04)
-------------------

* Improved, faster contact solver.
* Screenshot method, world-time display, server info on connect, ``TimedLoop``.
* Body type on object spawn; separate interaction-force function; improved world config files.
* Fixed wire detach, generalized force errors and server getter ordering.

v1.1.5 (2022-09-20)
-------------------

* ``World::exportToXML`` takes a single argument.
* ``heightOffset`` option and interaction mode for raisimUnreal.
* MJCF loading, Windows debug postfix, contact visualization and object focus fixes.

v1.1.4 (2022-08-27)
-------------------

* Visual tags (new server mode) and ``setMap``.
* Sensor API clean-up; sensor frame and timing fixes.
* Server reports request errors; Linux socket fix; joint-limit and robot visualization fixes.

v1.1.3 (2022-07-24)
-------------------

* New server protocol: visual instancing, synchronous mode, sensor data exchange (Unreal-compatible).
* Non-blocking, safety-checked server socket with version-mismatch warning; collision body changes shown in visualization.
* ``raisimODE`` built with an SOVERSION; ``setGeneralizedVelocity`` no longer updates kinematics.

v1.1.2 (2022-03-21)
-------------------

* New SAP broad phase and ray test (large speed-up); static friction.
* Initial sensors (URDF sensor reader, RealSense XML).
* Velocity limits, spring APIs, Jacobian time derivative; external force/torque in body frame; URDF pin joints.
* MJCF fixes (multiple joints, Euler order, refs, transparency); Cassie and Digit examples; default base joint is fixed.
* ``getBodyIdx``/``getFrameIdx`` return -1 when not found; heightmap and ray test fixes.
* License tooling fixes; SOVERSION library versioning; Apple M1 CMake fix.

v1.1.0 (2021-07-01)
-------------------

* New ABA dynamics with contacts, joint limits and external forces; large speed-up.
* Simplified contact solver; working self-collision.
* MJCF reader (frames, defaults, assets/materials, ground, single bodies).
* Fast Cholesky inverse; velocity computation separated from kinematics.

v1.0.5 (2021-04-29)
-------------------

* Server: video recording, ghost robots, external/contact force visualization, live visual shape changes, COM display.
* XML: recursive compounds, wires, parameters with unresolved-name checks, lower-case element aliases.
* ``setExternalForce`` variants, clear external force/torque, PD getters, ``setColor``, custom wires, joint-limit check, dynamic collision groups; ``getLinkCOM`` → ``getBodyCOM``; ``setLicenseFile`` → ``setActivationKey``.
* RK4 integration (no contact), new articulated-system constructor, constraint/spring fixes, URDF materials.
* Activation-key licensing (searched in ``~/.raisim``); Windows and Apple M1 support; bundled Eigen.

v1.0.0 (2020-05-04)
-------------------

* New math library and frame Jacobians; Bullet and Python dependencies removed.
* Worlds from XML (compounds, ``[THIS_DIR]``, time step); capsule-box collision; heightmap normals and ray collision.
* PD controller fixes, torque limits; base-only URDFs.
* LTO (~15% faster); C++14; Windows, MATLAB and raisimUnity compatibility.

v0.7.0 (2020-01-07)
-------------------

* World loading from XML (articulated systems, heightmaps, meshes, visuals, object classes).
* Server: add/remove visual objects, ``hibernate()``/``wakeup()``, float data transfer.
* Ray test; cleaned-up frame/joint methods; compliant wire, contact force and heightmap position fixes.
* Removed ``setBasePos``/``setBaseOrientation``; print movable joints.

v0.6.0 (2019-10-29)
-------------------

* Link/joint reference interface; access to collision and visual object sets.
* ``setExternalTorque``; multiple articulated systems per URDF; URDF from string.
* Server: configurable port, serialized contacts; ``getObject`` by index.
* rpath on binaries; ``getGravity`` is const.

v0.5.0 (2019-09-27)
-------------------

* Improved ERP.
* Mesh objects in the server; capsule and half-space server fixes.
* Joint order fix; missing resource directory is a warning.

v0.4.x (2019-08-23 – 2019-09-06)
--------------------------------

* 0.4.3: sparsity pattern fix.
* 0.4.2: ODE built as a shared library.
* 0.4.1: no space argument in the articulated-system constructor; matrix ``()`` operator.
* 0.4.0: mesh single bodies; more visualization methods.

v0.3.x (2019-08-13 – 2019-08-21)
--------------------------------

* 0.3.1: dimension sanity checks.
* 0.3.0: ``_W`` suffix removed from method names.

v0.2.x (2019-07-13 – 2019-08-07)
--------------------------------

* 0.2.1: shared library; change collision shape parameters, set material, get collision bodies by name, base pose getters; Python binding headers.
* 0.2.0: material system (``setDefaultMaterial``, ``setMaterialPairProp``); objects/wires by name; actuation limits; actual generalized force getter.

v0.1.x (2019-04-10 – 2019-05-10)
--------------------------------

* 0.1.2: heightmap from PNG; joint frame and solver state getters.
* 0.1.0: first versioned release — articulated systems, single bodies, compounds, heightmaps, meshes, wires; remote visualization server; Python interface.
