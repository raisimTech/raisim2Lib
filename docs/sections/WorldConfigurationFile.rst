########################
World Configuration File
########################

Examples are provided `here <https://github.com/raisimTech/raisimLib/tree/master/rsc/xmlScripts>`__.

The following section outlines the conventions for the RaiSim world configuration XML file.
Tags marked (optional) indicate elements that are not mandatory within their parent context.
Unmarked elements are mandatory if their parent element is present.
Tags marked (multiple) indicate that multiple instances of the element may exist under the same parent.

1. raisim: Top-level node.
    1. <attribute> ``version`` : Specifies the RaiSim version used to create the configuration file. Files may be compatible with different versions.
    2. <child> (optional) ``material`` : Refer to the `Material System <https://raisim.com/sections/MaterialSystem.html>`__ section for details and examples.
        1. <child> (optional) ``default`` : If omitted, default parameters are used as described `here <https://raisim.com/sections/MaterialSystem.html>`__.
            1. <attribute> ``friction`` [float]
            2. <attribute> ``restitution`` [float]
            3. <attribute> ``restitution_threshold`` [float]
        2. <child> (optional, multiple) ``pair_prop``
            1. <attribute> ``name1`` [string]
            2. <attribute> ``name2`` [string]
            3. <attribute> ``friction`` [float]
            4. <attribute> ``restitution`` [float]
            5. <attribute> ``restitution_threshold`` [float]
    3. <child> (optional) ``gravity``
        1. <attribute> ``value`` [Vec<3>]: default={0, 0, -9.81}
    4. <child> (optional) ``timestep``
        1. <attribute> ``value`` [float]: default = 0.005
    5. <child> (optional) ``erp`` : (Advanced) Defines spring and damping terms for error dynamics.
        1. <attribute> ``erp`` : spring term [float]
        2. <attribute> ``erp2`` : damping term [float]
    6. <child> ``objects`` : Detailed in the "Object XML Description" section.
    7. <child> ``wire`` : `DESCRIPTION <https://raisim.com/sections/Constraints.html>`_, `EXAMPLES <https://github.com/raisimTech/raisimLib/blob/master/rsc/xmlScripts/wire/newtonsCradle.xml>`__
        1. <attribute> ``name`` [string]: Wire name.
        2. <attribute> ``type`` [string]: Wire type ("stiff", "compliant", or "custom").
        3. <attribute> ``stretch_type`` [string]: Stretch type.
        4. <child> ``object1`` and ``object2`` : The objects to which the wire is attached.
            Option 1 (Single Body Object):
            1. <attribute> ``local_index`` [int]: The body index for attachment (0 for singleBodyObject).
            2. <attribute> ``pos`` [Vec<3>]: The attachment position in the local body frame.
            Option 2 (Articulated System):
            1. <attribute> ``frame`` [string]: The name of the frame where the wire is attached.

Object XML Description
----------------------------

Collision groups and masks can be specified as ``collision[1]`` or ``collision[1|4|6]``.
The former represents collision group 1.
The latter defines a collision mask interacting with groups 1, 4, and 6.
The mask ``collision[-1]`` indicates collision with any group (note: collision logic is an AND operation; both mask and group pairs must be compatible).

sphere
^^^^^^^^^^^^^
`EXAMPLES <https://github.com/raisimTech/raisimLib/blob/master/rsc/xmlScripts/objects/SingleBodies.xml>`__

**attributes**: (optional, default=1) ``collision_group`` [uint64_t], (optional, default=-1) ``collision_mask`` [uint64_t], (optional) ``appearance`` [string], (optional) ``body_type`` [string]: {dynamic, kinematic, static}, (optional) ``name`` [string], ``mass`` [float]

1. <child> (optional, default=Derived from geometry assuming uniform density) ``inertia`` : <attribute> ``xx`` [float], ``xy`` [float], ``xz`` [float], ``yy`` [float], ``yz`` [float], ``zz`` [float]

2. <child> dim: <attribute> radius[float]

3. <child> state: <attribute> ``pos`` [Vec<3>], (optional, default=1,0,0,0) ``quat`` [Vec<4>], (optional, default=0,0,0) ``lin_vel`` [Vec<3>], (optional, default=0,0,0) ``ang_vel`` [Vec<3>]

capsule and cylinder
^^^^^^^^^^^^^^^^^^^^^^^
`EXAMPLES <https://github.com/raisimTech/raisimLib/blob/master/rsc/xmlScripts/objects/SingleBodies.xml>`__

**attributes**: (optional, default=1) collision_group[uint64_t], (optional, default=-1) collision_mask[uint64_t], (optional) appearance[string], (optional) body_type[string]: {dynamic, kinematic, static}, (optional) name[string], mass[float]

1. <child> (optional, default=Derived from geometry assuming uniform density) ``inertia`` : <attribute> ``xx`` [float], ``xy`` [float], ``xz`` [float], ``yy`` [float], ``yz`` [float], ``zz`` [float]

2. <child> dim: <attribute> ``radius`` [float], ``height`` [float]

3. <child> state: <attribute> ``pos`` [Vec<3>], (optional, default=1,0,0,0) ``quat`` [Vec<4>], (optional, default=0,0,0) ``lin_vel`` [Vec<3>], (optional, default=0,0,0) ``ang_vel`` [Vec<3>]

box
^^^^^^^^^^^^^^^^^^^^^^^
`EXAMPLES <https://github.com/raisimTech/raisimLib/blob/master/rsc/xmlScripts/objects/SingleBodies.xml>`__

**attributes**: (optional, default=1) ``collision_group`` [uint64_t], (optional, default=-1) ``collision_mask`` [uint64_t], (optional) ``appearance`` [string], (optional) ``body_type`` [string]: {dynamic, kinematic, static}, (optional) ``name`` [string], ``mass`` [float]

1. <child> (optional, default=Derived from geometry assuming uniform density) ``inertia`` : <attribute> ``xx`` [float], ``xy`` [float], ``xz`` [float], ``yy`` [float], ``yz`` [float], ``zz`` [float]

2. <child> dim: <attribute> ``x`` [float], ``y`` [float], ``z`` [float]

3. <child> state: <attribute> ``pos`` [Vec<3>], (optional, default=1,0,0,0) ``quat`` [Vec<4>], (optional, default=0,0,0) ``lin_vel`` [Vec<3>], (optional, default=0,0,0) ``ang_vel`` [Vec<3>]

compound
^^^^^^^^^^^^^^^^^^^^^^^
`EXAMPLES <https://github.com/raisimTech/raisimLib/blob/master/rsc/xmlScripts/objects/SingleBodies.xml>`__

**attributes**: (optional, default=1) ``collision_group`` [uint64_t], (optional, default=-1) ``collision_mask`` [uint64_t], (optional) ``appearance`` [string], (optional) ``body_type`` [string]: {dynamic, kinematic, static}, (optional) ``name`` [string], ``com`` [Vec<3>], ``mass`` [float]

1. <child> (optional, default=Derived from geometry assuming uniform density) ``inertia``
    **attributes**: ``xx`` [float], ``xy`` [float], ``xz`` [float], ``yy`` [float], ``yz`` [float], ``zz`` [float]

2. <child> ``children``
   Common attributes of the children: ``appearance`` [string]

   ::

     <child> (optional, multiple) sphere
       <child> dim
         <attribute> radius [float]
       <attribute> (optional, default=default) material
     <child> (optional, multiple) cylinder
       <child> dim
         <attribute> radius [float]
         <attribute> height [float]
       <attribute> (optional, default=default) material
     <child> (optional, multiple) capsule
       <child> dim
         <attribute> radius [float]
         <attribute> height [float]
       <attribute> (optional, default=default) material
     <child> (optional, multiple) box
       <child> dim
         <attribute> x [float]
         <attribute> y [float]
         <attribute> z [float]
       <attribute> (optional, default=default) material

3. <child> state
    **attributes**: ``pos`` [Vec<3>], (optional, default=1,0,0,0) ``quat`` [Vec<4>], (optional, default=0,0,0) ``lin_vel`` [Vec<3>], (optional, default=0,0,0) ``ang_vel`` [Vec<3>]

mesh
^^^^^^^^^^^^^^^^^^^^^^^
`EXAMPLES <https://github.com/raisimTech/raisimLib/blob/master/rsc/xmlScripts/objects/SingleBodies.xml>`__

**attributes**: (optional, default=1) ``collision_group`` [uint64_t], (optional, default=-1) ``collision_mask`` [uint64_t], (optional) ``appearance`` [string], (optional) ``body_type`` [string]: {dynamic, kinematic, static}, (optional) ``name`` [string], ``mass`` [float], ``file_name`` [string], ``com`` [Vec<3>], ``scale`` [Vec<3>]

1. <child> (optional, default=Derived from geometry assuming uniform density) ``inertia``
    **attributes**: ``xx`` [float], ``xy`` [float], ``xz`` [float], ``yy`` [float], ``yz`` [float], ``zz`` [float]

2. <child> ``state``
    **attributes**: ``pos`` [Vec<3>], (optional, default=1,0,0,0) ``quat`` [Vec<4>], (optional, default=0,0,0) ``lin_vel`` [Vec<3>], (optional, default=0,0,0) ``ang_vel`` [Vec<3>]

ground
^^^^^^^^^^^^^^^^^^^^^^^
`EXAMPLES <https://github.com/raisimTech/raisimLib/blob/master/rsc/xmlScripts/material/material.xml>`__

**attributes**: (optional, default=-1) ``collision_mask`` [uint64_t], (optional) ``appearance`` [string], (optional) ``name`` [string], (optional, default=0) ``height`` [float]

heightmap
^^^^^^^^^^^^^^^^^^^^^^^
`EXAMPLES <https://github.com/raisimTech/raisimLib/tree/master/rsc/xmlScripts/heightMaps>`__

**Options**

1. **attributes**: (optional, default=-1) ``collision_mask`` [uint64_t], (optional) ``appearance`` [string], (optional, default=default) ``material`` [string], (optional) ``name`` [string], ``x_sample`` [size_t], ``y_sample`` [size_t], ``x_size`` [float], ``y_size`` [float], ``center_x`` [float], ``center_y`` [float], ``height`` [std::vector<float>]

2. **attributes**: (optional, default=-1) ``collision_mask`` [uint64_t], (optional) ``appearance`` [string], (optional, default=default) ``material`` [string], (optional) ``name`` [string], ``x_sample`` [size_t], ``y_sample`` [size_t], ``x_size`` [float], ``y_size`` [float], ``center_x`` [float], ``center_y`` [float], ``z_scale`` [float], ``z_offset`` [float], ``png`` [string]

3. **attributes**: (optional, default=-1) ``collision_mask`` [uint64_t], (optional) ``appearance`` [string], (optional, default=default) ``material`` [string], (optional) ``name`` [string], ``center_x`` [float], ``center_y`` [float], ``text`` [string]

4. **attributes**: (optional, default=-1) ``collision_mask`` [uint64_t], (optional) ``appearance`` [string], (optional, default=default) ``material`` [string], (optional) ``name`` [string], ``x_sample`` [size_t], ``y_sample`` [size_t], ``x_size`` [float], ``y_size`` [float], ``center_x`` [float], ``center_y`` [float]
    1. <child> ``terrain_properties``
        **attributes**: ``z_scale`` [float], ``fractal_octaves`` [size_t], ``fractal_lacunarity`` [float], ``fractal_gain`` [float], ``step_size`` [float], ``frequency`` [float], ``seed`` [size_t]

articulated_system
^^^^^^^^^^^^^^^^^^^^^
`EXAMPLES <https://github.com/raisimTech/raisimLib/blob/master/rsc/xmlScripts/heightMaps/heightMapUsingPng.xml>`__

**attributes**: (optional, default=1) ``collision_group`` [uint64_t], (optional, default=-1) ``collision_mask`` [uint64_t], (optional) ``name`` [string], (optional, default=URDF directory) ``res_dir`` [string], ``urdf_path`` [string]

1. <child> ``state``
    **attributes**: ``qpos`` [VecDyn], (optional, default=zeros) ``qvel`` [VecDyn]


Configuration Template
----------------------------
Configuration templates facilitate the systematic creation of simulation worlds.
An example is available `here <https://github.com/raisimTech/raisimLib/tree/master/rsc/xmlScripts/templatedWorld/templatedWorld.xml>`__.

External configuration files can be imported using the ``include`` tag.
This is beneficial for modularly combining sub-worlds.

Parameters are denoted by the prefix ``@@``.
Variables may be defined within the script using the ``params`` tag.
An example is available `here <https://github.com/raisimTech/raisimLib/tree/master/rsc/xmlScripts/templatedWorld/templatedWorld.xml>`__.
This allows for runtime modification of the world or the generation of configuration variants.

The ``array`` tag enables the creation of **loops** within the configuration file.
An example is available `here <https://github.com/raisimTech/raisimLib/tree/master/rsc/xmlScripts/templatedWorld/spheres.xml>`__.
The attributes ``idx``, ``start``, ``end``, and ``increment`` are mandatory.
The latter three attributes must be integers.

Mathematical expressions are also supported within the configuration file.
An example is available in the same file.
Expressions must be encapsulated within ``{}``.
Supported functions include basic operations and ``sin``, ``cos``, ``exp``, and ``log``.
Parentheses ``()`` are used for function calls and grouping, consistent with C++ syntax.

Objects support an optional ``exist`` attribute.
This enables conditional object existence based on runtime parameters.
An example file is available `here <https://github.com/raisimTech/raisimLib/tree/master/rsc/xmlScripts/templatedWorld/spheres.xml>`__.
