#############################
Height Map Using Raw Values
#############################

.. image:: ../../rsc/docs/image/heightMapUsingRawValues.gif

XML approach
-----------------------------

The matching asset is
``rsc/xmlScripts/heightMaps/heightMapUsingRawValues.xml``. Load it through the
world configuration constructor in your application:

.. code-block:: cpp

    raisim::World world("/path/to/raisim2Lib/rsc/xmlScripts/heightMaps/heightMapUsingRawValues.xml");

The XML file is equivalent to the following. The asset spells the attributes
in snake_case (``x_sample``, ``x_size``, ``center_x``, ``lin_vel``, ...); the
reader accepts both spellings. ``xSample`` and ``ySample`` are the numbers of
samples, and ``height`` lists ``xSample * ySample`` values row by row (x varies
fastest).

.. code-block:: xml

    <?xml version="1.0" ?>
    <raisim version="2.0.0">
        <timeStep value="0.001"/>
        <objects>
            <sphere name="sphere" mass="1">
                <dim radius="0.5" />
                <inertia xx="0.1" xy="0" xz="0" yy="0.1" yz="0" zz="0.1" />
                <state pos="0 0 5" quat="1 0 0 0" linVel="0 0 0" angVel="0 0 0" />
            </sphere>
            <capsule name="capsule" mass="1">
                <dim radius="0.5" height="1" />
                <inertia xx="0.1" xy="0" xz="0" yy="0.1" yz="0" zz="0.1" />
                <state pos="1 0 5" quat="1 0 0 0" linVel="0 0 0" angVel="0 0 0" />
            </capsule>
            <box name="box" mass="1">
                <dim x="0.5" y="1" z="2"/>
                <inertia xx="0.1" xy="0" xz="0" yy="0.1" yz="0" zz="0.1" />
                <state pos="1 1 5" quat="1 0 0 0" linVel="0 0 0" angVel="0 0 0" />
            </box>
            <heightmap name="terrain" xSample="5" ySample="5" xSize="10" ySize="10" centerX="0" centerY="0"
                      height="1 1 1 1 1 1 0 0 0 1 1 0.5 0 0 1 1 0 0 0 1 1 1 1 1 1"/>
        </objects>
    </raisim>

C++ approach
-----------------------------

The arguments are ``xSamples``, ``ySamples``, ``xSize``, ``ySize``,
``centerX``, ``centerY``, and the height samples (one row of five values per
line below):

.. code-block:: cpp

  std::vector<double> height = {
    1, 1,   1, 1, 1,
    1, 0,   0, 0, 1,
    1, 0.5, 0, 0, 1,
    1, 0,   0, 0, 1,
    1, 1,   1, 1, 1
  };
  auto* heightMap = world.addHeightMap(5, 5, 10, 10, 0, 0, height);

