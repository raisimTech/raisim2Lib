#############################
Height Map Using a PNG File
#############################

.. image:: ../../rsc/docs/image/heightMapUsingPNG.gif

The PNG must be an 8-bit or 16-bit grayscale image. Each pixel becomes one
height sample with the value ``pixelValue * heightScale + heightOffset``, and
the image width and height set the number of samples along x and y.

XML approach
-----------------------------

The matching asset is
``rsc/xmlScripts/heightMaps/heightMapUsingPng.xml``. Load it through the world
configuration constructor in your application:

.. code-block:: cpp

    raisim::World world("/path/to/raisim2Lib/rsc/xmlScripts/heightMaps/heightMapUsingPng.xml");

The XML file is equivalent to the following. The asset spells the attributes
in snake_case (``x_size``, ``center_x``, ``z_offset``, ``z_scale``, ...); the
reader accepts both spellings.

.. code-block:: xml

    <?xml version="1.0" ?>
    <raisim version="2.0.0">
        <timeStep value="0.001"/>
        <objects>
            <articulatedSystem name="anymal" resDir="[THIS_DIR]/../../anymal" urdfPath="[THIS_DIR]/../../anymal/urdf/anymal.urdf" collisionGroup="1" collisionMask="-1">
                <state qpos="0, 0, 10.84, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8"
                       qvel="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0" />
            </articulatedSystem>
            <heightmap name="terrain" png="[THIS_DIR]/zurichHeightMap.png" xSize="500" ySize="500" centerX="0" centerY="0" heightOffset="-10" heightScale="0.005"/>
        </objects>
    </raisim>

C++ approach
-----------------------------

The arguments are the PNG path, ``centerX``, ``centerY``, ``xSize``, ``ySize``,
``heightScale``, and ``heightOffset``:

.. code-block:: cpp

  auto* heightMap = world.addHeightMap("/path/to/zurichHeightMap.png",
                                       0, 0, 500, 500, 0.005, -10);
